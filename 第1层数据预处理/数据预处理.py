# -*- coding: utf-8 -*-
"""
Preprocess Amazon products + reviews tables (厨刀品类示例)

Input (default):
  - /mnt/data/merged_products_unique_asin.csv
  - /mnt/data/merged_reviews_unique_review_id.csv

Outputs (default folder: /mnt/data/preprocessed):
  - products_clean.csv
  - product_images.csv
  - product_subcat_ranks.csv
  - reviews_clean.csv
  - fact_review_enriched.csv
  - agg_product.csv
  - agg_product_week.csv

Notes:
  - products: asin 为主键（唯一）
  - reviews: review_id 为主键（唯一）
  - reviews 中 rating 重命名为 review_rating，避免与产品 rating 冲突
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def safe_json_load(x: Any) -> List[Any]:
    """Safely parse a JSON string into python object; fallback to empty list."""
    if pd.isna(x):
        return []
    s = str(x).strip()
    if s in ("", "nan", "None"):
        return []
    try:
        return json.loads(s)
    except Exception:
        return []


def parse_usd(x: Any) -> float:
    """Parse strings like '$1,234.56' into float."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return np.nan
    s = s.replace("$", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return np.nan


def to_bool(x: Any) -> bool | float:
    """Convert 0/1/true/false/yes/no into bool; unknown -> np.nan."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    return np.nan


def clean_text_basic(s: Any) -> str:
    """Lightweight text cleanup: strip HTML, normalize whitespace."""
    if pd.isna(s):
        return ""
    s = str(s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("\u200b", " ")  # zero-width space
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_stock_left(avail: Any) -> float:
    """Extract stock left from 'Only X left in stock - order soon.'."""
    if pd.isna(avail):
        return np.nan
    s = str(avail)
    m = re.search(r"Only\s+(\d+)\s+left in stock", s, flags=re.IGNORECASE)
    return int(m.group(1)) if m else np.nan


def in_stock_flag(avail: Any) -> int:
    """Binary stock flag from availability text."""
    if pd.isna(avail):
        return 0
    s = str(avail)
    return int(bool(re.search(r"\bin stock\b|Only\s+\d+\s+left in stock", s, flags=re.IGNORECASE)))


def parse_variant_kv(variant: Any) -> Dict[str, str]:
    """
    Parse Amazon-style variant string:
      'Color: Black/Gray; Size: 8 Inch; Style: Modern'
    Returns dict with normalized keys (lowercase).
    """
    if pd.isna(variant):
        return {}
    s = str(variant).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return {}

    parts = [p.strip() for p in re.split(r";\s*", s) if p.strip()]
    # If a single segment but multiple ':' and comma-separated, split by comma as fallback
    if len(parts) == 1 and s.count(":") > 1 and "," in s:
        parts = [p.strip() for p in re.split(r",\s*", s) if p.strip()]

    kv: Dict[str, str] = {}
    for p in parts:
        m = re.match(r"^\s*([^:]+?)\s*:\s*(.+?)\s*$", p)
        if not m:
            continue
        k, v = m.group(1).strip(), m.group(2).strip()
        k_norm = re.sub(r"\s+", " ", k).lower()
        v_norm = re.sub(r"\s+", " ", v)
        kv[k_norm] = v_norm
    return kv


def caps_ratio(s: str) -> float:
    """Uppercase-letter ratio among alphabetic chars."""
    if not s:
        return 0.0
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    caps = [c for c in letters if c.isupper()]
    return len(caps) / len(letters)


# -----------------------------
# Products preprocessing
# -----------------------------
def preprocess_products(products_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prod = pd.read_csv(products_path)

    # Drop non-business columns / constant columns
    drop_cols = ["#", "id", "variants", "rating_5_star", "rating_4_star", "rating_3_star", "rating_2_star", "rating_1_star"]
    prod = prod.drop(columns=[c for c in drop_cols if c in prod.columns], errors="ignore")

    # Currency -> float
    prod["price_num"] = prod["price"].apply(parse_usd)
    prod["original_price_num"] = prod["original_price"].apply(parse_usd)

    # list_price vs unit_price (避免 original_price < price 误算折扣)
    prod["list_price"] = np.where(prod["original_price_num"] >= prod["price_num"], prod["original_price_num"], np.nan)
    prod["unit_price"] = np.where(prod["original_price_num"] < prod["price_num"], prod["original_price_num"], np.nan)

    prod["discount_amount"] = prod["list_price"] - prod["price_num"]
    prod["discount_rate"] = prod["discount_amount"] / prod["list_price"]

    # Numeric fields
    prod["product_rating"] = pd.to_numeric(prod["rating"], errors="coerce")
    prod["product_rating_count"] = pd.to_numeric(prod["rating_count"], errors="coerce")
    prod["bsr_rank"] = pd.to_numeric(prod["bsr_rank"], errors="coerce")
    prod["bought_count_number"] = pd.to_numeric(prod["bought_count_number"], errors="coerce")

    prod["has_product_rating"] = prod["product_rating"].notna().astype(int)
    prod["product_rating"] = prod["product_rating"].fillna(0.0)
    prod["product_rating_count"] = prod["product_rating_count"].fillna(0).astype(int)

    # bought_count: 如果 bought_count 文本缺失且 number==0，则视为未知
    prod["has_bought_count_raw"] = prod["bought_count"].notna().astype(int)
    prod["bought_count_number_clean"] = prod["bought_count_number"].where(
        ~((prod["has_bought_count_raw"] == 0) & (prod["bought_count_number"] == 0)),
        np.nan,
    )

    # Booleans
    prod["is_fba"] = prod["is_fba"].apply(to_bool).fillna(False).astype(bool)
    prod["has_aplus"] = prod["has_aplus"].apply(to_bool).fillna(False).astype(bool)

    # Dates
    prod["created_at_dt"] = pd.to_datetime(prod["created_at"], errors="coerce")
    prod["updated_at_dt"] = pd.to_datetime(prod["updated_at"], errors="coerce")
    prod["first_available_dt"] = pd.to_datetime(prod["first_available"], errors="coerce")

    # Text fields
    prod["has_description"] = prod["description"].notna().astype(int)
    prod["description_clean"] = prod["description"].apply(clean_text_basic).fillna("")

    # JSON fields
    prod["bullet_list"] = prod["bullet_points"].apply(safe_json_load)
    prod["bullet_count"] = prod["bullet_list"].apply(lambda xs: len(xs) if isinstance(xs, list) else 0)
    prod["bullet_points_text"] = prod["bullet_list"].apply(
        lambda xs: " ".join([clean_text_basic(x) for x in xs]) if isinstance(xs, list) else ""
    )

    prod["image_list"] = prod["images"].apply(safe_json_load)
    prod["image_count"] = prod["image_list"].apply(lambda xs: len(xs) if isinstance(xs, list) else 0)
    prod["main_image"] = prod["image_list"].apply(lambda xs: xs[0] if isinstance(xs, list) and len(xs) > 0 else np.nan)

    # Availability -> structured
    prod["stock_left"] = prod["availability"].apply(parse_stock_left)
    prod["in_stock_flag"] = prod["availability"].apply(in_stock_flag).astype(int)

    # Normalize categoricals
    prod["brand_norm"] = prod["brand"].astype(str).str.strip().str.lower()
    prod["seller_norm"] = prod["seller"].fillna("").astype(str).str.strip().str.lower()
    prod["bsr_category_norm"] = prod["bsr_category"].fillna("").astype(str).str.strip().str.lower()

    # Sub-category ranks
    prod["subcat_list"] = prod["sub_category_ranks"].apply(safe_json_load)

    def filter_real_subcats(xs: Any) -> List[Dict[str, Any]]:
        if not isinstance(xs, list):
            return []
        out: List[Dict[str, Any]] = []
        for d in xs:
            if not isinstance(d, dict):
                continue
            cat = str(d.get("category", ""))
            if cat.lower().startswith("see top 100 in"):
                continue
            try:
                rank = int(d.get("rank"))
            except Exception:
                continue
            out.append({"rank": rank, "category": cat, "url": d.get("url", None)})
        return out

    prod["subcat_real"] = prod["subcat_list"].apply(filter_real_subcats)
    prod["subcat_count"] = prod["subcat_real"].apply(len)

    def best_subcat(xs: List[Dict[str, Any]]) -> tuple[float, float]:
        if not xs:
            return (np.nan, np.nan)
        best = min(xs, key=lambda d: d["rank"])
        return (best["rank"], best["category"])

    best_vals = prod["subcat_real"].apply(best_subcat)
    prod["best_subcat_rank"] = best_vals.apply(lambda t: t[0])
    prod["best_subcat_name"] = best_vals.apply(lambda t: t[1])

    # Detail table: images
    img_rows: List[Dict[str, Any]] = []
    for asin, lst in zip(prod["asin"], prod["image_list"]):
        if isinstance(lst, list):
            for i, url in enumerate(lst, start=1):
                img_rows.append({"asin": asin, "position": i, "image_url": url})
    product_images = pd.DataFrame(img_rows)

    # Detail table: subcategory ranks
    subcat_rows: List[Dict[str, Any]] = []
    for asin, lst in zip(prod["asin"], prod["subcat_real"]):
        if isinstance(lst, list):
            for d in lst:
                subcat_rows.append({"asin": asin, "rank": d["rank"], "category": d["category"], "url": d.get("url")})
    product_subcat_ranks = pd.DataFrame(subcat_rows)

    # Final products_clean
    products_clean = prod[
        [
            "asin",
            "title",
            "brand",
            "brand_norm",
            "seller",
            "seller_norm",
            "price",
            "price_num",
            "original_price",
            "original_price_num",
            "list_price",
            "unit_price",
            "discount_amount",
            "discount_rate",
            "product_rating",
            "product_rating_count",
            "has_product_rating",
            "bsr_rank",
            "bsr_category",
            "bsr_category_norm",
            "best_subcat_rank",
            "best_subcat_name",
            "subcat_count",
            "bullet_count",
            "bullet_points_text",
            "image_count",
            "main_image",
            "is_fba",
            "has_aplus",
            "first_available_dt",
            "availability",
            "in_stock_flag",
            "stock_left",
            "bought_count",
            "bought_count_number",
            "bought_count_number_clean",
            "has_bought_count_raw",
            "has_description",
            "description_clean",
            "created_at_dt",
            "updated_at_dt",
        ]
    ].copy()

    return products_clean, product_images, product_subcat_ranks


# -----------------------------
# Reviews preprocessing
# -----------------------------
def preprocess_reviews(reviews_path: Path) -> pd.DataFrame:
    rev = pd.read_csv(reviews_path)

    # Drop non-business columns + redundant columns
    rev = rev.drop(columns=[c for c in ["#", "id", "title", "brand"] if c in rev.columns], errors="ignore")

    # Rename to avoid conflicts
    rev = rev.rename(
        columns={
            "date": "review_date",
            "created_at": "scraped_at",
            "content": "review_text_raw",
            "rating": "review_rating",
        }
    )

    # Types
    rev["review_date_dt"] = pd.to_datetime(rev["review_date"], errors="coerce")
    rev["scraped_at_dt"] = pd.to_datetime(rev["scraped_at"], errors="coerce")
    rev["review_rating"] = pd.to_numeric(rev["review_rating"], errors="coerce").astype("Int64")
    rev["verified_purchase"] = rev["verified_purchase"].apply(to_bool).fillna(False).astype(bool)
    rev["helpful_votes"] = pd.to_numeric(rev["helpful_votes"], errors="coerce").fillna(0).astype(int)

    # Text
    rev["review_text_raw"] = rev["review_text_raw"].fillna("")
    rev["review_text_clean"] = rev["review_text_raw"].apply(clean_text_basic)
    rev["has_text"] = (rev["review_text_clean"].str.len() > 0).astype(int)

    # Text features
    rev["text_len"] = rev["review_text_clean"].str.len()
    rev["exclamation_cnt"] = rev["review_text_clean"].str.count("!")
    rev["question_cnt"] = rev["review_text_clean"].str.count(r"\?")
    rev["caps_ratio"] = rev["review_text_clean"].apply(caps_ratio)
    rev["text_hash"] = rev["review_text_clean"].apply(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest() if s else "")

    # Variant parsing
    rev["variant_kv"] = rev["variant"].apply(parse_variant_kv)

    def get_kv(d: Any, key: str) -> Any:
        if not isinstance(d, dict):
            return np.nan
        return d.get(key, np.nan)

    rev["variant_color"] = rev["variant_kv"].apply(lambda d: get_kv(d, "color"))
    rev["variant_style"] = rev["variant_kv"].apply(lambda d: get_kv(d, "style"))
    rev["variant_size"] = rev["variant_kv"].apply(lambda d: get_kv(d, "size"))
    rev["variant_number_of_items"] = rev["variant_kv"].apply(lambda d: get_kv(d, "number of items"))

    # Quality flags
    rev["bad_date_flag"] = (
        (rev["review_date_dt"].notna()) & (rev["scraped_at_dt"].notna()) & (rev["review_date_dt"] > rev["scraped_at_dt"])
    ).astype(int)
    rev["rating_out_of_range_flag"] = (~rev["review_rating"].between(1, 5)).fillna(True).astype(int)

    reviews_clean = rev[
        [
            "review_id",
            "asin",
            "author",
            "review_rating",
            "review_date",
            "review_date_dt",
            "verified_purchase",
            "helpful_votes",
            "variant",
            "variant_color",
            "variant_style",
            "variant_size",
            "variant_number_of_items",
            "review_text_raw",
            "review_text_clean",
            "has_text",
            "text_len",
            "exclamation_cnt",
            "question_cnt",
            "caps_ratio",
            "text_hash",
            "scraped_at",
            "scraped_at_dt",
            "bad_date_flag",
            "rating_out_of_range_flag",
        ]
    ].copy()

    return reviews_clean


# -----------------------------
# Join + Aggregations
# -----------------------------
def build_review_enriched_and_aggs(
    products_clean: pd.DataFrame, reviews_clean: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Enrich reviews with product dims (m:1)
    product_dim_cols = [
        "asin",
        "brand",
        "brand_norm",
        "seller",
        "seller_norm",
        "price_num",
        "list_price",
        "unit_price",
        "product_rating",
        "product_rating_count",
        "bsr_rank",
        "bsr_category_norm",
        "is_fba",
        "has_aplus",
        "image_count",
        "bullet_count",
        "best_subcat_rank",
        "best_subcat_name",
        "bought_count_number_clean",
    ]
    fact_review_enriched = reviews_clean.merge(products_clean[product_dim_cols], on="asin", how="left", validate="m:1")

    # asin-level agg
    agg_product = (
        fact_review_enriched.groupby("asin", as_index=False)
        .agg(
            sample_review_n=("review_id", "count"),
            avg_review_rating_sample=("review_rating", lambda x: pd.to_numeric(x, errors="coerce").mean()),
            review_rating_std_sample=("review_rating", lambda x: pd.to_numeric(x, errors="coerce").std()),
            verified_ratio=("verified_purchase", "mean"),
            helpful_sum=("helpful_votes", "sum"),
            helpful_mean=("helpful_votes", "mean"),
            has_text_ratio=("has_text", "mean"),
            avg_text_len=("text_len", "mean"),
            first_review_date=("review_date_dt", "min"),
            last_review_date=("review_date_dt", "max"),
        )
    )

    # Scraping-cap flags（采样偏差控制项）
    cap_counts = fact_review_enriched.groupby("asin")["review_id"].count()
    agg_product = agg_product.merge(cap_counts.rename("sample_review_n_check"), on="asin", how="left")
    agg_product["scrape_cap_10_flag"] = (agg_product["sample_review_n_check"] == 10).astype(int)
    agg_product["scrape_cap_100_flag"] = (agg_product["sample_review_n_check"] == 100).astype(int)
    agg_product = agg_product.drop(columns=["sample_review_n_check"])

    # Bring a few product fields into agg
    agg_product = agg_product.merge(
        products_clean[
            [
                "asin",
                "brand",
                "brand_norm",
                "price_num",
                "bsr_rank",
                "is_fba",
                "has_aplus",
                "image_count",
                "best_subcat_rank",
                "best_subcat_name",
                "bought_count_number_clean",
            ]
        ],
        on="asin",
        how="left",
    )

    # asin-week agg
    tmp = fact_review_enriched.copy()
    # W-SUN -> week starts Monday (period.start_time)
    tmp["week_start"] = tmp["review_date_dt"].dt.to_period("W-SUN").apply(lambda p: p.start_time if pd.notna(p) else pd.NaT)

    agg_product_week = (
        tmp.dropna(subset=["week_start"])
        .groupby(["asin", "week_start"], as_index=False)
        .agg(
            week_review_n=("review_id", "count"),
            week_avg_review_rating=("review_rating", lambda x: pd.to_numeric(x, errors="coerce").mean()),
            week_verified_ratio=("verified_purchase", "mean"),
            week_helpful_sum=("helpful_votes", "sum"),
            week_avg_text_len=("text_len", "mean"),
        )
    )

    return fact_review_enriched, agg_product, agg_product_week


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess products and reviews tables.")
    parser.add_argument("--products", type=str, default="merged_products_unique_asin.csv", help="Path to products CSV")
    parser.add_argument("--reviews", type=str, default="merged_reviews_unique_review_id.csv", help="Path to reviews CSV")
    parser.add_argument("--out_dir", type=str, default="preprocessed", help="Output directory")
    args = parser.parse_args()

    products_path = Path(args.products)
    reviews_path = Path(args.reviews)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    products_clean, product_images, product_subcat_ranks = preprocess_products(products_path)
    reviews_clean = preprocess_reviews(reviews_path)
    fact_review_enriched, agg_product, agg_product_week = build_review_enriched_and_aggs(products_clean, reviews_clean)

    # Save
    products_clean.to_csv(out_dir / "products_clean.csv", index=False, encoding="utf-8-sig")
    product_images.to_csv(out_dir / "product_images.csv", index=False, encoding="utf-8-sig")
    product_subcat_ranks.to_csv(out_dir / "product_subcat_ranks.csv", index=False, encoding="utf-8-sig")

    reviews_clean.to_csv(out_dir / "reviews_clean.csv", index=False, encoding="utf-8-sig")
    fact_review_enriched.to_csv(out_dir / "fact_review_enriched.csv", index=False, encoding="utf-8-sig")
    agg_product.to_csv(out_dir / "agg_product.csv", index=False, encoding="utf-8-sig")
    agg_product_week.to_csv(out_dir / "agg_product_week.csv", index=False, encoding="utf-8-sig")

    print("Done. Outputs written to:", out_dir.resolve())


if __name__ == "__main__":
    main()
