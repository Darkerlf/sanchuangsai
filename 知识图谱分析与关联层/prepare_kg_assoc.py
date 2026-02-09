#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三创赛：厨刀类目「知识图谱 + 关联分析」数据准备脚本

输入（默认）：
  - 预测建模/data/products_clean.csv
  - 预测建模/data/reviews_cleaned.csv
  - 预测建模/data/absa_detailed.csv
  - （可选）amazon_spasers_v2.0/data/amazon_data.db 的 search_results 表

输出（默认在本脚本同目录）：
  - dict/（词典与映射）
  - kg_export/（Neo4j 导入 CSV：节点表 + 边表）
  - assoc_export/（Apriori 输入 CSV）
  - reports/（数据质量与覆盖率摘要）
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_PAINPOINTS = [
    ("sharpness", "锋利/切割顺滑"),
    ("rust_resistance", "防锈/抗腐蚀"),
    ("durability", "耐用/不崩口"),
    ("handle_comfort", "握持/手柄舒适"),
    ("balance_weight", "平衡/重量手感"),
    ("appearance_finish", "外观/做工细节"),
    ("overall_quality", "总体质量"),
    ("value_for_money", "性价比"),
]


@dataclass(frozen=True)
class LexiconRule:
    pattern: str
    norm: str
    display_cn: str
    priority: int = 100

    def compile(self) -> re.Pattern[str]:
        return re.compile(self.pattern, flags=re.IGNORECASE)


def _read_csv_smart(path: Path) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"Failed to read CSV: {path} ({last_err})")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_bool_int(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    # common string/bool mixes
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["1", "true", "yes", "y", "on"]).astype(int)


def _safe_str(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x)


def _normalize_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _bucket_price_tier(price_num: pd.Series) -> pd.Series:
    # [0,30), [30,80), [80,200), [200, inf)
    bins = [0, 30, 80, 200, np.inf]
    labels = [0, 1, 2, 3]
    return pd.cut(price_num, bins=bins, labels=labels, right=False, include_lowest=True).astype("float")


def _bucket_rating(rating: pd.Series) -> pd.Series:
    def one(x: object) -> str:
        if x is None:
            return ""
        try:
            v = float(x)
        except Exception:  # noqa: BLE001
            return ""
        if math.isnan(v):
            return ""
        if v >= 4.5:
            return "rating>=4.5"
        if v >= 4.0:
            return "rating=4.0-4.5"
        return "rating<4.0"

    return rating.apply(one)


def _bucket_sales(sales: pd.Series) -> pd.Series:
    def one(x: object) -> str:
        if x is None:
            return ""
        try:
            v = float(x)
        except Exception:  # noqa: BLE001
            return ""
        if math.isnan(v):
            return ""
        if v < 100:
            return "sales<100"
        if v < 1000:
            return "sales=100-999"
        return "sales>=1000"

    return sales.apply(one)


def load_lexicon(path: Path, kind: str) -> list[LexiconRule]:
    df = _read_csv_smart(path)
    required = {"pattern", f"{kind}_norm", f"{kind}_display_cn"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Lexicon {path} missing columns: {sorted(missing)}")

    rules: list[LexiconRule] = []
    for _, row in df.iterrows():
        pattern = _safe_str(row["pattern"]).strip()
        norm = _safe_str(row[f"{kind}_norm"]).strip()
        disp = _safe_str(row[f"{kind}_display_cn"]).strip()
        if not pattern or not norm:
            continue
        priority = int(row["priority"]) if "priority" in df.columns and _safe_str(row["priority"]).strip() else 100
        rules.append(LexiconRule(pattern=pattern, norm=norm, display_cn=disp or norm, priority=priority))

    rules.sort(key=lambda r: (r.priority, r.norm))
    return rules


def extract_entities(
    products: pd.DataFrame,
    *,
    product_id_col: str,
    text_cols_in_order: list[str],
    rules: list[LexiconRule],
    entity_kind: str,
) -> pd.DataFrame:
    compiled = [(r, r.compile()) for r in rules]
    rows: list[dict] = []
    conf_by_source = {
        text_cols_in_order[0]: 1.0,
        text_cols_in_order[1] if len(text_cols_in_order) > 1 else "": 0.8,
        text_cols_in_order[2] if len(text_cols_in_order) > 2 else "": 0.6,
    }

    for _, prod in products.iterrows():
        pid = _safe_str(prod.get(product_id_col))
        if not pid:
            continue

        seen_norms: set[str] = set()
        for col in text_cols_in_order:
            text = _normalize_text(_safe_str(prod.get(col)))
            if not text:
                continue
            for rule, cre in compiled:
                if rule.norm in seen_norms:
                    continue
                if cre.search(text):
                    rows.append(
                        {
                            "product_id": pid,
                            f"{entity_kind}_id": rule.norm,
                            "source": col,
                            "confidence": conf_by_source.get(col, 0.5),
                        }
                    )
                    seen_norms.add(rule.norm)
        # done

    return pd.DataFrame(rows).drop_duplicates()


def build_painpoint_map(path: Path) -> pd.DataFrame:
    df = _read_csv_smart(path)
    if not {"aspect", "painpoint_norm"} <= set(df.columns):
        raise RuntimeError(f"painpoint_map must include columns: aspect, painpoint_norm ({path})")
    return df[["aspect", "painpoint_norm"]].copy()


def build_product_painpoint_agg(
    reviews: pd.DataFrame,
    absa: pd.DataFrame,
    painpoint_map: pd.DataFrame,
) -> pd.DataFrame:
    # reviews: review_id -> asin
    r = reviews[["review_id", "asin"]].copy()
    a = absa[["review_id", "aspect", "sentiment", "score"]].copy()

    a = a.merge(painpoint_map, on="aspect", how="left")
    a = a.merge(r, on="review_id", how="inner")
    a = a.dropna(subset=["painpoint_norm", "asin"])

    a["is_pos"] = a["sentiment"].astype(str).str.lower().isin(["positive", "pos", "1", "true"]).astype(int)
    a["is_neg"] = a["sentiment"].astype(str).str.lower().isin(["negative", "neg", "-1", "false"]).astype(int)

    g = a.groupby(["asin", "painpoint_norm"], as_index=False).agg(
        mention_n=("review_id", "count"),
        pos_n=("is_pos", "sum"),
        neg_n=("is_neg", "sum"),
        avg_score=("score", "mean"),
    )
    g["neg_ratio"] = g["neg_n"] / g["mention_n"].replace(0, np.nan)
    g["pos_ratio"] = g["pos_n"] / g["mention_n"].replace(0, np.nan)
    return g


def load_search_results_from_sqlite(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(str(db_path)) as conn:
            df = pd.read_sql_query(
                """
                SELECT
                    asin,
                    brand,
                    search_keyword,
                    search_rank,
                    is_sponsored,
                    is_amazon_choice,
                    is_best_seller
                FROM search_results
                """,
                conn,
            )
        return df
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _keyword_norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def build_brand_nodes(products: pd.DataFrame) -> pd.DataFrame:
    if "brand_norm" not in products.columns:
        raise RuntimeError("products_clean.csv must include brand_norm")
    brands = products[["brand_norm"]].copy()
    brands = brands.dropna()
    brands["brand_norm"] = brands["brand_norm"].astype(str).str.strip()
    brands = brands[brands["brand_norm"] != ""]

    if "brand" in products.columns:
        tmp = products[["brand_norm", "brand"]].copy()
        tmp["brand"] = tmp["brand"].astype(str).str.strip()
        tmp = tmp[tmp["brand"] != ""]
        # most frequent display per norm
        display = (
            tmp.groupby(["brand_norm", "brand"])
            .size()
            .reset_index(name="n")
            .sort_values(["brand_norm", "n"], ascending=[True, False])
            .drop_duplicates("brand_norm")[["brand_norm", "brand"]]
            .rename(columns={"brand": "brand_display"})
        )
    else:
        display = pd.DataFrame({"brand_norm": brands["brand_norm"].unique(), "brand_display": brands["brand_norm"]})

    out = brands.drop_duplicates("brand_norm").merge(display, on="brand_norm", how="left")
    return out.rename(columns={"brand_norm": "brand_id"})[["brand_id", "brand_display"]]


def build_product_nodes(products: pd.DataFrame) -> pd.DataFrame:
    required = [
        "asin",
        "title",
        "price_num",
        "discount_rate",
        "product_rating",
        "product_rating_count",
        "bsr_category_norm",
        "best_subcat_name",
        "best_subcat_rank",
        "bsr_rank",
        "is_fba",
        "has_aplus",
        "image_count",
        "bullet_count",
        "bought_count_number_clean",
    ]
    missing = [c for c in required if c not in products.columns]
    if missing:
        raise RuntimeError(f"products_clean.csv missing required columns: {missing}")

    df = products[required].copy()
    df = df.rename(columns={"asin": "product_id"})
    df["is_fba"] = _to_bool_int(df["is_fba"])
    df["has_aplus"] = _to_bool_int(df["has_aplus"])
    return df


def build_painpoint_nodes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"painpoint_id": pid, "painpoint_display_cn": disp}
            for pid, disp in DEFAULT_PAINPOINTS
        ]
    )


def build_competes_edges(
    products: pd.DataFrame,
    search_results: pd.DataFrame,
) -> pd.DataFrame:
    df = products.copy()
    df["price_tier"] = _bucket_price_tier(pd.to_numeric(df["price_num"], errors="coerce"))
    df["best_subcat_name"] = df["best_subcat_name"].fillna("")
    df["best_subcat_name"] = df["best_subcat_name"].astype(str).str.strip()
    df.loc[df["best_subcat_name"] == "", "best_subcat_name"] = df["bsr_category_norm"].fillna("unknown")

    seg = (
        df.groupby(["best_subcat_name", "price_tier", "brand_norm"], as_index=False)
        .agg(product_n=("asin", "nunique"), median_price=("price_num", "median"))
    )
    seg = seg[seg["product_n"] >= 3].copy()

    pair_stats: dict[tuple[str, str], dict] = {}
    for (subcat, tier), g in seg.groupby(["best_subcat_name", "price_tier"]):
        brands = g.sort_values("brand_norm")
        brand_rows = list(brands.to_dict(orient="records"))
        for i in range(len(brand_rows)):
            for j in range(i + 1, len(brand_rows)):
                b1 = str(brand_rows[i]["brand_norm"])
                b2 = str(brand_rows[j]["brand_norm"])
                key = (b1, b2)
                st = pair_stats.setdefault(key, {"shared_subcat_n": 0, "shared_keyword_n": 0, "price_gaps": []})
                st["shared_subcat_n"] += 1
                p1 = brand_rows[i]["median_price"]
                p2 = brand_rows[j]["median_price"]
                if pd.notna(p1) and pd.notna(p2):
                    st["price_gaps"].append(abs(float(p1) - float(p2)))

    if not search_results.empty:
        sr = search_results.copy()
        if "search_rank" in sr.columns:
            sr["search_rank"] = pd.to_numeric(sr["search_rank"], errors="coerce")
            sr = sr[sr["search_rank"].notna()]
            sr = sr[sr["search_rank"] <= 50]

        if {"search_keyword", "brand", "asin"} <= set(sr.columns):
            sr["keyword_norm"] = sr["search_keyword"].astype(str).apply(_keyword_norm)
            sr["brand_norm"] = sr["brand"].astype(str).str.strip().str.lower()
            kw = (
                sr.groupby(["keyword_norm", "brand_norm"], as_index=False)
                .agg(asin_n=("asin", "nunique"))
            )
            for keyword, g in kw.groupby("keyword_norm"):
                g = g.sort_values("brand_norm")
                rows = list(g.to_dict(orient="records"))
                for i in range(len(rows)):
                    for j in range(i + 1, len(rows)):
                        c = min(int(rows[i]["asin_n"]), int(rows[j]["asin_n"]))
                        if c < 3:
                            continue
                        b1 = str(rows[i]["brand_norm"])
                        b2 = str(rows[j]["brand_norm"])
                        key = (b1, b2) if b1 < b2 else (b2, b1)
                        st = pair_stats.setdefault(key, {"shared_subcat_n": 0, "shared_keyword_n": 0, "price_gaps": []})
                        st["shared_keyword_n"] += 1

    rows_out: list[dict] = []
    for (b1, b2), st in pair_stats.items():
        if st["shared_subcat_n"] <= 0 and st["shared_keyword_n"] <= 0:
            continue
        gaps = st["price_gaps"]
        avg_gap = float(np.mean(gaps)) if gaps else np.nan
        rows_out.append(
            {
                "brand_id_1": b1,
                "brand_id_2": b2,
                "shared_subcat_n": int(st["shared_subcat_n"]),
                "shared_keyword_n": int(st["shared_keyword_n"]),
                "avg_price_gap": avg_gap,
            }
        )
    return pd.DataFrame(rows_out).sort_values(["shared_subcat_n", "shared_keyword_n"], ascending=False)


def build_transactions(
    products: pd.DataFrame,
    product_material_edges: pd.DataFrame,
    product_knife_type_edges: pd.DataFrame,
    product_painpoint_agg: pd.DataFrame,
) -> pd.DataFrame:
    df = products[["asin", "price_num", "product_rating", "bought_count_number_clean", "is_fba", "has_aplus"]].copy()
    df = df.rename(columns={"asin": "transaction_id"})
    df["is_fba"] = _to_bool_int(df["is_fba"])
    df["has_aplus"] = _to_bool_int(df["has_aplus"])

    df["price_tier"] = _bucket_price_tier(pd.to_numeric(df["price_num"], errors="coerce"))
    df["rating_bucket"] = _bucket_rating(df["product_rating"])
    df["sales_bucket"] = _bucket_sales(df["bought_count_number_clean"])

    # entity lookups
    mats = (
        product_material_edges.groupby("product_id")["material_id"]
        .apply(lambda s: sorted(set(map(str, s))))
        .to_dict()
        if not product_material_edges.empty
        else {}
    )
    types = (
        product_knife_type_edges.groupby("product_id")["knife_type_id"]
        .apply(lambda s: sorted(set(map(str, s))))
        .to_dict()
        if not product_knife_type_edges.empty
        else {}
    )

    pp = product_painpoint_agg.copy()
    if not pp.empty:
        pp["neg_label"] = (pp["mention_n"] >= 5) & (pp["neg_ratio"] >= 0.30)
        pp["pos_label"] = (pp["mention_n"] >= 5) & (pp["pos_ratio"] >= 0.60)
        pp_labels = {}
        for asin, g in pp.groupby("asin"):
            items = []
            for _, r in g.iterrows():
                pid = str(r["painpoint_norm"])
                if bool(r["neg_label"]):
                    items.append(f"painpoint={pid}_negative")
                if bool(r["pos_label"]):
                    items.append(f"painpoint={pid}_positive")
            if items:
                pp_labels[str(asin)] = sorted(set(items))
    else:
        pp_labels = {}

    def items_for_row(r: pd.Series) -> str:
        tid = str(r["transaction_id"])
        items: set[str] = set()

        if pd.notna(r["price_tier"]):
            items.add(f"price_tier={int(r['price_tier'])}")

        if r["rating_bucket"]:
            items.add(str(r["rating_bucket"]))

        if r["sales_bucket"]:
            items.add(str(r["sales_bucket"]))

        if int(r["is_fba"]) == 1:
            items.add("is_fba=1")
        if int(r["has_aplus"]) == 1:
            items.add("has_aplus=1")

        for m in mats.get(tid, []):
            items.add(f"material={m}")
        for t in types.get(tid, []):
            items.add(f"knife_type={t}")
        for p in pp_labels.get(tid, []):
            items.add(p)

        return "|".join(sorted(items))

    df["items"] = df.apply(items_for_row, axis=1)
    return df[["transaction_id", "items"]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", type=str, default="预测建模/data/products_clean.csv")
    parser.add_argument("--reviews", type=str, default="预测建模/data/reviews_cleaned.csv")
    parser.add_argument("--absa", type=str, default="预测建模/data/absa_detailed.csv")
    parser.add_argument("--painpoint-map", type=str, default="知识图谱分析与关联层/dict/painpoint_map.csv")
    parser.add_argument("--material-lexicon", type=str, default="知识图谱分析与关联层/dict/material_lexicon.csv")
    parser.add_argument("--knife-type-lexicon", type=str, default="知识图谱分析与关联层/dict/knife_type_lexicon.csv")
    parser.add_argument("--search-db", type=str, default="amazon_spasers_v2.0/data/amazon_data.db")
    parser.add_argument("--out-dir", type=str, default="知识图谱分析与关联层")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    dict_dir = out_dir / "dict"
    kg_dir = out_dir / "kg_export"
    assoc_dir = out_dir / "assoc_export"
    reports_dir = out_dir / "reports"
    for d in (dict_dir, kg_dir, assoc_dir, reports_dir):
        _ensure_dir(d)

    products_path = Path(args.products)
    reviews_path = Path(args.reviews)
    absa_path = Path(args.absa)

    products = _read_csv_smart(products_path)
    reviews = _read_csv_smart(reviews_path)
    absa = _read_csv_smart(absa_path)

    # Data quality: uniqueness
    quality = {"inputs": {"products": str(products_path), "reviews": str(reviews_path), "absa": str(absa_path)}}
    quality["counts_raw"] = {"products": int(len(products)), "reviews": int(len(reviews)), "absa": int(len(absa))}

    if products["asin"].duplicated().any():
        dup_n = int(products["asin"].duplicated().sum())
        products = products.drop_duplicates(subset=["asin"], keep="first")
        quality["products_deduped"] = dup_n
    if reviews["review_id"].duplicated().any():
        dup_n = int(reviews["review_id"].duplicated().sum())
        reviews = reviews.drop_duplicates(subset=["review_id"], keep="last")
        quality["reviews_deduped"] = dup_n

    # Foreign keys: reviews.asin in products.asin
    reviews["asin"] = reviews["asin"].astype(str)
    products["asin"] = products["asin"].astype(str)
    missing_asin = set(reviews["asin"].unique()) - set(products["asin"].unique())
    quality["missing_review_asin_n"] = int(len(missing_asin))
    if missing_asin:
        reviews = reviews[~reviews["asin"].isin(list(missing_asin))].copy()

    # ABSA review_id in reviews.review_id
    absa["review_id"] = absa["review_id"].astype(str)
    reviews["review_id"] = reviews["review_id"].astype(str)
    missing_rid = set(absa["review_id"].unique()) - set(reviews["review_id"].unique())
    quality["missing_absa_review_id_n"] = int(len(missing_rid))
    if missing_rid:
        absa = absa[~absa["review_id"].isin(list(missing_rid))].copy()

    quality["counts_after_fk"] = {"products": int(len(products)), "reviews": int(len(reviews)), "absa": int(len(absa))}

    # Dictionaries
    painpoint_map = build_painpoint_map(Path(args.painpoint_map))
    material_rules = load_lexicon(Path(args.material_lexicon), kind="material")
    knife_rules = load_lexicon(Path(args.knife_type_lexicon), kind="knife_type")

    # Nodes
    node_brand = build_brand_nodes(products)
    node_product = build_product_nodes(products)
    node_painpoint = build_painpoint_nodes()

    # Edges: brand->product
    edge_brand_sells_product = products[["brand_norm", "asin"]].copy()
    edge_brand_sells_product = edge_brand_sells_product.rename(columns={"brand_norm": "brand_id", "asin": "product_id"})
    edge_brand_sells_product = edge_brand_sells_product.dropna()
    edge_brand_sells_product["brand_id"] = edge_brand_sells_product["brand_id"].astype(str).str.strip()
    edge_brand_sells_product = edge_brand_sells_product[edge_brand_sells_product["brand_id"] != ""].drop_duplicates()

    # Edges: product->material, product->knife_type
    text_cols = [c for c in ["title", "bullet_points_text", "description_clean"] if c in products.columns]
    if not text_cols:
        raise RuntimeError("products_clean.csv must include at least one of: title, bullet_points_text, description_clean")

    edge_product_has_material = extract_entities(
        products,
        product_id_col="asin",
        text_cols_in_order=text_cols,
        rules=material_rules,
        entity_kind="material",
    ).rename(columns={"material_id": "material_id"})
    edge_product_has_material = edge_product_has_material.rename(columns={"material_id": "material_id"})

    edge_product_has_knife_type = extract_entities(
        products,
        product_id_col="asin",
        text_cols_in_order=text_cols,
        rules=knife_rules,
        entity_kind="knife_type",
    ).rename(columns={"knife_type_id": "knife_type_id"})

    # Product-painpoint aggregation
    product_painpoint_agg = build_product_painpoint_agg(reviews, absa, painpoint_map)
    edge_product_has_painpoint = product_painpoint_agg.rename(columns={"asin": "product_id", "painpoint_norm": "painpoint_id"})

    # Nodes: material & knife_type only from what appears
    mat_ids = sorted(set(edge_product_has_material["material_id"].astype(str))) if not edge_product_has_material.empty else []
    type_ids = sorted(set(edge_product_has_knife_type["knife_type_id"].astype(str))) if not edge_product_has_knife_type.empty else []

    # lookup display_cn from lexicons
    mat_disp = {r.norm: r.display_cn for r in material_rules}
    type_disp = {r.norm: r.display_cn for r in knife_rules}
    node_material = pd.DataFrame([{"material_id": mid, "material_display_cn": mat_disp.get(mid, mid)} for mid in mat_ids])
    node_knife_type = pd.DataFrame([{"knife_type_id": tid, "knife_type_display_cn": type_disp.get(tid, tid)} for tid in type_ids])

    # Optional: search results
    search_df = load_search_results_from_sqlite(Path(args.search_db))
    if not search_df.empty:
        # 保证 keyword edges 不会指向不存在的 product 节点
        known_asins = set(products["asin"].astype(str).unique())
        search_df["asin"] = search_df["asin"].astype(str)
        search_df = search_df[search_df["asin"].isin(list(known_asins))].copy()

        search_df = search_df.dropna(subset=["asin", "search_keyword"])
        search_df["keyword_id"] = search_df["search_keyword"].astype(str).apply(_keyword_norm)
        node_keyword = (
            search_df[["keyword_id", "search_keyword"]]
            .drop_duplicates("keyword_id")
            .rename(columns={"search_keyword": "keyword_display"})
        )
        edge_keyword_ranks_product = search_df.rename(
            columns={
                "keyword_id": "keyword_id",
                "asin": "product_id",
            }
        )[
            ["keyword_id", "product_id", "search_rank", "is_sponsored", "is_best_seller", "is_amazon_choice"]
        ].drop_duplicates()
    else:
        node_keyword = pd.DataFrame()
        edge_keyword_ranks_product = pd.DataFrame()

    # Competitor edges
    edge_brand_competes_brand = build_competes_edges(products, search_df)

    # Transactions for Apriori
    tx = build_transactions(
        products,
        product_material_edges=edge_product_has_material.rename(columns={"product_id": "product_id"}),
        product_knife_type_edges=edge_product_has_knife_type.rename(columns={"product_id": "product_id"}),
        product_painpoint_agg=product_painpoint_agg,
    )

    # Coverage metrics
    total_products = int(products["asin"].nunique())
    quality["coverage"] = {
        "products_total": total_products,
        "has_painpoint_edge": int(edge_product_has_painpoint["product_id"].nunique()) if not edge_product_has_painpoint.empty else 0,
        "has_material_edge": int(edge_product_has_material["product_id"].nunique()) if not edge_product_has_material.empty else 0,
        "has_knife_type_edge": int(edge_product_has_knife_type["product_id"].nunique()) if not edge_product_has_knife_type.empty else 0,
        "has_keyword_edge": int(edge_keyword_ranks_product["product_id"].nunique()) if not edge_keyword_ranks_product.empty else 0,
    }

    # Write outputs (utf-8-sig for Excel friendliness)
    def w(df_: pd.DataFrame, path_: Path) -> None:
        df_.to_csv(path_, index=False, encoding="utf-8-sig")

    # nodes
    w(node_brand, kg_dir / "node_brand.csv")
    w(node_product, kg_dir / "node_product.csv")
    w(node_material, kg_dir / "node_material.csv")
    w(node_knife_type, kg_dir / "node_knife_type.csv")
    w(node_painpoint, kg_dir / "node_painpoint.csv")
    if not node_keyword.empty:
        w(node_keyword, kg_dir / "node_keyword.csv")

    # edges
    w(edge_brand_sells_product, kg_dir / "edge_brand_sells_product.csv")
    w(edge_product_has_material, kg_dir / "edge_product_has_material.csv")
    w(edge_product_has_knife_type, kg_dir / "edge_product_has_knife_type.csv")
    w(edge_product_has_painpoint, kg_dir / "edge_product_has_painpoint.csv")
    if not edge_keyword_ranks_product.empty:
        w(edge_keyword_ranks_product, kg_dir / "edge_keyword_ranks_product.csv")
    w(edge_brand_competes_brand, kg_dir / "edge_brand_competes_brand.csv")

    # assoc export
    w(tx, assoc_dir / "transactions_product.csv")

    # reports
    (reports_dir / "data_quality_summary.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports_dir / "painpoint_map_used.json").write_text(
        painpoint_map.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    # keep a copy with the original planned name for convenience
    w(product_painpoint_agg, reports_dir / "product_painpoint_agg.csv")

    print("Export completed.")
    print(f"- Neo4j CSV: {kg_dir}")
    print(f"- Apriori CSV: {assoc_dir}")
    print(f"- Reports: {reports_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
