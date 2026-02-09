#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate competition-ready visuals (painpoint storyline).

Outputs (PNG + SVG):
  1) painpoint_sentiment_overview
  2) painpoint_negative_rate_by_price_tier
  3) painpoint_negative_rate_by_knife_type
  4) painpoint_negative_rate_by_material
  5) brand_painpoint_profile_top5
  6) apriori_painpoint_rules_top
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT
KG_DIR = ROOT / "kg_export"
ASSOC_DIR = ROOT / "assoc_export"
REPORTS_DIR = ROOT / "reports"
VIS_DIR = ROOT / "visuals"


PAINPOINT_LABELS = {
    "sharpness": "Sharpness",
    "rust_resistance": "Rust Resistance",
    "durability": "Durability",
    "handle_comfort": "Handle Comfort",
    "balance_weight": "Balance & Weight",
    "appearance_finish": "Appearance & Finish",
    "overall_quality": "Overall Quality",
    "value_for_money": "Value for Money",
}

PAINPOINT_ORDER = [
    "sharpness",
    "rust_resistance",
    "durability",
    "handle_comfort",
    "balance_weight",
    "appearance_finish",
    "overall_quality",
    "value_for_money",
]

PRICE_TIER_LABELS = {
    0: "$0-30",
    1: "$30-80",
    2: "$80-200",
    3: "$200+",
}


def read_csv_smart(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(f"Failed to read CSV: {path}")


def ensure_dirs() -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)


def fig_base(title: str) -> tuple[plt.Figure, plt.Axes]:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 200
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_title(title, fontsize=18, pad=14, weight="bold")
    return fig, ax


def save_fig(fig: plt.Figure, name: str) -> None:
    png = VIS_DIR / f"{name}.png"
    svg = VIS_DIR / f"{name}.svg"
    fig.tight_layout()
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)


def normalize_painpoint_label(series: pd.Series) -> pd.Series:
    return series.map(lambda x: PAINPOINT_LABELS.get(str(x), str(x)))


def price_tier(series: pd.Series) -> pd.Series:
    bins = [0, 30, 80, 200, np.inf]
    labels = [0, 1, 2, 3]
    return pd.cut(series, bins=bins, labels=labels, right=False, include_lowest=True).astype("float")


def format_item(item: str) -> str:
    if item.startswith("painpoint="):
        val = item.split("=", 1)[1]
        if val.endswith("_positive"):
            base = val.replace("_positive", "")
            return f"{PAINPOINT_LABELS.get(base, base)} (+)"
        if val.endswith("_negative"):
            base = val.replace("_negative", "")
            return f"{PAINPOINT_LABELS.get(base, base)} (-)"
        return PAINPOINT_LABELS.get(val, val)
    if item.startswith("material="):
        return f"Material: {item.split('=', 1)[1].replace('_', ' ').title()}"
    if item.startswith("knife_type="):
        return f"Knife: {item.split('=', 1)[1].replace('_', ' ').title()}"
    if item.startswith("price_tier="):
        tier = int(item.split("=", 1)[1])
        return f"Price {PRICE_TIER_LABELS.get(tier, tier)}"
    if item.startswith("rating"):
        return item.replace("rating", "Rating ")
    if item.startswith("sales"):
        return item.replace("sales", "Sales ")
    if item == "is_fba=1":
        return "FBA=Yes"
    if item == "has_aplus=1":
        return "A+ Content=Yes"
    return item


def fig_painpoint_sentiment_overview(painpoint: pd.DataFrame) -> None:
    df = painpoint.copy()
    df["painpoint_label"] = normalize_painpoint_label(df["painpoint_norm"])
    agg = (
        df.groupby("painpoint_norm", as_index=False)
        .agg(pos_n=("pos_n", "sum"), neg_n=("neg_n", "sum"))
    )
    agg["painpoint_label"] = normalize_painpoint_label(agg["painpoint_norm"])
    agg["order"] = agg["painpoint_norm"].apply(lambda x: PAINPOINT_ORDER.index(x) if x in PAINPOINT_ORDER else 999)
    agg = agg.sort_values("order")

    fig, ax = fig_base("Painpoint Sentiment Overview")
    ax.bar(agg["painpoint_label"], agg["pos_n"], label="Positive", color="#4C78A8")
    ax.bar(agg["painpoint_label"], agg["neg_n"], bottom=agg["pos_n"], label="Negative", color="#E45756")
    ax.set_ylabel("Mentions")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="upper right")
    save_fig(fig, "painpoint_sentiment_overview")


def fig_negative_rate_by_price_tier(painpoint: pd.DataFrame, products: pd.DataFrame) -> None:
    df = painpoint.merge(products[["asin", "price_num"]], left_on="asin", right_on="asin", how="left")
    df["price_num"] = pd.to_numeric(df["price_num"], errors="coerce")
    df["price_tier"] = price_tier(df["price_num"])
    df = df[df["price_tier"].notna()].copy()
    df["price_tier_label"] = df["price_tier"].map(lambda x: PRICE_TIER_LABELS.get(int(x), str(x)))
    df["painpoint_label"] = normalize_painpoint_label(df["painpoint_norm"])

    agg = (
        df.groupby(["painpoint_norm", "price_tier_label"], as_index=False)
        .agg(neg_ratio=("neg_ratio", "mean"))
    )
    agg["painpoint_label"] = normalize_painpoint_label(agg["painpoint_norm"])
    agg["order"] = agg["painpoint_norm"].apply(lambda x: PAINPOINT_ORDER.index(x) if x in PAINPOINT_ORDER else 999)
    agg = agg.sort_values("order")

    fig, ax = fig_base("Negative Rate by Price Tier")
    sns.barplot(
        data=agg,
        x="painpoint_label",
        y="neg_ratio",
        hue="price_tier_label",
        palette="Blues",
        ax=ax,
    )
    ax.set_ylabel("Average Negative Rate")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Price Tier", loc="upper right")
    save_fig(fig, "painpoint_negative_rate_by_price_tier")


def fig_negative_rate_by_knife_type(
    painpoint: pd.DataFrame,
    edge_knife: pd.DataFrame,
) -> None:
    df = painpoint.merge(edge_knife, left_on="asin", right_on="product_id", how="inner")
    if df.empty:
        return
    coverage = df.groupby("knife_type_id")["product_id"].nunique().sort_values(ascending=False)
    top_types = list(coverage.head(6).index)
    df = df[df["knife_type_id"].isin(top_types)].copy()
    agg = (
        df.groupby("knife_type_id", as_index=False)
        .agg(neg_ratio=("neg_ratio", "mean"), product_n=("product_id", "nunique"))
        .sort_values("product_n", ascending=False)
    )
    agg["knife_label"] = agg["knife_type_id"].str.replace("_", " ").str.title()

    fig, ax = fig_base("Negative Rate by Knife Type (Top 6 Coverage)")
    sns.barplot(
        data=agg,
        x="knife_label",
        y="neg_ratio",
        hue="knife_label",
        palette="Oranges",
        legend=False,
        ax=ax,
    )
    ax.set_ylabel("Average Negative Rate")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    save_fig(fig, "painpoint_negative_rate_by_knife_type")


def fig_negative_rate_by_material(
    painpoint: pd.DataFrame,
    edge_material: pd.DataFrame,
) -> None:
    df = painpoint.merge(edge_material, left_on="asin", right_on="product_id", how="inner")
    if df.empty:
        return
    coverage = df.groupby("material_id")["product_id"].nunique().sort_values(ascending=False)
    top_materials = list(coverage.head(6).index)
    df = df[df["material_id"].isin(top_materials)].copy()
    agg = (
        df.groupby("material_id", as_index=False)
        .agg(neg_ratio=("neg_ratio", "mean"), product_n=("product_id", "nunique"))
        .sort_values("product_n", ascending=False)
    )
    agg["material_label"] = agg["material_id"].str.replace("_", " ").str.upper()

    fig, ax = fig_base("Negative Rate by Material (Top 6 Coverage)")
    sns.barplot(
        data=agg,
        x="material_label",
        y="neg_ratio",
        hue="material_label",
        palette="Reds",
        legend=False,
        ax=ax,
    )
    ax.set_ylabel("Average Negative Rate")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    save_fig(fig, "painpoint_negative_rate_by_material")


def fig_brand_painpoint_radar(
    painpoint: pd.DataFrame,
    products: pd.DataFrame,
) -> None:
    df = painpoint.merge(products[["asin", "brand_norm"]], on="asin", how="left")
    df = df.dropna(subset=["brand_norm"])
    brand_counts = products["brand_norm"].value_counts().head(5)
    top_brands = list(brand_counts.index)
    df = df[df["brand_norm"].isin(top_brands)].copy()

    pivot = (
        df.groupby(["brand_norm", "painpoint_norm"], as_index=False)
        .agg(neg_ratio=("neg_ratio", "mean"))
    )
    pivot["painpoint_norm"] = pd.Categorical(pivot["painpoint_norm"], categories=PAINPOINT_ORDER, ordered=True)
    pivot = pivot.sort_values("painpoint_norm")

    labels = [PAINPOINT_LABELS.get(p, p) for p in PAINPOINT_ORDER]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(16, 9))
    ax = plt.subplot(111, polar=True)
    ax.set_title("Brand Painpoint Profile (Top 5 Brands)", fontsize=18, pad=18, weight="bold")

    palette = sns.color_palette("tab10", n_colors=len(top_brands))
    for idx, brand in enumerate(top_brands):
        values = (
            pivot[pivot["brand_norm"] == brand]
            .set_index("painpoint_norm")
            .reindex(PAINPOINT_ORDER)["neg_ratio"]
            .fillna(0.0)
            .tolist()
        )
        values += values[:1]
        ax.plot(angles, values, color=palette[idx], linewidth=2, label=str(brand).title())
        ax.fill(angles, values, color=palette[idx], alpha=0.08)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    save_fig(fig, "brand_painpoint_profile_top5")


def fig_apriori_rules_top(rules: pd.DataFrame) -> None:
    df = rules.copy()
    if "consequent" not in df.columns:
        return
    df = df[df["consequent"].astype(str).str.contains("painpoint=")].copy()
    if df.empty:
        return
    df["lift"] = pd.to_numeric(df["lift"], errors="coerce")
    df = df.dropna(subset=["lift"]).sort_values("lift", ascending=False).head(10)

    def label_row(r: pd.Series) -> str:
        ante = str(r["antecedent"]).split("|") if r["antecedent"] else []
        cons = str(r["consequent"]).split("|") if r["consequent"] else []
        ante_fmt = " & ".join(format_item(a) for a in ante if a)
        cons_fmt = " & ".join(format_item(c) for c in cons if c)
        return f"{ante_fmt}  →  {cons_fmt}"

    df["rule_label"] = df.apply(label_row, axis=1)
    df = df.sort_values("lift", ascending=True)

    fig, ax = fig_base("Apriori Rules (Top 10 by Lift)")
    ax.barh(df["rule_label"], df["lift"], color="#4C78A8")
    ax.set_xlabel("Lift")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=9)
    save_fig(fig, "apriori_painpoint_rules_top")


def main() -> int:
    ensure_dirs()

    painpoint_path = REPORTS_DIR / "product_painpoint_agg.csv"
    products_path = Path("预测建模/data/products_clean.csv")
    knife_edge_path = KG_DIR / "edge_product_has_knife_type.csv"
    material_edge_path = KG_DIR / "edge_product_has_material.csv"
    rules_path = ASSOC_DIR / "apriori_rules.csv"

    painpoint = read_csv_smart(painpoint_path)
    products = read_csv_smart(products_path)
    knife_edges = read_csv_smart(knife_edge_path)
    material_edges = read_csv_smart(material_edge_path)
    rules = read_csv_smart(rules_path)

    # normalize columns
    painpoint["asin"] = painpoint["asin"].astype(str)
    products["asin"] = products["asin"].astype(str)
    knife_edges["product_id"] = knife_edges["product_id"].astype(str)
    material_edges["product_id"] = material_edges["product_id"].astype(str)

    sns.set_style("whitegrid")

    fig_painpoint_sentiment_overview(painpoint)
    fig_negative_rate_by_price_tier(painpoint, products)
    fig_negative_rate_by_knife_type(painpoint, knife_edges)
    fig_negative_rate_by_material(painpoint, material_edges)
    fig_brand_painpoint_radar(painpoint, products)
    fig_apriori_rules_top(rules)

    print("Visuals generated.")
    print(f"- Output dir: {VIS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
