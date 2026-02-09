#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 assoc_export/transactions_product.csv 生成 Apriori 关联规则（不依赖 mlxtend）。

默认策略：
  - 最大项集长度：3
  - 规则 consequent 限制为 1 项（更适合比赛展示）
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path


@dataclass(frozen=True)
class Rule:
    antecedent: tuple[str, ...]
    consequent: tuple[str, ...]
    support: float
    confidence: float
    lift: float
    antecedent_support: float
    consequent_support: float


def read_transactions(path: Path) -> list[set[str]]:
    txs: list[set[str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items_str = (row.get("items") or "").strip()
            items = {it for it in items_str.split("|") if it}
            if items:
                txs.append(items)
            else:
                txs.append(set())
    return txs


def supports_from_counts(counts: Counter[tuple[str, ...]], n: int) -> dict[tuple[str, ...], float]:
    return {k: v / n for k, v in counts.items()}


def mine_frequent_itemsets(
    txs: list[set[str]],
    *,
    min_support: float,
    max_len: int,
) -> dict[tuple[str, ...], float]:
    n = len(txs)
    if n == 0:
        return {}

    # L1
    c1: Counter[str] = Counter()
    for t in txs:
        c1.update(t)
    l1 = {tuple([k]): v / n for k, v in c1.items() if (v / n) >= min_support}
    frequent: dict[tuple[str, ...], float] = dict(l1)

    allowed_items = {k[0] for k in l1.keys()}
    if not allowed_items:
        return frequent

    # For k>=2: transaction-based counting using allowed items
    prev_level = {k for k in l1.keys()}
    for k in range(2, max_len + 1):
        ck: Counter[tuple[str, ...]] = Counter()
        for t in txs:
            items = sorted(i for i in t if i in allowed_items)
            if len(items) < k:
                continue
            for comb in combinations(items, k):
                # apriori pruning: all (k-1) subsets must be frequent in previous level
                if prev_level:
                    ok = True
                    for sub in combinations(comb, k - 1):
                        if sub not in prev_level:
                            ok = False
                            break
                    if not ok:
                        continue
                ck[comb] += 1

        lk = {itemset: cnt / n for itemset, cnt in ck.items() if (cnt / n) >= min_support}
        if not lk:
            break
        frequent.update(lk)
        prev_level = set(lk.keys())

    return frequent


def generate_rules(
    frequent: dict[tuple[str, ...], float],
    *,
    min_confidence: float,
    consequent_size: int,
) -> list[Rule]:
    rules: list[Rule] = []

    # For quick lookup
    support = frequent

    for itemset, s_itemset in frequent.items():
        if len(itemset) < 2:
            continue

        items = tuple(itemset)
        # consequents of fixed size
        for cons in combinations(items, consequent_size):
            cons = tuple(cons)
            ante = tuple(sorted(set(items) - set(cons)))
            if not ante:
                continue
            s_ante = support.get(tuple(sorted(ante)))
            s_cons = support.get(tuple(sorted(cons)))
            if not s_ante or not s_cons:
                continue
            conf = s_itemset / s_ante
            if conf < min_confidence:
                continue
            lift = conf / s_cons if s_cons > 0 else math.inf
            rules.append(
                Rule(
                    antecedent=tuple(sorted(ante)),
                    consequent=tuple(sorted(cons)),
                    support=s_itemset,
                    confidence=conf,
                    lift=lift,
                    antecedent_support=s_ante,
                    consequent_support=s_cons,
                )
            )

    rules.sort(key=lambda r: (r.lift, r.confidence, r.support), reverse=True)
    return rules


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transactions", type=str, default="知识图谱分析与关联层/assoc_export/transactions_product.csv")
    parser.add_argument("--out-csv", type=str, default="知识图谱分析与关联层/assoc_export/apriori_rules.csv")
    parser.add_argument("--out-md", type=str, default="知识图谱分析与关联层/reports/apriori_rules.md")
    parser.add_argument("--min-support", type=float, default=0.02)
    parser.add_argument("--min-confidence", type=float, default=0.30)
    parser.add_argument("--max-len", type=int, default=3)
    parser.add_argument("--consequent-size", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    tx_path = Path(args.transactions)
    txs = read_transactions(tx_path)
    n = len(txs)

    frequent = mine_frequent_itemsets(txs, min_support=args.min_support, max_len=args.max_len)
    rules = generate_rules(
        frequent,
        min_confidence=args.min_confidence,
        consequent_size=args.consequent_size,
    )[: args.top_k]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "antecedent",
                "consequent",
                "support",
                "confidence",
                "lift",
                "antecedent_support",
                "consequent_support",
            ],
        )
        writer.writeheader()
        for r in rules:
            writer.writerow(
                {
                    "antecedent": "|".join(r.antecedent),
                    "consequent": "|".join(r.consequent),
                    "support": r.support,
                    "confidence": r.confidence,
                    "lift": r.lift,
                    "antecedent_support": r.antecedent_support,
                    "consequent_support": r.consequent_support,
                }
            )

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w", encoding="utf-8", newline="") as f:
        f.write("# Apriori 关联规则（Top）\n\n")
        f.write(f"- Transactions: {n}\n")
        f.write(f"- Frequent itemsets: {len(frequent)} (min_support={args.min_support}, max_len={args.max_len})\n")
        f.write(f"- Rules saved: {len(rules)} (min_confidence={args.min_confidence})\n\n")
        f.write("| # | Antecedent | Consequent | Support | Confidence | Lift |\n")
        f.write("|---:|---|---|---:|---:|---:|\n")
        for i, r in enumerate(rules, start=1):
            f.write(
                f"| {i} | {'; '.join(r.antecedent)} | {'; '.join(r.consequent)} | "
                f"{r.support:.4f} | {r.confidence:.4f} | {r.lift:.4f} |\n"
            )

    print("Apriori rules exported.")
    print(f"- CSV: {out_csv}")
    print(f"- MD:  {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

