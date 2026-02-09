# 知识图谱与关联分析：数据准备（Neo4j + Apriori）

本目录用于把现有的结构化数据（`预测建模/data/*.csv` + 可选的爬虫 SQLite）整理成两套可直接使用的产物：

- `kg_export/`：Neo4j 导入 CSV（节点表 + 边表）
- `assoc_export/`：Apriori 输入 CSV（按 `asin` 生成交易 items）

## 依赖

- Python 3
- `pandas`, `numpy`

## 一键生成（推荐）

在 `e:\PycharmProjects` 下运行：

```powershell
powershell -NoProfile -Command "python .\\知识图谱分析与关联层\\prepare_kg_assoc.py"
```

## 生成 Apriori 规则（可直接用于展示）

```powershell
powershell -NoProfile -Command "python .\\知识图谱分析与关联层\\run_apriori_rules.py"
```

输出：

- `知识图谱分析与关联层/assoc_export/apriori_rules.csv`
- `知识图谱分析与关联层/reports/apriori_rules.md`

## 生成比赛展示图（6 张，PNG + SVG）

```powershell
powershell -NoProfile -Command "python .\\知识图谱分析与关联层\\generate_viz.py"
```

输出目录：

- `知识图谱分析与关联层/visuals/`

图表清单（同名 PNG+SVG）：

1. `painpoint_sentiment_overview`
2. `painpoint_negative_rate_by_price_tier`
3. `painpoint_negative_rate_by_knife_type`
4. `painpoint_negative_rate_by_material`
5. `brand_painpoint_profile_top5`
6. `apriori_painpoint_rules_top`

## Neo4j 导入参考

`知识图谱分析与关联层/neo4j_load_example.cypher` 提供了 `LOAD CSV` 的最小可用示例（需要把 `kg_export/*.csv` 放到 Neo4j 的 import 目录）。

## 输入与输出

默认输入：

- `预测建模/data/products_clean.csv`
- `预测建模/data/reviews_cleaned.csv`
- `预测建模/data/absa_detailed.csv`
- （可选）`amazon_spasers_v2.0/data/amazon_data.db` 的 `search_results` 表

默认输出：

- `知识图谱分析与关联层/kg_export/*.csv`
- `知识图谱分析与关联层/assoc_export/transactions_product.csv`
- `知识图谱分析与关联层/reports/data_quality_summary.json`

## 词典/映射（可编辑）

- `知识图谱分析与关联层/dict/painpoint_map.csv`
- `知识图谱分析与关联层/dict/material_lexicon.csv`
- `知识图谱分析与关联层/dict/knife_type_lexicon.csv`

词典更新后，重新运行脚本即可刷新导出结果。
