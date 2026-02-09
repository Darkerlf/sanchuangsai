import pandas as pd

# 读取表
products = pd.read_csv('merged_products_unique_asin.csv')
reviews = pd.read_csv('merged_reviews_unique_review_id.csv')

# 评论聚合统计
review_stats = reviews.groupby('asin').agg(
    review_count=('review_id', 'count'),
    avg_rating=('rating', 'mean'),
    verified_rate=('verified_purchase', 'mean'),
    positive_rate=('rating', lambda x: (x >= 4).mean()),  # 4-5星为正面
    negative_rate=('rating', lambda x: (x <= 2).mean())   # 1-2星为负面
).reset_index()

# 合并到商品表
master_df = products.merge(review_stats, on='asin', how='left')

# 填充无评论 商品的统计字段
fill_cols = ['review_count', 'avg_rating', 'verified_rate', 'positive_rate', 'negative_rate']
master_df[fill_cols] = master_df[fill_cols].fillna(0)

# 价格清洗（关键修复行）
master_df['price_num'] = master_df['price'].str.replace(r'[\$,]', '', regex=True).astype(float)

# 额外实用字段（推荐添加）
master_df['brand_lower'] = master_df['brand'].str.lower().str.strip()  # 品牌标准化，便于分组

# 保存主表
master_df.to_csv('master_analysis_table.csv', index=False)
print(f"主分析表生成完成，共 {len(master_df)} 行商品")
print(f"有关评论的高热度商品：{master_df['review_count'].gt(0).sum()} 个")