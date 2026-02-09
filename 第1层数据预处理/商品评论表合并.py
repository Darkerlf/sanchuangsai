import pandas as pd
import os

# 文件路径（假设文件在当前工作目录下，如果不是请修改路径）
file_paths = [
    'reviews-1-28.csv',
    'reviews-1-10.csv'
]

# 读取所有CSV文件
dfs = []
for file in file_paths:
    if os.path.exists(file):
        df = pd.read_csv(file)
        # 统一列结构：如果缺少 'brand' 列，补为空字符串
        if 'brand' not in df.columns:
            df['brand'] = ''
        dfs.append(df)
        print(f"成功读取 {file}，行数: {len(df)}，列数: {len(df.columns)}")
    else:
        print(f"文件 {file} 不存在，请检查路径")

# 如果没有读取到任何文件，直接退出
if not dfs:
    raise FileNotFoundError("没有找到任何CSV文件")

# 合并所有DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

print(f"\n合并后总行数（含潜在重复review_id）: {len(combined_df)}")
print(f"唯一review_id数量（合并前统计）: {combined_df['review_id'].nunique()}")

# 数据清洗与去重逻辑
# 1. 确保created_at是日期时间类型（用于优先保留最新采集的数据）
combined_df['created_at'] = pd.to_datetime(combined_df['created_at'], errors='coerce')

# 2. 按created_at降序排序（最新的采集记录排在前面）
combined_df = combined_df.sort_values('created_at', ascending=False)

# 3. 仅按review_id去重，保留created_at最新的一条（keep='first'）
dedup_df = combined_df.drop_duplicates(subset='review_id', keep='first').reset_index(drop=True)

print(f"去重后最终行数（review_id唯一）: {len(dedup_df)}")
print(f"去重后涉及的唯一asin数量: {dedup_df['asin'].nunique()}")

# 保存合并后的去重评论表
output_file = 'merged_reviews_unique_review_id.csv'
dedup_df.to_csv(output_file, index=False)
print(f"\n去重后的完整评论表已保存至: {output_file}")

# 显示前几行预览
print("\n去重后表格前5行预览:")
print(dedup_df[['asin', 'review_id', 'title', 'rating', 'date', 'author', 'created_at']].head())