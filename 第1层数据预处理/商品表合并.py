import pandas as pd
import os

# 文件路径（假设文件在当前工作目录下，如果不是请修改路径）
file_paths = [
    'products-1-28.csv',
    'products.csv',
    'products-1-10.csv'
]

# 读取所有CSV文件
dfs = []
for file in file_paths:
    if os.path.exists(file):
        df = pd.read_csv(file)
        dfs.append(df)
        print(f"成功读取 {file}，行数: {len(df)}")
    else:
        print(f"文件 {file} 不存在，请检查路径")

# 如果没有读取到任何文件，直接退出
if not dfs:
    raise FileNotFoundError("没有找到任何CSV文件")

# 合并所有DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

print(f"合并后总行数（含重复ASIN）: {len(combined_df)}")
print(f"唯一ASIN数量: {combined_df['asin'].nunique()}")

# 数据清洗与去重逻辑
# 1. 确保updated_at是日期时间类型（便于排序）
combined_df['updated_at'] = pd.to_datetime(combined_df['updated_at'], errors='coerce')

# 2. 按updated_at降序排序（最新的在前面）
combined_df = combined_df.sort_values('updated_at', ascending=False)

# 3. 按asin去重，保留updated_at最新的一条记录（keep='first'）
dedup_df = combined_df.drop_duplicates(subset='asin', keep='first').reset_index(drop=True)

print(f"去重后最终行数: {len(dedup_df)}")

# 可选：保存合并后的去重表
output_file = 'merged_products_unique_asin.csv'
dedup_df.to_csv(output_file, index=False)
print(f"去重后的表已保存至: {output_file}")

# 可选：显示前几行预览
print("\n去重后表格前5行预览:")
print(dedup_df[['asin', 'title', 'brand', 'price', 'rating', 'bsr_rank', 'updated_at']].head())