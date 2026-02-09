import pandas as pd
import re

# 1. 加载数据
df_reviews = pd.read_csv('preprocessed/reviews_clean.csv')
print(f"原始数据行数: {len(df_reviews)}")


# --- 定义清洗函数 ---

def clean_video_garbage(text):
    """
    移除评论中常见的视频播放器垃圾文本。
    这类文本通常以 'The video showcases...' 开头，以 'This is a modal window.' 结尾。
    """
    if not isinstance(text, str):
        return text

    # 特征字符串：这是垃圾文本块的结尾标志
    garbage_marker = "This is a modal window."

    if garbage_marker in text:
        # 如果包含该标志，则只保留标志之后的内容（即真正的评论）
        parts = text.split(garbage_marker)
        # 取最后一部分并去除首尾空格
        cleaned_text = parts[-1].strip()
        return cleaned_text
    return text


# 定义常用停用词列表 (因为环境中可能没有下载 NLTK 数据)
STOP_WORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
    "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
    'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
    'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
    "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
    'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
])


def remove_stopwords(text):
    """移除英文停用词"""
    if not isinstance(text, str):
        return text
    # 转换为小写并分词
    words = re.findall(r'\b\w+\b', text.lower())
    # 过滤停用词
    filtered_words = [w for w in words if w not in STOP_WORDS]
    return " ".join(filtered_words)


# --- 执行清洗 ---

# 1. 清除视频垃圾文本
# 创建新列 'review_text_cleaned_v2' 存储最终清洗结果
df_reviews['review_text_cleaned_v2'] = df_reviews['review_text_clean'].apply(clean_video_garbage)

# 2. (可选) 移除停用词
# 如果您需要用于词云或主题建模，建议使用此列
df_reviews['review_text_no_stopwords'] = df_reviews['review_text_cleaned_v2'].apply(remove_stopwords)

# --- 验证结果 ---

# 检查是否还有垃圾文本残留
garbage_phrases = ["video showcases", "Video Player", "liveRemaining", "play video"]
print("\n--- 垃圾文本残留检查 ---")
for phrase in garbage_phrases:
    count = df_reviews['review_text_cleaned_v2'].str.contains(phrase, case=False, regex=False).sum()
    print(f"包含 '{phrase}': {count} 行")

# 查看清洗前后的对比
print("\n--- 清洗前后对比 (示例) ---")
sample_idx = df_reviews[df_reviews['review_text_clean'].str.contains("video showcases", na=False)].index[0]
print(f"原始文本 (前100字符): {df_reviews.loc[sample_idx, 'review_text_clean'][:100]}...")
print(f"清洗后文本 (前100字符): {df_reviews.loc[sample_idx, 'review_text_cleaned_v2'][:100]}...")
print(f"去停用词后 (前100字符): {df_reviews.loc[sample_idx, 'review_text_no_stopwords'][:100]}...")

# --- 保存结果 ---
df_reviews.to_csv('reviews_cleaned_v2.csv', index=False)
print("\n已保存清洗后的文件: reviews_cleaned_v2.csv")