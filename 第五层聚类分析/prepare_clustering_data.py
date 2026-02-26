"""
厨刀市场聚类分析 - 数据准备模块 (更新版)
=========================================
适配 products_clean.csv 字段结构
"""

import pandas as pd
import numpy as np
import re
import os
import warnings
from typing import Dict, List, Optional, Tuple
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# 配置区：字段映射
# ============================================================================

# 字段名映射（实际字段名 → 标准字段名）
FIELD_MAPPING = {
    # 基础字段
    'asin': 'asin',
    'title': 'product_title',  # 更新
    'brand': 'brand',
    'brand_norm': 'brand_norm',

    # 价格相关
    'price_num': 'price',  # 更新：直接用数值型
    'original_price_num': 'original_price',
    'discount_rate': 'discount_rate',

    # 评分相关
    'product_rating': 'average_rating',  # 更新
    'product_rating_count': 'rating_number',  # 更新

    # 销量相关
    'bought_count_number_clean': 'sales',  # 更新：已清洗
    'bsr_rank': 'bsr_rank',

    # Listing质量
    'bullet_count': 'bullet_count',
    'image_count': 'image_count',
    'is_fba': 'is_fba',
    'has_aplus': 'has_aplus',

    # 时间
    'first_available_dt': 'first_available_date',
}


# ============================================================================
# 第一部分：标题解析模块
# ============================================================================

class KnifeTitleParser:
    """厨刀商品标题解析器"""

    def __init__(self):
        # 材质关键词模式
        self.materials = {
            'high_carbon': [
                r'high[\s\-]?carbon', r'carbon\s*steel', r'hc\s*steel',
                r'1\.4116', r'x50crmov15', r'vg[\-]?10', r'vg10', r'aus[\-]?10'
            ],
            'stainless': [
                r'stainless', r'stainless\s*steel', r'ss\s*steel',
                r'rust[\s\-]?free', r'rust[\s\-]?resistant', r'anti[\s\-]?rust'
            ],
            'damascus': [
                r'damascus', r'damaskus', r'67[\s\-]?layer', r'33[\s\-]?layer',
                r'damasc', r'pattern\s*steel'
            ],
            'ceramic': [
                r'ceramic', r'zirconia', r'kyocera'
            ],
            'german_steel': [
                r'german[\s\-]?steel', r'solingen', r'german[\s\-]?made',
                r'germany', r'deutsche'
            ],
            'japanese_steel': [
                r'japanese[\s\-]?steel', r'japan[\s\-]?made', r'nihon',
                r'nippon', r'japanese\s*knife', r'japan\s*quality'
            ]
        }

        # 刀型关键词模式
        self.knife_types = {
            'chef': [r"chef'?s?[\s\-]?knife", r'cook[\s\-]?knife', r'gyuto', r'chef\s*knife'],
            'santoku': [r'santoku', r'asian\s*knife'],
            'cleaver': [r'cleaver', r'butcher', r'chopper', r'meat\s*knife'],
            'paring': [r'paring', r'peeling', r'fruit\s*knife'],
            'bread': [r'bread[\s\-]?knife', r'serrated', r'bread\s*cutter'],
            'utility': [r'utility', r'all[\s\-]?purpose', r'multi[\s\-]?purpose'],
            'boning': [r'boning', r'fillet', r'filet', r'fish\s*knife'],
            'carving': [r'carving', r'slicing', r'roast\s*knife'],
            'steak': [r'steak[\s\-]?knife', r'steak[\s\-]?knives', r'table\s*knife'],
            'nakiri': [r'nakiri', r'vegetable[\s\-]?knife', r'usuba'],
            'kiritsuke': [r'kiritsuke', r'bunka'],
            'deba': [r'deba'],
        }

        # 套装识别模式
        self.set_patterns = [
            r'(\d+)[\s\-]?(?:piece|pcs?|pc)[\s\-]?(?:set|kit|block|collection)?',
            r'(?:set|kit|block|collection)[\s\-]?(?:of|with)?[\s\-]?(\d+)',
            r'(\d+)[\s\-]?(?:knife|knives)[\s\-]?(?:set|kit|block)',
            r'knife[\s\-]?(?:set|block|kit)',
            r'(\d+)\s*(?:in|pc)\s*(?:1|one)\s*set',
        ]

        # 尺寸提取模式
        self.size_pattern = r'(\d+(?:\.\d+)?)\s*[\"\'\-]?\s*(?:inch|in\b|"|\u201d)'

        # 品牌层级分类
        self.brand_tiers = {
            'premium': [
                'wusthof', 'wüsthof', 'zwilling', 'henckels', 'j.a. henckels',
                'shun', 'miyabi', 'global', 'mac', 'tojiro', 'masamoto',
                'misono', 'korin', 'yoshihiro', 'sakai takayuki'
            ],
            'mid': [
                'victorinox', 'mercer', 'dexter', 'dalstrong', 'cutco',
                'cuisinart', 'kitchenaid', 'calphalon', 'oxo', 'wusthof pro',
                'messermeister', 'lamson', 'sabatier'
            ],
            'budget': [
                'farberware', 'chicago cutlery', 'ginsu', 'utopia',
                'amazon basics', 'home hero', 'vremi', 'homgeek', 'imarku',
                'brodark', 'aroma house', 'cook n home', 'deik', 'emojoy'
            ]
        }

    def parse_title(self, title: str) -> Dict:
        """解析单个商品标题"""
        if pd.isna(title) or not isinstance(title, str):
            return self._empty_result()

        title_lower = title.lower().strip()

        result = {
            # 产品类型
            'is_set': self._is_set(title_lower),
            'set_pieces': self._extract_set_pieces(title_lower),

            # 材质特征
            'material': self._extract_material(title_lower),
            'is_damascus': self._contains_any(title_lower, self.materials['damascus']),
            'is_high_carbon': self._contains_any(title_lower, self.materials['high_carbon']),
            'is_german_steel': self._contains_any(title_lower, self.materials['german_steel']),
            'is_japanese_steel': self._contains_any(title_lower, self.materials['japanese_steel']),
            'is_ceramic': self._contains_any(title_lower, self.materials['ceramic']),

            # 刀型特征
            'knife_type': self._extract_knife_type(title_lower),
            'is_chef_knife': self._contains_any(title_lower, self.knife_types['chef']),
            'is_santoku': self._contains_any(title_lower, self.knife_types['santoku']),
            'is_steak_knife': self._contains_any(title_lower, self.knife_types['steak']),
            'is_cleaver': self._contains_any(title_lower, self.knife_types['cleaver']),
            'is_paring': self._contains_any(title_lower, self.knife_types['paring']),

            # 尺寸
            'blade_size_inch': self._extract_size(title_lower),

            # 附加特征
            'has_block': 'block' in title_lower,
            'has_sheath': any(w in title_lower for w in ['sheath', 'cover', 'guard', 'case']),
            'is_gift': any(w in title_lower for w in ['gift', 'present', 'box', 'packaging']),
            'is_professional': any(w in title_lower for w in ['professional', 'pro ', 'commercial', 'restaurant']),
            'has_sharpener': any(w in title_lower for w in ['sharpener', 'sharpening', 'honing']),
        }

        return result

    def _empty_result(self) -> Dict:
        """返回空结果模板"""
        return {
            'is_set': False, 'set_pieces': 1,
            'material': 'unknown', 'is_damascus': False,
            'is_high_carbon': False, 'is_german_steel': False,
            'is_japanese_steel': False, 'is_ceramic': False,
            'knife_type': 'unknown', 'is_chef_knife': False,
            'is_santoku': False, 'is_steak_knife': False,
            'is_cleaver': False, 'is_paring': False,
            'blade_size_inch': None, 'has_block': False,
            'has_sheath': False, 'is_gift': False,
            'is_professional': False, 'has_sharpener': False
        }

    def _contains_any(self, text: str, patterns: List[str]) -> bool:
        """检查文本是否匹配任一正则模式"""
        return any(re.search(p, text) for p in patterns)

    def _is_set(self, title: str) -> bool:
        """判断是否为套装商品"""
        set_indicators = ['set', 'kit', 'block', 'piece', 'pcs', 'knives', 'collection']
        single_indicators = ['knife sharpener', 'sharpening', 'single knife', 'only 1']

        has_set = any(ind in title for ind in set_indicators)
        is_single = any(ind in title for ind in single_indicators)

        pieces = self._extract_set_pieces(title)

        return has_set and not is_single and pieces > 1

    def _extract_set_pieces(self, title: str) -> int:
        """提取套装件数"""
        for pattern in self.set_patterns:
            match = re.search(pattern, title)
            if match:
                groups = match.groups()
                for g in groups:
                    if g and g.isdigit():
                        pieces = int(g)
                        if 2 <= pieces <= 30:
                            return pieces

        if 'set' in title or 'block' in title:
            if 'knives' in title or 'knife set' in title:
                return 5

        return 1

    def _extract_material(self, title: str) -> str:
        """提取主材质"""
        priority = ['damascus', 'high_carbon', 'german_steel',
                    'japanese_steel', 'ceramic', 'stainless']

        for mat in priority:
            if self._contains_any(title, self.materials[mat]):
                return mat

        return 'stainless'

    def _extract_knife_type(self, title: str) -> str:
        """提取刀型"""
        for knife_type, patterns in self.knife_types.items():
            if self._contains_any(title, patterns):
                return knife_type
        return 'general'

    def _extract_size(self, title: str) -> Optional[float]:
        """提取刀片尺寸"""
        match = re.search(self.size_pattern, title)
        if match:
            try:
                size = float(match.group(1))
                if 2 <= size <= 14:
                    return size
            except ValueError:
                pass
        return None

    def get_brand_tier(self, brand: str) -> str:
        """获取品牌层级"""
        if pd.isna(brand) or not isinstance(brand, str):
            return 'unknown'

        brand_lower = brand.lower().strip()

        for tier, brands in self.brand_tiers.items():
            if any(b in brand_lower for b in brands):
                return tier

        return 'other'

    def parse_dataframe(self, df: pd.DataFrame,
                        title_col: str = 'title',  # 更新默认值
                        brand_col: str = 'brand') -> pd.DataFrame:
        """批量解析 DataFrame"""
        print(f"  解析 {len(df)} 个商品标题...")

        parsed_features = df[title_col].apply(self.parse_title)
        parsed_df = pd.DataFrame(parsed_features.tolist())

        if brand_col in df.columns:
            parsed_df['brand_tier'] = df[brand_col].apply(self.get_brand_tier)
            tier_map = {'premium': 3, 'mid': 2, 'budget': 1, 'other': 0, 'unknown': 0}
            parsed_df['brand_tier_encoded'] = parsed_df['brand_tier'].map(tier_map)

        result = pd.concat([df.reset_index(drop=True), parsed_df.reset_index(drop=True)], axis=1)

        return result


# ============================================================================
# 第二部分：NLP特征聚合模块
# ============================================================================

class NLPFeatureAggregator:
    """将评论级NLP分析结果聚合到ASIN级别"""

    def __init__(self, reviews_df: pd.DataFrame,
                 review_id_col: str = 'review_id',
                 asin_col: str = 'asin'):
        self.review_to_asin = reviews_df.set_index(review_id_col)[asin_col].to_dict()
        self.reviews_df = reviews_df
        self.review_id_col = review_id_col
        self.asin_col = asin_col

        print(f"  建立映射: {len(self.review_to_asin)} 条评论 → ASIN")

    def aggregate_review_stats(self) -> pd.DataFrame:
        """聚合评论统计特征"""
        print("  聚合评论统计特征...")

        df = self.reviews_df.copy()
        agg_dict = {self.review_id_col: 'count'}

        if 'review_rating' in df.columns:
            df['review_rating'] = pd.to_numeric(df['review_rating'], errors='coerce')
            agg_dict['review_rating'] = ['mean', 'std']

        if 'verified_purchase' in df.columns:
            df['verified_purchase_num'] = df['verified_purchase'].map(
                {True: 1, False: 0, 'True': 1, 'False': 0, 'true': 1, 'false': 0}
            ).fillna(0)
            agg_dict['verified_purchase_num'] = 'mean'

        if 'helpful_votes' in df.columns:
            df['helpful_votes'] = pd.to_numeric(df['helpful_votes'], errors='coerce').fillna(0)
            agg_dict['helpful_votes'] = 'mean'

        if 'text_len' in df.columns:
            df['text_len'] = pd.to_numeric(df['text_len'], errors='coerce').fillna(0)
            agg_dict['text_len'] = 'mean'

        result = df.groupby(self.asin_col).agg(agg_dict)

        result.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col
                          for col in result.columns]
        result = result.reset_index()

        rename_map = {
            f'{self.review_id_col}_count': 'nlp_review_count',
            'review_rating_mean': 'nlp_rating_mean',
            'review_rating_std': 'nlp_rating_std',
            'verified_purchase_num_mean': 'verified_purchase_rate',
            'helpful_votes_mean': 'avg_helpful_votes',
            'text_len_mean': 'avg_review_length'
        }
        result = result.rename(columns=rename_map)

        return result

    def aggregate_bert_sentiment(self, bert_df: pd.DataFrame) -> pd.DataFrame:
        """聚合BERT情感分析结果"""
        print("  聚合BERT情感特征...")

        bert_df = bert_df.copy()
        bert_df['asin'] = bert_df['review_id'].map(self.review_to_asin)
        bert_df = bert_df.dropna(subset=['asin'])

        if len(bert_df) == 0:
            print("    警告: BERT数据无法匹配任何ASIN")
            return pd.DataFrame(columns=['asin'])

        label_map = {'negative': -1, 'neutral': 0, 'positive': 1,
                     'NEGATIVE': -1, 'NEUTRAL': 0, 'POSITIVE': 1,
                     'neg': -1, 'neu': 0, 'pos': 1}
        bert_df['label_value'] = bert_df['bert_label'].map(label_map)

        agg_result = bert_df.groupby('asin').agg({
            'label_value': ['mean', 'std'],
            'bert_score': ['mean', 'std'],
            'review_id': 'count'
        })

        agg_result.columns = [
            'bert_sentiment_mean', 'bert_sentiment_std',
            'bert_confidence_mean', 'bert_confidence_std',
            'bert_review_count'
        ]
        agg_result = agg_result.reset_index()

        sentiment_counts = bert_df.groupby(['asin', 'bert_label']).size().unstack(fill_value=0)
        total = sentiment_counts.sum(axis=1)

        sentiment_counts.columns = sentiment_counts.columns.str.lower()

        for label in ['positive', 'neutral', 'negative']:
            if label in sentiment_counts.columns:
                sentiment_counts[f'{label}_ratio'] = sentiment_counts[label] / total
            else:
                sentiment_counts[f'{label}_ratio'] = 0.0

        ratio_cols = [c for c in sentiment_counts.columns if c.endswith('_ratio')]
        sentiment_ratios = sentiment_counts[ratio_cols].reset_index()

        result = agg_result.merge(sentiment_ratios, on='asin', how='left')

        print(f"    BERT特征: {len(result)} 个ASIN")

        return result

    def aggregate_absa(self, absa_df: pd.DataFrame) -> pd.DataFrame:
        """聚合ABSA方面情感结果"""
        print("  聚合ABSA方面情感特征...")

        absa_df = absa_df.copy()
        absa_df['asin'] = absa_df['review_id'].map(self.review_to_asin)
        absa_df = absa_df.dropna(subset=['asin'])

        if len(absa_df) == 0:
            print("    警告: ABSA数据无法匹配任何ASIN")
            return pd.DataFrame(columns=['asin'])

        sentiment_map = {'negative': -1, 'neutral': 0, 'positive': 1,
                         'NEGATIVE': -1, 'NEUTRAL': 0, 'POSITIVE': 1}
        absa_df['sentiment_value'] = absa_df['sentiment'].map(sentiment_map)

        score_col = 'score' if 'score' in absa_df.columns else 'confidence'
        if score_col not in absa_df.columns:
            absa_df[score_col] = 1.0

        aspect_agg = absa_df.groupby(['asin', 'aspect']).agg({
            'sentiment_value': 'mean',
            score_col: 'mean',
            'review_id': 'count'
        }).reset_index()

        aspect_agg.columns = ['asin', 'aspect', 'avg_sentiment', 'avg_confidence', 'mention_count']

        sentiment_pivot = aspect_agg.pivot(
            index='asin',
            columns='aspect',
            values='avg_sentiment'
        ).fillna(0)
        sentiment_pivot.columns = [f'aspect_{col}_sentiment' for col in sentiment_pivot.columns]

        mention_pivot = aspect_agg.pivot(
            index='asin',
            columns='aspect',
            values='mention_count'
        ).fillna(0)
        mention_pivot.columns = [f'aspect_{col}_mentions' for col in mention_pivot.columns]

        result = pd.concat([sentiment_pivot, mention_pivot], axis=1).reset_index()

        sentiment_cols = [c for c in result.columns if c.endswith('_sentiment')]
        if sentiment_cols:
            result['aspect_sentiment_mean'] = result[sentiment_cols].mean(axis=1)
            result['aspect_sentiment_std'] = result[sentiment_cols].std(axis=1).fillna(0)
            result['aspect_count'] = (result[sentiment_cols] != 0).sum(axis=1)

        print(f"    ABSA特征: {len(result)} 个ASIN, {len(sentiment_cols)} 个方面")

        return result

    def create_full_nlp_features(self,
                                 bert_df: pd.DataFrame,
                                 absa_df: pd.DataFrame) -> pd.DataFrame:
        """创建完整NLP特征矩阵"""
        print("\n创建完整NLP特征矩阵...")

        review_stats = self.aggregate_review_stats()
        bert_features = self.aggregate_bert_sentiment(bert_df)
        absa_features = self.aggregate_absa(absa_df)

        nlp_features = review_stats

        if len(bert_features) > 0:
            nlp_features = nlp_features.merge(bert_features, on='asin', how='outer')

        if len(absa_features) > 0:
            nlp_features = nlp_features.merge(absa_features, on='asin', how='outer')

        numeric_cols = nlp_features.select_dtypes(include=[np.number]).columns
        nlp_features[numeric_cols] = nlp_features[numeric_cols].fillna(0)

        print(f"  NLP特征矩阵: {len(nlp_features)} 个ASIN, {len(nlp_features.columns)} 列")

        return nlp_features


# ============================================================================
# 第三部分：特征工程
# ============================================================================

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """创建衍生特征"""
    df = df.copy()

    # 对数变换（使用 products_clean.csv 的字段名）
    if 'price_num' in df.columns:
        df['log_price'] = np.log1p(pd.to_numeric(df['price_num'], errors='coerce').fillna(0))

    if 'product_rating_count' in df.columns:
        df['log_reviews'] = np.log1p(pd.to_numeric(df['product_rating_count'], errors='coerce').fillna(0))

    if 'bought_count_number_clean' in df.columns:
        df['log_sales'] = np.log1p(pd.to_numeric(df['bought_count_number_clean'], errors='coerce').fillna(0))

    if 'bsr_rank' in df.columns:
        df['log_bsr'] = np.log1p(pd.to_numeric(df['bsr_rank'], errors='coerce').fillna(0))

    # 价格/件数
    if 'price_num' in df.columns and 'set_pieces' in df.columns:
        price = pd.to_numeric(df['price_num'], errors='coerce').fillna(0)
        pieces = df['set_pieces'].replace(0, 1)
        df['price_per_piece'] = price / pieces
        df['log_price_per_piece'] = np.log1p(df['price_per_piece'])

    # 评分加权（贝叶斯平均）
    if 'product_rating' in df.columns and 'product_rating_count' in df.columns:
        rating = pd.to_numeric(df['product_rating'], errors='coerce')
        count = pd.to_numeric(df['product_rating_count'], errors='coerce')
        C = rating.mean()
        m = count.quantile(0.25)
        df['weighted_rating'] = (count * rating + m * C) / (count + m)

    # 产品上架时长（天）
    if 'first_available_dt' in df.columns:
        try:
            df['first_available_dt'] = pd.to_datetime(df['first_available_dt'], errors='coerce')
            today = pd.Timestamp.now()
            df['days_on_market'] = (today - df['first_available_dt']).dt.days
            df['log_days_on_market'] = np.log1p(df['days_on_market'].fillna(0))
        except Exception as e:
            print(f"  警告: 无法解析上架日期 - {e}")

    # 转换布尔型
    for col in ['is_fba', 'has_aplus']:
        if col in df.columns:
            df[col] = df[col].map({True: 1, False: 0, 'True': 1, 'False': 0,
                                   'true': 1, 'false': 0, 1: 1, 0: 0}).fillna(0).astype(int)

    return df


def select_clustering_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """选择聚类特征"""

    feature_priority = {
        'core_numeric': [
            'log_price', 'product_rating', 'log_reviews', 'log_sales',
            'weighted_rating', 'log_price_per_piece', 'log_bsr'
        ],
        'listing_quality': [
            'bullet_count', 'image_count', 'is_fba', 'has_aplus',
            'discount_rate', 'log_days_on_market'
        ],
        'brand': [
            'brand_tier_encoded'
        ],
        'product_type': [
            'is_set', 'set_pieces'
        ],
        'material': [
            'is_damascus', 'is_high_carbon', 'is_german_steel',
            'is_japanese_steel', 'is_ceramic'
        ],
        'knife_type': [
            'is_chef_knife', 'is_santoku', 'is_steak_knife',
            'is_cleaver', 'is_paring'
        ],
        'additional': [
            'has_block', 'is_professional', 'is_gift', 'blade_size_inch'
        ],
        'bert_sentiment': [
            'bert_sentiment_mean', 'positive_ratio', 'negative_ratio'
        ],
        'absa_sentiment': [
            'aspect_sharpness_sentiment', 'aspect_quality_sentiment',
            'aspect_durability_sentiment', 'aspect_handle_sentiment',
            'aspect_value_sentiment', 'aspect_rust_sentiment',
            'aspect_appearance_sentiment', 'aspect_balance_sentiment',
            'aspect_sentiment_mean'
        ],
        'review_stats': [
            'verified_purchase_rate', 'avg_helpful_votes'
        ]
    }

    selected_features = []
    feature_report = {}

    for category, features in feature_priority.items():
        available = [f for f in features if f in df.columns]
        selected_features.extend(available)
        feature_report[category] = {
            'requested': len(features),
            'available': len(available),
            'features': available
        }

    print("\n" + "=" * 60)
    print("特征选择报告")
    print("=" * 60)
    for category, info in feature_report.items():
        status = "✅" if info['available'] == info['requested'] else "⚠️"
        print(f"  {status} {category}: {info['available']}/{info['requested']} 特征")
        if info['features']:
            print(f"      → {info['features']}")
    print("-" * 60)
    print(f"  📊 总计: {len(selected_features)} 个聚类特征")
    print("=" * 60)

    return df, selected_features


# ============================================================================
# 第四部分：数据质量报告
# ============================================================================

def generate_quality_report(df: pd.DataFrame,
                            feature_cols: List[str],
                            output_path: str = 'data_quality_report.txt'):
    """生成数据质量报告"""

    lines = []
    lines.append("=" * 70)
    lines.append("          聚类数据质量检查报告")
    lines.append("=" * 70)
    lines.append(f"\n📊 数据概览:")
    lines.append(f"  - 总样本数: {len(df)}")
    lines.append(f"  - 聚类特征数: {len(feature_cols)}")
    lines.append(f"  - 总列数: {len(df.columns)}")

    lines.append(f"\n📈 特征覆盖率:")
    lines.append("-" * 50)

    coverage_data = []
    for col in feature_cols:
        if col in df.columns:
            coverage = df[col].notna().mean() * 100
            coverage_data.append((col, coverage))

    coverage_data.sort(key=lambda x: x[1], reverse=True)

    for col, coverage in coverage_data:
        bar = "█" * int(coverage / 5) + "░" * (20 - int(coverage / 5))
        status = "✅" if coverage >= 80 else "⚠️" if coverage >= 50 else "❌"
        lines.append(f"  {status} {col:<35} {bar} {coverage:5.1f}%")

    lines.append(f"\n📉 数值特征统计:")
    lines.append("-" * 70)

    numeric_features = [f for f in feature_cols if f in df.columns and
                        df[f].dtype in ['float64', 'int64', 'float32', 'int32']]

    if numeric_features:
        stats_df = df[numeric_features].describe().T
        stats_df = stats_df[['mean', 'std', 'min', '50%', 'max']]
        lines.append(stats_df.round(3).to_string())

    lines.append(f"\n🏷️ 布尔/类别特征分布:")
    lines.append("-" * 50)

    bool_features = [f for f in feature_cols if f in df.columns and
                     (df[f].dtype == 'bool' or f.startswith('is_') or f.startswith('has_'))]

    for col in bool_features:
        if col in df.columns:
            true_pct = df[col].sum() / len(df) * 100
            lines.append(f"  {col:<30} True: {true_pct:5.1f}%")

    lines.append("\n" + "=" * 70)
    lines.append("                    报告生成完成")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(report_text)
    print(f"\n📄 报告已保存至: {output_path}")

    return report_text


def plot_feature_distributions(df: pd.DataFrame,
                               feature_cols: List[str],
                               output_path: str = 'feature_distributions.png'):
    """绘制特征分布图"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.rcParams['font.size'] = 10

        numeric_features = [f for f in feature_cols if f in df.columns and
                            df[f].dtype in ['float64', 'int64', 'float32', 'int32']][:16]

        if not numeric_features:
            print("没有数值特征可供可视化")
            return

        n_features = len(numeric_features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
        axes = axes.flatten()

        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12',
                  '#1abc9c', '#34495e', '#e91e63']

        for i, col in enumerate(numeric_features):
            ax = axes[i]
            data = df[col].dropna()

            if len(data) > 0:
                color = colors[i % len(colors)]
                sns.histplot(data, kde=True, ax=ax, color=color, edgecolor='white', alpha=0.7)
                ax.set_title(col, fontsize=10, fontweight='bold')
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.axvline(data.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean: {data.mean():.2f}')
                ax.legend(fontsize=8)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle('聚类特征分布概览', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"📈 特征分布图已保存至: {output_path}")

    except ImportError:
        print("⚠️ matplotlib/seaborn 未安装，跳过可视化")


# ============================================================================
# 第五部分：主流程
# ============================================================================

def main():
    """主处理流程"""

    print("\n" + "=" * 70)
    print("          厨刀市场聚类分析 - 数据准备模块")
    print("          (适配 products_clean.csv 字段结构)")
    print("=" * 70)

    # ==================== 文件路径配置 ====================
    PRODUCTS_FILE = 'products_clean.csv'
    REVIEWS_FILE = 'reviews_cleaned.csv'
    ABSA_FILE = 'absa_detailed.csv'
    BERT_FILE = 'bert_sentiment_results.csv'
    OUTPUT_DIR = '.'  # 输出到当前目录

    # ==================== Step 1: 加载数据 ====================
    print("\n[Step 1/7] 加载原始数据...")

    try:
        products = pd.read_csv(PRODUCTS_FILE)
        print(f"  ✅ {PRODUCTS_FILE}: {len(products)} 条")
    except FileNotFoundError:
        print(f"  ❌ 未找到 {PRODUCTS_FILE}")
        return None, None

    try:
        reviews = pd.read_csv(REVIEWS_FILE)
        print(f"  ✅ {REVIEWS_FILE}: {len(reviews)} 条")
    except FileNotFoundError:
        print(f"  ⚠️ 未找到 {REVIEWS_FILE}")
        reviews = None

    try:
        absa_detailed = pd.read_csv(ABSA_FILE)
        print(f"  ✅ {ABSA_FILE}: {len(absa_detailed)} 条")
    except FileNotFoundError:
        print(f"  ⚠️ 未找到 {ABSA_FILE}")
        absa_detailed = None

    try:
        bert_results = pd.read_csv(BERT_FILE)
        print(f"  ✅ {BERT_FILE}: {len(bert_results)} 条")
    except FileNotFoundError:
        print(f"  ⚠️ 未找到 {BERT_FILE}")
        bert_results = None

    # ==================== Step 2: 标题解析 ====================
    print("\n[Step 2/7] 解析商品标题...")

    parser = KnifeTitleParser()
    products_parsed = parser.parse_dataframe(
        products,
        title_col='title',  # 使用 products_clean.csv 的字段名
        brand_col='brand'
    )

    # 打印统计
    print(f"\n  📊 标题解析统计:")
    print(f"    - 套装商品: {products_parsed['is_set'].sum()} ({products_parsed['is_set'].mean() * 100:.1f}%)")
    print(f"    - 大马士革钢: {products_parsed['is_damascus'].sum()}")
    print(f"    - 德国钢: {products_parsed['is_german_steel'].sum()}")
    print(f"    - 日本钢: {products_parsed['is_japanese_steel'].sum()}")
    print(f"    - 品牌层级:")
    for tier, count in products_parsed['brand_tier'].value_counts().items():
        print(f"        {tier}: {count}")

    # ==================== Step 3: 衍生特征 ====================
    print("\n[Step 3/7] 创建衍生特征...")

    products_parsed = create_derived_features(products_parsed)

    # ==================== Step 4: NLP特征聚合 ====================
    print("\n[Step 4/7] NLP特征聚合...")

    nlp_features = None
    if reviews is not None:
        aggregator = NLPFeatureAggregator(reviews)

        nlp_features = pd.DataFrame({'asin': products_parsed['asin'].unique()})

        if bert_results is not None:
            bert_features = aggregator.aggregate_bert_sentiment(bert_results)
            nlp_features = nlp_features.merge(bert_features, on='asin', how='left')

        if absa_detailed is not None:
            absa_features = aggregator.aggregate_absa(absa_detailed)
            nlp_features = nlp_features.merge(absa_features, on='asin', how='left')

        review_stats = aggregator.aggregate_review_stats()
        nlp_features = nlp_features.merge(review_stats, on='asin', how='left')

        print(f"  ✅ NLP特征维度: {nlp_features.shape}")
    else:
        print("  ⚠️ 跳过NLP特征聚合")

    # ==================== Step 5: 合并特征 ====================
    print("\n[Step 5/7] 合并特征...")

    final_df = products_parsed.copy()

    if nlp_features is not None:
        final_df = final_df.merge(nlp_features, on='asin', how='left')

    print(f"  ✅ 合并后维度: {final_df.shape}")

    # ==================== Step 6: 特征选择 ====================
    print("\n[Step 6/7] 特征选择...")

    final_df, feature_cols = select_clustering_features(final_df)

    # 缺失值处理
    print("\n处理缺失值...")
    for col in feature_cols:
        if col in final_df.columns:
            if final_df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                median_val = final_df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                final_df[col] = final_df[col].fillna(median_val)
            else:
                mode_val = final_df[col].mode()
                if len(mode_val) > 0:
                    final_df[col] = final_df[col].fillna(mode_val[0])

    # 布尔转整数
    bool_cols = [c for c in feature_cols if c in final_df.columns and
                 (final_df[c].dtype == 'bool' or c.startswith('is_') or c.startswith('has_'))]
    for col in bool_cols:
        final_df[col] = final_df[col].astype(int)

    # ==================== Step 7: 保存结果 ====================
    print("\n[Step 7/7] 保存结果...")

    # 完整特征矩阵
    output_full = os.path.join(OUTPUT_DIR, 'clustering_features.csv')
    final_df.to_csv(output_full, index=False)
    print(f"  ✅ {output_full} ({final_df.shape[0]} 行, {final_df.shape[1]} 列)")

    # 仅聚类特征
    cluster_only = final_df[['asin'] + [c for c in feature_cols if c in final_df.columns]]
    output_only = os.path.join(OUTPUT_DIR, 'clustering_features_only.csv')
    cluster_only.to_csv(output_only, index=False)
    print(f"  ✅ {output_only} ({cluster_only.shape[0]} 行, {cluster_only.shape[1]} 列)")

    # 特征列表
    output_cols = os.path.join(OUTPUT_DIR, 'feature_columns.txt')
    with open(output_cols, 'w') as f:
        f.write('\n'.join(feature_cols))
    print(f"  ✅ {output_cols} ({len(feature_cols)} 个特征)")

    # 质量报告
    output_report = os.path.join(OUTPUT_DIR, 'data_quality_report.txt')
    generate_quality_report(final_df, feature_cols, output_report)

    # 分布图
    output_fig = os.path.join(OUTPUT_DIR, 'feature_distributions.png')
    plot_feature_distributions(final_df, feature_cols, output_fig)

    # ==================== 完成 ====================
    print("\n" + "=" * 70)
    print("                    ✅ 数据准备完成！")
    print("=" * 70)
    print("\n📁 生成的文件:")
    print(f"  📊 clustering_features.csv      - 完整特征矩阵")
    print(f"  📊 clustering_features_only.csv - 仅聚类特征")
    print(f"  📋 feature_columns.txt          - 特征列表")
    print(f"  📋 data_quality_report.txt      - 数据质量报告")
    print(f"  📈 feature_distributions.png    - 特征分布图")
    print("\n🚀 下一步: 运行聚类分析代码")
    print("=" * 70)

    return final_df, feature_cols


if __name__ == '__main__':
    final_df, feature_cols = main()
