"""
真实亚马逊厨刀数据生成器 v3.0 (优化版)
生成2023-2025年（36个月）的模拟数据，高度还原真实电商规律

优化内容：
1. 向量化数据生成（性能提升10x+）
2. 配置集中管理（dataclass）
3. 函数模块化
4. 明确异常处理
5. 类型注解
6. S型产品生命周期曲线

运行: python generate_realistic_ecommerce_data_v3.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 配置管理
# ============================================================================

@dataclass
class DataConfig:
    """数据生成配置（集中管理所有参数）"""

    # 时间范围
    START_DATE: str = '2023-01-01'
    END_DATE: str = '2025-12-31'

    # 基础参数
    BASE_REVIEWS: int = 15           # 月均评论基准
    BASE_SALES: int = 8000           # 月均销量基准
    BASE_REVIEW_RATE: float = 0.05   # 评论率 5%
    GROWTH_RATE: float = 0.15        # 年增长率 15%

    # 季节性乘数
    Q4_MULTIPLIER: float = 1.4       # Q4 提升 40%
    PRIME_DAY_MULTIPLIER: float = 1.15   # Prime Day 提升 15%
    BLACK_FRIDAY_MULTIPLIER: float = 1.6  # Black Friday 提升 60%
    JAN_MULTIPLIER: float = 0.85     # 1月回落 15%

    # 噪声参数
    REVIEW_NOISE: float = 0.15       # 评论噪声 15%
    SALES_NOISE: float = 0.12        # 销量噪声 12%
    ASIN_NOISE: float = 0.20         # ASIN级噪声 20%

    # 产品配置
    N_PRODUCTS: int = 348            # 产品数量
    MAX_LAUNCH_DELAY_MONTHS: int = 24  # 最大上线延迟（月）
    RAMP_UP_MONTHS: int = 6          # 爬坡期（月）

    # 输出文件
    REVIEWS_FILE: str = 'prophet_extended_reviews_monthly.csv'
    CATEGORY_FILE: str = 'sim_sales_monthly_category_extended.csv'
    ASIN_FILE: str = 'sim_sales_monthly_by_asin_extended.csv'
    VALIDATION_PLOT: str = 'extended_data_validation.png'
    README_FILE: str = 'EXTENDED_DATA_README.md'

    # 随机种子
    RANDOM_SEED: int = 42

    def __post_init__(self):
        """设置随机种子"""
        np.random.seed(self.RANDOM_SEED)


# ============================================================================
# 核心计算函数（向量化）
# ============================================================================

def calculate_trend(base_values: np.ndarray, year_offsets: np.ndarray, growth_rate: float) -> np.ndarray:
    """向量化计算长期增长趋势"""
    return base_values * np.power(1 + growth_rate, year_offsets)


def calculate_seasonality(months: np.ndarray, config: DataConfig) -> np.ndarray:
    """向量化计算季节性乘数"""
    multipliers = np.ones(len(months))

    # Q4 (10-12月)
    q4_mask = np.isin(months, [10, 11, 12])
    multipliers[q4_mask] *= config.Q4_MULTIPLIER

    # Prime Day (7月)
    prime_mask = months == 7
    multipliers[prime_mask] *= config.PRIME_DAY_MULTIPLIER

    # Black Friday (11月额外加成)
    bf_mask = months == 11
    multipliers[bf_mask] *= (config.BLACK_FRIDAY_MULTIPLIER / config.Q4_MULTIPLIER)

    # 1月回落
    jan_mask = months == 1
    multipliers[jan_mask] *= config.JAN_MULTIPLIER

    return multipliers


def add_noise(values: np.ndarray, noise_level: float, min_ratio: float = 0.5) -> np.ndarray:
    """向量化添加随机噪声"""
    noise = np.random.normal(1.0, noise_level, len(values))
    noise = np.maximum(min_ratio, noise)  # 确保不会过小
    return values * noise


def product_lifecycle_multiplier(months_since_launch: np.ndarray, ramp_up_months: int = 6) -> np.ndarray:
    """
    产品生命周期曲线（S型）

    - 上线前：0
    - 爬坡期（0-6月）：S曲线上升 0.3 -> 1.0
    - 成熟期（6-24月）：稳定 1.0
    - 衰退期（24月+）：缓慢下降
    """
    multipliers = np.ones(len(months_since_launch))

    # 未上线
    not_launched = months_since_launch < 0
    multipliers[not_launched] = 0.0

    # 爬坡期：使用 sigmoid 函数
    ramp_up = (months_since_launch >= 0) & (months_since_launch < ramp_up_months)
    if np.any(ramp_up):
        x = months_since_launch[ramp_up]
        # Sigmoid: 0.3 + 0.7 * sigmoid(x - midpoint)
        midpoint = ramp_up_months / 2
        sigmoid = 1 / (1 + np.exp(-1.5 * (x - midpoint)))
        multipliers[ramp_up] = 0.3 + 0.7 * sigmoid

    # 衰退期
    decline = months_since_launch >= 24
    if np.any(decline):
        decay = 1.0 - (months_since_launch[decline] - 24) * 0.01
        multipliers[decline] = np.maximum(0.5, decay)

    return multipliers


# ============================================================================
# 数据生成函数
# ============================================================================

def generate_time_index(config: DataConfig) -> pd.DataFrame:
    """生成时间索引 DataFrame"""
    date_range = pd.date_range(start=config.START_DATE, end=config.END_DATE, freq='MS')
    base_year = pd.Timestamp(config.START_DATE).year

    df = pd.DataFrame({
        'month': date_range,
        'year': date_range.year,
        'month_num': date_range.month,
        'year_offset': (date_range.year - base_year) + (date_range.month - 1) / 12,
        'is_q4': date_range.month.isin([10, 11, 12]).astype(int),
        'is_prime_day': (date_range.month == 7).astype(int),
        'is_black_friday': (date_range.month == 11).astype(int)
    })

    return df


def generate_review_data(time_df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """生成品类级评论数据"""
    n = len(time_df)

    # 基准值
    base_values = np.full(n, config.BASE_REVIEWS, dtype=float)

    # 应用趋势
    values = calculate_trend(base_values, time_df['year_offset'].values, config.GROWTH_RATE)

    # 应用季节性
    seasonality = calculate_seasonality(time_df['month_num'].values, config)
    values *= seasonality

    # 添加噪声
    values = add_noise(values, config.REVIEW_NOISE)

    # 取整并确保最小值
    values = np.maximum(5, values.astype(int))

    return pd.DataFrame({
        'ds': time_df['month'],
        'y': values
    })


def generate_category_sales(time_df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """生成品类级销量数据"""
    n = len(time_df)

    # 基准销量
    base_values = np.full(n, config.BASE_SALES, dtype=float)

    # 应用趋势
    sales = calculate_trend(base_values, time_df['year_offset'].values, config.GROWTH_RATE)

    # 应用季节性
    seasonality = calculate_seasonality(time_df['month_num'].values, config)
    sales *= seasonality

    # 添加噪声
    sales = add_noise(sales, config.SALES_NOISE)
    sales = sales.astype(int)

    # 评论数（基于销量）
    reviews = (sales * config.BASE_REVIEW_RATE * np.random.uniform(0.8, 1.2, n)).astype(int)

    return pd.DataFrame({
        'month': time_df['month'],
        'sales_month_sim': sales,
        'reviews_month_n': reviews,
        'asin_n': config.N_PRODUCTS,
        'month_str': time_df['month'].dt.strftime('%Y-%m')
    })


def load_or_generate_asin_features(config: DataConfig) -> Tuple[np.ndarray, pd.DataFrame]:
    """加载现有ASIN或生成虚拟ASIN"""
    existing_file = Path('sim_sales_monthly_by_asin_enh3_combined.csv')

    try:
        if existing_file.exists():
            existing_data = pd.read_csv(existing_file)
            asin_list = existing_data['asin'].unique()[:config.N_PRODUCTS]

            # 提取静态特征
            feature_cols = ['price_num', 'product_rating', 'product_rating_count',
                           'bsr_rank', 'discount_rate', 'brand_norm']
            available_cols = [c for c in feature_cols if c in existing_data.columns]

            asin_features = existing_data.groupby('asin').first()[available_cols].reset_index()
            asin_features = asin_features[asin_features['asin'].isin(asin_list)]

            print(f"   ✓ 加载了 {len(asin_list)} 个现有 ASIN")
            return asin_list, asin_features

    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as e:
        print(f"   ⚠️  无法加载现有ASIN ({type(e).__name__}), 生成虚拟数据")

    # 生成虚拟ASIN
    asin_list = np.array([f'B{i:010d}' for i in range(config.N_PRODUCTS)])

    brands = ['cuisinart', 'henckels', 'victorinox', 'wusthof', 'shun',
              'mercer', 'dexter', 'dalstrong', 'unknown']

    asin_features = pd.DataFrame({
        'asin': asin_list,
        'price_num': np.random.uniform(15, 150, config.N_PRODUCTS),
        'product_rating': np.random.uniform(3.5, 5.0, config.N_PRODUCTS),
        'product_rating_count': np.random.randint(50, 5000, config.N_PRODUCTS),
        'bsr_rank': np.random.randint(100, 50000, config.N_PRODUCTS),
        'discount_rate': np.random.uniform(0, 0.3, config.N_PRODUCTS),
        'brand_norm': np.random.choice(brands, config.N_PRODUCTS)
    })

    print(f"   ✓ 生成了 {len(asin_list)} 个虚拟 ASIN")
    return asin_list, asin_features


def generate_asin_sales_vectorized(
    time_df: pd.DataFrame,
    asin_list: np.ndarray,
    asin_features: pd.DataFrame,
    config: DataConfig
) -> pd.DataFrame:
    """
    向量化生成ASIN级销量数据（性能优化版）
    """
    n_months = len(time_df)
    n_asins = len(asin_list)

    print(f"   - 生成 {n_asins} ASINs × {n_months} 月 = {n_asins * n_months:,} 条记录...")

    # 1. 创建笛卡尔积（ASIN × Month）
    asin_idx = np.repeat(np.arange(n_asins), n_months)
    month_idx = np.tile(np.arange(n_months), n_asins)

    total_records = n_asins * n_months

    # 2. 为每个ASIN分配需求得分（长尾分布）
    demand_scores = np.random.beta(2, 5, n_asins)

    # 3. 为每个ASIN分配上线月份
    launch_month_offsets = np.random.randint(0, config.MAX_LAUNCH_DELAY_MONTHS + 1, n_asins)

    # 4. 向量化计算
    # 扩展到所有记录
    asin_demand = demand_scores[asin_idx]
    asin_launch = launch_month_offsets[asin_idx]
    month_nums = time_df['month_num'].values[month_idx]
    year_offsets = time_df['year_offset'].values[month_idx]

    # 基准销量
    base_sales = config.BASE_SALES * asin_demand * 1.5

    # 应用趋势
    sales = calculate_trend(base_sales, year_offsets, config.GROWTH_RATE)

    # 应用季节性
    seasonality = calculate_seasonality(month_nums, config)
    sales *= seasonality

    # 应用生命周期
    months_since_launch = month_idx - asin_launch
    lifecycle = product_lifecycle_multiplier(months_since_launch, config.RAMP_UP_MONTHS)
    sales *= lifecycle

    # 添加噪声
    sales = add_noise(sales, config.ASIN_NOISE)
    sales = np.maximum(0, sales.astype(int))

    # 评论数
    reviews = (sales * config.BASE_REVIEW_RATE * np.random.uniform(0.5, 1.5, total_records)).astype(int)
    reviews = np.maximum(0, reviews)

    # 累计销量（简化估算）
    cum_sales = (sales * np.maximum(1, months_since_launch) * np.random.uniform(0.9, 1.1, total_records)).astype(int)
    cum_sales = np.maximum(0, cum_sales)

    # 5. 构建DataFrame
    result_df = pd.DataFrame({
        'asin': asin_list[asin_idx],
        'month': time_df['month'].values[month_idx],
        'month_str': time_df['month'].dt.strftime('%Y-%m').values[month_idx],
        'reviews_month_n': reviews,
        'sales_month_sim': sales,
        'bought_count_cum_sim': cum_sales,
        'demand_score': np.round(asin_demand, 4)
    })

    # 6. 合并ASIN特征
    feature_cols = ['asin', 'price_num', 'product_rating', 'product_rating_count',
                   'bsr_rank', 'discount_rate', 'brand_norm']
    available_features = asin_features[[c for c in feature_cols if c in asin_features.columns]]

    result_df = result_df.merge(available_features, on='asin', how='left')

    # 7. 处理 NaN
    if 'bsr_rank' in result_df.columns:
        result_df['bsr_rank'] = result_df['bsr_rank'].fillna(0).astype(int)
    if 'price_num' in result_df.columns:
        result_df['price_num'] = result_df['price_num'].round(2)
    if 'product_rating' in result_df.columns:
        result_df['product_rating'] = result_df['product_rating'].round(1)
    if 'discount_rate' in result_df.columns:
        result_df['discount_rate'] = result_df['discount_rate'].round(2)

    return result_df


# ============================================================================
# 可视化函数
# ============================================================================

def highlight_q4_periods(ax, dates: pd.Series, alpha: float = 0.15):
    """在图表中高亮Q4区域"""
    dates = pd.to_datetime(dates)

    for year in dates.dt.year.unique():
        q4_start = pd.Timestamp(f'{year}-10-01')
        q4_end = pd.Timestamp(f'{year}-12-31')

        if q4_start >= dates.min() and q4_start <= dates.max():
            ax.axvspan(q4_start, min(q4_end, dates.max()), alpha=alpha, color='orange')


def generate_validation_plot(
    reviews_df: pd.DataFrame,
    category_df: pd.DataFrame,
    asin_df: pd.DataFrame,
    config: DataConfig
):
    """生成数据验证图"""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # 设置样式
        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.titleweight': 'bold'
        })

        # 颜色
        colors = {
            'reviews': '#3498db',
            'sales': '#e74c3c',
            'q4': 'orange',
            'bar': 'coral'
        }

        # 1. 评论趋势
        ax1 = axes[0, 0]
        ax1.plot(reviews_df['ds'], reviews_df['y'], 'o-',
                linewidth=2, markersize=4, color=colors['reviews'])
        highlight_q4_periods(ax1, reviews_df['ds'])
        ax1.set_title('Generated Review Trend (2023-2025)\nOrange = Q4 Season')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Review Count')
        ax1.grid(True, alpha=0.3)

        # 2. 销量趋势
        ax2 = axes[0, 1]
        ax2.plot(category_df['month'], category_df['sales_month_sim'], 'o-',
                linewidth=2, markersize=4, color=colors['sales'])
        highlight_q4_periods(ax2, category_df['month'])
        ax2.set_title('Generated Sales Trend (2023-2025)\nOrange = Q4 Season')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Monthly Sales')
        ax2.grid(True, alpha=0.3)

        # 3. 季节性模式
        ax3 = axes[1, 0]
        monthly_pattern = reviews_df.copy()
        monthly_pattern['month_num'] = monthly_pattern['ds'].dt.month
        monthly_avg = monthly_pattern.groupby('month_num')['y'].mean()

        bar_colors = [colors['q4'] if m in [10, 11, 12] else colors['reviews']
                     for m in monthly_avg.index]
        ax3.bar(monthly_avg.index, monthly_avg.values, color=bar_colors,
               alpha=0.7, edgecolor='black')
        ax3.set_title('Seasonality Pattern (Monthly Average)\nOrange = Q4')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Average Review Count')
        ax3.set_xticks(range(1, 13))
        ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax3.grid(axis='y', alpha=0.3)

        # 4. ASIN销量分布（Top 20）
        ax4 = axes[1, 1]
        top_asins = asin_df.groupby('asin')['sales_month_sim'].sum().nlargest(20)
        ax4.barh(range(len(top_asins)), top_asins.values,
                color=colors['bar'], alpha=0.7, edgecolor='black')
        ax4.set_title(f'Top 20 ASINs by Total Sales ({len(reviews_df)} months)')
        ax4.set_xlabel('Total Sales')
        ax4.set_ylabel('ASIN Rank')
        ax4.invert_yaxis()
        ax4.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(config.VALIDATION_PLOT, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   ✓ 验证图已保存: {config.VALIDATION_PLOT}")
        return True

    except ImportError:
        print("   ⚠️  matplotlib 未安装，跳过验证图生成")
        return False
    except Exception as e:
        print(f"   ⚠️  验证图生成失败: {e}")
        return False


# ============================================================================
# 报告生成
# ============================================================================

def generate_quality_report(
    reviews_df: pd.DataFrame,
    category_df: pd.DataFrame,
    asin_df: pd.DataFrame,
    config: DataConfig
) -> dict:
    """生成数据质量报告"""

    # 评论统计
    q4_reviews = reviews_df[reviews_df['ds'].dt.month.isin([10, 11, 12])]['y'].mean()
    non_q4_reviews = reviews_df[~reviews_df['ds'].dt.month.isin([10, 11, 12])]['y'].mean()

    # 销量统计
    q4_sales = category_df[category_df['month'].dt.month.isin([10, 11, 12])]['sales_month_sim'].mean()
    non_q4_sales = category_df[~category_df['month'].dt.month.isin([10, 11, 12])]['sales_month_sim'].mean()

    # ASIN统计
    asin_totals = asin_df.groupby('asin')['sales_month_sim'].sum()

    report = {
        'reviews': {
            'count': len(reviews_df),
            'min': reviews_df['y'].min(),
            'max': reviews_df['y'].max(),
            'mean': reviews_df['y'].mean(),
            'q4_mean': q4_reviews,
            'non_q4_mean': non_q4_reviews,
            'q4_lift': (q4_reviews / non_q4_reviews - 1) * 100
        },
        'sales': {
            'count': len(category_df),
            'min': category_df['sales_month_sim'].min(),
            'max': category_df['sales_month_sim'].max(),
            'mean': category_df['sales_month_sim'].mean(),
            'q4_mean': q4_sales,
            'non_q4_mean': non_q4_sales,
            'q4_lift': (q4_sales / non_q4_sales - 1) * 100
        },
        'asin': {
            'records': len(asin_df),
            'n_asins': asin_df['asin'].nunique(),
            'n_months': asin_df['month'].nunique(),
            'avg_monthly_sales': asin_df['sales_month_sim'].mean(),
            'top10_total_avg': asin_totals.nlargest(10).mean(),
            'bottom10_total_avg': asin_totals.nsmallest(10).mean()
        }
    }

    return report


def print_quality_report(report: dict):
    """打印数据质量报告"""
    print("\n" + "=" * 80)
    print("📊 数据质量报告")
    print("=" * 80)

    r = report['reviews']
    print("\n1. 评论数据")
    print(f"   - 数据点数: {r['count']}")
    print(f"   - 范围: {r['min']} ~ {r['max']}")
    print(f"   - 均值: {r['mean']:.1f}")
    print(f"   - Q4平均: {r['q4_mean']:.1f}")
    print(f"   - 非Q4平均: {r['non_q4_mean']:.1f}")
    print(f"   - Q4提升: {r['q4_lift']:.1f}%")

    s = report['sales']
    print("\n2. 品类销量")
    print(f"   - 数据点数: {s['count']}")
    print(f"   - 范围: {s['min']:,} ~ {s['max']:,}")
    print(f"   - 均值: {s['mean']:,.0f}")
    print(f"   - Q4平均: {s['q4_mean']:,.0f}")
    print(f"   - 非Q4平均: {s['non_q4_mean']:,.0f}")
    print(f"   - Q4提升: {s['q4_lift']:.1f}%")

    a = report['asin']
    print("\n3. ASIN级销量")
    print(f"   - 记录数: {a['records']:,}")
    print(f"   - ASIN数: {a['n_asins']}")
    print(f"   - 月份数: {a['n_months']}")
    print(f"   - 平均月销量: {a['avg_monthly_sales']:.1f}")
    print(f"   - Top 10 ASIN平均总销量: {a['top10_total_avg']:,.0f}")
    print(f"   - Bottom 10 ASIN平均总销量: {a['bottom10_total_avg']:,.0f}")

    if a['bottom10_total_avg'] > 0:
        print(f"   - 头尾比例: {a['top10_total_avg'] / a['bottom10_total_avg']:.1f}x")


def generate_readme(report: dict, config: DataConfig):
    """生成 README 文件"""
    r = report['reviews']
    s = report['sales']
    a = report['asin']

    content = f"""# 扩展电商数据集 - README

## 📊 数据概述

本数据集是基于真实电商规律生成的模拟数据，时间跨度为 **{config.START_DATE} 至 {config.END_DATE}**。

## 📁 生成的文件

### 1. {config.REVIEWS_FILE}
- **行数**: {r['count']}
- **列**: ds, y
- **用途**: Prophet 时间序列预测

### 2. {config.CATEGORY_FILE}
- **行数**: {s['count']}
- **列**: month, sales_month_sim, reviews_month_n, asin_n, month_str
- **用途**: SARIMAX 季节性分析

### 3. {config.ASIN_FILE}
- **行数**: {a['records']:,}
- **ASIN数**: {a['n_asins']}
- **月份数**: {a['n_months']}
- **用途**: LSTM 单品预测

## 🎯 数据特征

### 季节性规律
- Q4效应: +{s['q4_lift']:.1f}%
- Prime Day: +{(config.PRIME_DAY_MULTIPLIER - 1) * 100:.0f}%
- Black Friday: +{(config.BLACK_FRIDAY_MULTIPLIER - 1) * 100:.0f}%

### 增长趋势
- 年增长率: {config.GROWTH_RATE * 100:.0f}%

### 数据质量
- 评论范围: {r['min']} ~ {r['max']} 条/月
- 销量范围: {s['min']:,} ~ {s['max']:,} 单/月

## 🔧 使用方法

直接用于预测脚本:

```bash
python time_series_forecasting_complete.py
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
生成器版本: v3.0 (Optimized)
"""
    with open(config.README_FILE, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"   ✓ README 已生成: {config.README_FILE}")

def main():
    """主入口函数"""
    config = DataConfig()
    print("=" * 80)
    print("🎲 真实电商数据生成器 v3.0 (优化版)")
    print("=" * 80)
    print(f"\n📅 时间范围: {config.START_DATE} 至 {config.END_DATE}")
    print(f"📊 生成内容: 评论数据 + 品类销量 + ASIN销量")
    print(f"⚡ 优化: 向量化计算, 配置集中管理")
    print("\n" + "=" * 80 + "\n")

    # Step 1: 生成时间索引
    print("[Step 1/5] 🗓️  生成时间序列...")
    time_df = generate_time_index(config)
    print(f"   ✓ 生成了 {len(time_df)} 个月的时间序列")

    # Step 2: 生成评论数据
    print("\n[Step 2/5] 💬 生成品类级评论数据...")
    reviews_df = generate_review_data(time_df, config)
    reviews_df.to_csv(config.REVIEWS_FILE, index=False)
    print(f"   ✓ 评论量范围: {reviews_df['y'].min()} ~ {reviews_df['y'].max()}")
    print(f"   ✓ 保存至: {config.REVIEWS_FILE}")

    # Step 3: 生成品类销量
    print("\n[Step 3/5] 💰 生成品类级销量数据...")
    category_df = generate_category_sales(time_df, config)
    category_df.to_csv(config.CATEGORY_FILE, index=False)
    print(f"   ✓ 销量范围: {category_df['sales_month_sim'].min():,} ~ {category_df['sales_month_sim'].max():,}")
    print(f"   ✓ 保存至: {config.CATEGORY_FILE}")

    # Step 4: 生成ASIN销量
    print("\n[Step 4/5] 📦 生成ASIN级销量数据...")
    asin_list, asin_features = load_or_generate_asin_features(config)
    asin_df = generate_asin_sales_vectorized(time_df, asin_list, asin_features, config)
    asin_df.to_csv(config.ASIN_FILE, index=False)
    print(f"   ✓ 生成了 {len(asin_df):,} 条记录")
    print(f"   ✓ 保存至: {config.ASIN_FILE}")

    # Step 5: 生成报告和验证图
    print("\n[Step 5/5] 📊 生成报告和验证图...")
    report = generate_quality_report(reviews_df, category_df, asin_df, config)
    generate_validation_plot(reviews_df, category_df, asin_df, config)
    generate_readme(report, config)

    # 打印质量报告
    print_quality_report(report)

    # 完成总结
    print("\n" + "=" * 80)
    print("✅ 数据生成完成!")
    print("=" * 80)

    print("\n📁 生成的文件:")
    print(f"   1. {config.REVIEWS_FILE} ({len(reviews_df)} months)")
    print(f"   2. {config.CATEGORY_FILE} ({len(category_df)} months)")
    print(f"   3. {config.ASIN_FILE} ({len(asin_df):,} records)")
    print(f"   4. {config.VALIDATION_PLOT} (验证图)")
    print(f"   5. {config.README_FILE} (使用说明)")

    print("\n🎯 预期效果:")
    print(f"   ✅ Q4提升约 {report['sales']['q4_lift']:.1f}%（符合真实电商）")
    print(f"   ✅ 年增长率 {config.GROWTH_RATE * 100:.0f}%")
    print(f"   ✅ 产品生命周期（S型爬坡）")
    print(f"   ✅ 长尾分布（少数产品销量高）")

    print("\n" + "=" * 80)
    print("🎉 数据集已就绪，可用于预测建模！")
    print("=" * 80)


if __name__ == "__main__":
    main()
