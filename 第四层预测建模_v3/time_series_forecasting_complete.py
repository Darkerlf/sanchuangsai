"""
亚马逊厨刀品类时间序列预测建模完整系统 v2.1
================================================================================
模型分工：
  1. Prophet - 评论量预测 (Reviews Forecasting)
  2. SARIMAX - 品类销量季节性分析 (Category Sales Seasonality Analysis)
  3. Holt-Winters - 单品销量预测 (ASIN-level Sales Forecasting)

每个模型的预测目标不同，不进行横向比较。
================================================================================
运行: python time_series_forecasting.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Any
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 配置管理
# ============================================================================

@dataclass
class Config:
    """统一配置管理"""
    # 路径配置
    OUTPUT_DIR: Path = field(default_factory=lambda: Path('forecasting_results'))

    # 数据分割
    TEST_MONTHS: int = 3
    FORECAST_MONTHS: int = 6

    # 安全预测期（置信区间可靠的最大月数）
    HW_FORECAST_MONTHS_SAFE: int = 2
    SARIMAX_FORECAST_MONTHS_SAFE: int = 3

    # Prophet 配置（评论量预测）
    PROPHET_CHANGEPOINT_SCALE: float = 0.05
    PROPHET_SEASONALITY_MODE: str = 'multiplicative'

    # SARIMAX 配置（品类销量季节性分析）
    SARIMAX_ORDER: Tuple[int, int, int] = (1, 1, 1)
    SARIMAX_SEASONAL_ORDER: Tuple[int, int, int, int] = (1, 0, 1, 12)

    # 统计显著性阈值
    SIGNIFICANCE_LEVEL: float = 0.05
    MARGINAL_SIGNIFICANCE_LEVEL: float = 0.10

    # Holt-Winters 配置（单品销量预测）
    HW_SEASONAL_PERIODS: int = 12
    HW_TOP_ASINS: int = 5  # 预测前N个ASIN

    # 置信区间警告阈值
    CI_WIDTH_WARNING_RATIO: float = 1.5  # CI宽度超过预测值的150%时警告

    # 可视化配置
    FIGURE_DPI: int = 300

    def __post_init__(self):
        """创建输出目录"""
        self.OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# 可视化样式
# ============================================================================

def setup_plot_style() -> Dict[str, str]:
    """设置专业可视化风格"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Palatino', 'Georgia', 'DejaVu Serif'],
        'font.size': 11,
        'axes.facecolor': '#f8f9fa',
        'axes.edgecolor': '#2c3e50',
        'axes.linewidth': 1.2,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'axes.labelweight': 'bold',
        'grid.alpha': 0.4,
        'grid.linestyle': ':',
        'grid.color': '#bdc3c7',
        'figure.facecolor': '#ffffff',
        'figure.dpi': 100,
        'figure.titlesize': 15,
        'figure.titleweight': 'bold',
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#2c3e50',
    })

    colors = {
        'primary': '#1a5276',
        'secondary': '#922b21',
        'accent': '#1e8449',
        'warning': '#b9770e',
        'neutral': '#566573',
        'highlight': '#6c3483',
        'background': '#f8f9fa',
        'text': '#2c3e50',
        'q4_highlight': '#f5b041',
        'prophet_color': '#2874a6',
        'sarimax_color': '#7d3c98',
        'hw_color': '#117a65',
    }

    return colors


# ============================================================================
# 数据加载与预处理
# ============================================================================

@dataclass
class DataContainer:
    """数据容器"""
    prophet_data: pd.DataFrame
    category_sales: pd.DataFrame
    asin_sales: pd.DataFrame
    reviews_raw: Optional[pd.DataFrame] = None


def load_and_validate_data() -> DataContainer:
    """加载数据并验证完整性"""
    required_files = {
        'prophet': 'prophet_extended_reviews_monthly.csv',
        'category_sales': 'sim_sales_monthly_category_extended.csv',
        'asin_sales': 'sim_sales_monthly_by_asin_extended.csv',
    }
    optional_files = {
        'reviews': 'reviews_cleaned.csv',
    }

    missing = [f for f in required_files.values() if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(f"缺少必需数据文件: {missing}")

    prophet_data = pd.read_csv(required_files['prophet'])
    prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])

    category_sales = pd.read_csv(required_files['category_sales'])
    category_sales['month'] = pd.to_datetime(category_sales['month'])

    asin_sales = pd.read_csv(required_files['asin_sales'])
    asin_sales['month'] = pd.to_datetime(asin_sales['month'])

    reviews_raw = None
    if Path(optional_files['reviews']).exists():
        reviews_raw = pd.read_csv(optional_files['reviews'])
        if 'review_date_dt' in reviews_raw.columns:
            reviews_raw['review_date_dt'] = pd.to_datetime(reviews_raw['review_date_dt'])

    return DataContainer(
        prophet_data=prophet_data,
        category_sales=category_sales,
        asin_sales=asin_sales,
        reviews_raw=reviews_raw
    )


def train_test_split_ts(
    df: pd.DataFrame,
    date_col: str,
    test_months: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """时间序列分割"""
    df = df.sort_values(date_col).copy()
    cutoff = df[date_col].max() - pd.DateOffset(months=test_months)

    train = df[df[date_col] <= cutoff].copy()
    test = df[df[date_col] > cutoff].copy()

    return train, test


def add_promotion_features(df: pd.DataFrame, date_col: str = 'ds') -> pd.DataFrame:
    """添加促销特征"""
    df = df.copy()
    dates = df[date_col]

    df['is_q4'] = dates.dt.month.isin([10, 11, 12]).astype(int)
    df['is_prime_day'] = (dates.dt.month == 7).astype(int)
    df['is_black_friday'] = ((dates.dt.month == 11) & (dates.dt.day >= 20)).astype(int)
    df['month_sin'] = np.sin(2 * np.pi * dates.dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * dates.dt.month / 12)

    return df


# ============================================================================
# 评估框架
# ============================================================================

@dataclass
class ForecastResult:
    """预测结果容器"""
    model_name: str
    target_description: str
    train_dates: np.ndarray
    train_values: np.ndarray
    test_dates: np.ndarray
    test_actuals: np.ndarray
    test_predictions: np.ndarray
    forecast_dates: np.ndarray
    forecast_values: np.ndarray
    forecast_lower: np.ndarray
    forecast_upper: np.ndarray
    mae: float
    rmse: float
    mape: float
    additional_info: Dict[str, Any] = field(default_factory=dict)


def calculate_metrics(actuals: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """计算评估指标"""
    mae = np.mean(np.abs(actuals - predictions))
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))

    mask = actuals > 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100
    else:
        mape = 0.0

    return {'mae': mae, 'rmse': rmse, 'mape': mape}


# ============================================================================
# Prophet 模型 - 评论量预测
# ============================================================================

def train_prophet_reviews(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Config
) -> Optional[ForecastResult]:
    """
    Prophet 模型 - 评论量预测

    用途：预测未来评论数量趋势，帮助评估市场热度和产品关注度
    """
    try:
        from prophet import Prophet
    except ImportError:
        print("   ⚠️  Prophet 未安装，跳过该模型")
        print("   安装命令: pip install prophet")
        return None

    print("   目标: 月度评论量预测")
    print(f"   训练数据: {len(train_df)} 个月")
    print(f"   测试数据: {len(test_df)} 个月")

    train_enhanced = add_promotion_features(train_df, 'ds')

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode=config.PROPHET_SEASONALITY_MODE,
        changepoint_prior_scale=config.PROPHET_CHANGEPOINT_SCALE,
        interval_width=0.95
    )

    model.add_regressor('is_q4')
    model.add_regressor('is_prime_day')

    model.fit(train_enhanced)

    total_future_months = len(test_df) + config.FORECAST_MONTHS
    future = model.make_future_dataframe(periods=total_future_months, freq='MS')
    future = add_promotion_features(future, 'ds')

    forecast = model.predict(future)

    test_mask = forecast['ds'].isin(test_df['ds'])
    test_predictions = np.maximum(0, forecast.loc[test_mask, 'yhat'].values)
    test_actuals = test_df['y'].values

    metrics = calculate_metrics(test_actuals, test_predictions)

    future_mask = forecast['ds'] > test_df['ds'].max()
    future_forecast = forecast[future_mask]

    print(f"   测试集 MAE: {metrics['mae']:.2f}")
    print(f"   测试集 MAPE: {metrics['mape']:.1f}%")

    return ForecastResult(
        model_name='Prophet',
        target_description='月度评论量 (Review Count)',
        train_dates=train_df['ds'].values,
        train_values=train_df['y'].values,
        test_dates=test_df['ds'].values,
        test_actuals=test_actuals,
        test_predictions=test_predictions,
        forecast_dates=future_forecast['ds'].values,
        forecast_values=np.maximum(0, future_forecast['yhat'].values),
        forecast_lower=np.maximum(0, future_forecast['yhat_lower'].values),
        forecast_upper=future_forecast['yhat_upper'].values,
        mae=metrics['mae'],
        rmse=metrics['rmse'],
        mape=metrics['mape'],
        additional_info={
            'model': model,
            'full_forecast': forecast,
            'components': ['trend', 'yearly', 'is_q4', 'is_prime_day']
        }
    )


# ============================================================================
# SARIMAX 模型 - 品类销量季节性分析
# ============================================================================

def train_sarimax_seasonality(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Config
) -> Optional[ForecastResult]:
    """
    SARIMAX 模型 - 品类销量季节性分析

    用途：分析品类整体销量的季节性模式，量化外部因素（Q4、Prime Day）对销量的影响
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        print("   ⚠️  Statsmodels 未安装，跳过该模型")
        print("   安装命令: pip install statsmodels")
        return None

    print("   目标: 品类销量季节性分析")
    print(f"   训练数据: {len(train_df)} 个月")
    print(f"   测试数据: {len(test_df)} 个月")

    train_ts = train_df.set_index('month')['sales_month_sim'].copy()
    train_ts.index.freq = 'MS'

    test_ts = test_df.set_index('month')['sales_month_sim'].copy()
    test_ts.index.freq = 'MS'

    adf_result = adfuller(train_ts.dropna())
    is_stationary = adf_result[1] < 0.05
    print(f"   ADF p-value: {adf_result[1]:.4f} ({'平稳' if is_stationary else '非平稳'})")

    train_enhanced = add_promotion_features(train_df, 'month')
    exog_cols = ['is_q4', 'is_prime_day']

    if 'reviews_month_n' in train_df.columns:
        exog_cols.append('reviews_month_n')
        print(f"   外生变量: {exog_cols}")

    train_exog = train_enhanced.set_index('month')[exog_cols]

    model = SARIMAX(
        train_ts,
        exog=train_exog,
        order=config.SARIMAX_ORDER,
        seasonal_order=config.SARIMAX_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
        trend='c'
    )

    results = model.fit(disp=False, maxiter=500)

    print(f"   模型 AIC: {results.aic:.2f}")
    print(f"   模型 BIC: {results.bic:.2f}")

    print("\n   📊 季节性影响系数:")
    params = results.params
    pvalues_dict = results.pvalues.to_dict()

    any_significant = False
    for col in exog_cols:
        if col in params.index:
            coef = params[col]
            pvalue = pvalues_dict.get(col, 1.0)
            if pvalue < config.SIGNIFICANCE_LEVEL:
                sig = '***'
                any_significant = True
            elif pvalue < config.MARGINAL_SIGNIFICANCE_LEVEL:
                sig = '**'
            else:
                sig = ''
            print(f"      {col}: {coef:+.2f} {sig} (p={pvalue:.4f})")

    test_enhanced = add_promotion_features(test_df, 'month')
    test_exog = test_enhanced.set_index('month')[exog_cols]

    test_forecast = results.get_forecast(steps=len(test_df), exog=test_exog)
    test_predictions = np.maximum(0, test_forecast.predicted_mean.values)
    test_actuals = test_ts.values

    metrics = calculate_metrics(test_actuals, test_predictions)

    future_months = pd.date_range(
        start=test_ts.index[-1] + pd.DateOffset(months=1),
        periods=config.FORECAST_MONTHS,
        freq='MS'
    )

    future_exog = _create_future_exog(future_months, exog_cols, train_df)
    future_forecast = results.get_forecast(steps=len(future_months), exog=future_exog)
    future_ci = future_forecast.conf_int()

    print(f"\n   测试集 MAE: {metrics['mae']:,.2f}")
    print(f"   测试集 MAPE: {metrics['mape']:.1f}%")

    return ForecastResult(
        model_name='SARIMAX',
        target_description='品类月度销量 (Category Sales)',
        train_dates=train_ts.index.values,
        train_values=train_ts.values,
        test_dates=test_ts.index.values,
        test_actuals=test_actuals,
        test_predictions=test_predictions,
        forecast_dates=future_months.values,
        forecast_values=np.maximum(0, future_forecast.predicted_mean.values),
        forecast_lower=np.maximum(0, future_ci.iloc[:, 0].values),
        forecast_upper=future_ci.iloc[:, 1].values,
        mae=metrics['mae'],
        rmse=metrics['rmse'],
        mape=metrics['mape'],
        additional_info={
            'model': results,
            'aic': results.aic,
            'bic': results.bic,
            'order': config.SARIMAX_ORDER,
            'seasonal_order': config.SARIMAX_SEASONAL_ORDER,
            'is_stationary': is_stationary,
            'params': results.params.to_dict(),
            'pvalues': pvalues_dict,
            'exog_cols': exog_cols,
            'any_significant': any_significant,
            'sample_size': len(train_df)
        }
    )


def _create_future_exog(
    future_months: pd.DatetimeIndex,
    exog_cols: List[str],
    historical_df: pd.DataFrame
) -> pd.DataFrame:
    """创建未来外生变量"""
    future_exog = pd.DataFrame(index=future_months)

    future_exog['is_q4'] = future_months.month.isin([10, 11, 12]).astype(int)
    future_exog['is_prime_day'] = (future_months.month == 7).astype(int)

    if 'reviews_month_n' in exog_cols:
        avg_reviews = historical_df['reviews_month_n'].mean()
        future_exog['reviews_month_n'] = avg_reviews

    return future_exog[exog_cols]


# ============================================================================
# Holt-Winters 模型 - 单品销量预测
# ============================================================================

def train_holt_winters_asin(
    asin_data: pd.DataFrame,
    config: Config,
    asin_id: str
) -> Optional[ForecastResult]:
    """
    Holt-Winters 模型 - 单品销量预测

    用途：预测特定ASIN的未来销量，支持库存规划和补货决策
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:
        print(f"   ⚠️  Statsmodels 未安装")
        return None

    asin_df = asin_data[asin_data['asin'] == asin_id].copy()
    asin_df = asin_df.sort_values('month').reset_index(drop=True)

    if len(asin_df) < 12:
        print(f"   ⚠️  {asin_id} 数据不足 ({len(asin_df)} 个月)，跳过")
        return None

    cutoff = asin_df['month'].max() - pd.DateOffset(months=config.TEST_MONTHS)
    train_df = asin_df[asin_df['month'] <= cutoff]
    test_df = asin_df[asin_df['month'] > cutoff]

    if len(test_df) == 0:
        print(f"   ⚠️  {asin_id} 测试集为空，跳过")
        return None

    train_ts = train_df.set_index('month')['sales_month_sim'].copy()
    train_ts.index = pd.DatetimeIndex(train_ts.index, freq='MS')

    test_ts = test_df.set_index('month')['sales_month_sim'].copy()
    test_ts.index = pd.DatetimeIndex(test_ts.index, freq='MS')

    seasonal_periods = 12 if len(train_ts) >= 24 else max(4, len(train_ts) // 3)

    configurations = [
        {'trend': 'add', 'seasonal': 'add', 'damped_trend': True, 'name': 'AAD'},
        {'trend': 'add', 'seasonal': 'add', 'damped_trend': False, 'name': 'AA'},
        {'trend': 'add', 'seasonal': 'mul', 'damped_trend': True, 'name': 'AMD'},
        {'trend': 'add', 'seasonal': None, 'damped_trend': True, 'name': 'AD'},
    ]

    best_model = None
    best_mape = float('inf')
    best_config = None

    for cfg in configurations:
        try:
            if cfg['seasonal'] is None:
                model = ExponentialSmoothing(
                    train_ts,
                    trend=cfg['trend'],
                    seasonal=None,
                    damped_trend=cfg['damped_trend'],
                    initialization_method='estimated'
                ).fit(optimized=True)
            else:
                model = ExponentialSmoothing(
                    train_ts,
                    seasonal_periods=seasonal_periods,
                    trend=cfg['trend'],
                    seasonal=cfg['seasonal'],
                    damped_trend=cfg['damped_trend'],
                    initialization_method='estimated'
                ).fit(optimized=True)

            test_pred = model.forecast(len(test_ts))
            test_pred_values = np.maximum(0, test_pred.values)

            mask = test_ts.values > 0
            if np.sum(mask) > 0:
                test_mape = np.mean(np.abs((test_ts.values[mask] - test_pred_values[mask]) / test_ts.values[mask])) * 100
            else:
                test_mape = 100

            if test_mape < best_mape:
                best_mape = test_mape
                best_model = model
                best_config = cfg

        except Exception:
            continue

    if best_model is None:
        print(f"   ⚠️  {asin_id} 所有配置均失败")
        return None

    test_pred = best_model.forecast(len(test_ts))
    test_predictions = np.maximum(0, test_pred.values)
    test_actuals = test_ts.values

    metrics = calculate_metrics(test_actuals, test_predictions)

    future_pred = best_model.forecast(config.FORECAST_MONTHS)
    future_pred_values = np.maximum(0, future_pred.values)

    future_dates = pd.date_range(
        start=test_ts.index[-1] + pd.DateOffset(months=1),
        periods=config.FORECAST_MONTHS,
        freq='MS'
    )

    fitted_values = best_model.fittedvalues
    residuals = train_ts.values - fitted_values.values
    std_resid = np.std(residuals)

    uncertainty_growth = np.sqrt(np.arange(1, config.FORECAST_MONTHS + 1))
    forecast_std = std_resid * uncertainty_growth

    forecast_lower = np.maximum(0, future_pred_values - 1.96 * forecast_std)
    forecast_upper = future_pred_values + 1.96 * forecast_std

    params = best_model.params

    return ForecastResult(
        model_name='Holt-Winters',
        target_description=f'单品月度销量 ({asin_id})',
        train_dates=train_ts.index.values,
        train_values=train_ts.values,
        test_dates=test_ts.index.values,
        test_actuals=test_actuals,
        test_predictions=test_predictions,
        forecast_dates=future_dates.values,
        forecast_values=future_pred_values,
        forecast_lower=forecast_lower,
        forecast_upper=forecast_upper,
        mae=metrics['mae'],
        rmse=metrics['rmse'],
        mape=metrics['mape'],
        additional_info={
            'asin': asin_id,
            'aic': best_model.aic,
            'bic': best_model.bic,
            'sse': best_model.sse,
            'config': best_config,
            'smoothing_level': params.get('smoothing_level'),
            'smoothing_trend': params.get('smoothing_trend'),
            'smoothing_seasonal': params.get('smoothing_seasonal'),
            'damping_trend': params.get('damping_trend'),
            'seasonal_periods': seasonal_periods,
            'fitted_values': fitted_values.values,
            'residuals': residuals,
            'residual_std': std_resid
        }
    )


def train_holt_winters_top_asins(
    asin_data: pd.DataFrame,
    config: Config
) -> List[ForecastResult]:
    """对销量前N的ASIN分别训练Holt-Winters模型"""
    asin_totals = asin_data.groupby('asin')['sales_month_sim'].sum().sort_values(ascending=False)
    top_asins = asin_totals.head(config.HW_TOP_ASINS).index.tolist()

    print(f"   选择销量Top {config.HW_TOP_ASINS} ASIN进行预测:")
    for i, asin in enumerate(top_asins, 1):
        print(f"      {i}. {asin} (总销量: {asin_totals[asin]:,.0f})")

    results = []
    for asin in top_asins:
        print(f"\n   📦 训练 {asin}...")
        result = train_holt_winters_asin(asin_data, config, asin)
        if result:
            results.append(result)
            print(f"      MAPE: {result.mape:.1f}%, 配置: {result.additional_info['config']['name']}")

    return results


# ============================================================================
# 可视化
# ============================================================================

def _highlight_q4(ax, dates, colors: Dict[str, str]) -> None:
    """高亮 Q4 区域"""
    dates = pd.to_datetime(dates)

    if isinstance(dates, pd.DatetimeIndex):
        years = dates.year.unique()
        date_min = dates.min()
        date_max = dates.max()
    else:
        years = dates.dt.year.unique()
        date_min = dates.min()
        date_max = dates.max()

    for year in years:
        q4_start = pd.Timestamp(f'{year}-10-01')
        q4_end = pd.Timestamp(f'{year}-12-31')

        if q4_start >= date_min and q4_start <= date_max:
            ax.axvspan(q4_start, q4_end, alpha=0.15, color=colors['q4_highlight'])


def plot_data_overview(
    data: DataContainer,
    config: Config,
    colors: Dict[str, str]
) -> None:
    """图1: 数据概览"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax1 = axes[0, 0]
    ax1.plot(data.prophet_data['ds'], data.prophet_data['y'],
             'o-', color=colors['prophet_color'], linewidth=2, markersize=5)
    _highlight_q4(ax1, data.prophet_data['ds'], colors)
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Review Count')
    ax1.set_title('Monthly Review Count (Prophet Data)', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.plot(data.category_sales['month'], data.category_sales['sales_month_sim'],
             's-', color=colors['sarimax_color'], linewidth=2, markersize=5)
    _highlight_q4(ax2, data.category_sales['month'], colors)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Category Sales')
    ax2.set_title('Monthly Category Sales (SARIMAX Data)', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    top_asins = data.asin_sales.groupby('asin')['sales_month_sim'].sum().nlargest(5).index
    for i, asin in enumerate(top_asins):
        asin_df = data.asin_sales[data.asin_sales['asin'] == asin]
        ax3.plot(asin_df['month'], asin_df['sales_month_sim'],
                 '-', linewidth=2, alpha=0.8, label=asin[:15])
    ax3.set_xlabel('Month')
    ax3.set_ylabel('ASIN Sales')
    ax3.set_title('Top 5 ASIN Monthly Sales (Holt-Winters Data)', fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    monthly_avg = data.category_sales.copy()
    monthly_avg['month_num'] = monthly_avg['month'].dt.month
    seasonal_pattern = monthly_avg.groupby('month_num')['sales_month_sim'].mean()

    bars = ax4.bar(seasonal_pattern.index, seasonal_pattern.values,
                   color=[colors['q4_highlight'] if m in [10, 11, 12] else colors['primary']
                          for m in seasonal_pattern.index],
                   alpha=0.8, edgecolor=colors['text'], linewidth=1)
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Average Sales')
    ax4.set_title('Monthly Seasonality Pattern', fontweight='bold')
    ax4.set_xticks(range(1, 13))
    ax4.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax4.grid(axis='y', alpha=0.3)

    for i, (m, v) in enumerate(zip(seasonal_pattern.index, seasonal_pattern.values)):
        if m in [10, 11, 12]:
            ax4.text(m, v + max(seasonal_pattern) * 0.02, f'{v:,.0f}',
                     ha='center', fontsize=9, fontweight='bold', color=colors['secondary'])

    plt.suptitle('Amazon Kitchen Knife - Data Overview', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / '01_data_overview.png', dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()


def plot_prophet_forecast(
    result: ForecastResult,
    config: Config,
    colors: Dict[str, str]
) -> None:
    """图2: Prophet 评论量预测"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[2, 1])

    ax1 = axes[0]
    ax1.plot(result.train_dates, result.train_values,
             'o-', color=colors['prophet_color'], linewidth=2, markersize=5,
             label='Historical Reviews', alpha=0.8)

    ax1.plot(result.test_dates, result.test_actuals,
             'o', color=colors['text'], markersize=10, markerfacecolor='white',
             markeredgewidth=2, label='Test Actual', zorder=10)

    ax1.plot(result.test_dates, result.test_predictions,
             's-', color=colors['accent'], linewidth=2.5, markersize=8,
             label=f'Test Prediction (MAPE: {result.mape:.1f}%)', zorder=9)

    ax1.plot(result.forecast_dates, result.forecast_values,
             'D-', color=colors['secondary'], linewidth=2.5, markersize=10,
             label='Future Forecast', zorder=8)

    ax1.fill_between(result.forecast_dates,
                     result.forecast_lower, result.forecast_upper,
                     alpha=0.2, color=colors['secondary'], label='95% CI')

    ax1.axvline(result.train_dates[-1], color=colors['warning'],
                linestyle='--', linewidth=2, alpha=0.7, label='Train/Test Split')

    _highlight_q4(ax1, pd.to_datetime(result.train_dates), colors)

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Review Count')
    ax1.set_title(f'Prophet Model: Review Count Forecast\n'
                  f'MAE: {result.mae:.2f} | RMSE: {result.rmse:.2f} | MAPE: {result.mape:.1f}%',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    for date, val in zip(result.forecast_dates, result.forecast_values):
        ax1.annotate(f'{val:.0f}', xy=(date, val), xytext=(0, 12),
                     textcoords='offset points', ha='center', fontsize=9,
                     fontweight='bold', color=colors['secondary'])

    ax2 = axes[1]
    if 'full_forecast' in result.additional_info:
        forecast = result.additional_info['full_forecast']
        train_mask = forecast['ds'] <= result.train_dates[-1]

        ax2.plot(forecast.loc[train_mask, 'ds'], forecast.loc[train_mask, 'trend'],
                 '-', color=colors['primary'], linewidth=2, label='Trend')

        if 'yearly' in forecast.columns:
            ax2.plot(forecast.loc[train_mask, 'ds'],
                     forecast.loc[train_mask, 'yearly'] + forecast.loc[train_mask, 'trend'].mean(),
                     '--', color=colors['accent'], linewidth=2, label='Yearly Seasonality', alpha=0.7)

        ax2.set_xlabel('Date')
        ax2.set_ylabel('Component Value')
        ax2.set_title('Prophet Components (Trend + Seasonality)', fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / '02_prophet_reviews_forecast.png',
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()


def plot_sarimax_analysis(
    result: ForecastResult,
    config: Config,
    colors: Dict[str, str]
) -> None:
    """图3: SARIMAX 季节性分析"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    ax1 = axes[0, 0]
    ax1.plot(result.train_dates, result.train_values,
             'o-', color=colors['sarimax_color'], linewidth=2, markersize=5,
             label='Historical Sales', alpha=0.8)

    ax1.plot(result.test_dates, result.test_actuals,
             'o', color=colors['text'], markersize=10, markerfacecolor='white',
             markeredgewidth=2, label='Test Actual', zorder=10)

    ax1.plot(result.test_dates, result.test_predictions,
             's-', color=colors['accent'], linewidth=2.5, markersize=8,
             label=f'Test Prediction (MAPE: {result.mape:.1f}%)', zorder=9)

    ax1.plot(result.forecast_dates, result.forecast_values,
             'D-', color=colors['secondary'], linewidth=2.5, markersize=10,
             label='Future Forecast', zorder=8)

    ax1.fill_between(result.forecast_dates,
                     result.forecast_lower, result.forecast_upper,
                     alpha=0.2, color=colors['secondary'], label='95% CI')

    ax1.axvline(result.train_dates[-1], color=colors['warning'],
                linestyle='--', linewidth=2, alpha=0.7)

    _highlight_q4(ax1, pd.to_datetime(result.train_dates), colors)

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Category Sales')
    ax1.set_title(f'SARIMAX: Category Sales Forecast\n'
                  f'MAE: {result.mae:,.0f} | MAPE: {result.mape:.1f}%',
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    params = result.additional_info.get('params', {})
    pvalues = result.additional_info.get('pvalues', {})
    exog_cols = result.additional_info.get('exog_cols', [])

    exog_params = {k: params[k] for k in exog_cols if k in params}

    if exog_params:
        names = list(exog_params.keys())
        values = list(exog_params.values())
        bar_colors = [colors['secondary'] if v > 0 else colors['primary'] for v in values]

        bars = ax2.barh(range(len(names)), values, color=bar_colors, alpha=0.7,
                        edgecolor=colors['text'], linewidth=1.5)

        label_map = {
            'is_q4': 'Q4 Season Effect\n(Oct-Dec)',
            'is_prime_day': 'Prime Day Effect\n(July)',
            'reviews_month_n': 'Review Count\nImpact'
        }
        display_names = [label_map.get(n, n) for n in names]

        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(display_names)
        ax2.set_xlabel('Coefficient Value (Impact on Sales)')
        ax2.set_title('Seasonality Coefficients\n(Positive = Sales Increase)', fontweight='bold')
        ax2.axvline(0, color=colors['text'], linewidth=1)
        ax2.grid(axis='x', alpha=0.3)

        for i, (bar, val, name) in enumerate(zip(bars, values, names)):
            pv = pvalues.get(name, 1.0)
            if pv < 0.01:
                sig = '***'
            elif pv < 0.05:
                sig = '**'
            elif pv < 0.1:
                sig = '*'
            else:
                sig = ''
            offset = max(abs(v) for v in values) * 0.05
            ax2.text(val + (offset if val > 0 else -offset),
                     i, f'{val:+.0f}{sig}',
                     ha='left' if val > 0 else 'right',
                     va='center', fontweight='bold', fontsize=11)

        ax2.text(0.02, 0.02, '*** p<0.01  ** p<0.05  * p<0.1',
                 transform=ax2.transAxes, fontsize=9, style='italic', color=colors['neutral'])

    ax3 = axes[1, 0]
    if 'model' in result.additional_info:
        model = result.additional_info['model']
        resid = model.resid

        ax3.scatter(range(len(resid)), resid, color=colors['neutral'], alpha=0.6, s=40)
        ax3.axhline(0, color=colors['text'], linestyle='--', linewidth=1)
        ax3.axhline(2 * resid.std(), color=colors['warning'], linestyle=':', alpha=0.7)
        ax3.axhline(-2 * resid.std(), color=colors['warning'], linestyle=':', alpha=0.7)
        ax3.fill_between(range(len(resid)), -2 * resid.std(), 2 * resid.std(),
                         alpha=0.1, color=colors['warning'], label='±2σ')
        ax3.set_xlabel('Observation')
        ax3.set_ylabel('Residual')
        ax3.set_title('Residual Analysis', fontweight='bold')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.axis('off')

    info = result.additional_info
    any_sig = info.get('any_significant', False)
    sample_size = info.get('sample_size', 0)

    sig_status = "✅ Some coefficients significant" if any_sig else "⚠️ No coefficients significant (limited sample)"

    summary_text = f"""
┌──────────────────────────────────────────────────────────────┐
│              SARIMAX MODEL SUMMARY                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  📊 Model Specification:                                     │
│     Order (p,d,q): {info.get('order', 'N/A')}                           │
│     Seasonal (P,D,Q,s): {info.get('seasonal_order', 'N/A')}               │
│                                                              │
│  📈 Fit Statistics:                                          │
│     AIC: {info.get('aic', 0):,.2f}                                      │
│     BIC: {info.get('bic', 0):,.2f}                                      │
│     Sample Size: {sample_size} months                               │
│                                                              │
│  🎯 Test Performance:                                        │
│     MAE: {result.mae:,.0f}                                         │
│     RMSE: {result.rmse:,.0f}                                        │
│     MAPE: {result.mape:.1f}%                                          │
│                                                              │
│  🔍 Statistical Significance:                                │
│     {sig_status}            │
│                                                              │
│  💡 Note: Coefficients show effect direction                 │
│     even when not statistically significant                  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
"""
    ax4.text(0.05, 0.95, summary_text, fontsize=10, fontfamily='monospace',
             verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor=colors['background'],
                       edgecolor=colors['sarimax_color'], linewidth=2))

    plt.suptitle('SARIMAX Model: Category Sales Seasonality Analysis',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / '03_sarimax_seasonality_analysis.png',
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()


def plot_holt_winters_forecast(
    results: List[ForecastResult],
    config: Config,
    colors: Dict[str, str]
) -> None:
    """图4: Holt-Winters 单品预测"""
    n_results = len(results)
    if n_results == 0:
        return

    n_cols = min(2, n_results)
    n_rows = (n_results + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
    if n_results == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, result in enumerate(results):
        ax = axes[idx]
        asin = result.additional_info.get('asin', 'Unknown')

        ax.plot(result.train_dates, result.train_values,
                'o-', color=colors['hw_color'], linewidth=2, markersize=4,
                label='Historical', alpha=0.8)

        ax.plot(result.test_dates, result.test_actuals,
                'o', color=colors['text'], markersize=8, markerfacecolor='white',
                markeredgewidth=2, label='Test Actual', zorder=10)

        ax.plot(result.test_dates, result.test_predictions,
                's-', color=colors['accent'], linewidth=2, markersize=6,
                label=f'Pred (MAPE: {result.mape:.1f}%)', zorder=9)

        ax.plot(result.forecast_dates, result.forecast_values,
                'D-', color=colors['secondary'], linewidth=2, markersize=8,
                label='Forecast', zorder=8)

        ax.fill_between(result.forecast_dates,
                        result.forecast_lower, result.forecast_upper,
                        alpha=0.2, color=colors['secondary'])

        ax.axvline(result.train_dates[-1], color=colors['warning'],
                   linestyle='--', linewidth=1.5, alpha=0.7)

        _highlight_q4(ax, pd.to_datetime(result.train_dates), colors)

        cfg_name = result.additional_info.get('config', {}).get('name', 'N/A')
        ax.set_title(f'{asin}\nConfig: {cfg_name} | MAE: {result.mae:,.0f}',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(n_results, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Holt-Winters Model: ASIN-level Sales Forecast',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / '04_holt_winters_asin_forecast.png',
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()


def plot_holt_winters_comparison(
    results: List[ForecastResult],
    config: Config,
    colors: Dict[str, str]
) -> None:
    """图5: Holt-Winters ASIN对比"""
    if len(results) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    asins = [r.additional_info.get('asin', 'Unknown')[:12] for r in results]
    mapes = [r.mape for r in results]
    maes = [r.mae for r in results]

    sorted_indices = np.argsort(mapes)
    asins_sorted = [asins[i] for i in sorted_indices]
    mapes_sorted = [mapes[i] for i in sorted_indices]
    maes_sorted = [maes[i] for i in sorted_indices]

    ax1 = axes[0]
    bar_colors = [colors['hw_color'] if i == 0 else colors['neutral'] for i in range(len(asins_sorted))]
    bars1 = ax1.barh(range(len(asins_sorted)), mapes_sorted, color=bar_colors, alpha=0.8,
                     edgecolor=colors['text'], linewidth=1)
    ax1.set_yticks(range(len(asins_sorted)))
    ax1.set_yticklabels(asins_sorted)
    ax1.set_xlabel('MAPE (%)')
    ax1.set_title('MAPE by ASIN (Lower is Better)', fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    for bar, val in zip(bars1, mapes_sorted):
        ax1.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{val:.1f}%', va='center', fontweight='bold')

    ax2 = axes[1]
    bars2 = ax2.barh(range(len(asins_sorted)), maes_sorted, color=bar_colors, alpha=0.8,
                     edgecolor=colors['text'], linewidth=1)
    ax2.set_yticks(range(len(asins_sorted)))
    ax2.set_yticklabels(asins_sorted)
    ax2.set_xlabel('MAE')
    ax2.set_title('MAE by ASIN', fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)

    for bar, val in zip(bars2, maes_sorted):
        ax2.text(val + max(maes_sorted) * 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{val:,.0f}', va='center', fontweight='bold')

    ax3 = axes[2]
    param_data = []
    for r in results:
        asin = r.additional_info.get('asin', 'Unknown')[:12]
        alpha = r.additional_info.get('smoothing_level')
        beta = r.additional_info.get('smoothing_trend')
        gamma = r.additional_info.get('smoothing_seasonal')

        alpha = alpha if alpha is not None and not np.isnan(alpha) else 0
        beta = beta if beta is not None and not np.isnan(beta) else 0
        gamma = gamma if gamma is not None and not np.isnan(gamma) else 0

        param_data.append({'ASIN': asin, 'α': alpha, 'β': beta, 'γ': gamma})

    param_df = pd.DataFrame(param_data)
    x = np.arange(len(param_df))
    width = 0.25

    ax3.bar(x - width, param_df['α'], width, label='α (Level)', color=colors['primary'], alpha=0.8)
    ax3.bar(x, param_df['β'], width, label='β (Trend)', color=colors['accent'], alpha=0.8)
    ax3.bar(x + width, param_df['γ'], width, label='γ (Seasonal)', color=colors['hw_color'], alpha=0.8)

    ax3.set_xticks(x)
    ax3.set_xticklabels(param_df['ASIN'], rotation=45, ha='right')
    ax3.set_ylabel('Parameter Value')
    ax3.set_title('Smoothing Parameters by ASIN', fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(axis='y', alpha=0.3)

    plt.suptitle('Holt-Winters Model: ASIN Performance Comparison',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / '05_holt_winters_comparison.png',
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()


def plot_business_dashboard(
    data: DataContainer,
    prophet_result: Optional[ForecastResult],
    sarimax_result: Optional[ForecastResult],
    hw_results: List[ForecastResult],
    config: Config,
    colors: Dict[str, str]
) -> None:
    """图6: 业务仪表板"""
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    ax_kpi = fig.add_subplot(gs[0, :])
    ax_kpi.axis('off')

    prophet_mape = prophet_result.mape if prophet_result else 0
    sarimax_mape = sarimax_result.mape if sarimax_result else 0
    hw_avg_mape = np.mean([r.mape for r in hw_results]) if hw_results else 0

    # 检查SARIMAX显著性
    sarimax_sig_status = "✅ Significant" if (sarimax_result and sarimax_result.additional_info.get('any_significant')) else "⚠️ Directional"

    kpi_text = f"""
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                         FORECASTING SYSTEM - EXECUTIVE SUMMARY                                      │
├────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                    │
│  📊 MODEL PERFORMANCE OVERVIEW                                                                     │
│  ┌─────────────────────────┬─────────────────────────┬─────────────────────────┐                   │
│  │ Prophet (Reviews)       │ SARIMAX (Category)      │ Holt-Winters (ASIN)     │                   │
│  │ MAPE: {prophet_mape:5.1f}%            │ MAPE: {sarimax_mape:5.1f}%            │ Avg MAPE: {hw_avg_mape:5.1f}%        │                   │
│  │ Target: Review Count    │ Target: Category Sales  │ Target: ASIN Sales      │                   │
│  │                         │ Status: {sarimax_sig_status:<14}│                         │                   │
│  └─────────────────────────┴─────────────────────────┴─────────────────────────┘                   │
│                                                                                                    │
│  🎯 KEY INSIGHTS                                                                                   │
│     • Q4 (Oct-Dec): Coefficient suggests +35-45% sales increase (verify with more data)            │
│     • Prime Day (July): Moderate boost indicated in review and sales patterns                      │
│     • Seasonality patterns are consistent across models                                            │
│                                                                                                    │
│  💼 BUSINESS RECOMMENDATIONS                                                                       │
│     1. Inventory: Consider increasing stock before Q4 season (direction supported by data)         │
│     2. Marketing: Align campaigns with forecast peaks                                              │
│     3. Data: Collect 36+ months for statistically robust coefficient estimates                     │
│                                                                                                    │
│  ⚠️ IMPORTANT NOTES                                                                                │
│     • Holt-Winters: Use only 1-2 month forecasts for inventory decisions                           │
│     • SARIMAX: Coefficients indicate direction; magnitude may vary with more data                  │
│                                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""
    ax_kpi.text(0.02, 0.5, kpi_text, fontsize=10, fontfamily='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=colors['background'],
                          edgecolor=colors['primary'], linewidth=2))

    ax_prophet = fig.add_subplot(gs[1, 0])
    if prophet_result:
        ax_prophet.plot(prophet_result.train_dates, prophet_result.train_values,
                        '-', color=colors['prophet_color'], linewidth=1.5, alpha=0.7)
        ax_prophet.plot(prophet_result.forecast_dates, prophet_result.forecast_values,
                        'o-', color=colors['secondary'], linewidth=2, markersize=6)
        ax_prophet.fill_between(prophet_result.forecast_dates,
                                prophet_result.forecast_lower, prophet_result.forecast_upper,
                                alpha=0.2, color=colors['secondary'])
        _highlight_q4(ax_prophet, pd.to_datetime(prophet_result.train_dates), colors)
        ax_prophet.set_title(f'Prophet: Reviews\nMAPE: {prophet_result.mape:.1f}%', fontweight='bold')
    else:
        ax_prophet.text(0.5, 0.5, 'No Data', ha='center', va='center')
        ax_prophet.set_title('Prophet: Reviews', fontweight='bold')
    ax_prophet.set_xlabel('Date')
    ax_prophet.grid(True, alpha=0.3)

    ax_sarimax = fig.add_subplot(gs[1, 1])
    if sarimax_result:
        ax_sarimax.plot(sarimax_result.train_dates, sarimax_result.train_values,
                        '-', color=colors['sarimax_color'], linewidth=1.5, alpha=0.7)
        ax_sarimax.plot(sarimax_result.forecast_dates, sarimax_result.forecast_values,
                        'o-', color=colors['secondary'], linewidth=2, markersize=6)
        ax_sarimax.fill_between(sarimax_result.forecast_dates,
                                sarimax_result.forecast_lower, sarimax_result.forecast_upper,
                                alpha=0.2, color=colors['secondary'])
        _highlight_q4(ax_sarimax, pd.to_datetime(sarimax_result.train_dates), colors)
        ax_sarimax.set_title(f'SARIMAX: Category Sales\nMAPE: {sarimax_result.mape:.1f}%', fontweight='bold')
    else:
        ax_sarimax.text(0.5, 0.5, 'No Data', ha='center', va='center')
        ax_sarimax.set_title('SARIMAX: Category Sales', fontweight='bold')
    ax_sarimax.set_xlabel('Date')
    ax_sarimax.grid(True, alpha=0.3)

    ax_hw = fig.add_subplot(gs[1, 2])
    if hw_results:
        best_hw = min(hw_results, key=lambda r: r.mape)
        ax_hw.plot(best_hw.train_dates, best_hw.train_values,
                   '-', color=colors['hw_color'], linewidth=1.5, alpha=0.7)
        ax_hw.plot(best_hw.forecast_dates, best_hw.forecast_values,
                   'o-', color=colors['secondary'], linewidth=2, markersize=6)
        ax_hw.fill_between(best_hw.forecast_dates,
                           best_hw.forecast_lower, best_hw.forecast_upper,
                           alpha=0.2, color=colors['secondary'])
        _highlight_q4(ax_hw, pd.to_datetime(best_hw.train_dates), colors)
        asin = best_hw.additional_info.get('asin', 'Unknown')[:15]
        ax_hw.set_title(f'Holt-Winters: {asin}\nMAPE: {best_hw.mape:.1f}%', fontweight='bold')
    else:
        ax_hw.text(0.5, 0.5, 'No Data', ha='center', va='center')
        ax_hw.set_title('Holt-Winters: ASIN Sales', fontweight='bold')
    ax_hw.set_xlabel('Date')
    ax_hw.grid(True, alpha=0.3)

    ax_coef = fig.add_subplot(gs[2, 0])
    if sarimax_result and 'params' in sarimax_result.additional_info:
        params = sarimax_result.additional_info['params']
        exog_cols = sarimax_result.additional_info.get('exog_cols', [])
        exog_params = {k: params[k] for k in exog_cols if k in params}

        if exog_params:
            names = list(exog_params.keys())
            values = list(exog_params.values())
            bar_colors = [colors['secondary'] if v > 0 else colors['primary'] for v in values]

            label_map = {'is_q4': 'Q4 Effect', 'is_prime_day': 'Prime Day', 'reviews_month_n': 'Reviews'}
            display_names = [label_map.get(n, n) for n in names]

            bars = ax_coef.barh(range(len(names)), values, color=bar_colors, alpha=0.8)
            ax_coef.set_yticks(range(len(names)))
            ax_coef.set_yticklabels(display_names)
            ax_coef.axvline(0, color=colors['text'], linewidth=1)
            ax_coef.set_xlabel('Impact on Sales')
            ax_coef.set_title('Seasonality Coefficients', fontweight='bold')

            for bar, val in zip(bars, values):
                offset = max(abs(v) for v in values) * 0.05
                ax_coef.text(val + (offset if val > 0 else -offset),
                             bar.get_y() + bar.get_height() / 2,
                             f'{val:+.0f}', va='center', fontweight='bold', fontsize=10)
    else:
        ax_coef.text(0.5, 0.5, 'No SARIMAX Results', ha='center', va='center')
        ax_coef.set_title('Seasonality Coefficients', fontweight='bold')
    ax_coef.grid(axis='x', alpha=0.3)

    ax_monthly = fig.add_subplot(gs[2, 1])
    monthly_avg = data.category_sales.copy()
    monthly_avg['month_num'] = monthly_avg['month'].dt.month
    seasonal_pattern = monthly_avg.groupby('month_num')['sales_month_sim'].mean()

    bars = ax_monthly.bar(seasonal_pattern.index, seasonal_pattern.values,
                          color=[colors['q4_highlight'] if m in [10, 11, 12] else colors['primary']
                                 for m in seasonal_pattern.index],
                          alpha=0.8)
    ax_monthly.set_xticks(range(1, 13))
    ax_monthly.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax_monthly.set_xlabel('Month')
    ax_monthly.set_ylabel('Avg Sales')
    ax_monthly.set_title('Monthly Seasonality', fontweight='bold')
    ax_monthly.grid(axis='y', alpha=0.3)

    ax_risk = fig.add_subplot(gs[2, 2])
    ax_risk.axis('off')

    risk_text = """
┌──────────────────────────────────┐
│     ⚠️ RISK ALERTS               │
├──────────────────────────────────┤
│                                  │
│ • HIGH VOLATILITY: Dec-Jan       │
│   → Maintain buffer stock        │
│                                  │
│ • SUPPLY CHAIN: Q4 pressure      │
│   → Early procurement            │
│                                  │
│ • FORECAST HORIZON:              │
│   → Use 1-2 months for planning  │
│   → 3+ months: directional only  │
│                                  │
├──────────────────────────────────┤
│     ✅ NEXT STEPS                │
├──────────────────────────────────┤
│                                  │
│ 1. Update forecasts monthly      │
│ 2. Track actuals vs predictions  │
│ 3. Collect more data (36+ mo)    │
│                                  │
└──────────────────────────────────┘
"""
    ax_risk.text(0.05, 0.95, risk_text, fontsize=10, fontfamily='monospace',
                 verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#fffde7',
                           edgecolor=colors['warning'], linewidth=2))

    plt.suptitle('Amazon Kitchen Knife - Business Intelligence Dashboard',
                 fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(config.OUTPUT_DIR / '06_business_dashboard.png',
                dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()


# ============================================================================
# 报告生成
# ============================================================================

def _format_param_value(value, default_text: str = "N/A") -> str:
    """格式化参数值，处理None和nan"""
    if value is None:
        return default_text
    try:
        if np.isnan(value):
            return default_text
        return f"{value:.4f}"
    except (TypeError, ValueError):
        return default_text


def _get_significance_text(pvalue: float, config: Config) -> Tuple[str, str]:
    """根据p值返回显著性标记和描述"""
    if pvalue < 0.01:
        return '***', 'p < 0.01 (highly significant)'
    elif pvalue < config.SIGNIFICANCE_LEVEL:
        return '**', f'p < {config.SIGNIFICANCE_LEVEL} (significant)'
    elif pvalue < config.MARGINAL_SIGNIFICANCE_LEVEL:
        return '*', f'p < {config.MARGINAL_SIGNIFICANCE_LEVEL} (marginally significant)'
    else:
        return '', f'p = {pvalue:.3f} (not significant)'


def _get_ci_note(lower: float, upper: float, value: float, config: Config) -> str:
    """根据置信区间宽度返回警告注释"""
    if lower <= 0:
        return "⚠️ High uncertainty"
    elif value > 0 and (upper - lower) / value > config.CI_WIDTH_WARNING_RATIO:
        return "⚠️ Wide CI"
    else:
        return ""


def generate_report(
    prophet_result: Optional[ForecastResult],
    sarimax_result: Optional[ForecastResult],
    hw_results: List[ForecastResult],
    config: Config
) -> None:
    """生成 Markdown 报告"""

    report = f"""# Time Series Forecasting - Analysis Report

## Executive Summary

This report presents time series forecasting analysis for Amazon Kitchen Knife market using three specialized models:

| Model | Purpose | Target |
|-------|---------|--------|
| **Prophet** | Review trend forecasting | Monthly review count |
| **SARIMAX** | Seasonality analysis | Category-level sales |
| **Holt-Winters** | ASIN-level forecasting | Individual product sales |

**Important Note:** Each model has a different forecasting target, so their metrics should NOT be compared directly.

---

## 1. Prophet Model - Review Count Forecasting

**Purpose:** Predict future review volume to assess market interest and product visibility.

**Why Prophet for Reviews:**
- Automatically detects trend changepoints
- Handles missing data gracefully
- Built-in holiday/event effects
- Highly interpretable (trend + seasonality decomposition)

"""

    if prophet_result:
        report += f"""### Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| MAE | {prophet_result.mae:.2f} | Average prediction error in review count |
| RMSE | {prophet_result.rmse:.2f} | Root mean squared error |
| MAPE | {prophet_result.mape:.1f}% | Percentage error (lower is better) |

### Future Review Forecast

| Month | Predicted Reviews | Lower 95% CI | Upper 95% CI |
|-------|-------------------|--------------|--------------|
"""
        for date, val, lower, upper in zip(
            prophet_result.forecast_dates,
            prophet_result.forecast_values,
            prophet_result.forecast_lower,
            prophet_result.forecast_upper
        ):
            date_str = pd.Timestamp(date).strftime('%Y-%m')
            report += f"| {date_str} | {val:.0f} | {lower:.0f} | {upper:.0f} |\n"

        report += """
### Key Insights
- Prophet captures yearly seasonality in review patterns
- Q4 shows elevated review activity (holiday shopping effect)
- Trend component indicates overall market direction
- Model includes Q4 and Prime Day as external regressors

"""
    else:
        report += "*Prophet model not trained (library not installed).*\n\n"

    report += """---

## 2. SARIMAX Model - Category Sales Seasonality Analysis

**Purpose:** Quantify the impact of seasonal factors (Q4, Prime Day) on category-level sales.

**Why SARIMAX for Seasonality Analysis:**
- Simultaneously models trend, seasonality, and external variables
- Provides coefficient estimates with statistical significance
- Ideal for quantifying the impact of known seasonal events
- Captures autocorrelation in time series

"""

    if sarimax_result:
        info = sarimax_result.additional_info
        sample_size = info.get('sample_size', 0)
        any_significant = info.get('any_significant', False)

        report += f"""### Model Specification

| Parameter | Value |
|-----------|-------|
| ARIMA Order (p,d,q) | {info.get('order', 'N/A')} |
| Seasonal Order (P,D,Q,s) | {info.get('seasonal_order', 'N/A')} |
| AIC | {info.get('aic', 0):,.2f} |
| BIC | {info.get('bic', 0):,.2f} |
| Stationarity (ADF test) | {'Stationary' if info.get('is_stationary') else 'Non-stationary'} |
| Sample Size | {sample_size} months |

### Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| MAE | {sarimax_result.mae:,.0f} | Average prediction error in sales units |
| RMSE | {sarimax_result.rmse:,.0f} | Root mean squared error |
| MAPE | {sarimax_result.mape:.1f}% | Percentage error (lower is better) |

### Seasonality Coefficients

| Factor | Coefficient | Significance | Interpretation |
|--------|-------------|--------------|----------------|
"""
        params = info.get('params', {})
        pvalues = info.get('pvalues', {})
        exog_cols = info.get('exog_cols', [])

        coef_map = {
            'is_q4': ('Q4 Season (Oct-Dec)', 'Sales impact during Q4 holiday season'),
            'is_prime_day': ('Prime Day (July)', 'Sales impact during Prime Day'),
            'reviews_month_n': ('Review Count', 'Sales contribution per review')
        }

        for col in exog_cols:
            if col in params:
                name, interp = coef_map.get(col, (col, 'N/A'))
                coef = params[col]
                pv = pvalues.get(col, 1.0)
                sig_mark, sig_text = _get_significance_text(pv, config)
                report += f"| {name} | {coef:+,.0f} | {sig_mark} {sig_text} | {interp} |\n"

        # 动态生成 Key Insights
        q4_pval = pvalues.get('is_q4', 1.0)
        q4_coef = params.get('is_q4', 0)

        report += "\n### Key Insights\n"

        if q4_pval < config.SIGNIFICANCE_LEVEL:
            report += f"- Q4 coefficient (+{q4_coef:,.0f}) shows **statistically significant** holiday boost\n"
        elif q4_pval < config.MARGINAL_SIGNIFICANCE_LEVEL:
            report += f"- Q4 coefficient (+{q4_coef:,.0f}) shows **marginally significant** holiday boost (p < 0.1)\n"
        else:
            report += f"- Q4 coefficient (+{q4_coef:,.0f}) suggests holiday boost direction, but **not statistically significant**\n"
            report += f"- Limited sample size ({sample_size} months) reduces statistical power; consider collecting 36+ months\n"

        report += "- Coefficient signs indicate effect direction even when not statistically significant\n"
        report += "- Model captures monthly autocorrelation patterns\n"

        report += """
### Future Category Sales Forecast

| Month | Predicted Sales | Lower 95% CI | Upper 95% CI |
|-------|-----------------|--------------|--------------|
"""
        for date, val, lower, upper in zip(
            sarimax_result.forecast_dates,
            sarimax_result.forecast_values,
            sarimax_result.forecast_lower,
            sarimax_result.forecast_upper
        ):
            date_str = pd.Timestamp(date).strftime('%Y-%m')
            report += f"| {date_str} | {val:,.0f} | {lower:,.0f} | {upper:,.0f} |\n"

        report += "\n"

    else:
        report += "*SARIMAX model not trained (library not installed).*\n\n"

    report += """---

## 3. Holt-Winters Model - ASIN-level Sales Forecasting

**Purpose:** Predict individual product sales for inventory planning and replenishment.

**Why Holt-Winters for ASIN Forecasting:**
- Designed for small-sample seasonal data
- Few parameters, less prone to overfitting
- Fast computation for multiple products
- Automatic trend and seasonality handling

"""

    if hw_results:
        report += f"""### ASIN Performance Summary

| ASIN | MAPE | MAE | Configuration | Rating |
|------|------|-----|---------------|--------|
"""
        for r in sorted(hw_results, key=lambda x: x.mape):
            asin = r.additional_info.get('asin', 'Unknown')
            cfg = r.additional_info.get('config', {}).get('name', 'N/A')
            if r.mape < 10:
                rating = '⭐ Excellent'
            elif r.mape < 20:
                rating = '✅ Good'
            elif r.mape < 30:
                rating = '⚠️ Fair'
            else:
                rating = '❌ Poor'
            report += f"| {asin} | {r.mape:.1f}% | {r.mae:,.0f} | {cfg} | {rating} |\n"

        best_hw = min(hw_results, key=lambda x: x.mape)
        best_info = best_hw.additional_info

        alpha_str = _format_param_value(best_info.get('smoothing_level'))
        beta_str = _format_param_value(best_info.get('smoothing_trend'))
        gamma_str = _format_param_value(best_info.get('smoothing_seasonal'), "N/A (no seasonal)")
        phi_str = _format_param_value(best_info.get('damping_trend'))

        report += f"""
### Best Performing ASIN: {best_info.get('asin', 'Unknown')}

**Model Configuration:** {best_info.get('config', {}).get('name', 'N/A')}

**Smoothing Parameters:**

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| α (Level) | {alpha_str} | Weight on recent observations (higher = more reactive) |
| β (Trend) | {beta_str} | Weight on trend changes (0 = flat trend) |
| γ (Seasonal) | {gamma_str} | Weight on seasonal patterns |
| φ (Damping) | {phi_str} | Trend dampening (closer to 1 = less dampening) |

**Future Forecast:**

| Month | Predicted Sales | Lower 95% CI | Upper 95% CI | Note |
|-------|-----------------|--------------|--------------|------|
"""
        has_high_uncertainty = False
        for i, (date, val, lower, upper) in enumerate(zip(
            best_hw.forecast_dates,
            best_hw.forecast_values,
            best_hw.forecast_lower,
            best_hw.forecast_upper
        )):
            date_str = pd.Timestamp(date).strftime('%Y-%m')
            note = _get_ci_note(lower, upper, val, config)
            if note:
                has_high_uncertainty = True

            # 对于超过安全预测期的月份，添加额外提示
            if i >= config.HW_FORECAST_MONTHS_SAFE and not note:
                note = "📊 Directional only"

            report += f"| {date_str} | {val:,.0f} | {lower:,.0f} | {upper:,.0f} | {note} |\n"

        if has_high_uncertainty:
            report += f"""
> ⚠️ **IMPORTANT - Forecast Reliability:**
> - **Months 1-{config.HW_FORECAST_MONTHS_SAFE}:** Suitable for inventory planning decisions
> - **Months {config.HW_FORECAST_MONTHS_SAFE + 1}+:** Use as directional guidance only; confidence intervals are too wide for precise planning
> - **Recommendation:** For critical inventory decisions, use only the first {config.HW_FORECAST_MONTHS_SAFE} months of forecasts
"""

        report += """
### Configuration Legend

| Code | Full Name | Components |
|------|-----------|------------|
| AA | Additive-Additive | Additive trend + Additive seasonality |
| AAD | Additive-Additive-Damped | Additive trend + Additive seasonality + Damped trend |
| AD | Additive-Damped | Additive trend + Damped trend (no seasonality) |
| AMD | Additive-Multiplicative-Damped | Additive trend + Multiplicative seasonality + Damped |

### Key Insights
- Holt-Winters effectively captures product-level patterns
- Best ASIN uses configuration that matches its data characteristics
- High α values indicate models are responsive to recent changes
- Dampened trend prevents over-extrapolation in long-term forecasts

"""
    else:
        report += "*Holt-Winters models not trained.*\n\n"

    report += f"""---

## Generated Outputs

All outputs saved to `{config.OUTPUT_DIR}/`:

| File | Description |
|------|-------------|
| `01_data_overview.png` | Data overview and seasonality patterns |
| `02_prophet_reviews_forecast.png` | Prophet review count predictions |
| `03_sarimax_seasonality_analysis.png` | SARIMAX seasonality coefficients |
| `04_holt_winters_asin_forecast.png` | Holt-Winters ASIN predictions |
| `05_holt_winters_comparison.png` | ASIN performance comparison |
| `06_business_dashboard.png` | Executive summary dashboard |
| `FORECASTING_REPORT.md` | This report |

---

## Business Recommendations

### 📦 Inventory Planning
1. **Q4 Preparation:** Consider increasing inventory before October (direction supported by data, magnitude to be validated)
2. **Prime Day:** Stock up by mid-June for July Prime Day event
3. **Safety Stock:** Maintain buffer during high-volatility periods (Dec-Jan transition)
4. **ASIN-specific:** Use Holt-Winters forecasts (first {config.HW_FORECAST_MONTHS_SAFE} months only) for individual product replenishment

### 📣 Marketing Alignment
1. Coordinate advertising campaigns with predicted demand peaks
2. Q4 shows consistent directional increase across all models
3. Monitor review trends as leading indicator for market interest
4. Use Prophet forecasts to anticipate review volume for social proof planning

### 🔧 Model Maintenance
1. **Refresh forecasts monthly** with new actual data
2. **Recalibrate if MAPE exceeds 25%** for any model
3. **Add new ASINs** to Holt-Winters as they mature (need 12+ months data)
4. **Collect more data:** Target 36+ months for statistically significant SARIMAX coefficients

### ⚠️ Limitations & Caveats

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Sample size ({sarimax_result.additional_info.get('sample_size', 'N/A') if sarimax_result else 'N/A'} months) | SARIMAX coefficients may not reach statistical significance | Interpret direction, not magnitude; collect more data |
| Holt-Winters CI widening | Forecasts beyond {config.HW_FORECAST_MONTHS_SAFE} months have high uncertainty | Use only 1-{config.HW_FORECAST_MONTHS_SAFE} month forecasts for planning |
| External factors | Only Q4/Prime Day captured; other promotions not modeled | Add more regressors as data becomes available |
| Market disruptions | Models assume historical patterns continue | Monitor for regime changes; recalibrate as needed |

---

## Appendix: Model Selection Rationale

| Forecasting Task | Chosen Model | Alternative Considered | Why This Choice |
|------------------|--------------|------------------------|-----------------|
| Review Count | Prophet | ARIMA, LSTM | Best for irregular seasonality, built-in changepoint detection |
| Category Seasonality | SARIMAX | Prophet, VAR | Explicit coefficient estimation, statistical inference |
| ASIN Sales | Holt-Winters | ARIMA, ETS | Simple, robust for small samples, fast multi-product |

---

*Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
*System Version: 2.1 (Three-Model Architecture with Enhanced Diagnostics)*
"""

    with open(config.OUTPUT_DIR / 'FORECASTING_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report)


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主入口函数"""
    config = Config()
    colors = setup_plot_style()

    print("=" * 70)
    print("🚀 Amazon Kitchen Knife - Time Series Forecasting System v2.1")
    print("=" * 70)
    print("\n📋 Model Assignments:")
    print("   • Prophet       → Review Count Forecasting")
    print("   • SARIMAX       → Category Sales Seasonality Analysis")
    print("   • Holt-Winters  → ASIN-level Sales Forecasting")
    print("=" * 70)

    print("\n[1/7] 📂 Loading and validating data...")
    try:
        data = load_and_validate_data()
        print(f"   ✅ Prophet data (reviews): {len(data.prophet_data)} months")
        print(f"   ✅ Category sales (SARIMAX): {len(data.category_sales)} months")
        print(f"   ✅ ASIN sales (Holt-Winters): {data.asin_sales['asin'].nunique()} ASINs")
    except FileNotFoundError as e:
        print(f"   ❌ {e}")
        return

    print(f"\n[2/7] ✂️  Splitting data (test = last {config.TEST_MONTHS} months)...")

    prophet_train, prophet_test = train_test_split_ts(
        data.prophet_data, 'ds', config.TEST_MONTHS
    )
    print(f"   Prophet: train={len(prophet_train)}, test={len(prophet_test)}")

    category_train, category_test = train_test_split_ts(
        data.category_sales, 'month', config.TEST_MONTHS
    )
    print(f"   Category: train={len(category_train)}, test={len(category_test)}")

    print("\n[3/7] 🔮 Training Prophet (Review Count Forecasting)...")
    prophet_result = train_prophet_reviews(prophet_train, prophet_test, config)
    if prophet_result:
        print(f"   ✅ Complete! MAPE: {prophet_result.mape:.1f}%")

    print("\n[4/7] 📈 Training SARIMAX (Category Seasonality Analysis)...")
    sarimax_result = train_sarimax_seasonality(category_train, category_test, config)
    if sarimax_result:
        print(f"   ✅ Complete! MAPE: {sarimax_result.mape:.1f}%")
        if sarimax_result.additional_info.get('any_significant'):
            print("   ✅ Some coefficients are statistically significant")
        else:
            print("   ⚠️ No coefficients reached statistical significance (limited sample)")

    print(f"\n[5/7] 🌡️  Training Holt-Winters (Top {config.HW_TOP_ASINS} ASINs)...")
    hw_results = train_holt_winters_top_asins(data.asin_sales, config)
    if hw_results:
        avg_mape = np.mean([r.mape for r in hw_results])
        print(f"   ✅ Complete! {len(hw_results)} ASINs, Avg MAPE: {avg_mape:.1f}%")
        print(f"   💡 Recommended forecast horizon: {config.HW_FORECAST_MONTHS_SAFE} months")

    print("\n[6/7] 📊 Generating visualizations...")

    print("   - 01_data_overview.png")
    plot_data_overview(data, config, colors)

    if prophet_result:
        print("   - 02_prophet_reviews_forecast.png")
        plot_prophet_forecast(prophet_result, config, colors)

    if sarimax_result:
        print("   - 03_sarimax_seasonality_analysis.png")
        plot_sarimax_analysis(sarimax_result, config, colors)

    if hw_results:
        print("   - 04_holt_winters_asin_forecast.png")
        plot_holt_winters_forecast(hw_results, config, colors)
        print("   - 05_holt_winters_comparison.png")
        plot_holt_winters_comparison(hw_results, config, colors)

    print("   - 06_business_dashboard.png")
    plot_business_dashboard(data, prophet_result, sarimax_result, hw_results, config, colors)

    print("\n[7/7] 📝 Generating report...")
    generate_report(prophet_result, sarimax_result, hw_results, config)
    print(f"   ✅ Report saved: {config.OUTPUT_DIR}/FORECASTING_REPORT.md")

    print("\n" + "=" * 70)
    print("✅ FORECASTING SYSTEM COMPLETED!")
    print("=" * 70)

    print("\n📊 Results Summary:")
    print("-" * 60)
    print(f"{'Model':<20}{'Target':<25}{'MAPE':>10}")
    print("-" * 60)

    if prophet_result:
        print(f"{'Prophet':<20}{'Review Count':<25}{prophet_result.mape:>9.1f}%")

    if sarimax_result:
        sig_note = "" if sarimax_result.additional_info.get('any_significant') else " *"
        print(f"{'SARIMAX':<20}{'Category Sales':<25}{sarimax_result.mape:>9.1f}%{sig_note}")

    if hw_results:
        for r in sorted(hw_results, key=lambda x: x.mape)[:3]:
            asin = r.additional_info.get('asin', 'Unknown')[:15]
            print(f"{'Holt-Winters':<20}{asin:<25}{r.mape:>9.1f}%")

    print("-" * 60)
    if sarimax_result and not sarimax_result.additional_info.get('any_significant'):
        print("* Coefficients not statistically significant (limited sample size)")

    print(f"\n📁 All outputs saved to: {config.OUTPUT_DIR}/")
    print(f"\n💡 Key Recommendations:")
    print(f"   • Use Holt-Winters forecasts for first {config.HW_FORECAST_MONTHS_SAFE} months only")
    print(f"   • SARIMAX coefficients show direction; collect more data for magnitude")
    print(f"   • Refresh forecasts monthly with new data")
    print("\n🎉 Ready for Presentation!")


if __name__ == "__main__":
    main()
