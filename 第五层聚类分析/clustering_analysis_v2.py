"""
厨刀市场聚类分析 - 优化版 v3.0
================================
核心改进：
1. 特征重构：情感8维→2维，移除冗余二值特征
2. K-Prototypes：正确处理混合数据类型
3. PCA降维后再聚类（连续特征部分）
4. Gap Statistic + Bootstrap稳定性验证
5. 更可靠的K值选择逻辑

依赖安装：
    pip install kmodes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score
)
from scipy.cluster.hierarchy import dendrogram, linkage

try:
    from kmodes.kprototypes import KPrototypes
    HAS_KMODES = True
except ImportError:
    HAS_KMODES = False
    print("⚠️  kmodes 未安装，将使用改进版 K-Means 作为备选")
    print("   安装命令: pip install kmodes\n")

warnings.filterwarnings('ignore')

# ============================================================================
# 全局配置
# ============================================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

CLUSTER_COLORS = [
    '#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12',
    '#1ABC9C', '#E91E63', '#34495E', '#00BCD4', '#FF5722',
]

# ============================================================================
# 特征定义（v3.0 重新分类）
# ============================================================================

# 连续型特征（适合标准化+欧氏距离）
CONTINUOUS_FEATURES = [
    'log_price_per_piece',   # 单价（保留，删除 log_price 避免冗余）
    'product_rating',
    'log_reviews',
    'log_sales',
    'log_bsr',
    'weighted_rating',
    'discount_rate',
    'blade_size_inch',
    'log_days_on_market',
    'verified_purchase_rate',
    'avg_helpful_votes',
    'set_pieces',            # 比 is_set 信息量更大
    # 以下两列由情感8维合并而来（在 preprocess 中生成）
    'sentiment_avg',
    'sentiment_std',
    'positive_ratio',
    'bullet_count',
    'image_count',
]

# 分类/二值型特征（适合汉明距离，K-Prototypes专用）
CATEGORICAL_FEATURES = [
    'is_fba',
    'has_aplus',
    'brand_tier_encoded',
    'is_damascus',
    'is_high_carbon',
    'is_german_steel',
    'is_japanese_steel',
    'is_ceramic',
    'is_chef_knife',
    'is_santoku',
    'is_steak_knife',
    'is_cleaver',
    'is_paring',
    'is_professional',
    'is_gift',
]

# 需要删除的冗余特征
DROP_FEATURES = [
    'negative_ratio',         # = 1 - positive_ratio
    'bert_sentiment_mean',    # ≈ positive_ratio
    'log_price',              # 与 log_price_per_piece 相关，套装场景下后者更准确
    'is_set',                 # set_pieces > 0 即为套装，信息重复
    'has_block',              # 与 is_set/set_pieces 强相关
    # 情感子维度：合并为 sentiment_avg + sentiment_std
    'aspect_sharpness_sentiment',
    'aspect_quality_sentiment',
    'aspect_durability_sentiment',
    'aspect_handle_sentiment',
    'aspect_value_sentiment',
    'aspect_rust_sentiment',
    'aspect_appearance_sentiment',
    'aspect_balance_sentiment',
    'aspect_sentiment_mean',
]


# ============================================================================
# 主类
# ============================================================================

class ClusteringPipelineV3:
    """
    厨刀市场聚类分析 v3.0

    核心改进：
    - 混合数据类型正确处理（K-Prototypes）
    - 情感特征降维聚合
    - Gap Statistic + Bootstrap 稳定性双重验证
    - 更保守的聚类结论输出
    """

    def __init__(self, data_path='clustering_features_only.csv',
                 output_dir='clustering_results_v3'):
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.df_raw = pd.read_csv(data_path)
        self.asin_col = 'asin' if 'asin' in self.df_raw.columns else self.df_raw.columns[0]
        self.asins = self.df_raw[self.asin_col].values

        # 运行时变量
        self.df_feat = None          # 预处理后特征 DataFrame
        self.X_cont_scaled = None    # 标准化连续特征
        self.X_cat = None            # 原始分类特征（整数）
        self.X_combined = None       # 拼接后（K-Means备选用）
        self.X_for_cluster = None    # 实际传入聚类器的数据
        self.cont_cols_final = None  # 最终有效连续列名
        self.cat_cols_final = None   # 最终有效分类列名
        self.scaler = None
        self.pca = None
        self.X_pca = None
        self.X_tsne = None
        self.best_k = None
        self.best_labels = None
        self.cluster_profiles = None
        self.cluster_profiles_z = None
        self.cluster_names = None
        self.cluster_descriptions = None
        self.stability_results = {}

        print("=" * 70)
        print("       厨刀市场聚类分析 v3.0 (混合数据类型优化版)")
        print("=" * 70)
        print(f"\n📊 原始数据: {len(self.df_raw)} 样本 × {len(self.df_raw.columns)-1} 特征")
        print(f"🔧 后端模式: {'K-Prototypes' if HAS_KMODES else 'K-Means (改进版)'}")

    # -------------------------------------------------------------------------
    # Step 1: 特征工程
    # -------------------------------------------------------------------------
    def preprocess(self):
        print("\n" + "=" * 70)
        print("[Step 1] 特征工程与预处理")
        print("=" * 70)

        df = self.df_raw.drop(columns=[self.asin_col]).copy()

        # ── 1.1 生成合并情感特征 ──────────────────────────────────────────
        aspect_cols = [c for c in DROP_FEATURES
                       if c.startswith('aspect_') and c in df.columns]
        if aspect_cols:
            df['sentiment_avg'] = df[aspect_cols].mean(axis=1)
            df['sentiment_std'] = df[aspect_cols].std(axis=1)
            print(f"\n  ✅ 情感维度合并: {len(aspect_cols)} 列 → sentiment_avg + sentiment_std")

        # ── 1.2 删除冗余特征 ─────────────────────────────────────────────
        drop_actual = [c for c in DROP_FEATURES if c in df.columns]
        df.drop(columns=drop_actual, inplace=True)
        print(f"  ✅ 删除冗余特征 {len(drop_actual)} 个: {drop_actual[:6]}{'...' if len(drop_actual)>6 else ''}")

        # ── 1.3 确定实际可用列 ───────────────────────────────────────────
        self.cont_cols_final = [c for c in CONTINUOUS_FEATURES if c in df.columns]
        self.cat_cols_final  = [c for c in CATEGORICAL_FEATURES  if c in df.columns]

        print(f"\n  📊 最终特征构成:")
        print(f"     连续型: {len(self.cont_cols_final)} 个")
        print(f"     分类型: {len(self.cat_cols_final)} 个")
        print(f"     合计:   {len(self.cont_cols_final)+len(self.cat_cols_final)} 个 (原始42→优化后)")

        # ── 1.4 缺失值/无穷值处理 ────────────────────────────────────────
        for col in self.cont_cols_final:
            col_data = df[col].replace([np.inf, -np.inf], np.nan)
            if col_data.isna().sum() > 0:
                df[col] = col_data.fillna(col_data.median())
            else:
                df[col] = col_data

        for col in self.cat_cols_final:
            df[col] = df[col].fillna(0).astype(int)

        self.df_feat = df

        # ── 1.5 标准化连续特征 ───────────────────────────────────────────
        self.scaler = StandardScaler()
        self.X_cont_scaled = self.scaler.fit_transform(df[self.cont_cols_final].values)
        self.X_cat = df[self.cat_cols_final].values.astype(int)

        # 拼接（K-Means备选 或 PCA可视化 用）
        self.X_combined = np.hstack([self.X_cont_scaled, self.X_cat])

        print(f"\n  ✅ 标准化完成 (StandardScaler on {len(self.cont_cols_final)} 连续特征)")

        # ── 1.6 相关性检查 ────────────────────────────────────────────────
        self._check_remaining_correlation()

        return self

    def _check_remaining_correlation(self):
        """检查处理后连续特征间的残余相关性"""
        df_cont = pd.DataFrame(self.X_cont_scaled, columns=self.cont_cols_final)
        corr = df_cont.corr().abs()
        high_pairs = []
        for i in range(len(self.cont_cols_final)):
            for j in range(i+1, len(self.cont_cols_final)):
                if corr.iloc[i, j] > 0.75:
                    high_pairs.append((self.cont_cols_final[i],
                                       self.cont_cols_final[j],
                                       corr.iloc[i, j]))
        if high_pairs:
            print(f"\n  ⚠️  残余高相关特征对 (|r|>0.75):")
            for f1, f2, r in sorted(high_pairs, key=lambda x: -x[2])[:5]:
                print(f"     {f1} ↔ {f2}: r={r:.3f}")
        else:
            print("  ✅ 无高度相关特征对残留")

    # -------------------------------------------------------------------------
    # Step 2: 降维
    # -------------------------------------------------------------------------
    def reduce_dimensions(self):
        print("\n" + "=" * 70)
        print("[Step 2] 降维分析")
        print("=" * 70)

        # ── PCA（仅对连续特征，用于聚类和可视化）────────────────────────
        # 自动确定维数（保留90%方差，但上限20维）
        pca_full = PCA(random_state=42)
        pca_full.fit(self.X_cont_scaled)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_90 = int(np.argmax(cumvar >= 0.90)) + 1
        n_components = min(n_90, 20, self.X_cont_scaled.shape[1])

        self.pca = PCA(n_components=n_components, random_state=42)
        self.X_pca = self.pca.fit_transform(self.X_cont_scaled)

        print(f"\n  📉 PCA 结果 (连续特征):")
        print(f"     原始连续维数: {self.X_cont_scaled.shape[1]}")
        print(f"     保留维数 (≥90%方差): {n_components}")
        print(f"     实际方差解释: {cumvar[n_components-1]*100:.1f}%")
        print(f"     前2PC方差: {cumvar[1]*100:.1f}%")

        # 保存方差图
        self._plot_pca_variance(pca_full, cumvar, n_components)

        # ── t-SNE（全特征用于可视化，PCA预处理加速）────────────────────
        print("\n  🔄 t-SNE 降维 (用于可视化)...")
        perplexity = min(30, len(self.X_combined) - 1)
        # 先用PCA压到50维加速t-SNE
        n_pre = min(50, self.X_combined.shape[1])
        pca_pre = PCA(n_components=n_pre, random_state=42)
        X_pre = pca_pre.fit_transform(self.X_combined)

        try:
            tsne = TSNE(n_components=2, perplexity=perplexity,
                        random_state=42, max_iter=1000)
        except TypeError:
            tsne = TSNE(n_components=2, perplexity=perplexity,
                        random_state=42, n_iter=1000)
        self.X_tsne = tsne.fit_transform(X_pre)
        print("  ✅ t-SNE 完成")

        return self

    def _plot_pca_variance(self, pca_full, cumvar, n_selected):
        n_show = min(20, len(cumvar))
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        ax = axes[0]
        ax.bar(range(1, n_show+1), pca_full.explained_variance_ratio_[:n_show],
               color='#3498db', alpha=0.8, edgecolor='white')
        ax.axvline(x=n_selected, color='red', linestyle='--', linewidth=2,
                   label=f'Selected: {n_selected}')
        ax.set_xlabel('Principal Component'); ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('PCA Explained Variance (Continuous Features)', fontweight='bold')
        ax.legend(); ax.set_xticks(range(1, n_show+1))

        ax = axes[1]
        ax.plot(range(1, n_show+1), cumvar[:n_show], 'o-', color='#e74c3c', linewidth=2)
        ax.axvline(x=n_selected, color='red', linestyle='--', linewidth=2)
        ax.axhline(y=0.90, color='gray', linestyle='--', alpha=0.7, label='90%')
        ax.axhline(y=0.95, color='gray', linestyle=':', alpha=0.7, label='95%')
        ax.fill_between(range(1, n_show+1), cumvar[:n_show], alpha=0.2, color='#e74c3c')
        ax.set_xlabel('Number of Components'); ax.set_ylabel('Cumulative Explained Variance')
        ax.set_title('Cumulative Variance Explained', fontweight='bold')
        ax.legend(loc='lower right'); ax.set_ylim(0, 1.05); ax.set_xticks(range(1, n_show+1))

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pca_variance_v3.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # -------------------------------------------------------------------------
    # Step 3: K值选择（Gap Statistic + 传统指标 + Bootstrap稳定性）
    # -------------------------------------------------------------------------
    def find_optimal_k(self, k_range=(2, 9), n_gap_refs=15, n_bootstrap=20):
        print("\n" + "=" * 70)
        print("[Step 3] 最优K值确定（多重验证）")
        print("=" * 70)

        k_min, k_max = k_range
        k_values = list(range(k_min, k_max + 1))

        # 聚类所用数据：PCA降维后连续特征 + 原始分类特征
        # （此处用于K值选择，实际聚类根据是否有kmodes决定）
        X_eval = np.hstack([self.X_pca, self.X_cat])

        # ── 3.1 传统指标 ─────────────────────────────────────────────────
        metrics = {'k': k_values, 'inertia': [], 'silhouette': [],
                   'calinski': [], 'davies_bouldin': []}

        print(f"\n  🔍 传统指标评估 (k = {k_min} ~ {k_max})")
        print("  " + "-" * 65)
        print(f"  {'K':<4} {'Inertia':<11} {'Silhouette':<12} {'CH Index':<11} {'DB Index'}")
        print("  " + "-" * 65)

        all_labels = {}
        for k in k_values:
            km = KMeans(n_clusters=k, init='k-means++', n_init=30,
                        max_iter=500, random_state=42)
            labels = km.fit_predict(X_eval)
            all_labels[k] = labels

            sil = silhouette_score(X_eval, labels)
            ch  = calinski_harabasz_score(X_eval, labels)
            db  = davies_bouldin_score(X_eval, labels)

            metrics['inertia'].append(km.inertia_)
            metrics['silhouette'].append(sil)
            metrics['calinski'].append(ch)
            metrics['davies_bouldin'].append(db)
            print(f"  {k:<4} {km.inertia_:<11.1f} {sil:<12.4f} {ch:<11.1f} {db:.4f}")

        print("  " + "-" * 65)

        # ── 3.2 Gap Statistic ────────────────────────────────────────────
        print(f"\n  📐 Gap Statistic 计算 (n_refs={n_gap_refs})...")
        gaps, gap_stds = self._gap_statistic(X_eval, k_values, n_gap_refs)

        print(f"\n  {'K':<4} {'Gap':<10} {'Std':<10} {'Gap(k)-Gap(k+1)+Std(k+1)'}")
        print("  " + "-" * 50)
        gap_k = None
        for i, k in enumerate(k_values):
            if i < len(k_values) - 1:
                criterion = gaps[i] - gaps[i+1] + gap_stds[i+1]
                flag = " ← 推荐" if criterion >= 0 and gap_k is None else ""
                if criterion >= 0 and gap_k is None:
                    gap_k = k
                print(f"  {k:<4} {gaps[i]:<10.4f} {gap_stds[i]:<10.4f} {criterion:.4f}{flag}")
            else:
                print(f"  {k:<4} {gaps[i]:<10.4f} {gap_stds[i]:<10.4f} -")

        if gap_k is None:
            gap_k = k_values[np.argmax(gaps)]
        print(f"\n  🎯 Gap Statistic 推荐: k = {gap_k}")

        # ── 3.3 Bootstrap 稳定性 ─────────────────────────────────────────
        print(f"\n  🔄 Bootstrap 稳定性检验 (n={n_bootstrap})...")
        stability_scores = {}
        print(f"  {'K':<4} {'ARI均值':<10} {'ARI标准差':<12} {'评级'}")
        print("  " + "-" * 45)

        for k in k_values:
            mean_ari, std_ari = self._bootstrap_stability(X_eval, k, n_bootstrap)
            stability_scores[k] = (mean_ari, std_ari)
            if mean_ari >= 0.85:
                grade = "🟢 稳定"
            elif mean_ari >= 0.65:
                grade = "🟡 中等"
            else:
                grade = "🔴 不稳定"
            print(f"  {k:<4} {mean_ari:<10.3f} {std_ari:<12.3f} {grade}")

        self.stability_results = stability_scores

        # ── 3.4 综合决策 ─────────────────────────────────────────────────
        self.best_k = self._decide_k(
            k_values, metrics, gaps, gap_stds, stability_scores, gap_k
        )

        print(f"\n  {'='*50}")
        print(f"  🏆 最终推荐 K = {self.best_k}")
        print(f"  {'='*50}")

        # 保存评估图
        self._plot_k_selection(k_values, metrics, gaps, gap_stds, stability_scores)

        self.metrics = metrics
        self.all_labels_eval = all_labels
        return self

    def _gap_statistic(self, X, k_values, n_refs):
        gaps, gap_stds = [], []
        rng = np.random.RandomState(42)

        for k in k_values:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            km.fit(X)
            log_wk = np.log(km.inertia_)

            ref_log_wks = []
            for _ in range(n_refs):
                X_ref = rng.uniform(X.min(axis=0), X.max(axis=0), X.shape)
                km_ref = KMeans(n_clusters=k, n_init=5, random_state=42)
                km_ref.fit(X_ref)
                ref_log_wks.append(np.log(km_ref.inertia_))

            gap = np.mean(ref_log_wks) - log_wk
            std = np.std(ref_log_wks) * np.sqrt(1 + 1/n_refs)
            gaps.append(gap)
            gap_stds.append(std)

        return gaps, gap_stds

    def _bootstrap_stability(self, X, k, n_bootstrap):
        base_km = KMeans(n_clusters=k, n_init=20, random_state=42)
        base_labels = base_km.fit_predict(X)
        ari_scores = []

        rng = np.random.RandomState(42)
        for i in range(n_bootstrap):
            # 兼容所有 sklearn 版本：手动生成 bootstrap 索引
            idx = rng.choice(len(X), size=len(X), replace=True)
            X_boot = X[idx]
            km_b = KMeans(n_clusters=k, n_init=5, random_state=i)
            boot_labels = km_b.fit_predict(X_boot)
            ari = adjusted_rand_score(base_labels[idx], boot_labels)
            ari_scores.append(ari)

        return float(np.mean(ari_scores)), float(np.std(ari_scores))

    def _decide_k(self, k_values, metrics, gaps, gap_stds, stability, gap_k):
        """综合多个指标的K值决策逻辑"""
        scores = np.zeros(len(k_values))

        # ── 轮廓系数归一化（权重40%）────────────────────────────────────
        sil = np.array(metrics['silhouette'])
        sil_n = (sil - sil.min()) / (sil.max() - sil.min() + 1e-8)
        scores += 0.40 * sil_n

        # ── Gap Statistic：用判定准则打分，而非原始值归一化 ──────────────
        # 准则：gap(k) >= gap(k+1) - std(k+1)  =>  该k是候选拐点
        # Gap单调递增时原始值归一化会错误地给最大k满分，改用准则得分
        gap_criterion_score = np.zeros(len(k_values))
        for i in range(len(k_values) - 1):
            criterion = gaps[i] - gaps[i+1] + gap_stds[i+1]
            # criterion越大说明越是拐点，正值才有意义
            gap_criterion_score[i] = max(0.0, criterion)
        # 归一化准则分
        if gap_criterion_score.max() > 0:
            gap_n = gap_criterion_score / gap_criterion_score.max()
        else:
            gap_n = gap_criterion_score
        scores += 0.30 * gap_n

        # ── Bootstrap稳定性归一化（权重30%）─────────────────────────────
        stab = np.array([stability[k][0] for k in k_values])
        stab_n = (stab - stab.min()) / (stab.max() - stab.min() + 1e-8)
        scores += 0.30 * stab_n

        best_idx = int(np.argmax(scores))
        best_k = k_values[best_idx]

        print(f"\n  📊 综合评分 (轮廓40% + Gap准则30% + 稳定性30%):")
        for i, k in enumerate(k_values):
            marker = " ← 最优" if k == best_k else ""
            print(f"     k={k}: {scores[i]:.4f}  "
                  f"[sil={sil[i]:.3f}, gap_crit={gap_criterion_score[i]:.4f}, "
                  f"ari={stab[i]:.3f}]{marker}")

        return best_k

    def _plot_k_selection(self, k_values, metrics, gaps, gap_stds, stability):
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)

        colors_bar = ['#e74c3c' if k == self.best_k else '#3498db' for k in k_values]

        # 1. 肘部
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(k_values, metrics['inertia'], 'bo-', lw=2, ms=7)
        ax1.axvline(x=self.best_k, color='red', ls='--', lw=2, label=f'Best k={self.best_k}')
        ax1.set_title('Elbow Method (SSE)', fontweight='bold')
        ax1.set_xlabel('K'); ax1.set_ylabel('Inertia')
        ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_xticks(k_values)

        # 2. 轮廓系数
        ax2 = fig.add_subplot(gs[0, 1])
        bars = ax2.bar(k_values, metrics['silhouette'], color=colors_bar, edgecolor='white', lw=1.5)
        for b, v in zip(bars, metrics['silhouette']):
            ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.001,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        ax2.set_title('Silhouette Score', fontweight='bold')
        ax2.set_xlabel('K'); ax2.set_ylabel('Score')
        ax2.grid(True, alpha=0.3, axis='y'); ax2.set_xticks(k_values)

        # 3. Gap Statistic
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.errorbar(k_values, gaps, yerr=gap_stds, fmt='go-', lw=2, ms=7,
                    ecolor='gray', capsize=4, label='Gap ± Std')
        ax3.axvline(x=self.best_k, color='red', ls='--', lw=2)
        ax3.set_title('Gap Statistic', fontweight='bold')
        ax3.set_xlabel('K'); ax3.set_ylabel('Gap Value')
        ax3.legend(); ax3.grid(True, alpha=0.3); ax3.set_xticks(k_values)

        # 4. Bootstrap 稳定性
        ax4 = fig.add_subplot(gs[1, 1])
        means = [stability[k][0] for k in k_values]
        stds  = [stability[k][1] for k in k_values]
        ax4.errorbar(k_values, means, yerr=stds, fmt='mo-', lw=2, ms=7,
                    ecolor='gray', capsize=4)
        ax4.axhline(y=0.85, color='green', ls='--', alpha=0.7, label='稳定阈值(0.85)')
        ax4.axhline(y=0.65, color='orange', ls='--', alpha=0.7, label='中等阈值(0.65)')
        ax4.axvline(x=self.best_k, color='red', ls='--', lw=2)
        ax4.set_title('Bootstrap Stability (ARI)', fontweight='bold')
        ax4.set_xlabel('K'); ax4.set_ylabel('Adjusted Rand Index')
        ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3); ax4.set_xticks(k_values)
        ax4.set_ylim(0, 1.05)

        # 5. CH Index
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(k_values, metrics['calinski'], 'co-', lw=2, ms=7)
        ax5.axvline(x=self.best_k, color='red', ls='--', lw=2)
        ax5.set_title('Calinski-Harabasz Index (Higher=Better)', fontweight='bold')
        ax5.set_xlabel('K'); ax5.set_ylabel('CH Index')
        ax5.grid(True, alpha=0.3); ax5.set_xticks(k_values)

        # 6. DB Index
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(k_values, metrics['davies_bouldin'], 'yo-', lw=2, ms=7,
                color='#e67e22')
        ax6.axvline(x=self.best_k, color='red', ls='--', lw=2)
        ax6.set_title('Davies-Bouldin Index (Lower=Better)', fontweight='bold')
        ax6.set_xlabel('K'); ax6.set_ylabel('DB Index')
        ax6.grid(True, alpha=0.3); ax6.set_xticks(k_values)

        fig.suptitle(f'Optimal K Analysis v3.0 — Best: k={self.best_k}',
                    fontsize=14, fontweight='bold')
        plt.savefig(os.path.join(self.output_dir, 'optimal_k_v3.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  📊 K值分析图已保存")

    # -------------------------------------------------------------------------
    # Step 4: 执行聚类
    # -------------------------------------------------------------------------
    def run_clustering(self, k=None):
        if k is None:
            k = self.best_k
        else:
            self.best_k = k

        print("\n" + "=" * 70)
        print(f"[Step 4] 执行聚类 (k = {k})")
        print("=" * 70)

        X_pca_cat = np.hstack([self.X_pca, self.X_cat])

        if HAS_KMODES:
            self._run_kprototypes(k)
        else:
            self._run_kmeans_improved(k, X_pca_cat)

        # 质量报告
        sil = silhouette_score(X_pca_cat, self.best_labels)
        ch  = calinski_harabasz_score(X_pca_cat, self.best_labels)
        db  = davies_bouldin_score(X_pca_cat, self.best_labels)

        print(f"\n  📊 聚类质量指标:")
        print(f"     轮廓系数:  {sil:.4f}  {'🟢 良好' if sil>0.3 else '🟡 中等' if sil>0.15 else '🔴 较差'}")
        print(f"     CH Index:  {ch:.1f}")
        print(f"     DB Index:  {db:.4f}")

        # Bootstrap最终稳定性
        mean_ari, std_ari = self.stability_results.get(k, (None, None))
        if mean_ari is not None:
            print(f"     稳定性ARI: {mean_ari:.3f} ± {std_ari:.3f}  "
                  f"{'🟢 稳定' if mean_ari>=0.85 else '🟡 中等' if mean_ari>=0.65 else '🔴 不稳定'}")

        if sil < 0.15:
            print("\n  ⚠️  警告：轮廓系数 < 0.15，聚类结构较弱")
            print("     建议：聚类结论仅用于探索性参考，勿作为强决策依据")

        # 簇大小
        sizes = pd.Series(self.best_labels).value_counts().sort_index()
        print(f"\n  📊 簇大小:")
        for cid, sz in sizes.items():
            pct = sz / len(self.best_labels) * 100
            print(f"     C{cid}: {sz:>4} ({pct:>5.1f}%) {'█'*int(pct/3)}")

        self._plot_clustering_scatter()
        return self

    def _run_kprototypes(self, k):
        """K-Prototypes：正确处理连续+分类混合数据"""
        print("\n  🔧 使用 K-Prototypes (连续+分类混合)")
        # 原始连续特征（标准化）+ 原始分类特征
        X_kp = np.hstack([self.X_cont_scaled, self.X_cat.astype(float)])
        cat_idx = list(range(self.X_cont_scaled.shape[1],
                              self.X_cont_scaled.shape[1] + self.X_cat.shape[1]))

        kp = KPrototypes(n_clusters=k, init='Cao', n_init=10,
                         random_state=42, verbose=0)
        self.best_labels = kp.fit_predict(X_kp, categorical=cat_idx)
        self.kp_model = kp
        print("  ✅ K-Prototypes 完成")

    def _run_kmeans_improved(self, k, X):
        """改进版 K-Means（kmodes不可用时的备选）"""
        print("\n  🔧 使用改进版 K-Means (PCA降维后连续+分类拼接)")
        km = KMeans(n_clusters=k, init='k-means++', n_init=50,
                    max_iter=500, random_state=42)
        self.best_labels = km.fit_predict(X)
        self.km_model = km
        print("  ✅ K-Means 完成")

    def _plot_clustering_scatter(self):
        k = len(np.unique(self.best_labels))
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, (X2d, xlabel, ylabel, title) in zip(axes, [
            (self.X_tsne, 't-SNE Dim 1', 't-SNE Dim 2', 't-SNE Visualization'),
            (self.X_pca[:, :2],
             f'PC1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)',
             f'PC2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)',
             'PCA Visualization (PC1 vs PC2)')
        ]):
            for i in range(k):
                m = self.best_labels == i
                ax.scatter(X2d[m, 0], X2d[m, 1],
                          c=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                          label=f'C{i} (n={m.sum()})',
                          alpha=0.6, s=45, edgecolors='white', lw=0.5)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
            ax.set_title(title, fontweight='bold')
            ax.legend(fontsize=8)

        fig.suptitle(f'Clustering Results (k={k}) — v3.0', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'clustering_scatter_v3.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        print("  📊 聚类散点图已保存")

    # -------------------------------------------------------------------------
    # Step 5: 簇分析
    # -------------------------------------------------------------------------
    def analyze_clusters(self):
        print("\n" + "=" * 70)
        print("[Step 5] 簇特征分析")
        print("=" * 70)

        # 用原始连续特征（未标准化）做画像，方便业务解读
        all_cols = self.cont_cols_final + self.cat_cols_final
        df_a = self.df_feat[all_cols].copy()
        df_a['cluster'] = self.best_labels

        profiles     = df_a.groupby('cluster')[all_cols].mean()
        overall_mean = df_a[all_cols].mean()
        overall_std  = df_a[all_cols].std()
        profiles_z   = (profiles - overall_mean) / (overall_std + 1e-8)

        self.cluster_profiles   = profiles
        self.cluster_profiles_z = profiles_z

        n_clusters = len(np.unique(self.best_labels))
        cluster_sizes = df_a['cluster'].value_counts().sort_index()
        cluster_descriptions = {}

        print("\n  🔍 各簇显著特征 (|z| > 0.5):")
        print("  " + "-" * 60)

        for cid in range(n_clusters):
            z = profiles_z.loc[cid]
            high = z[z >  0.5].sort_values(ascending=False)
            low  = z[z < -0.5].sort_values()
            sz   = cluster_sizes[cid]
            pct  = sz / len(df_a) * 100

            print(f"\n  【C{cid}】 {sz}个 ({pct:.1f}%)")
            if len(high): print(f"    ↑ {dict(list(high.head(5).items()))}")
            if len(low):  print(f"    ↓ {dict(list(low.head(5).items()))}")

            cluster_descriptions[cid] = {
                'size': sz, 'pct': pct,
                'high': high.head(5).to_dict(),
                'low':  low.head(5).to_dict()
            }

        self.cluster_descriptions = cluster_descriptions
        self.df_clustered = df_a

        # 可视化
        self._plot_analysis_dashboard(n_clusters)
        self._plot_feature_heatmap(n_clusters)
        self._plot_radar(n_clusters)
        self._plot_silhouette(n_clusters)

        return self

    def _plot_analysis_dashboard(self, n_clusters):
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))

        # 1. 饼图
        ax = axes[0, 0]
        sizes = [self.cluster_descriptions[i]['size'] for i in range(n_clusters)]
        colors_pie = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(n_clusters)]
        ax.pie(sizes,
               labels=[f"C{i}\n({self.cluster_descriptions[i]['pct']:.1f}%)"
                       for i in range(n_clusters)],
               autopct='%1.0f%%', colors=colors_pie,
               explode=[0.03]*n_clusters, textprops={'fontsize': 9})
        ax.set_title('Cluster Size Distribution', fontweight='bold')

        # 2. 关键连续特征箱线图
        ax = axes[0, 1]
        key = ['log_price_per_piece', 'product_rating', 'log_sales', 'positive_ratio']
        key = [c for c in key if c in self.cont_cols_final][:3]
        if key:
            melt = self.df_clustered[['cluster']+key].melt(
                id_vars='cluster', var_name='Feature', value_name='Value')
            sns.boxplot(data=melt, x='Feature', y='Value', hue='cluster',
                       palette=CLUSTER_COLORS[:n_clusters], ax=ax)
            ax.set_title('Key Features Distribution', fontweight='bold')
            ax.legend(title='C', fontsize=7, title_fontsize=8)
            ax.tick_params(axis='x', rotation=15)

        # 3. 情感对比（新增：sentiment_avg vs sentiment_std）
        ax = axes[1, 0]
        if 'sentiment_avg' in self.cont_cols_final and 'sentiment_std' in self.cont_cols_final:
            for i in range(n_clusters):
                m = self.best_labels == i
                ax.scatter(
                    self.df_feat.loc[m, 'sentiment_avg'].values,
                    self.df_feat.loc[m, 'sentiment_std'].values,
                    c=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                    label=f'C{i}', alpha=0.5, s=30
                )
            ax.set_xlabel('Sentiment Avg (整体口碑)')
            ax.set_ylabel('Sentiment Std (评价分化程度)')
            ax.set_title('Sentiment Landscape', fontweight='bold')
            ax.legend(fontsize=8)

        # 4. 价格 vs 销量
        ax = axes[1, 1]
        if 'log_price_per_piece' in self.cont_cols_final and 'log_sales' in self.cont_cols_final:
            for i in range(n_clusters):
                m = self.best_labels == i
                ax.scatter(
                    self.df_feat.loc[m, 'log_price_per_piece'].values,
                    self.df_feat.loc[m, 'log_sales'].values,
                    c=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                    label=f'C{i}', alpha=0.5, s=30
                )
            ax.set_xlabel('log_price_per_piece (单价)')
            ax.set_ylabel('log_sales (销量)')
            ax.set_title('Price vs Sales by Cluster', fontweight='bold')
            ax.legend(fontsize=8)

        plt.suptitle('Cluster Analysis Dashboard v3.0', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'dashboard_v3.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_feature_heatmap(self, n_clusters):
        # 选z-score绝对值变异最大的top20特征
        top_feats = self.cluster_profiles_z.abs().mean().nlargest(20).index.tolist()
        data = self.cluster_profiles_z[top_feats].T

        fig, ax = plt.subplots(figsize=(max(8, n_clusters*1.5), 12))
        sns.heatmap(data, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                   ax=ax, cbar_kws={'shrink': 0.8},
                   xticklabels=[f'C{i}' for i in range(n_clusters)],
                   annot_kws={'size': 8})
        ax.set_title('Feature Z-Score Heatmap (Top 20 Discriminative Features)',
                    fontweight='bold', fontsize=12)
        ax.tick_params(axis='y', rotation=0, labelsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'feature_heatmap_v3.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_radar(self, n_clusters):
        radar_feats = ['log_price_per_piece', 'product_rating', 'log_sales',
                       'log_reviews', 'sentiment_avg', 'positive_ratio',
                       'discount_rate', 'set_pieces']
        radar_feats = [f for f in radar_feats if f in self.cont_cols_final][:8]
        if len(radar_feats) < 3:
            return

        data = self.cluster_profiles[radar_feats].copy()
        norm = (data - data.min()) / (data.max() - data.min() + 1e-8)

        angles = np.linspace(0, 2*np.pi, len(radar_feats), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
        for i in range(n_clusters):
            vals = norm.loc[i].tolist() + [norm.loc[i].tolist()[0]]
            ax.plot(angles, vals, 'o-', lw=2, label=f'C{i}',
                   color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)], ms=5)
            ax.fill(angles, vals, alpha=0.12, color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_feats, fontsize=10)
        ax.set_title('Cluster Profiles Radar Chart v3.0', fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'radar_v3.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_silhouette(self, n_clusters):
        X_eval = np.hstack([self.X_pca, self.X_cat])
        sil_vals = silhouette_samples(X_eval, self.best_labels)

        fig, ax = plt.subplots(figsize=(8, 6))
        y_lower = 10
        for i in range(n_clusters):
            cluster_sil = np.sort(sil_vals[self.best_labels == i])
            y_upper = y_lower + len(cluster_sil)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                            facecolor=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                            alpha=0.7, edgecolor='white')
            ax.text(-0.05, y_lower + 0.5*len(cluster_sil), str(i), fontsize=9)
            y_lower = y_upper + 10

        avg = sil_vals.mean()
        ax.axvline(x=avg, color='red', ls='--', lw=2, label=f'Avg: {avg:.3f}')
        ax.set_xlabel('Silhouette Coefficient'); ax.set_ylabel('Cluster')
        ax.set_title('Silhouette Analysis v3.0', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'silhouette_v3.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()

    # -------------------------------------------------------------------------
    # Step 6: 命名与报告
    # -------------------------------------------------------------------------
    def generate_report(self):
        print("\n" + "=" * 70)
        print("[Step 6] 商业洞察报告")
        print("=" * 70)

        self.cluster_names = self._auto_name_clusters()

        print("\n  🏷️  簇命名:")
        for cid, name in self.cluster_names.items():
            d = self.cluster_descriptions[cid]
            print(f"     C{cid}: {name}  ({d['size']}个, {d['pct']:.1f}%)")

        lines = []
        lines += [
            "=" * 80,
            "        厨刀市场聚类分析 v3.0 — 商业洞察报告",
            "=" * 80,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"分析样本: {len(self.df_raw)} 个商品",
            f"有效特征: {len(self.cont_cols_final)+len(self.cat_cols_final)} 个"
            f" (连续{len(self.cont_cols_final)}+分类{len(self.cat_cols_final)})",
            f"聚类算法: {'K-Prototypes' if HAS_KMODES else 'K-Means (PCA+Cat)'}",
            f"聚类数量: {self.best_k}",
            f"轮廓系数: {silhouette_score(np.hstack([self.X_pca,self.X_cat]), self.best_labels):.4f}",
        ]

        # 稳定性评级
        ari_mean, ari_std = self.stability_results.get(self.best_k, (None, None))
        if ari_mean:
            grade = '稳定' if ari_mean >= 0.85 else ('中等' if ari_mean >= 0.65 else '不稳定')
            lines.append(f"聚类稳定性: ARI={ari_mean:.3f}±{ari_std:.3f} ({grade})")
            if ari_mean < 0.65:
                lines.append("⚠️  稳定性不足，以下结论仅供参考，不建议作为强决策依据")

        lines.append("\n" + "=" * 80)
        lines.append("                    市场细分详情")
        lines.append("=" * 80)

        for cid in range(self.best_k):
            name = self.cluster_names[cid]
            d    = self.cluster_descriptions[cid]
            z    = self.cluster_profiles_z.loc[cid]

            lines += [
                f"\n{'─'*80}",
                f"【C{cid}】 {name}   ({d['size']}个, {d['pct']:.1f}%)",
                f"{'─'*80}",
                "  核心优势:",
            ]
            for feat, val in d['high'].items():
                lines.append(f"    + {feat}: z={val:+.2f}")
            if not d['high']:
                lines.append("    (无显著高于平均)")

            lines.append("  改进空间:")
            for feat, val in d['low'].items():
                lines.append(f"    - {feat}: z={val:+.2f}")
            if not d['low']:
                lines.append("    (无显著低于平均)")

            lines.append("  建议:")
            for i, s in enumerate(self._suggest(cid, d, z.to_dict()), 1):
                lines.append(f"    {i}. {s}")

        lines += [
            "\n" + "=" * 80,
            "                    整体市场洞察",
            "=" * 80,
        ]
        for insight in self._overall_insights():
            lines.append(f"  {insight}")

        lines += ["", "=" * 80, "                      报告结束", "=" * 80]

        report = "\n".join(lines)
        path = os.path.join(self.output_dir, 'business_report_v3.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(report)
        print(f"\n  📄 报告已保存: {path}")
        return self

    def _auto_name_clusters(self):
        names = {}
        for cid in range(self.best_k):
            z    = self.cluster_profiles_z.loc[cid]
            d    = self.cluster_descriptions[cid]
            tags = []

            # 规模
            if d['pct'] > 35:   tags.append('主流')
            elif d['pct'] < 6:  tags.append('细分')

            # 价格
            pz = z.get('log_price_per_piece', 0)
            if   pz >  0.6: tags.append('高价')
            elif pz >  0.2: tags.append('中高价')
            elif pz < -0.6: tags.append('低价')
            elif pz < -0.2: tags.append('中低价')

            # 套装
            sz = self.cluster_profiles.loc[cid].get('set_pieces', 0)
            if   sz > 3:   tags.append('套装')
            elif sz < 1.5: tags.append('单品')

            # 口碑
            sa = z.get('sentiment_avg', z.get('positive_ratio', 0))
            if   sa >  1.5: tags.append('口碑佳')
            elif sa < -2.0: tags.append('口碑差')

            # 销量
            sv = z.get('log_sales', 0)
            if   sv >  0.8: tags.append('畅销')
            elif sv < -1.0: tags.append('滞销')

            # 材质
            if self.cluster_profiles.loc[cid].get('is_damascus', 0) > 0.15:
                tags.append('大马士革')
            elif self.cluster_profiles.loc[cid].get('is_japanese_steel', 0) > 0.12:
                tags.append('日系')
            elif self.cluster_profiles.loc[cid].get('is_german_steel', 0) > 0.12:
                tags.append('德系')

            names[cid] = '-'.join(tags[:3]) if tags else f'细分市场{cid}'
        return names

    def _suggest(self, cid, d, z):
        sug = []
        if z.get('log_price_per_piece', 0) > 0.5:
            sug.append("高单价定位，强化品质内容和品牌溢价叙事")
        if z.get('log_price_per_piece', 0) < -0.5:
            sug.append("价格敏感市场，考虑套装/捆绑销售提升客单价")
        if 'sentiment_avg' in d['low'] or 'positive_ratio' in d['low']:
            sug.append("口碑下滑，优先改善产品质量，回应负面评价")
        if 'sentiment_avg' in d['high'] or 'positive_ratio' in d['high']:
            sug.append("口碑是核心优势，鼓励晒单，强化社会证明")
        if 'log_sales' in d['high']:
            sug.append("热销品类，测试价格弹性空间")
        if 'log_sales' in d['low'] or 'log_reviews' in d['low']:
            sug.append("曝光不足，加强广告投放和关键词布局")
        if z.get('set_pieces', 0) > 0.5:
            sug.append("套装市场，优化刀组搭配和礼盒包装")
        if z.get('set_pieces', 0) < -0.3:
            sug.append("单品赛道，突出专业性能和使用场景")
        if not sug:
            sug.append("维持现有策略，持续监测竞品动态")
        return sug[:4]

    def _overall_insights(self):
        insights = []
        sizes = pd.Series(self.best_labels).value_counts()
        top2  = sizes.nlargest(2).sum() / len(self.best_labels)
        insights.append(f"市场集中度：前2簇占 {top2*100:.1f}%")

        if 'product_rating' in self.cont_cols_final:
            avg_r = self.df_feat['product_rating'].mean()
            insights.append(f"平均评分：{avg_r:.2f} 分")

        if 'positive_ratio' in self.cont_cols_final:
            avg_p = self.df_feat['positive_ratio'].mean()
            insights.append(f"整体正向情感比：{avg_p:.3f}")

        if 'sentiment_std' in self.cont_cols_final:
            avg_std = self.df_feat['sentiment_std'].mean()
            insights.append(
                f"评价分化指数（均值）：{avg_std:.3f} "
                f"({'评价较一致' if avg_std < 0.2 else '评价较分化'})"
            )

        if 'is_fba' in self.cat_cols_final:
            fba_r = self.df_feat['is_fba'].mean()
            insights.append(f"FBA占比：{fba_r*100:.1f}%")

        if 'set_pieces' in self.cont_cols_final:
            set_r = (self.df_feat['set_pieces'] > 1).mean()
            insights.append(f"套装商品占比：{set_r*100:.1f}%")

        return insights

    # -------------------------------------------------------------------------
    # Step 7: 保存
    # -------------------------------------------------------------------------
    def save_results(self):
        print("\n" + "=" * 70)
        print("[Step 7] 保存结果文件")
        print("=" * 70)

        files = {}

        # 带标签的完整数据
        out = self.df_raw.copy()
        out['cluster']      = self.best_labels
        out['cluster_name'] = pd.Series(self.best_labels).map(self.cluster_names)
        out['tsne_x']       = self.X_tsne[:, 0]
        out['tsne_y']       = self.X_tsne[:, 1]
        out['pca_x']        = self.X_pca[:, 0]
        out['pca_y']        = self.X_pca[:, 1]
        p = os.path.join(self.output_dir, 'clustered_products_v3.csv')
        out.to_csv(p, index=False, encoding='utf-8-sig'); files['产品聚类结果'] = p

        # 簇画像
        p = os.path.join(self.output_dir, 'cluster_profiles_v3.csv')
        self.cluster_profiles.to_csv(p, encoding='utf-8-sig'); files['簇特征均值'] = p

        p = os.path.join(self.output_dir, 'cluster_profiles_zscore_v3.csv')
        self.cluster_profiles_z.to_csv(p, encoding='utf-8-sig'); files['簇特征Z分'] = p

        # K值评估指标
        p = os.path.join(self.output_dir, 'clustering_metrics_v3.csv')
        pd.DataFrame(self.metrics).to_csv(p, index=False, encoding='utf-8-sig')
        files['K值评估'] = p

        # Bootstrap稳定性
        stab_rows = [{'k': k, 'ari_mean': v[0], 'ari_std': v[1]}
                     for k, v in self.stability_results.items()]
        p = os.path.join(self.output_dir, 'stability_results_v3.csv')
        pd.DataFrame(stab_rows).to_csv(p, index=False, encoding='utf-8-sig')
        files['稳定性检验'] = p

        # 簇名映射
        rows = [{'cluster': k, 'name': v,
                 'size': self.cluster_descriptions[k]['size'],
                 'pct':  self.cluster_descriptions[k]['pct']}
                for k, v in self.cluster_names.items()]
        p = os.path.join(self.output_dir, 'cluster_names_v3.csv')
        pd.DataFrame(rows).to_csv(p, index=False, encoding='utf-8-sig')
        files['簇命名'] = p

        for desc, path in files.items():
            print(f"  ✅ {desc}: {path}")

        print(f"\n  📁 输出目录: {self.output_dir}/")
        return self

    # -------------------------------------------------------------------------
    # 一键运行
    # -------------------------------------------------------------------------
    def run(self, k_range=(2, 9), final_k=None,
            n_gap_refs=15, n_bootstrap=20):
        """
        完整流水线入口

        Args:
            k_range:      K值搜索范围
            final_k:      强制指定K值（None=自动选择）
            n_gap_refs:   Gap Statistic 参考数据集数量
            n_bootstrap:  Bootstrap 重采样次数
        """
        self.preprocess()
        self.reduce_dimensions()
        self.find_optimal_k(k_range=k_range,
                            n_gap_refs=n_gap_refs,
                            n_bootstrap=n_bootstrap)
        if final_k is not None:
            print(f"\n  📌 用户指定 K = {final_k} (自动推荐: {self.best_k})")
            self.best_k = final_k

        self.run_clustering()
        self.analyze_clusters()
        self.generate_report()
        self.save_results()

        print("\n" + "=" * 70)
        print("  ✅ v3.0 分析完成！")
        sil = silhouette_score(np.hstack([self.X_pca, self.X_cat]), self.best_labels)
        ari_mean = self.stability_results.get(self.best_k, (None,))[0]
        ari_str = f"{ari_mean:.3f}" if ari_mean is not None else "N/A"
        print(f"     K={self.best_k} | 轮廓系数={sil:.4f} | 稳定性ARI={ari_str}")
        print("=" * 70)
        return self


# ============================================================================
# 入口
# ============================================================================
if __name__ == '__main__':
    pipeline = ClusteringPipelineV3(
        data_path  = 'clustering_features_only.csv',
        output_dir = 'clustering_results_v3'
    ).run(
        k_range     = (2, 9),   # K值搜索范围
        final_k     = 3,     # None=自动; 填数字=强制
        n_gap_refs  = 15,       # Gap Statistic 参考组数（越大越准，越慢）
        n_bootstrap = 20,       # Bootstrap 轮数（≥20 结果稳定）
    )