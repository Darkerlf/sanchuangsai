"""
厨刀市场聚类分析 - 优化版 v2.0
================================
优化内容：
1. 移除更多冗余特征（negative_ratio, bert_sentiment_mean）
2. 改进簇命名逻辑，增强差异化
3. 增强商业建议生成
4. 优化可视化效果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import warnings
import os
from datetime import datetime

# 机器学习库
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    calinski_harabasz_score, davies_bouldin_score
)
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================================
# 配置区
# ============================================================================

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 聚类颜色方案
CLUSTER_COLORS = [
    '#E74C3C',  # 红
    '#3498DB',  # 蓝
    '#2ECC71',  # 绿
    '#9B59B6',  # 紫
    '#F39C12',  # 橙
    '#1ABC9C',  # 青
    '#E91E63',  # 粉
    '#34495E',  # 深灰
    '#00BCD4',  # 青蓝
    '#FF5722',  # 深橙
]

# 冗余特征列表（完全相关或高度相关）
REDUNDANT_FEATURES = [
    'negative_ratio',       # 与 positive_ratio 完全互补 (r=-1)
    'bert_sentiment_mean',  # 与 positive_ratio 完全相关 (r=1)
]


class ClusteringPipelineOptimized:
    """
    厨刀市场聚类分析流水线 - 优化版

    Features:
    - 自动移除冗余特征
    - K-Means++ 聚类（稳定性最佳）
    - 智能簇命名
    - 丰富的商业洞察
    """

    def __init__(self, data_path: str = 'clustering_features_only.csv',
                 output_dir: str = 'clustering_results'):
        """
        初始化聚类分析流水线

        Args:
            data_path: 特征数据文件路径
            output_dir: 输出目录
        """
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 加载数据
        self.df = pd.read_csv(data_path)

        # 识别ID列
        self.asin_col = 'asin' if 'asin' in self.df.columns else self.df.columns[0]

        # 分离ID和特征
        self.asins = self.df[self.asin_col].values
        self.feature_cols = [c for c in self.df.columns if c != self.asin_col]
        self.X_raw = self.df[self.feature_cols].values

        # 初始化变量
        self.X_scaled = None
        self.scaler = None
        self.pca = None
        self.X_pca = None
        self.X_tsne = None
        self.best_k = None
        self.best_labels = None
        self.best_algorithm = None
        self.cluster_profiles = None
        self.cluster_profiles_z = None
        self.cluster_descriptions = None
        self.cluster_names = None
        self.df_clustered = None
        self.metrics = None
        self.algorithm_results = {}

        # 打印加载信息
        print("=" * 70)
        print("          厨刀市场聚类分析 (优化版 v2.0)")
        print("=" * 70)
        print(f"\n📊 数据加载完成:")
        print(f"   - 样本数: {len(self.df)}")
        print(f"   - 原始特征数: {len(self.feature_cols)}")
        print(f"   - ID列: {self.asin_col}")

    def preprocess(self, scaler_type: str = 'standard',
                   remove_redundant: bool = True):
        """
        特征预处理

        Args:
            scaler_type: 标准化方式 ('standard' 或 'robust')
            remove_redundant: 是否移除冗余特征
        """
        print("\n" + "=" * 70)
        print("[Step 1] 特征预处理")
        print("=" * 70)

        # ========== 移除冗余特征 ==========
        if remove_redundant:
            removed = []
            for feat in REDUNDANT_FEATURES:
                if feat in self.feature_cols:
                    removed.append(feat)
                    self.feature_cols.remove(feat)

            if removed:
                print(f"\n  🔧 移除冗余特征: {removed}")
                self.X_raw = self.df[self.feature_cols].values

        print(f"  📊 有效特征数: {len(self.feature_cols)}")

        # ========== 处理缺失值和无穷值 ==========
        X = self.X_raw.copy()

        # 缺失值填充
        nan_counts = np.isnan(X).sum(axis=0)
        if nan_counts.sum() > 0:
            nan_features = [self.feature_cols[i] for i, c in enumerate(nan_counts) if c > 0]
            print(f"\n  ⚠️ 发现缺失值特征: {nan_features[:5]}...")
            print(f"     使用中位数填充...")
            for i, count in enumerate(nan_counts):
                if count > 0:
                    col_median = np.nanmedian(X[:, i])
                    X[np.isnan(X[:, i]), i] = col_median

        # 无穷值处理
        inf_mask = np.isinf(X)
        if inf_mask.sum() > 0:
            print(f"  ⚠️ 发现无穷值，转换为中位数...")
            X[inf_mask] = np.nan
            for i in range(X.shape[1]):
                col_median = np.nanmedian(X[:, i])
                X[np.isnan(X[:, i]), i] = col_median

        # ========== 标准化 ==========
        print(f"\n  📐 使用 {scaler_type.title()}Scaler 标准化")

        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = RobustScaler()

        self.X_scaled = self.scaler.fit_transform(X)

        # 验证标准化结果
        means = np.mean(self.X_scaled, axis=0)
        stds = np.std(self.X_scaled, axis=0)

        print(f"\n  ✅ 标准化完成:")
        print(f"     - 均值范围: [{means.min():.4f}, {means.max():.4f}]")
        print(f"     - 标准差范围: [{stds.min():.4f}, {stds.max():.4f}]")

        return self

    def analyze_features(self):
        """特征相关性分析"""
        print("\n" + "-" * 50)
        print("特征相关性分析")
        print("-" * 50)

        # 计算相关矩阵
        corr_matrix = pd.DataFrame(self.X_scaled, columns=self.feature_cols).corr()

        # 找高相关特征对
        high_corr_pairs = []
        for i in range(len(self.feature_cols)):
            for j in range(i+1, len(self.feature_cols)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.8:
                    high_corr_pairs.append((self.feature_cols[i], self.feature_cols[j], corr))

        if high_corr_pairs:
            print("\n  ⚠️ 高相关特征对 (|r| > 0.8):")
            for f1, f2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]:
                print(f"     {f1} ↔ {f2}: r = {corr:.3f}")
        else:
            print("\n  ✅ 无高度相关的特征对")

        # 绘制相关性热力图
        plt.figure(figsize=(16, 14))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r',
                    center=0, square=True, linewidths=0.5,
                    cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'correlation_matrix.png'), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n  📊 相关性热力图已保存: correlation_matrix.png")

        return self

    def reduce_dimensions(self, n_pca_components: int = 10):
        """
        降维分析：PCA + t-SNE

        Args:
            n_pca_components: PCA保留的主成分数
        """
        print("\n" + "-" * 50)
        print("降维分析")
        print("-" * 50)

        # ========== PCA ==========
        n_components = min(n_pca_components, self.X_scaled.shape[1], self.X_scaled.shape[0])
        self.pca = PCA(n_components=n_components)
        self.X_pca = self.pca.fit_transform(self.X_scaled)

        # 累积方差解释
        cumsum_var = np.cumsum(self.pca.explained_variance_ratio_)

        print(f"\n  📉 PCA 降维结果:")
        print(f"     - 保留 {n_components} 个主成分")
        print(f"     - 前2个PC解释方差: {cumsum_var[1]*100:.1f}%")

        # 找到达到90%和95%方差所需的PC数
        n_90 = np.argmax(cumsum_var >= 0.90) + 1 if np.any(cumsum_var >= 0.90) else n_components
        n_95 = np.argmax(cumsum_var >= 0.95) + 1 if np.any(cumsum_var >= 0.95) else n_components
        print(f"     - 达到90%方差需: {n_90} 个PC")
        print(f"     - 达到95%方差需: {n_95} 个PC")

        # 绘制方差解释图
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 单独方差
        ax = axes[0]
        bars = ax.bar(range(1, n_components+1), self.pca.explained_variance_ratio_,
                     color='#3498db', alpha=0.8, edgecolor='white')
        ax.set_xlabel('Principal Component', fontsize=10)
        ax.set_ylabel('Explained Variance Ratio', fontsize=10)
        ax.set_title('PCA Explained Variance', fontsize=12, fontweight='bold')
        ax.set_xticks(range(1, n_components+1))

        # 累积方差
        ax = axes[1]
        ax.plot(range(1, n_components+1), cumsum_var, 'o-', color='#e74c3c',
               linewidth=2, markersize=8)
        ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.7, label='90%')
        ax.axhline(y=0.95, color='gray', linestyle=':', alpha=0.7, label='95%')
        ax.fill_between(range(1, n_components+1), cumsum_var, alpha=0.2, color='#e74c3c')
        ax.set_xlabel('Number of Components', fontsize=10)
        ax.set_ylabel('Cumulative Explained Variance', fontsize=10)
        ax.set_title('Cumulative Variance Explained', fontsize=12, fontweight='bold')
        ax.set_xticks(range(1, n_components+1))
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pca_variance.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # ========== t-SNE ==========
        print("\n  🔄 执行 t-SNE 降维...")
        perplexity = min(30, len(self.X_scaled) - 1)

        # 兼容新旧版本 scikit-learn
        try:
            tsne = TSNE(n_components=2, perplexity=perplexity,
                        random_state=42, max_iter=1000)
        except TypeError:
            tsne = TSNE(n_components=2, perplexity=perplexity,
                        random_state=42, n_iter=1000)

        self.X_tsne = tsne.fit_transform(self.X_scaled)
        print("  ✅ t-SNE 完成")

        return self

    def find_optimal_k(self, k_range: tuple = (3, 8)):
        """
        确定最优聚类数

        Args:
            k_range: 搜索范围 (min_k, max_k)
        """
        print("\n" + "=" * 70)
        print("[Step 2] 确定最优聚类数 (K-Means++)")
        print("=" * 70)

        k_min, k_max = k_range
        k_values = list(range(k_min, k_max + 1))

        metrics = {
            'k': k_values,
            'inertia': [],
            'silhouette': [],
            'calinski': [],
            'davies_bouldin': []
        }

        print(f"\n  🔍 搜索范围: k = {k_min} ~ {k_max}")
        print("\n  " + "-" * 60)
        print(f"  {'K':<5} {'Inertia':<12} {'Silhouette':<12} {'CH Index':<12} {'DB Index':<10}")
        print("  " + "-" * 60)

        for k in k_values:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=30,
                           max_iter=500, random_state=42)
            labels = kmeans.fit_predict(self.X_scaled)

            inertia = kmeans.inertia_
            silhouette = silhouette_score(self.X_scaled, labels)
            calinski = calinski_harabasz_score(self.X_scaled, labels)
            db_score = davies_bouldin_score(self.X_scaled, labels)

            metrics['inertia'].append(inertia)
            metrics['silhouette'].append(silhouette)
            metrics['calinski'].append(calinski)
            metrics['davies_bouldin'].append(db_score)

            print(f"  {k:<5} {inertia:<12.1f} {silhouette:<12.4f} {calinski:<12.1f} {db_score:<10.4f}")

        print("  " + "-" * 60)

        # ========== 综合评分选择最优k ==========
        # 标准化各指标到 [0, 1]
        sil_arr = np.array(metrics['silhouette'])
        ch_arr = np.array(metrics['calinski'])
        db_arr = np.array(metrics['davies_bouldin'])

        sil_norm = (sil_arr - sil_arr.min()) / (sil_arr.max() - sil_arr.min() + 1e-8)
        ch_norm = (ch_arr - ch_arr.min()) / (ch_arr.max() - ch_arr.min() + 1e-8)
        db_norm = 1 - (db_arr - db_arr.min()) / (db_arr.max() - db_arr.min() + 1e-8)  # DB越小越好

        # 综合得分（轮廓系数权重最高）
        composite_score = 0.5 * sil_norm + 0.3 * ch_norm + 0.2 * db_norm
        best_idx = np.argmax(composite_score)
        self.best_k = k_values[best_idx]

        print(f"\n  📊 各指标推荐:")
        print(f"     - 轮廓系数最优: k = {k_values[np.argmax(sil_arr)]} (score = {sil_arr.max():.4f})")
        print(f"     - CH Index最优: k = {k_values[np.argmax(ch_arr)]}")
        print(f"     - DB Index最优: k = {k_values[np.argmin(db_arr)]}")
        print(f"\n  🎯 综合评分最优: k = {self.best_k} (Silhouette = {metrics['silhouette'][best_idx]:.4f})")

        # 绘制评估图
        self._plot_metrics(metrics, k_values, self.best_k)

        self.metrics = metrics
        return self

    def _plot_metrics(self, metrics, k_values, best_k):
        """绘制聚类数评估图"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 肘部法则
        ax = axes[0, 0]
        ax.plot(k_values, metrics['inertia'], 'bo-', linewidth=2, markersize=8)
        ax.axvline(x=best_k, color='red', linestyle='--', linewidth=2, label=f'Best k={best_k}')
        ax.set_xlabel('Number of Clusters (K)', fontsize=10)
        ax.set_ylabel('Inertia (SSE)', fontsize=10)
        ax.set_title('Elbow Method', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)

        # 2. 轮廓系数
        ax = axes[0, 1]
        colors = ['#e74c3c' if k == best_k else '#3498db' for k in k_values]
        bars = ax.bar(k_values, metrics['silhouette'], color=colors, edgecolor='white', linewidth=1.5)
        ax.set_xlabel('Number of Clusters (K)', fontsize=10)
        ax.set_ylabel('Silhouette Score', fontsize=10)
        ax.set_title('Silhouette Score', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(k_values)

        # 添加数值标签
        for bar, val in zip(bars, metrics['silhouette']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        # 3. Calinski-Harabasz Index
        ax = axes[1, 0]
        ax.plot(k_values, metrics['calinski'], 'go-', linewidth=2, markersize=8)
        ax.axvline(x=best_k, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Number of Clusters (K)', fontsize=10)
        ax.set_ylabel('Calinski-Harabasz Index', fontsize=10)
        ax.set_title('Calinski-Harabasz Index (Higher=Better)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)

        # 4. Davies-Bouldin Index
        ax = axes[1, 1]
        ax.plot(k_values, metrics['davies_bouldin'], 'mo-', linewidth=2, markersize=8)
        ax.axvline(x=best_k, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Number of Clusters (K)', fontsize=10)
        ax.set_ylabel('Davies-Bouldin Index', fontsize=10)
        ax.set_title('Davies-Bouldin Index (Lower=Better)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)

        plt.suptitle(f'Optimal K Analysis (Best: k={best_k})', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'optimal_k_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n  📊 聚类数评估图已保存: optimal_k_analysis.png")

    def run_clustering(self, k: int = None):
        """
        执行 K-Means++ 聚类

        Args:
            k: 聚类数，默认使用自动推荐的最优k
        """
        if k is None:
            k = self.best_k

        print("\n" + "=" * 70)
        print(f"[Step 3] 执行 K-Means++ 聚类 (k = {k})")
        print("=" * 70)

        # K-Means++ 聚类（增加n_init以提高稳定性）
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=50,
                        max_iter=500, random_state=42)
        self.best_labels = kmeans.fit_predict(self.X_scaled)
        self.best_algorithm = 'K-Means++'
        self.kmeans_model = kmeans

        # 计算聚类质量指标
        sil = silhouette_score(self.X_scaled, self.best_labels)
        ch = calinski_harabasz_score(self.X_scaled, self.best_labels)
        db = davies_bouldin_score(self.X_scaled, self.best_labels)

        print(f"\n  📊 聚类质量指标:")
        print(f"     - 轮廓系数 (Silhouette): {sil:.4f}")
        print(f"     - CH Index: {ch:.1f}")
        print(f"     - DB Index: {db:.4f}")

        # ========== 簇大小分布 ==========
        cluster_sizes = pd.Series(self.best_labels).value_counts().sort_index()

        print(f"\n  📊 簇大小分布:")
        for cluster_id in sorted(cluster_sizes.index):
            size = cluster_sizes[cluster_id]
            pct = size / len(self.best_labels) * 100
            bar_len = int(pct / 3)
            bar = "█" * bar_len
            print(f"     Cluster {cluster_id}: {size:>4} ({pct:>5.1f}%) {bar}")

        # ========== 平衡性检查 ==========
        min_size = cluster_sizes.min()
        max_size = cluster_sizes.max()
        balance_ratio = min_size / max_size

        print(f"\n  📊 平衡性检查:")
        print(f"     - 最小簇: {min_size} ({min_size/len(self.best_labels)*100:.1f}%)")
        print(f"     - 最大簇: {max_size} ({max_size/len(self.best_labels)*100:.1f}%)")
        print(f"     - 平衡比: {balance_ratio:.3f}")

        if balance_ratio < 0.05:
            print(f"\n  ⚠️ 严重警告：簇大小极度不平衡！")
        elif balance_ratio < 0.1:
            print(f"\n  ⚠️ 警告：簇大小严重不平衡，考虑调整k值")
        elif balance_ratio < 0.2:
            print(f"\n  ⚠️ 注意：簇大小存在一定不平衡")
        else:
            print(f"\n  ✅ 簇大小分布较为均衡")

        # 存储算法比较结果
        self.algorithm_results = {
            'K-Means++': {
                'labels': self.best_labels,
                'silhouette': sil,
                'calinski': ch,
                'davies_bouldin': db
            }
        }

        # 绘制聚类结果可视化
        self._plot_clustering_results()

        return self

    def _plot_clustering_results(self):
        """绘制聚类结果可视化"""
        n_clusters = len(np.unique(self.best_labels))

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # t-SNE 可视化
        ax = axes[0]
        for i in range(n_clusters):
            mask = self.best_labels == i
            ax.scatter(self.X_tsne[mask, 0], self.X_tsne[mask, 1],
                      c=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                      label=f'Cluster {i} (n={mask.sum()})',
                      alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
        ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
        ax.set_title('Cluster Distribution (t-SNE)', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)

        # PCA 可视化
        ax = axes[1]
        for i in range(n_clusters):
            mask = self.best_labels == i
            ax.scatter(self.X_pca[mask, 0], self.X_pca[mask, 1],
                      c=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                      label=f'Cluster {i}',
                      alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
        ax.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
        ax.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
        ax.set_title('Cluster Distribution (PCA)', fontsize=12, fontweight='bold')

        plt.suptitle(f'K-Means++ Clustering Results (k={n_clusters})', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'clustering_results.png'), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n  📊 聚类结果图已保存: clustering_results.png")

    def analyze_clusters(self):
        """聚类结果深入分析"""
        print("\n" + "=" * 70)
        print("[Step 4] 聚类结果分析")
        print("=" * 70)

        # 创建分析数据框
        df_analysis = self.df[self.feature_cols].copy()
        df_analysis['cluster'] = self.best_labels
        df_analysis[self.asin_col] = self.asins

        n_clusters = len(np.unique(self.best_labels))

        # ========== 计算各簇特征均值 ==========
        cluster_profiles = df_analysis.groupby('cluster')[self.feature_cols].mean()

        # ========== 计算 Z-score 标准化的特征画像 ==========
        overall_mean = df_analysis[self.feature_cols].mean()
        overall_std = df_analysis[self.feature_cols].std()
        cluster_profiles_z = (cluster_profiles - overall_mean) / (overall_std + 1e-8)

        self.cluster_profiles = cluster_profiles
        self.cluster_profiles_z = cluster_profiles_z

        # ========== 各簇显著特征分析 ==========
        print("\n  🔍 各簇显著特征 (z-score > 0.5 或 < -0.5):")
        print("  " + "-" * 60)

        cluster_descriptions = {}
        cluster_sizes = df_analysis['cluster'].value_counts().sort_index()

        for cluster_id in range(n_clusters):
            z_scores = cluster_profiles_z.loc[cluster_id]

            # 高于平均的特征
            high_features = z_scores[z_scores > 0.5].sort_values(ascending=False)
            # 低于平均的特征
            low_features = z_scores[z_scores < -0.5].sort_values()

            size = cluster_sizes[cluster_id]
            pct = size / len(df_analysis) * 100

            print(f"\n  【Cluster {cluster_id}】 ({size} 个商品, {pct:.1f}%)")

            if len(high_features) > 0:
                print(f"    ↑ 高于平均:")
                for feat, val in high_features.head(5).items():
                    print(f"       {feat}: z = {val:+.2f}")

            if len(low_features) > 0:
                print(f"    ↓ 低于平均:")
                for feat, val in low_features.head(5).items():
                    print(f"       {feat}: z = {val:+.2f}")

            if len(high_features) == 0 and len(low_features) == 0:
                print(f"    (特征接近市场平均)")

            cluster_descriptions[cluster_id] = {
                'size': size,
                'pct': pct,
                'high_features': high_features.head(5).to_dict(),
                'low_features': low_features.head(5).to_dict()
            }

        self.cluster_descriptions = cluster_descriptions
        self.df_clustered = df_analysis

        # 生成可视化
        self._plot_cluster_analysis_dashboard(n_clusters)
        self._plot_radar_chart(n_clusters)
        self._plot_dendrogram()

        return self

    def _plot_cluster_analysis_dashboard(self, n_clusters):
        """生成聚类分析综合面板"""
        fig = plt.figure(figsize=(18, 14))

        # 1. t-SNE 可视化
        ax1 = fig.add_subplot(2, 3, 1)
        for i in range(n_clusters):
            mask = self.best_labels == i
            ax1.scatter(self.X_tsne[mask, 0], self.X_tsne[mask, 1],
                       c=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                       label=f'C{i} (n={mask.sum()})', alpha=0.6, s=40)
        ax1.set_title('t-SNE Visualization', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=7, loc='best')
        ax1.set_xlabel('t-SNE 1')
        ax1.set_ylabel('t-SNE 2')

        # 2. 簇大小分布饼图
        ax2 = fig.add_subplot(2, 3, 2)
        sizes = pd.Series(self.best_labels).value_counts().sort_index()
        colors_pie = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(n_clusters)]
        wedges, texts, autotexts = ax2.pie(
            sizes,
            labels=[f'C{i}' for i in range(n_clusters)],
            autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*len(self.best_labels))})',
            colors=colors_pie,
            explode=[0.02]*n_clusters,
            textprops={'fontsize': 9}
        )
        ax2.set_title('Cluster Size Distribution', fontsize=11, fontweight='bold')

        # 3. 轮廓系数分析图
        ax3 = fig.add_subplot(2, 3, 3)
        silhouette_vals = silhouette_samples(self.X_scaled, self.best_labels)
        y_lower = 10

        for i in range(n_clusters):
            cluster_silhouette = silhouette_vals[self.best_labels == i]
            cluster_silhouette.sort()
            size_cluster = cluster_silhouette.shape[0]
            y_upper = y_lower + size_cluster

            ax3.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette,
                             facecolor=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                             alpha=0.7, edgecolor='white')
            ax3.text(-0.05, y_lower + 0.5 * size_cluster, str(i), fontsize=9)
            y_lower = y_upper + 10

        avg_sil = silhouette_vals.mean()
        ax3.axvline(x=avg_sil, color='red', linestyle='--', linewidth=2,
                   label=f'Avg: {avg_sil:.3f}')
        ax3.set_xlabel('Silhouette Coefficient', fontsize=10)
        ax3.set_ylabel('Cluster', fontsize=10)
        ax3.set_title('Silhouette Analysis', fontsize=11, fontweight='bold')
        ax3.legend(loc='upper right')

        # 4. 关键特征箱线图
        ax4 = fig.add_subplot(2, 3, 4)
        key_feats = ['log_price', 'product_rating', 'log_sales', 'weighted_rating']
        key_feats = [f for f in key_feats if f in self.feature_cols][:3]

        if key_feats:
            plot_data = self.df_clustered[['cluster'] + key_feats].melt(
                id_vars='cluster', var_name='Feature', value_name='Value')
            sns.boxplot(data=plot_data, x='Feature', y='Value', hue='cluster',
                       palette=CLUSTER_COLORS[:n_clusters], ax=ax4)
            ax4.set_title('Key Features by Cluster', fontsize=11, fontweight='bold')
            ax4.legend(title='Cluster', fontsize=7, title_fontsize=8)
            ax4.tick_params(axis='x', rotation=15)

        # 5. 特征热力图（Top 15 特征）
        ax5 = fig.add_subplot(2, 3, 5)
        top_features = self.cluster_profiles_z.abs().mean().nlargest(15).index.tolist()
        heatmap_data = self.cluster_profiles_z[top_features].T

        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, ax=ax5, cbar_kws={'shrink': 0.8},
                   xticklabels=[f'C{i}' for i in range(n_clusters)],
                   annot_kws={'size': 8})
        ax5.set_title('Feature Z-Scores Heatmap (Top 15)', fontsize=11, fontweight='bold')
        ax5.tick_params(axis='y', rotation=0, labelsize=8)

        # 6. 簇间距离矩阵
        ax6 = fig.add_subplot(2, 3, 6)
        cluster_centers = self.cluster_profiles.values
        from scipy.spatial.distance import pdist, squareform
        dist_matrix = squareform(pdist(cluster_centers, metric='euclidean'))

        sns.heatmap(dist_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=[f'C{i}' for i in range(n_clusters)],
                   yticklabels=[f'C{i}' for i in range(n_clusters)],
                   ax=ax6, cbar_kws={'shrink': 0.8})
        ax6.set_title('Inter-Cluster Distance', fontsize=11, fontweight='bold')

        plt.suptitle('Cluster Analysis Dashboard', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cluster_analysis_dashboard.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n  📊 聚类分析面板已保存: cluster_analysis_dashboard.png")

    def _plot_radar_chart(self, n_clusters):
        """绘制雷达图"""
        # 选择关键特征
        radar_feats = [
            'log_price', 'product_rating', 'log_sales', 'log_reviews',
            'is_set', 'positive_ratio', 'is_fba', 'discount_rate'
        ]
        radar_feats = [f for f in radar_feats if f in self.feature_cols][:8]

        if len(radar_feats) < 4:
            # 如果关键特征不足，选择变异最大的特征
            var_rank = self.cluster_profiles_z.var().nlargest(8).index.tolist()
            radar_feats = var_rank

        # 归一化数据
        radar_data = self.cluster_profiles[radar_feats].copy()
        radar_norm = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min() + 1e-8)

        # 创建雷达图
        angles = np.linspace(0, 2*np.pi, len(radar_feats), endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        for i in range(n_clusters):
            values = radar_norm.loc[i].tolist()
            values += values[:1]  # 闭合

            ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {i}',
                   color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)], markersize=6)
            ax.fill(angles, values, alpha=0.15, color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_feats, fontsize=10)
        ax.set_title('Cluster Profiles Radar Chart', fontsize=12, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cluster_radar_chart.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  📊 雷达图已保存: cluster_radar_chart.png")

    def _plot_dendrogram(self):
        """绘制层次聚类树状图"""
        # 采样（大数据集时）
        n_sample = min(200, len(self.X_scaled))
        np.random.seed(42)
        sample_idx = np.random.choice(len(self.X_scaled), n_sample, replace=False)
        X_sample = self.X_scaled[sample_idx]

        # 计算层次聚类
        linkage_matrix = linkage(X_sample, method='ward')

        fig, ax = plt.subplots(figsize=(14, 6))
        dendrogram(linkage_matrix, ax=ax, truncate_mode='level', p=5,
                   leaf_rotation=90, leaf_font_size=8,
                   color_threshold=0.7*max(linkage_matrix[:,2]))

        ax.set_title('Hierarchical Clustering Dendrogram (Ward Method)',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Sample Index', fontsize=10)
        ax.set_ylabel('Distance', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'dendrogram.png'), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  📊 树状图已保存: dendrogram.png")

    def _auto_name_clusters(self):
        """智能命名簇 - 修正版"""
        cluster_names = {}
        n_clusters = len(np.unique(self.best_labels))

        for cluster_id in range(n_clusters):
            profile = self.cluster_profiles.loc[cluster_id]
            z = self.cluster_profiles_z.loc[cluster_id]
            desc = self.cluster_descriptions[cluster_id]
            size_pct = desc['pct']

            parts = []

            # ========== 规模标签 ==========
            if size_pct > 35:
                parts.append('主流')
            elif size_pct > 15:
                parts.append('中等')
            elif size_pct < 5:
                parts.append('细分')

            # ========== 价格定位（修正版）==========
            # 注意：使用 log_price_per_piece（单价）而非 log_price（总价）
            price_z = z.get('log_price_per_piece', z.get('log_price', 0))

            if price_z > 0.5:
                parts.append('高单价')  # 修正：正z值=高于平均
            elif price_z > 0.2:
                parts.append('中高单价')
            elif price_z < -0.5:
                parts.append('低单价')  # 修正：负z值=低于平均
            elif price_z < -0.2:
                parts.append('平价')

            # ========== 产品形态 ==========
            is_set_val = profile.get('is_set', 0.5)
            if is_set_val > 0.6:
                parts.append('套装')
            elif is_set_val < 0.4:
                parts.append('单品')

            # ========== 口碑特征 ==========
            sentiment_z = z.get('positive_ratio', z.get('aspect_sentiment_mean', 0))
            rating_z = z.get('product_rating', 0)

            if sentiment_z > 1.5 or rating_z > 1.0:
                parts.append('口碑佳')
            elif sentiment_z < -2 or rating_z < -1.5:
                parts.append('待改进')

            # ========== 销量/曝光特征 ==========
            sales_z = z.get('log_sales', 0)
            reviews_z = z.get('log_reviews', 0)
            bsr_z = z.get('log_bsr', 0)

            if sales_z > 0.5:
                parts.append('畅销')
            elif sales_z < -1.0 or reviews_z < -1.5 or bsr_z < -1.5:
                parts.append('低曝光')

            # ========== 材质/风格特色 ==========
            if profile.get('is_damascus', 0) > 0.15:
                parts.append('大马士革')
            elif profile.get('is_german_steel', 0) > 0.12:
                parts.append('德系')
            elif profile.get('is_japanese_steel', 0) > 0.12:
                parts.append('日系')
            elif profile.get('is_ceramic', 0) > 0.08:
                parts.append('陶瓷')

            # 组合名称（最多取3个标签）
            if parts:
                cluster_names[cluster_id] = '-'.join(parts[:3])
            else:
                cluster_names[cluster_id] = f'市场细分{cluster_id}'

            # ========== 特殊情况处理 ==========
            # 如果只有"待改进"没有其他标签，添加更多上下文
            if cluster_names[cluster_id] == '待改进':
                if size_pct < 15:
                    cluster_names[cluster_id] = '长尾-待激活'
                else:
                    cluster_names[cluster_id] = '质量待改进'

        return cluster_names

    def _generate_suggestions(self, cluster_id, desc, z_scores):
        """
        生成商业建议 - 基于簇特征
        """
        suggestions = []
        high = desc['high_features']
        low = desc['low_features']

        # 高端市场建议
        if z_scores.get('log_price', 0) > 0.5:
            suggestions.append("高端市场定位，强化品质故事和品牌溢价能力")

        # 经济型市场建议
        if z_scores.get('log_price', 0) < -0.5:
            suggestions.append("价格敏感群体，可通过套装组合或增值服务提升客单价")

        # 口碑问题
        if 'positive_ratio' in low or 'aspect_sentiment_mean' in low:
            suggestions.append("关注用户负面反馈，改善产品质量和使用体验")

        # 口碑优势
        if 'positive_ratio' in high or 'aspect_sentiment_mean' in high:
            suggestions.append("口碑是核心优势，鼓励用户评价，做好口碑营销")

        # 畅销品
        if 'log_sales' in high:
            suggestions.append("热销品类，可测试价格弹性，适当提价")

        # 小众品
        if 'log_sales' in low or 'log_reviews' in low:
            suggestions.append("曝光不足，加强广告投放和关键词优化")

        # 防锈痛点
        if 'aspect_rust_sentiment' in low:
            suggestions.append("防锈是用户痛点，强调不锈钢材质或提供保养指南")

        # 锋利优势
        if 'aspect_sharpness_sentiment' in high:
            suggestions.append("锋利度是产品优势，可作为核心卖点重点宣传")

        # 套装产品
        if z_scores.get('is_set', 0) > 0.5:
            suggestions.append("套装市场，关注刀座设计和组合搭配")

        # 单品市场
        if z_scores.get('is_set', 0) < -0.3:
            suggestions.append("单品市场，专注专业用户需求，突出专业性能")

        # 默认建议
        if not suggestions:
            suggestions.append("维持现有产品策略，持续监测市场竞争动态")

        return suggestions[:4]  # 最多返回4条建议

    def generate_business_insights(self):
        """生成商业洞察报告"""
        print("\n" + "=" * 70)
        print("[Step 5] 商业洞察报告")
        print("=" * 70)

        # 自动命名
        cluster_names = self._auto_name_clusters()
        self.cluster_names = cluster_names

        print("\n  🏷️ 智能簇命名:")
        for cid, name in cluster_names.items():
            size = self.cluster_descriptions[cid]['size']
            pct = self.cluster_descriptions[cid]['pct']
            print(f"     Cluster {cid}: {name} ({size}个, {pct:.1f}%)")

        # ========== 构建报告 ==========
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("          厨刀市场聚类分析 - 商业洞察报告")
        report_lines.append("=" * 80)
        report_lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"分析样本: {len(self.df)} 个商品")
        report_lines.append(f"有效特征: {len(self.feature_cols)} 个")
        report_lines.append(f"聚类数量: {len(np.unique(self.best_labels))} 个细分市场")
        report_lines.append(f"聚类算法: {self.best_algorithm}")
        report_lines.append(f"轮廓系数: {silhouette_score(self.X_scaled, self.best_labels):.4f}")

        report_lines.append("\n" + "=" * 80)
        report_lines.append("                      市场细分概览")
        report_lines.append("=" * 80)

        n_clusters = len(np.unique(self.best_labels))

        for cluster_id in range(n_clusters):
            name = cluster_names[cluster_id]
            desc = self.cluster_descriptions[cluster_id]
            z_scores = self.cluster_profiles_z.loc[cluster_id]

            size = desc['size']
            pct = desc['pct']

            report_lines.append(f"\n{'─' * 80}")
            report_lines.append(f"【Cluster {cluster_id}】 {name}")
            report_lines.append(f"{'─' * 80}")
            report_lines.append(f"  📊 规模: {size} 个商品 ({pct:.1f}%)")

            # 核心特征
            report_lines.append(f"\n  ✅ 核心优势 (高于市场平均):")
            if desc['high_features']:
                for feat, z in list(desc['high_features'].items())[:5]:
                    report_lines.append(f"      • {feat}: z = {z:+.2f}")
            else:
                report_lines.append(f"      (无显著高于平均的特征)")

            # 弱势特征
            report_lines.append(f"\n  ⚠️ 改进空间 (低于市场平均):")
            if desc['low_features']:
                for feat, z in list(desc['low_features'].items())[:5]:
                    report_lines.append(f"      • {feat}: z = {z:+.2f}")
            else:
                report_lines.append(f"      (无显著低于平均的特征)")

            # 商业建议
            report_lines.append(f"\n  💡 商业建议:")
            suggestions = self._generate_suggestions(cluster_id, desc, z_scores.to_dict())
            for i, sug in enumerate(suggestions, 1):
                report_lines.append(f"      {i}. {sug}")

        # ========== 整体市场洞察 ==========
        report_lines.append("\n" + "=" * 80)
        report_lines.append("                      整体市场洞察")
        report_lines.append("=" * 80)

        insights = self._generate_overall_insights()
        for insight in insights:
            report_lines.append(f"\n  {insight}")

        # ========== 行动建议汇总 ==========
        report_lines.append("\n" + "=" * 80)
        report_lines.append("                      战略建议")
        report_lines.append("=" * 80)

        strategic_suggestions = self._generate_strategic_suggestions()
        for i, sug in enumerate(strategic_suggestions, 1):
            report_lines.append(f"\n  {i}. {sug}")

        report_lines.append("\n" + "=" * 80)
        report_lines.append("                        报告结束")
        report_lines.append("=" * 80)

        report_text = "\n".join(report_lines)

        # 保存报告
        report_path = os.path.join(self.output_dir, 'business_insights_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        # 打印报告
        print(report_text)

        print(f"\n  📄 报告已保存: {report_path}")

        return self

    def _generate_overall_insights(self):
        """生成整体市场洞察"""
        insights = []

        # 市场集中度
        cluster_sizes = pd.Series(self.best_labels).value_counts()
        top2_share = cluster_sizes.nlargest(2).sum() / len(self.best_labels)

        if top2_share > 0.7:
            insights.append(f"📈 市场高度集中：前2个细分市场占据 {top2_share*100:.1f}% 的份额")
        elif top2_share > 0.5:
            insights.append(f"📊 市场中等集中：前2个细分市场占 {top2_share*100:.1f}% 的份额")
        else:
            insights.append(f"📊 市场较为分散：前2个细分市场仅占 {top2_share*100:.1f}%")

        # 价格分析
        if 'log_price' in self.feature_cols:
            price_range = self.df_clustered['log_price'].max() - self.df_clustered['log_price'].min()
            insights.append(f"💰 价格跨度：对数价格范围 {price_range:.2f}，存在明显价格分层")

        # 情感分析
        if 'positive_ratio' in self.feature_cols:
            avg_sent = self.df_clustered['positive_ratio'].mean()
            insights.append(f"💬 整体用户情感：平均正向比例 {avg_sent:.3f}，{'整体正面' if avg_sent > 0.5 else '有提升空间'}")

        # 套装分析
        if 'is_set' in self.feature_cols:
            set_ratio = self.df_clustered['is_set'].mean()
            insights.append(f"📦 产品形态：{set_ratio*100:.1f}% 为套装商品，{'套装为主' if set_ratio > 0.5 else '单品为主'}")

        # FBA分析
        if 'is_fba' in self.feature_cols:
            fba_ratio = self.df_clustered['is_fba'].mean()
            insights.append(f"🚚 物流模式：{fba_ratio*100:.1f}% 使用FBA配送")

        # 评分分布
        if 'product_rating' in self.feature_cols:
            avg_rating = self.df_clustered['product_rating'].mean()
            insights.append(f"⭐ 平均评分：{avg_rating:.2f} 分")

        return insights

    def _generate_strategic_suggestions(self):
        """生成战略建议"""
        suggestions = []

        # 基于市场结构的建议
        cluster_sizes = pd.Series(self.best_labels).value_counts()
        largest_cluster = cluster_sizes.idxmax()
        smallest_cluster = cluster_sizes.idxmin()

        largest_name = self.cluster_names.get(largest_cluster, f'Cluster {largest_cluster}')
        smallest_name = self.cluster_names.get(smallest_cluster, f'Cluster {smallest_cluster}')

        suggestions.append(f"【主战场】'{largest_name}' 是最大细分市场，竞争激烈，需差异化定位")
        suggestions.append(f"【蓝海机会】'{smallest_name}' 是小众市场，可评估是否存在未被满足的需求")

        # 基于口碑的建议
        if 'positive_ratio' in self.feature_cols:
            low_sentiment_clusters = []
            for cid in range(len(cluster_sizes)):
                z = self.cluster_profiles_z.loc[cid].get('positive_ratio', 0)
                if z < -1.5:
                    low_sentiment_clusters.append(cid)

            if low_sentiment_clusters:
                names = [self.cluster_names.get(c, f'C{c}') for c in low_sentiment_clusters]
                suggestions.append(f"【质量警示】{', '.join(names)} 用户反馈较差，亟需产品质量改进")

        # 基于价格的建议
        if 'log_price' in self.feature_cols:
            price_variance = self.cluster_profiles_z['log_price'].std()
            if price_variance > 0.8:
                suggestions.append("【价格策略】各细分市场价格差异明显，可针对性制定定价策略")
            else:
                suggestions.append("【价格机会】各细分市场价格趋同，存在高端化或性价比差异化空间")

        # 套装vs单品
        if 'is_set' in self.feature_cols:
            set_clusters = []
            single_clusters = []
            for cid in range(len(cluster_sizes)):
                is_set_val = self.cluster_profiles.loc[cid].get('is_set', 0.5)
                if is_set_val > 0.7:
                    set_clusters.append(cid)
                elif is_set_val < 0.3:
                    single_clusters.append(cid)

            if set_clusters and single_clusters:
                suggestions.append("【产品形态】套装与单品形成明显分野，可根据目标市场选择产品形态")

        return suggestions

    def save_results(self):
        """保存所有结果"""
        print("\n" + "=" * 70)
        print("[Step 6] 保存结果")
        print("=" * 70)

        # 1. 带聚类标签的完整数据
        output_df = self.df.copy()
        output_df['cluster'] = self.best_labels
        output_df['cluster_name'] = output_df['cluster'].map(self.cluster_names)
        output_df['tsne_x'] = self.X_tsne[:, 0]
        output_df['tsne_y'] = self.X_tsne[:, 1]
        output_df['pca_x'] = self.X_pca[:, 0]
        output_df['pca_y'] = self.X_pca[:, 1]

        output_path = os.path.join(self.output_dir, 'clustered_products.csv')
        output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"  ✅ {output_path}")

        # 2. 簇特征画像（原始均值）
        profiles_path = os.path.join(self.output_dir, 'cluster_profiles.csv')
        self.cluster_profiles.to_csv(profiles_path, encoding='utf-8-sig')
        print(f"  ✅ {profiles_path}")

        # 3. 簇特征画像（Z-score）
        profiles_z_path = os.path.join(self.output_dir, 'cluster_profiles_zscore.csv')
        self.cluster_profiles_z.to_csv(profiles_z_path, encoding='utf-8-sig')
        print(f"  ✅ {profiles_z_path}")

        # 4. 聚类评估指标
        metrics_df = pd.DataFrame(self.metrics)
        metrics_path = os.path.join(self.output_dir, 'clustering_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
        print(f"  ✅ {metrics_path}")

        # 5. 算法比较结果
        algo_results = []
        for algo, res in self.algorithm_results.items():
            algo_results.append({
                'algorithm': algo,
                'silhouette': res['silhouette'],
                'calinski_harabasz': res['calinski'],
                'davies_bouldin': res['davies_bouldin']
            })
        algo_df = pd.DataFrame(algo_results)
        algo_path = os.path.join(self.output_dir, 'algorithm_comparison.csv')
        algo_df.to_csv(algo_path, index=False, encoding='utf-8-sig')
        print(f"  ✅ {algo_path}")

        # 6. 簇命名映射
        names_df = pd.DataFrame([
            {'cluster': k, 'name': v,
             'size': self.cluster_descriptions[k]['size'],
             'percentage': self.cluster_descriptions[k]['pct']}
            for k, v in self.cluster_names.items()
        ])
        names_path = os.path.join(self.output_dir, 'cluster_names.csv')
        names_df.to_csv(names_path, index=False, encoding='utf-8-sig')
        print(f"  ✅ {names_path}")

        print(f"\n  📁 所有结果已保存至: {self.output_dir}/")

        return self

    def run_full_pipeline(self, k_range: tuple = (3, 8), final_k: int = None,
                          scaler_type: str = 'standard'):
        """
        运行完整聚类分析流水线

        Args:
            k_range: 搜索最优k的范围
            final_k: 指定最终使用的k值，None则使用自动推荐
            scaler_type: 标准化方式
        """
        # Step 1: 预处理
        self.preprocess(scaler_type=scaler_type)
        self.analyze_features()
        self.reduce_dimensions()

        # Step 2: 确定最优k
        self.find_optimal_k(k_range=k_range)

        # 如果指定了final_k，使用指定值
        if final_k is not None:
            print(f"\n  📌 使用指定聚类数: k = {final_k}")
            self.best_k = final_k

        # Step 3: 执行聚类
        self.run_clustering(k=self.best_k)

        # Step 4: 分析结果
        self.analyze_clusters()

        # Step 5: 商业洞察
        self.generate_business_insights()

        # Step 6: 保存结果
        self.save_results()

        # 打印完成信息
        self._print_summary()

        return self

    def _print_summary(self):
        """打印分析摘要"""
        sil_score = silhouette_score(self.X_scaled, self.best_labels)

        print("\n" + "=" * 70)
        print("                    ✅ 聚类分析完成！")
        print("=" * 70)

        print(f"\n  📊 最终聚类数: {self.best_k}")
        print(f"  📈 聚类算法: {self.best_algorithm}")
        print(f"  📉 轮廓系数: {sil_score:.4f}")

        print(f"\n  🏷️ 市场细分:")
        for cid, name in self.cluster_names.items():
            size = self.cluster_descriptions[cid]['size']
            pct = self.cluster_descriptions[cid]['pct']
            print(f"     C{cid}: {name} ({size}个, {pct:.1f}%)")

        print(f"\n  📁 输出文件夹: {self.output_dir}/")
        print("     ├── clustered_products.csv        (带聚类标签的商品数据)")
        print("     ├── cluster_profiles.csv          (各簇特征均值)")
        print("     ├── cluster_profiles_zscore.csv   (各簇特征Z分数)")
        print("     ├── cluster_names.csv             (簇命名映射)")
        print("     ├── clustering_metrics.csv        (聚类评估指标)")
        print("     ├── algorithm_comparison.csv      (算法比较结果)")
        print("     ├── business_insights_report.txt  (商业洞察报告)")
        print("     ├── cluster_analysis_dashboard.png(聚类分析面板)")
        print("     ├── cluster_radar_chart.png       (雷达图)")
        print("     ├── clustering_results.png        (聚类可视化)")
        print("     ├── correlation_matrix.png        (相关性热力图)")
        print("     ├── optimal_k_analysis.png        (最优K分析)")
        print("     ├── dendrogram.png                (层次聚类树状图)")
        print("     └── pca_variance.png              (PCA方差解释)")
        print("=" * 70)


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""

    # ==================== 配置参数 ====================
    DATA_PATH = 'clustering_features_only.csv'  # 输入数据
    OUTPUT_DIR = 'clustering_results'            # 输出目录
    K_RANGE = (4, 8)                             # K值搜索范围
    FINAL_K = 5                                  # 最终使用的K值（None=自动选择）
    SCALER_TYPE = 'standard'                     # 标准化方式
    # ================================================

    # 创建流水线
    pipeline = ClusteringPipelineOptimized(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR
    )

    # 运行完整流水线
    pipeline.run_full_pipeline(
        k_range=K_RANGE,
        final_k=FINAL_K,
        scaler_type=SCALER_TYPE
    )

    return pipeline


if __name__ == '__main__':
    pipeline = main()
