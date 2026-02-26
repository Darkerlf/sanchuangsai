import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

from clustering_analysis import ClusteringPipelineOptimized

# Ensure UTF-8 output to avoid GBK encoding errors on Windows consoles
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / 'clustering_features_only.csv'
ORIGINAL_RESULTS_DIR = BASE_DIR / 'clustering_results'

# ====== Rerun config ======
K_RANGE = (4, 8)
FINAL_K = 5
SCALER_TYPE = 'standard'

# Loose filter rule: remove rows only when all are zero (product_rating==0 AND log_sales==0 AND log_reviews==0)
FILTER_COLS = ['product_rating', 'log_sales', 'log_reviews']

RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = BASE_DIR / 'clustering_results_rerun' / RUN_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILTERED_PATH = OUTPUT_DIR / 'clustering_features_only_rerun.csv'


def _require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _load_algo_metrics(path: Path):
    df = pd.read_csv(path)
    if 'algorithm' in df.columns:
        row = df[df['algorithm'] == 'K-Means++']
        if not row.empty:
            return row.iloc[0]
    return df.iloc[0]


def _cluster_sizes(path: Path):
    df = pd.read_csv(path)
    return df['cluster'].value_counts().sort_index()


def main():
    print("=" * 70)
    print("Rerun clustering with strict filter (no overwrite)")
    print("=" * 70)
    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_DIR}")

    df = pd.read_csv(INPUT_PATH)
    _require_columns(df, FILTER_COLS)

    before = len(df)
    # Keep rows unless all three are zero
    mask = ~((df['product_rating'] == 0) & (df['log_sales'] == 0) & (df['log_reviews'] == 0))
    filtered = df[mask].copy()
    after = len(filtered)
    removed = before - after

    print(f"\nFilter rule: remove rows only when all of {FILTER_COLS} == 0")
    print(f"Rows before: {before}")
    print(f"Rows after : {after}")
    print(f"Removed    : {removed} ({removed / max(before, 1) * 100:.1f}%)")

    filtered.to_csv(FILTERED_PATH, index=False, encoding='utf-8-sig')
    print(f"\nFiltered data saved: {FILTERED_PATH}")

    pipeline = ClusteringPipelineOptimized(
        data_path=str(FILTERED_PATH),
        output_dir=str(OUTPUT_DIR)
    )

    pipeline.run_full_pipeline(
        k_range=K_RANGE,
        final_k=FINAL_K,
        scaler_type=SCALER_TYPE
    )

    # ====== Comparison outputs ======
    try:
        orig_algo = _load_algo_metrics(ORIGINAL_RESULTS_DIR / 'algorithm_comparison.csv')
        new_algo = _load_algo_metrics(OUTPUT_DIR / 'algorithm_comparison.csv')

        comparison_df = pd.DataFrame({
            'metric': ['silhouette', 'calinski_harabasz', 'davies_bouldin'],
            'original': [orig_algo['silhouette'], orig_algo['calinski_harabasz'], orig_algo['davies_bouldin']],
            'rerun': [new_algo['silhouette'], new_algo['calinski_harabasz'], new_algo['davies_bouldin']]
        })
        comparison_df.to_csv(OUTPUT_DIR / 'comparison_metrics.csv', index=False, encoding='utf-8-sig')

        orig_sizes = _cluster_sizes(ORIGINAL_RESULTS_DIR / 'clustered_products.csv')
        new_sizes = _cluster_sizes(OUTPUT_DIR / 'clustered_products.csv')

        # Write summary
        summary_path = OUTPUT_DIR / 'comparison_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Rerun Comparison Summary\n")
            f.write("=" * 60 + "\n")
            f.write(f"Run ID: {RUN_ID}\n")
            f.write(f"Input: {INPUT_PATH}\n")
            f.write(f"Filtered data: {FILTERED_PATH}\n")
            f.write(f"Output dir: {OUTPUT_DIR}\n\n")
            f.write("Filter rule:\n")
            f.write("  Remove rows only when all of product_rating/log_sales/log_reviews == 0\n")
            f.write(f"Rows before: {before}\n")
            f.write(f"Rows after : {after}\n")
            f.write(f"Removed    : {removed} ({removed / max(before, 1) * 100:.1f}%)\n\n")

            f.write("Metrics (K-Means++):\n")
            f.write(f"  Original silhouette: {orig_algo['silhouette']:.4f}\n")
            f.write(f"  Rerun    silhouette: {new_algo['silhouette']:.4f}\n")
            f.write(f"  Original CH index  : {orig_algo['calinski_harabasz']:.2f}\n")
            f.write(f"  Rerun    CH index  : {new_algo['calinski_harabasz']:.2f}\n")
            f.write(f"  Original DB index  : {orig_algo['davies_bouldin']:.4f}\n")
            f.write(f"  Rerun    DB index  : {new_algo['davies_bouldin']:.4f}\n\n")

            f.write("Cluster size distribution (original):\n")
            f.write("  " + ", ".join([f"C{k}={v}" for k, v in orig_sizes.items()]) + "\n")
            f.write("Cluster size distribution (rerun):\n")
            f.write("  " + ", ".join([f"C{k}={v}" for k, v in new_sizes.items()]) + "\n")

        # Simple comparison plot
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        # Metrics bar
        metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
        orig_vals = comparison_df['original'].values
        new_vals = comparison_df['rerun'].values
        x = np.arange(len(metrics))
        width = 0.35

        ax = axes[0, 0]
        ax.bar(x - width/2, orig_vals, width, label='original')
        ax.bar(x + width/2, new_vals, width, label='rerun')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=15)
        ax.set_title('Metrics Comparison')
        ax.legend()

        # Cluster size distribution
        ax = axes[0, 1]
        orig_sizes = orig_sizes.sort_index()
        new_sizes = new_sizes.sort_index()
        k_labels = sorted(set(orig_sizes.index.tolist() + new_sizes.index.tolist()))
        orig_plot = [orig_sizes.get(k, 0) for k in k_labels]
        new_plot = [new_sizes.get(k, 0) for k in k_labels]
        x = np.arange(len(k_labels))
        ax.bar(x - width/2, orig_plot, width, label='original')
        ax.bar(x + width/2, new_plot, width, label='rerun')
        ax.set_xticks(x)
        ax.set_xticklabels([f"C{k}" for k in k_labels])
        ax.set_title('Cluster Size Distribution')
        ax.legend()

        # Sample counts
        ax = axes[1, 0]
        ax.axis('off')
        text = (
            f"Rows before: {before}\n"
            f"Rows after : {after}\n"
            f"Removed    : {removed} ({removed / max(before, 1) * 100:.1f}%)\n\n"
            f"Original k: {len(orig_sizes)}\n"
            f"Rerun k   : {len(new_sizes)}"
        )
        ax.text(0.02, 0.98, text, va='top', ha='left', fontsize=11)
        ax.set_title('Sample Summary')

        # Placeholder for notes
        ax = axes[1, 1]
        ax.axis('off')
        ax.text(0.02, 0.98, "Notes:\n- Strict filter applied\n- New outputs saved separately",
                va='top', ha='left', fontsize=11)
        ax.set_title('Notes')

        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / 'comparison_dashboard.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"\nComparison outputs saved:\n- {OUTPUT_DIR / 'comparison_metrics.csv'}\n- {OUTPUT_DIR / 'comparison_summary.txt'}\n- {OUTPUT_DIR / 'comparison_dashboard.png'}")

    except Exception as e:
        print(f"\n[WARN] Comparison step skipped due to: {e}")


if __name__ == '__main__':
    main()
