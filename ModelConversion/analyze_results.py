#!/usr/bin/env python3
"""
Benchmark Results Analysis Script
Analyzes CSV outputs from TFLite and ORT iOS apps
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_results(tflite_csv, ort_csv):
    """Load CSV results from both frameworks"""
    df_tflite = pd.read_csv(tflite_csv)
    df_ort = pd.read_csv(ort_csv)
    
    print(f"✅ Loaded TFLite results: {len(df_tflite)} rows")
    print(f"✅ Loaded ORT results: {len(df_ort)} rows")
    
    return df_tflite, df_ort


def calculate_statistics(df, name):
    """Calculate latency statistics"""
    stats = {
        'framework': name,
        'count': len(df),
        'mean': df['latency_ms'].mean(),
        'std': df['latency_ms'].std(),
        'min': df['latency_ms'].min(),
        'max': df['latency_ms'].max(),
        'p50': df['latency_ms'].quantile(0.50),
        'p90': df['latency_ms'].quantile(0.90),
        'p95': df['latency_ms'].quantile(0.95),
        'p99': df['latency_ms'].quantile(0.99),
    }
    
    if 'memory_mb' in df.columns:
        stats['peak_memory_mb'] = df['memory_mb'].max()
        stats['avg_memory_mb'] = df['memory_mb'].mean()
    
    return stats


def print_comparison_table(stats_tflite, stats_ort):
    """Print side-by-side comparison table"""
    print("\n" + "="*80)
    print("📊 BENCHMARK RESULTS COMPARISON")
    print("="*80)
    
    metrics = [
        ('Runs', 'count', ''),
        ('Mean', 'mean', 'ms'),
        ('Std Dev', 'std', 'ms'),
        ('Min', 'min', 'ms'),
        ('Max', 'max', 'ms'),
        ('p50 (Median)', 'p50', 'ms'),
        ('p90', 'p90', 'ms'),
        ('p95', 'p95', 'ms'),
        ('p99', 'p99', 'ms'),
    ]
    
    if 'peak_memory_mb' in stats_tflite:
        metrics.extend([
            ('Peak Memory', 'peak_memory_mb', 'MB'),
            ('Avg Memory', 'avg_memory_mb', 'MB'),
        ])
    
    print(f"{'Metric':<20} {'TFLite':<20} {'ORT':<20} {'Difference':<20}")
    print("-"*80)
    
    for label, key, unit in metrics:
        tflite_val = stats_tflite[key]
        ort_val = stats_ort[key]
        
        if key == 'count':
            diff_str = ""
            tflite_str = f"{tflite_val:.0f}"
            ort_str = f"{ort_val:.0f}"
        else:
            diff = ((ort_val - tflite_val) / tflite_val) * 100
            diff_str = f"{diff:+.2f}%"
            tflite_str = f"{tflite_val:.3f} {unit}"
            ort_str = f"{ort_val:.3f} {unit}"
        
        print(f"{label:<20} {tflite_str:<20} {ort_str:<20} {diff_str:<20}")
    
    print("="*80 + "\n")


def plot_latency_distribution(df_tflite, df_ort, output_dir):
    """Plot latency distribution histograms"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Combined histogram
    ax = axes[0, 0]
    ax.hist(df_tflite['latency_ms'], bins=30, alpha=0.6, label='TFLite', color='blue', edgecolor='black')
    ax.hist(df_ort['latency_ms'], bins=30, alpha=0.6, label='ORT', color='purple', edgecolor='black')
    ax.set_xlabel('Latency (ms)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Latency Distribution (Overlaid)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Box plot
    ax = axes[0, 1]
    data_to_plot = [df_tflite['latency_ms'], df_ort['latency_ms']]
    bp = ax.boxplot(data_to_plot, labels=['TFLite', 'ORT'], patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('purple')
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Latency Box Plot', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Time series (latency over runs)
    ax = axes[1, 0]
    ax.plot(df_tflite['run_id'], df_tflite['latency_ms'], alpha=0.7, label='TFLite', color='blue', linewidth=1)
    ax.plot(df_ort['run_id'], df_ort['latency_ms'], alpha=0.7, label='ORT', color='purple', linewidth=1)
    ax.set_xlabel('Run ID', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Latency Over Time (Stability)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Percentile comparison
    ax = axes[1, 1]
    percentiles = [50, 75, 90, 95, 99]
    tflite_p = [df_tflite['latency_ms'].quantile(p/100) for p in percentiles]
    ort_p = [df_ort['latency_ms'].quantile(p/100) for p in percentiles]
    
    x = np.arange(len(percentiles))
    width = 0.35
    ax.bar(x - width/2, tflite_p, width, label='TFLite', color='blue', alpha=0.8)
    ax.bar(x + width/2, ort_p, width, label='ORT', color='purple', alpha=0.8)
    ax.set_xlabel('Percentile', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Percentile Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'p{p}' for p in percentiles])
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'latency_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📈 Saved plot: {output_path}")
    plt.show()


def plot_memory_usage(df_tflite, df_ort, output_dir):
    """Plot memory usage over time if available"""
    if 'memory_mb' not in df_tflite.columns or 'memory_mb' not in df_ort.columns:
        print("⚠️  Memory data not available, skipping memory plots")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Memory over time
    ax = axes[0]
    ax.plot(df_tflite['run_id'], df_tflite['memory_mb'], alpha=0.7, label='TFLite', color='blue', linewidth=1)
    ax.plot(df_ort['run_id'], df_ort['memory_mb'], alpha=0.7, label='ORT', color='purple', linewidth=1)
    ax.set_xlabel('Run ID', fontsize=12)
    ax.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax.set_title('Memory Usage Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Memory distribution
    ax = axes[1]
    data_to_plot = [df_tflite['memory_mb'], df_ort['memory_mb']]
    bp = ax.boxplot(data_to_plot, labels=['TFLite', 'ORT'], patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('purple')
    ax.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax.set_title('Memory Usage Distribution', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'memory_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📈 Saved plot: {output_path}")
    plt.show()


def export_summary_table(stats_tflite, stats_ort, output_dir):
    """Export summary table as CSV and Markdown"""
    # Create DataFrame
    df_summary = pd.DataFrame([stats_tflite, stats_ort])
    
    # Export CSV
    csv_path = Path(output_dir) / 'summary_statistics.csv'
    df_summary.to_csv(csv_path, index=False)
    print(f"📄 Saved summary CSV: {csv_path}")
    
    # Export Markdown table
    md_path = Path(output_dir) / 'summary_statistics.md'
    with open(md_path, 'w') as f:
        f.write("# Benchmark Results Summary\n\n")
        f.write(df_summary.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n")
    print(f"📄 Saved summary Markdown: {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze TFLite vs ORT benchmark results"
    )
    parser.add_argument(
        "--tflite-csv",
        type=str,
        required=True,
        help="Path to TFLite results CSV"
    )
    parser.add_argument(
        "--ort-csv",
        type=str,
        required=True,
        help="Path to ORT results CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./analysis_output",
        help="Output directory for plots and reports"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("🔬 TFLite vs ORT Benchmark Analysis")
    print("="*80)
    
    # Load data
    df_tflite, df_ort = load_results(args.tflite_csv, args.ort_csv)
    
    # Calculate statistics
    stats_tflite = calculate_statistics(df_tflite, 'TFLite')
    stats_ort = calculate_statistics(df_ort, 'ORT')
    
    # Print comparison
    print_comparison_table(stats_tflite, stats_ort)
    
    # Export summary
    export_summary_table(stats_tflite, stats_ort, output_dir)
    
    # Generate plots
    if not args.no_plots:
        plot_latency_distribution(df_tflite, df_ort, output_dir)
        plot_memory_usage(df_tflite, df_ort, output_dir)
    
    print("\n✅ Analysis complete!")
    print(f"📁 Results saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()

