import argparse
from scipy import stats
import seaborn as sns
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.ioff()  # Turn off interactive mode
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

def load_all_testruns(base_dir: Path) -> tuple[pd.DataFrame, Path]:
    all_frames = []

    # Create results folder
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    for testrun_dir in sorted(base_dir.glob("testrun_*")):
        if not testrun_dir.is_dir():
            continue

        testrun_name = testrun_dir.name

        build_data = load_csv(testrun_dir / "bvh_build_times.csv")
        build_data = build_data.rename(columns={"time_seconds":"construction_time"})
        render_data = load_csv(testrun_dir / "render_times.csv")
        render_data = render_data.rename(columns={"time_seconds":"traversal_time"})
        shader_data = load_csv(testrun_dir / "shading_times.csv")
        shader_data = shader_data.rename(columns={"time_seconds": "hitray_count"})

        merge_keys = ["model_name", "model_scale", "algorithm_name", "cam_pos_x", "cam_pos_y", "cam_pos_z", "cam_dir_x", "cam_dir_y", "cam_dir_z"]

        # Set Types
        numeric_cols = ["model_scale", "cam_pos_x", "cam_pos_y", "cam_pos_z", "cam_dir_x", "cam_dir_y", "cam_dir_z"]
        for col in numeric_cols:
            if col in build_data.columns:
                build_data[col] = build_data[col].astype('float64')
            if col in render_data.columns:
                render_data[col] = render_data[col].astype('float64')
            if col in shader_data.columns:
                shader_data[col] = shader_data[col].astype('float64')

        string_cols = ["model_name", "algorithm_name"]
        for col in string_cols:
            if col in build_data.columns:
                build_data[col] = build_data[col].astype('string')
            if col in render_data.columns:
                render_data[col] = render_data[col].astype('string')
            if col in shader_data.columns:
                shader_data[col] = shader_data[col].astype('string')

        combined = pd.merge(build_data, render_data, on=merge_keys, how="inner", suffixes=("_build", "_render"))
        combined = pd.merge(combined, shader_data, on=merge_keys, how="inner")

        combined["testrun_index"] = testrun_name.split("_")[-1]
        combined["testrun_index"] = combined["testrun_index"].astype(int)

        print(f"Testrun {testrun_name}: {len(combined)} measurements")
        all_frames.append(combined)
    if not all_frames:
        raise RuntimeError(f"Keine CSV-Dateien unter {base_dir} gefunden")

    return pd.concat(all_frames, ignore_index=True), results_dir


def load_csv(csv_file: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_file)

    # polish
    df["model_name"] = df["model_name"].apply(lambda x: Path(x).stem)
    df["algorithm_degree"] = df["algorithm_name"].apply(lambda x: x.split("-")[-1]).astype(int)
    df["algorithm_type"] = df["algorithm_name"].apply(lambda x: "collapsed" if x.__contains__("-c-") else "k-way")
    df["algorithm_prefix"] = df["algorithm_name"].apply(lambda x: x.split("-")[0])
    return df


def detailed_analysis(df, results_dir: Path):
    df = df.copy()
    # Check if there is a significant difference between k = 2 and k € {4, 8 ,16}

    print("OVERVIEW")
    models = df["model_name"].unique()
    print("Model names: ", models)
    testrun_indices = df["testrun_index"].unique()
    print("Testrun Indices: ", sorted(testrun_indices))
    algorithm_prefixes = df["algorithm_prefix"].unique()
    print("Algorithm prefixes: ", algorithm_prefixes)
    algorithm_types = df["algorithm_type"].unique()
    print("Algorithm types: ", algorithm_types)
    algorithm_degrees = df["algorithm_degree"].unique()
    print("Algorithm degrees: ", algorithm_degrees)

    # 1. Aggregate over all camera steps to get mean per testrun
    print("Aggregating camera steps...")
    testrun_means = df.groupby([
        "model_name", "algorithm_prefix", "algorithm_type", "algorithm_degree", "testrun_index"
    ]).agg({
        "traversal_time": "mean",
        "construction_time": "mean",
        "hitray_count": "mean"
    }).reset_index()
    print(f"After aggregation: {len(testrun_means)} testrun measurements")

    results_summary = []

    # 2. Statistical analysis over repetitions for each combination
    print("Starting statistical analysis...")
    for model in models:
        for algorithm_prefix in algorithm_prefixes:
            for algorithm_type in algorithm_types:
                for algorithm_degree in algorithm_degrees:
                    if algorithm_degree == 2:
                        continue

                    print(f'Analyzing { algorithm_prefix } , { algorithm_degree }, {algorithm_type}')
                    # Filter for this specific model and algorithm combination
                    baseline_data = testrun_means[
                        (testrun_means["model_name"] == model) &
                        (testrun_means["algorithm_prefix"] == algorithm_prefix) &
                        (testrun_means["algorithm_degree"] == 2)
                        ]

                    test_data = testrun_means[
                        (testrun_means["model_name"] == model) &
                        (testrun_means["algorithm_prefix"] == algorithm_prefix) &
                        (testrun_means["algorithm_degree"] == algorithm_degree) &
                        (testrun_means["algorithm_type"] == algorithm_type)
                        ]

                    # Skip if either dataset is incomplete
                    if len(baseline_data) < 2 or len(test_data) < 2:
                        print(f"Insufficient data: baseline={len(baseline_data)}, test={len(test_data)}")
                        continue

                    # calculate difference between traversal times
                    baseline_times = baseline_data['traversal_time']
                    test_times = test_data['traversal_time']

                    # Welch's t-test (assumes unequal variances)
                    t_stat, p_value = stats.ttest_ind(baseline_times, test_times, equal_var=False)

                    baseline_mean = baseline_times.mean()
                    test_mean = test_times.mean()
                    speedup_factor = baseline_mean / test_mean

                    alpha = 0.05
                    is_significant = p_value < alpha
                    percent_change = ((test_mean - baseline_mean) / baseline_mean) * 100

                    results_summary.append({
                        'model': model,
                        'algorithm_prefix': algorithm_prefix,
                        'degree': algorithm_degree,
                        'type': algorithm_type,
                        'baseline_mean': baseline_mean,
                        'test_mean': test_mean,
                        'speedup_factor': speedup_factor,
                        'percent_change': percent_change,
                        't_stat': t_stat,
                        'p_value': p_value,
                        'significant': is_significant,
                        'baseline_n': len(baseline_times),
                        'test_n': len(test_times)
                    })

                    # Create plots
                    detailed_plot(results_dir, baseline_times, test_times, model, algorithm_prefix, algorithm_degree, algorithm_type,
                                    baseline_mean, test_mean, speedup_factor, percent_change, t_stat, p_value, is_significant, len(baseline_times), len(test_times))
    print("Detailed analysis completed.")
    return pd.DataFrame(results_summary)

def detailed_plot(results_dir, baseline_times, test_times, model, algorithm_prefix, algorithm_degree, algorithm_type, baseline_mean, test_mean, speedup_factor, percent_change, t_stat, p_value, significant, baseline_n, test_n):

    plot_data = pd.DataFrame({
        'Algorithm': ['k=2'] * len(baseline_times) + [f'k={algorithm_degree}'] * len(test_times),
        'Traversal_Time': pd.concat([baseline_times.reset_index(drop=True), test_times.reset_index(drop=True)])
    })

    fig, axes = plt.subplots(1, 2, figsize=(12,8))
    # Histogram of differences (only for matched pairs)
    min_len = min(len(baseline_times), len(test_times))
    differences = baseline_times.iloc[:min_len].values - test_times.iloc[:min_len].values
    axes[0].hist(differences, bins='auto')
    axes[0].set_xlabel(f'Time Difference (k=2 - k={algorithm_degree}) [s]')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f"{model} | {algorithm_prefix} | {algorithm_degree} | {algorithm_type}")
    axes[0].grid(True)

    text = '\n'.join((
        r'$t=%.2f$' % (t_stat,),
        r'$p=%.2e$' % (p_value,),
        r'Reject H0' if significant else r'Fail to reject H0',
        r'%.2f%% speed' % percent_change
    ))
    if significant:
        print("Significant result!")
        if percent_change < 0:
            print("Algorithm is faster!")
            text +=" (FASTER)"
        else:
            print("Algorithm is slower!")
            text += " (SLOWER)"

    # Boxplot comparing k=2 vs k=X
    sns.boxplot(x='Algorithm', y='Traversal_Time', hue='Algorithm', data=plot_data, palette='pastel', legend=False, ax=axes[1])
    axes[1].set_title(f'Traversal Time Distribution: k=2 vs k={algorithm_degree}')
    axes[1].set_ylabel('Traversal Time [s]')

    axes[1].text(
        0.95, 0.95, text,
        transform=axes[1].transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # Figure layout
    fig.tight_layout()

    # Save plot to results folder instead of showing
    plot_filename = results_dir / f"detailed_{model}_{algorithm_prefix}_{algorithm_type}_{algorithm_degree}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_filename}")
    plt.close()  # Close figure to free memory

# This part is ai generated for easy reading of results
def create_detailed_results_table(results_df, results_dir: Path):
    # Create sorting key: degree first, then type, then algorithm, then model
    def create_sort_key(row):
        # Primary sort: degree (k)
        # Secondary sort: type (k-way=0, collapsed=1)
        # Tertiary sort: algorithm
        # Quaternary sort: model

        if row['type'] == 'k-way':
            type_order = 0  # k-way comes first
        else:  # collapsed
            type_order = 1  # collapsed comes second

        return (row['degree'], type_order, row['algorithm_prefix'], row['model'])

    results_df['sort_key'] = results_df.apply(create_sort_key, axis=1)
    results_df = results_df.sort_values('sort_key').drop('sort_key', axis=1)

    # Create a formatted version for display
    formatted_df = results_df.copy()

    # Round numerical values for better presentation
    formatted_df['baseline_mean'] = formatted_df['baseline_mean'].round(4)
    formatted_df['test_mean'] = formatted_df['test_mean'].round(4)
    formatted_df['speedup_factor'] = formatted_df['speedup_factor'].round(3)
    formatted_df['percent_change'] = formatted_df['percent_change'].round(1)
    formatted_df['t_stat'] = formatted_df['t_stat'].round(3)

    # Combine p-value with significance stars
    def format_p_value_with_stars(p_val):
        p_str = f"{p_val:.2e}"
        if p_val < 0.001:
            return f"{p_str}***"
        elif p_val < 0.01:
            return f"{p_str}**"
        elif p_val < 0.05:
            return f"{p_str}*"
        else:
            return p_str

    formatted_df['p_value_with_stars'] = formatted_df['p_value'].apply(format_p_value_with_stars)

    # Rename columns for publication
    column_mapping = {
        'model': 'Model',
        'algorithm_prefix': 'Algorithm',
        'degree': 'k',
        'type': 'Type',
        'baseline_mean': 'k=2 Time (s)',
        'test_mean': f'k=X Time (s)',
        'speedup_factor': 'Speedup',
        'percent_change': '% Change',
        't_stat': 't-statistic',
        'p_value_with_stars': 'p-value',
        'baseline_n': 'n (k=2)',
        'test_n': 'n (k=X)'
    }

    publication_df = formatted_df.rename(columns=column_mapping)

    # Select columns for publication table (no separate significance column)
    pub_columns = ['Model', 'Algorithm', 'k', 'Type', 'k=2 Time (s)', 'k=X Time (s)',
                   'Speedup', '% Change', 't-statistic', 'p-value']
    publication_df = publication_df[pub_columns]

    # Save as CSV
    csv_path = results_dir / "statistical_results_table.csv"
    publication_df.to_csv(csv_path, index=False)
    print(f"Results table saved to: {csv_path}")

    # Create a fancy visualization of the table
    fig, ax = plt.subplots(figsize=(16, max(8, int(len(publication_df) * 0.4))))
    ax.axis('tight')
    ax.axis('off')

    # Create table with better styling
    table = ax.table(cellText=publication_df.values,
                    colLabels=publication_df.columns,
                    cellLoc='center',
                    loc='center')

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Color the % Change column based on performance
    pct_change_col_idx = publication_df.columns.get_loc('% Change')
    for table_row_idx, (_, row) in enumerate(publication_df.iterrows()):
        pct_change = float(row['% Change'])
        if pct_change < 0:  # Faster (improvement)
            color = '#90EE90'  # Light green
        else:  # Slower (degradation)
            color = '#FFB6C1'  # Light red

        # Color the % Change column (table_row_idx + 1 because row 0 is header)
        table[(table_row_idx + 1, pct_change_col_idx)].set_facecolor(color)

    # Style header
    for j in range(len(publication_df.columns)):
        table[(0, j)].set_facecolor('#cccccc')
        table[(0, j)].set_text_props(weight='bold')

    plt.title('BVH Performance Comparison\n' +
              'Green = Faster, Red = Slower | * p < 0.05, ** p < 0.01, *** p < 0.001',
              fontsize=14, fontweight='bold', pad=5)

    # Save the table image
    table_path = results_dir / "statistical_results_table.png"
    plt.savefig(table_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Table image saved to: {table_path}")

    # Print summary statistics
    significant_results = publication_df[publication_df['p-value'].str.contains(r'\*', regex=True, na=False)]
    print(f"\nSummary:")
    print(f"Total comparisons: {len(publication_df)}")
    print(f"Significant results: {len(significant_results)}")

    if len(significant_results) > 0:
        print("\nSignificant improvements (faster algorithms):")
        improvements = significant_results[significant_results['% Change'] < 0]
        for _, row in improvements.iterrows():
            print(f"  {row['Model']} | {row['Algorithm']} k={row['k']} ({row['Type']}): "
                  f"{row['% Change']:.1f}% faster (p={row['p-value']})")

        print("\nSignificant degradations (slower algorithms):")
        degradations = significant_results[significant_results['% Change'] >= 0]
        for _, row in degradations.iterrows():
            print(f"  {row['Model']} | {row['Algorithm']} k={row['k']} ({row['Type']}): "
                  f"{row['% Change']:.1f}% slower (p={row['p-value']})")


def create_summarized_results_table(results_df, results_dir):
    if results_df.empty:
        print(" No comparison results to make a table from!")
        return

    # Sort by degree first, then algorithm, then model (like detailed results)
    results_df = results_df.sort_values(['degree', 'algorithm_prefix', 'model'])

    # Create a formatted version for display
    formatted_df = results_df.copy()

    # Round numerical values for better presentation
    formatted_df['kway_mean'] = formatted_df['kway_mean'].round(4)
    formatted_df['collapsed_mean'] = formatted_df['collapsed_mean'].round(4)
    formatted_df['kway_std'] = formatted_df['kway_std'].round(4)
    formatted_df['collapsed_std'] = formatted_df['collapsed_std'].round(4)
    formatted_df['speedup_factor'] = formatted_df['speedup_factor'].round(3)
    formatted_df['percent_change'] = formatted_df['percent_change'].round(1)
    formatted_df['t_stat'] = formatted_df['t_stat'].round(3)

    # Combine p-value with significance stars (like detailed results)
    def format_p_value_with_stars(p_val):
        p_str = f"{p_val:.2e}"
        if p_val < 0.001:
            return f"{p_str}***"
        elif p_val < 0.01:
            return f"{p_str}**"
        elif p_val < 0.05:
            return f"{p_str}*"
        else:
            return p_str

    formatted_df['p_value_with_stars'] = formatted_df['p_value'].apply(format_p_value_with_stars)

    # Rename columns for publication
    column_mapping = {
        'model': 'Model',
        'algorithm_prefix': 'Algorithm',
        'degree': 'k',
        'kway_mean': 'k-way Time (s)',
        'collapsed_mean': 'Collapsed Time (s)',
        'speedup_factor': 'Speedup',
        'percent_change': '% Change',
        't_stat': 't-statistic',
        'p_value_with_stars': 'p-value',
    }

    publication_df = formatted_df.rename(columns=column_mapping)

    # Select columns for publication table
    pub_columns = ['Model', 'Algorithm', 'k', 'k-way Time (s)', 'Collapsed Time (s)',
                   'Speedup', '% Change', 't-statistic', 'p-value',]
    publication_df = publication_df[pub_columns]

    # Save as CSV
    csv_path = results_dir / "comparison_results_table.csv"
    publication_df.to_csv(csv_path, index=False)
    print(f"Comparison results table saved to: {csv_path}")

    # Create a fancy visualization of the table (like detailed results)
    fig, ax = plt.subplots(figsize=(18, max(8, int(len(publication_df) * 0.4))))
    ax.axis('tight')
    ax.axis('off')

    # Create table with better styling
    table = ax.table(cellText=publication_df.values,
                    colLabels=publication_df.columns,
                    cellLoc='center',
                    loc='center')

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Color the % Change column based on performance
    pct_change_col_idx = publication_df.columns.get_loc('% Change')
    for table_row_idx, (_, row) in enumerate(publication_df.iterrows()):
        pct_change = float(row['% Change'])
        if pct_change < 0:  # Collapsed faster than k-way
            color = '#90EE90'  # Light green
        else:  # Collapsed slower than k-way
            color = '#FFB6C1'  # Light red

        # Color the % Change column (table_row_idx + 1 because row 0 is header)
        table[(table_row_idx + 1, pct_change_col_idx)].set_facecolor(color)

    # Style header
    for j in range(len(publication_df.columns)):
        table[(0, j)].set_facecolor('#cccccc')
        table[(0, j)].set_text_props(weight='bold')

    plt.title('Collapsed vs k-way Algorithm Comparison\n' +
              'Green = Collapsed Faster, Red = Collapsed Slower | * p < 0.05, ** p < 0.01, *** p < 0.001',
              fontsize=14, fontweight='bold', pad=20)

    # Save the table image
    table_path = results_dir / "comparison_results_table.png"
    plt.savefig(table_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Comparison table image saved to: {table_path}")

    # Print summary statistics (like detailed results)
    significant_results = publication_df[publication_df['p-value'].str.contains(r'\*', regex=True, na=False)]
    print(f"\nComparison Summary:")
    print(f"Total comparisons: {len(publication_df)}")
    print(f"Significant differences: {len(significant_results)}")

    if len(significant_results) > 0:
        print("\nCases where collapsed is significantly faster:")
        faster_collapsed = significant_results[significant_results['% Change'] < 0]
        for _, row in faster_collapsed.iterrows():
            print(f"  {row['Model']} | {row['Algorithm']} k={row['k']}: "
                  f"{row['% Change']:.1f}% faster (p={row['p-value']})")

        print("\nCases where collapsed is significantly slower:")
        slower_collapsed = significant_results[significant_results['% Change'] >= 0]
        for _, row in slower_collapsed.iterrows():
            print(f"  {row['Model']} | {row['Algorithm']} k={row['k']}: "
                  f"{row['% Change']:.1f}% slower (p={row['p-value']})")

def collapsed_vs_kway_analysis(df, results_dir: Path):
    # 1. Aggregate over all camera steps to get mean per testrun (same as detailed analysis)
    print("Aggregating camera steps for comparison analysis...")
    testrun_means = df.groupby([
        "model_name", "algorithm_prefix", "algorithm_type", "algorithm_degree", "testrun_index"
    ]).agg({
        "traversal_time": "mean",
        "construction_time": "mean",
        "hitray_count": "mean"
    }).reset_index()

    results_summary = []

    # 2. Group by model, algorithm prefix, and degree
    grouped = testrun_means.groupby(['model_name', 'algorithm_prefix', 'algorithm_degree'])

    for (model, algorithm_prefix, degree), group_df in grouped:
        if degree == 2:  # Skip k=2 as it doesn't have collapsed variant
            continue

        print(f"\nAnalyzing {model} | {algorithm_prefix} | k={degree}")

        # Get k-way and collapsed data
        kway_data = group_df[group_df['algorithm_type'] == 'k-way']['traversal_time']
        collapsed_data = group_df[group_df['algorithm_type'] == 'collapsed']['traversal_time']

        if len(kway_data) == 0 or len(collapsed_data) == 0:
            print(f"  Skipping - missing data (k-way: {len(kway_data)}, collapsed: {len(collapsed_data)})")
            continue

        if len(kway_data) < 2 or len(collapsed_data) < 2:
            print(f"  Skipping - insufficient data (k-way: {len(kway_data)}, collapsed: {len(collapsed_data)})")
            continue

        # Perform t-test (Welch's t-test for unequal variances)
        t_stat, p_value = stats.ttest_ind(kway_data, collapsed_data, equal_var=False)
        # Calculate descriptive statistics
        kway_mean = kway_data.mean()
        collapsed_mean = collapsed_data.mean()
        kway_std = kway_data.std()
        collapsed_std = collapsed_data.std()
        # Calculate effect metrics
        speedup_factor = kway_mean / collapsed_mean if collapsed_mean > 0 else float('inf')
        percent_change = ((collapsed_mean - kway_mean) / kway_mean) * 100
        # Significance test
        alpha = 0.05
        is_significant = p_value < alpha

        results_summary.append({
            'model': model,
            'algorithm_prefix': algorithm_prefix,
            'degree': degree,
            'kway_mean': kway_mean,
            'kway_std': kway_std,
            'kway_n': len(kway_data),
            'collapsed_mean': collapsed_mean,
            'collapsed_std': collapsed_std,
            'collapsed_n': len(collapsed_data),
            'speedup_factor': speedup_factor,
            'percent_change': percent_change,
            't_stat': t_stat,
            'p_value': p_value,
            'significant': is_significant,
        })

        print(f"  k-way: {kway_mean:.4f}s ± {kway_std:.4f} (n={len(kway_data)})")
        print(f"  collapsed: {collapsed_mean:.4f}s ± {collapsed_std:.4f} (n={len(collapsed_data)})")
        print(f"  % Change (collapsed vs k-way): {percent_change:+.1f}%")
        print(f"  t-statistic: {t_stat:.3f}, p-value: {p_value:.2e}")

        # Create comparison plot
        try:
            comparison_plot(results_dir, kway_data, collapsed_data, model, algorithm_prefix, degree,
                          kway_mean, collapsed_mean, speedup_factor, percent_change, t_stat, p_value,
                          is_significant, len(kway_data), len(collapsed_data))
        except Exception as e:
            print(f"Error creating comparison plot: {e}")

    print("Collapsed vs k-way analysis completed.")
    return pd.DataFrame(results_summary)

def comparison_plot(results_dir, kway_data, collapsed_data, model, algorithm_prefix, degree,
                   kway_mean, collapsed_mean, speedup_factor, percent_change, t_stat, p_value,
                   significant, kway_n, collapsed_n):

    plot_data = pd.DataFrame({
        'Algorithm': ['k-way'] * len(kway_data) + ['collapsed'] * len(collapsed_data),
        'Traversal_Time': pd.concat([kway_data.reset_index(drop=True), collapsed_data.reset_index(drop=True)])
    })

    fig, axes = plt.subplots(1, 2, figsize=(12,8))

    # Histogram of differences (only for matched pairs)
    min_len = min(len(kway_data), len(collapsed_data))
    differences = kway_data.iloc[:min_len].values - collapsed_data.iloc[:min_len].values
    axes[0].hist(differences, bins='auto')
    axes[0].set_xlabel(f'Time Difference (k-way - collapsed) [s]')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f"{model} | {algorithm_prefix} | k={degree}")
    axes[0].grid(True)

    text = '\n'.join((
        r'$t=%.2f$' % (t_stat,),
        r'$p=%.2e$' % (p_value,),
        r'Reject H0' if significant else r'Fail to reject H0',
        r'%.2f%% change' % percent_change
    ))

    if significant:
        print("Significant result!")
        if percent_change < 0:
            print("Collapsed is faster!")
            text += " (COLLAPSED FASTER)"
        else:
            print("Collapsed is slower!")
            text += " (COLLAPSED SLOWER)"

    # Boxplot comparing k-way vs collapsed
    sns.boxplot(x='Algorithm', y='Traversal_Time', hue='Algorithm', data=plot_data, palette='pastel', legend=False, ax=axes[1])
    axes[1].set_title(f'Traversal Time Distribution: k-way vs collapsed')
    axes[1].set_ylabel('Traversal Time [s]')

    axes[1].text(
        0.95, 0.95, text,
        transform=axes[1].transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # Figure layout
    fig.tight_layout()

    # Save plot to results folder
    plot_filename = results_dir / f"comparison_{model}_{algorithm_prefix}_k{degree}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_filename}")
    plt.close()  # Close figure to free memory


def main():
    parser = argparse.ArgumentParser(description="Loads testrun CSV-Data")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("./testruns"),
        help="Default: ./testruns"
    )

    args = parser.parse_args()

    base_dir = args.dir.resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Folder not found: {base_dir}")

    df, results_dir = load_all_testruns(base_dir)

    print("Successfully loaded data:")
    print(f"Dataset rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Results will be saved to: {results_dir}")

    print(f"Sample of data:")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)

    detailed_results_df = detailed_analysis(df, results_dir)
    create_detailed_results_table(detailed_results_df, results_dir)

    summarized_results = collapsed_vs_kway_analysis(df, results_dir)
    create_summarized_results_table(summarized_results, results_dir)

if __name__ == "__main__":
    main()
