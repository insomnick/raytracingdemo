import argparse
from scipy import stats
import seaborn as sns
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import jinja2

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

    # Color the % Change column based on performance (only if significant)
    pct_change_col_idx = publication_df.columns.get_loc('% Change')
    for table_row_idx, (_, row) in enumerate(publication_df.iterrows()):
        pct_change = float(row['% Change'])
        p_value_str = row['p-value']

        # Check if result is significant (has asterisks in p-value)
        is_significant = '*' in p_value_str

        if is_significant:
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
              'Green/Red = Significant Results Only | * p < 0.05, ** p < 0.01, *** p < 0.001',
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


def k_way_vs_collapsed_results_table(results_df, results_dir):
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

    # Color the % Change column based on performance (only if significant)
    pct_change_col_idx = publication_df.columns.get_loc('% Change')
    for table_row_idx, (_, row) in enumerate(publication_df.iterrows()):
        pct_change = float(row['% Change'])
        p_value_str = row['p-value']

        # Check if result is significant (has asterisks in p-value)
        is_significant = '*' in p_value_str

        if is_significant:
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
              'Green/Red = Significant Results Only | * p < 0.05, ** p < 0.01, *** p < 0.001',
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

def create_comprehensive_summary_graphs(detailed_results_df, comparison_results, dynamic_results_df, dynamic_comparison_results, results_dir: Path):
    """Create comprehensive summary graphs that combine all analysis results"""
    import numpy as np

    print("\nCreating comprehensive summary graphs...")

    # Prepare data for static analysis (detailed + comparison)
    static_data = []

    # Add detailed results (k=2 vs k=X)
    if not detailed_results_df.empty:
        for _, row in detailed_results_df.iterrows():
            static_data.append({
                'model': row['model'],
                'algorithm_prefix': row['algorithm_prefix'],
                'degree': row['degree'],
                'type': row['type'],
                'percent_change': row['percent_change'],
                'analysis_type': 'k=2_vs_kX'
            })

    # Add comparison results (k-way vs collapsed) - but mark as comparison
    if comparison_results:
        for result in comparison_results:
            static_data.append({
                'model': result['model'],
                'algorithm_prefix': result['algorithm_prefix'],
                'degree': result['degree'],
                'type': 'comparison',  # Special marker for k-way vs collapsed
                'percent_change': result['percent_change'],
                'analysis_type': 'kway_vs_collapsed'
            })

    # Prepare data for dynamic analysis (dynamic + dynamic comparison)
    dynamic_data = []

    # Add dynamic results (k=2 vs k=X combined time)
    if not dynamic_results_df.empty:
        for _, row in dynamic_results_df.iterrows():
            dynamic_data.append({
                'model': row['model'],
                'algorithm_prefix': row['algorithm_prefix'],
                'degree': row['degree'],
                'type': row['type'],
                'percent_change': row['percent_change'],
                'analysis_type': 'k=2_vs_kX_combined'
            })

    # Add dynamic comparison results (k-way vs collapsed combined time)
    if dynamic_comparison_results:
        for result in dynamic_comparison_results:
            dynamic_data.append({
                'model': result['model'],
                'algorithm_prefix': result['algorithm_prefix'],
                'degree': result['degree'],
                'type': 'comparison',  # Special marker for k-way vs collapsed
                'percent_change': result['percent_change'],
                'analysis_type': 'kway_vs_collapsed_combined'
            })

    # Convert to DataFrames
    static_df = pd.DataFrame(static_data)
    dynamic_df = pd.DataFrame(dynamic_data)

    # Create the graphs - separate for each k-value
    create_summary_graph_type1_split(static_df, dynamic_df, results_dir)
    create_summary_graph_type2_split(static_df, dynamic_df, results_dir)

def create_summary_graph_type1_split(static_df, dynamic_df, results_dir):
    """Graph Type 1 Split: Separate boxplot for each k-value, with mean and std annotations"""

    # Type 1a: Static Analysis - separate plot for each k-value
    if not static_df.empty:
        static_filtered = static_df[static_df['analysis_type'] == 'k=2_vs_kX'].copy()

        if not static_filtered.empty:
            k_values = sorted(static_filtered['degree'].unique())

            for k in k_values:
                k_data = static_filtered[static_filtered['degree'] == k]

                if len(k_data) == 0:
                    continue

                fig, ax = plt.subplots(figsize=(10, 8))

                # Create boxplot for this k-value
                boxplot_data = [k_data['percent_change'].values]

                bp = ax.boxplot(boxplot_data, tick_labels=[f'k={k}'], patch_artist=True)

                # Color box
                bp['boxes'][0].set_facecolor('#87CEEB')  # Sky blue
                bp['boxes'][0].set_alpha(0.7)

                # Add mean and std annotation inside the box
                data = k_data['percent_change']
                mean_val = data.mean()
                std_val = data.std()
                median_val = data.median()

                # Position text inside the box (at median position)
                ax.text(1, median_val, f'μ={mean_val:.1f}%\nσ={std_val:.1f}%\nn={len(data)}',
                       ha='center', va='center', fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor='black'))

                ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
                ax.set_xlabel('Algorithm Degree')
                ax.set_ylabel('% Change (k=X vs k=2)')
                ax.set_title(f'Static Analysis: Performance Change for k={k}\n(Negative = k={k} Faster than k=2)')
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(results_dir / f"q_static_type1_k{k}_percent_change.png", dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  Created: q_static_type1_k{k}_percent_change.png")

    # Type 1b: Dynamic Analysis - separate plot for each k-value
    if not dynamic_df.empty:
        dynamic_filtered = dynamic_df[dynamic_df['analysis_type'] == 'k=2_vs_kX_combined'].copy()

        if not dynamic_filtered.empty:
            k_values = sorted(dynamic_filtered['degree'].unique())

            for k in k_values:
                k_data = dynamic_filtered[dynamic_filtered['degree'] == k]

                if len(k_data) == 0:
                    continue

                fig, ax = plt.subplots(figsize=(10, 8))

                # Create boxplot for this k-value
                boxplot_data = [k_data['percent_change'].values]

                bp = ax.boxplot(boxplot_data, tick_labels=[f'k={k}'], patch_artist=True)

                # Color box
                bp['boxes'][0].set_facecolor('#98FB98')  # Pale green
                bp['boxes'][0].set_alpha(0.7)

                # Add mean and std annotation inside the box
                data = k_data['percent_change']
                mean_val = data.mean()
                std_val = data.std()
                median_val = data.median()

                # Position text inside the box (at median position)
                ax.text(1, median_val, f'μ={mean_val:.1f}%\nσ={std_val:.1f}%\nn={len(data)}',
                       ha='center', va='center', fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor='black'))

                ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
                ax.set_xlabel('Algorithm Degree')
                ax.set_ylabel('% Change (k=X vs k=2 Combined Time)')
                ax.set_title(f'Dynamic Analysis: Performance Change for k={k}\n(Negative = k={k} Faster than k=2, Combined Construction + Traversal)')
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(results_dir / f"q_dynamic_type1_k{k}_percent_change.png", dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  Created: q_dynamic_type1_k{k}_percent_change.png")

def create_summary_graph_type2_split(static_df, dynamic_df, results_dir):
    """Graph Type 2 Split: k-way and collapsed separated for each k-value"""

    # Type 2a: Static Analysis - separate plot for each k-value
    if not static_df.empty:
        static_filtered = static_df[static_df['analysis_type'] == 'k=2_vs_kX'].copy()

        if not static_filtered.empty:
            k_values = sorted(static_filtered['degree'].unique())
            types = ['k-way', 'collapsed']

            for k in k_values:
                k_data = static_filtered[static_filtered['degree'] == k]

                if len(k_data) == 0:
                    continue

                fig, ax = plt.subplots(figsize=(10, 8))

                positions = []
                labels = []
                boxplot_data = []
                colors = []

                for j, alg_type in enumerate(types):
                    data = k_data[k_data['type'] == alg_type]['percent_change']
                    if len(data) > 0:
                        pos = j + 1  # Position 1 and 2
                        positions.append(pos)
                        labels.append(alg_type)
                        boxplot_data.append(data.values)
                        colors.append('lightblue' if alg_type == 'k-way' else 'lightcoral')

                if boxplot_data:
                    bp = ax.boxplot(boxplot_data, positions=positions, patch_artist=True, widths=0.6)

                    # Color boxes
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)

                    # Add mean and std annotations inside the boxes
                    for pos, data, label in zip(positions, boxplot_data, labels):
                        mean_val = np.mean(data)
                        std_val = np.std(data)
                        median_val = np.median(data)

                        # Position text inside the box (at median position)
                        ax.text(pos, median_val, f'μ={mean_val:.1f}%\nσ={std_val:.1f}%\nn={len(data)}',
                               ha='center', va='center', fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, edgecolor='black'))

                    ax.set_xticks(positions)
                    ax.set_xticklabels(labels)
                    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
                    ax.set_ylabel('% Change (k=X vs k=2)')
                    ax.set_title(f'Static Analysis: k-way vs collapsed for k={k}\n(Blue = k-way, Red = collapsed)')
                    ax.grid(True, alpha=0.3)

                    plt.tight_layout()
                    plt.savefig(results_dir / f"q_static_type2_k{k}_kway_vs_collapsed.png", dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"  Created: q_static_type2_k{k}_kway_vs_collapsed.png")

    # Type 2b: Dynamic Analysis - separate plot for each k-value
    if not dynamic_df.empty:
        dynamic_filtered = dynamic_df[dynamic_df['analysis_type'] == 'k=2_vs_kX_combined'].copy()

        if not dynamic_filtered.empty:
            k_values = sorted(dynamic_filtered['degree'].unique())
            types = ['k-way', 'collapsed']

            for k in k_values:
                k_data = dynamic_filtered[dynamic_filtered['degree'] == k]

                if len(k_data) == 0:
                    continue

                fig, ax = plt.subplots(figsize=(10, 8))

                positions = []
                labels = []
                boxplot_data = []
                colors = []

                for j, alg_type in enumerate(types):
                    data = k_data[k_data['type'] == alg_type]['percent_change']
                    if len(data) > 0:
                        pos = j + 1  # Position 1 and 2
                        positions.append(pos)
                        labels.append(alg_type)
                        boxplot_data.append(data.values)
                        colors.append('lightblue' if alg_type == 'k-way' else 'lightcoral')

                if boxplot_data:
                    bp = ax.boxplot(boxplot_data, positions=positions, patch_artist=True, widths=0.6)

                    # Color boxes
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)

                    # Add mean and std annotations inside the boxes
                    for pos, data, label in zip(positions, boxplot_data, labels):
                        mean_val = np.mean(data)
                        std_val = np.std(data)
                        median_val = np.median(data)

                        # Position text inside the box (at median position)
                        ax.text(pos, median_val, f'μ={mean_val:.1f}%\nσ={std_val:.1f}%\nn={len(data)}',
                               ha='center', va='center', fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, edgecolor='black'))

                    ax.set_xticks(positions)
                    ax.set_xticklabels(labels)
                    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
                    ax.set_ylabel('% Change (k=X vs k=2 Combined Time)')
                    ax.set_title(f'Dynamic Analysis: k-way vs collapsed for k={k}\n(Blue = k-way, Red = collapsed, Combined Construction + Traversal)')
                    ax.grid(True, alpha=0.3)

                    plt.tight_layout()
                    plt.savefig(results_dir / f"q_dynamic_type2_k{k}_kway_vs_collapsed.png", dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"  Created: q_dynamic_type2_k{k}_kway_vs_collapsed.png")

def collapsed_vs_kway_analysis(df, results_dir: Path):

    # 1. Aggregate over all camera steps to get mean per testrun (same as detailed analysis)
    print("Aggregating camera steps for dynamic model analysis...")
    testrun_means = df.groupby([
        "model_name", "algorithm_prefix", "algorithm_type", "algorithm_degree", "testrun_index"
    ]).agg({
        "traversal_time": "mean",
        "construction_time": "mean",
        "hitray_count": "mean"
    }).reset_index()

    # Calculate combined time
    testrun_means['combined_time'] = testrun_means['traversal_time'] + testrun_means['construction_time']

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
            'collapsed_std': collapsed_data.std(),
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
    return results_summary

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


def dynamic_model_results(df, results_dir: Path):

    # 1. Aggregate over all camera steps to get mean per testrun (same as detailed analysis)
    print("Aggregating camera steps for dynamic model analysis...")
    testrun_means = df.groupby([
        "model_name", "algorithm_prefix", "algorithm_type", "algorithm_degree", "testrun_index"
    ]).agg({
        "traversal_time": "mean",
        "construction_time": "mean",
        "hitray_count": "mean"
    }).reset_index()

    # Calculate combined time
    testrun_means['combined_time'] = testrun_means['traversal_time'] + testrun_means['construction_time']

    results_summary = []

    # 2. Group by model, algorithm prefix, and degree
    grouped = testrun_means.groupby(['model_name', 'algorithm_prefix', 'algorithm_degree'])

    for (model, algorithm_prefix, degree), group_df in grouped:
        if degree == 2:  # Skip k=2 as it doesn't have collapsed variant
            continue

        print(f"\nAnalyzing {model} | {algorithm_prefix} | k={degree}")

        # Get k=2 baseline data
        baseline_data = testrun_means[
            (testrun_means["model_name"] == model) &
            (testrun_means["algorithm_prefix"] == algorithm_prefix) &
            (testrun_means["algorithm_degree"] == 2)
        ]['combined_time']

        # Get k=X data for both k-way and collapsed
        for algorithm_type in ['k-way', 'collapsed']:
            test_data = group_df[group_df['algorithm_type'] == algorithm_type]['combined_time']

            if len(baseline_data) == 0 or len(test_data) == 0:
                print(f"  Skipping {algorithm_type} - missing data (baseline: {len(baseline_data)}, test: {len(test_data)})")
                continue

            if len(baseline_data) < 2 or len(test_data) < 2:
                print(f"  Skipping {algorithm_type} - insufficient data (baseline: {len(baseline_data)}, test: {len(test_data)})")
                continue

            # Perform t-test (Welch's t-test for unequal variances)
            t_stat, p_value = stats.ttest_ind(baseline_data, test_data, equal_var=False)

            # Calculate descriptive statistics
            baseline_mean = baseline_data.mean()
            test_mean = test_data.mean()

            # Calculate effect metrics
            speedup_factor = baseline_mean / test_mean if test_mean > 0 else float('inf')
            percent_change = ((test_mean - baseline_mean) / baseline_mean) * 100

            # Significance test
            alpha = 0.05
            is_significant = p_value < alpha

            results_summary.append({
                'model': model,
                'algorithm_prefix': algorithm_prefix,
                'degree': degree,
                'type': algorithm_type,
                'baseline_mean': baseline_mean,
                'test_mean': test_mean,
                'speedup_factor': speedup_factor,
                'percent_change': percent_change,
                't_stat': t_stat,
                'p_value': p_value,
                'significant': is_significant,
                'baseline_n': len(baseline_data),
                'test_n': len(test_data)
            })

            print(f"  {algorithm_type}: baseline={baseline_mean:.4f}s, test={test_mean:.4f}s")
            print(f"  % Change: {percent_change:+.1f}%")
            print(f"  t-statistic: {t_stat:.3f}, p-value: {p_value:.2e}")

            # Create comparison plot
            try:
                dynamic_plot(results_dir, baseline_data, test_data, model, algorithm_prefix, degree, algorithm_type,
                           baseline_mean, test_mean, speedup_factor, percent_change, t_stat, p_value,
                           is_significant, len(baseline_data), len(test_data))
            except Exception as e:
                print(f"Error creating dynamic plot: {e}")

    print("Dynamic model analysis completed.")
    return pd.DataFrame(results_summary)

def dynamic_plot(results_dir, baseline_data, test_data, model, algorithm_prefix, degree, algorithm_type,
                baseline_mean, test_mean, speedup_factor, percent_change, t_stat, p_value,
                significant, baseline_n, test_n):
    """Create a detailed comparison plot for combined construction + traversal time"""

    plot_data = pd.DataFrame({
        'Algorithm': ['k=2'] * len(baseline_data) + [f'k={degree}'] * len(test_data),
        'Combined_Time': pd.concat([baseline_data.reset_index(drop=True), test_data.reset_index(drop=True)])
    })

    fig, axes = plt.subplots(1, 2, figsize=(12,8))

    # Histogram of differences (only for matched pairs)
    min_len = min(len(baseline_data), len(test_data))
    differences = baseline_data.iloc[:min_len].values - test_data.iloc[:min_len].values
    axes[0].hist(differences, bins='auto')
    axes[0].set_xlabel(f'Combined Time Difference (k=2 - k={degree}) [s]')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f"{model} | {algorithm_prefix} | {degree} | {algorithm_type}")
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
    sns.boxplot(x='Algorithm', y='Combined_Time', hue='Algorithm', data=plot_data, palette='pastel', legend=False, ax=axes[1])
    axes[1].set_title(f'Combined Time Distribution: k=2 vs k={degree}')
    axes[1].set_ylabel('Combined Time (Construction + Traversal) [s]')

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
    plot_filename = results_dir / f"dynamic_{model}_{algorithm_prefix}_{algorithm_type}_{degree}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Dynamic plot saved to: {plot_filename}")
    plt.close()  # Close figure to free memory

def dynamic_model_results_table(results_df, results_dir):
    """Create a formatted table for dynamic model results (combined construction + traversal time)"""

    if results_df.empty:
        print("No dynamic model results to make a table from!")
        return

    # Create sorting key: degree first, then type, then algorithm, then model (same as detailed results)
    def create_sort_key(row):
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
        'baseline_mean': 'k=2 Combined Time (s)',
        'test_mean': 'k=X Combined Time (s)',
        'speedup_factor': 'Speedup',
        'percent_change': '% Change',
        't_stat': 't-statistic',
        'p_value_with_stars': 'p-value',
        'baseline_n': 'n (k=2)',
        'test_n': 'n (k=X)'
    }

    publication_df = formatted_df.rename(columns=column_mapping)

    # Select columns for publication table
    pub_columns = ['Model', 'Algorithm', 'k', 'Type', 'k=2 Combined Time (s)', 'k=X Combined Time (s)',
                   'Speedup', '% Change', 't-statistic', 'p-value']
    publication_df = publication_df[pub_columns]

    # Save as CSV
    csv_path = results_dir / "dynamic_model_results_table.csv"
    publication_df.to_csv(csv_path, index=False)
    print(f"Dynamic model results table saved to: {csv_path}")


    # Create a fancy visualization of the table
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

    # Color the % Change column based on performance (only if significant)
    pct_change_col_idx = publication_df.columns.get_loc('% Change')
    for table_row_idx, (_, row) in enumerate(publication_df.iterrows()):
        pct_change = float(row['% Change'])
        p_value_str = row['p-value']

        # Check if result is significant (has asterisks in p-value)
        is_significant = '*' in p_value_str

        if is_significant:
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

    plt.title('Dynamic Model Analysis: Combined Construction + Traversal Time\n' +
              'Green/Red = Significant Results Only | * p < 0.05, ** p < 0.01, *** p < 0.001',
              fontsize=14, fontweight='bold', pad=20)

    # Save the table image
    table_path = results_dir / "dynamic_model_results_table.png"
    plt.savefig(table_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Dynamic model table image saved to: {table_path}")

    # Print summary statistics
    significant_results = publication_df[publication_df['p-value'].str.contains(r'\*', regex=True, na=False)]
    print(f"\nDynamic Model Summary:")
    print(f"Total comparisons: {len(publication_df)}")
    print(f"Significant results: {len(significant_results)}")

    if len(significant_results) > 0:
        print("\nSignificant improvements (faster combined time):")
        improvements = significant_results[significant_results['% Change'] < 0]
        for _, row in improvements.iterrows():
            print(f"  {row['Model']} | {row['Algorithm']} k={row['k']} ({row['Type']}): "
                  f"{row['% Change']:.1f}% faster (p={row['p-value']})")

        print("\nSignificant degradations (slower combined time):")
        degradations = significant_results[significant_results['% Change'] >= 0]
        for _, row in degradations.iterrows():
            print(f"  {row['Model']} | {row['Algorithm']} k={row['k']} ({row['Type']}): "
                  f"{row['% Change']:.1f}% slower (p={row['p-value']})")

def dynamic_comparison_analysis(df, results_dir: Path):

    # 1. Aggregate over all camera steps to get mean per testrun (same as other analyses)
    print("Aggregating camera steps for dynamic comparison analysis...")
    testrun_means = df.groupby([
        "model_name", "algorithm_prefix", "algorithm_type", "algorithm_degree", "testrun_index"
    ]).agg({
        "traversal_time": "mean",
        "construction_time": "mean",
        "hitray_count": "mean"
    }).reset_index()

    # Calculate combined time
    testrun_means['combined_time'] = testrun_means['traversal_time'] + testrun_means['construction_time']

    results_summary = []

    # 2. Group by model, algorithm prefix, and degree
    grouped = testrun_means.groupby(['model_name', 'algorithm_prefix', 'algorithm_degree'])

    for (model, algorithm_prefix, degree), group_df in grouped:
        if degree == 2:  # Skip k=2 as it doesn't have collapsed variant
            continue

        print(f"\nAnalyzing dynamic comparison {model} | {algorithm_prefix} | k={degree}")

        # Get k-way and collapsed data
        kway_data = group_df[group_df['algorithm_type'] == 'k-way']['combined_time']
        collapsed_data = group_df[group_df['algorithm_type'] == 'collapsed']['combined_time']

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
            'collapsed_std': collapsed_data.std(),
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
            dynamic_comparison_plot(group_df, model, algorithm_prefix, degree, results_dir)
        except Exception as e:
            print(f"Error creating dynamic comparison plot: {e}")

    print("Dynamic comparison analysis completed.")
    return results_summary

def dynamic_comparison_plot(group_df, model, algorithm_prefix, degree, results_dir):
    """Create a detailed comparison plot for collapsed vs k-way combined time (similar to comparison_plot)"""

    # Get data for both types
    kway_data = group_df[group_df['algorithm_type'] == 'k-way']['combined_time']
    collapsed_data = group_df[group_df['algorithm_type'] == 'collapsed']['combined_time']

    plot_data = pd.DataFrame({
        'Algorithm': ['k-way'] * len(kway_data) + ['collapsed'] * len(collapsed_data),
        'Combined_Time': pd.concat([kway_data.reset_index(drop=True), collapsed_data.reset_index(drop=True)])
    })

    fig, axes = plt.subplots(1, 2, figsize=(12,8))

    # Histogram of differences (only for matched pairs)
    min_len = min(len(kway_data), len(collapsed_data))
    differences = kway_data.iloc[:min_len].values - collapsed_data.iloc[:min_len].values
    axes[0].hist(differences, bins='auto')
    axes[0].set_xlabel(f'Combined Time Difference (k-way - collapsed) [s]')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f"{model} | {algorithm_prefix} | k={degree}")
    axes[0].grid(True)

    # Calculate statistics for display
    kway_mean = kway_data.mean() if len(kway_data) > 0 else 0
    collapsed_mean = collapsed_data.mean() if len(collapsed_data) > 0 else 0
    percent_change = ((collapsed_mean - kway_mean) / kway_mean) * 100 if kway_mean > 0 else 0

    if len(kway_data) > 0 and len(collapsed_data) > 0:
        t_stat, p_value = stats.ttest_ind(kway_data, collapsed_data)

        text = '\n'.join((
            r'$t=%.2f$' % (t_stat,),
            r'$p=%.2e$' % (p_value,),
            r'Reject H0' if p_value < 0.05 else r'Fail to reject H0',
            r'%.2f%% change' % percent_change
        ))

        if p_value < 0.05:
            print("Significant result!")
            if percent_change < 0:
                print("Collapsed is faster!")
                text += " (COLLAPSED FASTER)"
            else:
                print("Collapsed is slower!")
                text += " (COLLAPSED SLOWER)"
    else:
        text = "Insufficient data"

    # Boxplot comparing k-way vs collapsed
    sns.boxplot(x='Algorithm', y='Combined_Time', hue='Algorithm', data=plot_data, palette='pastel', legend=False, ax=axes[1])
    axes[1].set_title(f'Combined Time Distribution: k-way vs collapsed')
    axes[1].set_ylabel('Combined Time (Construction + Traversal) [s]')

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
    plot_filename = results_dir / f"dynamic_comparison_{model}_{algorithm_prefix}_k{degree}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"  Dynamic comparison plot saved to: {plot_filename}")
    plt.close()  # Close figure to free memory

def dynamic_comparison_results_table(results_summary, results_dir: Path):
    """Create a formatted table for dynamic comparison results (collapsed vs k-way combined time)"""

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_summary)

    if results_df.empty:
        print("Gawrsh! No dynamic comparison results to make a table from!")
        return

    # Sort by degree first, then algorithm, then model
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
        'kway_mean': 'k-way Combined Time (s)',
        'collapsed_mean': 'Collapsed Combined Time (s)',
        'speedup_factor': 'Speedup',
        'percent_change': '% Change',
        't_stat': 't-statistic',
        'p_value_with_stars': 'p-value',
    }

    publication_df = formatted_df.rename(columns=column_mapping)

    # Select columns for publication table
    pub_columns = ['Model', 'Algorithm', 'k', 'k-way Combined Time (s)', 'Collapsed Combined Time (s)',
                   'Speedup', '% Change', 't-statistic', 'p-value']
    publication_df = publication_df[pub_columns]

    # Save as CSV
    csv_path = results_dir / "dynamic_comparison_results_table.csv"
    publication_df.to_csv(csv_path, index=False)
    print(f"Dynamic comparison results table saved to: {csv_path}")


    # Create a fancy visualization of the table
    fig, ax = plt.subplots(figsize=(20, max(8, int(len(publication_df) * 0.4))))
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

    # Color the % Change column based on performance (only if significant)
    pct_change_col_idx = publication_df.columns.get_loc('% Change')
    for table_row_idx, (_, row) in enumerate(publication_df.iterrows()):
        pct_change = float(row['% Change'])
        p_value_str = row['p-value']

        # Check if result is significant (has asterisks in p-value)
        is_significant = '*' in p_value_str

        if is_significant:
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

    plt.title('Dynamic Comparison: Collapsed vs k-way Combined Time\n' +
              'Green/Red = Significant Results Only | * p < 0.05, ** p < 0.01, *** p < 0.001',
              fontsize=14, fontweight='bold', pad=20)

    # Save the table image
    table_path = results_dir / "dynamic_comparison_results_table.png"
    plt.savefig(table_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Dynamic comparison table image saved to: {table_path}")

    # Print summary statistics
    significant_results = publication_df[publication_df['p-value'].str.contains(r'\*', regex=True, na=False)]
    print(f"\nDynamic Comparison Summary:")
    print(f"Total comparisons: {len(publication_df)}")
    print(f"Significant differences: {len(significant_results)}")

    if len(significant_results) > 0:
        print("\nCases where collapsed is significantly faster (combined time):")
        faster_collapsed = significant_results[significant_results['% Change'] < 0]
        for _, row in faster_collapsed.iterrows():
            print(f"  {row['Model']} | {row['Algorithm']} k={row['k']}: "
                  f"{row['% Change']:.1f}% faster (p={row['p-value']})")

        print("\nCases where collapsed is significantly slower (combined time):")
        slower_collapsed = significant_results[significant_results['% Change'] >= 0]
        for _, row in slower_collapsed.iterrows():
            print(f"  {row['Model']} | {row['Algorithm']} k={row['k']}: "
                  f"{row['% Change']:.1f}% slower (p={row['p-value']})")

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

    # static
    detailed_results_df = detailed_analysis(df, results_dir)
    create_detailed_results_table(detailed_results_df, results_dir)

    k_way_vs_collapsed_results = collapsed_vs_kway_analysis(df, results_dir)
    k_way_vs_collapsed_results_table(pd.DataFrame(k_way_vs_collapsed_results), results_dir)

    # dynamic
    dynamic_results = dynamic_model_results(df, results_dir)
    dynamic_model_results_table(dynamic_results, results_dir)

    dynamic_comparison_results = dynamic_comparison_analysis(df, results_dir)
    dynamic_comparison_results_table(dynamic_comparison_results, results_dir)

    # summary
    create_comprehensive_summary_graphs(detailed_results_df, k_way_vs_collapsed_results,
                                      dynamic_results, dynamic_comparison_results, results_dir)

if __name__ == "__main__":
    main()
