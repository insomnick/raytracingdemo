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
        combined["combined_time"] = combined["construction_time"] + combined["traversal_time"]

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


def detailed_quality_analysis(df, baseline_filter, metric, results_dir) -> pd.DataFrame:
    df = df.copy()
    # Check if there is a significant difference between baseline and test

    models = df["model_name"].unique()
    algorithm_prefixes = df["algorithm_prefix"].unique()
    algorithm_types = df["algorithm_type"].unique()
    algorithm_degrees = df["algorithm_degree"].unique()

    # 1. Aggregate over all camera steps to get mean per testrun
    testrun_means = df.groupby([
        "model_name", "algorithm_prefix", "algorithm_type", "algorithm_degree", "testrun_index"
    ]).agg({
        "traversal_time": "mean",
        "construction_time": "mean",
        "combined_time": "mean",
        "hitray_count": "mean"
    }).reset_index()

    results_summary = []

    # 2. Statistical analysis over repetitions for each combination
    for model in models:
        for algorithm_prefix in algorithm_prefixes:
            for algorithm_type in algorithm_types:
                for algorithm_degree in algorithm_degrees:
                    # when one of values is in baseline_filter, skip
                    if (algorithm_degree == baseline_filter.get('algorithm_degree', None)
                            or algorithm_type == baseline_filter.get('algorithm_type', None)
                            or algorithm_prefix == baseline_filter.get('algorithm_prefix', None)
                            or model == baseline_filter.get('model', None)):
                        continue


                    print(f'Analyzing { algorithm_prefix } , { algorithm_degree }, {algorithm_type}')
                    # Filter for this specific model and algorithm combination
                    baseline_data = testrun_means[
                        ((testrun_means["model_name"] == model) if 'model' not in baseline_filter else testrun_means["model_name"] == baseline_filter.get('model')) &
                        (testrun_means["algorithm_prefix"] == algorithm_prefix if 'algorithm_prefix' not in baseline_filter else testrun_means["algorithm_prefix"] == baseline_filter.get('algorithm_prefix')) &
                        (testrun_means["algorithm_degree"] == algorithm_degree if 'algorithm_degree' not in baseline_filter else testrun_means["algorithm_degree"] == baseline_filter.get('algorithm_degree'))
                        ]

                    # special case because there is no k=2 collapsed
                    if baseline_filter.get("algorithm_degree", None) != 2:
                        baseline_data = baseline_data[baseline_data["algorithm_type"] == algorithm_type if 'algorithm_type' not in baseline_filter else baseline_data["algorithm_type"] == baseline_filter.get('algorithm_type')]

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

                    baseline_times = baseline_data[metric]
                    test_times = test_data[metric]

                    if len(baseline_times) != len(test_times):
                        print(f"Unequal sample sizes (not expected): baseline={len(baseline_times)}, test={len(test_times)}. Skipping.")
                        continue

                    t_stat, p_value = stats.ttest_ind(baseline_times, test_times, equal_var=False)

                    baseline_mean = baseline_times.mean()
                    test_mean = test_times.mean()
                    speedup_factor = baseline_mean / test_mean
                    alpha = 0.05
                    is_significant = p_value < alpha

                    ci_low, ci_high = bootstrap_ci_speedup(baseline_times, test_times, 20000, alpha=alpha, random_state=42)

                    # Capture construction times for dynamic analysis
                    baseline_construction_mean = baseline_data['construction_time'].mean()
                    test_construction_mean = test_data['construction_time'].mean()

                    results_summary.append({
                        'model': model,
                        'algorithm_prefix': algorithm_prefix,
                        'degree': algorithm_degree,
                        'type': algorithm_type,
                        'baseline_mean': baseline_mean,
                        'test_mean': test_mean,
                        'baseline_construction_time': baseline_construction_mean,
                        'test_construction_time': test_construction_mean,
                        'speedup_factor': speedup_factor,
                        'speedup_factor_ci_low': ci_low,
                        'speedup_factor_ci_high': ci_high,
                        'percent_change': ((test_mean - baseline_mean) / baseline_mean) * 100,
                        't_stat': t_stat,
                        'p_value': p_value,
                        'significant': is_significant,
                        'sample_n': len(baseline_times)
                    })
                    #
                    # detailed_plot(results_dir=results_dir,
                    #               baseline_times=baseline_times,
                    #               test_times=test_times,
                    #               model=model,
                    #               algorithm_prefix=algorithm_prefix,
                    #               algorithm_degree=algorithm_degree,
                    #               algorithm_type=algorithm_type,
                    #               baseline_mean=baseline_mean,
                    #               test_mean=test_mean,
                    #               speedup_factor=speedup_factor,
                    #               percent_change=((test_mean - baseline_mean) / baseline_mean) * 100,
                    #               t_stat=t_stat,
                    #               p_value=p_value,
                    #               significant=is_significant,
                    #               baseline_n=len(baseline_times),
                    #               test_n=len(test_times)
                    #               )

    return pd.DataFrame(results_summary)

def bootstrap_ci_speedup(baseline_times, test_times, n_boot=20000, alpha=0.05, random_state=0):
    rng = np.random.default_rng(random_state)

    baseline = np.asarray(baseline_times, dtype=float)
    test = np.asarray(test_times, dtype=float)

    n_b = baseline.size
    n_t = test.size
    if n_b < 2 or n_t < 2:
        return np.nan, np.nan

    if np.mean(test) == 0:
        return np.nan, np.nan

    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        b_s = rng.choice(baseline, size=n_b, replace=True)
        t_s = rng.choice(test, size=n_t, replace=True)
        boot[i] = np.mean(b_s) / np.mean(t_s)

    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return lo, hi

def unified_plot(results_dir, data1, data2, model, algorithm_prefix, degree,
                 data1_label, data2_label, data1_mean, data2_mean,
                 speedup_factor, percent_change, t_stat, p_value, significant,
                 data1_n, data2_n, plot_type, time_unit="[s]", algorithm_type=None):

    results_dir.mkdir(exist_ok=True)

    # Create plot data DataFrame
    plot_data = pd.DataFrame({
        'Algorithm': [data1_label] * len(data1) + [data2_label] * len(data2),
        'Time': pd.concat([data1.reset_index(drop=True), data2.reset_index(drop=True)])
    })

    fig, axes = plt.subplots(1, 2, figsize=(12,8))

    # Left plot: Histogram of differences (for matched pairs)
    min_len = min(len(data1), len(data2))
    differences = data1.iloc[:min_len].values - data2.iloc[:min_len].values
    axes[0].hist(differences, bins='auto')

    # Configure labels based on plot type
    y_label = f'Time {time_unit}'  # Default
    box_title = f'Time Distribution: {data1_label} vs {data2_label}'  # Default
    plot_filename = results_dir / f"plot_{model}_{algorithm_prefix}_{degree}.png"  # Default

    if plot_type == "detailed":
        axes[0].set_xlabel(f'Time Difference ({data1_label} - {data2_label}) {time_unit}')
        axes[0].set_title(f"{model} | {algorithm_prefix} | {degree} | {algorithm_type}")
        y_label = f'Traversal Time {time_unit}'
        box_title = f'Traversal Time Distribution: {data1_label} vs {data2_label}'
        plot_filename = results_dir / f"detailed_{model}_{algorithm_prefix}_{algorithm_type}_{degree}.png"
    elif plot_type == "comparison":
        axes[0].set_xlabel(f'Time Difference ({data1_label} - {data2_label}) {time_unit}')
        axes[0].set_title(f"{model} | {algorithm_prefix} | k={degree}")
        y_label = f'Traversal Time {time_unit}'
        box_title = f'Traversal Time Distribution: {data1_label} vs {data2_label}'
        plot_filename = results_dir / f"comparison_{model}_{algorithm_prefix}_k{degree}.png"
    elif plot_type == "dynamic":
        axes[0].set_xlabel(f'Combined Time Difference ({data1_label} - {data2_label}) {time_unit}')
        if algorithm_type:
            axes[0].set_title(f"{model} | {algorithm_prefix} | {degree} | {algorithm_type}")
            plot_filename = results_dir / f"dynamic_{model}_{algorithm_prefix}_{algorithm_type}_{degree}.png"
        else:
            axes[0].set_title(f"{model} | {algorithm_prefix} | k={degree}")
            plot_filename = results_dir / f"dynamic_comparison_{model}_{algorithm_prefix}_k{degree}.png"
        y_label = f'Combined Time (Construction + Traversal) {time_unit}'
        box_title = f'Combined Time Distribution: {data1_label} vs {data2_label}'

    axes[0].set_ylabel('Count')
    axes[0].grid(True)

    # Create statistics text
    text = '\n'.join((
        r'$t=%.2f$' % (t_stat,),
        r'$p=%.2e$' % (p_value,),
        r'Reject H0' if significant else r'Fail to reject H0',
        r'%.2f%% change' % percent_change
    ))

    # Add significance indicators based on plot type
    if significant:
        if plot_type == "detailed":
            if percent_change < 0:
                text += " (FASTER)"
            else:
                text += " (SLOWER)"
        elif plot_type == "comparison":
            if percent_change < 0:
                text += " (COLLAPSED FASTER)"
            else:
                text += " (COLLAPSED SLOWER)"
        elif plot_type == "dynamic":
            if data2_label == "collapsed":  # dynamic comparison
                if percent_change < 0:
                    text += " (COLLAPSED FASTER)"
                else:
                    text += " (COLLAPSED SLOWER)"
            else:  # dynamic detailed
                if percent_change < 0:
                    text += " (FASTER)"
                else:
                    text += " (SLOWER)"

    # Right plot: Boxplot comparing the two algorithms
    sns.boxplot(x='Algorithm', y='Time', hue='Algorithm', data=plot_data,
                palette='pastel', legend=False, ax=axes[1])
    axes[1].set_title(box_title)
    axes[1].set_ylabel(y_label)

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

    # Save plot
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')

    # Print appropriate message
    if plot_type == "detailed":
        print(f"Plot saved to: {plot_filename}")
    elif plot_type == "comparison":
        print(f"Comparison plot saved to: {plot_filename}")
    elif plot_type == "dynamic":
        if algorithm_type:
            print(f"Dynamic plot saved to: {plot_filename}")
        else:
            print(f"  Dynamic comparison plot saved to: {plot_filename}")

    plt.close()  # Close figure to free memory


def detailed_plot(results_dir, baseline_times, test_times, model, algorithm_prefix, algorithm_degree, algorithm_type, baseline_mean, test_mean, speedup_factor, percent_change, t_stat, p_value, significant, baseline_n, test_n):
    unified_plot(results_dir, baseline_times, test_times, model, algorithm_prefix, algorithm_degree,
                 "k=2", f"k={algorithm_degree}", baseline_mean, test_mean,
                 speedup_factor, percent_change, t_stat, p_value, significant,
                 baseline_n, test_n, "detailed", "[s]", algorithm_type)

def create_detailed_results_table(results_df, results_dir: Path, column_mapping):

    results_dir.mkdir(exist_ok=True)
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
    formatted_df['baseline_mean'] = formatted_df['baseline_mean'].round(5)
    formatted_df['test_mean'] = formatted_df['test_mean'].round(5)
    if 'baseline_construction_time' in formatted_df.columns:
        formatted_df['baseline_construction_time'] = formatted_df['baseline_construction_time'].round(5)
    if 'test_construction_time' in formatted_df.columns:
        formatted_df['test_construction_time'] = formatted_df['test_construction_time'].round(5)
    formatted_df['speedup_factor'] = formatted_df['speedup_factor'].round(5)
    formatted_df['speedup_factor_ci_low'] = formatted_df['speedup_factor_ci_low'].round(5)
    formatted_df['speedup_factor_ci_high'] = formatted_df['speedup_factor_ci_high'].round(5)
    formatted_df['percent_change'] = formatted_df['percent_change'].round(5)
    formatted_df['t_stat'] = formatted_df['t_stat'].round(5)

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

    # delete all columns not in column mapping keys
    formatted_df = formatted_df[column_mapping.keys()]
    publication_df = formatted_df.rename(columns=column_mapping)


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

    # Color the Speedup column based on performance (only if significant)
    speedup_col_idx = publication_df.columns.get_loc('Speedup')
    for table_row_idx, (_, row) in enumerate(publication_df.iterrows()):
        speedup_f = float(row['Speedup'])
        p_value_str = row['p-value']

        # Check if result is significant (has asterisks in p-value)
        is_significant = '*' in p_value_str

        if is_significant:
            if speedup_f > 1:  # Faster (improvement)
                color = '#90EE90'  # Light green
            else:  # Slower (degradation)
                color = '#FFB6C1'  # Light red

            # Color the % Change column (table_row_idx + 1 because row 0 is header)
            table[(table_row_idx + 1, speedup_col_idx)].set_facecolor(color)

    # Style header
    for j in range(len(publication_df.columns)):
        table[(0, j)].set_facecolor('#cccccc')
        table[(0, j)].set_text_props(weight='bold')

    plt.title('BVH Performance Comparison\n' +
              'Green/Red = Significant Results Only | * p < 0.05, ** p < 0.01, *** p < 0.001\n'
              'n = 10 repetitions per configuration',
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
            print(f"  {row['Model']} | {row['Algorithm']} k={row['k']}: "
                  f"{row['% Change']:.1f}% faster (p={row['p-value']})")

        print("\nSignificant degradations (slower algorithms):")
        degradations = significant_results[significant_results['% Change'] >= 0]
        for _, row in degradations.iterrows():
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
    detailed_results_df = detailed_quality_analysis(df, {'algorithm_degree' : 2}, 'traversal_time', results_dir / 'static')
    detailed_column_mapping = {
        'model': 'Model',
        'algorithm_prefix': 'Algorithm',
        'degree': 'k',
        'type': 'type',
        'baseline_mean': 'k=2 Time (s)',
        'test_mean': f'k=X Time (s)',
        'speedup_factor': 'Speedup',
        'speedup_factor_ci_low': 'Speedup CI Min',
        'speedup_factor_ci_high': 'Speedup CI Max',
        'percent_change': '% Change',
        't_stat': 't-statistic',
        'p_value_with_stars': 'p-value',
    }
    create_detailed_results_table(detailed_results_df, results_dir / 'static', detailed_column_mapping)

    # collapsed vs k-way
    comparison_results_df = detailed_quality_analysis(df, { 'algorithm_type' : 'collapsed' }, 'traversal_time', results_dir / 'comparison')
    comparison_column_mapping = {
        'model': 'Model',
        'algorithm_prefix': 'Algorithm',
        'degree': 'k',
        'baseline_mean': 'k-way Time (s)',
        'test_mean': f'collapsed Time (s)',
        'speedup_factor': 'Speedup',
        'speedup_factor_ci_low': 'Speedup CI Min',
        'speedup_factor_ci_high': 'Speedup CI Max',        'percent_change': '% Change',
        't_stat': 't-statistic',
        'p_value_with_stars': 'p-value',
    }
    create_detailed_results_table(comparison_results_df, results_dir / 'comparison', comparison_column_mapping)

    # dynamic
    dynamic_results_df = detailed_quality_analysis(df, { 'algorithm_degree' : 2 }, 'combined_time', results_dir / 'dynamic')
    dynamic_column_mapping = {
        'model': 'Model',
        'algorithm_prefix': 'Algorithm',
        'degree': 'k',
        'type': 'type',
        'baseline_mean': 'k=2 Time (s)',
        'test_mean': 'k=X Time (s)',
        'baseline_construction_time': 'C-Time k=2 (s)',
        'test_construction_time': 'C-Time k=X (s)',
        'speedup_factor': 'Speedup',
        'speedup_factor_ci_low': 'Speedup CI Min',
        'speedup_factor_ci_high': 'Speedup CI Max',
        'percent_change': '% Change',
        't_stat': 't-statistic',
        'p_value_with_stars': 'p-value',
    }
    create_detailed_results_table(dynamic_results_df, results_dir / 'dynamic', dynamic_column_mapping)


if __name__ == "__main__":
    main()
