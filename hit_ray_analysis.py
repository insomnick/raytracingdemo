import os
import glob
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def find_ray_data_files(base_dir="./"):
    ray_data_files = []

    testrun_dirs = glob.glob(os.path.join(base_dir, "testruns*"))

    for testrun_dir in testrun_dirs:
        aabb_pos_files = glob.glob(os.path.join(testrun_dir, "**/aabb_pos_*ray_data*.csv"), recursive=True)
        aabb_neg_files = glob.glob(os.path.join(testrun_dir, "**/aabb_neg_*ray_data*.csv"), recursive=True)
        tri_pos_files = glob.glob(os.path.join(testrun_dir, "**/tri_pos_*ray_data*.csv"), recursive=True)
        tri_neg_files = glob.glob(os.path.join(testrun_dir, "**/tri_neg_*ray_data*.csv"), recursive=True)

        # Also check for old format files as fallback
        aabb_files = glob.glob(os.path.join(testrun_dir, "**/aabb*ray_data*.csv"), recursive=True)
        tri_files = glob.glob(os.path.join(testrun_dir, "**/tri_*ray_data*.csv"), recursive=True)

        ray_data_files.extend(aabb_pos_files)
        ray_data_files.extend(aabb_neg_files)
        ray_data_files.extend(tri_pos_files)
        ray_data_files.extend(tri_neg_files)

        # Add old format files only if no new format files found
        if not (aabb_pos_files or aabb_neg_files or tri_pos_files or tri_neg_files):
            ray_data_files.extend(aabb_files)
            ray_data_files.extend(tri_files)

    return ray_data_files

def load_and_process_data(files):
    all_data = []

    for file_path in files:
        try:
            df = pd.read_csv(file_path)

            file_name = os.path.basename(file_path)

            # Determine test type and result type from filename
            if file_name.startswith('aabb_pos_'):
                test_type = 'aabb_pos'
                base_type = 'aabb'
            elif file_name.startswith('aabb_neg_'):
                test_type = 'aabb_neg'
                base_type = 'aabb'
            elif file_name.startswith('tri_pos_'):
                test_type = 'tri_pos'
                base_type = 'tri'
            elif file_name.startswith('tri_neg_'):
                test_type = 'tri_neg'
                base_type = 'tri'
            elif file_name.startswith('aabb'):
                test_type = 'aabb'
                base_type = 'aabb'
            elif file_name.startswith('tri_'):
                test_type = 'tri'
                base_type = 'tri'
            else:
                continue

            df['test_type'] = test_type
            df['base_type'] = base_type
            df['count'] = df['time_seconds'].astype(int)

            all_data.append(df)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    combined_df = pd.concat(all_data, ignore_index=True)

    # Check if we have pos/neg data
    has_pos_neg = any(test_type in ['aabb_pos', 'aabb_neg', 'tri_pos', 'tri_neg']
                      for test_type in combined_df['test_type'].unique())

    if has_pos_neg:
        # Group pos and neg tests for tri to get total tri count
        tri_data = []
        for name, group in combined_df.groupby(['model_name', 'algorithm_name', 'cam_pos_x', 'cam_pos_y', 'cam_pos_z']):
            tri_pos = group[group['test_type'] == 'tri_pos']['count'].sum() if 'tri_pos' in group['test_type'].values else 0
            tri_neg = group[group['test_type'] == 'tri_neg']['count'].sum() if 'tri_neg' in group['test_type'].values else 0

            if tri_pos > 0 or tri_neg > 0:
                tri_total = tri_pos + tri_neg
                # Create a new row for total tri tests
                tri_row = group.iloc[0].copy()
                tri_row['test_type'] = 'tri'
                tri_row['base_type'] = 'tri'
                tri_row['count'] = tri_total
                tri_data.append(tri_row)

        if tri_data:
            tri_df = pd.DataFrame(tri_data)
            # Remove individual tri_pos and tri_neg rows and add tri totals
            combined_df = combined_df[~combined_df['test_type'].isin(['tri_pos', 'tri_neg'])]
            combined_df = pd.concat([combined_df, tri_df], ignore_index=True)

    return combined_df

def aggregate_data(df):
    df['camera_position'] = df['cam_pos_x'].astype(str) + ',' + df['cam_pos_y'].astype(str) + ',' + df['cam_pos_z'].astype(str)

    aggregated = df.groupby(['model_name', 'algorithm_name', 'camera_position', 'test_type'])['count'].sum().reset_index()

    return aggregated

def create_stacked_bar_chart(df):
    algorithms = ['median-2', 'median-c-4', 'median-c-8', 'median-c-16']

    filtered_df = df[df['algorithm_name'].isin(algorithms)]

    # Create pivot tables for different test types
    pivot_df = filtered_df.pivot_table(
        index='algorithm_name',
        columns='test_type',
        values='count',
        aggfunc='mean',
        fill_value=0
    )

    pivot_df = pivot_df.reindex(algorithms)

    fig, ax = plt.subplots(figsize=(14, 8))

    width = 0.6
    x = np.arange(len(algorithms))

    # Colors for different test types - softer tones with better differentiation
    colors = ['#A8E6CF', '#7FCDCD', '#FFB3BA']  # Soft mint green for AABB pos, soft teal for AABB neg, soft pink for TRI

    bottom = np.zeros(len(algorithms))
    bar_plots = []

    # Order of stacking: aabb_pos, aabb_neg, tri
    test_order = ['aabb_pos', 'aabb_neg', 'tri']
    labels = ['AABB Positive Tests', 'AABB Negative Tests', 'Triangle Tests']

    for i, (test_type, label) in enumerate(zip(test_order, labels)):
        if test_type in pivot_df.columns:
            values = pivot_df[test_type].values
            bars = ax.bar(x, values, width, label=label,
                   bottom=bottom, color=colors[i],
                   edgecolor='white', linewidth=1.5, alpha=0.8)
            bar_plots.append(bars)

            # Add count labels on each bar section
            for j, (bar, value) in enumerate(zip(bars, values)):
                if value > 1000:  # Only show label if there's a significant value
                    height = bar.get_height()
                    y_pos = bottom[j] + height/2
                    # Format large numbers
                    if value >= 1000000:
                        text = f'{value/1000000:.1f}M'
                    elif value >= 1000:
                        text = f'{value/1000:.0f}k'
                    else:
                        text = f'{int(value)}'

                    ax.text(bar.get_x() + bar.get_width()/2, y_pos, text,
                           ha='center', va='center', fontweight='bold',
                           fontsize=9, color='black')

            bottom += values

    # Add total sum labels above each bar
    for i, algorithm in enumerate(algorithms):
        total_tests = bottom[i]
        if total_tests > 0:
            if total_tests >= 1000000:
                total_text = f'Total: {total_tests/1000000:.1f}M'
            elif total_tests >= 1000:
                total_text = f'Total: {total_tests/1000:.0f}k'
            else:
                total_text = f'Total: {int(total_tests)}'

            ax.text(i, total_tests + max(bottom) * 0.02, total_text,
                    ha='center', va='bottom', fontweight='bold', fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))

    ax.set_xlabel('Branching Factor k (Median, collapse)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Intersection Tests per Frame', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['2', '4', '8', '16'])

    # Format y-axis to show thousands separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    ax.legend(loc='upper left', framealpha=0.9, fontsize=11)

    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    output_path = 'hit_ray_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved as {output_path}")

    return pivot_df

def print_detailed_summary(df):
    algorithms = ['median-2', 'median-4', 'median-8', 'median-16']

    filtered_df = df[df['algorithm_name'].isin(algorithms)]

    pivot_df = filtered_df.pivot_table(
        index='algorithm_name',
        columns='test_type',
        values='count',
        aggfunc='mean',
        fill_value=0
    )

    pivot_df = pivot_df.reindex(algorithms)

def main():
    print("Searching for ray data files...")
    ray_data_files = find_ray_data_files()

    if ray_data_files:
        print(f"Found {len(ray_data_files)} ray data files:")
        for file in ray_data_files:
            print(f"  - {file}")
    else:
        print("No ray data files found. Using sample data for demonstration.")

    print("\nLoading and processing data...")
    df = load_and_process_data(ray_data_files)

    print("\nAggregating data...")
    aggregated_df = aggregate_data(df)

    print(f"Data shape after aggregation: {aggregated_df.shape}")
    print(f"Available algorithms: {sorted(aggregated_df['algorithm_name'].unique())}")

    print("\nCreating stacked bar chart...")
    result_df = create_stacked_bar_chart(aggregated_df)

    print_detailed_summary(aggregated_df)

if __name__ == "__main__":
    main()
