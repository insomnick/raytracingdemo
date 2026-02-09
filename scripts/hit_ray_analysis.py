import os
import glob
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def get_algorithm_name_for_testrun(testrun_dir):
    """Look up the real algorithm name (with k-value) from render_times.csv."""
    render_times_path = os.path.join(testrun_dir, "render_times.csv")
    if os.path.exists(render_times_path):
        df = pd.read_csv(render_times_path)
        if 'algorithm_name' in df.columns and len(df) > 0:
            return df['algorithm_name'].iloc[0]
    return None


def find_ray_data_files(base_dir="./"):
    ray_data_files = []

    testrun_dirs = sorted(glob.glob(os.path.join(base_dir, "testruns_hitrays", "testrun_*")))

    for testrun_dir in testrun_dirs:
        algo_name = get_algorithm_name_for_testrun(testrun_dir)
        if algo_name is None:
            print(f"Warning: could not determine algorithm name for {testrun_dir}, skipping")
            continue

        for pattern in ['aabb_pos_*ray_data*.csv', 'aabb_neg_*ray_data*.csv',
                        'tri_pos_*ray_data*.csv', 'tri_neg_*ray_data*.csv']:
            files = glob.glob(os.path.join(testrun_dir, pattern))
            for f in files:
                ray_data_files.append((f, algo_name))

    return ray_data_files


def load_and_process_data(files):
    all_data = []

    for file_path, algo_name in files:
        try:
            df = pd.read_csv(file_path)

            file_name = os.path.basename(file_path)

            if file_name.startswith('aabb_pos_'):
                test_type = 'aabb_pos'
            elif file_name.startswith('aabb_neg_'):
                test_type = 'aabb_neg'
            elif file_name.startswith('tri_pos_'):
                test_type = 'tri_pos'
            elif file_name.startswith('tri_neg_'):
                test_type = 'tri_neg'
            else:
                continue

            # Override the generic algorithm_name with the real one from render_times
            df['algorithm_name'] = algo_name
            df['test_type'] = test_type
            df['count'] = df['time_seconds'].astype(int)

            all_data.append(df)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    combined_df = pd.concat(all_data, ignore_index=True)

    # Combine tri_pos and tri_neg into a single 'tri' test type
    tri_data = []
    for name, group in combined_df.groupby(['model_name', 'algorithm_name', 'cam_pos_x', 'cam_pos_y', 'cam_pos_z']):
        tri_pos = group[group['test_type'] == 'tri_pos']['count'].sum()
        tri_neg = group[group['test_type'] == 'tri_neg']['count'].sum()

        if tri_pos > 0 or tri_neg > 0:
            tri_row = group.iloc[0].copy()
            tri_row['test_type'] = 'tri'
            tri_row['count'] = tri_pos + tri_neg
            tri_data.append(tri_row)

    if tri_data:
        tri_df = pd.DataFrame(tri_data)
        combined_df = combined_df[~combined_df['test_type'].isin(['tri_pos', 'tri_neg'])]
        combined_df = pd.concat([combined_df, tri_df], ignore_index=True)

    return combined_df


def aggregate_data(df):
    df['camera_position'] = df['cam_pos_x'].astype(str) + ',' + df['cam_pos_y'].astype(str) + ',' + df['cam_pos_z'].astype(str)

    aggregated = df.groupby(['model_name', 'algorithm_name', 'camera_position', 'test_type'])['count'].sum().reset_index()

    return aggregated


def _draw_stacked_bar_chart(ax, pivot_df, algorithms, x_labels, xlabel):
    """Helper to draw a single stacked bar chart on a given axis."""
    width = 0.6
    x = np.arange(len(algorithms))

    colors = ['#A8E6CF', '#7FCDCD', '#FFB3BA']

    bottom = np.zeros(len(algorithms))

    test_order = ['aabb_pos', 'aabb_neg', 'tri']
    labels = ['AABB Positive Tests', 'AABB Negative Tests', 'Triangle Tests']

    for i, (test_type, label) in enumerate(zip(test_order, labels)):
        if test_type in pivot_df.columns:
            values = pivot_df[test_type].values
            bars = ax.bar(x, values, width, label=label,
                          bottom=bottom, color=colors[i],
                          edgecolor='white', linewidth=1.5, alpha=0.8)

            for j, (bar, value) in enumerate(zip(bars, values)):
                if value > 1000:
                    y_pos = bottom[j] + bar.get_height() / 2
                    if value >= 1000000:
                        text = f'{value/1000000:.1f}M'
                    elif value >= 1000:
                        text = f'{value/1000:.0f}k'
                    else:
                        text = f'{int(value)}'
                    ax.text(bar.get_x() + bar.get_width() / 2, y_pos, text,
                            ha='center', va='center', fontweight='bold',
                            fontsize=9, color='black')

            bottom += values

    # Total labels above bars
    for i in range(len(algorithms)):
        total = bottom[i]
        if total > 0:
            if total >= 1000000:
                total_text = f'Total: {total/1000000:.1f}M'
            elif total >= 1000:
                total_text = f'Total: {total/1000:.0f}k'
            else:
                total_text = f'Total: {int(total)}'
            ax.text(i, total + max(bottom) * 0.02, total_text,
                    ha='center', va='bottom', fontweight='bold', fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Intersection Tests per Frame', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, p: f'{int(val):,}'))
    ax.legend(loc='upper left', framealpha=0.9, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _draw_mixed_chart(ax, df):
    """Draw mixed chart with k-way and collapsing bars side by side per k-value."""
    k_values = [4, 8, 16]
    kway_algos = [f'bsah-{k}' for k in k_values]
    collapse_algos = [f'bsah-c-{k}' for k in k_values]
    all_algos = []
    for k, c in zip(kway_algos, collapse_algos):
        all_algos.extend([k, c])

    filtered_df = df[df['algorithm_name'].isin(all_algos)]
    pivot_df = filtered_df.pivot_table(
        index='algorithm_name', columns='test_type',
        values='count', aggfunc='mean', fill_value=0
    )
    pivot_df = pivot_df.reindex(all_algos)

    bar_width = 0.35
    colors = ['#A8E6CF', '#7FCDCD', '#FFB3BA']
    test_order = ['aabb_pos', 'aabb_neg', 'tri']
    labels = ['AABB Positive Tests', 'AABB Negative Tests', 'Triangle Tests']

    # Group positions: each group has 2 bars, groups are at 0, 1, 2
    group_positions = np.arange(len(k_values))
    offsets = [-bar_width / 2, bar_width / 2]

    legend_added = set()

    for group_idx, k in enumerate(k_values):
        for bar_idx, algo in enumerate([f'bsah-{k}', f'bsah-c-{k}']):
            if algo not in pivot_df.index:
                continue
            x_pos = group_positions[group_idx] + offsets[bar_idx]
            bottom = 0
            for test_idx, (test_type, label) in enumerate(zip(test_order, labels)):
                if test_type not in pivot_df.columns:
                    continue
                value = pivot_df.loc[algo, test_type]
                show_label = label if label not in legend_added else None
                ax.bar(x_pos, value, bar_width, bottom=bottom,
                       color=colors[test_idx], edgecolor='white',
                       linewidth=1.5, alpha=0.8, label=show_label)
                if show_label:
                    legend_added.add(label)

                if value > 1000:
                    y_pos = bottom + value / 2
                    if value >= 1000000:
                        text = f'{value/1000000:.1f}M'
                    elif value >= 1000:
                        text = f'{value/1000:.0f}k'
                    else:
                        text = f'{int(value)}'
                    ax.text(x_pos, y_pos, text, ha='center', va='center',
                            fontweight='bold', fontsize=8, color='black')

                bottom += value

            # Total label
            if bottom > 0:
                if bottom >= 1000000:
                    total_text = f'{bottom/1000000:.1f}M'
                elif bottom >= 1000:
                    total_text = f'{bottom/1000:.0f}k'
                else:
                    total_text = f'{int(bottom)}'
                ax.text(x_pos, bottom + ax.get_ylim()[1] * 0.01 if ax.get_ylim()[1] > 0 else bottom * 1.02,
                        total_text, ha='center', va='bottom', fontweight='bold', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', alpha=0.7))

    # X-axis: group labels showing k-value, with sub-labels for kway/collapse
    ax.set_xticks(group_positions)
    ax.set_xticklabels([f'k={k}' for k in k_values])

    # Add sub-labels
    for group_idx, k in enumerate(k_values):
        for bar_idx, sublabel in enumerate(['k-way', 'collapse']):
            x_pos = group_positions[group_idx] + offsets[bar_idx]
            ax.text(x_pos, -ax.get_ylim()[1] * 0.04 if ax.get_ylim()[1] > 0 else 0,
                    sublabel, ha='center', va='top', fontsize=8, fontstyle='italic')

    ax.set_xlabel('Branching Factor k (k-way vs collapse)', fontsize=12, fontweight='bold', labelpad=20)
    ax.set_ylabel('Intersection Tests per Frame', fontsize=12, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, p: f'{int(val):,}'))
    ax.legend(loc='upper left', framealpha=0.9, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def create_charts(df):
    # --- Chart 1: K-way splitting (no -c-): bsah-2, bsah-4, bsah-8, bsah-16 ---
    kway_algorithms = ['bsah-2', 'bsah-4', 'bsah-8', 'bsah-16']
    kway_labels = ['2', '4', '8', '16']

    kway_df = df[df['algorithm_name'].isin(kway_algorithms)]
    kway_pivot = kway_df.pivot_table(
        index='algorithm_name', columns='test_type',
        values='count', aggfunc='mean', fill_value=0
    ).reindex(kway_algorithms)

    fig1, ax1 = plt.subplots(figsize=(14, 8))
    _draw_stacked_bar_chart(ax1, kway_pivot, kway_algorithms, kway_labels,
                            'Branching Factor k (k-way splitting)')
    plt.tight_layout()
    fig1.savefig('hit_ray_analysis_kway.png', dpi=300, bbox_inches='tight')
    print("Chart saved as hit_ray_analysis_kway.png")
    plt.close(fig1)

    # --- Chart 2: Collapsing (with -c-) + bsah-2 baseline ---
    collapse_algorithms = ['bsah-2', 'bsah-c-4', 'bsah-c-8', 'bsah-c-16']
    collapse_labels = ['2', 'c-4', 'c-8', 'c-16']

    collapse_df = df[df['algorithm_name'].isin(collapse_algorithms)]
    collapse_pivot = collapse_df.pivot_table(
        index='algorithm_name', columns='test_type',
        values='count', aggfunc='mean', fill_value=0
    ).reindex(collapse_algorithms)

    fig2, ax2 = plt.subplots(figsize=(14, 8))
    _draw_stacked_bar_chart(ax2, collapse_pivot, collapse_algorithms, collapse_labels,
                            'Branching Factor k (collapsing)')
    plt.tight_layout()
    fig2.savefig('hit_ray_analysis_collapse.png', dpi=300, bbox_inches='tight')
    print("Chart saved as hit_ray_analysis_collapse.png")
    plt.close(fig2)

    # --- Chart 3: Mixed - k-way vs collapse side by side per k-value ---
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    _draw_mixed_chart(ax3, df)
    plt.tight_layout()
    fig3.savefig('hit_ray_analysis_mixed.png', dpi=300, bbox_inches='tight')
    print("Chart saved as hit_ray_analysis_mixed.png")
    plt.close(fig3)


def main():
    print("Searching for ray data files...")
    ray_data_files = find_ray_data_files()

    if ray_data_files:
        print(f"Found {len(ray_data_files)} ray data files:")
        for file_path, algo in ray_data_files:
            print(f"  - {file_path} ({algo})")
    else:
        print("No ray data files found.")
        return

    print("\nLoading and processing data...")
    df = load_and_process_data(ray_data_files)

    print("\nAggregating data...")
    aggregated_df = aggregate_data(df)

    print(f"Data shape after aggregation: {aggregated_df.shape}")
    print(f"Available algorithms: {sorted(aggregated_df['algorithm_name'].unique())}")

    print("\nCreating charts...")
    create_charts(aggregated_df)


if __name__ == "__main__":
    main()
