#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.ioff()

# Helper function to capitalize model names
def capitalize_model_name(name):
    """Capitalize model names for display (e.g., 'stanford-bunny' -> 'Stanford-Bunny')."""
    if pd.isna(name):
        return name
    # Split on hyphen, capitalize each part, rejoin
    parts = str(name).split('-')
    return '-'.join(part.capitalize() for part in parts)

# Color scheme: similar colors for same algorithm, different shades for types
COLOR_SCHEME = {
    'bsah-k-way': '#0066CC',      # Dark blue
    'bsah-collapsed': '#66B3FF',  # Light blue
    'sah-k-way': '#FF6600',       # Dark orange
    'sah-collapsed': '#FFAA66',   # Light orange
    'median-k-way': '#00AA00',    # Dark green
    'median-collapsed': '#66DD66', # Light green
    # For comparison graphs (no type suffix)
    'bsah': '#3399FF',            # Medium blue (between k-way and collapsed)
    'sah': '#FF9933',             # Medium orange
    'median': '#33CC33'           # Medium green
}

MARKER_SCHEME = {
    'k-way': 'o',      # Circle
    'collapsed': 's',  # Square
    None: 'D'          # Diamond for comparison (no specific type)
}

# Fallback colors for any unexpected combinations
FALLBACK_COLORS = ['#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2']


def load_polygon_counts(example_dir: Path) -> pd.DataFrame:
    """Load polygon counts from object_meta.csv."""
    meta_file = example_dir / "object_meta.csv"
    if not meta_file.exists():
        raise FileNotFoundError(f"object_meta.csv not found at {meta_file}")

    df = pd.read_csv(meta_file)
    # Clean model names (remove .obj extension if present)
    df['model_name'] = df['model_name'].str.replace('.obj', '', regex=False)
    return df[['model_name', 'polygon_count']]


def create_graph_for_k(df: pd.DataFrame, k_value: int, output_path: Path, result_type: str):
    """
    Create a scatter plot for a specific k value.

    Args:
        df: DataFrame with columns: Model, Algorithm, k, type, Speedup, Speedup CI Min, Speedup CI Max, polygon_count
        k_value: The k value to plot
        output_path: Path to save the graph
        result_type: Type of result (static, dynamic, comparison)
    """
    # Filter data for this k value
    k_data = df[df['k'] == k_value].copy()

    if len(k_data) == 0:
        print(f"  No data for k={k_value}, skipping")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get unique algorithm-type combinations
    if 'type' in k_data.columns:
        k_data['algorithm_type'] = k_data['Algorithm'] + '-' + k_data['type']
        algo_types = sorted(k_data['algorithm_type'].unique())
    else:
        # For comparison results, we might not have a type column
        k_data['algorithm_type'] = k_data['Algorithm']
        algo_types = sorted(k_data['algorithm_type'].unique())

    # Plot each algorithm-type combination
    fallback_idx = 0
    for algo_type in algo_types:
        subset = k_data[k_data['algorithm_type'] == algo_type].copy()
        subset = subset.sort_values('polygon_count')

        if len(subset) == 0:
            continue

        # Get color and marker
        if algo_type in COLOR_SCHEME:
            color = COLOR_SCHEME[algo_type]
        else:
            # Use fallback color
            color = FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)]
            fallback_idx += 1
            print(f"    Warning: Using fallback color for {algo_type}")

        # Determine marker based on type
        if 'type' in subset.columns and len(subset) > 0:
            type_value = subset['type'].iloc[0]
            marker = MARKER_SCHEME.get(type_value, 'o')
        else:
            # For comparison graphs without type column, use diamond
            marker = MARKER_SCHEME.get(None, 'D')

        # Calculate error bars (asymmetric)
        yerr_lower = subset['Speedup'] - subset['Speedup CI Min']
        yerr_upper = subset['Speedup CI Max'] - subset['Speedup']

        # Plot with error bars - make them very visible
        ax.errorbar(
            subset['polygon_count'],
            subset['Speedup'],
            yerr=[yerr_lower, yerr_upper],
            fmt='none',  # Don't use fmt for line/marker - specify separately
            color=color,
            capsize=10,     # Larger caps
            capthick=1,     # Thicker caps
            elinewidth=1,   # Much thicker error bar lines
            alpha=1.0,      # Full opacity
            zorder=2
        )

        # Plot markers on top - small so error bars are visible
        ax.scatter(
            subset['polygon_count'],
            subset['Speedup'],
            marker=marker,
            s=40,  # Small markers to show error bars better
            color=color,
            edgecolors='black',
            linewidths=1,
            label=algo_type,
            alpha=0.95,
            zorder=3  # Ensure markers are on top
        )

    # Configure axes
    ax.set_xlabel('Model Polygon Count', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    # Add horizontal line at y=1 (no speedup)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Add model name labels on x-axis
    unique_models = k_data[['Model', 'polygon_count']].drop_duplicates()
    unique_models = unique_models.sort_values('polygon_count')

    # Set custom x-ticks at model positions with both name and count
    ax.set_xticks(unique_models['polygon_count'])
    tick_labels = [f"{capitalize_model_name(row['Model'])}\n({int(row['polygon_count'])})"
                   for _, row in unique_models.iterrows()]
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)

    # Add vertical lines at each model position for clarity
    for _, row in unique_models.iterrows():
        ax.axvline(x=row['polygon_count'], color='lightgray', linestyle=':', linewidth=1, alpha=0.5, zorder=1)

    # Tight layout
    fig.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def process_result_directory(result_dir: Path, polygon_df: pd.DataFrame, testrun_dir: Path):
    """
    Process a result directory (static, dynamic, or comparison).

    Args:
        result_dir: Path to result directory (e.g., results/static)
        polygon_df: DataFrame with polygon counts
        testrun_dir: Path to testrun directory (for output)
    """
    csv_file = result_dir / "statistical_results_table.csv"

    if not csv_file.exists():
        print(f"No results found in {result_dir}, skipping")
        return

    # Load results
    results_df = pd.read_csv(csv_file)

    # Merge with polygon counts
    results_df = results_df.merge(
        polygon_df,
        left_on='Model',
        right_on='model_name',
        how='left'
    )

    # Drop rows with missing polygon counts
    missing = results_df[results_df['polygon_count'].isna()]
    if len(missing) > 0:
        print(f"  Warning: Skipping {len(missing)} rows with missing polygon data:")
        for model in missing['Model'].unique():
            print(f"    - {model}")
        results_df = results_df.dropna(subset=['polygon_count'])

    if len(results_df) == 0:
        print(f"  No valid data after merging, skipping")
        return

    # Get unique k values
    k_values = sorted(results_df['k'].unique())

    print(f"  Found k values: {k_values}")

    # Create graphs directory in testrun/graphs/<type>/
    graphs_dir = testrun_dir / "graphs" / result_dir.name

    # Create a graph for each k value
    for k_value in k_values:
        output_filename = f"speedup_k{k_value}.png"
        output_path = graphs_dir / output_filename

        create_graph_for_k(results_df, k_value, output_path, result_dir.name)


# ============================================================================
# Extended Graphs for Research Questions
# ============================================================================

def create_rq1_lineplot(df: pd.DataFrame, output_path: Path, result_type: str):
    """
    RQ1 Graph 1: Line plot of speedup vs k for each model (excluding k=2).
    """
    # Filter out k=2 (baseline)
    df_filtered = df[df['k'] != 2].copy()

    if len(df_filtered) == 0:
        print(f"  No data for k>2, skipping RQ1 lineplot")
        return

    # Get data for each model
    models = sorted(df_filtered['Model'].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors by polygon count (gradient)
    polygon_counts = df_filtered.groupby('Model')['polygon_count'].first().sort_values()
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=np.log10(polygon_counts.min()),
                         vmax=np.log10(polygon_counts.max()))

    for model in models:
        model_data = df_filtered[df_filtered['Model'] == model].copy()
        model_data = model_data.sort_values('k')

        # Get average speedup per k value and algorithm-type
        if 'type' in model_data.columns:
            # For static/dynamic: average over algorithm-type combinations
            plot_data = model_data.groupby('k')['Speedup'].mean().reset_index()
        else:
            # For comparison: just use as is
            plot_data = model_data.groupby('k')['Speedup'].mean().reset_index()

        polygon_count = polygon_counts[model]
        color = cmap(norm(np.log10(polygon_count)))

        ax.plot(plot_data['k'], plot_data['Speedup'],
                marker='o', linewidth=2, markersize=8,
                label=f"{capitalize_model_name(model)} ({int(polygon_count):,})",
                color=color)

    ax.set_xlabel('Branching Factor k', fontsize=12)
    ax.set_ylabel('Speedup Factor', fontsize=12)
    ax.set_xticks([4, 8, 16])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9, title='Model (Polygons)')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def create_rq1_heatmap_model_algo_grid(df: pd.DataFrame, output_path: Path, result_type: str):
    """
    RQ1 Graph: Heatmap with 3 subplots (k=4, 8, 16), models as rows, algorithms as columns.
    Creates one for k-way, one for collapsed, and one combined (averaged).
    """
    # Filter out k=2
    df_filtered = df[df['k'] != 2].copy()

    if len(df_filtered) == 0:
        print(f"  No data for k>2, skipping model-algo grid heatmap")
        return

    if 'Algorithm' not in df_filtered.columns:
        print(f"  Missing Algorithm column for model-algo grid heatmap")
        return

    k_values = sorted(df_filtered['k'].unique())
    # Order algorithms as: median, bsah, sah
    algorithm_order = ['median', 'bsah', 'sah']
    algorithms = [a for a in algorithm_order if a in df_filtered['Algorithm'].unique()]

    # Get models sorted by polygon count
    model_order = (df_filtered.groupby('Model')['polygon_count']
                   .first().sort_values().index.tolist())

    # Create separate figures for k-way and collapsed
    if 'type' in df_filtered.columns:
        types = sorted(df_filtered['type'].unique())

        for algo_type in types:
            type_df = df_filtered[df_filtered['type'] == algo_type]

            fig, axes = plt.subplots(1, len(k_values), figsize=(6*len(k_values), 6))
            if len(k_values) == 1:
                axes = [axes]

            # Calculate global min/max for consistent color scale
            all_speedups = type_df['Speedup'].values
            data_min = np.percentile(all_speedups, 2)  # Use percentile to avoid outliers
            data_max = np.percentile(all_speedups, 98)

            # Extend range slightly for better contrast
            vmin = max(0.5, min(data_min - 0.05, 0.85))
            vmax = min(1.5, max(data_max + 0.05, 1.15))

            for idx, k_val in enumerate(k_values):
                k_data = type_df[type_df['k'] == k_val]

                # Pivot: models as rows, algorithms as columns
                pivot_data = k_data.groupby(['Model', 'Algorithm'])['Speedup'].mean().reset_index()
                heatmap_data = pivot_data.pivot(index='Model', columns='Algorithm', values='Speedup')

                # Reorder rows and columns
                heatmap_data = heatmap_data.reindex(index=model_order, columns=algorithms)

                # Capitalize model names
                heatmap_data.index = [capitalize_model_name(name) for name in heatmap_data.index]

                # Create heatmap without individual colorbar (except last one)
                cbar = (idx == len(k_values) - 1)
                cbar_kws = {'label': 'Speedup Factor'} if cbar else None

                sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                            center=1.0, vmin=vmin, vmax=vmax,
                            cbar=cbar, cbar_kws=cbar_kws,
                            linewidths=1, linecolor='gray', ax=axes[idx])

                axes[idx].set_xlabel(f'Algorithm (k={k_val})', fontsize=11)
                if idx == 0:
                    axes[idx].set_ylabel('Model', fontsize=11)
                else:
                    axes[idx].set_ylabel('')

            fig.tight_layout()

            # Save
            type_output = output_path.parent / f"{output_path.stem}_grid_{algo_type}{output_path.suffix}"
            type_output.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(type_output, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {type_output}")

        # Create combined (averaged over types)
        fig, axes = plt.subplots(1, len(k_values), figsize=(6*len(k_values), 6))
        if len(k_values) == 1:
            axes = [axes]

        # Calculate global min/max for consistent color scale
        all_speedups = df_filtered['Speedup'].values
        data_min = np.percentile(all_speedups, 2)
        data_max = np.percentile(all_speedups, 98)
        vmin = max(0.5, min(data_min - 0.05, 0.85))
        vmax = min(1.5, max(data_max + 0.05, 1.15))

        for idx, k_val in enumerate(k_values):
            k_data = df_filtered[df_filtered['k'] == k_val]

            # Average over types
            pivot_data = k_data.groupby(['Model', 'Algorithm'])['Speedup'].mean().reset_index()
            heatmap_data = pivot_data.pivot(index='Model', columns='Algorithm', values='Speedup')

            # Reorder
            heatmap_data = heatmap_data.reindex(index=model_order, columns=algorithms)

            # Capitalize model names
            heatmap_data.index = [capitalize_model_name(name) for name in heatmap_data.index]

            # Create heatmap with colorbar only on last subplot
            cbar = (idx == len(k_values) - 1)
            cbar_kws = {'label': 'Speedup Factor'} if cbar else None

            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                        center=1.0, vmin=vmin, vmax=vmax,
                        cbar=cbar, cbar_kws=cbar_kws,
                        linewidths=1, linecolor='gray', ax=axes[idx])

            axes[idx].set_xlabel(f'Algorithm (k={k_val})', fontsize=11)
            if idx == 0:
                axes[idx].set_ylabel('Model', fontsize=11)
            else:
                axes[idx].set_ylabel('')

        fig.tight_layout()

        # Save combined
        combined_output = output_path.parent / f"{output_path.stem}_grid_combined{output_path.suffix}"
        combined_output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(combined_output, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {combined_output}")

    else:
        # No type column - just create single combined
        fig, axes = plt.subplots(1, len(k_values), figsize=(6*len(k_values), 6))
        if len(k_values) == 1:
            axes = [axes]

        # Calculate global min/max for consistent color scale
        all_speedups = df_filtered['Speedup'].values
        data_min = np.percentile(all_speedups, 2)
        data_max = np.percentile(all_speedups, 98)
        vmin = max(0.5, min(data_min - 0.05, 0.85))
        vmax = min(1.5, max(data_max + 0.05, 1.15))

        for idx, k_val in enumerate(k_values):
            k_data = df_filtered[df_filtered['k'] == k_val]

            pivot_data = k_data.groupby(['Model', 'Algorithm'])['Speedup'].mean().reset_index()
            heatmap_data = pivot_data.pivot(index='Model', columns='Algorithm', values='Speedup')

            heatmap_data = heatmap_data.reindex(index=model_order, columns=algorithms)
            heatmap_data.index = [capitalize_model_name(name) for name in heatmap_data.index]

            # Create heatmap with colorbar only on last subplot
            cbar = (idx == len(k_values) - 1)
            cbar_kws = {'label': 'Speedup Factor'} if cbar else None

            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                        center=1.0, vmin=vmin, vmax=vmax,
                        cbar=cbar, cbar_kws=cbar_kws,
                        linewidths=1, linecolor='gray', ax=axes[idx])

            axes[idx].set_xlabel(f'Algorithm (k={k_val})', fontsize=11)
            if idx == 0:
                axes[idx].set_ylabel('Model', fontsize=11)
            else:
                axes[idx].set_ylabel('')

        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")


def create_rq2_grouped_bar(df: pd.DataFrame, output_path: Path):
    """
    RQ2 Graph: Grouped bar chart comparing k-way vs collapsed.
    """
    if 'type' not in df.columns:
        print(f"  No 'type' column, skipping RQ2 grouped bar")
        return

    # Use the 'k=X Time (s)' column which has the actual times
    if 'k=X Time (s)' not in df.columns:
        print(f"  Cannot find 'k=X Time (s)' column, skipping RQ2 grouped bar")
        return

    time_col = 'k=X Time (s)'

    # Get data for plotting
    plot_data = df[['Model', 'Algorithm', 'k', 'type', time_col]].copy()

    models = sorted(plot_data['Model'].unique())
    k_values = sorted(plot_data['k'].unique())

    # Create subplots for each k value
    fig, axes = plt.subplots(1, len(k_values), figsize=(5*len(k_values), 6), sharey=True)
    if len(k_values) == 1:
        axes = [axes]

    for idx, k_val in enumerate(k_values):
        ax = axes[idx]
        k_data = plot_data[plot_data['k'] == k_val]

        # Pivot: models as index, type as columns
        pivot = k_data.groupby(['Model', 'type'])[time_col].mean().unstack()

        # Sort by polygon count if available
        if 'polygon_count' in df.columns:
            model_order = df.groupby('Model')['polygon_count'].first().sort_values().index
            pivot = pivot.reindex(model_order, fill_value=0)

        # Capitalize model names in index
        pivot.index = [capitalize_model_name(name) for name in pivot.index]

        pivot.plot(kind='bar', ax=ax, color=['#66B3FF', '#0066CC'], width=0.8)
        ax.set_title(f"k={k_val}", fontsize=12)
        ax.set_xlabel('Model', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Time (s)', fontsize=11)
        ax.legend(title='Type', fontsize=9, labels=['collapsed', 'k-way'])
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def create_scaling_plot(df: pd.DataFrame, output_path: Path, result_type: str):
    """
    Cross-cutting: Log-log plot of time vs polygon count.
    """
    # For static/dynamic results, use 'k=X Time (s)'
    # For comparison results, average k-way and collapsed times
    if 'k=X Time (s)' in df.columns:
        time_col = 'k=X Time (s)'
    elif 'k-way Time (s)' in df.columns and 'collapsed Time (s)' in df.columns:
        # For comparison: average the two
        df = df.copy()
        df['avg_time'] = (df['k-way Time (s)'] + df['collapsed Time (s)']) / 2
        time_col = 'avg_time'
    else:
        print(f"  Cannot find time column for scaling plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    k_values = sorted(df['k'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))

    for k_val, color in zip(k_values, colors):
        k_data = df[df['k'] == k_val]

        # Average over algorithms/types
        plot_data = k_data.groupby(['Model', 'polygon_count'])[time_col].mean().reset_index()
        plot_data = plot_data.sort_values('polygon_count')

        ax.plot(plot_data['polygon_count'], plot_data[time_col],
                marker='o', linewidth=2, markersize=8,
                label=f"k={k_val}", color=color)

    ax.set_xlabel('Polygon Count', fontsize=12)
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def create_extended_graphs(result_dir: Path, polygon_df: pd.DataFrame, testrun_dir: Path):
    """
    Create extended analysis graphs for research questions.
    """
    csv_file = result_dir / "statistical_results_table.csv"

    if not csv_file.exists():
        print(f"No results found in {result_dir}, skipping extended graphs")
        return

    # Load results
    results_df = pd.read_csv(csv_file)

    # Merge with polygon counts
    results_df = results_df.merge(
        polygon_df,
        left_on='Model',
        right_on='model_name',
        how='left'
    )

    results_df = results_df.dropna(subset=['polygon_count'])

    if len(results_df) == 0:
        print(f"  No valid data for extended graphs")
        return

    # Create extended directory
    extended_dir = testrun_dir / "graphs" / "extended" / result_dir.name
    extended_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Creating extended graphs...")

    # RQ1 graphs - only for static
    if result_dir.name == 'static':
        create_rq1_lineplot(results_df, extended_dir / "rq1_speedup_vs_k.png", result_dir.name)
        create_rq1_heatmap_model_algo_grid(results_df, extended_dir / "rq1_heatmap.png", result_dir.name)

    # RQ2 graphs - only for comparison
    if result_dir.name == 'comparison':
        if 'type' in results_df.columns:
            create_rq2_grouped_bar(results_df, extended_dir / "rq2_kway_vs_collapsed.png")

    # RQ3 graphs - only for dynamic (TODO: need raw data)
    if result_dir.name == 'dynamic':
        # create_rq3_stacked_bar(results_df, extended_dir / "rq3_stacked_bar.png")
        # create_rq3_scatter(results_df, extended_dir / "rq3_scatter.png")
        pass

    # Scaling plot for all types
    create_scaling_plot(results_df, extended_dir / "scaling_plot.png", result_dir.name)


def main():
    parser = argparse.ArgumentParser(
        description="Create speedup graphs from BVH analysis results"
    )
    parser.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Path to testrun directory (e.g., testruns_2026_02_04)"
    )

    args = parser.parse_args()

    testrun_dir = args.dir.resolve()

    if not testrun_dir.exists():
        print(f"Error: Directory not found: {testrun_dir}")
        return 1

    results_dir = testrun_dir / "results"

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print("Run bvh_analysis.py first to generate results")
        return 1

    # Load polygon counts
    example_dir = Path("example")
    try:
        polygon_df = load_polygon_counts(example_dir)
        print(f"Loaded polygon counts for {len(polygon_df)} models")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    print(f"\nProcessing results from: {results_dir}")

    # Process each result type
    result_types = ['static', 'dynamic', 'comparison']

    for result_type in result_types:
        result_dir = results_dir / result_type

        if not result_dir.exists():
            print(f"\n{result_type.capitalize()}: Directory not found, skipping")
            continue

        print(f"\n{result_type.capitalize()}:")
        process_result_directory(result_dir, polygon_df, testrun_dir)

        # Create extended graphs
        print(f"\n{result_type.capitalize()} - Extended Graphs:")
        create_extended_graphs(result_dir, polygon_df, testrun_dir)

    print("\n" + "="*60)
    print("Done!")
    print("="*60)

    return 0


if __name__ == "__main__":
    exit(main())
