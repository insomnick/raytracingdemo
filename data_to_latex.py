#!/usr/bin/env python3

import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def format_p_value_with_stars(p_val_str):
    """Format p-value strings with significance stars"""
    # Handle case where p-value might already have stars
    if '*' in str(p_val_str):
        return str(p_val_str)

    try:
        p_val = float(p_val_str)
        p_str = f"{p_val:.2e}"
        if p_val < 0.001:
            return f"{p_str}***"
        elif p_val < 0.01:
            return f"{p_str}**"
        elif p_val < 0.05:
            return f"{p_str}*"
        else:
            return p_str
    except (ValueError, TypeError):
        return str(p_val_str)


def process_csv_to_latex(csv_path, output_dir, table_config):
    """Process a single CSV file to LaTeX format"""
    print(f"Processing: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return

    if df.empty:
        print(f"Empty CSV file: {csv_path}")
        return

    # Apply table-specific formatting
    formatted_df = df.copy()

    # unite min and max speedup ci into one column if both exist - DO THIS FIRST
    if 'Speedup CI Min' in formatted_df.columns and 'Speedup CI Max' in formatted_df.columns:
        # Convert to float first to handle any string values
        formatted_df['Speedup CI Min'] = pd.to_numeric(formatted_df['Speedup CI Min'], errors='coerce')
        formatted_df['Speedup CI Max'] = pd.to_numeric(formatted_df['Speedup CI Max'], errors='coerce')

        formatted_df['Speedup CI'] = formatted_df.apply(
            lambda row: f"[{float(row['Speedup CI Min']):.3f}, {float(row['Speedup CI Max']):.3f}]"
            if pd.notnull(row['Speedup CI Min']) and pd.notnull(row['Speedup CI Max'])
            else '', axis=1)
        formatted_df = formatted_df.drop(columns=['Speedup CI Min', 'Speedup CI Max'])

    # Format p-values if they exist
    if 'p-value' in formatted_df.columns:
        formatted_df['p-value'] = formatted_df['p-value'].apply(format_p_value_with_stars)
    # rename p-value column to just 'p'
    if 'p-value' in formatted_df.columns:
        formatted_df = formatted_df.rename(columns={'p-value': 'p'})

    for col in formatted_df.columns:
        if 'Speedup' in col and col != 'Speedup CI':  # Don't format the CI column as it's already a string
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else x)

    # round time columns to 4 decimal places
    for col in formatted_df.columns:
        if 'Time' in col:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else x)

    # Escape % symbols for LaTeX in column names
    for col in formatted_df.columns:
        if '% Change' in col or 'Change' in col:
            formatted_df = formatted_df.rename(columns={col: col.replace('%', '\\%')})

    # Delete unwanted columns AFTER formatting
    # delete t-statistic column if it exists
    if 't-statistic' in formatted_df.columns:
        formatted_df = formatted_df.drop(columns=['t-statistic'])

    # delete Time columns (but keep formatted ones if needed)
    cols_to_drop = []
    for col in formatted_df.columns:
        if 'Time' in col:
            cols_to_drop.append(col)
    if cols_to_drop:
        formatted_df = formatted_df.drop(columns=cols_to_drop)

    # delete percentage change columns
    cols_to_drop = []
    for col in formatted_df.columns:
        if '\\% Change' in col or '% Change' in col:
            cols_to_drop.append(col)
    if cols_to_drop:
        formatted_df = formatted_df.drop(columns=cols_to_drop)

    # Rename only Speedup column for LaTeX
    if 'Speedup' in formatted_df.columns:
        formatted_df = formatted_df.rename(columns={'Speedup': r'Speedup $\mu$'})

    column_order = ['Model', 'Algorithm', 'k', 'Type', r'Speedup $\mu$', 'Speedup CI', 'p']

    # Reorder the DataFrame to match the desired column order
    available_columns = [col for col in column_order if col in formatted_df.columns]
    formatted_df = formatted_df[available_columns + [col for col in formatted_df.columns if col not in available_columns]]

    # Generate LaTeX
    latex_table = formatted_df.to_latex(
        index=False,
        escape=False,
        longtable=True,
        caption=table_config['caption'],
        label=table_config['label'],
        column_format=table_config.get('column_format', None)
    )

    # Save individual LaTeX file with folder prefix
    folder_name = csv_path.parent.name
    if folder_name != "results":
        latex_filename = f"{folder_name}_{csv_path.stem}.tex"
    else:
        latex_filename = f"{csv_path.stem}.tex"
    latex_path = output_dir / latex_filename

    with open(latex_path, 'w') as f:
        f.write(latex_table)

    print(f"LaTeX table saved to: {latex_path}")

    return {
        'name': csv_path.stem,
        'caption': table_config['caption'],
        'latex_content': latex_table,
        'section_title': table_config['section_title']
    }


def create_consolidated_latex_document(table_data, output_dir):
    latex_file = output_dir / "all_tables_consolidated.tex"

    with open(latex_file, 'w') as f:
        # Write LaTeX document header
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage[utf8]{inputenc}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage{longtable}\n")
        f.write("\\usepackage{geometry}\n")
        f.write("\\geometry{margin=0.8in}\n")
        f.write("\\usepackage{rotating}\n")
        f.write("\\usepackage{pdflscape}\n")
        f.write("\\begin{document}\n\n")
        f.write("\\title{BVH Analysis Results - All Tables}\n")
        f.write("\\author{Generated by data\\_to\\_latex.py}\n")
        f.write("\\date{\\today}\n")
        f.write("\\maketitle\n\n")
        f.write("\\tableofcontents\n")
        f.write("\\clearpage\n\n")

        # Write each table as a section
        for table_info in table_data:
            f.write(f"\\section{{{table_info['section_title']}}}\n")
            f.write(table_info['latex_content'])
            f.write("\n\\clearpage\n\n")

        f.write("\\end{document}\n")

    print(f"Consolidated LaTeX document saved to: {latex_file}")


def get_polygon_count_label(model_name):
    # Mapping of model names to polygon counts (approximate values)
    model_info = {
        'armadillo': (99976, 'Armadillo (100k)'),
        'bunny': (69451, 'Stanford Bunny (69k)'),
        'stanford-bunny': (69451, 'Stanford Bunny (69k)'),
        'suzanne': (968, 'Suzanne (1k)'),
        'teapot': (6320, 'Teapot (6k)')
    }

    # Clean the model name
    clean_name = model_name.lower().strip()
    for key in model_info:
        if key in clean_name:
            return model_info[key]

    # Default fallback
    return (0, model_name)


def create_speedup_graphs_for_type(results_dir, output_dir, graph_type="comparison"):
    # Find CSV files with speedup data based on type
    csv_files = []

    if graph_type == "dynamic":
        # Look specifically for files in dynamic directories
        dynamic_files = list(results_dir.rglob("dynamic/*.csv"))
        if dynamic_files:
            csv_files = dynamic_files
            print(f"Using dynamic files: {[f.name for f in dynamic_files]}")
    elif graph_type == "static":
        # Look specifically for files in static directories
        static_files = list(results_dir.rglob("static/*.csv"))
        if static_files:
            csv_files = static_files
            print(f"Using static files: {[f.name for f in static_files]}")
    else:
        # Try to find comparison files first (these typically have the cleanest speedup data)
        comparison_files = list(results_dir.rglob("comparison/*.csv"))
        if comparison_files:
            csv_files = comparison_files
            print(f"Using comparison files: {[f.name for f in comparison_files]}")
        else:
            # Fall back to files in dynamic directories
            dynamic_files = list(results_dir.rglob("dynamic/*.csv"))
            if dynamic_files:
                csv_files = dynamic_files
                print(f"Using dynamic files: {[f.name for f in dynamic_files]}")
            else:
                # Last resort - any table files
                csv_files = list(results_dir.rglob("*table*.csv"))
                print(f"Using any table files: {[f.name for f in csv_files]}")

    if not csv_files:
        print("No CSV files found for graph generation!")
        return

    # Read and combine all data
    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'Model' in df.columns and 'Algorithm' in df.columns and 'Speedup' in df.columns:
                # Add source file info
                df['source_file'] = csv_file.name
                all_data.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue

    if not all_data:
        print("No suitable data found for graph generation!")
        return

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)

    # Remove rows with missing speedup values
    combined_df = combined_df.dropna(subset=['Speedup'])

    # Add polygon count information
    combined_df['polygon_count'] = combined_df['Model'].apply(lambda x: get_polygon_count_label(x)[0])
    combined_df['model_label'] = combined_df['Model'].apply(lambda x: get_polygon_count_label(x)[1])

    # Filter out rows with 0 polygon count (unknown models)
    combined_df = combined_df[combined_df['polygon_count'] > 0]

    if combined_df.empty:
        print("No valid model data found for graphing!")
        return

    # Handle duplicate entries by averaging speedup values for same model-algorithm-k-type combinations
    grouping_cols = ['Model', 'Algorithm', 'polygon_count', 'model_label']
    if 'k' in combined_df.columns:
        grouping_cols.append('k')
    if 'Type' in combined_df.columns:
        grouping_cols.append('Type')

    # Remove duplicates but don't aggregate - we want individual data points with their CI values
    combined_df = combined_df.drop_duplicates(subset=grouping_cols + ['Speedup'])

    # Get unique combinations of k and Algorithm Type
    if 'k' in combined_df.columns:
        k_values = sorted([k for k in combined_df['k'].dropna().unique() if pd.notna(k)])
    else:
        k_values = [None]

    if 'Type' in combined_df.columns:
        algorithm_types = sorted([t for t in combined_df['Type'].dropna().unique() if pd.notna(t)])
    else:
        algorithm_types = [None]

    # Create graphs directory
    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(exist_ok=True)

    # Create one graph per k and algorithm type combination
    for k_val in k_values:
        for algo_type in algorithm_types:
            # Filter data for this combination
            if k_val is not None and algo_type is not None:
                filtered_df = combined_df[
                    (pd.notna(combined_df['k'])) & (combined_df['k'] == k_val) &
                    (pd.notna(combined_df['Type'])) & (combined_df['Type'] == algo_type)
                ]
                graph_title = f"Speedup Analysis - k={k_val}, Type={algo_type}"
                filename = f"{graph_type}_speedup_k{k_val}_{algo_type.replace(' ', '_').replace('-', '_')}.png"
            elif k_val is not None:
                filtered_df = combined_df[(pd.notna(combined_df['k'])) & (combined_df['k'] == k_val)]
                graph_title = f"Speedup Analysis - k={k_val}"
                filename = f"{graph_type}_speedup_k{k_val}.png"
            elif algo_type is not None:
                filtered_df = combined_df[(pd.notna(combined_df['Type'])) & (combined_df['Type'] == algo_type)]
                graph_title = f"Speedup Analysis - Type={algo_type}"
                filename = f"{graph_type}_speedup_{algo_type.replace(' ', '_').replace('-', '_')}.png"
            else:
                filtered_df = combined_df.copy()
                graph_title = "Speedup Analysis"
                filename = f"{graph_type}_speedup_all.png"

            if filtered_df.empty:
                continue

            # Create the plot
            plt.figure(figsize=(12, 8))

            # Get unique algorithms for this filtered data
            algorithms = sorted(filtered_df['Algorithm'].unique())
            colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(algorithms)))

            for i, algorithm in enumerate(algorithms):
                algo_data = filtered_df[filtered_df['Algorithm'] == algorithm]

                if algo_data.empty:
                    continue

                # Sort by polygon count for proper plotting
                algo_data = algo_data.sort_values('polygon_count')

                # Calculate positions with small offset for multiple algorithms to avoid overlap
                algo_count = len(algorithms)

                for j, (count, count_data) in enumerate(algo_data.groupby('polygon_count')):
                    # Get all data points for this polygon count and algorithm
                    for idx, row in count_data.iterrows():
                        # Position points with small offset for different algorithms
                        base_position = j
                        offset = (i - (algo_count - 1) / 2) * 0.1  # Small offset to avoid overlap
                        x_pos = base_position + offset

                        # Get speedup value and CI bounds
                        speedup = row['Speedup']

                        # Check if CI data exists in DataFrame columns
                        if 'Speedup CI Min' in count_data.columns and 'Speedup CI Max' in count_data.columns:
                            ci_min = row['Speedup CI Min']
                            ci_max = row['Speedup CI Max']

                            # Create error bars if CI data is valid
                            if pd.notna(ci_min) and pd.notna(ci_max) and ci_min <= speedup <= ci_max:
                                # Plot point with error bar
                                plt.errorbar(x_pos, speedup,
                                           yerr=[[speedup - ci_min], [ci_max - speedup]],
                                           fmt='o', color=colors[i], markersize=8,
                                           capsize=4, capthick=2, elinewidth=2,
                                           label=algorithm if j == 0 and idx == count_data.index[0] else "",
                                           alpha=0.8)
                            else:
                                # Plot point without error bar if CI data is invalid
                                plt.scatter(x_pos, speedup, color=colors[i], s=60,
                                          label=algorithm if j == 0 and idx == count_data.index[0] else "",
                                          alpha=0.8, edgecolor='white', linewidth=1)
                        else:
                            # Plot point without error bar if no CI columns
                            plt.scatter(x_pos, speedup, color=colors[i], s=60,
                                      label=algorithm if j == 0 and idx == count_data.index[0] else "",
                                      alpha=0.8, edgecolor='white', linewidth=1)

            plt.xlabel('Model (Polygon Count)', fontsize=12)
            plt.ylabel('Speedup', fontsize=12)
            plt.title(graph_title, fontsize=14, fontweight='bold')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)

            # Set x-tick labels for scatter plot
            unique_counts = sorted(filtered_df['polygon_count'].unique())
            unique_labels = []
            for count in unique_counts:
                label_row = filtered_df[filtered_df['polygon_count'] == count].iloc[0]
                unique_labels.append(label_row['model_label'])

            plt.xticks(range(len(unique_labels)), unique_labels, rotation=45, ha='right')

            # Ensure y=1.0 is always visible while keeping all data and error bars well visible
            # Calculate the actual range including error bars by examining the data
            all_speedup_values = []
            all_ci_min_values = []
            all_ci_max_values = []

            for _, row in filtered_df.iterrows():
                all_speedup_values.append(row['Speedup'])
                if 'Speedup CI Min' in row and pd.notna(row['Speedup CI Min']):
                    all_ci_min_values.append(row['Speedup CI Min'])
                if 'Speedup CI Max' in row and pd.notna(row['Speedup CI Max']):
                    all_ci_max_values.append(row['Speedup CI Max'])

            # Find actual data range including error bars
            all_values = all_speedup_values + all_ci_min_values + all_ci_max_values
            if all_values:
                actual_min = min(all_values)
                actual_max = max(all_values)

                # Ensure y=1.0 is included in visible range
                range_min = min(actual_min, 1.0)
                range_max = max(actual_max, 1.0)
                data_range = range_max - range_min

                # Add 15% padding for good visibility
                padding = data_range * 0.15

                # Set limits to show y=1.0 and all data with padding
                bottom_limit = max(0, range_min - padding)  # Don't go below 0 for speedup
                top_limit = range_max + padding

                plt.ylim(bottom=bottom_limit, top=top_limit)
            else:
                # Fallback if no data - ensure y=1.0 is visible
                plt.ylim(bottom=0.5, top=1.5)

            # Adjust layout and save
            plt.tight_layout()
            graph_path = graphs_dir / filename
            plt.savefig(graph_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Graph saved to: {graph_path}")

    print(f"All speedup graphs saved to: {graphs_dir}")


def create_traversal_vs_construction_graph(results_dir, output_dir):
    """Create traversal vs construction time graphs - Y=Traversal Time, X=Construction Time, one line per k, one graph per scene"""
    print("Creating traversal vs construction time graphs...")

    dynamic_file = results_dir / "dynamic" / "statistical_results_table.csv"
    static_file = results_dir / "static" / "statistical_results_table.csv"

    if not dynamic_file.exists() or not static_file.exists():
        print("Required dynamic and static CSV files not found!")
        return

    try:
        dynamic_df = pd.read_csv(dynamic_file)
        static_df = pd.read_csv(static_file)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    # Merge dynamic and static data
    merged_df = dynamic_df.merge(
        static_df,
        on=['Model', 'Algorithm', 'k', 'Type'],
        suffixes=('_dynamic', '_static')
    )

    # Calculate construction time and traversal time
    merged_df['construction_time'] = merged_df['k=X Time (s)_dynamic'] - merged_df['k=X Time (s)_static']
    merged_df['traversal_time'] = merged_df['k=X Time (s)_static']

    # Also process k=2 data separately and add to dataset
    k2_data = []
    for _, row in merged_df.iterrows():
        k2_construction = row['k=2 Time (s)_dynamic'] - row['k=2 Time (s)_static']
        k2_traversal = row['k=2 Time (s)_static']

        k2_row = {
            'Model': row['Model'],
            'Algorithm': row['Algorithm'],
            'k': 2,
            'Type': row['Type'],
            'construction_time': k2_construction,
            'traversal_time': k2_traversal
        }
        k2_data.append(k2_row)

    k2_df = pd.DataFrame(k2_data)

    # Combine k=X and k=2 data
    time_data = merged_df[['Model', 'Algorithm', 'k', 'Type', 'construction_time', 'traversal_time']].copy()
    combined_data = pd.concat([time_data, k2_df], ignore_index=True)

    # Average k-way and collapsed values for each Algorithm
    averaged_data = combined_data.groupby(['Model', 'Algorithm', 'k'])[['construction_time', 'traversal_time']].mean().reset_index()

    # Debug: Print data availability
    print("Debug: Available data overview:")
    print(f"Models: {sorted(averaged_data['Model'].unique())}")
    print(f"Algorithms: {sorted(averaged_data['Algorithm'].unique())}")
    print(f"K values: {sorted(averaged_data['k'].unique())}")

    # Get unique models and algorithms
    models = sorted(averaged_data['Model'].unique())
    algorithms = sorted(averaged_data['Algorithm'].unique())

    # Create subplot figure - one subplot per model (scene)
    num_models = len(models)
    if num_models <= 2:
        fig, axes = plt.subplots(1, num_models, figsize=(8 * num_models, 6))
        if num_models == 1:
            axes = [axes]
    else:
        # If more than 2 models, use 2x2 layout
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

    # Use distinct colors and markers for different k values
    k_values = sorted(averaged_data['k'].unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Blue, Orange, Green, Red, Purple
    markers = ['o', 's', '^', 'D', 'v']  # Circle, Square, Triangle, Diamond, Triangle-down

    for i, model in enumerate(models):
        if i >= len(axes):
            break

        ax = axes[i]
        model_data = averaged_data[averaged_data['Model'] == model]

        print(f"Debug: Model {model} has {len(model_data)} data points")

        # Plot one line per k value
        for j, k_val in enumerate(k_values):
            k_data = model_data[model_data['k'] == k_val]

            if k_data.empty:
                print(f"Debug: No data for {model} + k={k_val}")
                continue

            # Sort by algorithm for consistent line plotting
            k_data = k_data.sort_values('Algorithm')

            # Extract x (construction time) and y (traversal time) data
            x_data = k_data['construction_time'].tolist()
            y_data = k_data['traversal_time'].tolist()
            algorithm_labels = k_data['Algorithm'].tolist()

            if x_data and y_data:
                # Plot scatter points for algorithms for this k value
                ax.scatter(x_data, y_data,
                          color=colors[j % len(colors)],
                          marker=markers[j % len(markers)],
                          label=f'k={k_val}',
                          s=80,
                          edgecolors='white',
                          linewidth=1,
                          alpha=0.8)

                # Add algorithm labels as annotations with smart positioning and increased distance
                for idx, (x, y, alg) in enumerate(zip(x_data, y_data, algorithm_labels)):
                    # Calculate smart offset positions with larger distance for better point visibility
                    angle = 2 * np.pi * idx / len(algorithm_labels)  # Distribute around circle
                    offset_x = 35 * np.cos(angle)  # Increased from 15 to 35 for better visibility
                    offset_y = 35 * np.sin(angle)  # Increased from 15 to 35 for better visibility

                    ax.annotate(alg, (x, y), xytext=(offset_x, offset_y),
                               textcoords='offset points', fontsize=8, alpha=0.9,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                        edgecolor=colors[j % len(colors)], alpha=0.9),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1',
                                             color=colors[j % len(colors)], alpha=0.7, lw=1.5))

                print(f"Debug: {model} k={k_val} has {len(x_data)} algorithm points")

        ax.set_xlabel('Construction Time (s)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Traversal Time (s)', fontsize=11, fontweight='bold')
        ax.set_title(f'{model}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')

        # Ensure both axes start at 0 and have equal scaling for better comparison
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Find the maximum range to make axes equal
        max_val = max(xlim[1], ylim[1])

        ax.set_xlim(left=0, right=max_val)
        ax.set_ylim(bottom=0, top=max_val)
        ax.set_aspect('equal', adjustable='box')

    # Hide unused subplots if less models than subplots
    for i in range(len(models), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Traversal Time vs Construction Time Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / "graphs" / "t-vs-c.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Traversal vs Construction graph saved to: {output_path}")
    print(f"Graph layout: {len(models)} model(s), x-axis = Construction Time, y-axis = Traversal Time, lines per k-value")


def create_traversal_vs_construction_graph_no_k16(results_dir, output_dir):
    """Create traversal vs construction time graphs without k=16 - Y=Traversal Time, X=Construction Time, one line per k, one graph per scene"""
    print("Creating traversal vs construction time graphs (without k=16)...")

    dynamic_file = results_dir / "dynamic" / "statistical_results_table.csv"
    static_file = results_dir / "static" / "statistical_results_table.csv"

    if not dynamic_file.exists() or not static_file.exists():
        print("Required dynamic and static CSV files not found!")
        return

    try:
        dynamic_df = pd.read_csv(dynamic_file)
        static_df = pd.read_csv(static_file)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    # Merge dynamic and static data
    merged_df = dynamic_df.merge(
        static_df,
        on=['Model', 'Algorithm', 'k', 'Type'],
        suffixes=('_dynamic', '_static')
    )

    # Filter out k=16 data
    merged_df = merged_df[merged_df['k'] != 16]

    # Calculate construction time and traversal time
    merged_df['construction_time'] = merged_df['k=X Time (s)_dynamic'] - merged_df['k=X Time (s)_static']
    merged_df['traversal_time'] = merged_df['k=X Time (s)_static']

    # Also process k=2 data separately and add to dataset
    k2_data = []
    for _, row in merged_df.iterrows():
        k2_construction = row['k=2 Time (s)_dynamic'] - row['k=2 Time (s)_static']
        k2_traversal = row['k=2 Time (s)_static']

        k2_row = {
            'Model': row['Model'],
            'Algorithm': row['Algorithm'],
            'k': 2,
            'Type': row['Type'],
            'construction_time': k2_construction,
            'traversal_time': k2_traversal
        }
        k2_data.append(k2_row)

    k2_df = pd.DataFrame(k2_data)

    # Combine k=X and k=2 data, filter out any remaining k=16
    time_data = merged_df[['Model', 'Algorithm', 'k', 'Type', 'construction_time', 'traversal_time']].copy()
    combined_data = pd.concat([time_data, k2_df], ignore_index=True)
    combined_data = combined_data[combined_data['k'] != 16]

    # Average k-way and collapsed values for each Algorithm
    averaged_data = combined_data.groupby(['Model', 'Algorithm', 'k'])[['construction_time', 'traversal_time']].mean().reset_index()

    # Debug: Print data availability
    print("Debug: Available data overview (without k=16):")
    print(f"Models: {sorted(averaged_data['Model'].unique())}")
    print(f"Algorithms: {sorted(averaged_data['Algorithm'].unique())}")
    print(f"K values: {sorted(averaged_data['k'].unique())}")

    # Get unique models and algorithms
    models = sorted(averaged_data['Model'].unique())
    algorithms = sorted(averaged_data['Algorithm'].unique())

    # Create subplot figure - one subplot per model (scene)
    num_models = len(models)
    if num_models <= 2:
        fig, axes = plt.subplots(1, num_models, figsize=(8 * num_models, 6))
        if num_models == 1:
            axes = [axes]
    else:
        # If more than 2 models, use 2x2 layout
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

    # Use distinct colors and markers for different k values (without k=16)
    k_values = sorted([k for k in averaged_data['k'].unique() if k != 16])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green for k=2,4,8
    markers = ['o', 's', '^']  # Circle, Square, Triangle for k=2,4,8

    for i, model in enumerate(models):
        if i >= len(axes):
            break

        ax = axes[i]
        model_data = averaged_data[averaged_data['Model'] == model]

        print(f"Debug: Model {model} has {len(model_data)} data points (without k=16)")

        # Plot one line per k value
        for j, k_val in enumerate(k_values):
            k_data = model_data[model_data['k'] == k_val]

            if k_data.empty:
                print(f"Debug: No data for {model} + k={k_val}")
                continue

            # Sort by algorithm for consistent line plotting
            k_data = k_data.sort_values('Algorithm')

            # Extract x (construction time) and y (traversal time) data
            x_data = k_data['construction_time'].tolist()
            y_data = k_data['traversal_time'].tolist()
            algorithm_labels = k_data['Algorithm'].tolist()

            if x_data and y_data:
                # Plot scatter points for algorithms for this k value
                ax.scatter(x_data, y_data,
                          color=colors[j % len(colors)],
                          marker=markers[j % len(markers)],
                          label=f'k={k_val}',
                          s=80,
                          edgecolors='white',
                          linewidth=1,
                          alpha=0.8)

                # Add algorithm labels as annotations with smart positioning
                for idx, (x, y, alg) in enumerate(zip(x_data, y_data, algorithm_labels)):
                    # Calculate smart offset positions with larger distance for better point visibility
                    angle = 2 * np.pi * idx / len(algorithm_labels)  # Distribute around circle
                    offset_x = 35 * np.cos(angle)  # Increased from 15 to 35 for better visibility
                    offset_y = 35 * np.sin(angle)  # Increased from 15 to 35 for better visibility

                    ax.annotate(alg, (x, y), xytext=(offset_x, offset_y),
                               textcoords='offset points', fontsize=8, alpha=0.9,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                        edgecolor=colors[j % len(colors)], alpha=0.9),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1',
                                             color=colors[j % len(colors)], alpha=0.7, lw=1.5))

                print(f"Debug: {model} k={k_val} has {len(x_data)} algorithm points")

        ax.set_xlabel('Construction Time (s)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Traversal Time (s)', fontsize=11, fontweight='bold')
        ax.set_title(f'{model} (k=2,4,8)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')

        # Ensure both axes start at 0 and have equal scaling for better comparison
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Find the maximum range to make axes equal
        max_val = max(xlim[1], ylim[1])

        ax.set_xlim(left=0, right=max_val)
        ax.set_ylim(bottom=0, top=max_val)
        ax.set_aspect('equal', adjustable='box')

    # Hide unused subplots if less models than subplots
    for i in range(len(models), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Traversal Time vs Construction Time Analysis (k=2,4,8)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / "graphs" / "t-vs-c-no-k16.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Traversal vs Construction graph (without k=16) saved to: {output_path}")
    print(f"Graph layout: {len(models)} model(s), k-values: {k_values}, x-axis = Construction Time, y-axis = Traversal Time")


def create_combined_speedup_graphs(results_dir, output_dir, graph_type="comparison"):
    """Create combined speedup graphs that average k-way and collapsed data with extended CI bounds"""
    # Find CSV files with speedup data based on type
    csv_files = []

    if graph_type == "dynamic":
        dynamic_files = list(results_dir.rglob("dynamic/*.csv"))
        if dynamic_files:
            csv_files = dynamic_files
            print(f"Using dynamic files: {[f.name for f in dynamic_files]}")
    elif graph_type == "static":
        static_files = list(results_dir.rglob("static/*.csv"))
        if static_files:
            csv_files = static_files
            print(f"Using static files: {[f.name for f in static_files]}")
    else:
        comparison_files = list(results_dir.rglob("comparison/*.csv"))
        if comparison_files:
            csv_files = comparison_files
            print(f"Using comparison files: {[f.name for f in comparison_files]}")
        else:
            dynamic_files = list(results_dir.rglob("dynamic/*.csv"))
            if dynamic_files:
                csv_files = dynamic_files
                print(f"Using dynamic files: {[f.name for f in dynamic_files]}")
            else:
                csv_files = list(results_dir.rglob("*table*.csv"))
                print(f"Using any table files: {[f.name for f in csv_files]}")

    if not csv_files:
        print("No CSV files found for combined graph generation!")
        return

    # Read and combine all data
    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'Model' in df.columns and 'Algorithm' in df.columns and 'Speedup' in df.columns:
                df['source_file'] = csv_file.name
                all_data.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue

    if not all_data:
        print("No suitable data found for combined graph generation!")
        return

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.dropna(subset=['Speedup'])

    # Add polygon count information
    combined_df['polygon_count'] = combined_df['Model'].apply(lambda x: get_polygon_count_label(x)[0])
    combined_df['model_label'] = combined_df['Model'].apply(lambda x: get_polygon_count_label(x)[1])
    combined_df = combined_df[combined_df['polygon_count'] > 0]

    if combined_df.empty:
        print("No valid model data found for combined graphing!")
        return

    # Filter for k-way and collapsed types only
    if 'Type' in combined_df.columns:
        combined_df = combined_df[combined_df['Type'].isin(['k-way', 'collapsed'])]

    if combined_df.empty:
        print("No k-way or collapsed data found!")
        return

    # Group by Model, Algorithm, k and combine k-way and collapsed
    grouping_cols = ['Model', 'Algorithm', 'polygon_count', 'model_label']
    if 'k' in combined_df.columns:
        grouping_cols.append('k')

    # Aggregate k-way and collapsed data
    def combine_types(group):
        if len(group) == 0:
            return None

        # Calculate average speedup between k-way and collapsed
        avg_speedup = group['Speedup'].mean()

        # Use minimum CI Min and maximum CI Max for extended bounds
        min_ci_min = group['Speedup CI Min'].min() if 'Speedup CI Min' in group.columns else None
        max_ci_max = group['Speedup CI Max'].max() if 'Speedup CI Max' in group.columns else None

        result = group.iloc[0].copy()
        result['Speedup'] = avg_speedup
        if pd.notna(min_ci_min):
            result['Speedup CI Min'] = min_ci_min
        if pd.notna(max_ci_max):
            result['Speedup CI Max'] = max_ci_max

        return result

    combined_summary = combined_df.groupby(grouping_cols).apply(combine_types).reset_index(drop=True)
    combined_summary = combined_summary.dropna()

    # Get unique combinations of k
    if 'k' in combined_summary.columns:
        k_values = sorted([k for k in combined_summary['k'].dropna().unique() if pd.notna(k)])
    else:
        k_values = [None]

    # Create graphs directory
    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(exist_ok=True)

    # Create one graph per k value
    for k_val in k_values:
        # Filter data for this k value
        if k_val is not None:
            filtered_df = combined_summary[(pd.notna(combined_summary['k'])) & (combined_summary['k'] == k_val)]
            graph_title = f"Speedup Analysis - k={k_val}"
            filename = f"{graph_type}_combined_speedup_k{k_val}.png"
        else:
            filtered_df = combined_summary.copy()
            graph_title = "Speedup Analysis"
            filename = f"{graph_type}_combined_speedup_all.png"

        if filtered_df.empty:
            continue

        # Create the plot
        plt.figure(figsize=(12, 8))

        # Get unique algorithms for this filtered data
        algorithms = sorted(filtered_df['Algorithm'].unique())
        colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(algorithms)))

        for i, algorithm in enumerate(algorithms):
            algo_data = filtered_df[filtered_df['Algorithm'] == algorithm]

            if algo_data.empty:
                continue

            # Sort by polygon count for proper plotting
            algo_data = algo_data.sort_values('polygon_count')
            algo_count = len(algorithms)

            for j, (count, count_data) in enumerate(algo_data.groupby('polygon_count')):
                for idx, row in count_data.iterrows():
                    base_position = j
                    offset = (i - (algo_count - 1) / 2) * 0.1
                    x_pos = base_position + offset

                    speedup = row['Speedup']

                    # Check if CI data exists in DataFrame columns
                    if 'Speedup CI Min' in count_data.columns and 'Speedup CI Max' in count_data.columns:
                        ci_min = row['Speedup CI Min']
                        ci_max = row['Speedup CI Max']

                        if pd.notna(ci_min) and pd.notna(ci_max) and ci_min <= speedup <= ci_max:
                            plt.errorbar(x_pos, speedup,
                                       yerr=[[speedup - ci_min], [ci_max - speedup]],
                                       fmt='o', color=colors[i], markersize=8,
                                       capsize=4, capthick=2, elinewidth=2,
                                       label=algorithm if j == 0 and idx == count_data.index[0] else "",
                                       alpha=0.8)
                        else:
                            plt.scatter(x_pos, speedup, color=colors[i], s=60,
                                      label=algorithm if j == 0 and idx == count_data.index[0] else "",
                                      alpha=0.8, edgecolor='white', linewidth=1)
                    else:
                        plt.scatter(x_pos, speedup, color=colors[i], s=60,
                                  label=algorithm if j == 0 and idx == count_data.index[0] else "",
                                  alpha=0.8, edgecolor='white', linewidth=1)

        plt.xlabel('Model (Polygon Count)', fontsize=12)
        plt.ylabel('Speedup', fontsize=12)
        plt.title(graph_title, fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        # Set x-tick labels for scatter plot
        unique_counts = sorted(filtered_df['polygon_count'].unique())
        unique_labels = []
        for count in unique_counts:
            label_row = filtered_df[filtered_df['polygon_count'] == count].iloc[0]
            unique_labels.append(label_row['model_label'])

        plt.xticks(range(len(unique_labels)), unique_labels, rotation=45, ha='right')

        # Ensure y=1.0 is always visible while keeping all data and error bars well visible
        # Calculate the actual range including error bars by examining the data
        all_speedup_values = []
        all_ci_min_values = []
        all_ci_max_values = []

        for _, row in filtered_df.iterrows():
            all_speedup_values.append(row['Speedup'])
            if 'Speedup CI Min' in row and pd.notna(row['Speedup CI Min']):
                all_ci_min_values.append(row['Speedup CI Min'])
            if 'Speedup CI Max' in row and pd.notna(row['Speedup CI Max']):
                all_ci_max_values.append(row['Speedup CI Max'])

        # Find actual data range including error bars
        all_values = all_speedup_values + all_ci_min_values + all_ci_max_values
        if all_values:
            actual_min = min(all_values)
            actual_max = max(all_values)

            # Ensure y=1.0 is included in visible range
            range_min = min(actual_min, 1.0)
            range_max = max(actual_max, 1.0)
            data_range = range_max - range_min

            # Add 15% padding for good visibility
            padding = data_range * 0.15

            # Set limits to show y=1.0 and all data with padding
            bottom_limit = max(0, range_min - padding)  # Don't go below 0 for speedup
            top_limit = range_max + padding

            plt.ylim(bottom=bottom_limit, top=top_limit)
        else:
            # Fallback if no data - ensure y=1.0 is visible
            plt.ylim(bottom=0.5, top=1.5)

        # Adjust layout and save
        plt.tight_layout()
        graph_path = graphs_dir / filename
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Combined graph saved to: {graph_path}")

    print(f"All combined speedup graphs saved to: {graphs_dir}")


def create_averaged_models_speedup_graph(results_dir, output_dir, graph_type="comparison"):
    """Create speedup graphs using average values across all models"""
    print(f"Creating averaged models speedup graph for {graph_type}...")

    # Find CSV files with speedup data based on type
    csv_files = []

    if graph_type == "dynamic":
        dynamic_files = list(results_dir.rglob("dynamic/*.csv"))
        if dynamic_files:
            csv_files = dynamic_files
            print(f"Using dynamic files: {[f.name for f in dynamic_files]}")
    elif graph_type == "static":
        static_files = list(results_dir.rglob("static/*.csv"))
        if static_files:
            csv_files = static_files
            print(f"Using static files: {[f.name for f in static_files]}")
    else:
        comparison_files = list(results_dir.rglob("comparison/*.csv"))
        if comparison_files:
            csv_files = comparison_files
            print(f"Using comparison files: {[f.name for f in comparison_files]}")
        else:
            dynamic_files = list(results_dir.rglob("dynamic/*.csv"))
            if dynamic_files:
                csv_files = dynamic_files
                print(f"Using dynamic files: {[f.name for f in dynamic_files]}")
            else:
                csv_files = list(results_dir.rglob("*table*.csv"))
                print(f"Using any table files: {[f.name for f in csv_files]}")

    if not csv_files:
        print("No CSV files found for averaged graph generation!")
        return

    # Read and combine all data
    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'Model' in df.columns and 'Algorithm' in df.columns and 'Speedup' in df.columns:
                df['source_file'] = csv_file.name
                all_data.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue

    if not all_data:
        print("No suitable data found for averaged graph generation!")
        return

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.dropna(subset=['Speedup'])

    # Add polygon count information for filtering
    combined_df['polygon_count'] = combined_df['Model'].apply(lambda x: get_polygon_count_label(x)[0])
    combined_df = combined_df[combined_df['polygon_count'] > 0]

    if combined_df.empty:
        print("No valid model data found for averaged graphing!")
        return

    # Group by Algorithm, k, and Type to calculate averages across models
    grouping_cols = ['Algorithm']
    if 'k' in combined_df.columns:
        grouping_cols.append('k')
    if 'Type' in combined_df.columns:
        grouping_cols.append('Type')

    # Calculate average speedup and CI bounds across all models
    def calculate_averages(group):
        result = {
            'Speedup': group['Speedup'].mean(),
            'Model_Count': len(group),  # Track how many models were averaged
        }

        # Calculate averaged CI bounds if available
        if 'Speedup CI Min' in group.columns:
            result['Speedup CI Min'] = group['Speedup CI Min'].mean()
        if 'Speedup CI Max' in group.columns:
            result['Speedup CI Max'] = group['Speedup CI Max'].mean()

        return pd.Series(result)

    averaged_df = combined_df.groupby(grouping_cols).apply(calculate_averages).reset_index()

    print(f"Debug: Averaged data overview:")
    print(f"Algorithms: {sorted(averaged_df['Algorithm'].unique())}")
    if 'k' in averaged_df.columns:
        print(f"K values: {sorted(averaged_df['k'].unique())}")
    if 'Type' in averaged_df.columns:
        print(f"Types: {sorted(averaged_df['Type'].unique())}")
    print(f"Average model count per group: {averaged_df['Model_Count'].mean():.1f}")

    # Get unique combinations for graphing
    if 'k' in averaged_df.columns:
        k_values = sorted([k for k in averaged_df['k'].dropna().unique() if pd.notna(k)])
    else:
        k_values = [None]

    if 'Type' in averaged_df.columns:
        algorithm_types = sorted([t for t in averaged_df['Type'].dropna().unique() if pd.notna(t)])
    else:
        algorithm_types = [None]

    # Create graphs directory
    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(exist_ok=True)

    # Create one graph per k and algorithm type combination
    for k_val in k_values:
        for algo_type in algorithm_types:
            # Filter data for this combination
            if k_val is not None and algo_type is not None:
                filtered_df = averaged_df[
                    (pd.notna(averaged_df['k'])) & (averaged_df['k'] == k_val) &
                    (pd.notna(averaged_df['Type'])) & (averaged_df['Type'] == algo_type)
                ]
                graph_title = f"Average Speedup Analysis - k={k_val}, Type={algo_type}"
                filename = f"{graph_type}_averaged_speedup_k{k_val}_{algo_type.replace(' ', '_').replace('-', '_')}.png"
            elif k_val is not None:
                filtered_df = averaged_df[(pd.notna(averaged_df['k'])) & (averaged_df['k'] == k_val)]
                graph_title = f"Average Speedup Analysis - k={k_val}"
                filename = f"{graph_type}_averaged_speedup_k{k_val}.png"
            elif algo_type is not None:
                filtered_df = averaged_df[(pd.notna(averaged_df['Type'])) & (averaged_df['Type'] == algo_type)]
                graph_title = f"Average Speedup Analysis - Type={algo_type}"
                filename = f"{graph_type}_averaged_speedup_{algo_type.replace(' ', '_').replace('-', '_')}.png"
            else:
                filtered_df = averaged_df.copy()
                graph_title = "Average Speedup Analysis"
                filename = f"{graph_type}_averaged_speedup_all.png"

            if filtered_df.empty:
                continue

            # Create the plot
            plt.figure(figsize=(10, 6))

            # Get unique algorithms for this filtered data
            algorithms = sorted(filtered_df['Algorithm'].unique())
            colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(algorithms)))

            x_positions = range(len(algorithms))

            for i, algorithm in enumerate(algorithms):
                algo_data = filtered_df[filtered_df['Algorithm'] == algorithm].iloc[0]

                speedup = algo_data['Speedup']

                # Plot with error bars if CI data is available
                if 'Speedup CI Min' in algo_data and 'Speedup CI Max' in algo_data:
                    ci_min = algo_data['Speedup CI Min']
                    ci_max = algo_data['Speedup CI Max']

                    if pd.notna(ci_min) and pd.notna(ci_max):
                        plt.errorbar(x_positions[i], speedup,
                                   yerr=[[speedup - ci_min], [ci_max - speedup]],
                                   fmt='o', color=colors[i], markersize=10,
                                   capsize=6, capthick=2, elinewidth=2,
                                   alpha=0.8, label=algorithm)
                    else:
                        plt.scatter(x_positions[i], speedup, color=colors[i], s=100,
                                  alpha=0.8, edgecolor='white', linewidth=2, label=algorithm)
                else:
                    plt.scatter(x_positions[i], speedup, color=colors[i], s=100,
                              alpha=0.8, edgecolor='white', linewidth=2, label=algorithm)

                # Add speedup value as text annotation
                plt.annotate(f'{speedup:.3f}', (x_positions[i], speedup),
                           textcoords="offset points", xytext=(0,10), ha='center',
                           fontsize=9, fontweight='bold')

            plt.xlabel('Algorithm', fontsize=12, fontweight='bold')
            plt.ylabel('Average Speedup', fontsize=12, fontweight='bold')
            plt.title(f'{graph_title}\n(Averaged across all models)', fontsize=14, fontweight='bold')
            plt.xticks(x_positions, algorithms, rotation=45, ha='right')
            plt.grid(True, alpha=0.3)

            # Ensure y=1.0 is always visible while keeping all data and error bars well visible
            all_speedup_values = filtered_df['Speedup'].tolist()
            all_ci_min_values = []
            all_ci_max_values = []

            for _, row in filtered_df.iterrows():
                if 'Speedup CI Min' in row and pd.notna(row['Speedup CI Min']):
                    all_ci_min_values.append(row['Speedup CI Min'])
                if 'Speedup CI Max' in row and pd.notna(row['Speedup CI Max']):
                    all_ci_max_values.append(row['Speedup CI Max'])

            # Find actual data range including error bars
            all_values = all_speedup_values + all_ci_min_values + all_ci_max_values
            if all_values:
                actual_min = min(all_values)
                actual_max = max(all_values)

                # Ensure y=1.0 is included in visible range
                range_min = min(actual_min, 1.0)
                range_max = max(actual_max, 1.0)
                data_range = range_max - range_min

                # Add 15% padding for good visibility
                padding = data_range * 0.15

                # Set limits to show y=1.0 and all data with padding
                bottom_limit = max(0, range_min - padding)
                top_limit = range_max + padding

                plt.ylim(bottom=bottom_limit, top=top_limit)
            else:
                plt.ylim(bottom=0.5, top=1.5)

            # Add y=1.0 baseline
            plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.6, linewidth=1, label='Speedup=1.0')
            plt.legend()

            # Adjust layout and save
            plt.tight_layout()
            graph_path = graphs_dir / filename
            plt.savefig(graph_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Averaged models graph saved to: {graph_path}")

    print(f"All averaged models speedup graphs saved to: {graphs_dir}")


def create_averaged_models_t_vs_c_graph(results_dir, output_dir):
    """Create averaged traversal vs construction time graph - average values across all models"""
    print("Creating averaged traversal vs construction time graph...")

    dynamic_file = results_dir / "dynamic" / "statistical_results_table.csv"
    static_file = results_dir / "static" / "statistical_results_table.csv"

    if not dynamic_file.exists() or not static_file.exists():
        print("Required dynamic and static CSV files not found!")
        return

    try:
        dynamic_df = pd.read_csv(dynamic_file)
        static_df = pd.read_csv(static_file)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    # Merge dynamic and static data
    merged_df = dynamic_df.merge(
        static_df,
        on=['Model', 'Algorithm', 'k', 'Type'],
        suffixes=('_dynamic', '_static')
    )

    # Calculate construction time and traversal time
    merged_df['construction_time'] = merged_df['k=X Time (s)_dynamic'] - merged_df['k=X Time (s)_static']
    merged_df['traversal_time'] = merged_df['k=X Time (s)_static']

    # Also process k=2 data separately and add to dataset
    k2_data = []
    for _, row in merged_df.iterrows():
        k2_construction = row['k=2 Time (s)_dynamic'] - row['k=2 Time (s)_static']
        k2_traversal = row['k=2 Time (s)_static']

        k2_row = {
            'Model': row['Model'],
            'Algorithm': row['Algorithm'],
            'k': 2,
            'Type': row['Type'],
            'construction_time': k2_construction,
            'traversal_time': k2_traversal
        }
        k2_data.append(k2_row)

    k2_df = pd.DataFrame(k2_data)

    # Combine k=X and k=2 data
    time_data = merged_df[['Model', 'Algorithm', 'k', 'Type', 'construction_time', 'traversal_time']].copy()
    combined_data = pd.concat([time_data, k2_df], ignore_index=True)

    # Average k-way and collapsed values for each Algorithm first
    step1_averaged = combined_data.groupby(['Model', 'Algorithm', 'k'])[['construction_time', 'traversal_time']].mean().reset_index()

    # Now average across all models for each Algorithm + k combination
    final_averaged = step1_averaged.groupby(['Algorithm', 'k'])[['construction_time', 'traversal_time']].mean().reset_index()

    # Add model count information
    model_counts = step1_averaged.groupby(['Algorithm', 'k']).size().reset_index(name='model_count')
    final_averaged = final_averaged.merge(model_counts, on=['Algorithm', 'k'])

    # Debug: Print data availability
    print("Debug: Averaged t-vs-c data overview:")
    print(f"Algorithms: {sorted(final_averaged['Algorithm'].unique())}")
    print(f"K values: {sorted(final_averaged['k'].unique())}")
    print(f"Average model count per combination: {final_averaged['model_count'].mean():.1f}")

    # Get unique algorithms and k values
    algorithms = sorted(final_averaged['Algorithm'].unique())
    k_values = sorted(final_averaged['k'].unique())

    # Create single plot showing all data
    plt.figure(figsize=(10, 8))

    # Use distinct colors and markers for different k values
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Blue, Orange, Green, Red, Purple
    markers = ['o', 's', '^', 'D', 'v']  # Circle, Square, Triangle, Diamond, Triangle-down

    for j, k_val in enumerate(k_values):
        k_data = final_averaged[final_averaged['k'] == k_val]

        if k_data.empty:
            print(f"Debug: No data for k={k_val}")
            continue

        # Sort by algorithm for consistent plotting
        k_data = k_data.sort_values('Algorithm')

        # Extract x (construction time) and y (traversal time) data
        x_data = k_data['construction_time'].tolist()
        y_data = k_data['traversal_time'].tolist()
        algorithm_labels = k_data['Algorithm'].tolist()
        model_counts = k_data['model_count'].tolist()

        if x_data and y_data:
            # Plot scatter points for algorithms for this k value
            plt.scatter(x_data, y_data,
                       color=colors[j % len(colors)],
                       marker=markers[j % len(markers)],
                       label=f'k={k_val}',
                       s=120,
                       edgecolors='white',
                       linewidth=2,
                       alpha=0.8)

            # Add algorithm labels as annotations with smart positioning to avoid overlaps
            for idx, (x, y, alg, count) in enumerate(zip(x_data, y_data, algorithm_labels, model_counts)):
                # Calculate smart offset positions with larger distance for better point visibility
                angle = 2 * np.pi * idx / len(algorithm_labels)
                offset_x = 50 * np.cos(angle) * 0.5  # Increased to 50 for better visibility and less overlap
                offset_y = 50 * np.sin(angle) * 0.5  # Increased to 50 for better visibility and less overlap

                plt.annotate(f'{alg}', (x, y), xytext=(offset_x, offset_y),
                           textcoords='offset points', fontsize=9, alpha=0.95,
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                    edgecolor=colors[j % len(colors)], alpha=0.95),
                           arrowprops=dict(arrowstyle='->',
                                         color=colors[j % len(colors)], alpha=0.8, lw=2.0))

            print(f"Debug: k={k_val} has {len(x_data)} algorithm points (averaged over {model_counts[0]} models)")

    plt.xlabel('Construction Time (s)', fontsize=12, fontweight='bold')
    plt.ylabel('Traversal Time (s)', fontsize=12, fontweight='bold')
    plt.title('Traversal Time vs Construction Time Analysis\n(Averaged across all models)',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='best')

    # Ensure both axes start at 0 and have equal scaling for better comparison
    xlim = plt.xlim()
    ylim = plt.ylim()

    # Find the maximum range to make axes equal
    max_val = max(xlim[1], ylim[1])

    plt.xlim(left=0, right=max_val)
    plt.ylim(bottom=0, top=max_val)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.tight_layout()

    output_path = output_dir / "graphs" / "t-vs-c-averaged.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Averaged t-vs-c graph saved to: {output_path}")
    print(f"Graph shows average construction vs traversal times across all models")


def create_averaged_models_t_vs_c_graph_no_k16(results_dir, output_dir):
    """Create averaged traversal vs construction time graph without k=16 - average values across all models"""
    print("Creating averaged traversal vs construction time graph (without k=16)...")

    dynamic_file = results_dir / "dynamic" / "statistical_results_table.csv"
    static_file = results_dir / "static" / "statistical_results_table.csv"

    if not dynamic_file.exists() or not static_file.exists():
        print("Required dynamic and static CSV files not found!")
        return

    try:
        dynamic_df = pd.read_csv(dynamic_file)
        static_df = pd.read_csv(static_file)
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    # Merge dynamic and static data
    merged_df = dynamic_df.merge(
        static_df,
        on=['Model', 'Algorithm', 'k', 'Type'],
        suffixes=('_dynamic', '_static')
    )

    # Filter out k=16 data
    merged_df = merged_df[merged_df['k'] != 16]

    # Calculate construction time and traversal time
    merged_df['construction_time'] = merged_df['k=X Time (s)_dynamic'] - merged_df['k=X Time (s)_static']
    merged_df['traversal_time'] = merged_df['k=X Time (s)_static']

    # Also process k=2 data separately and add to dataset
    k2_data = []
    for _, row in merged_df.iterrows():
        k2_construction = row['k=2 Time (s)_dynamic'] - row['k=2 Time (s)_static']
        k2_traversal = row['k=2 Time (s)_static']

        k2_row = {
            'Model': row['Model'],
            'Algorithm': row['Algorithm'],
            'k': 2,
            'Type': row['Type'],
            'construction_time': k2_construction,
            'traversal_time': k2_traversal
        }
        k2_data.append(k2_row)

    k2_df = pd.DataFrame(k2_data)

    # Combine k=X and k=2 data, filter out any remaining k=16
    time_data = merged_df[['Model', 'Algorithm', 'k', 'Type', 'construction_time', 'traversal_time']].copy()
    combined_data = pd.concat([time_data, k2_df], ignore_index=True)
    combined_data = combined_data[combined_data['k'] != 16]

    # Average k-way and collapsed values for each Algorithm first
    step1_averaged = combined_data.groupby(['Model', 'Algorithm', 'k'])[['construction_time', 'traversal_time']].mean().reset_index()

    # Now average across all models for each Algorithm + k combination
    final_averaged = step1_averaged.groupby(['Algorithm', 'k'])[['construction_time', 'traversal_time']].mean().reset_index()

    # Add model count information
    model_counts = step1_averaged.groupby(['Algorithm', 'k']).size().reset_index(name='model_count')
    final_averaged = final_averaged.merge(model_counts, on=['Algorithm', 'k'])

    # Debug: Print data availability
    print("Debug: Averaged t-vs-c data overview (without k=16):")
    print(f"Algorithms: {sorted(final_averaged['Algorithm'].unique())}")
    print(f"K values: {sorted(final_averaged['k'].unique())}")
    print(f"Average model count per combination: {final_averaged['model_count'].mean():.1f}")

    # Get unique algorithms and k values
    algorithms = sorted(final_averaged['Algorithm'].unique())
    k_values = sorted([k for k in final_averaged['k'].unique() if k != 16])

    # Create single plot showing all data
    plt.figure(figsize=(10, 8))

    # Use distinct colors and markers for different k values (without k=16)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green for k=2,4,8
    markers = ['o', 's', '^']  # Circle, Square, Triangle for k=2,4,8

    for j, k_val in enumerate(k_values):
        k_data = final_averaged[final_averaged['k'] == k_val]

        if k_data.empty:
            print(f"Debug: No data for k={k_val}")
            continue

        # Sort by algorithm for consistent plotting
        k_data = k_data.sort_values('Algorithm')

        # Extract x (construction time) and y (traversal time) data
        x_data = k_data['construction_time'].tolist()
        y_data = k_data['traversal_time'].tolist()
        algorithm_labels = k_data['Algorithm'].tolist()
        model_counts = k_data['model_count'].tolist()

        if x_data and y_data:
            # Plot scatter points for algorithms for this k value
            plt.scatter(x_data, y_data,
                       color=colors[j % len(colors)],
                       marker=markers[j % len(markers)],
                       label=f'k={k_val}',
                       s=120,
                       edgecolors='white',
                       linewidth=2,
                       alpha=0.8)

            # Add algorithm labels as annotations with smart positioning to avoid overlaps
            for idx, (x, y, alg, count) in enumerate(zip(x_data, y_data, algorithm_labels, model_counts)):
                # Calculate smart offset positions to avoid overlap
                angle = 2 * np.pi * idx / len(algorithm_labels)
                offset_x = -50 * np.cos(angle) * 0.5 # Increased to 50 for better visibility and less overlap
                offset_y = -50 * np.sin(angle) * 0.5# Increased to 50 for better visibility and less overlap

                plt.annotate(f'{alg}', (x, y), xytext=(offset_x, offset_y),
                           textcoords='offset points', fontsize=9, alpha=0.95,
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                    edgecolor=colors[j % len(colors)], alpha=0.95),
                           arrowprops=dict(arrowstyle='->',
                                         color=colors[j % len(colors)], alpha=0.8, lw=2.0))

            print(f"Debug: k={k_val} has {len(x_data)} algorithm points (averaged over {model_counts[0]} models)")

    plt.xlabel('Construction Time (s)', fontsize=12, fontweight='bold')
    plt.ylabel('Traversal Time (s)', fontsize=12, fontweight='bold')
    plt.title('Traversal Time vs Construction Time Analysis (k=2,4,8)\n(Averaged across all models)',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='best')

    # Ensure both axes start at 0 and have equal scaling for better comparison
    xlim = plt.xlim()
    ylim = plt.ylim()

    # Find the maximum range to make axes equal
    max_val = max(xlim[1], ylim[1])

    plt.xlim(left=0, right=max_val)
    plt.ylim(bottom=0, top=max_val)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.tight_layout()

    output_path = output_dir / "graphs" / "t-vs-c-averaged-no-k16.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Averaged t-vs-c graph (without k=16) saved to: {output_path}")
    print(f"Graph shows average construction vs traversal times across all models (k=2,4,8)")


def main():
    parser = argparse.ArgumentParser(description="Convert CSV tables to LaTeX format - Ah-hyuck!")
    parser.add_argument(
        "--results_dir",
        type=Path,
        help="Path to results directory containing CSV files"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Output directory for LaTeX files (default: same as results_dir)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*table*.csv",
        help="File pattern to match CSV files (default: *table*.csv)"
    )
    parser.add_argument(
        "--create_graphs",
        action="store_true",
        help="Create speedup graphs in addition to LaTeX tables"
    )

    args = parser.parse_args()

    # Find results directory automatically if not specified
    if not args.results_dir:
        # Look for the most recent results directory
        base_dir = Path(".")
        possible_dirs = list(base_dir.glob("**/results"))
        if possible_dirs:
            args.results_dir = max(possible_dirs, key=lambda p: p.stat().st_mtime)
            print(f"Auto-detected results directory: {args.results_dir}")
        else:
            print("No results directory found! Please specify --results_dir")
            return

    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else results_dir

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define table configurations
    table_configs = {
        'statistical': {
            'caption': 'Statistical Analysis Results',
            'label': 'tab:statistical',
            'section_title': 'Statistical Analysis'
        },
        'comparison': {
            'caption': 'Algorithm Comparison Results',
            'label': 'tab:comparison',
            'section_title': 'Algorithm Comparison'
        },
        'dynamic': {
            'caption': 'Dynamic Analysis Results',
            'label': 'tab:dynamic',
            'section_title': 'Dynamic Analysis'
        },
        'detailed': {
            'caption': 'Detailed Quality Analysis Results',
            'label': 'tab:detailed',
            'section_title': 'Detailed Quality Analysis'
        }
    }

    # Find CSV files recursively in all subdirectories
    csv_files = list(results_dir.rglob(args.pattern))
    if not csv_files:
        print(f"No CSV files found matching pattern '{args.pattern}' in {results_dir} or its subdirectories")
        return

    print(f"Found {len(csv_files)} CSV files to process:")
    for csv_file in csv_files:
        print(f"  - {csv_file.name}")

    # Process each CSV file
    processed_tables = []

    for csv_file in csv_files:
        # Get the folder name to determine table type
        folder_name = csv_file.parent.name
        relative_path = csv_file.relative_to(results_dir)

        # Try to match with known table types based on folder name
        table_config = None
        for config_key, config in table_configs.items():
            if config_key in folder_name or config_key in csv_file.stem:
                table_config = config
                break

        # Use default config if no match found
        if not table_config:
            # Include folder info in the title for better identification
            folder_info = f" - {folder_name}" if folder_name != "results" else ""
            table_config = {
                'caption': f'Analysis Results Table: {csv_file.stem.replace("_", " ").title()}{folder_info}',
                'label': f'tab:{csv_file.stem}_{folder_name}',
                'section_title': f'{csv_file.stem.replace("_", " ").title()}{folder_info}'
            }

        table_info = process_csv_to_latex(csv_file, output_dir, table_config)
        if table_info:
            processed_tables.append(table_info)

    # Create consolidated document
    if processed_tables:
        create_consolidated_latex_document(processed_tables, output_dir)
        print(f"Individual LaTeX files and consolidated document saved to: {output_dir}")
    else:
        print("No tables were processed successfully.")

    # Create graphs if requested
    if args.create_graphs:
        # Create comparison graphs
        create_speedup_graphs_for_type(results_dir, output_dir, "comparison")
        # Create dynamic graphs
        create_speedup_graphs_for_type(results_dir, output_dir, "dynamic")
        # Create static graphs
        create_speedup_graphs_for_type(results_dir, output_dir, "static")
        # Create combined speedup graphs (k-way + collapsed averaged)
        create_combined_speedup_graphs(results_dir, output_dir, "comparison")
        create_combined_speedup_graphs(results_dir, output_dir, "dynamic")
        create_combined_speedup_graphs(results_dir, output_dir, "static")
        # Create traversal vs construction time ratio graph
        create_traversal_vs_construction_graph(results_dir, output_dir)
        # Create traversal vs construction time ratio graph without k=16
        create_traversal_vs_construction_graph_no_k16(results_dir, output_dir)
        # Create averaged models speedup graphs
        create_averaged_models_speedup_graph(results_dir, output_dir, "comparison")
        create_averaged_models_speedup_graph(results_dir, output_dir, "dynamic")
        create_averaged_models_speedup_graph(results_dir, output_dir, "static")
        # Create averaged traversal vs construction time graphs
        create_averaged_models_t_vs_c_graph(results_dir, output_dir)
        create_averaged_models_t_vs_c_graph_no_k16(results_dir, output_dir)


if __name__ == "__main__":
    main()

