#!/usr/bin/env python3
"""
Converts statistical results CSV tables to LaTeX format.
Produces three tables (static, comparison, dynamic) with consistent column formatting.
"""

import pandas as pd
import argparse
from pathlib import Path


def format_p_value(p_val_str):
    """Format p-value strings with significance stars."""
    clean = str(p_val_str).replace('*', '')
    try:
        p_val = float(clean)
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


def format_ci(ci_min, ci_max):
    """Format confidence interval as [CI-Min, CI-Max]."""
    return f"[{ci_min:.3f}, {ci_max:.3f}]"


def capitalize_model_name(name):
    """Capitalize model names for display (e.g., 'stanford-bunny' -> 'Bunny')."""
    if pd.isna(name):
        return name
    return str(name).split('-')[-1].capitalize()


def process_table(csv_path, table_type):
    """Process a single CSV file into a formatted DataFrame."""
    df = pd.read_csv(csv_path)

    p_col = 'p-value' if 'p-value' in df.columns else None

    out = pd.DataFrame()
    out['Model'] = df['Model'].apply(capitalize_model_name)
    out['Algorithm'] = df['Algorithm']
    out['k'] = df['k']

    # Static and dynamic have a Type column; comparison does not
    if table_type != 'comparison' and 'type' in df.columns:
        out['Type'] = df['type']

    out['Speedup $\\mu$'] = df['Speedup'].apply(lambda x: f"{x:.3f}")
    out['Speedup CI'] = df.apply(
        lambda row: format_ci(row['Speedup CI Min'], row['Speedup CI Max']), axis=1
    )

    if p_col:
        out['p'] = df[p_col].apply(format_p_value)

    return out


def df_to_longtable(df, caption, label, table_type):
    """Convert a DataFrame to a LaTeX longtable string."""
    n_cols = len(df.columns)
    # Build column format: l for text, r for numeric
    col_fmt = '@{\\extracolsep{\\fill}}'
    for col in df.columns:
        if col in ('k',):
            col_fmt += 'r'
        elif col in ('Speedup $\\mu$', 'p'):
            col_fmt += 'r'
        else:
            col_fmt += 'l'

    header = ' & '.join(df.columns) + ' \\\\'

    lines = []
    lines.append(f'\\begin{{longtable}}{{{col_fmt}}}')
    lines.append(f'\\caption{{{caption}}} \\label{{tab:{label}}} \\\\')
    lines.append('\\toprule')
    lines.append(header)
    lines.append('\\midrule')
    lines.append('\\endfirsthead')
    lines.append(f'\\caption[]{{{caption}}} \\\\')
    lines.append('\\toprule')
    lines.append(header)
    lines.append('\\midrule')
    lines.append('\\endhead')
    lines.append('\\midrule')
    lines.append(f'\\multicolumn{{{n_cols}}}{{r}}{{Continued on next page}} \\\\')
    lines.append('\\midrule')
    lines.append('\\endfoot')
    lines.append('\\bottomrule')
    lines.append('\\endlastfoot')

    for _, row in df.iterrows():
        row_str = ' & '.join(str(v) for v in row.values) + ' \\\\'
        lines.append(row_str)

    lines.append('\\end{longtable}')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description="Convert CSV tables to LaTeX format")
    parser.add_argument(
        "--results_dir",
        type=Path,
        help="Path to results directory containing static/, comparison/, dynamic/ subdirs"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Output directory for LaTeX files (default: same as results_dir)"
    )

    args = parser.parse_args()

    # Find results directory automatically if not specified
    if not args.results_dir:
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

    output_dir.mkdir(parents=True, exist_ok=True)

    # Table definitions: (subdir, table_type, caption, label)
    tables = [
        ('static', 'static', 'Static Statistical Results', 'static'),
        ('comparison', 'comparison', 'Comparison Statistical Results', 'comparison'),
        ('dynamic', 'dynamic', 'Dynamic Statistical Results', 'dynamic'),
    ]

    for subdir, table_type, caption, label in tables:
        csv_path = results_dir / subdir / 'statistical_results_table.csv'
        if not csv_path.exists():
            # Try flat structure
            csv_path = results_dir / f'{subdir}_statistical_results_table.csv'
        if not csv_path.exists():
            print(f"Skipping {table_type}: CSV not found")
            continue

        print(f"Processing: {csv_path}")
        df = process_table(csv_path, table_type)
        latex = df_to_longtable(df, caption, label, table_type)

        out_path = output_dir / f'{table_type}_statistical_results_table.tex'
        with open(out_path, 'w') as f:
            f.write(latex)
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
