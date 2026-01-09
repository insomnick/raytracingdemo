import argparse
from pathlib import Path

import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------------------
# Laden & Vorverarbeitung
# ------------------------------------------------------------

def load_csv(csv_file: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_file)

    # data_type: 'render_times', 'bvh_build_times', 'shading_times'
    data_type = csv_file.stem
    df["data_type"] = data_type

    # Modellnamen säubern
    df["model_name"] = df["model_name"].apply(lambda x: Path(x).stem)

    # Algorithmus-Infos zerlegen
    df["algorithm_degree"] = df["algorithm_name"].apply(
        lambda x: int(x.split("-")[-1])
    )
    df["algorithm_type"] = df["algorithm_name"].apply(
        lambda x: "collapsed" if "-c-" in x else "k-way"
    )
    df["algorithm_prefix"] = df["algorithm_name"].apply(
        lambda x: x.split("-")[0]
    )

    return df


def load_all_testruns(base_dir: Path) -> pd.DataFrame:
    all_frames = []

    for testrun_dir in sorted(base_dir.glob("testrun_*")):
        if not testrun_dir.is_dir():
            continue

        testrun_name = testrun_dir.name
        testrun_index = int(testrun_name.split("_")[-1])

        # CSVs laden
        build_data = load_csv(testrun_dir / "bvh_build_times.csv")
        build_data = build_data.rename(columns={"time_seconds": "construction_time"})

        render_data = load_csv(testrun_dir / "render_times.csv")
        render_data = render_data.rename(columns={"time_seconds": "traversal_time"})

        # optional: shading
        try:
            shader_data = load_csv(testrun_dir / "shading_times.csv")
            shader_data = shader_data.rename(columns={"time_seconds": "hitray_count"})
            combined = pd.concat([build_data, render_data, shader_data], ignore_index=True)
        except FileNotFoundError:
            combined = pd.concat([build_data, render_data], ignore_index=True)

        combined["testrun_index"] = testrun_index

        all_frames.append(combined)

    if not all_frames:
        raise RuntimeError(f"Keine CSV-Dateien unter {base_dir} gefunden")

    df = pd.concat(all_frames, ignore_index=True)
    return df


# ------------------------------------------------------------
# Aggregation auf Testrun-Ebene
# ------------------------------------------------------------

def aggregate_times(df: pd.DataFrame):
    """
    Liefert zwei DataFrames:

    - render_agg: mittlere traversal_time pro Testrun & Konfiguration
    - dyn_agg: wie oben, plus construction_time und dynamic_time
    """

    # Nur Render- und Build-Daten
    render_df = df[df["data_type"] == "render_times"].copy()
    build_df = df[df["data_type"] == "bvh_build_times"].copy()

    group_keys = [
        "testrun_index",
        "model_name",
        "algorithm_prefix",
        "algorithm_type",
        "algorithm_degree",
    ]

    # Traversal: Mittelwert über alle Frames/Kamerapositionen pro Testrun & Konfiguration
    render_agg = (
        render_df
        .groupby(group_keys, as_index=False)["traversal_time"]
        .mean()
    )

    # Construction: Mittelwert über die 10 Wiederholungen
    build_agg = (
        build_df
        .groupby(group_keys, as_index=False)["construction_time"]
        .mean()
    )

    # Zusammenführen und dynamische Zeit berechnen
    dyn_agg = pd.merge(
        render_agg,
        build_agg,
        on=group_keys,
        how="inner",
    )
    dyn_agg["dynamic_time"] = dyn_agg["traversal_time"] + dyn_agg["construction_time"]

    return render_agg, dyn_agg


# ------------------------------------------------------------
# Statistische Tests: Branching-Grad
# ------------------------------------------------------------
def run_degree_tests(agg_df: pd.DataFrame, time_col: str, label: str, alpha: float = 0.05) -> pd.DataFrame:
    """
    Vergleicht für jede Kombination (Modell, Algorithmus)
    k=2 (immer k-way) mit k in {4,8,16}, jeweils:
      - Kandidat k-way
      - Kandidat collapsed

    Pairing über testrun_index.
    time_col: 'traversal_time' oder 'dynamic_time'
    label: z.B. 'Statisch' oder 'Dynamisch' für die Ausgabe
    """
    results = []

    models = sorted(agg_df["model_name"].unique())
    prefixes = sorted(agg_df["algorithm_prefix"].unique())
    degrees = sorted(agg_df["algorithm_degree"].unique())

    print(f"\n===== Branching-Grad-Tests ({label}, Variable: {time_col}) =====\n")

    for model in models:
        for prefix in prefixes:
            # Subset nur nach Modell + Algorithmus, algorithm_type wird erst bei Kandidaten betrachtet
            subset = agg_df[
                (agg_df["model_name"] == model)
                & (agg_df["algorithm_prefix"] == prefix)
                ]

            if subset.empty:
                continue

            # Baseline: k=2 (typischerweise k-way)
            baseline = subset[subset["algorithm_degree"] == 2]

            if baseline.empty:
                # Ohne Baseline macht der Test keinen Sinn
                continue

            for degree in degrees:
                if degree == 2:
                    continue

                for algo_type in ["k-way", "collapsed"]:
                    candidate = subset[
                        (subset["algorithm_degree"] == degree)
                        & (subset["algorithm_type"] == algo_type)
                        ]

                    if candidate.empty:
                        continue

                    merged = pd.merge(
                        baseline,
                        candidate,
                        on="testrun_index",
                        suffixes=("_k2", "_kX"),
                    )

                    if len(merged) < 2:
                        # Zu wenig Paare für einen sinnvollen Test
                        continue

                    diff = merged[f"{time_col}_k2"] - merged[f"{time_col}_kX"]
                    mean_diff = diff.mean()
                    std_diff = diff.std(ddof=1)
                    t_stat, p_val = stats.ttest_1samp(diff, popmean=0)

                    mean_k2 = merged[f"{time_col}_k2"].mean()
                    rel_diff = 100.0 * mean_diff / mean_k2 if mean_k2 != 0 else float("nan")
                    significant = p_val < alpha

                    results.append(
                        {
                            "scene_type": label,
                            "time_col": time_col,
                            "model_name": model,
                            "algorithm_prefix": prefix,
                            "algorithm_type_candidate": algo_type,
                            "degree_baseline": 2,
                            "degree_candidate": degree,
                            "n_pairs": len(diff),
                            "mean_diff": mean_diff,
                            "std_diff": std_diff,
                            "rel_diff_percent": rel_diff,
                            "t_stat": t_stat,
                            "p_value": p_val,
                            "alpha": alpha,
                            "significant": significant,
                        }
                    )

                    direction = "schneller" if mean_diff > 0 else "langsamer"
                    sig_str = "SIGNIFIKANT" if significant else "nicht signifikant"

                    print(
                        f"{model} | {prefix} | Kandidat={algo_type} | k=2 vs k={degree} "
                        f"(n={len(diff)}): mean_diff={mean_diff:.3e}s "
                        f"({rel_diff:.2f}%), t={t_stat:.2f}, p={p_val:.2e} -> {sig_str}, {direction}"
                    )

    return pd.DataFrame(results)

# ------------------------------------------------------------
# Statistische Tests: Erstellungsmethode (k-way vs. collapsed)
# ------------------------------------------------------------

def run_method_tests(agg_df: pd.DataFrame, time_col: str, label: str, alpha: float = 0.05) -> pd.DataFrame:
    """
    Vergleicht k-way vs. collapsed für jeden Branching-Grad, gepaart über testrun_index.
    """
    results = []

    models = sorted(agg_df["model_name"].unique())
    prefixes = sorted(agg_df["algorithm_prefix"].unique())
    degrees = sorted(agg_df["algorithm_degree"].unique())

    print(f"\n===== Methoden-Tests k-way vs. collapsed ({label}, Variable: {time_col}) =====\n")

    for model in models:
        for prefix in prefixes:
            for degree in degrees:
                subset = agg_df[
                    (agg_df["model_name"] == model)
                    & (agg_df["algorithm_prefix"] == prefix)
                    & (agg_df["algorithm_degree"] == degree)
                    ]

                if subset.empty:
                    continue

                k_way = subset[subset["algorithm_type"] == "k-way"]
                collapsed = subset[subset["algorithm_type"] == "collapsed"]

                merged = pd.merge(
                    k_way,
                    collapsed,
                    on="testrun_index",
                    suffixes=("_kway", "_collapsed"),
                )

                if len(merged) < 2:
                    continue

                diff = merged[f"{time_col}_kway"] - merged[f"{time_col}_collapsed"]
                mean_diff = diff.mean()
                std_diff = diff.std(ddof=1)
                t_stat, p_val = stats.ttest_1samp(diff, popmean=0)

                mean_kway = merged[f"{time_col}_kway"].mean()
                rel_diff = 100.0 * mean_diff / mean_kway if mean_kway != 0 else float("nan")
                significant = p_val < alpha

                results.append(
                    {
                        "scene_type": label,
                        "time_col": time_col,
                        "model_name": model,
                        "algorithm_prefix": prefix,
                        "degree": degree,
                        "n_pairs": len(diff),
                        "mean_diff": mean_diff,
                        "std_diff": std_diff,
                        "rel_diff_percent": rel_diff,
                        "t_stat": t_stat,
                        "p_value": p_val,
                        "alpha": alpha,
                        "significant": significant,
                    }
                )

                direction = "schneller" if mean_diff > 0 else "langsamer"
                sig_str = "SIGNIFIKANT" if significant else "nicht signifikant"

                print(
                    f"{model} | {prefix} | degree={degree} | k-way vs. collapsed "
                    f"(n={len(diff)}): mean_diff={mean_diff:.3e}s "
                    f"({rel_diff:.2f}%), t={t_stat:.2f}, p={p_val:.2e} -> {sig_str}, {direction}"
                )

    return pd.DataFrame(results)


# ------------------------------------------------------------
# Optionale Visualisierung (ein Beispiel-Plot)
# ------------------------------------------------------------

def plot_example_diff(diff_series, title: str):
    plt.figure(figsize=(6, 4))
    sns.histplot(diff_series, kde=True)
    plt.xlabel("Differenz (Baseline - Kandidat)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse BVH-Benchmarkdaten")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("./testruns"),
        help="Ordner mit testrun_* (Default: ./testruns)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Signifikanzniveau für die Tests (Default: 0.05)",
    )

    args = parser.parse_args()

    base_dir = args.dir.resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Folder not found: {base_dir}")

    df = load_all_testruns(base_dir)

    print("Successfully loaded data:")
    print(df.info())
    print(df.head())
    print(f"\nDataset rows: {len(df)}\n")

    render_agg, dyn_agg = aggregate_times(df)

    # 1) Branching-Grad: statische Szene (nur traversal_time)
    deg_results_static = run_degree_tests(
        render_agg,
        time_col="traversal_time",
        label="Statisch",
        alpha=args.alpha,
    )

    # 2) Branching-Grad: dynamische Szene (dynamic_time)
    deg_results_dynamic = run_degree_tests(
        dyn_agg,
        time_col="dynamic_time",
        label="Dynamisch",
        alpha=args.alpha,
    )

    # 3) Methodenvergleich: k-way vs. collapsed (statisch)
    method_results_static = run_method_tests(
        render_agg,
        time_col="traversal_time",
        label="Statisch",
        alpha=args.alpha,
    )

    # 4) Methodenvergleich: k-way vs. collapsed (dynamisch)
    method_results_dynamic = run_method_tests(
        dyn_agg,
        time_col="dynamic_time",
        label="Dynamisch",
        alpha=args.alpha,
    )

    # Optional: Ergebnisse als CSV speichern
    out_dir = base_dir / "analysis_results"
    out_dir.mkdir(exist_ok=True)

    deg_results_static.to_csv(out_dir / "degree_tests_static.csv", index=False)
    deg_results_dynamic.to_csv(out_dir / "degree_tests_dynamic.csv", index=False)
    method_results_static.to_csv(out_dir / "method_tests_static.csv", index=False)
    method_results_dynamic.to_csv(out_dir / "method_tests_dynamic.csv", index=False)

    print(f"\nErgebnisse gespeichert unter: {out_dir}")


if __name__ == "__main__":
    main()
