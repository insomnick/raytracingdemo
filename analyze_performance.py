import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

class PerformanceAnalyzer:
    def __init__(self, base_dir="testruns"):   # TODO: think about this when dockerize
        self.base_dir = Path(base_dir)
        self.result_dir = self.base_dir / "results"
        # Create results directory if it doesn't exist
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.data = {}
        self.metrics = ['bvh_build_times', 'render_times', 'shading_times']
        
    def load_data(self):
        print("Load data...")

        #For all testrun directories
        testrun_dirs = list(self.base_dir.glob("testrun_*"))
        if not testrun_dirs:
            print("No directories found")
            return False
            
        print(f"Found {len(testrun_dirs)} test runs")

        # Load data from each metric type
        for metric in self.metrics:
            self.data[metric] = []

            for testrun_dir in testrun_dirs:
                csv_file = testrun_dir / f"{metric}.csv"
                if csv_file.exists():
                    try:
                        df = pd.read_csv(csv_file, header=0)
                        if len(df.columns) == 11:
                            df.columns = [
                                'file_name', 'model_name', 'model_scale', 'algorithm_name',
                                'cam_pos_x', 'cam_pos_y', 'cam_pos_z',
                                'cam_dir_x', 'cam_dir_y', 'cam_dir_z', 'time_seconds'
                            ]
                        else:
                            print(f"Warning: {csv_file} has {len(df.columns)} columns, expected 10")
                        df['testrun'] = testrun_dir.name
                        df['metric_type'] = metric
                        self.data[metric].append(df)
                    except Exception as e:
                        print(f"Error loading {csv_file}: {e}")

            if self.data[metric]:
                self.data[metric] = pd.concat(self.data[metric], ignore_index=True)
                print(f"  Total {metric} data: {len(self.data[metric])} rows")
            else:
                print(f"  No data found for {metric}")

        # Summary
        total_data = sum(len(df) if df is not None and not df.empty else 0
                        for df in self.data.values() if isinstance(df, pd.DataFrame))
        print(f"\nTotal data loaded: {total_data} rows")
        return total_data > 0

    def print_summary(self):
        print("\nDATA SUMMARY")
        print("=" * 50)
        
        for metric in self.metrics:
            if metric in self.data and not self.data[metric].empty:
                df = self.data[metric]
                print(f"\n{metric.upper().replace('_', ' ')}:")
                print(f"Total measurements: {len(df)}")
                print(f"Algorithms:         {', '.join(df['algorithm_name'].unique())}")
                print(f"Models:             {', '.join(df['model_name'].unique())}")
                print(f"Test runs:          {', '.join(df['testrun'].unique())}")
                print(f"Time range:         {df['time_seconds'].min():.4f}s - {df['time_seconds'].max():.4f}s")
                print(f"Mean time:          {df['time_seconds'].mean():.4f}s")

    def compare_key_metrics(self, save_plots=True):
        print("\nKEY METRIC COMPARISON")
        print("=" * 50)

        fig, axes = plt.subplots(len(self.metrics), 1, figsize=(12, 4 * len(self.metrics)))
        if len(self.metrics) == 1:
            axes = [axes]

        fig.suptitle('Key Metric Comparison Across BVH Algorithms', fontsize=16, fontweight='bold')

        # Performance by algorithm (box plots)
        for i, metric in enumerate(self.metrics):
            if metric in self.data and not self.data[metric].empty:
                df = self.data[metric]
                try:
                    # Box plot comparing algorithms
                    sns.boxplot(data=df, x='algorithm_name', y='time_seconds', ax=axes[i])
                    axes[i].set_title(f'{metric.replace("_", " ").title()}')
                    axes[i].set_xlabel('Algorithm')
                    axes[i].set_ylabel('Time (seconds)')
                    axes[i].tick_params(axis='x', rotation=45)

                    # Add mean values as text
                    for j, algorithm in enumerate(df['algorithm_name'].unique()):
                        mean_time = df[df['algorithm_name'] == algorithm]['time_seconds'].mean()
                        axes[i].text(j, mean_time, f'μ={mean_time:.4f}s',
                            ha='center', va='bottom', fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                except Exception as e:
                    print(f"Error plotting {metric}: {e}")

        plt.tight_layout()

        if save_plots:
            try:
                output_path = self.result_dir / 'performance_key_metrics.png'
                plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png', facecolor='white')
                print(f"Saved algorithm comparison plot as '{output_path}'")
            except Exception as e:
                print(f"Error saving plot: {e}")

        # Only show if in interactive environment
        try:
            plt.show()
        except:
            pass  # Ignore display errors in headless environments
        finally:
            plt.close('all')

    def analyze_performance_trends(self, save_plots=True):
        """Analyze performance trends over camera path"""
        print("\nPERFORMANCE TRENDS ANALYSIS")
        print("=" * 50)
        
        fig, axes = plt.subplots(len(self.metrics), 1, figsize=(12, 4 * len(self.metrics)))
        if len(self.metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(self.metrics):
            if metric in self.data and not self.data[metric].empty:
                df = self.data[metric]
                
                # Group by algorithm and plot trends
                for algorithm in df['algorithm_name'].unique():
                    alg_data = df[df['algorithm_name'] == algorithm].copy()
                    
                    # Sort by some order (using index as proxy for time sequence)
                    alg_data = alg_data.reset_index().sort_values('index')
                    
                    axes[i].plot(alg_data.index, alg_data['time_seconds'], 
                               label=f'{algorithm}', marker='o', alpha=0.7)
                
                axes[i].set_title(f'{metric.replace("_", " ").title()} Over Time')
                axes[i].set_xlabel('Measurement Sequence')
                axes[i].set_ylabel('Time (seconds)')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            try:
                plt.savefig(self.result_dir / 'performance_trends.png', dpi=300, bbox_inches='tight')
                print("Saved performance trends plot as 'performance_trends.png'")
            except Exception as e:
                print(f"Error saving performance trends plot: {e}")

        # Only show if in interactive environment
        try:
            plt.show()
        except:
            pass  # Ignore display errors in headless environments
        finally:
            plt.close()

    def generate_statistical_report(self):
        """Generate detailed statistical analysis"""
        print("\nSTATISTICAL ANALYSIS REPORT")
        print("=" * 50)
        
        for metric in self.metrics:
            if metric in self.data and not self.data[metric].empty:
                df = self.data[metric]
                print(f"\n{metric.upper().replace('_', ' ')}")
                print("-" * 40)
                
                # Group by algorithm
                stats = df.groupby('algorithm_name')['time_seconds'].agg([
                    'count', 'mean', 'std', 'min', 'max', 'median'
                ]).round(6)
                
                print(stats)
                
                # Statistical significance testing (ANOVA)
                try:
                    from scipy import stats as scipy_stats
                    algorithms = df['algorithm_name'].unique()
                    if len(algorithms) > 1:
                        groups = [df[df['algorithm_name'] == alg]['time_seconds'].values 
                                for alg in algorithms]
                        f_stat, p_value = scipy_stats.f_oneway(*groups)
                        print(f"\nANOVA Test Results:")
                        print(f"F-statistic:    {f_stat:.4f}")
                        print(f"P-value:        {p_value:.6f}")
                        
                        if p_value < 0.05:
                            print("Statistically significant difference between algorithms!")
                        else:
                            print("No statistically significant difference detected.")
                except ImportError:
                    print("Scipy not available for statistical testing")
                except Exception as e:
                    print(f"Error in statistical testing: {e}")
    
    def find_best_performers(self):
        """Identify best performing configurations"""
        print("\nBEST PERFORMERS")
        print("=" * 50)
        
        for metric in self.metrics:
            if metric in self.data and not self.data[metric].empty:
                df = self.data[metric]
                print(f"\n{metric.replace('_', ' ').title()}:")
                
                # Best algorithm overall
                best_alg = df.groupby('algorithm_name')['time_seconds'].mean().idxmin()
                best_time = df.groupby('algorithm_name')['time_seconds'].mean().min()
                print(f"Best Algorithm: {best_alg} (avg: {best_time:.4f}s)")
                
                # Best single measurement
                best_single = df.loc[df['time_seconds'].idxmin()]
                print(f"Fastest Single Run: {best_single['algorithm_name']} "
                      f"({best_single['time_seconds']:.4f}s)")
                
                # Performance improvement
                worst_alg = df.groupby('algorithm_name')['time_seconds'].mean().idxmax()
                worst_time = df.groupby('algorithm_name')['time_seconds'].mean().max()
                improvement = ((worst_time - best_time) / worst_time) * 100
                print(f"Improvement: {improvement:.1f}% faster than worst ({worst_alg})")

def main():
    parser = argparse.ArgumentParser(description='Analyze raytracing performance data')
    parser.add_argument('--dir', '-d', default='testruns',
                       help='Base directory containing testrun folders')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--save-plots', action='store_true', default=True,
                       help='Save plots to files')

    args = parser.parse_args()

    print("Raytracing Performance Analyzer")
    print("=" * 50)

    analyzer = PerformanceAnalyzer(args.dir)

    if not analyzer.load_data():
        print("Failed to load data. Make sure you have testrun directories with CSV files.")
        return

    analyzer.print_summary()
    analyzer.generate_statistical_report()
    # analyzer.find_best_performers()

    if not args.no_plots:
        analyzer.compare_key_metrics(args.save_plots)
        analyzer.analyze_performance_trends(args.save_plots)

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
