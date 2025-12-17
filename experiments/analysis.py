"""Analysis tools for CPR experiment results.

Provides functions for loading, analyzing, and visualizing
experiment results.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List


def load_results(filepath: str) -> Dict:
    """Load experiment results from JSON file.

    Args:
        filepath: Path to results file

    Returns:
        Dict: Loaded results
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_persona_results(results: Dict) -> pd.DataFrame:
    """Analyze persona comparison results.

    Args:
        results: Results from persona comparison experiment

    Returns:
        DataFrame with summary statistics
    """
    data = []

    for matchup, stats in results.items():
        data.append({
            "matchup": matchup,
            "tragedy_rate": stats["tragedy_rate"],
            "avg_cooperation": stats["avg_cooperation"],
            "avg_rounds": stats["avg_rounds"],
        })

    df = pd.DataFrame(data)
    return df


def plot_tragedy_rates(results_dict: Dict, title: str = "Tragedy Rates by Configuration"):
    """Plot tragedy occurrence rates.

    Args:
        results_dict: Dictionary of results
        title: Plot title
    """
    configs = []
    tragedy_rates = []

    for config, result in results_dict.items():
        configs.append(config)

        # Check if this is a multi-game result
        if isinstance(result, dict) and "tragedy_rate" in result:
            tragedy_rates.append(result["tragedy_rate"])
        elif isinstance(result, dict) and "tragedy_occurred" in result:
            tragedy_rates.append(1.0 if result["tragedy_occurred"] else 0.0)
        else:
            tragedy_rates.append(0.0)

    plt.figure(figsize=(10, 6))
    plt.bar(configs, tragedy_rates, color='coral')
    plt.xlabel('Configuration')
    plt.ylabel('Tragedy Rate')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)

    return plt.gcf()


def plot_cooperation_trends(results_dict: Dict):
    """Plot cooperation index trends.

    Args:
        results_dict: Dictionary of results
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for config, result in results_dict.items():
        if "avg_cooperation" in result:
            ax.scatter([config], [result["avg_cooperation"]], s=100, label=config)

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Average Cooperation Index')
    ax.set_title('Cooperation Index by Configuration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig


def plot_resource_dynamics(game_state: Dict):
    """Plot resource level over time for a single game.

    Args:
        game_state: Game state dictionary
    """
    resource_history = game_state.get("resource_history", [])

    if not resource_history:
        print("No resource history available")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(resource_history, linewidth=2, color='green', label='Resource Level')
    ax.axhline(y=500, color='red', linestyle='--', label='Sustainability Threshold', alpha=0.7)
    ax.set_xlabel('Round')
    ax.set_ylabel('Resource Level')
    ax.set_title('Resource Dynamics Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def compare_metrics_across_experiments(results_files: List[str]):
    """Compare key metrics across multiple experiments.

    Args:
        results_files: List of paths to result files
    """
    all_data = []

    for filepath in results_files:
        experiment_name = Path(filepath).stem
        results = load_results(filepath)

        for config, result in results.items():
            # Handle both single game and multi-game results
            if isinstance(result, dict):
                if "games" in result:
                    # Multi-game result
                    for game in result["games"]:
                        all_data.append({
                            "experiment": experiment_name,
                            "config": config,
                            "tragedy": game.get("tragedy_occurred", False),
                            "cooperation": game.get("avg_cooperation_index", 0),
                            "rounds": game.get("total_rounds", 0),
                            "final_resource": game.get("final_resource_level", 0),
                            "gini": game.get("gini_coefficient", 0),
                        })
                else:
                    # Single game result
                    all_data.append({
                        "experiment": experiment_name,
                        "config": config,
                        "tragedy": result.get("tragedy_occurred", False),
                        "cooperation": result.get("avg_cooperation_index", 0),
                        "rounds": result.get("total_rounds", 0),
                        "final_resource": result.get("final_resource_level", 0),
                        "gini": result.get("gini_coefficient", 0),
                    })

    df = pd.DataFrame(all_data)
    return df


def create_summary_report(df: pd.DataFrame) -> str:
    """Create text summary report.

    Args:
        df: DataFrame with experiment results

    Returns:
        str: Formatted report
    """
    report = []
    report.append("=" * 70)
    report.append("CPR EXPERIMENT SUMMARY REPORT")
    report.append("=" * 70)
    report.append("")

    # Overall statistics
    report.append("OVERALL STATISTICS:")
    report.append(f"Total Games: {len(df)}")
    report.append(f"Tragedy Rate: {df['tragedy'].mean():.1%}")
    report.append(f"Avg Cooperation: {df['cooperation'].mean():.3f}")
    report.append(f"Avg Rounds per Game: {df['rounds'].mean():.1f}")
    report.append(f"Avg Final Resource: {df['final_resource'].mean():.1f}")
    report.append("")

    # By experiment
    report.append("BY EXPERIMENT:")
    for exp in df['experiment'].unique():
        exp_df = df[df['experiment'] == exp]
        report.append(f"\n{exp}:")
        report.append(f"  Games: {len(exp_df)}")
        report.append(f"  Tragedy Rate: {exp_df['tragedy'].mean():.1%}")
        report.append(f"  Avg Cooperation: {exp_df['cooperation'].mean():.3f}")

    report.append("")
    report.append("=" * 70)

    return "\n".join(report)


def visualize_all_experiments(results_dir: str = None):
    """Create comprehensive visualization of all experiments.

    Args:
        results_dir: Directory containing result files
    """
    if results_dir is None:
        results_dir = Path(__file__).parent

    results_dir = Path(results_dir)
    result_files = list(results_dir.glob("results_*.json"))

    if not result_files:
        print("No result files found")
        return

    print(f"Found {len(result_files)} result files")

    # Load and combine all results
    df = compare_metrics_across_experiments([str(f) for f in result_files])

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Tragedy rates
    tragedy_by_exp = df.groupby('experiment')['tragedy'].mean()
    axes[0, 0].bar(range(len(tragedy_by_exp)), tragedy_by_exp.values, color='coral')
    axes[0, 0].set_xticks(range(len(tragedy_by_exp)))
    axes[0, 0].set_xticklabels(tragedy_by_exp.index, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Tragedy Rate')
    axes[0, 0].set_title('Tragedy Rates by Experiment')
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Cooperation
    axes[0, 1].boxplot([df[df['experiment'] == exp]['cooperation'].values
                        for exp in df['experiment'].unique()],
                       labels=df['experiment'].unique())
    axes[0, 1].set_ylabel('Cooperation Index')
    axes[0, 1].set_title('Cooperation Distribution')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Rounds
    axes[1, 0].scatter(df['cooperation'], df['rounds'], alpha=0.6, c=df['tragedy'].astype(int), cmap='RdYlGn_r')
    axes[1, 0].set_xlabel('Cooperation Index')
    axes[1, 0].set_ylabel('Rounds Played')
    axes[1, 0].set_title('Cooperation vs Game Length')
    axes[1, 0].grid(True, alpha=0.3)

    # Final resource
    axes[1, 1].scatter(df['cooperation'], df['final_resource'], alpha=0.6)
    axes[1, 1].set_xlabel('Cooperation Index')
    axes[1, 1].set_ylabel('Final Resource Level')
    axes[1, 1].set_title('Cooperation vs Resource Survival')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = results_dir / "analysis_summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")

    # Print summary report
    report = create_summary_report(df)
    print("\n" + report)

    # Save report
    report_path = results_dir / "analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    return df


if __name__ == "__main__":
    print("Running analysis on experiment results...\n")
    df = visualize_all_experiments()

    if df is not None:
        print("\n✓ Analysis complete!")
        print(f"Analyzed {len(df)} games across {df['experiment'].nunique()} experiments")
