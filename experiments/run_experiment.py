"""Run CPR game experiments with different configurations.

This script demonstrates how to run systematic experiments exploring
different game parameters and agent configurations.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cpr_game.game_runner import GameRunner
from cpr_game.config import CONFIG
import json
import numpy as np


def experiment_persona_comparison(n_games: int = 5):
    """Compare different persona matchups.

    Args:
        n_games: Number of games per matchup
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT: Persona Comparison")
    print("=" * 70 + "\n")

    matchups = [
        ("rational_selfish", "rational_selfish"),
        ("cooperative", "cooperative"),
        ("rational_selfish", "cooperative"),
    ]

    results = {}

    for persona_0, persona_1 in matchups:
        print(f"\nMatchup: {persona_0} vs {persona_1}")
        print("-" * 50)

        config = CONFIG.copy()
        config["player_personas"] = [persona_0, persona_1]
        config["max_steps"] = 50

        runner = GameRunner(config, use_mock_agents=True, use_mock_logging=True)

        matchup_results = []
        for i in range(n_games):
            runner.setup_game(f"{persona_0}_vs_{persona_1}_game_{i}")
            summary = runner.run_episode(visualize=False, verbose=False)
            matchup_results.append(summary)

        # Aggregate
        tragedy_rate = sum(1 for r in matchup_results if r['tragedy_occurred']) / n_games
        avg_cooperation = np.mean([r['avg_cooperation_index'] for r in matchup_results])
        avg_rounds = np.mean([r['total_rounds'] for r in matchup_results])

        results[f"{persona_0}_vs_{persona_1}"] = {
            "tragedy_rate": tragedy_rate,
            "avg_cooperation": avg_cooperation,
            "avg_rounds": avg_rounds,
            "games": matchup_results,
        }

        print(f"Tragedy Rate: {tragedy_rate:.1%}")
        print(f"Avg Cooperation: {avg_cooperation:.3f}")
        print(f"Avg Rounds: {avg_rounds:.1f}")

    # Save results
    output_path = Path(__file__).parent / "results_persona_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Results saved to {output_path}")

    return results


def experiment_regeneration_rate(rates: list = None):
    """Test different regeneration rates.

    Args:
        rates: List of regeneration rates to test
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT: Regeneration Rate Sensitivity")
    print("=" * 70 + "\n")

    if rates is None:
        rates = [1.5, 2.0, 2.5, 3.0]

    results = {}

    for rate in rates:
        print(f"\nRegeneration Rate: {rate}")
        print("-" * 50)

        config = CONFIG.copy()
        config["regeneration_rate"] = rate
        config["max_steps"] = 50

        runner = GameRunner(config, use_mock_agents=True, use_mock_logging=True)
        runner.setup_game(f"regen_rate_{rate}")
        summary = runner.run_episode(visualize=False, verbose=False)

        results[f"rate_{rate}"] = summary

        print(f"Tragedy: {'Yes' if summary['tragedy_occurred'] else 'No'}")
        print(f"Final Resource: {summary['final_resource_level']:.1f}")
        print(f"Avg Cooperation: {summary['avg_cooperation_index']:.3f}")

    # Save results
    output_path = Path(__file__).parent / "results_regeneration_rates.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Results saved to {output_path}")

    return results


def experiment_player_count(n_players_list: list = None):
    """Test different numbers of players.

    Args:
        n_players_list: List of player counts to test
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT: Player Count Scaling")
    print("=" * 70 + "\n")

    if n_players_list is None:
        n_players_list = [2, 3, 4]

    results = {}

    for n_players in n_players_list:
        print(f"\nNumber of Players: {n_players}")
        print("-" * 50)

        config = CONFIG.copy()
        config["n_players"] = n_players
        config["max_steps"] = 50

        # Extend personas
        config["player_personas"] = ["cooperative"] * n_players

        runner = GameRunner(config, use_mock_agents=True, use_mock_logging=True)
        runner.setup_game(f"n_players_{n_players}")
        summary = runner.run_episode(visualize=False, verbose=False)

        results[f"{n_players}_players"] = summary

        print(f"Tragedy: {'Yes' if summary['tragedy_occurred'] else 'No'}")
        print(f"Final Resource: {summary['final_resource_level']:.1f}")
        print(f"Avg Cooperation: {summary['avg_cooperation_index']:.3f}")

    # Save results
    output_path = Path(__file__).parent / "results_player_count.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Results saved to {output_path}")

    return results


def experiment_sustainability_threshold(thresholds: list = None):
    """Test different sustainability thresholds.

    Args:
        thresholds: List of threshold values to test
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT: Sustainability Threshold")
    print("=" * 70 + "\n")

    if thresholds is None:
        thresholds = [300.0, 500.0, 700.0, 900.0]

    results = {}

    for threshold in thresholds:
        print(f"\nSustainability Threshold: {threshold}")
        print("-" * 50)

        config = CONFIG.copy()
        config["sustainability_threshold"] = threshold
        config["max_steps"] = 50

        runner = GameRunner(config, use_mock_agents=True, use_mock_logging=True)
        runner.setup_game(f"threshold_{threshold}")
        summary = runner.run_episode(visualize=False, verbose=False)

        results[f"threshold_{threshold}"] = summary

        sustainability_score = summary.get('sustainability_score', 0.0)
        if sustainability_score > 0:
            print(f"Sustainability Score: {sustainability_score:.1%}")
        print(f"Final Resource: {summary['final_resource_level']:.1f}")

    # Save results
    output_path = Path(__file__).parent / "results_thresholds.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Results saved to {output_path}")

    return results


def experiment_null_personas(n_games: int = 5, n_players: int = 2):
    """Run experiment with null personas (no personality traits).

    Args:
        n_games: Number of games to run
        n_players: Number of players
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT: Null Personas")
    print("=" * 70 + "\n")

    config = CONFIG.copy()
    config["n_players"] = n_players
    config["player_personas"] = [""] * n_players  # Empty string = null persona
    config["max_steps"] = 50

    runner = GameRunner(config, use_mock_agents=True, use_mock_logging=True)

    results = []
    for i in range(n_games):
        print(f"\nGame {i + 1}/{n_games}")
        print("-" * 50)
        
        runner.setup_game(f"null_personas_game_{i}")
        summary = runner.run_episode(visualize=False, verbose=False)
        results.append(summary)

        print(f"Tragedy: {'Yes' if summary['tragedy_occurred'] else 'No'}")
        print(f"Rounds: {summary['total_rounds']}")
        print(f"Final Resource: {summary['final_resource_level']:.1f}")
        print(f"Avg Cooperation: {summary['avg_cooperation_index']:.3f}")

    # Aggregate statistics
    tragedy_rate = sum(1 for r in results if r['tragedy_occurred']) / n_games
    avg_cooperation = np.mean([r['avg_cooperation_index'] for r in results])
    avg_rounds = np.mean([r['total_rounds'] for r in results])
    avg_final_resource = np.mean([r['final_resource_level'] for r in results])
    avg_gini = np.mean([r['gini_coefficient'] for r in results])

    aggregated = {
        "tragedy_rate": float(tragedy_rate),
        "avg_cooperation": float(avg_cooperation),
        "avg_rounds": float(avg_rounds),
        "avg_final_resource": float(avg_final_resource),
        "avg_gini_coefficient": float(avg_gini),
        "n_games": n_games,
        "n_players": n_players,
        "games": results,
    }

    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS")
    print("=" * 70)
    print(f"Tragedy Rate: {tragedy_rate:.1%}")
    print(f"Avg Cooperation: {avg_cooperation:.3f}")
    print(f"Avg Rounds: {avg_rounds:.1f}")
    print(f"Avg Final Resource: {avg_final_resource:.1f}")
    print(f"Avg Gini Coefficient: {avg_gini:.3f}")

    # Save results
    output_path = Path(__file__).parent / "results_null_personas.json"
    with open(output_path, 'w') as f:
        json.dump(aggregated, f, indent=2, default=str)

    print(f"\n✓ Results saved to {output_path}")

    return aggregated


def main():
    """Run all experiments."""
    print("\n" + "=" * 70)
    print("RUNNING ALL CPR EXPERIMENTS")
    print("=" * 70)

    # Run experiments
    experiment_persona_comparison(n_games=3)
    experiment_regeneration_rate()
    experiment_player_count()
    experiment_sustainability_threshold()
    experiment_null_personas(n_games=5)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Allow running specific experiments
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "null":
            experiment_null_personas(n_games=5)
        elif sys.argv[1] == "personas":
            experiment_persona_comparison(n_games=5)
        elif sys.argv[1] == "all":
            main()
        else:
            print(f"Unknown experiment: {sys.argv[1]}")
            print("Available: null, personas, all")
    else:
        main()


if __name__ == "__main__":
    main()
