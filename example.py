#!/usr/bin/env python
"""Example usage of the CPR game environment.

This script demonstrates basic usage patterns for running games,
tournaments, and analyzing results.
"""

from cpr_game import GameRunner
from cpr_game.config import CONFIG
import json
import logging

# Setup logging for error reporting
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def example_1_basic_game():
    """Run a basic game with mock agents."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Game with Mock Agents")
    print("=" * 70 + "\n")

    # Create runner with mock agents (no API keys needed)
    runner = GameRunner(use_mock_agents=True)

    # Setup and run
    runner.setup_game("example_basic")
    summary = runner.run_episode(verbose=True)

    # Print key results
    print("\n📊 Key Results:")
    print(f"  Tragedy occurred: {summary['tragedy_occurred']}")
    print(f"  Final resource: {summary['final_resource_level']:.1f}")
    print(f"  Cooperation index: {summary['avg_cooperation_index']:.3f}")
    print(f"  Gini coefficient: {summary['gini_coefficient']:.3f}")


def example_2_custom_config():
    """Run a game with custom configuration."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Custom Configuration")
    print("=" * 70 + "\n")

    # Customize config
    custom_config = CONFIG.copy()
    custom_config['max_steps'] = 30
    custom_config['regeneration_rate'] = 1.5
    custom_config['initial_resource'] = 500.0

    print(f"Custom settings:")
    print(f"  Max steps: {custom_config['max_steps']}")
    print(f"  Regeneration rate: {custom_config['regeneration_rate']}")
    print(f"  Initial resource: {custom_config['initial_resource']}")
    print()

    runner = GameRunner(custom_config, use_mock_agents=True)
    runner.setup_game("example_custom")
    summary = runner.run_episode(verbose=False)

    print(f"\n✓ Game completed in {summary['total_rounds']} rounds")


def example_3_tournament():
    """Run a tournament of multiple games."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Tournament Mode")
    print("=" * 70 + "\n")

    runner = GameRunner(use_mock_agents=True)

    # Run 5 games
    results = runner.run_tournament(n_games=5, verbose=False)

    # Analyze results
    tragedy_count = sum(1 for r in results if r['tragedy_occurred'])
    print(f"\n📊 Tournament Results:")
    print(f"  Games played: {len(results)}")
    print(f"  Tragedies: {tragedy_count} ({tragedy_count/len(results):.1%})")
    print(f"  Resources survived: {len(results) - tragedy_count}")


def example_4_persona_comparison():
    """Compare different persona matchups."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Persona Comparison")
    print("=" * 70 + "\n")

    matchups = [
        ("rational_selfish", "rational_selfish"),
        ("cooperative", "cooperative"),
        ("rational_selfish", "cooperative"),
    ]

    results = {}

    for persona_0, persona_1 in matchups:
        config = CONFIG.copy()
        config['player_personas'] = [persona_0, persona_1]
        config['max_steps'] = 50

        runner = GameRunner(config, use_mock_agents=True)
        runner.setup_game(f"{persona_0}_vs_{persona_1}")
        summary = runner.run_episode(verbose=False)

        results[f"{persona_0} vs {persona_1}"] = summary

        print(f"\n{persona_0} vs {persona_1}:")
        print(f"  Rounds: {summary['total_rounds']}")
        print(f"  Tragedy: {'Yes' if summary['tragedy_occurred'] else 'No'}")
        print(f"  Cooperation: {summary['avg_cooperation_index']:.3f}")


def example_5_export_results():
    """Run games and export results."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Export Results")
    print("=" * 70 + "\n")

    runner = GameRunner(use_mock_agents=True)

    # Run multiple games
    for i in range(3):
        runner.setup_game(f"export_game_{i}")
        runner.run_episode(verbose=False)
        print(f"✓ Completed game {i+1}/3")

    # Export to JSON
    output_file = "example_results.json"
    runner.export_results(output_file)

    print(f"\n✓ Results exported to {output_file}")

    # Load and display
    with open(output_file, 'r') as f:
        data = json.load(f)

    print(f"  Games in export: {len(data['games'])}")


def example_6_environment_only():
    """Use the environment directly without GameRunner."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Direct Environment Usage")
    print("=" * 70 + "\n")

    from cpr_game.cpr_environment import CPREnvironment
    import numpy as np

    # Create environment
    env = CPREnvironment(CONFIG)
    obs, info = env.reset()

    print("Running 10 rounds with random actions...\n")

    for i in range(10):
        # Random actions
        actions = np.random.uniform(0, 50, size=env.n_players)

        # Step
        obs, rewards, terminated, truncated, info = env.step(actions)

        print(f"Round {i+1}: Resource={info['resource']:.1f}, "
              f"Extractions={actions}, Rewards={rewards}")

        if terminated or truncated:
            break

    summary = env.get_summary_stats()
    print(f"\n✓ Final resource: {summary['final_resource_level']:.1f}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("CPR GAME - EXAMPLE USAGE")
    print("=" * 80)

    examples = [
        ("Basic Game", example_1_basic_game),
        ("Custom Config", example_2_custom_config),
        ("Tournament", example_3_tournament),
        ("Persona Comparison", example_4_persona_comparison),
        ("Export Results", example_5_export_results),
        ("Direct Environment", example_6_environment_only),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nRunning all examples...\n")

    for name, func in examples:
        try:
            func()
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Error in {name}: {type(e).__name__}: {e}", exc_info=True)
            print(f"\n❌ Error in {name}: {type(e).__name__}: {e}")
        except (KeyError, IndexError) as e:
            logger.error(f"Configuration error in {name}: {type(e).__name__}: {e}", exc_info=True)
            print(f"\n❌ Configuration error in {name}: {type(e).__name__}: {e}")
        except Exception as e:
            # Catch any other unexpected exceptions
            logger.error(f"Unexpected error in {name}: {type(e).__name__}: {e}", exc_info=True)
            print(f"\n❌ Unexpected error in {name}: {type(e).__name__}: {e}")

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
