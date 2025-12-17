"""Tests for all simple game examples from simple_example.py.

Tests cover:
1. Basic 20-round Game
2. Shorter Game with Custom Settings
3. Tournament (5 games)
4. Using Environment Directly
"""

import pytest
import numpy as np
from cpr_game import GameRunner
from cpr_game.cpr_environment import CPREnvironment
from cpr_game.config import CONFIG
from cpr_game.utils import compute_sustainability_score


class TestSimpleGame1_BasicGame:
    """Test Example 1: Basic 20-round Game."""

    def test_basic_game_setup(self):
        """Test that basic game can be set up."""
        runner = GameRunner(use_mock_agents=True, use_mock_logging=True)
        game_id = runner.setup_game("demo_game")
        
        assert game_id == "demo_game"
        assert runner.env is not None
        assert len(runner.agents) == 2
        assert runner.logger is not None

    def test_basic_game_run(self):
        """Test that basic game runs to completion."""
        runner = GameRunner(use_mock_agents=True, use_mock_logging=True)
        runner.setup_game("demo_game")
        
        summary = runner.run_episode(visualize=False, verbose=False)
        
        # Check summary structure
        assert "total_rounds" in summary
        assert "final_resource_level" in summary
        assert "tragedy_occurred" in summary
        assert "avg_cooperation_index" in summary
        assert "gini_coefficient" in summary
        assert "cumulative_payoffs" in summary
        
        # Check values are reasonable
        assert summary["total_rounds"] > 0
        assert summary["total_rounds"] <= CONFIG["max_steps"]
        assert isinstance(summary["tragedy_occurred"], bool)
        assert 0.0 <= summary["avg_cooperation_index"] <= 1.0
        assert 0.0 <= summary["gini_coefficient"] <= 1.0
        assert len(summary["cumulative_payoffs"]) == 2

    def test_basic_game_results_structure(self):
        """Test that basic game results have expected structure."""
        runner = GameRunner(use_mock_agents=True, use_mock_logging=True)
        runner.setup_game("demo_game")
        summary = runner.run_episode(visualize=False, verbose=False)
        
        # Check all expected keys are present
        expected_keys = [
            "total_rounds",
            "final_resource_level",
            "cumulative_payoffs",
            "tragedy_occurred",
            "avg_cooperation_index",
            "gini_coefficient",
            "total_extracted",
            "avg_extraction_per_player",
        ]
        
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"

    def test_basic_game_payoffs_are_positive(self):
        """Test that players receive positive payoffs."""
        runner = GameRunner(use_mock_agents=True, use_mock_logging=True)
        runner.setup_game("demo_game")
        summary = runner.run_episode(visualize=False, verbose=False)
        
        # Payoffs should be non-negative (at least 0)
        for payoff in summary["cumulative_payoffs"]:
            assert payoff >= 0.0


class TestSimpleGame2_CustomSettings:
    """Test Example 2: Shorter Game with Custom Settings."""

    def test_custom_settings_game_setup(self):
        """Test that custom settings game can be set up."""
        config = CONFIG.copy()
        config['max_steps'] = 30
        config['regeneration_rate'] = 1.5
        config['initial_resource'] = 500
        
        runner = GameRunner(config, use_mock_agents=True, use_mock_logging=True)
        game_id = runner.setup_game("custom_game")
        
        assert game_id == "custom_game"
        assert runner.env.max_steps == 30
        assert runner.env.regeneration_rate == 1.5
        assert runner.env.initial_resource == 500

    def test_custom_settings_game_run(self):
        """Test that custom settings game runs correctly."""
        config = CONFIG.copy()
        config['max_steps'] = 30
        config['regeneration_rate'] = 1.5
        config['initial_resource'] = 500
        
        runner = GameRunner(config, use_mock_agents=True, use_mock_logging=True)
        runner.setup_game("custom_game")
        summary = runner.run_episode(visualize=False, verbose=False)
        
        # Check that game respects custom settings
        assert summary["total_rounds"] <= 30
        assert runner.env.initial_resource == 500
        assert runner.env.regeneration_rate == 1.5

    def test_custom_settings_affect_gameplay(self):
        """Test that custom settings actually affect gameplay."""
        # Run with default settings
        runner1 = GameRunner(use_mock_agents=True, use_mock_logging=True)
        runner1.setup_game("default_game")
        summary1 = runner1.run_episode(visualize=False, verbose=False)
        
        # Run with custom settings (shorter, different regen)
        config = CONFIG.copy()
        config['max_steps'] = 10
        config['regeneration_rate'] = 1.2
        config['initial_resource'] = 200
        
        runner2 = GameRunner(config, use_mock_agents=True, use_mock_logging=True)
        runner2.setup_game("custom_game")
        summary2 = runner2.run_episode(visualize=False, verbose=False)
        
        # Games should be different
        # At minimum, total_rounds should be different if one is shorter
        if summary1["total_rounds"] > 10:
            assert summary2["total_rounds"] <= 10


class TestSimpleGame3_Tournament:
    """Test Example 3: Tournament (5 games)."""

    def test_tournament_setup(self):
        """Test that tournament can be set up."""
        runner = GameRunner(use_mock_agents=True, use_mock_logging=True)
        runner.setup_game("tournament_0")
        
        assert runner.env is not None
        assert len(runner.agents) == 2

    def test_tournament_runs_multiple_games(self):
        """Test that tournament runs multiple games."""
        runner = GameRunner(use_mock_agents=True, use_mock_logging=True)
        results = []
        
        for i in range(5):
            runner.setup_game(f"tournament_{i}")
            result = runner.run_episode(visualize=False, verbose=False)
            results.append(result)
        
        # Should have 5 results
        assert len(results) == 5
        
        # Each result should be valid
        for result in results:
            assert "total_rounds" in result
            assert "tragedy_occurred" in result
            assert "avg_cooperation_index" in result
            assert result["total_rounds"] > 0

    def test_tournament_aggregate_stats(self):
        """Test tournament aggregate statistics."""
        runner = GameRunner(use_mock_agents=True, use_mock_logging=True)
        results = []
        
        for i in range(5):
            runner.setup_game(f"tournament_{i}")
            result = runner.run_episode(visualize=False, verbose=False)
            results.append(result)
        
        # Calculate aggregate stats
        tragedy_rate = sum(1 for r in results if r['tragedy_occurred']) / len(results)
        avg_rounds = np.mean([r['total_rounds'] for r in results])
        avg_cooperation = np.mean([r['avg_cooperation_index'] for r in results])
        avg_gini = np.mean([r['gini_coefficient'] for r in results])
        
        # Check stats are valid
        assert 0.0 <= tragedy_rate <= 1.0
        assert avg_rounds > 0
        assert 0.0 <= avg_cooperation <= 1.0
        assert 0.0 <= avg_gini <= 1.0

    def test_tournament_consistency(self):
        """Test that tournament games are consistent."""
        runner = GameRunner(use_mock_agents=True, use_mock_logging=True)
        results = []
        
        for i in range(3):  # Smaller number for faster test
            runner.setup_game(f"tournament_{i}")
            result = runner.run_episode(visualize=False, verbose=False)
            results.append(result)
        
        # All games should have same structure
        first_keys = set(results[0].keys())
        for result in results[1:]:
            assert set(result.keys()) == first_keys

    def test_run_tournament_method(self):
        """Test the run_tournament method."""
        runner = GameRunner(use_mock_agents=True, use_mock_logging=True)
        results = runner.run_tournament(n_games=3, verbose=False)
        
        assert len(results) == 3
        for result in results:
            assert "total_rounds" in result
            assert "tragedy_occurred" in result


class TestSimpleGame4_DirectEnvironment:
    """Test Example 4: Using Environment Directly."""

    def test_environment_initialization(self):
        """Test that environment can be initialized directly."""
        env = CPREnvironment()
        obs, info = env.reset()
        
        assert env.current_resource == env.initial_resource
        assert env.current_step == 0
        assert "resource" in info
        assert info["resource"] == env.initial_resource

    def test_environment_reset(self):
        """Test environment reset functionality."""
        env = CPREnvironment()
        obs1, info1 = env.reset()
        
        # Take some steps
        for _ in range(3):
            actions = np.array([10.0, 15.0])
            env.step(actions)
        
        # Reset
        obs2, info2 = env.reset()
        
        # Should be back to initial state
        assert env.current_resource == env.initial_resource
        assert env.current_step == 0
        assert len(env.extraction_history) == 0
        assert info2["resource"] == env.initial_resource

    def test_environment_step_with_random_actions(self):
        """Test environment step with random actions."""
        env = CPREnvironment()
        obs, info = env.reset()
        
        initial_resource = info['resource']
        
        # Run 5 rounds with random actions
        for i in range(5):
            actions = np.random.uniform(0, 30, size=2)
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Check that resource is updated
            assert "resource" in info
            assert isinstance(info["resource"], (int, float))
            
            # Check that rewards are returned
            assert len(rewards) == 2
            assert all(r >= 0 for r in rewards)
            
            # Check cooperation index
            assert "cooperation_index" in info
            assert 0.0 <= info["cooperation_index"] <= 1.0
            
            if terminated or truncated:
                break
        
        # Should have made some progress
        assert env.current_step > 0

    def test_environment_resource_dynamics(self):
        """Test that resource dynamics work correctly."""
        env = CPREnvironment()
        obs, info = env.reset()
        
        initial_resource = env.current_resource
        
        # Extract nothing - resource should regenerate
        actions = np.array([0.0, 0.0])
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # Resource should have regenerated (doubled by default)
        expected_resource = min(initial_resource * env.regeneration_rate, env.max_fishes)
        assert env.current_resource == expected_resource

    def test_environment_observations_structure(self):
        """Test that observations have correct structure."""
        env = CPREnvironment()
        obs, info = env.reset()
        
        # Check observations for each player
        for i in range(env.n_players):
            player_key = f"player_{i}"
            assert player_key in obs
            
            player_obs = obs[player_key]
            assert "resource_level" in player_obs
            assert "step" in player_obs
            assert "my_recent_extractions" in player_obs
            assert "other_players_recent_extractions" in player_obs
            assert "my_cumulative_payoff" in player_obs
            assert "other_players_cumulative_payoffs" in player_obs

    def test_environment_termination_conditions(self):
        """Test environment termination conditions."""
        env = CPREnvironment()
        obs, info = env.reset()
        
        # Test max steps truncation
        config = CONFIG.copy()
        config["max_steps"] = 5
        env_short = CPREnvironment(config)
        obs, info = env_short.reset()
        
        for _ in range(10):  # More than max_steps
            actions = np.array([5.0, 5.0])
            obs, rewards, terminated, truncated, info = env_short.step(actions)
            if truncated:
                break
        
        assert truncated is True or env_short.current_step >= 5

    def test_environment_summary_stats(self):
        """Test environment summary statistics."""
        env = CPREnvironment()
        obs, info = env.reset()
        
        # Play a few rounds
        for _ in range(5):
            actions = np.array([10.0, 15.0])
            obs, rewards, terminated, truncated, info = env.step(actions)
            if terminated or truncated:
                break
        
        summary = env.get_summary_stats()
        
        # Check summary structure
        assert "total_rounds" in summary
        assert "final_resource_level" in summary
        assert "tragedy_occurred" in summary
        assert "avg_cooperation_index" in summary
        assert "gini_coefficient" in summary
        assert "cumulative_payoffs" in summary
        assert "total_extracted" in summary
        assert "avg_extraction_per_player" in summary
        
        # Check values
        assert summary["total_rounds"] == env.current_step
        assert summary["final_resource_level"] == env.current_resource
        assert len(summary["cumulative_payoffs"]) == env.n_players


class TestSimpleGamesIntegration:
    """Integration tests across all simple game examples."""

    def test_all_games_use_same_config_structure(self):
        """Test that all games use consistent config structure."""
        # Basic game
        runner1 = GameRunner(use_mock_agents=True, use_mock_logging=True)
        runner1.setup_game("game1")
        
        # Custom game
        config = CONFIG.copy()
        config['max_steps'] = 20
        runner2 = GameRunner(config, use_mock_agents=True, use_mock_logging=True)
        runner2.setup_game("game2")
        
        # Direct environment
        env = CPREnvironment()
        
        # All should have same basic structure
        assert runner1.env.n_players == runner2.env.n_players == env.n_players
        assert runner1.env.min_extraction == runner2.env.min_extraction == env.min_extraction
        assert runner1.env.max_extraction == runner2.env.max_extraction == env.max_extraction

    def test_all_games_produce_valid_summaries(self):
        """Test that all game types produce valid summaries."""
        # GameRunner basic
        runner1 = GameRunner(use_mock_agents=True, use_mock_logging=True)
        runner1.setup_game("test1")
        summary1 = runner1.run_episode(visualize=False, verbose=False)
        
        # GameRunner custom
        config = CONFIG.copy()
        config['max_steps'] = 10
        runner2 = GameRunner(config, use_mock_agents=True, use_mock_logging=True)
        runner2.setup_game("test2")
        summary2 = runner2.run_episode(visualize=False, verbose=False)
        
        # Direct environment
        env = CPREnvironment()
        env.reset()
        for _ in range(5):
            actions = np.array([10.0, 10.0])
            env.step(actions)
        summary3 = env.get_summary_stats()
        
        # All summaries should have same keys
        keys1 = set(summary1.keys())
        keys2 = set(summary2.keys())
        keys3 = set(summary3.keys())
        
        # Core keys should be present in all
        core_keys = {
            "total_rounds",
            "final_resource_level",
            "cumulative_payoffs",
            "tragedy_occurred",
            "avg_cooperation_index",
            "gini_coefficient",
        }
        
        for keys in [keys1, keys2, keys3]:
            assert core_keys.issubset(keys)

    def test_sustainability_score_calculation(self):
        """Test sustainability score calculation if needed."""
        env = CPREnvironment()
        obs, info = env.reset()
        
        # Play some rounds
        for _ in range(10):
            actions = np.array([5.0, 5.0])  # Small extractions
            obs, rewards, terminated, truncated, info = env.step(actions)
            if terminated or truncated:
                break
        
        # Calculate sustainability score manually
        threshold = 50.0  # Example threshold
        sustainability_score = compute_sustainability_score(
            env.resource_history,
            threshold
        )
        
        assert 0.0 <= sustainability_score <= 1.0

