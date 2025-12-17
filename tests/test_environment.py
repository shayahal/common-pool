"""Unit tests for CPR Environment."""

import pytest
import numpy as np
from cpr_game.cpr_environment import CPREnvironment
from cpr_game.config import CONFIG


class TestCPREnvironment:
    """Test cases for CPR environment."""

    @pytest.fixture
    def env(self):
        """Create test environment."""
        config = CONFIG.copy()
        config["max_steps"] = 10
        config["n_players"] = 2
        config["initial_resource"] = 1000.0
        config["regeneration_rate"] = 2.0
        config["sustainability_threshold"] = 500.0
        config["sustainability_bonus"] = 10.0
        config["depletion_penalty"] = -1000.0
        config["max_fishes"] = float("inf")  # Remove capacity cap for tests
        config["max_extraction"] = 2000.0  # Allow large extractions for depletion tests
        config["min_resource"] = 0.0  # Allow resource to go to 0
        return CPREnvironment(config)

    def test_initialization(self, env):
        """Test environment initialization."""
        assert env.n_players == 2
        assert env.max_steps == 10
        assert env.initial_resource == 1000.0
        assert env.regeneration_rate == 2.0

    def test_reset(self, env):
        """Test environment reset."""
        obs, info = env.reset()

        assert env.current_resource == env.initial_resource
        assert env.current_step == 0
        assert len(env.extraction_history) == 0
        assert len(env.resource_history) == 1
        assert env.done is False

        # Check observations structure
        assert "player_0" in obs
        assert "player_1" in obs
        assert "resource_level" in obs["player_0"]

    def test_step_basic(self, env):
        """Test basic step execution."""
        env.reset()

        # Take actions
        actions = np.array([10.0, 15.0])
        obs, rewards, terminated, truncated, info = env.step(actions)

        # Check resource dynamics: R(t+1) = R(t) * 2.0 - total_extraction
        expected_resource = 1000.0 * 2.0 - 25.0  # 2000 - 25 = 1975
        assert env.current_resource == expected_resource

        # Check rewards (should include sustainability bonus since resource is above threshold)
        assert len(rewards) == 2
        # Resource after step is 1975, which is above threshold 500, so bonus applies
        assert rewards[0] == 10.0 + env.sustainability_bonus  # extraction + bonus
        assert rewards[1] == 15.0 + env.sustainability_bonus

        # Check step counter
        assert env.current_step == 1

        # Not done yet
        assert not terminated
        assert not truncated

    def test_resource_regeneration(self, env):
        """Test resource regeneration mechanics."""
        env.reset()

        # Extract nothing
        actions = np.array([0.0, 0.0])
        env.step(actions)

        # Resource should double
        expected = 1000.0 * 2.0
        assert env.current_resource == expected

    def test_resource_depletion(self, env):
        """Test resource depletion termination."""
        env.reset()

        # Extract more than regenerated resource
        # After reset: resource = 1000
        # After regeneration: resource = 1000 * 2 = 2000
        # Extract 2000 to deplete it
        actions = np.array([1000.0, 1000.0])  # Total: 2000, exactly the regenerated amount
        obs, rewards, terminated, truncated, info = env.step(actions)

        # Should be depleted (2000 - 2000 = 0)
        assert env.current_resource == 0.0
        assert terminated is True
        assert bool(info["tragedy_occurred"]) is True

        # Should receive depletion penalty
        penalty_per_player = env.depletion_penalty / env.n_players
        assert penalty_per_player < 0

    def test_sustainability_bonus(self, env):
        """Test sustainability bonus."""
        env.reset()

        # Extract small amount to stay above threshold
        actions = np.array([10.0, 10.0])
        obs, rewards, terminated, truncated, info = env.step(actions)

        # Resource should be well above threshold (1000 * 2 - 20 = 1980)
        assert env.current_resource > env.sustainability_threshold

        # Should receive sustainability bonus
        expected_reward = 10.0 + env.sustainability_bonus
        assert rewards[0] == expected_reward

    def test_max_steps_truncation(self, env):
        """Test max steps truncation."""
        env.reset()

        # Run for max_steps
        for _ in range(env.max_steps):
            actions = np.array([5.0, 5.0])
            obs, rewards, terminated, truncated, info = env.step(actions)

        # Should be truncated
        assert truncated is True

    def test_action_clipping(self, env):
        """Test that actions are clipped to valid range."""
        env.reset()

        # Try invalid actions (use values outside the configured max_extraction)
        actions = np.array([-10.0, env.max_extraction + 100.0])  # Below min and above max
        obs, rewards, terminated, truncated, info = env.step(actions)

        # Should be clipped
        assert env.extraction_history[0][0] == env.min_extraction
        assert env.extraction_history[0][1] == env.max_extraction

    def test_observations_structure(self, env):
        """Test observation space structure."""
        obs, _ = env.reset()

        player_0_obs = obs["player_0"]

        # Check all required keys
        assert "resource_level" in player_0_obs
        assert "step" in player_0_obs
        assert "my_recent_extractions" in player_0_obs
        assert "other_players_recent_extractions" in player_0_obs
        assert "my_cumulative_payoff" in player_0_obs
        assert "other_players_cumulative_payoffs" in player_0_obs

        # Check shapes
        assert player_0_obs["resource_level"].shape == (1,)
        assert player_0_obs["step"].shape == (1,)
        assert len(player_0_obs["my_recent_extractions"]) == env.config["include_history_rounds"]

    def test_cumulative_payoffs(self, env):
        """Test cumulative payoff tracking."""
        env.reset()

        # Take several actions
        for _ in range(3):
            actions = np.array([10.0, 10.0])
            env.step(actions)

        # Check cumulative payoffs
        # Each round: 10 (extraction) + 10 (sustainability bonus) = 20
        expected_total = 20.0 * 3
        np.testing.assert_almost_equal(
            env.player_cumulative_payoffs[0],
            expected_total,
            decimal=2
        )

    def test_cooperation_index_tracking(self, env):
        """Test cooperation index calculation."""
        env.reset()

        # Similar extractions (high cooperation)
        actions = np.array([10.0, 11.0])
        env.step(actions)

        assert len(env.cooperation_history) == 1
        assert env.cooperation_history[0] > 0.9  # High cooperation

        # Very different extractions (low cooperation)
        actions = np.array([0.0, 100.0])
        env.step(actions)

        assert env.cooperation_history[1] < 0.5  # Low cooperation

    def test_render_human(self, env):
        """Test human-readable rendering."""
        env.reset()
        env.step(np.array([10.0, 15.0]))

        output = env.render(mode="human")

        assert isinstance(output, str)
        assert "Resource Level" in output
        assert "Player 0" in output

    def test_render_dict(self, env):
        """Test dictionary rendering."""
        env.reset()
        env.step(np.array([10.0, 15.0]))

        output = env.render(mode="dict")

        assert isinstance(output, dict)
        assert "resource" in output
        assert "step" in output
        assert "resource_history" in output
        assert "extraction_history" in output

    def test_summary_stats(self, env):
        """Test summary statistics."""
        env.reset()

        # Play a few rounds
        for _ in range(5):
            env.step(np.array([10.0, 15.0]))

        summary = env.get_summary_stats()

        assert "total_rounds" in summary
        assert "final_resource_level" in summary
        assert "tragedy_occurred" in summary
        assert "avg_cooperation_index" in summary
        assert "gini_coefficient" in summary
        assert "sustainability_score" in summary

        assert summary["total_rounds"] == 5

    def test_episode_after_done(self, env):
        """Test that step after done raises error."""
        env.reset()

        # Deplete resource by extracting more than available
        # Resource after reset: 1000, after regeneration: 2000
        # Extract 2000 to deplete
        env.step(np.array([1000.0, 1000.0]))
        
        # Verify it's done
        assert env.done is True

        # Try to step again - should raise RuntimeError
        with pytest.raises(RuntimeError):
            env.step(np.array([10.0, 10.0]))

    def test_multiple_episodes(self, env):
        """Test multiple episodes."""
        # First episode
        env.reset()
        for _ in range(5):
            env.step(np.array([10.0, 10.0]))

        # Reset for second episode
        obs, info = env.reset()

        assert env.current_step == 0
        assert env.current_resource == env.initial_resource
        assert len(env.extraction_history) == 0
