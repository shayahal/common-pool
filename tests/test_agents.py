"""Unit tests for LLM agents."""

import pytest
import numpy as np
from cpr_game.llm_agent import MockLLMAgent
from cpr_game.config import CONFIG


class TestMockLLMAgent:
    """Test cases for MockLLMAgent."""

    @pytest.fixture
    def agent(self):
        """Create test agent."""
        config = CONFIG.copy()
        return MockLLMAgent(player_id=0, persona="cooperative", config=config)

    @pytest.fixture
    def observation(self):
        """Create test observation."""
        return {
            "resource_level": np.array([1000.0]),
            "step": np.array([0]),
            "my_recent_extractions": np.zeros(5),
            "other_players_recent_extractions": np.zeros((1, 5)),
            "my_cumulative_payoff": np.array([0.0]),
            "other_players_cumulative_payoffs": np.array([0.0]),
        }

    def test_initialization(self, agent):
        """Test agent initialization."""
        assert agent.player_id == 0
        assert agent.persona == "cooperative"
        assert len(agent.action_history) == 0
        assert len(agent.reasoning_history) == 0

    def test_act_returns_valid_action(self, agent, observation):
        """Test that act returns valid action in range."""
        action, reasoning = agent.act(observation)

        assert isinstance(action, float)
        assert agent.min_extraction <= action <= agent.max_extraction
        assert isinstance(reasoning, str)

    def test_act_stores_history(self, agent, observation):
        """Test that act stores action and reasoning."""
        agent.act(observation)

        assert len(agent.action_history) == 1
        assert len(agent.reasoning_history) == 1

    def test_persona_affects_behavior(self, observation):
        """Test that different personas behave differently."""
        # Create agents with different personas
        selfish_agent = MockLLMAgent(0, "rational_selfish", CONFIG)
        coop_agent = MockLLMAgent(1, "cooperative", CONFIG)

        # Collect actions over multiple rounds
        selfish_actions = []
        coop_actions = []

        for _ in range(10):
            selfish_action, _ = selfish_agent.act(observation)
            coop_action, _ = coop_agent.act(observation)

            selfish_actions.append(selfish_action)
            coop_actions.append(coop_action)

        # Selfish agent should extract more on average
        assert np.mean(selfish_actions) > np.mean(coop_actions)

    def test_update_memory(self, agent, observation):
        """Test memory update."""
        action = 10.0
        reward = 15.0

        agent.act(observation)
        agent.update_memory(observation, action, reward)

        assert len(agent.observation_history) == 1
        assert len(agent.reward_history) == 1
        assert agent.reward_history[0] == reward

    def test_get_last_reasoning(self, agent, observation):
        """Test getting last reasoning."""
        agent.act(observation, return_reasoning=True)

        reasoning = agent.get_last_reasoning()
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0

    def test_reset(self, agent, observation):
        """Test agent reset."""
        # Take some actions
        agent.act(observation)
        agent.act(observation)

        # Reset
        agent.reset()

        assert len(agent.action_history) == 0
        assert len(agent.reasoning_history) == 0
        assert len(agent.observation_history) == 0
        assert len(agent.reward_history) == 0

    def test_multiple_actions(self, agent, observation):
        """Test multiple actions."""
        n_actions = 5

        for _ in range(n_actions):
            action, reasoning = agent.act(observation)
            assert action is not None
            assert reasoning is not None

        assert len(agent.action_history) == n_actions

    def test_resource_level_affects_action(self, agent):
        """Test that resource level affects extraction."""
        # High resource
        high_resource_obs = {
            "resource_level": np.array([1000.0]),
            "step": np.array([0]),
            "my_recent_extractions": np.zeros(5),
            "other_players_recent_extractions": np.zeros((1, 5)),
            "my_cumulative_payoff": np.array([0.0]),
            "other_players_cumulative_payoffs": np.array([0.0]),
        }

        # Low resource
        low_resource_obs = {
            "resource_level": np.array([100.0]),
            "step": np.array([0]),
            "my_recent_extractions": np.zeros(5),
            "other_players_recent_extractions": np.zeros((1, 5)),
            "my_cumulative_payoff": np.array([0.0]),
            "other_players_cumulative_payoffs": np.array([0.0]),
        }

        high_action, _ = agent.act(high_resource_obs)
        low_action, _ = agent.act(low_resource_obs)

        # Higher resource should lead to higher extraction
        assert high_action > low_action

    def test_repr(self, agent):
        """Test string representation."""
        repr_str = repr(agent)
        assert "MockLLMAgent" in repr_str or "LLMAgent" in repr_str
        assert "player_id=0" in repr_str
        assert "cooperative" in repr_str


class TestAgentIntegration:
    """Integration tests with environment."""

    def test_agent_with_environment(self):
        """Test agent working with environment."""
        from cpr_game.cpr_environment import CPREnvironment

        config = CONFIG.copy()
        config["n_players"] = 2
        config["max_steps"] = 5

        env = CPREnvironment(config)
        agents = [
            MockLLMAgent(0, "rational_selfish", config),
            MockLLMAgent(1, "cooperative", config)
        ]

        # Run episode
        obs, _ = env.reset()

        for _ in range(5):
            actions = []
            for i, agent in enumerate(agents):
                action, _ = agent.act(obs[f"player_{i}"])
                actions.append(action)

            obs, rewards, terminated, truncated, info = env.step(np.array(actions))

            # Update agent memories
            for i, agent in enumerate(agents):
                agent.update_memory(obs[f"player_{i}"], actions[i], rewards[i])

            if terminated or truncated:
                break

        # Check that episode completed
        assert env.current_step > 0
        assert all(len(agent.action_history) > 0 for agent in agents)
