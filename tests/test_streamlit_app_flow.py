"""Tests that simulate the actual Streamlit app flow.

These tests mimic what happens when a user presses "Run Game" in the app.
"""

import pytest
from unittest.mock import patch, MagicMock
from cpr_game import GameRunner
from cpr_game.config import CONFIG


# Mock Streamlit for all tests
@pytest.fixture(autouse=True)
def mock_streamlit():
    """Mock Streamlit session state and functions for all tests."""
    mock_session_state = MagicMock()
    mock_session_state.dashboard_run_history = []
    mock_session_state.all_runs = []
    
    def getitem(key):
        if hasattr(mock_session_state, key):
            return getattr(mock_session_state, key)
        if key in ['dashboard_run_history', 'all_runs']:
            return []
        raise KeyError(f"Key {key} not found")
    
    def setitem(key, value):
        setattr(mock_session_state, key, value)
    
    mock_session_state.__getitem__ = getitem
    mock_session_state.__setitem__ = setitem
    mock_session_state.get = lambda key, default=None: getattr(mock_session_state, key, default) if hasattr(mock_session_state, key) else default
    
    mock_st = MagicMock()
    mock_st.session_state = mock_session_state
    mock_st.empty.return_value = MagicMock()
    mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
    mock_st.tabs.return_value = [MagicMock()]
    mock_st.metric.return_value = None
    mock_st.dataframe.return_value = None
    mock_st.plotly_chart.return_value = None
    mock_st.info.return_value = None
    mock_st.write.return_value = None
    mock_st.markdown.return_value = None
    mock_st.button.return_value = True
    mock_st.slider.return_value = 1.0
    mock_st.selectbox.return_value = "rational_selfish"
    mock_st.checkbox.return_value = False
    
    with patch('cpr_game.dashboard.st', mock_st):
        with patch('cpr_game.dashboard.st.session_state', mock_session_state):
            yield mock_st


class TestStreamlitAppFlow:
    """Test the actual Streamlit app execution flow."""

    @patch('cpr_game.logging_manager.Langfuse')
    def test_run_game_button_press_simulation(self, mock_langfuse_class):
        """Simulate pressing Run Game button in Streamlit app."""
        # Setup mock Langfuse client
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_client.start_generation.return_value = MagicMock()
        mock_client.score_current_trace.return_value = MagicMock()
        mock_client.update_current_trace.return_value = MagicMock()
        mock_client.flush.return_value = None
        mock_langfuse_class.return_value = mock_client
        
        # Simulate what app.py does when Run Game is pressed
        config = CONFIG.copy()
        config["n_players"] = 2
        config["max_steps"] = 5
        config["initial_resource"] = 1000
        config["regeneration_rate"] = 2.0
        config["sustainability_threshold"] = 500.0
        config["max_fishes"] = 100
        config["player_personas"] = ["rational_selfish", "cooperative"]
        
        # Check if Langfuse keys are required
        # In app.py, these should come from environment or config
        if "langfuse_public_key" not in config or not config["langfuse_public_key"]:
            # This is likely the issue - keys might be missing
            config["langfuse_public_key"] = "test_public_key"
            config["langfuse_secret_key"] = "test_secret_key"
        
        # Initialize game runner (this is what app.py does)
        runner = GameRunner(
            config=config,
            use_mock_agents=True  # Use mock agents for testing
        )
        
        # Setup game (this happens in app.py)
        game_id = runner.setup_game()
        assert game_id is not None
        
        # Start game trace (this happens in app.py line 211)
        runner.logger.start_game_trace(game_id, config)
        
        # Reset environment (this happens in app.py line 214)
        observations, info = runner.env.reset()
        for agent in runner.agents:
            agent.reset()
        
        # Run a few steps to simulate game execution
        done = False
        step = 0
        max_steps = 2  # Just test a couple steps
        
        while not done and step < max_steps:
            # Get actions from agents
            actions = []
            for i, agent in enumerate(runner.agents):
                obs = observations[f"player_{i}"]
                action, reasoning = agent.act(obs, return_reasoning=True)
                actions.append(action)
                
                # Log generation (this happens in game_runner)
                prompt = agent._build_prompt(obs) if hasattr(agent, '_build_prompt') else ""
                runner.logger.log_generation(
                    player_id=i,
                    prompt=prompt,
                    response=reasoning or "",
                    action=action,
                    reasoning=reasoning
                )
            
            # Execute step
            observations, rewards, terminated, truncated, info = runner.env.step(actions)
            done = terminated or truncated
            
            # Log round metrics (this happens in game_runner)
            round_metrics = {
                "resource_level": info["resource"],
                "total_extraction": info["total_extraction"],
                "cooperation_index": info["cooperation_index"],
                "individual_extractions": actions,
                "individual_payoffs": rewards.tolist(),
            }
            runner.logger.log_round_metrics(step, round_metrics)
            
            step += 1
        
        # End game trace
        summary = runner.env.get_summary_stats()
        runner.logger.end_game_trace(summary)
        
        # Should complete without exceptions
        assert summary["total_rounds"] > 0

    @patch('cpr_game.logging_manager.Langfuse')
    def test_app_flow_with_missing_langfuse_keys(self, mock_langfuse_class):
        """Test what happens when Langfuse keys are missing (common issue)."""
        config = CONFIG.copy()
        config["langfuse_public_key"] = ""  # Missing key
        config["langfuse_secret_key"] = ""  # Missing key
        config["n_players"] = 2
        config["max_steps"] = 2
        
        # This should raise ValueError
        with pytest.raises(ValueError, match="Langfuse API keys are required"):
            runner = GameRunner(config=config, use_mock_agents=True)
            runner.setup_game()

    @patch('cpr_game.logging_manager.Langfuse')
    def test_app_flow_with_config_from_app_py(self, mock_langfuse_class):
        """Test with exact config structure from app.py."""
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_client.start_generation.return_value = MagicMock()
        mock_client.score_current_trace.return_value = MagicMock()
        mock_client.update_current_trace.return_value = MagicMock()
        mock_client.flush.return_value = None
        mock_langfuse_class.return_value = mock_client
        
        # This is exactly what app.py creates
        config = CONFIG.copy()
        config["n_players"] = 2
        config["max_steps"] = 5
        config["initial_resource"] = 1000
        config["regeneration_rate"] = 2.0
        config["sustainability_threshold"] = 500.0
        config["max_fishes"] = 100
        config["player_personas"] = ["rational_selfish", "cooperative"]
        
        # Check if keys exist in CONFIG or need to be set
        if not config.get("langfuse_public_key"):
            config["langfuse_public_key"] = "test_public_key"
        if not config.get("langfuse_secret_key"):
            config["langfuse_secret_key"] = "test_secret_key"
        
        # This is the exact flow from app.py
        runner = GameRunner(
            config=config,
            use_mock_agents=True
        )
        
        game_id = runner.setup_game()
        
        # This line from app.py line 211
        runner.logger.start_game_trace(game_id, config)
        
        # These lines from app.py lines 214-216
        observations, info = runner.env.reset()
        for agent in runner.agents:
            agent.reset()
        
        # Should not raise any exceptions
        assert game_id is not None
        assert runner.logger is not None
        assert runner.env is not None

    def test_config_validation_before_game_run(self):
        """Test that config validation happens and catches issues."""
        from cpr_game.config import validate_config
        
        # Test with invalid config
        invalid_config = CONFIG.copy()
        invalid_config["n_players"] = -1  # Invalid
        
        with pytest.raises((ValueError, AssertionError)):
            validate_config(invalid_config)

    @patch('cpr_game.logging_manager.Langfuse')
    def test_dashboard_initialization_in_app_flow(self, mock_langfuse_class):
        """Test dashboard initialization which happens in app.py."""
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_key"
        config["langfuse_secret_key"] = "test_secret"
        config["n_players"] = 2
        
        runner = GameRunner(config=config, use_mock_agents=True)
        
        # Dashboard is created in setup_game
        game_id = runner.setup_game()
        
        # Dashboard should be initialized
        assert runner.dashboard is not None
        
        # Should be able to access dashboard methods
        assert hasattr(runner.dashboard, 'update')
        assert hasattr(runner.dashboard, 'show_summary')

    @patch('cpr_game.logging_manager.Langfuse')
    def test_full_app_execution_with_all_components(self, mock_langfuse_class):
        """Test complete app execution with all components."""
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_client.start_generation.return_value = MagicMock()
        mock_client.score_current_trace.return_value = MagicMock()
        mock_client.update_current_trace.return_value = MagicMock()
        mock_client.flush.return_value = None
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_key"
        config["langfuse_secret_key"] = "test_secret"
        config["n_players"] = 2
        config["max_steps"] = 3
        
        # Full execution
        runner = GameRunner(config=config, use_mock_agents=True)
        game_id = runner.setup_game()
        
        # This is what app.py does - start trace before running
        runner.logger.start_game_trace(game_id, config)
        
        # Reset
        observations, info = runner.env.reset()
        for agent in runner.agents:
            agent.reset()
        
        # Run episode (this is what should happen)
        summary = runner.run_episode(visualize=False, verbose=False)
        
        # Should complete successfully
        assert summary["total_rounds"] > 0
        assert "final_resource_level" in summary

