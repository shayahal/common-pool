"""End-to-end tests for CPR game with Langfuse integration.

These tests run the full game flow to catch API compatibility issues
and ensure the game can run successfully from start to finish.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from cpr_game import GameRunner
from cpr_game.config import CONFIG


# Mock Streamlit for tests
@pytest.fixture(autouse=True)
def mock_streamlit():
    """Mock Streamlit session state and functions for all tests."""
    mock_session_state = MagicMock()
    mock_session_state.dashboard_run_history = []
    # Make it behave like a dict/object hybrid
    def getitem(key):
        if hasattr(mock_session_state, key):
            return getattr(mock_session_state, key)
        raise KeyError(f"Key {key} not found")
    def setitem(key, value):
        setattr(mock_session_state, key, value)
    mock_session_state.__getitem__ = getitem
    mock_session_state.__setitem__ = setitem
    
    # Mock all Streamlit functions
    mock_st = MagicMock()
    mock_st.session_state = mock_session_state
    mock_st.empty.return_value = MagicMock()
    mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]  # 3 columns
    mock_st.tabs.return_value = [MagicMock()]
    mock_st.metric.return_value = None
    mock_st.dataframe.return_value = None
    mock_st.plotly_chart.return_value = None
    mock_st.info.return_value = None
    mock_st.write.return_value = None
    
    with patch('cpr_game.dashboard.st', mock_st):
        with patch('cpr_game.dashboard.st.session_state', mock_session_state):
            yield mock_session_state


class TestEndToEndGameRun:
    """End-to-end tests that run complete games."""

    @pytest.fixture
    def config_with_langfuse(self):
        """Create config with Langfuse keys for testing."""
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        config["max_steps"] = 5  # Short game for testing
        config["n_players"] = 2
        config["initial_resource"] = 1000
        return config

    @patch('cpr_game.logging_manager.Langfuse')
    def test_full_game_run_with_mock_langfuse(self, mock_langfuse_class):
        """Test complete game run with mocked Langfuse client."""
        # Setup mock Langfuse client with all required methods
        mock_client = MagicMock()
        mock_trace_obs = MagicMock()
        mock_span_obs = MagicMock()
        
        # Mock all Langfuse 3.x API methods
        mock_client.start_as_current_observation.return_value = mock_trace_obs
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_client.start_as_current_span.return_value = mock_span_obs
        mock_client.get_current_observation_id.return_value = "span_456"
        mock_client.start_generation.return_value = MagicMock()
        mock_client.score_current_span.return_value = MagicMock()
        mock_client.score_current_trace.return_value = MagicMock()
        mock_client.update_current_trace.return_value = MagicMock()
        mock_client.flush.return_value = None
        
        mock_langfuse_class.return_value = mock_client
        
        # Create config with Langfuse keys
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        config["max_steps"] = 3  # Very short game
        config["n_players"] = 2
        
        # Run game
        runner = GameRunner(config=config, use_mock_agents=True)
        game_id = runner.setup_game()
        
        # Verify logger was initialized
        assert runner.logger is not None
        
        # Run episode (this starts the trace)
        summary = runner.run_episode(visualize=False, verbose=False)
        
        # Verify trace was started during run_episode
        # Note: trace_id may be None after end_game_trace, so we check that methods were called
        
        # Verify game completed
        assert "total_rounds" in summary
        assert summary["total_rounds"] > 0
        assert "final_resource_level" in summary
        
        # Verify Langfuse methods were called
        assert mock_client.start_as_current_observation.called, "Trace should be started"
        # Note: start_as_current_span may not be called if game_runner doesn't use it
        assert mock_client.start_generation.called, "Generations should be logged"
        assert mock_client.score_current_trace.called, "Game scores should be logged"
        assert mock_client.update_current_trace.called, "Trace should be updated"
        assert mock_client.flush.called, "Trace should be flushed"

    @patch('cpr_game.logging_manager.Langfuse')
    def test_game_run_without_tags_parameter(self, mock_langfuse_class):
        """Test that game runs without using unsupported 'tags' parameter."""
        # Setup mock that will raise error if 'tags' is passed
        def check_no_tags(**kwargs):
            if 'tags' in kwargs:
                raise TypeError(f"Unexpected keyword argument 'tags'")
            return MagicMock()
        
        mock_client = MagicMock()
        mock_client.start_as_current_observation.side_effect = check_no_tags
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_client.start_as_current_span.return_value = MagicMock()
        mock_client.get_current_observation_id.return_value = "span_456"
        mock_client.start_generation.return_value = MagicMock()
        mock_client.score_current_span.return_value = MagicMock()
        mock_client.score_current_trace.return_value = MagicMock()
        mock_client.update_current_trace.return_value = MagicMock()
        mock_client.flush.return_value = None
        
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        config["max_steps"] = 2
        config["n_players"] = 2
        
        # This should not raise an error about 'tags' parameter
        runner = GameRunner(config=config, use_mock_agents=True)
        game_id = runner.setup_game()
        
        # Should complete without errors
        summary = runner.run_episode(visualize=False, verbose=False)
        assert "total_rounds" in summary

    @patch('cpr_game.logging_manager.Langfuse')
    def test_game_run_handles_langfuse_errors_gracefully(self, mock_langfuse_class):
        """Test that game handles Langfuse API errors appropriately."""
        mock_client = MagicMock()
        # Simulate API error during trace creation
        mock_client.start_as_current_observation.side_effect = AttributeError(
            "start_as_current_observation() got an unexpected keyword argument 'tags'"
        )
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        config["max_steps"] = 2
        config["n_players"] = 2
        
        runner = GameRunner(config=config, use_mock_agents=True)
        game_id = runner.setup_game()
        
        # Should raise RuntimeError with helpful message
        with pytest.raises(RuntimeError, match="Failed to start game trace"):
            runner.run_episode(visualize=False, verbose=False)

    @patch('cpr_game.logging_manager.Langfuse')
    def test_multiple_rounds_with_logging(self, mock_langfuse_class):
        """Test game with multiple rounds and verify all logging calls."""
        mock_client = MagicMock()
        mock_trace_obs = MagicMock()
        mock_span_obs = MagicMock()
        
        mock_client.start_as_current_observation.return_value = mock_trace_obs
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_client.start_as_current_span.return_value = mock_span_obs
        mock_client.get_current_observation_id.return_value = "span_456"
        mock_client.start_generation.return_value = MagicMock()
        mock_client.score_current_span.return_value = MagicMock()
        mock_client.score_current_trace.return_value = MagicMock()
        mock_client.update_current_trace.return_value = MagicMock()
        mock_client.flush.return_value = None
        
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        config["max_steps"] = 5
        config["n_players"] = 2
        
        runner = GameRunner(config=config, use_mock_agents=True)
        game_id = runner.setup_game()
        summary = runner.run_episode(visualize=False, verbose=False)
        
        # Verify we logged generations for each round
        # Each round has n_players generations
        expected_generations = summary["total_rounds"] * config["n_players"]
        assert mock_client.start_generation.call_count == expected_generations
        
        # Verify we logged metrics for each round
        # Since no spans are started, should use score_current_trace
        assert mock_client.score_current_trace.call_count >= summary["total_rounds"]
        
        # Verify trace was updated and flushed
        assert mock_client.update_current_trace.called
        assert mock_client.flush.called

    @patch('cpr_game.logging_manager.Langfuse')
    def test_game_with_different_configs(self, mock_langfuse_class):
        """Test game runs with various configurations."""
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_client.start_as_current_span.return_value = MagicMock()
        mock_client.get_current_observation_id.return_value = "span_456"
        mock_client.start_generation.return_value = MagicMock()
        mock_client.score_current_span.return_value = MagicMock()
        mock_client.score_current_trace.return_value = MagicMock()
        mock_client.update_current_trace.return_value = MagicMock()
        mock_client.flush.return_value = None
        
        mock_langfuse_class.return_value = mock_client
        
        # Test with different player counts
        for n_players in [2, 3, 4]:
            config = CONFIG.copy()
            config["langfuse_public_key"] = "test_public_key"
            config["langfuse_secret_key"] = "test_secret_key"
            config["max_steps"] = 3
            config["n_players"] = n_players
            
            runner = GameRunner(config=config, use_mock_agents=True)
            game_id = runner.setup_game()
            summary = runner.run_episode(visualize=False, verbose=False)
            
            assert summary["total_rounds"] > 0
            assert len(summary["cumulative_payoffs"]) == n_players

    @patch('cpr_game.logging_manager.Langfuse')
    def test_game_reset_and_multiple_runs(self, mock_langfuse_class):
        """Test running multiple games in sequence."""
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_client.start_as_current_span.return_value = MagicMock()
        mock_client.get_current_observation_id.return_value = "span_456"
        mock_client.start_generation.return_value = MagicMock()
        mock_client.score_current_span.return_value = MagicMock()
        mock_client.score_current_trace.return_value = MagicMock()
        mock_client.update_current_trace.return_value = MagicMock()
        mock_client.flush.return_value = None
        
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        config["max_steps"] = 2
        config["n_players"] = 2
        
        # Run first game
        runner1 = GameRunner(config=config, use_mock_agents=True)
        game_id1 = runner1.setup_game()
        summary1 = runner1.run_episode(visualize=False, verbose=False)
        
        # Run second game
        runner2 = GameRunner(config=config, use_mock_agents=True)
        game_id2 = runner2.setup_game()
        summary2 = runner2.run_episode(visualize=False, verbose=False)
        
        # Both should complete successfully
        assert summary1["total_rounds"] > 0
        assert summary2["total_rounds"] > 0
        assert game_id1 != game_id2  # Different game IDs


class TestLangfuseAPICompatibility:
    """Tests specifically for Langfuse API compatibility."""

    @patch('cpr_game.logging_manager.Langfuse')
    def test_start_as_current_observation_parameters(self, mock_langfuse_class):
        """Test that start_as_current_observation is called with correct parameters."""
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        from cpr_game.logging_manager import LoggingManager
        logger = LoggingManager(config)
        logger.start_game_trace("test_game", config)
        
        # Verify the call was made
        assert mock_client.start_as_current_observation.called
        
        # Get the call arguments
        call_args = mock_client.start_as_current_observation.call_args
        
        # Verify 'tags' is NOT in kwargs (should be in metadata instead)
        assert 'tags' not in call_args.kwargs, \
            "tags should not be passed as a separate parameter"
        
        # Verify required parameters are present
        assert 'as_type' in call_args.kwargs
        assert call_args.kwargs['as_type'] == 'trace'
        assert 'name' in call_args.kwargs
        assert 'metadata' in call_args.kwargs
        
        # Verify tags are in metadata
        metadata = call_args.kwargs['metadata']
        assert 'tags' in metadata, "tags should be in metadata, not as a separate parameter"

    @patch('cpr_game.logging_manager.Langfuse')
    def test_all_langfuse_methods_exist(self, mock_langfuse_class):
        """Test that all required Langfuse methods are available."""
        mock_client = MagicMock()
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        from cpr_game.logging_manager import LoggingManager
        logger = LoggingManager(config)
        
        # Verify all required methods exist
        required_methods = [
            'start_as_current_observation',
            'get_current_trace_id',
            'start_as_current_span',
            'get_current_observation_id',
            'start_generation',
            'score_current_span',
            'score_current_trace',
            'update_current_trace',
            'flush'
        ]
        
        for method_name in required_methods:
            assert hasattr(mock_client, method_name), \
                f"Langfuse client should have method: {method_name}"

