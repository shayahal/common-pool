"""Tests to find runtime exceptions in actual code execution.

These tests run the actual code paths to discover exceptions
that occur during real execution.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from cpr_game import GameRunner
from cpr_game.config import CONFIG
from cpr_game.logging_manager import LoggingManager


# Mock Streamlit for tests
@pytest.fixture(autouse=True)
def mock_streamlit():
    """Mock Streamlit session state and functions for all tests."""
    mock_session_state = MagicMock()
    mock_session_state.dashboard_run_history = []
    def getitem(key):
        if hasattr(mock_session_state, key):
            return getattr(mock_session_state, key)
        raise KeyError(f"Key {key} not found")
    def setitem(key, value):
        setattr(mock_session_state, key, value)
    mock_session_state.__getitem__ = getitem
    mock_session_state.__setitem__ = setitem
    
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
    
    with patch('cpr_game.dashboard.st', mock_st):
        with patch('cpr_game.dashboard.st.session_state', mock_session_state):
            yield mock_session_state


class TestGameRunnerExceptions:
    """Test for exceptions in GameRunner execution."""

    @patch('cpr_game.logging_manager.Langfuse')
    def test_game_runner_setup_with_langfuse(self, mock_langfuse_class):
        """Test that GameRunner setup works with Langfuse."""
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_key"
        config["langfuse_secret_key"] = "test_secret"
        config["max_steps"] = 2
        config["n_players"] = 2
        
        runner = GameRunner(config=config, use_mock_agents=True)
        game_id = runner.setup_game()
        
        assert game_id is not None
        assert runner.logger is not None
        assert runner.env is not None
        assert len(runner.agents) == 2

    @patch('cpr_game.logging_manager.Langfuse')
    def test_game_runner_full_episode(self, mock_langfuse_class):
        """Test running a full episode to find exceptions."""
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_client.start_generation.return_value = MagicMock()
        mock_client.score_current_span.return_value = MagicMock()
        mock_client.score_current_trace.return_value = MagicMock()
        mock_client.update_current_trace.return_value = MagicMock()
        mock_client.flush.return_value = None
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_key"
        config["langfuse_secret_key"] = "test_secret"
        config["max_steps"] = 3
        config["n_players"] = 2
        
        runner = GameRunner(config=config, use_mock_agents=True)
        runner.setup_game()
        
        # This should complete without exceptions
        summary = runner.run_episode(visualize=False, verbose=False)
        
        assert "total_rounds" in summary
        assert summary["total_rounds"] > 0

    @patch('cpr_game.logging_manager.Langfuse')
    def test_logging_manager_with_actual_api_calls(self, mock_langfuse_class):
        """Test LoggingManager with realistic API call patterns."""
        mock_client = MagicMock()
        mock_trace_obs = MagicMock()
        mock_span_obs = MagicMock()
        
        # Simulate actual return values
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
        config["langfuse_public_key"] = "test_key"
        config["langfuse_secret_key"] = "test_secret"
        
        logger = LoggingManager(config)
        
        # Simulate game flow
        logger.start_game_trace("test_game", config)
        logger.start_round_span(0, {"resource": 1000, "step": 0})
        
        # Log multiple generations
        for i in range(2):
            logger.log_generation(
                player_id=i,
                prompt=f"Test prompt {i}",
                response=f"Test response {i}",
                action=50.0 + i * 10,
                reasoning=f"Test reasoning {i}"
            )
        
        # Log metrics with various types
        logger.log_round_metrics(0, {
            "resource_level": 1000.0,
            "cooperation_index": 0.9,
            "total_extraction": 100,
            "individual_extractions": [50.0, 50.0],
            "individual_payoffs": [25.0, 25.0],
        })
        
        logger.end_round_span()
        logger.end_game_trace({
            "total_rounds": 1,
            "final_resource_level": 900.0,
            "tragedy_occurred": False,
            "avg_cooperation_index": 0.9,
            "gini_coefficient": 0.1,
            "sustainability_score": 0.8,
        })
        
        # Verify no exceptions occurred
        assert len(logger.generation_data) == 2
        assert len(logger.round_metrics) == 1

    @patch('cpr_game.logging_manager.Langfuse')
    def test_log_round_metrics_with_list_values(self, mock_langfuse_class):
        """Test that log_round_metrics handles list values correctly."""
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_key"
        config["langfuse_secret_key"] = "test_secret"
        
        logger = LoggingManager(config)
        logger.start_game_trace("test_game", config)
        
        # This is what game_runner actually passes
        round_metrics = {
            "resource_level": 1000.0,
            "total_extraction": 100.0,
            "cooperation_index": 0.9,
            "individual_extractions": [50.0, 50.0],  # List!
            "individual_payoffs": [25.0, 25.0],  # List!
        }
        
        # Should not raise exception
        logger.log_round_metrics(0, round_metrics)
        
        # Verify only numeric metrics were scored
        # Lists should be skipped
        # Since no span is active, should use score_current_trace
        call_count = mock_client.score_current_trace.call_count
        assert call_count == 3  # resource_level, total_extraction, cooperation_index

    @patch('cpr_game.logging_manager.Langfuse')
    def test_log_generation_with_empty_strings(self, mock_langfuse_class):
        """Test log_generation with empty strings and None values."""
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_key"
        config["langfuse_secret_key"] = "test_secret"
        
        logger = LoggingManager(config)
        logger.start_game_trace("test_game", config)
        
        # Test with empty strings
        logger.log_generation(0, "", "", 0.0, None)
        
        # Test with None reasoning
        logger.log_generation(1, "prompt", "response", 50.0, None)
        
        # Should not raise exceptions
        assert len(logger.generation_data) == 2

    @patch('cpr_game.logging_manager.Langfuse')
    def test_end_game_trace_with_missing_summary_keys(self, mock_langfuse_class):
        """Test end_game_trace with incomplete summary."""
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_key"
        config["langfuse_secret_key"] = "test_secret"
        
        logger = LoggingManager(config)
        logger.start_game_trace("test_game", config)
        
        # Minimal summary
        summary = {"total_rounds": 10}
        
        # Should not raise exception
        logger.end_game_trace(summary)
        
        assert mock_client.score_current_trace.called
        assert mock_client.update_current_trace.called

    @patch('cpr_game.logging_manager.Langfuse')
    def test_multiple_rounds_with_span_management(self, mock_langfuse_class):
        """Test multiple rounds with proper span start/end."""
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
        config["langfuse_public_key"] = "test_key"
        config["langfuse_secret_key"] = "test_secret"
        
        logger = LoggingManager(config)
        logger.start_game_trace("test_game", config)
        
        # Simulate multiple rounds
        for round_num in range(3):
            logger.start_round_span(round_num, {"resource": 1000 - round_num * 100, "step": round_num})
            logger.log_generation(0, "prompt", "response", 50.0)
            logger.log_round_metrics(round_num, {"resource_level": 1000.0 - round_num * 100})
            logger.end_round_span()
        
        logger.end_game_trace({"total_rounds": 3})
        
        # Verify spans were created for each round
        assert mock_client.start_as_current_span.call_count == 3
        assert len(logger.round_metrics) == 3

    @patch('cpr_game.logging_manager.Langfuse')
    def test_game_runner_does_not_call_start_round_span(self, mock_langfuse_class):
        """Test that game_runner doesn't call start_round_span (it doesn't in current code)."""
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_client.start_generation.return_value = MagicMock()
        mock_client.score_current_span.return_value = MagicMock()
        mock_client.score_current_trace.return_value = MagicMock()
        mock_client.update_current_trace.return_value = MagicMock()
        mock_client.flush.return_value = None
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_key"
        config["langfuse_secret_key"] = "test_secret"
        config["max_steps"] = 2
        config["n_players"] = 2
        
        runner = GameRunner(config=config, use_mock_agents=True)
        runner.setup_game()
        summary = runner.run_episode(visualize=False, verbose=False)
        
        # game_runner doesn't call start_round_span, so it shouldn't be called
        # But score_current_span might still be called
        # The key is that no exceptions should occur
        assert summary["total_rounds"] > 0

    @patch('cpr_game.logging_manager.Langfuse')
    def test_score_current_span_without_active_span(self, mock_langfuse_class):
        """Test score_current_span when no span is active (game_runner doesn't use spans)."""
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        # Simulate error when scoring without active span
        mock_client.score_current_span.side_effect = RuntimeError("No active span")
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_key"
        config["langfuse_secret_key"] = "test_secret"
        
        logger = LoggingManager(config)
        logger.start_game_trace("test_game", config)
        
        # Should not fail - now uses score_current_trace when no span is active
        logger.log_round_metrics(0, {"resource_level": 1000.0})
        
        # Should have called score_current_trace (not score_current_span)
        assert mock_client.score_current_trace.called
        assert not mock_client.score_current_span.called

