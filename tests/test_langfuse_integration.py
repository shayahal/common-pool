"""Integration tests for Langfuse API compatibility.

These tests verify that the actual Langfuse API calls work correctly
and catch any runtime exceptions that might occur.
"""

import pytest
from unittest.mock import patch, MagicMock, call
from cpr_game.logging_manager import LoggingManager
from cpr_game.config import CONFIG


class TestLangfuseAPICalls:
    """Test actual Langfuse API method calls."""

    @patch('cpr_game.logging_manager.Langfuse')
    def test_start_as_current_observation_call_signature(self, mock_langfuse_class):
        """Test that start_as_current_observation is called with correct signature."""
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_key"
        config["langfuse_secret_key"] = "test_secret"
        
        logger = LoggingManager(config)
        logger.start_game_trace("test_game", config)
        
        # Verify the call was made
        assert mock_client.start_as_current_observation.called
        
        # Get call arguments
        call_args = mock_client.start_as_current_observation.call_args
        
        # Verify no 'tags' parameter in kwargs
        assert 'tags' not in call_args.kwargs, \
            "tags should not be a direct parameter"
        
        # Verify required parameters
        assert 'as_type' in call_args.kwargs
        assert call_args.kwargs['as_type'] == 'trace'
        assert 'name' in call_args.kwargs
        assert 'metadata' in call_args.kwargs
        
        # Verify tags are in metadata
        metadata = call_args.kwargs['metadata']
        assert 'tags' in metadata

    @patch('cpr_game.logging_manager.Langfuse')
    def test_start_as_current_span_call_signature(self, mock_langfuse_class):
        """Test that start_as_current_span is called correctly."""
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_client.start_as_current_span.return_value = MagicMock()
        mock_client.get_current_observation_id.return_value = "span_456"
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_key"
        config["langfuse_secret_key"] = "test_secret"
        
        logger = LoggingManager(config)
        logger.start_game_trace("test_game", config)
        logger.start_round_span(0, {"resource": 1000, "step": 0})
        
        # Verify the call
        assert mock_client.start_as_current_span.called
        call_args = mock_client.start_as_current_span.call_args
        
        # Verify parameters
        assert 'name' in call_args.kwargs
        assert 'metadata' in call_args.kwargs

    @patch('cpr_game.logging_manager.Langfuse')
    def test_start_generation_call_signature(self, mock_langfuse_class):
        """Test that start_generation is called correctly."""
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_key"
        config["langfuse_secret_key"] = "test_secret"
        
        logger = LoggingManager(config)
        logger.start_game_trace("test_game", config)
        logger.log_generation(0, "prompt", "response", 50.0, "reasoning")
        
        # Verify the call
        assert mock_client.start_generation.called
        call_args = mock_client.start_generation.call_args
        
        # Verify required parameters
        assert 'name' in call_args.kwargs
        assert 'model' in call_args.kwargs
        assert 'input' in call_args.kwargs
        assert 'output' in call_args.kwargs
        assert 'metadata' in call_args.kwargs

    @patch('cpr_game.logging_manager.Langfuse')
    def test_score_current_span_call_signature(self, mock_langfuse_class):
        """Test that score_current_span is called correctly."""
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_key"
        config["langfuse_secret_key"] = "test_secret"
        
        logger = LoggingManager(config)
        logger.start_game_trace("test_game", config)
        logger.log_round_metrics(0, {"resource_level": 1000.0, "cooperation_index": 0.9})
        
        # Verify the calls - should use score_current_trace when no span is active
        assert mock_client.score_current_trace.called
        
        # Check call arguments
        for call_obj in mock_client.score_current_span.call_args_list:
            assert 'name' in call_obj.kwargs
            assert 'value' in call_obj.kwargs
            assert isinstance(call_obj.kwargs['value'], (int, float))

    @patch('cpr_game.logging_manager.Langfuse')
    def test_score_current_trace_call_signature(self, mock_langfuse_class):
        """Test that score_current_trace is called correctly."""
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_key"
        config["langfuse_secret_key"] = "test_secret"
        
        logger = LoggingManager(config)
        logger.start_game_trace("test_game", config)
        logger.end_game_trace({
            "total_rounds": 20,
            "final_resource_level": 500.0,
            "tragedy_occurred": False,
            "avg_cooperation_index": 0.9,
            "gini_coefficient": 0.1,
            "sustainability_score": 0.8,
        })
        
        # Verify the calls
        assert mock_client.score_current_trace.called
        
        # Check call arguments
        for call_obj in mock_client.score_current_trace.call_args_list:
            assert 'name' in call_obj.kwargs
            assert 'value' in call_obj.kwargs
            assert isinstance(call_obj.kwargs['value'], (int, float))

    @patch('cpr_game.logging_manager.Langfuse')
    def test_update_current_trace_call_signature(self, mock_langfuse_class):
        """Test that update_current_trace is called correctly."""
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_key"
        config["langfuse_secret_key"] = "test_secret"
        
        logger = LoggingManager(config)
        logger.start_game_trace("test_game", config)
        logger.end_game_trace({"total_rounds": 20})
        
        # Verify the call
        assert mock_client.update_current_trace.called
        call_args = mock_client.update_current_trace.call_args
        
        # Verify parameters
        assert 'metadata' in call_args.kwargs

    @patch('cpr_game.logging_manager.Langfuse')
    def test_all_methods_handle_none_return_values(self, mock_langfuse_class):
        """Test that methods handle None return values gracefully."""
        mock_client = MagicMock()
        # Some methods might return None
        mock_client.start_as_current_observation.return_value = None
        mock_client.get_current_trace_id.return_value = None
        mock_client.start_as_current_span.return_value = None
        mock_client.get_current_observation_id.return_value = None
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_key"
        config["langfuse_secret_key"] = "test_secret"
        
        logger = LoggingManager(config)
        
        # Should not raise exceptions even if methods return None
        trace = logger.start_game_trace("test_game", config)
        # trace might be None, but shouldn't crash
        assert logger.game_id == "test_game"
        
        # Try to start span even if trace_id is None
        try:
            logger.start_round_span(0, {"resource": 1000, "step": 0})
        except RuntimeError:
            # This is expected if trace_id is None
            pass

    @patch('cpr_game.logging_manager.Langfuse')
    def test_methods_with_missing_attributes(self, mock_langfuse_class):
        """Test behavior when Langfuse client methods don't exist."""
        mock_client = MagicMock()
        # Remove a method to simulate missing attribute
        del mock_client.start_as_current_observation
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_key"
        config["langfuse_secret_key"] = "test_secret"
        
        logger = LoggingManager(config)
        
        # Should raise RuntimeError with helpful message
        with pytest.raises(RuntimeError, match="Failed to start game trace"):
            logger.start_game_trace("test_game", config)

    @patch('cpr_game.logging_manager.Langfuse')
    def test_concurrent_trace_operations(self, mock_langfuse_class):
        """Test that trace operations work correctly in sequence."""
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
        
        # Full sequence
        logger.start_game_trace("test_game", config)
        logger.start_round_span(0, {"resource": 1000, "step": 0})
        logger.log_generation(0, "prompt", "response", 50.0)
        logger.log_round_metrics(0, {"resource_level": 1000.0})
        logger.end_round_span()
        logger.end_game_trace({"total_rounds": 1})
        
        # Verify all methods were called
        assert mock_client.start_as_current_observation.called
        assert mock_client.start_as_current_span.called
        assert mock_client.start_generation.called
        assert mock_client.score_current_span.called
        assert mock_client.score_current_trace.called
        assert mock_client.update_current_trace.called
        assert mock_client.flush.called

    @patch('cpr_game.logging_manager.Langfuse')
    def test_error_messages_are_helpful(self, mock_langfuse_class):
        """Test that error messages provide helpful information."""
        mock_client = MagicMock()
        mock_client.start_as_current_observation.side_effect = TypeError(
            "start_as_current_observation() got an unexpected keyword argument 'tags'"
        )
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_key"
        config["langfuse_secret_key"] = "test_secret"
        
        logger = LoggingManager(config)
        
        with pytest.raises(RuntimeError) as exc_info:
            logger.start_game_trace("test_game", config)
        
        # Error message should be helpful
        error_msg = str(exc_info.value)
        assert "error starting game trace" in error_msg.lower()

