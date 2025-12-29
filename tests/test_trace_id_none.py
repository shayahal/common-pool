"""Test that handles the case where get_current_trace_id returns None."""

import pytest
from unittest.mock import patch, MagicMock
from cpr_game.logging_manager import LoggingManager
from cpr_game.config import CONFIG


class TestTraceIdNone:
    """Test handling when get_current_trace_id returns None."""

    @patch('cpr_game.logging_manager.Langfuse')
    def test_start_game_trace_with_none_trace_id(self, mock_langfuse_class):
        """Test that start_game_trace handles None trace ID gracefully."""
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        # This is the key: get_current_trace_id returns None
        mock_client.get_current_trace_id.return_value = None
        mock_client.start_generation.return_value = MagicMock()
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_key"
        config["langfuse_secret_key"] = "test_secret"
        
        logger = LoggingManager(config)
        
        # Should not raise an exception even if trace_id is None
        trace = logger.start_game_trace("test_game", config)
        
        # current_trace_id should be set to "active" placeholder
        assert logger.current_trace_id == "active"
        
        # Now log_generation should work even though trace_id was None
        logger.log_generation(
            player_id=0,
            prompt="test prompt",
            response="test response",
            action=10.0,
            reasoning="test reasoning"
        )
        
        # Should have called start_generation
        assert mock_client.start_generation.called


