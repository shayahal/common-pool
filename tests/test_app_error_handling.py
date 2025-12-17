"""Test error handling in app.py to catch immediate failures."""

import pytest
from unittest.mock import patch, MagicMock
from cpr_game.config import CONFIG


class TestAppErrorHandling:
    """Test that app.py handles errors gracefully."""

    def test_app_fails_with_missing_langfuse_keys(self):
        """Test that app fails immediately if Langfuse keys are missing."""
        from cpr_game import GameRunner
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = ""
        config["langfuse_secret_key"] = ""
        
        # This should raise ValueError immediately
        with pytest.raises(ValueError, match="Langfuse API keys are required"):
            runner = GameRunner(config=config, use_mock_agents=True)

    @patch('cpr_game.logging_manager.Langfuse')
    def test_app_flow_with_valid_config(self, mock_langfuse_class):
        """Test that app works with valid config."""
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_client.start_generation.return_value = MagicMock()
        mock_client.score_current_trace.return_value = MagicMock()
        mock_client.update_current_trace.return_value = MagicMock()
        mock_client.flush.return_value = None
        mock_langfuse_class.return_value = mock_client
        
        # Mock Streamlit
        mock_st = MagicMock()
        mock_st.session_state = MagicMock()
        mock_st.session_state.dashboard_run_history = []
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
            config = CONFIG.copy()
            config["langfuse_public_key"] = "test_key"
            config["langfuse_secret_key"] = "test_secret"
            config["n_players"] = 2
            config["max_steps"] = 2
            
            from cpr_game import GameRunner
            runner = GameRunner(config=config, use_mock_agents=True)
            game_id = runner.setup_game()
            
            # Should succeed
            assert game_id is not None

    def test_validate_config_error_message(self):
        """Test that validate_config gives helpful error messages."""
        from cpr_game.config import validate_config
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = ""
        config["langfuse_secret_key"] = ""
        
        with pytest.raises(ValueError) as exc_info:
            validate_config(config)
        
        error_msg = str(exc_info.value)
        assert "LANGFUSE_PUBLIC_KEY" in error_msg
        assert "LANGFUSE_SECRET_KEY" in error_msg

