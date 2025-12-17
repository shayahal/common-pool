"""Unit tests for LoggingManager functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from cpr_game.logging_manager import LoggingManager, MockLoggingManager
from cpr_game.config import CONFIG


class TestMockLoggingManager:
    """Test cases for MockLoggingManager."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock logging manager."""
        return MockLoggingManager()

    def test_initialization(self, mock_logger):
        """Test mock logger initialization."""
        assert mock_logger.client is None
        assert mock_logger.current_trace_data is None
        assert len(mock_logger.traces) == 0
        assert len(mock_logger.generations) == 0

    def test_start_game_trace(self, mock_logger):
        """Test starting game trace."""
        config = CONFIG.copy()
        trace = mock_logger.start_game_trace("test_game", config)
        
        assert trace is not None
        assert trace["game_id"] == "test_game"
        assert "config" in trace
        assert "start_time" in trace
        assert mock_logger.game_id == "test_game"

    def test_start_round_span(self, mock_logger):
        """Test starting round span."""
        config = CONFIG.copy()
        mock_logger.start_game_trace("test_game", config)
        
        game_state = {"resource": 1000.0, "step": 0}
        span = mock_logger.start_round_span(0, game_state)
        
        assert span is not None
        assert span["round"] == 0
        assert span["game_state"] == game_state

    def test_log_generation(self, mock_logger):
        """Test logging generation."""
        config = CONFIG.copy()
        mock_logger.start_game_trace("test_game", config)
        
        mock_logger.log_generation(
            player_id=0,
            prompt="Test prompt",
            response="Test response",
            action=50.0,
            reasoning="Test reasoning"
        )
        
        assert len(mock_logger.generations) == 1
        assert mock_logger.generations[0]["player_id"] == 0
        assert mock_logger.generations[0]["action"] == 50.0

    def test_log_round_metrics(self, mock_logger):
        """Test logging round metrics."""
        config = CONFIG.copy()
        mock_logger.start_game_trace("test_game", config)
        
        metrics = {
            "resource_level": 1000.0,
            "cooperation_index": 0.9,
        }
        mock_logger.log_round_metrics(0, metrics)
        
        assert len(mock_logger.round_metrics) == 1
        assert mock_logger.round_metrics[0]["round"] == 0
        assert mock_logger.round_metrics[0]["resource_level"] == 1000.0

    def test_end_game_trace(self, mock_logger):
        """Test ending game trace."""
        config = CONFIG.copy()
        mock_logger.start_game_trace("test_game", config)
        
        summary = {
            "total_rounds": 20,
            "final_resource_level": 500.0,
        }
        mock_logger.end_game_trace(summary)
        
        assert len(mock_logger.traces) == 1
        assert mock_logger.traces[0]["summary"] == summary
        assert "end_time" in mock_logger.traces[0]

    def test_reset(self, mock_logger):
        """Test resetting logger."""
        config = CONFIG.copy()
        mock_logger.start_game_trace("test_game", config)
        mock_logger.log_generation(0, "prompt", "response", 50.0)
        mock_logger.log_round_metrics(0, {"resource": 1000.0})
        
        mock_logger.reset()
        
        assert mock_logger.current_trace_data is None
        assert mock_logger.game_id is None
        assert len(mock_logger.round_metrics) == 0
        assert len(mock_logger.generation_data) == 0

    def test_get_all_traces(self, mock_logger):
        """Test getting all traces."""
        config = CONFIG.copy()
        mock_logger.start_game_trace("game1", config)
        mock_logger.end_game_trace({"total_rounds": 10})
        
        mock_logger.start_game_trace("game2", config)
        mock_logger.end_game_trace({"total_rounds": 20})
        
        traces = mock_logger.get_all_traces()
        assert len(traces) == 2
        assert traces[0]["game_id"] == "game1"
        assert traces[1]["game_id"] == "game2"


class TestLoggingManager:
    """Test cases for LoggingManager (with Langfuse)."""

    @pytest.fixture
    def config_with_langfuse_disabled(self):
        """Config with Langfuse disabled."""
        config = CONFIG.copy()
        config["langfuse_enabled"] = False
        return config

    @pytest.fixture
    def config_with_langfuse_enabled(self):
        """Config with Langfuse enabled but invalid keys."""
        config = CONFIG.copy()
        config["langfuse_enabled"] = True
        config["langfuse_public_key"] = ""
        config["langfuse_secret_key"] = ""
        return config

    def test_initialization_langfuse_disabled(self, config_with_langfuse_disabled):
        """Test initialization with Langfuse disabled."""
        logger = LoggingManager(config_with_langfuse_disabled)
        assert logger.client is None

    def test_initialization_langfuse_enabled_no_keys(self, config_with_langfuse_enabled):
        """Test initialization with Langfuse enabled but no keys."""
        logger = LoggingManager(config_with_langfuse_enabled)
        assert logger.client is None

    @patch('cpr_game.logging_manager.Langfuse')
    def test_initialization_langfuse_success(self, mock_langfuse_class):
        """Test successful Langfuse initialization."""
        # Mock Langfuse client with trace method
        mock_client = MagicMock()
        mock_client.trace = MagicMock()
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_enabled"] = True
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        logger = LoggingManager(config)
        assert logger.client is not None
        mock_langfuse_class.assert_called_once()

    @patch('cpr_game.logging_manager.Langfuse')
    def test_initialization_langfuse_no_trace_method(self, mock_langfuse_class):
        """Test initialization when Langfuse client lacks trace method."""
        # Mock Langfuse client without trace method
        mock_client = MagicMock()
        del mock_client.trace  # Remove trace method
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_enabled"] = True
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        logger = LoggingManager(config)
        assert logger.client is None  # Should be None when trace method missing

    @patch('cpr_game.logging_manager.Langfuse')
    def test_initialization_langfuse_exception(self, mock_langfuse_class):
        """Test initialization when Langfuse raises exception."""
        mock_langfuse_class.side_effect = Exception("Connection failed")
        
        config = CONFIG.copy()
        config["langfuse_enabled"] = True
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        logger = LoggingManager(config)
        assert logger.client is None  # Should be None on exception

    def test_start_game_trace_no_client(self, config_with_langfuse_disabled):
        """Test starting trace when client is None."""
        logger = LoggingManager(config_with_langfuse_disabled)
        trace = logger.start_game_trace("test_game", CONFIG)
        assert trace is None

    @patch('cpr_game.logging_manager.Langfuse')
    def test_start_game_trace_success(self, mock_langfuse_class):
        """Test successful trace start."""
        mock_trace = MagicMock()
        mock_client = MagicMock()
        mock_client.trace.return_value = mock_trace
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_enabled"] = True
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        logger = LoggingManager(config)
        trace = logger.start_game_trace("test_game", config)
        
        assert trace == mock_trace
        assert logger.current_trace == mock_trace
        assert logger.game_id == "test_game"
        mock_client.trace.assert_called_once()

    @patch('cpr_game.logging_manager.Langfuse')
    def test_start_game_trace_exception(self, mock_langfuse_class):
        """Test trace start when exception occurs."""
        mock_client = MagicMock()
        mock_client.trace.side_effect = Exception("Trace failed")
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_enabled"] = True
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        logger = LoggingManager(config)
        trace = logger.start_game_trace("test_game", config)
        
        assert trace is None  # Should return None on exception

    def test_log_generation_no_client(self, config_with_langfuse_disabled):
        """Test logging generation when client is None."""
        logger = LoggingManager(config_with_langfuse_disabled)
        logger.log_generation(0, "prompt", "response", 50.0)
        # Should not raise exception

    @patch('cpr_game.logging_manager.Langfuse')
    def test_log_generation_success(self, mock_langfuse_class):
        """Test successful generation logging."""
        mock_trace = MagicMock()
        mock_trace.generation = MagicMock()
        mock_client = MagicMock()
        mock_client.trace.return_value = mock_trace
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_enabled"] = True
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        logger = LoggingManager(config)
        logger.start_game_trace("test_game", config)
        logger.log_generation(0, "prompt", "response", 50.0, "reasoning")
        
        mock_trace.generation.assert_called_once()

    def test_log_round_metrics_no_client(self, config_with_langfuse_disabled):
        """Test logging metrics when client is None."""
        logger = LoggingManager(config_with_langfuse_disabled)
        logger.log_round_metrics(0, {"resource": 1000.0})
        # Should not raise exception

    @patch('cpr_game.logging_manager.Langfuse')
    def test_end_game_trace_success(self, mock_langfuse_class):
        """Test successful game trace ending."""
        mock_trace = MagicMock()
        mock_trace.score = MagicMock()
        mock_trace.update = MagicMock()
        mock_trace.metadata = {}
        mock_client = MagicMock()
        mock_client.trace.return_value = mock_trace
        mock_client.flush = MagicMock()
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_enabled"] = True
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        logger = LoggingManager(config)
        logger.start_game_trace("test_game", config)
        
        summary = {
            "total_rounds": 20,
            "final_resource_level": 500.0,
            "tragedy_occurred": False,
            "avg_cooperation_index": 0.9,
            "gini_coefficient": 0.1,
            "sustainability_score": 0.8,
        }
        logger.end_game_trace(summary)
        
        assert mock_trace.score.call_count > 0
        mock_trace.update.assert_called_once()
        mock_client.flush.assert_called_once()

    def test_reset(self, config_with_langfuse_disabled):
        """Test resetting logger."""
        logger = LoggingManager(config_with_langfuse_disabled)
        logger.current_trace = MagicMock()
        logger.game_id = "test_game"
        logger.round_metrics = [{"round": 0}]
        logger.generation_data = [{"player_id": 0}]
        
        logger.reset()
        
        assert logger.current_trace is None
        assert logger.game_id is None
        assert len(logger.round_metrics) == 0
        assert len(logger.generation_data) == 0

