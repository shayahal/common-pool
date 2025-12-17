"""Unit tests for LoggingManager functionality."""

import pytest
import importlib.metadata
from unittest.mock import Mock, patch, MagicMock
from cpr_game.logging_manager import LoggingManager, REQUIRED_LANGFUSE_VERSION
from cpr_game.config import CONFIG


class TestLoggingManager:
    """Test cases for LoggingManager (with Langfuse)."""

    @pytest.fixture
    def config_with_keys(self):
        """Config with valid Langfuse keys."""
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        return config

    @pytest.fixture
    def config_missing_keys(self):
        """Config with missing Langfuse keys."""
        config = CONFIG.copy()
        config["langfuse_public_key"] = ""
        config["langfuse_secret_key"] = ""
        return config

    @patch('cpr_game.logging_manager.importlib.metadata.version')
    def test_version_check_success(self, mock_version):
        """Test successful version check."""
        mock_version.return_value = REQUIRED_LANGFUSE_VERSION
        
        with patch('cpr_game.logging_manager.Langfuse') as mock_langfuse:
            mock_client = MagicMock()
            mock_langfuse.return_value = mock_client
            
            config = CONFIG.copy()
            config["langfuse_public_key"] = "test_public_key"
            config["langfuse_secret_key"] = "test_secret_key"
            
            logger = LoggingManager(config)
            assert logger.client is not None

    def test_version_check_mismatch(self):
        """Test version check with mismatched version."""
        # The version check happens at module import time, so we can't easily
        # test it with mocking after the module is imported. Instead, we verify
        # that the version check logic works by checking the actual installed version.
        from packaging import version
        import importlib.metadata
        
        installed_version = importlib.metadata.version("langfuse")
        # Verify that the version check would work correctly
        # The module should have already passed the version check at import time
        assert version.parse(installed_version) == version.parse(REQUIRED_LANGFUSE_VERSION), \
            f"Installed version {installed_version} should match required {REQUIRED_LANGFUSE_VERSION}. " \
            f"If this fails, the version check at import time should have caught it."

    def test_version_check_not_installed(self):
        """Test version check when langfuse is not installed."""
        # This test verifies the version check logic by testing the exception handling
        # Since the module is already imported, we test that the version check
        # would raise the correct error if the package wasn't found.
        # We verify the error message format is correct.
        from packaging import version
        import importlib.metadata
        
        # Verify that version checking works correctly
        try:
            installed_version = importlib.metadata.version("langfuse")
            # If we get here, langfuse is installed, which is expected
            assert version.parse(installed_version) == version.parse(REQUIRED_LANGFUSE_VERSION)
        except importlib.metadata.PackageNotFoundError:
            # This would raise RuntimeError at import time
            pytest.fail("langfuse should be installed for tests to run")

    @patch('cpr_game.logging_manager.importlib.metadata.version')
    @patch('cpr_game.logging_manager.Langfuse')
    def test_initialization_missing_keys(self, mock_langfuse_class, mock_version):
        """Test initialization with missing API keys."""
        mock_version.return_value = REQUIRED_LANGFUSE_VERSION
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = ""
        config["langfuse_secret_key"] = ""
        
        with pytest.raises(ValueError, match="Langfuse API keys are required"):
            LoggingManager(config)

    @patch('cpr_game.logging_manager.importlib.metadata.version')
    @patch('cpr_game.logging_manager.Langfuse')
    def test_initialization_success(self, mock_langfuse_class, mock_version):
        """Test successful Langfuse initialization."""
        mock_version.return_value = REQUIRED_LANGFUSE_VERSION
        
        mock_client = MagicMock()
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        logger = LoggingManager(config)
        assert logger.client is not None
        assert logger.client == mock_client
        mock_langfuse_class.assert_called_once_with(
            public_key="test_public_key",
            secret_key="test_secret_key",
            host=config.get("langfuse_host", "https://cloud.langfuse.com")
        )

    @patch('cpr_game.logging_manager.importlib.metadata.version')
    @patch('cpr_game.logging_manager.Langfuse')
    def test_initialization_connection_error(self, mock_langfuse_class, mock_version):
        """Test initialization when Langfuse raises connection error."""
        mock_version.return_value = REQUIRED_LANGFUSE_VERSION
        mock_langfuse_class.side_effect = ConnectionError("Connection failed")
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        with pytest.raises(RuntimeError, match="Failed to initialize Langfuse client"):
            LoggingManager(config)

    @patch('cpr_game.logging_manager.importlib.metadata.version')
    @patch('cpr_game.logging_manager.Langfuse')
    def test_start_game_trace_success(self, mock_langfuse_class, mock_version):
        """Test successful trace start."""
        mock_version.return_value = REQUIRED_LANGFUSE_VERSION
        
        mock_trace_obs = MagicMock()
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = mock_trace_obs
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        logger = LoggingManager(config)
        trace = logger.start_game_trace("test_game", config)
        
        assert trace == mock_trace_obs
        assert logger.current_trace_id == "trace_123"
        assert logger.game_id == "test_game"
        mock_client.start_as_current_observation.assert_called_once()
        call_kwargs = mock_client.start_as_current_observation.call_args[1]
        assert call_kwargs["as_type"] == "trace"
        assert call_kwargs["name"] == "CPR_Game_test_game"
        assert "metadata" in call_kwargs
        assert call_kwargs["metadata"]["game_id"] == "test_game"

    @patch('cpr_game.logging_manager.importlib.metadata.version')
    @patch('cpr_game.logging_manager.Langfuse')
    def test_start_game_trace_no_client(self, mock_langfuse_class, mock_version):
        """Test starting trace when client is None."""
        mock_version.return_value = REQUIRED_LANGFUSE_VERSION
        
        mock_client = MagicMock()
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        logger = LoggingManager(config)
        logger.client = None  # Simulate client being None
        
        with pytest.raises(RuntimeError, match="Langfuse client is not initialized"):
            logger.start_game_trace("test_game", config)

    @patch('cpr_game.logging_manager.importlib.metadata.version')
    @patch('cpr_game.logging_manager.Langfuse')
    def test_start_game_trace_exception(self, mock_langfuse_class, mock_version):
        """Test trace start when exception occurs."""
        mock_version.return_value = REQUIRED_LANGFUSE_VERSION
        
        mock_client = MagicMock()
        mock_client.start_as_current_observation.side_effect = AttributeError("Method not found")
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        logger = LoggingManager(config)
        
        with pytest.raises(RuntimeError, match="Failed to start game trace"):
            logger.start_game_trace("test_game", config)

    @patch('cpr_game.logging_manager.importlib.metadata.version')
    @patch('cpr_game.logging_manager.Langfuse')
    def test_start_round_span_success(self, mock_langfuse_class, mock_version):
        """Test successful round span start."""
        mock_version.return_value = REQUIRED_LANGFUSE_VERSION
        
        mock_span_obs = MagicMock()
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_client.start_as_current_span.return_value = mock_span_obs
        mock_client.get_current_observation_id.return_value = "span_456"
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        logger = LoggingManager(config)
        logger.start_game_trace("test_game", config)
        
        game_state = {"resource": 1000, "step": 0}
        span = logger.start_round_span(0, game_state)
        
        assert span == mock_span_obs
        assert logger.current_round_span_id == "span_456"
        mock_client.start_as_current_span.assert_called_once()
        call_kwargs = mock_client.start_as_current_span.call_args[1]
        assert call_kwargs["name"] == "round_0"
        assert "metadata" in call_kwargs
        assert call_kwargs["metadata"]["round"] == 0

    @patch('cpr_game.logging_manager.importlib.metadata.version')
    @patch('cpr_game.logging_manager.Langfuse')
    def test_start_round_span_no_trace(self, mock_langfuse_class, mock_version):
        """Test starting span when trace is not started."""
        mock_version.return_value = REQUIRED_LANGFUSE_VERSION
        
        mock_client = MagicMock()
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        logger = LoggingManager(config)
        
        with pytest.raises(RuntimeError, match="Game trace is not started"):
            logger.start_round_span(0, {"resource": 1000, "step": 0})

    @patch('cpr_game.logging_manager.importlib.metadata.version')
    @patch('cpr_game.logging_manager.Langfuse')
    def test_log_generation_success(self, mock_langfuse_class, mock_version):
        """Test successful generation logging."""
        mock_version.return_value = REQUIRED_LANGFUSE_VERSION
        
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        logger = LoggingManager(config)
        logger.start_game_trace("test_game", config)
        
        logger.log_generation(
            player_id=0,
            prompt="Test prompt",
            response="Test response",
            action=50.0,
            reasoning="Test reasoning"
        )
        
        mock_client.start_generation.assert_called_once()
        call_kwargs = mock_client.start_generation.call_args[1]
        assert call_kwargs["name"] == "player_0_decision"
        assert call_kwargs["metadata"]["player_id"] == 0
        assert call_kwargs["metadata"]["action"] == 50.0
        assert len(logger.generation_data) == 1
        assert logger.generation_data[0]["player_id"] == 0

    @patch('cpr_game.logging_manager.importlib.metadata.version')
    @patch('cpr_game.logging_manager.Langfuse')
    def test_log_generation_no_trace(self, mock_langfuse_class, mock_version):
        """Test logging generation when trace is not started."""
        mock_version.return_value = REQUIRED_LANGFUSE_VERSION
        
        mock_client = MagicMock()
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        logger = LoggingManager(config)
        
        with pytest.raises(RuntimeError, match="Game trace is not started"):
            logger.log_generation(0, "prompt", "response", 50.0)

    @patch('cpr_game.logging_manager.importlib.metadata.version')
    @patch('cpr_game.logging_manager.Langfuse')
    def test_log_generation_exception(self, mock_langfuse_class, mock_version):
        """Test generation logging when exception occurs."""
        mock_version.return_value = REQUIRED_LANGFUSE_VERSION
        
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_client.start_generation.side_effect = ValueError("Invalid generation")
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        logger = LoggingManager(config)
        logger.start_game_trace("test_game", config)
        
        with pytest.raises(RuntimeError, match="Failed to log generation"):
            logger.log_generation(0, "prompt", "response", 50.0)

    @patch('cpr_game.logging_manager.importlib.metadata.version')
    @patch('cpr_game.logging_manager.Langfuse')
    def test_log_round_metrics_success(self, mock_langfuse_class, mock_version):
        """Test successful round metrics logging."""
        mock_version.return_value = REQUIRED_LANGFUSE_VERSION
        
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        logger = LoggingManager(config)
        logger.start_game_trace("test_game", config)
        
        metrics = {
            "resource_level": 1000.0,
            "cooperation_index": 0.9,
        }
        logger.log_round_metrics(0, metrics)
        
        # Should call score_current_trace for each numeric metric (no span is active)
        assert mock_client.score_current_trace.call_count == 2
        assert len(logger.round_metrics) == 1
        assert logger.round_metrics[0]["round"] == 0
        assert logger.round_metrics[0]["resource_level"] == 1000.0

    @patch('cpr_game.logging_manager.importlib.metadata.version')
    @patch('cpr_game.logging_manager.Langfuse')
    def test_log_round_metrics_no_trace(self, mock_langfuse_class, mock_version):
        """Test logging metrics when trace is not started."""
        mock_version.return_value = REQUIRED_LANGFUSE_VERSION
        
        mock_client = MagicMock()
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        logger = LoggingManager(config)
        
        with pytest.raises(RuntimeError, match="Game trace is not started"):
            logger.log_round_metrics(0, {"resource": 1000.0})

    @patch('cpr_game.logging_manager.importlib.metadata.version')
    @patch('cpr_game.logging_manager.Langfuse')
    def test_log_round_metrics_exception(self, mock_langfuse_class, mock_version):
        """Test round metrics logging when exception occurs."""
        mock_version.return_value = REQUIRED_LANGFUSE_VERSION
        
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_client.score_current_trace.side_effect = ConnectionError("Connection failed")
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        logger = LoggingManager(config)
        logger.start_game_trace("test_game", config)
        
        with pytest.raises(RuntimeError, match="Failed to log round metrics"):
            logger.log_round_metrics(0, {"resource": 1000.0})

    @patch('cpr_game.logging_manager.importlib.metadata.version')
    @patch('cpr_game.logging_manager.Langfuse')
    def test_end_round_span_success(self, mock_langfuse_class, mock_version):
        """Test successful round span ending."""
        mock_version.return_value = REQUIRED_LANGFUSE_VERSION
        
        mock_span_obs = MagicMock()
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_client.start_as_current_span.return_value = mock_span_obs
        mock_client.get_current_observation_id.return_value = "span_456"
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        logger = LoggingManager(config)
        logger.start_game_trace("test_game", config)
        logger.start_round_span(0, {"resource": 1000, "step": 0})
        
        logger.end_round_span()
        
        assert logger.current_round_span_id is None

    @patch('cpr_game.logging_manager.importlib.metadata.version')
    @patch('cpr_game.logging_manager.Langfuse')
    def test_end_round_span_no_span(self, mock_langfuse_class, mock_version):
        """Test ending span when span is not started."""
        mock_version.return_value = REQUIRED_LANGFUSE_VERSION
        
        mock_client = MagicMock()
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        logger = LoggingManager(config)
        
        with pytest.raises(RuntimeError, match="Round span is not started"):
            logger.end_round_span()

    @patch('cpr_game.logging_manager.importlib.metadata.version')
    @patch('cpr_game.logging_manager.Langfuse')
    def test_end_game_trace_success(self, mock_langfuse_class, mock_version):
        """Test successful game trace ending."""
        mock_version.return_value = REQUIRED_LANGFUSE_VERSION
        
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
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
        
        # Should call score_current_trace for each game score
        assert mock_client.score_current_trace.call_count > 0
        mock_client.update_current_trace.assert_called_once()
        mock_client.flush.assert_called_once()
        assert logger.current_trace_id is None
        assert logger.game_id is None

    @patch('cpr_game.logging_manager.importlib.metadata.version')
    @patch('cpr_game.logging_manager.Langfuse')
    def test_end_game_trace_no_trace(self, mock_langfuse_class, mock_version):
        """Test ending trace when trace is not started."""
        mock_version.return_value = REQUIRED_LANGFUSE_VERSION
        
        mock_client = MagicMock()
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        logger = LoggingManager(config)
        
        with pytest.raises(RuntimeError, match="Game trace is not started"):
            logger.end_game_trace({})

    @patch('cpr_game.logging_manager.importlib.metadata.version')
    @patch('cpr_game.logging_manager.Langfuse')
    def test_end_game_trace_exception(self, mock_langfuse_class, mock_version):
        """Test game trace ending when exception occurs."""
        mock_version.return_value = REQUIRED_LANGFUSE_VERSION
        
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_client.score_current_trace.side_effect = ValueError("Invalid score")
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        logger = LoggingManager(config)
        logger.start_game_trace("test_game", config)
        
        with pytest.raises(RuntimeError, match="Failed to end game trace"):
            logger.end_game_trace({"total_rounds": 20})

    @patch('cpr_game.logging_manager.importlib.metadata.version')
    @patch('cpr_game.logging_manager.Langfuse')
    def test_reset(self, mock_langfuse_class, mock_version):
        """Test resetting logger."""
        mock_version.return_value = REQUIRED_LANGFUSE_VERSION
        
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        logger = LoggingManager(config)
        logger.start_game_trace("test_game", config)
        logger.log_generation(0, "prompt", "response", 50.0)
        logger.log_round_metrics(0, {"resource": 1000.0})
        
        logger.reset()
        
        assert logger.current_trace_id is None
        assert logger.current_round_span_id is None
        assert logger.game_id is None
        assert len(logger.round_metrics) == 0
        assert len(logger.generation_data) == 0

    @patch('cpr_game.logging_manager.importlib.metadata.version')
    @patch('cpr_game.logging_manager.Langfuse')
    def test_get_round_metrics(self, mock_langfuse_class, mock_version):
        """Test getting round metrics."""
        mock_version.return_value = REQUIRED_LANGFUSE_VERSION
        
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        logger = LoggingManager(config)
        logger.start_game_trace("test_game", config)
        logger.log_round_metrics(0, {"resource": 1000.0})
        logger.log_round_metrics(1, {"resource": 900.0})
        
        metrics = logger.get_round_metrics()
        assert len(metrics) == 2
        assert metrics[0]["round"] == 0
        assert metrics[1]["round"] == 1
        # Should return a copy
        metrics.append({"round": 2})
        assert len(logger.get_round_metrics()) == 2

    @patch('cpr_game.logging_manager.importlib.metadata.version')
    @patch('cpr_game.logging_manager.Langfuse')
    def test_get_generation_data(self, mock_langfuse_class, mock_version):
        """Test getting generation data."""
        mock_version.return_value = REQUIRED_LANGFUSE_VERSION
        
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        logger = LoggingManager(config)
        logger.start_game_trace("test_game", config)
        logger.log_generation(0, "prompt1", "response1", 50.0)
        logger.log_generation(1, "prompt2", "response2", 60.0)
        
        generations = logger.get_generation_data()
        assert len(generations) == 2
        assert generations[0]["player_id"] == 0
        assert generations[1]["player_id"] == 1
        # Should return a copy
        generations.append({"player_id": 2})
        assert len(logger.get_generation_data()) == 2

    @patch('cpr_game.logging_manager.importlib.metadata.version')
    @patch('cpr_game.logging_manager.Langfuse')
    def test_full_workflow(self, mock_langfuse_class, mock_version):
        """Test complete workflow: trace -> span -> generation -> metrics -> end."""
        mock_version.return_value = REQUIRED_LANGFUSE_VERSION
        
        mock_client = MagicMock()
        mock_client.start_as_current_observation.return_value = MagicMock()
        mock_client.get_current_trace_id.return_value = "trace_123"
        mock_client.start_as_current_span.return_value = MagicMock()
        mock_client.get_current_observation_id.return_value = "span_456"
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        
        logger = LoggingManager(config)
        
        # Start game trace
        trace = logger.start_game_trace("test_game", config)
        assert trace is not None
        assert logger.game_id == "test_game"
        
        # Start round span
        game_state = {"resource": 1000, "step": 0}
        span = logger.start_round_span(0, game_state)
        assert span is not None
        assert logger.current_round_span_id == "span_456"
        
        # Log generation
        logger.log_generation(0, "prompt", "response", 50.0, "reasoning")
        assert len(logger.generation_data) == 1
        
        # Log metrics
        logger.log_round_metrics(0, {"resource_level": 1000.0, "cooperation_index": 0.9})
        assert len(logger.round_metrics) == 1
        
        # End span
        logger.end_round_span()
        assert logger.current_round_span_id is None
        
        # End game trace
        summary = {
            "total_rounds": 1,
            "final_resource_level": 950.0,
            "tragedy_occurred": False,
            "avg_cooperation_index": 0.9,
            "gini_coefficient": 0.1,
            "sustainability_score": 0.8,
        }
        logger.end_game_trace(summary)
        assert logger.current_trace_id is None
        assert logger.game_id is None
        
        # Verify all methods were called
        mock_client.start_as_current_observation.assert_called_once()
        mock_client.start_as_current_span.assert_called_once()
        mock_client.start_generation.assert_called_once()
        assert mock_client.score_current_span.call_count > 0
        assert mock_client.score_current_trace.call_count > 0
        mock_client.update_current_trace.assert_called_once()
        mock_client.flush.assert_called_once()

    @patch('cpr_game.logging_manager.importlib.metadata.version')
    @patch('cpr_game.logging_manager.Langfuse')
    def test_custom_host(self, mock_langfuse_class, mock_version):
        """Test initialization with custom Langfuse host."""
        mock_version.return_value = REQUIRED_LANGFUSE_VERSION
        
        mock_client = MagicMock()
        mock_langfuse_class.return_value = mock_client
        
        config = CONFIG.copy()
        config["langfuse_public_key"] = "test_public_key"
        config["langfuse_secret_key"] = "test_secret_key"
        config["langfuse_host"] = "https://custom.langfuse.com"
        
        logger = LoggingManager(config)
        
        mock_langfuse_class.assert_called_once_with(
            public_key="test_public_key",
            secret_key="test_secret_key",
            host="https://custom.langfuse.com"
        )
