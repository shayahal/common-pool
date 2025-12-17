"""Test actual config loading from app.py to find initialization issues."""

import pytest
import os
from unittest.mock import patch, MagicMock
from cpr_game.config import CONFIG, validate_config


class TestAppConfigLoading:
    """Test how config is loaded in the actual app."""

    def test_config_has_langfuse_keys_from_env(self):
        """Test that config loads Langfuse keys from environment."""
        # Check if CONFIG has langfuse keys
        # They should come from environment variables or be empty
        public_key = CONFIG.get("langfuse_public_key", "")
        secret_key = CONFIG.get("langfuse_secret_key", "")
        
        # If keys are empty, that's the problem!
        if not public_key or not secret_key:
            # This would cause the app to fail
            with pytest.raises(ValueError, match="Langfuse API keys are required"):
                from cpr_game.logging_manager import LoggingManager
                logger = LoggingManager(CONFIG)

    @patch.dict(os.environ, {
        'LANGFUSE_PUBLIC_KEY': 'test_public_key',
        'LANGFUSE_SECRET_KEY': 'test_secret_key'
    })
    def test_config_loads_from_environment(self):
        """Test that config loads from environment variables."""
        # Reload config to pick up environment variables
        from importlib import reload
        from cpr_game import config as config_module
        reload(config_module)
        
        # Check if keys are loaded
        config = config_module.CONFIG.copy()
        assert config.get("langfuse_public_key") == "test_public_key"
        assert config.get("langfuse_secret_key") == "test_secret_key"

    def test_app_py_uses_config_copy(self):
        """Test that app.py creates a copy of CONFIG."""
        # This is what app.py does: config = CONFIG.copy()
        config = CONFIG.copy()
        
        # Modify it
        config["n_players"] = 3
        
        # Original should be unchanged
        assert CONFIG["n_players"] != 3
        
        # But if CONFIG doesn't have langfuse keys, the copy won't either!
        if not config.get("langfuse_public_key") or not config.get("langfuse_secret_key"):
            # This would cause failure when GameRunner tries to create LoggingManager
            with pytest.raises(ValueError, match="Langfuse API keys are required"):
                from cpr_game.logging_manager import LoggingManager
                logger = LoggingManager(config)

    def test_game_runner_with_config_missing_keys(self):
        """Test GameRunner when config is missing Langfuse keys."""
        from cpr_game import GameRunner
        
        config = CONFIG.copy()
        # Remove keys if they exist
        config["langfuse_public_key"] = ""
        config["langfuse_secret_key"] = ""
        
        # This should fail immediately in __init__ when validate_config is called
        with pytest.raises(ValueError, match="Langfuse API keys are required"):
            runner = GameRunner(config=config, use_mock_agents=True)

    def test_validate_config_checks_langfuse_keys(self):
        """Test if validate_config checks for Langfuse keys."""
        config = CONFIG.copy()
        config["langfuse_public_key"] = ""
        config["langfuse_secret_key"] = ""
        
        # validate_config might not check this, but LoggingManager will
        try:
            validate_config(config)
            # If validate_config doesn't check, LoggingManager will fail later
        except ValueError:
            # If it does check, that's good
            pass

