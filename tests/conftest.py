"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np


@pytest.fixture
def sample_config():
    """Provide a sample configuration for testing."""
    return {
        "n_players": 2,
        "max_steps": 10,
        "initial_resource": 1000.0,
        "regeneration_rate": 2.0,
        "min_resource": 0.0,
        "min_extraction": 0.0,
        "max_extraction": 100.0,
        "extraction_value": 1.0,
        "depletion_penalty": -1000.0,
        "sustainability_bonus": 10.0,
        "sustainability_threshold": 500.0,
        "include_history_rounds": 5,
        "show_other_players_actions": True,
        "player_personas": ["rational_selfish", "cooperative"],
        "langfuse_enabled": False,
    }


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42
