"""Configuration file for Common Pool Resource Game.

All game parameters, LLM settings, logging, and visualization options.
Modify these constants to customize game behavior.
"""

import os
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# Core Game Settings
# ============================================================================

N_PLAYERS: int = 2
MAX_STEPS: int = 50
INITIAL_RESOURCE: int = 100
REGENERATION_RATE: int = 2  # Resource doubles each round (before extraction)
SUSTAINABILITY_THRESHOLD: int = N_PLAYERS
# MIN_RESOURCE is set to N_PLAYERS in CONFIG dict below

# ============================================================================
# Action Space
# ============================================================================

MIN_EXTRACTION: int = 0
MAX_EXTRACTION: int = 35  # Maximum a player can extract per round
ACTION_TYPE: str = "continuous"

# ============================================================================
# Resource Capacity
# ============================================================================

MAX_FISHES: int = 100  # Maximum resource level at any given step (capacity cap)

# ============================================================================
# Reward Parameters
# ============================================================================

EXTRACTION_VALUE: int = 1  # Value per unit extracted

# ============================================================================
# Game Mechanics
# ============================================================================

MOVE_TYPE: str = "simultaneous"  # Players act at the same time
TERMINATION_CONDITIONS: List[str] = ["max_steps", "resource_depletion"]

# ============================================================================
# LLM Configuration
# ============================================================================

# Model Settings
LLM_PROVIDER: str = "openai"
LLM_MODEL: str = "gpt-3.5-turbo"
LLM_TEMPERATURE: float = 0.7
LLM_MAX_TOKENS: int = 500
LLM_TIMEOUT: int = 30  # seconds

# Persona System Prompts
PERSONA_PROMPTS: Dict[str, str] = {
    "rational_selfish": """You are a rational, self-interested player who aims to maximize your own payoff.
You understand game theory and will extract resources strategically to maximize your individual gains.
You may cooperate if it serves your long-term interests, but your primary goal is personal profit.""",

    "cooperative": """You are a cooperative player who values group welfare and sustainability.
You aim to maintain the resource for long-term benefit of all players.
You prefer fair distribution and will try to establish trust and cooperation patterns.""",

    "aggressive": """You are an aggressive player who prioritizes immediate gains.
You tend to extract as much as possible each round, focusing on short-term profits.
You are less concerned about long-term resource sustainability.""",

    "conservative": """You are a conservative player who is risk-averse.
You prefer to extract small amounts to ensure the resource remains available.
You prioritize stability and avoiding resource depletion over maximizing gains.""",

    "opportunistic": """You are an opportunistic player who adapts your strategy based on others' actions.
You observe what other players do and adjust your extraction accordingly.
You may cooperate when others cooperate, but exploit when others are aggressive.""",

    "altruistic": """You are an altruistic player who prioritizes the welfare of others.
You are willing to extract less to ensure others can benefit from the resource.
You value fairness and equitable distribution above personal gain.""",

    "null": "",  # Null persona with no characteristics

    "": """You are a player in a resource management game.
Make decisions that you think are best given the situation."""
}

# Agent Personas (first N_PLAYERS will be used)
PLAYER_PERSONAS: list[str] = list(PERSONA_PROMPTS.keys()) 

# Prompt Engineering
INCLUDE_HISTORY_ROUNDS: int = 5  # How many past rounds to include in context
SHOW_OTHER_PLAYERS_ACTIONS: bool = True  # Full observability
ALLOW_REASONING_OUTPUT: bool = True  # Let LLM explain its thinking

# ============================================================================
# Langfuse Configuration
# ============================================================================

# API Settings (set via environment variables)
LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST: str = "https://cloud.langfuse.com"

# Enable/disable Langfuse logging (useful when API keys not available)
LANGFUSE_ENABLED: bool = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

# Trace Settings
LOG_LEVEL: str = "detailed"  # "minimal", "standard", "detailed"
LOG_LLM_PROMPTS: bool = True
LOG_LLM_RESPONSES: bool = True
LOG_GAME_STATE: bool = True
LOG_REASONING: bool = True

# Metrics to Track Per Round
ROUND_METRICS: List[str] = [
    "resource_level",
    "total_extraction",
    "individual_extractions",
    "individual_payoffs",
    "cooperation_index",
]

# Metrics to Track Per Game
GAME_METRICS: List[str] = [
    "total_rounds",
    "final_resource_level",
    "cumulative_payoffs",
    "tragedy_occurred",  # Boolean: did resource deplete?
    "avg_cooperation_index",
    "gini_coefficient",  # Payoff inequality measure
]

# ============================================================================
# Visualization Configuration
# ============================================================================

# Dashboard Settings
VISUALIZATION_TOOL: str = "streamlit"  # or "gradio"
UPDATE_MODE: str = "realtime"  # "realtime" or "post_game"
REFRESH_RATE: float = 0.5  # seconds between updates

# Chart Configuration
CHART_HEIGHT: int = 400
CHART_WIDTH: int = 800
SHOW_GRID: bool = True

# Plots to Display
PLOTS: Dict[str, bool] = {
    "resource_over_time": True,
    "individual_extractions": True,
    "cumulative_payoffs": True,
    "cooperation_index": True,
    "reasoning_log": True,  # Text display of LLM reasoning
}

# Styling
PLAYER_COLORS: List[str] = ["#FF6B6B", "#4ECDC4", "#95E1D3", "#F38181"]
RESOURCE_COLOR: str = "#2ECC71"
THRESHOLD_COLOR: str = "#E74C3C"
BACKGROUND_COLOR: str = "#FFFFFF"

# ============================================================================
# OpenAI Configuration
# ============================================================================

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# ============================================================================
# Configuration Dictionary
# ============================================================================

CONFIG: Dict = {
    # Game parameters
    "n_players": N_PLAYERS,
    "max_steps": MAX_STEPS,
    "initial_resource": INITIAL_RESOURCE,
    "regeneration_rate": REGENERATION_RATE,
    "min_resource": float(N_PLAYERS),  # MIN_RESOURCE = N_PLAYERS
    "min_extraction": MIN_EXTRACTION,
    "max_extraction": MAX_EXTRACTION,
    "action_type": ACTION_TYPE,
    "max_fishes": MAX_FISHES,  # Maximum resource capacity

    # Rewards (extraction value is always 1)
    "sustainability_threshold": SUSTAINABILITY_THRESHOLD,

    # Game mechanics
    "move_type": MOVE_TYPE,
    "termination_conditions": TERMINATION_CONDITIONS,

    # LLM settings
    "llm_provider": LLM_PROVIDER,
    "llm_model": LLM_MODEL,
    "llm_temperature": LLM_TEMPERATURE,
    "llm_max_tokens": LLM_MAX_TOKENS,
    "llm_timeout": LLM_TIMEOUT,
    "player_personas": PLAYER_PERSONAS,
    "persona_prompts": PERSONA_PROMPTS,
    "include_history_rounds": INCLUDE_HISTORY_ROUNDS,
    "show_other_players_actions": SHOW_OTHER_PLAYERS_ACTIONS,
    "allow_reasoning_output": ALLOW_REASONING_OUTPUT,

    # Langfuse
    "langfuse_enabled": LANGFUSE_ENABLED,
    "langfuse_public_key": LANGFUSE_PUBLIC_KEY,
    "langfuse_secret_key": LANGFUSE_SECRET_KEY,
    "langfuse_host": LANGFUSE_HOST,
    "log_level": LOG_LEVEL,
    "log_llm_prompts": LOG_LLM_PROMPTS,
    "log_llm_responses": LOG_LLM_RESPONSES,
    "log_game_state": LOG_GAME_STATE,
    "log_reasoning": LOG_REASONING,
    "round_metrics": ROUND_METRICS,
    "game_metrics": GAME_METRICS,

    # Visualization
    "visualization_tool": VISUALIZATION_TOOL,
    "update_mode": UPDATE_MODE,
    "refresh_rate": REFRESH_RATE,
    "chart_height": CHART_HEIGHT,
    "chart_width": CHART_WIDTH,
    "show_grid": SHOW_GRID,
    "plots": PLOTS,
    "player_colors": PLAYER_COLORS,
    "resource_color": RESOURCE_COLOR,
    "threshold_color": THRESHOLD_COLOR,
    "background_color": BACKGROUND_COLOR,

    # API keys
    "openai_api_key": OPENAI_API_KEY,
}


def get_config() -> Dict:
    """Return the complete configuration dictionary.

    Returns:
        Dict: Configuration parameters for the CPR game
    """
    return CONFIG.copy()


def validate_config(config: Dict) -> bool:
    """Validate configuration parameters.

    Args:
        config: Configuration dictionary to validate

    Returns:
        bool: True if config is valid

    Raises:
        ValueError: If configuration has invalid values
    """
    if config["n_players"] < 2:
        raise ValueError("n_players must be at least 2")

    if config["max_steps"] < 1:
        raise ValueError("max_steps must be at least 1")

    if config["initial_resource"] <= 0:
        raise ValueError("initial_resource must be positive")

    if config["regeneration_rate"] < 0:
        raise ValueError("regeneration_rate cannot be negative")

    if config["min_extraction"] < 0:
        raise ValueError("min_extraction cannot be negative")

    if config["max_extraction"] <= config["min_extraction"]:
        raise ValueError("max_extraction must be greater than min_extraction")

    if len(config["player_personas"]) < config["n_players"]:
        raise ValueError(f"Not enough personas defined for {config['n_players']} players")

    return True
