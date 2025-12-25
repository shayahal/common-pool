"""Configuration file for Common Pool Resource Game.

All game parameters, LLM settings, logging, and visualization options.
Modify these constants to customize game behavior.
"""

import os
from typing import Dict, List, Optional
from dotenv import load_dotenv
from .persona_prompts import PERSONA_PROMPTS, GAME_RULES_PROMPT

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# Core Game Settings
# ============================================================================

N_PLAYERS: int = 2
MAX_STEPS: int = 20  # Default: 20
INITIAL_RESOURCE: int = 1000  # Default: 1000
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

Max_fish: int = 1000  # Maximum resource level at any given step (capacity cap), renamed from MAX_FISHES

# ============================================================================
# Reward Parameters
# ============================================================================

# Note: Extraction value is always 1 (reward = extraction amount)
# No separate EXTRACTION_VALUE constant needed

# ============================================================================
# Game Mechanics
# ============================================================================

MOVE_TYPE: str = "simultaneous"  # Players act at the same time
TERMINATION_CONDITIONS: List[str] = ["max_steps", "resource_depletion"]

# ============================================================================
# LLM Configuration
# ============================================================================

# Model Settings
LLM_PROVIDER: str = "openrouter"
LLM_MODEL: str = "gpt-3.5-turbo"
LLM_TEMPERATURE: float = 0.7
LLM_MAX_TOKENS: int = 500
LLM_TIMEOUT: int = 30  # seconds

# Persona System Prompts (imported from persona_prompts module)
# PERSONA_PROMPTS is now defined in persona_prompts.py

# Agent Personas (first N_PLAYERS will be used)
PLAYER_PERSONAS: list[str] = list(PERSONA_PROMPTS.keys()) 

# Prompt Engineering
INCLUDE_HISTORY_ROUNDS: int = 3  # How many past rounds to include in context
SHOW_OTHER_PLAYERS_ACTIONS: bool = True  # Full observability
ALLOW_REASONING_OUTPUT: bool = True  # Let LLM explain its thinking
COMPACT_PROMPTS: bool = True  # Use compact prompt format to reduce token usage
USE_INCREMENTAL_PROMPTS: bool = False  # Use delta prompts (only send changes) - advanced optimization

# ============================================================================
# OpenTelemetry Configuration (Single Source of Truth)
# ============================================================================

# OTel Settings (set via environment variables)
OTEL_SERVICE_NAME: str = os.getenv("OTEL_SERVICE_NAME", "cpr-game")
OTEL_ENDPOINT: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")  # HTTP port - OTLP exporter will append /v1/traces automatically
OTEL_PROTOCOL: str = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")  # "grpc" or "http/protobuf" - Langfuse requires HTTP
OTEL_ENABLED: bool = os.getenv("OTEL_ENABLED", "true").lower() == "true"
OTEL_SERVICE_VERSION: str = os.getenv("OTEL_SERVICE_VERSION", "1.0.0")

# OpenTelemetry Receiver Selection
# Controls which receiver(s) receive traces: "langfuse", "langsmith", or "both"
OTEL_RECEIVER: str = os.getenv("OTEL_RECEIVER", "both").lower()

# Langfuse as OTel Receiver
LANGFUSE_OTEL_ENABLED: bool = True
LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_OTEL_ENDPOINT: str = "https://cloud.langfuse.com/api/public/otel"
LANGFUSE_HOST: str = "https://cloud.langfuse.com"  # Legacy, kept for backward compatibility

# LangSmith as OTel Receiver
LANGSMITH_OTEL_ENABLED: bool = True
LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "cpr-game")
LANGSMITH_ENDPOINT: str = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

# FalkorDB as OTel Receiver (via Graphiti)
FALKORDB_ENABLED: bool = os.getenv("FALKORDB_ENABLED", "true").lower() == "true"
FALKORDB_HOST: str = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT: int = int(os.getenv("FALKORDB_PORT", "6379"))
FALKORDB_USERNAME: Optional[str] = os.getenv("FALKORDB_USERNAME", None)
FALKORDB_PASSWORD: Optional[str] = os.getenv("FALKORDB_PASSWORD", None)
FALKORDB_GROUP_ID: str = os.getenv("FALKORDB_GROUP_ID", "cpr-game-traces")

# FalkorDB Exporter Retry Configuration
FALKORDB_MAX_RETRIES: int = int(os.getenv("FALKORDB_MAX_RETRIES", "10"))  # Increased
FALKORDB_BASE_RETRY_DELAY: float = float(os.getenv("FALKORDB_BASE_RETRY_DELAY", "5.0"))  # Increased
FALKORDB_MAX_RETRY_DELAY: float = float(os.getenv("FALKORDB_MAX_RETRY_DELAY", "300.0"))  # Increased
FALKORDB_EXPORT_TIMEOUT: float = float(os.getenv("FALKORDB_EXPORT_TIMEOUT", "3600.0"))  # 60 minutes
FALKORDB_EPISODE_RATE_LIMIT: float = float(os.getenv("FALKORDB_EPISODE_RATE_LIMIT", "1.0"))  # Minimum seconds between episode additions (throttling)

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
# OpenRouter Configuration
# ============================================================================

OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

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
    "max_fishes": Max_fish,  # Maximum resource capacity (kept as max_fishes for backward compatibility in code)

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
    "game_rules_prompt": GAME_RULES_PROMPT,
    "include_history_rounds": INCLUDE_HISTORY_ROUNDS,
    "show_other_players_actions": SHOW_OTHER_PLAYERS_ACTIONS,
    "allow_reasoning_output": ALLOW_REASONING_OUTPUT,
    "compact_prompts": COMPACT_PROMPTS,
    "use_incremental_prompts": USE_INCREMENTAL_PROMPTS,

    # OpenTelemetry (Single Source of Truth)
    "otel_enabled": OTEL_ENABLED,
    "otel_service_name": OTEL_SERVICE_NAME,
    "otel_endpoint": OTEL_ENDPOINT,
    "otel_protocol": OTEL_PROTOCOL,
    "otel_service_version": OTEL_SERVICE_VERSION,
    "otel_receiver": OTEL_RECEIVER,  # Which receiver(s) to use: "langfuse", "langsmith", or "both"
    "otel_resource_attributes": {
        "service.name": OTEL_SERVICE_NAME,
        "service.version": OTEL_SERVICE_VERSION,
        "deployment.environment": os.getenv("DEPLOYMENT_ENVIRONMENT", "development"),
    },
    
    # Langfuse (OTel Receiver - API keys for authentication)
    "langfuse_public_key": LANGFUSE_PUBLIC_KEY,
    "langfuse_secret_key": LANGFUSE_SECRET_KEY,
    "langfuse_host": LANGFUSE_HOST,  # Legacy
    "langfuse_otel_enabled": LANGFUSE_OTEL_ENABLED,
    "langfuse_otel_endpoint": LANGFUSE_OTEL_ENDPOINT,
    
    # LangSmith (OTel Receiver)
    "langsmith_otel_enabled": LANGSMITH_OTEL_ENABLED,
    "langsmith_api_key": LANGSMITH_API_KEY,
    "langsmith_project": LANGSMITH_PROJECT,
    "langsmith_endpoint": LANGSMITH_ENDPOINT,
    
    # FalkorDB (OTel Receiver via Graphiti)
    "falkordb_enabled": FALKORDB_ENABLED,
    "falkordb_host": FALKORDB_HOST,
    "falkordb_port": FALKORDB_PORT,
    "falkordb_username": FALKORDB_USERNAME,
    "falkordb_password": FALKORDB_PASSWORD,
    "falkordb_group_id": FALKORDB_GROUP_ID,
    "falkordb_max_retries": FALKORDB_MAX_RETRIES,
    "falkordb_base_retry_delay": FALKORDB_BASE_RETRY_DELAY,
    "falkordb_max_retry_delay": FALKORDB_MAX_RETRY_DELAY,
    "falkordb_export_timeout": FALKORDB_EXPORT_TIMEOUT,
    "falkordb_episode_rate_limit": FALKORDB_EPISODE_RATE_LIMIT,
    
    # Trace Settings
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
    "openrouter_api_key": OPENROUTER_API_KEY,
    
    # Logging
    "log_dir": "logs",
    
    # Database
    "db_path": "data/game_results.db",
    "db_enabled": True,
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

    # OpenTelemetry configuration - check and auto-adjust based on available keys
    if config.get("otel_enabled", True):
        # Validate OTEL_RECEIVER and check required API keys
        receiver = config.get("otel_receiver", "both").lower()
        if receiver not in ["langfuse", "langsmith", "both"]:
            raise ValueError(
                f"Invalid OTEL_RECEIVER value: {receiver}. Must be one of: 'langfuse', 'langsmith', 'both'"
            )
        
        # Check which receivers have valid API keys
        langfuse_public_key = config.get("langfuse_public_key", "")
        langfuse_secret_key = config.get("langfuse_secret_key", "")
        langfuse_available = bool(langfuse_public_key and langfuse_secret_key)
        
        langsmith_api_key = config.get("langsmith_api_key", "")
        langsmith_available = bool(langsmith_api_key)
        
        # Auto-adjust receiver based on available keys
        if receiver == "both":
            if langfuse_available and langsmith_available:
                # Both available, keep "both"
                pass
            elif langfuse_available:
                # Only Langfuse available
                import logging
                logger = logging.getLogger(__name__)
                error_msg = (
                    "OTEL_RECEIVER was set to 'both' but LangSmith API key is missing. "
                    "Set LANGSMITH_API_KEY to use LangSmith, or set OTEL_RECEIVER to 'langfuse'."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            elif langsmith_available:
                # Only LangSmith available
                import logging
                logger = logging.getLogger(__name__)
                error_msg = (
                    "OTEL_RECEIVER was set to 'both' but Langfuse API keys are missing. "
                    "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to use Langfuse, "
                    "or set OTEL_RECEIVER to 'langsmith'."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            else:
                # Neither available
                import logging
                logger = logging.getLogger(__name__)
                error_msg = (
                    "OTEL_RECEIVER was set to 'both' but neither Langfuse nor LangSmith API keys are available. "
                    "Set API keys to enable tracing, or set OTEL_RECEIVER to 'langfuse' or 'langsmith'."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
        elif receiver == "langfuse":
            if not langfuse_available:
                raise ValueError(
                    "Langfuse API keys are required when OTEL_RECEIVER is 'langfuse'. "
                    "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables."
                )
        elif receiver == "langsmith":
            if not langsmith_available:
                raise ValueError(
                    "LangSmith API key is required when OTEL_RECEIVER is 'langsmith'. "
                    "Set LANGSMITH_API_KEY environment variable."
                )
        
        # Check endpoint only if OTEL is still enabled after key validation
        if config.get("otel_enabled", True):
            endpoint = config.get("otel_endpoint", "")
            if not endpoint:
                raise ValueError(
                    "OpenTelemetry endpoint is required. Set OTEL_EXPORTER_OTLP_ENDPOINT environment variable "
                    "or provide otel_endpoint in config."
                )

    return True
