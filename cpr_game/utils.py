"""Utility functions for CPR game analysis and metrics calculation."""

import re
from typing import List, Tuple, Optional
import numpy as np


def compute_gini_coefficient(payoffs: np.ndarray) -> float:
    """Calculate Gini coefficient for payoff inequality.

    The Gini coefficient measures inequality in a distribution.
    0 = perfect equality, 1 = maximum inequality.

    Args:
        payoffs: Array of player payoffs

    Returns:
        float: Gini coefficient (0-1)
    """
    if len(payoffs) == 0:
        return 0.0

    # Handle edge cases
    if np.all(payoffs == payoffs[0]):  # All equal
        return 0.0

    # Sort payoffs
    sorted_payoffs = np.sort(payoffs)
    n = len(sorted_payoffs)

    # Calculate Gini coefficient
    cumsum = np.cumsum(sorted_payoffs)
    gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_payoffs)) / (n * np.sum(sorted_payoffs)) - (n + 1) / n

    return float(gini)


def compute_cooperation_index(extractions: np.ndarray, max_extraction: float = 100.0) -> float:
    """Calculate cooperation index based on extraction variance.

    Lower variance indicates more coordinated behavior.
    Returns normalized value where 0 = maximum variance, 1 = perfect coordination.

    Args:
        extractions: Array of player extractions for current round
        max_extraction: Maximum possible extraction value

    Returns:
        float: Cooperation index (0-1)
    """
    if len(extractions) <= 1:
        return 1.0

    # Calculate coefficient of variation (normalized std dev)
    mean_extraction = np.mean(extractions)

    if mean_extraction == 0:
        # All players extracted 0 - perfect cooperation
        return 1.0

    std_extraction = np.std(extractions)
    cv = std_extraction / mean_extraction

    # Normalize: maximum CV occurs when one player extracts max and others extract 0
    # For 2 players: max_cv = 1.0 when one extracts max_extraction and other extracts 0
    max_cv = 1.0

    # Return inverted and normalized value
    cooperation = max(0.0, min(1.0, 1.0 - (cv / max_cv)))

    return float(cooperation)


def compute_sustainability_score(resource_history: List[float], threshold: float) -> float:
    """Calculate what percentage of rounds had resource above threshold.

    Args:
        resource_history: List of resource levels over time
        threshold: Sustainability threshold value

    Returns:
        float: Percentage of rounds above threshold (0-1)
    """
    if len(resource_history) == 0:
        return 0.0

    above_threshold = sum(1 for r in resource_history if r >= threshold)
    return above_threshold / len(resource_history)


def parse_extraction_from_text(text: str, min_val: float = 0.0, max_val: float = 100.0) -> Tuple[float, Optional[str]]:
    """Extract numerical action from LLM response text.

    Looks for pattern "EXTRACT: <number>" first, then falls back to
    finding any number in the text. Returns float extraction amount.

    Args:
        text: LLM response text
        min_val: Minimum allowed extraction value
        max_val: Maximum allowed extraction value

    Returns:
        Tuple of (extraction_amount, reasoning_text)
        extraction_amount is a float
        reasoning_text is the part before "EXTRACT:" if found
    """
    # Try to find explicit "EXTRACT: number" pattern
    extract_pattern = r"EXTRACT:\s*([+-]?\d+\.?\d*)"
    match = re.search(extract_pattern, text, re.IGNORECASE)

    reasoning = None

    if match:
        value = float(match.group(1))
        # Extract reasoning (text before EXTRACT:)
        reasoning_end = match.start()
        if reasoning_end > 0:
            reasoning = text[:reasoning_end].strip()
    else:
        # No EXTRACT pattern found - raise error instead of falling back
        raise ValueError(
            f"Could not find 'EXTRACT: <number>' pattern in text: {text[:100]}"
        )

    # Clip to valid range (keep as float)
    value = float(np.clip(value, min_val, max_val))

    return value, reasoning


def format_round_summary(
    round_num: int,
    resource: float,
    extractions: np.ndarray,
    rewards: np.ndarray,
    cumulative_payoffs: np.ndarray
) -> str:
    """Format a human-readable summary of a game round.

    Args:
        round_num: Current round number
        resource: Resource level after this round
        extractions: Player extractions this round
        rewards: Player rewards this round
        cumulative_payoffs: Player cumulative payoffs

    Returns:
        str: Formatted summary text
    """
    summary = f"\n{'=' * 60}\n"
    summary += f"Round {round_num}\n"
    summary += f"{'=' * 60}\n"
    summary += f"Resource Level: {resource:.2f}\n"
    summary += f"\nPlayer Actions:\n"

    for i, (extraction, reward, total) in enumerate(zip(extractions, rewards, cumulative_payoffs)):
        summary += f"  Player {i}: Extracted {int(extraction)} | Reward {reward:.2f} | Total {total:.2f}\n"

    summary += f"\nTotal Extracted: {int(np.sum(extractions))}\n"

    return summary


def detect_turn_taking_pattern(extraction_history: List[np.ndarray], window: int = 5) -> float:
    """Detect if players are taking turns with high/low extraction.

    Measures negative correlation between players' extractions over time.

    Args:
        extraction_history: List of extraction arrays (one per round)
        window: Number of recent rounds to analyze

    Returns:
        float: Turn-taking score (-1 to 1, where -1 indicates perfect alternation)
    """
    if len(extraction_history) < 2:
        return 0.0

    # Get recent history
    recent = extraction_history[-window:] if len(extraction_history) > window else extraction_history

    if len(recent) < 2:
        return 0.0

    # Convert to array (rounds x players)
    extractions_matrix = np.array(recent)

    if extractions_matrix.shape[1] < 2:
        return 0.0

    # Calculate correlation between player 0 and player 1
    player_0 = extractions_matrix[:, 0]
    player_1 = extractions_matrix[:, 1]

    # Need variance to compute correlation
    if np.std(player_0) == 0 or np.std(player_1) == 0:
        return 0.0

    correlation = np.corrcoef(player_0, player_1)[0, 1]

    return float(correlation)


def calculate_extraction_trend(extraction_history: List[np.ndarray], player_id: int) -> float:
    """Calculate trend in player's extraction over time.

    Positive = increasing extraction, Negative = decreasing extraction.

    Args:
        extraction_history: List of extraction arrays
        player_id: Which player to analyze

    Returns:
        float: Slope of linear regression (trend)
    """
    if len(extraction_history) < 2:
        return 0.0

    extractions = [round_ex[player_id] for round_ex in extraction_history]

    # Simple linear regression
    n = len(extractions)
    x = np.arange(n)
    y = np.array(extractions)

    # Calculate slope
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        return 0.0

    slope = numerator / denominator

    return float(slope)


def validate_action(action: float, min_val: float, max_val: float) -> int:
    """Validate and clip action to valid range.

    Args:
        action: Raw action value
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        int: Clipped and rounded action (integer)
    """
    clipped = np.clip(action, min_val, max_val)
    return int(round(clipped))


def calculate_nash_extraction(
    resource: float,
    n_players: int,
    regeneration_rate: float,
    discount_factor: float = 0.9
) -> float:
    """Calculate theoretical Nash equilibrium extraction (simplified).

    This is a simplified calculation for reference.
    In reality, Nash equilibrium depends on full game dynamics.

    Args:
        resource: Current resource level
        n_players: Number of players
        regeneration_rate: Resource regeneration rate
        discount_factor: Time preference parameter

    Returns:
        float: Suggested Nash extraction per player
    """
    # Simplified formula: extract enough to maximize immediate gain
    # while considering regeneration for future rounds
    # This is a heuristic, not true Nash equilibrium

    sustainable_extraction = resource * (regeneration_rate - 1.0) / n_players

    return max(0.0, sustainable_extraction)


def calculate_social_optimum(
    resource: float,
    n_players: int,
    regeneration_rate: float,
    max_extraction: float
) -> float:
    """Calculate socially optimal extraction per player.

    Social optimum maximizes total welfare while maintaining sustainability.

    Args:
        resource: Current resource level
        n_players: Number of players
        regeneration_rate: Resource regeneration rate
        max_extraction: Maximum extraction allowed

    Returns:
        float: Suggested optimal extraction per player
    """
    # Social optimum: extract regenerated amount equally
    regenerated = resource * (regeneration_rate - 1.0)
    per_player = regenerated / n_players

    return float(np.clip(per_player, 0.0, max_extraction))
