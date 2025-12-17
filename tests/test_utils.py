"""Unit tests for utility functions."""

import pytest
import numpy as np
from cpr_game.utils import (
    compute_gini_coefficient,
    compute_cooperation_index,
    compute_sustainability_score,
    parse_extraction_from_text,
    validate_action,
    calculate_nash_extraction,
    calculate_social_optimum,
)


class TestGiniCoefficient:
    """Test Gini coefficient calculation."""

    def test_perfect_equality(self):
        """Test with equal payoffs."""
        payoffs = np.array([100.0, 100.0, 100.0])
        gini = compute_gini_coefficient(payoffs)
        assert gini == 0.0

    def test_perfect_inequality(self):
        """Test with maximum inequality."""
        payoffs = np.array([0.0, 0.0, 100.0])
        gini = compute_gini_coefficient(payoffs)
        assert gini > 0.5  # High inequality

    def test_moderate_inequality(self):
        """Test with moderate inequality."""
        payoffs = np.array([50.0, 75.0, 100.0])
        gini = compute_gini_coefficient(payoffs)
        assert 0.0 < gini < 0.5

    def test_empty_array(self):
        """Test with empty array."""
        payoffs = np.array([])
        gini = compute_gini_coefficient(payoffs)
        assert gini == 0.0


class TestCooperationIndex:
    """Test cooperation index calculation."""

    def test_perfect_cooperation(self):
        """Test with identical extractions."""
        extractions = np.array([50.0, 50.0])
        coop = compute_cooperation_index(extractions)
        assert coop == 1.0

    def test_no_cooperation(self):
        """Test with very different extractions."""
        extractions = np.array([0.0, 100.0])
        coop = compute_cooperation_index(extractions)
        assert coop < 0.5

    def test_all_zero_extraction(self):
        """Test with all zero extractions."""
        extractions = np.array([0.0, 0.0])
        coop = compute_cooperation_index(extractions)
        assert coop == 1.0  # Perfect cooperation (all abstain)

    def test_single_player(self):
        """Test with single player."""
        extractions = np.array([50.0])
        coop = compute_cooperation_index(extractions)
        assert coop == 1.0


class TestSustainabilityScore:
    """Test sustainability score calculation."""

    def test_always_above_threshold(self):
        """Test when always above threshold."""
        history = [600.0, 700.0, 800.0, 900.0]
        threshold = 500.0
        score = compute_sustainability_score(history, threshold)
        assert score == 1.0

    def test_never_above_threshold(self):
        """Test when never above threshold."""
        history = [100.0, 200.0, 300.0, 400.0]
        threshold = 500.0
        score = compute_sustainability_score(history, threshold)
        assert score == 0.0

    def test_partial_sustainability(self):
        """Test partial sustainability."""
        history = [600.0, 400.0, 700.0, 300.0]  # 2 out of 4 above 500
        threshold = 500.0
        score = compute_sustainability_score(history, threshold)
        assert score == 0.5

    def test_empty_history(self):
        """Test with empty history."""
        history = []
        score = compute_sustainability_score(history, 500.0)
        assert score == 0.0


class TestParseExtraction:
    """Test extraction parsing from text."""

    def test_explicit_extract_pattern(self):
        """Test parsing explicit EXTRACT: pattern."""
        text = "I think we should be careful. EXTRACT: 25.5"
        value, reasoning = parse_extraction_from_text(text)
        assert value == 25.5
        assert "careful" in reasoning.lower()

    def test_case_insensitive(self):
        """Test case-insensitive parsing."""
        text = "My decision: extract: 30"
        value, reasoning = parse_extraction_from_text(text)
        assert value == 30.0

    def test_fallback_to_last_number(self):
        """Test fallback to last number in text."""
        text = "I will extract 42 units this round."
        value, reasoning = parse_extraction_from_text(text)
        assert value == 42.0

    def test_clipping_to_range(self):
        """Test clipping to valid range."""
        text = "EXTRACT: 150"
        value, reasoning = parse_extraction_from_text(text, min_val=0.0, max_val=100.0)
        assert value == 100.0

        text = "EXTRACT: -10"
        value, reasoning = parse_extraction_from_text(text, min_val=0.0, max_val=100.0)
        assert value == 0.0

    def test_no_number_found(self):
        """Test when no number found."""
        text = "I don't know what to do."
        value, reasoning = parse_extraction_from_text(text, min_val=0.0, max_val=100.0)
        assert value == 0.0  # Default to minimum

    def test_decimal_numbers(self):
        """Test parsing decimal numbers."""
        text = "EXTRACT: 45.75"
        value, reasoning = parse_extraction_from_text(text)
        assert value == 45.75


class TestValidateAction:
    """Test action validation."""

    def test_valid_action(self):
        """Test valid action passes through."""
        action = validate_action(50.0, 0.0, 100.0)
        assert action == 50.0

    def test_clip_above_max(self):
        """Test clipping above maximum."""
        action = validate_action(150.0, 0.0, 100.0)
        assert action == 100.0

    def test_clip_below_min(self):
        """Test clipping below minimum."""
        action = validate_action(-10.0, 0.0, 100.0)
        assert action == 0.0

    def test_boundary_values(self):
        """Test boundary values."""
        assert validate_action(0.0, 0.0, 100.0) == 0.0
        assert validate_action(100.0, 0.0, 100.0) == 100.0


class TestNashExtraction:
    """Test Nash equilibrium calculation."""

    def test_nash_extraction_positive(self):
        """Test Nash extraction is non-negative."""
        nash = calculate_nash_extraction(
            resource=1000.0,
            n_players=2,
            regeneration_rate=2.0,
            discount_factor=0.9
        )
        assert nash >= 0.0

    def test_nash_with_zero_resource(self):
        """Test with zero resource."""
        nash = calculate_nash_extraction(
            resource=0.0,
            n_players=2,
            regeneration_rate=2.0
        )
        assert nash == 0.0


class TestSocialOptimum:
    """Test social optimum calculation."""

    def test_social_optimum_basic(self):
        """Test basic social optimum calculation."""
        optimum = calculate_social_optimum(
            resource=1000.0,
            n_players=2,
            regeneration_rate=2.0,
            max_extraction=100.0
        )
        # Regenerated = 1000 * (2-1) = 1000
        # Per player = 1000 / 2 = 500, but capped at 100
        assert optimum == 100.0

    def test_social_optimum_low_resource(self):
        """Test with low resource."""
        optimum = calculate_social_optimum(
            resource=100.0,
            n_players=2,
            regeneration_rate=1.5,
            max_extraction=100.0
        )
        # Regenerated = 100 * 0.5 = 50
        # Per player = 25
        assert optimum == 25.0

    def test_social_optimum_no_regeneration(self):
        """Test with no regeneration."""
        optimum = calculate_social_optimum(
            resource=1000.0,
            n_players=2,
            regeneration_rate=1.0,
            max_extraction=100.0
        )
        # No regeneration, should be 0
        assert optimum == 0.0
