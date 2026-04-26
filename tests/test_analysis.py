"""
Tests for Analysis Engines
==========================
Tests metrics calculator, league normalizer, and player comparison.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from analysis.metrics_calculator import (
    calculate_xT, possession_adjusted_tackles,
    pressing_intensity, progression_score,
    goal_threat_score, creative_output_score,
)
from analysis.league_normalizer import (
    apply_league_tax, league_adaptability_score, get_coefficient,
)
from analysis.player_comparison import PlayerComparisonEngine


# ─── Metrics Calculator Tests ────────────────────────────────────

class TestXTCalculation:
    def test_positive_xT_gain(self):
        """A pass from defensive half to attacking half should have positive xT."""
        pass_data = [((20, 34), (90, 34))]  # Deep to attacking zone
        result = calculate_xT(pass_data)
        assert result > 0, "Pass towards goal should generate positive xT"

    def test_backward_pass_no_xT(self):
        """A backward pass should generate 0 xT."""
        pass_data = [((90, 34), (20, 34))]  # Back pass
        result = calculate_xT(pass_data)
        assert result == 0.0, "Backward pass should not generate xT"

    def test_empty_passes(self):
        """No passes = 0 xT."""
        assert calculate_xT([]) == 0.0

    def test_multiple_passes(self):
        """Multiple forward passes should accumulate xT."""
        pass_data = [
            ((30, 34), (60, 34)),
            ((60, 34), (85, 34)),
        ]
        result = calculate_xT(pass_data)
        assert result > 0


class TestPossessionAdjustedTackles:
    def test_low_possession_inflates(self):
        """A team with 35% possession should have tackles deflated."""
        adjusted = possession_adjusted_tackles(3.0, 35)
        # 50 / (100-35) = 50/65 ≈ 0.77 → 3.0 * 0.77 ≈ 2.31
        assert adjusted < 3.0

    def test_high_possession_team(self):
        """A team with 70% possession should have tackles inflated."""
        adjusted = possession_adjusted_tackles(2.0, 70)
        # 50 / (100-70) = 50/30 ≈ 1.67 → 2.0 * 1.67 ≈ 3.33
        assert adjusted > 2.0

    def test_50_percent_no_change(self):
        """50% possession should leave tackles unchanged."""
        adjusted = possession_adjusted_tackles(3.0, 50)
        assert adjusted == 3.0


class TestPressingIntensity:
    def test_zero_pressures(self):
        assert pressing_intensity(0, 0) == 0.0

    def test_high_pressing(self):
        result = pressing_intensity(25, 10)
        assert result > 0


class TestCompositeScores:
    def test_progression_score(self):
        score = progression_score(6.0, 8.0, 3.0, 500)
        assert 0 <= score <= 100

    def test_goal_threat_clinical(self):
        """Clinical finisher (goals > xG) should get bonus."""
        clinical = goal_threat_score(0.6, 0.4, 0.4, 3.0)
        not_clinical = goal_threat_score(0.3, 0.4, 0.4, 3.0)
        assert clinical > not_clinical


# ─── League Normalizer Tests ────────────────────────────────────

class TestLeagueNormalizer:
    def test_premier_league_no_change(self):
        """PL stats should remain unchanged (coefficient = 1.0)."""
        stats = {"goals_p90": 0.5, "tackles_p90": 2.0}
        adjusted = apply_league_tax(stats, "premier_league")
        assert adjusted["goals_p90"] == 0.5
        assert adjusted["tackles_p90"] == 2.0

    def test_eredivisie_reduces_goals(self):
        """Eredivisie goals should be reduced (coefficient ~0.7)."""
        stats = {"goals_p90": 1.0, "tackles_p90": 2.0}
        adjusted = apply_league_tax(stats, "eredivisie")
        assert adjusted["goals_p90"] < 1.0
        assert adjusted["goals_p90"] == round(1.0 * 0.70, 3)

    def test_unknown_league(self):
        """Unknown league should use default coefficient."""
        coef = get_coefficient("random_league")
        assert coef == 0.55

    def test_adaptability_young_top_league(self):
        """Young player from a top league should adapt well."""
        score = league_adaptability_score("la_liga", 22, ["la_liga"])
        assert score > 80

    def test_adaptability_old_weak_league(self):
        """Older player from weak league scores lower."""
        score = league_adaptability_score("mls", 31, [])
        assert score < 70


# ─── Player Comparison Tests ────────────────────────────────────

class TestPlayerComparison:
    def test_add_and_find(self):
        engine = PlayerComparisonEngine(collection_name="test_collection")
        engine.reset()

        # Add two similar attacking players
        stats_a = {"goals_p90": 0.5, "assists_p90": 0.3, "xG_p90": 0.45,
                    "xA_p90": 0.25, "xT_p90": 0.4, "tackles_p90": 0.5}
        stats_b = {"goals_p90": 0.48, "assists_p90": 0.28, "xG_p90": 0.42,
                    "xA_p90": 0.22, "xT_p90": 0.38, "tackles_p90": 0.6}

        # Add a very different defender
        stats_c = {"goals_p90": 0.02, "assists_p90": 0.05, "xG_p90": 0.01,
                    "xA_p90": 0.03, "xT_p90": 0.05, "tackles_p90": 4.5}

        engine.add_player("p1", "Attacker A", stats_a)
        engine.add_player("p2", "Attacker B", stats_b)
        engine.add_player("p3", "Defender C", stats_c)

        similar = engine.find_similar(stats_a, n=2, exclude_id="p1")

        assert len(similar) == 2
        # Attacker B should be more similar to Attacker A than Defender C
        assert similar[0]["name"] == "Attacker B"

        engine.reset()
