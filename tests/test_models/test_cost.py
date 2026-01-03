"""Tests for cost tracking models."""

from datetime import datetime
from unittest.mock import Mock

import pytest

from vibesignal.models.cost import (
    APICallMetrics,
    CostBreakdown,
    TokenUsage,
    MODEL_PRICING,
)


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_total_input_tokens(self):
        """Test total_input_tokens calculation."""
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            cache_creation_input_tokens=200,
            cache_read_input_tokens=300,
        )
        assert usage.total_input_tokens == 1500  # 1000 + 200 + 300

    def test_total_tokens(self):
        """Test total_tokens calculation."""
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            cache_creation_input_tokens=200,
            cache_read_input_tokens=300,
        )
        assert usage.total_tokens == 2000  # 1500 input + 500 output

    def test_defaults_to_zero(self):
        """Test default values are zero."""
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cache_creation_input_tokens == 0
        assert usage.cache_read_input_tokens == 0
        assert usage.total_tokens == 0


class TestCostBreakdown:
    """Tests for CostBreakdown dataclass."""

    def test_total_cost(self):
        """Test total_cost calculation."""
        cost = CostBreakdown(
            input_cost=0.003,
            output_cost=0.015,
            cache_write_cost=0.001,
            cache_read_cost=0.0001,
        )
        assert abs(cost.total_cost - 0.0191) < 0.0001

    def test_cache_savings(self):
        """Test cache_savings calculation."""
        cost = CostBreakdown(
            input_cost=0.003,
            output_cost=0.015,
            cache_write_cost=0.0,
            cache_read_cost=0.0003,  # Cache read is 0.1x normal price
        )
        # Full price would be 0.003 (10x cache read)
        # Savings is 0.003 - 0.0003 = 0.0027
        expected_savings = 0.003 - 0.0003
        assert abs(cost.cache_savings - expected_savings) < 0.0001

    def test_no_cache_savings_when_no_cache_reads(self):
        """Test cache_savings is 0 when no cache reads."""
        cost = CostBreakdown(
            input_cost=0.003,
            output_cost=0.015,
        )
        assert cost.cache_savings == 0.0


class TestModelPricing:
    """Tests for MODEL_PRICING configuration."""

    def test_sonnet_pricing_exists(self):
        """Test that Sonnet pricing is configured."""
        assert "claude-sonnet-4-5-20250929" in MODEL_PRICING
        pricing = MODEL_PRICING["claude-sonnet-4-5-20250929"]
        assert "input" in pricing
        assert "output" in pricing
        assert "cache_write" in pricing
        assert "cache_read" in pricing

    def test_default_pricing_exists(self):
        """Test that default pricing fallback exists."""
        assert "default" in MODEL_PRICING

    def test_cache_pricing_ratios(self):
        """Test that cache pricing follows expected ratios."""
        for model, pricing in MODEL_PRICING.items():
            # Cache write should be ~1.25x input price
            assert pricing["cache_write"] >= pricing["input"]
            # Cache read should be ~0.1x input price
            assert pricing["cache_read"] < pricing["input"]


class TestAPICallMetrics:
    """Tests for APICallMetrics dataclass."""

    def test_from_response_extracts_usage(self):
        """Test from_response extracts token usage correctly."""
        # Create mock response
        mock_response = Mock()
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 1000
        mock_response.usage.output_tokens = 500
        mock_response.usage.cache_creation_input_tokens = 0
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.id = "msg_123"

        metrics = APICallMetrics.from_response(
            response=mock_response,
            model="claude-sonnet-4-5-20250929",
            wall_time_seconds=2.5,
        )

        assert metrics.usage.input_tokens == 1000
        assert metrics.usage.output_tokens == 500
        assert metrics.wall_time_seconds == 2.5
        assert metrics.model == "claude-sonnet-4-5-20250929"
        assert metrics.request_id == "msg_123"

    def test_from_response_calculates_costs(self):
        """Test from_response calculates costs correctly."""
        mock_response = Mock()
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 1_000_000  # 1M tokens
        mock_response.usage.output_tokens = 100_000   # 100K tokens
        mock_response.usage.cache_creation_input_tokens = 0
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.id = None

        metrics = APICallMetrics.from_response(
            response=mock_response,
            model="claude-sonnet-4-5-20250929",
            wall_time_seconds=10.0,
        )

        # Sonnet: $3/M input, $15/M output
        assert abs(metrics.cost.input_cost - 3.0) < 0.001
        assert abs(metrics.cost.output_cost - 1.5) < 0.001
        assert abs(metrics.cost.total_cost - 4.5) < 0.001

    def test_from_response_with_cache_tokens(self):
        """Test from_response handles cache tokens."""
        mock_response = Mock()
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 500
        mock_response.usage.output_tokens = 200
        mock_response.usage.cache_creation_input_tokens = 1000
        mock_response.usage.cache_read_input_tokens = 2000
        mock_response.id = None

        metrics = APICallMetrics.from_response(
            response=mock_response,
            model="claude-sonnet-4-5-20250929",
            wall_time_seconds=1.0,
        )

        assert metrics.usage.cache_creation_input_tokens == 1000
        assert metrics.usage.cache_read_input_tokens == 2000
        assert metrics.cost.cache_write_cost > 0
        assert metrics.cost.cache_read_cost > 0

    def test_from_response_uses_default_pricing_for_unknown_model(self):
        """Test from_response falls back to default pricing."""
        mock_response = Mock()
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 1000
        mock_response.usage.output_tokens = 500
        mock_response.usage.cache_creation_input_tokens = 0
        mock_response.usage.cache_read_input_tokens = 0
        mock_response.id = None

        metrics = APICallMetrics.from_response(
            response=mock_response,
            model="unknown-model-xyz",
            wall_time_seconds=1.0,
        )

        # Should still calculate costs using default pricing
        assert metrics.cost.total_cost > 0

    def test_format_summary(self):
        """Test format_summary produces readable output."""
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            cache_creation_input_tokens=100,
            cache_read_input_tokens=200,
        )
        cost = CostBreakdown(
            input_cost=0.003,
            output_cost=0.0075,
            cache_write_cost=0.000375,
            cache_read_cost=0.00006,
        )
        metrics = APICallMetrics(
            model="claude-sonnet-4-5-20250929",
            usage=usage,
            cost=cost,
            wall_time_seconds=2.5,
        )

        summary = metrics.format_summary()

        assert "claude-sonnet-4-5-20250929" in summary
        assert "2.50s" in summary
        assert "1,000" in summary  # input tokens
        assert "500" in summary    # output tokens
        assert "Cache Write" in summary
        assert "Cache Read" in summary
        assert "$0.003000" in summary  # input cost

    def test_to_dict(self):
        """Test to_dict produces serializable output."""
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=500,
        )
        cost = CostBreakdown(
            input_cost=0.003,
            output_cost=0.0075,
        )
        metrics = APICallMetrics(
            model="claude-sonnet-4-5-20250929",
            usage=usage,
            cost=cost,
            wall_time_seconds=2.5,
            request_id="msg_123",
        )

        data = metrics.to_dict()

        assert data["model"] == "claude-sonnet-4-5-20250929"
        assert data["wall_time_seconds"] == 2.5
        assert data["request_id"] == "msg_123"
        assert data["usage"]["input_tokens"] == 1000
        assert data["usage"]["output_tokens"] == 500
        assert data["cost"]["input_cost"] == 0.003
        assert data["cost"]["output_cost"] == 0.0075
        assert "timestamp" in data

    def test_timestamp_is_set(self):
        """Test that timestamp is automatically set."""
        usage = TokenUsage()
        cost = CostBreakdown()
        metrics = APICallMetrics(
            model="test",
            usage=usage,
            cost=cost,
            wall_time_seconds=1.0,
        )

        assert isinstance(metrics.timestamp, datetime)
        # Should be recent
        delta = datetime.now() - metrics.timestamp
        assert delta.total_seconds() < 10
