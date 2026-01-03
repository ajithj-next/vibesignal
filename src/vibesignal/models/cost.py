"""Cost tracking models for Claude API usage."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# Pricing per million tokens (as of Jan 2025)
# https://www.anthropic.com/pricing
MODEL_PRICING = {
    # Claude 3.5 Sonnet
    "claude-3-5-sonnet-20241022": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,  # 1.25x input price
        "cache_read": 0.30,   # 0.1x input price
    },
    "claude-sonnet-4-5-20250929": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    # Claude 3.5 Haiku
    "claude-3-5-haiku-20241022": {
        "input": 0.80,
        "output": 4.00,
        "cache_write": 1.00,
        "cache_read": 0.08,
    },
    # Claude 3 Opus
    "claude-3-opus-20240229": {
        "input": 15.00,
        "output": 75.00,
        "cache_write": 18.75,
        "cache_read": 1.50,
    },
    # Default fallback (use sonnet pricing)
    "default": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
}


@dataclass
class TokenUsage:
    """Token usage breakdown from a Claude API call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens including cache tokens."""
        return self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.total_input_tokens + self.output_tokens


@dataclass
class CostBreakdown:
    """Cost breakdown for a Claude API call."""

    input_cost: float = 0.0
    output_cost: float = 0.0
    cache_write_cost: float = 0.0
    cache_read_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        """Total cost in USD."""
        return self.input_cost + self.output_cost + self.cache_write_cost + self.cache_read_cost

    @property
    def cache_savings(self) -> float:
        """Estimated savings from cache usage.

        Cache reads cost 0.1x input price, so savings is 0.9x what those tokens
        would have cost at full price.
        """
        # If we had cache reads, we saved 90% on those tokens
        if self.cache_read_cost > 0:
            full_price_would_be = self.cache_read_cost * 10  # Cache is 0.1x
            return full_price_would_be - self.cache_read_cost
        return 0.0


@dataclass
class APICallMetrics:
    """Complete metrics for a Claude API call."""

    model: str
    usage: TokenUsage
    cost: CostBreakdown
    wall_time_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)
    request_id: Optional[str] = None

    @classmethod
    def from_response(
        cls,
        response,
        model: str,
        wall_time_seconds: float,
    ) -> "APICallMetrics":
        """Create metrics from an Anthropic API response.

        Args:
            response: The Message response from anthropic SDK
            model: Model name used
            wall_time_seconds: Wall clock time for the API call

        Returns:
            APICallMetrics: Populated metrics object
        """
        # Extract usage from response
        usage_data = response.usage
        usage = TokenUsage(
            input_tokens=getattr(usage_data, "input_tokens", 0),
            output_tokens=getattr(usage_data, "output_tokens", 0),
            cache_creation_input_tokens=getattr(usage_data, "cache_creation_input_tokens", 0),
            cache_read_input_tokens=getattr(usage_data, "cache_read_input_tokens", 0),
        )

        # Calculate costs
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])
        cost = CostBreakdown(
            input_cost=(usage.input_tokens / 1_000_000) * pricing["input"],
            output_cost=(usage.output_tokens / 1_000_000) * pricing["output"],
            cache_write_cost=(usage.cache_creation_input_tokens / 1_000_000) * pricing["cache_write"],
            cache_read_cost=(usage.cache_read_input_tokens / 1_000_000) * pricing["cache_read"],
        )

        # Extract request ID if available
        request_id = getattr(response, "id", None)

        return cls(
            model=model,
            usage=usage,
            cost=cost,
            wall_time_seconds=wall_time_seconds,
            request_id=request_id,
        )

    def format_summary(self) -> str:
        """Format a human-readable summary of the metrics.

        Returns:
            str: Formatted summary string
        """
        lines = [
            f"Model: {self.model}",
            f"Wall Time: {self.wall_time_seconds:.2f}s",
            "",
            "Token Usage:",
            f"  Input:        {self.usage.input_tokens:,}",
            f"  Output:       {self.usage.output_tokens:,}",
        ]

        if self.usage.cache_creation_input_tokens > 0:
            lines.append(f"  Cache Write:  {self.usage.cache_creation_input_tokens:,}")
        if self.usage.cache_read_input_tokens > 0:
            lines.append(f"  Cache Read:   {self.usage.cache_read_input_tokens:,}")

        lines.append(f"  Total:        {self.usage.total_tokens:,}")
        lines.append("")
        lines.append("Cost Breakdown:")
        lines.append(f"  Input:        ${self.cost.input_cost:.6f}")
        lines.append(f"  Output:       ${self.cost.output_cost:.6f}")

        if self.cost.cache_write_cost > 0:
            lines.append(f"  Cache Write:  ${self.cost.cache_write_cost:.6f}")
        if self.cost.cache_read_cost > 0:
            lines.append(f"  Cache Read:   ${self.cost.cache_read_cost:.6f}")

        lines.append(f"  Total:        ${self.cost.total_cost:.6f}")

        if self.cost.cache_savings > 0:
            lines.append(f"  Cache Saved:  ${self.cost.cache_savings:.6f}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.

        Returns:
            dict: Serializable dictionary
        """
        return {
            "model": self.model,
            "wall_time_seconds": self.wall_time_seconds,
            "timestamp": self.timestamp.isoformat(),
            "request_id": self.request_id,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "cache_creation_input_tokens": self.usage.cache_creation_input_tokens,
                "cache_read_input_tokens": self.usage.cache_read_input_tokens,
                "total_tokens": self.usage.total_tokens,
            },
            "cost": {
                "input_cost": self.cost.input_cost,
                "output_cost": self.cost.output_cost,
                "cache_write_cost": self.cost.cache_write_cost,
                "cache_read_cost": self.cost.cache_read_cost,
                "total_cost": self.cost.total_cost,
                "cache_savings": self.cost.cache_savings,
            },
        }
