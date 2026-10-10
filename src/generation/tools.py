"""
Analytics tools for numeric reasoning over SEC 10-Q figures.

Plain retrieval + generation (src/generation/generator.py) works for lookup
questions ("what was Apple's iPhone revenue in Q3 2023?") but LLMs are
unreliable at multi-step arithmetic performed silently over numbers pulled
from prose context. Questions that require computing a period-over-period
change, a trend across several quarters, a ratio (margin, effective tax
rate), or a share-of-total are implemented here as callable tools instead,
exposed via OpenAI's function-calling schema (`FinancialAnalyticsTools.TOOLS`)
so the model calls them rather than computing inline.

Derived from qna_data_gold_page_present_25.csv — the analytics-style
questions there are:
  - trend across quarters: Apple/Microsoft net sales, Apple/NVIDIA inventory
  - ratio: Amazon gross margin drivers, Amazon effective tax rate
  - share-of-total: NVIDIA data-center share of total revenue
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _require_number(value: Any, label: str) -> float:
    """
    Validate a tool argument is an actual number before it reaches arithmetic.

    Forced tool-calling guarantees a call happens, not that every argument is
    populated — when the model can't find a figure in the retrieved context it
    may still emit null for a "number" field. Failing here with a clear
    message (instead of, e.g., a bare TypeError from inside sorted()) makes
    that case obvious in logs rather than a confusing stack trace.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number, got {value!r}")
    return float(value)


def _safe_divide(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _direction(percent_change: Optional[float]) -> str:
    if percent_change is None:
        return "undefined"
    if percent_change > 0:
        return "increase"
    if percent_change < 0:
        return "decrease"
    return "flat"


@dataclass
class TrendPoint:
    period: str
    value: float


class FinancialAnalyticsTools:
    """
    Numeric tools for financial-analytics questions that plain text
    generation answers unreliably: period-over-period change, multi-period
    trend, ratio/margin, share-of-total, and cross-entity comparison.

    Use `TOOLS` as the `tools=` payload for an OpenAI chat.completions call;
    route a returned tool_call's name/arguments through `call()`.
    """

    # ── calculate_change ──────────────────────────────────────────────
    @staticmethod
    def calculate_change(
        metric_name: str,
        previous_value: float,
        current_value: float,
        previous_period: str = "previous period",
        current_period: str = "current period",
    ) -> Dict[str, Any]:
        """Absolute and percent change between two values of the same metric."""
        previous_value = _require_number(previous_value, "previous_value")
        current_value = _require_number(current_value, "current_value")
        absolute_change = current_value - previous_value
        percent_change = _safe_divide(absolute_change, abs(previous_value))
        percent_change = percent_change * 100 if percent_change is not None else None
        return {
            "metric": metric_name,
            "previous_period": previous_period,
            "current_period": current_period,
            "previous_value": previous_value,
            "current_value": current_value,
            "absolute_change": absolute_change,
            "percent_change": round(percent_change, 2) if percent_change is not None else None,
            "direction": _direction(percent_change),
        }

    # ── analyze_trend ─────────────────────────────────────────────────
    @staticmethod
    def analyze_trend(
        metric_name: str,
        data_points: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Sequential and overall change across chronologically ordered data
        points, e.g. one figure per quarter across several 10-Q filings.

        data_points: [{"period": "2022 Q3", "value": 82959}, ...] in
        chronological order.
        """
        points = [
            TrendPoint(
                period=str(d.get("period", f"point {i}")),
                value=_require_number(d.get("value"), f"data_points[{i}].value"),
            )
            for i, d in enumerate(data_points)
        ]
        if len(points) < 2:
            raise ValueError("analyze_trend requires at least 2 data points")

        sequential_changes = [
            FinancialAnalyticsTools.calculate_change(
                metric_name, prev.value, curr.value, prev.period, curr.period,
            )
            for prev, curr in zip(points, points[1:])
        ]

        overall_change = FinancialAnalyticsTools.calculate_change(
            metric_name, points[0].value, points[-1].value, points[0].period, points[-1].period,
        )

        directions = {c["direction"] for c in sequential_changes}
        if directions == {"increase"}:
            pattern = "consistently increasing"
        elif directions == {"decrease"}:
            pattern = "consistently decreasing"
        elif directions == {"flat"}:
            pattern = "flat"
        else:
            pattern = "mixed / non-monotonic"

        return {
            "metric": metric_name,
            "data_points": [{"period": p.period, "value": p.value} for p in points],
            "sequential_changes": sequential_changes,
            "overall_change": overall_change,
            "pattern": pattern,
        }

    # ── calculate_ratio ───────────────────────────────────────────────
    @staticmethod
    def calculate_ratio(
        ratio_name: str,
        numerator_value: float,
        denominator_value: float,
        numerator_label: str = "numerator",
        denominator_label: str = "denominator",
    ) -> Dict[str, Any]:
        """
        Generic numerator/denominator ratio as a percentage — covers margins
        (gross margin = gross profit / revenue), effective tax rate (tax
        provision / pretax income), and similar single-period ratios.
        """
        numerator_value = _require_number(numerator_value, "numerator_value")
        denominator_value = _require_number(denominator_value, "denominator_value")
        ratio = _safe_divide(numerator_value, denominator_value)
        return {
            "ratio_name": ratio_name,
            "numerator_label": numerator_label,
            "numerator_value": numerator_value,
            "denominator_label": denominator_label,
            "denominator_value": denominator_value,
            "ratio_percent": round(ratio * 100, 2) if ratio is not None else None,
        }

    # ── calculate_share_of_total ──────────────────────────────────────
    @staticmethod
    def calculate_share_of_total(
        part_label: str,
        part_value: float,
        total_label: str,
        total_value: float,
    ) -> Dict[str, Any]:
        """
        What share of a total a part represents, e.g. NVIDIA data-center
        revenue as a percentage of total revenue.
        """
        ratio = FinancialAnalyticsTools.calculate_ratio(
            ratio_name=f"{part_label}_share_of_{total_label}",
            numerator_value=part_value,
            denominator_value=total_value,
            numerator_label=part_label,
            denominator_label=total_label,
        )
        return {
            "part_label": part_label,
            "part_value": part_value,
            "total_label": total_label,
            "total_value": total_value,
            "share_percent": ratio["ratio_percent"],
        }

    # ── compare_across_entities ────────────────────────────────────────
    @staticmethod
    def compare_across_entities(
        metric_name: str,
        values: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Rank and diff a metric reported for two or more labeled entities or
        periods, e.g. comparing the same metric across companies.
        """
        if len(values) < 2:
            raise ValueError("compare_across_entities requires at least 2 values")
        values = {label: _require_number(v, f"values[{label!r}]") for label, v in values.items()}
        ranked = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
        highest_label, highest_value = ranked[0]
        lowest_label, lowest_value = ranked[-1]
        return {
            "metric": metric_name,
            "ranked": [{"label": label, "value": value} for label, value in ranked],
            "highest": {"label": highest_label, "value": highest_value},
            "lowest": {"label": lowest_label, "value": lowest_value},
            "spread": highest_value - lowest_value,
        }

    # ── OpenAI function-calling glue ──────────────────────────────────
    TOOLS: List[Dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "calculate_change",
                "description": (
                    "Compute the absolute and percent change between two values of "
                    "the same financial metric across two periods (e.g. "
                    "quarter-over-quarter revenue change)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric_name": {"type": "string", "description": "e.g. 'total net sales'"},
                        "previous_value": {"type": "number"},
                        "current_value": {"type": "number"},
                        "previous_period": {"type": "string", "description": "e.g. '2022 Q3'"},
                        "current_period": {"type": "string", "description": "e.g. '2023 Q1'"},
                    },
                    "required": ["metric_name", "previous_value", "current_value"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_trend",
                "description": (
                    "Analyze how a single financial metric moved across 3 or more "
                    "chronologically ordered periods (e.g. one figure per quarter "
                    "across several 10-Q filings). Returns sequential "
                    "period-over-period changes, the overall change from first to "
                    "last period, and whether the trend is consistently "
                    "increasing, decreasing, or mixed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric_name": {"type": "string"},
                        "data_points": {
                            "type": "array",
                            "description": "Chronologically ordered values, one per period.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "period": {"type": "string", "description": "e.g. '2022 Q3'"},
                                    "value": {"type": "number"},
                                },
                                "required": ["period", "value"],
                            },
                            "minItems": 2,
                        },
                    },
                    "required": ["metric_name", "data_points"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_ratio",
                "description": (
                    "Compute a numerator/denominator ratio as a percentage. Use "
                    "for financial ratios such as gross margin (gross profit / "
                    "revenue), effective tax rate (tax provision / pretax "
                    "income), or operating margin."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ratio_name": {"type": "string", "description": "e.g. 'gross_margin'"},
                        "numerator_value": {"type": "number"},
                        "denominator_value": {"type": "number"},
                        "numerator_label": {"type": "string"},
                        "denominator_label": {"type": "string"},
                    },
                    "required": ["ratio_name", "numerator_value", "denominator_value"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_share_of_total",
                "description": (
                    "Compute what percentage share a part represents of a total, "
                    "e.g. one product segment's revenue as a percentage of total "
                    "company revenue."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "part_label": {"type": "string"},
                        "part_value": {"type": "number"},
                        "total_label": {"type": "string"},
                        "total_value": {"type": "number"},
                    },
                    "required": ["part_label", "part_value", "total_label", "total_value"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compare_across_entities",
                "description": (
                    "Rank and diff a metric reported for two or more labeled "
                    "entities or periods, e.g. comparing the same metric across "
                    "companies, or ranking multiple quarters by value."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric_name": {"type": "string"},
                        "values": {
                            "type": "object",
                            "description": 'Mapping of label -> numeric value, e.g. {"AAPL": 82959, "MSFT": 50122}',
                            "additionalProperties": {"type": "number"},
                        },
                    },
                    "required": ["metric_name", "values"],
                },
            },
        },
    ]

    @classmethod
    def call(cls, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dispatch a tool_call's name/arguments to the matching method.

        Raises KeyError for an unrecognized tool name (a real bug — the
        caller passed a name outside TOOLS) and ValueError/TypeError for
        missing or malformed arguments (e.g. a required figure came back
        null because it wasn't in the source text) — callers should treat
        those two failure modes differently: KeyError as a genuine defect,
        ValueError/TypeError as "this calculation can't be run" and fall
        back to answering without it.
        """
        dispatch = {
            "calculate_change": cls.calculate_change,
            "analyze_trend": cls.analyze_trend,
            "calculate_ratio": cls.calculate_ratio,
            "calculate_share_of_total": cls.calculate_share_of_total,
            "compare_across_entities": cls.compare_across_entities,
        }
        if name not in dispatch:
            raise KeyError(f"Unknown tool: {name}")
        return dispatch[name](**arguments)


if __name__ == "__main__":
    import json

    # Apple total net sales — Multi-Doc RAG trend question from the gold set.
    trend = FinancialAnalyticsTools.call(
        "analyze_trend",
        {
            "metric_name": "Apple total net sales",
            "data_points": [
                {"period": "2022 Q3", "value": 82959},
                {"period": "2023 Q1", "value": 117154},
                {"period": "2023 Q2", "value": 94836},
                {"period": "2023 Q3", "value": 81797},
            ],
        },
    )
    print(json.dumps(trend, indent=2))

    # NVIDIA data-center share of Q3 2023 revenue — Single-Doc Single-Chunk question.
    share = FinancialAnalyticsTools.call(
        "calculate_share_of_total",
        {
            "part_label": "data center revenue",
            "part_value": 14514,
            "total_label": "total revenue",
            "total_value": 18120,
        },
    )
    print(json.dumps(share, indent=2))
