"""
Serialization helpers for turning Recommendation / EqualizationRecommendation
dataclasses into the flat dict/JSON structure consumed by the AI narrative
generator and the Power BI export step.
"""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from src.recommendation_engine.rule_engine import EqualizationRecommendation, Recommendation


def recommendation_to_dict(rec: Recommendation) -> dict:
    d = asdict(rec)
    d["recommendation_type"] = rec.recommendation_type.value
    d["risk_level"] = rec.risk_level.value
    return d


def recommendations_to_dataframe(recs: list[Recommendation]) -> pd.DataFrame:
    return pd.DataFrame([recommendation_to_dict(r) for r in recs])


def equalization_to_dict(rec: EqualizationRecommendation) -> dict:
    d = asdict(rec)
    d["risk_level"] = rec.risk_level.value
    return d


def equalization_to_dataframe(recs: list[EqualizationRecommendation]) -> pd.DataFrame:
    if not recs:
        return pd.DataFrame(
            columns=[
                "resource_type",
                "overloaded_mac_id",
                "overloaded_project",
                "overloaded_utilization_pct",
                "underloaded_mac_id",
                "underloaded_project",
                "underloaded_utilization_pct",
                "fleet_average_utilization_pct",
                "suggested_action",
                "risk_level",
            ]
        )
    return pd.DataFrame([equalization_to_dict(r) for r in recs])
