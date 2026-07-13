"""
Serialization helpers for turning Recommendation dataclasses into the flat
dict/JSON structure consumed by the AI narrative generator and the Power BI
export step.
"""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from src.recommendation_engine.rule_engine import Recommendation


def recommendation_to_dict(rec: Recommendation) -> dict:
    d = asdict(rec)
    d["recommendation_type"] = rec.recommendation_type.value
    d["risk_level"] = rec.risk_level.value
    return d


def recommendations_to_dataframe(recs: list[Recommendation]) -> pd.DataFrame:
    return pd.DataFrame([recommendation_to_dict(r) for r in recs])
