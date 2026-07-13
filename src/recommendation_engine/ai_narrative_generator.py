"""
Generates human-readable AI narratives for recommendations using either
Google AI Studio (Gemini) or a local Ollama model. Falls back to a
deterministic template if neither is configured/reachable, so the pipeline
never breaks in offline/CI environments.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import requests

from src.recommendation_engine.rule_engine import Recommendation

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "narrative_prompt_template.txt"


def _build_prompt(rec: Recommendation) -> str:
    template = PROMPT_TEMPLATE_PATH.read_text()
    return template.format(
        resource_id=rec.resource_id,
        resource_type=rec.resource_type,
        recommendation_type=rec.recommendation_type.value,
        risk_level=rec.risk_level.value,
        current_value=round(rec.current_value, 2),
        forecasted_value=round(rec.forecasted_value, 2),
        forecast_horizon_weeks=rec.forecast_horizon_weeks,
        details=rec.details,
    )


def _template_fallback(rec: Recommendation) -> str:
    """Deterministic narrative with no external API call - used as a safe default."""
    action_text = rec.recommendation_type.value.replace("_", " ")
    if rec.recommendation_type.value == "no_action":
        return (
            f"{rec.resource_id} is currently operating within normal thresholds "
            f"({round(rec.current_value, 1)}%). No action is required at this time."
        )
    horizon_text = f" within {rec.forecast_horizon_weeks} weeks" if rec.forecast_horizon_weeks else ""
    return (
        f"{rec.resource_id} is projected to reach {round(rec.forecasted_value, 1)}{horizon_text} "
        f"based on current trends (risk level: {rec.risk_level.value}). "
        f"Recommended action: {action_text}."
    )


def _call_gemini(prompt: str, model: str = "gemini-1.5-flash") -> str | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        resp = requests.post(
            url,
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gemini call failed, falling back: %s", exc)
        return None


def _call_ollama(prompt: str, model: str = "llama3", host: str = "http://localhost:11434") -> str | None:
    try:
        resp = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ollama call failed, falling back: %s", exc)
        return None


def generate_narrative(rec: Recommendation, provider: str = "gemini") -> str:
    """
    Args:
        rec: a single Recommendation
        provider: "gemini" | "ollama" | "template_fallback"
    """
    prompt = _build_prompt(rec)

    if provider == "gemini":
        result = _call_gemini(prompt, model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
    elif provider == "ollama":
        result = _call_ollama(
            prompt,
            model=os.getenv("OLLAMA_MODEL", "llama3"),
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )
    else:
        result = None

    return result or _template_fallback(rec)


def generate_narratives(recs: list[Recommendation], provider: str = "gemini") -> list[dict]:
    return [{"resource_id": r.resource_id, "narrative": generate_narrative(r, provider)} for r in recs]
