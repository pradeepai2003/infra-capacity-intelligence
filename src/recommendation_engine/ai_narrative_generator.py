"""
Generates human-readable AI narratives for both per-Mac recommendations and
fleet-wide equalization recommendations, using either Google AI Studio
(Gemini) or a local Ollama model. Falls back to a deterministic template if
neither is configured/reachable, so the pipeline never breaks in
offline/CI environments.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import requests

from src.recommendation_engine.rule_engine import EqualizationRecommendation, Recommendation

logger = logging.getLogger(__name__)

# Google's free tier caps requests-per-minute for Gemini models. Pacing calls
# at this interval keeps a batch of narratives well under that ceiling instead
# of firing them all at once and immediately triggering 429s.
GEMINI_PACING_SECONDS = 4.5

PROMPTS_DIR = Path(__file__).parent / "prompts"
NARRATIVE_TEMPLATE_PATH = PROMPTS_DIR / "narrative_prompt_template.txt"
EQUALIZATION_TEMPLATE_PATH = PROMPTS_DIR / "equalization_prompt_template.txt"


def _build_prompt(rec: Recommendation) -> str:
    template = NARRATIVE_TEMPLATE_PATH.read_text()
    return template.format(
        mac_id=rec.mac_id,
        project_name=rec.project_name,
        resource_type=rec.resource_type,
        recommendation_type=rec.recommendation_type.value,
        risk_level=rec.risk_level.value,
        current_value=round(rec.current_value, 2),
        forecasted_value=round(rec.forecasted_value, 2),
        forecast_horizon_weeks=rec.forecast_horizon_weeks,
        details=rec.details,
    )


def _build_equalization_prompt(rec: EqualizationRecommendation) -> str:
    template = EQUALIZATION_TEMPLATE_PATH.read_text()
    return template.format(
        resource_type=rec.resource_type,
        overloaded_mac_id=rec.overloaded_mac_id,
        overloaded_project=rec.overloaded_project,
        overloaded_utilization_pct=rec.overloaded_utilization_pct,
        underloaded_mac_id=rec.underloaded_mac_id,
        underloaded_project=rec.underloaded_project,
        underloaded_utilization_pct=rec.underloaded_utilization_pct,
        fleet_average_utilization_pct=rec.fleet_average_utilization_pct,
    )


def _template_fallback(rec: Recommendation) -> str:
    """Deterministic narrative with no external API call - used as a safe default."""
    if rec.recommendation_type.value == "no_action":
        return (
            f"{rec.mac_id} ({rec.project_name}) is currently operating within normal thresholds on "
            f"{rec.resource_type.upper()} ({round(rec.current_value, 1)}%). No action is required at this time."
        )
    if rec.recommendation_type.value == "increase_allocation":
        return (
            f"{rec.mac_id} ({rec.project_name}) is projected to reach {round(rec.forecasted_value, 1)}% "
            f"{rec.resource_type.upper()} utilization, risking a capacity shortfall. Recommended action: "
            f"increase its {rec.resource_type.upper()} allocation before this becomes a bottleneck."
        )
    # reduce_allocation
    return (
        f"{rec.mac_id} ({rec.project_name}) has been below {rec.details.get('threshold', 'the healthy')}% "
        f"{rec.resource_type.upper()} utilization for {rec.details.get('days_below_threshold', 'a sustained period')} "
        f"days, indicating it is over-allocated relative to its actual need. Recommended action: reduce its "
        f"{rec.resource_type.upper()} allocation to save cost and free up capacity for other projects."
    )


def _equalization_template_fallback(rec: EqualizationRecommendation) -> str:
    return rec.suggested_action


def _call_gemini(prompt: str, model: str = "gemini-flash-latest", max_retries: int = 3) -> str | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                url,
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()

        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            # 429 (rate limit) and 503 (transient overload) are worth retrying;
            # anything else (400/401/403/404) won't be fixed by waiting, so fail fast.
            if status not in (429, 503) or attempt == max_retries:
                logger.warning("Gemini call failed, falling back: %s", exc)
                return None

            retry_after = None
            if exc.response is not None:
                retry_after = exc.response.headers.get("Retry-After")
            wait_seconds = float(retry_after) if retry_after else (2**attempt) * 2  # 2s, 4s, 8s
            logger.warning(
                "Gemini call hit %s, retrying in %.0fs (attempt %d/%d)...",
                status,
                wait_seconds,
                attempt + 1,
                max_retries,
            )
            time.sleep(wait_seconds)

        except Exception as exc:  # noqa: BLE001 -- timeouts, connection errors, malformed responses, etc.
            if attempt == max_retries:
                logger.warning("Gemini call failed, falling back: %s", exc)
                return None
            wait_seconds = (2**attempt) * 2
            logger.warning("Gemini call error (%s), retrying in %.0fs...", exc, wait_seconds)
            time.sleep(wait_seconds)

    return None  # pragma: no cover -- unreachable, loop always returns or raises above


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


def _dispatch(prompt: str, provider: str) -> str | None:
    if provider == "gemini":
        return _call_gemini(prompt, model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"))
    if provider == "ollama":
        return _call_ollama(
            prompt,
            model=os.getenv("OLLAMA_MODEL", "llama3"),
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )
    return None


def generate_narrative(rec: Recommendation, provider: str = "gemini") -> str:
    result = _dispatch(_build_prompt(rec), provider)
    return result or _template_fallback(rec)


def generate_narratives(recs: list[Recommendation], provider: str = "gemini") -> list[dict]:
    return [
        {"mac_id": r.mac_id, "resource_type": r.resource_type, "narrative": generate_narrative(r, provider)}
        for r in recs
    ]


def generate_equalization_narrative(rec: EqualizationRecommendation, provider: str = "gemini") -> str:
    result = _dispatch(_build_equalization_prompt(rec), provider)
    return result or _equalization_template_fallback(rec)


def generate_equalization_narratives(recs: list[EqualizationRecommendation], provider: str = "gemini") -> list[dict]:
    return [
        {
            "overloaded_mac_id": r.overloaded_mac_id,
            "underloaded_mac_id": r.underloaded_mac_id,
            "narrative": generate_equalization_narrative(r, provider),
        }
        for r in recs
    ]
