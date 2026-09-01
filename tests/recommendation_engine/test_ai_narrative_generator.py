from unittest.mock import MagicMock, patch

from src.recommendation_engine.ai_narrative_generator import (
    _call_gemini,
    _call_ollama,
    generate_equalization_narrative,
    generate_equalization_narratives,
    generate_narrative,
    generate_narratives,
)
from src.recommendation_engine.rule_engine import (
    EqualizationRecommendation,
    Recommendation,
    RecommendationType,
    RiskLevel,
)


def _sample_recommendation(rec_type=RecommendationType.INCREASE_ALLOCATION):
    return Recommendation(
        mac_id="mac-01",
        resource_type="disk",
        project_name="Project Atlas",
        recommendation_type=rec_type,
        risk_level=RiskLevel.CRITICAL,
        current_value=82.0,
        forecasted_value=95.0,
        forecast_horizon_weeks=10,
        details={"threshold": 90},
    )


def _sample_equalization():
    return EqualizationRecommendation(
        resource_type="cpu",
        overloaded_mac_id="mac-01",
        overloaded_project="Project Atlas",
        overloaded_utilization_pct=92.0,
        underloaded_mac_id="mac-05",
        underloaded_project="Project Falcon",
        underloaded_utilization_pct=12.0,
        fleet_average_utilization_pct=50.0,
        suggested_action="Shift CPU load from mac-01 to mac-05 to balance the fleet.",
        risk_level=RiskLevel.CRITICAL,
    )


def test_generate_narrative_falls_back_to_template_when_no_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    rec = _sample_recommendation()
    narrative = generate_narrative(rec, provider="gemini")
    assert "mac-01" in narrative
    assert isinstance(narrative, str) and len(narrative) > 0


def test_generate_narrative_template_fallback_increase_allocation():
    rec = _sample_recommendation(RecommendationType.INCREASE_ALLOCATION)
    narrative = generate_narrative(rec, provider="template_fallback")
    assert "mac-01" in narrative
    assert "increase" in narrative.lower()


def test_generate_narrative_template_fallback_reduce_allocation_avoids_waste_wording():
    rec = Recommendation(
        mac_id="mac-02",
        resource_type="ram",
        project_name="Project Vega",
        recommendation_type=RecommendationType.REDUCE_ALLOCATION,
        risk_level=RiskLevel.WARNING,
        current_value=12.0,
        forecasted_value=12.0,
        forecast_horizon_weeks=0,
        details={"threshold": 30, "days_below_threshold": 25},
    )
    narrative = generate_narrative(rec, provider="template_fallback")
    assert "over-allocated" in narrative.lower()
    assert "wasted" not in narrative.lower() and "waste" not in narrative.lower()


def test_generate_narrative_no_action_case():
    rec = Recommendation(
        mac_id="mac-06",
        resource_type="cpu",
        project_name="Project Comet",
        recommendation_type=RecommendationType.NO_ACTION,
        risk_level=RiskLevel.INFO,
        current_value=55.0,
        forecasted_value=55.0,
        forecast_horizon_weeks=0,
    )
    narrative = generate_narrative(rec, provider="template_fallback")
    assert "no action" in narrative.lower()


@patch("src.recommendation_engine.ai_narrative_generator._call_gemini")
def test_generate_narrative_uses_gemini_response_when_available(mock_gemini):
    mock_gemini.return_value = "Disk is projected to reach capacity soon; increase allocation."
    rec = _sample_recommendation()
    narrative = generate_narrative(rec, provider="gemini")
    assert narrative == "Disk is projected to reach capacity soon; increase allocation."
    mock_gemini.assert_called_once()


def test_generate_narratives_batch():
    recs = [_sample_recommendation()]
    results = generate_narratives(recs, provider="template_fallback")
    assert len(results) == 1
    assert results[0]["mac_id"] == "mac-01"
    assert "narrative" in results[0]


@patch("src.recommendation_engine.ai_narrative_generator.requests.post")
def test_call_gemini_success(mock_post, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    mock_response = MagicMock()
    mock_response.json.return_value = {"candidates": [{"content": {"parts": [{"text": "Mac is nearly full."}]}}]}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    result = _call_gemini("some prompt")
    assert result == "Mac is nearly full."


def test_call_gemini_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert _call_gemini("some prompt") is None


@patch("src.recommendation_engine.ai_narrative_generator.requests.post")
def test_call_gemini_returns_none_on_request_failure(mock_post, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    mock_post.side_effect = Exception("network error")
    assert _call_gemini("some prompt") is None


@patch("src.recommendation_engine.ai_narrative_generator.requests.post")
def test_call_ollama_success(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "Mac is underutilized."}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response
    assert _call_ollama("some prompt") == "Mac is underutilized."


@patch("src.recommendation_engine.ai_narrative_generator.requests.post")
def test_call_ollama_returns_none_on_request_failure(mock_post):
    mock_post.side_effect = Exception("connection refused")
    assert _call_ollama("some prompt") is None


@patch("src.recommendation_engine.ai_narrative_generator._call_ollama")
def test_generate_narrative_uses_ollama_provider(mock_ollama):
    mock_ollama.return_value = "Fleet imbalance detected."
    rec = _sample_recommendation()
    result = generate_narrative(rec, provider="ollama")
    assert result == "Fleet imbalance detected."
    mock_ollama.assert_called_once()


def test_generate_equalization_narrative_falls_back_to_suggested_action():
    rec = _sample_equalization()
    narrative = generate_equalization_narrative(rec, provider="template_fallback")
    assert narrative == rec.suggested_action


@patch("src.recommendation_engine.ai_narrative_generator._call_gemini")
def test_generate_equalization_narrative_uses_gemini_when_available(mock_gemini):
    mock_gemini.return_value = "Rebalance CPU from mac-01 to mac-05 to equalize the fleet."
    rec = _sample_equalization()
    narrative = generate_equalization_narrative(rec, provider="gemini")
    assert narrative == "Rebalance CPU from mac-01 to mac-05 to equalize the fleet."


def test_generate_equalization_narratives_batch():
    recs = [_sample_equalization()]
    results = generate_equalization_narratives(recs, provider="template_fallback")
    assert len(results) == 1
    assert results[0]["overloaded_mac_id"] == "mac-01"
    assert results[0]["underloaded_mac_id"] == "mac-05"
    assert "narrative" in results[0]
