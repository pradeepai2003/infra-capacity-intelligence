from unittest.mock import MagicMock, patch

from src.recommendation_engine.ai_narrative_generator import (
    _call_gemini,
    _call_ollama,
    generate_narrative,
    generate_narratives,
)
from src.recommendation_engine.rule_engine import Recommendation, RecommendationType, RiskLevel


def _sample_recommendation():
    return Recommendation(
        resource_id="storage-01",
        resource_type="storage",
        recommendation_type=RecommendationType.INCREASE_STORAGE,
        risk_level=RiskLevel.CRITICAL,
        current_value=82.0,
        forecasted_value=95.0,
        forecast_horizon_weeks=10,
        details={"threshold": 90},
    )


def test_generate_narrative_falls_back_to_template_when_no_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    rec = _sample_recommendation()
    narrative = generate_narrative(rec, provider="gemini")
    assert "storage-01" in narrative
    assert isinstance(narrative, str)
    assert len(narrative) > 0


def test_generate_narrative_template_fallback_directly():
    rec = _sample_recommendation()
    narrative = generate_narrative(rec, provider="template_fallback")
    assert "storage-01" in narrative
    assert "increase storage capacity" in narrative.lower()


def test_generate_narrative_no_action_case():
    rec = Recommendation(
        resource_id="cluster-05",
        resource_type="compute",
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
    mock_gemini.return_value = "Storage is projected to reach capacity soon; increase allocation."
    rec = _sample_recommendation()
    narrative = generate_narrative(rec, provider="gemini")
    assert narrative == "Storage is projected to reach capacity soon; increase allocation."
    mock_gemini.assert_called_once()


def test_generate_narratives_batch():
    recs = [_sample_recommendation()]
    results = generate_narratives(recs, provider="template_fallback")
    assert len(results) == 1
    assert results[0]["resource_id"] == "storage-01"
    assert "narrative" in results[0]


@patch("src.recommendation_engine.ai_narrative_generator.requests.post")
def test_call_gemini_success(mock_post, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    mock_response = MagicMock()
    mock_response.json.return_value = {"candidates": [{"content": {"parts": [{"text": "Storage is nearly full."}]}}]}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    result = _call_gemini("some prompt")
    assert result == "Storage is nearly full."
    mock_post.assert_called_once()


def test_call_gemini_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert _call_gemini("some prompt") is None


@patch("src.recommendation_engine.ai_narrative_generator.requests.post")
def test_call_gemini_returns_none_on_request_failure(mock_post, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    mock_post.side_effect = Exception("network error")

    result = _call_gemini("some prompt")
    assert result is None


@patch("src.recommendation_engine.ai_narrative_generator.requests.post")
def test_call_ollama_success(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "Cluster is underutilized."}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    result = _call_ollama("some prompt")
    assert result == "Cluster is underutilized."
    mock_post.assert_called_once()


@patch("src.recommendation_engine.ai_narrative_generator.requests.post")
def test_call_ollama_returns_none_on_request_failure(mock_post):
    mock_post.side_effect = Exception("connection refused")

    result = _call_ollama("some prompt")
    assert result is None


@patch("src.recommendation_engine.ai_narrative_generator._call_ollama")
def test_generate_narrative_uses_ollama_provider(mock_ollama):
    mock_ollama.return_value = "Network link is spiking."
    rec = _sample_recommendation()

    result = generate_narrative(rec, provider="ollama")
    assert result == "Network link is spiking."
    mock_ollama.assert_called_once()
