from unittest.mock import patch

from src.recommendation_engine.ai_narrative_generator import generate_narrative, generate_narratives
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
