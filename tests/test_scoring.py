"""건강점수의 범위와 결측 분석 재가중 테스트."""

import pandas as pd

from src.scoring import calculate_health_score


def test_health_score_stays_in_range_without_labels():
    quality = pd.DataFrame({"상태": ["안정", "주의", "위험"]})
    drift = pd.DataFrame({"상태": ["안정", "주의"]})
    result = calculate_health_score(
        quality_table=quality,
        drift_table=drift,
        performance={"baseline": {}, "current": None},
        calibration=None,
        weak_segments=pd.DataFrame(),
        current_rows=120,
    )
    assert 0 <= result["score"] <= 100
    assert "모델 성능" in result["unavailable"]
    assert result["confidence"] == "낮음"

