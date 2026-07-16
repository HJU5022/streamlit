"""진단 결과를 공개된 내부 평가 기준에 따라 건강점수로 변환합니다."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import HEALTH_WEIGHTS


def _clip(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


def calculate_health_score(
    quality_table: pd.DataFrame,
    drift_table: pd.DataFrame,
    performance: dict[str, Any],
    calibration: dict[str, Any] | None,
    weak_segments: pd.DataFrame,
    current_rows: int,
    task_kind: str = "binary",
) -> dict[str, Any]:
    """분석 가능한 구성 요소만 재가중하여 0~100점을 계산합니다."""
    components: dict[str, float | None] = {}
    reasons: dict[str, str] = {}

    high_quality = int((quality_table["상태"] == "위험").sum()) if not quality_table.empty else 0
    warn_quality = int((quality_table["상태"] == "주의").sum()) if not quality_table.empty else 0
    components["데이터 품질"] = _clip(100 - high_quality * 18 - warn_quality * 7)
    reasons["데이터 품질"] = f"위험 {high_quality}개, 주의 {warn_quality}개"

    if drift_table.empty:
        components["데이터 안정성"] = None
        reasons["데이터 안정성"] = "비교할 특징이 없음"
    else:
        high_drift = int((drift_table["상태"] == "위험").sum())
        warn_drift = int((drift_table["상태"] == "주의").sum())
        components["데이터 안정성"] = _clip(100 - high_drift * 16 - warn_drift * 6)
        reasons["데이터 안정성"] = f"위험 드리프트 {high_drift}개, 주의 {warn_drift}개"

    current_metrics = performance.get("current")
    if not current_metrics:
        components["모델 성능"] = None
        reasons["모델 성능"] = "현재 실제 정답이 없어 계산하지 않음"
    else:
        baseline = performance["baseline"]
        if task_kind == "regression":
            mae_increase = max(0.0, current_metrics["MAE"] / max(baseline["MAE"], 1e-9) - 1)
            rmse_increase = max(0.0, current_metrics["RMSE"] / max(baseline["RMSE"], 1e-9) - 1)
            r2_drop = max(0.0, baseline["R²"] - current_metrics["R²"])
            components["모델 성능"] = _clip(100 - mae_increase*55 - rmse_increase*35 - r2_drop*35)
            reasons["모델 성능"] = f"MAE {mae_increase*100:.1f}% 증가, RMSE {rmse_increase*100:.1f}% 증가, R² {r2_drop:.3f} 하락"
        elif task_kind == "multiclass":
            recall_drop = max(0.0, baseline.get("Macro Recall", 0)-current_metrics.get("Macro Recall", 0))
            f1_drop = max(0.0, baseline.get("Macro F1", 0)-current_metrics.get("Macro F1", 0))
            accuracy_drop = max(0.0, baseline.get("Balanced Accuracy", 0)-current_metrics.get("Balanced Accuracy", 0))
            components["모델 성능"] = _clip(100-recall_drop*150-f1_drop*140-accuracy_drop*80)
            reasons["모델 성능"] = f"Macro Recall {recall_drop*100:.1f}%p, Macro F1 {f1_drop*100:.1f}%p 하락"
        else:
            recall_drop = max(0.0, baseline.get("Recall", 0) - current_metrics.get("Recall", 0))
            f1_drop = max(0.0, baseline.get("F1", 0) - current_metrics.get("F1", 0))
            auc_drop = max(0.0, baseline.get("ROC-AUC", 0) - current_metrics.get("ROC-AUC", 0))
            components["모델 성능"] = _clip(100 - recall_drop * 170 - f1_drop * 120 - auc_drop * 80)
            reasons["모델 성능"] = f"Recall {recall_drop*100:.1f}%p, F1 {f1_drop*100:.1f}%p, ROC-AUC {auc_drop*100:.1f}%p 하락"

    if calibration is None:
        components["예측 신뢰성"] = None
        reasons["예측 신뢰성"] = "이진 분류의 실제 정답과 예측 확률이 모두 있을 때만 계산"
    else:
        current_brier = calibration["current_brier"]
        delta = max(0.0, calibration["brier_delta"])
        components["예측 신뢰성"] = _clip(100 - current_brier * 170 - delta * 180)
        reasons["예측 신뢰성"] = f"현재 Brier {current_brier:.3f}, 기준 대비 {delta:+.3f}"

    if weak_segments.empty:
        components["취약 구간 안정성"] = None
        reasons["취약 구간 안정성"] = "분석 가능한 범주형 그룹 또는 실제 정답이 없음"
    else:
        current_segments = weak_segments[weak_segments["기간"] == "현재"]
        high_segments = int((current_segments["상태"] == "위험").sum())
        warn_segments = int((current_segments["상태"] == "주의").sum())
        components["취약 구간 안정성"] = _clip(100 - high_segments * 18 - warn_segments * 7)
        reasons["취약 구간 안정성"] = f"위험 그룹 {high_segments}개, 주의 그룹 {warn_segments}개"

    available = {name: score for name, score in components.items() if score is not None}
    weight_sum = sum(HEALTH_WEIGHTS[name] for name in available)
    total = sum(float(score) * HEALTH_WEIGHTS[name] for name, score in available.items()) / weight_sum
    total = round(_clip(total), 1)
    risk = "낮음" if total >= 80 else "주의" if total >= 60 else "높음"

    unavailable = [name for name, score in components.items() if score is None]
    confidence = "높음" if current_rows >= 500 and len(unavailable) == 0 else "중간" if current_rows >= 200 else "낮음"
    component_table = pd.DataFrame(
        [
            {
                "구성 항목": name,
                "점수": components[name],
                "기본 가중치": HEALTH_WEIGHTS[name],
                "계산 근거": reasons[name],
                "상태": "계산 완료" if components[name] is not None else "분석 불가",
            }
            for name in HEALTH_WEIGHTS
        ]
    )
    return {
        "score": total,
        "risk": risk,
        "confidence": confidence,
        "components": components,
        "component_table": component_table,
        "unavailable": unavailable,
        "disclaimer": "건강점수는 점검 우선순위를 정하기 위한 내부 평가 지표이며 전문 인증이나 절대적인 안전 보장이 아닙니다.",
    }
