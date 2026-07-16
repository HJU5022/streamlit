"""여러 진단 Tool을 안전한 순서로 실행하는 서비스 계층."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .agent import build_rule_based_report
from .data import TaskSpec, infer_feature_columns
from .diagnostics import (
    check_calibration,
    compare_model_performance,
    detect_data_quality_issues,
    detect_drift,
    find_weak_segments,
    profile_data,
    suggest_threshold,
)
from .modeling import train_and_predict
from .scoring import calculate_health_score


def run_full_analysis(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_column: str,
    task: TaskSpec,
    positive_class: object | None,
    excluded_columns: list[str],
    threshold: float = 0.50,
    prediction_column: str | None = None,
    probability_column: str | None = None,
) -> dict[str, Any]:
    """통계 도구 → 모델 → 점수 → 규칙 보고서를 순서대로 실행합니다."""
    numeric, categorical = infer_feature_columns(
        reference,
        current,
        target_column=target_column,
        excluded=excluded_columns,
    )
    if not numeric and not categorical:
        raise ValueError("공통 특징 컬럼이 없습니다. 제외 컬럼 설정을 확인해 주세요.")

    reference_profile = profile_data(reference, target_column)
    current_profile = profile_data(current, target_column if target_column in current.columns else None)
    quality = detect_data_quality_issues(reference, current, target_column)
    drift = detect_drift(reference, current, numeric, categorical)

    model_run = train_and_predict(
        reference=reference,
        current=current,
        target_column=target_column,
        task=task,
        positive_class=positive_class,
        numeric_columns=numeric,
        categorical_columns=categorical,
        threshold=threshold,
        prediction_column=prediction_column,
        probability_column=probability_column,
    )
    performance = compare_model_performance(
        model_run.baseline_y,
        model_run.baseline_pred,
        model_run.baseline_proba,
        model_run.current_y,
        model_run.current_pred,
        model_run.current_proba,
        task_kind=task.kind,
        class_labels=model_run.class_labels,
    )
    calibration = check_calibration(
        model_run.baseline_y,
        model_run.baseline_proba,
        model_run.current_y,
        model_run.current_proba,
    ) if task.kind == "binary" else None
    segments = find_weak_segments(
        model_run.baseline_frame,
        current,
        model_run.baseline_y,
        model_run.baseline_pred,
        model_run.current_y,
        model_run.current_pred,
        categorical,
        task_kind=task.kind,
    )
    threshold_table = suggest_threshold(model_run.current_y, model_run.current_proba) if task.kind == "binary" else pd.DataFrame()
    health = calculate_health_score(
        quality,
        drift,
        performance,
        calibration,
        segments,
        current_rows=len(current),
        task_kind=task.kind,
    )

    limitations: list[str] = []
    if model_run.current_y is None:
        limitations.append("현재 데이터에 실제 결과가 없어 성능 변화는 확정하지 않았습니다.")
    if task.kind != "binary":
        limitations.append("예측 확률 보정과 Threshold 비교는 이진 분류 모델에만 적용됩니다.")
    if len(current) < 200:
        limitations.append("현재 데이터의 표본 수가 200건 미만이라 진단 결과의 변동성이 클 수 있습니다.")
    if not segments.empty and (segments["상태"] == "표본 부족").any():
        limitations.append("표본이 30건 미만인 그룹은 취약 구간 확정 판정에서 제외했습니다.")
    limitations.append("자동 재학습이나 임곗값 변경은 수행하지 않으며 의사결정 전 비용 검토가 필요합니다.")

    context: dict[str, Any] = {
        "reference": reference,
        "current": current,
        "reference_profile": reference_profile,
        "current_profile": current_profile,
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "quality": quality,
        "drift": drift,
        "model_run": model_run,
        "performance": performance,
        "calibration": calibration,
        "segments": segments,
        "threshold": threshold_table,
        "health": health,
        "limitations": limitations,
        "task": task,
        "settings": {
            "target_column": target_column,
            "task_kind": task.kind,
            "task_label": task.label,
            "positive_class": str(positive_class),
            "prediction_column": prediction_column,
            "probability_column": probability_column,
            "excluded_columns": excluded_columns,
            "threshold": threshold,
        },
    }
    context["rule_report"] = build_rule_based_report(context)
    return context
