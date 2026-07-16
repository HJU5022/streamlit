"""데이터 품질, 드리프트, 성능, 신뢰성, 취약 구간 진단 Tool."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import JS_DANGER, JS_WARNING, MIN_SEGMENT_SIZE, PSI_DANGER, PSI_WARNING
from .data import detect_constant_columns, detect_id_candidates, encode_binary_target


# =============================================================================
# 공통 보조 함수
# =============================================================================
def _status(score: float, warning: float, danger: float) -> str:
    if score >= danger:
        return "위험"
    if score >= warning:
        return "주의"
    return "안정"


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> dict[str, float]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "오탐률": _safe_rate(fp, fp + tn),
        "미탐률": _safe_rate(fn, fn + tp),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_proba)
        metrics["PR-AUC"] = average_precision_score(y_true, y_proba)
    return {key: float(value) for key, value in metrics.items()}


# =============================================================================
# Tool 1: 데이터 프로파일
# =============================================================================
def profile_data(df: pd.DataFrame, target_column: str | None = None) -> dict[str, Any]:
    excluded = [target_column] if target_column else []
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in df.columns if column not in numeric]
    id_candidates = detect_id_candidates(df, excluded=excluded)
    constant_columns = detect_constant_columns(df, excluded=excluded)

    leakage_candidates: list[dict[str, Any]] = []
    if target_column and target_column in df.columns and df[target_column].nunique(dropna=True) == 2:
        target = df[target_column].astype(str)
        for column in df.columns:
            if column == target_column:
                continue
            series = df[column]
            comparable = series.notna() & df[target_column].notna()
            if comparable.sum() < 30:
                continue
            equality = (series[comparable].astype(str) == target[comparable]).mean()
            reason = None
            strength = float(equality)
            if equality >= 0.98:
                reason = "타깃 값과 거의 동일"
            elif pd.api.types.is_numeric_dtype(series):
                encoded = pd.Series(pd.factorize(target[comparable])[0], index=target[comparable].index)
                corr = series[comparable].astype(float).corr(encoded.astype(float))
                if pd.notna(corr) and abs(corr) >= 0.98:
                    reason = "타깃과 비정상적으로 높은 상관"
                    strength = float(abs(corr))
            if reason:
                leakage_candidates.append({"column": column, "reason": reason, "strength": strength})

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "id_candidates": id_candidates,
        "constant_columns": constant_columns,
        "leakage_candidates": leakage_candidates,
        "target_distribution": (
            df[target_column].value_counts(dropna=False).to_dict()
            if target_column and target_column in df.columns
            else {}
        ),
    }


# =============================================================================
# Tool 2: 데이터 품질 비교
# =============================================================================
def detect_data_quality_issues(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_column: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_columns = list(dict.fromkeys(reference.columns.tolist() + current.columns.tolist()))
    for column in all_columns:
        ref_exists = column in reference.columns
        cur_exists = column in current.columns
        ref_missing = float(reference[column].isna().mean()) if ref_exists else np.nan
        cur_missing = float(current[column].isna().mean()) if cur_exists else np.nan
        ref_unique = int(reference[column].nunique(dropna=True)) if ref_exists else 0
        cur_unique = int(current[column].nunique(dropna=True)) if cur_exists else 0
        new_categories = 0
        dtype_changed = False
        if ref_exists and cur_exists:
            dtype_changed = str(reference[column].dtype) != str(current[column].dtype)
            # 식별자·시간처럼 거의 모든 행이 고유한 열은 "새 카테고리"로 보지 않습니다.
            is_low_cardinality = ref_unique <= min(100, max(20, int(len(reference) * .20)))
            if not pd.api.types.is_numeric_dtype(reference[column]) and is_low_cardinality:
                ref_values = set(reference[column].dropna().astype(str).unique())
                cur_values = set(current[column].dropna().astype(str).unique())
                new_categories = len(cur_values - ref_values)

        missing_delta = cur_missing - ref_missing if ref_exists and cur_exists else np.nan
        status = "안정"
        reason = "특이사항 없음"
        if not ref_exists or not cur_exists:
            status, reason = "위험", "기준/현재 중 한쪽에 컬럼이 없음"
        elif dtype_changed:
            status, reason = "위험", "자료형이 변경됨"
        elif pd.notna(missing_delta) and missing_delta >= 0.10:
            status, reason = "위험", "결측률이 10%p 이상 증가"
        elif new_categories > 0 or (pd.notna(missing_delta) and missing_delta >= 0.03):
            status, reason = "주의", "새 범주 또는 결측률 증가"

        rows.append(
            {
                "컬럼": column,
                "자료형(기준)": str(reference[column].dtype) if ref_exists else "없음",
                "자료형(현재)": str(current[column].dtype) if cur_exists else "없음",
                "기준 결측률": ref_missing,
                "현재 결측률": cur_missing,
                "결측률 변화": missing_delta,
                "기준 고유값": ref_unique,
                "현재 고유값": cur_unique,
                "새 카테고리 수": new_categories,
                "상태": status,
                "근거": reason,
                "타깃 여부": column == target_column,
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Tool 3: 데이터 드리프트
# =============================================================================
def _numeric_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if len(ref) < 20 or len(cur) < 20:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_counts = np.histogram(cur, bins=edges)[0] / len(cur)
    ref_counts = np.clip(ref_counts, 1e-6, None)
    cur_counts = np.clip(cur_counts, 1e-6, None)
    return float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))


def _categorical_js(reference: pd.Series, current: pd.Series) -> float:
    ref = reference.fillna("(결측)").astype(str)
    cur = current.fillna("(결측)").astype(str)
    categories = sorted(set(ref.unique()) | set(cur.unique()))
    ref_prob = ref.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    cur_prob = cur.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    return float(jensenshannon(ref_prob, cur_prob, base=2) ** 2)


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in numeric_columns:
        ref_values = pd.to_numeric(reference[column], errors="coerce").dropna()
        cur_values = pd.to_numeric(current[column], errors="coerce").dropna()
        psi = _numeric_psi(ref_values, cur_values)
        ks_stat, ks_p = (0.0, 1.0)
        if len(ref_values) >= 20 and len(cur_values) >= 20:
            ks_stat, ks_p = ks_2samp(ref_values, cur_values)
        ref_mean = float(ref_values.mean()) if not ref_values.empty else np.nan
        cur_mean = float(cur_values.mean()) if not cur_values.empty else np.nan
        rows.append(
            {
                "컬럼": column,
                "유형": "숫자형",
                "드리프트 점수": psi,
                "측정 기준": "PSI",
                "보조 통계": float(ks_stat),
                "보조 p-value": float(ks_p),
                "기준 대표값": ref_mean,
                "현재 대표값": cur_mean,
                "대표값 변화": cur_mean - ref_mean,
                "상태": _status(psi, PSI_WARNING, PSI_DANGER),
            }
        )

    for column in categorical_columns:
        js = _categorical_js(reference[column], current[column])
        ref_mode = reference[column].fillna("(결측)").astype(str).mode()
        cur_mode = current[column].fillna("(결측)").astype(str).mode()
        rows.append(
            {
                "컬럼": column,
                "유형": "범주형",
                "드리프트 점수": js,
                "측정 기준": "JS divergence",
                "보조 통계": np.nan,
                "보조 p-value": np.nan,
                "기준 대표값": ref_mode.iloc[0] if not ref_mode.empty else "-",
                "현재 대표값": cur_mode.iloc[0] if not cur_mode.empty else "-",
                "대표값 변화": "최빈값 비교",
                "상태": _status(js, JS_WARNING, JS_DANGER),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["컬럼", "유형", "드리프트 점수", "상태"])
    return pd.DataFrame(rows).sort_values("드리프트 점수", ascending=False).reset_index(drop=True)


# =============================================================================
# Tool 4: 성능 비교
# =============================================================================
def compare_model_performance(
    baseline_y: np.ndarray,
    baseline_pred: np.ndarray,
    baseline_proba: np.ndarray | None,
    current_y: np.ndarray | None,
    current_pred: np.ndarray,
    current_proba: np.ndarray | None,
    task_kind: str = "binary",
    class_labels: list[str] | None = None,
) -> dict[str, Any]:
    labels = class_labels or []
    if task_kind == "regression":
        def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
            return {
                "MAE": float(mean_absolute_error(y_true, y_pred)),
                "RMSE": float(mean_squared_error(y_true, y_pred) ** .5),
                "R²": float(r2_score(y_true, y_pred)),
            }
        baseline = regression_metrics(baseline_y, baseline_pred)
        result: dict[str, Any] = {
            "task_kind": task_kind, "baseline": baseline, "current": None,
            "table": pd.DataFrame(), "baseline_confusion": None, "current_confusion": None,
            "curves": {}, "class_labels": [],
            "baseline_actual_pred": pd.DataFrame({"실제값": baseline_y, "예측값": baseline_pred, "오차": baseline_y-baseline_pred}),
            "current_actual_pred": pd.DataFrame(),
        }
        if current_y is None:
            return result
        current = regression_metrics(current_y, current_pred)
        rows = []
        for metric in baseline:
            before, after = baseline[metric], current[metric]
            rows.append({"지표": metric, "기준": before, "현재": after, "변화": after-before,
                         "상대 변화율": (after-before)/abs(before) if before else np.nan})
        result.update({
            "current": current, "table": pd.DataFrame(rows),
            "current_actual_pred": pd.DataFrame({"실제값": current_y, "예측값": current_pred, "오차": current_y-current_pred}),
        })
        return result

    if task_kind == "binary":
        baseline = _classification_metrics(baseline_y, baseline_pred, baseline_proba)
    else:
        def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None) -> dict[str, float]:
            metrics = {
                "Accuracy": accuracy_score(y_true, y_pred),
                "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
                "Macro Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
                "Macro Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
                "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            }
            if proba is not None:
                metrics["Log Loss"] = log_loss(y_true, proba, labels=np.arange(proba.shape[1]))
            return {key: float(value) for key, value in metrics.items()}
        baseline = multiclass_metrics(baseline_y, baseline_pred, baseline_proba)

    result: dict[str, Any] = {
        "task_kind": task_kind,
        "baseline": baseline,
        "current": None,
        "table": pd.DataFrame(),
        "baseline_confusion": confusion_matrix(baseline_y, baseline_pred),
        "current_confusion": None,
        "curves": {},
        "class_labels": labels,
    }
    if current_y is None:
        return result
    current = (_classification_metrics(current_y, current_pred, current_proba)
               if task_kind == "binary" else multiclass_metrics(current_y, current_pred, current_proba))
    rows = []
    for metric in list(dict.fromkeys(list(baseline) + list(current))):
        before = baseline.get(metric, np.nan)
        after = current.get(metric, np.nan)
        absolute = after - before
        relative = absolute / abs(before) if before not in (0, np.nan) and pd.notna(before) else np.nan
        rows.append(
            {
                "지표": metric,
                "기준": before,
                "현재": after,
                "변화": absolute,
                "변화(%p)": absolute * 100 if metric != "Log Loss" else np.nan,
                "상대 변화율": relative,
            }
        )

    curves: dict[str, Any] = {}
    if task_kind == "binary" and baseline_proba is not None and current_proba is not None and len(np.unique(current_y)) == 2:
        base_fpr, base_tpr, _ = roc_curve(baseline_y, baseline_proba)
        cur_fpr, cur_tpr, _ = roc_curve(current_y, current_proba)
        base_precision, base_recall, _ = precision_recall_curve(baseline_y, baseline_proba)
        cur_precision, cur_recall, _ = precision_recall_curve(current_y, current_proba)
        curves = {
            "roc": {"baseline_x": base_fpr, "baseline_y": base_tpr, "current_x": cur_fpr, "current_y": cur_tpr},
            "pr": {"baseline_x": base_recall, "baseline_y": base_precision, "current_x": cur_recall, "current_y": cur_precision},
        }
    result.update(
        {
            "current": current,
            "table": pd.DataFrame(rows),
            "current_confusion": confusion_matrix(current_y, current_pred),
            "curves": curves,
        }
    )
    return result


# =============================================================================
# Tool 5: 확률 신뢰성
# =============================================================================
def check_calibration(
    baseline_y: np.ndarray,
    baseline_proba: np.ndarray,
    current_y: np.ndarray | None,
    current_proba: np.ndarray,
    bins: int = 10,
) -> dict[str, Any] | None:
    if baseline_proba is None or current_proba is None or baseline_proba.ndim != 1 or current_proba.ndim != 1:
        return None
    if current_y is None or len(np.unique(current_y)) < 2:
        return None
    base_true, base_pred = calibration_curve(baseline_y, baseline_proba, n_bins=bins, strategy="uniform")
    cur_true, cur_pred = calibration_curve(current_y, current_proba, n_bins=bins, strategy="uniform")
    base_brier = float(brier_score_loss(baseline_y, baseline_proba))
    cur_brier = float(brier_score_loss(current_y, current_proba))
    return {
        "baseline_brier": base_brier,
        "current_brier": cur_brier,
        "brier_delta": cur_brier - base_brier,
        "baseline_curve": pd.DataFrame({"예측 확률": base_pred, "실제 양성률": base_true}),
        "current_curve": pd.DataFrame({"예측 확률": cur_pred, "실제 양성률": cur_true}),
        "assessment": "과신 위험 증가" if cur_brier - base_brier >= 0.03 else "큰 변화 없음",
    }


# =============================================================================
# Tool 6: 취약 구간
# =============================================================================
def find_weak_segments(
    baseline_frame: pd.DataFrame,
    current_frame: pd.DataFrame,
    baseline_y: np.ndarray,
    baseline_pred: np.ndarray,
    current_y: np.ndarray | None,
    current_pred: np.ndarray,
    categorical_columns: list[str],
    task_kind: str = "binary",
    min_size: int = MIN_SEGMENT_SIZE,
) -> pd.DataFrame:
    if current_y is None:
        return pd.DataFrame()

    def collect(period: str, frame: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        if task_kind == "regression":
            overall_error = float(mean_absolute_error(y_true, y_pred))
        elif task_kind == "binary":
            overall = _classification_metrics(y_true, y_pred, None)
        else:
            overall_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        for column in categorical_columns:
            if column not in frame.columns or frame[column].nunique(dropna=False) > 30:
                continue
            groups = frame[column].fillna("(결측)").astype(str)
            for group in groups.unique():
                mask = groups.eq(group).to_numpy()
                count = int(mask.sum())
                if count == 0:
                    continue
                status = "표본 부족" if count < min_size else "안정"
                row: dict[str, Any] = {"기간": period, "분석 컬럼": column, "그룹": group, "표본 수": count}
                if task_kind == "regression":
                    mae = float(mean_absolute_error(y_true[mask], y_pred[mask]))
                    ratio = mae / max(overall_error, 1e-9)
                    if count >= min_size and ratio >= 1.5: status = "위험"
                    elif count >= min_size and ratio >= 1.25: status = "주의"
                    row.update({"MAE": mae, "전체 대비 오차 배수": ratio})
                elif task_kind == "binary":
                    positive_count = int(y_true[mask].sum())
                    negative_count = int(count-positive_count)
                    metrics = _classification_metrics(y_true[mask], y_pred[mask], None)
                    recall_gap = metrics["Recall"] - overall["Recall"]
                    fpr_gap = metrics["오탐률"] - overall["오탐률"]
                    enough = positive_count >= 10 and negative_count >= 10
                    if count < min_size or not enough: status = "표본 부족"
                    elif recall_gap <= -.20 or fpr_gap >= .20: status = "위험"
                    elif recall_gap <= -.10 or fpr_gap >= .10: status = "주의"
                    row.update({"양성 표본 수": positive_count, "음성 표본 수": negative_count, **metrics,
                                "Recall 전체 대비": recall_gap, "오탐률 전체 대비": fpr_gap})
                else:
                    macro_f1 = float(f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0))
                    gap = macro_f1-overall_f1
                    if count >= min_size and gap <= -.20: status = "위험"
                    elif count >= min_size and gap <= -.10: status = "주의"
                    row.update({"Accuracy": float(accuracy_score(y_true[mask], y_pred[mask])), "Macro F1": macro_f1,
                                "Macro F1 전체 대비": gap})
                row["상태"] = status
                collected.append(
                    row
                )
        return collected

    rows = collect("기준", baseline_frame, baseline_y, baseline_pred)
    rows += collect("현재", current_frame.reset_index(drop=True), current_y, current_pred)
    return pd.DataFrame(rows)


# =============================================================================
# Tool 7: 임곗값 시뮬레이션
# =============================================================================
def suggest_threshold(y_true: np.ndarray | None, probabilities: np.ndarray | None) -> pd.DataFrame:
    if y_true is None or probabilities is None or probabilities.ndim != 1 or len(np.unique(y_true)) < 2:
        return pd.DataFrame()
    rows = []
    for threshold in np.round(np.arange(0.10, 0.91, 0.05), 2):
        prediction = (probabilities >= threshold).astype(int)
        metrics = _classification_metrics(y_true, prediction, None)
        matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        rows.append(
            {
                "임곗값": threshold,
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1": metrics["F1"],
                "오탐 수": int(fp),
                "미탐 수": int(fn),
            }
        )
    return pd.DataFrame(rows)
