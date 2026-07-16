"""CSV 로딩, 분리, 컬럼 탐지, 문제 유형 자동 판별."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import BinaryIO

import numpy as np
import pandas as pd


TASK_LABELS = {
    "binary": "이진 분류 (Binary Classification)",
    "multiclass": "다중 분류 (Multiclass Classification)",
    "regression": "회귀 분석 (Regression)",
}


@dataclass(frozen=True)
class TaskSpec:
    kind: str
    label: str
    classes: tuple[object, ...] = ()


def read_csv_flexible(file_or_path: str | BinaryIO | BytesIO) -> pd.DataFrame:
    """UTF-8과 CP949를 순서대로 시도해 CSV를 읽습니다."""
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(0)
            frame = pd.read_csv(file_or_path, encoding=encoding)
            if frame.empty:
                raise ValueError("CSV에 분석할 행이 없습니다.")
            return frame
        except UnicodeDecodeError as error:
            last_error = error
    raise ValueError("CSV를 읽지 못했습니다. UTF-8 또는 CP949로 다시 저장해 주세요.") from last_error


def detect_id_candidates(df: pd.DataFrame, excluded: list[str] | None = None) -> list[str]:
    """모델 입력으로 부적절한 식별자·시간 컬럼 후보를 찾습니다."""
    excluded_set = set(excluded or [])
    candidates: list[str] = []
    rows = max(len(df), 1)
    for column in df.columns:
        if column in excluded_set:
            continue
        series = df[column].dropna()
        if series.empty:
            continue
        name = column.lower()
        uniqueness = series.nunique() / rows
        name_hint = any(token in name for token in ("id", "uuid", "key", "index", "timestamp", "datetime"))
        datetime_dtype = pd.api.types.is_datetime64_any_dtype(series)
        high_cardinality_text = uniqueness >= 0.95 and not pd.api.types.is_numeric_dtype(series)
        if datetime_dtype or high_cardinality_text or (name_hint and uniqueness >= 0.5):
            candidates.append(column)
    return candidates


def detect_constant_columns(df: pd.DataFrame, excluded: list[str] | None = None) -> list[str]:
    excluded_set = set(excluded or [])
    return [c for c in df.columns if c not in excluded_set and df[c].nunique(dropna=False) <= 1]


def infer_feature_columns(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_column: str,
    excluded: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    excluded_set = set(excluded or []) | {target_column}
    common = [c for c in reference.columns if c in current.columns and c not in excluded_set]
    numeric = [c for c in common if pd.api.types.is_numeric_dtype(reference[c])]
    categorical = [c for c in common if c not in numeric]
    return numeric, categorical


def infer_task_type(series: pd.Series, override: str = "auto") -> TaskSpec:
    """정답 열을 보고 이진 분류·다중 분류·회귀 분석을 판별합니다."""
    clean = series.dropna()
    if clean.empty:
        raise ValueError("선택한 정답 열에 값이 없습니다.")
    unique = clean.unique().tolist()
    if len(unique) < 2:
        raise ValueError("정답이 한 종류뿐이라 모델을 학습하거나 비교할 수 없습니다.")

    if override != "auto":
        kind = override
    elif len(unique) == 2:
        kind = "binary"
    elif pd.api.types.is_numeric_dtype(clean) and len(unique) > max(20, int(len(clean) * 0.05)):
        kind = "regression"
    else:
        kind = "multiclass"

    if kind == "binary" and len(unique) != 2:
        raise ValueError("이진 분류를 선택했지만 정답 값이 정확히 두 종류가 아닙니다.")
    if kind == "multiclass" and not 3 <= len(unique) <= 100:
        raise ValueError("다중 분류는 정답 값이 3~100종류인 데이터에 적합합니다.")
    if kind == "regression" and not pd.api.types.is_numeric_dtype(clean):
        raise ValueError("회귀 분석을 사용하려면 정답 열이 숫자형이어야 합니다.")
    classes = tuple(sorted(unique, key=str)) if kind != "regression" else ()
    return TaskSpec(kind=kind, label=TASK_LABELS[kind], classes=classes)


def target_candidates(df: pd.DataFrame) -> list[str]:
    """정답일 가능성이 높은 열을 앞에 배치합니다. 모든 열 사용도 허용합니다."""
    hints = ("target", "label", "y_true", "outcome", "failure", "fraud", "churn", "defect", "price", "score")
    return sorted(df.columns, key=lambda c: (not any(h in c.lower() for h in hints), df[c].nunique(dropna=True), c))


def split_single_dataframe(
    df: pd.DataFrame,
    reference_ratio: float = 0.60,
    time_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    if not 0.3 <= reference_ratio <= 0.85:
        raise ValueError("기준 데이터 비율은 30%~85% 사이여야 합니다.")
    if len(df) < 100:
        raise ValueError("안정적인 비교를 위해 최소 100행 이상의 데이터가 필요합니다.")
    working = df.copy()
    warning = None
    if time_column and time_column in working.columns:
        parsed = pd.to_datetime(working[time_column], errors="coerce")
        if parsed.notna().mean() >= 0.8:
            working = working.assign(_time=parsed).sort_values("_time").drop(columns="_time")
        else:
            warning = "선택한 시간 열의 일부 값을 날짜로 읽지 못해 현재 행 순서를 사용했습니다."
    else:
        warning = "시간 정보가 없어 현재 행 순서를 과거→현재 순서로 가정했습니다."
    cut = int(len(working) * reference_ratio)
    return working.iloc[:cut].reset_index(drop=True), working.iloc[cut:].reset_index(drop=True), warning


def encode_binary_target(series: pd.Series, positive_class: object) -> np.ndarray:
    return (series.astype(str) == str(positive_class)).astype(int).to_numpy()
