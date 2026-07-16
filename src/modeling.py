"""분류·회귀 공통 전처리 파이프라인과 임시 모델."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from .data import TaskSpec, encode_binary_target


@dataclass
class ModelRun:
    pipeline: Pipeline
    task: TaskSpec
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    baseline_frame: pd.DataFrame
    baseline_y: np.ndarray
    baseline_pred: np.ndarray
    baseline_proba: np.ndarray | None
    current_y: np.ndarray | None
    current_pred: np.ndarray
    current_proba: np.ndarray | None
    class_labels: list[str]
    prediction_source: str


def build_pipeline(numeric: list[str], categorical: list[str], task_kind: str) -> Pipeline:
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    if not transformers:
        raise ValueError("모델이 참고할 입력 정보가 없습니다. 제외 열 설정을 확인해 주세요.")
    estimator = (
        RandomForestRegressor(n_estimators=220, max_depth=10, min_samples_leaf=4, random_state=42, n_jobs=-1)
        if task_kind == "regression"
        else RandomForestClassifier(n_estimators=220, max_depth=9, min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=-1)
    )
    return Pipeline([("preprocessor", ColumnTransformer(transformers)), ("model", estimator)])


def _encode_targets(
    train: pd.Series,
    baseline: pd.Series,
    current: pd.Series | None,
    task: TaskSpec,
    positive_class: object | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[str], LabelEncoder | None]:
    if task.kind == "regression":
        return (
            pd.to_numeric(train, errors="raise").to_numpy(float),
            pd.to_numeric(baseline, errors="raise").to_numpy(float),
            None if current is None else pd.to_numeric(current, errors="raise").to_numpy(float),
            [], None,
        )
    if task.kind == "binary":
        positive = task.classes[-1] if positive_class is None else positive_class
        negative = next(value for value in task.classes if str(value) != str(positive))
        return (
            encode_binary_target(train, positive), encode_binary_target(baseline, positive),
            None if current is None else encode_binary_target(current, positive),
            [str(negative), str(positive)], None,
        )
    encoder = LabelEncoder().fit(pd.concat([train, baseline]).astype(str))
    return (
        encoder.transform(train.astype(str)), encoder.transform(baseline.astype(str)),
        None if current is None else encoder.transform(current.astype(str)),
        encoder.classes_.astype(str).tolist(), encoder,
    )


def train_and_predict(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_column: str,
    task: TaskSpec,
    positive_class: object | None,
    numeric_columns: list[str],
    categorical_columns: list[str],
    threshold: float = 0.50,
    prediction_column: str | None = None,
    probability_column: str | None = None,
) -> ModelRun:
    features = numeric_columns + categorical_columns
    if len(reference) < 80:
        raise ValueError("기준 데이터가 너무 적습니다. 최소 80행 이상을 권장합니다.")
    clean = reference.dropna(subset=[target_column]).reset_index(drop=True)
    cut = min(len(clean) - 1, max(1, int(len(clean) * .8)))
    train, baseline = clean.iloc[:cut], clean.iloc[cut:].reset_index(drop=True)
    current_target = current[target_column] if target_column in current and current[target_column].notna().all() else None
    train_y, baseline_y, current_y, labels, encoder = _encode_targets(
        train[target_column], baseline[target_column], current_target, task, positive_class
    )
    if task.kind != "regression" and len(np.unique(train_y)) < 2:
        raise ValueError("과거 학습 구간에 결과가 한 종류뿐입니다. 데이터 분할 또는 정답 열을 확인해 주세요.")

    pipeline = build_pipeline(numeric_columns, categorical_columns, task.kind)
    pipeline.fit(train[features], train_y)
    baseline_pred = pipeline.predict(baseline[features])
    model_current_pred = pipeline.predict(current[features])
    baseline_proba = pipeline.predict_proba(baseline[features]) if task.kind != "regression" else None
    model_current_proba = pipeline.predict_proba(current[features]) if task.kind != "regression" else None

    prediction_source = "과거 데이터로 학습한 임시 모델"
    current_pred = model_current_pred
    current_proba = model_current_proba
    if prediction_column and prediction_column in current:
        raw = current[prediction_column]
        if task.kind == "regression":
            current_pred = pd.to_numeric(raw, errors="raise").to_numpy(float)
        elif task.kind == "binary":
            current_pred = encode_binary_target(raw, positive_class if positive_class is not None else task.classes[-1])
        else:
            if encoder is None:
                raise ValueError("다중 분류 예측값을 변환하지 못했습니다.")
            unknown = set(raw.dropna().astype(str)) - set(encoder.classes_)
            if unknown:
                raise ValueError(f"예측 결과 열에 과거 정답에 없던 값이 있습니다: {sorted(unknown)[:5]}")
            current_pred = encoder.transform(raw.astype(str))
        prediction_source = f"업로드한 예측 결과 열 · {prediction_column}"

    if task.kind == "binary":
        baseline_proba = baseline_proba[:, 1]
        current_proba = model_current_proba[:, 1]
        if probability_column and probability_column in current:
            current_proba = pd.to_numeric(current[probability_column], errors="raise").to_numpy(float)
            if ((current_proba < 0) | (current_proba > 1)).any():
                raise ValueError("예측 확률은 0과 1 사이여야 합니다.")
            if not prediction_column:
                current_pred = (current_proba >= threshold).astype(int)
            prediction_source += f" · 확률 열 {probability_column}"
        baseline_pred = (baseline_proba >= threshold).astype(int)
        if not prediction_column:
            current_pred = (current_proba >= threshold).astype(int)

    return ModelRun(
        pipeline=pipeline, task=task, feature_columns=features,
        numeric_columns=numeric_columns, categorical_columns=categorical_columns,
        baseline_frame=baseline, baseline_y=np.asarray(baseline_y), baseline_pred=np.asarray(baseline_pred),
        baseline_proba=None if baseline_proba is None else np.asarray(baseline_proba),
        current_y=None if current_y is None else np.asarray(current_y), current_pred=np.asarray(current_pred),
        current_proba=None if current_proba is None else np.asarray(current_proba),
        class_labels=labels, prediction_source=prediction_source,
    )
