"""데모 데이터와 데이터 분리 테스트."""

import pandas as pd
from src.data import detect_id_candidates, infer_task_type, split_single_dataframe
from src.demo_data import generate_demo_frames, generate_multiclass_demo_frames, generate_regression_demo_frames


def test_demo_frames_have_expected_shape_and_target():
    reference, normal, drift = generate_demo_frames()
    assert len(reference) == 1600
    assert len(normal) == 700
    assert len(drift) == 700
    assert reference["y_true"].nunique() == 2
    assert "Jeju" in set(drift["region"].dropna())
    assert drift["signal_score"].isna().sum() > normal["signal_score"].isna().sum()


def test_single_dataframe_split_and_id_detection():
    reference, _, _ = generate_demo_frames()
    past, current, warning = split_single_dataframe(reference, 0.6, "timestamp")
    assert len(past) + len(current) == len(reference)
    assert warning is None
    assert "record_id" in detect_id_candidates(reference, excluded=["y_true"])


def test_task_type_detection_supports_all_model_types():
    binary, _, _ = generate_demo_frames()
    multi, _, _ = generate_multiclass_demo_frames()
    regression, _, _ = generate_regression_demo_frames()
    assert infer_task_type(binary["y_true"]).kind == "binary"
    assert infer_task_type(multi["service_tier"]).kind == "multiclass"
    assert infer_task_type(regression["next_month_value"]).kind == "regression"
