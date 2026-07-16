"""드리프트·성능 진단 테스트."""

from src.data import infer_feature_columns, infer_task_type
from src.demo_data import generate_demo_frames, generate_multiclass_demo_frames, generate_regression_demo_frames
from src.diagnostics import detect_data_quality_issues, detect_drift
from src.service import run_full_analysis


def test_injected_drift_is_detected():
    reference, _, drift = generate_demo_frames()
    numeric, categorical = infer_feature_columns(
        reference,
        drift,
        target_column="y_true",
        excluded=["record_id", "timestamp"],
    )
    table = detect_drift(reference, drift, numeric, categorical)
    dangerous = set(table.loc[table["상태"] == "위험", "컬럼"])
    assert dangerous & {"usage_frequency", "transaction_amount", "signal_score", "product_type", "region"}


def test_quality_detects_new_category_and_missingness():
    reference, _, drift = generate_demo_frames()
    table = detect_data_quality_issues(reference, drift, "y_true")
    region = table.loc[table["컬럼"] == "region"].iloc[0]
    signal = table.loc[table["컬럼"] == "signal_score"].iloc[0]
    assert region["새 카테고리 수"] >= 1
    assert signal["현재 결측률"] > signal["기준 결측률"]


def test_full_analysis_runs_for_normal_and_drift():
    reference, normal, drift = generate_demo_frames()
    task = infer_task_type(reference["y_true"])
    normal_result = run_full_analysis(reference, normal, "y_true", task, 1, ["record_id", "timestamp"])
    drift_result = run_full_analysis(reference, drift, "y_true", task, 1, ["record_id", "timestamp"])
    assert normal_result["performance"]["current"] is not None
    assert drift_result["performance"]["current"] is not None
    assert drift_result["health"]["score"] <= normal_result["health"]["score"]
    assert not drift_result["threshold"].empty


def test_multiclass_and_regression_analysis_run():
    for generator, target in [
        (generate_multiclass_demo_frames, "service_tier"),
        (generate_regression_demo_frames, "next_month_value"),
    ]:
        reference, normal, drift = generator()
        task = infer_task_type(reference[target])
        normal_result = run_full_analysis(reference, normal, target, task, None, ["record_id", "timestamp"])
        drift_result = run_full_analysis(reference, drift, target, task, None, ["record_id", "timestamp"])
        assert normal_result["performance"]["current"] is not None
        assert drift_result["health"]["score"] < normal_result["health"]["score"]
