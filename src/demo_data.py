"""원클릭 데모용 합성 데이터 생성기.

중요: 여기의 컬럼명은 '데모 데이터'에만 사용됩니다.
진단 엔진은 이 컬럼명을 전혀 가정하지 않습니다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 내부 데이터 생성 함수
# =============================================================================
def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


def _make_frame(
    n_rows: int,
    seed: int,
    start_date: str,
    scenario: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    product_type = rng.choice(
        np.array(["L", "M", "H"], dtype=object),
        size=n_rows,
        p=[0.45, 0.35, 0.20],
    )
    region = rng.choice(
        np.array(["Seoul", "Busan", "Daejeon"], dtype=object),
        size=n_rows,
        p=[0.52, 0.30, 0.18],
    )
    usage = rng.normal(12.0, 3.2, n_rows).clip(0.5)
    amount = rng.lognormal(10.25, 0.42, n_rows)
    account_age = rng.gamma(3.0, 180.0, n_rows).clip(10, 2500)
    tickets = rng.poisson(1.4, n_rows)
    signal = rng.normal(0.2, 1.0, n_rows)

    if scenario == "drift":
        usage += 3.0
        amount *= 1.38
        signal -= 1.15
        product_type = rng.choice(
            np.array(["L", "M", "H"], dtype=object),
            size=n_rows,
            p=[0.68, 0.22, 0.10],
        )
        region = rng.choice(
            np.array(["Seoul", "Busan", "Daejeon", "Jeju"], dtype=object),
            size=n_rows,
            p=[0.37, 0.24, 0.14, 0.25],
        )

    # 기준 환경의 실제 관계입니다.
    logit = (
        -2.0
        + 0.22 * tickets
        - 0.035 * usage
        + 0.000012 * amount
        - 0.0010 * account_age
        - 0.85 * signal
        + 0.35 * (product_type == "L")
        + 0.18 * (region == "Busan")
    )
    if scenario == "drift":
        # 운영 환경에서 입력 분포뿐 아니라 정답을 만드는 관계도 바뀐
        # concept drift를 재현합니다. 기준 모델이 낮은 signal을 위험하다고
        # 배웠지만 현재는 반대 관계가 강해져 Recall/F1 및 확률 보정이 악화됩니다.
        logit = (
            -1.35
            + 0.12 * tickets
            + 0.020 * usage
            + 0.000004 * amount
            - 0.00035 * account_age
            + 0.95 * signal
            + 1.00 * (product_type == "L")
            + 0.75 * (region == "Jeju")
            + 0.40 * ((product_type == "L") & (region == "Busan"))
        )

    probability = _sigmoid(logit)
    target = rng.binomial(1, probability)

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(start_date, periods=n_rows, freq="h"),
            "record_id": [f"REC-{seed}-{i:05d}" for i in range(n_rows)],
            "product_type": product_type,
            "region": region,
            "usage_frequency": np.round(usage, 2),
            "transaction_amount": np.round(amount, 0),
            "account_age_days": np.round(account_age, 0),
            "support_tickets": tickets,
            "signal_score": np.round(signal, 3),
            "y_true": target,
        }
    )

    if scenario == "drift":
        # 운영 환경 악화를 재현하는 결측치 증가(의도적 시뮬레이션)
        for column, rate in [("signal_score", 0.08), ("region", 0.04)]:
            idx = rng.choice(frame.index, size=max(1, int(n_rows * rate)), replace=False)
            frame.loc[idx, column] = np.nan

    return frame


# =============================================================================
# 외부에서 호출하는 공개 함수
# =============================================================================
def generate_demo_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """기준, 정상 현재, 위험 현재 데이터 세 개를 반환합니다."""
    reference = _make_frame(1600, 42, "2025-01-01", "reference")
    normal = _make_frame(700, 43, "2025-04-01", "normal")
    drift = _make_frame(700, 44, "2025-05-01", "drift")
    return reference, normal, drift


def _multiclass_frame(rows: int, seed: int, start: str, risk: bool) -> pd.DataFrame:
    base = _make_frame(rows, seed, start, "drift" if risk else "reference").drop(columns="y_true")
    rng = np.random.default_rng(seed + 300)
    signal = base["signal_score"].fillna(base["signal_score"].median()).to_numpy()
    usage = base["usage_frequency"].to_numpy()
    amount = base["transaction_amount"].to_numpy()
    scores = np.column_stack([
        .5*signal - .04*usage,
        -.25*signal + .000018*amount,
        .05*usage - .000010*amount,
    ])
    if risk:
        scores[:, 0] -= .8
        scores[:, 2] += .7*(base["product_type"].to_numpy() == "L")
    scores += rng.normal(0, .55, scores.shape)
    chosen = scores.argmax(axis=1)
    if risk:
        # 특정 고객군의 등급 결정 규칙이 바뀌는 concept drift를 재현합니다.
        changed = base["product_type"].to_numpy() == "L"
        chosen[changed] = (chosen[changed] + 1) % 3
    base["service_tier"] = np.array(["Basic", "Plus", "Premium"])[chosen]
    return base


def generate_multiclass_demo_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        _multiclass_frame(1600, 142, "2025-01-01", False),
        _multiclass_frame(700, 143, "2025-04-01", False),
        _multiclass_frame(700, 144, "2025-05-01", True),
    )


def _regression_frame(rows: int, seed: int, start: str, risk: bool) -> pd.DataFrame:
    base = _make_frame(rows, seed, start, "drift" if risk else "reference").drop(columns="y_true")
    rng = np.random.default_rng(seed + 500)
    signal = base["signal_score"].fillna(base["signal_score"].median()).to_numpy()
    value = (12000 + .42*base["transaction_amount"].to_numpy() + 850*base["usage_frequency"].to_numpy()
             - 9*base["account_age_days"].to_numpy() - 2200*signal)
    if risk:
        value += 10000*(base["region"].fillna("Unknown").to_numpy() == "Jeju") - 3500*signal
        noise = rng.normal(0, 13000, rows)
    else:
        noise = rng.normal(0, 5500, rows)
    base["next_month_value"] = np.maximum(0, value + noise).round(0)
    return base


def generate_regression_demo_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        _regression_frame(1600, 242, "2025-01-01", False),
        _regression_frame(700, 243, "2025-04-01", False),
        _regression_frame(700, 244, "2025-05-01", True),
    )


def generate_demo_suite() -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """세 가지 모델 유형을 원클릭으로 시험할 수 있는 데모 모음."""
    return {
        "이진 분류 (Binary Classification)": generate_demo_frames(),
        "다중 분류 (Multiclass Classification)": generate_multiclass_demo_frames(),
        "회귀 분석 (Regression)": generate_regression_demo_frames(),
    }


def save_demo_csvs(output_dir: str | Path) -> list[Path]:
    """데모 프레임을 CSV 파일로 저장합니다."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    suites = generate_demo_suite()
    frames = [frame for trio in suites.values() for frame in trio]
    names = [
        "reference_demo.csv", "current_normal_demo.csv", "current_drift_demo.csv",
        "reference_multiclass_demo.csv", "current_multiclass_normal_demo.csv", "current_multiclass_drift_demo.csv",
        "reference_regression_demo.csv", "current_regression_normal_demo.csv", "current_regression_drift_demo.csv",
    ]
    paths: list[Path] = []
    for name, frame in zip(names, frames):
        path = directory / name
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        paths.append(path)
    return paths


if __name__ == "__main__":
    save_demo_csvs(Path(__file__).resolve().parents[1] / "data")
