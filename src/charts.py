"""ModelGuard AI의 Plotly 시각화 모음."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .config import RISK_COLORS


# =============================================================================
# 공통 스타일
# =============================================================================
def _layout(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=24, r=24, t=58, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system, BlinkMacSystemFont, Apple SD Gothic Neo, sans-serif", color="#1D1D1F"),
        legend_title_text="",
        hoverlabel=dict(bgcolor="white", font_color="#1D1D1F"),
    )
    fig.update_xaxes(gridcolor="rgba(0,0,0,.07)", zerolinecolor="rgba(0,0,0,.12)")
    fig.update_yaxes(gridcolor="rgba(0,0,0,.07)", zerolinecolor="rgba(0,0,0,.12)")
    return fig


def health_gauge(score: float, risk: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " / 100", "font": {"size": 38}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": RISK_COLORS.get(risk, "#14B8A6")},
                "steps": [
                    {"range": [0, 60], "color": "rgba(239,68,68,.20)"},
                    {"range": [60, 80], "color": "rgba(245,158,11,.20)"},
                    {"range": [80, 100], "color": "rgba(16,185,129,.20)"},
                ],
            },
        )
    )
    return _layout(fig, "모델 건강점수", 300)


def component_scores(component_table: pd.DataFrame) -> go.Figure:
    data = component_table.dropna(subset=["점수"]).copy()
    fig = px.bar(data, x="점수", y="구성 항목", orientation="h", color="점수", range_x=[0, 100], color_continuous_scale="Teal")
    fig.update_layout(coloraxis_showscale=False)
    return _layout(fig, "건강점수 구성", 330)


def problem_severity_chart(problems: list[dict]) -> go.Figure:
    counts = pd.Series([item["severity"] for item in problems]).value_counts().reindex(["높음", "주의", "낮음"], fill_value=0)
    fig = px.pie(values=counts.values, names=counts.index, hole=.68,
                 color=counts.index, color_discrete_map={"높음":"#FF3B30","주의":"#FF9F0A","낮음":"#34C759"})
    fig.update_traces(textinfo="label+value")
    return _layout(fig, "발견된 문제의 심각도", 300)


# =============================================================================
# 데이터 품질·분포
# =============================================================================
def missing_comparison(quality_table: pd.DataFrame, top_n: int = 15) -> go.Figure:
    data = quality_table[~quality_table["타깃 여부"]].copy()
    data["최대 결측률"] = data[["기준 결측률", "현재 결측률"]].max(axis=1)
    data = data.nlargest(top_n, "최대 결측률")
    long = data.melt(id_vars="컬럼", value_vars=["기준 결측률", "현재 결측률"], var_name="기간", value_name="결측률")
    fig = px.bar(long, x="컬럼", y="결측률", color="기간", barmode="group", color_discrete_map={"기준 결측률": "#38BDF8", "현재 결측률": "#F59E0B"})
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, "컬럼별 결측률 비교")


def quality_issue_chart(quality_table: pd.DataFrame) -> go.Figure:
    counts = quality_table["상태"].value_counts().reindex(["위험","주의","안정"], fill_value=0).reset_index()
    counts.columns = ["상태","컬럼 수"]
    fig = px.bar(counts, x="상태", y="컬럼 수", color="상태",
                 color_discrete_map={"위험":"#FF3B30","주의":"#FF9F0A","안정":"#34C759"}, text_auto=True)
    return _layout(fig, "데이터 품질 상태 요약", 320)


def unique_value_chart(quality_table: pd.DataFrame, top_n: int = 12) -> go.Figure:
    data = quality_table.assign(변화=lambda x: (x["현재 고유값"]-x["기준 고유값"]).abs()).nlargest(top_n,"변화")
    long = data.melt(id_vars="컬럼", value_vars=["기준 고유값","현재 고유값"], var_name="기간", value_name="고유값 수")
    fig = px.bar(long, x="컬럼", y="고유값 수", color="기간", barmode="group",
                 color_discrete_map={"기준 고유값":"#0071E3","현재 고유값":"#64D2FF"})
    return _layout(fig, "컬럼별 고유값 수 변화", 340)


def class_distribution(reference: pd.Series, current: pd.Series) -> go.Figure:
    frames = []
    for name, series in [("기준", reference), ("현재", current)]:
        counts_series = series.astype(str).value_counts(normalize=True)
        counts_series.index.name = "클래스"
        counts = counts_series.to_frame("비율").reset_index()
        counts["기간"] = name
        frames.append(counts)
    data = pd.concat(frames, ignore_index=True)
    fig = px.bar(data, x="클래스", y="비율", color="기간", barmode="group", color_discrete_map={"기준": "#38BDF8", "현재": "#14B8A6"})
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, "타깃 클래스 비율 비교", 340)


def drift_ranking(drift_table: pd.DataFrame, top_n: int = 10) -> go.Figure:
    data = drift_table.head(top_n).sort_values("드리프트 점수")
    fig = px.bar(
        data,
        x="드리프트 점수",
        y="컬럼",
        orientation="h",
        color="상태",
        color_discrete_map={"안정": "#10B981", "주의": "#F59E0B", "위험": "#EF4444"},
        hover_data=["측정 기준", "기준 대표값", "현재 대표값"],
    )
    return _layout(fig, f"드리프트 상위 {min(top_n, len(data))}개 컬럼")


def drift_heatmap(drift_table: pd.DataFrame) -> go.Figure:
    data = drift_table.sort_values("드리프트 점수", ascending=False)
    fig = px.imshow([data["드리프트 점수"].to_numpy()], x=data["컬럼"], y=["변화 강도"],
                    color_continuous_scale=[[0,"#E8F5EE"],[.5,"#FFCC00"],[1,"#FF3B30"]], aspect="auto", text_auto=".2f")
    return _layout(fig, "전체 컬럼 변화 지도", 230)


def numeric_distribution(reference: pd.Series, current: pd.Series, column: str) -> go.Figure:
    data = pd.concat(
        [
            pd.DataFrame({column: pd.to_numeric(reference, errors="coerce"), "기간": "기준"}),
            pd.DataFrame({column: pd.to_numeric(current, errors="coerce"), "기간": "현재"}),
        ],
        ignore_index=True,
    ).dropna()
    fig = px.histogram(data, x=column, color="기간", barmode="overlay", opacity=0.58, nbins=35, color_discrete_map={"기준": "#38BDF8", "현재": "#F59E0B"})
    return _layout(fig, f"{column}: 기준·현재 분포")


def numeric_box(reference: pd.Series, current: pd.Series, column: str) -> go.Figure:
    data = pd.concat(
        [
            pd.DataFrame({"값": pd.to_numeric(reference, errors="coerce"), "기간": "기준"}),
            pd.DataFrame({"값": pd.to_numeric(current, errors="coerce"), "기간": "현재"}),
        ],
        ignore_index=True,
    ).dropna()
    fig = px.box(data, x="기간", y="값", color="기간", points="outliers", color_discrete_map={"기준": "#38BDF8", "현재": "#F59E0B"})
    return _layout(fig, f"{column}: 범위와 이상치 비교", 330)


def categorical_distribution(reference: pd.Series, current: pd.Series, column: str, top_n: int = 15) -> go.Figure:
    categories = reference.fillna("(결측)").astype(str).value_counts().head(top_n).index.union(
        current.fillna("(결측)").astype(str).value_counts().head(top_n).index
    )
    frames = []
    for name, series in [("기준", reference), ("현재", current)]:
        rates = series.fillna("(결측)").astype(str).value_counts(normalize=True).reindex(categories, fill_value=0)
        frames.append(pd.DataFrame({"카테고리": rates.index, "비율": rates.values, "기간": name}))
    data = pd.concat(frames, ignore_index=True)
    fig = px.bar(data, x="카테고리", y="비율", color="기간", barmode="group", color_discrete_map={"기준": "#38BDF8", "현재": "#F59E0B"})
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, f"{column}: 카테고리 비율 비교")


# =============================================================================
# 성능·신뢰성·취약 구간
# =============================================================================
def performance_comparison(table: pd.DataFrame) -> go.Figure:
    data = table.copy()
    long = data.melt(id_vars="지표", value_vars=["기준", "현재"], var_name="기간", value_name="점수")
    fig = px.bar(long, x="지표", y="점수", color="기간", barmode="group", color_discrete_map={"기준": "#0071E3", "현재": "#64D2FF"})
    return _layout(fig, "기준·현재 성능지표")


def confusion_heatmap(matrix: np.ndarray, title: str, class_labels: list[str] | None = None) -> go.Figure:
    labels = class_labels if class_labels and len(class_labels) == matrix.shape[0] else [str(i) for i in range(matrix.shape[0])]
    fig = px.imshow(matrix, text_auto=True, color_continuous_scale="Blues",
                    x=[f"예측 {x}" for x in labels], y=[f"실제 {x}" for x in labels])
    fig.update_layout(coloraxis_showscale=False)
    return _layout(fig, title, 330)


def curves_chart(curves: dict, kind: str) -> go.Figure:
    data = curves[kind]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["baseline_x"], y=data["baseline_y"], name="기준", mode="lines", line=dict(color="#38BDF8", width=3)))
    fig.add_trace(go.Scatter(x=data["current_x"], y=data["current_y"], name="현재", mode="lines", line=dict(color="#14B8A6", width=3)))
    if kind == "roc":
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="무작위", mode="lines", line=dict(color="#64748B", dash="dash")))
        fig.update_xaxes(title="False Positive Rate")
        fig.update_yaxes(title="True Positive Rate")
        return _layout(fig, "ROC Curve")
    fig.update_xaxes(title="Recall")
    fig.update_yaxes(title="Precision")
    return _layout(fig, "Precision-Recall Curve")


def calibration_chart(calibration: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="완벽한 보정", mode="lines", line=dict(color="#64748B", dash="dash")))
    for label, key, color in [("기준", "baseline_curve", "#38BDF8"), ("현재", "current_curve", "#F59E0B")]:
        table = calibration[key]
        fig.add_trace(go.Scatter(x=table["예측 확률"], y=table["실제 양성률"], name=label, mode="lines+markers", line=dict(color=color, width=3)))
    fig.update_xaxes(title="평균 예측 확률", range=[0, 1])
    fig.update_yaxes(title="실제 양성률", range=[0, 1])
    return _layout(fig, "예측 확률 신뢰성")


def probability_histogram(baseline: np.ndarray, current: np.ndarray) -> go.Figure:
    data = pd.concat([pd.DataFrame({"예측 확률":baseline,"기간":"기준"}), pd.DataFrame({"예측 확률":current,"기간":"현재"})])
    fig = px.histogram(data, x="예측 확률", color="기간", barmode="overlay", opacity=.62, nbins=20,
                       color_discrete_map={"기준":"#0071E3","현재":"#FF9F0A"})
    return _layout(fig, "예측 확률 분포", 340)


def regression_scatter(table: pd.DataFrame, title: str) -> go.Figure:
    fig = px.scatter(table, x="실제값", y="예측값", color="오차", color_continuous_scale="RdBu_r", opacity=.68)
    low = min(table["실제값"].min(), table["예측값"].min()); high = max(table["실제값"].max(), table["예측값"].max())
    fig.add_shape(type="line", x0=low, y0=low, x1=high, y1=high, line=dict(color="#86868B", dash="dash"))
    return _layout(fig, title, 380)


def residual_histogram(table: pd.DataFrame, title: str) -> go.Figure:
    fig = px.histogram(table, x="오차", nbins=35, color_discrete_sequence=["#0071E3"])
    fig.add_vline(x=0, line_dash="dash", line_color="#FF3B30")
    return _layout(fig, title, 340)


def weak_segment_chart(segments: pd.DataFrame, column: str, task_kind: str = "binary") -> go.Figure:
    data = segments[(segments["분석 컬럼"] == column) & (segments["상태"] != "표본 부족")]
    metrics = ["MAE"] if task_kind == "regression" else ["Macro F1","Accuracy"] if task_kind == "multiclass" else ["Recall", "F1", "오탐률", "미탐률"]
    metrics = [metric for metric in metrics if metric in data.columns]
    long = data.melt(id_vars=["기간", "그룹"], value_vars=metrics, var_name="지표", value_name="점수")
    fig = px.bar(long, x="그룹", y="점수", color="기간", facet_row="지표", barmode="group", color_discrete_map={"기준": "#38BDF8", "현재": "#F59E0B"})
    fig.update_layout(height=650)
    return _layout(fig, f"{column}: 그룹별 성능 비교", 650)


def segment_error_heatmap(segments: pd.DataFrame, column: str, task_kind: str = "binary") -> go.Figure:
    data = segments[(segments["분석 컬럼"] == column) & (segments["기간"] == "현재") & (segments["상태"] != "표본 부족")]
    metrics = ["MAE"] if task_kind == "regression" else ["Macro F1","Accuracy"] if task_kind == "multiclass" else ["오탐률","미탐률","Recall","F1"]
    metrics = [m for m in metrics if m in data]
    if data.empty or not metrics:
        return go.Figure()
    matrix = data.set_index("그룹")[metrics].T
    fig = px.imshow(matrix, text_auto=".2f", color_continuous_scale="Blues", aspect="auto")
    return _layout(fig, f"{column}: 현재 그룹별 상세 지도", 320)


def threshold_chart(table: pd.DataFrame, current_threshold: float = 0.5) -> go.Figure:
    fig = go.Figure()
    for metric, color in [("Precision", "#38BDF8"), ("Recall", "#F59E0B"), ("F1", "#14B8A6")]:
        fig.add_trace(go.Scatter(x=table["임곗값"], y=table[metric], name=metric, mode="lines+markers", line=dict(color=color)))
    fig.add_vline(x=current_threshold, line_dash="dash", line_color="#EF4444", annotation_text="현재 임곗값")
    fig.update_yaxes(range=[0, 1])
    return _layout(fig, "임곗값별 Precision·Recall·F1")


def threshold_error_chart(table: pd.DataFrame, current_threshold: float = 0.5) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=table["임곗값"], y=table["오탐 수"], name="오탐 수", mode="lines+markers", line=dict(color="#38BDF8")))
    fig.add_trace(go.Scatter(x=table["임곗값"], y=table["미탐 수"], name="미탐 수", mode="lines+markers", line=dict(color="#EF4444")))
    fig.add_vline(x=current_threshold, line_dash="dash", line_color="#F59E0B")
    return _layout(fig, "임곗값별 오탐·미탐 개수")
