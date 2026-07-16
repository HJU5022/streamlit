"""ModelGuard AI Streamlit 진입점. 실행: python -m streamlit run app.py"""

from __future__ import annotations

import os
from io import BytesIO

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.agent import run_tool_calling_agent
from src.charts import (
    calibration_chart, categorical_distribution, class_distribution, component_scores,
    confusion_heatmap, curves_chart, drift_heatmap, drift_ranking, health_gauge,
    missing_comparison, numeric_box, numeric_distribution, performance_comparison,
    probability_histogram, problem_severity_chart, quality_issue_chart, regression_scatter,
    residual_histogram, segment_error_heatmap, threshold_chart, threshold_error_chart,
    unique_value_chart, weak_segment_chart,
)
from src.data import (
    TASK_LABELS, detect_constant_columns, detect_id_candidates, infer_task_type,
    read_csv_flexible, split_single_dataframe, target_candidates,
)
from src.demo_data import generate_demo_suite
from src.service import run_full_analysis
from src.ui import apply_page_style, download_table, lock_page_language, render_hero, render_risk_card, render_section


# =============================================================================
# 1. 앱 설정과 캐시
# =============================================================================
st.set_page_config(page_title="ModelGuard AI", page_icon="◉", layout="wide")
load_dotenv()
lock_page_language()
apply_page_style()


@st.cache_data(show_spinner=False)
def cached_demo_suite():
    return generate_demo_suite()


@st.cache_data(show_spinner=False)
def cached_read_csv(raw: bytes) -> pd.DataFrame:
    return read_csv_flexible(BytesIO(raw))


@st.cache_resource(show_spinner=False)
def cached_analysis(reference, current, target, task, positive, excluded, threshold, prediction, probability):
    return run_full_analysis(reference, current, target, task, positive, list(excluded), threshold, prediction, probability)


def get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, os.getenv(name, default)))
    except Exception:
        return os.getenv(name, default)


# =============================================================================
# 2. 데이터 선택
# =============================================================================
render_hero()
with st.sidebar:
    st.markdown("## 검사 설정")
    st.caption("처음이라면 원클릭 데모를 선택하세요. 설정은 아래에서 자동으로 완성됩니다.")
    mode = st.radio("데이터 준비 방법", ["원클릭 데모", "CSV 한 파일", "과거·현재 CSV 두 파일"])

reference_df: pd.DataFrame | None = None
current_df: pd.DataFrame | None = None
scenario_label = ""
split_warning: str | None = None
demo_target: str | None = None

if mode == "원클릭 데모":
    suite = cached_demo_suite()
    with st.sidebar:
        demo_type = st.selectbox("분석할 모델 유형", list(suite), help="정답 값의 형태에 따라 이진 분류, 다중 분류 또는 회귀 분석을 선택합니다.")
        scenario = st.radio("데이터 상태", ["정상 데이터", "이상 징후 데이터"], horizontal=True)
        st.caption("개인정보가 없는 합성 데이터입니다. 실제 운영 결과가 아닙니다.")
    reference_demo, normal_demo, risk_demo = suite[demo_type]
    reference_df = reference_demo.copy()
    current_df = normal_demo.copy() if scenario == "정상 데이터" else risk_demo.copy()
    demo_target = {"이진 분류 (Binary Classification)":"y_true", "다중 분류 (Multiclass Classification)":"service_tier", "회귀 분석 (Regression)":"next_month_value"}[demo_type]
    scenario_label = f"원클릭 데모 · {demo_type} · {scenario}"

elif mode == "CSV 한 파일":
    with st.sidebar:
        uploaded = st.file_uploader("분석할 CSV", type=["csv"], key="single_csv")
    if uploaded:
        full = cached_read_csv(uploaded.getvalue())
        with st.sidebar:
            time_options = ["시간 정보 없음"] + [str(c) for c in full.columns]
            time_choice = st.selectbox("시간을 나타내는 열 (선택)", time_options)
            ratio = st.slider("과거 기준 데이터 비율", .40, .80, .60, .05)
        reference_df, current_df, split_warning = split_single_dataframe(
            full, ratio, None if time_choice == "시간 정보 없음" else time_choice
        )
        scenario_label = f"한 CSV를 과거·현재로 분리 · {uploaded.name}"

else:
    with st.sidebar:
        reference_file = st.file_uploader("과거 기준 CSV", type=["csv"], key="reference_csv")
        current_file = st.file_uploader("현재 운영 CSV", type=["csv"], key="current_csv")
    if reference_file and current_file:
        reference_df = cached_read_csv(reference_file.getvalue())
        current_df = cached_read_csv(current_file.getvalue())
        scenario_label = f"두 CSV 비교 · {reference_file.name} / {current_file.name}"

if reference_df is None or current_df is None:
    st.info("왼쪽에서 원클릭 데모를 선택하거나 CSV 파일을 업로드해 주세요.")
    st.stop()
if split_warning:
    st.warning(split_warning)


# =============================================================================
# 3. 정답·모델 유형 자동 설정
# =============================================================================
with st.sidebar:
    if demo_target:
        target_column = demo_target
        task = infer_task_type(reference_df[target_column])
        st.success(f"정답 열과 모델 유형을 자동으로 설정했습니다.\n\n**{task.label}**")
    else:
        ordered_targets = target_candidates(reference_df)
        target_column = st.selectbox(
            "정답 열 (Target column)",
            ordered_targets,
            help="예: 고장 여부, 이탈 여부, 상품 등급, 다음 달 매출. 지역·ID·시간처럼 모델이 참고만 하는 열은 선택하지 마세요.",
        )
        with st.expander("모델 유형을 직접 선택하려면 펼치세요"):
            labels = ["자동 판별"] + list(TASK_LABELS.values())
            override_label = st.selectbox("모델 유형", labels)
            reverse = {value:key for key,value in TASK_LABELS.items()}
            override = "auto" if override_label == "자동 판별" else reverse[override_label]
        try:
            task = infer_task_type(reference_df[target_column], override)
            st.caption(f"선택된 모델 유형: {task.label} · 정답 값 {reference_df[target_column].nunique(dropna=True):,}종류")
        except ValueError as error:
            st.error(str(error))
            st.caption("실제 결과가 기록된 다른 정답 열을 선택하거나 모델 유형을 직접 지정해 주세요.")
            st.stop()

    positive_class = None
    if task.kind == "binary":
        positive_class = st.selectbox(
            "중점적으로 탐지할 값 (Positive class)",
            list(task.classes),
            index=len(task.classes)-1,
            help="예: 1이 고장을 뜻한다면 1을 선택하세요. 선택한 값의 탐지율과 놓친 건수를 중점적으로 계산합니다.",
        )

    prediction_column = probability_column = None
    if mode == "과거·현재 CSV 두 파일":
        with st.expander("운영 모델의 예측 결과가 CSV에 있나요?"):
            prediction_options = ["없음 · 임시 모델 사용"] + [str(c) for c in current_df.columns if c != target_column]
            prediction_choice = st.selectbox("운영 모델 예측값 열 (Prediction)", prediction_options)
            prediction_column = None if prediction_choice.startswith("없음") else prediction_choice
            if task.kind == "binary":
                probability_options = ["없음"] + [str(c) for c in current_df.columns if c not in {target_column, prediction_column}]
                probability_choice = st.selectbox("Positive class 확률 열 (선택)", probability_options)
                probability_column = None if probability_choice == "없음" else probability_choice

    suggested_exclusions = list(dict.fromkeys(
        detect_id_candidates(reference_df, [target_column]) + detect_constant_columns(reference_df, [target_column])
        + [c for c in [prediction_column, probability_column] if c]
    ))
    excluded_columns = st.multiselect(
        "모델 입력에서 제외할 열",
        [c for c in reference_df.columns if c != target_column],
        default=[c for c in suggested_exclusions if c in reference_df.columns],
        help="고객 ID, 기록 ID, 시간처럼 정답과 직접 관계없는 식별 정보는 기본 제외합니다.",
    )
    threshold = .50
    if task.kind == "binary":
        threshold = st.slider("분류 기준 확률 (Threshold)", .10, .90, .50, .05,
                              help="예측 확률이 이 값 이상이면 선택한 Positive class로 분류합니다. 앱은 이 값을 자동으로 변경하지 않습니다.")


# =============================================================================
# 4. 진단 실행
# =============================================================================
try:
    with st.spinner("과거와 현재 데이터를 비교하고 모델 상태를 계산하고 있습니다..."):
        context = cached_analysis(reference_df, current_df, target_column, task, positive_class,
                                  tuple(excluded_columns), threshold, prediction_column, probability_column)
except Exception as error:
    st.error(f"분석을 완료하지 못했습니다: {error}")
    with st.expander("확인할 내용", expanded=True):
        st.markdown("- 과거와 현재 CSV에 같은 입력 열이 있는지 확인하세요.\n- 정답 열이 실제 결과인지 확인하세요.\n- 숫자 예측을 선택했다면 정답 값이 숫자인지 확인하세요.\n- 모든 입력 열을 제외하지 않았는지 확인하세요.")
    st.stop()

health = context["health"]
performance = context["performance"]
report = context["rule_report"]
model_run = context["model_run"]
high_drift = int((context["drift"]["상태"] == "위험").sum())
weak_count = int(((context["segments"]["기간"] == "현재") & context["segments"]["상태"].isin(["주의","위험"])).sum()) if not context["segments"].empty else 0
immediate_count = sum(action["priority"] == "즉시" for action in report["actions"])

st.markdown(f"<div class='context-strip notranslate' translate='no' lang='ko'><b>{scenario_label}</b> &nbsp;·&nbsp; {task.label} &nbsp;·&nbsp; 과거 {len(reference_df):,}행 / 현재 {len(current_df):,}행 &nbsp;·&nbsp; {model_run.prediction_source}</div>", unsafe_allow_html=True)


# =============================================================================
# 5. 핵심 결과와 탐색 구조
# =============================================================================
primary_cards = st.columns(3)
with primary_cards[0]: render_risk_card("모델 건강점수", f"{health['score']:.1f} / 100", health["risk"], "낮을수록 점검 우선순위가 높습니다.")
with primary_cards[1]: render_risk_card("현재 위험도", health["risk"], health["risk"], "건강점수와 핵심 문제를 종합한 단계입니다.")
with primary_cards[2]: render_risk_card("진단 신뢰도", health["confidence"], "분석 불가", "표본 수와 실제로 계산된 진단 항목을 기준으로 표시합니다.")

secondary_cards = st.columns(2)
with secondary_cards[0]: render_risk_card("변화가 큰 입력 열", f"{high_drift}개", "높음" if high_drift else "낮음", "과거와 분포가 크게 달라진 입력 열입니다.")
with secondary_cards[1]: render_risk_card("취약 그룹 / 즉시 조치", f"{weak_count} / {immediate_count}", "주의" if weak_count or immediate_count else "낮음", "취약 그룹 수와 지금 바로 확인할 조치 수입니다.")

st.write("")
tab_overview, tab_change, tab_performance, tab_segments, tab_agent = st.tabs(
    ["한눈에 보기", "데이터 변화", "예측 성능", "취약 그룹", "권장 조치"]
)


# =============================================================================
# 6. 한눈에 보기
# =============================================================================
with tab_overview:
    render_section("먼저 확인할 세 가지", "건강점수, 구성 점수, 문제의 심각도를 한 화면에서 봅니다. 점수가 낮은 구성 항목부터 상세 탭에서 원인을 확인하세요.", "OVERVIEW")
    a,b,c = st.columns([1,1.1,.85])
    with a: st.plotly_chart(health_gauge(health["score"], health["risk"]), width="stretch")
    with b: st.plotly_chart(component_scores(health["component_table"]), width="stretch")
    with c: st.plotly_chart(problem_severity_chart(report["problems"]), width="stretch")
    st.caption(health["disclaimer"])

    render_section("발견된 핵심 문제", "각 문제를 펼치면 계산 근거와 영향을 받은 입력 열을 확인할 수 있습니다.", "FINDINGS")
    for index, problem in enumerate(report["problems"], 1):
        icon = "●" if problem["severity"] == "높음" else "▲" if problem["severity"] == "주의" else "✓"
        with st.expander(f"{icon} {index}. {problem['title']} · {problem['severity']}", expanded=index <= 2):
            st.markdown(f"**근거**  \n{problem['evidence']}")
            if problem.get("affected_columns"): st.write("관련 입력 열:", ", ".join(problem["affected_columns"]))

    with st.expander("건강점수 계산 방법 보기"):
        st.dataframe(health["component_table"], width="stretch", hide_index=True)
        download_table("점수 계산표 다운로드", health["component_table"], "health_score_breakdown.csv", "health_download")


# =============================================================================
# 7. 데이터 변화
# =============================================================================
with tab_change:
    quality_tab, drift_tab = st.tabs(["데이터 품질", "입력값 변화"])
    with quality_tab:
        render_section("데이터 자체에 문제가 생겼나요?", "결측값, 자료형 변경, 새로운 범주를 과거와 현재 사이에서 비교합니다. 모델 성능이 떨어지기 전에 수집 오류를 먼저 찾는 단계입니다.", "DATA QUALITY")
        q1,q2 = st.columns(2)
        with q1: st.plotly_chart(missing_comparison(context["quality"]), width="stretch")
        with q2: st.plotly_chart(quality_issue_chart(context["quality"]), width="stretch")
        st.plotly_chart(unique_value_chart(context["quality"]), width="stretch")
        if target_column in current_df:
            st.plotly_chart(class_distribution(reference_df[target_column], current_df[target_column]), width="stretch")
            st.caption("실제 결과의 비율이 크게 바뀌면 운영 환경 자체가 달라졌을 수 있습니다.")
        with st.expander("전체 열 품질표", expanded=False):
            st.dataframe(context["quality"], width="stretch", hide_index=True)
            download_table("품질표 다운로드", context["quality"], "data_quality.csv", "quality_download")
        with st.expander("정답 유출 가능성 확인"):
            leakage = context["reference_profile"]["leakage_candidates"]
            if leakage:
                st.dataframe(pd.DataFrame(leakage), width="stretch", hide_index=True)
                st.warning("예측 시점에는 알 수 없는 정보가 입력에 포함됐는지 확인하세요. 자동 확정 결과는 아닙니다.")
            else: st.success("현재 검사 기준에서 뚜렷한 데이터 누수(Data Leakage) 후보가 발견되지 않았습니다.")

    with drift_tab:
        render_section("모델이 처음 보는 환경으로 바뀌었나요?", "과거와 현재의 분포 차이를 열별로 측정합니다. 점수가 클수록 모델이 학습하지 못한 환경일 가능성이 큽니다.", "DATA DRIFT")
        st.plotly_chart(drift_ranking(context["drift"]), width="stretch")
        st.plotly_chart(drift_heatmap(context["drift"]), width="stretch")
        with st.expander("전체 변화 측정표"):
            display = context["drift"].copy()
            for column in ["기준 대표값","현재 대표값","대표값 변화"]:
                if column in display: display[column] = display[column].astype(str)
            st.dataframe(display, width="stretch", hide_index=True)
            download_table("변화 측정표 다운로드", display, "drift_results.csv", "drift_download")
        options = context["numeric_columns"] + context["categorical_columns"]
        if options:
            selected = st.selectbox("자세히 비교할 입력 열", options)
            if selected in context["numeric_columns"]:
                x,y = st.columns(2)
                with x: st.plotly_chart(numeric_distribution(reference_df[selected], current_df[selected], selected), width="stretch")
                with y: st.plotly_chart(numeric_box(reference_df[selected], current_df[selected], selected), width="stretch")
                st.caption("분포의 중심과 범위가 함께 이동하면 고객 구성, 운영 정책 또는 수집 방식이 변했는지 확인하세요.")
            else:
                st.plotly_chart(categorical_distribution(reference_df[selected], current_df[selected], selected), width="stretch")
                st.caption("새로운 항목이나 비율 급변은 모델이 과거에 충분히 학습하지 못한 패턴일 수 있습니다.")


# =============================================================================
# 8. 예측 성능
# =============================================================================
with tab_performance:
    render_section("모델의 예측력이 과거보다 떨어졌나요?", "과거 검증 구간과 현재 데이터를 같은 기준으로 비교합니다. 현재 실제 결과가 없으면 성능을 추측하지 않고 이유를 표시합니다.", "MODEL PERFORMANCE")
    if performance.get("current") is None:
        st.info("현재 CSV에 실제 결과가 없어 성능은 계산하지 않았습니다. 데이터 품질과 입력값 변화는 계속 확인할 수 있습니다.")
    else:
        st.plotly_chart(performance_comparison(performance["table"]), width="stretch")
        st.dataframe(performance["table"], width="stretch", hide_index=True)
        download_table("성능 비교표 다운로드", performance["table"], "performance_comparison.csv", "performance_download")

        if task.kind == "regression":
            p1,p2 = st.columns(2)
            with p1: st.plotly_chart(regression_scatter(performance["current_actual_pred"], "현재 실제값과 예측값"), width="stretch")
            with p2: st.plotly_chart(residual_histogram(performance["current_actual_pred"], "현재 예측 오차 분포"), width="stretch")
            st.caption("점이 대각선에 가까울수록 정확합니다. 오차가 0을 중심으로 고르게 모이면 편향이 작은 모델입니다.")
        else:
            cm1,cm2 = st.columns(2)
            with cm1: st.plotly_chart(confusion_heatmap(performance["baseline_confusion"], "과거 검증 결과", performance["class_labels"]), width="stretch")
            with cm2: st.plotly_chart(confusion_heatmap(performance["current_confusion"], "현재 운영 결과", performance["class_labels"]), width="stretch")
            st.caption("대각선 칸은 맞힌 건수, 대각선 밖은 서로 잘못 구분한 건수입니다.")

        if task.kind == "binary":
            if performance["curves"]:
                c1,c2 = st.columns(2)
                with c1: st.plotly_chart(curves_chart(performance["curves"], "roc"), width="stretch")
                with c2: st.plotly_chart(curves_chart(performance["curves"], "pr"), width="stretch")
                st.caption("곡선이 좋은 방향에 가까울수록 정상과 문제를 더 잘 구분합니다. 데이터 불균형이 크면 PR Curve를 함께 보세요.")
            render_section("예측 확률을 믿을 수 있나요?", "모델이 80%라고 말한 사례가 실제로도 약 80% 발생하는지 확인합니다.", "PREDICTION RELIABILITY")
            if context["calibration"] and model_run.current_proba is not None:
                c1,c2 = st.columns([1.35,1])
                with c1: st.plotly_chart(calibration_chart(context["calibration"]), width="stretch")
                with c2: st.plotly_chart(probability_histogram(model_run.baseline_proba, model_run.current_proba), width="stretch")
                st.metric("현재 Brier Score · 낮을수록 좋음", f"{context['calibration']['current_brier']:.3f}", f"{context['calibration']['brier_delta']:+.3f}", delta_color="inverse")
            else: st.info("실제 결과와 예측 확률이 모두 있어야 확률 신뢰성을 계산할 수 있습니다.")
            render_section("판정 기준을 바꾸면 결과가 어떻게 달라지나요?", "확률 기준을 낮추면 문제를 더 많이 찾지만 정상 사례를 잘못 경고할 수 있습니다. 추천값은 자동 적용하지 않습니다.", "THRESHOLD SIMULATION")
            if not context["threshold"].empty:
                t1,t2 = st.columns(2)
                with t1: st.plotly_chart(threshold_chart(context["threshold"], threshold), width="stretch")
                with t2: st.plotly_chart(threshold_error_chart(context["threshold"], threshold), width="stretch")
                with st.expander("전체 기준값 비교표"): st.dataframe(context["threshold"], width="stretch", hide_index=True)
        else:
            st.info("예측 확률 보정과 Threshold 시뮬레이션은 이진 분류 모델에만 적용됩니다.")


# =============================================================================
# 9. 취약 그룹과 권장 조치
# =============================================================================
with tab_segments:
    render_section("어떤 그룹에서 유독 많이 틀리나요?", "전체 평균만 보면 가려지는 지역·상품 유형별 문제를 찾습니다. 표본이 작은 그룹은 확정하지 않습니다.", "WEAK SEGMENTS")
    segments = context["segments"]
    if segments.empty:
        st.info("현재 실제 결과 또는 비교 가능한 그룹 정보가 없어 취약 그룹을 계산하지 않았습니다.")
    else:
        segment_columns = segments["분석 컬럼"].unique().tolist()
        selected_segment = st.selectbox("비교할 그룹 열", segment_columns)
        st.plotly_chart(weak_segment_chart(segments, selected_segment, task.kind), width="stretch")
        st.plotly_chart(segment_error_heatmap(segments, selected_segment, task.kind), width="stretch")
        filtered = segments[segments["분석 컬럼"] == selected_segment]
        st.dataframe(filtered, width="stretch", hide_index=True)
        download_table("취약 그룹표 다운로드", filtered, "weak_segments.csv", "segment_download")
        st.caption("표본 30건 미만 또는 필요한 정답 사례가 부족한 그룹은 ‘표본 부족’으로 표시합니다.")

with tab_agent:
    render_section("지금 무엇을 해야 하나요?", "코드로 계산한 수치만 사용해 즉시·단기·지속 모니터링 조치를 나눠 제안합니다. 자동 재학습이나 자동 배포는 하지 않습니다.", "ACTION PLAN")
    st.markdown(f"### {report['summary']}")
    for action in report["actions"]:
        label = "즉시 확인" if action["priority"] == "즉시" else "단기 개선" if action["priority"] == "단기" else "지속 관찰"
        st.markdown(f"- **{label} · {action['action']}**  \n  <span style='color:#6e6e73'>{action['reason']}</span>", unsafe_allow_html=True)

    api_key = get_secret("OPENAI_API_KEY")
    ai_agent_enabled = get_secret("ENABLE_AI_AGENT", "false").strip().lower() in {"1", "true", "yes", "on"}
    if not api_key:
        st.info("현재 보고서는 계산 규칙으로 작성했습니다. API 키를 연결하면 실제 AI가 진단 도구를 선택해 보고서를 다시 구성할 수 있습니다.")
    elif not ai_agent_enabled:
        st.info("AI Agent 호출은 기본적으로 꺼져 있습니다. 비용과 데이터 전송 정책을 확인한 뒤 ENABLE_AI_AGENT=true로 명시적으로 활성화하세요.")
    else:
        st.caption("AI Agent를 실행하면 원본 CSV 전체가 아닌 진단 지표, 컬럼명, 그룹값이 설정된 OpenAI 호환 API로 전송됩니다. 민감한 이름과 그룹값은 사용 전에 익명화하세요.")
    if api_key and ai_agent_enabled and st.button("AI Agent로 보고서 다시 작성", type="primary"):
        try:
            with st.status("AI가 필요한 진단 도구를 선택하고 있습니다...", expanded=True) as status:
                answer, trace = run_tool_calling_agent(context, api_key, get_secret("OPENAI_MODEL", "gpt-4.1-mini"), get_secret("OPENAI_BASE_URL") or None)
                for item in trace: st.write(f"✓ {item['tool']}")
                status.update(label="AI 보고서 작성 완료", state="complete")
            st.session_state["agent_answer"] = answer
        except Exception as error: st.error(f"AI 보고서 생성 실패: {error}")
    if st.session_state.get("agent_answer"): st.markdown(st.session_state["agent_answer"])

    with st.expander("분석 한계와 안전장치", expanded=True):
        for limitation in report["limitations"]: st.markdown(f"- {limitation}")
        st.caption("건강점수는 모니터링 우선순위용 지표이며 전문 인증이나 절대적 안전 보장이 아닙니다.")
    with st.expander("실행된 진단 단계"):
        for step in ["데이터 구조 확인", "품질 비교", "입력값 변화 측정", "성능 비교", "예측 신뢰성 확인", "취약 그룹 탐색", "판정 기준 비교", "대응 조치 작성"]:
            st.markdown(f"✓ {step}")

st.divider()
st.caption("ModelGuard AI · 업로드한 데이터는 현재 앱 세션에서만 분석하며 별도 데이터베이스에 저장하지 않습니다. AI Agent를 실행하면 일부 진단 결과가 설정된 외부 API로 전송될 수 있습니다.")
