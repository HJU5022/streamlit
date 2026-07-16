"""규칙 기반 보고서와 실제 LLM Tool Calling Agent."""

from __future__ import annotations

import json
from typing import Any, Callable

import pandas as pd


# =============================================================================
# JSON 변환 보조
# =============================================================================
def _records(value: Any, limit: int = 20) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.head(limit).replace({float("nan"): None}).to_dict(orient="records")
    return value


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, pd.DataFrame):
        return _records(value)
    return str(value)


# =============================================================================
# API 키가 없어도 동작하는 근거 기반 보고서
# =============================================================================
def build_rule_based_report(context: dict[str, Any]) -> dict[str, Any]:
    health = context["health"]
    drift = context["drift"]
    performance = context["performance"]
    calibration = context["calibration"]
    segments = context["segments"]
    threshold = context["threshold"]
    task_kind = context["task"].kind

    problems: list[dict[str, Any]] = []
    high_drift = drift[drift["상태"] == "위험"] if not drift.empty else drift
    warn_drift = drift[drift["상태"] == "주의"] if not drift.empty else drift
    if not high_drift.empty:
        names = high_drift["컬럼"].head(5).tolist()
        problems.append({"title": "강한 데이터 드리프트", "severity": "높음", "evidence": f"위험 기준을 넘은 컬럼 {len(high_drift)}개", "affected_columns": names})
    elif not warn_drift.empty:
        names = warn_drift["컬럼"].head(5).tolist()
        problems.append({"title": "데이터 분포 변화 관찰", "severity": "주의", "evidence": f"주의 기준을 넘은 컬럼 {len(warn_drift)}개", "affected_columns": names})

    if performance.get("current"):
        base, current = performance["baseline"], performance["current"]
        if task_kind == "regression":
            increase = current["MAE"] / max(base["MAE"], 1e-9) - 1
            if increase >= .15:
                problems.append({"title":"숫자 예측 오차 증가", "severity":"높음" if increase >= .35 else "주의",
                                 "evidence":f"MAE {base['MAE']:.3f}→{current['MAE']:.3f} ({increase*100:.1f}% 증가)", "affected_columns":[]})
        else:
            recall_name = "Recall" if task_kind == "binary" else "Macro Recall"
            f1_name = "F1" if task_kind == "binary" else "Macro F1"
            recall_drop = base[recall_name]-current[recall_name]
            f1_drop = base[f1_name]-current[f1_name]
            if recall_drop >= .10 or f1_drop >= .10:
                problems.append({"title":"모델 성능 저하", "severity":"높음" if recall_drop >= .20 else "주의",
                                 "evidence":f"{recall_name} {base[recall_name]:.3f}→{current[recall_name]:.3f}, {f1_name} {f1_drop*100:.1f}%p 하락", "affected_columns":[]})

    if calibration and calibration["brier_delta"] >= 0.03:
        problems.append({"title": "예측 확률 신뢰성 저하", "severity": "주의", "evidence": f"Brier Score {calibration['baseline_brier']:.3f}→{calibration['current_brier']:.3f}", "affected_columns": []})

    if not segments.empty:
        weak = segments[(segments["기간"] == "현재") & (segments["상태"].isin(["주의", "위험"]))]
        if not weak.empty:
            metric_name = "MAE" if task_kind == "regression" else "Recall" if task_kind == "binary" else "Macro F1"
            top = weak.sort_values(["상태", metric_name], ascending=[True, task_kind == "regression"]).iloc[0]
            severity = "높음" if top["상태"] == "위험" else "주의"
            problems.append({"title": "취약 구간 후보 발견", "severity": severity, "evidence": f"{top['분석 컬럼']}={top['그룹']} · n={int(top['표본 수'])}, {metric_name}={top[metric_name]:.3f}", "affected_columns": [top["분석 컬럼"]]})

    actions: list[dict[str, str]] = []
    if not high_drift.empty:
        actions.append({"priority": "즉시", "action": "드리프트 상위 컬럼의 수집·전처리 변경 여부를 확인", "reason": "입력 분포 변화가 모델 성능 저하의 원인일 수 있습니다."})
    if performance.get("current"):
        key = "MAE" if task_kind == "regression" else "Recall" if task_kind == "binary" else "Macro Recall"
        degraded = (performance["current"][key] > performance["baseline"][key]*1.15 if task_kind == "regression"
                    else performance["baseline"][key]-performance["current"][key] >= .10)
        if degraded:
            actions.append({"priority":"단기", "action":"최근 정답 데이터로 재학습 실험을 수행", "reason":f"현재 환경에서 {key}가 기준보다 악화되었습니다."})
    if task_kind == "binary" and not threshold.empty:
        best = threshold.loc[threshold["F1"].idxmax()]
        current_row = threshold.iloc[(threshold["임곗값"]-context["settings"]["threshold"]).abs().argmin()]
        improvement = float(best["F1"]-current_row["F1"])
        priority = "단기" if improvement >= .03 else "모니터링"
        actions.append({"priority": priority, "action": f"임곗값 {best['임곗값']:.2f} 후보를 비용·오류 관점에서 검토", "reason": f"현재 기준 대비 F1 변화 후보 {improvement:+.3f}; 자동 적용하지 않습니다."})
    if calibration and calibration["brier_delta"] >= 0.03:
        actions.append({"priority": "단기", "action": "확률 보정 실험을 검토", "reason": "현재 Brier Score가 기준보다 악화되었습니다."})
    actions.append({"priority": "모니터링", "action": "주간 단위로 드리프트와 그룹별 성능을 분리 추적", "reason": "일시적 변화와 지속적 변화를 구분하기 위해서입니다."})

    limitations = list(context.get("limitations", []))
    if not problems:
        problems.append({"title": "즉시 대응이 필요한 강한 신호 없음", "severity": "낮음", "evidence": "설정된 위험 기준을 넘는 핵심 지표가 없습니다.", "affected_columns": []})
    return {
        "mode": "규칙 기반 진단",
        "summary": f"모델 건강점수는 {health['score']:.1f}/100, 현재 위험도는 {health['risk']}입니다.",
        "risk_level": health["risk"],
        "confidence": health["confidence"],
        "problems": problems,
        "actions": actions,
        "limitations": limitations,
    }


# =============================================================================
# 실제 LLM Tool Calling
# =============================================================================
def build_tool_registry(context: dict[str, Any]) -> dict[str, Callable[[], Any]]:
    return {
        "get_health_summary": lambda: {
            "score": context["health"]["score"],
            "risk": context["health"]["risk"],
            "confidence": context["health"]["confidence"],
            "components": _records(context["health"]["component_table"]),
        },
        "get_top_drift": lambda: _records(context["drift"], 10),
        "get_performance_change": lambda: {
            "baseline": context["performance"].get("baseline"),
            "current": context["performance"].get("current"),
            "comparison": _records(context["performance"].get("table", pd.DataFrame())),
        },
        "get_calibration": lambda: None if context["calibration"] is None else {
            "baseline_brier": context["calibration"]["baseline_brier"],
            "current_brier": context["calibration"]["current_brier"],
            "assessment": context["calibration"]["assessment"],
        },
        "get_weak_segments": lambda: _records(context["segments"].query("기간 == '현재'") if not context["segments"].empty else context["segments"], 15),
        "get_threshold_options": lambda: _records(context["threshold"].sort_values("F1", ascending=False) if not context["threshold"].empty else context["threshold"], 5),
    }


def run_tool_calling_agent(
    context: dict[str, Any],
    api_key: str,
    model: str,
    base_url: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """OpenAI 호환 API로 실제 tool-calling 루프를 실행합니다."""
    from openai import OpenAI

    registry = build_tool_registry(context)
    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        }
        for name, description in {
            "get_health_summary": "코드로 계산된 건강점수와 구성 점수를 조회한다.",
            "get_top_drift": "드리프트가 큰 컬럼과 실제 측정값을 조회한다.",
            "get_performance_change": "기준과 현재 모델 성능의 실제 계산 결과를 조회한다.",
            "get_calibration": "예측 확률 보정 결과를 조회한다.",
            "get_weak_segments": "표본 조건을 통과한 현재 취약 구간 후보를 조회한다.",
            "get_threshold_options": "자동 적용하지 않을 임곗값 비교 후보를 조회한다.",
        }.items()
    ]
    client = OpenAI(api_key=api_key, base_url=base_url or None)
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "당신은 ML 모델 모니터링 Agent다. 반드시 제공된 도구의 수치만 사용하고 숫자, 컬럼명, "
                "표본 수를 만들지 마라. 필요한 도구를 호출한 뒤 한국어로 요약, 근거가 있는 문제, 즉시/단기/모니터링 "
                "조치, 한계를 작성하라. 재학습이나 임곗값 변경을 자동 실행한다고 말하지 마라."
            ),
        },
        {"role": "user", "content": "현재 모델 상태를 진단하고 근거 기반 대응 계획을 작성해 주세요."},
    ]
    trace: list[dict[str, Any]] = []
    for _ in range(8):
        response = client.chat.completions.create(model=model, messages=messages, tools=tools, tool_choice="auto")
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        if not message.tool_calls:
            return message.content or "Agent가 빈 응답을 반환했습니다.", trace
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            if name not in registry:
                result = {"error": f"허용되지 않은 도구: {name}"}
            else:
                result = registry[name]()
            trace.append({"tool": name, "result": result})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False, default=_json_default),
                }
            )
    raise RuntimeError("Agent가 최대 도구 호출 횟수를 초과했습니다.")
