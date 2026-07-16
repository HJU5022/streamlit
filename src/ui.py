"""Streamlit 화면 스타일과 재사용 UI 부품."""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from .config import RISK_COLORS


def lock_page_language() -> None:
    """브라우저 자동 번역이 이미 작성된 한국어를 다시 번역하지 않도록 고정합니다."""
    locale_file = Path(__file__).resolve().parents[1] / "static" / "locale_lock.html"
    st.iframe(locale_file, height=1, width="stretch", tab_index=-1)


def apply_page_style() -> None:
    st.markdown("""
    <style>
    :root { --ink:#1d1d1f; --muted:#6e6e73; --line:#e5e5e7; --blue:#0071e3; --surface:#fff; }
    .stApp { background:#f5f5f7; color:var(--ink); }
    iframe[src*="locale_lock.html"] { position:absolute; width:1px !important; height:1px !important; opacity:0; pointer-events:none; }
    [data-testid="stSidebar"] { background:rgba(255,255,255,.96); border-right:1px solid var(--line); }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label { color:var(--ink); }
    .block-container { max-width:1280px; padding-top:2.2rem; padding-bottom:5rem; }
    h1,h2,h3 { color:var(--ink); letter-spacing:-.035em; font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif; }
    p,div,label,button,input { font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif; }
    .hero { padding:4.2rem 4.4rem; border-radius:32px; background:linear-gradient(145deg,#fff 0%,#f1f5ff 58%,#e9f7ff 100%);
            border:1px solid rgba(0,0,0,.05); box-shadow:0 18px 60px rgba(0,0,0,.06); margin-bottom:1.5rem; }
    .eyebrow { color:#0071e3; letter-spacing:.13em; font-size:.76rem; font-weight:700; text-transform:uppercase; }
    .hero-title { color:#1d1d1f; font-size:3.2rem; line-height:1.08; font-weight:750; margin:.65rem 0 1rem; letter-spacing:-.055em; }
    .hero-copy { color:#515154; font-size:1.12rem; max-width:820px; line-height:1.7; letter-spacing:-.015em; }
    .section-kicker { color:#0071e3; font-size:.78rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; margin-bottom:.35rem; }
    .section-copy { color:#6e6e73; font-size:.91rem; line-height:1.65; margin:-.35rem 0 1.1rem; max-width:900px; }
    .risk-card { min-height:118px; padding:1.35rem 1.45rem; border-radius:22px; background:#fff; border:1px solid rgba(0,0,0,.06);
                 box-shadow:0 8px 30px rgba(0,0,0,.045); }
    .risk-label { color:#6e6e73; font-size:.82rem; margin-bottom:.65rem; }
    .risk-value { font-size:1.55rem; font-weight:720; letter-spacing:-.04em; }
    .risk-help { color:#86868b; font-size:.72rem; margin-top:.4rem; line-height:1.45; }
    .context-strip { padding:1rem 1.25rem; background:#fff; border:1px solid var(--line); border-radius:16px; color:#515154; font-size:.86rem; }
    div[data-testid="stMetric"] { background:#fff; border:1px solid rgba(0,0,0,.06); padding:1.2rem; border-radius:20px; box-shadow:0 8px 26px rgba(0,0,0,.04); }
    div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:18px; overflow:hidden; background:#fff; }
    div[data-testid="stPlotlyChart"] { background:#fff; border:1px solid rgba(0,0,0,.055); border-radius:22px; padding:.35rem; box-shadow:0 8px 28px rgba(0,0,0,.035); }
    button[data-baseweb="tab"] { font-weight:650; color:#6e6e73; padding-left:1rem; padding-right:1rem; }
    button[data-baseweb="tab"][aria-selected="true"] { color:#0071e3; }
    [data-testid="stExpander"] { background:#fff; border:1px solid var(--line); border-radius:18px; }
    .stAlert { border-radius:16px; }
    .small-note { color:#86868b; font-size:.78rem; line-height:1.55; }
    @media(max-width:800px){ .hero{padding:2.2rem 1.5rem}.hero-title{font-size:2.25rem}.block-container{padding-top:1rem} }
    </style>
    """, unsafe_allow_html=True)


def render_hero() -> None:
    st.markdown("""
    <div class="hero notranslate" translate="no" lang="ko">
      <div class="eyebrow">ModelGuard AI · Model Monitoring Workspace</div>
      <div class="hero-title">AI 모델의 현재 상태를,<br>근거로 확인하세요.</div>
      <div class="hero-copy">과거에는 잘 작동했던 모델을 지금도 믿을 수 있을까요? 기준 데이터와 현재 데이터를 비교해
      입력 변화, 성능 저하, 예측 신뢰성, 취약 그룹을 한곳에서 점검하고 다음 조치를 정리합니다.</div>
    </div>
    """, unsafe_allow_html=True)


def render_section(title: str, description: str, kicker: str = "INSIGHT") -> None:
    st.markdown(f"<div class='notranslate' translate='no' lang='ko'><div class='section-kicker'>{html.escape(kicker)}</div><h3>{html.escape(title)}</h3>"
                f"<div class='section-copy'>{html.escape(description)}</div></div>", unsafe_allow_html=True)


def render_risk_card(label: str, value: str, risk: str = "분석 불가", help_text: str = "") -> None:
    color = RISK_COLORS.get(risk, "#0071E3")
    help_html = f"<div class='risk-help'>{html.escape(help_text)}</div>" if help_text else ""
    st.markdown(f"<div class='risk-card notranslate' translate='no' lang='ko'><div class='risk-label'>{html.escape(label)}</div>"
                f"<div class='risk-value' style='color:{color}'>{html.escape(value)}</div>{help_html}</div>", unsafe_allow_html=True)


def download_table(label: str, table, file_name: str, key: str) -> None:
    st.download_button(label, table.to_csv(index=False).encode("utf-8-sig"), file_name=file_name, mime="text/csv", key=key)
