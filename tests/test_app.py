"""Streamlit 앱 시작 화면 스모크 테스트."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_starts_without_exception():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=60).run()
    assert not app.exception
    assert app.title or app.markdown

