# ModelGuard AI

> **AI Model Health Inspector** — 기준 데이터와 현재 운영 데이터를 비교해 데이터 변화, 성능 저하, 예측 신뢰성, 취약 구간을 진단하는 Streamlit AI Agent MVP입니다.

## 왜 만들었나요?

모델은 배포 당시 정확했다고 해서 계속 정확한 것이 아닙니다. 고객 구성, 센서 범위, 상품 비율처럼 운영 환경이 바뀌면 입력 분포와 예측 성능도 달라질 수 있습니다. ModelGuard AI는 그 변화를 표·그래프·근거 수치로 보여주고, Python Tool의 결과를 바탕으로 대응 우선순위를 제안합니다.

## 핵심 기능

- 이진 분류·다중 분류·회귀 분석 모델의 정상/이상 징후 **원클릭 데모**
- CSV 한 개의 시간순 기준/현재 분리
- 기준 CSV와 현재 CSV 직접 비교
- 결측치, 새 범주, 자료형, ID·누수 후보 검사
- 숫자형 PSI·KS, 범주형 JS divergence
- Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- 다중 분류 Macro Precision·Recall·F1, Balanced Accuracy, Log Loss
- 회귀 MAE, RMSE, R², 실제값-예측값 및 오차 분포
- 혼동행렬, ROC, Precision-Recall, Calibration 비교
- 범주형 그룹별 취약 구간 후보 탐색
- 임곗값별 Precision·Recall·F1·오탐·미탐 비교
- 계산 근거가 공개된 건강점수
- API 키가 있을 때 실제 OpenAI 호환 Tool Calling
- API 키가 없어도 완전히 작동하는 규칙 기반 진단

## 빠른 실행

Python 3.11을 권장합니다.

```bash
cd ModelGuard_AI
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

처음에는 사이드바에서 `원클릭 데모 → 분석할 모델 유형 → 이상 징후 데이터`를 선택해 보세요.

## 프로젝트 구조

```text
app.py                 Streamlit 대표 실행 파일
ModelGuard_app_셀별학습.ipynb  app.py를 의미별 셀로 나눈 학습용 노트북
app_cells.py           VS Code의 `# %%` 셀 형식 학습용 사본
src/data.py            CSV 로딩·분리·컬럼 자동 탐지
src/demo_data.py       투명한 합성 시나리오 생성
src/modeling.py        누수 방지 Pipeline과 RandomForest
src/diagnostics.py     실제 통계·성능 Tool
src/scoring.py         건강점수 계산
src/charts.py          Plotly 비교 시각화
src/agent.py           규칙 보고서·실제 Tool Calling
src/service.py         전체 진단 실행 순서
src/ui.py              디자인과 공통 UI
tests/                 자동 테스트
```

노트북과 `app_cells.py`는 코드 흐름을 셀별로 공부하거나 복사하기 위한 자료입니다.
Streamlit 앱을 실행할 때는 항상 대표 파일인 `app.py`를 사용하세요.

## 입력 데이터

현재 앱은 다음 세 가지 표 형태 머신러닝 문제를 자동 판별합니다.

- 이진 분류(Binary Classification): 정상/고장, 이탈/유지, 0/1
- 다중 분류(Multiclass Classification): Basic/Plus/Premium처럼 세 종류 이상의 결과
- 회귀 분석(Regression): 매출, 금액, 수요처럼 연속적인 숫자

### CSV 한 개

타깃과 선택적 시간 컬럼을 지정하면 앞 60%를 기준, 뒤 40%를 현재 데이터로 사용합니다. 기준 데이터 내부에서 다시 학습/검증을 분리하므로 학습 데이터 성능을 기준값으로 사용하지 않습니다.

### CSV 두 개

- 기준 CSV: 타깃을 포함해야 임시 모델을 학습할 수 있습니다.
- 현재 CSV: 타깃이 있으면 성능까지, 없으면 데이터 품질과 드리프트까지만 확정합니다.
- 현재 CSV에 운영 모델의 예측 결과 열이 있으면 직접 선택할 수 있습니다.
- 이진 분류 모델은 예측 확률 열도 선택해 Calibration과 Threshold를 검사할 수 있습니다.

## 건강점수

기본 가중치는 데이터 품질 15%, 안정성 25%, 성능 30%, 신뢰성 10%, 취약 구간 20%입니다. 계산할 수 없는 항목은 0점 처리하지 않고 나머지 항목의 가중치를 다시 정규화합니다.

이 점수는 인증이 아니라 **점검 우선순위를 정하기 위한 내부 평가 지표**입니다. 앱의 한눈에 보기 탭에서 점수와 감점 근거를 모두 공개합니다.

## 실제 Agent 연결

로컬에서는 `.streamlit/secrets.toml.example`을 `secrets.toml`로 복사한 뒤 값을 입력할 수 있습니다.

```toml
OPENAI_API_KEY = "..."
OPENAI_MODEL = "gpt-4.1-mini"
OPENAI_BASE_URL = ""
ENABLE_AI_AGENT = false
```

`secrets.toml`은 절대 GitHub에 올리지 마세요. 키가 없을 때는 앱이 규칙 기반 모드임을 명확히 표시합니다.

AI Agent를 실행하면 원본 CSV 전체가 아니라 코드로 계산한 진단 지표, 컬럼명, 그룹값이 설정된 OpenAI 호환 API로 전송됩니다. 민감한 컬럼명이나 그룹값은 업로드 전에 익명화하고, 조직의 데이터 처리 정책과 API 제공자의 보존 정책을 확인하세요.

AI 호출은 기본적으로 비활성화됩니다. 비용과 데이터 전송 정책을 확인한 뒤 `ENABLE_AI_AGENT = true`를 명시해야만 AI Agent 버튼이 활성화됩니다.

공개 배포에 서버 API 키와 AI Agent를 활성화하면 모든 방문자가 호출 비용을 발생시킬 수 있습니다. 비용 한도, 사용량 알림, 접근 제어를 설정할 수 없는 공개 데모에서는 API 키를 연결하지 않고 규칙 기반 모드를 사용하는 것을 권장합니다.

## 테스트

```bash
python -m pytest -q
```

테스트는 데모 생성, 드리프트 주입 탐지, 건강점수 범위, 단일 클래스 방어, Streamlit 시작 화면을 확인합니다.

## Streamlit Community Cloud 배포

1. GitHub 저장소에 프로젝트 파일을 push합니다.
2. `.env`와 `.streamlit/secrets.toml`이 제외되었는지 `git status`로 확인합니다.
3. Streamlit Cloud에서 저장소와 `app.py`를 선택합니다.
4. 실제 Agent가 필요하면 Cloud의 Advanced settings → Secrets에 키를 입력합니다.
5. 공개 배포에서는 API 비용 한도와 접근 제어를 확인합니다.
6. 배포 후 정상·위험 데모, 그래프, 앱 로그를 확인합니다.

## 기술 스택

Python 3.11, Streamlit, pandas, NumPy, SciPy, scikit-learn, Plotly, OpenAI SDK, pytest

## 한계

- 실제 정답이 없는 현재 데이터에서는 성능 저하를 확정하지 않습니다.
- Calibration과 Threshold 비교는 이진 분류 모델에만 적용합니다.
- 그룹 표본이 30건 미만이면 취약 구간으로 확정하지 않습니다.
- 자동 재학습, 자동 임곗값 변경, 자동 배포를 수행하지 않습니다.
- 기본 데모는 실제 운영 로그가 아닌 합성 시뮬레이션입니다.

## 다음 단계

- 실제 ELEC2 등 시간순 공개 데이터 검증
- 모델 파일과 실제 예측 로그 연결
- 기간별 모니터링 이력 저장
- 비용 기반 임곗값 최적화
- 사용자 피드백을 포함한 Agent 평가
