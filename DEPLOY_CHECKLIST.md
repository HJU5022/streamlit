# 배포 전 체크리스트

- [ ] `python -m pytest -q` 통과
- [ ] `python -m streamlit run app.py` 로컬 실행
- [ ] 정상/위험 원클릭 데모 확인
- [ ] CSV 한 개 업로드 확인
- [ ] `git status`에 `.env`와 `secrets.toml`이 없는지 확인
- [ ] requirements.txt가 저장소 루트에 있는지 확인
- [ ] Streamlit Cloud main file이 `app.py`인지 확인
- [ ] Cloud Secrets는 웹 설정에만 입력
- [ ] 배포 URL에서 그래프와 다운로드 확인
- [ ] Manage app 로그에 재시작 반복이 없는지 확인
