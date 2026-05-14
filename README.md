# Mining Process Quality Prediction Dashboard

광물 선별 공정 데이터로 `% Silica Concentrate` 고위험 여부를 예측하고, 실험 결과와 오분류 분석을 Streamlit 대시보드로 정리한 프로젝트입니다.

## 제출 산출물

- 전처리 데이터: `data/processed/mining_process/train.csv`, `data/processed/mining_process/test.csv`
- 학습 모델: `models/best_model.pkl`
- Streamlit 대시보드: `code/app.py`

## 실행 방법

```powershell
pip install -r requirements.txt
streamlit run code/app.py
```

## 주요 내용

- LightGBM 기본 모델 학습 및 저장
- 5-fold stratified CV 기반 모델 비교
- LightGBM, XGBoost, ExtraTrees 등 모델 비교
- 불균형 데이터 샘플링 실험
- Feature importance 및 변수별 통계 비교
- 오분류 데이터 심화 분석
- 라벨 지연/센서 이상 가능성 기반 모델 고도화 실험
