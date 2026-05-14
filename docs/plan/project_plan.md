# Mining Process Quality Prediction Project Plan

## Project Goal

광물 선별 공정의 운전 변수와 원료 특성을 활용해 최종 품질을 예측한다.

1차 목표:

> `% Silica Concentrate`를 예측하는 회귀 모델을 만든다.

확장 목표:

> 실리카 함량이 품질 기준을 초과할 위험이 있는지 예측하는 분류 모델을 만든다.

## Why This Dataset

이 데이터는 이전 펌프 고장 데이터보다 프로젝트 진행에 더 적합하다.

- 라벨이 충분히 많다.
- 결측치가 없다.
- 제조/공정/품질 예측 주제와 잘 맞는다.
- 회귀와 분류를 모두 설계할 수 있다.
- EDA, 전처리, 모델 비교, 성능 대시보드, 오차 분석까지 자연스럽게 이어진다.

## Dataset

- Kaggle ID: `edumagalhaes/quality-prediction-in-a-mining-process`
- Local path: `data/raw/mining_process/MiningProcess_Flotation_Plant_Database.csv`
- Rows: 737,453
- Columns: 24
- Date range: 2017-03-10 ~ 2017-09-09

## Key Preprocessing

- `date`를 datetime으로 변환
- 숫자 컬럼의 콤마 소수점 문자열을 float으로 변환
- timestamp 중복 구조 분석
- 시간 기준 train/test split 설계
- 필요 시 timestamp 단위 집계 실험

## Modeling Plan

Baseline:

- Linear Regression
- Ridge / Lasso
- RandomForestRegressor

Main models:

- LightGBMRegressor
- XGBoostRegressor

분류 확장:

- `% Silica Concentrate` 기준값 초과 여부
- Logistic Regression
- RandomForestClassifier
- LightGBMClassifier
- XGBoostClassifier

## Evaluation

회귀 지표:

- MAE
- RMSE
- R2
- MAPE
- target 구간별 오차

분류 확장 지표:

- micro/macro AUROC
- micro/macro F1
- Precision
- Recall

## Dashboard Direction

Streamlit 대시보드 주요 화면:

- 데이터 개요
- 공정 변수 분포
- 시간에 따른 품질 변화
- 예측값 vs 실제값
- 오차가 큰 구간 분석
- 주요 변수 중요도
- 품질 기준 초과 위험도
# Pump Sensor Predictive Maintenance Project Plan

## Project Goal

펌프 설비의 1분 단위 센서 데이터를 활용해 고장 전조를 탐지하는 머신러닝 프로젝트를 수행한다.

최종 목표는 단순히 현재 상태를 분류하는 것이 아니라, 고장 발생 전 일정 시간 안에 위험 신호를 포착해 설비 점검이나 정비 의사결정에 활용할 수 있는 모델과 대시보드를 만드는 것이다.

## Dataset

- Kaggle ID: `nphantawee/pump-sensor-data`
- Local path: `data/raw/pump_sensor/sensor.csv`
- Type: tabular time-series
- Rows: 220,320
- Columns: 55
- Timestamp range: 2018-04-01 ~ 2018-08-31
- Label column: `machine_status`

## Problem Definition

원본 라벨은 다음 3개 상태로 구성된다.

- `NORMAL`
- `RECOVERING`
- `BROKEN`

하지만 `BROKEN` 라벨이 7건뿐이므로, 원본 라벨을 그대로 3분류로 모델링하면 실무적 의미와 학습 안정성이 떨어질 수 있다.

따라서 프로젝트의 핵심 타깃은 다음과 같이 재정의한다.

> 현재 시점의 센서 패턴을 보고, 앞으로 N시간 이내 펌프 고장이 발생할 위험이 있는지 예측한다.

예시 타깃:

- `failure_within_1h`
- `failure_within_3h`
- `failure_within_6h`
- `failure_within_24h`

EDA 결과를 바탕으로 가장 적절한 예측 시간창을 선택한다.

## Main Phases

상세 비교실험과 통계검증 계획은 `docs/plan/experiment_plan.md`에 별도로 기록한다.

### 1. EDA

- 시간 범위와 샘플링 주기 확인
- 센서별 결측률 확인
- `sensor_15`처럼 전체 결측인 컬럼 제거 후보 식별
- 고장 시점 전후 센서 패턴 비교
- `NORMAL`, `RECOVERING`, `BROKEN` 구간의 시간적 흐름 분석
- 고장 이벤트별 특성 차이 확인

### 2. Preprocessing

- 불필요한 인덱스 컬럼 제거
- `timestamp` datetime 변환
- 전체 결측 컬럼 제거
- 결측치 보간 또는 rolling 기반 대체
- 시간 기준 train/validation/test split
- 고장 이후 `RECOVERING` 구간 처리 정책 정의

### 3. Feature Engineering

- lag feature
- rolling mean, std, min, max
- 센서 변화량
- 센서 간 차이 또는 비율
- 고장 이벤트까지 남은 시간 기반 타깃 생성

### 4. Modeling

Baseline:

- Logistic Regression
- RandomForest

Main models:

- LightGBM
- XGBoost

평가 지표:

- Precision
- Recall
- F1-score
- PR-AUC
- Confusion matrix
- 고장 이벤트 기준 사전 탐지 성공률

### 5. Misclassification Analysis

- False Negative: 고장 전조를 놓친 케이스
- False Positive: 정상인데 위험하다고 판단한 케이스
- 오분류 구간의 센서 트렌드 비교
- 고장 이벤트별 탐지 가능성과 한계 정리

### 6. Streamlit Dashboard

대시보드 주요 화면:

- 데이터 개요
- 센서 트렌드 시각화
- 고장 이벤트 타임라인
- 모델 성능 지표
- Confusion matrix
- 예측 위험도 추이
- 오분류 케이스 상세 분석

## Code Folder Plan

앞으로 작성하는 코드는 `code` 폴더에 저장한다.

예상 구조:

```text
code/
  01_eda.py
  02_make_features.py
  03_train_model.py
  04_evaluate_model.py
  app.py
```

필요에 따라 공통 함수는 `code/src` 아래로 분리한다.
