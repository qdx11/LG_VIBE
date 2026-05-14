# Mining Process Experiment Plan

## Objective

1차 분석 목표는 `% Silica Concentrate`의 품질 위험을 예측하고, 어떤 전처리/모델/샘플링 전략이 통계적으로 안정적인 성능 개선을 만드는지 검증하는 것이다.

기본 분류 타깃:

```text
high_silica_risk = 1 if % Silica Concentrate >= 4.0 else 0
```

`% Silica Concentrate`는 연속형 품질 지표이므로 회귀 문제도 가능하지만, 사용자가 요청한 stratified CV, 불균형 샘플링, 오분류 심화 분석은 품질 기준 초과 여부 분류 문제에서 더 직접적으로 평가한다.

## Leakage Control

이 데이터는 동일 `date`가 여러 행에서 반복된다. 같은 timestamp의 행이 train/validation에 동시에 들어가면 품질 라벨과 공정 상태가 거의 중복되어 성능이 과대평가될 수 있다.

따라서 기본 실험은 다음 정책을 따른다.

- timestamp 단위로 평균 집계
- timestamp 내부 공정 변동성 보존을 위해 입력 변수별 표준편차 feature 추가
- 5-fold stratified CV는 집계 후 샘플 기준으로 수행

향후 비교 실험:

- row-level `StratifiedGroupKFold` with `date` as group
- timestamp 집계 vs row-level group CV 성능 차이 비교

## Experiment Axes

### Models

- Dummy prior baseline
- Logistic Regression
- RandomForest
- ExtraTrees
- LightGBM
- XGBoost

### Preprocessing

- median imputation
- missing indicator
- no outlier clipping
- percentile clipping: p01/p99
- IQR clipping

현재 데이터는 결측치가 없지만, pipeline 안정성과 미래 데이터 대응을 위해 imputation 실험축을 유지한다.

### Imbalance Handling

- no sampling
- class weight / scale_pos_weight
- random undersampling
- random oversampling
- SMOTE

샘플링은 반드시 각 fold의 train split 안에서만 적용하고 validation fold에는 원본 분포를 유지한다.

## Metrics

주요 지표:

- PR-AUC: 불균형 분류에서 우선 지표
- ROC-AUC: ranking quality 보조 지표
- F1: threshold 기반 균형 지표
- Precision / Recall: 운영 목적에 따라 trade-off 확인
- Balanced accuracy
- Brier score / log loss: 확률 품질 확인

## Statistical Validation

각 실험은 동일 5-fold split에서 평가한다.

통계검증:

- fold별 PR-AUC 차이에 대한 paired t-test
- fold별 PR-AUC 차이에 대한 Wilcoxon signed-rank test
- 다중 비교 보정: Holm-Bonferroni
- 평균, 표준편차, 95% CI 함께 보고

주의:

- 5-fold만으로는 검정력이 낮으므로 p-value는 보조 근거로 사용한다.
- 최종 판단은 effect size, fold 안정성, 실무 지표 개선폭을 함께 본다.

## Dashboard Mapping

Streamlit 앱은 `output/mining_process` 산출물을 읽어 다음 탭으로 구성한다.

### Tab 1: 실험 결과

- 전체 실험 결과 테이블
- fold-level 성능 분포
- paired statistical tests
- 방법론별 효과 해석
- 향후 추가 실험 제안

### Tab 2: 입력 변수 분석

- feature importance
- 중요 변수/비중요 변수의 class별 통계 차이
- top feature correlation
- 심화 EDA 힌트

### Tab 3: 오분류 데이터 심화 분석

- 최고 모델의 OOF 예측 기반 confusion matrix
- 임계치 조정
- 임계치 근접 오분류 샘플
- 큰 마진 오분류 샘플
- near-error vs large-margin-error 통계 비교
- 라벨 오류 가능성 및 모델 고도화 아이디어
# Experiment Plan: Pump Failure Early Warning

## Objective

펌프 센서 시계열 데이터를 사용해 다음 4개 예측 문제를 비교 실험한다.

- `failure_within_1h`
- `failure_within_3h`
- `failure_within_6h`
- `failure_within_24h`

각 문제는 현재 시점의 센서 패턴을 보고 앞으로 해당 시간창 안에 `BROKEN` 이벤트가 발생할지 예측하는 이진 분류 문제다.

## Important Constraint

원본 `BROKEN` 이벤트는 7개뿐이다. 따라서 모델 성능을 해석할 때 단순 row-level 정확도보다 이벤트 단위 탐지 성공 여부와 시간 누수를 더 중요하게 본다.

파생 타깃별 양성 샘플 수는 다음과 같다.

```text
failure_within_1h     420 positive rows, 0.19%
failure_within_3h   1,260 positive rows, 0.57%
failure_within_6h   2,520 positive rows, 1.14%
failure_within_24h 10,080 positive rows, 4.58%
```

## Feasibility Strategy

고장 이벤트가 7개뿐이므로 순수 지도학습 모델만으로 고장 전조를 안정적으로 일반화하기는 어렵다. 이 프로젝트의 현실적인 전략은 다음과 같은 하이브리드 접근이다.

1. `normal_pattern_learning`
   - 정상 구간의 센서 패턴을 학습한다.
   - 정상 패턴에서 크게 벗어나는 시점을 위험 신호로 본다.
   - Isolation Forest, rolling z-score, rolling MAD, autoencoder 후보를 검토한다.

2. `supervised_early_warning`
   - `failure_within_1h`, `3h`, `6h`, `24h` 라벨을 사용해 LightGBM/XGBoost 같은 지도학습 모델을 학습한다.
   - 고장 이벤트가 적으므로 복잡한 모델보다 누수 방지, 검증 설계, threshold tuning이 더 중요하다.

3. `hybrid_model`
   - 정상 패턴 기반 anomaly score를 지도학습 모델의 feature로 넣는다.
   - 예: sensor 원본값 + rolling feature + anomaly score + missing flag
   - 최종적으로 고장 전조 라벨을 맞히되, 정상 패턴 이탈 정보를 함께 활용한다.

이 프로젝트에서 기대할 수 있는 것은 모든 고장을 완벽히 맞히는 모델이 아니라, 제한된 고장 사례 안에서 어떤 센서 패턴이 고장 전조와 연결되는지 검증하고 조기경보 체계를 설계하는 것이다.

성공 기준은 다음처럼 정의한다.

- 단순 baseline보다 macro F1과 macro AUROC가 개선되는가
- 마지막 2개 test 고장 이벤트를 사전에 탐지하는가
- 알람이 고장 몇 분/몇 시간 전에 발생하는가
- false positive가 운영 가능한 수준인가
- 모델이 특정 이벤트만 암기한 것이 아니라 event-aware CV에서 일관된 성능을 보이는가

## Data Loading Plan

1. `pandas`로 `data/raw/pump_sensor/sensor.csv`를 읽는다.
2. 기본 정보를 출력한다.
   - `shape`
   - `columns`
   - `dtypes`
   - timestamp 범위
   - `machine_status` 분포
3. `timestamp`를 datetime으로 변환한다.
4. `Unnamed: 0`은 CSV 저장 과정의 인덱스로 보고 제거한다.

## Target Definition

원본 타깃 컬럼:

- `machine_status`

모델링 타깃:

- `failure_within_1h`
- `failure_within_3h`
- `failure_within_6h`
- `failure_within_24h`

생성 방식:

1. `machine_status == "BROKEN"`인 timestamp를 고장 이벤트로 정의한다.
2. 각 고장 이벤트 이전 N시간 구간의 `NORMAL` row를 positive로 지정한다.
3. 그 외 row는 negative로 지정한다.
4. `RECOVERING` 구간은 기본 실험에서 학습 대상에서 제외한다.

`RECOVERING` 제외 이유:

- 이미 고장이 발생한 뒤 복구 중인 상태이므로, 고장 전조 예측 문제의 목적과 다르다.
- 복구 상태를 negative로 넣으면 모델이 복구 패턴을 정상 패턴으로 오해할 수 있다.
- 별도 실험에서는 `RECOVERING` 포함/제외를 비교한다.

## Feature Definition

제외 컬럼:

- `Unnamed: 0`
- `timestamp`
- `machine_status`
- 파생 타깃 이외의 label-related columns

기본 피처:

- `sensor_00` ~ `sensor_51`

추가 피처 후보:

- lag features: 5분, 15분, 30분, 60분 전 센서값
- rolling mean: 15분, 30분, 60분, 180분
- rolling std: 15분, 30분, 60분, 180분
- rolling min/max
- 현재값 - rolling mean
- 현재값 - lag value
- 센서별 결측 여부 indicator

## Train/Test Split Strategy

### Recommended Official Split

이 데이터는 시계열이고 고장 이벤트 수가 적으므로, 공식 평가는 랜덤 `train_test_split`이 아니라 이벤트 기반 시간 분할을 사용한다.

공식 분할:

- Train: 2018-04-01 00:00:00 ~ 2018-07-04 17:50:00
- Test: 2018-07-04 17:51:00 ~ 2018-08-31 23:59:00

이 분할은 앞쪽 5개 고장 이벤트를 train/CV에 사용하고, 마지막 2개 고장 이벤트를 최종 test에 남긴다. 5번째 고장 이벤트의 복구 구간까지 train에 포함해 하나의 고장 에피소드가 train/test에 걸쳐 쪼개지지 않도록 한다.

```text
Train rows: 136,431
Test rows: 83,889
Train broken events: 5
Test broken events: 2
```

파생 타깃별 train/test 양성 샘플 수:

```text
1h  train positives=300,  test positives=120
3h  train positives=900,  test positives=360
6h  train positives=1800, test positives=720
24h train positives=7200, test positives=2880
```

생성된 CSV:

```text
data/processed/pump_sensor/sensor_train.csv
data/processed/pump_sensor/sensor_test.csv
data/processed/pump_sensor/split_metadata.json
```

위 CSV에는 원본 컬럼과 함께 `failure_within_1h`, `failure_within_3h`, `failure_within_6h`, `failure_within_24h` 파생 타깃이 포함된다.

### Why Not Random Split as Official Evaluation

사용자가 제시한 `train_test_split(test_size=0.2, random_state=42, stratify=y)`는 일반 분류 문제에서는 적절하다. 하지만 이 데이터에서는 인접한 시점의 센서값이 매우 비슷하므로 랜덤 분할을 하면 거의 같은 시간대의 row가 train과 test에 동시에 들어갈 수 있다.

그 결과 모델이 미래 고장을 예측한 것이 아니라 근처 시간 패턴을 암기한 것처럼 보일 수 있다. 따라서 랜덤 split은 baseline sanity check로만 사용하고, 공식 성능은 시간/이벤트 기반 holdout으로 평가한다.

## 5-Fold CV Strategy

### Primary CV: Event-Aware 5-Fold

Train 구간에는 고장 이벤트가 5개 있으므로, 5-fold CV는 각 fold가 고장 이벤트 1개를 validation으로 갖도록 설계한다.

각 fold:

- validation: 특정 고장 이벤트 1개의 positive window와 주변 negative window
- train: 나머지 4개 고장 이벤트 기반 데이터
- purge gap: validation 고장 이벤트 전후 일정 시간은 train에서 제거해 시간 누수를 줄인다.

이 방식은 사용자가 요청한 5-fold 구조를 유지하면서도, 고장 이벤트 단위 일반화 성능을 확인할 수 있다.

### Secondary CV: StratifiedKFold

비교 목적으로 `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`도 수행한다.

단, 이 결과는 시간 누수 가능성이 있으므로 최종 결론에서는 보조 지표로만 사용한다.

### Out-of-Fold Prediction

OOF는 test data가 아니라 train data 내부의 fold별 validation 예측값이다.

사용 목적:

- threshold tuning
- calibration
- 모델 간 비교
- stacking 후보 검토
- 오분류 케이스 분석

최종 test는 모든 실험 설계와 threshold 결정이 끝난 뒤 한 번만 평가한다.

## Preprocessing Experiment Grid

### Missing Value Handling

실험 후보:

1. `drop_high_missing`
   - `sensor_15` 제거
   - `sensor_50` 제거
   - 나머지는 median imputation

2. `time_interpolation`
   - `sensor_15` 제거
   - `sensor_50` 유지
   - 시간 기반 interpolation 후 forward/backward fill

3. `missing_indicator`
   - `sensor_15` 제거
   - 결측률이 있는 센서는 결측 여부 indicator 추가
   - LightGBM/XGBoost의 missing value handling 활용

4. `robust_fill`
   - sensor별 rolling median으로 보간
   - 남은 결측은 train median으로 보간

### Outlier Handling

이상치는 자동 제거하지 않고 비교 실험한다.

실험 후보:

1. `none`
   - 원본 유지

2. `iqr_clip`
   - train set 기준 Q1, Q3, IQR 계산
   - Q1 - 3*IQR, Q3 + 3*IQR로 clipping

3. `percentile_clip`
   - train set 기준 p1, p99 clipping

4. `robust_scaling`
   - clipping 없이 RobustScaler 적용

주의:

- 이상치는 실제 고장 전조일 수 있으므로 무조건 제거하지 않는다.
- 모든 clipping 기준은 train fold에서만 학습하고 validation/test에 적용한다.

### Anomaly Detection Preprocessing

이상 탐지는 필요하다. 다만 이 프로젝트에서는 이상치를 단순히 제거하는 전처리로 쓰면 안 된다. 센서 이상 패턴 자체가 고장 전조일 수 있기 때문이다.

따라서 이상 탐지는 다음 3가지 역할로 나누어 실험한다.

1. `anomaly_as_feature`
   - 이상 여부 또는 이상 점수를 모델 입력 피처로 추가한다.
   - 원본 센서값은 유지한다.
   - 기본 추천 방식이다.

2. `anomaly_for_cleaning`
   - 명백한 데이터 오류만 제거하거나 보정한다.
   - 예: 불가능한 값, 센서 전체 범위를 벗어난 값, 긴 구간의 비정상 고정값
   - 고장 직전의 극단값은 제거하지 않는다.

3. `anomaly_as_baseline_model`
   - Isolation Forest, One-Class SVM, rolling z-score 같은 비지도 이상 탐지 모델을 별도 baseline으로 평가한다.
   - 이 모델들은 고장 전조 라벨을 직접 학습하지 않으므로, ML 분류 모델과 비교 기준으로 사용한다.

실험 후보:

1. `rolling_zscore`
   - 센서별 rolling mean/std를 계산한다.
   - 현재값이 rolling 평균에서 몇 표준편차 떨어져 있는지 feature로 추가한다.
   - window 후보: 30분, 60분, 180분, 360분

2. `rolling_mad_score`
   - median absolute deviation 기반 robust anomaly score를 만든다.
   - 센서 분포가 비정규적이거나 outlier가 많은 경우 z-score보다 안정적이다.

3. `iqr_anomaly_flag`
   - train fold 기준 IQR 범위를 벗어나는지 여부를 센서별 flag로 추가한다.
   - clipping과 달리 원본값은 유지한다.

4. `isolation_forest_score`
   - `NORMAL` train row만 사용해 Isolation Forest를 학습한다.
   - validation/test에는 anomaly score만 산출해 피처로 추가한다.
   - fold 밖 정보를 사용하지 않도록 반드시 fold 내부에서만 fit한다.

5. `sensor_stuck_flag`
   - 특정 센서값이 오랜 시간 변하지 않는 구간을 flag로 만든다.
   - 설비 센서에서는 값 고착이 센서 이상 또는 설비 이상 신호일 수 있다.

6. `sudden_change_feature`
   - 직전 값 또는 rolling 평균 대비 급격한 변화량을 feature로 추가한다.
   - 고장 전 급격한 진동, 압력, 유량 변화가 있는지 포착하기 위한 목적이다.

중요한 누수 방지 원칙:

- anomaly threshold는 train fold에서만 계산한다.
- rolling feature는 반드시 과거 데이터만 사용한다.
- 전체 데이터의 평균/표준편차/분위수를 먼저 계산한 뒤 train/test에 적용하지 않는다.
- test 구간의 이상 패턴을 보고 threshold를 조정하지 않는다.

최종 비교 축:

```text
raw_sensor_only
raw_sensor + rolling_features
raw_sensor + anomaly_flags
raw_sensor + anomaly_scores
raw_sensor + rolling_features + anomaly_features
```

## Imbalanced Data Experiment Grid

불균형이 매우 크므로 샘플링과 모델 내 class weight를 모두 비교한다.

후보:

1. `no_sampling`
   - 샘플링 없음
   - 모델의 class weight 또는 scale_pos_weight만 사용

2. `class_weight`
   - Logistic Regression, RandomForest: `class_weight="balanced"`
   - LightGBM: `is_unbalance=True` 또는 `scale_pos_weight`
   - XGBoost: `scale_pos_weight = negative / positive`

3. `random_under_sampling`
   - positive:negative 비율을 1:10, 1:20, 1:50으로 비교

4. `random_over_sampling`
   - positive window를 단순 복제

5. `smote_tabular`
   - 센서 피처 기반 SMOTE
   - 단, 시계열 구조를 깨뜨릴 수 있으므로 보조 실험으로만 사용

6. `event_balanced_sampling`
   - 각 고장 이벤트 주변 positive window는 유지
   - negative는 이벤트와 먼 정상 구간에서 샘플링

## Model Experiment Grid

### Baselines

1. Dummy classifier
   - 항상 negative 예측
   - 모든 모델이 넘어야 할 최저 기준

2. Logistic Regression
   - 해석 가능한 baseline
   - scaling 필요

3. RandomForest
   - 비선형 baseline
   - class weight 비교

### Main Models

#### LightGBM

주요 탐색 파라미터:

- `n_estimators`: 300, 700, 1200
- `learning_rate`: 0.01, 0.03, 0.05
- `num_leaves`: 15, 31, 63
- `max_depth`: -1, 5, 8
- `min_child_samples`: 20, 50, 100
- `subsample`: 0.7, 0.9, 1.0
- `colsample_bytree`: 0.7, 0.9, 1.0
- `reg_alpha`: 0, 0.1, 1.0
- `reg_lambda`: 0, 0.1, 1.0
- `scale_pos_weight`: negative / positive

#### XGBoost

주요 탐색 파라미터:

- `n_estimators`: 300, 700, 1200
- `learning_rate`: 0.01, 0.03, 0.05
- `max_depth`: 3, 5, 7
- `min_child_weight`: 1, 5, 10
- `subsample`: 0.7, 0.9, 1.0
- `colsample_bytree`: 0.7, 0.9, 1.0
- `gamma`: 0, 1, 5
- `reg_alpha`: 0, 0.1, 1.0
- `reg_lambda`: 1, 5, 10
- `scale_pos_weight`: negative / positive

#### Additional Candidates

- CatBoost
- HistGradientBoostingClassifier
- BalancedRandomForestClassifier
- EasyEnsembleClassifier

추가 모델은 baseline과 LightGBM/XGBoost 결과를 본 뒤 확장한다.

## Threshold Tuning

기본 threshold 0.5는 불균형 문제에 적합하지 않을 가능성이 높다.

OOF prediction에서 다음 기준으로 threshold를 선택한다.

- macro F1-score 최대
- micro F1-score 최대
- Recall 우선, Precision 최소 기준 만족
- Precision-Recall curve 기반 operating point
- 이벤트 단위 사전 탐지 성공률 최대

예시 운영 기준:

- 고장 전조를 놓치는 비용이 크므로 Recall을 우선한다.
- 다만 False Positive가 너무 많으면 현장 알람 피로가 커지므로 Precision 하한을 둔다.

## Evaluation Metrics

### Main Metrics

이 프로젝트의 메인 평가지표는 다음 4개로 둔다.

- micro AUROC
- macro AUROC
- micro F1
- macro F1

`failure_within_1h`, `failure_within_3h`, `failure_within_6h`, `failure_within_24h`는 서로 다른 난이도의 조기경보 문제다. 따라서 각 타깃을 따로 평가하면서도, 전체 모델 실험을 비교할 때는 4개 타깃을 함께 요약하는 micro/macro 지표를 사용한다.

### Why Micro And Macro Both Matter

`micro` 평균은 모든 타깃의 예측 결과를 한꺼번에 합쳐 계산한다. 샘플 수가 많은 타깃, 특히 `failure_within_24h`처럼 positive가 상대적으로 많은 문제의 영향이 더 커진다. 전체 운영 관점에서 모델이 얼마나 안정적으로 맞히는지 보기 좋다.

`macro` 평균은 각 타깃별 점수를 먼저 계산한 뒤 단순 평균한다. `failure_within_1h`처럼 positive가 매우 적고 어려운 문제도 `failure_within_24h`와 같은 비중으로 반영된다. 불균형 데이터에서 쉬운 타깃만 잘 맞히는 모델을 걸러내는 데 중요하다.

해석 기준:

- micro가 높고 macro가 낮으면 쉬운/큰 타깃 위주로 성능이 좋은 모델일 수 있다.
- macro가 높으면 어려운 시간창에서도 상대적으로 균형 잡힌 성능을 낸다.
- 최종 모델은 macro F1과 macro AUROC를 우선하되, 운영 안정성을 위해 micro 지표도 함께 본다.

### AUROC

AUROC는 threshold를 정하기 전, 모델이 positive row를 negative row보다 더 높은 위험도로 정렬할 수 있는지 보는 지표다.

- `micro AUROC`: 4개 타깃의 예측을 모두 합쳐 전체 ranking 성능을 평가한다.
- `macro AUROC`: 4개 타깃별 AUROC를 구한 뒤 평균해 시간창별 균형을 평가한다.

주의:

- 불균형이 매우 심하면 AUROC가 좋아 보여도 실제 positive 탐지 성능이 부족할 수 있다.
- 따라서 AUROC는 F1, Precision-Recall, 이벤트 단위 탐지 지표와 함께 해석한다.

### F1

F1은 Precision과 Recall의 조화평균이다. 고장 전조 탐지에서는 고장을 놓치지 않는 Recall과 과도한 오탐을 줄이는 Precision 사이 균형이 중요하므로 핵심 지표로 둔다.

- `micro F1`: 전체 row/target 조합 기준으로 TP, FP, FN을 합쳐 계산한다.
- `macro F1`: 각 타깃별 F1을 계산한 뒤 평균한다.

최종 비교에서는 `macro F1`을 가장 중요한 단일 지표로 보고, `micro F1`, `macro AUROC`, `micro AUROC`를 함께 확인한다.

### Secondary Row-Level Metrics

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score
- Balanced accuracy
- Confusion matrix

Event-level metrics:

- 고장 이벤트 7개 중 몇 개를 사전에 탐지했는가
- 고장 몇 분/시간 전에 처음 알람이 발생했는가
- 이벤트별 false alarm 수
- 고장 이벤트 사이 정상 기간의 false positive rate

Calibration metrics:

- Brier score
- calibration curve

## Statistical Validation

모델 비교는 단일 점수 차이만 보지 않는다.

### Cross-Validation Score Comparison

5-fold CV 결과에 대해 모델별 fold score를 기록한다.

비교 방법:

- paired t-test
- Wilcoxon signed-rank test
- bootstrap confidence interval

주의:

- fold 수가 5개로 작으므로 p-value만으로 결론 내리지 않는다.
- 평균 차이, 신뢰구간, 이벤트 단위 성능을 함께 본다.

### Bootstrap on OOF Predictions

OOF prediction을 기준으로 bootstrap resampling을 수행한다.

비교 대상:

- PR-AUC 차이
- F1 차이
- Recall 차이
- Precision 차이

출력:

- 평균 차이
- 95% confidence interval
- p-value 또는 유사 확률

### McNemar Test

동일 validation/test row에 대한 두 모델의 정오분류 차이를 비교할 때 사용한다.

주의:

- row 간 독립성이 약하므로 최종 결론에서는 보조 지표로 사용한다.
- 이벤트 단위 분석과 함께 해석한다.

### Multiple Comparisons

많은 모델과 전처리 조합을 비교하므로 다중비교 문제가 발생한다.

대응:

- 탐색 단계와 확증 단계를 분리한다.
- 최종 후보 2~3개만 test에서 비교한다.
- 필요 시 Bonferroni 또는 Benjamini-Hochberg 보정을 적용한다.

## Experiment Stages

### Stage 0: Data Audit

- 데이터 로드
- schema 확인
- 시간 간격 검증
- 결측치/이상치/라벨 분포 확인

### Stage 1: Target Audit

- 1h, 3h, 6h, 24h 타깃 생성
- 각 타깃별 positive 수 확인
- 고장 이벤트별 positive window 확인
- `RECOVERING` 처리 정책 비교

### Stage 2: Baseline

- Dummy classifier
- Logistic Regression
- RandomForest
- 단순 sensor 원본 피처만 사용
- median imputation만 적용

### Stage 3: Preprocessing Comparison

- 결측치 처리 방식 비교
- 이상치 처리 방식 비교
- scaling 여부 비교

### Stage 4: Feature Engineering

- lag features
- rolling statistics
- difference features
- missing indicators

### Stage 5: Imbalance Handling

- class weight
- scale_pos_weight
- under/over sampling
- event-balanced sampling

### Stage 6: Main Model Tuning

- LightGBM
- XGBoost
- randomized search
- top 후보에 대해 narrower grid search

### Stage 7: Statistical Comparison

- fold별 성능 비교
- bootstrap confidence interval
- McNemar test
- 이벤트 단위 탐지 비교

### Stage 8: Final Test Evaluation

- 최종 후보 1~3개만 untouched test set에서 평가
- threshold는 OOF에서 정한 값을 사용
- test set으로 hyperparameter를 다시 고르지 않는다.

## Expected Risks

### Risk 1: Too Few Failure Events

`BROKEN` 이벤트가 7개뿐이라 모델이 고장 전조를 일반화하기 어렵다.

대응:

- row-level metric보다 event-level metric을 함께 사용
- 24h 타깃부터 시작해 6h, 3h, 1h로 점점 어려운 문제로 이동
- 고장 이벤트별 leave-one-event-out 검증 수행

### Risk 2: Time Leakage

랜덤 split, rolling feature 생성, imputation 과정에서 미래 정보가 섞일 수 있다.

대응:

- split 이후 전처리 fit
- rolling feature는 과거 방향으로만 생성
- validation/test 정보를 사용해 imputer/scaler를 fit하지 않음

### Risk 3: False Positive Explosion

불균형 데이터에서 recall만 높이면 정상 구간 알람이 너무 많아질 수 있다.

대응:

- Precision 하한 설정
- 일 단위 false alarm 수 측정
- 이벤트 단위 조기 탐지와 알람 피로도를 함께 평가

## Initial Implementation Files

앞으로 작성할 코드는 `code`에 둔다.

예상 파일:

```text
code/
  01_data_audit.py
  02_make_targets.py
  03_split_data.py
  04_feature_engineering.py
  05_run_baseline.py
  06_run_experiments.py
  07_statistical_tests.py
  app.py
```
