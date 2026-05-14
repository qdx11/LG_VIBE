# Mining Process Dataset Notes

## Source

- Kaggle dataset: `edumagalhaes/quality-prediction-in-a-mining-process`
- Local file: `data/raw/mining_process/MiningProcess_Flotation_Plant_Database.csv`
- File size: about 184MB

## Basic Structure

```text
Rows: 737,453
Columns: 24
Date range: 2017-03-10 01:00:00 ~ 2017-09-09 23:00:00
```

## What This Data Represents

이 데이터는 광물 선별 공정의 운전 조건과 최종 품질 지표를 담고 있다.

주요 입력 변수:

- 원료 품질: `% Iron Feed`, `% Silica Feed`
- 약품 투입량: `Starch Flow`, `Amina Flow`
- 광석 펄프 조건: `Ore Pulp Flow`, `Ore Pulp pH`, `Ore Pulp Density`
- 부유선별 컬럼 공기 유량: `Flotation Column 01~07 Air Flow`
- 부유선별 컬럼 레벨: `Flotation Column 01~07 Level`

품질 결과 후보:

- `% Iron Concentrate`
- `% Silica Concentrate`

## Initial Target Recommendation

1차 추천 타깃:

```text
% Silica Concentrate
```

이유:

- 최종 제품 내 실리카 함량은 낮을수록 좋은 품질 지표로 해석하기 쉽다.
- 연속형 값이므로 회귀 문제로 시작하기 좋다.
- 이후 특정 기준 초과 여부를 분류 문제로 바꿀 수도 있다.

보조 타깃:

```text
% Iron Concentrate
```

## Initial Data Quality Findings

- 결측치 없음
- 숫자 컬럼이 콤마 소수점 문자열로 저장되어 있음
- 예: `55,2`, `16,98`
- 모델링 전에 모든 숫자 컬럼을 `,` -> `.` 변환 후 numeric으로 변환해야 함

## Timestamp Note

`date` 컬럼은 초 단위까지 표시되지만 같은 timestamp가 여러 번 반복된다.

```text
unique timestamps: 4,097
duplicate timestamp rows: 733,356
```

즉, 한 timestamp 안에 여러 센서 샘플이 들어있는 구조로 보인다. 시계열 분할은 가능하지만, timestamp 단위 집계 여부를 EDA에서 검토해야 한다.

## Target Summary

`% Silica Concentrate`:

```text
mean: 2.33
std:  1.13
min:  0.60
p25:  1.44
p50:  2.00
p75:  3.01
p95:  4.73
max:  5.53
```

`% Iron Concentrate`:

```text
mean: 65.05
std:   1.12
min:  62.05
p25:  64.37
p50:  65.21
p75:  65.86
p95:  66.61
max:  68.01
```

## Initial Modeling Direction

권장 시작 문제:

> 공정 운전 조건을 바탕으로 최종 `% Silica Concentrate`를 예측하는 회귀 모델

확장 문제:

> `% Silica Concentrate`가 기준치 이상으로 높아질 위험이 있는지 예측하는 분류 모델

이 데이터는 이전 펌프 고장 데이터보다 라벨이 훨씬 풍부하므로, ML 비교실험과 통계검증을 진행하기에 더 안정적이다.
