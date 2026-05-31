# 🏫 소규모 학교 위기 예측 및 시뮬레이션

> 제8회 교육 공공데이터 AI활용 대회 출품작

<br>

## 서비스 바로가기
https://small-school-crisis-simulation.onrender.com

<br>

## 프로젝트 개요

소규모 학교 위기는 단순한 학생 수 감소를 넘어 교육 여건 저하와 지역 소멸로 이어지는 복합 문제다. 기존 연구는 단일 지역·사례 중심의 사후 분석에 머물러 전국 단위 조기 예측과 대안 비교가 부재하다.

이에 본 프로젝트는 전국 초·중학교 공공데이터를 기반으로 XGBoost 앙상블 AI 위기 예측 모델을 구축하고, OSRM 실도로거리 기반 통폐합·통합운영·유지 3가지 시나리오를 자동 비교하는 데이터 기반 의사결정 지원 대시보드를 제안한다.

<br>


## 주요 기능

- 전국 고위험·중위험 학교 분포 지도
- XGBoost 앙상블 모델 기반 통폐합 위기확률 산출 (0~100점)
- LIME 분석으로 학교별 위험 원인 시각화
- K-Means 군집 분석으로 6개 유형 태그 제공
- OSRM 실제 도로거리 기반 통폐합·통합운영 후보 추천
- Prophet 시계열 모델 기반 10년 후 학생수 예측

<br>

## 사용자 흐름
<img width="886" height="744" alt="image" src="https://github.com/user-attachments/assets/59f27c9a-4406-409e-b67d-857fe0c873bc" />

<br>
<br>

## 파이프라인

1. 데이터 수집 및 전처리  
2. EDA / 피처 선정  
3. XGBoost 모델 학습 (언더샘플링 앙상블 N=20, Optuna 튜닝)  
4. LIME 분석 / K-Means 군집화  
5. 소규모 위험 학교 예측 및 분석
6. 시나리오 시뮬레이션 (통폐합 / 통합운영 / 분교유지)  
7. Dash 대시보드

<br>

## 모델 성능

### Baseline vs Final Model

| 항목 | Baseline | Final Model |
|------|----------|-------------|
| 피처 수 | 11개 | 9개 |
| 모델 구조 | 단일 XGBClassifier | 언더샘플링 앙상블 (N=20) |
| n_estimators | 300 | 200 + early_stopping=15 |
| max_depth | 6 | Optuna 탐색 (3~7) |
| learning_rate | 0.05 | Optuna 탐색 (0.01~0.1, log) |
| min_child_weight | 1 (기본값) | Optuna 탐색 (1~7) |
| subsample | 기본값 | Optuna 탐색 (0.6~1.0) |
| colsample_bytree | 기본값 | Optuna 탐색 (0.6~1.0) |
| reg_alpha | 기본값 | Optuna 탐색 (0.0~3.0) |
| reg_lambda | 기본값 | Optuna 탐색 (0.5~5.0) |
| scale_pos_weight | 불균형 비율 자동 계산 | Optuna 탐색 (0.8~3.0) |
| 클래스 불균형 처리 | scale_pos_weight | 언더샘플링 앙상블 |
| 샘플 가중치 | 없음 | 폐교 근접도 기반 차등 (1.0~1.5) |
| 하이퍼파라미터 탐색 | 없음 | Optuna 100 trials |

<br>

### 성능 평가 비교
> 동일한 Recall 수준에서 오탐(FP) 993개 → 60개, **93% 감소**

<br>

**Baseline**

| 기준 | threshold | accuracy | balanced_accuracy | precision | recall | f1 | roc_auc | pr_auc | mcc | fn_rate | tn | fp | fn | tp |
|------|-----------|----------|-------------------|-----------|--------|----|---------|--------|-----|---------|----|----|----|----|
| Best Threshold 0.0271 | 0.027124 | 0.844152 | 0.827008 | 0.048804 | 0.809524 | 0.092058 | 0.931534 | 0.645015 | 0.174550 | 0.190476 | 5398 | 994 | 12 | 51 |


<br>

**Final Model**

| 기준 | threshold | accuracy | balanced_accuracy | precision | recall | f1 | roc_auc | pr_auc | mcc | fn_rate | tn | fp | fn | tp |
|------|-----------|----------|-------------------|-----------|--------|----|---------|--------|-----|---------|----|----|----|----|
| Best Threshold 0.5901 | 0.590109 | 0.988846 | 0.900069 | 0.459459 | 0.809524 | 0.586207 | 0.985565 | 0.701188 | 0.605076 | 0.190476 | 6332 | 60 | 12 | 51 |



<br>
<br>

## 기술 스택

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-4B8BBE?style=for-the-badge&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Dash](https://img.shields.io/badge/Dash-008DE4?style=for-the-badge&logo=plotly&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

<br>
<br>

## 파일 구조

```
├── data/
│   ├── (결과)LIME_분석.csv
│   ├── (결과)고위험학교.csv
│   ├── (결과)고위험학교_군집분석.csv
│   ├── (결과)분교유지_시뮬.csv
│   ├── (결과)통폐합_시뮬.csv
│   ├── (결과)통합운영_시뮬.csv
│   ├── 2008_2025_초등_국공립_학급학생수교사수_통합.csv
│   └── 전국학교_위경도_정제.csv
├── app.py
├── Procfile
└── requirements.txt
```

<br>





