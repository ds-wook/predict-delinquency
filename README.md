# PredictCreditCardDelinquency
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[신용카드 사용자 연체 예측 AI 경진대회](https://dacon.io/competitions/official/235713/overview/description)

## Feature Engineering
+ income_type: 기본 변수
+ edu_type: 기본 변수
+ family_type: 기본 변수
+ house_type: 기본 변수
+ occyp_type: 기본 변수
+ income_total: numeric한 변수를 category로 바꿈
+ begin_month: 중복 데이터를 구별할 수 있는 핵심 변수
+ DAYS_BIRTH_month: DAYS_BIRTH의 달
+ DAYS_BIRTH_week: DAYS_BIRTH의 주
+ Age: DAYS_BIRTH의 년도
+ DAYS_EMPLOYED_month: DAYS_EMPLOYED의 달
+ DAYS_EMPLOYED_week: DAYS_EMPLOYED의 주
+ EMPLOYED: DAYS_EMPLOYED의 년
+ before_EMPLOYED: DAYS_BIRTH와 DAYS_EMPLOYED의 차
+ before_EMPLOYED_month: 고용되기 전의 달
+ before_EMPLOYED_week: 고용 되기 전의 주
+ gender_car_reality: 성별, 차, 부동산 변수를 합침


## 핵심 Model
+ category feature의 전처리가 필수적으로 중요
+ **CatBoost**를 활용하여 category feature들을 지정하여 모델의 성능을 높힘
+ LightGBM, XGBoost와 비교 후 더 나은 성능을 보임


## Cross Validation 전략
+ Stratified K-Fold: 다중 분류문제에서 자주 쓰이는 기법 labeling의 sample을 잘 맞춰서 학습을 진행하도록 함
+ 10-fold로 진행하여 성능을 높힘


## Hyperparameter 전략
+ Bayesian TPE 방식으로 빠르게 하이퍼파라미터 튜닝
+ 최단시간에 최고효율이 나오게끔 함


## Benchmark
|model|OOF(10-fold)|Public LB|Private LB|
|:-----|:---------|:--------|:--------|
|LightGBM|0.68714|0.68674|-|
|XGBoost|0.68901|0.68900|-|
|RandomForest|0.69137|0.69296|-|
|TabNet|0.80392|0.77971|-|
|**CatBoost**|**0.67234**|**0.67288**|-|


## Ensemble Model
예측값 $ \hat y = \beta_0 * catboost + \beta_1 * lightgbm + \beta_2 * xgboost + \beta_3 * random forest + \beta_4 * TabNet$ 을 모델링하여 logloss를 최대한 줄임
각 $\beta_i(i=0, 1, 2...)$의 최적의 파라미터를 찾음
