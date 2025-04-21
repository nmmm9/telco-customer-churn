# :telephone: 통신사 고객 이탈 예측 프로젝트

## 📜 프로젝트 개요
## 📅 개발 기간  
### 2025.04.17 ~ 2025.04.18
## 프로젝트 필요성
### 📌 고객 유지 비용 보다 큰 신규 고객 유치 비용
-  신규 고객 확보 비용은 광고, 프로모션, 제휴 마케팅 등 많은 비용이 드는 반면, 기존 고객을 유지하는 비용은 훨씬 낮음
-  이탈 가능성이 높은 고객을 사전에 파악하고 선제 대응하는 것이 훨씬 효율적임

### 📌 맞춤형 리텐션 마케팅 가능성 확보
-  이탈 가능성이 높은 고객군을 타겟팅하여 개인화된 리텐션 마케팅을 가능하게 함
-  고객 만족도 상승과 브랜드 충성도 증가로 이어짐

### 📌 고객 생애가치(LTV, Lifetime Value) 극대화
-  장기 고객일수록 ARPU(가입자 평균 매출)도 높고, 부가서비스 사용률도 높음
-  고객 이탈을 줄이면 고객 생애 가치가 증가하고, 이는 곧 장기 수익 증가로 이어짐


### 📌 정체된 시장에서의 경쟁력 확보
-  이미 포화 상태인 통신 시장에서 시장 점유율을 지키기 위해 기존 고객의 이탈을 줄이는 것이 가장 적절함
<br>

# 🎯프로젝트 목표
### 통신사 이탈 고위험성 고객 사전 식별, 높은 고객 유지율, 시장 경쟁력 극대화를 위한 고객 이탈 예측 모델 개발 및 시각화

<br>

## 🛠️기술 스택
|    분야    |사용기술|
|:------:|:------:|
|    언어   |![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white)|
|데이터분석   |![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=Pandas&logoColor=white)  ![Numpy](https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)|
|시각화   |![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-43B6C7?style=for-the-badge&logo=seaborn&logoColor=white) ![SHAP](https://img.shields.io/badge/SHAP-5A20CB?style=for-the-badge&logoColor=white)
|    모델 설계    |![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) ![Random Forest](https://img.shields.io/badge/Random%20Forest-228B22?style=for-the-badge&logoColor=white) ![Logistic Regression](https://img.shields.io/badge/Logistic%20Regression-1E90FF?style=for-the-badge&logoColor=white) ![SVM](https://img.shields.io/badge/SVM-800080?style=for-the-badge&logoColor=white) ![LightGBM](https://img.shields.io/badge/LightGBM-9ACD32?style=for-the-badge&logoColor=white) ![Gradient Boosting](https://img.shields.io/badge/Gradient%20Boosting-F5B041?style=for-the-badge&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)|
|화면구현    |![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)|

<br> 

------------------------------------------------------------------------------------------------------------
## 데이터셋 소개
IBM이 제공하는 통신사 고객 이탈 데이터셋을 사용  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- 데이터 수 : 7043명
- 사용 변수

|변수명 (column name)|변수 설명 (description)|변수 유형 (data type)|
|---|---|---|
| `customerID` | 고객 고유 ID |📝 문자열 (String)|
| `gender` | 고객 성별 (남성/여성) |📝 문자열 (String)|
| `SeniorCitizen` | 고령자 여부 (1 = 예, 0 = 아니오) |🔢 정수형 (Integer)|
| `Partner` | 배우자 유무 (예/아니오) |📝 문자열 (String)|
| `Dependents` | 부양 가족 유무 (예/아니오) |📝 문자열 (String)|
| `tenure` | 회사 가입 개월 수 |🔢 정수형 (Integer)|
| `PhoneService` | 전화 서비스 이용 여부 (예/아니오) |📝 문자열 (String)|
| `MultipleLines` | 다중 회선 이용 여부 (예/아니오/전화 서비스 없음) |📝 문자열 (String)|
| `InternetService` | 인터넷 서비스 종류 (DSL/광섬유/인터넷 서비스 없음) |📝 문자열 (String)|
| `OnlineSecurity` | 온라인 보안 서비스 이용 여부 (예/아니오) |📝 문자열 (String)|
| `OnlineBackup` | 온라인 백업 서비스 이용 여부 (예/아니오) |📝 문자열 (String)|
| `DeviceProtection` | 기기 보호 서비스 이용 여부 (예/아니오) |📝 문자열 (String)|
| `TechSupport` | 기술 지원 서비스 이용 여부 (예/아니오) |📝 문자열 (String)|
| `StreamingTV` | 스트리밍 TV 이용 여부 (예/아니오) |📝 문자열 (String)|
| `StreamingMovies` | 영화 스트리밍 서비스 이용 여부 (예/아니오) |📝 문자열 (String)|
| `Contract` | 계약 유형 (월별 계약/1년 계약/2년 계약) |📝 문자열 (String)|
| `PaperlessBilling` | 종이 없는 청구서 여부 (예/아니오) |📝 문자열 (String)|
| `PaymentMethod` | 결제 방식 (전자 수표/우편 수표/계좌 이체/신용카드) |📝 문자열 (String)|
| `MonthlyCharges` | 월별 서비스 요금 |🔢 실수형 (Float)|
| `TotalCharges` | 누적 총 지불 요금 |📝 문자열 (String)|
| `Churn` | 이탈여부 (예/아니오)| 📝 문자열 (String)|

------------------------------------------------------------------------------------------------------------    
## 📊EDA
### 고객 이탈 여부     

<img src="https://github.com/user-attachments/assets/d4f8a461-5fbf-45e6-b9f0-a8f5c3ac3ca3" width="850">   

- 고객이 이탈하지 않은 경우(No)는 73.5%, 고객이 이탈한 경우(Yes)는 26.5% 로 나타남
- 전체 고객 중 약 4명 중 1명 정도가 이탈한 것으로 분석

### 성별에 따른 이탈 여부     

<img src="https://github.com/user-attachments/assets/ef7ceb86-e563-489f-9de6-71cc7c19046a" width="850" height="450">

-  남성(Male)이 50.5%, 여성(Female)이 49.5%로 거의 비슷한 비율을 보임
-  성별 이탈률 그래프는 남성과 여성 모두 약 25%의 이탈률을 보이며, 성별에 따른 고객 이탈률의 차이가 거의 없음 의미

### 고령자 이탈 여부     

<img src="https://github.com/user-attachments/assets/1642eabd-39b8-48e2-96bc-6511f3bcc869" width="850" height="450">  

-  비노인 고객은 전체의 83.8%로 대부분을 차지하며, 노인은 16.2%를 차지함
-  노인 고객의 이탈률은 약 40%로 비노인(약 20%)보다 훨씬 높게 기록함

### 이용기간 별 이탈 여부   

<img src="https://github.com/user-attachments/assets/5bb420b7-e7d5-40e5-ad86-b3b7dc020961" width="850">   

-  서비스 이용 초기(0~5개월)에 고객 이탈이 두드러지게 높으며, 시간이 지날수록 점차 감소하는 경향을 보임
-  이탈 고객의 이용 기간 중앙값은 약 10개월로 나타나, 서비스 초기 단계에서 고객 이탈이 집중됨을 알 수 있음
-  반면, 비이탈 고객의 경우, 서비스 이용 기간 중앙값이 약 40개월로 장기적인 이용 경향을 보임
-  60개월 이상 장기 이용 고객층은 이탈률이 현저히 낮아 안정적 고객군으로 분류가 가능함


  

### 월별 요금 별 이탈 여부    

<img src="https://github.com/user-attachments/assets/be2f2c78-cf68-41ce-b2ec-fa634cb2a88b" width="850">   

- 월별 요금이 낮은 구간(약 20달러 이하)에서는 이탈하지 않은 고객 비율이 매우 높음
- 요금이 증가할수록 이탈 고객 비율도 점차 높아지는 경향을 보임
- 월별 요금이 70달러 이상인 구간에서는 이탈 고객 비율이 상대적으로 높아짐
- 월별 요금이 높을수록 고객 이탈 가능성이 커짐을 유추할 수 있음


### 결제 방법 별 이탈 여부    

<img src="https://github.com/user-attachments/assets/e800c8f3-ef5f-4759-af6a-65bcf2691614" width="850" height="450">   

- ‘Electronic check’ 방식은 약 45%에 달하는 가장 높은 이탈률을 보이고 있음
- ‘Bank transfer (automatic)’와 ‘Credit card (automatic)’ 방식의 이탈률은 각각 약 15% 수준을 보임
- ‘Mailed check’ 방식은 중간 수준의 이탈률을 보이고 있으며, 약 20% 내외로 추정됨
- 자동 결제 방식이 고객의 지속적인 이용을 유도하는 데 긍정적인 영향을 미치고 있음을 시사함

### 인터넷 서비스 별 이탈 여부    

<img src="https://github.com/user-attachments/assets/cf116353-4996-4752-bb88-51d800ffa906" width="850" height="450">   

- Fiber optic(광섬유) 서비스를 이용하는 고객의 이탈률이 약 40%로 가장 높음
- DSL 서비스의 이탈률은 약 20%로 중간 수준이며, 인터넷 서비스를 사용하지 않는 고객(No)의 이탈률은 가장 낮음

### 계약 형태 별 이탈 여부    

<img src="https://github.com/user-attachments/assets/56c1a163-5664-4bfb-9697-377e6ccdaab9" width="850">   

- 월 단위(Month-to-month) 계약 고객의 이탈률이 약 40%로 가장 높음음
- 1년(One year) 계약 고객의 이탈률은 약 10%로 상대적으로 낮음
- 2년(Two year) 계약 고객의 이탈률은 가장 낮음
- 장기 계약 고객일수록 이탈 가능성이 낮아짐

### 지불 방법 별 이탈 여부     

<img src="https://github.com/user-attachments/assets/a3df36e8-34d7-48b0-929d-ecf3d415cff7" width="850">   

- Electronic check(전자 수표) 방식을 사용하는 고객이 다른 방법에 비해 월 요금이 상대적으로 높고 이탈률도 높은 경향을 보임
- Mailed check(우편 수표) 방식을 사용하는 고객은 상대적으로 낮은 월 요금을 지불하며, 이탈 고객과 비이탈 고객 간 요금 차이가 크지 않음
- ank transfer(자동 은행 이체) 및 Credit card(자동 신용카드 결제)를 사용하는 고객들은 월별 요금 수준이 비슷하며, 이탈률 또한 낮음
- Electronic check 사용 고객의 높은 요금 수준이 이탈률 증가와 연관될 가능성이 크며, 자동 결제 방식이 이탈 관리에 더 유리함

### 부가 서비스 별 이탈 여부    

<img src="https://github.com/user-attachments/assets/c12a0cb0-7a2c-4a73-a49b-b2063ba12af6" width="850" height="350">  

- 온라인 보안 서비스에 가입하지 않은(No) 고객의 이탈률이 40% 이상으로 매우 높음
- 서비스에 가입한(Yes) 고객의 이탈률은 15% 이하로 상대적으로 낮음
- 온라인 보안 서비스 가입이 고객의 이탈 방지에 긍정적인 영향을 줌
------------------------------------------------------------------------------------------------------------
## 데이터 전처리 
- 결측치 및 중복값 제거
- 사용하지 않는 feature 제거 (`Customer_ID`)
- Churn의 컬럼 값 1 / 0으로 변환
- TotalCharges 컬럼에서 `NaN` 값 제거 및 데이터 타입 변환(float)

## 특성 스케일링 및 데이터 불균형 처리
- StandardScaler를 이용한 특성 스케일링
- SMOTE 적용
------------------------------------------------------------------------------------------------------------
## 사용 모델
- 전통 머신러닝 모델
  - Logistic Regression
  - Random Forest
  - LightGBM
  - SVM
  - XGBoost
  - Gradient Boosting
<br>

 - MoonyungStacking : StackingClassifier를 사용하여 앙상블 모델 구성
   - XGBClassifier
     - n_estimators=90
     - max_depth=5
     - learning_rate=0.08
     - subsample=0.9
     - colsample_bytree=0.7
     - eval_metric="logloss"
     - use_label_encoder=False
     - random_state=42  

   - RandomForestClassifier
      - n_estimators=100
      - max_depth=8
      - random_state=42
        
   - LogisticRegression
     - max_iter=1000

   - StackingClassifier(
     - estimators=[("rf", rf), ("xgb", xgb)]
     - final_estimator=lr
     - passthrough=True

## 📊 모델 평가 및 결과
- 모델별 Accuracy, Recall, Precision, f1-score 비교
  
![image](https://github.com/user-attachments/assets/9643fca8-7916-4b7a-a183-0c41450be2ad)

- ROC-Curve
  
![newplot (2)](https://github.com/user-attachments/assets/beae7185-77de-4c8a-9791-3e6d1077bd53)

- 모델 종합 비교
  
![newplot (1)](https://github.com/user-attachments/assets/9706f6ee-4ae7-46a6-987f-991368530622)


### 성능 결과 분석  
MoonyungStacking 모델이 정확도 0.914, F1 score 0.843로 가장 높은 성능을 보임  
MoonyungStacking를 제외한 머신러닝 모델은 성능이 비슷함  
SMOTE를 통한 소수 클래스 오버샘플링이 모델 성능을 더욱 향상시킴


### 특성 중요도(Feature Importance) 분석 (상위 5개)
![image](https://github.com/user-attachments/assets/a6a95442-0f7b-461e-9230-3f89706a8411)

Contract (계약 유형)  
tenure (이용 기간)  
MonthlyCharges (월 요금)  
OnlineSecurity (온라인 보안 서비스)  
TechSupport (기술지원)  

------------------------------------------------------------------------------------------------------------
## 결론
### 종합 결론
고객 이탈의 주요 원인을 다각도로 분석하고, 머신러닝 기반 예측 모델을 통해 이탈 고위험 고객을 효과적으로 식별함으로써, 고객 유지 전략 수립과 시장 경쟁력 확보에 기여할 수 있는 중요한 인사이트를 제공하였음
- 고객 특성 분석: 고령자, 고요금 고객, 초기 이용자 등에서 이탈률이 높아, 맞춤형 관리 전략이 필요함
- 계약 및 결제 방식 영향: 자동 결제와 장기 계약이 이탈률을 낮추는 데 효과적임을 확인함
- 부가서비스 효과: 온라인 보안 서비스와 같은 부가서비스 가입이 이탈 방지에 긍정적 영향을 미침
- 머신러닝 모델 성과: MoonyungStacking 모델이 높은 정확도(0.914)와 F1 점수(0.843)를 기록, 실무 적용 가능성이 큼
- 비즈니스 가치: 기존 고객 유지는 신규 고객 확보보다 비용 효율적이며, 고객 생애가치(LTV) 향상과 장기적 수익에 기여함

### 한계점
- 데이터 불균형 문제의 본질적인 한계  
SMOTE를 통해 소수 클래스(이탈 고객)를 보완했지만, 인위적으로 생성된 데이터는 실제 현장 데이터의 복잡한 특성과 다를 수 있음

- 시간적 요인 미반영
본 모델은 시계열 데이터 특성(예: 시간에 따라 변화하는 고객의 이용 패턴, 이탈 트렌드 등)을 반영하지 않았음   
이탈까지의 경과 시간이나 이벤트 순서를 고려하지 않았기 때문에, 고객의 행동 변화 추이 예측에는 제한이 있음   

- 도메인 특화 피처 부족
현재 사용된 피처는 기본적인 고객 정보, 서비스 가입 정보 등으로 구성되어 있으며, 심화된 고객 행동 데이터가 포함되지 않아
이탈 예측의 정밀도 향상에 한계가 있음

### 개선 방안
- 행동 및 시계열 데이터 반영
이용 패턴, 최근 변경 이력 등 시계열 기반 피처를 추가하여 이탈 징후를 조기 포착하는 데 효과성을 높임
- 도메인 특화 피처 확장
고객센터 문의, 서비스 불만, 장애 이력 등 피처 추가
------------------------------------------------------------------------------------------------------------
## 기대효과 및 활용방안
###  기대효과
#### 1. 고객 유지율 향상
- 이탈 가능성이 높은 고객을 조기 식별하여 선제적 대응(프로모션, 상담, 혜택 제공) 가능
- 장기 고객으로 전환 유도 → LTV(고객 생애가치) 증가
#### 2. 마케팅 효율 극대화
- 전체 고객이 아닌 타겟 고객군에 집중 마케팅 가능
- 비용 대비 효과 높은 맞춤형 리텐션 캠페인 설계 가능
#### 3. 매출 손실 최소화
- 고수익 고객의 이탈을 사전에 방지하여 예상 매출 감소 최소화
- 고객 이탈로 인한 광고·유치 비용 부담 감소
#### 4. 경쟁사 이탈 방어 및 시장 점유율 유지
- 이탈 위험 고객이 경쟁사로 넘어가는 것을 막아 시장 점유율 보호
- 포화된 시장에서 수익성 중심의 운영 전략 수립 가능

### 활용방안
#### 1. 고객 등급별 리스크 모니터링 시스템 구축
- 실시간으로 고객 이탈 위험 점수를 산출하여 고위험 고객 자동 태깅
#### 2. CRM 시스템과 연계한 자동 알림 및 대응 트리거 설정
- 예: 이탈 위험 고객 발생 시 자동으로 할인 쿠폰, 상담 연결 유도 등
#### 3. 부서별 전략 수립 자료로 활용
- 마케팅팀: 타겟 리텐션 캠페인
- 고객센터: 선제적 상담 스크립트
- 경영진: 고객 이탈 현황 보고 및 KPI 설정
#### 4. 프로모션 우선순위 결정 기준으로 활용
- 한정된 자원(예산, 상담인력 등)을 이탈 가능성이 높은 고객에 집중
------------------------------------------------------------------------------------------------------------
<br>



  
