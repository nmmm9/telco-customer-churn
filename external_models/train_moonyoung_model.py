#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# 데이터 로드
print("데이터 로드 중...")
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 컬럼명 변경
print("데이터 전처리 중...")
df.columns = [
    "고객ID", "성별", "시니어여부", "배우자여부", "부양가족여부", "가입개월수", "전화서비스",
    "복수회선여부", "인터넷서비스종류", "온라인보안서비스", "온라인백업", "디바이스보호",
    "기술지원", "TV스트리밍", "영화스트리밍", "계약종류", "전자청구서여부", "결제방법",
    "월요금", "총요금", "이탈여부"
]

# 전처리
df["총요금"] = pd.to_numeric(df["총요금"], errors="coerce")
df.dropna(subset=["총요금"], inplace=True)
df["이탈여부"] = df["이탈여부"].map({"Yes": 1, "No": 0})
df.drop(columns=["고객ID"], inplace=True)

# 인코딩
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# 파생 변수
df["누적지불금액"] = df["가입개월수"] * df["월요금"]
df["장기계약여부"] = (df["계약종류"] != 0).astype(int)
df["인터넷없음"] = (df["인터넷서비스종류"] == 0).astype(int)
df["요금대"] = pd.cut(df["월요금"], bins=[0, 35, 70, 120], labels=[0, 1, 2])
df["요금대"] = le.fit_transform(df["요금대"].astype(str))
df["가입비율"] = df["가입개월수"] / (df["가입개월수"].max() + 1e-5)

# 특성/타겟 분리
X = df.drop("이탈여부", axis=1)
y = df["이탈여부"]

# 특성 이름 저장
os.makedirs('models', exist_ok=True)
with open('models/moonyoung_feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# SMOTE 적용 (클래스 불균형 해결)
print("SMOTE를 적용하여 클래스 불균형 해결 중...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 모델 정의
print("모델 학습 중...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(n_estimators=100, random_state=42)

base_models = [
    ('rf', rf),
    ('xgb', xgb)
]

stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=5
)

# 모델 학습
stacking.fit(X_train_smote, y_train_smote)

# 모델 저장
print("모델 저장 중...")
with open('models/moonyoung.pkl', 'wb') as f:
    pickle.dump(stacking, f)

with open('models/moonyoung_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 특성 중요도 시각화
print("특성 중요도 시각화 중...")
try:
    # 특성 중요도와 이름 추출
    importances = stacking.named_estimators_['xgb'].feature_importances_
    features = X.columns
    
    # 데이터프레임 생성 및 정렬
    importance_df = pd.DataFrame({
        'feature': list(features),
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # 한글 특성 이름을 영어로 변환하는 매핑 (필요한 경우)
    korean_to_english = {
        "성별": "Gender",
        "시니어여부": "Senior",
        "배우자여부": "Partner",
        "부양가족여부": "Dependents",
        "가입개월수": "Tenure",
        "전화서비스": "Phone",
        "복수회선여부": "MultipleLines",
        "인터넷서비스종류": "InternetService",
        "온라인보안서비스": "OnlineSecurity",
        "온라인백업": "OnlineBackup",
        "디바이스보호": "DeviceProtection",
        "기술지원": "TechSupport",
        "TV스트리밍": "StreamingTV",
        "영화스트리밍": "StreamingMovies",
        "계약종류": "Contract",
        "전자청구서여부": "PaperlessBilling",
        "결제방법": "PaymentMethod",
        "월요금": "MonthlyCharges",
        "총요금": "TotalCharges",
        "누적지불금액": "CumulativePayment",
        "장기계약여부": "LongTermContract",
        "인터넷없음": "NoInternet",
        "요금대": "PriceRange",
        "가입비율": "TenureRatio"
    }
    
    # 특성 이름을 영어로 변환
    importance_df['feature_en'] = importance_df['feature'].map(korean_to_english)
    
    # 영어 이름 기준으로 상위 15개 추출
    top_features = importance_df.head(15)
    
    # 시각화 
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(top_features)), top_features['importance'], align='center')
    plt.yticks(range(len(top_features)), top_features['feature_en'])
    plt.xlabel('Importance Score', fontsize=12)
    plt.title('XGBoost Feature Importance (Top 15)', fontsize=14)
    plt.grid(axis='x')
    
    # 막대 옆에 수치 출력
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', va='center')
    
    plt.tight_layout()
    
    # 그래프 저장
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig('visualizations/feature_importance.png')
    print("특성 중요도 시각화 저장 완료: visualizations/feature_importance.png")
    
    # 중요도 데이터 저장 (한글 및 영어 이름 모두 포함)
    importance_df.to_csv('visualizations/feature_importance.csv', index=False, encoding='utf-8-sig')
    print("특성 중요도 데이터 저장 완료: visualizations/feature_importance.csv")
    
except Exception as e:
    print(f"특성 중요도 시각화 중 오류 발생: {e}")

print("문영모델 학습 및 저장 완료!") 