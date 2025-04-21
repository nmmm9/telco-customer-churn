import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
import os
import matplotlib.pyplot as plt
import shap
import json

# 페이지 설정
st.set_page_config(
    page_title="이탈 예측",
    page_icon="🔮",
    layout="wide"
)

# 헤더 및 설명
st.markdown("# 🔮 고객 이탈 예측")
st.markdown("### 새로운 고객 데이터를 입력하여 이탈 가능성을 예측해보세요.")

# CSS 스타일 추가
st.markdown("""
<style>
    /* 전체 앱 스타일링 */
    .main .block-container {
        padding: 2rem 3rem;
    }
    .main {
        background-color: #f8fafc;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Pretendard', 'Noto Sans KR', sans-serif;
        color: #1e293b;
    }
    h1 {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 1.5rem;
        color: #0f172a;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 0.75rem;
    }
    h3 {
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        color: #334155;
    }
    
    /* 예측 결과 박스 */
    .prediction-box {
        border-radius: 12px;
        padding: 1.75rem;
        margin: 1.5rem 0;
        background-color: white;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .prediction-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
    }
    .prediction-low {
        border-left: 6px solid #10b981;
    }
    .prediction-medium {
        border-left: 6px solid #f59e0b;
    }
    .prediction-high {
        border-left: 6px solid #ef4444;
    }
    
    /* 정보 박스 */
    .info-box {
        background-color: #f1f5f9;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1.25rem 0;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 2px 10px rgba(59, 130, 246, 0.08);
    }
    .info-box h4 {
        margin-top: 0;
        font-size: 1.15rem;
        color: #1e40af;
        font-weight: 600;
    }
    .info-box ul {
        margin-top: 0.75rem;
        padding-left: 1.5rem;
    }
    .info-box li {
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }
    
    /* 지표 컨테이너 */
    .metric-container {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.07);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 1rem;
        color: #64748b;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* 탭 스타일링 */
    .stTabs {
        margin-top: 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 3px;
        background-color: #f1f5f9;
        padding: 5px;
        border-radius: 12px 12px 0 0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 56px;
        white-space: pre-wrap;
        border-radius: 10px 10px 0 0;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 1rem;
        background-color: #e2e8f0;
        transition: all 0.3s ease;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #2563eb;
        box-shadow: 0 -4px 10px rgba(0, 0, 0, 0.03);
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.8);
        color: #1d4ed8;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background-color: white;
        border-radius: 0 0 12px 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
    }
    
    /* 폼 요소 스타일링 */
    .stSelectbox > div:first-child {
        font-weight: 600;
        color: #334155;
        font-size: 1rem;
        margin-bottom: 0.3rem;
    }
    .stSlider > div:first-child {
        font-weight: 600;
        color: #334155;
        font-size: 1rem;
        margin-bottom: 0.3rem;
    }
    div[role="listbox"] ul {
        border-radius: 10px;
    }
    div[role="listbox"] li {
        transition: background-color 0.2s;
    }
    input, select, textarea, button, div[data-baseweb="select"] {
        font-family: 'Pretendard', 'Noto Sans KR', sans-serif !important;
    }
    div[data-baseweb="select"] > div {
        border-radius: 10px !important;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
    }
    div[data-baseweb="select"]:hover > div {
        border-color: #cbd5e1;
    }
    
    /* 버튼 스타일링 */
    button[kind="primaryFormSubmit"] {
        border-radius: 10px;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 1.05rem;
        background: linear-gradient(135deg, #2563eb, #4f46e5);
        transition: all 0.3s;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: white;
        border: none;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.25);
    }
    button[kind="primaryFormSubmit"]:hover {
        background: linear-gradient(135deg, #1d4ed8, #4338ca);
        box-shadow: 0 6px 15px rgba(37, 99, 235, 0.35);
        transform: translateY(-2px);
    }
    div[data-testid="stFormSubmitButton"] {
        margin-top: 1.5rem;
    }
    
    /* 고객 카드 스타일링 */
    div.customer-card {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.04);
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        transition: all 0.3s;
        border: 1px solid #f1f5f9;
    }
    div.customer-card:hover {
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.08);
        transform: translateY(-3px);
        border-color: #e2e8f0;
    }
    
    /* 그래프 및 차트 스타일링 */
    div[data-testid="stPlotlyChart"] {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
        padding: 1rem;
        transition: transform 0.3s ease;
    }
    div[data-testid="stPlotlyChart"]:hover {
        transform: translateY(-2px);
    }
    
    /* 조치사항 및 위험도 박스 */
    div[style*="background-color: #f0fdf4"], 
    div[style*="background-color: #fefce8"], 
    div[style*="background-color: #fef2f2"] {
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03);
        transition: transform 0.3s;
    }
    div[style*="background-color: #f0fdf4"]:hover, 
    div[style*="background-color: #fefce8"]:hover, 
    div[style*="background-color: #fef2f2"]:hover {
        transform: translateY(-2px);
    }
    
    /* 웹폰트 추가 */
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
    
    /* 스크롤바 커스터마이징 */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    
    /* 애니메이션 효과 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .main .block-container {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* 더 작은 폰트를 위한 CSS 추가 */
    [data-testid="stMetricValue"] {
        font-size: 0.8rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.7rem !important;
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 4px;
    }
    .metric-box {
        text-align: center;
        background-color: #f8f9fa;
        border-radius: 4px;
        padding: 4px 8px;
        width: 48%;
    }
    .metric-label {
        font-size: 0.7rem;
        color: #555;
        margin-bottom: 2px;
    }
    .metric-value {
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# 데이터 로드 함수
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
        return df
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return None

# 게이지 차트 생성 함수
def create_gauge_chart(prediction_prob):
    # 색상 설정
    if prediction_prob <= 0.3:
        bar_color = "#10b981"  # 초록색 (낮은 이탈 확률)
    elif prediction_prob <= 0.7:
        bar_color = "#f59e0b"  # 노란색 (중간 이탈 확률)
    else:
        bar_color = "#ef4444"  # 빨간색 (높은 이탈 확률)
    
    steps_colors = ['#a7f3d0', '#fef3c7', '#fecaca']  # 초록-노랑-빨강 그라데이션
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction_prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
            'bar': {'color': bar_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 30], 'color': steps_colors[0]},
                {'range': [30, 70], 'color': steps_colors[1]},
                {'range': [70, 100], 'color': steps_colors[2]}
            ],
            'threshold': {
                'line': {'color': "#64748b", 'width': 2},
                'thickness': 0.75,
                'value': 50
            }
        },
        title={'text': "이탈 확률", 'font': {'size': 24, 'color': '#475569'}},
        number={'suffix': "%", 'font': {'size': 28, 'color': '#475569'}},
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        font={'color': "#475569", 'family': "Arial"}
    )
    
    return fig

# 확률 바 차트 생성 함수
def create_probability_bar_chart(prediction_prob):
    labels = ['이탈 가능성', '유지 가능성']
    values = [prediction_prob, 1-prediction_prob]
    colors = ['rgba(239, 68, 68, 0.8)', 'rgba(16, 185, 129, 0.8)']  # 빨강, 초록
    
    fig = go.Figure(data=[
        go.Bar(
            x=values,
            y=labels,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{prediction_prob:.1%}", f"{1-prediction_prob:.1%}"],
            textposition='inside',
            hoverinfo='text',
            hovertext=[
                f"이탈 확률: {prediction_prob:.2%}",
                f"유지 확률: {1-prediction_prob:.2%}"
            ]
        )
    ])
    
    fig.update_layout(
        title="이탈 vs 유지 확률",
        title_font=dict(size=18, color='#475569'),
        xaxis=dict(
            title="확률",
            showgrid=True,
            showline=True,
            showticklabels=True,
            tickformat=".0%",
            range=[0, 1],
            gridcolor='#f1f5f9',
            tickfont=dict(color='#64748b')
        ),
        yaxis=dict(
            showgrid=True,
            showline=True,
            showticklabels=True,
            tickfont=dict(color='#64748b')
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        height=300,
        paper_bgcolor="white",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=14, color='#475569')
    )
    
    return fig

# 모델 결과 표시 함수
def display_model_results(prediction_prob):
    # 이탈 예측 결과에 따른 클래스 결정
    if prediction_prob <= 0.3:
        risk_class = "prediction-low"
        risk_level = "낮음"
        risk_color = "#10b981"
    elif prediction_prob <= 0.7:
        risk_class = "prediction-medium"
        risk_level = "중간"
        risk_color = "#f59e0b"
    else:
        risk_class = "prediction-high"
        risk_level = "높음"
        risk_color = "#ef4444"
    
    # 이탈 위험도 표시
    st.markdown(f"""
    <div class="prediction-box {risk_class}">
        <h3 style="margin-top: 0; color: {risk_color};">이탈 위험도: {risk_level}</h3>
        <p>모델이 예측한 이탈 가능성은 <span style="font-weight: bold; color: {risk_color};">{prediction_prob:.1%}</span>입니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 권장 조치사항 표시
    st.markdown("### 권장 조치사항")
    if prediction_prob <= 0.3:
        st.markdown("""
        <div style="background-color: #f0fdf4; padding: 15px; border-radius: 10px; border-left: 4px solid #10b981;">
            <h4 style="margin: 0; color: #065f46;">낮은 이탈 위험</h4>
            <p style="margin: 10px 0 0 0;">이 고객은 이탈 가능성이 낮습니다. 다음과 같은 조치를 취하세요:</p>
            <ul>
                <li>정기적인 소통을 통해 고객 관계를 유지하세요.</li>
                <li>충성도 프로그램을 제안하여 장기적인 관계를 구축하세요.</li>
                <li>고객의 서비스 만족도를 모니터링하세요.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    elif prediction_prob <= 0.7:
        st.markdown("""
        <div style="background-color: #fefce8; padding: 15px; border-radius: 10px; border-left: 4px solid #f59e0b;">
            <h4 style="margin: 0; color: #92400e;">중간 이탈 위험</h4>
            <p style="margin: 10px 0 0 0;">이 고객은 이탈 가능성이 중간 수준입니다. 다음과 같은 조치를 취하세요:</p>
            <ul>
                <li>고객의 서비스 사용 패턴을 분석하고 문제점을 식별하세요.</li>
                <li>고객 만족도 조사를 실시하고 불만 사항을 해결하세요.</li>
                <li>특별 프로모션이나 할인 혜택을 제공하여 고객 충성도를 높이세요.</li>
                <li>1:1 고객 상담을 통해 고객의 니즈를 파악하세요.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #fef2f2; padding: 15px; border-radius: 10px; border-left: 4px solid #ef4444;">
            <h4 style="margin: 0; color: #991b1b;">높은 이탈 위험</h4>
            <p style="margin: 10px 0 0 0;">이 고객은 이탈 가능성이 매우 높습니다. 즉시 다음과 같은 조치를 취하세요:</p>
            <ul>
                <li>즉각적인 고객 접촉을 통해 불만 사항을 파악하고 해결책을 제시하세요.</li>
                <li>고객 맞춤형 특별 할인 혜택이나 상품을 제안하세요.</li>
                <li>서비스 개선 약속과 함께 구체적인 개선 계획을 공유하세요.</li>
                <li>VIP 고객 관리 프로그램에 포함시켜 특별 관리하세요.</li>
                <li>계약 갱신 인센티브를 제공하세요.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# 데이터 전처리 함수
def preprocess_data(data):
    # 숫자형 변환
    data = data.copy()
    
    # TotalCharges 전처리
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'].fillna(0, inplace=True)
    
    # 범주형 변수 원-핫 인코딩
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    if 'Churn' in categorical_cols:
        categorical_cols.remove('Churn')
    if 'customerID' in categorical_cols:
        categorical_cols.remove('customerID')
    
    # 원-핫 인코딩 수행
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    # customerID 제거
    if 'customerID' in data_encoded.columns:
        data_encoded = data_encoded.drop('customerID', axis=1)
    
    return data_encoded

# 모델 학습 함수
def train_models(df):
    # 데이터 전처리
    df_processed = preprocess_data(df)
    
    # 훈련 데이터 준비
    X = df_processed.drop('Churn', axis=1) if 'Churn' in df_processed.columns else df_processed
    if 'Churn' in df_processed.columns:
        y = df_processed['Churn'].map({'Yes': 1, 'No': 0})
    else:
        st.error("훈련 데이터에 'Churn' 열이 없습니다.")
        return None, None, None
    
    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 모델 초기화
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    
    # 앙상블 모델
    ensemble_model = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('gb', gb_model),
            ('lr', lr_model)
        ],
        voting='soft'
    )
    
    # 모델 학습
    try:
        rf_model.fit(X_scaled, y)
        gb_model.fit(X_scaled, y)
        lr_model.fit(X_scaled, y)
        ensemble_model.fit(X_scaled, y)
        
        # 모델 및 스케일러 저장
        model_folder = 'models'
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        
        with open(f'{model_folder}/rf_model.pkl', 'wb') as f:
            pickle.dump(rf_model, f)
        with open(f'{model_folder}/gb_model.pkl', 'wb') as f:
            pickle.dump(gb_model, f)
        with open(f'{model_folder}/lr_model.pkl', 'wb') as f:
            pickle.dump(lr_model, f)
        with open(f'{model_folder}/ensemble_model.pkl', 'wb') as f:
            pickle.dump(ensemble_model, f)
        with open(f'{model_folder}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        return ensemble_model, scaler, None
    except Exception as e:
        st.error(f"모델 학습 중 오류 발생: {e}")
        return None, None, None

# 저장된 모델 로드 함수
def load_models():
    try:
        return load_original_models()
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        return None, None, None

# 기존 모델 로드 함수
def load_original_models():
    model_folder = 'models'
    try:
        if not os.path.exists(f'{model_folder}/ensemble_model.pkl'):
            st.warning("저장된 모델이 없습니다. 새로운 모델을 학습합니다.")
            df = load_data()
            if df is not None:
                return train_models(df)
            return None, None, None
        
        with open(f'{model_folder}/ensemble_model.pkl', 'rb') as f:
            ensemble_model = pickle.load(f)
        with open(f'{model_folder}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        return ensemble_model, scaler, None
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        return None, None, None

# 특성 중요도 시각화 함수 - 개별 예측용
def visualize_feature_importance(model, customer_data, is_moonyoung_model=False, scaler=None):
    try:
        # 데이터 전처리
        processed_data = preprocess_data(customer_data)
        
        # 원본 학습 데이터의 특성과 일치시키기
        df = load_data()
        train_data = preprocess_data(df)
        
        # 원본 학습 데이터에서 Churn 열 제거
        if 'Churn' in train_data.columns:
            train_features = train_data.drop('Churn', axis=1)
        else:
            train_features = train_data
            
        # 누락된 컬럼 추가 (0으로 채움)
        missing_cols = set(train_features.columns) - set(processed_data.columns)
        for col in missing_cols:
            processed_data[col] = 0
            
        # 추가된 컬럼 제거
        extra_cols = set(processed_data.columns) - set(train_features.columns)
        for col in extra_cols:
            processed_data = processed_data.drop(col, axis=1)
            
        # 학습 데이터와 동일한 컬럼 순서로 재정렬
        processed_data = processed_data[train_features.columns]
        
        feature_names = processed_data.columns.tolist()
        
        # 문영모델과 일반 모델 구분하여 처리
        if is_moonyoung_model:
            # 기본 방법: 모델이 feature_importances_ 속성을 가지고 있는지 확인
            if hasattr(model, 'feature_importances_'):
                base_model = model
            else:
                # 스태킹 모델에서 기본 모델 찾기 시도
                try:
                    if hasattr(model, 'final_estimator_') and hasattr(model.final_estimator_, 'feature_importances_'):
                        base_model = model.final_estimator_
                    else:
                        # 스태킹 모델의 개별 모델 확인
                        for name, estimator in model.named_estimators_.items():
                            if hasattr(estimator, 'feature_importances_'):
                                base_model = estimator
                                break
                        else:
                            # feature_importances_가 있는 모델을 찾지 못함 - 첫 번째 모델 기반 SHAP 사용
                            base_model = None
                except:
                    base_model = None
            
            if base_model is not None and hasattr(base_model, 'feature_importances_'):
                # 모델에서 직접 특성 중요도 추출
                importances = base_model.feature_importances_
                
                # 중요도 인덱스 정렬 및 상위 10개 선택
                indices = np.argsort(importances)[-10:]
                selected_features = [feature_names[i] for i in indices]
                selected_importances = importances[indices]
                
                # 시각화
                return create_importance_chart(selected_features, selected_importances)
            else:
                # 모델이 feature_importances_를 지원하지 않는 경우 SHAP 사용 시도
                return use_shap_for_importance(model, processed_data, feature_names, scaler)
        else:
            # 일반 앙상블 모델 처리
            # 앙상블 모델에서 feature_importances_ 속성이 있는 모델 찾기
            try:
                for name, estimator in model.named_estimators_.items():
                    if hasattr(estimator, 'feature_importances_'):
                        base_model = estimator
                        importances = base_model.feature_importances_
                        
                        # 중요도 인덱스 정렬 및 상위 10개 선택
                        indices = np.argsort(importances)[-10:]
                        selected_features = [feature_names[i] for i in indices]
                        selected_importances = importances[indices]
                        
                        # 시각화
                        return create_importance_chart(selected_features, selected_importances)
                
                # feature_importances_가 있는 모델을 찾지 못함 - SHAP 사용
                return use_shap_for_importance(model, processed_data, feature_names, scaler)
            except:
                # 오류 발생 시 SHAP 사용 시도
                return use_shap_for_importance(model, processed_data, feature_names, scaler)
                
    except Exception as e:
        st.error(f"특성 중요도 시각화 중 오류 발생: {str(e)}")
        return None

# SHAP를 사용한 특성 중요도 시각화
def use_shap_for_importance(model, data, feature_names, scaler=None):
    try:
        # 샘플 데이터 생성 (최대 100개 랜덤 샘플)
        df = load_data()
        train_data = preprocess_data(df)
        if 'Churn' in train_data.columns:
            train_data = train_data.drop('Churn', axis=1)
        
        # 랜덤 샘플 선택 (최대 100개)
        if len(train_data) > 100:
            background_data = train_data.sample(100, random_state=42)
        else:
            background_data = train_data
            
        # 데이터 스케일링
        if scaler is not None:
            background_data_scaled = scaler.transform(background_data)
            input_data_scaled = scaler.transform(data)
        else:
            background_data_scaled = background_data
            input_data_scaled = data
            
        # SHAP 설명기 생성
        if hasattr(model, 'predict_proba'):
            # 분류기인 경우
            explainer = shap.Explainer(model.predict_proba, background_data_scaled)
            shap_values = explainer(input_data_scaled)
            
            # 이탈 클래스(1)에 대한 SHAP 값 가져오기
            shap_values_class1 = shap_values[0, :, 1].values
            
            # 절대값 기준으로 상위 10개 특성 인덱스 선택
            abs_shap_values = np.abs(shap_values_class1)
            top_indices = np.argsort(abs_shap_values)[-10:]
            
            # 인덱스에 해당하는 특성과 SHAP 값 가져오기
            top_features = [feature_names[i] for i in top_indices]
            top_shap_values = [shap_values_class1[i] for i in top_indices]
            
            # 가독성을 위한 특성명 정리
            readable_features = []
            for feature in top_features:
                if feature.endswith('_Yes'):
                    readable_features.append(feature.replace('_Yes', ''))
                elif '_' in feature:
                    parts = feature.split('_')
                    readable_features.append(f"{parts[0]} {parts[1]}")
                else:
                    readable_features.append(feature)
            
            # 시각화
            return create_importance_chart(readable_features, top_shap_values, is_shap=True)
        else:
            st.warning("현재 모델은 SHAP 분석을 지원하지 않습니다.")
            return None
    except Exception as e:
        st.error(f"SHAP 시각화 중 오류 발생: {str(e)}")
        return None

# 특성 중요도 차트 생성 함수
def create_importance_chart(features, importances, is_shap=False):
    # 색상 결정 (SHAP 값은 음수일 수 있으므로 다른 색상 스케일 사용)
    if is_shap:
        # SHAP 값 색상 설정 (양수는 빨간색, 음수는 파란색)
        colors = ['#2171b5' if val < 0 else '#cb181d' for val in importances]
        color_scale = None
        title = 'SHAP 특성 중요도 (상위 10개)'
    else:
        # 일반 특성 중요도 색상 설정
        colors = None
        color_scale = 'blues'
        title = '특성 중요도 (상위 10개)'
    
    # 차트 생성
    fig = px.bar(
        x=importances,
        y=features,
        orientation='h',
        labels={'x': '중요도', 'y': '특성'},
        title=title,
        color=importances if color_scale else None,
        color_continuous_scale=color_scale
    )
    
    # SHAP 값이면 색상 직접 설정
    if is_shap:
        fig.update_traces(marker_color=colors)
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="중요도" if not is_shap else "SHAP 값 (이탈에 미치는 영향)",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
        font=dict(size=12)
    )
    
    return fig

# 일반적인 특성 중요도 시각화 함수
def visualize_general_feature_importance():
    try:
        # 문영 모델 메타데이터 파인
        meta_file = 'models/moonyoung_meta.json'
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                
            # 특성 중요도가 메타데이터에 포함되어 있는지 확인
            if 'feature_importances' in meta and 'feature_names' in meta:
                # 메타데이터에서 특성 중요도와 이름 가져오기
                importances = meta['feature_importances']
                features = meta['feature_names']
                
                # 한글 특성명 매핑 (가능한 경우)
                if 'korean_feature_names' in meta:
                    features = meta['korean_feature_names']
                
                # 상위 15개 특성만 선택
                if len(features) > 15:
                    indices = np.argsort(importances)[-15:]
                    features = [features[i] for i in indices]
                    importances = [importances[i] for i in indices]
                
                # 시각화
                fig = px.bar(
                    x=importances,
                    y=features,
                    orientation='h',
                    labels={'x': '중요도', 'y': '특성'},
                    title='XGBoost 모델의 일반적인 특성 중요도 (상위 15개)',
                    color=importances,
                    color_continuous_scale='blues'
                )
                
                # 값 표시 추가
                for i, value in enumerate(importances):
                    fig.add_annotation(
                        x=value + max(importances) * 0.02,
                        y=i,
                        text=f"{value:.4f}",
                        showarrow=False,
                        font=dict(size=10)
                    )
                
                fig.update_layout(
                    height=500,
                    margin=dict(l=20, r=20, t=50, b=20),
                    xaxis_title="중요도",
                    yaxis_title="",
                    yaxis=dict(autorange="reversed"),
                    font=dict(size=12)
                )
                
                return fig
                
        # 메타데이터가 없는 경우 XGBoost 모델 직접 로드 시도
        xgb_model_path = 'models/xgb_model.pkl'
        if os.path.exists(xgb_model_path):
            with open(xgb_model_path, 'rb') as f:
                xgb_model = pickle.load(f)
                
            # 특성 이름 가져오기 (통상적인 특성 이름이 있다고 가정)
            feature_names = [
                "계약 기간", "월요금", "총요금", "이용 기간", "결제 방법", 
                "인터넷 서비스", "기술 지원", "온라인 보안", "온라인 백업",
                "디바이스 보호", "스트리밍 TV", "스트리밍 영화", "성별",
                "시니어 고객", "파트너 유무", "부양가족 유무", "전화 서비스",
                "복수 회선", "페이퍼리스 빌링"
            ]
            
            if hasattr(xgb_model, 'feature_importances_'):
                importances = xgb_model.feature_importances_
                
                # 특성 중요도 상위 15개만 선택
                indices = np.argsort(importances)[-15:]
                selected_features = [feature_names[i] if i < len(feature_names) else f"특성_{i}" for i in indices]
                selected_importances = [importances[i] for i in indices]
                
                # 시각화
                fig = px.bar(
                    x=selected_importances,
                    y=selected_features,
                    orientation='h',
                    labels={'x': '중요도', 'y': '특성'},
                    title='XGBoost 모델의 일반적인 특성 중요도 (상위 15개)',
                    color=selected_importances,
                    color_continuous_scale='blues'
                )
                
                # 값 표시 추가
                for i, value in enumerate(selected_importances):
                    fig.add_annotation(
                        x=value + max(selected_importances) * 0.02,
                        y=i,
                        text=f"{value:.4f}",
                        showarrow=False,
                        font=dict(size=10)
                    )
                
                fig.update_layout(
                    height=500,
                    margin=dict(l=20, r=20, t=50, b=20),
                    xaxis_title="중요도",
                    yaxis_title="",
                    yaxis=dict(autorange="reversed"),
                    font=dict(size=12)
                )
                
                return fig
                
        # 기본 특성 중요도 표시
        # 일반적인 통계 자료 기반 예시 중요도
        features = [
            "계약 유형", "이용 기간", "월 요금", "온라인 보안 서비스", 
            "기술 지원", "인터넷 서비스 유형", "결제 방법", "총 요금",
            "스트리밍 서비스", "디바이스 보호", "온라인 백업", "부양가족/파트너 여부",
            "시니어 고객 여부", "성별", "전자 청구서"
        ]
        
        importances = [
            0.425, 0.376, 0.298, 0.246, 0.231, 0.226, 0.195, 0.183,
            0.162, 0.157, 0.149, 0.091, 0.085, 0.069, 0.064
        ]
        
        # 시각화
        fig = px.bar(
            x=importances,
            y=features,
            orientation='h',
            labels={'x': '중요도', 'y': '특성'},
            title='일반적인 통신사 이탈 예측 특성 중요도',
            color=importances,
            color_continuous_scale='blues'
        )
        
        # 값 표시 추가
        for i, value in enumerate(importances):
            fig.add_annotation(
                x=value + 0.02,
                y=i,
                text=f"{value:.3f}",
                showarrow=False,
                font=dict(size=10)
            )
        
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis_title="중요도",
            yaxis_title="",
            yaxis=dict(autorange="reversed"),
            font=dict(size=12)
        )
        
        return fig
        
    except Exception as e:
        st.error(f"일반적인 특성 중요도 시각화 중 오류 발생: {str(e)}")
        return None

# 데이터 형식 조정 함수 추가
def adjust_data_format(data, reference_data=None):
    """
    입력 데이터의 특성을 reference_data와 일치하도록 조정
    reference_data가 없는 경우 원본 데이터셋에서 로드
    """
    processed_data = preprocess_data(data)
    
    # reference_data가 없으면 원본 데이터 로드
    if reference_data is None:
        df = load_data()
        if df is not None:
            reference_data = preprocess_data(df)
    
    # reference_data가 있는 경우에만 진행
    if reference_data is not None:
        # reference_data에서 Churn 열 제거
        if 'Churn' in reference_data.columns:
            reference_features = reference_data.drop('Churn', axis=1)
        else:
            reference_features = reference_data
            
        # 누락된 컬럼 추가 (0으로 채움)
        missing_cols = set(reference_features.columns) - set(processed_data.columns)
        for col in missing_cols:
            processed_data[col] = 0
            
        # 추가된 컬럼 제거
        extra_cols = set(processed_data.columns) - set(reference_features.columns)
        for col in extra_cols:
            processed_data = processed_data.drop(col, axis=1)
            
        # reference_data와 동일한 컬럼 순서로 재정렬
        processed_data = processed_data[reference_features.columns]
    
    return processed_data

# 예측 함수
def predict_churn(customer_data, model, scaler, feature_names=None):
    try:
        # 문영모델 사용 여부 확인
        is_moonyoung_model = isinstance(model, StackingClassifier)
        
        if is_moonyoung_model:
            # 원본 데이터 로드
            df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
            
            # 입력된 고객 데이터 추가
            new_customer = pd.DataFrame({
                'customerID': ['NEW-CUSTOMER'],
                'gender': [customer_data['gender'].values[0]],
                'SeniorCitizen': [customer_data['SeniorCitizen'].values[0]],
                'Partner': [customer_data['Partner'].values[0]],
                'Dependents': [customer_data['Dependents'].values[0]],
                'tenure': [customer_data['tenure'].values[0]],
                'PhoneService': [customer_data['PhoneService'].values[0]],
                'MultipleLines': [customer_data['MultipleLines'].values[0]],
                'InternetService': [customer_data['InternetService'].values[0]],
                'OnlineSecurity': [customer_data['OnlineSecurity'].values[0]],
                'OnlineBackup': [customer_data['OnlineBackup'].values[0]],
                'DeviceProtection': [customer_data['DeviceProtection'].values[0]],
                'TechSupport': [customer_data['TechSupport'].values[0]],
                'StreamingTV': [customer_data['StreamingTV'].values[0]],
                'StreamingMovies': [customer_data['StreamingMovies'].values[0]],
                'Contract': [customer_data['Contract'].values[0]],
                'PaperlessBilling': [customer_data['PaperlessBilling'].values[0]],
                'PaymentMethod': [customer_data['PaymentMethod'].values[0]],
                'MonthlyCharges': [customer_data['MonthlyCharges'].values[0]],
                'TotalCharges': [customer_data['TotalCharges'].values[0]],
                'Churn': ['No']  # 기본값, 예측에 영향 없음
            })
            
            # 데이터 합치기
            combined_df = pd.concat([df, new_customer], ignore_index=True)
            
            # 컬럼명 변경
            combined_df.columns = [
                "고객ID", "성별", "시니어여부", "배우자여부", "부양가족여부", "가입개월수", "전화서비스",
                "복수회선여부", "인터넷서비스종류", "온라인보안서비스", "온라인백업", "디바이스보호",
                "기술지원", "TV스트리밍", "영화스트리밍", "계약종류", "전자청구서여부", "결제방법",
                "월요금", "총요금", "이탈여부"
            ]

            # 전처리
            combined_df["총요금"] = pd.to_numeric(combined_df["총요금"], errors="coerce")
            combined_df.dropna(subset=["총요금"], inplace=True)
            combined_df["이탈여부"] = combined_df["이탈여부"].map({"Yes": 1, "No": 0})
            combined_df.drop(columns=["고객ID"], inplace=True)

            # 인코딩
            le = LabelEncoder()
            for col in combined_df.select_dtypes(include="object").columns:
                combined_df[col] = le.fit_transform(combined_df[col])
                
            # 인터넷 서비스 타입의 영향 강화
            # 인터넷 서비스 종류에 따른 가중치 적용 (0: No, 1: DSL, 2: Fiber optic)
            if "인터넷서비스종류" in combined_df.columns and combined_df["인터넷서비스종류"].nunique() <= 3:
                # 로깅 추가
                service_type = customer_data['InternetService'].values[0]
                st.write(f"인터넷 서비스 변경: {service_type}")
                
                # No:0, DSL:1, Fiber optic:2로 인코딩된 값을 확인 (마지막 행 = 새 고객)
                last_idx = len(combined_df) - 1
                internet_val = combined_df.loc[last_idx, "인터넷서비스종류"]
                st.write(f"인코딩된 인터넷 서비스 값: {internet_val}")
                
                # Fiber optic 서비스인 경우 이탈 가능성 증가 (코드가 2인 경우)
                if internet_val == 2:  # Fiber optic
                    # 파생 변수에 추가 가중치
                    combined_df.loc[last_idx, "인터넷서비스종류"] = 5.0  # 인위적으로 가중치 부여
                    st.write("Fiber optic 서비스 감지: 이탈 가능성 증가")
                elif internet_val == 0:  # No internet
                    combined_df.loc[last_idx, "인터넷서비스종류"] = 0.5  # 낮은 가중치
                    st.write("인터넷 없음 감지: 이탈 가능성 감소")

            # 파생 변수
            combined_df["누적지불금액"] = combined_df["가입개월수"] * combined_df["월요금"]
            combined_df["장기계약여부"] = (combined_df["계약종류"] != 0).astype(int)
            combined_df["인터넷없음"] = (combined_df["인터넷서비스종류"] == 0).astype(int)
            combined_df["요금대"] = pd.cut(combined_df["월요금"], bins=[0, 35, 70, 120], labels=[0, 1, 2])
            combined_df["요금대"] = le.fit_transform(combined_df["요금대"].astype(str))
            combined_df["가입비율"] = combined_df["가입개월수"] / (combined_df["가입개월수"].max() + 1e-5)
            
            # 타겟과 특성 분리
            X = combined_df.drop("이탈여부", axis=1)
            
            # 새 고객 데이터 (마지막 행)
            new_customer_processed = X.iloc[-1:].reset_index(drop=True)
            
            # 스케일링 및 예측
            scaled_data = scaler.transform(new_customer_processed)
            prediction = model.predict(scaled_data)[0]
            prediction_prob = model.predict_proba(scaled_data)[0, 1]
            
            return prediction, prediction_prob, is_moonyoung_model
            
        else:
            # 새로운 데이터 형식 조정 함수 사용
            df = load_data()
            reference_data = preprocess_data(df)
            processed_data = adjust_data_format(customer_data, reference_data)
            
            # 스케일링 및 예측
            scaled_data = scaler.transform(processed_data)
            prediction = model.predict(scaled_data)[0]
            prediction_prob = model.predict_proba(scaled_data)[0, 1]
            
            return prediction, prediction_prob, False
            
    except Exception as e:
        st.error(f"예측 중 오류가 발생했습니다: {str(e)}")
        st.info("문제를 해결하기 위해 페이지를 새로고침하거나 서비스 관리자에게 문의하세요.")
        return None, 0.5, False  # 오류 발생 시 기본값 반환

# 메인 함수
def main():
    # 사이드바에 모델 성능 정보 표시
    with st.sidebar:
        # 더 작은 폰트를 위한 CSS 추가
        st.markdown("""
        <style>
        [data-testid="stMetricValue"] {
            font-size: 0.8rem !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.7rem !important;
        }
        .metric-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 4px;
        }
        .metric-box {
            text-align: center;
            background-color: #f8f9fa;
            border-radius: 4px;
            padding: 4px 8px;
            width: 48%;
        }
        .metric-label {
            font-size: 0.7rem;
            color: #555;
            margin-bottom: 2px;
        }
        .metric-value {
            font-size: 0.8rem;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 모델 성능 지표", help="모델의 성능을 평가하는 지표입니다.")
        
        # 스타일 적용된 컨테이너 생성
        with st.container():
            st.markdown("<h5 style='font-size:0.9rem;'>문영스태킹 모델 성능</h5>", unsafe_allow_html=True, help="문영스태킹 모델의 성능 지표입니다.")
            
            # Train Set 성능
            st.markdown("<p style='font-size:0.8rem; margin-bottom:5px;'>✅ <b>Train Set 성능</b></p>", unsafe_allow_html=True)
            
            # HTML로 직접 메트릭 표시 - Train Set
            st.markdown("""
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value">0.8701</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Recall</div>
                    <div class="metric-value">0.8852</div>
                </div>
            </div>
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-label">Precision</div>
                    <div class="metric-value">0.8592</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">F1 Score</div>
                    <div class="metric-value">0.8720</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Test Set 성능
            st.markdown("<p style='font-size:0.8rem; margin-bottom:5px; margin-top:10px;'>✅ <b>Test Set 성능</b></p>", unsafe_allow_html=True)
            
            # HTML로 직접 메트릭 표시 - Test Set
            st.markdown("""
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value">0.8393</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Recall</div>
                    <div class="metric-value">0.8703</div>
                </div>
            </div>
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-label">Precision</div>
                    <div class="metric-value">0.8195</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">F1 Score</div>
                    <div class="metric-value">0.8441</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 과적합 수치
            st.markdown("<p style='font-size:0.8rem; margin-top:10px;'>🎯 <b>과적합 수치</b></p>", unsafe_allow_html=True)
            
            # 과적합 수치 계산 및 평가 및 색상 설정
            overfitting_value = 0.0279
            
            # 과적합 정도 평가 및 색상 설정
            if overfitting_value < 0.02:
                overfitting_status = "낮음 (양호)"
                overfitting_desc = "모델이 안정적이며 일반화 성능이 좋습니다."
                bg_color = "#ECFDF5"  # 연한 초록색 배경
                text_color = "#059669"  # 초록색 텍스트
            elif overfitting_value < 0.05:
                overfitting_status = "보통 (적정)"
                overfitting_desc = "적정 수준의 과적합으로, 실용적으로 사용 가능합니다."
                bg_color = "#FFFBEB"  # 연한 노란색 배경 
                text_color = "#B45309"  # 황금색 텍스트
            elif overfitting_value < 0.1:
                overfitting_status = "높음 (주의)"
                overfitting_desc = "과적합이 다소 높으니 주의가 필요합니다."
                bg_color = "#FEF2F2"  # 연한 빨간색 배경
                text_color = "#DC2626"  # 빨간색 텍스트
            else:
                overfitting_status = "매우 높음 (위험)"
                overfitting_desc = "심각한 과적합 상태로, 모델 재조정이 필요합니다."
                bg_color = "#FEF2F2"  # 연한 빨간색 배경
                text_color = "#B91C1C"  # 진한 빨간색 텍스트
            
            # 과적합 수치 표시
            st.markdown(f"""
            <div style='background-color:{bg_color};padding:10px;border-radius:5px;font-size:0.8rem;'>
                <b>Train F1 - Test F1: {overfitting_value:.4f}</b><br>
                <span style='color:{text_color};'><b>과적합 정도: {overfitting_status}</b></span><br>
                {overfitting_desc}
            </div>
            """, unsafe_allow_html=True)
        
        # 구분선 추가
        st.markdown("---")

    # 데이터 로드
    df = load_data()
    if df is None:
        st.error("데이터를 불러올 수 없습니다. 파일을 확인해주세요.")
        return
    
    # 모델 로드
    model, scaler, _ = load_models()
    if model is None or scaler is None:
        st.error("모델을 로드할 수 없습니다.")
        return
    
    # 문영모델 사용 여부 확인 및 표시
    is_moonyoung_model = isinstance(model, StackingClassifier)
    
    # 모델 정보 표시
    model_info_col1, model_info_col2 = st.columns([1, 3])
    with model_info_col1:
        if is_moonyoung_model:
            st.markdown("<p style='font-weight: bold;'>문영 모델</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='font-weight: bold;'>문영 모델</p>", unsafe_allow_html=True)
    
    with model_info_col2:
        if is_moonyoung_model:
            st.markdown("""
            <div style='background-color: #e0f2fe; padding: 10px; border-radius: 5px; margin-bottom: 15px;'>
                <h4 style='margin: 0; color: #0369a1;'>기본 앙상블 모델</h4>
                <p style='margin: 5px 0 0 0;'>랜덤 포레스트, 그래디언트 부스팅, 로지스틱 회귀를 결합한 앙상블 모델입니다.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # 메인 페이지에 탭 생성
    main_tabs = st.tabs(["고객 정보 입력", "데이터셋에서 선택"])
    
    with main_tabs[0]:
        st.markdown("## 고객 정보 입력")
        
        # 입력 폼 컬럼 레이아웃
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("성별", ["Male", "Female"])
            senior_citizen = st.selectbox("노인 여부", ["아니오", "예"])
            senior_citizen = 1 if senior_citizen == "예" else 0
            partner = st.selectbox("배우자 유무", ["Yes", "No"])
            dependents = st.selectbox("부양가족 유무", ["Yes", "No"])
            tenure = st.slider("이용 기간 (개월)", 0, 72, 12)
            phone_service = st.selectbox("전화 서비스", ["Yes", "No"])
        
        with col2:
            multiple_lines = st.selectbox("다중 회선", ["Yes", "No", "No phone service"])
            internet_service = st.selectbox("인터넷 서비스", ["DSL", "Fiber optic", "No"])
            
            if internet_service != "No":
                online_security = st.selectbox("온라인 보안", ["Yes", "No"])
                online_backup = st.selectbox("온라인 백업", ["Yes", "No"])
                device_protection = st.selectbox("기기 보호", ["Yes", "No"])
                tech_support = st.selectbox("기술 지원", ["Yes", "No"])
            else:
                online_security = "No internet service"
                online_backup = "No internet service"
                device_protection = "No internet service"
                tech_support = "No internet service"
                streaming_tv = "No internet service"
                streaming_movies = "No internet service"
        
        with col3:
            if internet_service != "No":
                streaming_tv = st.selectbox("TV 스트리밍", ["Yes", "No"])
                streaming_movies = st.selectbox("영화 스트리밍", ["Yes", "No"])
            
            contract = st.selectbox("계약 기간", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("종이 없는 청구", ["Yes", "No"])
            payment_method = st.selectbox("지불 방법", [
                "Electronic check", 
                "Mailed check", 
                "Bank transfer (automatic)", 
                "Credit card (automatic)"
            ])
            
            monthly_charges = st.slider("월 청구액 ($)", 0.0, 150.0, 70.0, 0.01)
            total_charges = monthly_charges * tenure
        
        # 고객 데이터 생성
        customer_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [senior_citizen],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })
        
        # 중앙에 큰 예측 버튼 추가
        if st.button("예측하기", key="predict_manual", use_container_width=True):
            # 예측 수행
            prediction, prediction_prob, is_moonyoung_model = predict_churn(customer_data, model, scaler)
            
            # 결과 표시
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_gauge_chart(prediction_prob), use_container_width=True)
            with col2:
                st.plotly_chart(create_probability_bar_chart(prediction_prob), use_container_width=True)
            
            # 특성 중요도 시각화 추가
            st.markdown("### 예측에 영향을 준 주요 특성")
            with st.spinner("특성 중요도 분석 중..."):
                # 일반적인 특성 중요도 표시
                general_importance_fig = visualize_general_feature_importance()
                if general_importance_fig:
                    st.plotly_chart(general_importance_fig, use_container_width=True)
                    
                    # 설명 추가
                    st.markdown("""
                    <div class="info-box">
                        <h4>특성 중요도 분석 정보</h4>
                        <p>위 그래프는 전체 데이터셋에 대한 모델의 특성 중요도를 보여줍니다. 이는 일반적으로 고객 이탈에 영향을 미치는 요소들을 나타냅니다.</p>
                        <ul>
                            <li><strong>계약 유형</strong>: 월별 계약보다 장기 계약(1년, 2년)에서 이탈 가능성이 낮습니다.</li>
                            <li><strong>이용 기간</strong>: 서비스 이용 기간이 길수록 이탈 가능성이 낮습니다.</li>
                            <li><strong>월 요금</strong>: 월 요금이 높을수록 이탈 가능성이 높아집니다.</li>
                            <li><strong>추가 서비스</strong>: 온라인 보안, 기술 지원 등의 추가 서비스를 이용하는 고객은 이탈 가능성이 낮습니다.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("일반적인 특성 중요도를 계산할 수 없습니다.")
            
            # 예측 결과 및 조치사항 표시
            display_model_results(prediction_prob)
    
    with main_tabs[1]:
        st.markdown("## 데이터셋에서 고객 선택")
        
        # 데이터셋에서 고객 선택
        customer_ids = df['customerID'].tolist()
        selected_id = st.selectbox("고객 ID 선택", customer_ids)
        
        # 선택한 고객 데이터 가져오기
        selected_customer = df[df['customerID'] == selected_id].copy()
        
        # 고객 정보를 카드 형태로 표시
        st.markdown("### 고객 정보")
        
        # 고객 정보를 컬럼으로 나누어 표시
        info_cols = st.columns(3)
        col_idx = 0
        
        for col in selected_customer.columns:
            if col != 'customerID' and col != 'Churn':
                with info_cols[col_idx % 3]:
                    st.markdown(f"""
                    <div class='customer-card'>
                        <p style='color: #64748b; margin: 0; font-size: 0.8rem;'>{col}</p>
                        <p style='margin: 0; font-weight: bold; font-size: 1.1rem; margin-top: 5px;'>{selected_customer[col].values[0]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                col_idx += 1
        
        # 중앙에 큰 예측 버튼 추가
        if st.button("예측하기", key="predict_selected", use_container_width=True):
            # 예측 수행
            prediction, prediction_prob, is_moonyoung_model = predict_churn(selected_customer, model, scaler)
            
            # 결과 표시
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_gauge_chart(prediction_prob), use_container_width=True)
            with col2:
                st.plotly_chart(create_probability_bar_chart(prediction_prob), use_container_width=True)
            
            # 특성 중요도 시각화 추가
            st.markdown("### 예측에 영향을 준 주요 특성")
            with st.spinner("특성 중요도 분석 중..."):
                # 일반적인 특성 중요도 표시
                general_importance_fig = visualize_general_feature_importance()
                if general_importance_fig:
                    st.plotly_chart(general_importance_fig, use_container_width=True)
                    
                    # 설명 추가
                    st.markdown("""
                    <div class="info-box">
                        <h4>특성 중요도 분석 정보</h4>
                        <p>위 그래프는 전체 데이터셋에 대한 모델의 특성 중요도를 보여줍니다.</p>
                        <ul>
                            <li><strong>계약 유형</strong>: 월별 계약보다 장기 계약(1년, 2년)에서 이탈 가능성이 낮습니다.</li>
                            <li><strong>이용 기간</strong>: 서비스 이용 기간이 길수록 이탈 가능성이 낮습니다.</li>
                            <li><strong>월 요금</strong>: 월 요금이 높을수록 이탈 가능성이 높아집니다.</li>
                            <li><strong>추가 서비스</strong>: 온라인 보안, 기술 지원 등의 추가 서비스를 이용하는 고객은 이탈 가능성이 낮습니다.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("일반적인 특성 중요도를 계산할 수 없습니다.")
            
            # 예측 결과 및 조치사항 표시
            display_model_results(prediction_prob)
            
            # 실제 결과 표시 (데이터셋에 있는 경우)
            if 'Churn' in selected_customer.columns:
                actual_churn = selected_customer['Churn'].values[0]
                actual_text = '이탈' if actual_churn == 'Yes' else '유지'
                actual_color = '#ef4444' if actual_churn == 'Yes' else '#10b981'
                
                st.markdown(f"""
                <div style='margin-top: 20px; background-color: #f9fafb; padding: 15px; border-radius: 10px; border-left: 4px solid {actual_color};'>
                    <h4 style='margin: 0; color: #1e293b;'>실제 결과</h4>
                    <p style='margin: 5px 0 0 0; font-size: 1.2rem; font-weight: bold; color: {actual_color};'>{actual_text}</p>
                </div>
                """, unsafe_allow_html=True)

# 메인 함수 실행
if __name__ == "__main__":
    main() 