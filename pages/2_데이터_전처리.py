import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import time

# 페이지 구성
st.set_page_config(
    page_title="데이터 전처리",
    page_icon="🔍",
    layout="wide"
)

# 스타일 정의
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
        background: linear-gradient(90deg, #1E88E5, #64B5F6);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(30, 136, 229, 0.7);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(30, 136, 229, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(30, 136, 229, 0);
        }
    }
    
    .section-header {
        font-size: 1.5rem;
        color: #1976D2;
        font-weight: 600;
        margin-top: 1rem;
        border-left: 5px solid #1976D2;
        padding-left: 10px;
    }
    
    .card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .info-text {
        background-color: #e3f2fd;
        border-left: 4px solid #1976D2;
        padding: 0.8rem;
        margin-bottom: 1rem;
        border-radius: 0 5px 5px 0;
    }
    
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4CAF50;
        padding: 0.8rem;
        margin-bottom: 1rem;
        border-radius: 0 5px 5px 0;
    }
    
    .progress-container {
        width: 100%;
        height: 20px;
        background-color: #f0f0f0;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .highlight {
        background-color: yellow;
        padding: 0 4px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# 애니메이션 헤더
st.markdown("<h1 class='main-header'>데이터 전처리 Dashboard</h1>", unsafe_allow_html=True)

# 데이터 폴더 확인
if not os.path.exists('data'):
    os.makedirs('data')

# 데이터 로드 섹션
st.markdown("<h2 class='section-header'>1. 데이터 로드</h2>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    data_load_state = st.text('데이터 로딩 중...')
    try:
        # 데이터 로드 - 올바른 파일 경로 사용
        correct_file_path = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
        if os.path.exists(correct_file_path):
            df = pd.read_csv(correct_file_path)
            
            # 로딩 애니메이션
            with st.spinner('데이터 처리 중...'):
                time.sleep(0.5)  # 로딩 효과를 위한 지연
            
            data_load_state.markdown("<div class='success-box'>✅ 데이터 로드 완료!</div>", unsafe_allow_html=True)
            
            # 데이터 요약 정보
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📊 데이터 형태:**")
                st.markdown(f"<div class='info-text'>행: {df.shape[0]:,} | 열: {df.shape[1]}</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("**🔍 메모리 사용량:**")
                memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
                st.markdown(f"<div class='info-text'>{memory_usage:.2f} MB</div>", unsafe_allow_html=True)
            
            # 데이터 미리보기
            with st.expander("데이터 미리보기", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
        else:
            st.error(f"데이터 파일을 찾을 수 없습니다: {correct_file_path}")
            # 계속 진행할 수 있도록 샘플 데이터 생성
            st.warning("샘플 데이터로 계속 진행합니다.")
            # 간단한 샘플 데이터 생성
            df = pd.DataFrame({
                'customerID': [f'customer_{i}' for i in range(100)],
                'gender': np.random.choice(['Male', 'Female'], 100),
                'SeniorCitizen': np.random.choice([0, 1], 100),
                'Partner': np.random.choice(['Yes', 'No'], 100),
                'Dependents': np.random.choice(['Yes', 'No'], 100),
                'tenure': np.random.randint(1, 72, 100),
                'PhoneService': np.random.choice(['Yes', 'No'], 100),
                'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], 100),
                'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], 100),
                'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], 100),
                'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], 100),
                'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], 100),
                'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], 100),
                'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], 100),
                'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], 100),
                'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 100),
                'PaperlessBilling': np.random.choice(['Yes', 'No'], 100),
                'PaymentMethod': np.random.choice([
                    'Electronic check', 'Mailed check', 
                    'Bank transfer (automatic)', 'Credit card (automatic)'
                ], 100),
                'MonthlyCharges': np.random.uniform(20, 120, 100),
                'TotalCharges': np.random.uniform(200, 8000, 100),
                'Churn': np.random.choice(['Yes', 'No'], 100)
            })
    
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        # 계속 진행할 수 있도록 샘플 데이터 생성
        st.warning("오류가 발생해 샘플 데이터로 계속 진행합니다.")
        # 간단한 샘플 데이터 생성
        df = pd.DataFrame({
            'customerID': [f'customer_{i}' for i in range(100)],
            'gender': np.random.choice(['Male', 'Female'], 100),
            'SeniorCitizen': np.random.choice([0, 1], 100),
            'Partner': np.random.choice(['Yes', 'No'], 100),
            'Dependents': np.random.choice(['Yes', 'No'], 100),
            'tenure': np.random.randint(1, 72, 100),
            'PhoneService': np.random.choice(['Yes', 'No'], 100),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], 100),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], 100),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], 100),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], 100),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], 100),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], 100),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], 100),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], 100),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 100),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], 100),
            'PaymentMethod': np.random.choice([
                'Electronic check', 'Mailed check', 
                'Bank transfer (automatic)', 'Credit card (automatic)'
            ], 100),
            'MonthlyCharges': np.random.uniform(20, 120, 100),
            'TotalCharges': np.random.uniform(200, 8000, 100),
            'Churn': np.random.choice(['Yes', 'No'], 100)
        })
    
    st.markdown("</div>", unsafe_allow_html=True)

# 전역 변수로 df 선언 (이후 코드에서 df가 참조될 때 에러 방지)
if 'df' not in locals():
    # 샘플 데이터 생성 (실제로는 위에서 이미 생성되었을 것임)
    df = pd.DataFrame()

# 데이터 전처리 섹션
st.markdown("<h2 class='section-header'>2. 데이터 전처리</h2>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    # 데이터프레임이 비어있는지 확인
    if df.empty:
        st.error("데이터가 로드되지 않았습니다. 먼저 데이터를 로드해주세요.")
    else:
        # 데이터 전처리
        try:
            st.markdown("<div class='info-text'>📋 고객 이탈 예측을 위한 데이터 전처리를 수행합니다.</div>", unsafe_allow_html=True)
            
            # 결측치 처리
            col1, col2 = st.columns(2)
            with col1:
                missing_values = df.isnull().sum()
                missing_values = missing_values[missing_values > 0]
                
                if len(missing_values) > 0:
                    st.markdown("**⚠️ 결측치 발견:**")
                    st.dataframe(pd.DataFrame({'결측치 수': missing_values}))
                    
                    # 결측치 처리
                    for col in missing_values.index:
                        if df[col].dtype in ['int64', 'float64']:
                            df[col].fillna(df[col].mean(), inplace=True)
                        else:
                            df[col].fillna(df[col].mode()[0], inplace=True)
                    
                    st.markdown("<div class='success-box'>✅ 결측치 처리 완료!</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='success-box'>✅ 결측치 없음</div>", unsafe_allow_html=True)
            
            with col2:
                # 중복 데이터 확인
                duplicates = df.duplicated().sum()
                if duplicates > 0:
                    st.markdown(f"**⚠️ 중복 데이터 발견: {duplicates}개**")
                    df = df.drop_duplicates()
                    st.markdown("<div class='success-box'>✅ 중복 데이터 제거 완료!</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='success-box'>✅ 중복 데이터 없음</div>", unsafe_allow_html=True)
            
            # 데이터 타입 변환
            st.markdown("**🔄 데이터 타입 변환:**")
            
            # TotalCharges 열을 숫자형으로 변환
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'].fillna(0, inplace=True)
            
            # SeniorCitizen을 범주형으로 변환 (필요한 경우)
            if 'SeniorCitizen' in df.columns:
                df['SeniorCitizen'] = df['SeniorCitizen'].astype(str).replace({'0': 'No', '1': 'Yes'})
            
            # 프로그레스 바를 이용한 전처리 단계 표시
            steps = ['데이터 로드', '결측치 처리', '중복 제거', '데이터 타입 변환', '인코딩 준비']
            step_idx = 4  # 현재 단계
            
            st.markdown("<div class='progress-container'>", unsafe_allow_html=True)
            st.progress(step_idx / (len(steps) - 1))
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown(f"진행 단계: **{steps[step_idx]}** ({step_idx + 1}/{len(steps)})", unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"데이터 전처리 중 오류 발생: {e}")
        
    st.markdown("</div>", unsafe_allow_html=True)

# 인코딩 및 스케일링 섹션
st.markdown("<h2 class='section-header'>3. 특성 변환 및 인코딩</h2>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    # 데이터프레임이 비어있는지 확인
    if df.empty:
        st.error("데이터가 로드되지 않았습니다. 먼저 데이터를 로드해주세요.")
    else:
        try:
            # 전처리할 열 분리
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            # customerID가 있으면 제외
            if 'customerID' in categorical_cols:
                categorical_cols.remove('customerID')
            
            # Churn이 있으면 제외
            if 'Churn' in categorical_cols:
                categorical_cols.remove('Churn')
            
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # 디렉토리 확인 및 생성
            if not os.path.exists('models'):
                os.makedirs('models')
            
            if not os.path.exists('data'):
                os.makedirs('data')
            
            # 열 정보 표시
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔠 범주형 변수:**")
                for col in categorical_cols:
                    st.markdown(f"- {col} (유니크 값: {df[col].nunique()})")
            
            with col2:
                st.markdown("**🔢 수치형 변수:**")
                for col in numerical_cols:
                    st.markdown(f"- {col} (범위: {df[col].min():.2f} ~ {df[col].max():.2f})")
            
            # 인코딩 및 스케일링 파이프라인 생성
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_cols),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                ])
            
            # 결과 표시
            st.markdown("<div class='info-text'>⚙️ 전처리 파이프라인이 생성되었습니다.</div>", unsafe_allow_html=True)
            
            # Churn 인코딩
            st.markdown("**🎯 타겟 변수(Churn) 인코딩:**")
            if 'Churn' in df.columns:
                df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
                st.markdown("<div class='success-box'>✅ 타겟 변수 인코딩 완료!</div>", unsafe_allow_html=True)
            else:
                st.warning("'Churn' 컬럼이 데이터에 없습니다.")
            
            # 피처와 타겟 분리
            X = df.drop(['customerID', 'Churn'] if 'Churn' in df.columns else ['customerID'], axis=1)
            y = df['Churn'] if 'Churn' in df.columns else pd.Series(np.zeros(len(df)))
            
            # 전처리 파이프라인 적용
            with st.spinner('데이터 변환 중...'):
                time.sleep(0.5)  # 로딩 효과를 위한 지연
                X_preprocessed = preprocessor.fit_transform(X)
            
            # 전처리 결과 요약
            st.markdown("**📊 전처리 결과:**")
            st.markdown(f"<div class='info-text'>- 원본 데이터 형태: {X.shape}<br>- 변환된 데이터 형태: {X_preprocessed.shape}</div>", unsafe_allow_html=True)
            
            # 전처리 모델 저장
            with open('models/preprocessor.pkl', 'wb') as f:
                pickle.dump(preprocessor, f)
            
            # 전처리된 데이터 저장
            processed_data = {
                'X_preprocessed': X_preprocessed,
                'X': X,
                'y': y,
                'feature_names': X.columns.tolist(),
                'categorical_cols': categorical_cols,
                'numerical_cols': numerical_cols
            }
            
            with open('data/processed_data.pkl', 'wb') as f:
                pickle.dump(processed_data, f)
            
            st.markdown("<div class='success-box'>✅ 전처리 데이터 저장 완료! 다음 단계(모델 학습)로 이동할 수 있습니다.</div>", unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"특성 변환 중 오류 발생: {e}")
        
    st.markdown("</div>", unsafe_allow_html=True)

# 데이터 시각화 섹션
st.markdown("<h2 class='section-header'>4. 전처리 데이터 시각화</h2>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    # 데이터프레임이 비어있는지 확인
    if df.empty:
        st.error("데이터가 로드되지 않았습니다. 먼저 데이터를 로드해주세요.")
    else:
        try:
            # 수치형/범주형 변수가 비어 있는지 확인
            if not numerical_cols:
                st.warning("수치형 변수가 없습니다.")
            else:
                # 수치형 변수의 분포 시각화
                st.markdown("**📊 수치형 변수 분포:**")
                
                selected_num_col = st.selectbox(
                    "시각화할 수치형 변수 선택:",
                    numerical_cols
                )
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=df[selected_num_col],
                    name=selected_num_col,
                    opacity=0.75,
                    marker=dict(color='#1976D2')
                ))
                
                fig.update_layout(
                    title=f"{selected_num_col} 분포",
                    xaxis_title=selected_num_col,
                    yaxis_title="빈도",
                    template="plotly_white",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            if not categorical_cols:
                st.warning("범주형 변수가 없습니다.")
            else:
                # 범주형 변수 분포
                st.markdown("**📊 범주형 변수 분포:**")
                
                selected_cat_col = st.selectbox(
                    "시각화할 범주형 변수 선택:",
                    categorical_cols
                )
                
                cat_counts = df[selected_cat_col].value_counts().reset_index()
                cat_counts.columns = [selected_cat_col, 'Count']
                
                fig = px.bar(
                    cat_counts, 
                    x=selected_cat_col, 
                    y='Count',
                    color='Count',
                    color_continuous_scale='Blues',
                    template="plotly_white"
                )
                
                fig.update_layout(
                    title=f"{selected_cat_col} 분포",
                    xaxis_title=selected_cat_col,
                    yaxis_title="빈도",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"데이터 시각화 중 오류 발생: {e}")
        
    st.markdown("</div>", unsafe_allow_html=True)

# 전처리 요약 섹션
st.markdown("<h2 class='section-header'>5. 전처리 요약</h2>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    st.markdown("**📋 전처리 단계 요약:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        1. **데이터 로드**: 
           - 텔레콤 고객 이탈 데이터셋 로드
           - 형태: 7,043행 × 21열
        
        2. **결측치 처리**:
           - 결측치 평균값으로 대체
           
        3. **데이터 타입 변환**:
           - TotalCharges를 숫자형으로 변환
        """)
    
    with col2:
        st.markdown("""
        4. **특성 변환**:
           - 수치형 변수: StandardScaler 적용
           - 범주형 변수: OneHotEncoder 적용
        
        5. **타겟 변수 인코딩**:
           - Churn: Yes → 1, No → 0
           
        6. **데이터 저장**:
           - 전처리 모델 및 변환된 데이터 저장
        """)
    
    # 다음 단계 안내
    st.markdown("<div class='info-text'>⏭️ <b>다음 단계</b>: '모델 학습' 페이지로 이동하여 전처리된 데이터를 바탕으로 머신러닝 모델을 학습해보세요.</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True) 