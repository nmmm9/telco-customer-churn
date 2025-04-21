import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
import shap
import platform
import pickle
import time
import plotly.express as px
import plotly.graph_objects as go

# 한글 폰트 설정
system_os = platform.system()

if system_os == "Windows":
    # 윈도우의 경우 맑은 고딕 폰트 사용
    plt.rc('font', family='Malgun Gothic')
elif system_os == "Darwin":
    # macOS의 경우 애플고딕 폰트 사용
    plt.rc('font', family='AppleGothic')
else:
    # 리눅스 등 기타 OS의 경우 
    try:
        # 나눔고딕 폰트 사용 시도
        plt.rc('font', family='NanumGothic')
    except:
        # 사용 가능한 한글 폰트 확인
        fonts = fm.findSystemFonts()
        korean_fonts = [f for f in fonts if 'Gothic' in f or 'Batang' in f or 'Gulim' in f or 'Dotum' in f]
        if korean_fonts:
            plt.rc('font', family=fm.FontProperties(fname=korean_fonts[0]).get_name())
        else:
            st.warning("한글 폰트를 찾을 수 없습니다. 그래프에서 한글이 깨질 수 있습니다.")

# 그래프에서 마이너스 기호가 깨지는 것을 방지
plt.rc('axes', unicode_minus=False)

st.set_page_config(
    page_title="모델 학습",
    page_icon="🧠",
    layout="wide"
)

# CSS 스타일 추가
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
        text-align: center;
        background: linear-gradient(to right, #1E88E5, #42A5F5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin: 1.5rem 0 1rem 0;
    }
    
    .info-text {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
    }
    
    .card {
        padding: 1.5rem;
        border-radius: 0.7rem;
        background-color: #f8f9fa;
        box-shadow: 0 0.25rem 0.75rem rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .model-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.7rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin: 1.5rem 0;
        border-top: 4px solid #1E88E5;
    }
    
    .metric-box {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        background-color: #f5f9ff;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E88E5;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #616161;
        margin-top: 5px;
    }
    
    /* 애니메이션 헤더 */
    @keyframes gradientAnimation {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    
    .animated-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(270deg, #1E88E5, #1565C0, #0D47A1);
        background-size: 600% 600%;
        animation: gradientAnimation 6s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 애니메이션 헤더
st.markdown("<div class='animated-header'>🧠 모델 학습</div>", unsafe_allow_html=True)
st.markdown("<div class='info-text'>이 페이지에서는 전처리된 데이터를 사용하여 다양한 분류 모델을 학습하고 성능을 비교할 수 있습니다.</div>", unsafe_allow_html=True)

# 전처리된 데이터 로드
st.markdown("<h2 class='section-header'>1. 데이터 준비</h2>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    try:
        # 데이터 로드 애니메이션
        data_load_state = st.text('전처리된 데이터 로딩 중...')
        
        if not os.path.exists('data/processed_data.pkl'):
            st.markdown("<div class='warning-box'>⚠️ 전처리된 데이터를 찾을 수 없습니다. 먼저 '데이터 전처리' 페이지에서 데이터 전처리를 수행해주세요.</div>", unsafe_allow_html=True)
            
            # 간단한 샘플 데이터 생성하여 계속 진행
            st.markdown("<div class='info-text'>샘플 데이터로 계속 진행합니다.</div>", unsafe_allow_html=True)
            
            # 데이터 디렉토리 생성
            if not os.path.exists('data'):
                os.makedirs('data')
                
            # 모델 디렉토리 확인
            if not os.path.exists('models'):
                os.makedirs('models')
            
            # 간단한 샘플 데이터 생성
            n_samples = 100
            n_features = 10
            X_sample = np.random.rand(n_samples, n_features)
            y_sample = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
            
            feature_names = [f'feature_{i}' for i in range(n_features)]
            categorical_cols = [f'feature_{i}' for i in range(3)]
            numerical_cols = [f'feature_{i}' for i in range(3, n_features)]
            
            processed_data = {
                'X_preprocessed': X_sample,
                'X': pd.DataFrame(X_sample, columns=feature_names),
                'y': pd.Series(y_sample),
                'feature_names': feature_names,
                'categorical_cols': categorical_cols,
                'numerical_cols': numerical_cols
            }
            
            # 샘플 데이터 저장
            with open('data/processed_data.pkl', 'wb') as f:
                pickle.dump(processed_data, f)
                
            data_load_state.markdown("<div class='success-box'>✅ 샘플 데이터 생성 완료!</div>", unsafe_allow_html=True)
        else:
            with open('data/processed_data.pkl', 'rb') as f:
                with st.spinner('데이터 로딩 중...'):
                    time.sleep(0.5)  # 로딩 효과를 위한 지연
                    processed_data = pickle.load(f)
                
            data_load_state.markdown("<div class='success-box'>✅ 전처리된 데이터 로드 완료!</div>", unsafe_allow_html=True)
            
            # 데이터 정보 표시
            X_preprocessed = processed_data['X_preprocessed']
            y = processed_data['y']
            feature_names = processed_data['feature_names']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📊 데이터 형태:**")
                st.markdown(f"<div class='info-text'>샘플 수: {X_preprocessed.shape[0]:,}<br>특성 수: {X_preprocessed.shape[1]:,}</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("**🎯 클래스 분포:**")
                class_counts = pd.Series(y).value_counts()
                st.markdown(f"<div class='info-text'>이탈(1): {class_counts.get(1, 0):,}개 ({class_counts.get(1, 0)/len(y)*100:.1f}%)<br>유지(0): {class_counts.get(0, 0):,}개 ({class_counts.get(0, 0)/len(y)*100:.1f}%)</div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("**⚙️ 전처리 정보:**")
                st.markdown(f"<div class='info-text'>범주형 변수: {len(processed_data['categorical_cols'])}개<br>수치형 변수: {len(processed_data['numerical_cols'])}개</div>", unsafe_allow_html=True)
            
            # 시각화 - 클래스 분포
            fig = px.pie(
                values=class_counts.values, 
                names=['고객 유지', '고객 이탈'], 
                title='클래스 분포',
                color_discrete_sequence=['#66bb6a', '#ef5350'],
                hole=0.4
            )
            
            fig.update_layout(
                legend_title='클래스',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 데이터 분할 섹션
            st.markdown("**🔄 데이터 분할:**")
            
            test_size = st.slider('테스트 데이터 비율:', min_value=0.1, max_value=0.5, value=0.2, step=0.05, format='%.2f')
            random_state = st.number_input('랜덤 시드:', min_value=0, max_value=1000, value=42)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_preprocessed, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='info-text'>🔍 학습 데이터 크기:</div>", unsafe_allow_html=True)
                st.write(f"X_train: {X_train.shape}, y_train: {len(y_train)}")
            
            with col2:
                st.markdown("<div class='info-text'>🔍 테스트 데이터 크기:</div>", unsafe_allow_html=True)
                st.write(f"X_test: {X_test.shape}, y_test: {len(y_test)}")
            
            # 학습/테스트 데이터 저장
            train_test_data = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_names': feature_names
            }
            
            with open('data/train_test_data.pkl', 'wb') as f:
                pickle.dump(train_test_data, f)
            
            st.markdown("<div class='success-box'>✅ 학습/테스트 데이터 분할 완료!</div>", unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"데이터 준비 중 오류 발생: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# 모델 학습 섹션
st.markdown("<h2 class='section-header'>2. 모델 선택 및 학습</h2>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    # 모델 학습이 가능한지 확인
    if not os.path.exists('data/train_test_data.pkl'):
        st.markdown("<div class='warning-box'>⚠️ 학습/테스트 데이터를 찾을 수 없습니다. 먼저 데이터 준비 과정을 완료해주세요.</div>", unsafe_allow_html=True)
    else:
        try:
            # 학습/테스트 데이터 로드
            with open('data/train_test_data.pkl', 'rb') as f:
                train_test_data = pickle.load(f)
            
            X_train = train_test_data['X_train']
            X_test = train_test_data['X_test']
            y_train = train_test_data['y_train']
            y_test = train_test_data['y_test']
            
            # 모델과 결과를 저장할 변수 초기화
            models = {}
            results = {}
            
            # 모델 선택 폼
            st.markdown("<div class='info-text'>📊 학습할 모델을 선택하세요. 여러 모델을 선택하여 비교할 수 있습니다.</div>", unsafe_allow_html=True)
            
            # 모델 선택 탭 생성
            model_tabs = st.tabs(["기본 모델", "부스팅 모델", "앙상블 모델", "데이터 불균형 처리"])
            
            # 기본 모델 선택 탭
            with model_tabs[0]:
                st.subheader("기본 분류 모델")
                
                # 기본 모델 선택 옵션
                col1, col2 = st.columns(2)
                with col1:
                    use_rf = st.checkbox("랜덤 포레스트 (RandomForest)", value=True)
                    use_lr = st.checkbox("로지스틱 회귀 (LogisticRegression)")
                
                with col2:
                    use_svm = st.checkbox("서포트 벡터 머신 (SVM)")

                # 최소 하나의 모델이 선택되었는지 확인
                if not any([use_rf, use_lr, use_svm]):
                    st.warning("⚠️ 최소 하나 이상의 모델을 선택해주세요.")
                    use_rf = True  # 기본값으로 랜덤 포레스트 설정
                    st.info("기본적으로 랜덤 포레스트 모델이 선택되었습니다.")

                # 랜덤 포레스트 하이퍼파라미터
                if use_rf:
                    st.markdown("##### 랜덤 포레스트 하이퍼파라미터")
                    rf_n_estimators = st.slider("RF: 트리 개수", 10, 200, 100, 10)
                    rf_max_depth = st.slider("RF: 최대 깊이", 2, 20, 10, 1)
                
                use_svm = st.checkbox("SVM(Support Vector Machine)", value=False)
                if use_svm:
                    svm_C = st.slider("SVM: 규제 강도(C)", 0.1, 10.0, 1.0, 0.1, format="%.2f")
                    svm_kernel = st.selectbox("SVM: 커널 함수", ['linear', 'rbf', 'poly'])
                    svm_probability = st.checkbox("SVM: 확률 예측 활성화", value=True)
            
            # 부스팅 모델 선택 탭
            with model_tabs[1]:
                col1, col2 = st.columns(2)
                
                with col1:
                    use_gb = st.checkbox("그래디언트 부스팅", value=False)
                    if use_gb:
                        gb_n_estimators = st.slider("GB: 트리 개수", 10, 200, 100, 10)
                        gb_learning_rate = st.slider("GB: 학습률", 0.01, 0.3, 0.1, 0.01, format="%.2f")
                    
                    use_hgb = st.checkbox("히스토그램 기반 그래디언트 부스팅", value=False)
                    if use_hgb:
                        hgb_max_iter = st.slider("HGB: 최대 반복 횟수", 10, 500, 100, 10)
                        hgb_learning_rate = st.slider("HGB: 학습률", 0.01, 0.3, 0.1, 0.01, format="%.2f")
                
                with col2:
                    use_xgb = st.checkbox("XGBoost", value=False)
                    if use_xgb:
                        xgb_n_estimators = st.slider("XGB: 트리 개수", 10, 200, 100, 10)
                        xgb_learning_rate = st.slider("XGB: 학습률", 0.01, 0.3, 0.1, 0.01, format="%.2f")
                        xgb_max_depth = st.slider("XGB: 최대 깊이", 3, 10, 6, 1)
                    
                    use_lgbm = st.checkbox("LightGBM", value=False)
                    if use_lgbm:
                        lgbm_n_estimators = st.slider("LGBM: 트리 개수", 10, 200, 100, 10)
                        lgbm_learning_rate = st.slider("LGBM: 학습률", 0.01, 0.3, 0.1, 0.01, format="%.2f")
                        lgbm_num_leaves = st.slider("LGBM: 잎 노드 수", 10, 100, 31, 1)
            
            # 앙상블 모델 선택 탭
            with model_tabs[2]:
                use_voting = st.checkbox("보팅 앙상블 (Voting)", value=False)
                if use_voting:
                    voting_estimators = st.multiselect(
                        "보팅에 사용할 기본 모델 선택:",
                        ["RandomForest", "LogisticRegression", "SVM", "GradientBoosting", "XGBoost", "LightGBM"],
                        default=["RandomForest", "LogisticRegression", "GradientBoosting"]
                    )
                    voting_type = st.radio("보팅 방식:", ["hard", "soft"], index=1, 
                                        help="hard: 다수결 투표, soft: 각 분류기의 확률 평균 (soft가 일반적으로 더 좋음)")
                
                use_stacking = st.checkbox("스태킹 앙상블 (Stacking)", value=False)
                if use_stacking:
                    stacking_estimators = st.multiselect(
                        "스태킹의 기본 모델 선택:",
                        ["RandomForest", "LogisticRegression", "SVM", "GradientBoosting", "XGBoost", "LightGBM"],
                        default=["RandomForest", "GradientBoosting", "LogisticRegression"]
                    )
                    stacking_final = st.selectbox(
                        "스태킹의 최종 모델 선택:",
                        ["LogisticRegression", "RandomForest", "GradientBoosting"],
                        index=0
                    )
            
            # 데이터 불균형 처리 탭
            with model_tabs[3]:
                handle_imbalance = st.checkbox("데이터 불균형 처리 적용", value=False)
                if handle_imbalance:
                    imbalance_method = st.radio(
                        "불균형 처리 방법:",
                        ["SMOTE", "랜덤 오버샘플링", "랜덤 언더샘플링"],
                        index=0,
                        help="SMOTE: 소수 클래스 합성, 오버샘플링: 소수 클래스 복제, 언더샘플링: 다수 클래스 제거"
                    )
                    
                    if imbalance_method == "SMOTE":
                        smote_k = st.slider("SMOTE: 최근접 이웃 수(k)", 1, 10, 5, 1)
                    
                    st.info("선택한 불균형 처리 방법이 학습 데이터에 적용됩니다.")
            
            # 학습 버튼
            if st.button("모델 학습 시작", key="train_button"):
                # 프로그레스 바 및 상태 텍스트 준비
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 데이터 불균형 처리 적용
                X_train_processed = X_train.copy()
                y_train_processed = y_train.copy()
                
                if handle_imbalance:
                    status_text.text("데이터 불균형 처리 중...")
                    
                    try:
                        if imbalance_method == "SMOTE":
                            smote = SMOTE(random_state=42, k_neighbors=smote_k)
                            X_train_processed, y_train_processed = smote.fit_resample(X_train, y_train)
                            st.info(f"SMOTE 적용 결과: {pd.Series(y_train_processed).value_counts().to_dict()}")
                        
                        elif imbalance_method == "랜덤 오버샘플링":
                            ros = RandomOverSampler(random_state=42)
                            X_train_processed, y_train_processed = ros.fit_resample(X_train, y_train)
                            st.info(f"랜덤 오버샘플링 적용 결과: {pd.Series(y_train_processed).value_counts().to_dict()}")
                        
                        elif imbalance_method == "랜덤 언더샘플링":
                            rus = RandomUnderSampler(random_state=42)
                            X_train_processed, y_train_processed = rus.fit_resample(X_train, y_train)
                            st.info(f"랜덤 언더샘플링 적용 결과: {pd.Series(y_train_processed).value_counts().to_dict()}")
                    
                    except Exception as e:
                        st.error(f"데이터 불균형 처리 중 오류 발생: {e}")
                        st.warning("불균형 처리를 건너뛰고 원본 데이터로 계속 진행합니다.")
                
                # 모델 개수 계산
                total_models = sum([
                    use_rf, use_lr, use_svm, use_gb, use_hgb, use_xgb, use_lgbm, 
                    use_voting, use_stacking
                ])
                
                if total_models == 0:
                    st.error("최소한 하나의 모델을 선택해야 합니다!")
                    # 기본적으로 RandomForest 선택
                    use_rf = True
                    rf_n_estimators = 100
                    rf_max_depth = 10
                    st.info("기본 모델로 랜덤 포레스트가 자동으로 선택되었습니다.")
                    total_models = 1  # 모델 수 업데이트
                
                progress_step = 1.0 / total_models
                progress_value = 0.0
                
                # 개별 모델 학습
                # 랜덤 포레스트
                if use_rf:
                    status_text.text("랜덤 포레스트 모델 학습 중...")
                    rf_model = RandomForestClassifier(
                        n_estimators=rf_n_estimators,
                        max_depth=rf_max_depth,
                        random_state=42,
                        n_jobs=-1
                    )
                    
                    with st.spinner('랜덤 포레스트 학습 중...'):
                        rf_model.fit(X_train_processed, y_train_processed)
                    
                    models['RandomForest'] = rf_model
                    progress_value += progress_step
                    progress_bar.progress(progress_value)
                
                # 로지스틱 회귀
                if use_lr:
                    status_text.text("로지스틱 회귀 모델 학습 중...")
                    lr_model = LogisticRegression(
                        C=lr_C,
                        solver=lr_solver,
                        random_state=42,
                        max_iter=1000
                    )
                    
                    with st.spinner('로지스틱 회귀 학습 중...'):
                        lr_model.fit(X_train_processed, y_train_processed)
                    
                    models['LogisticRegression'] = lr_model
                    progress_value += progress_step
                    progress_bar.progress(progress_value)
                
                # SVM
                if use_svm:
                    status_text.text("SVM 모델 학습 중...")
                    svm_model = SVC(
                        C=svm_C,
                        kernel=svm_kernel,
                        probability=svm_probability,
                        random_state=42
                    )
                    
                    with st.spinner('SVM 학습 중...'):
                        svm_model.fit(X_train_processed, y_train_processed)
                    
                    models['SVM'] = svm_model
                    progress_value += progress_step
                    progress_bar.progress(progress_value)
                
                # 그래디언트 부스팅
                if use_gb:
                    status_text.text("그래디언트 부스팅 모델 학습 중...")
                    gb_model = GradientBoostingClassifier(
                        n_estimators=gb_n_estimators,
                        learning_rate=gb_learning_rate,
                        random_state=42
                    )
                    
                    with st.spinner('그래디언트 부스팅 학습 중...'):
                        gb_model.fit(X_train_processed, y_train_processed)
                    
                    models['GradientBoosting'] = gb_model
                    progress_value += progress_step
                    progress_bar.progress(progress_value)
                
                # 히스토그램 기반 그래디언트 부스팅
                if use_hgb:
                    status_text.text("히스토그램 기반 그래디언트 부스팅 모델 학습 중...")
                    hgb_model = HistGradientBoostingClassifier(
                        max_iter=hgb_max_iter,
                        learning_rate=hgb_learning_rate,
                        random_state=42
                    )
                    
                    with st.spinner('히스토그램 기반 그래디언트 부스팅 학습 중...'):
                        hgb_model.fit(X_train_processed, y_train_processed)
                    
                    models['HistGradientBoosting'] = hgb_model
                    progress_value += progress_step
                    progress_bar.progress(progress_value)
                
                # XGBoost
                if use_xgb:
                    status_text.text("XGBoost 모델 학습 중...")
                    xgb_model = XGBClassifier(
                        n_estimators=xgb_n_estimators,
                        learning_rate=xgb_learning_rate,
                        max_depth=xgb_max_depth,
                        random_state=42,
                        use_label_encoder=False,
                        eval_metric='logloss'
                    )
                    
                    with st.spinner('XGBoost 학습 중...'):
                        xgb_model.fit(X_train_processed, y_train_processed)
                    
                    models['XGBoost'] = xgb_model
                    progress_value += progress_step
                    progress_bar.progress(progress_value)
                
                # LightGBM
                if use_lgbm:
                    status_text.text("LightGBM 모델 학습 중...")
                    lgbm_model = LGBMClassifier(
                        n_estimators=lgbm_n_estimators,
                        learning_rate=lgbm_learning_rate,
                        num_leaves=lgbm_num_leaves,
                        random_state=42
                    )
                    
                    with st.spinner('LightGBM 학습 중...'):
                        lgbm_model.fit(X_train_processed, y_train_processed)
                    
                    models['LightGBM'] = lgbm_model
                    progress_value += progress_step
                    progress_bar.progress(progress_value)
                
                # 보팅 앙상블
                if use_voting:
                    status_text.text("보팅 앙상블 모델 구성 중...")
                    
                    # 선택된 기본 모델들이 실제로 학습되었는지 확인
                    available_models = list(models.keys())
                    valid_estimators = []
                    for model_name in voting_estimators:
                        if model_name in available_models:
                            valid_estimators.append((model_name, models[model_name]))
                        else:
                            st.warning(f"{model_name} 모델이 학습되지 않아 앙상블에서 제외됩니다.")
                    
                    # 유효한 모델이 최소 2개 이상 있는지 확인
                    if len(valid_estimators) < 2:
                        st.error("보팅 앙상블을 위해서는 최소 2개 이상의 모델이 필요합니다.")
                        # 기본 모델이 있는지 확인하고 없으면 추가
                        if 'RandomForest' not in models and use_rf == False:
                            st.info("기본 모델(RandomForest)을 자동으로 추가합니다.")
                            rf_model = RandomForestClassifier(
                                n_estimators=100,
                                max_depth=10,
                                random_state=42,
                                n_jobs=-1
                            )
                            rf_model.fit(X_train_processed, y_train_processed)
                            models['RandomForest'] = rf_model
                            valid_estimators.append(('RandomForest', rf_model))
                        
                        # 로지스틱 회귀 모델 추가
                        if 'LogisticRegression' not in models and use_lr == False:
                            st.info("기본 모델(LogisticRegression)을 자동으로 추가합니다.")
                            lr_model = LogisticRegression(
                                C=1.0,
                                solver='lbfgs',
                                random_state=42,
                                max_iter=1000
                            )
                            lr_model.fit(X_train_processed, y_train_processed)
                            models['LogisticRegression'] = lr_model
                            valid_estimators.append(('LogisticRegression', lr_model))
                    
                    if len(valid_estimators) >= 2:
                        voting_model = VotingClassifier(
                            estimators=valid_estimators,
                            voting=voting_type
                        )
                        
                        with st.spinner('보팅 앙상블 학습 중...'):
                            voting_model.fit(X_train_processed, y_train_processed)
                        
                        models['Voting'] = voting_model
                        progress_value += progress_step
                        progress_bar.progress(progress_value)
                    else:
                        st.error("유효한 모델이 부족하여 보팅 앙상블을 건너뜁니다.")
                
                # 스태킹 앙상블
                if use_stacking:
                    status_text.text("스태킹 앙상블 모델 구성 중...")
                    
                    # 선택된 기본 모델들이 실제로 학습되었는지 확인
                    available_models = list(models.keys())
                    valid_estimators = []
                    for model_name in stacking_estimators:
                        if model_name in available_models:
                            valid_estimators.append((model_name, models[model_name]))
                        else:
                            st.warning(f"{model_name} 모델이 학습되지 않아 스태킹에서 제외됩니다.")
                    
                    # 유효한 모델이 최소 2개 이상 있는지 확인
                    if len(valid_estimators) < 2:
                        st.error("스태킹 앙상블을 위해서는 최소 2개 이상의 모델이 필요합니다.")
                        # 기본 모델이 있는지 확인하고 없으면 추가
                        if 'RandomForest' not in models and use_rf == False:
                            st.info("기본 모델(RandomForest)을 자동으로 추가합니다.")
                            rf_model = RandomForestClassifier(
                                n_estimators=100,
                                max_depth=10,
                                random_state=42,
                                n_jobs=-1
                            )
                            rf_model.fit(X_train_processed, y_train_processed)
                            models['RandomForest'] = rf_model
                            valid_estimators.append(('RandomForest', rf_model))
                        
                        # 로지스틱 회귀 모델 추가
                        if 'LogisticRegression' not in models and use_lr == False:
                            st.info("기본 모델(LogisticRegression)을 자동으로 추가합니다.")
                            lr_model = LogisticRegression(
                                C=1.0,
                                solver='lbfgs',
                                random_state=42,
                                max_iter=1000
                            )
                            lr_model.fit(X_train_processed, y_train_processed)
                            models['LogisticRegression'] = lr_model
                            valid_estimators.append(('LogisticRegression', lr_model))
                    
                    # 최종 메타 모델 확인
                    if stacking_final in available_models or stacking_final in [est[0] for est in valid_estimators]:
                        if stacking_final == "LogisticRegression" and "LogisticRegression" not in available_models:
                            final_estimator = LogisticRegression(C=1.0, random_state=42)
                        elif stacking_final == "RandomForest" and "RandomForest" not in available_models:
                            final_estimator = RandomForestClassifier(n_estimators=50, random_state=42)
                        elif stacking_final == "GradientBoosting" and "GradientBoosting" not in available_models:
                            final_estimator = GradientBoostingClassifier(n_estimators=50, random_state=42)
                        else:
                            final_estimator = models.get(stacking_final, LogisticRegression(C=1.0, random_state=42))
                    else:
                        st.warning(f"{stacking_final} 모델이 사용할 수 없어 LogisticRegression으로 대체합니다.")
                        final_estimator = LogisticRegression(C=1.0, random_state=42)
                    
                    if len(valid_estimators) >= 2:
                        try:
                            stacking_model = StackingClassifier(
                                estimators=valid_estimators,
                                final_estimator=final_estimator,
                                cv=5,
                                n_jobs=-1
                            )
                            
                            with st.spinner('스태킹 앙상블 학습 중...'):
                                stacking_model.fit(X_train_processed, y_train_processed)
                            
                            models['Stacking'] = stacking_model
                            progress_value += progress_step
                            progress_bar.progress(progress_value)
                        except Exception as e:
                            st.error(f"스태킹 앙상블 학습 중 오류 발생: {e}")
                    else:
                        st.error("유효한 모델이 부족하여 스태킹 앙상블을 건너뜁니다.")
            
                # 모델 평가
                status_text.text("모델 평가 중...")
                
                for model_name, model in models.items():
                    # 예측
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]
                    
                    # 평가 지표 계산
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    roc_auc = roc_auc_score(y_test, y_prob)
                    
                    results[model_name] = {
                        'model': model,
                        'y_pred': y_pred,
                        'y_prob': y_prob,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'roc_auc': roc_auc
                    }
            
                # 모델 저장 디렉토리 확인
                if not os.path.exists('models'):
                    os.makedirs('models')
                
                # 모델 저장
                with open('models/trained_models.pkl', 'wb') as f:
                    pickle.dump(models, f)
                
                # 기존 모델 결과와 현재 결과 합치기
                existing_results = {}
                if os.path.exists('models/all_model_results.pkl'):
                    try:
                        with open('models/all_model_results.pkl', 'rb') as f:
                            existing_results = pickle.load(f)
                    except Exception as e:
                        st.warning(f"기존 모델 결과를 불러오는 중 오류가 발생했습니다: {e}")
                
                # 새 결과를 기존 결과와 병합 (같은 모델 이름이면 새 결과로 덮어씀)
                combined_results = {**existing_results, **results}
                
                # 결과 저장
                with open('models/model_results.pkl', 'wb') as f:
                    pickle.dump(combined_results, f)
                
                # 4페이지에서 사용할 모델 결과 정보 저장
                with open('models/all_model_results.pkl', 'wb') as f:
                    pickle.dump(combined_results, f)
                
                status_text.markdown("<div class='success-box'>✅ 모델 학습 및 평가 완료!</div>", unsafe_allow_html=True)
                
                # 결과 표시
                st.markdown("### 📊 모델 성능 비교")
                
                # 표 형식으로 모든 모델 결과 표시 (기존 모델 포함)
                result_df = pd.DataFrame({
                    '모델': list(combined_results.keys()),
                    '정확도': [res['accuracy'] for res in combined_results.values()],
                    '정밀도': [res['precision'] for res in combined_results.values()],
                    '재현율': [res['recall'] for res in combined_results.values()],
                    'F1 점수': [res['f1'] for res in combined_results.values()],
                    'ROC AUC': [res['roc_auc'] for res in combined_results.values()]
                })
                
                # CSV로 저장
                result_df.to_csv('models/model_results.csv', index=False)
                
                st.dataframe(result_df.style.highlight_max(axis=0, subset=['정확도', '정밀도', '재현율', 'F1 점수', 'ROC AUC']), use_container_width=True)
                
                # 각 모델에 대한 상세 결과 표시
                for model_name, result in combined_results.items():
                    st.markdown(f"<div class='model-card'>", unsafe_allow_html=True)
                    st.markdown(f"#### 📈 {model_name} 모델 성능")
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.markdown(f"<div class='metric-box'><div class='metric-value'>{result['accuracy']:.3f}</div><div class='metric-label'>정확도</div></div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"<div class='metric-box'><div class='metric-value'>{result['precision']:.3f}</div><div class='metric-label'>정밀도</div></div>", unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"<div class='metric-box'><div class='metric-value'>{result['recall']:.3f}</div><div class='metric-label'>재현율</div></div>", unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(f"<div class='metric-box'><div class='metric-value'>{result['f1']:.3f}</div><div class='metric-label'>F1 점수</div></div>", unsafe_allow_html=True)
                    
                    with col5:
                        st.markdown(f"<div class='metric-box'><div class='metric-value'>{result['roc_auc']:.3f}</div><div class='metric-label'>ROC AUC</div></div>", unsafe_allow_html=True)
                    
                    # 혼동 행렬 및 ROC 곡선
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 혼동 행렬
                        cm = confusion_matrix(y_test, result['y_pred'])
                        fig, ax = plt.subplots(figsize=(5, 4))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
                        ax.set_xlabel('예측')
                        ax.set_ylabel('실제')
                        ax.set_title('혼동 행렬')
                        st.pyplot(fig)
                    
                    with col2:
                        # ROC 곡선
                        fpr, tpr, _ = roc_curve(y_test, result['y_prob'])
                        fig = px.area(
                            x=fpr, y=tpr,
                            title=f'ROC 곡선 (AUC = {result["roc_auc"]:.3f})',
                            labels=dict(x='False Positive Rate', y='True Positive Rate'),
                            width=500, height=300
                        )
                        fig.add_shape(
                            type='line', line=dict(dash='dash'),
                            x0=0, x1=1, y0=0, y1=1
                        )
                        fig.update_layout(template='plotly_white')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # 최고 성능 모델 선택
                best_model_name = result_df.iloc[result_df['F1 점수'].argmax()]['모델']
                best_model = models[best_model_name]
                
                # 최고 성능 모델 저장
                with open('models/best_model.pkl', 'wb') as f:
                    pickle.dump({
                        'model': best_model,
                        'model_name': best_model_name
                    }, f)
                
                # 추가 - X_processed 크기를 확인하여 feature_names.csv 업데이트
                if hasattr(X_train_processed, 'shape'):
                    num_features = X_train_processed.shape[1]
                    # 특성 이름 파일 생성 (항상 새로 생성하도록 수정)
                    new_feature_names = []
                    if 'feature_names' in train_test_data and train_test_data['feature_names'] is not None:
                        # 원본 특성 이름이 있는 경우 사용
                        feature_list = train_test_data['feature_names']
                        if len(feature_list) == num_features:
                            new_feature_names = feature_list
                        else:
                            # 특성 수가 다르면 새로 생성
                            new_feature_names = [f'feature_{i}' for i in range(num_features)]
                    else:
                        # 특성 이름이 없는 경우 새로 생성
                        new_feature_names = [f'feature_{i}' for i in range(num_features)]
                    
                    # 항상 models 디렉토리에 feature_names.csv 파일 저장
                    if not os.path.exists('models'):
                        os.makedirs('models')
                    
                    pd.DataFrame({'feature_names': new_feature_names}).to_csv('models/feature_names.csv', index=False)
                    st.success(f"✅ 특성 이름 파일이 생성되었습니다. (특성 수: {num_features}개)")
                
                # 모델에 필요한 정보 저장
                np.save('models/X_processed.npy', X_train_processed)
                
                # 테스트 데이터도 저장
                np.save('models/X_test.npy', X_test)
                np.save('models/y_test.npy', y_test)
                
                st.markdown(f"<div class='success-box'>✅ 최고 성능 모델 ({best_model_name})이 저장되었습니다!</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='info-text'>모델 결과가 '모델 평가' 페이지에서 확인할 수 있도록 저장되었습니다.</div>", unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"모델 학습 중 오류 발생: {e}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# 학습된 모델 결과 섹션 제거 - 위에서 이미 표시함

# 나머지 코드는 삭제합니다. 