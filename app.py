import streamlit as st
import pandas as pd

# 페이지 설정 - 테마 개선 및 넓은 레이아웃
st.set_page_config(
    page_title="가입 고객 이탈 예측",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일 추가
st.markdown("""
<style>
    /* 전체 테마 색상 */
    :root {
        --main-color: #3498db;
        --accent-color: #2980b9;
        --background-color: #f8f9fa;
        --text-color: #2c3e50;
        --success-color: #2ecc71;
        --warning-color: #f39c12;
        --danger-color: #e74c3c;
    }
    
    /* 헤더 스타일링 */
    h1, h2, h3 {
        color: var(--text-color);
        padding-bottom: 0.3rem;
        border-bottom: 2px solid var(--main-color);
    }
    
    /* 카드 컴포넌트 */
    .card {
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* 데이터프레임 스타일링 */
    .dataframe {
        border-radius: 5px;
        overflow: hidden;
    }
    .dataframe th {
        background-color: var(--main-color);
        color: white;
    }
    .dataframe td {
        text-align: center;
    }
    
    /* 페이지 모듈 아이콘 */
    .page-icon {
        font-size: 2rem;
        margin-right: 0.5rem;
        color: var(--main-color);
    }
    
    /* 메트릭 컨테이너 */
    .metric-container {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: var(--main-color);
    }
    .metric-label {
        font-size: 0.9rem;
        color: var(--text-color);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideIn {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# 헤더 애니메이션 추가
st.markdown("""
<div style="text-align: center; padding: 1.5rem 0;">
    <h1 style="color: #3498db; font-size: 2.8rem; margin-bottom: 0.5rem; animation: fadeIn 1.5s;">
        📊 가입 고객 이탈 예측 시스템
    </h1>
    <p style="font-size: 1.2rem; color: #7f8c8d; animation: slideIn 1.8s;">
        데이터 분석부터 AI 예측까지, 한 눈에 확인하는 고객 이탈 솔루션
    </p>
</div>
""", unsafe_allow_html=True)

# 데이터 로드
@st.cache_data
def load_data():
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    return df

# 데이터 개요 표시
try:
    df = load_data()
    
    # 2개의 행으로 구성된 대시보드 레이아웃
    # 첫 번째 행: 주요 지표
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{df.shape[0]:,}</div>
            <div class="metric-label">총 고객 수</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        churn_count = df['Churn'].value_counts()
        churn_percent = 100 * churn_count / len(df)
        st.markdown(f"""
        <div class="metric-container" style="border-left: 4px solid #e74c3c;">
            <div class="metric-value" style="color: #e74c3c;">{churn_percent['Yes']:.1f}%</div>
            <div class="metric-label">이탈률</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container" style="border-left: 4px solid #2ecc71;">
            <div class="metric-value" style="color: #2ecc71;">{churn_count['No']:,}</div>
            <div class="metric-label">유지 고객 수</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container" style="border-left: 4px solid #e74c3c;">
            <div class="metric-value" style="color: #e74c3c;">{churn_count['Yes']:,}</div>
            <div class="metric-label">이탈 고객 수</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 두 번째 행: 데이터 미리보기 및 주요 기능 안내
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📋 데이터 미리보기")
        st.dataframe(df.head(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📌 주요 기능")
        
        # 카드형 기능 안내
        features = [
            {"icon": "🔍", "title": "데이터 탐색", "desc": "데이터의 분포와 특성을 시각적으로 확인"},
            {"icon": "🧹", "title": "데이터 전처리", "desc": "모델 학습을 위한 데이터 정제 과정"},
            {"icon": "🤖", "title": "모델 학습", "desc": "다양한 머신러닝 모델 훈련 및 성능 비교"},
            {"icon": "📊", "title": "모델 평가", "desc": "학습된 모델의 성능 평가 및 시각화"},
            {"icon": "🔮", "title": "이탈 예측", "desc": "새로운 고객 데이터로 이탈 가능성 예측"}
        ]
        
        for feature in features:
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <div style="font-size: 1.5rem; margin-right: 0.7rem; min-width: 30px; text-align: center;">{feature['icon']}</div>
                <div>
                    <div style="font-weight: bold; color: #3498db;">{feature['title']}</div>
                    <div style="font-size: 0.9rem; color: #7f8c8d;">{feature['desc']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 세 번째 행: 사용 안내 및 팁
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💡 시작하기")
    st.markdown("""
    1. 왼쪽 사이드바에서 원하는 페이지를 선택하세요.
    2. 데이터 탐색부터 시작하여 데이터의 특성을 파악하세요.
    3. 데이터 전처리를 통해 모델 학습에 적합한 형태로 변환하세요.
    4. 모델 학습 페이지에서 다양한 AI 모델을 훈련시키고 비교하세요.
    5. 모델 평가를 통해 각 모델의 성능을 자세히 분석하세요.
    6. 이탈 예측 페이지에서 새로운 고객 데이터로 이탈 가능성을 예측해보세요.
    
    <div style="background-color: #ebf5fb; border-left: 4px solid #3498db; padding: 0.7rem; margin-top: 1rem; border-radius: 5px;">
        <b>💡 TIP:</b> 모든 페이지에는 설명과 도움말이 포함되어 있어 쉽게 이용하실 수 있습니다.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.info("'data/WA_Fn-UseC_-Telco-Customer-Churn.csv' 파일이 있는지 확인해주세요.") 