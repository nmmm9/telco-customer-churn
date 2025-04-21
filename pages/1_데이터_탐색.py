import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 페이지 설정
st.set_page_config(
    page_title="데이터 탐색",
    page_icon="🔍",
    layout="wide"
)

# CSS 스타일 추가
st.markdown("""
<style>
    /* 카드 컴포넌트 */
    .card {
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* 헤더 스타일 */
    h1, h2, h3 {
        color: #2c3e50;
        padding-bottom: 0.3rem;
    }
    h1 {
        border-bottom: 2px solid #3498db;
    }
    
    /* 데이터 인사이트 박스 */
    .insight-box {
        padding: 1rem;
        background-color: #f8f9fa;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    /* 탭 스타일링 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        background-color: #f1f1f1;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# 애니메이션 헤더
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 style="font-size: 2.5rem; animation: fade-in 1.5s;">
        🔍 데이터 탐색
    </h1>
    <p style="font-size: 1.2rem; color: #7f8c8d; animation: slide-up 1.8s;">
        고객 이탈 데이터의 심층 분석 및 시각화
    </p>
</div>
<style>
    @keyframes fade-in {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slide-up {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# 데이터 로드
@st.cache_data
def load_data():
    df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    # TotalCharges를 숫자형으로 변환
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df

try:
    df = load_data()
    
    # 데이터 개요 카드
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # 데이터 요약 메트릭
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 고객 수", f"{df.shape[0]:,}명")
    
    with col2:
        churn_percent = 100 * df['Churn'].value_counts(normalize=True)['Yes']
        st.metric("이탈률", f"{churn_percent:.1f}%")
    
    with col3:
        avg_tenure = df['tenure'].mean()
        st.metric("평균 이용 기간", f"{avg_tenure:.1f}개월")
    
    with col4:
        avg_monthly = df['MonthlyCharges'].mean()
        st.metric("평균 월 청구액", f"${avg_monthly:.2f}")
    
    # 데이터 개요 확장 패널
    with st.expander("데이터 세부 정보 보기", expanded=False):
        st.write("#### 데이터 구조")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**데이터셋 크기:**", df.shape)
            st.write("**고유 고객 ID 수:**", df['customerID'].nunique())
            
            # 데이터 타입 요약
            dtype_counts = df.dtypes.value_counts().to_dict()
            st.write("**데이터 타입 분포:**")
            for dtype, count in dtype_counts.items():
                st.write(f"- {dtype}: {count}개 컬럼")
        
        with col2:
            # 결측치 정보
            missing_values = df.isnull().sum()
            missing_cols = missing_values[missing_values > 0]
            
            if not missing_cols.empty:
                st.write("**결측치가 있는 컬럼:**")
                for col, count in missing_cols.items():
                    st.write(f"- {col}: {count}개 ({100*count/len(df):.2f}%)")
            else:
                st.write("**결측치 없음**")
        
        # 데이터 미리보기
        st.write("#### 데이터 미리보기")
        st.dataframe(df.head(), use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 데이터 인사이트 카드
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 주요 인사이트")
    
    # 인사이트 박스 - 핵심 정보 제공
    st.markdown("""
    <div class="insight-box">
        <h4 style="margin-top: 0;">💡 이탈 고객 특성 요약</h4>
        <ul>
            <li>월별 계약보다 <b>계약 기간이 짧을수록</b> 이탈률이 높습니다.</li>
            <li>전자 청구서를 사용하지 않는 고객이 이탈할 확률이 더 높습니다.</li>
            <li>인터넷 서비스 중 <b>Fiber Optic</b> 사용자의 이탈률이 가장 높습니다.</li>
            <li>추가 서비스(온라인 보안, 기기 보호 등)를 사용하지 않는 고객의 이탈률이 높습니다.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 데이터 시각화 탭
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 데이터 시각화")
    
    tabs = st.tabs(["📊 이탈 분석", "👥 고객 특성", "💰 요금 분석", "📱 서비스 분석"])
    
    with tabs[0]:
        st.write("#### 이탈 고객 특성 분석")
        
        # 이탈률 개요 차트
        fig = px.pie(df, names='Churn', title='고객 이탈 비율',
                     color='Churn', 
                     color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'},
                     hole=0.4)
        fig.update_layout(
            legend_title="이탈 여부",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 계약 기간별 이탈률
        contract_churn = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()
        contract_churn = contract_churn.reset_index()
        
        # 인터랙티브 바 차트
        fig = px.bar(
            contract_churn, x='Contract', y='Yes', 
            title="계약 형태별 이탈률",
            labels={'Yes': '이탈률', 'Contract': '계약 형태'},
            color_discrete_sequence=['#e74c3c']
        )
        fig.update_layout(
            xaxis_title="계약 형태",
            yaxis_title="이탈률",
            yaxis_tickformat='.0%',
            hovermode="x",
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 이탈에 영향을 미치는 상위 요인
        col1, col2 = st.columns(2)
        
        with col1:
            # 인터넷 서비스별 이탈률
            internet_churn = df.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack()
            internet_churn = internet_churn * 100  # 퍼센트로 변환
            
            fig = px.bar(
                internet_churn.reset_index(), x='InternetService', y='Yes',
                title="인터넷 서비스별 이탈률",
                labels={'Yes': '이탈률 (%)', 'InternetService': '인터넷 서비스'},
                color_discrete_sequence=['#e74c3c']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 지불 방법별 이탈률
            payment_churn = df.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack()
            payment_churn = payment_churn * 100  # 퍼센트로 변환
            
            fig = px.bar(
                payment_churn.reset_index(), x='PaymentMethod', y='Yes',
                title="결제 방법별 이탈률",
                labels={'Yes': '이탈률 (%)', 'PaymentMethod': '결제 방법'},
                color_discrete_sequence=['#e74c3c']
            )
            fig.update_layout(
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 이용 기간별 이탈 분포
        fig = px.histogram(
            df, x='tenure', color='Churn',
            marginal='box',
            title="이용 기간별 이탈 분포",
            labels={'tenure': '이용 기간(개월)', 'count': '고객 수'},
            color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'},
            opacity=0.7,
            barmode='overlay'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        st.write("#### 고객 인구통계 특성")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 성별에 따른 이탈 분포
            gender_churn = df.groupby(['gender', 'Churn']).size().unstack()
            gender_churn['이탈률'] = gender_churn['Yes'] / (gender_churn['Yes'] + gender_churn['No'])
            
            fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "bar"}]],
                               subplot_titles=("성별 분포", "성별 이탈률"))
            
            # 1. 성별 분포 - 파이 차트
            fig.add_trace(
                go.Pie(
                    labels=df['gender'].value_counts().index,
                    values=df['gender'].value_counts().values,
                    hole=0.4,
                    marker_colors=['#3498db', '#e74c3c']
                ),
                row=1, col=1
            )
            
            # 2. 성별 이탈률 - 바 차트
            fig.add_trace(
                go.Bar(
                    x=gender_churn.index,
                    y=gender_churn['이탈률'],
                    marker_color='#e74c3c'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                height=400,
                title_text="성별 분포 및 이탈률",
                showlegend=False
            )
            
            # y축을 퍼센트로 표시
            fig.update_yaxes(tickformat='.0%', row=1, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 노인 여부에 따른 이탈률
            senior_churn = df.groupby(['SeniorCitizen', 'Churn']).size().unstack()
            senior_churn.index = ['비노인', '노인']  # 0, 1 대신 명확한 레이블 사용
            senior_churn['이탈률'] = senior_churn['Yes'] / (senior_churn['Yes'] + senior_churn['No'])
            
            fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "bar"}]],
                               subplot_titles=("노인 여부 분포", "노인 여부에 따른 이탈률"))
            
            # 1. 노인 여부 분포 - 파이 차트
            senior_counts = df['SeniorCitizen'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=['비노인', '노인'],
                    values=[senior_counts[0], senior_counts[1]],
                    hole=0.4,
                    marker_colors=['#3498db', '#e74c3c']
                ),
                row=1, col=1
            )
            
            # 2. 노인 여부 이탈률 - 바 차트
            fig.add_trace(
                go.Bar(
                    x=senior_churn.index,
                    y=senior_churn['이탈률'],
                    marker_color='#e74c3c'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                height=400,
                title_text="노인 여부 분포 및 이탈률",
                showlegend=False
            )
            
            # y축을 퍼센트로 표시
            fig.update_yaxes(tickformat='.0%', row=1, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 가족 관련 특성
        st.write("#### 가족 관련 특성")
        
        # 파트너, 부양가족 여부에 따른 이탈률
        family_cols = ['Partner', 'Dependents']
        
        family_data = []
        for col in family_cols:
            group = df.groupby([col, 'Churn']).size().unstack()
            group['이탈률'] = group['Yes'] / (group['Yes'] + group['No'])
            group = group.reset_index()
            group['특성'] = col
            family_data.append(group)
        
        family_df = pd.concat(family_data)
        
        # 특성별 이탈률 시각화
        fig = px.bar(
            family_df, x=family_df[family_cols[0]], y='이탈률', color='특성',
            facet_col='특성', title="가족 관련 특성별 이탈률",
            labels={family_cols[0]: '여부', '이탈률': '이탈률', '특성': '특성'},
            color_discrete_sequence=['#3498db', '#e74c3c'],
            barmode='group',
            category_orders={'특성': family_cols}
        )
        fig.update_layout(
            xaxis_title="여부",
            yaxis_title="이탈률",
            yaxis_tickformat='.0%',
        )
        
        # facet 타이틀 업데이트
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        fig.update_annotations(font_size=14)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.write("#### 요금 분석")
        
        # 월별 요금 분포
        fig = px.histogram(
            df, x='MonthlyCharges', color='Churn',
            title="월별 요금 분포",
            labels={'MonthlyCharges': '월별 요금($)', 'count': '고객 수'},
            color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'},
            marginal='box',
            opacity=0.7,
            barmode='overlay'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 이용 기간과 월별 요금의 관계
        st.write("#### 이용 기간과 월별 요금의 관계")
        
        # 산점도 + 추세선
        fig = px.scatter(
            df, x='tenure', y='MonthlyCharges', color='Churn',
            title="이용 기간과 월별 요금의 관계",
            labels={'tenure': '이용 기간(개월)', 'MonthlyCharges': '월별 요금($)'},
            color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'},
            opacity=0.7,
            trendline='ols'  # 추세선 추가
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 지불 방법별 월별 요금 분포
        fig = px.box(
            df, x='PaymentMethod', y='MonthlyCharges', color='Churn',
            title="지불 방법별 월별 요금 분포",
            labels={
                'PaymentMethod': '지불 방법', 
                'MonthlyCharges': '월별 요금($)'
            },
            color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'}
        )
        fig.update_layout(
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        st.write("#### 서비스 분석")
        
        # 인터넷 서비스별 분포
        fig = px.pie(
            df, names='InternetService',
            title="인터넷 서비스 분포",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hole=0.4
        )
        fig.update_layout(
            legend_title="인터넷 서비스",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 부가 서비스 이용 현황
        service_cols = [
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        # 부가 서비스별 이탈률
        service_data = []
        for col in service_cols:
            group = df.groupby([col, 'Churn']).size().unstack()
            total = group.sum(axis=1)
            group['이탈률'] = group['Yes'] / total
            group = group.reset_index()
            group['서비스'] = col
            service_data.append(group)
        
        service_df = pd.concat(service_data)
        
        # 서비스 가입 여부에 따른 이탈률
        fig = px.bar(
            service_df[service_df[service_cols[0]].isin(['Yes', 'No'])],
            x=service_cols[0], y='이탈률', color='서비스', 
            facet_col='서비스', facet_col_wrap=3,
            title="부가 서비스별 이탈률",
            labels={service_cols[0]: '가입 여부', '이탈률': '이탈률', '서비스': '서비스 종류'},
            color_discrete_sequence=px.colors.qualitative.Bold,
            barmode='group',
            category_orders={service_cols[0]: ['Yes', 'No']}
        )
        
        # facet 타이틀 업데이트 (서비스 이름만 표시)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        fig.update_annotations(font_size=14)
        
        fig.update_layout(
            xaxis_title="가입 여부",
            yaxis_title="이탈률",
            yaxis_tickformat='.0%',
            height=700
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 분석 결론 및 권장사항
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔎 분석 결론")
    
    st.markdown("""
    <div style="padding: 20px; background-color: #f8f9fa; border-radius: 5px;">
        <h4 style="color: #2980b9;">핵심 분석 결과</h4>
        <ul>
            <li><strong>계약 유형</strong>이 이탈에 가장 큰 영향을 미치는 요인으로 확인됨</li>
            <li><strong>광케이블(Fiber Optic) 서비스</strong> 사용자의 이탈률이 높음</li>
            <li><strong>전자 청구서</strong>를 사용하지 않는 고객은 이탈 위험이 증가</li>
            <li><strong>부가 서비스</strong> 미사용 고객의 이탈률이 높음</li>
            <li>노인 고객의 이탈률이 비노인 고객보다 높음</li>
        </ul>
        
        <h4 style="color: #2980b9; margin-top: 20px;">권장사항</h4>
        <ol>
            <li>광케이블 서비스 사용자에게 특화된 유지 프로그램 개발</li>
            <li>계약 기간이 만료되는 월별 계약 고객에게 장기 계약 전환 혜택 제공</li>
            <li>부가 서비스 사용률을 높이기 위한 홍보 및 교육 캠페인 실시</li>
            <li>노인 고객을 위한 맞춤형 서비스 패키지 개발</li>
            <li>전자 청구서 사용 유도를 위한 인센티브 제공</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"데이터를 처리하는 중 오류가 발생했습니다: {e}")
    # 오류 발생 시 자세한 정보 제공
    st.markdown("""
    <div style="padding: 15px; background-color: #ffebee; border-left: 5px solid #e57373; border-radius: 4px;">
        <h3 style="color: #c62828; margin-top: 0;">오류 해결 방법</h3>
        <p>다음 사항을 확인해 주세요:</p>
        <ul>
            <li>데이터 파일('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')이 존재하는지 확인</li>
            <li>파일 경로가 올바른지 확인</li>
            <li>데이터 파일이 손상되지 않았는지 확인</li>
        </ul>
    </div>
    """, unsafe_allow_html=True) 