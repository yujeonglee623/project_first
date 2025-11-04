import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# 파일 시스템 모듈을 import 해줘야 해.
import os 
# ... (다른 import 문)

@st.cache_data
def load_market_data():
    """국가별 시장 데이터"""
    # 🌟 추가된 부분: 파일의 마지막 수정 시간을 캐시의 입력으로 사용
    last_modified = os.path.getmtime('market_data.xlsx') 
    
    # 💡 Streamlit이 이 변수가 바뀔 때마다 캐시를 무효화하도록 감지함
    return pd.read_excel('market_data.xlsx')

@st.cache_data
def load_formulation_data():
    """제형별 트렌드 데이터"""
    # 🌟 다른 파일들도 마찬가지로 수정
    os.path.getmtime('formulation_data.xlsx')
    return pd.read_excel('formulation_data.xlsx')

@st.cache_data
def load_ingredient_data():
    """성분별 트렌드 데이터"""
    # 🌟 다른 파일들도 마찬가지로 수정
    os.path.getmtime('ingredient_data.xlsx')
    return pd.read_excel('ingredient_data.xlsx')
# ...

# 페이지 설정
st.set_page_config(
    page_title="화장품 ODM 시장 예측 대시보드",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# 데이터 로드 함수
@st.cache_data
def load_market_data():
    """국가별 시장 데이터"""
    data = {
        'country': ['China', 'USA', 'Japan', 'India', 'South Korea', 'Europe', 'Southeast Asia'],
        'marketSize': [41310, 88810, 20750, 25570, 6183, 95460, 4266],
        'cagr': [9.84, 6.8, 5.3, 12.1, 5.9, 6.36, 7.8],
        'entryBarrier': [75, 60, 80, 40, 65, 70, 45],
        'competition': [85, 90, 75, 60, 95, 80, 65],
        'regulation': [80, 70, 85, 50, 60, 90, 55],
        'profit': [65, 80, 70, 85, 75, 75, 80],
        'topFormulation': ['Skincare', 'Skincare', 'Skincare', 'Skincare', 'K-Beauty Multi-step', 'Clean Beauty', 'Skincare'],
        'topIngredient': ['Natural/Organic', 'Clean Beauty', 'Biotech', 'Natural/Organic', 'Innovative', 'Sustainable', 'K-Beauty'],
        'preferredPrice': ['Mass & Premium', 'Premium', 'Premium', 'Mass', 'Mass & Premium', 'Premium', 'Mass']
    }
    return pd.DataFrame(data)

@st.cache_data
def load_formulation_data():
    """제형별 트렌드 데이터"""
    data = {
        'name': ['Skincare', 'Cleanser', 'Serum', 'Moisturizer', 'Sheet Mask', 'Suncare', 'Makeup'],
        'demand': [95, 88, 92, 90, 85, 87, 75],
        'growth': [8.5, 12.7, 10.5, 9.2, 8.0, 11.5, 5.5],
        'profitMargin': [75, 70, 85, 80, 65, 78, 70],
        'competition': [90, 75, 85, 88, 80, 70, 85],
        'innovation': [85, 80, 95, 75, 70, 82, 65],
        'roi': [80, 85, 88, 82, 75, 83, 68]
    }
    return pd.DataFrame(data)

@st.cache_data
def load_ingredient_data():
    """성분별 트렌드 데이터"""
    data = {
        'name': ['Peptides', 'Niacinamide', 'Retinol', 'Hyaluronic Acid', 'Vitamin C', 
                 'Ceramides', 'Bakuchiol', 'Natural/Organic', 'Probiotics', 'Exosomes'],
        'popularity': [98, 95, 93, 90, 88, 86, 82, 92, 80, 75],
        'efficacy': [92, 90, 95, 88, 85, 90, 80, 75, 82, 88],
        'cost': [70, 85, 75, 80, 75, 78, 65, 60, 70, 45],
        'regulation': [85, 95, 70, 90, 85, 88, 82, 80, 75, 60],
        'trend': ['Rising', 'Stable', 'Stable', 'Stable', 'Stable', 'Rising', 'Rising', 'Rising', 'Rising', 'Emerging'],
        'searchVolume': [95, 92, 90, 88, 85, 82, 78, 90, 75, 70],
        'successRate': [90, 93, 88, 90, 85, 87, 80, 85, 78, 72]
    }
    return pd.DataFrame(data)

def calculate_success_score(market_row, formulation_row, ingredient_row):
    """성공률 계산 알고리즘"""
    market_attractiveness = (market_row['cagr'] * 2 + market_row['marketSize'] / 1000) / 3
    competitive_advantage = (100 - market_row['competition'] + market_row['profit']) / 2
    product_fit = (formulation_row['demand'] + formulation_row['growth'] * 2 + formulation_row['profitMargin']) / 4
    ingredient_score = (ingredient_row['popularity'] + ingredient_row['efficacy'] + ingredient_row['successRate']) / 3
    regulatory_ease = (100 - market_row['regulation'] + ingredient_row['regulation']) / 2
    
    success_score = (
        market_attractiveness * 0.25 +
        competitive_advantage * 0.20 +
        product_fit * 0.25 +
        ingredient_score * 0.20 +
        regulatory_ease * 0.10
    )
    
    roi = formulation_row['roi'] * (ingredient_score / 100) * (competitive_advantage / 100) * 100
    
    return round(success_score), round(roi)

def generate_success_matrix(market_df, formulation_df, ingredient_df):
    """전체 성공률 매트릭스 생성"""
    results = []
    
    for _, market in market_df.iterrows():
        for _, formulation in formulation_df.iterrows():
            for _, ingredient in ingredient_df.iterrows():
                success_score, roi = calculate_success_score(market, formulation, ingredient)
                
                results.append({
                    'country': market['country'],
                    'formulation': formulation['name'],
                    'ingredient': ingredient['name'],
                    'successScore': success_score,
                    'roi': roi,
                    'marketSize': market['marketSize'],
                    'growth': market['cagr']
                })
    
    return pd.DataFrame(results)

# 메인 애플리케이션
def main():
    # 헤더
    st.markdown("""
        <h1 style='text-align: center; color: #667eea; font-size: 3em;'>
            💄 화장품 ODM 시장 예측 대시보드
        </h1>
        <p style='text-align: center; font-size: 1.2em; color: #666;'>
            2025년 최신 데이터 기반 | 글로벌 ODM 시장 규모: $67.81B → $104.69B (2032)
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 데이터 로드
    market_df = load_market_data()
    formulation_df = load_formulation_data()
    ingredient_df = load_ingredient_data()
    success_matrix = generate_success_matrix(market_df, formulation_df, ingredient_df)
    
    # 사이드바 필터
    st.sidebar.header("🎯 맞춤 분석 필터")
    
    selected_country = st.sidebar.selectbox(
        "국가 선택",
        ['전체'] + list(market_df['country'].unique())
    )
    
    selected_formulation = st.sidebar.selectbox(
        "제형 선택",
        ['전체'] + list(formulation_df['name'].unique())
    )
    
    selected_ingredient = st.sidebar.selectbox(
        "성분 선택",
        ['전체'] + list(ingredient_df['name'].unique())
    )
    
    # 필터링
    filtered_matrix = success_matrix.copy()
    if selected_country != '전체':
        filtered_matrix = filtered_matrix[filtered_matrix['country'] == selected_country]
    if selected_formulation != '전체':
        filtered_matrix = filtered_matrix[filtered_matrix['formulation'] == selected_formulation]
    if selected_ingredient != '전체':
        filtered_matrix = filtered_matrix[filtered_matrix['ingredient'] == selected_ingredient]
    
    # TOP 5 추천
    st.markdown("## 🏆 최고 성공률 예측 TOP 5")
    top_5 = success_matrix.nlargest(5, 'successScore')
    
    cols = st.columns(5)
    for idx, (_, row) in enumerate(top_5.iterrows()):
        with cols[idx]:
            st.markdown(f"""
                <div class="recommendation-card">
                    <h2 style='margin: 0;'>#{idx + 1}</h2>
                    <h3 style='margin: 5px 0;'>{row['country']}</h3>
                    <p style='margin: 3px 0; font-size: 0.9em;'>{row['formulation']}</p>
                    <p style='margin: 3px 0; font-size: 0.9em;'>{row['ingredient']}</p>
                    <div style='background-color: #ffd700; color: #333; padding: 8px; border-radius: 20px; margin-top: 10px; font-weight: bold;'>
                        성공률 {row['successScore']}%
                    </div>
                    <p style='margin-top: 10px; font-size: 0.85em;'>ROI: {row['roi']}%</p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 메인 대시보드
    tab1, tab2, tab3, tab4 = st.tabs(["📊 시장 분석", "💰 제형 분석", "🧪 성분 분석", "🎯 맞춤 추천"])
    
    with tab1:
        st.markdown("### 🌍 국가별 시장 매력도")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 시장 규모 vs 성장률
            fig1 = go.Figure()
            
            for _, row in market_df.iterrows():
                success_score = success_matrix[success_matrix['country'] == row['country']]['successScore'].mean()
                
                fig1.add_trace(go.Scatter(
                    x=[row['marketSize']],
                    y=[row['cagr']],
                    mode='markers+text',
                    marker=dict(size=success_score/2, color=success_score, colorscale='Viridis', showscale=True),
                    text=row['country'],
                    textposition="top center",
                    name=row['country']
                ))
            
            fig1.update_layout(
                title="시장 규모 vs 성장률",
                xaxis_title="시장 규모 (백만 USD)",
                yaxis_title="CAGR (%)",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # 국가별 성공률
            avg_success = success_matrix.groupby('country')['successScore'].mean().reset_index()
            avg_success = avg_success.sort_values('successScore', ascending=True)
            
            fig2 = px.bar(
                avg_success,
                x='successScore',
                y='country',
                orientation='h',
                title="국가별 평균 성공률",
                color='successScore',
                color_continuous_scale='RdYlGn',
                labels={'successScore': '평균 성공률 (%)'}
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        # 국가별 상세 지표
        st.markdown("### 📈 국가별 상세 지표")
        
        fig3 = go.Figure()
        
        metrics = ['entryBarrier', 'competition', 'regulation', 'profit']
        metric_names = ['진입장벽', '경쟁강도', '규제수준', '수익성']
        
        for metric, name in zip(metrics, metric_names):
            fig3.add_trace(go.Bar(
                name=name,
                x=market_df['country'],
                y=market_df[metric]
            ))
        
        fig3.update_layout(
            barmode='group',
            title="국가별 비즈니스 환경 비교",
            xaxis_title="국가",
            yaxis_title="점수",
            height=400
        )
        st.plotly_chart(fig3, use_container_width=True)
    
        st.markdown("### 🏆 선택 국가별 핵심 트렌드 분석")

        if selected_country != '전체':
            # 선택된 국가의 데이터만 가져옴
            current_market = market_df[market_df['country'] == selected_country].iloc[0]
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric(
                    label="1순위 제형 (Top Formulation)", 
                    value=current_market['topFormulation'], 
                    delta="시장 집중도: 높음"
                )
            with col_b:
                st.metric(
                    label="주요 성분 (Top Ingredient)", 
                    value=current_market['topIngredient'], 
                    delta="핵심 R&D 초점"
                )
            with col_c:
                st.metric(
                    label="선호 가격대 (Preferred Price)", 
                    value=current_market['preferredPrice'], 
                    delta="마진 전략 수립"
                )
            
            st.markdown(f"""
                <div style='background-color: #fffbe6; padding: 15px; border-radius: 8px; border-left: 5px solid #facc15; margin-top: 15px;'>
                    **인사이트:** {current_market['country']} 시장은 주로 **{current_market['topFormulation']}**에 대한 수요가 높으며, 특히 **{current_market['topIngredient']}** 성분을 활용하여 **{current_market['preferredPrice']}** 가격대로 진입하는 전략이 유효할 것으로 분석됩니다.
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("사이드바에서 국가를 선택하면 해당 국가의 핵심 트렌드 분석을 볼 수 있습니다.")

    with tab2:
        st.markdown("### 💰 제형별 ROI 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 레이더 차트
            fig4 = go.Figure()
            
            categories = ['수요', '성장률', '수익률', '혁신성', 'ROI']
            
            for _, row in formulation_df.iterrows():
                fig4.add_trace(go.Scatterpolar(
                    r=[row['demand'], row['growth']*5, row['profitMargin'], row['innovation'], row['roi']],
                    theta=categories,
                    fill='toself',
                    name=row['name']
                ))
            
            fig4.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                title="제형별 종합 평가",
                height=500
            )
            st.plotly_chart(fig4, use_container_width=True)
        
        with col2:
            # 제형별 ROI 랭킹
            formulation_sorted = formulation_df.sort_values('roi', ascending=True)
            
            fig5 = px.bar(
                formulation_sorted,
                x='roi',
                y='name',
                orientation='h',
                title="제형별 ROI 랭킹",
                color='roi',
                color_continuous_scale='Plasma',
                labels={'roi': 'ROI (%)', 'name': '제형'}
            )
            fig5.update_layout(height=500)
            st.plotly_chart(fig5, use_container_width=True)
        
        # 제형별 상세 데이터
        st.markdown("### 📋 제형별 상세 데이터")
        st.dataframe(
            formulation_df.style.background_gradient(cmap='YlOrRd', subset=['demand', 'growth', 'profitMargin', 'roi']),
            use_container_width=True
        )
    
    with tab3:
        st.markdown("### 🧪 2025 핵심 성분 트렌드")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 성분별 인기도
            ingredient_sorted = ingredient_df.sort_values('popularity', ascending=True)
            
            fig6 = go.Figure()
            fig6.add_trace(go.Bar(
                y=ingredient_sorted['name'],
                x=ingredient_sorted['popularity'],
                name='인기도',
                orientation='h',
                marker=dict(color='#8b5cf6')
            ))
            fig6.add_trace(go.Bar(
                y=ingredient_sorted['name'],
                x=ingredient_sorted['efficacy'],
                name='효능',
                orientation='h',
                marker=dict(color='#ec4899')
            ))
            fig6.add_trace(go.Bar(
                y=ingredient_sorted['name'],
                x=ingredient_sorted['successRate'],
                name='성공률',
                orientation='h',
                marker=dict(color='#3b82f6')
            ))
            
            fig6.update_layout(
                title="성분별 종합 평가",
                barmode='group',
                height=500,
                xaxis_title="점수"
            )
            st.plotly_chart(fig6, use_container_width=True)
        
        with col2:
            # 트렌드별 분포
            trend_counts = ingredient_df['trend'].value_counts()
            
            fig7 = px.pie(
                values=trend_counts.values,
                names=trend_counts.index,
                title="성분 트렌드 분포",
                color_discrete_sequence=px.colors.sequential.RdBu,
                hole=0.4
            )
            fig7.update_layout(height=500)
            st.plotly_chart(fig7, use_container_width=True)
            
            # 상위 성분 정보
            st.markdown("#### 🔥 TOP 5 성분")
            top_ingredients = ingredient_df.nlargest(5, 'popularity')
            for _, ing in top_ingredients.iterrows():
                st.markdown(f"""
                    **{ing['name']}** - {ing['trend']}  
                    인기도: {ing['popularity']}% | 효능: {ing['efficacy']}% | 성공률: {ing['successRate']}%
                """)
    
    with tab4:
        st.markdown("### 🎯 필터 기반 맞춤 추천")
        
        # 필터링된 결과 표시
        filtered_top = filtered_matrix.nlargest(10, 'successScore')
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 📋 추천 순위")
            
            for idx, (_, row) in enumerate(filtered_top.iterrows(), 1):
                st.markdown(f"""
                    <div style='background: linear-gradient(to right, #f3f4f6, #e5e7eb); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #8b5cf6;'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <span style='font-size: 1.5em; font-weight: bold; color: #8b5cf6;'>#{idx}</span>
                                <span style='font-size: 1.2em; font-weight: bold; margin-left: 10px;'>{row['country']}</span>
                            </div>
                            <div style='background-color: #8b5cf6; color: white; padding: 8px 15px; border-radius: 20px; font-weight: bold;'>
                                {row['successScore']}%
                            </div>
                        </div>
                        <div style='margin-top: 10px; color: #666;'>
                            <strong>제형:</strong> {row['formulation']} | <strong>성분:</strong> {row['ingredient']}<br/>
                            <strong style='color: #10b981;'>ROI: {row['roi']}%</strong> | 
                            <strong style='color: #3b82f6;'>성장률: {row['growth']}%</strong>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 📊 분석 통계")
            
            st.metric("분석된 조합 수", len(filtered_matrix))
            st.metric("평균 성공률", f"{filtered_matrix['successScore'].mean():.1f}%")
            st.metric("평균 ROI", f"{filtered_matrix['roi'].mean():.1f}%")
            
            # 최적 조합 하이라이트
            best = filtered_matrix.nlargest(1, 'successScore').iloc[0]
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 10px; margin-top: 20px;'>
                    <h4 style='margin: 0;'>🌟 최적 조합 컨설팅 제안</h4>
                    <p style='margin: 5px 0;'><strong>국가: {best['country']}</strong></p>
                    <p style='margin: 5px 0;'><strong>제형: {best['formulation']}</strong> | <strong>성분: {best['ingredient']}</strong></p>
                    <p style='margin: 5px 0; font-size: 1.5em;'>**예상 성공률: {best['successScore']}%**</p>
                    <p style='margin: 5px 0; font-size: 1.1em;'>**예상 ROI: {best['roi']}%**</p>
                </div>

                <div style='background-color: #f7f3ff; padding: 15px; border-radius: 10px; margin-top: 15px; border-left: 5px solid #a855f7;'>
                    <h5 style='color: #6b21a8; margin: 0;'>👉 컨설팅 보고서 요약</h5>
                    <p style='margin-top: 10px; font-size: 0.9em;'>
                        선택된 조합은 높은 시장 매력도와 (Success Score 산출 기준: **{best['successScore']}%**로 근거 제시), 
                        경쟁 강도 대비 높은 수익성 우위를 확보하여 (ROI: **{best['roi']}%**),
                        ODM 파트너의 다음 핵심 개발 제품으로 강력하게 추천됩니다.
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    # 푸터 인사이트
    st.markdown("---")
    st.markdown("## 📊 2025 ODM 시장 핵심 인사이트")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div style='background-color: #f0f9ff; padding: 20px; border-radius: 10px; border-left: 4px solid #3b82f6;'>
                <h4 style='color: #1e40af; margin: 0;'>🌿 자연/유기농 트렌드</h4>
                <p style='color: #1e3a8a; margin-top: 10px;'>52% 소비자가 무독성 제품 선호</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background-color: #fef3f2; padding: 20px; border-radius: 10px; border-left: 4px solid #ef4444;'>
                <h4 style='color: #991b1b; margin: 0;'>✨ 맞춤형 뷰티</h4>
                <p style='color: #7f1d1d; margin-top: 10px;'>45% 밀레니얼이 개인화 요구</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style='background-color: #f0fdf4; padding: 20px; border-radius: 10px; border-left: 4px solid #22c55e;'>
                <h4 style='color: #15803d; margin: 0;'>🛒 온라인 채널</h4>
                <p style='color: #14532d; margin-top: 10px;'>60% ODM 제품이 온라인 판매</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div style='background-color: #faf5ff; padding: 20px; border-radius: 10px; border-left: 4px solid #a855f7;'>
                <h4 style='color: #6b21a8; margin: 0;'>🔬 바이오테크 혁신</h4>
                <p style='color: #581c87; margin-top: 10px;'>AI 기반 성분 개발 가속화</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()