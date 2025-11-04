import streamlit as st
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from wordcloud import WordCloud

# 페이지 설정
st.set_page_config(
    page_title="Reddit 트렌드 분석 대시보드",
    page_icon="📊",
    layout="wide"
)

# 스타일 설정
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# 함수 정의
# ============================================

@st.cache_data
def load_data(file):
    """데이터 로드"""
    df = pd.read_csv(file)
    return df

def clean_text(text):
    """텍스트 전처리"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return text

def extract_words(text, min_length=3):
    """단어 추출"""
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'can', 'my', 'your', 'their',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
        'we', 'they', 'me', 'him', 'her', 'us', 'them', 'what', 'which',
        'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
        'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'really',
        'also', 'like', 'get', 'got', 'use', 'used', 'using', 'one', 'two'
    }
    
    words = clean_text(text).split()
    return [w for w in words if len(w) >= min_length and w not in stopwords]

# ============================================
# 메인 대시보드
# ============================================

st.title("📊 Reddit 트렌드 분석 대시보드")
st.markdown("---")

# 사이드바 - 파일 업로드 및 설정
with st.sidebar:
    st.header("⚙️ 설정")
    
    uploaded_file = st.file_uploader(
        "댓글 분석 CSV 파일 업로드",
        type=['csv'],
        help="comment_analysis.csv 파일을 업로드하세요"
    )
    
    st.markdown("---")
    
    if uploaded_file:
        st.success("✅ 파일 로드 완료!")
        df = load_data(uploaded_file)
        st.metric("총 댓글 수", f"{len(df):,}개")
        
        if 'sentiment' in df.columns:
            positive = len(df[df['sentiment'] == 'POSITIVE'])
            negative = len(df[df['sentiment'] == 'NEGATIVE'])
            st.metric("긍정 비율", f"{positive/len(df)*100:.1f}%")
            st.metric("부정 비율", f"{negative/len(df)*100:.1f}%")
    
    st.markdown("---")
    st.markdown("### 📖 사용 방법")
    st.markdown("""
    1. CSV 파일 업로드
    2. 원하는 분석 탭 선택
    3. 설정 조정 후 분석 실행
    """)

# 파일 업로드 확인
if not uploaded_file:
    st.info("👈 왼쪽 사이드바에서 CSV 파일을 업로드해주세요.")
    st.stop()

# 탭 생성
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "☁️ 워드클라우드",
    "📊 키워드 빈도",
    "😊😞 감성 키워드",
    "📈 시간 트렌드",
    "🔗 키워드 연관",
    "🏷️ 토픽 비교"
])

# ============================================
# 탭 1: 워드클라우드
# ============================================
with tab1:
    st.header("☁️ 워드클라우드 분석")
    st.markdown("댓글에서 자주 등장하는 단어를 시각화합니다.")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("설정")
        min_word_length = st.slider("최소 단어 길이", 2, 5, 3)
        top_n_words = st.slider("표시할 단어 수", 20, 100, 50)
        
        if st.button("🎨 워드클라우드 생성", key="wc_btn"):
            with st.spinner("생성 중..."):
                # 모든 댓글 합치기
                all_text = ' '.join(df['comment_body'].apply(clean_text))
                words = extract_words(all_text, min_word_length)
                word_freq = Counter(words)
                
                # 워드클라우드 생성
                wordcloud = WordCloud(
                    width=1200,
                    height=600,
                    background_color='white',
                    colormap='viridis',
                    relative_scaling=0.5,
                    min_font_size=10
                ).generate_from_frequencies(dict(word_freq.most_common(top_n_words)))
                
                # 시각화
                fig, ax = plt.subplots(figsize=(15, 8))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                
                with col2:
                    st.pyplot(fig)
                
                # 상위 단어 테이블
                st.subheader("📋 상위 키워드")
                top_words_df = pd.DataFrame(
                    word_freq.most_common(20),
                    columns=['단어', '빈도']
                )
                st.dataframe(top_words_df, use_container_width=True)
    
    with col2:
        if 'wordcloud' not in locals():
            st.info("👈 왼쪽에서 '워드클라우드 생성' 버튼을 클릭하세요.")

# ============================================
# 탭 2: 키워드 빈도 분석
# ============================================
with tab2:
    st.header("📊 키워드 빈도 분석")
    st.markdown("특정 키워드들의 언급 빈도와 감성을 분석합니다.")
    
    # 키워드 입력
    st.subheader("🔍 분석할 키워드 입력")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        keywords_input = st.text_area(
            "키워드 입력 (쉼표로 구분)",
            value="toner, serum, cream, cleanser, mask, sunscreen, niacinamide, retinol, vitamin c, hyaluronic acid",
            height=100,
            help="분석하고 싶은 키워드를 쉼표로 구분해서 입력하세요"
        )
    
    with col2:
        top_n = st.number_input("표시할 상위 개수", 5, 30, 15)
        show_sentiment = st.checkbox("감성 분석 포함", value=True)
    
    if st.button("📊 분석 시작", key="kf_btn"):
        with st.spinner("분석 중..."):
            custom_keywords = [kw.strip().lower() for kw in keywords_input.split(',')]
            
            keyword_stats = []
            has_sentiment = 'sentiment' in df.columns
            
            for keyword in custom_keywords:
                total_count = 0
                positive_count = 0
                negative_count = 0
                total_score = 0
                
                for idx, row in df.iterrows():
                    text = row['comment_body']
                    if pd.notna(text) and keyword.lower() in str(text).lower():
                        total_count += 1
                        total_score += row.get('comment_score', 0)
                        
                        if has_sentiment and show_sentiment:
                            if row['sentiment'] == 'POSITIVE':
                                positive_count += 1
                            elif row['sentiment'] == 'NEGATIVE':
                                negative_count += 1
                
                keyword_stats.append({
                    'keyword': keyword,
                    'count': total_count,
                    'percentage': round(total_count / len(df) * 100, 2),
                    'positive': positive_count,
                    'negative': negative_count,
                    'positive_rate': round(positive_count / total_count * 100, 1) if total_count > 0 else 0,
                    'negative_rate': round(negative_count / total_count * 100, 1) if total_count > 0 else 0,
                    'avg_score': round(total_score / total_count, 2) if total_count > 0 else 0
                })
            
            result_df = pd.DataFrame(keyword_stats)
            result_df = result_df.sort_values('count', ascending=False).head(top_n)
            
            # 그래프
            if has_sentiment and show_sentiment:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("언급 빈도")
                    fig1, ax1 = plt.subplots(figsize=(10, 8))
                    ax1.barh(result_df['keyword'], result_df['count'], color='steelblue')
                    ax1.set_xlabel('Mentions')
                    ax1.invert_yaxis()
                    st.pyplot(fig1)
                
                with col2:
                    st.subheader("감성 비율")
                    fig2, ax2 = plt.subplots(figsize=(10, 8))
                    y_pos = range(len(result_df))
                    ax2.barh(y_pos, result_df['positive_rate'], color='lightgreen', label='Positive')
                    ax2.barh(y_pos, result_df['negative_rate'], left=result_df['positive_rate'], 
                            color='lightcoral', label='Negative')
                    ax2.set_yticks(y_pos)
                    ax2.set_yticklabels(result_df['keyword'])
                    ax2.set_xlabel('Sentiment Rate (%)')
                    ax2.set_xlim(0, 100)
                    ax2.legend()
                    ax2.invert_yaxis()
                    st.pyplot(fig2)
            else:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.barh(result_df['keyword'], result_df['count'], color='skyblue')
                ax.set_xlabel('Mentions')
                ax.invert_yaxis()
                st.pyplot(fig)
            
            # 테이블
            st.subheader("📋 상세 데이터")
            if has_sentiment and show_sentiment:
                display_df = result_df[['keyword', 'count', 'percentage', 'positive_rate', 'negative_rate', 'avg_score']]
                display_df.columns = ['키워드', '언급수', '비율(%)', '긍정률(%)', '부정률(%)', '평균점수']
            else:
                display_df = result_df[['keyword', 'count', 'percentage']]
                display_df.columns = ['키워드', '언급수', '비율(%)']
            
            st.dataframe(display_df, use_container_width=True)
            
            # 다운로드
            csv = result_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "📥 CSV 다운로드",
                csv,
                "keyword_frequency.csv",
                "text/csv"
            )

# ============================================
# 탭 3: 감성별 키워드
# ============================================
with tab3:
    st.header("😊😞 감성별 키워드 분석")
    st.markdown("긍정 댓글과 부정 댓글에서 자주 나오는 단어를 비교합니다.")
    
    if 'sentiment' not in df.columns:
        st.warning("⚠️ 감성 분석 데이터가 없습니다.")
    else:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("설정")
            top_n_sentiment = st.slider("표시할 키워드 수", 10, 30, 15, key="sent_slider")
            min_length = st.slider("최소 단어 길이", 2, 5, 3, key="sent_length")
            
            if st.button("🔍 분석 시작", key="sent_btn"):
                with st.spinner("분석 중..."):
                    positive_comments = df[df['sentiment'] == 'POSITIVE']['comment_body']
                    negative_comments = df[df['sentiment'] == 'NEGATIVE']['comment_body']
                    
                    # 긍정 키워드
                    positive_text = ' '.join(positive_comments.apply(clean_text))
                    positive_words = extract_words(positive_text, min_length)
                    positive_freq = Counter(positive_words).most_common(top_n_sentiment)
                    
                    # 부정 키워드
                    negative_text = ' '.join(negative_comments.apply(clean_text))
                    negative_words = extract_words(negative_text, min_length)
                    negative_freq = Counter(negative_words).most_common(top_n_sentiment)
                    
                    # 그래프
                    with col2:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                        
                        # 긍정
                        pos_words = [w for w, c in positive_freq]
                        pos_counts = [c for w, c in positive_freq]
                        ax1.barh(pos_words, pos_counts, color='lightgreen')
                        ax1.set_xlabel('Frequency')
                        ax1.set_title('😊 Positive Keywords', color='green', fontsize=14)
                        ax1.invert_yaxis()
                        
                        # 부정
                        neg_words = [w for w, c in negative_freq]
                        neg_counts = [c for w, c in negative_freq]
                        ax2.barh(neg_words, neg_counts, color='lightcoral')
                        ax2.set_xlabel('Frequency')
                        ax2.set_title('😞 Negative Keywords', color='red', fontsize=14)
                        ax2.invert_yaxis()
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # 테이블
                    st.subheader("📋 상세 데이터")
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown("**😊 긍정 키워드**")
                        pos_df = pd.DataFrame(positive_freq, columns=['단어', '빈도'])
                        st.dataframe(pos_df, use_container_width=True)
                    
                    with col_b:
                        st.markdown("**😞 부정 키워드**")
                        neg_df = pd.DataFrame(negative_freq, columns=['단어', '빈도'])
                        st.dataframe(neg_df, use_container_width=True)
        
        with col2:
            if 'positive_freq' not in locals():
                st.info("👈 왼쪽에서 '분석 시작' 버튼을 클릭하세요.")

# ============================================
# 탭 4: 시간 트렌드
# ============================================
with tab4:
    st.header("📈 시간대별 키워드 트렌드")
    st.markdown("시간에 따른 키워드 언급량 변화를 추적합니다.")
    
    if 'comment_created' not in df.columns:
        st.warning("⚠️ 날짜 데이터가 없습니다.")
    else:
        st.subheader("🔍 분석할 키워드 선택")
        
        trend_keywords = st.text_input(
            "키워드 입력 (쉼표로 구분)",
            value="hydrating, brightening, anti-aging",
            help="최대 5개까지 추천"
        )
        
        if st.button("📈 트렌드 분석", key="trend_btn"):
            with st.spinner("분석 중..."):
                df['date'] = pd.to_datetime(df['comment_created']).dt.date
                keywords = [kw.strip().lower() for kw in trend_keywords.split(',')][:5]
                
                trend_data = []
                for keyword in keywords:
                    for date in df['date'].unique():
                        date_comments = df[df['date'] == date]['comment_body']
                        count = sum(1 for text in date_comments if pd.notna(text) and keyword in str(text).lower())
                        trend_data.append({
                            'date': date,
                            'keyword': keyword,
                            'count': count
                        })
                
                trend_df = pd.DataFrame(trend_data)
                
                # 그래프
                fig, ax = plt.subplots(figsize=(14, 8))
                for keyword in keywords:
                    keyword_data = trend_df[trend_df['keyword'] == keyword]
                    ax.plot(keyword_data['date'], keyword_data['count'], 
                           marker='o', label=keyword, linewidth=2)
                
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Mentions', fontsize=12)
                ax.set_title('Keyword Trends Over Time', fontsize=16)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # 테이블
                st.subheader("📋 상세 데이터")
                pivot_df = trend_df.pivot(index='date', columns='keyword', values='count')
                st.dataframe(pivot_df, use_container_width=True)

# ============================================
# 탭 5: 키워드 공출현
# ============================================
with tab5:
    st.header("🔗 키워드 공출현 분석")
    st.markdown("함께 언급되는 키워드들의 관계를 분석합니다.")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("설정")
        top_words_co = st.slider("분석할 상위 키워드 수", 10, 40, 20, key="co_slider")
        
        if st.button("🔗 분석 시작", key="co_btn"):
            with st.spinner("분석 중..."):
                # 상위 키워드 추출
                all_text = ' '.join(df['comment_body'].apply(clean_text))
                words = extract_words(all_text, 3)
                top_words = [w for w, c in Counter(words).most_common(top_words_co)]
                
                # 공출현 매트릭스
                cooccurrence = np.zeros((len(top_words), len(top_words)))
                
                for text in df['comment_body']:
                    if pd.isna(text):
                        continue
                    comment_words = set(extract_words(str(text), 3))
                    for i, word1 in enumerate(top_words):
                        for j, word2 in enumerate(top_words):
                            if word1 in comment_words and word2 in comment_words:
                                cooccurrence[i][j] += 1
                
                # 히트맵
                with col2:
                    fig, ax = plt.subplots(figsize=(14, 12))
                    sns.heatmap(cooccurrence, xticklabels=top_words, yticklabels=top_words,
                               cmap='YlOrRd', annot=False, fmt='g', cbar_kws={'label': 'Co-occurrence Count'})
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # 상위 조합
                st.subheader("🔥 가장 많이 함께 언급되는 키워드 조합")
                pairs = []
                for i in range(len(top_words)):
                    for j in range(i+1, len(top_words)):
                        if cooccurrence[i][j] > 0:
                            pairs.append((top_words[i], top_words[j], int(cooccurrence[i][j])))
                
                pairs.sort(key=lambda x: x[2], reverse=True)
                pairs_df = pd.DataFrame(pairs[:15], columns=['키워드 1', '키워드 2', '공출현 횟수'])
                st.dataframe(pairs_df, use_container_width=True)
    
    with col2:
        if 'cooccurrence' not in locals():
            st.info("👈 왼쪽에서 '분석 시작' 버튼을 클릭하세요.")

# ============================================
# 탭 6: 토픽 비교
# ============================================
with tab6:
    st.header("🏷️ 토픽 그룹 비교")
    st.markdown("여러 키워드를 그룹으로 묶어 토픽별 인기도를 비교합니다.")
    
    st.subheader("🎯 토픽 그룹 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        topic1_name = st.text_input("토픽 1 이름", "Hydration")
        topic1_keywords = st.text_input("토픽 1 키워드", "hydrating, moisture, dewy, plump")
        
        topic2_name = st.text_input("토픽 2 이름", "Brightening")
        topic2_keywords = st.text_input("토픽 2 키워드", "brightening, glow, radiant, luminous")
    
    with col2:
        topic3_name = st.text_input("토픽 3 이름", "Anti-Aging")
        topic3_keywords = st.text_input("토픽 3 키워드", "anti-aging, wrinkle, firm, lifting")
        
        topic4_name = st.text_input("토픽 4 이름", "Acne")
        topic4_keywords = st.text_input("토픽 4 키워드", "acne, breakout, pimple, blemish")
    
    if st.button("🏷️ 토픽 비교 분석", key="topic_btn"):
        with st.spinner("분석 중..."):
            topic_groups = {
                topic1_name: [k.strip().lower() for k in topic1_keywords.split(',')],
                topic2_name: [k.strip().lower() for k in topic2_keywords.split(',')],
                topic3_name: [k.strip().lower() for k in topic3_keywords.split(',')],
                topic4_name: [k.strip().lower() for k in topic4_keywords.split(',')]
            }
            
            topic_stats = []
            has_sentiment = 'sentiment' in df.columns
            
            for topic, keywords in topic_groups.items():
                mentions = 0
                positive = 0
                negative = 0
                total_score = 0
                
                for idx, row in df.iterrows():
                    text = str(row['comment_body']).lower()
                    if any(kw in text for kw in keywords):
                        mentions += 1
                        total_score += row.get('comment_score', 0)
                        
                        if has_sentiment:
                            if row['sentiment'] == 'POSITIVE':
                                positive += 1
                            elif row['sentiment'] == 'NEGATIVE':
                                negative += 1
                
                topic_stats.append({
                    'topic': topic,
                    'mentions': mentions,
                    'positive': positive,
                    'negative': negative,
                    'positive_rate': round(positive / mentions * 100, 1) if mentions > 0 else 0,
                    'avg_score': round(total_score / mentions, 2) if mentions > 0 else 0
                })
            
            result_df = pd.DataFrame(topic_stats)
            
            # 그래프
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("언급 빈도")
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                ax1.bar(result_df['topic'], result_df['mentions'], color='steelblue')
                ax1.set_ylabel('Mentions')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig1)
            
            if has_sentiment:
                with col2:
                    st.subheader("긍정률")
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    ax2.bar(result_df['topic'], result_df['positive_rate'], color='lightgreen')
                    ax2.set_ylabel('Positive Rate (%)')
                    ax2.set_ylim(0, 100)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig2)
            
            # 테이블
            st.subheader("📋 토픽별 통계")
            display_df = result_df[['topic', 'mentions', 'positive_rate', 'avg_score']]
            display_df.columns = ['토픽', '언급수', '긍정률(%)', '평균점수']
            st.dataframe(display_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Reddit Trend Analysis Dashboard")