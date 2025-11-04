import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
import re
from wordcloud import WordCloud
import praw
from dotenv import load_dotenv
import os
from prawcore.exceptions import ResponseException, RequestException

# 페이지 설정
st.set_page_config(
    page_title="Reddit 분석 대시보드",
    page_icon="🔴",
    layout="wide"
)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
# plt.rcParams['font.family'] = 'AppleGothic'  # Mac
plt.rcParams['axes.unicode_minus'] = False


class RedditAnalyzer:
    """Reddit 분석 클래스"""
    
    def __init__(self, posts_df, comments_df=None):
        self.posts_df = posts_df.copy()
        self.comments_df = comments_df.copy() if comments_df is not None else None
        
        # 날짜 컬럼 변환
        if 'created_utc' in self.posts_df.columns:
            self.posts_df['created_utc'] = pd.to_datetime(self.posts_df['created_utc'], unit='s')
        if self.comments_df is not None and 'created_utc' in self.comments_df.columns:
            self.comments_df['created_utc'] = pd.to_datetime(self.comments_df['created_utc'], unit='s')
    
    
    def preprocess_text(self, text):
        """텍스트 전처리"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^가-힣a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    
    def extract_keywords(self, text_series, min_length=2, top_n=50):
        """키워드 추출"""
        all_text = ' '.join(text_series.apply(self.preprocess_text))
        words = all_text.split()
        words = [w for w in words if len(w) >= min_length]
        
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had',
                    '그', '이', '저', '것', '수', '등', '들', '및', '또한', '하다', '있다', '되다',
                    '이것', '그것', '저것', '그런', '이런', '저런', 'removed', 'deleted'}
        
        words = [w for w in words if w not in stopwords]
        word_freq = Counter(words)
        
        return word_freq.most_common(top_n)
    
    
    def wordcloud(self, text_series, width=1200, height=800):
        """워드클라우드 생성"""
        all_text = ' '.join(text_series.apply(self.preprocess_text))
        
        wordcloud = WordCloud(
            font_path='malgun.ttf',
            width=width,
            height=height,
            background_color='white',
            max_words=100,
            relative_scaling=0.3,
            colormap='viridis'
        ).generate(all_text)
        
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Reddit 텍스트 워드클라우드', fontsize=20, pad=20)
        plt.tight_layout()
        
        return fig
    
    
    def keyword_frequency(self, text_series, top_n=20):
        """키워드 빈도 분석"""
        keywords = self.extract_keywords(text_series, top_n=top_n)
        words, counts = zip(*keywords)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(range(len(words)), counts, color='orangered')
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.set_xlabel('빈도', fontsize=12)
        ax.set_title(f'상위 {top_n}개 키워드 빈도', fontsize=16, pad=20)
        ax.invert_yaxis()
        plt.tight_layout()
        
        freq_df = pd.DataFrame(keywords, columns=['키워드', '빈도'])
        
        return fig, freq_df
    
    
    def sentiment_analysis(self, text_series):
        """감성 분석"""
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'best', 'love', 'awesome', 'perfect', 'nice', 'happy', 'thank',
            '좋다', '최고', '대박', '예쁘다', '멋지다', '완벽', '감사', '행복'
        }
        
        negative_words = {
            'bad', 'worst', 'terrible', 'awful', 'horrible', 'hate',
            'poor', 'disappointing', 'useless', 'waste', 'crap',
            '싫다', '별로', '나쁘다', '최악', '형편없다', '실망'
        }
        
        def calculate_sentiment(text):
            text = self.preprocess_text(text)
            words = text.split()
            
            pos_count = sum(1 for w in words if w in positive_words)
            neg_count = sum(1 for w in words if w in negative_words)
            
            if pos_count > neg_count:
                return '긍정'
            elif pos_count < neg_count:
                return '부정'
            else:
                return '중립'
        
        sentiments = text_series.apply(calculate_sentiment)
        sentiment_counts = sentiments.value_counts()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        colors = ['#90EE90', '#FFB6C1', '#D3D3D3']
        axes[0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                   autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0].set_title('감성 분포', fontsize=14, pad=20)
        
        axes[1].bar(sentiment_counts.index, sentiment_counts.values, color=colors)
        axes[1].set_xlabel('감성', fontsize=12)
        axes[1].set_ylabel('개수', fontsize=12)
        axes[1].set_title('감성별 개수', fontsize=14, pad=20)
        
        plt.tight_layout()
        
        return fig, sentiment_counts
    
    
    def time_trend(self, df, date_col='created_utc', interval='D'):
        """시간대별 트렌드 분석"""
        if date_col not in df.columns:
            return None, None
        
        # 수정: set_index 후 sort_index() 추가하여 인덱스를 정렬 (resample을 위한 monotonic index 확보)
        df_sorted = df.set_index(date_col).sort_index()
        time_counts = df_sorted.resample(interval).size()
        
        if 'score' in df.columns:
            time_scores = df_sorted['score'].resample(interval).sum()
        else:
            time_scores = None
        
        if time_scores is not None:
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            axes[0].plot(time_counts.index, time_counts.values, marker='o', linewidth=2, color='orangered')
            axes[0].set_xlabel('날짜', fontsize=12)
            axes[0].set_ylabel('게시물/댓글 수', fontsize=12)
            axes[0].set_title('시간대별 게시물/댓글 수 추이', fontsize=14, pad=20)
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(time_scores.index, time_scores.values, marker='o', 
                        color='coral', linewidth=2)
            axes[1].set_xlabel('날짜', fontsize=12)
            axes[1].set_ylabel('점수 합계', fontsize=12)
            axes[1].set_title('시간대별 점수 추이', fontsize=14, pad=20)
            axes[1].grid(True, alpha=0.3)
            
            trend_df = pd.DataFrame({
                '날짜': time_counts.index,
                '개수': time_counts.values,
                '점수': time_scores.values
            })
        else:
            fig, ax = plt.subplots(figsize=(14, 6))
            
            ax.plot(time_counts.index, time_counts.values, marker='o', linewidth=2, color='orangered')
            ax.set_xlabel('날짜', fontsize=12)
            ax.set_ylabel('게시물/댓글 수', fontsize=12)
            ax.set_title('시간대별 게시물/댓글 수 추이', fontsize=14, pad=20)
            ax.grid(True, alpha=0.3)
            
            trend_df = pd.DataFrame({
                '날짜': time_counts.index,
                '개수': time_counts.values
            })
        
        plt.tight_layout()
        
        return fig, trend_df
    
    
    def subreddit_comparison(self):
        """서브레딧별 비교 분석"""
        if 'subreddit' not in self.posts_df.columns:
            return None, None
        
        subreddit_stats = self.posts_df.groupby('subreddit').agg({
            'score': ['mean', 'sum', 'count'],
            'num_comments': 'mean'
        }).round(2)
        
        subreddit_stats.columns = ['평균_점수', '총_점수', '게시물_수', '평균_댓글수']
        subreddit_stats = subreddit_stats.sort_values('총_점수', ascending=False)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 게시물 수
        axes[0, 0].bar(subreddit_stats.index, subreddit_stats['게시물_수'], color='orangered')
        axes[0, 0].set_title('서브레딧별 게시물 수', fontsize=14)
        axes[0, 0].set_ylabel('게시물 수')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 총 점수
        axes[0, 1].bar(subreddit_stats.index, subreddit_stats['총_점수'], color='coral')
        axes[0, 1].set_title('서브레딧별 총 점수', fontsize=14)
        axes[0, 1].set_ylabel('총 점수')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 평균 점수
        axes[1, 0].bar(subreddit_stats.index, subreddit_stats['평균_점수'], color='tomato')
        axes[1, 0].set_title('서브레딧별 평균 점수', fontsize=14)
        axes[1, 0].set_ylabel('평균 점수')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 평균 댓글 수
        axes[1, 1].bar(subreddit_stats.index, subreddit_stats['평균_댓글수'], color='lightsalmon')
        axes[1, 1].set_title('서브레딧별 평균 댓글 수', fontsize=14)
        axes[1, 1].set_ylabel('평균 댓글 수')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        return fig, subreddit_stats


def search_and_collect_reddit_data(subreddit_names, search_query, post_limit, sort_by, time_filter, collect_comments, comment_limit):
    """Reddit API를 통한 데이터 수집"""
    load_dotenv()
    
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "RedditAnalyzer/1.0")
    
    if not client_id or not client_secret:
        st.error("Reddit API 자격증명이 설정되지 않았습니다. .env 파일을 확인하세요.")
        st.info("필요한 환경 변수: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT")
        return None, None
    
    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
    except Exception as e:
        st.error(f"Reddit API 연결 오류: {e}")
        return None, None
    
    all_posts = []
    all_comments = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    subreddit_list = [s.strip() for s in subreddit_names.split(',')]
    total_subreddits = len(subreddit_list)
    
    for idx, subreddit_name in enumerate(subreddit_list):
        status_text.text(f"서브레딧 {idx+1}/{total_subreddits}: r/{subreddit_name} 수집 중...")
        
        try:
            subreddit = reddit.subreddit(subreddit_name)
            
            # 게시물 검색/수집
            if search_query:
                posts = subreddit.search(search_query, sort=sort_by, time_filter=time_filter, limit=post_limit)
            else:
                if sort_by == 'hot':
                    posts = subreddit.hot(limit=post_limit)
                elif sort_by == 'new':
                    posts = subreddit.new(limit=post_limit)
                elif sort_by == 'top':
                    posts = subreddit.top(time_filter=time_filter, limit=post_limit)
                elif sort_by == 'rising':
                    posts = subreddit.rising(limit=post_limit)
                else:
                    posts = subreddit.hot(limit=post_limit)
            
            for post in posts:
                post_data = {
                    'post_id': post.id,
                    'subreddit': post.subreddit.display_name,
                    'title': post.title,
                    'selftext': post.selftext,
                    'author': str(post.author),
                    'score': post.score,
                    'upvote_ratio': post.upvote_ratio,
                    'num_comments': post.num_comments,
                    'created_utc': post.created_utc,
                    'url': post.url,
                    'permalink': f"https://reddit.com{post.permalink}"
                }
                all_posts.append(post_data)
                
                # 댓글 수집
                if collect_comments:
                    try:
                        post.comments.replace_more(limit=0)
                        comments = post.comments.list()[:comment_limit]
                        
                        for comment in comments:
                            if hasattr(comment, 'body'):
                                comment_data = {
                                    'comment_id': comment.id,
                                    'post_id': post.id,
                                    'subreddit': post.subreddit.display_name,
                                    'author': str(comment.author),
                                    'body': comment.body,
                                    'score': comment.score,
                                    'created_utc': comment.created_utc,
                                    'post_title': post.title
                                }
                                all_comments.append(comment_data)
                    except Exception as e:
                        st.warning(f"게시물 {post.id} 댓글 수집 오류: {e}")
        
        except Exception as e:
            st.error(f"서브레딧 r/{subreddit_name} 수집 오류: {e}")
        
        progress_bar.progress((idx + 1) / total_subreddits)
    
    progress_bar.empty()
    status_text.empty()
    
    posts_df = pd.DataFrame(all_posts)
    comments_df = pd.DataFrame(all_comments) if all_comments else None
    
    return posts_df, comments_df


# ========================================
# Streamlit 메인 앱
# ========================================

def main():
    st.title("🔴 Reddit 분석 대시보드")
    st.markdown("---")
    
    # 사이드바 - 데이터 수집/업로드
    st.sidebar.header("📂 데이터 소스")
    data_source = st.sidebar.radio(
        "데이터 입력 방식 선택",
        ["API로 실시간 수집", "CSV 파일 업로드"]
    )
    
    posts_df = None
    comments_df = None
    
    if data_source == "API로 실시간 수집":
        st.sidebar.subheader("🔍 검색 설정")
        subreddit_names = st.sidebar.text_input(
            "서브레딧 (쉼표로 구분)",
            value="kbeauty,AsianBeauty",
            help="예: kbeauty,AsianBeauty,SkincareAddiction"
        )
        search_query = st.sidebar.text_input(
            "검색어 (선택)",
            value="",
            help="비워두면 서브레딧의 모든 게시물 수집"
        )
        post_limit = st.sidebar.slider("게시물 개수", 10, 500, 100)
        
        sort_by = st.sidebar.selectbox(
            "정렬 방식",
            ["hot", "new", "top", "rising"],
            format_func=lambda x: {
                "hot": "인기순",
                "new": "최신순",
                "top": "최고 평점",
                "rising": "급상승"
            }[x]
        )
        
        time_filter = st.sidebar.selectbox(
            "기간 필터 (top/search만 적용)",
            ["all", "day", "week", "month", "year"],
            format_func=lambda x: {
                "all": "전체",
                "day": "오늘",
                "week": "이번 주",
                "month": "이번 달",
                "year": "올해"
            }[x]
        )
        
        collect_comments = st.sidebar.checkbox("댓글도 수집", value=True)
        comment_limit = st.sidebar.slider("게시물당 댓글 수", 10, 200, 50) if collect_comments else 0
        
        if st.sidebar.button("🚀 데이터 수집 시작"):
            with st.spinner("Reddit 데이터 수집 중..."):
                posts_df, comments_df = search_and_collect_reddit_data(
                    subreddit_names, search_query, post_limit, 
                    sort_by, time_filter, collect_comments, comment_limit
                )
            
            if posts_df is not None and not posts_df.empty:
                st.success(f"✅ 게시물 {len(posts_df)}개 수집 완료!")
                if comments_df is not None:
                    st.success(f"✅ 댓글 {len(comments_df)}개 수집 완료!")
                
                # 세션 스테이트에 저장
                st.session_state['posts_df'] = posts_df
                st.session_state['comments_df'] = comments_df
            else:
                st.warning("수집된 데이터가 없습니다.")
    
    else:  # CSV 파일 업로드
        st.sidebar.subheader("📤 파일 업로드")
        posts_file = st.sidebar.file_uploader("게시물 CSV 파일", type=['csv'])
        comments_file = st.sidebar.file_uploader("댓글 CSV 파일 (선택)", type=['csv'])
        
        if posts_file:
            posts_df = pd.read_csv(posts_file)
            st.session_state['posts_df'] = posts_df
            st.sidebar.success(f"✅ 게시물 {len(posts_df)}개 로드")
        
        if comments_file:
            comments_df = pd.read_csv(comments_file)
            st.session_state['comments_df'] = comments_df
            st.sidebar.success(f"✅ 댓글 {len(comments_df)}개 로드")
    
    # 세션 스테이트에서 데이터 로드
    if 'posts_df' in st.session_state:
        posts_df = st.session_state['posts_df']
    if 'comments_df' in st.session_state:
        comments_df = st.session_state['comments_df']
    # 기본 통계
    st.header("📈 기본 통계")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("총 게시물 수", f"{len(posts_df):,}")
    with col2:
        st.metric("평균 점수", f"{posts_df['score'].mean():.1f}")
    with col3:
        st.metric("평균 댓글 수", f"{posts_df['num_comments'].mean():.1f}")
    with col4:
    if comments_df is not None:
        st.metric("총 댓글 수", f"{len(comments_df):,}")

    st.markdown("---")

    # 탭으로 분석 모드 구분
    tabs = st.tabs([
    "☁️ 워드클라우드",
    "📊 키워드 빈도",
    "😊😢 감성 분석",
    "📈 시간 트렌드",
    "🎯 서브레딧 비교",
    "📋 원본 데이터"
    ])

    analyzer = RedditAnalyzer(posts_df, comments_df)

    # 탭 1: 워드클라우드
    with tabs[0]:
    st.header("☁️ 워드클라우드")

    text_source = st.radio(
        "텍스트 소스",
        ["게시물 제목", "게시물 본문", "댓글"] if comments_df is not None else ["게시물 제목", "게시물 본문"],
        horizontal=True
    )

    if st.button("🔍 워드클라우드 생성", key="btn_wordcloud"):
        with st.spinner("워드클라우드 생성 중..."):
            if text_source == "게시물 제목":
                fig = analyzer.wordcloud(posts_df['title'])
            elif text_source == "게시물 본문":
                fig = analyzer.wordcloud(posts_df['selftext'])
            else:  # 댓글
                fig = analyzer.wordcloud(comments_df['body'])
            st.pyplot(fig)
    else:
        st.info("👆 텍스트 소스를 선택하고 버튼을 클릭하세요.")

    # 탭 2: 키워드 빈도
    with tabs[1]:
    st.header("📊 키워드 빈도 분석")

    text_source = st.radio(
        "텍스트 소스",
        ["게시물 제목", "게시물 본문", "댓글"] if comments_df is not None else ["게시물 제목", "게시물 본문"],
        horizontal=True,
        key="keyword_source"
    )
    top_n = st.slider("표시할 키워드 개수", 10, 50, 20, key="keyword_top_n")

    if st.button("🔍 키워드 빈도 분석", key="btn_keyword"):
        with st.spinner("키워드 빈도 분석 중..."):
            if text_source == "게시물 제목":
                fig, freq_df = analyzer.keyword_frequency(posts_df['title'], top_n=top_n)
            elif text_source == "게시물 본문":
                fig, freq_df = analyzer.keyword_frequency(posts_df['selftext'], top_n=top_n)
            else:
                fig, freq_df = analyzer.keyword_frequency(comments_df['body'], top_n=top_n)
            
            st.pyplot(fig)
            
            st.subheader("📋 키워드 데이터")
            st.dataframe(freq_df, use_container_width=True)
            
            csv = freq_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button(
                "💾 CSV 다운로드",
                csv,
                "reddit_keyword_frequency.csv",
                "text/csv",
                key='download-keyword-csv'
            )
    else:
        st.info("👆 텍스트 소스를 선택하고 버튼을 클릭하세요.")

    # 탭 3: 감성 분석
    with tabs[2]:
    st.header("😊😢 감성 분석")

    text_source = st.radio(
        "텍스트 소스",
        ["게시물 제목", "게시물 본문", "댓글"] if comments_df is not None else ["게시물 제목", "게시물 본문"],
        horizontal=True,
        key="sentiment_source"
    )

    if st.button("🔍 감성 분석 실행", key="btn_sentiment"):
        with st.spinner("감성 분석 중..."):
            if text_source == "게시물 제목":
                fig, sentiment_counts = analyzer.sentiment_analysis(posts_df['title'])
            elif text_source == "게시물 본문":
                fig, sentiment_counts = analyzer.sentiment_analysis(posts_df['selftext'])
            else:
                fig, sentiment_counts = analyzer.sentiment_analysis(comments_df['body'])
            
            st.pyplot(fig)
            
            col1, col2, col3 = st.columns(3)
            for idx, (sentiment, count) in enumerate(sentiment_counts.items()):
                with [col1, col2, col3][idx % 3]:
                    st.metric(sentiment, f"{count:,}개")
    else:
        st.info("👆 텍스트 소스를 선택하고 버튼을 클릭하세요.")

    # 탭 4: 시간 트렌드
    with tabs[3]:
    st.header("📈 시간 트렌드 분석")

    data_source_trend = st.radio(
        "데이터 소스",
        ["게시물", "댓글"] if comments_df is not None else ["게시물"],
        horizontal=True
    )
    interval = st.radio("시간 간격", ["D (일)", "W (주)", "M (월)"], horizontal=True, key="time_interval")
    interval_code = interval.split()[0]

    if st.button("🔍 시간 트렌드 분석", key="btn_time"):
        with st.spinner("시간 트렌드 분석 중..."):
            if data_source_trend == "게시물":
                fig, trend_df = analyzer.time_trend(posts_df, interval=interval_code)
            else:
                fig, trend_df = analyzer.time_trend(comments_df, interval=interval_code)
            
            if fig:
                st.pyplot(fig)
                
                st.subheader("📋 트렌드 데이터")
                st.dataframe(trend_df, use_container_width=True)
                
                csv = trend_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                st.download_button(
                    "💾 CSV 다운로드",
                    csv,
                    "reddit_time_trend.csv",
                    "text/csv",
                    key='download-trend-csv'
                )
            else:
                st.warning("날짜 정보가 없어 시간 트렌드 분석을 수행할 수 없습니다.")
    else:
        st.info("👆 데이터 소스와 시간 간격을 선택하고 버튼을 클릭하세요.")

    # 탭 5: 서브레딧 비교
    with tabs[4]:
    st.header("🎯 서브레딧 비교 분석")

    if st.button("🔍 서브레딧 비교 분석", key="btn_subreddit"):
        with st.spinner("서브레딧 비교 분석 중..."):
            fig, comparison_df = analyzer.subreddit_comparison()
            if fig:
                st.pyplot(fig)
                
                st.subheader("📋 서브레딧 통계")
                st.dataframe(comparison_df, use_container_width=True)
                
                csv = comparison_df.to_csv(encoding='utf-8-sig').encode('utf-8-sig')
                st.download_button(
                    "💾 CSV 다운로드",
                    csv,
                    "reddit_subreddit_comparison.csv",
                    "text/csv",
                    key='download-subreddit-csv'
                )
            else:
                st.warning("서브레딧 정보가 없어 비교 분석을 수행할 수 없습니다.")
    else:
        st.info("👆 버튼을 클릭하여 서브레딧별 통계를 확인하세요.")

    # 탭 6: 원본 데이터
    with tabs[5]:
    st.header("📋 원본 데이터")

    data_type = st.radio(
        "데이터 유형 선택",
        ["게시물 데이터", "댓글 데이터"] if comments_df is not None else ["게시물 데이터"],
        horizontal=True
    )

    if data_type == "게시물 데이터":
        st.subheader("📝 게시물 데이터")
        
        # 컬럼 선택
        display_columns = st.multiselect(
            "표시할 컬럼 선택",
            posts_df.columns.tolist(),
            default=['title', 'subreddit', 'score', 'num_comments', 'author'][:min(5, len(posts_df.columns))]
        )
        
        if display_columns:
            st.dataframe(posts_df[display_columns], use_container_width=True, height=600)
        else:
            st.dataframe(posts_df, use_container_width=True, height=600)
        
        csv = posts_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button(
            "💾 게시물 데이터 CSV 다운로드",
            csv,
            "reddit_posts_data.csv",
            "text/csv",
            key='download-posts-raw'
        )

    else:
        if comments_df is not None and not comments_df.empty:
            st.subheader("💬 댓글 데이터")
            
            # 컬럼 선택
            display_columns = st.multiselect(
                "표시할 컬럼 선택",
                comments_df.columns.tolist(),
                default=['body', 'subreddit', 'score', 'author', 'post_title'][:min(5, len(comments_df.columns))]
            )
            
            if display_columns:
                st.dataframe(comments_df[display_columns], use_container_width=True, height=600)
            else:
                st.dataframe(comments_df, use_container_width=True, height=600)
            
            csv = comments_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button(
                "💾 댓글 데이터 CSV 다운로드",
                csv,
                "reddit_comments_data.csv",
                "text/csv",
                key='download-comments-raw'
            )
        else:
            st.warning("댓글 데이터가 없습니다.")


    if __name__ == "__main__":
    main()