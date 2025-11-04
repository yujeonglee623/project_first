import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
import re
from wordcloud import WordCloud
import io
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
import time
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch

# 페이지 설정
st.set_page_config(
    page_title="YouTube 댓글 분석 대시보드",
    page_icon="🎥",
    layout="wide"
)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
# plt.rcParams['font.family'] = 'AppleGothic'  # Mac
plt.rcParams['axes.unicode_minus'] = False


class YouTubeCommentAnalyzer:
    """YouTube 댓글 분석 클래스"""
    
    def __init__(self, comments_df, videos_df=None):
        self.comments_df = comments_df.copy()
        self.videos_df = videos_df.copy() if videos_df is not None else None
        
        # 날짜 컬럼 변환
        if 'published_at' in self.comments_df.columns:
            self.comments_df['published_at'] = pd.to_datetime(self.comments_df['published_at'])
    
    
    def preprocess_text(self, text):
        """텍스트 전처리"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^가-힣a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    
    def extract_keywords(self, min_length=2, top_n=50):
        """키워드 추출"""
        all_text = ' '.join(self.comments_df['text'].apply(self.preprocess_text))
        words = all_text.split()
        words = [w for w in words if len(w) >= min_length]
        
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had',
                    '그', '이', '저', '것', '수', '등', '들', '및', '또한', '하다', '있다', '되다',
                    '이것', '그것', '저것', '그런', '이런', '저런'}
        
        words = [w for w in words if w not in stopwords]
        word_freq = Counter(words)
        
        return word_freq.most_common(top_n)
    
    
    def wordcloud(self, width=1200, height=800):
        """워드클라우드 생성"""
        all_text = ' '.join(self.comments_df['text'].apply(self.preprocess_text))
        
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
        ax.set_title('댓글 워드클라우드', fontsize=20, pad=20)
        plt.tight_layout()
        
        return fig
    
    
    def keyword_frequency(self, top_n=20):
        """키워드 빈도 분석"""
        keywords = self.extract_keywords(top_n=top_n)
        words, counts = zip(*keywords)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(range(len(words)), counts, color='skyblue')
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.set_xlabel('빈도', fontsize=12)
        ax.set_title(f'상위 {top_n}개 키워드 빈도', fontsize=16, pad=20)
        ax.invert_yaxis()
        plt.tight_layout()
        
        freq_df = pd.DataFrame(keywords, columns=['키워드', '빈도'])
        
        return fig, freq_df
    
    
    def sentiment_keywords(self):
        """감성 키워드 분석"""
        positive_words = {
            '좋다', '최고', '대박', '예쁘다', '이쁘다', '멋지다', '훌륭하다', 
            '완벽', '좋아', '감사', '사랑', '행복', '추천', '굿', 'good', 
            'best', 'love', 'amazing', 'perfect', 'great', 'excellent',
            '좋아요', '좋네요', '멋있다', '아름답다', '최고다', '짱'
        }
        
        negative_words = {
            '싫다', '별로', '안좋다', '나쁘다', '최악', '형편없다',
            '싫어', '실망', '별로네', '아쉽다', 'bad', 'worst', 'hate',
            '싫어요', '별로예요', '그저그렇다', '지루하다'
        }
        
        def calculate_sentiment(text):
            text = self.preprocess_text(text)
            words = text.split()
            
            pos_count = sum(1 for w in words if w in positive_words)
            neg_count = sum(1 for w in words if w in negative_words)
            
            return pos_count, neg_count
        
        self.comments_df[['positive_count', 'negative_count']] = \
            self.comments_df['text'].apply(lambda x: pd.Series(calculate_sentiment(x)))
        
        def classify_sentiment(row):
            if row['positive_count'] > row['negative_count']:
                return '긍정'
            elif row['positive_count'] < row['negative_count']:
                return '부정'
            else:
                return '중립'
        
        self.comments_df['sentiment'] = self.comments_df.apply(classify_sentiment, axis=1)
        sentiment_counts = self.comments_df['sentiment'].value_counts()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        colors = ['#90EE90', '#FFB6C1', '#D3D3D3']
        axes[0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                   autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0].set_title('댓글 감성 분포', fontsize=14, pad=20)
        
        axes[1].bar(sentiment_counts.index, sentiment_counts.values, color=colors)
        axes[1].set_xlabel('감성', fontsize=12)
        axes[1].set_ylabel('댓글 수', fontsize=12)
        axes[1].set_title('감성별 댓글 수', fontsize=14, pad=20)
        
        plt.tight_layout()
        
        sentiment_df = self.comments_df[['text', 'sentiment', 'positive_count', 'negative_count']]
        
        return fig, sentiment_counts, sentiment_df
    
    
    def time_trend(self, interval='D'):
        """시간대별 트렌드 분석"""
        if 'published_at' not in self.comments_df.columns:
            return None, None
        
        time_counts = self.comments_df.set_index('published_at').resample(interval).size()
        time_likes = self.comments_df.set_index('published_at')['like_count'].resample(interval).sum()
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        axes[0].plot(time_counts.index, time_counts.values, marker='o', linewidth=2)
        axes[0].set_xlabel('날짜', fontsize=12)
        axes[0].set_ylabel('댓글 수', fontsize=12)
        axes[0].set_title('시간대별 댓글 수 추이', fontsize=14, pad=20)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(time_likes.index, time_likes.values, marker='o', 
                    color='coral', linewidth=2)
        axes[1].set_xlabel('날짜', fontsize=12)
        axes[1].set_ylabel('좋아요 수', fontsize=12)
        axes[1].set_title('시간대별 좋아요 수 추이', fontsize=14, pad=20)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        trend_df = pd.DataFrame({
            '날짜': time_counts.index,
            '댓글_수': time_counts.values,
            '좋아요_수': time_likes.values
        })
        
        return fig, trend_df
    
    
    def cooccurrence(self, top_n=15):
        """키워드 동시출현 분석"""
        top_keywords = [word for word, _ in self.extract_keywords(top_n=top_n)]
        cooc_matrix = pd.DataFrame(0, index=top_keywords, columns=top_keywords)
        
        for text in self.comments_df['text']:
            text = self.preprocess_text(text)
            words = set(text.split())
            
            for word1 in top_keywords:
                if word1 in words:
                    for word2 in top_keywords:
                        if word2 in words:
                            cooc_matrix.loc[word1, word2] += 1
        
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(cooc_matrix, annot=True, fmt='d', cmap='YlOrRd', 
                   cbar_kws={'label': '동시출현 빈도'}, ax=ax)
        ax.set_title(f'상위 {top_n}개 키워드 동시출현 분석', fontsize=16, pad=20)
        ax.set_xlabel('키워드', fontsize=12)
        ax.set_ylabel('키워드', fontsize=12)
        plt.tight_layout()
        
        return fig, cooc_matrix
    
    
    def topic_comparison(self):
        """영상별 토픽 비교 분석"""
        if 'video_title' not in self.comments_df.columns:
            return None, None
        
        video_keywords = {}
        
        for video_title in self.comments_df['video_title'].unique()[:10]:
            video_comments = self.comments_df[
                self.comments_df['video_title'] == video_title
            ]['text']
            
            all_text = ' '.join(video_comments.apply(self.preprocess_text))
            words = all_text.split()
            words = [w for w in words if len(w) >= 2]
            
            word_freq = Counter(words)
            top_words = [word for word, _ in word_freq.most_common(5)]
            
            video_keywords[video_title[:30] + '...'] = top_words
        
        comparison_df = pd.DataFrame(video_keywords).T
        comparison_df.columns = [f'키워드{i+1}' for i in range(comparison_df.shape[1])]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=comparison_df.values,
                        rowLabels=comparison_df.index,
                        colLabels=comparison_df.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        for i in range(len(comparison_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(len(comparison_df)):
            table[(i+1, -1)].set_facecolor('#E8F5E9')
            table[(i+1, -1)].set_text_props(weight='bold')
        
        plt.title('영상별 주요 키워드 비교', fontsize=16, pad=20)
        
        return fig, comparison_df


def search_and_collect_data(keyword, max_videos, max_comments_per_video, order):
    """YouTube API를 통한 데이터 수집"""
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    
    if not api_key:
        st.error("YouTube API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return None, None
    
    youtube = build("youtube", "v3", developerKey=api_key)
    
    # 영상 검색
    try:
        search_response = youtube.search().list(
            q=keyword,
            part="snippet",
            maxResults=min(max_videos, 50),
            type="video",
            order=order,
            regionCode="KR"
        ).execute()
        
        video_ids = [item["id"]["videoId"] for item in search_response["items"]]
    except Exception as e:
        st.error(f"검색 오류: {e}")
        return None, None
    
    # 영상 상세 정보
    videos_data = []
    try:
        for i in range(0, len(video_ids), 50):
            batch_ids = video_ids[i:i+50]
            video_response = youtube.videos().list(
                part="snippet,statistics,contentDetails",
                id=",".join(batch_ids)
            ).execute()
            
            for item in video_response["items"]:
                video_info = {
                    "video_id": item["id"],
                    "title": item["snippet"]["title"],
                    "channel": item["snippet"]["channelTitle"],
                    "published_at": item["snippet"]["publishedAt"],
                    "description": item["snippet"]["description"],
                    "view_count": int(item["statistics"].get("viewCount", 0)),
                    "like_count": int(item["statistics"].get("likeCount", 0)),
                    "comment_count": int(item["statistics"].get("commentCount", 0)),
                    "duration": item["contentDetails"]["duration"],
                    "tags": ", ".join(item["snippet"].get("tags", [])),
                    "url": f"https://www.youtube.com/watch?v={item['id']}"
                }
                videos_data.append(video_info)
    except Exception as e:
        st.error(f"영상 정보 수집 오류: {e}")
    
    videos_df = pd.DataFrame(videos_data)
    
    # 댓글 수집
    all_comments = []
    video_info_dict = {}
    for _, row in videos_df.iterrows():
        video_info_dict[row['video_id']] = {
            'title': row['title'],
            'channel': row['channel'],
            'url': row['url']
        }
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, video_id in enumerate(video_ids):
        status_text.text(f"영상 {idx+1}/{len(video_ids)} 댓글 수집 중...")
        progress_bar.progress((idx + 1) / len(video_ids))
        
        try:
            comments = []
            next_page_token = None
            
            while len(comments) < max_comments_per_video:
                request = youtube.commentThreads().list(
                    part="snippet,replies",
                    videoId=video_id,
                    maxResults=min(100, max_comments_per_video - len(comments)),
                    pageToken=next_page_token,
                    textFormat="plainText",
                    order="relevance"
                )
                response = request.execute()
                
                for item in response["items"]:
                    top_comment = item["snippet"]["topLevelComment"]["snippet"]
                    
                    comment_info = {
                        "comment_id": item["snippet"]["topLevelComment"]["id"],
                        "video_id": video_id,
                        "author": top_comment["authorDisplayName"],
                        "text": top_comment["textDisplay"],
                        "like_count": top_comment["likeCount"],
                        "published_at": top_comment["publishedAt"],
                        "reply_count": item["snippet"]["totalReplyCount"]
                    }
                    
                    if video_id in video_info_dict:
                        comment_info['video_title'] = video_info_dict[video_id]['title']
                        comment_info['video_channel'] = video_info_dict[video_id]['channel']
                        comment_info['video_url'] = video_info_dict[video_id]['url']
                    
                    comments.append(comment_info)
                
                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break
                
                time.sleep(0.5)
            
            all_comments.extend(comments)
        
        except Exception as e:
            if "commentsDisabled" not in str(e):
                st.warning(f"영상 {video_id} 댓글 수집 오류: {e}")
        
        time.sleep(1)
    
    progress_bar.empty()
    status_text.empty()
    
    comments_df = pd.DataFrame(all_comments)
    
    return videos_df, comments_df


# ========================================
# Streamlit 메인 앱
# ========================================

def main():
    st.title("🎥 YouTube 댓글 분석 대시보드")
    st.markdown("---")
    
    # 사이드바 - 데이터 수집/업로드만
    st.sidebar.header("📂 데이터 소스")
    data_source = st.sidebar.radio(
        "데이터 입력 방식 선택",
        ["API로 실시간 수집", "CSV 파일 업로드"]
    )
    
    videos_df = None
    comments_df = None
    
    if data_source == "API로 실시간 수집":
        st.sidebar.subheader("🔍 검색 설정")
        keyword = st.sidebar.text_input("검색 키워드", value="K-beauty")
        max_videos = st.sidebar.slider("영상 개수", 1, 50, 10)
        max_comments = st.sidebar.slider("영상당 댓글 수", 10, 200, 50)
        order = st.sidebar.selectbox(
            "정렬 방식",
            ["relevance", "date", "viewCount"],
            format_func=lambda x: {"relevance": "관련성순", "date": "최신순", "viewCount": "조회수순"}[x]
        )
        
        if st.sidebar.button("🚀 데이터 수집 시작"):
            with st.spinner("데이터 수집 중..."):
                videos_df, comments_df = search_and_collect_data(
                    keyword, max_videos, max_comments, order
                )
            
            if videos_df is not None and comments_df is not None:
                st.success(f"✅ 영상 {len(videos_df)}개, 댓글 {len(comments_df)}개 수집 완료!")
                
                # 세션 스테이트에 저장
                st.session_state['videos_df'] = videos_df
                st.session_state['comments_df'] = comments_df
    
    else:  # CSV 파일 업로드
        st.sidebar.subheader("📤 파일 업로드")
        comments_file = st.sidebar.file_uploader("댓글 CSV 파일", type=['csv'])
        videos_file = st.sidebar.file_uploader("영상 CSV 파일 (선택)", type=['csv'])
        
        if comments_file:
            comments_df = pd.read_csv(comments_file)
            st.session_state['comments_df'] = comments_df
            st.sidebar.success(f"✅ 댓글 {len(comments_df)}개 로드")
        
        if videos_file:
            videos_df = pd.read_csv(videos_file)
            st.session_state['videos_df'] = videos_df
            st.sidebar.success(f"✅ 영상 {len(videos_df)}개 로드")
    
    # 세션 스테이트에서 데이터 로드
    if 'comments_df' in st.session_state:
        comments_df = st.session_state['comments_df']
    if 'videos_df' in st.session_state:
        videos_df = st.session_state['videos_df']
    
    # 데이터가 없으면 안내 메시지
    if comments_df is None or comments_df.empty:
        st.info("👆 왼쪽 사이드바에서 데이터를 수집하거나 업로드해주세요.")
        return
    
    # 기본 통계
    st.header("📈 기본 통계")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 댓글 수", f"{len(comments_df):,}")
    with col2:
        st.metric("평균 좋아요", f"{comments_df['like_count'].mean():.1f}")
    with col3:
        st.metric("총 좋아요", f"{comments_df['like_count'].sum():,}")
    with col4:
        if videos_df is not None:
            st.metric("분석 영상 수", f"{len(videos_df)}")
    
    st.markdown("---")
    
    # 탭으로 분석 모드 구분
    tabs = st.tabs([
        "☁️ 워드클라우드",
        "📊 키워드 빈도",
        "😊😢 감성 분석",
        "📈 시간 트렌드",
        "🔗 동시출현",
        "🎬 토픽 비교",
        "📋 원본 데이터"
    ])
    
    analyzer = YouTubeCommentAnalyzer(comments_df, videos_df)
    
    # 탭 1: 워드클라우드
    with tabs[0]:
        st.header("☁️ 워드클라우드")
        if st.button("🔍 워드클라우드 생성", key="btn_wordcloud"):
            with st.spinner("워드클라우드 생성 중..."):
                fig = analyzer.wordcloud()
                st.pyplot(fig)
        else:
            st.info("👆 버튼을 클릭하여 워드클라우드를 생성하세요.")
    
    # 탭 2: 키워드 빈도
    with tabs[1]:
        st.header("📊 키워드 빈도 분석")
        top_n = st.slider("표시할 키워드 개수", 10, 50, 20, key="keyword_top_n")
        
        if st.button("🔍 키워드 빈도 분석", key="btn_keyword"):
            with st.spinner("키워드 빈도 분석 중..."):
                fig, freq_df = analyzer.keyword_frequency(top_n=top_n)
                st.pyplot(fig)
                
                st.subheader("📋 키워드 데이터")
                st.dataframe(freq_df, use_container_width=True)
                
                csv = freq_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                st.download_button(
                    "💾 CSV 다운로드",
                    csv,
                    "keyword_frequency.csv",
                    "text/csv",
                    key='download-keyword-csv'
                )
        else:
            st.info("👆 버튼을 클릭하여 키워드 빈도를 분석하세요.")
    
    # 탭 3: 감성 분석
    with tabs[2]:
        st.header("😊😢 감성 분석")
        
        if st.button("🔍 감성 분석 실행", key="btn_sentiment"):
            with st.spinner("감성 분석 중..."):
                fig, sentiment_counts, sentiment_df = analyzer.sentiment_keywords()
                st.pyplot(fig)
                
                col1, col2, col3 = st.columns(3)
                for idx, (sentiment, count) in enumerate(sentiment_counts.items()):
                    with [col1, col2, col3][idx]:
                        st.metric(sentiment, f"{count:,}개")
                
                st.subheader("📋 감성 분류 데이터")
                st.dataframe(sentiment_df.head(100), use_container_width=True)
                
                csv = sentiment_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                st.download_button(
                    "💾 CSV 다운로드",
                    csv,
                    "sentiment_analysis.csv",
                    "text/csv",
                    key='download-sentiment-csv'
                )
        else:
            st.info("👆 버튼을 클릭하여 감성 분석을 실행하세요.")
    
    # 탭 4: 시간 트렌드
    with tabs[3]:
        st.header("📈 시간 트렌드 분석")
        interval = st.radio("시간 간격", ["D (일)", "W (주)", "M (월)"], horizontal=True, key="time_interval")
        interval_code = interval.split()[0]
        
        if st.button("🔍 시간 트렌드 분석", key="btn_time"):
            with st.spinner("시간 트렌드 분석 중..."):
                fig, trend_df = analyzer.time_trend(interval=interval_code)
                if fig:
                    st.pyplot(fig)
                    
                    st.subheader("📋 트렌드 데이터")
                    st.dataframe(trend_df, use_container_width=True)
                    
                    csv = trend_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                    st.download_button(
                        "💾 CSV 다운로드",
                        csv,
                        "time_trend.csv",
                        "text/csv",
                        key='download-trend-csv'
                    )
                else:
                    st.warning("published_at 컬럼이 없어 시간 트렌드 분석을 수행할 수 없습니다.")
        else:
            st.info("👆 시간 간격을 선택하고 버튼을 클릭하여 분석하세요.")
    
    # 탭 5: 동시출현
    with tabs[4]:
        st.header("🔗 키워드 동시출현 분석")
        cooc_n = st.slider("분석할 키워드 개수", 5, 20, 15, key="cooc_n")
        
        if st.button("🔍 동시출현 분석", key="btn_cooc"):
            with st.spinner("동시출현 분석 중..."):
                fig, cooc_matrix = analyzer.cooccurrence(top_n=cooc_n)
                st.pyplot(fig)
                
                st.subheader("📋 동시출현 매트릭스")
                st.dataframe(cooc_matrix, use_container_width=True)
                
                csv = cooc_matrix.to_csv(encoding='utf-8-sig').encode('utf-8-sig')
                st.download_button(
                    "💾 CSV 다운로드",
                    csv,
                    "cooccurrence_matrix.csv",
                    "text/csv",
                    key='download-cooc-csv'
                )
        else:
            st.info("👆 키워드 개수를 선택하고 버튼을 클릭하여 분석하세요.")
    
    # 탭 6: 토픽 비교
    with tabs[5]:
        st.header("🎬 영상별 토픽 비교")
        
        if st.button("🔍 토픽 비교 분석", key="btn_topic"):
            with st.spinner("토픽 비교 분석 중..."):
                fig, comparison_df = analyzer.topic_comparison()
                if fig:
                    st.pyplot(fig)
                    
                    st.subheader("📋 토픽 비교 데이터")
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    csv = comparison_df.to_csv(encoding='utf-8-sig').encode('utf-8-sig')
                    st.download_button(
                        "💾 CSV 다운로드",
                        csv,
                        "topic_comparison.csv",
                        "text/csv",
                        key='download-topic-csv'
                    )
                else:
                    st.warning("video_title 컬럼이 없어 토픽 비교 분석을 수행할 수 없습니다.")
        else:
            st.info("👆 버튼을 클릭하여 토픽 비교를 분석하세요.")
    
    # 탭 7: 원본 데이터
    with tabs[6]:
        st.header("📋 원본 데이터")
        
        data_type = st.radio("데이터 유형 선택", ["댓글 데이터", "영상 데이터"], horizontal=True)
        
        if data_type == "댓글 데이터":
            st.subheader("💬 댓글 데이터")
            st.dataframe(comments_df, use_container_width=True, height=600)
            
            csv = comments_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button(
                "💾 댓글 데이터 CSV 다운로드",
                csv,
                "comments_data.csv",
                "text/csv",
                key='download-comments-raw'
            )
        
        else:
            if videos_df is not None and not videos_df.empty:
                st.subheader("🎥 영상 데이터")
                st.dataframe(videos_df, use_container_width=True, height=600)
                
                csv = videos_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                st.download_button(
                    "💾 영상 데이터 CSV 다운로드",
                    csv,
                    "videos_data.csv",
                    "text/csv",
                    key='download-videos-raw'
                )
            else:
                st.warning("영상 데이터가 없습니다.")

    st.markdown("---")
    st.header("📄 Market Insight Report Generator (flan-t5-base 기반)")
    st.write("저장된 분석 CSV 파일을 기반으로 요약 보고서를 자동 생성합니다.")

    available_files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".csv")]

    if not available_files:
        st.warning("분석 결과 CSV 파일이 없습니다. 먼저 위 분석 탭에서 분석을 실행하고 CSV를 다운로드/저장하세요.")
    else:
        selected_files = st.multiselect("📂 보고서에 포함할 파일 선택", available_files, default=available_files)

        if st.button("🧠 보고서 생성"):
            full_text = ""
            for f in selected_files:
                file_path = os.path.join(SAVE_DIR, f)
                try:
                    # CSV를 pandas로 로드하고 서술적 텍스트로 변환 (이전 제안처럼)
                    df = pd.read_csv(file_path, encoding="utf-8-sig")
                    content = f"Market insight from {f}:\n"
                    content += f"This dataset has {len(df)} rows.\n"
                    content += "Key statistics:\n" + df.describe().to_string() + "\n\n"
                    content += "Sample data:\n" + df.head(5).to_string(index=False) + "\n\n"
                    full_text += content + "\n\n"
                except Exception as e:
                    st.error(f"파일 {f} 처리 오류: {str(e)}")
                    continue

            if full_text:
                with st.spinner("flan-t5-base 모델이 요약 보고서를 생성 중..."):
                    summary = summarize_text(full_text, tokenizer, model, max_input=1024, max_output=500)

                st.subheader("📈 AI 자동 생성 보고서 초안")
                st.text_area("요약 결과", summary, height=400)

                st.download_button(
                    "💾 요약 보고서 저장",
                    summary.encode("utf-8-sig"),
                    "Market_Insight_Report.txt",
                    "text/plain"
                )



SAVE_DIR = "analysis_results"
os.makedirs(SAVE_DIR, exist_ok=True)

from transformers import AutoTokenizer, AutoModelWithLMHead  # WithLMHead 사용 (모델 카드 추천)

@st.cache_resource
def load_common_gen_model():
    """Hugging Face mrm8488/t5-base-finetuned-common_gen 모델 로드"""
    model_name = "mrm8488/t5-base-finetuned-common_gen"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    if torch.cuda.is_available():
        model.half()  # FP16으로 VRAM 절약
        model.to("cuda")
    return tokenizer, model

tokenizer, model = load_common_gen_model()

def generate_report(text, tokenizer, model, max_length=64):
    """mrm8488/t5-base-finetuned-common_gen을 이용한 보고서 생성 함수"""
    # 입력 텍스트를 키워드 문자열로 변환 (당신 CSV 예시처럼)
    # full_text가 "Insight from file: keywords beauty 50 makeup 30 ..." 식이면 split으로 키워드 추출
    keywords = ' '.join(text.split()[:20])  # 너무 길면 자름 (모델 입력 제한 ~512 토큰)
    
    inputs = tokenizer([keywords], return_tensors="pt", truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # 생성: max_length로 길이 제어, num_beams로 품질 향상
    output_ids = model.generate(**inputs, max_length=max_length, num_beams=4, early_stopping=True)
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generate

if __name__ == "__main__":
    main()