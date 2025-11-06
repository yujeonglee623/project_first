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
import requests 

# ========================================
# Streamlit 기본 설정 및 AI 설정
# ========================================

# 페이지 설정
st.set_page_config(
    page_title="YouTube 댓글 분석 대시보드",
    page_icon="🎥",
    layout="wide"
)

# 한글 폰트 설정 (시스템에 'Malgun Gothic'이 설치되어 있어야 함)
plt.rcParams['font.family'] = 'Malgun Gothic' 
plt.rcParams['axes.unicode_minus'] = False

SAVE_DIR = "analysis_results"
os.makedirs(SAVE_DIR, exist_ok=True)


def generate_openai_report(keywords, user_focus_prompt, api_key, model_name="gpt-4o"):
    """사용자 프롬프트와 분석 데이터를 조합하여 OpenAI API를 이용한 일반 보고서 문장 생성 함수 (항목당 최대 5문장 제한)"""
    
    if not api_key:
        return "Error: OpenAI API Key is missing. Please set the OPENAI_API_KEY in the .env file."

    # 시스템 프롬프트: 길이를 '최대 5문장'으로 제한하고 분석가 역할 강조
    system_prompt = (
        "You are a professional YouTube Market Analyst. "
        "Your task is to analyze the provided raw data summary or statistical analysis "
        "and generate a comprehensive, insightful, and professional English summary **limited to a maximum of 5 detailed sentences**. "
        "The summary must strictly address the User Focus prompt, covering key trends, sentiment drivers, quantitative findings, and strategic implications. "
        "If positive or negative comments are provided in the sample, use them as evidence to explain the sentiment drivers. "
        "Do not use markdown headers or lists. Just provide the summary text."
    )
    
    # User Prompt: 사용자 입력과 분석 데이터를 합쳐서 전달
    full_user_prompt = (
        f"**User Focus:** {user_focus_prompt}\n\n"
        f"**Analysis Data for Context:** {keywords}\n\n"
        "Generate the insightful report summary in English, focusing on the User Focus above."
    )


    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # API Payload (5문장 분량에 맞게 토큰 제한)
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_user_prompt}
        ],
        "max_tokens": 250, 
        "temperature": 0.3, 
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=40)
        response.raise_for_status() 
        
        result = response.json()
        
        summary = result['choices'][0]['message']['content'].strip()
        return summary

    except requests.exceptions.HTTPError as errh:
        return f"HTTP Error: {errh}. Check if your API key is valid and the model name is correct. (Status: {response.status_code})"
    except requests.exceptions.ConnectionError as errc:
        return f"Error Connecting: {errc}"
    except requests.exceptions.Timeout as errt:
        return f"Timeout Error: {errt}"
    except requests.exceptions.RequestException as err:
        return f"An Unexpected Error: {err}"
    except Exception as e:
        return f"API Error occurred: {e}. Check response structure."


def generate_executive_report(keywords, user_exec_prompt, api_key, model_name="gpt-4o"):
    """사용자 프롬프트와 분석 데이터를 조합하여 OpenAI API를 이용한 임원진 보고서 생성 (국문+영문 분리)"""
    
    if not api_key:
        return "Error: OpenAI API Key is missing."

    # 시스템 프롬프트: 임원진 보고서 형식 요청
    system_prompt = (
        "You are a Senior Executive Market Analyst specializing in YouTube Trends. "
        "Your task is to analyze the provided data and generate a highly summarized, actionable **Executive Report** based on the user's focus. "
        "The response MUST contain two clearly separated sections: **English Summary** (5-7 sentences) and its **Korean Summary** (5-7 sentences). "
        "The output format MUST strictly follow the structure: 'English Summary: [English Text] Korean Summary: [Korean Text]'. "
        "Focus on key insights, strategic implications, and high-level trends. "
    )
    
    # User Prompt: 사용자 입력과 분석 데이터를 합쳐서 전달
    full_user_exec_prompt = (
        f"**User Focus for Executive Report:** {user_exec_prompt}\n\n"
        f"**Analysis Data for Context:** {keywords}\n\n"
        "Generate the Executive Report in the required dual-language format."
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # API Payload
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_user_exec_prompt}
        ],
        "max_tokens": 800, 
        "temperature": 0.3, 
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=50)
        response.raise_for_status()
        result = response.json()
        summary = result['choices'][0]['message']['content'].strip()
        
        # 결과 파싱
        eng_match = re.search(r'English Summary:\s*(.*?)\s*Korean Summary:', summary, re.DOTALL)
        kor_match = re.search(r'Korean Summary:\s*(.*)', summary, re.DOTALL)

        english_summary = eng_match.group(1).strip() if eng_match else "Parsing Error: English Summary not found."
        korean_summary = kor_match.group(1).strip() if kor_match else "Parsing Error: Korean Summary not found."

        return english_summary, korean_summary

    except Exception as e:
        error_msg = f"API Error occurred: {e}. Check API key or rate limits."
        return error_msg, error_msg

# ========================================
# 분석 클래스 (YouTubeCommentAnalyzer)
# ========================================

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
            # font_path='malgun.ttf', 
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
        
        # CSV 출력을 위해 컬럼명을 영어로 변경
        freq_df = pd.DataFrame(keywords, columns=['Keyword', 'Frequency'])
        
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
        
        # 내부 컬럼명 변경
        self.comments_df[['PositiveCount', 'NegativeCount']] = \
            self.comments_df['text'].apply(lambda x: pd.Series(calculate_sentiment(x)))
        
        def classify_sentiment(row):
            if row['PositiveCount'] > row['NegativeCount']:
                return '긍정'
            elif row['PositiveCount'] < row['NegativeCount']:
                return '부정'
            else:
                return '중립'
        
        self.comments_df['sentiment'] = self.comments_df.apply(classify_sentiment, axis=1)
        sentiment_counts = self.comments_df['sentiment'].value_counts()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        colors = ['#90EE90', '#FFB6C1', '#D3D3D3']
        
        order = ['긍정', '부정', '중립']
        ordered_counts = sentiment_counts.reindex(order, fill_value=0)
        ordered_counts = ordered_counts[ordered_counts > 0]
        ordered_colors = [c for s, c in zip(order, colors) if s in ordered_counts.index]
        
        axes[0].pie(ordered_counts.values, labels=ordered_counts.index, 
                      autopct='%1.1f%%', colors=ordered_colors, startangle=90)
        axes[0].set_title('댓글 감성 분포', fontsize=14, pad=20)
        
        axes[1].bar(ordered_counts.index, ordered_counts.values, color=ordered_colors)
        axes[1].set_xlabel('감성', fontsize=12)
        axes[1].set_ylabel('댓글 수', fontsize=12)
        axes[1].set_title('감성별 댓글 수', fontsize=14, pad=20)
        
        plt.tight_layout()
        
        # CSV 출력을 위해 컬럼명을 영어로 변경
        sentiment_df = self.comments_df[['text', 'sentiment', 'PositiveCount', 'NegativeCount', 'like_count']].rename(
            columns={'sentiment': 'Sentiment', 'text': 'Text', 'like_count': 'LikeCount'}
        )
        
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
        
        # CSV 출력을 위해 컬럼명을 영어로 변경
        trend_df = pd.DataFrame({
            'Date': time_counts.index,
            'CommentCount': time_counts.values,
            'LikeCount': time_likes.values
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
        ax.set_xlabel('Keyword2', fontsize=12) 
        ax.set_ylabel('Keyword1', fontsize=12) 
        plt.tight_layout()
        
        # CSV 출력을 위해 컬럼명을 영어로 변경
        cooc_matrix.index.name = 'Keyword1'
        cooc_matrix.columns.name = 'Keyword2'
        
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
        # CSV 출력을 위해 컬럼명을 영어로 변경
        comparison_df.columns = [f'Keyword{i+1}' for i in range(comparison_df.shape[1])]
        comparison_df.index.name = 'VideoTitle' 
        
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        cell_text = comparison_df.reset_index().values.tolist() 
        col_labels = ['VideoTitle'] + comparison_df.columns.tolist()
        
        table = ax.table(cellText=cell_text,
                          rowLabels=None, 
                          colLabels=col_labels,
                          cellLoc='center',
                          loc='center',
                          bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # 헤더 셀 스타일링
        for i in range(len(col_labels)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # 로우 레이블 스타일링 (첫 번째 데이터 컬럼)
        for i in range(len(comparison_df)):
            table[(i+1, 0)].set_facecolor('#E8F5E9') 
            table[(i+1, 0)].set_text_props(weight='bold')
            
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
    
    # OpenAI API Key 설정 (.env 파일에서 로드)
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # 사이드바 - 데이터 수집/업로드만
    st.sidebar.header("📂 데이터 소스")
    data_source = st.sidebar.radio(
        "데이터 입력 방식 선택",
        ["API로 실시간 수집", "CSV 파일 업로드"]
    )
    
    # 세션 상태 초기화 및 로드
    if 'videos_df' not in st.session_state:
        st.session_state['videos_df'] = None
    if 'comments_df' not in st.session_state:
        st.session_state['comments_df'] = None
    
    videos_df = st.session_state['videos_df']
    comments_df = st.session_state['comments_df']
    
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
                videos_df_new, comments_df_new = search_and_collect_data(
                    keyword, max_videos, max_comments, order
                )
            
            if videos_df_new is not None and comments_df_new is not None and not comments_df_new.empty:
                st.success(f"✅ 영상 {len(videos_df_new)}개, 댓글 {len(comments_df_new)}개 수집 완료!")
                
                # 🔴 API 수집 직후 원본 파일 타임스탬프 저장
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # 1. 영상 데이터 저장
                videos_file_name = f"videos_data_{timestamp}.csv"
                videos_file_path = os.path.join(SAVE_DIR, videos_file_name)
                videos_df_new.to_csv(videos_file_path, index=False, encoding='utf-8-sig')
                st.success(f"💾 영상 데이터가 **{videos_file_name}**로 저장되었습니다.")

                # 2. 댓글 데이터 저장
                comments_file_name = f"comments_data_{timestamp}.csv"
                comments_file_path = os.path.join(SAVE_DIR, comments_file_name)
                comments_df_new.to_csv(comments_file_path, index=False, encoding='utf-8-sig')
                st.success(f"💾 댓글 데이터가 **{comments_file_name}**로 저장되었습니다.")

                # 세션 스테이트에 저장 (UI 업데이트를 위해 재할당)
                st.session_state['videos_df'] = videos_df_new
                st.session_state['comments_df'] = comments_df_new
                st.rerun() 
            elif comments_df_new is not None and comments_df_new.empty:
                st.warning("수집된 댓글이 없습니다. 검색 조건이나 API 상태를 확인하세요.")
    
    else: # CSV 파일 업로드
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
        
        if comments_file or videos_file:
            st.rerun()

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
    
    # 탭으로 분석 모드 구분 (총 9개 탭으로 재구성)
    tabs = st.tabs([
        "☁️ 워드클라우드",
        "📊 키워드 빈도",
        "😊😢 감성 분석",
        "📈 시간 트렌드",
        "🔗 동시출현",
        "🎬 토픽 비교",
        "📋 원본 데이터",
        "📄 일반 보고서", 
        "💼 임원진 보고서" # 🔴 [추가] 임원진 보고서 탭
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

    # 탭 2: 키워드 빈도 (타임스탬프 저장 적용)
    with tabs[1]:
        st.header("📊 키워드 빈도 분석")
        top_n = st.slider("표시할 키워드 개수", 10, 50, 20, key="keyword_top_n")
        
        if st.button("🔍 키워드 빈도 분석", key="btn_keyword"):
            with st.spinner("키워드 빈도 분석 중..."):
                fig, freq_df = analyzer.keyword_frequency(top_n=top_n)
                st.pyplot(fig)
                st.session_state['freq_df'] = freq_df 
                
                st.subheader("📋 키워드 데이터 (English Column)")
                st.dataframe(freq_df, use_container_width=True)
                
                # 🔴 [수정] 타임스탬프 저장
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_file_name = f"keyword_frequency_{timestamp}.csv"
                freq_df.to_csv(os.path.join(SAVE_DIR, csv_file_name), index=False, encoding='utf-8-sig')
                st.success(f"분석 결과가 서버 폴더 `{SAVE_DIR}`에 **{csv_file_name}**로 저장되었습니다.")

                csv = freq_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                st.download_button("💾 CSV 다운로드", csv, csv_file_name, "text/csv", key='download-keyword-csv')
        else:
            if 'freq_df' in st.session_state:
                freq_df = st.session_state['freq_df']
                st.subheader("📋 마지막 분석 결과 (키워드 데이터 - English Column)")
                st.dataframe(freq_df, use_container_width=True)
            else:
                st.info("👆 버튼을 클릭하여 키워드 빈도를 분석하세요.")
    
    # 탭 3: 감성 분석 (타임스탬프 저장 적용)
    with tabs[2]:
        st.header("😊😢 감성 분석")
        
        if st.button("🔍 감성 분석 실행", key="btn_sentiment"):
            with st.spinner("감성 분석 중..."):
                fig, sentiment_counts, sentiment_df = analyzer.sentiment_keywords()
                st.pyplot(fig)
                st.session_state['sentiment_df'] = sentiment_df 
                
                col1, col2, col3 = st.columns(3)
                for idx, (sentiment, count) in enumerate(sentiment_counts.items()):
                    with [col1, col2, col3][idx]:
                        st.metric(sentiment, f"{count:,}개")
                
                st.subheader("📋 감성 분류 데이터 (English Column)")
                st.dataframe(sentiment_df.head(100), use_container_width=True)
                
                # 🔴 [수정] 타임스탬프 저장
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_file_name = f"sentiment_analysis_{timestamp}.csv"
                sentiment_df.to_csv(os.path.join(SAVE_DIR, csv_file_name), index=False, encoding='utf-8-sig')
                st.success(f"분석 결과가 서버 폴더 `{SAVE_DIR}`에 **{csv_file_name}**로 저장되었습니다.")

                csv = sentiment_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                st.download_button("💾 CSV 다운로드", csv, csv_file_name, "text/csv", key='download-sentiment-csv')
        else:
            if 'sentiment_df' in st.session_state:
                sentiment_df = st.session_state['sentiment_df']
                st.subheader("📋 마지막 분석 결과 (감성 분류 데이터 - English Column)")
                st.dataframe(sentiment_df.head(100), use_container_width=True)
            else:
                st.info("👆 버튼을 클릭하여 감성 분석을 실행하세요.")
    
    # 탭 4: 시간 트렌드 (타임스탬프 저장 적용)
    with tabs[3]:
        st.header("📈 시간 트렌드 분석")
        interval = st.radio("시간 간격", ["D (일)", "W (주)", "M (월)"], horizontal=True, key="time_interval")
        interval_code = interval.split()[0]
        
        if st.button("🔍 시간 트렌드 분석", key="btn_time"):
            with st.spinner("시간 트렌드 분석 중..."):
                fig, trend_df = analyzer.time_trend(interval=interval_code)
                if fig:
                    st.pyplot(fig)
                    st.session_state['trend_df'] = trend_df 
                    
                    st.subheader("📋 트렌드 데이터 (English Column)")
                    st.dataframe(trend_df, use_container_width=True)
                    
                    # 🔴 [수정] 타임스탬프 저장
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv_file_name = f"time_trend_{timestamp}.csv"
                    trend_df.to_csv(os.path.join(SAVE_DIR, csv_file_name), index=False, encoding='utf-8-sig')
                    st.success(f"분석 결과가 서버 폴더 `{SAVE_DIR}`에 **{csv_file_name}**로 저장되었습니다.")

                    csv = trend_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                    st.download_button("💾 CSV 다운로드", csv, csv_file_name, "text/csv", key='download-trend-csv')
                else:
                    st.warning("published_at 컬럼이 없어 시간 트렌드 분석을 수행할 수 없습니다.")
        else:
            if 'trend_df' in st.session_state:
                trend_df = st.session_state['trend_df']
                st.subheader("📋 마지막 분석 결과 (트렌드 데이터 - English Column)")
                st.dataframe(trend_df, use_container_width=True)
            else:
                st.info("👆 시간 간격을 선택하고 버튼을 클릭하여 분석하세요.")
    
    # 탭 5: 동시출현 (타임스탬프 저장 적용)
    with tabs[4]:
        st.header("🔗 키워드 동시출현 분석")
        cooc_n = st.slider("분석할 키워드 개수", 5, 20, 15, key="cooc_n")
        
        if st.button("🔍 동시출현 분석", key="btn_cooc"):
            with st.spinner("동시출현 분석 중..."):
                fig, cooc_matrix = analyzer.cooccurrence(top_n=cooc_n)
                st.pyplot(fig)
                st.session_state['cooc_df'] = cooc_matrix 
                
                st.subheader("📋 동시출현 매트릭스 (English Index/Column)")
                st.dataframe(cooc_matrix, use_container_width=True)
                
                # 🔴 [수정] 타임스탬프 저장
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_file_name = f"cooccurrence_matrix_{timestamp}.csv"
                cooc_matrix.to_csv(os.path.join(SAVE_DIR, csv_file_name), encoding='utf-8-sig') 
                st.success(f"분석 결과가 서버 폴더 `{SAVE_DIR}`에 **{csv_file_name}**로 저장되었습니다.")
                
                csv = cooc_matrix.to_csv(encoding='utf-8-sig').encode('utf-8-sig')
                st.download_button("💾 CSV 다운로드", csv, csv_file_name, "text/csv", key='download-cooc-csv')
        else:
            if 'cooc_df' in st.session_state:
                cooc_matrix = st.session_state['cooc_df']
                st.subheader("📋 마지막 분석 결과 (동시출현 매트릭스 - English Index/Column)")
                st.dataframe(cooc_matrix, use_container_width=True)
            else:
                st.info("👆 키워드 개수를 선택하고 버튼을 클릭하여 분석하세요.")
    
    # 탭 6: 토픽 비교 (타임스탬프 저장 적용)
    with tabs[5]:
        st.header("🎬 영상별 토픽 비교")
        
        if st.button("🔍 토픽 비교 분석", key="btn_topic"):
            with st.spinner("토픽 비교 분석 중..."):
                fig, comparison_df = analyzer.topic_comparison()
                if fig:
                    st.pyplot(fig)
                    st.session_state['topic_df'] = comparison_df 
                    
                    st.subheader("📋 토픽 비교 데이터 (English Column)")
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # 🔴 [수정] 타임스탬프 저장
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv_file_name = f"topic_comparison_{timestamp}.csv"
                    comparison_df.to_csv(os.path.join(SAVE_DIR, csv_file_name), encoding='utf-8-sig') 
                    st.success(f"분석 결과가 서버 폴더 `{SAVE_DIR}`에 **{csv_file_name}**로 저장되었습니다.")

                    csv = comparison_df.to_csv(encoding='utf-8-sig').encode('utf-8-sig')
                    st.download_button("💾 CSV 다운로드", csv, csv_file_name, "text/csv", key='download-topic-csv')
                else:
                    st.warning("video_title 컬럼이 없어 토픽 비교 분석을 수행할 수 없습니다.")
        else:
            if 'topic_df' in st.session_state:
                comparison_df = st.session_state['topic_df']
                st.subheader("📋 마지막 분석 결과 (토픽 비교 데이터 - English Column)")
                st.dataframe(comparison_df, use_container_width=True)
            else:
                st.info("👆 버튼을 클릭하여 토픽 비교를 분석하세요.")

    # 탭 7: 원본 데이터 (다운로드 파일에 타임스탬프 적용)
    with tabs[6]:
        st.header("📋 원본 데이터")
        
        data_type = st.radio("데이터 유형 선택", ["댓글 데이터", "영상 데이터"], horizontal=True)
        
        # 댓글 데이터
        if data_type == "댓글 데이터":
            st.subheader("💬 댓글 데이터")
            st.dataframe(comments_df, use_container_width=True, height=600)
            
            # 🔴 [수정] 다운로드 파일에 타임스탬프 적용
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file_name = f"comments_data_{timestamp}.csv"
            
            csv = comments_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button(
                "💾 댓글 데이터 CSV 다운로드",
                csv,
                csv_file_name,
                "text/csv",
                key='download-comments-raw'
            )
        
        # 영상 데이터
        else:
            if videos_df is not None and not videos_df.empty:
                st.subheader("🎥 영상 데이터")
                st.dataframe(videos_df, use_container_width=True, height=600)
                
                # 🔴 [수정] 다운로드 파일에 타임스탬프 적용
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_file_name = f"videos_data_{timestamp}.csv"

                csv = videos_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                st.download_button(
                    "💾 영상 데이터 CSV 다운로드",
                    csv,
                    csv_file_name,
                    "text/csv",
                    key='download-videos-raw'
                )
            else:
                st.warning("영상 데이터가 없습니다.")

    # ========================================
    # 탭 8: 📄 일반 보고서 생성 섹션 (프롬프트 입력 기능 포함)
    # ========================================
    with tabs[7]:
        st.header("📄 Market Insight Report Generator (OpenAI API 기반)")
        st.write("분석 CSV 파일에 포함된 핵심 키워드와 통계를 기반으로 **사용자 지정 프롬프트**에 맞춘 요약 보고서를 자동 생성합니다.")
        st.warning("⚠️ **OpenAI API Key**가 `.env` 파일에 설정되어 있어야 하며, 보고서에 포함할 CSV 파일은 먼저 분석 탭에서 실행하고 **`analysis_results` 폴더에 저장**해야 합니다.")

        user_focus_prompt = st.text_area(
            "✍️ 보고서의 핵심 분석 주제 및 질문 (Focus Prompt)",
            value="Analyze the key positive and negative sentiment drivers in the selected datasets. What strategic recommendations can be derived from the time trend data for content planning?",
            height=150, key="report_general_prompt"
        )
        
        try:
            available_files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".csv")]
        except FileNotFoundError:
            available_files = []

        if not available_files:
            st.warning("분석 결과 CSV 파일이 없습니다. 먼저 위 분석 탭에서 분석을 실행하고 CSV를 `analysis_results` 폴더에 저장하세요.")
        else:
            selected_files = st.multiselect("📂 보고서에 포함할 파일 선택", available_files, default=available_files)

            if st.button("🧠 보고서 생성 실행"):
                report_sentences = [] 
                
                if not selected_files or not user_focus_prompt.strip() or not OPENAI_API_KEY:
                    st.error("입력값을 확인하세요. (파일 선택, 프롬프트 입력, API Key 확인)")
                    return
                
                raw_comments_df = st.session_state.get('comments_df')
                if raw_comments_df is None or raw_comments_df.empty:
                    st.error("댓글 원본 데이터가 세션에 없습니다. 데이터 수집 또는 로드를 확인하세요.")
                    return
                
                temp_analyzer = YouTubeCommentAnalyzer(raw_comments_df)
                if 'like_count' in raw_comments_df.columns:
                     raw_comments_df['LikeCount'] = raw_comments_df['like_count']
                
                _, _, sentiment_classified_df_full = temp_analyzer.sentiment_keywords() 


                with st.spinner("OpenAI GPT 모델이 보고서를 생성 중..."):
                    for f in selected_files:
                        file_path = os.path.join(SAVE_DIR, f)
                        try:
                            df = pd.read_csv(file_path, encoding="utf-8-sig")

                            keywords = f"File: {f}. "
                            
                            # 데이터 추출 및 키워드 문자열 생성 
                            if 'Frequency' in df.columns: 
                                top_keyword = df.iloc[0]['Keyword']
                                top_count = df.iloc[0]['Frequency']
                                keywords += f"Top keyword is '{top_keyword}' with count {top_count}. Total unique keywords: {len(df)}. "
                            
                            elif 'Sentiment' in df.columns: 
                                sentiment_counts = df['Sentiment'].value_counts()
                                pos = sentiment_counts.get('긍정', 0)
                                neg = sentiment_counts.get('부정', 0)
                                total = len(df)
                                pos_ratio = pos / total * 100 if total > 0 else 0
                                
                                # 긍정 댓글 샘플링 (LikeCount 기준)
                                positive_samples = sentiment_classified_df_full[
                                    (sentiment_classified_df_full['Sentiment'] == '긍정') & 
                                    (sentiment_classified_df_full['LikeCount'] > 0) 
                                ].sort_values(by='LikeCount', ascending=False)['Text'].head(3).tolist()
                                
                                if positive_samples:
                                    clean_samples = [temp_analyzer.preprocess_text(s) for s in positive_samples]
                                    sample_text = "Sample positive comments (high like count): " + " | ".join(clean_samples)
                                    keywords += sample_text + " "

                                keywords += f"Total comments {total}. Positive comments: {pos} ({pos_ratio:.1f}%). Negative comments: {neg}. The overall sentiment is mostly Positive. "
                            
                            elif 'CommentCount' in df.columns: 
                                df['Date'] = pd.to_datetime(df['Date'])
                                max_comments_date = df.loc[df['CommentCount'].idxmax(), 'Date'].strftime('%Y-%m-%d')
                                max_comments_count = df['CommentCount'].max()
                                keywords += f"Peak comment count {max_comments_count} occurred on {max_comments_date}. Average comments per period is {df['CommentCount'].mean():.1f}. "
                            
                            elif 'Keyword1' in df.columns and 'Keyword2' in df.columns: 
                                keywords += f"Co-occurrence matrix data. Analyzing relationships between {len(df)} keywords. "
                            
                            elif 'Keyword1' in df.columns: # Topic Comparison
                                top_video_topic = df.index[0]
                                key_terms = [str(x) for x in df.iloc[0].dropna().tolist()]
                                keywords += f"The top video topic is '{top_video_topic}' with key terms: {', '.join(key_terms)}. "
                            
                            elif 'text' in df.columns and 'like_count' in df.columns:
                                total_comments = len(df)
                                avg_likes = df['like_count'].mean()
                                top_comments = df.sort_values(by='like_count', ascending=False)['text'].head(3).tolist()
                                clean_samples = [temp_analyzer.preprocess_text(s) for s in top_comments]
                                top_comment_text = " | ".join(clean_samples)
                                keywords += f"Raw comment data summary. Total records: {total_comments}. Average likes per comment: {avg_likes:.1f}. Top comments by like count: {top_comment_text}. "
                            
                            else: 
                                keywords += f"Dataset rows: {len(df)}. Columns: {', '.join(df.columns)}. Data statistics available. "
                            
                            sentence = generate_openai_report(keywords, user_focus_prompt, OPENAI_API_KEY) 
                            report_sentences.append(f"**{f} Insight:** {sentence}") 
                            
                        except Exception as e:
                            st.error(f"파일 {f} 처리 오류: CSV 파일 구조 확인 필요. {str(e)}")
                            continue

                if report_sentences:
                    summary = "\n\n".join(report_sentences)
                    
                    final_report = f"""
# YouTube Analysis Auto-Generated Report
## Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
## User Focus Prompt: {user_focus_prompt}
---
{summary}
"""
                    st.subheader("📈 AI 자동 생성 보고서 초안")
                    st.text_area("요약 결과", final_report, height=400)
                    
                    st.download_button(
                        "💾 요약 보고서 저장",
                        final_report.encode("utf-8-sig"),
                        "Market_Insight_Report.txt",
                        "text/plain"
                    )
                else:
                    st.error("보고서 생성 실패. 파일 선택 및 구조를 확인하세요.")

    # ========================================
    # 탭 9: 💼 임원진 보고서 (Executive Summary)
    # ========================================
    with tabs[8]:
        st.header("💼 임원진 보고서 (Executive Summary)")
        st.write("핵심 데이터를 기반으로 **국문 및 영문**으로 분리된, 임원진 제출용으로 적합한 요약 보고서를 생성합니다.")
        st.warning("⚠️ 이 보고서 생성을 위해서는 **OpenAI API Key**가 필수이며, 분석 탭에서 CSV 파일을 **`analysis_results`** 폴더에 저장해야 합니다.")
        
        # 🔴 [추가] 임원진 보고서 프롬프트 입력 영역
        user_exec_prompt = st.text_area(
            "✍️ 임원진 보고서의 핵심 분석 주제 및 질문 (Focus Prompt)",
            value="Identify the 3 most critical market insights regarding K-Beauty trends and propose concise strategic actions for brand positioning based on the competitive analysis.",
            height=100, key="report_exec_prompt"
        )
        
        # SAVE_DIR에서 파일 목록 가져오기
        try:
            available_files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".csv")]
        except FileNotFoundError:
            available_files = []

        if not available_files:
            st.warning("분석 결과 CSV 파일이 없습니다. 먼저 분석 탭에서 파일을 저장하세요.")
        else:
            selected_exec_files = st.multiselect("📂 보고서에 포함할 파일 선택", available_files, default=available_files, key="report_exec_files")

            if st.button("🧠 국/영문 임원진 보고서 생성", key="btn_generate_exec_report"):
                
                if not selected_exec_files or not user_exec_prompt.strip() or not OPENAI_API_KEY:
                    st.error("입력값을 확인하세요. (파일 선택, 프롬프트 입력, API Key 확인)")
                    return

                # --- 1. 통합된 키워드/데이터 요약 문자열 생성 ---
                full_keywords_for_exec = ""
                with st.spinner("OpenAI GPT 모델이 국/영문 보고서를 생성하기 위해 데이터 준비 중..."):
                    
                    # Sentiment 분석을 위한 기본 Analyzer 객체 생성 및 분류 데이터 준비
                    raw_comments_df = st.session_state.get('comments_df')
                    temp_analyzer = YouTubeCommentAnalyzer(raw_comments_df)
                    if 'like_count' in raw_comments_df.columns:
                         raw_comments_df['LikeCount'] = raw_comments_df['like_count']
                    _, _, sentiment_classified_df_full = temp_analyzer.sentiment_keywords() 
                    
                    # 모든 선택된 파일의 데이터를 하나의 키워드 문자열로 조합 
                    for f in selected_exec_files:
                        file_path = os.path.join(SAVE_DIR, f)
                        keywords_chunk = f"File: {f}. "
                        try:
                            df = pd.read_csv(file_path, encoding="utf-8-sig")

                            # --- 키워드 추출 로직 (일반 보고서 탭과 동일) ---
                            if 'Frequency' in df.columns: 
                                keywords_chunk += f"Top keyword is '{df.iloc[0]['Keyword']}' with count {df.iloc[0]['Frequency']}. Total unique keywords: {len(df)}. "
                            
                            elif 'Sentiment' in df.columns: 
                                sentiment_counts = df['Sentiment'].value_counts()
                                pos_ratio = sentiment_counts.get('긍정', 0) / len(df) * 100 if len(df) > 0 else 0
                                keywords_chunk += f"Total comments {len(df)}. Positive ratio: {pos_ratio:.1f}%. "
                            
                            elif 'CommentCount' in df.columns: 
                                max_count = df['CommentCount'].max()
                                keywords_chunk += f"Peak comment count {max_count}. Average count per period is {df['CommentCount'].mean():.1f}. "
                            
                            elif 'Keyword1' in df.columns and 'Keyword2' in df.columns: 
                                keywords_chunk += f"Co-occurrence matrix data. Analyzing relationships between {len(df)} keywords. "

                            elif 'Keyword1' in df.columns: # Topic Comparison
                                keywords_chunk += f"The top video topic is '{df.index[0]}' with key terms: {', '.join([str(x) for x in df.iloc[0].dropna().tolist()])}. "
                            
                            elif 'text' in df.columns and 'like_count' in df.columns:
                                avg_likes = df['like_count'].mean()
                                keywords_chunk += f"Raw comment data summary. Total records: {len(df)}. Average likes per comment: {avg_likes:.1f}. "
                            
                            else: keywords_chunk += f"Dataset rows: {len(df)}. Columns: {', '.join(df.columns)}. "
                            
                            full_keywords_for_exec += keywords_chunk
                        except Exception as e:
                            st.error(f"파일 {f} 처리 오류: {str(e)}")
                            continue
                            
                # --- 2. 통합된 키워드와 사용자 프롬프트를 바탕으로 임원진 보고서 생성 ---
                with st.spinner("OpenAI GPT 모델이 국/영문 보고서를 생성 중..."):
                    english_summary, korean_summary = generate_executive_report(full_keywords_for_exec, user_exec_prompt, OPENAI_API_KEY)
                    
                    # 3. 결과 출력 및 다운로드
                    
                    korean_final_report = f"""
# 🔴 YouTube 임원진 보고서 (국문)
## 작성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
## 분석 주제: {user_exec_prompt}
---
### 핵심 요약 (Korean Summary)
{korean_summary}

---
### 분석에 사용된 데이터 파일
{', '.join(selected_exec_files)}
"""
                    english_final_report = f"""
# 🔴 YouTube Executive Summary (English)
## Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
## Focus Prompt: {user_exec_prompt}
---
### Executive Summary
{english_summary}

---
### Data Files Used
{', '.join(selected_exec_files)}
"""
                    
                    st.markdown("---")
                    st.subheader("🇰🇷 국문 보고서 초안")
                    st.text_area("국문 요약 결과", korean_final_report, height=300)
                    
                    st.subheader("🇺🇸 영문 보고서 초안")
                    st.text_area("영문 요약 결과", english_final_report, height=300)
                    
                    col_kor, col_eng = st.columns(2)
                    
                    with col_kor:
                        st.download_button(
                            "💾 국문 보고서 다운로드 (KR.txt)",
                            korean_final_report.encode("utf-8-sig"),
                            "Executive_Report_KR.txt",
                            "text/plain"
                        )
                    with col_eng:
                        st.download_button(
                            "💾 영문 보고서 다운로드 (EN.txt)",
                            english_final_report.encode("utf-8-sig"),
                            "Executive_Report_EN.txt",
                            "text/plain"
                        )
                    
                    if "Parsing Error" in english_summary:
                        st.error("보고서 파싱에 실패했습니다. OpenAI 출력 형식을 확인하세요.")


if __name__ == "__main__":
    main()