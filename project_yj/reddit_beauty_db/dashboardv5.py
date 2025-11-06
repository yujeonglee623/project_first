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
import requests 
import io 
from prawcore.exceptions import ResponseException, RequestException

# ========================================
# Streamlit 기본 설정 및 OpenAI 설정
# ========================================

# 페이지 설정
st.set_page_config(
    page_title="Reddit 분석 대시보드",
    page_icon="🔴",
    layout="wide"
)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 보고서 저장을 위한 디렉토리 설정 및 자동 생성
SAVE_DIR = "analysis_results"
os.makedirs(SAVE_DIR, exist_ok=True)


def generate_openai_report(keywords, user_prompt, api_key, model_name="gpt-4o"):
    """사용자 프롬프트와 분석 데이터를 조합하여 OpenAI API를 이용한 일반 보고서 문장 생성 (영문)"""
    
    if not api_key:
        return "Error: OpenAI API Key is missing. Please set the OPENAI_API_KEY in the .env file."

    # System Prompt: AI의 역할과 원하는 출력 형식을 명확히 정의
    system_prompt = (
        "You are a professional Social Media Market Analyst. "
        "Your task is to analyze the provided raw data summary or statistical analysis "
        "and generate a comprehensive, insightful, and professional English summary based *strictly* on the user's focus prompt. "
        "The summary should be concise (around 5 detailed sentences) and must address the user's specific questions/focus. "
        "If Korean content is provided, translate and interpret it within the context of the user's request. "
        "Do not use markdown headers or lists. Just provide the summary text."
    )
    
    # User Prompt: 사용자 입력과 분석 데이터(keywords)를 합쳐서 전달
    full_user_prompt = (
        f"**User Focus:** {user_prompt}\n\n"
        f"**Analysis Data for Context:** {keywords}\n\n"
        "Generate the insightful report summary in English, focusing on the User Focus above."
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
            {"role": "user", "content": full_user_prompt}
        ],
        "max_tokens": 400,
        "temperature": 0.3, 
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=40)
        response.raise_for_status()
        result = response.json()
        summary = result['choices'][0]['message']['content'].strip()
        return summary
    except Exception as e:
        return f"API Error occurred: {e}. Check API key or rate limits."


def generate_executive_report(keywords, user_prompt, api_key, model_name="gpt-4o"):
    """사용자 프롬프트와 분석 데이터를 조합하여 OpenAI API를 이용한 임원진 보고서 생성 (국문+영문 분리)"""
    
    if not api_key:
        return "Error: OpenAI API Key is missing."

    # System Prompt: AI의 역할과 원하는 출력 형식을 명확히 정의 (국문/영문 분리 요청)
    system_prompt = (
        "You are a Senior Executive Market Analyst. "
        "Your task is to analyze the provided data and generate a highly summarized, actionable **Executive Report** based on the user's focus. "
        "The response MUST contain two clearly separated sections: **English Summary** (5-7 sentences) and its **Korean Summary** (5-7 sentences). "
        "The output format MUST strictly follow the structure: 'English Summary: [English Text] Korean Summary: [Korean Text]'. "
        "Focus on key insights, strategic implications, and high-level trends. "
    )
    
    # User Prompt: 사용자 입력과 분석 데이터(keywords)를 합쳐서 전달
    full_user_prompt = (
        f"**User Focus for Executive Report:** {user_prompt}\n\n"
        f"**Analysis Data for Context:** {keywords}\n\n"
        "Generate the Executive Report in the required dual-language format."
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # API Payload (토큰을 조금 더 늘려 국문/영문 모두 생성 가능하도록 함)
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_user_prompt}
        ],
        "max_tokens": 800, # 두 언어를 모두 담기 위해 토큰 증가
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
# Reddit 분석 클래스
# ========================================
class RedditAnalyzer:
    """Reddit 분석 클래스"""
    
    def __init__(self, posts_df, comments_df=None):
        self.posts_df = posts_df.copy()
        self.comments_df = comments_df.copy() if comments_df is not None and not comments_df.empty else None
        
        # 날짜 컬럼 변환 
        # API 수집 시 created_utc가 UTC 타임스탬프(초)로 들어와야 하지만, 
        # 이미 datetime 객체로 변환된 경우를 위해 오류 처리 추가
        if 'created_utc' in self.posts_df.columns:
            # 문자열 형태일 경우 pd.to_datetime으로 변환 시도
            if self.posts_df['created_utc'].dtype == 'object':
                 self.posts_df['created_utc'] = pd.to_datetime(self.posts_df['created_utc'], errors='coerce') 
            # 정수형태일 경우 UTC(초)로 간주하고 변환 시도
            elif self.posts_df['created_utc'].dtype in ['int64', 'float64']:
                 self.posts_df['created_utc'] = pd.to_datetime(self.posts_df['created_utc'], unit='s', errors='coerce') 

        if self.comments_df is not None and 'created_utc' in self.comments_df.columns:
             # 문자열 형태일 경우 pd.to_datetime으로 변환 시도
            if self.comments_df['created_utc'].dtype == 'object':
                 self.comments_df['created_utc'] = pd.to_datetime(self.comments_df['created_utc'], errors='coerce') 
            # 정수형태일 경우 UTC(초)로 간주하고 변환 시도
            elif self.comments_df['created_utc'].dtype in ['int64', 'float64']:
                 self.comments_df['created_utc'] = pd.to_datetime(self.comments_df['created_utc'], unit='s', errors='coerce')
    
    
    def preprocess_text(self, text):
        """텍스트 전처리"""
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^가-힣a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    
    def extract_keywords(self, text_series, top_n=50):
        """키워드 추출"""
        all_text = ' '.join(text_series.fillna('').apply(self.preprocess_text))
        words = all_text.split()
        words = [w for w in words if len(w) >= 2]
        
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had',
                     'i', 'me', 'my', 'you', 'your', 'it', 'its', 'not', 'no', 'yes', 'we',
                     '그', '이', '저', '것', '수', '등', '들', '및', '또한', '하다', '있다', '되다',
                     '이것', '그것', '저것', '그런', '이런', '저런', 'removed', 'deleted'}
        
        words = [w for w in words if w not in stopwords]
        word_freq = Counter(words)
        
        return word_freq.most_common(top_n)
    
    
    def wordcloud(self, text_series, width=1200, height=800):
        """워드클라우드 생성"""
        all_text = ' '.join(text_series.fillna('').apply(self.preprocess_text))
        
        try:
             font_path = 'C:/Windows/Fonts/malgun.ttf' 
        except:
             font_path = None

        wordcloud = WordCloud(
            font_path=font_path,
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
        
        if not keywords:
             return None, pd.DataFrame(columns=['키워드', '빈도'])

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
    
    
    def sentiment_analysis(self, text_series, data_df):
        """감성 분석"""
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'best', 'love', 'awesome', 'perfect', 'nice', 'happy', 'thank',
            '좋다', '최고', '대박', '예쁘다', '멋지다', '완벽', '감사', '행복', '좋아요'
        }
        
        negative_words = {
            'bad', 'worst', 'terrible', 'awful', 'horrible', 'hate',
            'poor', 'disappointing', 'useless', 'waste', 'crap',
            '싫다', '별로', '나쁘다', '최악', '형편없다', '실망', '별로네'
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
        
        sentiments = text_series.fillna('').apply(calculate_sentiment)
        
        df_copy = data_df.copy().reset_index(drop=True)
        df_copy['Sentiment'] = sentiments
        sentiment_counts = sentiments.value_counts()
        
        if sentiment_counts.empty:
            return None, pd.Series(), df_copy[['Sentiment']]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        colors = ['#90EE90', '#FFB6C1', '#D3D3D3']
        
        order = ['긍정', '부정', '중립']
        ordered_counts = sentiment_counts.reindex(order, fill_value=0)
        ordered_counts = ordered_counts[ordered_counts > 0]
        ordered_colors = [c for s, c in zip(order, colors) if s in ordered_counts.index]

        axes[0].pie(ordered_counts.values, labels=ordered_counts.index, 
                      autopct='%1.1f%%', colors=ordered_colors, startangle=90)
        axes[0].set_title('감성 분포', fontsize=14, pad=20)
        
        axes[1].bar(ordered_counts.index, ordered_counts.values, color=ordered_colors)
        axes[1].set_xlabel('감성', fontsize=12)
        axes[1].set_ylabel('개수', fontsize=12)
        axes[1].set_title('감성별 개수', fontsize=14, pad=20)
        
        plt.tight_layout()
        
        sentiment_df = df_copy.rename(columns={'title': '제목', 'selftext': '본문', 'body': '본문_또는_내용', 'score': '점수'})
        
        if text_series.name == 'title':
            sentiment_df = sentiment_df[['Sentiment', '제목', '점수']]
        elif text_series.name == 'selftext':
            sentiment_df = sentiment_df[['Sentiment', '본문', '점수']]
        else: # 댓글 본문
            sentiment_df = sentiment_df[['Sentiment', '본문_또는_내용', '점수']]
            
        return fig, sentiment_counts, sentiment_df
    
    
    def time_trend(self, df, date_col='created_utc', interval='D'):
        """시간대별 트렌드 분석"""
        if date_col not in df.columns:
            return None, None
        
        # 1. 날짜 컬럼을 에러를 무시하고 datetime 타입으로 강제 변환 (CSV 업로드 대비)
        df_valid = df.copy()
        df_valid[date_col] = pd.to_datetime(df_valid[date_col], errors='coerce') 

        # 2. 유효한 날짜가 있는 행만 사용
        df_valid = df_valid.dropna(subset=[date_col])
        if df_valid.empty:
            st.warning("유효한 날짜 데이터가 없어 시간 트렌드 분석을 수행할 수 없습니다.")
            return None, None
        
        # 3. 날짜 컬럼을 인덱스로 설정 (이때 DatetimeIndex가 생성됨)
        df_sorted = df_valid.reset_index(drop=True).set_index(date_col).sort_index()

        time_counts = df_sorted.resample(interval).size()
        
        # 'score' 컬럼이 df_sorted에 있는지 확인
        if 'score' in df_sorted.columns:
            time_scores = df_sorted['score'].resample(interval).sum()
        else:
            time_scores = None
        
        fig = None
        
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
        if 'subreddit' not in self.posts_df.columns or self.posts_df['subreddit'].nunique() < 2:
            return None, None
        
        subreddit_stats = self.posts_df.groupby('subreddit').agg({
            'score': ['mean', 'sum', 'count'],
            'num_comments': 'mean'
        }).round(2)
        
        subreddit_stats.columns = ['평균_점수', '총_점수', '게시물_수', '평균_댓글수']
        subreddit_stats.index.name = '서브레딧'
        subreddit_stats = subreddit_stats.sort_values('총_점수', ascending=False)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        axes[0, 0].bar(subreddit_stats.index, subreddit_stats['게시물_수'], color='orangered')
        axes[0, 0].set_title('서브레딧별 게시물 수', fontsize=14)
        axes[0, 0].set_ylabel('게시물 수')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].bar(subreddit_stats.index, subreddit_stats['총_점수'], color='coral')
        axes[0, 1].set_title('서브레딧별 총 점수', fontsize=14)
        axes[0, 1].set_ylabel('총 점수')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        axes[1, 0].bar(subreddit_stats.index, subreddit_stats['평균_점수'], color='tomato')
        axes[1, 0].set_title('서브레딧별 평균 점수', fontsize=14)
        axes[1, 0].set_ylabel('평균 점수')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        axes[1, 1].bar(subreddit_stats.index, subreddit_stats['평균_댓글수'], color='lightsalmon')
        axes[1, 1].set_title('서브레딧별 평균 댓글 수', fontsize=14)
        axes[1, 1].set_ylabel('평균 댓글 수')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        return fig, subreddit_stats


def search_and_collect_reddit_data(subreddit_names, search_query, post_limit, sort_by, time_filter, collect_comments, comment_limit):
    # (API 수집 함수)
    load_dotenv()
    
    # User-Agent를 .env 파일에서 가져오지 못할 경우 기본값 설정
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    # PRAW는 client_id와 secret을 사용하는 인증(Application Only) 시에도 User-Agent를 요구함
    user_agent = os.getenv("REDDIT_USER_AGENT", "RedditAnalyzer_Streamlit_v1.0 (by Python)") 
    
    # 사용자 이름과 비밀번호를 추가로 로드 (User/Password 인증 시 필요)
    username = os.getenv("REDDIT_USERNAME")
    password = os.getenv("REDDIT_PASSWORD")
    
    if not client_id or not client_secret:
        st.error("Reddit API 자격증명이 설정되지 않았습니다. .env 파일을 확인하세요.")
        st.info("필요한 환경 변수: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME(선택), REDDIT_PASSWORD(선택)")
        return None, None
    
    try:
        if username and password:
            # User/Password 인증 (가장 일반적인 스크립트 접근 방식)
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
                username=username,
                password=password
            )
        else:
            # Application Only 인증 (읽기 전용, 일부 기능 제한될 수 있음)
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
    
    subreddit_list = [s.strip() for s in subreddit_names.split(',') if s.strip()]
    total_subreddits = len(subreddit_list)
    
    if total_subreddits == 0:
        st.warning("수집할 서브레딧 이름이 입력되지 않았습니다.")
        return None, None

    for idx, subreddit_name in enumerate(subreddit_list):
        status_text.text(f"서브레딧 {idx+1}/{total_subreddits}: r/{subreddit_name} 수집 중...")
        
        try:
            try:
                subreddit = reddit.subreddit(subreddit_name)
                # 서브레딧 존재 여부 확인을 위해 title에 접근 시도
                _ = subreddit.title 
            except ResponseException as e:
                st.warning(f"서브레딧 r/{subreddit_name}에 접근할 수 없습니다. 스킵합니다. 오류: {e}")
                progress_bar.progress((idx + 1) / total_subreddits)
                continue

            # 검색 쿼리가 있는 경우 search() 사용
            if search_query:
                posts = subreddit.search(search_query, sort=sort_by, time_filter=time_filter, limit=post_limit)
            # 검색 쿼리가 없는 경우 정렬 방식에 따라 해당 서브레딧의 글 목록 가져옴
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
                    posts = subreddit.hot(limit=post_limit) # 기본값

            post_count = 0
            for post in posts:
                post_data = {
                    'post_id': post.id,
                    'subreddit': post.subreddit.display_name if hasattr(post.subreddit, 'display_name') else subreddit_name,
                    'title': post.title,
                    'selftext': post.selftext,
                    'author': str(post.author) if post.author else '[deleted]',
                    'score': post.score,
                    'upvote_ratio': post.upvote_ratio,
                    'num_comments': post.num_comments,
                    'created_utc': post.created_utc,
                    'url': post.url,
                    'permalink': f"https://reddit.com{post.permalink}"
                }
                all_posts.append(post_data)
                post_count += 1
                
                if collect_comments and post.num_comments > 0:
                    try:
                        # 'More Comments' 객체를 제거하고 실제 댓글만 가져옴
                        post.comments.replace_more(limit=0)
                        comments = post.comments.list()[:comment_limit]
                        
                        for comment in comments:
                            if hasattr(comment, 'body') and comment.body is not None:
                                comment_data = {
                                    'comment_id': comment.id,
                                    'post_id': post.id,
                                    'subreddit': post.subreddit.display_name if hasattr(post.subreddit, 'display_name') else subreddit_name,
                                    'author': str(comment.author) if comment.author else '[deleted]',
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
    
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # 사이드바 - 데이터 수집/업로드
    st.sidebar.header("📂 데이터 소스")
    data_source = st.sidebar.radio(
        "데이터 입력 방식 선택",
        ["API로 실시간 수집", "CSV 파일 업로드"]
    )
    
    # 세션 상태 초기화 및 로드
    if 'posts_df' not in st.session_state:
        st.session_state['posts_df'] = pd.DataFrame(columns=['post_id', 'title', 'subreddit', 'score', 'num_comments'])
    if 'comments_df' not in st.session_state:
        st.session_state['comments_df'] = None 
    
    posts_df = st.session_state['posts_df']
    comments_df = st.session_state['comments_df']
    
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
            help="비워두면 서브레딧의 'sort_by'에 해당하는 게시물 수집"
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
                posts_df_new, comments_df_new = search_and_collect_reddit_data(
                    subreddit_names, search_query, post_limit, 
                    sort_by, time_filter, collect_comments, comment_limit
                )
            
            if posts_df_new is not None and not posts_df_new.empty:
                st.success(f"✅ 게시물 **{len(posts_df_new):,}**개 수집 완료!")

                # ========================================
                # 🔴 [수정 반영] 타임스탬프 CSV 저장 로직 추가
                # ========================================
                
                # 현재 시각을 YYYYMMDD_HHMMSS 형식으로 포맷
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # 1. 게시물 데이터 저장 (필수)
                posts_file_name = f"posts_data_{timestamp}.csv"
                posts_file_path = os.path.join(SAVE_DIR, posts_file_name)
                posts_df_new.to_csv(posts_file_path, index=False, encoding='utf-8-sig')
                st.success(f"💾 게시물 데이터가 **{posts_file_name}**로 저장되었습니다.")
                
                # 🔴 디버깅 라인 추가
                current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.write(f"디버깅: 생성된 타임스탬프 문자열: {current_time_str}") # Streamlit 화면에 출력
                
                timestamp = current_time_str # 생성된 문자열을 사용
                # ... (나머지 저장 로직 유지)

                # 2. 댓글 데이터 저장 (선택)
                if comments_df_new is not None and not comments_df_new.empty:
                    st.success(f"✅ 댓글 **{len(comments_df_new):,}**개 수집 완료!")
                    comments_file_name = f"comments_data_{timestamp}.csv"
                    comments_file_path = os.path.join(SAVE_DIR, comments_file_name)
                    comments_df_new.to_csv(comments_file_path, index=False, encoding='utf-8-sig')
                    st.success(f"💾 댓글 데이터가 **{comments_file_name}**로 저장되었습니다.")
                
                # ========================================
                
                st.session_state['posts_df'] = posts_df_new
                st.session_state['comments_df'] = comments_df_new
                st.rerun()
            else:
                st.warning("수집된 데이터가 없습니다.")
    
    else:  # CSV 파일 업로드
        st.sidebar.subheader("📤 파일 업로드")
        posts_file = st.sidebar.file_uploader("게시물 CSV 파일", type=['csv'], key="posts_upload")
        comments_file = st.sidebar.file_uploader("댓글 CSV 파일 (선택)", type=['csv'], key="comments_upload")
        
        if posts_file:
            posts_df_loaded = pd.read_csv(posts_file)
            st.session_state['posts_df'] = posts_df_loaded
            st.sidebar.success(f"✅ 게시물 **{len(posts_df_loaded):,}**개 로드")
        
        if comments_file:
            comments_df_loaded = pd.read_csv(comments_file)
            st.session_state['comments_df'] = comments_df_loaded
            st.sidebar.success(f"✅ 댓글 **{len(comments_df_loaded):,}**개 로드")
        
        if posts_file or comments_file:
            st.rerun()

    
    posts_df = st.session_state['posts_df']
    comments_df = st.session_state['comments_df']

    if posts_df.empty:
        st.info("👆 왼쪽 사이드바에서 데이터를 수집하거나 업로드해주세요.")
        return 

    # 기본 통계 (생략: 기존 코드 유지)
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

    # 탭으로 분석 모드 구분 (생략: 기존 코드 유지)
    tabs = st.tabs([
    "☁️ 워드클라우드",
    "📊 키워드 빈도",
    "😊😢 감성 분석",
    "📈 시간 트렌드",
    "🎯 서브레딧 비교",
    "📋 원본 데이터",
    "📄 보고서 생성",
    "💼 임원진 보고서"
    ])

    analyzer = RedditAnalyzer(posts_df, comments_df)
    
    # 텍스트 분석에 사용할 수 있는 데이터프레임 확인
    text_sources_available = ["게시물 제목"]
    if 'selftext' in posts_df.columns and not posts_df['selftext'].isnull().all():
        text_sources_available.append("게시물 본문")
    if comments_df is not None and 'body' in comments_df.columns and not comments_df['body'].isnull().all():
        text_sources_available.append("댓글")


    # 탭 1~8은 기존 로직 유지
    
    with tabs[0]: # 워드클라우드
        st.header("☁️ 워드클라우드")
        
        if not text_sources_available:
            st.warning("텍스트 데이터(제목, 본문, 댓글)가 없어 분석을 수행할 수 없습니다.")
        else:
            text_source = st.radio("텍스트 소스", text_sources_available, horizontal=True, key="wordcloud_source")

            if st.button("🔍 워드클라우드 생성", key="btn_wordcloud"):
                with st.spinner(f"{text_source} 워드클라우드 생성 중..."):
                    if text_source == "게시물 제목":
                        fig = analyzer.wordcloud(posts_df['title'])
                    elif text_source == "게시물 본문":
                        fig = analyzer.wordcloud(posts_df['selftext'])
                    else: fig = analyzer.wordcloud(comments_df['body'])
                    st.pyplot(fig)
            else: st.info("👆 텍스트 소스를 선택하고 버튼을 클릭하세요.")

    with tabs[1]: # 키워드 빈도
        st.header("📊 키워드 빈도 분석")
        
        if not text_sources_available: st.warning("텍스트 데이터(제목, 본문, 댓글)가 없어 분석을 수행할 수 없습니다.")
        else:
            text_source = st.radio("텍스트 소스", text_sources_available, horizontal=True, key="keyword_source")
            top_n = st.slider("표시할 키워드 개수", 10, 50, 20, key="keyword_top_n")

            if st.button("🔍 키워드 빈도 분석", key="btn_keyword"):
                with st.spinner(f"{text_source} 키워드 빈도 분석 중..."):
                    if text_source == "게시물 제목": fig, freq_df = analyzer.keyword_frequency(posts_df['title'], top_n=top_n)
                    elif text_source == "게시물 본문": fig, freq_df = analyzer.keyword_frequency(posts_df['selftext'], top_n=top_n)
                    else: fig, freq_df = analyzer.keyword_frequency(comments_df['body'], top_n=top_n)
                    
                    if fig:
                        st.pyplot(fig)
                        st.subheader("📋 키워드 데이터")
                        st.dataframe(freq_df, use_container_width=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        csv_file_name = f"reddit_keyword_frequency_{timestamp}.csv"
                        freq_df.to_csv(os.path.join(SAVE_DIR, csv_file_name), index=False, encoding='utf-8-sig')
                        st.session_state['freq_df_report'] = freq_df 
                        csv_data = freq_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                        st.download_button("💾 CSV 다운로드", csv_data, csv_file_name, "text/csv", key='download-keyword-csv')
                        st.success(f"분석 결과가 서버 폴더 `{SAVE_DIR}`에 **{csv_file_name}**로 저장되었습니다.")
                    else: st.warning("분석할 유효한 키워드가 없습니다.")
            else:
                st.info("👆 텍스트 소스를 선택하고 버튼을 클릭하세요.")
                if 'freq_df_report' in st.session_state:
                    st.subheader("📋 마지막 분석 결과 (키워드 데이터)")
                    st.dataframe(st.session_state['freq_df_report'], use_container_width=True)


    with tabs[2]: # 감성 분석
        st.header("😊😢 감성 분석")
        
        if not text_sources_available: st.warning("텍스트 데이터(제목, 본문, 댓글)가 없어 분석을 수행할 수 없습니다.")
        else:
            text_source = st.radio("텍스트 소스", text_sources_available, horizontal=True, key="sentiment_source")

            if st.button("🔍 감성 분석 실행", key="btn_sentiment"):
                with st.spinner(f"{text_source} 감성 분석 중..."):
                    if text_source == "게시물 제목": fig, sentiment_counts, sentiment_df = analyzer.sentiment_analysis(posts_df['title'], posts_df)
                    elif text_source == "게시물 본문": fig, sentiment_counts, sentiment_df = analyzer.sentiment_analysis(posts_df['selftext'], posts_df)
                    else: fig, sentiment_counts, sentiment_df = analyzer.sentiment_analysis(comments_df['body'], comments_df)
                    
                    if fig:
                        st.pyplot(fig)
                        st.subheader("📊 감성 요약")
                        col1_s, col2_s, col3_s = st.columns(3)
                        for idx, sentiment in enumerate(['긍정', '부정', '중립']):
                             count = sentiment_counts.get(sentiment, 0)
                             with [col1_s, col2_s, col3_s][idx]: st.metric(sentiment, f"{count:,}개")
                                 
                        st.subheader("📋 감성 분류 데이터 (상위 100개)")
                        st.dataframe(sentiment_df.head(100), use_container_width=True)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        csv_file_name = f"reddit_sentiment_analysis_{timestamp}.csv"
                        sentiment_df.to_csv(os.path.join(SAVE_DIR, csv_file_name), index=False, encoding='utf-8-sig')
                        st.session_state['sentiment_df_report'] = sentiment_df 
                        csv_data = sentiment_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                        st.download_button("💾 CSV 다운로드", csv_data, csv_file_name, "text/csv", key='download-sentiment-csv')
                        st.success(f"분석 결과가 서버 폴더 `{SAVE_DIR}`에 **{csv_file_name}**로 저장되었습니다.")
                    else: st.warning("감성 분석을 수행할 텍스트가 부족하거나 없습니다.")
            else:
                st.info("👆 텍스트 소스를 선택하고 버튼을 클릭하세요.")
                if 'sentiment_df_report' in st.session_state:
                    st.subheader("📋 마지막 분석 결과 (감성 분류 데이터 - 상위 100개)")
                    st.dataframe(st.session_state['sentiment_df_report'].head(100), use_container_width=True)


    with tabs[3]: # 시간 트렌드
        st.header("📈 시간 트렌드 분석")
        data_source_trend_options = ["게시물"]
        if comments_df is not None: data_source_trend_options.append("댓글")
        data_source_trend = st.radio("데이터 소스", data_source_trend_options, horizontal=True)
        interval = st.radio("시간 간격", ["D (일)", "W (주)", "M (월)"], horizontal=True, key="time_interval")
        interval_code = interval.split()[0]

        if st.button("🔍 시간 트렌드 분석", key="btn_time"):
            with st.spinner(f"{data_source_trend} 시간 트렌드 분석 중..."):
                if data_source_trend == "게시물": fig, trend_df = analyzer.time_trend(posts_df, interval=interval_code)
                else: fig, trend_df = analyzer.time_trend(comments_df, interval=interval_code)
                
                if fig:
                    st.pyplot(fig)
                    st.subheader("📋 트렌드 데이터")
                    st.dataframe(trend_df, use_container_width=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv_file_name = f"reddit_time_trend_{timestamp}.csv"
                    trend_df.to_csv(os.path.join(SAVE_DIR, csv_file_name), index=False, encoding='utf-8-sig')
                    st.session_state['trend_df_report'] = trend_df 
                    csv_data = trend_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                    st.download_button("💾 CSV 다운로드", csv_data, csv_file_name, "text/csv", key='download-trend-csv')
                    st.success(f"분석 결과가 서버 폴더 `{SAVE_DIR}`에 **{csv_file_name}**로 저장되었습니다.")
                else: st.warning("날짜 정보가 없거나 유효한 데이터가 없어 시간 트렌드 분석을 수행할 수 없습니다.")
        else:
            st.info("👆 데이터 소스와 시간 간격을 선택하고 버튼을 클릭하세요.")
            if 'trend_df_report' in st.session_state:
                st.subheader("📋 마지막 분석 결과 (트렌드 데이터)")
                st.dataframe(st.session_state['trend_df_report'], use_container_width=True)


    with tabs[4]: # 서브레딧 비교
        st.header("🎯 서브레딧 비교 분석")

        if st.button("🔍 서브레딧 비교 분석", key="btn_subreddit"):
            with st.spinner("서브레딧 비교 분석 중..."):
                fig, comparison_df = analyzer.subreddit_comparison()
                if fig:
                    st.pyplot(fig)
                    st.subheader("📋 서브레딧 통계")
                    st.dataframe(comparison_df, use_container_width=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv_file_name = f"reddit_subreddit_comparison_{timestamp}.csv"
                    comparison_df.to_csv(os.path.join(SAVE_DIR, csv_file_name), encoding='utf-8-sig') 
                    st.session_state['comparison_df_report'] = comparison_df 
                    csv_data = comparison_df.to_csv(encoding='utf-8-sig').encode('utf-8-sig')
                    st.download_button("💾 CSV 다운로드", csv_data, csv_file_name, "text/csv", key='download-subreddit-csv')
                    st.success(f"분석 결과가 서버 폴더 `{SAVE_DIR}`에 **{csv_file_name}**로 저장되었습니다.")
                else: st.warning("서브레딧 정보가 없거나 비교할 서브레딧이 2개 미만입니다.")
        else:
            st.info("👆 버튼을 클릭하여 서브레딧별 통계를 확인하세요.")
            if 'comparison_df_report' in st.session_state:
                st.subheader("📋 마지막 분석 결과 (서브레딧 통계)")
                st.dataframe(st.session_state['comparison_df_report'], use_container_width=True)


    with tabs[5]: # 원본 데이터
        st.header("📋 원본 데이터")
        data_type_options = ["게시물 데이터"]
        if comments_df is not None: data_type_options.append("댓글 데이터")
        data_type = st.radio("데이터 유형 선택", data_type_options, horizontal=True)

        if data_type == "게시물 데이터":
            st.subheader("📝 게시물 데이터")
            display_columns = st.multiselect("표시할 컬럼 선택", posts_df.columns.tolist(),
                default=['title', 'subreddit', 'score', 'num_comments', 'author'][:min(5, len(posts_df.columns))], key='posts_cols_select')
            
            if display_columns: st.dataframe(posts_df[display_columns], use_container_width=True, height=600)
            else: st.dataframe(posts_df, use_container_width=True, height=600)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file_name = f"reddit_posts_data_{timestamp}.csv"
            posts_df.to_csv(os.path.join(SAVE_DIR, csv_file_name), index=False, encoding='utf-8-sig')
            st.success(f"원본 데이터가 서버 폴더 `{SAVE_DIR}`에 **{csv_file_name}**로 저장되었습니다.")
            csv_data = posts_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button("💾 게시물 데이터 CSV 다운로드", csv_data, csv_file_name, "text/csv", key='download-posts-raw')
        else:
            if comments_df is not None and not comments_df.empty:
                st.subheader("💬 댓글 데이터")
                display_columns = st.multiselect("표시할 컬럼 선택", comments_df.columns.tolist(),
                    default=['body', 'subreddit', 'score', 'author', 'post_title'][:min(5, len(comments_df.columns))], key='comments_cols_select')
                
                if display_columns: st.dataframe(comments_df[display_columns], use_container_width=True, height=600)
                else: st.dataframe(comments_df, use_container_width=True, height=600)
                
                csv_file_name = "reddit_comments_data.csv"
                comments_df.to_csv(os.path.join(SAVE_DIR, csv_file_name), index=False, encoding='utf-8-sig')
                st.success(f"원본 데이터가 서버 폴더 `{SAVE_DIR}`에 저장되었습니다.")
                csv_data = comments_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                st.download_button("💾 댓글 데이터 CSV 다운로드", csv_data, csv_file_name, "text/csv", key='download-comments-raw')
            else: st.warning("댓글 데이터가 없습니다.")


    with tabs[6]: # AI 자동 보고서 생성 (일반)
        st.header("📄 Market Insight Report Generator (OpenAI API 기반)")
        st.write("분석 CSV 파일의 데이터를 기반으로 **사용자 지정 프롬프트**에 맞춘 보고서를 생성합니다.")
        st.warning("⚠️ **OpenAI API Key**가 `.env` 파일에 설정되어 있어야 하며, 보고서에 포함할 CSV 파일은 먼저 분석 탭에서 실행 후 **`analysis_results` 폴더에 저장**해야 합니다.")
        
        user_focus_prompt = st.text_area(
            "✍️ 보고서의 핵심 분석 주제 및 질문 (Focus Prompt)",
            value="Analyze the overall sentiment and identify the key positive drivers across the selected datasets. What strategic recommendations can be derived from the trend data and subreddit comparison?",
            height=150, key="report_general_prompt"
        )
        
        try:
            available_files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".csv")]
        except FileNotFoundError:
            available_files = []

        if not available_files:
            st.error("분석 결과 CSV 파일이 없습니다. 먼저 다른 분석 탭에서 분석을 실행하고 CSV를 다운로드하여 `analysis_results` 폴더에 저장하세요.")
        else:
            selected_files = st.multiselect("📂 보고서에 포함할 파일 선택", available_files, default=available_files, key="report_general_files")

            if st.button("🧠 보고서 생성 실행", key="btn_generate_report"):
                report_sentences = [] 
                
                if not selected_files or not user_focus_prompt.strip() or not OPENAI_API_KEY:
                    st.error("입력값을 확인하세요.")
                    return
                
                temp_analyzer = RedditAnalyzer(posts_df, comments_df)

                with st.spinner("OpenAI GPT 모델이 보고서를 생성 중..."):
                    
                    for f in selected_files:
                        file_path = os.path.join(SAVE_DIR, f)
                        keywords = f"File: {f}. "
                        try:
                            df = pd.read_csv(file_path, encoding="utf-8-sig")

                            if '키워드' in df.columns and '빈도' in df.columns: keywords += f"Top keyword is '{df.iloc[0]['키워드']}' with count {df.iloc[0]['빈도']}. Total unique keywords: {len(df)}. "
                            elif 'Sentiment' in df.columns:
                                sentiment_counts = df['Sentiment'].value_counts()
                                pos = sentiment_counts.get('긍정', 0)
                                neg = sentiment_counts.get('부정', 0)
                                total = len(df)
                                pos_ratio = pos / total * 100 if total > 0 else 0
                                keywords += f"Total comments/posts {total}. Positive ratio: {pos_ratio:.1f}%. Negative comments: {neg}. The overall sentiment is mostly Positive. "
                            elif '개수' in df.columns and '날짜' in df.columns:
                                df['날짜'] = pd.to_datetime(df['날짜'])
                                max_count_date = df.loc[df['개수'].idxmax(), '날짜'].strftime('%Y-%m-%d')
                                max_count = df['개수'].max()
                                keywords += f"Peak count {max_count} occurred on {max_count_date}. Average count per period is {df['개수'].mean():.1f}. "
                            elif '총_점수' in df.columns and '서브레딧' in df.columns:
                                top_subreddit = df.loc[df['총_점수'].idxmax(), '서브레딧']
                                top_score = df['총_점수'].max()
                                avg_comments = df['평균_댓글수'].mean()
                                keywords += f"Top subreddit by total score is '{top_subreddit}' with score {top_score}. Average comments per post across all subreddits: {avg_comments:.1f}. "
                            elif ('title' in df.columns and 'score' in df.columns) or ('body' in df.columns and 'score' in df.columns):
                                total_records = len(df)
                                avg_score = df['score'].mean()
                                keywords += f"Raw data summary. Total records: {total_records}. Average score: {avg_score:.1f}. "
                            else: keywords += f"Dataset rows: {len(df)}. Columns: {', '.join(df.columns)}. Data statistics available. "
                            
                            sentence = generate_openai_report(keywords, user_focus_prompt, OPENAI_API_KEY) 
                            report_sentences.append(f"**{f} Insight:** {sentence}")
                        except Exception as e:
                            st.error(f"파일 {f} 처리 오류: CSV 파일 구조 확인 필요. {str(e)}")
                            continue

                if report_sentences:
                    summary = "\n\n".join(report_sentences)
                    final_report = f"""
# Reddit Analysis Auto-Generated Report
## Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
## User Focus Prompt: {user_focus_prompt}

{summary}
"""
                    st.subheader("📈 AI 자동 생성 보고서 초안")
                    st.text_area("요약 결과", final_report, height=400)
                    st.download_button("💾 요약 보고서 저장", final_report.encode("utf-8-sig"), "Market_Insight_Report_Reddit.txt", "text/plain")
                else: st.error("보고서 생성 실패. 파일 선택 및 구조를 확인하세요.")


    with tabs[7]: # 임원진 보고서
        st.header("💼 임원진 보고서 (Executive Summary)")
        st.write("핵심 데이터를 기반으로 **국문 및 영문**으로 분리된, 임원진 제출용으로 적합한 요약 보고서를 생성합니다.")
        st.warning("⚠️ 이 보고서 생성을 위해서는 **OpenAI API Key**가 필수이며, 다른 탭에서 CSV 파일을 **`analysis_results`** 폴더에 저장해야 합니다.")
        
        user_exec_prompt = st.text_area(
            "✍️ 임원진 보고서의 핵심 분석 주제 및 질문",
            value="Identify the 3 most critical market insights and propose concise strategic actions for brand positioning based on the competitive analysis.",
            height=100, key="report_exec_prompt"
        )
        
        try:
            available_files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".csv")]
        except FileNotFoundError:
            available_files = []

        if not available_files:
            st.error("분석 결과 CSV 파일이 없습니다. 먼저 분석 탭에서 파일을 저장하세요.")
        else:
            selected_exec_files = st.multiselect("📂 보고서에 포함할 파일 선택", available_files, default=available_files, key="report_exec_files")

            if st.button("🧠 국/영문 임원진 보고서 생성", key="btn_generate_exec_report"):
                
                if not selected_exec_files or not user_exec_prompt.strip() or not OPENAI_API_KEY:
                    st.error("입력값을 확인하세요.")
                    return

                temp_analyzer = RedditAnalyzer(posts_df, comments_df)
                full_keywords_for_exec = ""
                
                with st.spinner("OpenAI GPT 모델이 국/영문 보고서를 생성 중..."):
                    
                    # 1. 모든 선택된 파일의 데이터를 하나의 키워드 문자열로 조합 (데이터 추출 로직 재사용)
                    for f in selected_exec_files:
                        file_path = os.path.join(SAVE_DIR, f)
                        keywords_chunk = f"File: {f}. "
                        try:
                            df = pd.read_csv(file_path, encoding="utf-8-sig")

                            if '키워드' in df.columns and '빈도' in df.columns: keywords_chunk += f"Top keyword is '{df.iloc[0]['키워드']}' with count {df.iloc[0]['빈도']}. Total unique keywords: {len(df)}. "
                            elif 'Sentiment' in df.columns:
                                sentiment_counts = df['Sentiment'].value_counts()
                                pos_ratio = sentiment_counts.get('긍정', 0) / len(df) * 100 if len(df) > 0 else 0
                                keywords_chunk += f"Total comments/posts {len(df)}. Positive ratio: {pos_ratio:.1f}%. "
                            elif '개수' in df.columns and '날짜' in df.columns:
                                df['날짜'] = pd.to_datetime(df['날짜'])
                                max_count = df['개수'].max()
                                keywords_chunk += f"Peak count {max_count}. Average count per period is {df['개수'].mean():.1f}. "
                            elif '총_점수' in df.columns and '서브레딧' in df.columns:
                                top_subreddit = df.loc[df['총_점수'].idxmax(), '서브레딧']
                                top_score = df['총_점수'].max()
                                keywords_chunk += f"Top subreddit is '{top_subreddit}' with score {top_score}. "
                            elif ('title' in df.columns and 'score' in df.columns) or ('body' in df.columns and 'score' in df.columns):
                                avg_score = df['score'].mean()
                                keywords_chunk += f"Raw data summary. Total records: {len(df)}. Average score: {avg_score:.1f}. "
                            else: keywords_chunk += f"Dataset rows: {len(df)}. Columns: {', '.join(df.columns)}. "
                            
                            full_keywords_for_exec += keywords_chunk
                        except Exception as e:
                            st.error(f"파일 {f} 처리 오류: {str(e)}")
                            continue
                            
                    # 2. 통합된 키워드와 사용자 프롬프트를 바탕으로 임원진 보고서 생성
                    english_summary, korean_summary = generate_executive_report(full_keywords_for_exec, user_exec_prompt, OPENAI_API_KEY)
                    
                    # 3. 결과 출력 및 다운로드
                    
                    # 국문 보고서 구성
                    korean_final_report = f"""
# 🔴 Reddit 임원진 보고서 (국문)
## 작성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
## 분석 주제: {user_exec_prompt}
---
### 핵심 요약 (Korean Summary)
{korean_summary}

---
### 분석에 사용된 데이터 파일
{', '.join(selected_exec_files)}
"""
                    # 영문 보고서 구성
                    english_final_report = f"""
# 🔴 Reddit Executive Summary (English)
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
                    st.error("보고서 파싱에 실패했습니다. OpenAI 출력 형식이 제대로 지켜지지 않았을 수 있습니다. 일반 보고서 탭에서 API 오류 메시지를 확인해보세요.")


if __name__ == "__main__":
    main()