import os
from dotenv import load_dotenv
import praw

# .env 파일 불러오기
load_dotenv()

# 환경변수에서 값 읽기
client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
username = os.getenv("REDDIT_USERNAME")
password = os.getenv("REDDIT_PASSWORD")

# Reddit API 연결
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent="yj_beauty_analysis by u/{}".format(username),
    username=username,
    password=password
)

SUBREDDIT_NAME = "koreanskincare"  # 분석하려는 서브레딧 이름
LIMIT = 1000  # 가져올 게시글 수 (최대 1000개까지 가능하지만, 한 번에 너무 많이 가져오지 않도록 주의)

# 특정 서브레딧 객체 지정
subreddit = reddit.subreddit(SUBREDDIT_NAME)

print(f"--- r/{SUBREDDIT_NAME} 의 인기 게시글 {LIMIT}개 수집 시작 ---")

# 'hot' 게시글을 순회하며 데이터 수집
for submission in subreddit.hot(limit=LIMIT):
    # 게시글 제목과 URL 출력
    print(f"\n[제목] {submission.title}")
    print(f"[URL] {submission.url}")
    print(f"[점수] {submission.score} | [댓글] {submission.num_comments}")
    print("---------------------------------")
    
    # 여기서 더 나아가 댓글까지 수집할 수 있어
    # submission.comments.replace_more(limit=0) # '더 보기' 댓글 제거
    # for top_comment in submission.comments.list()[:3]: # 상위 3개 댓글만 출력
    #     print(f"    [댓글] {top_comment.body}")

SEARCH_TERM = "dalba"
SEARCH_LIMIT = 1000

print(f"--- r/{SUBREDDIT_NAME} 내에서 '{SEARCH_TERM}' 검색 결과 {SEARCH_LIMIT}개 ---")

# .search() 함수 사용
for submission in subreddit.search(query=SEARCH_TERM, limit=SEARCH_LIMIT, sort='new'): # 'new'는 최신순
    print(f"\n[제목] {submission.title}")
    print(f"[작성자] {submission.author}")
    print(f"[텍스트 미리보기] {submission.selftext[:100]}...")
    print(f"[URL] https://reddit.com{submission.permalink}")

    # 댓글 가져오기
    print(f"\n--- 댓글 ({submission.num_comments}개) ---")
    
    # 모든 댓글을 로드 (MoreComments 객체 제거)
    submission.comments.replace_more(limit=0)
    
    # 최상위 댓글만 가져오기
    for comment in submission.comments.list()[:5]:  # 상위 5개만
        if hasattr(comment, 'body'):  # 실제 댓글인지 확인
            print(f"\n  [{comment.author}] {comment.score}점")
            print(f"  {comment.body[:200]}...")  # 댓글 내용 미리보기

from transformers import pipeline
import pandas as pd

# 감성분석 모델 로드
# 영어용
sentiment_analyzer_en = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# 한국어용 (선택사항)
# sentiment_analyzer_ko = pipeline("sentiment-analysis", model="matthewburke/korean_sentiment")

SEARCH_TERM = "dalba"
SEARCH_LIMIT = 1000

results = []

print(f"--- r/{SUBREDDIT_NAME} 내에서 '{SEARCH_TERM}' 검색 및 감성분석 ---\n")

for submission in subreddit.search(query=SEARCH_TERM, limit=SEARCH_LIMIT, sort='new'):
    print(f"\n{'='*80}")
    print(f"[제목] {submission.title}")
    print(f"[댓글 수] {submission.num_comments}개")
    
    # 댓글 로드
    submission.comments.replace_more(limit=0)
    
    for comment in submission.comments.list()[:10]:  # 상위 10개 댓글
        if hasattr(comment, 'body') and len(comment.body) > 10:  # 너무 짧은 댓글 제외
            try:
                # 감성분석 수행
                sentiment = sentiment_analyzer_en(comment.body[:512])[0]  # 최대 512자
                
                label = sentiment['label']  # POSITIVE or NEGATIVE
                confidence = sentiment['score']  # 확신도 (0~1)
                
                # 결과 저장
                results.append({
                    'post_title': submission.title,
                    'author': str(comment.author),
                    'comment': comment.body[:200],
                    'score': comment.score,
                    'sentiment': label,
                    'confidence': round(confidence, 3)
                })
                
                # 이모지로 표시
                emoji = "😊" if label == "POSITIVE" else "😞"
                print(f"\n{emoji} [{comment.author}] (점수: {comment.score})")
                print(f"   감성: {label} ({confidence:.2%} 확신도)")
                print(f"   내용: {comment.body[:150]}...")
                
            except Exception as e:
                print(f"   [오류] {e}")
                continue

# 결과를 DataFrame으로 변환
df = pd.DataFrame(results)

# 통계 출력
print(f"\n{'='*80}")
print(f"총 분석된 댓글: {len(df)}개")
print(f"긍정(POSITIVE): {len(df[df['sentiment']=='POSITIVE'])}개 ({len(df[df['sentiment']=='POSITIVE'])/len(df)*100:.1f}%)")
print(f"부정(NEGATIVE): {len(df[df['sentiment']=='NEGATIVE'])}개 ({len(df[df['sentiment']=='NEGATIVE'])/len(df)*100:.1f}%)")
print(f"\n평균 확신도: {df['confidence'].mean():.2%}")

# CSV로 저장
df.to_csv('reddit_sentiment_analysis.csv', index=False, encoding='utf-8-sig')
print(f"\n결과가 'reddit_sentiment_analysis.csv'에 저장되었습니다.")