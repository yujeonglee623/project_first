import streamlit as st
import os

# Streamlit 페이지 설정을 메인 파일에서 한 번만 합니다.
st.set_page_config(
    page_title="통합 소셜 데이터 분석 허브",
    page_icon="✨",
    layout="wide"
)

def main():
    """메인 페이지 콘텐츠 (환영 메시지 및 안내)"""
    st.title("✨ 통합 소셜 데이터 분석 허브")
    st.markdown("---")
    st.markdown(f"""
        **환영합니다!** 👋
        
        왼쪽 사이드바에서 분석을 원하는 플랫폼을 선택하세요.
        
        * **1. Reddit 대시보드:** 레딧 댓글 및 게시물 트렌드 분석
        * **2. YouTube 대시보드:** 유튜브 댓글 및 영상 토픽 분석
        
        **📌 주의 사항:** 각 대시보드를 사용하기 전에, 프로젝트 폴더 내 `.env` 파일에
        필요한 API 키가 설정되어 있는지 확인해 주세요.
    """)
    
    # 분석 결과 폴더가 없다면 생성 (전역으로 두는 것이 좋음)
    SAVE_DIR = "analysis_results"
    os.makedirs(SAVE_DIR, exist_ok=True)


if __name__ == "__main__":
    main()