import os

folder = "pptx"  # 이미지가 들어있는 폴더명
output_file = "README.md"  # 결과 파일명

with open(output_file, "w", encoding="utf-8") as f:
    for file in sorted(os.listdir(folder)):
        if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
            f.write(f"![{file}]({folder}/{file})\n")

print(f"✅ 이미지 링크가 {output_file}에 생성되었습니다!")
