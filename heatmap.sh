#!/bin/bash

# 프로젝트 루트로 이동
cd "$(dirname "$0")"

# 히트맵 생성 실행
python3 src/heatmap/heatmap.py

echo "포트폴리오 히트맵이 output/heatmap/portfolio_heatmap.png로 저장되었습니다."
