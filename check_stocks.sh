#!/bin/bash

# 반등 신호 / 반등 기대 / 상승 지속 대상 추출 스크립트
# 최근 30일 내 매수 신호를 분석하여 반등 신호, 반등 기대, 상승 지속 대상으로 분류

# 가상 환경 활성화 (있는 경우)
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# 기본값 설정
DAYS=30  # 최근 N일 내 매수 신호 기준

# 명령행 인자 처리
while [[ $# -gt 0 ]]; do
  case $1 in
    -d|--days)
      DAYS="$2"
      shift 2
      ;;
    -h|--help)
      echo "사용법: $0 [옵션]"
      echo ""
      echo "옵션:"
      echo "  -d, --days DAYS    최근 N일 내 매수 신호 기준 (기본값: 30)"
      echo "  -h, --help         도움말 출력"
      echo ""
      echo "예시:"
      echo "  $0                  # 최근 30일 내 매수 신호 분석"
      echo "  $0 -d 60            # 최근 60일 내 매수 신호 분석"
      exit 0
      ;;
    *)
      echo "알 수 없는 옵션: $1"
      echo "사용법: $0 [-d DAYS] [-h]"
      exit 1
      ;;
  esac
done

# 스크립트 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Python 스크립트 실행
echo "📊 반등 신호 / 반등 기대 / 상승 지속 대상 분석 중..."
echo "기준: 최근 ${DAYS}일 내 매수 신호"
echo ""

# Python 스크립트 실행 및 결과 파일 경로 추출
RESULT=$(python extract_rebound_signals.py --days "$DAYS" 2>&1)

# 결과 출력
echo "$RESULT"

# 결과 파일 경로 추출
RESULT_FILE=$(echo "$RESULT" | grep -E "파일 경로:|📄 파일 경로:" | tail -1 | sed 's/.*: //' | tr -d '\n')

if [ ! -z "$RESULT_FILE" ] && [ -f "$RESULT_FILE" ]; then
    echo ""
    echo "📄 결과 파일:"
    echo "   $RESULT_FILE"
    
    # 결과 요약 출력
    echo ""
    echo "📊 분석 요약:"
    grep -E "반등 신호:|반등 기대:|상승 지속:" "$RESULT_FILE" | head -3
fi


