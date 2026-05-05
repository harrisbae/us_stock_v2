#!/bin/bash

# 컨솔 출력에서 PNG 파일 목록을 추출하여 정리하는 스크립트

# 기본값 설정
DAYS=7  # 최근 N일 내 생성된 파일
OUTPUT_DIR="output/hma_mantra"  # 기본 검색 디렉토리
SORT_BY="time"  # 정렬 기준: time (시간), name (이름)

# 명령행 인자 처리
while [[ $# -gt 0 ]]; do
  case $1 in
    -d|--days)
      DAYS="$2"
      shift 2
      ;;
    -o|--output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -s|--sort)
      SORT_BY="$2"
      shift 2
      ;;
    -h|--help)
      echo "사용법: $0 [옵션]"
      echo ""
      echo "옵션:"
      echo "  -d, --days DAYS        최근 N일 내 생성된 파일 (기본값: 7)"
      echo "  -o, --output-dir DIR    검색 디렉토리 (기본값: output/hma_mantra)"
      echo "  -s, --sort SORT         정렬 기준: time 또는 name (기본값: time)"
      echo "  -h, --help             도움말 출력"
      echo ""
      echo "예시:"
      echo "  $0                      # 최근 7일 내 PNG 파일 목록"
      echo "  $0 -d 30                # 최근 30일 내 PNG 파일 목록"
      echo "  $0 -s name              # 파일명으로 정렬"
      exit 0
      ;;
    *)
      echo "알 수 없는 옵션: $1"
      echo "사용법: $0 [-d DAYS] [-o DIR] [-s SORT] [-h]"
      exit 1
      ;;
  esac
done

# 스크립트 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "📊 PNG 파일 목록 정리 중..."
echo "검색 디렉토리: $OUTPUT_DIR"
echo "기준: 최근 ${DAYS}일 내 생성된 파일"
echo ""

# PNG 파일 찾기
if [ "$SORT_BY" = "time" ]; then
    # 시간순 정렬 (최신순)
    PNG_FILES=$(find "$OUTPUT_DIR" -name "*.png" -type f -mtime -${DAYS} 2>/dev/null | sort -r)
else
    # 이름순 정렬
    PNG_FILES=$(find "$OUTPUT_DIR" -name "*.png" -type f -mtime -${DAYS} 2>/dev/null | sort)
fi

if [ -z "$PNG_FILES" ]; then
    echo "❌ 최근 ${DAYS}일 내 생성된 PNG 파일이 없습니다."
    exit 0
fi

# 파일 개수
FILE_COUNT=$(echo "$PNG_FILES" | grep -c '^' || echo "0")
echo "총 ${FILE_COUNT}개의 PNG 파일 발견"
echo ""

# 날짜별로 그룹화하여 출력
echo "===================================================================================================="
echo "📅 날짜별 PNG 파일 목록"
echo "===================================================================================================="
echo ""

# 임시 파일에 날짜별로 정리
TEMP_FILE=$(mktemp)
while IFS= read -r file; do
    if [ -z "$file" ]; then
        continue
    fi
    
    # 파일 수정 시간 가져오기 (macOS와 Linux 호환)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        mod_time=$(stat -f "%Sm" -t "%Y-%m-%d" "$file" 2>/dev/null)
    else
        mod_time=$(stat -c "%y" "$file" 2>/dev/null | cut -d' ' -f1)
    fi
    
    if [ -z "$mod_time" ]; then
        mod_time="알 수 없음"
    fi
    
    echo "$mod_time|$file" >> "$TEMP_FILE"
done <<< "$PNG_FILES"

# 날짜별로 그룹화하여 출력
prev_date=""
while IFS='|' read -r date file; do
    if [ "$date" != "$prev_date" ]; then
        if [ ! -z "$prev_date" ]; then
            echo ""
        fi
        echo "📅 $date"
        echo "----------------------------------------------------------------------------------------------------"
        prev_date="$date"
    fi
    
    # 종목명 추출
    symbol=$(basename "$(dirname "$file")")
    filename=$(basename "$file")
    
    echo "  📊 $symbol"
    echo "     $file"
done < <(sort -t'|' -k1,1r -k2,2 "$TEMP_FILE")

rm -f "$TEMP_FILE"

echo ""
echo "===================================================================================================="
echo "📄 전체 파일 경로 (클릭 가능)"
echo "===================================================================================================="
echo ""

if [ "$SORT_BY" = "time" ]; then
    echo "$PNG_FILES"
else
    echo "$PNG_FILES" | sort
fi

echo ""
echo "✅ 총 ${FILE_COUNT}개의 PNG 파일"
