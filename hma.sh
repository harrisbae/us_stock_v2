#!/bin/bash

# 가상 환경 활성화
source .venv/bin/activate

# 기본값 설정
PERIOD="120d"
DAYS=""  # 일수 옵션 추가
FROM_DATE=""  # 시작 날짜 옵션 추가
TO_DATE=""    # 종료 날짜 옵션 추가
INTERVAL="1d"
PREPOST="false"
SYMBOL=""
SYMBOL_FILE=""
ANALYSIS_TYPE="technical"  # 기본값: 기술적 분석
VOLUME_PROFILE_TYPE="none"  # 기본값: Volume Profile 없음

# 명령행 인자 처리
while [[ $# -gt 0 ]]; do
  case $1 in
    -s)
      SYMBOL="$2"
      shift 2
      ;;
    -f)
      SYMBOL_FILE="$2"
      shift 2
      ;;
    -p)
      PERIOD="$2"
      shift 2
      ;;
    -d)
      DAYS="$2"
      shift 2
      ;;
    -i)
      INTERVAL="$2"
      shift 2
      ;;
    -t)
      PREPOST="$2"
      shift 2
      ;;
    -a)
      ANALYSIS_TYPE="$2"
      shift 2
      ;;
    -v)
      VOLUME_PROFILE_TYPE="$2"
      shift 2
      ;;
    --from)
      FROM_DATE="$2"
      shift 2
      ;;
    --to)
      TO_DATE="$2"
      shift 2
      ;;
    --file)
      SYMBOL_FILE="$2"
      shift 2
      ;;
    *)
      echo "알 수 없는 옵션: $1"
      exit 1
      ;;
  esac
done

# 일수 옵션이 지정된 경우 PERIOD를 업데이트
if [ ! -z "$DAYS" ]; then
  PERIOD="${DAYS}d"
fi

# 날짜 범위 처리
if [ ! -z "$FROM_DATE" ] || [ ! -z "$TO_DATE" ]; then
  # --to와 -p가 모두 지정된 경우: --to 날짜로부터 -p 기간만큼 이전 계산
  if [ ! -z "$TO_DATE" ] && [ ! -z "$PERIOD" ] && [ "$PERIOD" != "120d" ] && [ -z "$FROM_DATE" ]; then
    # PERIOD에서 숫자와 단위 분리 (예: 30d -> 30, d, 6mo -> 6, mo)
    if [[ "$PERIOD" =~ ^([0-9]+)([dmy]|mo)$ ]]; then
      amount=${BASH_REMATCH[1]}
      unit=${BASH_REMATCH[2]}
      
      # Python을 사용하여 날짜 계산
      FROM_DATE=$(python3 -c "
from datetime import datetime, timedelta
import sys

to_date = '$TO_DATE'
amount = $amount
unit = '$unit'

# 단위에 따라 날짜 계산
if unit == 'd':
    days = amount
elif unit == 'm' or unit == 'mo':  # m 또는 mo (월)
    days = amount * 30  # 월을 30일로 근사
elif unit == 'y':
    days = amount * 365  # 년을 365일로 근사
else:
    days = amount

# FROM_DATE 계산 (TO_DATE에서 days일 전)
try:
    to_dt = datetime.strptime(to_date, '%Y-%m-%d')
    from_dt = to_dt - timedelta(days=days)
    print(from_dt.strftime('%Y-%m-%d'))
except Exception as e:
    print(to_date)  # 오류 시 TO_DATE 반환
")
      echo "종료 날짜($TO_DATE)로부터 $PERIOD 전까지 계산: $FROM_DATE ~ $TO_DATE"
    fi
  fi
  
  # --to가 지정되지 않은 경우 오늘 날짜로 설정
  if [ -z "$TO_DATE" ]; then
    TO_DATE=$(date +"%Y-%m-%d")
    echo "종료 날짜가 지정되지 않아 오늘 날짜($TO_DATE)로 설정합니다."
  fi
  
  # --from이 지정되지 않은 경우 TO_DATE로 설정 (같은 날짜)
  if [ -z "$FROM_DATE" ]; then
    FROM_DATE="$TO_DATE"
  fi
  
  PERIOD="${FROM_DATE}_${TO_DATE}"
fi

# 필수 인자 확인 (-s 또는 --file 중 하나는 필수)
if [ -z "$SYMBOL" ] && [ -z "$SYMBOL_FILE" ]; then
  echo "Usage: $0 (-s SYMBOL | --file SYMBOL_FILE) [-p PERIOD | -d DAYS | --from START_DATE --to END_DATE] [-i INTERVAL] [-t PREPOST] [-a ANALYSIS_TYPE] [-v VOLUME_PROFILE_TYPE]"
  echo "Options:"
  echo "  -s SYMBOL           단일 종목 분석"
  echo "  --file SYMBOL_FILE  종목 파일에서 읽어서 분석 (한 줄에 하나의 종목코드)"
  echo "  -p PERIOD          기간 (예: 120d, 6mo, 1y, 기본값: 120d)"
  echo "  -d DAYS            일수 (예: 30, 60, 90, 365, 기본값: 사용 안함)"
  echo "  --from START_DATE  시작 날짜 (예: 2024-01-01)"
  echo "  --to END_DATE      종료 날짜 (예: 2024-12-31, 생략 시 오늘 날짜)"
  echo "  -i INTERVAL        간격 (기본값: 1d)"
  echo "  -t PREPOST         장전/장후 데이터 포함 여부 (기본값: false)"
  echo "  -a TYPE            분석 유형 (technical/macro/sector/news/chart/financial/strategy/all, 기본값: technical)"
  echo "  -v TYPE            Volume Profile 유형 (none/separate/overlay, 기본값: none)"
  echo "                      none: Volume Profile 없음"
  echo "                      separate: 별도 영역에 Volume Profile"
  echo "                      overlay: 메인차트에 Volume Profile 오버레이"
  echo ""
  echo "기간 설정 예시:"
  echo "  -p 30d            30일"
  echo "  -p 6mo            6개월"
  echo "  -p 1y             1년"
  echo "  -d 30             30일 (자동으로 30d로 변환)"
  echo "  -d 365            1년 (자동으로 365d로 변환)"
  echo "  --from 2024-01-01 --to 2024-12-31  특정 날짜 범위"
  echo "  --from 2024-08-01 --to 2025-08-01  12개월 특정 기간"
  echo "  --from 2025-01-01                   2025년 1월 1일부터 오늘까지"
  echo "  --to 2025-08-01 -p 30d             2025-08-01로부터 30일 전까지"
  echo "  --to 2025-08-01 -p 6mo             2025-08-01로부터 6개월 전까지"
  echo "  --to 2025-08-01 -p 1y              2025-08-01로부터 1년 전까지"
  exit 1
fi

# 분석 함수 정의
analyze_stock() {
    local symbol=$1
    local output_dir="output/analysis/$symbol"
    mkdir -p "$output_dir"
    
    case $ANALYSIS_TYPE in
        "technical"|"all")
            echo "기술적 분석 시작..."
            case $VOLUME_PROFILE_TYPE in
                "separate")
                    echo "Volume Profile (별도 영역) 생성 중..."
                    python test/volume_profile_test.py "$symbol" "$PERIOD"
                    ;;
                "overlay")
                    echo "Volume Profile (오버레이) 생성 중..."
                    python test/volume_profile_overlay_test.py "$symbol" "$PERIOD"
                    ;;
                "none"|*)
                    echo "기본 기술적 분석 실행..."
                    python src/indicators/hma_mantra_example.py "$symbol" "$PERIOD" "$INTERVAL" "$PREPOST"
                    ;;
            esac
            ;;
        "macro"|"all")
            echo "거시경제 분석 시작..."
            python src/analysis/macro/economic_indicators.py "$symbol" "$output_dir/${symbol}_macro.png"
            ;;
        "sector"|"all")
            echo "섹터 분석 시작..."
            python src/analysis/sector/gics_analysis.py "$symbol" "$output_dir/${symbol}_sector.png"
            ;;
        "news"|"all")
            echo "뉴스 분석 시작..."
            python src/analysis/news/news_analysis.py "$symbol" "$output_dir/${symbol}_news.png"
            ;;
        "chart"|"all")
            echo "차트 분석 시작..."
            python src/analysis/visualization/chart_analysis.py "$symbol" "$output_dir/${symbol}_chart.png"
            ;;
        "financial"|"all")
            echo "재무 분석 시작..."
            python test/financial_analysis_test.py "$symbol" "$PERIOD"
            ;;
        "strategy"|"all")
            echo "차트 기반 투자 전략 분석 시작..."
            echo "분석할 파일 정보:"
            echo "  - 종목: $symbol"
            echo "  - 기간: $PERIOD"
            echo "  - Volume Profile: $VOLUME_PROFILE_TYPE"
            echo "  - 출력 디렉토리: output/hma_mantra/$symbol/"
            echo "  - 신호 파일: output/hma_mantra/$symbol/${symbol}_signal.txt"
            
            # Volume Profile 차트 생성
            if [ "$VOLUME_PROFILE_TYPE" = "overlay" ]; then
                echo "Volume Profile (오버레이) 차트 생성 중..."
                python test/volume_profile_overlay_test.py "$symbol" "$PERIOD"
                echo "Volume Profile 차트 저장 완료: output/hma_mantra/$symbol/${symbol}_volume_profile_overlay_${PERIOD}_chart.png"
            elif [ "$VOLUME_PROFILE_TYPE" = "separate" ]; then
                echo "Volume Profile (별도 영역) 차트 생성 중..."
                python test/volume_profile_test.py "$symbol" "$PERIOD"
                echo "Volume Profile 차트 저장 완료: output/hma_mantra/$symbol/${symbol}_volume_profile_${PERIOD}_chart.png"
            fi
            
            # 투자 전략 분석 실행
            python test/strategy_analysis_test.py "$symbol" "$PERIOD"
            echo "투자 전략 분석 완료!"
            ;;
        *)
            echo "알 수 없는 분석 유형: $ANALYSIS_TYPE"
            exit 1
            ;;
    esac
}

# 종목 파일이 지정된 경우
if [ ! -z "$SYMBOL_FILE" ]; then
  if [ ! -f "$SYMBOL_FILE" ]; then
    echo "오류: 종목 파일을 찾을 수 없습니다: $SYMBOL_FILE"
    exit 1
  fi
  
  echo "종목 파일 처리 시작: $SYMBOL_FILE"
  while IFS= read -r symbol || [ -n "$symbol" ]; do
    # 주석 라인과 빈 라인 건너뛰기
    [[ $symbol =~ ^#.*$ ]] && continue
    [[ -z "${symbol// }" ]] && continue
    
    echo "$symbol 분석 시작..."
    analyze_stock "$symbol"
    echo "$symbol 분석 완료"
    echo "-------------------"
  done < "$SYMBOL_FILE"
  echo "모든 종목 분석 완료"

# 단일 종목이 지정된 경우
else
  echo "$SYMBOL 분석 시작..."
  analyze_stock "$SYMBOL"
fi

# 결과 파일 이름 변경
if [ -f "output/hma_mantra/$SYMBOL/${SYMBOL}_analysis.png" ]; then
  mv "output/hma_mantra/$SYMBOL/${SYMBOL}_analysis.png" "output/hma_mantra/$SYMBOL/${SYMBOL}_analysis.png"
fi 

# 매수/매도 신호 종합 기능
SUMMARY_FILE="output/analysis/summary_signal.txt"
echo "종목,신호,날짜" > "$SUMMARY_FILE"

# 현재 날짜 가져오기 (한국 시간 기준)
CURRENT_DATE=$(TZ=Asia/Seoul date +"%Y-%m-%d")

if [ ! -z "$SYMBOL_FILE" ]; then
  while IFS= read -r symbol || [ -n "$symbol" ]; do
    [[ $symbol =~ ^#.*$ ]] && continue
    [[ -z "${symbol// }" ]] && continue
    
    SIGNAL_FILE="output/hma_mantra/$symbol/${symbol}_signal.txt"
    if [ -f "$SIGNAL_FILE" ]; then
      # 신호 요약 섹션에서 마지막 줄의 신호만 추출
      SIGNAL=$(grep "=== 신호 요약 ===" -A 1 "$SIGNAL_FILE" | tail -1 | tr -d '\r')
      if [ -z "$SIGNAL" ]; then
        # 신호 요약이 없으면 파일의 마지막 줄 사용
        SIGNAL=$(tail -1 "$SIGNAL_FILE" | tr -d '\r')
      fi
      echo "$symbol,$SIGNAL,$CURRENT_DATE" >> "$SUMMARY_FILE"
    else
      echo "$symbol,NO_SIGNAL,$CURRENT_DATE" >> "$SUMMARY_FILE"
    fi
  done < "$SYMBOL_FILE"
else
  SIGNAL_FILE="output/hma_mantra/$SYMBOL/${SYMBOL}_signal.txt"
  if [ -f "$SIGNAL_FILE" ]; then
    # 신호 요약 섹션에서 마지막 줄의 신호만 추출
    SIGNAL=$(grep "=== 신호 요약 ===" -A 1 "$SIGNAL_FILE" | tail -1 | tr -d '\r')
    if [ -z "$SIGNAL" ]; then
      # 신호 요약이 없으면 파일의 마지막 줄 사용
      SIGNAL=$(tail -1 "$SIGNAL_FILE" | tr -d '\r')
    fi
    echo "$SYMBOL,$SIGNAL,$CURRENT_DATE" >> "$SUMMARY_FILE"
  else
    echo "$SYMBOL,NO_SIGNAL,$CURRENT_DATE" >> "$SUMMARY_FILE"
  fi
fi

echo -e "\n=== 매수/매도 신호 요약 ==="
echo "파일 위치: $SUMMARY_FILE"
echo "-------------------"
cat "$SUMMARY_FILE"
echo "-------------------" 
