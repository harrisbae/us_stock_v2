#!/bin/bash

# 가상 환경 활성화
source .venv/bin/activate

# 기본값 설정
PERIOD="120d"
INTERVAL="1d"
PREPOST="false"
SYMBOL=""
SYMBOL_FILE=""
ANALYSIS_TYPE="technical"  # 기본값: 기술적 분석
VOLUME_PROFILE_TYPE="none"  # 기본값: Volume Profile 없음

# 명령행 인자 처리
while getopts "s:f:p:i:t:a:v:" opt; do
  case $opt in
    s) SYMBOL="$OPTARG";;
    f) SYMBOL_FILE="$OPTARG";;
    p) PERIOD="$OPTARG";;
    i) INTERVAL="$OPTARG";;
    t) PREPOST="$OPTARG";;
    a) ANALYSIS_TYPE="$OPTARG";;
    v) VOLUME_PROFILE_TYPE="$OPTARG";;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1;;
  esac
done

# 필수 인자 확인 (-s 또는 -f 중 하나는 필수)
if [ -z "$SYMBOL" ] && [ -z "$SYMBOL_FILE" ]; then
  echo "Usage: $0 (-s SYMBOL | -f SYMBOL_FILE) [-p PERIOD] [-i INTERVAL] [-t PREPOST] [-a ANALYSIS_TYPE] [-v VOLUME_PROFILE_TYPE]"
  echo "Options:"
  echo "  -s SYMBOL       단일 종목 분석"
  echo "  -f SYMBOL_FILE  종목 파일에서 읽어서 분석 (한 줄에 하나의 종목코드)"
  echo "  -p PERIOD      기간 (기본값: 120d)"
  echo "  -i INTERVAL    간격 (기본값: 1d)"
  echo "  -t PREPOST     장전/장후 데이터 포함 여부 (기본값: false)"
  echo "  -a TYPE        분석 유형 (technical/macro/sector/news/chart/all, 기본값: technical)"
  echo "  -v TYPE        Volume Profile 유형 (none/separate/overlay, 기본값: none)"
  echo "                  none: Volume Profile 없음"
  echo "                  separate: 별도 영역에 Volume Profile"
  echo "                  overlay: 메인차트에 Volume Profile 오버레이"
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
                    python test/volume_profile_test.py "$symbol"
                    ;;
                "overlay")
                    echo "Volume Profile (오버레이) 생성 중..."
                    python test/volume_profile_overlay_test.py "$symbol"
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
      SIGNAL=$(cat "$SIGNAL_FILE")
      echo "$symbol,$SIGNAL,$CURRENT_DATE" >> "$SUMMARY_FILE"
    else
      echo "$symbol,NO_SIGNAL,$CURRENT_DATE" >> "$SUMMARY_FILE"
    fi
  done < "$SYMBOL_FILE"
else
  SIGNAL_FILE="output/hma_mantra/$SYMBOL/${SYMBOL}_signal.txt"
  if [ -f "$SIGNAL_FILE" ]; then
    SIGNAL=$(cat "$SIGNAL_FILE")
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
