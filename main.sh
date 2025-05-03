#!/bin/bash

# 주식 기술적 분석 시스템 실행 스크립트
# 사용법: ./main.sh [옵션] [종목코드1] [종목코드2] ...

# 기본 값 설정
PERIOD="1mo"
INTERVAL="1d"
VISUALIZE=true
START_DATE=""
END_DATE=""
OUTPUT_DIR=""
HELP=false
STOCKS=()

# 현재 날짜 (YYYY-MM-DD 형식)
CURRENT_DATE=$(date +%Y-%m-%d)

# 도움말 표시 함수
show_help() {
    echo "주식 기술적 분석 시스템 (v1.0)"
    echo ""
    echo "사용법: ./main.sh [옵션] [종목코드1] [종목코드2] ..."
    echo ""
    echo "옵션:"
    echo "  -h, --help                도움말 표시"
    echo "  -p, --period PERIOD       데이터 기간 (기본값: 1mo)"
    echo "                            유효값: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max"
    echo "  -i, --interval INTERVAL   봉 간격 (기본값: 1d)"
    echo "                            유효값: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo"
    echo "  -s, --start-date DATE     시작일 (YYYY-MM-DD 형식)"
    echo "  -e, --end-date DATE       종료일 (YYYY-MM-DD 형식)"
    echo "  -o, --output DIR          출력 디렉토리 지정"
    echo "  -nv, --no-visualize       차트 시각화 생략"
    echo ""
    echo "예제:"
    echo "  ./main.sh AAPL                      # 애플 주식 기본 분석"
    echo "  ./main.sh -p 3mo AAPL TSLA AMZN     # 3개월 데이터로 여러 종목 분석"
    echo "  ./main.sh -s 2025-01-01 -e 2025-05-01 TSLA  # 특정 기간 테슬라 분석"
    echo ""
}

# 매개변수 처리
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            HELP=true
            shift
            ;;
        -p|--period)
            PERIOD="$2"
            shift 2
            ;;
        -i|--interval)
            INTERVAL="$2"
            shift 2
            ;;
        -s|--start-date)
            START_DATE="$2"
            shift 2
            ;;
        -e|--end-date)
            END_DATE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -nv|--no-visualize)
            VISUALIZE=false
            shift
            ;;
        *)
            # 종목코드로 해석
            STOCKS+=("$1")
            shift
            ;;
    esac
done

# 도움말 표시 및 종료
if [ "$HELP" = true ]; then
    show_help
    exit 0
fi

# 종목코드가 없는 경우 기본값 설정
if [ ${#STOCKS[@]} -eq 0 ]; then
    echo "종목코드가 지정되지 않았습니다. 기본 종목(AAPL, TSLA, AMZN)을 분석합니다."
    STOCKS=("AAPL" "TSLA" "AMZN")
fi

# 종료일이 없는 경우 현재 날짜로 설정
if [ -z "$END_DATE" ]; then
    END_DATE=$CURRENT_DATE
fi

# 각 종목 분석 실행
for stock in "${STOCKS[@]}"; do
    echo "=========================="
    echo "분석 중: $stock"
    echo "기간: $PERIOD, 간격: $INTERVAL"
    if [ -n "$START_DATE" ]; then
        echo "분석 기간: $START_DATE ~ $END_DATE"
    fi
    echo "=========================="
    
    # 명령 구성
    cmd="python3 main.py --ticker $stock --period $PERIOD --interval $INTERVAL"
    
    # 시각화 옵션 추가
    if [ "$VISUALIZE" = true ]; then
        cmd="$cmd --visualize"
    fi
    
    # 날짜 범위 옵션 추가
    if [ -n "$START_DATE" ]; then
        cmd="$cmd --start-date $START_DATE --end-date $END_DATE"
    fi
    
    # 출력 디렉토리 옵션 추가
    if [ -n "$OUTPUT_DIR" ]; then
        cmd="$cmd --output $OUTPUT_DIR"
    fi
    
    # 명령 실행
    echo "실행 명령: $cmd"
    echo ""
    eval $cmd
    
    echo ""
done

echo "모든 분석이 완료되었습니다." 