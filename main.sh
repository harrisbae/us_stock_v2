#!/bin/bash

# 주식 기술적 분석 시스템 실행 스크립트
# 사용법: ./main.sh [옵션] [종목코드1] [종목코드2] ...

# 기본 값 설정
PERIOD="3mo"
INTERVAL="1d"
VISUALIZE=true
START_DATE=""
END_DATE=""
OUTPUT_DIR=""
HELP=false
SELL_ONLY=false
BUY_ONLY=false
BOX_WINDOW=15
BOX_THRESHOLD=0.1
STOCKS=()
STOCK_LIST_FILE="stock_list.txt"

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
    echo "  --buy                     매수 신호 분석 강조 모드 (매수 신호 분석에 초점)"
    echo "  --sell                    매도 신호 분석 강조 모드 (매도 신호 분석에 초점)"
    echo "  --box-window WINDOW       박스권 분석 윈도우 크기 (기본값: 15)"
    echo "  --box-threshold THRESHOLD 박스권 분석 변동폭 임계값 (기본값: 0.1 = 10%)"
    echo "  -f, --file FILE           종목 목록 파일 지정 (기본값: stock_list.txt)"
    echo ""
    echo "예제:"
    echo "  ./main.sh AAPL                      # 애플 주식 기본 분석"
    echo "  ./main.sh -p 3mo AAPL TSLA AMZN     # 3개월 데이터로 여러 종목 분석"
    echo "  ./main.sh -s 2025-01-01 -e 2025-05-01 TSLA  # 특정 기간 테슬라 분석"
    echo "  ./main.sh --buy AAPL                # 애플 주식 매수 신호 분석 모드"
    echo "  ./main.sh --sell AAPL               # 애플 주식 매도 신호 분석 모드"
    echo "  ./main.sh --box-window 20 --box-threshold 0.05 AAPL  # 애플 주식 박스권 분석 (변동폭 5%)"
    echo "  ./main.sh -f my_stocks.txt          # my_stocks.txt 파일의 종목 목록으로 분석"
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
        --sell)
            SELL_ONLY=true
            shift
            ;;
        --buy)
            BUY_ONLY=true
            shift
            ;;
        --box-window)
            BOX_WINDOW="$2"
            shift 2
            ;;
        --box-threshold)
            BOX_THRESHOLD="$2"
            shift 2
            ;;
        -f|--file)
            STOCK_LIST_FILE="$2"
            shift 2
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

# 종목코드가 없는 경우 stock_list.txt 파일에서 읽기
if [ ${#STOCKS[@]} -eq 0 ]; then
    if [ -f "$STOCK_LIST_FILE" ]; then
        echo "종목코드가 지정되지 않았습니다. $STOCK_LIST_FILE 파일에서 종목 목록을 읽어옵니다."
        while IFS= read -r line || [ -n "$line" ]; do
            # 빈 줄이나 주석(#) 제외
            if [[ ! -z "$line" && ! "$line" =~ ^[[:space:]]*# ]]; then
                STOCKS+=("$line")
            fi
        done < "$STOCK_LIST_FILE"
    else
        echo "종목코드가 지정되지 않았고, $STOCK_LIST_FILE 파일도 없습니다. 기본 종목(AAPL, TSLA, AMZN)을 분석합니다."
    STOCKS=("AAPL" "TSLA" "AMZN")
    fi
fi

# 종료일이 없는 경우 현재 날짜로 설정
if [ -z "$END_DATE" ]; then
    END_DATE=$CURRENT_DATE
fi

# 각 종목 분석 실행
for stock in "${STOCKS[@]}"; do
    echo "=========================="
    if [ "$SELL_ONLY" = true ]; then
        echo "매도 신호 분석 중: $stock"
    elif [ "$BUY_ONLY" = true ]; then
        echo "매수 신호 분석 중: $stock"
    else
        echo "분석 중: $stock"
    fi
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
    
    # 매도 신호 분석 모드 옵션 추가
    if [ "$SELL_ONLY" = true ]; then
        cmd="$cmd --sell-only"
    fi
    
    # 매수 신호 분석 모드 옵션 추가
    if [ "$BUY_ONLY" = true ]; then
        cmd="$cmd --buy-only"
    fi
    
    # 박스권 분석 옵션 추가
    cmd="$cmd --box-window $BOX_WINDOW --box-threshold $BOX_THRESHOLD"
    
    # 명령 실행
    echo "실행 명령: $cmd"
    echo ""
    eval $cmd
    
    echo ""
done

# HMA와 만트라 밴드 시각화
echo "HMA와 만트라 밴드 시각화 중..."
python3 -c "
import pandas as pd
from src.visualization import plot_hma_mantra_bands
from src.data_loader import load_stock_data

symbols = ['PLTR', 'UVIX', 'VIXY', 'TQQQ', 'QQQ', 'AAPL', 'AMZN', 'GOOGL', 'META', 'MSFT', 'SMR']
for symbol in symbols:
    df = load_stock_data(symbol)
    plot_hma_mantra_bands(df, symbol)
"

echo "모든 분석이 완료되었습니다." 