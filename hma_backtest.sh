#!/bin/bash

# 가상환경 활성화
source .venv/bin/activate

# 기본값 설정
SYMBOL=""
PERIOD="24mo"
CAPITAL=1000
VIX_LOW=0
VIX_HIGH=999
VIX_BANDS=""
ENABLE_VIX_FILTER=false

# 도움말 함수
show_help() {
    echo "Usage: $0 <SYMBOL> [OPTIONS]"
    echo ""
    echo "필수 인자:"
    echo "  SYMBOL                    종목코드 (예: BAC, AAPL, TSLA)"
    echo ""
    echo "옵션:"
    echo "  -p, --period PERIOD       분석 기간 (기본값: 24mo)"
    echo "                            예: 6mo, 12mo, 24mo, 60mo"
    echo ""
    echo "  -c, --capital CAPITAL     기본 매수 금액 (기본값: 1000)"
    echo "                            VIX 필터가 비활성화된 경우 사용"
    echo ""
    echo "  --vix-low VALUE           VIX 하한값 (기본값: 0)"
    echo "                            이 값 이상일 때만 매수 신호 생성"
    echo ""
    echo "  --vix-high VALUE          VIX 상한값 (기본값: 999)"
    echo "                            이 값 이하일 때만 매수 신호 생성"
    echo ""
    echo "  --vix-bands BANDS         VIX 대역별 매수비용 설정"
    echo "                            형식: 'low:cost,mid:cost,high:cost'"
    echo "                            예: '0-20:1000,20-30:800,30+:500'"
    echo "                            또는 'low:1000,mid:800,high:500'"
    echo ""
    echo "  --enable-vix-filter       VIX 필터 활성화"
    echo "                            이 옵션이 없으면 VIX 조건 무시"
    echo "                            VIX 필터 사용 시 더 긴 기간의 VIX 데이터 로드"
    echo ""
    echo "예시:"
    echo "  $0 BAC                                    # 기본 백테스트"
    echo "  $0 BAC -p 12mo -c 1500                   # 12개월, 1500달러 매수"
    echo "  $0 BAC --vix-low 15 --vix-high 25        # VIX 15-25 범위에서만 매수"
    echo "  $0 BAC --vix-low 15 --vix-high 25 --enable-vix-filter -p 12mo"
    echo "                                          # 12개월, VIX 15-25 필터 (VIX 24개월 데이터 사용)"
    echo "  $0 BAC --vix-bands 'low:1000,mid:800,high:500' --enable-vix-filter"
    echo "                                          # VIX 대역별 매수비용 설정"
    echo "  $0 BAC --vix-bands 'low:1000,mid:800,high:500' --enable-vix-filter -p 6mo"
    echo "                                          # 6개월, VIX 대역별 설정 (VIX 12개월 데이터 사용)"
    echo ""
             echo "VIX 대역 설정:"
         echo "  low:  VIX < 20  (낮은 변동성)"
         echo "  mid:  20 ≤ VIX ≤ 25 (중간 변동성)"
         echo "  high: VIX > 25  (높은 변동성)"
    echo ""
    echo "VIX 데이터 기간 설정:"
    echo "  VIX 필터 사용 시 주식 데이터보다 더 긴 기간의 VIX 데이터를 로드합니다:"
    echo "  - 주식 6개월 → VIX 12개월"
    echo "  - 주식 12개월 → VIX 24개월"
    echo "  - 주식 24개월 → VIX 60개월"
}

# 결과 디렉토리명 생성 함수
generate_result_dir() {
    local base_dir="test/backtest_results/${SYMBOL}"
    local suffix=""
    
    # 기본 설정이 아닌 경우에만 접미사 추가
    if [ "$PERIOD" != "24mo" ]; then
        suffix="${suffix}_p${PERIOD}"
    fi
    
    if [ "$CAPITAL" != "1000" ]; then
        suffix="${suffix}_c${CAPITAL}"
    fi
    
    if [ "$ENABLE_VIX_FILTER" = true ]; then
        if [ -n "$VIX_BANDS" ]; then
            # VIX 대역 설정이 있는 경우
            suffix="${suffix}_vixbands"
        else
            # VIX 범위 설정이 있는 경우
            if [ "$VIX_LOW" != "0" ] || [ "$VIX_HIGH" != "999" ]; then
                suffix="${suffix}_vix${VIX_LOW}-${VIX_HIGH}"
            fi
        fi
    fi
    
    echo "${base_dir}${suffix}"
}

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -p|--period)
            PERIOD="$2"
            shift 2
            ;;
        -c|--capital)
            CAPITAL="$2"
            shift 2
            ;;
        --vix-low)
            VIX_LOW="$2"
            shift 2
            ;;
        --vix-high)
            VIX_HIGH="$2"
            shift 2
            ;;
        --vix-bands)
            VIX_BANDS="$2"
            shift 2
            ;;
        --enable-vix-filter)
            ENABLE_VIX_FILTER=true
            shift
            ;;
        -*)
            echo "알 수 없는 옵션: $1"
            show_help
            exit 1
            ;;
        *)
            if [ -z "$SYMBOL" ]; then
                SYMBOL="$1"
            else
                echo "중복된 SYMBOL: $1"
                show_help
                exit 1
            fi
            shift
            ;;
    esac
done

# 필수 인자 검증
if [ -z "$SYMBOL" ]; then
    echo "오류: SYMBOL이 필요합니다."
    show_help
    exit 1
fi

# VIX 필터 설정 검증
if [ "$ENABLE_VIX_FILTER" = true ]; then
    if [ -z "$VIX_BANDS" ]; then
        echo "VIX 필터 활성화됨: $VIX_LOW ≤ VIX ≤ $VIX_HIGH"
    else
        echo "VIX 대역별 매수비용 설정: $VIX_BANDS"
    fi
else
    if [ "$VIX_LOW" != "0" ] || [ "$VIX_HIGH" != "999" ] || [ -n "$VIX_BANDS" ]; then
        echo "경고: VIX 옵션이 설정되었지만 --enable-vix-filter가 없습니다."
        echo "VIX 조건이 무시되고 기본 매수 금액($CAPITAL)이 사용됩니다."
    fi
fi

# 결과 디렉토리 생성
RESULT_DIR=$(generate_result_dir)
mkdir -p "$RESULT_DIR"

# 백테스트 실행
echo "=== BAC 백테스트 시작 ==="
echo "종목: $SYMBOL"
echo "기간: $PERIOD"
echo "기본 매수 금액: $CAPITAL"
echo "VIX 필터: $ENABLE_VIX_FILTER"
echo "결과 디렉토리: $RESULT_DIR"

if [ "$ENABLE_VIX_FILTER" = true ]; then
    if [ -n "$VIX_BANDS" ]; then
        python test/hma_backtest_program.py "$SYMBOL" "$PERIOD" "$CAPITAL" --vix-bands "$VIX_BANDS" --result-dir "$RESULT_DIR"
    else
        python test/hma_backtest_program.py "$SYMBOL" "$PERIOD" "$CAPITAL" --vix-low "$VIX_LOW" --vix-high "$VIX_HIGH" --result-dir "$RESULT_DIR"
    fi
else
    python test/hma_backtest_program.py "$SYMBOL" "$PERIOD" "$CAPITAL" --result-dir "$RESULT_DIR"
fi 