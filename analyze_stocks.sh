#!/bin/bash

# 여러 종목을 일괄 처리하는 스크립트
# 사용법: ./analyze_stocks.sh [옵션]

# 기본 값 설정
PERIOD="1mo"
INTERVAL="1d"
START_DATE=""
END_DATE=""
OUTPUT_DIR=""
STOCK_LIST_FILE="stock_list.txt"
SUMMARY_FILE="stock_summary.md"

# 명령줄 인자 처리
while [[ $# -gt 0 ]]; do
    case $1 in
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
        -f|--file)
            STOCK_LIST_FILE="$2"
            shift 2
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            exit 1
            ;;
    esac
done

# 종목 리스트 파일 확인
if [ ! -f "$STOCK_LIST_FILE" ]; then
    echo "종목 리스트 파일($STOCK_LIST_FILE)이 존재하지 않습니다."
    exit 1
fi

# 현재 날짜 (YYYYMMDD 형식)
CURRENT_DATE=$(date +%Y%m%d)

# 출력 디렉토리 설정
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="output/${CURRENT_DATE}"
fi

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

# 요약 파일 헤더 생성
echo "# 종목별 기술적 분석 종합 요약 ($(date +%Y-%m-%d) 기준)" > "${OUTPUT_DIR}/${SUMMARY_FILE}"
echo "" >> "${OUTPUT_DIR}/${SUMMARY_FILE}"
echo "## 종목별 핵심 지표 비교" >> "${OUTPUT_DIR}/${SUMMARY_FILE}"
echo "" >> "${OUTPUT_DIR}/${SUMMARY_FILE}"
echo "| 종목 | 종가 | 전일대비 | 단기추세 | 중기추세 | 장기추세 | RSI | 매수점수 | 매도점수 | 매수추천 | 매도추천 |" >> "${OUTPUT_DIR}/${SUMMARY_FILE}"
echo "|------|------|---------|---------|---------|---------|-----|---------|---------|---------|---------|" >> "${OUTPUT_DIR}/${SUMMARY_FILE}"

# CSV 파일에서 마지막 행의 종가 가져오기
get_last_price() {
    local csv_file="$1"
    if [ -f "$csv_file" ]; then
        # CSV의 마지막 행 가져오기 (헤더 제외)
        tail -n 1 "$csv_file" | cut -d',' -f5  # Close 열은 보통 5번째 열
    else
        echo "N/A"
    fi
}

# 종목 분석 파일 생성 함수
create_analysis_file() {
    local stock="$1"
    local log_file="$2"
    local output_file="${OUTPUT_DIR}/${stock}/${stock}_analysis.md"
    local csv_file="${OUTPUT_DIR}/${stock}/${stock}_data.csv"
    
    # 데이터 파일이 존재하지 않으면 실패
    if [ ! -f "$csv_file" ]; then
        echo "경고: ${stock}의 데이터 파일이 없습니다."
        return 1
    fi
    
    # CSV 파일에서 직접 종가 가져오기
    local close_price=$(get_last_price "$csv_file")
    
    # 로그 파일에서 주요 데이터 추출
    local date=$(grep "기준일:" "$log_file" | head -1 | cut -d':' -f2 | tr -d ' ')
    local change_pct=$(grep "전일대비:" "$log_file" | head -1 | cut -d':' -f2 | tr -d ' ')
    local ma5=$(grep "5일 이동평균:" "$log_file" | head -1 | cut -d':' -f2 | tr -d ' ')
    local ma20=$(grep "20일 이동평균:" "$log_file" | head -1 | cut -d':' -f2 | tr -d ' ')
    local bb_upper=$(grep "볼린저 밴드 상단:" "$log_file" | head -1 | cut -d':' -f2 | tr -d ' ')
    local bb_lower=$(grep "볼린저 밴드 하단:" "$log_file" | head -1 | cut -d':' -f2 | tr -d ' ')
    local bb_pb=$(grep "볼린저 밴드 %B:" "$log_file" | head -1 | awk -F':' '{print $2}' | awk '{print $1}')
    local rsi=$(grep "RSI:" "$log_file" | head -1 | cut -d':' -f2 | tr -d ' ')
    local short_trend=$(grep "단기 추세" "$log_file" | head -1 | awk '{print $NF}')
    local mid_trend=$(grep "중기 추세" "$log_file" | head -1 | awk '{print $NF}')
    local long_trend=$(grep "장기 추세" "$log_file" | head -1 | awk '{print $NF}')
    
    # 심볼에 따라 가격 표시 방식 변경
    local price_display=""
    if [[ "$stock" == *".KS"* || "$stock" == *".KQ"* ]]; then
        price_display="₩$close_price"
    else
        price_display="\$$close_price"
    fi
    
    # 파일 헤더 생성
    echo "# ${stock} 기술적 분석 결과" > "$output_file"
    echo "" >> "$output_file"
    echo "## 1. 기본 정보 및 추세 분석" >> "$output_file"
    echo "- **현재 종가**: ${price_display} (${date} 기준)" >> "$output_file"
    echo "- **전일대비**: ${change_pct}" >> "$output_file"
    
    # 추세 상태 기록
    echo "- **추세 상태**: " >> "$output_file"
    echo "  - **단기 추세(20일)**: ${short_trend}" >> "$output_file"
    echo "  - **중기 추세(60일)**: ${mid_trend}" >> "$output_file"
    echo "  - **장기 추세(120일)**: ${long_trend}" >> "$output_file"
    echo "" >> "$output_file"
    
    # 기술적 지표 기록
    echo "## 2. 주요 기술적 지표" >> "$output_file"
    echo "- **이동평균선**:" >> "$output_file"
    echo "  - MA5: ${ma5}" >> "$output_file"
    echo "  - MA20: ${ma20}" >> "$output_file"
    echo "  " >> "$output_file"
    echo "- **볼린저 밴드**:" >> "$output_file"
    echo "  - 상단: ${bb_upper}" >> "$output_file"
    echo "  - 하단: ${bb_lower}" >> "$output_file"
    echo "  - %B: ${bb_pb}" >> "$output_file"
    echo "  " >> "$output_file"
    echo "- **모멘텀 지표**:" >> "$output_file"
    echo "  - RSI: ${rsi}" >> "$output_file"
    echo "" >> "$output_file"
    
    # 매수/매도 신호 분석
    echo "## 3. 매수/매도 신호 분석" >> "$output_file"
    
    # 매수 신호 섹션 추출 및 기록
    echo "### 매수 신호" >> "$output_file"
    echo '```' >> "$output_file"
    awk '/최근 10일간 매수 신호 점수:/,/최종 추천:/' "$log_file" >> "$output_file"
    echo '```' >> "$output_file"
    echo "" >> "$output_file"
    
    # 매도 신호 섹션 추출 및 기록
    echo "### 매도 신호" >> "$output_file"
    echo '```' >> "$output_file"
    awk '/최근 10일간 매도 신호 점수:/,/최종 매도 추천:/' "$log_file" >> "$output_file"
    echo '```' >> "$output_file"
    echo "" >> "$output_file"
    
    # 최근 매수/매도 점수 추출
    local buy_score=$(grep -A 12 "최근 10일간 매수 신호 점수:" "$log_file" | grep "2025-05-" | tail -1 | awk -F':' '{print $2}' | awk '{print $1}')
    local sell_score=$(grep -A 12 "최근 10일간 매도 신호 점수:" "$log_file" | grep "2025-05-" | tail -1 | awk -F':' '{print $2}' | awk '{print $1}')
    local buy_rec=$(grep -A 1 "최종 추천:" "$log_file" | tail -1)
    local sell_rec=$(grep -A 1 "최종 매도 추천:" "$log_file" | tail -1)
    
    # 종합 평가
    echo "## 4. 종합 평가" >> "$output_file"
    echo "" >> "$output_file"
    
    # 매수/매도 신호 강도에 따른 평가
    echo "### 매수 관점" >> "$output_file"
    # 소수점 처리를 위해 정수로 변환
    local buy_score_int=$(echo "$buy_score * 10" | bc | cut -d. -f1)
    
    if [ "$buy_score_int" -gt 500 ]; then
        echo "- 현재 매수 신호가 강합니다 (점수: ${buy_score}점)" >> "$output_file"
        echo "- 적극적인 매수 고려 가능합니다" >> "$output_file"
    elif [ "$buy_score_int" -gt 300 ]; then
        echo "- 현재 매수 신호가 중간 수준입니다 (점수: ${buy_score}점)" >> "$output_file"
        echo "- 부분적인 매수 고려 가능합니다" >> "$output_file"
    else
        echo "- 현재 매수 신호가 약합니다 (점수: ${buy_score}점)" >> "$output_file"
        echo "- 신규 매수는 신중하게 접근해야 합니다" >> "$output_file"
    fi
    
    echo "" >> "$output_file"
    echo "### 매도 관점" >> "$output_file"
    # 소수점 처리를 위해 정수로 변환
    local sell_score_int=$(echo "$sell_score * 10" | bc | cut -d. -f1)
    
    if [ "$sell_score_int" -gt 500 ]; then
        echo "- 현재 매도 신호가 강합니다 (점수: ${sell_score}점)" >> "$output_file"
        echo "- 보유 중인 경우 매도 고려가 필요합니다" >> "$output_file"
    elif [ "$sell_score_int" -gt 300 ]; then
        echo "- 현재 매도 신호가 중간 수준입니다 (점수: ${sell_score}점)" >> "$output_file"
        echo "- 일부 이익 실현 고려 가능합니다" >> "$output_file"
    else
        echo "- 현재 매도 신호가 약합니다 (점수: ${sell_score}점)" >> "$output_file"
        echo "- 급격한 매도 압력은 없습니다" >> "$output_file"
    fi
    
    echo "" >> "$output_file"
    echo "## 5. 주의사항" >> "$output_file"
    echo "- 이 분석은 기술적 지표에 기반한 것으로, 투자 결정을 위한 참고 자료입니다." >> "$output_file"
    echo "- 실제 투자 결정은 개인의 투자 목표, 위험 감수 능력, 시장 상황 등을 종합적으로 고려해야 합니다." >> "$output_file"
    echo "- 과거 데이터에 기반한 분석이므로 미래 주가를 정확히 예측하지 못할 수 있습니다." >> "$output_file"
    
    echo "${stock} 분석 파일 생성 완료: ${output_file}"
    
    # 종합 파일에 정보 추가
    add_to_summary "$stock" "$price_display" "$change_pct" "$short_trend" "$mid_trend" "$long_trend" "$rsi" "$buy_score" "$sell_score" "$buy_rec" "$sell_rec"
}

# 요약 파일에 정보 추가 함수
add_to_summary() {
    local stock="$1"
    local price="$2"
    local change="$3"
    local short_trend="$4"
    local mid_trend="$5"
    local long_trend="$6"
    local rsi="$7"
    local buy_score="$8"
    local sell_score="$9"
    local buy_rec="${10}"
    local sell_rec="${11}"
    
    # 요약 파일에 추가
    echo "| $stock | $price | $change | $short_trend | $mid_trend | $long_trend | $rsi | ${buy_score}점 | ${sell_score}점 | $buy_rec | $sell_rec |" >> "${OUTPUT_DIR}/${SUMMARY_FILE}"
}

# 메인 처리 함수
process_stocks() {
    local total_stocks=$(grep -v '^#' "$STOCK_LIST_FILE" | grep -v '^$' | wc -l)
    local current=0
    
    echo "총 ${total_stocks}개 종목 분석 시작..."
    
    # 종목 리스트를 순회하며 처리
    while IFS= read -r stock; do
        # 빈 줄이나 주석은 무시
        if [[ -z "$stock" || "$stock" =~ ^# ]]; then
            continue
        fi
        
        current=$((current + 1))
        echo "[$current/$total_stocks] $stock 분석 중..."
        
        # 로그 파일 경로
        log_file="${OUTPUT_DIR}/${stock}_analysis.log"
        
        # 기본 명령 구성
        cmd="./main.sh -p $PERIOD -i $INTERVAL"
        
        # 날짜 범위 추가
        if [ -n "$START_DATE" ]; then
            cmd="$cmd -s $START_DATE"
        fi
        if [ -n "$END_DATE" ]; then
            cmd="$cmd -e $END_DATE"
        fi
        
        # 출력 디렉토리 추가
        if [ -n "$OUTPUT_DIR" ]; then
            cmd="$cmd -o $OUTPUT_DIR/$stock"
        fi
        
        # 종목 코드 추가
        cmd="$cmd $stock"
        
        # 명령 실행 및 로그 저장
        echo "실행: $cmd"
        mkdir -p "${OUTPUT_DIR}/${stock}"
        eval $cmd > "$log_file" 2>&1
        
        # 분석 파일 생성
        create_analysis_file "$stock" "$log_file"
        
        echo "-----------------------------------"
    done < "$STOCK_LIST_FILE"
    
    echo "모든 종목 분석 완료!"
    echo "종합 요약 파일: ${OUTPUT_DIR}/${SUMMARY_FILE}"
}

# 메인 처리 실행
process_stocks

# 종합 요약 보완
echo "" >> "${OUTPUT_DIR}/${SUMMARY_FILE}"
echo "## 종목별 상세 분석" >> "${OUTPUT_DIR}/${SUMMARY_FILE}"
echo "" >> "${OUTPUT_DIR}/${SUMMARY_FILE}"
echo "각 종목의 상세 분석 결과는 다음 파일에서 확인할 수 있습니다:" >> "${OUTPUT_DIR}/${SUMMARY_FILE}"
echo "" >> "${OUTPUT_DIR}/${SUMMARY_FILE}"

# 종목별 파일 경로 추가
while IFS= read -r stock; do
    if [[ -z "$stock" || "$stock" =~ ^# ]]; then
        continue
    fi
    
    if [ -f "${OUTPUT_DIR}/${stock}/${stock}_analysis.md" ]; then
        echo "- [${stock} 분석](${stock}/${stock}_analysis.md)" >> "${OUTPUT_DIR}/${SUMMARY_FILE}"
    fi
done < "$STOCK_LIST_FILE"

echo "" >> "${OUTPUT_DIR}/${SUMMARY_FILE}"
echo "---" >> "${OUTPUT_DIR}/${SUMMARY_FILE}"
echo "분석 기간: $PERIOD, 간격: $INTERVAL" >> "${OUTPUT_DIR}/${SUMMARY_FILE}"
if [ -n "$START_DATE" ]; then
    echo "분석 날짜 범위: $START_DATE ~ $END_DATE" >> "${OUTPUT_DIR}/${SUMMARY_FILE}"
fi
echo "생성 시간: $(date '+%Y-%m-%d %H:%M:%S')" >> "${OUTPUT_DIR}/${SUMMARY_FILE}" 