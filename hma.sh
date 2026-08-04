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
USE_CURRENT_DATE=false  # 현재일시 기준 데이터 수집 옵션 추가
AUTO_ADJUST="false"  # 기본값: unadjusted 가격 사용
ELLIOTT_WAVE="false"  # 엘리엇 파동 분석 옵션
ELLIOTT_MIN_WAVE_SIZE="3.0"  # 엘리엇 파동 최소 크기
ELLIOTT_ZIGZAG_THRESHOLD="5.0"  # 엘리엇 파동 ZigZag 임계값
# 박스권 옵션들
SHOW_BOX_RANGES="true"  # 박스권 표시 여부
BOX_PERIOD="20"  # 박스권 계산 기간
NUM_BOXES="2"  # 표시할 박스권 개수
BOX_OVERLAP="5"  # 박스권 간 겹치는 일수
BOX_STYLE="default"  # 박스권 스타일 ('default', 'gradient', 'rainbow')
AVOID_TIME_OVERLAP="true"  # 시간축 겹침 방지 여부
SHOW_PATTERN_STRIP="false"  # -v overlay 시 메인 아래 차트 패턴 스트립(CHART_PATTERN_CRITERIA 행 수)
# 메인 차트 패턴 오버레이 (--pattern-main, 쉼표 구분)
# 패턴 ID: double_top_m, bear_flag, bear_diamond, range_box, triple_bottom_w, bull_flag, asc_triangle | all 또는 *
PATTERN_MAIN=""
PATTERN_RANGE_BOX_MAIN_MAX="1"  # 메인 range_box 표시 최대 개수(선별). 스트립은 전체
PATTERN_MAIN_MIN_CONFIDENCE="0.0"  # 메인 패턴 최소 신뢰도(스트립 영향 없음)
SHOW_DISPARITY_STRATEGY="false"
DISPARITY_MA="20"
DISPARITY_LOW="95"
DISPARITY_HIGH="105"
DISPARITY_CONF_WEIGHT_DEPTH="0.22"
DISPARITY_CONF_WEIGHT_VOLUME="0.16"
DISPARITY_CONF_WEIGHT_TREND="0.10"
DISPARITY_CONF_PRESET=""
SHOW_ADX_DMI="false"
ADX_PERIOD="14"
ADX_SIDEWAYS="20"
ADX_TREND="25"
ADX_STRONG="40"
TECH_CHART="false"
TECH_CHART_ONLY="false"
SHOW_BB="false"
FILTER_SURGE_SETUP="false"
FILTER_SURGE_MIN_SCORE="40"
FILTER_SURGE_INCLUDE_ACTIVE="false"
FILTER_SURGE_KINDS=""
FILTER_SURGE_OUT_DIR="output/analysis"

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
    --current)
      USE_CURRENT_DATE=true
      shift
      ;;
    --adjusted)
      AUTO_ADJUST="true"
      shift
      ;;
    --unadjusted)
      AUTO_ADJUST="false"
      shift
      ;;
    -e|--elliott)
      ELLIOTT_WAVE="true"
      shift
      ;;
    --elliott-min-wave)
      ELLIOTT_MIN_WAVE_SIZE="$2"
      shift 2
      ;;
    --elliott-zigzag)
      ELLIOTT_ZIGZAG_THRESHOLD="$2"
      shift 2
      ;;
    --no-box-ranges)
      SHOW_BOX_RANGES="false"
      shift
      ;;
    --box-period)
      BOX_PERIOD="$2"
      shift 2
      ;;
    --num-boxes)
      NUM_BOXES="$2"
      shift 2
      ;;
    --box-overlap)
      BOX_OVERLAP="$2"
      shift 2
      ;;
    --box-style)
      BOX_STYLE="$2"
      shift 2
      ;;
    --allow-time-overlap)
      AVOID_TIME_OVERLAP="false"
      shift
      ;;
    --show-pattern-strip)
      SHOW_PATTERN_STRIP="true"
      shift
      ;;
    --pattern-main)
      PATTERN_MAIN="$2"
      shift 2
      ;;
    --pattern-range-box-main-max)
      PATTERN_RANGE_BOX_MAIN_MAX="$2"
      shift 2
      ;;
    --pattern-main-min-confidence)
      PATTERN_MAIN_MIN_CONFIDENCE="$2"
      shift 2
      ;;
    --show-disparity-strategy)
      SHOW_DISPARITY_STRATEGY="true"
      shift
      ;;
    --disparity-ma)
      DISPARITY_MA="$2"
      shift 2
      ;;
    --disparity-low)
      DISPARITY_LOW="$2"
      shift 2
      ;;
    --disparity-high)
      DISPARITY_HIGH="$2"
      shift 2
      ;;
    --disparity-conf-weight-depth)
      DISPARITY_CONF_WEIGHT_DEPTH="$2"
      shift 2
      ;;
    --disparity-conf-weight-volume)
      DISPARITY_CONF_WEIGHT_VOLUME="$2"
      shift 2
      ;;
    --disparity-conf-weight-trend)
      DISPARITY_CONF_WEIGHT_TREND="$2"
      shift 2
      ;;
    --disparity-conf-preset)
      DISPARITY_CONF_PRESET="$2"
      shift 2
      ;;
    --show-adx-dmi)
      SHOW_ADX_DMI="true"
      shift
      ;;
    --adx-period)
      ADX_PERIOD="$2"
      shift 2
      ;;
    --adx-sideways)
      ADX_SIDEWAYS="$2"
      shift 2
      ;;
    --adx-trend)
      ADX_TREND="$2"
      shift 2
      ;;
    --adx-strong)
      ADX_STRONG="$2"
      shift 2
      ;;
    --tech-chart)
      TECH_CHART="true"
      shift
      ;;
    --tech-chart-only)
      TECH_CHART_ONLY="true"
      shift
      ;;
    --show-bb)
      SHOW_BB="true"
      shift
      ;;
    --filter-surge-setup)
      FILTER_SURGE_SETUP="true"
      shift
      ;;
    --filter-surge-min-score)
      FILTER_SURGE_MIN_SCORE="$2"
      shift 2
      ;;
    --filter-surge-include-active)
      FILTER_SURGE_INCLUDE_ACTIVE="true"
      shift
      ;;
    --filter-surge-kinds)
      FILTER_SURGE_KINDS="$2"
      shift 2
      ;;
    --filter-surge-out-dir)
      FILTER_SURGE_OUT_DIR="$2"
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

# 현재일시 기준 데이터 수집 처리
if [ "$USE_CURRENT_DATE" = true ]; then
  echo "현재일시 기준으로 데이터를 수집합니다."
  # 현재 한국 시간 기준으로 오늘 날짜 계산
  CURRENT_DATE=$(TZ=Asia/Seoul date +"%Y-%m-%d")
  TO_DATE="$CURRENT_DATE"
  
  # PERIOD에서 숫자와 단위 분리하여 FROM_DATE 계산
  if [[ "$PERIOD" =~ ^([0-9]+)([dmy]|mo)$ ]]; then
    amount=${BASH_REMATCH[1]}
    unit=${BASH_REMATCH[2]}
    
    # Python을 사용하여 현재일시로부터 이전 날짜 계산
    FROM_DATE=$(python3 -c "
from datetime import datetime, timedelta
import sys

current_date = '$CURRENT_DATE'
amount = $amount
unit = '$unit'

# 현재일시를 기준으로 FROM_DATE 계산
current = datetime.strptime(current_date, '%Y-%m-%d')

if unit == 'd':
    from_date = current - timedelta(days=amount)
elif unit == 'm' or unit == 'mo':
    from_date = current - timedelta(days=amount * 30)
elif unit == 'y':
    from_date = current - timedelta(days=amount * 365)
else:
    from_date = current - timedelta(days=amount)

print(from_date.strftime('%Y-%m-%d'))
")
    echo "현재일시: $CURRENT_DATE"
    echo "분석 기간: $FROM_DATE ~ $TO_DATE (${PERIOD})"
  fi
fi

# 날짜 범위 처리 (기존 로직)
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
  echo "Usage: $0 (-s SYMBOL | --file SYMBOL_FILE) [-p PERIOD | -d DAYS | --from START_DATE --to END_DATE] [-i INTERVAL] [-t PREPOST] [-a ANALYSIS_TYPE] [-v VOLUME_PROFILE_TYPE] [-e|--elliott] [박스권 옵션들]"
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
  echo "  -v TYPE            Volume Profile 유형 (none/separate/overlay/compare, 기본값: none)"
  echo "                      none: Volume Profile 없음"
  echo "                      separate: 별도 영역에 Volume Profile"
  echo "                      overlay: 메인차트에 Volume Profile 오버레이"
  echo "                      compare: 다중 종목 비교 분석 (QQQ,TQQQ,SQQQ,SPY)"
  echo "  -e, --elliott      엘리엇 파동 분석 활성화"
  echo "  --elliott-min-wave 엘리엇 파동 최소 크기 (%, 기본값: 3.0)"
  echo "  --elliott-zigzag   엘리엇 파동 ZigZag 임계값 (%, 기본값: 5.0)"
  echo ""
  echo "박스권 옵션들:"
  echo "  --no-box-ranges    박스권 표시 비활성화"
  echo "  --box-period       박스권 계산 기간 (일수, 기본값: 20)"
  echo "  --num-boxes        표시할 박스권 개수 (기본값: 2)"
  echo "  --box-overlap      박스권 간 겹치는 일수 (기본값: 5)"
  echo "  --box-style        박스권 스타일 (default/gradient/rainbow, 기본값: default)"
  echo "  --allow-time-overlap 시간축 겹침 허용 (기본값: 겹침방지)"
  echo "  --show-pattern-strip  -v overlay일 때 차트 패턴 타임라인 서브플롯 표시"
  echo "  --pattern-main LIST   메인 차트 패턴 오버레이 (쉼표로 패턴 ID 나열). all 또는 *=전 패턴 id. -v overlay 전용"
  echo "                        패턴 ID (chart_patterns.py CHART_PATTERN_CRITERIA와 동일):"
  echo "                          double_top_m      쌍봉(M)"
  echo "                          bear_flag         하락깃발"
  echo "                          bear_diamond      하락다이아"
  echo "                          range_box         박스권"
  echo "                          triple_bottom_w   역삼중창(W)"
  echo "                          bull_flag         상승깃발"
  echo "                          asc_triangle      상승삼각"
  echo "                          post_box_bull     박스후장대양봉"
  echo "                          fvg_gap           FVG(갭)"
  echo "                        예: --pattern-main triple_bottom_w,double_top_m  |  --pattern-main all"
  echo "  --pattern-range-box-main-max N  메인 range_box 최대 N건(신뢰도·종료일 선별, 기본 1). 0=메인 미표시"
  echo "  --pattern-main-min-confidence F 메인 패턴 최소 신뢰도(0~1, 기본 0.0). 스트립에는 영향 없음"
  echo "  --show-disparity-strategy    이격도(95/105) 전략 마커/요약 표시"
  echo "  --disparity-ma N             이격도 이동평균 기간(기본 20)"
  echo "  --disparity-low F            이격도 저평가 임계값(기본 95)"
  echo "  --disparity-high F           이격도 고평가 임계값(기본 105)"
  echo "  --disparity-conf-weight-depth F  이격도 conf 깊이 가중치(기본 0.22)"
  echo "  --disparity-conf-weight-volume F 이격도 conf 거래량 가중치(기본 0.16)"
  echo "  --disparity-conf-weight-trend F  이격도 conf 추세 가중치(기본 0.10)"
  echo "  --disparity-conf-preset P   이격도 conf 프리셋(growth|balanced|defensive)"
  echo "  --show-adx-dmi              ADX·DMI(+DI/−DI) 추세·강도 서브플롯 표시"
  echo "  --adx-period N              ADX/DMI 기간 (기본 14)"
  echo "  --adx-sideways F            ADX 횡보 임계 (기본 20)"
  echo "  --adx-trend F               ADX 추세 시작 (기본 25)"
  echo "  --adx-strong F              ADX 강추세 (기본 40)"
  echo "  --tech-chart                기술 분석 주석 차트(별도 PNG) 추가 생성"
  echo "  --tech-chart-only           기술 분석 주석 차트만 생성 (메인 차트 생략)"
  echo "  --show-bb                   기술 분석 주석 차트에 볼린저 밴드 선 표시"
  echo ""
  echo "급등 셋업 필터:"
  echo "  --filter-surge-setup              종목 목록(--file)에서 급등 셋업/급등 후보만 필터 (분석 생략)"
  echo "  --filter-surge-min-score N        최소 점수 (기본 40)"
  echo "  --filter-surge-include-active     이미 급등진행/급등과열 종목도 포함"
  echo "  --filter-surge-kinds A,B          셋업 종류만 (예: A 또는 A,B)"
  echo "  --filter-surge-out-dir DIR        결과 저장 경로 (기본 output/analysis)"
  echo "  예: ./hma.sh --filter-surge-setup -f stocks/us_Harris_Wish.txt -p 6mo"
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
  echo "  overlay + 패턴스트립 예: ./hma.sh --from 2025-04-01 -v overlay --num-boxes 7 -s NVDA --show-pattern-strip"
  echo "  --to 2025-08-01 -p 30d             2025-08-01로부터 30일 전까지"
  echo "  --to 2025-08-01 -p 6mo             2025-08-01로부터 6개월 전까지"
  echo "  --to 2025-08-01 -p 1y              2025-08-01로부터 1년 전까지"
  exit 1
fi

# 급등 셋업 필터 모드: 목록만 걸러내고 종료 (전체 분석 루프 생략)
if [ "$FILTER_SURGE_SETUP" = "true" ]; then
  if [ -z "$SYMBOL_FILE" ]; then
    echo "오류: --filter-surge-setup 은 --file SYMBOL_FILE 이 필요합니다."
    exit 1
  fi
  if [ ! -f "$SYMBOL_FILE" ]; then
    echo "오류: 종목 파일을 찾을 수 없습니다: $SYMBOL_FILE"
    exit 1
  fi
  FILTER_ARGS=(
    python filter_surge_setup.py
    --file "$SYMBOL_FILE"
    -p "$PERIOD"
    --min-score "$FILTER_SURGE_MIN_SCORE"
    --out-dir "$FILTER_SURGE_OUT_DIR"
  )
  if [ "$FILTER_SURGE_INCLUDE_ACTIVE" = "true" ]; then
    FILTER_ARGS+=(--include-active)
  fi
  if [ -n "$FILTER_SURGE_KINDS" ]; then
    FILTER_ARGS+=(--kinds "$FILTER_SURGE_KINDS")
  fi
  echo "급등 셋업 필터 실행: $SYMBOL_FILE"
  "${FILTER_ARGS[@]}"
  exit $?
fi

# 분석 함수 정의
analyze_stock() {
    local symbol=$1
    local output_dir="output/analysis/$symbol"
    mkdir -p "$output_dir"
    
    case $ANALYSIS_TYPE in
        "technical"|"all")
            # 비교 분석 모드 확인
            if [ "$VOLUME_PROFILE_TYPE" = "compare" ]; then
                echo "다중 종목 비교 분석 시작..."
                if [ -z "$SYMBOL" ]; then
                    echo "오류: 비교 분석을 위해서는 -s 옵션으로 종목을 지정해야 합니다."
                    echo "예: ./hma.sh -a technical -v compare --from 2024-01-01 -s QQQ,TQQQ,SQQQ,SPY"
                    exit 1
                fi
                
                # 종목 리스트를 쉼표로 분리
                IFS=',' read -ra SYMBOLS <<< "$SYMBOL"
                if [ ${#SYMBOLS[@]} -lt 2 ]; then
                    echo "오류: 비교 분석을 위해서는 최소 2개 이상의 종목이 필요합니다."
                    echo "예: ./hma.sh -a technical -v compare --from 2024-01-01 -s QQQ,TQQQ,SQQQ,SPY"
                    exit 1
                fi
                
                # 비교 분석 실행
                python src/indicators/hma_mantra/visualization/comparison_chart.py \
                    --symbols "$SYMBOL" \
                    --from "$FROM_DATE" \
                    --to "$TO_DATE" \
                    --auto_adjust "$AUTO_ADJUST"
                echo "다중 종목 비교 분석 완료!"
                return
            fi
            
            echo "기술적 분석 시작..."
            case $VOLUME_PROFILE_TYPE in
                "separate")
                    echo "Volume Profile (별도 영역) 생성 중..."
                    python test/volume_profile_test.py "$symbol" "$PERIOD"
                    ;;
                "overlay")
                    echo "Volume Profile (오버레이) 생성 중..."
                    python test/volume_profile_overlay_test.py "$symbol" "$PERIOD" --auto_adjust "$AUTO_ADJUST" \
                        --show-box-ranges "$SHOW_BOX_RANGES" --box-period "$BOX_PERIOD" \
                        --num-boxes "$NUM_BOXES" --box-overlap "$BOX_OVERLAP" --box-style "$BOX_STYLE" \
                        --avoid-time-overlap "$AVOID_TIME_OVERLAP" \
                        --pattern-range-box-main-max "$PATTERN_RANGE_BOX_MAIN_MAX" \
                        --pattern-main-min-confidence "$PATTERN_MAIN_MIN_CONFIDENCE" \
                        --disparity-ma "$DISPARITY_MA" \
                        --disparity-low "$DISPARITY_LOW" \
                        --disparity-high "$DISPARITY_HIGH" \
                        --disparity-conf-weight-depth "$DISPARITY_CONF_WEIGHT_DEPTH" \
                        --disparity-conf-weight-volume "$DISPARITY_CONF_WEIGHT_VOLUME" \
                        --disparity-conf-weight-trend "$DISPARITY_CONF_WEIGHT_TREND" \
                        $( [ -n "$DISPARITY_CONF_PRESET" ] && echo --disparity-conf-preset "$DISPARITY_CONF_PRESET" ) \
                        $( [ "$SHOW_DISPARITY_STRATEGY" = "true" ] && echo --show-disparity-strategy ) \
                        $( [ "$SHOW_ADX_DMI" = "true" ] && echo --show-adx-dmi ) \
                        --adx-period "$ADX_PERIOD" \
                        --adx-sideways "$ADX_SIDEWAYS" \
                        --adx-trend "$ADX_TREND" \
                        --adx-strong "$ADX_STRONG" \
                        $( [ "$TECH_CHART" = "true" ] && echo --tech-chart ) \
                        $( [ "$TECH_CHART_ONLY" = "true" ] && echo --tech-chart-only ) \
                        $( [ "$SHOW_BB" = "true" ] && echo --show-bb ) \
                        $( [ "$SHOW_PATTERN_STRIP" = "true" ] && echo --show-pattern-strip ) \
                        $( [ -n "$PATTERN_MAIN" ] && echo --pattern-main "$PATTERN_MAIN" )
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
                python test/volume_profile_overlay_test.py "$symbol" "$PERIOD" --auto_adjust "$AUTO_ADJUST" \
                    --show-box-ranges "$SHOW_BOX_RANGES" --box-period "$BOX_PERIOD" \
                    --num-boxes "$NUM_BOXES" --box-overlap "$BOX_OVERLAP" --box-style "$BOX_STYLE" \
                    --avoid-time-overlap "$AVOID_TIME_OVERLAP" \
                    --pattern-range-box-main-max "$PATTERN_RANGE_BOX_MAIN_MAX" \
                    --pattern-main-min-confidence "$PATTERN_MAIN_MIN_CONFIDENCE" \
                    --disparity-ma "$DISPARITY_MA" \
                    --disparity-low "$DISPARITY_LOW" \
                    --disparity-high "$DISPARITY_HIGH" \
                    --disparity-conf-weight-depth "$DISPARITY_CONF_WEIGHT_DEPTH" \
                    --disparity-conf-weight-volume "$DISPARITY_CONF_WEIGHT_VOLUME" \
                    --disparity-conf-weight-trend "$DISPARITY_CONF_WEIGHT_TREND" \
                    $( [ -n "$DISPARITY_CONF_PRESET" ] && echo --disparity-conf-preset "$DISPARITY_CONF_PRESET" ) \
                    $( [ "$SHOW_DISPARITY_STRATEGY" = "true" ] && echo --show-disparity-strategy ) \
                    $( [ "$SHOW_ADX_DMI" = "true" ] && echo --show-adx-dmi ) \
                    --adx-period "$ADX_PERIOD" \
                    --adx-sideways "$ADX_SIDEWAYS" \
                    --adx-trend "$ADX_TREND" \
                    --adx-strong "$ADX_STRONG" \
                    $( [ "$TECH_CHART" = "true" ] && echo --tech-chart ) \
                    $( [ "$TECH_CHART_ONLY" = "true" ] && echo --tech-chart-only ) \
                    $( [ "$SHOW_BB" = "true" ] && echo --show-bb ) \
                    $( [ "$SHOW_PATTERN_STRIP" = "true" ] && echo --show-pattern-strip ) \
                    $( [ -n "$PATTERN_MAIN" ] && echo --pattern-main "$PATTERN_MAIN" )
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
    
    # 엘리엇 파동 분석 (옵션이 활성화된 경우)
    if [ "$ELLIOTT_WAVE" = "true" ]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "🌊 엘리엇 파동 분석 시작..."
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # FROM_DATE와 TO_DATE 설정
        if [ -z "$FROM_DATE" ] || [ -z "$TO_DATE" ]; then
            # PERIOD 파싱해서 사용
            python test/elliott_wave_test.py "$symbol" "$PERIOD" \
                --min-wave-size "$ELLIOTT_MIN_WAVE_SIZE" \
                --zigzag-threshold "$ELLIOTT_ZIGZAG_THRESHOLD" \
                --auto_adjust "$AUTO_ADJUST"
        else
            # 날짜 범위로 사용
            python test/elliott_wave_test.py "$symbol" \
                --from "$FROM_DATE" \
                --to "$TO_DATE" \
                --min-wave-size "$ELLIOTT_MIN_WAVE_SIZE" \
                --zigzag-threshold "$ELLIOTT_ZIGZAG_THRESHOLD" \
                --auto_adjust "$AUTO_ADJUST"
        fi
        
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "✅ 엘리엇 파동 분석 완료!"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    fi
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
echo "종목,신호(HMA),T-POC강도,C-POC강도,PCR전략,투자액션(RSI+ISI),날짜,현재가,C-POC범위,T-POC범위" > "$SUMMARY_FILE"

# 현재 날짜 가져오기 (한국 시간 기준)
CURRENT_DATE=$(TZ=Asia/Seoul date +"%Y-%m-%d")

if [ ! -z "$SYMBOL_FILE" ]; then
  while IFS= read -r symbol || [ -n "$symbol" ]; do
    [[ $symbol =~ ^#.*$ ]] && continue
    [[ -z "${symbol// }" ]] && continue
    
    # 가장 최근에 생성된 신호 파일 찾기
  SIGNAL_FILE=$(find "output/hma_mantra/$symbol/" -name "*_signal.txt" -type f | sort | tail -1)
    if [ -f "$SIGNAL_FILE" ]; then
      # 신호 요약 섹션에서 마지막 줄의 신호만 추출
      SIGNAL=$(grep "=== 신호 요약 ===" -A 1 "$SIGNAL_FILE" | tail -1 | tr -d '\r')
      if [ -z "$SIGNAL" ]; then
        # 신호 요약이 없으면 파일의 마지막 줄 사용
        SIGNAL=$(tail -1 "$SIGNAL_FILE" | tr -d '\r')
      fi
      # 현재가 추출
      CURRENT_PRICE=$(grep "현재가:" "$SIGNAL_FILE" | sed 's/.*현재가: //' | tr -d '\r')
      
      # PCR 전략과 투자액션 추출
      PCR_STRATEGY=$(grep "PCR 전략:" "$SIGNAL_FILE" | sed 's/.*PCR 전략: //' | tr -d '\r')
      INVESTMENT_ACTION=$(grep "투자액션(RSI+ISI):" "$SIGNAL_FILE" | sed 's/.*투자액션(RSI+ISI): //' | tr -d '\r')
      
      # POC 정보 추출
      C_POC_RANGE=$(grep "C-POC:" "$SIGNAL_FILE" | sed 's/.*C-POC: [0-9.]* (\([^)]*\)), 강도: \([^)]*\).*/\1/' | tr -d '\r')
      C_POC_STRENGTH=$(grep "C-POC:" "$SIGNAL_FILE" | sed 's/.*강도: \([^)]*\).*/\1/' | tr -d '\r')
      T_POC_RANGE=$(grep "T-POC:" "$SIGNAL_FILE" | sed 's/.*T-POC: [0-9.]* (\([^)]*\)), 강도: \([^)]*\).*/\1/' | tr -d '\r')
      T_POC_STRENGTH=$(grep "T-POC:" "$SIGNAL_FILE" | sed 's/.*강도: \([^)]*\).*/\1/' | tr -d '\r')
      
      echo "$symbol,$SIGNAL,$T_POC_STRENGTH,$C_POC_STRENGTH,$PCR_STRATEGY,$INVESTMENT_ACTION,$CURRENT_DATE,$CURRENT_PRICE,$C_POC_RANGE,$T_POC_RANGE" >> "$SUMMARY_FILE"
    else
      echo "$symbol,NO_SIGNAL,,,,,,$CURRENT_DATE,,,,," >> "$SUMMARY_FILE"
    fi
  done < "$SYMBOL_FILE"
else
  # 가장 최근에 생성된 신호 파일 찾기
  SIGNAL_FILE=$(find "output/hma_mantra/$SYMBOL/" -name "*_signal.txt" -type f | sort | tail -1)
  if [ -f "$SIGNAL_FILE" ]; then
    # 신호 요약 섹션에서 마지막 줄의 신호만 추출
    SIGNAL=$(grep "=== 신호 요약 ===" -A 1 "$SIGNAL_FILE" | tail -1 | tr -d '\r')
    if [ -z "$SIGNAL" ]; then
      # 신호 요약이 없으면 파일의 마지막 줄 사용
      SIGNAL=$(tail -1 "$SIGNAL_FILE" | tr -d '\r')
    fi
            # 현재가 추출
        CURRENT_PRICE=$(grep "현재가:" "$SIGNAL_FILE" | sed 's/.*현재가: //' | tr -d '\r')
        
        # PCR 전략과 투자액션 추출
        PCR_STRATEGY=$(grep "PCR 전략:" "$SIGNAL_FILE" | sed 's/.*PCR 전략: //' | tr -d '\r')
        INVESTMENT_ACTION=$(grep "투자액션(RSI+ISI):" "$SIGNAL_FILE" | sed 's/.*투자액션(RSI+ISI): //' | tr -d '\r')
        
        # POC 정보 추출
        C_POC_RANGE=$(grep "C-POC:" "$SIGNAL_FILE" | sed 's/.*C-POC: [0-9.]* (\([^)]*\)), 강도: \([^)]*\).*/\1/' | tr -d '\r')
        C_POC_STRENGTH=$(grep "C-POC:" "$SIGNAL_FILE" | sed 's/.*강도: \([^)]*\).*/\1/' | tr -d '\r')
        T_POC_RANGE=$(grep "T-POC:" "$SIGNAL_FILE" | sed 's/.*T-POC: [0-9.]* (\([^)]*\)), 강도: \([^)]*\).*/\1/' | tr -d '\r')
        T_POC_STRENGTH=$(grep "T-POC:" "$SIGNAL_FILE" | sed 's/.*강도: \([^)]*\).*/\1/' | tr -d '\r')
        
        echo "$SYMBOL,$SIGNAL,$T_POC_STRENGTH,$C_POC_STRENGTH,$PCR_STRATEGY,$INVESTMENT_ACTION,$CURRENT_DATE,$CURRENT_PRICE,$C_POC_RANGE,$T_POC_RANGE" >> "$SUMMARY_FILE"
  else
    echo "$SYMBOL,NO_SIGNAL,,,,,,$CURRENT_DATE,,,,," >> "$SUMMARY_FILE"
  fi
fi

echo -e "\n=== 매수/매도 신호 요약 ==="
echo "파일 위치: $SUMMARY_FILE"
echo "-------------------"
cat "$SUMMARY_FILE"
echo "-------------------"

# 매수시그널 테이블 파일 경로 출력
if [ ! -z "$SYMBOL" ]; then
    BUY_SIGNALS_TABLE="output/hma_mantra/${SYMBOL}/${SYMBOL}_buy_signals_table.txt"
    if [ -f "$BUY_SIGNALS_TABLE" ]; then
        echo -e "\n📊 매수시그널 테이블 파일:"
        echo "   $BUY_SIGNALS_TABLE"
    fi
elif [ ! -z "$SYMBOL_FILE" ]; then
    # 파일 목록인 경우 일자별 매수시그널 테이블 생성
    echo -e "\n📊 일자별 매수시그널 테이블 생성 중..."
    
    # 날짜 범위 추출
    if [ ! -z "$FROM_DATE" ]; then
        START_DATE="$FROM_DATE"
    else
        START_DATE="2025-01-01"
    fi
    
    if [ ! -z "$TO_DATE" ]; then
        END_DATE="$TO_DATE"
    else
        END_DATE=$(date +%Y-%m-%d)
    fi
    
    # Python 스크립트 실행하여 일자별 테이블 생성
    BUY_SIGNALS_BY_DATE=$(python generate_buy_signals_by_date.py "$SYMBOL_FILE" "$START_DATE" "$END_DATE" 2>&1 | tail -1)
    
    if [ ! -z "$BUY_SIGNALS_BY_DATE" ] && [ -f "$BUY_SIGNALS_BY_DATE" ]; then
        echo -e "\n📅 일자별 매수시그널 테이블 파일:"
        echo "   $BUY_SIGNALS_BY_DATE"
    fi
fi 
