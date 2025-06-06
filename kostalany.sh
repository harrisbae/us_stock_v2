#!/bin/bash

# 코스탈라니 달걀모형 통합 분석 실행 스크립트

# 디폴트값 설정
DEFAULT_COUNTRY="us"
DEFAULT_GDP="1.6"
DEFAULT_INFLATION="2.5"
DEFAULT_INTEREST="3.0"
DEFAULT_UNEMPLOYMENT="4.0"
DEFAULT_VIX="20.0"
DEFAULT_DXY="105.0"

# 명령줄 인자 파싱
COUNTRY=${1:-$DEFAULT_COUNTRY}
GDP=${2:-$DEFAULT_GDP}
INFLATION=${3:-$DEFAULT_INFLATION}
INTEREST=${4:-$DEFAULT_INTEREST}
UNEMPLOYMENT=${5:-$DEFAULT_UNEMPLOYMENT}
VIX=${6:-$DEFAULT_VIX}
DXY=${7:-$DEFAULT_DXY}

# 실행
python src/kostalany_integrated.py \
    --country=$COUNTRY \
    --default_gdp=$GDP \
    --default_inflation=$INFLATION \
    --default_interest=$INTEREST \
    --default_unemployment=$UNEMPLOYMENT \
    --default_vix=$VIX \
    --default_dxy=$DXY

# 사용법 출력
echo "사용법: ./kostalany.sh [country] [gdp] [inflation] [interest] [unemployment] [vix] [dxy]"
echo "예시: ./kostalany.sh us $GDP 2.5 3.0 4.0 20.0 105.0"