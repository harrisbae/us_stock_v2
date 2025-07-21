#!/bin/bash

# 가상환경 활성화
source .venv/bin/activate

# 인자 파싱
SYMBOL=$1
PERIOD=${2:-24mo}
CAPITAL=${3:-1000}

if [ -z "$SYMBOL" ]; then
  echo "Usage: $0 <SYMBOL> [PERIOD] [CAPITAL]"
  echo "  SYMBOL: 종목코드 (필수)"
  echo "  PERIOD: 기간 (기본값: 24mo)"
  echo "  CAPITAL: 매수 금액 (기본값: 1000)"
  exit 1
fi

python test/hma_backtest_program.py "$SYMBOL" "$PERIOD" "$CAPITAL" 