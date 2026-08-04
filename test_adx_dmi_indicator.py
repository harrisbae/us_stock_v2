#!/usr/bin/env python3
"""ADX/DMI 지표 테스트"""

import os
import sys

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(__file__))

from src.indicators.hma_mantra.adx_dmi_indicator import analyze_adx_dmi, detect_adx_buy_signals


def main():
    sym = sys.argv[1] if len(sys.argv) > 1 else "SATS"
    data = yf.download(sym, period="6mo", progress=False)
    if data.empty:
        print("데이터 없음")
        return 1
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    r = analyze_adx_dmi(data)
    print(f"{sym}: {r.get('summary')} ADX={r.get('adx')} +DI={r.get('plus_di')} -DI={r.get('minus_di')}")
    buys = r.get("buy_signals") or []
    print(f"ADX buy 신호: {len(buys)}건")
    for b in buys[-3:]:
        print(f"  {b['date'].strftime('%Y-%m-%d')} ADX={b['adx']} +DI={b['plus_di']} -DI={b['minus_di']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
