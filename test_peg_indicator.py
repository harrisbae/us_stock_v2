#!/usr/bin/env python3
"""PEG 1년·3년 테스트"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.indicators.hma_mantra.peg_indicator import calculate_peg_indicator


def main():
    for sym in sys.argv[1:] or ["MSFT", "006400.KS", "NVDA"]:
        print(f"\n{'='*40}\n{sym}")
        d = calculate_peg_indicator(sym)
        if not d:
            print("  실패")
            continue
        print(
            f"  1년 PEG={d.get('peg_1y')} YoY={d.get('growth_1y_pct')}% | "
            f"3년 PEG={d.get('peg_3y')} CAGR={d.get('growth_3y_cagr_pct')}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
