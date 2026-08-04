#!/usr/bin/env python3
"""CNN 공포·탐욕 지수 테스트"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.indicators.hma_mantra.fear_greed_indicator import calculate_fear_greed_indicator


def main():
    data = calculate_fear_greed_indicator()
    if not data:
        print("❌ 조회 실패")
        return 1
    print(f"\n✅ score={data['score']} ({data['rating_ko']})")
    print(f"   guide: {data.get('guide')}")
    print(f"   source: {data.get('source')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
