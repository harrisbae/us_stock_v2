#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
버핏 지수 계산·검증 스크립트
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.indicators.hma_mantra.buffett_indicator import (
    calculate_buffett_indicator,
    get_buffett_sentiment,
    validate_buffett_result,
    WILSHIRE_CALIB_DATE,
    WILSHIRE_CALIB_INDEX,
    WILSHIRE_CALIB_CAP_TRILLIONS,
)

# buffettindicator.org 교차검증 기준 (주기적 갱신)
EXTERNAL_REFERENCE_RATIO_PCT = 253.4
EXTERNAL_TOLERANCE_PCT = 15.0


def run_test(*, external_ratio: float | None = EXTERNAL_REFERENCE_RATIO_PCT) -> int:
    print("=== 버핏 지수 계산·검증 ===\n")

    buffett_data = calculate_buffett_indicator()
    if not buffett_data:
        print("❌ 버핏 지수 계산 실패")
        return 1

    print(f"\n📊 결과")
    print(f"  - 버핏 지수: {buffett_data['value']}%")
    print(f"  - Wilshire 시총: {buffett_data['wilshire_market_cap']}조 ({buffett_data['wilshire_date']})")
    if buffett_data.get("wilshire_index"):
        print(f"  - Wilshire 지수: {buffett_data['wilshire_index']}")
    print(f"  - US GDP: {buffett_data['us_gdp']}조 ({buffett_data['gdp_date']})")
    print(f"  - 소스: {buffett_data.get('data_sources', 'N/A')}")

    sentiment, color, guide = get_buffett_sentiment(buffett_data["value"])
    print(f"\n🎯 심리도: {sentiment} ({color})")
    print(f"   가이드: {guide}")

    recalc = (buffett_data["wilshire_market_cap"] / buffett_data["us_gdp"]) * 100
    print(f"\n🧮 공식 검증: ({buffett_data['wilshire_market_cap']} / {buffett_data['us_gdp']}) × 100 = {recalc:.1f}%")

    checks = validate_buffett_result(
        buffett_data,
        external_ratio_pct=external_ratio,
        external_tolerance_pct=EXTERNAL_TOLERANCE_PCT,
    )
    print("\n✅ 검증 체크리스트:")
    labels = {
        "internal_formula": "내부 공식 일치",
        "plausible_range": "합리적 범위 (40~300%)",
        "data_fresh": "실시간 데이터 (fallback 없음·10일 이내)",
        "external_reference": f"외부 참조 ±{EXTERNAL_TOLERANCE_PCT}%p"
        + (f" (기준 {external_ratio}%)" if external_ratio else " (스킵)"),
    }
    failed = 0
    for key, ok in checks.items():
        mark = "✅" if ok else "❌"
        print(f"  {mark} {labels[key]}")
        if not ok:
            failed += 1

    print(f"\n📌 캘리브레이션: {WILSHIRE_CALIB_DATE} 지수 {WILSHIRE_CALIB_INDEX:.0f} → {WILSHIRE_CALIB_CAP_TRILLIONS}조")
    if failed:
        print(f"\n⚠️ {failed}개 검증 실패 (네트워크·캘리브 기준일·외부 사이트 차이 확인)")
        return 2
    print("\n✅ 모든 검증 통과")
    return 0


def main():
    parser = argparse.ArgumentParser(description="버핏 지수 테스트")
    parser.add_argument("--no-external", action="store_true", help="외부 참조 비교 생략")
    args = parser.parse_args()
    ext = None if args.no_external else EXTERNAL_REFERENCE_RATIO_PCT
    raise SystemExit(run_test(external_ratio=ext))


if __name__ == "__main__":
    main()
