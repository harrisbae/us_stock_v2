#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
버핏 지수 계산 테스트 스크립트
GDP와 시총 기준일과 값을 확인할 수 있습니다.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.indicators.hma_mantra.visualization.volume_profile_overlay_chart import calculate_buffett_indicator, get_buffett_sentiment

def test_buffett_indicator():
    """버핏 지수를 테스트합니다."""
    
    print("=== 버핏 지수 계산 테스트 (2025년 8월 최신) ===")
    
    try:
        # 버핏 지수 계산
        print("버핏 지수를 계산하는 중... (2025년 8월 최신 데이터)")
        buffett_data = calculate_buffett_indicator()
        
        if buffett_data:
            print(f"\n✅ 버핏 지수 계산 성공!")
            print(f"\n📊 계산 결과:")
            print(f"  - 버핏 지수: {buffett_data['value']}%")
            print(f"  - Wilshire 5000 시가총액: {buffett_data['wilshire_market_cap']}조 달러")
            print(f"  - Wilshire 5000 기준일: {buffett_data['wilshire_date']}")
            print(f"  - US GDP: {buffett_data['us_gdp']}조 달러")
            print(f"  - GDP 기준일: {buffett_data['gdp_date']}")
            
            # 투자 심리도 분석
            sentiment, color, guide = get_buffett_sentiment(buffett_data['value'])
            print(f"\n🎯 투자 심리도 분석:")
            print(f"  - 심리도: {sentiment}")
            print(f"  - 색상: {color}")
            print(f"  - 투자 가이드: {guide}")
            
            # 계산 공식 확인
            calculated_value = (buffett_data['wilshire_market_cap'] / buffett_data['us_gdp']) * 100
            print(f"\n🧮 계산 공식 확인:")
            print(f"  - 공식: (시가총액 / GDP) × 100")
            print(f"  - 계산: ({buffett_data['wilshire_market_cap']} / {buffett_data['us_gdp']}) × 100 = {calculated_value:.1f}%")
            print(f"  - 결과 일치: {'✅' if abs(calculated_value - buffett_data['value']) < 0.1 else '❌'}")
            
        else:
            print("❌ 버핏 지수 계산 실패")
            
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def explain_buffett_indicator():
    """버핏 지수에 대해 설명합니다."""
    print("\n=== 버핏 지수란? ===")
    print("버핏 지수는 미국 주식시장의 전체 시가총액을 GDP로 나눈 값입니다.")
    print("워렌 버핏이 '주식시장의 가치를 측정하는 가장 좋은 단일 지표'라고 평가한 지표입니다.")
    
    print("\n📈 계산 공식:")
    print("버핏 지수 = (Wilshire 5000 시가총액 / US GDP) × 100")
    
    print("\n🎯 해석 기준:")
    print("  - 50% 이하: 극도 과매도 (강력한 매수 기회)")
    print("  - 50-75%: 과매도 (매수 기회)")
    print("  - 75-90%: 극형 (균형적 배분)")
    print("  - 90-115%: 과열 (주의 필요)")
    print("  - 115% 이상: 극도 과열 (강력한 매도 신호)")
    
    print("\n📊 데이터 소스:")
    print("  - Wilshire 5000: 미국 전체 주식시장 시가총액 지수")
    print("  - US GDP: 미국 국내총생산 (분기별 발표)")
    
    print("\n⚠️ 주의사항:")
    print("  - 이 지표는 투자 판단의 참고용으로만 사용하세요")
    print("  - 단일 지표에 의존하지 말고 다양한 분석을 병행하세요")
    print("  - 과거 데이터 기반이므로 미래를 정확히 예측하지 못할 수 있습니다")

if __name__ == "__main__":
    print("버핏 지수 테스트를 시작합니다...")
    
    # 버핏 지수 설명
    explain_buffett_indicator()
    
    # 사용자 확인
    response = input("\n테스트를 실행하시겠습니까? (y/n): ").lower().strip()
    
    if response in ['y', 'yes', '예']:
        test_buffett_indicator()
    else:
        print("테스트를 건너뜁니다.")
        print("\n사용법:")
        print("python test_buffett_indicator.py")
        print("\n또는 코드에서 직접 호출:")
        print("from src.indicators.hma_mantra.visualization.volume_profile_overlay_chart import calculate_buffett_indicator")
        print("buffett_data = calculate_buffett_indicator()")
