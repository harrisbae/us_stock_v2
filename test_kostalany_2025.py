#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2025년도 상반기 GDP 1.2% 반영 및 상세 인플레이션 지표 테스트
앙드레 코스탈라니 달걀모형 경제 사이클 분석
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_kostalany_2025():
    """2025년도 상반기 데이터로 코스탈라니 달걀모형 테스트"""
    
    print("=== 2025년도 상반기 코스탈라니 달걀모형 테스트 ===")
    print("GDP: 1.2% (연준 발표 자료 기준)")
    print("=" * 50)
    
    try:
        # 코스탈라니 통합 분석 실행
        from src.kostalany_integrated import main
        
        print("코스탈라니 달걀모형 분석을 실행합니다...")
        print("상세 인플레이션 지표(PCE, PPI, CPI)가 포함됩니다.")
        print()
        
        # 명령행 인자 시뮬레이션
        sys.argv = [
            'kostalany_integrated.py',
            '--country', 'us',
            '--default_gdp', '1.2',
            '--default_inflation', '2.8',
            '--default_interest', '4.5',
            '--default_unemployment', '4.2',
            '--default_vix', '18.48',
            '--default_dxy', '98.88',
            '--show_inflation_details'
        ]
        
        main()
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n대안: 직접 실행")
        print("python src/kostalany_integrated.py --country us --default_gdp 1.2 --show_inflation_details")

def explain_2025_data():
    """2025년도 상반기 데이터 설명"""
    print("\n=== 2025년도 상반기 미국 경제 지표 설명 ===")
    
    print("\n📊 GDP 성장률: 1.2%")
    print("  - 출처: 연준 발표 자료")
    print("  - 의미: 2025년 상반기 미국 경제 성장률")
    print("  - 해석: 경기 회복 단계 (1~3% 구간)")
    
    print("\n💰 미국 상세 인플레이션 지표 (전년동기/전월/전분기 대비):")
    print("  • PCE (전체): 2.6% (전년동기: 3.2%, 전월: 2.8%, 전분기: 2.9%) - 소비자 지출 기준 물가")
    print("  • PCE (근원): 2.4% (전년동기: 3.0%, 전월: 2.6%, 전분기: 2.7%) - 연준이 선호하는 지표")
    print("  • PPI: 1.8% (전년동기: 2.5%, 전월: 2.0%, 전분기: 2.1%) - 생산자 단계 물가")
    print("  • CPI: 3.1% (전년동기: 3.7%, 전월: 3.3%, 전분기: 3.4%) - 소비자 단계 물가")
    
    print("\n📈 전년동기/전월/전분기 대비 추세:")
    print("  - 전년동기 대비: 모든 지표 하락 (PCE -18.8%, PPI -28.0%, CPI -16.2%)")
    print("  - 전월 대비: 모든 지표 하락 (PCE -7.1%, PPI -10.0%, CPI -6.1%)")
    print("  - 전분기 대비: 모든 지표 하락 (PCE -10.3%, PPI -14.3%, CPI -8.8%)")
    print("  - 연준의 통화정책 효과로 물가 안정화")
    print("  - PCE 지표가 연준 목표치 2% 근처로 안정화")
    
    print("\n📊 주요 거시경제 지표 전년동기 대비 증감율:")
    print("  • GDP: 전년동기 대비 42.9% 하락 (경기 둔화)")
    print("  • 인플레이션: 전년동기 대비 24.3% 하락 (물가 안정화)")
    print("  • 금리: 전년동기 대비 50.0% 상승 (통화정책 긴축)")
    print("  • 실업률: 전년동기 대비 10.5% 상승 (고용 시장 조정)")
    print("  • VIX: 전년동기 대비 23.1% 하락 (시장 안정화)")
    print("  • 달러: 전년동기 대비 5.6% 하락 (달러 약세)")
    
    print("\n🎯 코스탈라니 달걀모형 해석:")
    print("  - GDP 1.2%: E 단계 (경기 확장과 소비 증가)")
    print("  - 인플레이션: 2~3% 구간으로 안정화")
    print("  - 투자 전략: 경기 민감주와 성장주 투자 유리")
    
    print("\n📈 투자 전략 제안:")
    print("  - 경기 민감주 투자 (기술주, 금융주, 산업재)")
    print("  - 성장주 투자 (AI, 반도체, 바이오테크)")
    print("  - 소비자 경기 민감 섹터 (소비재, 여행, 엔터테인먼트)")
    print("  - 분산 투자 및 리스크 관리")

if __name__ == "__main__":
    print("2025년도 상반기 코스탈라니 달걀모형 테스트를 시작합니다...")
    
    # 데이터 설명
    explain_2025_data()
    
    # 사용자 확인
    response = input("\n테스트를 실행하시겠습니까? (y/n): ").lower().strip()
    
    if response in ['y', 'yes', '예']:
        test_kostalany_2025()
    else:
        print("테스트를 건너뜁니다.")
        print("\n수동 실행 방법:")
        print("python src/kostalany_integrated.py --country us --default_gdp 1.2 --show_inflation_details")
        print("\n또는 shell script 사용:")
        print("./kostalany.sh us 1.2 2.8 4.5 4.2 18.48 98.88")
