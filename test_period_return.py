#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
검색기간 수익률 계산 테스트 스크립트
주가 상승률과 금액을 계산하여 출력합니다.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def test_period_return():
    """검색기간 수익률을 테스트합니다."""
    
    # 테스트할 종목과 기간 설정
    ticker = 'AAPL'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6개월 데이터
    
    print(f"=== {ticker} 검색기간 수익률 테스트 ===")
    print(f"분석 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    
    try:
        # 주식 데이터 가져오기
        print("주식 데이터를 가져오는 중...")
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            print("데이터를 가져올 수 없습니다.")
            return
        
        print(f"데이터 로드 완료: {len(data)}개 캔들")
        
        # 검색기간 수익률 계산
        from src.indicators.hma_mantra.visualization.volume_profile_overlay_chart import calculate_period_return
        
        period_return = calculate_period_return(data, data.index[0], data.index[-1])
        
        if period_return:
            print(f"\n✅ 검색기간 수익률 계산 성공!")
            print(f"\n📊 계산 결과:")
            print(f"  - 검색기간: {period_return['start_date']} → {period_return['end_date']}")
            print(f"  - 시작가: ${period_return['start_price']:.2f}")
            print(f"  - 종료가: ${period_return['end_price']:.2f}")
            print(f"  - 수익률: {period_return['direction']} {period_return['return_percentage']:.2f}%")
            print(f"  - 수익금액: ${period_return['return_amount']:.2f}")
            print(f"  - 색상: {period_return['color']}")
            
            # 표시 형식 예시
            print(f"\n📋 실제 표시 형식:")
            print(f"  [검색기간] {period_return['start_date']} → {period_return['end_date']}")
            print(f"  {period_return['direction']} {period_return['return_percentage']:.2f}% (${period_return['return_amount']:.2f})")
            
            # 수동 계산 검증
            manual_return = ((period_return['end_price'] - period_return['start_price']) / period_return['start_price']) * 100
            manual_amount = period_return['end_price'] - period_return['start_price']
            
            print(f"\n🧮 수동 계산 검증:")
            print(f"  - 수익률: {manual_return:.2f}% (자동: {period_return['return_percentage']:.2f}%)")
            print(f"  - 수익금액: ${manual_amount:.2f} (자동: ${period_return['return_amount']:.2f})")
            print(f"  - 계산 일치: {'✅' if abs(manual_return - period_return['return_percentage']) < 0.01 else '❌'}")
            
        else:
            print("❌ 검색기간 수익률 계산 실패")
            
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def explain_period_return():
    """검색기간 수익률에 대해 설명합니다."""
    print("\n=== 검색기간 수익률이란? ===")
    print("검색기간 수익률은 분석 시작일부터 종료일까지의 주가 변화를 나타내는 지표입니다.")
    
    print("\n📈 계산 공식:")
    print("수익률 = ((종료가 - 시작가) / 시작가) × 100")
    print("수익금액 = 종료가 - 시작가")
    
    print("\n🎯 표시 정보:")
    print("  - 검색기간: 시작일 → 종료일")
    print("  - 시작가: 분석 시작일의 종가")
    print("  - 종료가: 분석 종료일의 종가")
    print("  - 수익률: 퍼센트 단위의 변화율")
    print("  - 수익금액: 달러 단위의 절대 변화량")
    
    print("\n📊 표시 위치:")
    print("  - 차트 내 통합 정보 박스: 실시간가/현재가 아래")
    print("  - 신호 분석 파일: 현재가 다음 줄")
    print("  - 콘솔 출력: 투자심리도 정보 앞")
    
    print("\n⚠️ 주의사항:")
    print("  - 이 지표는 과거 데이터 기반 분석입니다")
    print("  - 미래 수익률을 보장하지 않습니다")
    print("  - 투자 판단의 참고용으로만 사용하세요")

if __name__ == "__main__":
    print("검색기간 수익률 테스트를 시작합니다...")
    
    # 검색기간 수익률 설명
    explain_period_return()
    
    # 사용자 확인
    response = input("\n테스트를 실행하시겠습니까? (y/n): ").lower().strip()
    
    if response in ['y', 'yes', '예']:
        test_period_return()
    else:
        print("테스트를 건너뜁니다.")
        print("\n사용법:")
        print("python test_period_return.py")
        print("\n또는 코드에서 직접 호출:")
        print("from src.indicators.hma_mantra.visualization.volume_profile_overlay_chart import calculate_period_return")
        print("period_return = calculate_period_return(ohlcv_data, start_date, end_date)")
