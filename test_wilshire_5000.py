#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wilshire 5000 지수 데이터 접근 문제 진단
"""

import yfinance as yf
import pandas as pd
import sys
import os

def test_wilshire_5000():
    """Wilshire 5000 지수 데이터 접근 테스트"""
    print("🔍 Wilshire 5000 지수 데이터 접근 진단")
    print("=" * 60)
    
    # 테스트할 티커들
    test_tickers = [
        "^W5000",       # Wilshire 5000 (권장)
        "W5000",        # ^ 제거
        "^W5000FLT",    # 이전 티커 (비교용)
        "W5000FLT",     # 이전 티커 (비교용)
        "^RUA",         # Russell 3000 (대안)
        "^GSPC",        # S&P 500 (대안)
        "^DJI",         # Dow Jones (대안)
        "^VIX",         # VIX (비교용)
        "AAPL"          # 개별주 (비교용)
    ]
    
    results = []
    
    for ticker in test_tickers:
        print(f"\n📊 테스트 티커: {ticker}")
        print(f"   {'-' * 40}")
        
        try:
            # 1. 기본 Ticker 객체 생성
            print(f"   1️⃣ Ticker 객체 생성 시도...")
            yf_ticker = yf.Ticker(ticker)
            print(f"      ✅ Ticker 객체 생성 성공")
            
            # 2. 기본 정보 가져오기
            print(f"   2️⃣ 기본 정보 가져오기 시도...")
            try:
                info = yf_ticker.info
                if info:
                    print(f"      ✅ 기본 정보 성공: {len(info)}개 항목")
                    # 주요 정보 출력
                    key_info = {
                        'longName': info.get('longName', 'N/A'),
                        'shortName': info.get('shortName', 'N/A'),
                        'sector': info.get('sector', 'N/A'),
                        'industry': info.get('industry', 'N/A'),
                        'marketCap': info.get('marketCap', 'N/A'),
                        'currentPrice': info.get('currentPrice', 'N/A'),
                        'regularMarketPrice': info.get('regularMarketPrice', 'N/A')
                    }
                    for key, value in key_info.items():
                        if value != 'N/A':
                            print(f"        {key}: {value}")
                else:
                    print(f"      ⚠️ 기본 정보 없음")
            except Exception as e:
                print(f"      ❌ 기본 정보 실패: {e}")
            
            # 3. 다양한 기간으로 히스토리 데이터 테스트
            print(f"   3️⃣ 히스토리 데이터 테스트...")
            periods = ["1d", "1wk", "1mo", "3mo", "6mo", "1y"]
            
            for period in periods:
                try:
                    hist = yf_ticker.history(period=period)
                    if not hist.empty:
                        print(f"      ✅ {period}: {len(hist)}개 행, 최신: ${hist['Close'].iloc[-1]:.2f}")
                        break
                    else:
                        print(f"      ⚠️ {period}: 데이터 없음")
                except Exception as e:
                    print(f"      ❌ {period}: {e}")
            
            # 4. 특정 날짜 범위로 테스트
            print(f"   4️⃣ 특정 날짜 범위 테스트...")
            try:
                # 최근 1주일 데이터
                hist_range = yf_ticker.history(start="2025-08-18", end="2025-08-25")
                if not hist_range.empty:
                    print(f"      ✅ 날짜 범위 성공: {len(hist_range)}개 행")
                    print(f"        시작: ${hist_range['Close'].iloc[0]:.2f}")
                    print(f"        종료: ${hist_range['Close'].iloc[-1]:.2f}")
                else:
                    print(f"      ⚠️ 날짜 범위: 데이터 없음")
            except Exception as e:
                print(f"      ❌ 날짜 범위: {e}")
            
            # 5. 실시간 가격 테스트
            print(f"   5️⃣ 실시간 가격 테스트...")
            try:
                current_price = yf_ticker.history(period='1d', interval='1m')
                if not current_price.empty:
                    print(f"      ✅ 실시간 가격: ${current_price['Close'].iloc[-1]:.2f}")
                else:
                    print(f"      ⚠️ 실시간 가격: 데이터 없음")
            except Exception as e:
                print(f"      ❌ 실시간 가격: {e}")
            
            # 결과 저장
            results.append({
                'ticker': ticker,
                'success': True,
                'info_count': len(info) if 'info' in locals() and info else 0,
                'hist_count': len(hist) if 'hist' in locals() and not hist.empty else 0,
                'current_price': current_price['Close'].iloc[-1] if 'current_price' in locals() and not current_price.empty else None
            })
            
        except Exception as e:
            print(f"      ❌ 전체 실패: {e}")
            results.append({
                'ticker': ticker,
                'success': False,
                'error': str(e)
            })
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"총 테스트: {total}")
    print(f"성공: {successful}")
    print(f"실패: {total - successful}")
    print(f"성공률: {successful/total*100:.1f}%")
    
    print("\n상세 결과:")
    for result in results:
        status_icon = "✅" if result['success'] else "❌"
        if result['success']:
            info_str = f"정보: {result['info_count']}, 히스토리: {result['hist_count']}"
            if result['current_price']:
                info_str += f", 가격: ${result['current_price']:.2f}"
            print(f"{status_icon} {result['ticker']}: {info_str}")
        else:
            print(f"{status_icon} {result['ticker']}: {result['error']}")
    
    # Wilshire 5000 특별 분석
    print("\n" + "=" * 60)
    print("🔍 Wilshire 5000 특별 분석")
    print("=" * 60)
    
    print("Wilshire 5000 지수는 다음 이유로 접근이 어려울 수 있습니다:")
    print("1. 실시간 데이터 접근 제한")
    print("2. 티커 표기법 차이")
    print("3. 데이터 제공업체 정책")
    print("4. 시장 시간 외 접근 제한")
    
    print("\n대안 지수들:")
    print("- ^GSPC: S&P 500 (가장 널리 사용)")
    print("- ^RUA: Russell 3000 (대형주 포함)")
    print("- ^DJI: Dow Jones Industrial Average")
    print("- ^IXIC: NASDAQ Composite")
    
    print("\n" + "=" * 60)
    print("✅ Wilshire 5000 진단 완료!")

if __name__ == "__main__":
    test_wilshire_5000()
