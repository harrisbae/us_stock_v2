#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BRK.A 종목 검색 문제 진단 스크립트
"""

import yfinance as yf
import pandas as pd
import sys
import os

def test_brk_a_ticker():
    """BRK.A 종목 검색 테스트"""
    print("🔍 BRK.A 종목 검색 문제 진단")
    print("=" * 50)
    
    # 테스트할 티커들
    test_tickers = [
        "BRK.A",      # Berkshire Hathaway A클래스
        "BRK-B",      # Berkshire Hathaway B클래스 (하이픈 포함)
        "BRKB",       # Berkshire Hathaway B클래스 (하이픈 없음)
        "BRK-A",      # 하이픈 포함
        "BRKA",       # 하이픈 없음
        "AAPL",       # 비교용: 정상 작동하는 종목
        "MSFT"        # 비교용: 정상 작동하는 종목
    ]
    
    results = []
    
    for ticker in test_tickers:
        print(f"\n📊 테스트 티커: {ticker}")
        print(f"   {'-' * 30}")
        
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
                        'currentPrice': info.get('currentPrice', 'N/A')
                    }
                    for key, value in key_info.items():
                        if value != 'N/A':
                            print(f"        {key}: {value}")
                else:
                    print(f"      ⚠️ 기본 정보 없음")
            except Exception as e:
                print(f"      ❌ 기본 정보 실패: {e}")
            
            # 3. 히스토리 데이터 가져오기
            print(f"   3️⃣ 히스토리 데이터 가져오기 시도...")
            try:
                hist = yf_ticker.history(period='1d')
                if not hist.empty:
                    print(f"      ✅ 히스토리 데이터 성공: {len(hist)}개 행")
                    latest = hist.iloc[-1]
                    print(f"        최신 가격: ${latest['Close']:.2f}")
                    print(f"        거래량: {latest['Volume']:,}")
                else:
                    print(f"      ⚠️ 히스토리 데이터 없음")
            except Exception as e:
                print(f"      ❌ 히스토리 데이터 실패: {e}")
            
            # 4. 실시간 가격 가져오기
            print(f"   4️⃣ 실시간 가격 가져오기 시도...")
            try:
                current_price = yf_ticker.history(period='1d', interval='1m')
                if not current_price.empty:
                    print(f"      ✅ 실시간 가격 성공: ${current_price['Close'].iloc[-1]:.2f}")
                else:
                    print(f"      ⚠️ 실시간 가격 없음")
            except Exception as e:
                print(f"      ❌ 실시간 가격 실패: {e}")
            
            # 5. 배당 정보 가져오기
            print(f"   5️⃣ 배당 정보 가져오기 시도...")
            try:
                dividends = yf_ticker.dividends
                if not dividends.empty:
                    print(f"      ✅ 배당 정보 성공: {len(dividends)}개 배당")
                else:
                    print(f"      ⚠️ 배당 정보 없음")
            except Exception as e:
                print(f"      ❌ 배당 정보 실패: {e}")
            
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
    print("\n" + "=" * 50)
    print("📊 테스트 결과 요약")
    print("=" * 50)
    
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
    
    # BRK.A 특별 분석
    print("\n" + "=" * 50)
    print("🔍 BRK.A 특별 분석")
    print("=" * 50)
    
    try:
        brk_a = yf.Ticker("BRK-A")
        print("BRK-A 티커로 시도:")
        
        # 다양한 방법으로 데이터 가져오기 시도
        methods = [
            ("기본 정보", lambda: brk_a.info),
            ("1일 히스토리", lambda: brk_a.history(period='1d')),
            ("1주 히스토리", lambda: brk_a.history(period='1wk')),
            ("1개월 히스토리", lambda: brk_a.history(period='1mo')),
            ("분봉 데이터", lambda: brk_a.history(period='1d', interval='1m')),
            ("배당 정보", lambda: brk_a.dividends),
            ("분할 정보", lambda: brk_a.splits)
        ]
        
        for method_name, method_func in methods:
            try:
                result = method_func()
                if hasattr(result, 'empty') and result.empty:
                    print(f"  {method_name}: 데이터 없음")
                elif hasattr(result, '__len__'):
                    print(f"  {method_name}: {len(result)}개 항목")
                else:
                    print(f"  {method_name}: 성공")
            except Exception as e:
                print(f"  {method_name}: 실패 - {e}")
                
    except Exception as e:
        print(f"BRK-A 분석 중 오류: {e}")
    
    print("\n" + "=" * 50)
    print("✅ BRK.A 종목 검색 진단 완료!")

if __name__ == "__main__":
    test_brk_a_ticker()
