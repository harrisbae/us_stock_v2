#!/usr/bin/env python3
"""
VST HMA Mantra 매수신호 분석 - 디버깅 버전
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def calculate_rsi(data, window=14):
    """RSI 계산"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def analyze_vst_signals():
    """VST HMA Mantra 매수신호 분석"""
    
    print("VST 데이터 다운로드 중...")
    vst = yf.download('VST', start='2024-01-01', end=None)
    
    if vst.empty:
        print("데이터 다운로드 실패")
        return
    
    print(f"데이터 다운로드 완료: {len(vst)} 개 데이터")
    print(f"데이터 기간: {vst.index[0].strftime('%Y-%m-%d')} ~ {vst.index[-1].strftime('%Y-%m-%d')}")
    
    # RSI 계산
    vst['RSI'] = calculate_rsi(vst)
    
    # HMA Mantra 매수신호 발생일들 (백테스트 결과 기반)
    buy_signals = [
        '2024-01-08', '2024-01-19', '2024-02-13', '2024-03-12', '2024-03-19',
        '2024-04-01', '2024-04-05', '2024-04-24', '2024-05-24', '2024-06-24',
        '2024-07-31', '2024-08-29', '2024-09-11', '2024-10-16', '2024-11-05',
        '2024-11-26', '2024-12-24', '2025-01-30', '2025-03-12', '2025-04-01',
        '2025-04-09', '2025-04-22', '2025-05-12', '2025-05-23', '2025-06-02',
        '2025-06-16', '2025-07-07', '2025-07-18'
    ]
    
    print(f"\n총 {len(buy_signals)}개의 매수신호 분석:")
    print("=" * 80)
    
    results = []
    
    for signal_date in buy_signals:
        try:
            # 날짜 파싱
            buy_date = pd.to_datetime(signal_date)
            
            print(f"처리 중: {signal_date} -> {buy_date}")
            
            # 해당 날짜의 데이터 확인
            if buy_date in vst.index:
                print(f"  ✓ 데이터 존재")
                
                # 가격 정보
                close_price = vst.loc[buy_date, 'Close']
                print(f"  ✓ 종가: {close_price}")
                
                # RSI 값
                rsi_value = vst.loc[buy_date, 'RSI']
                print(f"  ✓ RSI: {rsi_value}")
                
                # 10일 전 가격
                start_date = buy_date - timedelta(days=10)
                if start_date in vst.index:
                    start_price = vst.loc[start_date, 'Close']
                    price_change_10d = ((close_price - start_price) / start_price) * 100
                    print(f"  ✓ 10일간 변화: {price_change_10d:.2f}%")
                else:
                    price_change_10d = None
                    print(f"  ⚠ 10일 전 데이터 없음")
                
                # 투자심리도 계산 (10일간 변화율 기반)
                if price_change_10d is not None:
                    sentiment = max(0, min(100, 50 + price_change_10d * 2))
                    print(f"  ✓ 투자심리도: {sentiment:.1f}")
                else:
                    sentiment = 50
                    print(f"  ⚠ 투자심리도: 기본값 50")
                
                # RSI 상태 분석
                if pd.isna(rsi_value):
                    rsi_status = "계산불가"
                elif rsi_value < 30:
                    rsi_status = "과매도"
                elif rsi_value > 70:
                    rsi_status = "과매수"
                else:
                    rsi_status = "중립"
                
                print(f"  ✓ RSI 상태: {rsi_status}")
                
                # 투자심리도 상태 분석
                if sentiment < 30:
                    sentiment_status = "매우 부정적"
                elif sentiment < 50:
                    sentiment_status = "부정적"
                elif sentiment < 70:
                    sentiment_status = "중립"
                else:
                    sentiment_status = "긍정적"
                
                print(f"  ✓ 심리도 상태: {sentiment_status}")
                
                result = {
                    '매수신호일': signal_date,
                    '종가': f"${close_price:.2f}",
                    'RSI': f"{rsi_value:.1f}" if not pd.isna(rsi_value) else 'N/A',
                    'RSI_상태': rsi_status,
                    '투자심리도': f"{sentiment:.1f}",
                    '심리도_상태': sentiment_status,
                    '10일간_가격변화': f"{price_change_10d:.2f}%" if price_change_10d is not None else 'N/A'
                }
                
                results.append(result)
                print(f"  ✓ 결과 저장 완료")
                
            else:
                print(f"  ❌ 데이터 없음")
                
        except Exception as e:
            print(f"  ❌ 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 결과 요약
    if results:
        print(f"\n📊 분석 완료: {len(results)}개 신호")
        
        # CSV 저장
        output_file = 'vst_signal_analysis_results.csv'
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"결과 저장: {output_file}")
        
        # 간단한 통계
        rsi_values = []
        sentiment_values = []
        
        for r in results:
            if r['RSI'] != 'N/A':
                try:
                    rsi_values.append(float(r['RSI']))
                except:
                    pass
            
            if r['투자심리도'] != 'N/A':
                try:
                    sentiment_values.append(float(r['투자심리도']))
                except:
                    pass
        
        if rsi_values:
            print(f"\nRSI 통계:")
            print(f"  평균: {np.mean(rsi_values):.1f}")
            print(f"  최소: {np.min(rsi_values):.1f}")
            print(f"  최대: {np.max(rsi_values):.1f}")
        
        if sentiment_values:
            print(f"\n투자심리도 통계:")
            print(f"  평균: {np.mean(sentiment_values):.1f}")
            print(f"  최소: {np.min(sentiment_values):.1f}")
            print(f"  최대: {np.max(sentiment_values):.1f}")
    
    else:
        print("분석할 결과가 없습니다.")

if __name__ == "__main__":
    print("VST HMA Mantra 매수신호 분석 - 디버깅 버전")
    print("=" * 80)
    analyze_vst_signals()
