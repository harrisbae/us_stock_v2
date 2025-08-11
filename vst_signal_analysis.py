#!/usr/bin/env python3
"""
VST HMA Mantra 매수신호 분석 - 투자심리도 및 RSI 계산
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

def calculate_investor_sentiment(data, date):
    """특정 날짜 기준 10일간 주가상승 비율 계산 (투자심리도)"""
    try:
        # 해당 날짜 또는 가장 가까운 이전 날짜 찾기
        if date in data.index:
            current_date = date
        else:
            prev_dates = data.index[data.index <= date]
            if len(prev_dates) > 0:
                current_date = prev_dates[-1]
            else:
                return 50  # 기본값
        
        # 10일 전 날짜 찾기
        start_date = current_date - timedelta(days=10)
        if start_date in data.index:
            start_price = data.loc[start_date, 'Close']
            current_price = data.loc[current_date, 'Close']
            
            # 10일간 수익률 계산
            return_10d = ((current_price - start_price) / start_price) * 100
            
            # 투자심리도: -100% ~ +100% 범위를 0~100으로 정규화
            sentiment = max(0, min(100, 50 + return_10d * 2))
            return round(sentiment, 2)
        else:
            return 50  # 기본값
    except:
        return 50  # 기본값

def analyze_vst_signals():
    """VST HMA Mantra 매수신호 분석"""
    
    # VST 데이터 다운로드 (2024-01-01부터)
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
            
            # 해당 날짜의 데이터 확인
            if buy_date in vst.index:
                # 가격 정보
                close_price = float(vst.loc[buy_date, 'Close'])
                rsi_value = vst.loc[buy_date, 'RSI']
                
                # RSI 값이 NaN인지 확인
                if pd.isna(rsi_value):
                    rsi_value = None
                    rsi_status = "계산불가"
                else:
                    rsi_value = float(rsi_value)
                    if rsi_value < 30:
                        rsi_status = "과매도"
                    elif rsi_value > 70:
                        rsi_status = "과매수"
                    else:
                        rsi_status = "중립"
                
                # 투자심리도 계산
                sentiment = calculate_investor_sentiment(vst, buy_date)
                
                # 10일 전 가격
                start_date = buy_date - timedelta(days=10)
                if start_date in vst.index:
                    start_price = float(vst.loc[start_date, 'Close'])
                    price_change_10d = ((close_price - start_price) / start_price) * 100
                else:
                    price_change_10d = None
                
                # 20일 이동평균
                data_before_date = vst.loc[:buy_date]
                if len(data_before_date) >= 20:
                    ma20 = float(data_before_date['Close'].tail(20).mean())
                else:
                    ma20 = None
                
                # 50일 이동평균
                if len(data_before_date) >= 50:
                    ma50 = float(data_before_date['Close'].tail(50).mean())
                else:
                    ma50 = None
                
                # 현재가 대비 이동평균 위치
                if ma20 is not None:
                    ma20_position = ((close_price - ma20) / ma20) * 100
                else:
                    ma20_position = None
                    
                if ma50 is not None:
                    ma50_position = ((close_price - ma50) / ma50) * 100
                else:
                    ma50_position = None
                
                # 투자심리도 상태 분석
                if sentiment < 30:
                    sentiment_status = "매우 부정적"
                elif sentiment < 50:
                    sentiment_status = "부정적"
                elif sentiment < 70:
                    sentiment_status = "중립"
                else:
                    sentiment_status = "긍정적"
                
                result = {
                    '매수신호일': signal_date,
                    '종가': f"${close_price:.2f}",
                    'RSI': f"{rsi_value:.1f}" if rsi_value is not None else 'N/A',
                    'RSI_상태': rsi_status,
                    '투자심리도': f"{sentiment:.1f}",
                    '심리도_상태': sentiment_status,
                    '10일간_가격변화': f"{price_change_10d:.2f}%" if price_change_10d is not None else 'N/A',
                    'MA20': f"${ma20:.2f}" if ma20 is not None else 'N/A',
                    'MA50': f"${ma50:.2f}" if ma50 is not None else 'N/A',
                    'MA20_대비': f"{ma20_position:.2f}%" if ma20_position is not None else 'N/A',
                    'MA50_대비': f"{ma50_position:.2f}%" if ma50_position is not None else 'N/A'
                }
                
                results.append(result)
                
                print(f"📅 {signal_date}: 종가 ${close_price:.2f} | RSI: {rsi_value:.1f if rsi_value is not None else 'N/A'} ({rsi_status}) | 투자심리도: {sentiment:.1f} ({sentiment_status})")
                
            else:
                print(f"⚠️  {signal_date}: 데이터 없음")
                
        except Exception as e:
            print(f"❌ {signal_date}: 오류 발생 - {str(e)}")
    
    # 결과를 DataFrame으로 변환하여 CSV로 저장
    if results:
        df_results = pd.DataFrame(results)
        
        # CSV 저장
        output_file = 'vst_signal_analysis_results.csv'
        df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n📊 분석 결과가 '{output_file}'에 저장되었습니다.")
        
        # 요약 통계
        print("\n" + "=" * 80)
        print("📈 요약 통계")
        print("=" * 80)
        
        # RSI 통계
        rsi_values = []
        for r in results:
            if r['RSI'] != 'N/A':
                try:
                    rsi_values.append(float(r['RSI']))
                except:
                    pass
        
        if rsi_values:
            print(f"RSI 평균: {np.mean(rsi_values):.1f}")
            print(f"RSI 최소: {np.min(rsi_values):.1f}")
            print(f"RSI 최대: {np.max(rsi_values):.1f}")
            print(f"과매도 신호 (RSI < 30): {len([r for r in results if r['RSI_상태'] == '과매도'])}회")
            print(f"과매수 신호 (RSI > 70): {len([r for r in results if r['RSI_상태'] == '과매수'])}회")
            print(f"중립 신호 (30 ≤ RSI ≤ 70): {len([r for r in results if r['RSI_상태'] == '중립'])}회")
        
        # 투자심리도 통계
        sentiment_values = []
        for r in results:
            if r['투자심리도'] != 'N/A':
                try:
                    sentiment_values.append(float(r['투자심리도']))
                except:
                    pass
        
        if sentiment_values:
            print(f"\n투자심리도 평균: {np.mean(sentiment_values):.1f}")
            print(f"투자심리도 최소: {np.min(sentiment_values):.1f}")
            print(f"투자심리도 최대: {np.max(sentiment_values):.1f}")
            print(f"매우 부정적 (심리도 < 30): {len([r for r in results if r['심리도_상태'] == '매우 부정적'])}회")
            print(f"부정적 (심리도 < 50): {len([r for r in results if r['심리도_상태'] == '부정적'])}회")
            print(f"중립 (50 ≤ 심리도 ≤ 70): {len([r for r in results if r['심리도_상태'] == '중립'])}회")
            print(f"긍정적 (심리도 > 70): {len([r for r in results if r['심리도_상태'] == '긍정적'])}회")
        
        # 이동평균 대비 위치 통계
        ma20_positions = []
        for r in results:
            if r['MA20_대비'] != 'N/A':
                try:
                    ma20_positions.append(float(r['MA20_대비'].replace('%', '')))
                except:
                    pass
        
        if ma20_positions:
            print(f"\nMA20 대비 평균: {np.mean(ma20_positions):.2f}%")
            print(f"MA20 위 거래: {len([p for p in ma20_positions if p > 0])}회")
            print(f"MA20 아래 거래: {len([p for p in ma20_positions if p < 0])}회")
        
        ma50_positions = []
        for r in results:
            if r['MA50_대비'] != 'N/A':
                try:
                    ma50_positions.append(float(r['MA50_대비'].replace('%', '')))
                except:
                    pass
        
        if ma50_positions:
            print(f"MA50 대비 평균: {np.mean(ma50_positions):.2f}%")
            print(f"MA50 위 거래: {len([p for p in ma50_positions if p > 0])}회")
            print(f"MA50 아래 거래: {len([p for p in ma50_positions if p < 0])}회")
    
    else:
        print("분석할 결과가 없습니다.")

if __name__ == "__main__":
    print("VST HMA Mantra 매수신호 분석 - 투자심리도 및 RSI 계산")
    print("=" * 80)
    analyze_vst_signals()
