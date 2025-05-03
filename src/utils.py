import os
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

def setup_argparser():
    """명령행 인수 설정"""
    parser = argparse.ArgumentParser(description='주식 기술적 분석 도구')
    
    parser.add_argument('--ticker', type=str, required=True,
                        help='종목 코드 (예: AAPL, 005930.KS)')
    
    parser.add_argument('--period', type=str, default='1y',
                        choices=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max'],
                        help='조회 기간 (기본값: 1y)')
    
    parser.add_argument('--interval', type=str, default='1d',
                        choices=['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'],
                        help='시간 간격 (기본값: 1d)')
    
    parser.add_argument('--visualize', action='store_true',
                        help='차트 시각화 여부')
    
    parser.add_argument('--output', type=str, default=None,
                        help='출력 디렉토리 (기본값: output/YYYYMMDD/TICKER)')
    
    parser.add_argument('--start-date', type=str, default=None,
                        help='데이터 시작 날짜 (YYYY-MM-DD 형식)')
    
    parser.add_argument('--end-date', type=str, default=None,
                        help='데이터 종료 날짜 (YYYY-MM-DD 형식, 기본값: 오늘)')
    
    return parser

def create_output_dir(ticker, base_dir='output'):
    """출력 디렉토리 생성"""
    # 현재 날짜 가져오기
    today = datetime.now().strftime("%Y%m%d")
    
    # 출력 디렉토리 경로
    output_dir = os.path.join(base_dir, today, ticker)
    
    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def print_analysis_summary(data, ticker):
    """분석 결과 요약 출력"""
    if data is None or data.empty:
        print("데이터가 없습니다.")
        return
        
    latest = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else None
    
    print("\n" + "="*50)
    print(f"종목: {ticker} 기술적 분석 요약")
    print("="*50)
    
    # 기본 가격 정보
    print(f"기준일: {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"종가: {latest['Close']:.2f}")
    if prev is not None:
        daily_change = (latest['Close'] - prev['Close']) / prev['Close'] * 100
        print(f"전일대비: {daily_change:.2f}%")
    
    # 이동평균선
    if 'MA5' in data.columns:
        print(f"5일 이동평균: {latest['MA5']:.2f}")
    if 'MA20' in data.columns:
        print(f"20일 이동평균: {latest['MA20']:.2f}")
    if 'MA60' in data.columns:
        print(f"60일 이동평균: {latest['MA60']:.2f}")
    
    # 추세 정보
    if 'Short_Trend' in data.columns:
        short_trend = "상승" if latest['Short_Trend'] > 0 else "하락"
        print(f"단기 추세(20일): {short_trend}")
    if 'Mid_Trend' in data.columns:
        mid_trend = "상승" if latest['Mid_Trend'] > 0 else "하락"
        print(f"중기 추세(60일): {mid_trend}")
    if 'Long_Trend' in data.columns:
        long_trend = "상승" if latest['Long_Trend'] > 0 else "하락"
        print(f"장기 추세(120일): {long_trend}")
    
    # 볼린저 밴드
    if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
        print(f"볼린저 밴드 상단: {latest['BB_Upper']:.2f}")
        print(f"볼린저 밴드 하단: {latest['BB_Lower']:.2f}")
    if 'BB_PB' in data.columns:
        bb_status = ""
        if latest['BB_PB'] > 1:
            bb_status = "과매수 구간"
        elif latest['BB_PB'] < 0:
            bb_status = "과매도 구간"
        else:
            bb_status = "중간 구간"
        print(f"볼린저 밴드 %B: {latest['BB_PB']:.2f} ({bb_status})")
    
    # RSI
    if 'RSI' in data.columns:
        rsi_status = ""
        if latest['RSI'] > 70:
            rsi_status = "과매수 구간"
        elif latest['RSI'] < 30:
            rsi_status = "과매도 구간"
        else:
            rsi_status = "중립 구간"
        print(f"RSI: {latest['RSI']:.2f} ({rsi_status})")
    
    # MACD
    if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
        macd_signal = "매수 신호" if latest['MACD'] > latest['MACD_Signal'] else "매도 신호"
        print(f"MACD: {latest['MACD']:.4f}, 시그널: {latest['MACD_Signal']:.4f} ({macd_signal})")
    
    print("="*50)
    print("* 참고: 이 분석은 단순 참고용이며, 투자 결정은 추가적인 연구가 필요합니다.")
    print("="*50)

def is_valid_ticker(ticker):
    """유효한 종목코드인지 확인"""
    # 기본 형식 검증
    if not ticker or not isinstance(ticker, str):
        return False
    
    # 영문, 숫자, 점(.)만 허용
    valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.")
    if not all(c in valid_chars for c in ticker.upper()):
        return False
    
    return True 