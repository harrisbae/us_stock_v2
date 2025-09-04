#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yfinance 데이터 형태 디버깅
"""

import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

def debug_data_shape():
    """yfinance 데이터 형태 디버깅"""
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    # AAPL 데이터 다운로드
    aapl = yf.download('AAPL', start=start_date, end=end_date, progress=False)
    
    print("=== yfinance 데이터 형태 분석 ===")
    print(f"전체 데이터 형태: {aapl.shape}")
    print(f"컬럼: {aapl.columns.tolist()}")
    print(f"인덱스 타입: {type(aapl.index)}")
    print(f"인덱스 길이: {len(aapl.index)}")
    
    print("\n=== 각 컬럼별 형태 ===")
    for col in aapl.columns:
        print(f"{col}: {aapl[col].shape}, 타입: {type(aapl[col])}")
        print(f"  첫 번째 값: {aapl[col].iloc[0]}, 타입: {type(aapl[col].iloc[0])}")
        print(f"  .values 형태: {aapl[col].values.shape}")
        print()
    
    print("=== 데이터 샘플 ===")
    print(aapl.head())
    
    print("\n=== 올바른 DataFrame 생성 방법 ===")
    
    # 방법 1: 직접 사용
    print("방법 1: 원본 데이터 직접 사용")
    data1 = aapl[['Open', 'High', 'Low', 'Close', 'Volume']]
    print(f"결과 형태: {data1.shape}")
    
    # 방법 2: copy() 사용
    print("방법 2: copy() 사용")
    data2 = aapl[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    print(f"결과 형태: {data2.shape}")
    
    # 방법 3: 새로운 DataFrame 생성
    print("방법 3: 새로운 DataFrame 생성")
    data3 = pd.DataFrame({
        'Open': aapl['Open'],
        'High': aapl['High'],
        'Low': aapl['Low'],
        'Close': aapl['Close'],
        'Volume': aapl['Volume']
    })
    print(f"결과 형태: {data3.shape}")

if __name__ == "__main__":
    debug_data_shape()
