#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기존 차트에 통합 시장 심리 지표를 추가한 완전한 구현 테스트
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from indicators.hma_mantra.visualization.volume_profile_overlay_chart import plot_main_chart_with_volume_profile_overlay
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

def test_existing_chart_with_sentiment():
    """기존 차트에 통합 시장 심리 지표를 추가한 테스트"""
    
    print("=== 기존 차트 + 통합 시장 심리 지표 테스트 ===")
    
    # 테스트 데이터 생성
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    # AAPL 데이터 다운로드
    try:
        print("AAPL 데이터 다운로드 중...")
        aapl = yf.download('AAPL', start=start_date, end=end_date, progress=False)
        print(f"AAPL 데이터 로드 완료: {len(aapl)}개")
    except Exception as e:
        print(f"AAPL 데이터 로드 실패: {e}")
        return
    
    # 데이터를 DataFrame으로 변환 (MultiIndex 컬럼 처리)
    data = aapl[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    print("차트 생성 시작...")
    
    # 기존 차트 함수 호출 (통합 시장 심리 지표 포함)
    plot_main_chart_with_volume_profile_overlay(
        data=data,
        ticker='AAPL',
        save_path='test_output/aapl_with_sentiment.png',
        current_price=aapl['Close'].iloc[-1],
        rsi_divergence_window=60,
        rsi_pivot_span=5,
        include_hidden_divergence=True,
        current_price_line_style='thin'
    )
    
    print("차트 생성 완료!")
    print("저장된 파일: test_output/aapl_with_sentiment.png")

if __name__ == "__main__":
    # test_output 디렉토리 생성
    os.makedirs('test_output', exist_ok=True)
    
    test_existing_chart_with_sentiment()
