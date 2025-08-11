#!/usr/bin/env python3
"""
Volume Profile 오버레이 차트 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from src.indicators.hma_mantra.visualization.volume_profile_overlay_chart import plot_main_chart_with_volume_profile_overlay

def main():
    import sys
    
    # 명령행 인자 처리
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    else:
        ticker = "TSLA"  # 기본값
    
    # 기간 설정 (명령행 인자에서 받기)
    if len(sys.argv) > 2:
        period = sys.argv[2]
    else:
        period = "6mo"  # 기본값
    
    # RSI 다이버전스 옵션: [유효 윈도우, 피벗 스팬]
    if len(sys.argv) > 3:
        try:
            rsi_window = int(sys.argv[3])
        except Exception:
            rsi_window = 20
    else:
        rsi_window = 20
    
    if len(sys.argv) > 4:
        try:
            rsi_pivot = int(sys.argv[4])
        except Exception:
            rsi_pivot = 3
    else:
        rsi_pivot = 3
    
    print(f"데이터 다운로드 중: {ticker} (period={period})")
    
    # 날짜 범위인지 확인 (YYYY-MM-DD_YYYY-MM-DD 형식)
    if '_' in period and len(period.split('_')) == 2:
        start_date, end_date = period.split('_')
        print(f"날짜 범위: {start_date} ~ {end_date}")
        
        # yfinance는 end 날짜를 포함하지 않으므로 하루를 더해서 다운로드
        from datetime import datetime, timedelta
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        end_date_plus_one = (end_dt + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # 날짜 범위로 데이터 다운로드 (종료일 포함을 위해 하루 추가)
        data = yf.download(ticker, start=start_date, end=end_date_plus_one, progress=False)
    else:
        # 기존 period 방식으로 데이터 다운로드
        data = yf.download(ticker, period=period, progress=False)
    
    if data.empty:
        print(f"데이터를 가져올 수 없습니다: {ticker}")
        return
    
    print(f"다운로드 완료: {len(data)} 개 데이터")
    print(f"데이터 기간: {data.index[0].strftime('%Y-%m-%d')} ~ {data.index[-1].strftime('%Y-%m-%d')}")
    
    # 출력 경로 설정
    output_dir = f"output/hma_mantra/{ticker}"
    os.makedirs(output_dir, exist_ok=True)
    save_path = f"{output_dir}/{ticker}_volume_profile_overlay_{period}_chart.png"
    
    print(f"Volume Profile 오버레이 차트 생성 중...")
    
    # Volume Profile 오버레이 차트 생성
    plot_main_chart_with_volume_profile_overlay(
        data=data,
        ticker=ticker,
        save_path=save_path,
        rsi_divergence_window=rsi_window,
        rsi_pivot_span=rsi_pivot,
        include_hidden_divergence=True
    )
    
    print(f"차트 저장 완료: {save_path}")

if __name__ == "__main__":
    main() 