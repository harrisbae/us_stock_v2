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
    import argparse
    
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='Volume Profile 오버레이 차트 생성기')
    parser.add_argument('ticker', nargs='?', default='TSLA', help='종목코드 (기본값: TSLA)')
    parser.add_argument('period', nargs='?', default='6mo', help='데이터 기간 (기본값: 6mo)')
    parser.add_argument('rsi_window', nargs='?', type=int, default=20, help='RSI 다이버전스 윈도우 (기본값: 20)')
    parser.add_argument('rsi_pivot', nargs='?', type=int, default=3, help='RSI 피벗 스팬 (기본값: 3)')
    parser.add_argument('target_prices', nargs='?', default='', help='Target 가격들 (형식: buy:sell:stop)')
    parser.add_argument('show_targets', nargs='?', default='true', help='Target 가격 표시 여부 (기본값: true)')
    parser.add_argument('--auto_adjust', default='false', help='가격 조정 여부 (기본값: false)')
    
    # argparse로 인자 파싱
    args = parser.parse_args()
    
    # 변수 할당
    ticker = args.ticker
    period = args.period
    rsi_window = args.rsi_window
    rsi_pivot = args.rsi_pivot
    auto_adjust = args.auto_adjust.lower() in ['true', '1', 'yes', 'y']
    
    # Target 가격 옵션들
    target_buy_price = None
    target_sell_price = None
    stop_loss_price = None
    show_target_prices = args.show_targets.lower() in ['true', '1', 'yes', 'y']
    
    # Target 가격들을 명령행 인자에서 받기 (형식: buy:sell:stop)
    if args.target_prices:
        target_prices = args.target_prices.split(':')
        if len(target_prices) >= 1 and target_prices[0]:
            try:
                target_buy_price = float(target_prices[0])
            except ValueError:
                target_buy_price = None
        
        if len(target_prices) >= 2 and target_prices[1]:
            try:
                target_sell_price = float(target_prices[1])
            except ValueError:
                target_sell_price = None
        
        if len(target_prices) >= 3 and target_prices[2]:
            try:
                stop_loss_price = float(target_prices[2])
            except ValueError:
                stop_loss_price = None
    
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
        data = yf.download(ticker, start=start_date, end=end_date_plus_one, progress=False, auto_adjust=auto_adjust)
    else:
        # 기존 period 방식으로 데이터 다운로드
        data = yf.download(ticker, period=period, progress=False, auto_adjust=auto_adjust)
    
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
        include_hidden_divergence=True,
        # Target 가격 옵션들 추가
        target_buy_price=target_buy_price,
        target_sell_price=target_sell_price,
        stop_loss_price=stop_loss_price,
        show_target_prices=show_target_prices
    )
    
    print(f"차트 저장 완료: {save_path}")

if __name__ == "__main__":
    main() 