import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from src.indicators.hma_mantra.visualization.advanced import plot_hma_mantra_md_signals
import matplotlib.pyplot as plt

def get_scalar(val):
    # 값이 Series면 첫 번째 값만 추출, 아니면 그대로 반환
    if isinstance(val, pd.Series):
        return val.values[0]
    return val

def calculate_rsi_signals(data):
    """RSI 기반 매매 신호 생성"""
    # RSI 계산 (3, 14, 50 기간)
    delta = data['Close'].diff()
    
    # RSI(3) 계산
    gain_3 = (delta.where(delta > 0, 0)).rolling(window=3).mean()
    loss_3 = (-delta.where(delta < 0, 0)).rolling(window=3).mean()
    rs_3 = gain_3 / loss_3
    rsi_3 = 100 - (100 / (1 + rs_3))
    
    # RSI(14) 계산
    gain_14 = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss_14 = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs_14 = gain_14 / loss_14
    rsi_14 = 100 - (100 / (1 + rs_14))
    
    # RSI(50) 계산
    gain_50 = (delta.where(delta > 0, 0)).rolling(window=50).mean()
    loss_50 = (-delta.where(delta < 0, 0)).rolling(window=50).mean()
    rs_50 = gain_50 / loss_50
    rsi_50 = 100 - (100 / (1 + rs_50))
    
    # 신호 생성
    signal = "HOLD"
    
    # 최근 RSI 값들
    current_rsi_3 = get_scalar(rsi_3.iloc[-1])
    prev_rsi_3 = get_scalar(rsi_3.iloc[-2])
    current_rsi_14 = get_scalar(rsi_14.iloc[-1])
    current_rsi_50 = get_scalar(rsi_50.iloc[-1])

    if (pd.isna(current_rsi_3) or pd.isna(prev_rsi_3) or
        pd.isna(current_rsi_14) or pd.isna(current_rsi_50)):
        return "HOLD"

    current_rsi_3 = float(current_rsi_3)
    prev_rsi_3 = float(prev_rsi_3)
    current_rsi_14 = float(current_rsi_14)
    current_rsi_50 = float(current_rsi_50)
    
    # 매수 조건:
    # 1. RSI(3), RSI(14)가 30 이하
    # 2. RSI(50)이 50 이하
    # 3. RSI(3)가 30을 상향 돌파
    if (current_rsi_3 > 30 and prev_rsi_3 <= 30 and current_rsi_14 <= 30 and current_rsi_50 <= 50):
        signal = "BUY"
    
    # 매도 조건:
    # 1. RSI(3), RSI(14)가 70 이상
    # 2. RSI(50)이 50 이상
    # 3. RSI(3)가 70을 하향 돌파
    elif (current_rsi_3 < 70 and prev_rsi_3 >= 70 and current_rsi_14 >= 70 and current_rsi_50 >= 50):
        signal = "SELL"
    
    return signal

def main():
    # 명령행 인자 처리
    if len(sys.argv) < 2:
        print("Usage: python hma_mantra_example.py <ticker> [period] [interval] [prepost] [current_price]")
        sys.exit(1)

    ticker = sys.argv[1]
    period = sys.argv[2] if len(sys.argv) > 2 else "120d"
    interval = sys.argv[3] if len(sys.argv) > 3 else "1d"
    prepost = sys.argv[4].lower() == "true" if len(sys.argv) > 4 else False
    current_price = float(sys.argv[5]) if len(sys.argv) > 5 else None

    try:
        print(f"데이터 다운로드 시작: {ticker}, 기간: {period}")
        # 데이터 다운로드
        data = yf.download(ticker, period=period, interval=interval, prepost=prepost)
        if data.empty:
            print(f"데이터를 가져올 수 없습니다: {ticker}")
            sys.exit(1)

        # 저장 경로 설정
        save_dir = Path("output/hma_mantra") / ticker
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{ticker}_hma_mantra_md_signals.png"

        # 차트 생성
        plot_hma_mantra_md_signals(data, ticker, str(save_path), current_price)
        print(f"분석 완료: {save_path}")

        # RSI 기반 신호 생성
        signal = calculate_rsi_signals(data)
        
        # 신호 파일 생성
        signal_path = save_dir / f"{ticker}_signal.txt"
        with open(signal_path, "w") as f:
            f.write(signal)
        print(f"신호 파일 생성 완료: {signal_path} (신호: {signal})")

    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def create_sector_dashboard(symbol):
    """섹터 분석 대시보드 생성"""
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 섹터 구분도
    ax1 = fig.add_subplot(221)
    plot_sector_tree(ax1, symbol)
    
    # 2. 섹터 성과
    ax2 = fig.add_subplot(222)
    plot_sector_performance(ax2)
    
    # 3. 경쟁사 비교
    ax3 = fig.add_subplot(223)
    plot_peer_comparison(ax3, symbol)
    
    # 4. 투자전략
    ax4 = fig.add_subplot(224)
    plot_investment_strategy(ax4)

if __name__ == "__main__":
    main() 