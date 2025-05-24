import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import yfinance as yf
import pandas as pd
from pathlib import Path
from src.indicators.hma_mantra.visualization.advanced import plot_hma_mantra_md_signals
import matplotlib.pyplot as plt

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