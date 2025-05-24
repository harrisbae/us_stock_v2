import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
from indicators.hma_mantra import (
    calculate_hma,
    calculate_mantra_bands,
    plot_hma_mantra,
    plot_comparison,
    plot_signals_with_strength,
    get_hma_signals,
    get_mantra_signals
)
from datetime import datetime, timedelta

def analyze_stock(ticker: str, period: str = '6mo', interval: str = '1d'):
    """
    주식 분석 실행
    
    Args:
        ticker (str): 종목 코드
        period (str): 기간
        interval (str): 간격
    """
    # 데이터 다운로드
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1년치 데이터
    
    print(f"{ticker} 데이터 다운로드 중...")
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        print(f"{ticker} 데이터를 다운로드할 수 없습니다.")
        return
    
    # 출력 디렉토리 생성
    output_dir = f"output/hma_mantra/{ticker}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 분석 실행
    print(f"{ticker} 분석 중...")
    plot_signals_with_strength(
        data=data,
        ticker=ticker,
        save_path=f"{output_dir}/{ticker}_analysis.png"
    )
    print(f"분석 완료: {output_dir}/{ticker}_analysis.png")
    
    # 신호 출력
    hma_signals = get_hma_signals(data)
    mantra_signals = get_mantra_signals(data)
    
    print(f"\n{ticker} 분석 결과:")
    print("=" * 50)
    
    print("\n최근 HMA 신호:")
    print("-" * 30)
    print("날짜            신호    강도    가격")
    print("-" * 30)
    if hma_signals:
        for signal in hma_signals[-5:]:  # 최근 5개 신호만 출력
            date_str = signal['date'].strftime('%Y-%m-%d')
            print(f"{date_str}      {signal['type']}     {signal['strength']}  ${signal['price']:.2f}")
    else:
        print("(신호 없음)")
    
    print("\n최근 만트라 밴드 신호:")
    print("-" * 30)
    print("날짜            신호            강도    가격")
    print("-" * 30)
    if mantra_signals:
        for signal in mantra_signals[-5:]:  # 최근 5개 신호만 출력
            date_str = signal['date'].strftime('%Y-%m-%d')
            print(f"{date_str}      {signal['type']}      {signal['strength']}  ${signal['price']:.2f}")
    else:
        print("(신호 없음)")
    
    # 그래프 저장
    plot_hma_mantra(data, ticker, f"{output_dir}/{ticker}_hma_mantra.png")
    
    # 비교 그래프 저장
    plot_comparison(data, ticker, f"{output_dir}/{ticker}_comparison.png")
    
    # 시그널 강도 그래프 저장
    plot_signals_with_strength(data, ticker, f"{output_dir}/{ticker}_signals.png")
    
    print(f"\n그래프가 {output_dir} 디렉토리에 저장되었습니다.")
    print(f"- HMA & 만트라 밴드: {ticker}_hma_mantra.png")
    print(f"- 비교 그래프: {ticker}_comparison.png")
    print(f"- 시그널 강도 그래프: {ticker}_signals.png")

if __name__ == "__main__":
    # 사용자로부터 종목 코드, 기간, 간격 입력 받기
    print("\n분석할 종목 코드를 입력하세요 (여러 종목은 쉼표로 구분)")
    print("예시: AAPL,MSFT,GOOGL 또는 086790.KS")
    user_input = input("종목 코드: ").strip()
    period_input = input("기간(예: 12mo, 6mo, 3mo 등): ").strip()
    interval_input = input("간격(예: 1d, 1wk 등): ").strip()

    # 입력된 종목 코드 처리
    if user_input:
        tickers = [ticker.strip() for ticker in user_input.split(',')]
    else:
        print("종목 코드가 입력되지 않았습니다. 기본 종목을 분석합니다.")
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

    # 각 종목 분석 실행
    for ticker in tickers:
        try:
            print(f"\n{ticker} 분석을 시작합니다...")
            analyze_stock(ticker, period=period_input, interval=interval_input)
        except Exception as e:
            print(f"\n{ticker} 분석 중 오류가 발생했습니다: {type(e)} - {str(e)}")
            continue 