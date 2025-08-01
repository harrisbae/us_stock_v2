#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import yfinance as yf
from datetime import datetime, timedelta

# 프로젝트 루트 경로를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indicators.hma_mantra.visualization.financial_analysis_chart import (
    get_financial_data, 
    plot_main_chart_with_financial_analysis,
    calculate_financial_correlation
)

def main():
    import sys
    
    # 명령행 인자 처리
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    else:
        ticker = "AAPL"  # 기본값
    
    # 기간 설정 (명령행 인자에서 받기)
    if len(sys.argv) > 2:
        period = sys.argv[2]
    else:
        period = "2y"  # 기본값 (재무 데이터는 더 긴 기간이 필요)
    
    print(f"재무 데이터 분석 시작: {ticker} (period={period})")
    
    # 재무 데이터 가져오기
    financial_metrics, price_data = get_financial_data(ticker, period)
    
    if price_data.empty:
        print(f"주가 데이터를 가져올 수 없습니다: {ticker}")
        return
    
    if not financial_metrics:
        print(f"재무 데이터를 가져올 수 없습니다: {ticker}")
        print("일부 기업의 경우 재무 데이터가 제한적일 수 있습니다.")
        return
    
    print(f"데이터 다운로드 완료: {len(price_data)} 개 주가 데이터")
    print(f"주가 기간: {price_data.index[0].strftime('%Y-%m-%d')} ~ {price_data.index[-1].strftime('%Y-%m-%d')}")
    
    # 사용 가능한 재무 지표 출력
    available_metrics = [k for k, v in financial_metrics.items() if not v.empty]
    print(f"사용 가능한 재무 지표: {', '.join(available_metrics)}")
    

    
    # 출력 경로 설정
    output_dir = f"output/hma_mantra/{ticker}"
    os.makedirs(output_dir, exist_ok=True)
    save_path = f"{output_dir}/{ticker}_financial_analysis_{period}_chart.png"
    
    print(f"재무 분석 차트 생성 중...")
    
    # 재무 분석 차트 생성
    plot_main_chart_with_financial_analysis(
        data=price_data,
        financial_metrics=financial_metrics,
        ticker=ticker,
        save_path=save_path
    )
    
    # 상관관계 분석
    print(f"상관관계 분석 중...")
    correlations = calculate_financial_correlation(price_data, financial_metrics)
    
    if correlations:
        print("\n=== 주가와 재무지표 간 상관관계 ===")
        for indicator, corr in correlations.items():
            print(f"{indicator}: {corr:.4f}")
    
    print(f"재무 분석 완료: {save_path}")

if __name__ == "__main__":
    main() 