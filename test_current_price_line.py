#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
현재가 수직선 스타일 테스트 스크립트
두 가지 스타일의 현재가 수직선을 비교하여 확인할 수 있습니다.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from src.indicators.hma_mantra.visualization.volume_profile_overlay_chart import plot_main_chart_with_volume_profile_overlay

def test_current_price_line_styles():
    """현재가 수직선 스타일을 테스트합니다."""
    
    # 테스트할 종목과 기간 설정
    ticker = 'AAPL'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6개월 데이터
    
    print(f"=== {ticker} 현재가 수직선 스타일 테스트 ===")
    print(f"분석 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    
    try:
        # 주식 데이터 가져오기
        print("주식 데이터를 가져오는 중...")
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            print("데이터를 가져올 수 없습니다.")
            return
        
        print(f"데이터 로드 완료: {len(data)}개 캔들")
        print(f"현재가: ${data['Close'].iloc[-1]:.2f}")
        
        # 출력 디렉토리 생성
        output_dir = "test/current_price_line_test"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 얇은 수직선 스타일 테스트
        print("\n1. 얇은 수직선 스타일 테스트 중...")
        thin_chart_path = os.path.join(output_dir, f"{ticker}_thin_style_chart.png")
        plot_main_chart_with_volume_profile_overlay(
            data=data,
            ticker=ticker,
            save_path=thin_chart_path,
            current_price_line_style='thin'
        )
        print(f"   완료: {thin_chart_path}")
        
        # 2. 하단 시작 수직선 스타일 테스트
        print("\n2. 하단 시작 수직선 스타일 테스트 중...")
        bottom_chart_path = os.path.join(output_dir, f"{ticker}_bottom_start_style_chart.png")
        plot_main_chart_with_volume_profile_overlay(
            data=data,
            ticker=ticker,
            save_path=bottom_chart_path,
            current_price_line_style='bottom_start'
        )
        print(f"   완료: {bottom_chart_path}")
        
        # 3. 기본 스타일 테스트 (매개변수 생략)
        print("\n3. 기본 스타일 테스트 중...")
        default_chart_path = os.path.join(output_dir, f"{ticker}_default_style_chart.png")
        plot_main_chart_with_volume_profile_overlay(
            data=data,
            ticker=ticker,
            save_path=default_chart_path
        )
        print(f"   완료: {default_chart_path}")
        
        print(f"\n=== 테스트 완료 ===")
        print(f"생성된 차트 파일들:")
        print(f"  - 얇은 수직선: {thin_chart_path}")
        print(f"  - 하단 시작: {bottom_chart_path}")
        print(f"  - 기본 스타일: {default_chart_path}")
        print(f"\n차트를 비교하여 현재가 수직선 스타일의 차이를 확인하세요.")
        
    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def compare_line_styles():
    """두 스타일의 차이점을 설명합니다."""
    print("\n=== 현재가 수직선 스타일 비교 ===")
    print("\n1. 얇은 수직선 (thin):")
    print("   - linewidth: 0.5")
    print("   - alpha: 0.6")
    print("   - ymin: 0, ymax: 1 (전체 차트)")
    print("   - 캔들바와 겹쳐도 잘 보임")
    
    print("\n2. 하단 시작 수직선 (bottom_start):")
    print("   - linewidth: 1.0")
    print("   - alpha: 0.7")
    print("   - ymin: 현재 캔들바 하단, ymax: 1")
    print("   - 캔들바와 겹치지 않음")
    
    print("\n3. 기본값:")
    print("   - current_price_line_style 매개변수를 생략하면 'thin' 사용")

if __name__ == "__main__":
    print("현재가 수직선 스타일 테스트를 시작합니다...")
    
    # 스타일 비교 정보 출력
    compare_line_styles()
    
    # 사용자 확인
    response = input("\n테스트를 실행하시겠습니까? (y/n): ").lower().strip()
    
    if response in ['y', 'yes', '예']:
        test_current_price_line_styles()
    else:
        print("테스트를 건너뜁니다.")
        print("\n사용법:")
        print("python test_current_price_line.py")
        print("\n또는 코드에서 직접 호출:")
        print("from src.indicators.hma_mantra.visualization.volume_profile_overlay_chart import plot_main_chart_with_volume_profile_overlay")
        print("plot_main_chart_with_volume_profile_overlay(data, ticker, save_path, current_price_line_style='thin')")
