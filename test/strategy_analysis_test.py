#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def calculate_volume_profile(data, num_bins=50):
    """Volume Profile 계산"""
    try:
        # 가격 범위 계산
        price_min = data['Low'].min().item()
        price_max = data['High'].max().item()
        price_range = price_max - price_min
        
        if price_range == 0:
            return None, None, None, None, None
        
        # 가격 구간 설정
        price_bins = np.linspace(price_min, price_max, num_bins + 1)
        
        # 각 거래일의 가격과 거래량을 구간에 할당
        volume_profile = np.zeros(num_bins)
        
        for i in range(len(data)):
            low_price = data['Low'].iloc[i].item()
            high_price = data['High'].iloc[i].item()
            volume = data['Volume'].iloc[i].item()
            
            # 해당 거래일의 가격 범위가 포함되는 구간들 찾기
            start_bin = max(0, int((low_price - price_min) / price_range * num_bins))
            end_bin = min(num_bins - 1, int((high_price - price_min) / price_range * num_bins))
            
            # 각 구간에 거래량 분배
            for bin_idx in range(start_bin, end_bin + 1):
                volume_profile[bin_idx] += volume
        
        # POC (Point of Control) - 거래량이 가장 많은 가격 구간
        poc_bin = np.argmax(volume_profile)
        poc_price = price_bins[poc_bin]
        
        # Value Area 계산 (총 거래량의 70%를 포함하는 구간)
        total_volume = np.sum(volume_profile)
        target_volume = total_volume * 0.7
        
        # POC에서 시작하여 양쪽으로 확장
        current_volume = volume_profile[poc_bin]
        left_bin = poc_bin
        right_bin = poc_bin
        
        while current_volume < target_volume and (left_bin > 0 or right_bin < num_bins - 1):
            left_volume = volume_profile[left_bin - 1] if left_bin > 0 else 0
            right_volume = volume_profile[right_bin + 1] if right_bin < num_bins - 1 else 0
            
            if left_volume > right_volume and left_bin > 0:
                left_bin -= 1
                current_volume += left_volume
            elif right_bin < num_bins - 1:
                right_bin += 1
                current_volume += right_volume
            else:
                break
        
        value_area_min = price_bins[left_bin]
        value_area_max = price_bins[right_bin + 1]
        
        return price_bins, volume_profile, poc_price, value_area_min, value_area_max
        
    except Exception as e:
        print(f"Volume Profile 계산 오류: {e}")
        return None, None, None, None, None

def analyze_strategy_signal(ticker, period="12mo"):
    """차트 기반 투자 전략 분석"""
    try:
        # 날짜 범위인지 확인 (YYYY-MM-DD_YYYY-MM-DD 형식)
        if '_' in period and len(period.split('_')) == 2:
            start_date, end_date = period.split('_')
            print(f"날짜 범위로 데이터 다운로드: {start_date} ~ {end_date}")
            # 날짜 범위로 데이터 다운로드
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        else:
            # 기존 period 방식으로 데이터 다운로드
            data = yf.download(ticker, period=period, progress=False)
        
        if data.empty:
            return None, "데이터 없음"
        
        # Volume Profile 계산
        price_bins, volume_profile, poc_price, value_area_min, value_area_max = calculate_volume_profile(data)
        
        if poc_price is None:
            return None, "Volume Profile 계산 실패"
        
        # 현재가
        current_price = data['Close'].iloc[-1].item()
        
        # POC 가격의 +/- 10% 범위
        poc_range_min = poc_price * 0.9
        poc_range_max = poc_price * 1.1
        
        # 매수 조건 확인
        in_value_area = value_area_min <= current_price <= value_area_max
        in_poc_range = poc_range_min <= current_price <= poc_range_max
        
        # 추가 기술적 지표 계산
        # RSI 계산 (안전하게)
        try:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            if pd.isna(current_rsi):
                current_rsi = 50.0  # 기본값
        except:
            current_rsi = 50.0  # 오류 시 기본값
        
        # 이동평균 계산 (안전하게)
        try:
            ma20 = data['Close'].rolling(window=20).mean().iloc[-1]
            ma50 = data['Close'].rolling(window=50).mean().iloc[-1]
            if pd.isna(ma20):
                ma20 = current_price
            if pd.isna(ma50):
                ma50 = current_price
        except:
            ma20 = current_price
            ma50 = current_price
        
        # 매수 시그널 판단
        signal = "HOLD"
        signal_strength = 0
        
        # Volume Profile 조건
        if in_value_area and in_poc_range:
            signal_strength += 3  # 강한 매수 조건
        elif in_value_area:
            signal_strength += 2  # 일반 매수 조건
        
        # RSI 조건
        if current_rsi < 30:
            signal_strength += 1  # 과매도
        elif current_rsi > 70:
            signal_strength -= 1  # 과매수
        
        # 이동평균 조건
        if current_price > ma20 and ma20 > ma50:
            signal_strength += 1  # 상승 추세
        elif current_price < ma20 and ma20 < ma50:
            signal_strength -= 1  # 하락 추세
        
        # 최종 시그널 결정
        if signal_strength >= 3:
            signal = "BUY_STRONG"
        elif signal_strength >= 2:
            signal = "BUY"
        else:
            signal = "HOLD"
        
        # 분석 결과
        analysis = {
            'ticker': ticker,
            'current_price': current_price,
            'poc_price': poc_price,
            'value_area_min': value_area_min,
            'value_area_max': value_area_max,
            'poc_range_min': poc_range_min,
            'poc_range_max': poc_range_max,
            'in_value_area': in_value_area,
            'in_poc_range': in_poc_range,
            'rsi': current_rsi,
            'ma20': ma20,
            'ma50': ma50,
            'signal': signal,
            'signal_strength': signal_strength,
            'period': period
        }
        
        return analysis, None
        
    except Exception as e:
        return None, str(e)

def main():
    # 명령행 인자 처리
    if len(sys.argv) < 2:
        print("Usage: python strategy_analysis_test.py <ticker> [period]")
        sys.exit(1)
    
    ticker = sys.argv[1]
    period = sys.argv[2] if len(sys.argv) > 2 else "12mo"
    
    print(f"차트 기반 투자 전략 분석 시작: {ticker} (period={period})")
    print(f"분석할 파일 정보:")
    print(f"  - 종목 코드: {ticker}")
    print(f"  - 분석 기간: {period}")
    print(f"  - 데이터 소스: Yahoo Finance (yfinance)")
    print(f"  - 출력 디렉토리: output/hma_mantra/{ticker}/")
    print(f"  - 신호 파일: output/hma_mantra/{ticker}/{ticker}_signal.txt")
    print(f"  - 분석 지표: Volume Profile, RSI, MA20, MA50")
    print(f"  - 신호 유형: BUY_STRONG, BUY, HOLD")
    print(f"분석 시작...")
    
    # 분석 실행
    analysis, error = analyze_strategy_signal(ticker, period)
    
    if error:
        print(f"분석 실패: {error}")
        sys.exit(1)
    
    # 결과 출력
    print(f"\n=== {ticker} 투자 전략 분석 결과 ===")
    print(f"현재가: ${analysis['current_price']:.2f}")
    print(f"POC: ${analysis['poc_price']:.2f}")
    print(f"Value Area: ${analysis['value_area_min']:.2f} - ${analysis['value_area_max']:.2f}")
    print(f"RSI: {analysis['rsi']:.1f}")
    print(f"MA20: ${analysis['ma20']:.2f}, MA50: ${analysis['ma50']:.2f}")
    print(f"매수 시그널: {analysis['signal']}")
    print(f"신호 강도: {analysis['signal_strength']}")
    
    # 매수 근거 설명
    print(f"\n매수 근거:")
    if analysis['in_value_area']:
        print("- Value Area 내에서 거래 중")
    if analysis['in_poc_range']:
        print("- POC 근처에서 거래 중")
    if analysis['rsi'] < 30:
        print("- RSI 과매도 구간")
    elif analysis['rsi'] > 70:
        print("- RSI 과매수 구간")
    if analysis['current_price'] > analysis['ma20'] and analysis['ma20'] > analysis['ma50']:
        print("- 상승 추세")
    elif analysis['current_price'] < analysis['ma20'] and analysis['ma20'] < analysis['ma50']:
        print("- 하락 추세")
    
    # 투자 전략 제안
    print(f"\n투자 전략:")
    if analysis['signal'] == 'BUY_STRONG':
        print("- 강한 매수 신호: Value Area 내 + POC 근처 + 기술적 지지")
        print(f"- 목표가: ${analysis['poc_price']:.2f} (POC)")
        print(f"- 손절가: ${analysis['value_area_min']:.2f} (Value Area 하단)")
    elif analysis['signal'] == 'BUY':
        print("- 일반 매수 신호: Value Area 내에서 거래 중")
        print(f"- 목표가: ${analysis['poc_price']:.2f} (POC)")
        print(f"- 손절가: ${analysis['value_area_min']:.2f} (Value Area 하단)")
    else:
        print("- 보유 신호: 현재 매수 조건 미충족")
    
    # 신호 파일 저장
    output_dir = Path(f"output/hma_mantra/{ticker}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    signal_file = output_dir / f"{ticker}_signal.txt"
    with open(signal_file, "w", encoding='utf-8') as f:
        f.write(f"=== {ticker} 투자 전략 분석 결과 ===\n")
        f.write(f"분석 날짜: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"분석 기간: {analysis['period']}\n")
        f.write(f"현재가: ${analysis['current_price']:.2f}\n")
        f.write(f"POC: ${analysis['poc_price']:.2f}\n")
        f.write(f"Value Area: ${analysis['value_area_min']:.2f} - ${analysis['value_area_max']:.2f}\n")
        f.write(f"POC 범위: ${analysis['poc_range_min']:.2f} - ${analysis['poc_range_max']:.2f}\n")
        f.write(f"RSI: {analysis['rsi']:.1f}\n")
        f.write(f"MA20: ${analysis['ma20']:.2f}\n")
        f.write(f"MA50: ${analysis['ma50']:.2f}\n")
        f.write(f"신호 강도: {analysis['signal_strength']}\n")
        f.write(f"최종 신호: {analysis['signal']}\n\n")
        
        f.write("=== 매수 조건 분석 ===\n")
        f.write(f"Value Area 내 거래: {'예' if analysis['in_value_area'] else '아니오'}\n")
        f.write(f"POC 범위 내 거래: {'예' if analysis['in_poc_range'] else '아니오'}\n")
        f.write(f"RSI 과매도 (<30): {'예' if analysis['rsi'] < 30 else '아니오'}\n")
        f.write(f"RSI 과매수 (>70): {'예' if analysis['rsi'] > 70 else '아니오'}\n")
        f.write(f"상승 추세 (현재가 > MA20 > MA50): {'예' if analysis['current_price'] > analysis['ma20'] and analysis['ma20'] > analysis['ma50'] else '아니오'}\n")
        f.write(f"하락 추세 (현재가 < MA20 < MA50): {'예' if analysis['current_price'] < analysis['ma20'] and analysis['ma20'] < analysis['ma50'] else '아니오'}\n\n")
        
        f.write("=== 투자 전략 제안 ===\n")
        if analysis['signal'] == 'BUY_STRONG':
            f.write("강한 매수 신호: Value Area 내 + POC 근처 + 기술적 지지\n")
            f.write(f"목표가: ${analysis['poc_price']:.2f} (POC)\n")
            f.write(f"손절가: ${analysis['value_area_min']:.2f} (Value Area 하단)\n")
            f.write("매수 근거: Volume Profile과 기술적 지표가 모두 긍정적\n")
        elif analysis['signal'] == 'BUY':
            f.write("일반 매수 신호: Value Area 내에서 거래 중\n")
            f.write(f"목표가: ${analysis['poc_price']:.2f} (POC)\n")
            f.write(f"손절가: ${analysis['value_area_min']:.2f} (Value Area 하단)\n")
            f.write("매수 근거: Volume Profile 조건 충족\n")
        else:
            f.write("보유 신호: 현재 매수 조건 미충족\n")
            f.write("권장사항: 추가 모니터링 필요\n")
        
        f.write(f"\n=== 신호 요약 ===\n")
        f.write(f"{analysis['signal']}")
    
    print(f"\n=== 생성된 파일 정보 ===")
    print(f"📁 출력 디렉토리: {output_dir}")
    print(f"📄 신호 파일: {signal_file}")
    print(f"📊 신호 내용: {analysis['signal']}")
    
    # 기존 파일들 확인
    existing_files = list(output_dir.glob("*"))
    if existing_files:
        print(f"\n📂 기존 분석 파일들:")
        for file in existing_files:
            if file.is_file():
                file_size = file.stat().st_size
                print(f"  - {file.name} ({file_size} bytes)")
    
    print(f"\n✅ 분석 완료!")
    print(f"💡 다음 명령어로 결과 확인:")
    print(f"   cat {signal_file}")
    print(f"   ls -la {output_dir}")

if __name__ == "__main__":
    main() 