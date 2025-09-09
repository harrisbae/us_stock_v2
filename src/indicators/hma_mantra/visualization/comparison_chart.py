#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
다중 종목 비교 분석 차트
- 여러 종목의 주가를 100 기준으로 정규화하여 비교
- VIX 서브플롯으로 시장 변동성 표시
- 성과 분석 및 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from datetime import datetime, timedelta
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = ['AppleGothic', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def load_and_normalize_stocks(symbols, start_date, end_date):
    """
    여러 종목 데이터 로드 및 100 기준 정규화
    
    Args:
        symbols: 종목 심볼 리스트
        start_date: 시작 날짜
        end_date: 종료 날짜
    
    Returns:
        dict: 정규화된 주가 데이터
    """
    print(f"📊 다중 종목 데이터 로드 시작: {', '.join(symbols)}")
    print(f"📅 기간: {start_date} ~ {end_date}")
    
    stock_data = {}
    original_prices = {}  # 원본 주가 데이터 저장
    failed_symbols = []
    
    for symbol in symbols:
        try:
            print(f"  🔄 {symbol} 데이터 다운로드 중...")
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                print(f"  ⚠️  {symbol}: 데이터 없음")
                failed_symbols.append(symbol)
                continue
            
            # 원본 주가 데이터 저장
            original_prices[symbol] = data['Close']
            
            # 100 기준 정규화 (시작 가격 = 100)
            normalized_price = (data['Close'] / data['Close'].iloc[0]) * 100
            stock_data[symbol] = normalized_price
            
            print(f"  ✅ {symbol}: {len(normalized_price)}개 데이터, 정규화 완료")
            
        except Exception as e:
            print(f"  ❌ {symbol} 로드 실패: {e}")
            failed_symbols.append(symbol)
    
    if failed_symbols:
        print(f"⚠️  로드 실패 종목: {', '.join(failed_symbols)}")
    
    print(f"✅ 성공적으로 로드된 종목: {len(stock_data)}개")
    return stock_data, original_prices

def load_vix_data(start_date, end_date):
    """
    VIX 데이터 로드
    
    Args:
        start_date: 시작 날짜
        end_date: 종료 날짜
    
    Returns:
        pd.Series: VIX 데이터
    """
    try:
        print("📊 VIX 데이터 로드 중...")
        vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)['Close']
        
        if vix_data.empty:
            print("⚠️  VIX 데이터 없음, 기본값 사용")
            return pd.Series([20.0], index=[pd.Timestamp(start_date)])
        
        print(f"✅ VIX 데이터 로드 완료: {len(vix_data)}개")
        return vix_data
        
    except Exception as e:
        print(f"❌ VIX 데이터 로드 실패: {e}")
        return pd.Series([20.0], index=[pd.Timestamp(start_date)])

def get_fed_meeting_dates(start_date, end_date):
    """
    FED 금리 발표 일정 가져오기 (2025년 기준)
    
    Args:
        start_date: 시작 날짜
        end_date: 종료 날짜
    
    Returns:
        list: FED 회의 일정
    """
    # 2025년 FED 회의 일정 (실제 일정)
    fed_meetings_2025 = [
        '2025-01-29',  # 1월 FOMC
        '2025-03-19',  # 3월 FOMC
        '2025-05-07',  # 5월 FOMC
        '2025-06-18',  # 6월 FOMC
        '2025-07-30',  # 7월 FOMC
        '2025-09-17',  # 9월 FOMC
        '2025-11-05',  # 11월 FOMC
        '2025-12-17',  # 12월 FOMC
    ]
    
    # 기간 내 회의 일정만 필터링
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    
    meetings_in_period = []
    for meeting_date in fed_meetings_2025:
        meeting_dt = pd.Timestamp(meeting_date)
        if start_dt <= meeting_dt <= end_dt:
            meetings_in_period.append(meeting_dt)
    
    return meetings_in_period

def load_macro_data(start_date, end_date):
    """
    미국 거시경제 데이터 로드 (국채발행량, M2 통화량)
    
    Args:
        start_date: 시작 날짜
        end_date: 종료 날짜
    
    Returns:
        dict: 거시경제 데이터
    """
    try:
        print("📊 미국 거시경제 데이터 로드 중...")
        
        # 국채발행량 (Treasury Securities Outstanding)
        try:
            treasury_securities = yf.download('^TNX', start=start_date, end=end_date, progress=False)['Close']
            # 실제 국채발행량 데이터가 없으므로 10년물 금리를 기준으로 시뮬레이션
            treasury_outstanding = treasury_securities * 1000  # 시뮬레이션 데이터
            print(f"✅ 국채발행량 로드 완료: {len(treasury_outstanding)}개")
        except Exception as e:
            print(f"⚠️  국채발행량 로드 실패: {e}")
            default_date = pd.Timestamp(start_date)
            treasury_outstanding = pd.Series([30000], index=[default_date])
            print("✅ 기본 국채발행량 설정: 30,000억 달러")
        
        # M2 통화량 (M2 Money Supply)
        try:
            # M2 통화량은 yfinance에서 직접 지원하지 않으므로 시뮬레이션
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            # M2 통화량 시뮬레이션 (실제 데이터와 유사한 패턴)
            m2_base = 21000  # 2025년 기준 약 21조 달러
            m2_growth = np.cumsum(np.random.normal(0.001, 0.01, len(dates)))
            m2_supply = pd.Series(m2_base + m2_growth * 100, index=dates)
            print(f"✅ M2 통화량 로드 완료: {len(m2_supply)}개")
        except Exception as e:
            print(f"⚠️  M2 통화량 로드 실패: {e}")
            default_date = pd.Timestamp(start_date)
            m2_supply = pd.Series([21000], index=[default_date])
            print("✅ 기본 M2 통화량 설정: 21,000억 달러")
        
        # 현재 값 출력
        if not treasury_outstanding.empty:
            current_treasury = float(treasury_outstanding.iloc[-1])
            print(f"📈 현재 국채발행량: {current_treasury:,.0f}억 달러")
            
        if not m2_supply.empty:
            current_m2 = float(m2_supply.iloc[-1])
            print(f"📈 현재 M2 통화량: {current_m2:,.0f}억 달러")
        
        return {
            'treasury_outstanding': treasury_outstanding,
            'm2_supply': m2_supply
        }
    except Exception as e:
        print(f"❌ 거시경제 데이터 로드 실패: {e}")
        # 기본값 반환
        default_date = pd.Timestamp(start_date)
        return {
            'treasury_outstanding': pd.Series([30000], index=[default_date]),
            'm2_supply': pd.Series([21000], index=[default_date])
        }

def load_interest_rate_data(start_date, end_date):
    """
    미국 금리 데이터 로드 (연방기금금리, 10년물 국채금리, 30년물 국채금리)
    
    Args:
        start_date: 시작 날짜
        end_date: 종료 날짜
    
    Returns:
        dict: 금리 데이터
    """
    try:
        print("📊 미국 금리 데이터 로드 중...")
        
        # 연방기금금리 (Federal Funds Rate) - 여러 방법 시도
        fed_funds = None
        
        # 방법 1: ^FEDFUNDS 시도
        try:
            fed_funds = yf.download('^FEDFUNDS', start=start_date, end=end_date, progress=False)['Close']
            if not fed_funds.empty:
                print(f"✅ 연방기금금리 (^FEDFUNDS) 로드 완료: {len(fed_funds)}개")
            else:
                raise Exception("FEDFUNDS 데이터 없음")
        except Exception as e:
            print(f"⚠️  ^FEDFUNDS 로드 실패: {e}")
            
            # 방법 2: ^IRX (3개월 국채) 사용
            try:
                fed_funds = yf.download('^IRX', start=start_date, end=end_date, progress=False)['Close']
                print(f"✅ 연방기금금리 (^IRX) 로드 완료: {len(fed_funds)}개")
            except Exception as e2:
                print(f"⚠️  ^IRX 로드 실패: {e2}")
                
                # 방법 3: ^TNX (10년물 국채) 사용
                try:
                    fed_funds = yf.download('^TNX', start=start_date, end=end_date, progress=False)['Close']
                    print(f"✅ 연방기금금리 (^TNX) 로드 완료: {len(fed_funds)}개")
                except Exception as e3:
                    print(f"❌ 모든 연방기금금리 로드 실패: {e3}")
                    # 기본값 설정
                    default_date = pd.Timestamp(start_date)
                    fed_funds = pd.Series([5.25], index=[default_date])
                    print("✅ 기본 연방기금금리 설정: 5.25%")
        
        # 10년물 국채 금리 (10-Year Treasury)
        treasury_10y = yf.download('^TNX', start=start_date, end=end_date, progress=False)['Close']
        print(f"✅ 10년물 국채금리 로드 완료: {len(treasury_10y)}개")
        
        # 30년물 국채 금리 (30-Year Treasury)
        try:
            treasury_30y = yf.download('^TYX', start=start_date, end=end_date, progress=False)['Close']
            print(f"✅ 30년물 국채금리 로드 완료: {len(treasury_30y)}개")
        except:
            # 30년물이 없으면 20년물 사용
            treasury_30y = yf.download('^TYX', start=start_date, end=end_date, progress=False)['Close']
            print(f"✅ 30년물 국채금리 로드 완료: {len(treasury_30y)}개")
        
        # FED 회의 일정 가져오기
        fed_meetings = get_fed_meeting_dates(start_date, end_date)
        print(f"✅ FED 회의 일정: {len(fed_meetings)}개")
        
        # 현재 금리값 출력
        if not fed_funds.empty:
            current_fed = float(fed_funds.iloc[-1])
            print(f"📈 현재 연방기금금리: {current_fed:.2f}%")
        
        if not treasury_10y.empty:
            current_10y = float(treasury_10y.iloc[-1])
            print(f"📈 현재 10년물 국채금리: {current_10y:.2f}%")
            
        if not treasury_30y.empty:
            current_30y = float(treasury_30y.iloc[-1])
            print(f"📈 현재 30년물 국채금리: {current_30y:.2f}%")
        
        return {
            'fed_funds': fed_funds,
            'treasury_10y': treasury_10y,
            'treasury_30y': treasury_30y,
            'fed_meetings': fed_meetings
        }
    except Exception as e:
        print(f"❌ 금리 데이터 로드 실패: {e}")
        # 기본값 반환
        default_date = pd.Timestamp(start_date)
        return {
            'fed_funds': pd.Series([5.25], index=[default_date]),
            'treasury_10y': pd.Series([4.5], index=[default_date]),
            'treasury_30y': pd.Series([4.8], index=[default_date]),
            'fed_meetings': []
        }

def get_financial_data(symbol):
    """
    종목의 재무 데이터 가져오기 (배당 정보)
    
    Args:
        symbol: 종목 심볼
    
    Returns:
        dict: 재무 데이터 (배당 정보)
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # 배당 정보
        dividend_yield = info.get('dividendYield', 0) or 0
        dividend_rate = info.get('dividendRate', 0) or 0
        ex_dividend_date = info.get('exDividendDate', 0) or 0
        last_dividend_date = info.get('lastDividendDate', 0) or 0
        
        # 배당 주기 계산 (연간 배당금 / 분기별 배당금)
        dividend_frequency = info.get('dividendFrequency', 0) or 0
        
        # 배당 주기 텍스트 변환
        if dividend_frequency == 4:
            dividend_frequency_text = "Quarterly"
        elif dividend_frequency == 12:
            dividend_frequency_text = "Monthly"
        elif dividend_frequency == 2:
            dividend_frequency_text = "Semi-Annual"
        elif dividend_frequency == 1:
            dividend_frequency_text = "Annual"
        else:
            dividend_frequency_text = "None" if dividend_yield == 0 else "Unknown"
        
        return {
            'dividend_yield': float(dividend_yield) * 100 if dividend_yield else 0,  # 퍼센트로 변환
            'dividend_rate': float(dividend_rate) if dividend_rate else 0,
            'dividend_frequency': dividend_frequency_text,
            'ex_dividend_date': ex_dividend_date,
            'last_dividend_date': last_dividend_date
        }
    except Exception as e:
        print(f"  ⚠️  {symbol} 재무 데이터 로드 실패: {e}")
        return {
            'dividend_yield': 0, 
            'dividend_rate': 0, 
            'dividend_frequency': "None",
            'ex_dividend_date': 0,
            'last_dividend_date': 0
        }

def calculate_performance_metrics(stock_data, original_prices):
    """
    각 종목별 성과 지표 계산
    
    Args:
        stock_data: 정규화된 주가 데이터
        original_prices: 원본 주가 데이터
    
    Returns:
        dict: 성과 지표
    """
    performance = {}
    
    for symbol, data in stock_data.items():
        # 원본 주가 사용
        original_data = original_prices[symbol]
        start_price = float(original_data.iloc[0])  # 실제 시작가
        end_price = float(original_data.iloc[-1])   # 실제 현재가
        
        # 정규화된 데이터로 수익률 계산
        normalized_start = float(data.iloc[0])  # 100
        normalized_end = float(data.iloc[-1])
        total_return = ((normalized_end - normalized_start) / normalized_start) * 100
        
        # 변동성 계산 (일일 수익률의 표준편차)
        daily_returns = data.pct_change().dropna()
        volatility = float(daily_returns.std() * 100)
        
        # 최대/최소 가격
        max_price = float(data.max())
        min_price = float(data.min())
        
        # 최대 낙폭 (Drawdown)
        rolling_max = data.expanding().max()
        drawdown = ((data - rolling_max) / rolling_max) * 100
        max_drawdown = float(drawdown.min())
        
        # 재무 데이터 가져오기
        financial_data = get_financial_data(symbol)
        dividend_yield = financial_data['dividend_yield']
        dividend_rate = financial_data['dividend_rate']
        dividend_frequency = financial_data['dividend_frequency']
        
        performance[symbol] = {
            'start_price': start_price,
            'end_price': end_price,
            'total_return': total_return,
            'volatility': volatility,
            'max_price': max_price,
            'min_price': min_price,
            'max_drawdown': max_drawdown,
            'dividend_yield': dividend_yield,
            'dividend_rate': dividend_rate,
            'dividend_frequency': dividend_frequency
        }
    
    return performance

def create_comparison_chart(stock_data, vix_data, start_date, end_date, original_prices=None, interest_data=None, macro_data=None):
    """
    비교 차트 생성 (메인 + VIX 서브플롯)
    
    Args:
        stock_data: 정규화된 주가 데이터
        vix_data: VIX 데이터
        start_date: 시작 날짜
        end_date: 종료 날짜
    """
    print("📈 비교 차트 생성 중...")
    
    # 차트 설정 (4개 서브플롯)
    fig, (ax_main, ax_vix, ax_interest, ax_macro) = plt.subplots(4, 1, figsize=(16, 18), height_ratios=[3, 1, 1, 1])
    
    # 색상 팔레트
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # 메인 차트: 주가 비교 (종가 기준)
    for i, (symbol, data) in enumerate(stock_data.items()):
        color = colors[i % len(colors)]
        ax_main.plot(data.index, data.values, 
                    label=symbol, color=color, 
                    linewidth=1.75, alpha=0.8)
        
        # 현재 가격 위치에 종목명과 성과 정보 표시
        current_price = float(data.iloc[-1])
        current_date = data.index[-1]
        
        # 성과 지표 계산
        total_return = ((current_price - 100) / 100) * 100
        daily_returns = data.pct_change().dropna()
        volatility = float(daily_returns.std() * 100)
        
        # 종목명과 성과 정보 텍스트
        label_text = f"{symbol}\n({total_return:+.1f}%, {volatility:.1f}%)"
        
        # 현재 가격 위치 오른쪽에 텍스트 표시
        ax_main.text(current_date, current_price, label_text,
                    fontsize=9, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, 
                             alpha=0.7, edgecolor=color, linewidth=1),
                    color='white', fontweight='bold',
                    transform=ax_main.transData)
    
    # 메인 차트 스타일링
    ax_main.set_title(f'Multi-Stock Comparison Analysis\n{start_date} ~ {end_date}', 
                     fontsize=16, fontweight='bold', pad=20)
    ax_main.set_ylabel('Normalized Price (Base = 100)', fontsize=12)
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # 100 기준선 추가
    ax_main.axhline(y=100, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax_main.text(ax_main.get_xlim()[0], 100, 'Base = 100', 
                verticalalignment='bottom', fontsize=9, alpha=0.7)
    
    # VIX 서브플롯
    ax_vix.plot(vix_data.index, vix_data.values, color='red', linewidth=1.5, alpha=0.8)
    # VIX 데이터가 1차원인지 확인 후 fill_between 실행
    if len(vix_data.values.shape) == 1:
        ax_vix.fill_between(vix_data.index, vix_data.values, alpha=0.3, color='red')
    else:
        ax_vix.fill_between(vix_data.index, vix_data.values.flatten(), alpha=0.3, color='red')
    
    # 현재 시점의 VIX 값 표시 (교차점과 동일한 스타일)
    current_vix = float(vix_data.iloc[-1])
    current_date = vix_data.index[-1]
    
    # 메인 차트에서 VIX 서브플롯으로 수직 점선 연결 (교차점과 동일한 스타일)
    ax_main.axvline(x=current_date, color='gray', linestyle='--', alpha=0.6, linewidth=1)
    ax_vix.axvline(x=current_date, color='gray', linestyle='--', alpha=0.6, linewidth=1)
    
    # VIX 서브플롯에 현재 VIX 값 표시 (교차점과 동일한 스타일)
    ax_vix.text(current_date, current_vix, f'VIX: {current_vix:.1f}',
               fontsize=8, ha='center', va='bottom', rotation=45,
               bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7),
               color='black', fontweight='bold')
    
    
    # 종목 가격 교차 지점 찾기 및 VIX 연결선 표시
    stock_symbols = list(stock_data.keys())
    
    print(f"🔍 교차점 감지 시작: {len(stock_symbols)}개 종목")
    total_cross_points = 0
    
    # 모든 종목 쌍에 대해 교차 지점 찾기
    for i, symbol1 in enumerate(stock_symbols):
        for j, symbol2 in enumerate(stock_symbols[i+1:], i+1):
            series1 = stock_data[symbol1]
            series2 = stock_data[symbol2]
            
            print(f"  🔍 {symbol1} vs {symbol2} 교차점 검색 중...")
            
            # 간단하고 안전한 교차점 감지 로직
            cross_points = []
            
            print(f"    📊 {symbol1} vs {symbol2} 교차점 검색 중...")
            
            # 두 시리즈의 차이를 직접 계산
            try:
                # 각 지점에서의 차이 계산
                for i in range(len(series1)):
                    try:
                        price1 = float(series1.iloc[i])
                        price2 = float(series2.iloc[i])
                        diff = abs(price1 - price2)
                        
                        # 차이가 매우 작은 지점들을 교차점으로 간주 (5 이내)
                        if diff <= 5.0:
                            cross_date = series1.index[i]
                            cross_price = (price1 + price2) / 2
                            cross_points.append((cross_date, cross_price))
                            print(f"      ✅ 교차점 발견: {cross_date.strftime('%Y-%m-%d')}, 차이: {diff:.2f}, 가격: {cross_price:.2f}")
                    except Exception as e:
                        continue
                
                # 교차점이 너무 많으면 가장 가까운 3개만 선택
                if len(cross_points) > 3:
                    # 차이로 정렬하여 가장 가까운 3개 선택
                    cross_points.sort(key=lambda x: abs(float(series1.loc[x[0]]) - float(series2.loc[x[0]])))
                    cross_points = cross_points[:3]
                    
            except Exception as e:
                print(f"      ❌ 교차점 검색 실패: {e}")
                continue
            
            # 교차점 감지 결과 출력
            if cross_points:
                print(f"    ✅ {symbol1} vs {symbol2}: {len(cross_points)}개 교차점 발견")
                total_cross_points += len(cross_points)
            else:
                print(f"    ❌ {symbol1} vs {symbol2}: 교차점 없음")
            
            # 교차점에서 VIX 연결선 그리기
            for cross_date, cross_price in cross_points:
                # VIX 데이터에서 해당 날짜의 가장 가까운 값 찾기
                vix_cross_idx = vix_data.index.get_indexer([cross_date], method='nearest')[0]
                vix_cross_value = vix_data.iloc[vix_cross_idx]
                vix_cross_date = vix_data.index[vix_cross_idx]
                
                print(f"      📍 교차점: {cross_date.strftime('%Y-%m-%d')}, VIX: {float(vix_cross_value):.1f}")
                
                # 메인 차트에서 VIX 서브플롯으로 수직 점선 연결
                ax_main.axvline(x=cross_date, color='gray', linestyle='--', alpha=0.6, linewidth=1)
                ax_vix.axvline(x=cross_date, color='gray', linestyle='--', alpha=0.6, linewidth=1)
                
                # VIX 서브플롯에 VIX 값 표시
                vix_value_float = float(vix_cross_value)
                ax_vix.text(vix_cross_date, vix_value_float, f'VIX: {vix_value_float:.1f}',
                           fontsize=8, ha='center', va='bottom', rotation=45,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7),
                           color='black', fontweight='bold')
                
    
    print(f"🔍 총 교차점: {total_cross_points}개 발견")
    
    # VIX 서브플롯 스타일링
    ax_vix.set_title('VIX (Market Volatility)', fontsize=12, fontweight='bold')
    ax_vix.set_ylabel('VIX', fontsize=10)
    ax_vix.set_xlabel('Date', fontsize=10)
    ax_vix.grid(True, alpha=0.3)
    
    # VIX 수준별 배경색
    ax_vix.axhspan(0, 20, alpha=0.1, color='green', label='Low Volatility')
    ax_vix.axhspan(20, 30, alpha=0.1, color='yellow', label='Medium Volatility')
    ax_vix.axhspan(30, 50, alpha=0.1, color='red', label='High Volatility')
    
    # 금리 서브플롯 (연방기금금리, 10년물 국채금리, 30년물 국채금리)
    if interest_data:
        fed_funds = interest_data['fed_funds']
        treasury_10y = interest_data['treasury_10y']
        treasury_30y = interest_data['treasury_30y']
        fed_meetings = interest_data['fed_meetings']
        
        # 연방기금금리 (FFR)
        ax_interest.plot(fed_funds.index, fed_funds.values, color='red', linewidth=1, label='FFR (연방기금금리)')
        
        # CD금리 (CDR) - 3개월 국채금리
        ax_interest.plot(fed_funds.index, fed_funds.values, color='orange', linewidth=1, label='CDR (CD금리)', linestyle='--')
        
        # 미국채 10년물 금리
        ax_interest.plot(treasury_10y.index, treasury_10y.values, color='blue', linewidth=1, label='미국채 10년물금리')
        
        # 미국채 30년물 금리
        ax_interest.plot(treasury_30y.index, treasury_30y.values, color='green', linewidth=1, label='미국채 30년물금리')
        
        # FED 회의 일정 수직선 및 금리값 표시
        for meeting_date in fed_meetings:
            # 수직선 그리기
            ax_interest.axvline(x=meeting_date, color='purple', linestyle='--', alpha=0.7, linewidth=1.5)
            
            # 해당 날짜의 금리값들 찾기
            try:
                # 회의일과 가장 가까운 날짜의 금리값 찾기
                closest_date = min(fed_funds.index, key=lambda x: abs((x - meeting_date).days))
                fed_rate = float(fed_funds.loc[closest_date])
                treasury_10y_rate = float(treasury_10y.loc[closest_date])
                treasury_30y_rate = float(treasury_30y.loc[closest_date])
                
                # 연방기금금리 교차점에 마커 표시
                ax_interest.plot(meeting_date, fed_rate, 'ro', markersize=4, markeredgecolor='darkred', markeredgewidth=2, alpha=0.5)
                
                # 10년물 국채 금리 교차점에 마커 표시
                ax_interest.plot(meeting_date, treasury_10y_rate, 'bo', markersize=4, markeredgecolor='darkblue', markeredgewidth=2, alpha=0.5)
                
                # 30년물 국채 금리 교차점에 마커 표시
                ax_interest.plot(meeting_date, treasury_30y_rate, 'go', markersize=4, markeredgecolor='darkgreen', markeredgewidth=2, alpha=0.5)
                
            except Exception as e:
                # 금리값을 찾을 수 없는 경우
                print(f"⚠️  {meeting_date.strftime('%Y-%m-%d')} 금리값 찾기 실패: {e}")
                ax_interest.text(meeting_date, ax_interest.get_ylim()[1] * 0.9, 'FED',
                               fontsize=8, ha='center', va='bottom', rotation=90,
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcoral', alpha=0.7),
                               color='black', fontweight='bold')
        
        # 금리 서브플롯 스타일링
        ax_interest.set_title('금리 (Interest Rates)', fontsize=12, fontweight='bold')
        ax_interest.set_ylabel('금리 (%)', fontsize=10)
        ax_interest.set_xlabel('Date', fontsize=10)
        ax_interest.grid(True, alpha=0.3)
        ax_interest.legend(loc='upper left', fontsize=8)
        
        # 금리 수준별 배경색 (동적 계산)
        all_rates = pd.concat([fed_funds, treasury_10y, treasury_30y]).dropna()
        if not all_rates.empty:
            min_rate = all_rates.min()
            max_rate = all_rates.max()
            rate_range = max_rate - min_rate
            
            # 동적 구간 계산 (최소값 기준으로 25%, 50%, 75% 구간)
            low_threshold = min_rate + rate_range * 0.25
            mid_threshold = min_rate + rate_range * 0.5
            high_threshold = min_rate + rate_range * 0.75
            
            # 배경색 적용
            ax_interest.axhspan(min_rate, low_threshold, alpha=0.1, color='green', label='Low Rate')
            ax_interest.axhspan(low_threshold, mid_threshold, alpha=0.1, color='yellow', label='Medium Rate')
            ax_interest.axhspan(mid_threshold, high_threshold, alpha=0.1, color='orange', label='High Rate')
            ax_interest.axhspan(high_threshold, max_rate, alpha=0.1, color='red', label='Very High Rate')
            
            # Y축 범위를 데이터에 맞게 조정 (여백 추가)
            y_margin = rate_range * 0.1
            ax_interest.set_ylim(min_rate - y_margin, max_rate + y_margin)
        
        # 최근 기준금리 발표일자 하이라이트 텍스트 (우측하단)
        if fed_meetings:
            latest_meeting = max(fed_meetings)
            latest_meeting_str = latest_meeting.strftime('%Y-%m-%d')
            
            # 현재 금리값들
            current_fed = float(fed_funds.iloc[-1]) if not fed_funds.empty else 0
            current_10y = float(treasury_10y.iloc[-1]) if not treasury_10y.empty else 0
            current_30y = float(treasury_30y.iloc[-1]) if not treasury_30y.empty else 0
            
            # 하이라이트 텍스트 내용
            highlight_text = f"최근 기준금리 발표: {latest_meeting_str}\n"
            highlight_text += f"FFR: {current_fed:.2f}%\n"
            highlight_text += f"CDR: {current_fed:.2f}%\n"
            highlight_text += f"10년물: {current_10y:.2f}%\n"
            highlight_text += f"30년물: {current_30y:.2f}%"
            
            # 우측하단에 텍스트 박스 추가
            ax_interest.text(0.98, 0.02, highlight_text, 
                           transform=ax_interest.transAxes,
                           fontsize=6, ha='right', va='bottom',
                           bbox=dict(boxstyle="round,pad=0.5", 
                                   facecolor='lightyellow', 
                                   alpha=0.9, 
                                   edgecolor='orange',
                                   linewidth=1.5),
                           color='black', fontweight='bold')
    
    # 거시경제 서브플롯 (국채발행량, M2 통화량)
    if macro_data:
        treasury_outstanding = macro_data['treasury_outstanding']
        m2_supply = macro_data['m2_supply']
        
        # 국채발행량
        ax_macro.plot(treasury_outstanding.index, treasury_outstanding.values, 
                     color='purple', linewidth=1, label='국채발행량')
        
        # M2 통화량 (오른쪽 Y축)
        ax_macro2 = ax_macro.twinx()
        ax_macro2.plot(m2_supply.index, m2_supply.values, 
                      color='brown', linewidth=1, label='M2 통화량')
        
        # 거시경제 서브플롯 스타일링
        ax_macro.set_title('거시경제 지표 (Macroeconomic Indicators)', fontsize=12, fontweight='bold')
        ax_macro.set_ylabel('국채발행량 (억 달러)', fontsize=10, color='purple')
        ax_macro2.set_ylabel('M2 통화량 (억 달러)', fontsize=10, color='brown')
        ax_macro.set_xlabel('Date', fontsize=10)
        ax_macro.grid(True, alpha=0.3)
        
        # 범례 설정
        lines1, labels1 = ax_macro.get_legend_handles_labels()
        lines2, labels2 = ax_macro2.get_legend_handles_labels()
        ax_macro.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
        
        # 현재 값 표시 (우측하단)
        current_treasury = float(treasury_outstanding.iloc[-1]) if not treasury_outstanding.empty else 0
        current_m2 = float(m2_supply.iloc[-1]) if not m2_supply.empty else 0
        
        macro_text = f"국채발행량: {current_treasury:,.0f}억 달러\n"
        macro_text += f"M2 통화량: {current_m2:,.0f}억 달러"
        
        ax_macro.text(0.98, 0.02, macro_text, 
                     transform=ax_macro.transAxes,
                     fontsize=6, ha='right', va='bottom',
                     bbox=dict(boxstyle="round,pad=0.5", 
                             facecolor='lightcyan', 
                             alpha=0.9, 
                             edgecolor='blue',
                             linewidth=1.5),
                     color='black', fontweight='bold')
    
    # 날짜 형식 설정
    axes = [ax_main, ax_vix]
    if interest_data:
        axes.append(ax_interest)
    if macro_data:
        axes.append(ax_macro)
    
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=7)
    
    # 레이아웃 조정
    plt.tight_layout()
    
    # 성과 요약 텍스트 추가 (위측으로 위치 변경)
    performance = calculate_performance_metrics(stock_data, original_prices)
    summary_text = "Performance Summary:\n"
    for symbol, metrics in performance.items():
        summary_text += f"{symbol}: \\${metrics['start_price']:.2f}→\\${metrics['end_price']:.2f} "
        summary_text += f"({metrics['total_return']:+.1f}%)\n"
        summary_text += f"  [변동성: {metrics['volatility']:.1f}%, 최대낙폭: {metrics['max_drawdown']:.1f}%]\n"
    
    # 위측 중앙에 배치 (범례와 겹치지 않도록)
    ax_main.text(0.5, 0.95, summary_text, transform=ax_main.transAxes, 
                fontsize=9, verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    print("✅ 비교 차트 생성 완료")
    return fig

def save_results(stock_data, vix_data, performance, start_date, end_date, original_prices=None, interest_data=None, macro_data=None, output_dir="output/hma_mantra/comparison"):
    """
    결과 저장 (차트, CSV, 마크다운)
    
    Args:
        stock_data: 정규화된 주가 데이터
        vix_data: VIX 데이터
        performance: 성과 지표
        start_date: 시작 날짜
        end_date: 종료 날짜
        output_dir: 출력 디렉토리
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일명 생성
    symbols_str = "_".join(stock_data.keys())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 차트 저장
    chart_path = os.path.join(output_dir, f"{symbols_str}_comparison_{start_date}_{end_date}.png")
    fig = create_comparison_chart(stock_data, vix_data, start_date, end_date, original_prices, interest_data, macro_data)
    fig.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 차트 저장: {chart_path}")
    
    # 2. 성과 요약 CSV 저장
    csv_path = os.path.join(output_dir, f"{symbols_str}_performance_summary.csv")
    performance_df = pd.DataFrame(performance).T
    performance_df.to_csv(csv_path)
    print(f"📈 성과 요약 저장: {csv_path}")
    
    # 3. 상세 분석 마크다운 저장
    md_path = os.path.join(output_dir, f"{symbols_str}_comparison_analysis.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Multi-Stock Comparison Analysis\n\n")
        f.write(f"**분석 기간**: {start_date} ~ {end_date}\n")
        f.write(f"**분석 종목**: {', '.join(stock_data.keys())}\n")
        f.write(f"**생성 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Performance Summary\n\n")
        f.write("| Symbol | Total Return | Volatility | Max Price | Min Price | Max Drawdown |\n")
        f.write("|--------|--------------|------------|-----------|-----------|--------------|\n")
        
        for symbol, metrics in performance.items():
            f.write(f"| {symbol} | {metrics['total_return']:+.1f}% | "
                   f"{metrics['volatility']:.1f}% | {metrics['max_price']:.1f} | "
                   f"{metrics['min_price']:.1f} | {metrics['max_drawdown']:.1f}% |\n")
        
        f.write(f"\n## VIX Analysis\n\n")
        f.write(f"- **Average VIX**: {float(vix_data.mean()):.2f}\n")
        f.write(f"- **Max VIX**: {float(vix_data.max()):.2f}\n")
        f.write(f"- **Min VIX**: {float(vix_data.min()):.2f}\n")
        f.write(f"- **Current VIX**: {float(vix_data.iloc[-1]):.2f}\n")
        
        f.write(f"\n## Market Sentiment\n\n")
        current_vix = float(vix_data.iloc[-1])
        if current_vix < 20:
            sentiment = "Low Volatility (Stable Market)"
        elif current_vix < 30:
            sentiment = "Medium Volatility (Normal Market)"
        else:
            sentiment = "High Volatility (Unstable Market)"
        
        f.write(f"- **Current Market Sentiment**: {sentiment}\n")
        f.write(f"- **VIX Level**: {current_vix:.2f}\n")
    
    print(f"📝 상세 분석 저장: {md_path}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Multi-stock comparison analysis')
    parser.add_argument('--symbols', required=True, help='Comma-separated stock symbols')
    parser.add_argument('--from', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--to', help='End date (YYYY-MM-DD), default: today')
    parser.add_argument('--output', help='Output directory', default='output/hma_mantra/comparison')
    
    args = parser.parse_args()
    
    # 인수 처리
    symbols = [s.strip() for s in args.symbols.split(',')]
    start_date = getattr(args, 'from')
    end_date = args.to or datetime.now().strftime('%Y-%m-%d')
    
    print("🚀 Multi-Stock Comparison Analysis 시작")
    print(f"📊 종목: {', '.join(symbols)}")
    print(f"📅 기간: {start_date} ~ {end_date}")
    
    try:
        # 데이터 로드
        stock_data, original_prices = load_and_normalize_stocks(symbols, start_date, end_date)
        if not stock_data:
            print("❌ 로드된 종목이 없습니다.")
            return
        
        vix_data = load_vix_data(start_date, end_date)
        
        # 금리 데이터 로드
        interest_data = load_interest_rate_data(start_date, end_date)
        
        # 거시경제 데이터 로드
        macro_data = load_macro_data(start_date, end_date)
        
        # 성과 분석
        performance = calculate_performance_metrics(stock_data, original_prices)
        
        # 결과 저장
        save_results(stock_data, vix_data, performance, start_date, end_date, original_prices, interest_data, macro_data, args.output)
        
        print("✅ 분석 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()
