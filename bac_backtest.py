#!/usr/bin/env python3
"""
BAC 백테스트 스크립트
매수 신호 시 1000달러씩 종가 기준 매수 vs 익일 종가 매수 비교
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.indicators.hma_mantra_example import calculate_combined_signals

def get_buy_signals(data):
    """매수 신호 날짜들을 반환"""
    buy_signals = []
    
    for i in range(len(data)):
        if i < 50:  # 충분한 데이터가 없으면 건너뛰기
            continue
            
        # 현재까지의 데이터로 신호 계산
        current_data = data.iloc[:i+1]
        signal = calculate_combined_signals(current_data)
        
        if signal in ['BUY', 'BUY_STRONG']:
            buy_signals.append(data.index[i])
    
    return buy_signals

def calculate_returns(data, buy_signals, strategy_type='same_day'):
    """
    백테스트 수익률 계산
    
    Args:
        data: 주가 데이터
        buy_signals: 매수 신호 날짜들
        strategy_type: 'same_day' (같은 날 종가 매수) 또는 'next_day' (익일 종가 매수)
    """
    portfolio_value = 1000  # 초기 포트폴리오 가치
    shares_owned = 0
    total_invested = 0
    trades = []
    
    for signal_date in buy_signals:
        if strategy_type == 'same_day':
            buy_date = signal_date
        else:  # next_day
            # 익일 날짜 찾기
            next_day_idx = data.index.get_loc(signal_date) + 1
            if next_day_idx >= len(data):
                continue
            buy_date = data.index[next_day_idx]
        
        # 매수 가격 (종가)
        buy_price = data.loc[buy_date, 'Close']
        
        # 1000달러로 매수할 수 있는 주식 수
        shares_to_buy = int(1000 / buy_price)
        if shares_to_buy > 0:
            shares_owned += shares_to_buy
            total_invested += shares_to_buy * buy_price
            
            trades.append({
                'signal_date': signal_date,
                'buy_date': buy_date,
                'buy_price': buy_price,
                'shares': shares_to_buy,
                'invested': shares_to_buy * buy_price
            })
    
    return shares_owned, total_invested, trades

def calculate_period_returns(data, shares_owned, total_invested, periods):
    """각 기간별 수익률 계산"""
    results = {}
    
    for period_name, days in periods.items():
        if days == 0:  # 현재까지
            end_date = data.index[-1]
        else:
            end_date = data.index[-1] - timedelta(days=days)
        
        # 해당 기간의 마지막 거래일 찾기
        valid_dates = data.index[data.index <= end_date]
        if len(valid_dates) == 0:
            results[period_name] = {'return_pct': 0, 'total_value': total_invested}
            continue
            
        last_date = valid_dates[-1]
        last_price = data.loc[last_date, 'Close']
        
        # 포트폴리오 가치
        portfolio_value = shares_owned * last_price
        
        # 수익률 계산
        if total_invested > 0:
            return_pct = ((portfolio_value - total_invested) / total_invested) * 100
        else:
            return_pct = 0
        
        results[period_name] = {
            'return_pct': return_pct,
            'total_value': portfolio_value,
            'last_date': last_date,
            'last_price': last_price
        }
    
    return results

def main():
    print("BAC 백테스트 시작...")
    
    # BAC 데이터 다운로드 (최근 2년)
    ticker = "BAC"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2년
    
    print(f"데이터 다운로드 중: {ticker} ({start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')})")
    
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if data.empty:
        print("데이터를 가져올 수 없습니다.")
        return
    
    print(f"다운로드 완료: {len(data)} 개 데이터")
    print(f"데이터 기간: {data.index[0].strftime('%Y-%m-%d')} ~ {data.index[-1].strftime('%Y-%m-%d')}")
    
    # 매수 신호 찾기
    print("매수 신호 분석 중...")
    buy_signals = get_buy_signals(data)
    print(f"총 매수 신호: {len(buy_signals)} 개")
    
    if len(buy_signals) == 0:
        print("매수 신호가 없습니다.")
        return
    
    # 전략별 백테스트
    strategies = {
        'same_day': '같은 날 종가 매수',
        'next_day': '익일 종가 매수'
    }
    
    periods = {
        '3개월': 90,
        '6개월': 180,
        '1년': 365,
        '현재까지': 0
    }
    
    results = {}
    
    for strategy_type, strategy_name in strategies.items():
        print(f"\n=== {strategy_name} 전략 ===")
        
        # 수익률 계산
        shares_owned, total_invested, trades = calculate_returns(data, buy_signals, strategy_type)
        
        print(f"총 매수 횟수: {len(trades)}")
        print(f"총 투자 금액: ${total_invested:,.2f}")
        print(f"보유 주식 수: {shares_owned:,} 주")
        
        # 기간별 수익률 계산
        period_returns = calculate_period_returns(data, shares_owned, total_invested, periods)
        
        results[strategy_type] = {
            'trades': trades,
            'shares_owned': shares_owned,
            'total_invested': total_invested,
            'period_returns': period_returns
        }
        
        print("\n기간별 수익률:")
        for period_name, period_data in period_returns.items():
            print(f"  {period_name}: {period_data['return_pct']:+.2f}% (${period_data['total_value']:,.2f})")
    
    # 결과 비교
    print("\n" + "="*60)
    print("전략별 수익률 비교")
    print("="*60)
    
    comparison_df = pd.DataFrame()
    
    for strategy_type, strategy_name in strategies.items():
        period_returns = results[strategy_type]['period_returns']
        returns_data = {}
        
        for period_name, period_data in period_returns.items():
            returns_data[f"{period_name}_수익률"] = period_data['return_pct']
            returns_data[f"{period_name}_포트폴리오가치"] = period_data['total_value']
        
        comparison_df[strategy_name] = pd.Series(returns_data)
    
    print(comparison_df.round(2))
    
    # 최적 전략 찾기
    print("\n" + "="*60)
    print("최적 전략 분석")
    print("="*60)
    
    for period_name in periods.keys():
        same_day_return = results['same_day']['period_returns'][period_name]['return_pct']
        next_day_return = results['next_day']['period_returns'][period_name]['return_pct']
        
        if same_day_return > next_day_return:
            better_strategy = "같은 날 종가 매수"
            better_return = same_day_return
        else:
            better_strategy = "익일 종가 매수"
            better_return = next_day_return
        
        print(f"{period_name}: {better_strategy} ({better_return:+.2f}%)")
    
    # 상세 거래 내역
    print("\n" + "="*60)
    print("상세 거래 내역 (최근 10건)")
    print("="*60)
    
    for strategy_type, strategy_name in strategies.items():
        print(f"\n{strategy_name}:")
        trades = results[strategy_type]['trades']
        
        if len(trades) > 0:
            recent_trades = trades[-10:]  # 최근 10건
            for trade in recent_trades:
                print(f"  신호: {trade['signal_date'].strftime('%Y-%m-%d')} -> "
                      f"매수: {trade['buy_date'].strftime('%Y-%m-%d')} "
                      f"@ ${trade['buy_price']:.2f} "
                      f"({trade['shares']}주, ${trade['invested']:.2f})")

if __name__ == "__main__":
    main()
