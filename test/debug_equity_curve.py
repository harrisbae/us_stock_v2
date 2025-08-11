#!/usr/bin/env python3
"""
calculate_equity_curve 함수 디버깅 테스트
"""

import pandas as pd
import numpy as np

def test_equity_curve_logic():
    """calculate_equity_curve 함수의 수익률 처리 로직 테스트"""
    
    # 테스트 데이터
    test_results = [
        {
            '매수 신호 발생일': '2023-08-28',
            '매수 가격': 6.85,
            '매수 금액': 1000,
            '1개월 수익률(%)': -9.2,
            '3개월 수익률(%)': 20.0,
            '6개월 수익률(%)': 38.98,
            '현재까지 수익률(%)': 658.25
        },
        {
            '매수 신호 발생일': '2023-09-26',
            '매수 가격': 6.16,
            '매수 금액': 1000,
            '1개월 수익률(%)': -1.3,
            '3개월 수익률(%)': 39.45,
            '6개월 수익률(%)': 165.1,
            '현재까지 수익률(%)': 743.18
        }
    ]

    # 포지션 생성 테스트
    positions = []
    for r in test_results:
        buy_date = pd.to_datetime(r['매수 신호 발생일']).date()
        buy_price = r['매수 가격']
        buy_amount = r['매수 금액']
        shares = buy_amount / buy_price
        
        returns_1m = r.get('1개월 수익률(%)', 0)
        returns_3m = r.get('3개월 수익률(%)', 0)
        returns_6m = r.get('6개월 수익률(%)', 0)
        returns_now = r.get('현재까지 수익률(%)', 0)
        
        positions.append({
            'buy_date': buy_date,
            'shares': shares,
            'buy_price': buy_price,
            'buy_amount': buy_amount,
            'returns_1m': returns_1m,
            'returns_3m': returns_3m,
            'returns_6m': returns_6m,
            'returns_now': returns_now
        })

    print('포지션 생성 결과:')
    for i, pos in enumerate(positions):
        print(f'포지션 {i+1}: {pos}')

    # 일별 포트폴리오 가치 계산 테스트
    start_date = min(pos['buy_date'] for pos in positions)
    end_date = pd.Timestamp.now().date()
    date_range = pd.date_range(start_date, end_date, freq='D')

    print(f'\n날짜 범위: {start_date} ~ {end_date}')
    print(f'총 일수: {len(date_range)}')

    # 첫 번째 날짜의 포트폴리오 가치 계산
    first_date = date_range[0].date()
    total_value = 0

    for position in positions:
        if first_date >= position['buy_date']:
            days_since_buy = (first_date - position['buy_date']).days
            print(f'포지션 {position["buy_date"]}: 경과일수 {days_since_buy}일')
            
            # 기간별 수익률 적용
            if days_since_buy <= 30:
                current_price = position['buy_price'] * (1 + position['returns_1m'] / 100)
                print(f'  1개월 수익률 적용: ${position["buy_price"]:.2f} -> ${current_price:.2f}')
            elif days_since_buy <= 90:
                current_price = position['buy_price'] * (1 + position['returns_3m'] / 100)
                print(f'  3개월 수익률 적용: ${position["buy_price"]:.2f} -> ${current_price:.2f}')
            elif days_since_buy <= 180:
                current_price = position['buy_price'] * (1 + position['returns_6m'] / 100)
                print(f'  6개월 수익률 적용: ${position["buy_price"]:.2f} -> ${current_price:.2f}')
            else:
                current_price = position['buy_price'] * (1 + position['returns_now'] / 100)
                print(f'  현재 수익률 적용: ${position["buy_price"]:.2f} -> ${current_price:.2f}')
            
            total_value += position['shares'] * current_price

    print(f'\n첫 번째 날짜 포트폴리오 가치: ${total_value:.2f}')
    
    # MDD 계산 테스트
    values = [total_value, total_value * 1.1, total_value * 0.95, total_value * 1.2]
    running_max = pd.Series(values).expanding().max()
    drawdown = (pd.Series(values) - running_max) / running_max * 100
    mdd = drawdown.min()
    
    print(f'\nMDD 계산 테스트:')
    print(f'포트폴리오 가치: {values}')
    print(f'누적 최고점: {running_max.values}')
    print(f'Drawdown: {drawdown.values}')
    print(f'MDD: {mdd:.2f}%')

if __name__ == "__main__":
    test_equity_curve_logic()
