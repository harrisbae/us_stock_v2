"""
Signal generation module
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Tuple
from .core import calculate_hma, calculate_mantra_bands, calculate_rsi, calculate_macd
from .utils import to_float

def get_hma_signals(data: pd.DataFrame) -> List[Dict]:
    """HMA 신호 생성"""
    signals = []
    
    if isinstance(data.columns, pd.MultiIndex):
        close_data = data[('Close', data.columns.levels[1][0])]
    else:
        close_data = data['Close']
    
    hma = calculate_hma(close_data)
    
    for i in range(1, len(data)):
        close = close_data.iloc[i]
        prev_close = close_data.iloc[i-1]
        hma_now = hma.iloc[i]
        hma_prev = hma.iloc[i-1]
        
        # 매수 신호: 종가가 HMA를 상향 돌파
        if prev_close < hma_prev and close >= hma_now:
            signals.append({
                'date': data.index[i],
                'type': 'BUY',
                'price': close,
                'reason': 'HMA 상향돌파'
            })
        
        # 매도 신호: 종가가 HMA를 하향 돌파
        elif prev_close > hma_prev and close <= hma_now:
            signals.append({
                'date': data.index[i],
                'type': 'SELL',
                'price': close,
                'reason': 'HMA 하향돌파'
            })
    
    return signals

def get_mantra_signals(data: pd.DataFrame) -> List[Dict]:
    """만트라 밴드 신호 생성"""
    signals = []
    
    if isinstance(data.columns, pd.MultiIndex):
        close_data = data[('Close', data.columns.levels[1][0])]
    else:
        close_data = data['Close']
    
    upper_band, lower_band = calculate_mantra_bands(close_data)
    
    for i in range(1, len(data)):
        close = close_data.iloc[i]
        prev_close = close_data.iloc[i-1]
        upper = upper_band.iloc[i]
        lower = lower_band.iloc[i]
        
        # 매수 신호: 종가가 하단 밴드를 상향 돌파
        if prev_close < lower and close >= lower:
            signals.append({
                'date': data.index[i],
                'type': 'BUY',
                'price': close,
                'reason': '하단밴드 상향돌파'
            })
        
        # 매도 신호: 종가가 상단 밴드를 하향 돌파
        elif prev_close > upper and close <= upper:
            signals.append({
                'date': data.index[i],
                'type': 'SELL',
                'price': close,
                'reason': '상단밴드 하향돌파'
            })
    
    return signals

def get_hma_mantra_md_signals(data: pd.DataFrame, ticker: str = None) -> List[Dict]:
    """HMA + 만트라 밴드 매수/매도 신호를 생성합니다."""
    if isinstance(data.columns, pd.MultiIndex):
        ohlcv_data = data.xs(ticker, axis=1, level=1)
    else:
        ohlcv_data = data

    # 기술적 지표 계산
    hma = calculate_hma(ohlcv_data['Close'])
    upper_band, lower_band = calculate_mantra_bands(ohlcv_data['Close'])
    rsi3 = calculate_rsi(ohlcv_data['Close'], period=3)
    rsi14 = calculate_rsi(ohlcv_data['Close'], period=14)
    macd, signal, _ = calculate_macd(ohlcv_data['Close'])

    signals = []
    last_signal_type = None
    
    for i in range(1, len(ohlcv_data)):
        current_price = ohlcv_data['Close'].iloc[i]
        current_date = ohlcv_data.index[i]
        prev_price = ohlcv_data['Close'].iloc[i-1]
        
        # RSI 매수/매도 구간 확인
        is_rsi_buy = rsi3.iloc[i] >= rsi14.iloc[i]
        is_rsi_sell = rsi3.iloc[i] < rsi14.iloc[i]
        
        # MACD 상태 확인
        macd_state = macd.iloc[i] - signal.iloc[i]
        
        # 매수 조건 확인
        if last_signal_type != 'BUY':
            # B1: HMA 상향돌파 + RSI 매수
            if prev_price < hma.iloc[i-1] and current_price >= hma.iloc[i] and is_rsi_buy:
                signals.append({
                    'date': current_date,
                    'price': current_price,
                    'type': 'BUY',
                    'reason': 'HMA 상향돌파 + RSI 매수',
                    'macd_state': macd_state
                })
                last_signal_type = 'BUY'
            
            # B2: 밴드 하단 상향돌파 + RSI 매수
            elif prev_price < lower_band.iloc[i-1] and current_price >= lower_band.iloc[i] and is_rsi_buy:
                signals.append({
                    'date': current_date,
                    'price': current_price,
                    'type': 'BUY',
                    'reason': '하단밴드 상향돌파 + RSI 매수',
                    'macd_state': macd_state
                })
                last_signal_type = 'BUY'
            
            # B3: HMA 상향돌파 + RSI 비매수 + MACD 매수
            elif prev_price < hma.iloc[i-1] and current_price >= hma.iloc[i] and not is_rsi_buy and macd_state > 0:
                signals.append({
                    'date': current_date,
                    'price': current_price,
                    'type': 'BUY',
                    'reason': 'HMA 상향돌파 + MACD 매수',
                    'macd_state': macd_state
                })
                last_signal_type = 'BUY'
        
        # 매도 조건 확인
        elif last_signal_type != 'SELL':
            # T1: HMA 하향돌파 + RSI 매도
            if prev_price > hma.iloc[i-1] and current_price <= hma.iloc[i] and is_rsi_sell:
                signals.append({
                    'date': current_date,
                    'price': current_price,
                    'type': 'SELL',
                    'reason': 'HMA 하향돌파 + RSI 매도',
                    'macd_state': macd_state
                })
                last_signal_type = 'SELL'
            
            # T2: 밴드 상단 하향돌파 + RSI 매도
            elif prev_price > upper_band.iloc[i-1] and current_price <= upper_band.iloc[i] and is_rsi_sell:
                signals.append({
                    'date': current_date,
                    'price': current_price,
                    'type': 'SELL',
                    'reason': '상단밴드 하향돌파 + RSI 매도',
                    'macd_state': macd_state
                })
                last_signal_type = 'SELL'
    
    return signals 