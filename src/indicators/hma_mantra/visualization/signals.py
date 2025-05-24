"""
신호 시각화 모듈
"""

import mplfinance as mpf
import pandas as pd
import numpy as np
from ..core import calculate_hma, calculate_mantra_bands
from ..signals import get_hma_signals, get_mantra_signals
from ..utils import get_available_font

def plot_signals_with_strength(data: pd.DataFrame, ticker: str = None, save_path: str = None):
    """HMA + 만트라 밴드 신호 강도 차트"""
    if isinstance(data.columns, pd.MultiIndex):
        ohlcv_data = data.xs(ticker, axis=1, level=1)
    else:
        ohlcv_data = data

    # 기술적 지표 계산
    hma = calculate_hma(ohlcv_data['Close'])
    upper_band, lower_band = calculate_mantra_bands(ohlcv_data['Close'])
    
    # 신호 생성
    hma_signals = get_hma_signals(ohlcv_data)
    mantra_signals = get_mantra_signals(ohlcv_data)
    
    # 신호 데이터프레임 생성
    buy_dates = [s['date'] for s in hma_signals if s['type'] == 'BUY']
    sell_dates = [s['date'] for s in hma_signals if s['type'] == 'SELL']
    overbought_dates = [s['date'] for s in mantra_signals if s['type'] == 'OVERBOUGHT']
    oversold_dates = [s['date'] for s in mantra_signals if s['type'] == 'OVERSOLD']
    
    # 신호 강도에 따른 마커 크기 설정
    buy_sizes = [100 if s['strength'] == 'STRONG' else 50 for s in hma_signals if s['type'] == 'BUY']
    sell_sizes = [100 if s['strength'] == 'STRONG' else 50 for s in hma_signals if s['type'] == 'SELL']
    overbought_sizes = [100 if s['strength'] == 'STRONG' else 50 for s in mantra_signals if s['type'] == 'OVERBOUGHT']
    oversold_sizes = [100 if s['strength'] == 'STRONG' else 50 for s in mantra_signals if s['type'] == 'OVERSOLD']
    
    # 추가 플롯 설정
    apds = [
        mpf.make_addplot(hma, color='blue', width=0.7, label='HMA'),
        mpf.make_addplot(upper_band, color='red', width=0.7, label='Upper Mantra'),
        mpf.make_addplot(lower_band, color='green', width=0.7, label='Lower Mantra')
    ]
    
    # 스타일 설정
    style = mpf.make_mpf_style(
        marketcolors=mpf.make_marketcolors(
            up='red', down='blue',
            edge='inherit',
            wick='inherit',
            volume='in'
        ),
        gridstyle=':', 
        gridcolor='gray',
        y_on_right=False,
        rc={
            'font.family': get_available_font(),
            'axes.unicode_minus': False
        }
    )
    
    # 차트 생성
    fig, axlist = mpf.plot(
        ohlcv_data,
        type='candle',
        style=style,
        addplot=apds,
        volume=True,
        panel_ratios=(6,1),
        title=f'\n{ticker} HMA + Mantra Bands Signals\n',
        returnfig=True
    )
    
    # 신호 표시
    ax = axlist[0]
    for date, size in zip(buy_dates, buy_sizes):
        ax.scatter(date, ohlcv_data.loc[date, 'Low'] * 0.99, 
                  marker='^', color='red', s=size, alpha=0.6)
    for date, size in zip(sell_dates, sell_sizes):
        ax.scatter(date, ohlcv_data.loc[date, 'High'] * 1.01, 
                  marker='v', color='blue', s=size, alpha=0.6)
    for date, size in zip(overbought_dates, overbought_sizes):
        ax.scatter(date, ohlcv_data.loc[date, 'High'] * 1.02, 
                  marker='s', color='purple', s=size, alpha=0.6)
    for date, size in zip(oversold_dates, oversold_sizes):
        ax.scatter(date, ohlcv_data.loc[date, 'Low'] * 0.98, 
                  marker='s', color='green', s=size, alpha=0.6)
    
    # 범례 추가
    ax.legend(['HMA', 'Upper Mantra', 'Lower Mantra', 
              'Buy Signal', 'Sell Signal', 'Overbought', 'Oversold'])
    
    # 저장 또는 표시
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        fig.show() 