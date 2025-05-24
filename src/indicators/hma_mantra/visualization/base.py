"""
기본 차트 시각화 모듈
"""

import mplfinance as mpf
import pandas as pd
import numpy as np
from ..core import calculate_hma, calculate_mantra_bands
from ..utils import get_available_font

def plot_hma_mantra(data: pd.DataFrame, ticker: str = None, save_path: str = None):
    """HMA + 만트라 밴드 차트 플로팅"""
    if isinstance(data.columns, pd.MultiIndex):
        ohlcv_data = data.xs(ticker, axis=1, level=1)
    else:
        ohlcv_data = data

    # 기술적 지표 계산
    hma = calculate_hma(ohlcv_data['Close'])
    upper_band, lower_band = calculate_mantra_bands(ohlcv_data['Close'])
    
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
        title=f'\n{ticker} HMA + Mantra Bands\n',
        returnfig=True
    )
    
    # 범례 추가
    ax = axlist[0]
    ax.legend(['HMA', 'Upper Mantra', 'Lower Mantra'])
    
    # 저장 또는 표시
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        fig.show() 