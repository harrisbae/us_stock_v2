"""
비교 차트 시각화 모듈
"""

import mplfinance as mpf
import pandas as pd
import numpy as np
from ..core import calculate_hma, calculate_mantra_bands, calculate_ma
from ..utils import get_available_font

def plot_comparison(data: pd.DataFrame, ticker: str = None, save_path: str = None):
    """HMA와 일반 이동평균선 비교 차트"""
    if isinstance(data.columns, pd.MultiIndex):
        ohlcv_data = data.xs(ticker, axis=1, level=1)
    else:
        ohlcv_data = data

    # 기술적 지표 계산
    hma = calculate_hma(ohlcv_data['Close'])
    ma20 = calculate_ma(ohlcv_data['Close'], 20)
    ma60 = calculate_ma(ohlcv_data['Close'], 60)
    ma120 = calculate_ma(ohlcv_data['Close'], 120)
    
    # 추가 플롯 설정
    apds = [
        mpf.make_addplot(hma, color='blue', width=1.0, label='HMA(20)'),
        mpf.make_addplot(ma20, color='red', width=0.7, label='MA(20)'),
        mpf.make_addplot(ma60, color='green', width=0.7, label='MA(60)'),
        mpf.make_addplot(ma120, color='purple', width=0.7, label='MA(120)')
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
        title=f'\n{ticker} HMA vs Moving Averages\n',
        returnfig=True
    )
    
    # 범례 추가
    ax = axlist[0]
    ax.legend(['HMA(20)', 'MA(20)', 'MA(60)', 'MA(120)'])
    
    # 저장 또는 표시
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        fig.show() 