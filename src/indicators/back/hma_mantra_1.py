import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from datetime import datetime
import matplotlib.dates as mdates
import yfinance as yf
from mplfinance.original_flavor import candlestick_ohlc

# HMA 계산
def calculate_hma(data: pd.Series, period: int = 20) -> pd.Series:
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    wma1 = data.ewm(span=half_period, adjust=False).mean()
    wma2 = data.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=sqrt_period, adjust=False).mean()
    return hma

# 만트라 밴드 계산
def calculate_mantra_bands(data: pd.Series, period: int = 20, multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series]:
    middle_line = calculate_hma(data, period)
    std = data.rolling(window=period).std()
    upper_band = middle_line + (multiplier * std)
    lower_band = middle_line - (multiplier * std)
    return upper_band, lower_band

# HMA 신호 생성
def get_hma_signals(data: pd.DataFrame) -> List[dict]:
    signals = []
    hma = calculate_hma(data['Close'])
    for i in range(1, len(data)):
        prev_close = data['Close'].iloc[i-1]
        curr_close = data['Close'].iloc[i]
        prev_hma = hma.iloc[i-1]
        curr_hma = hma.iloc[i]
        if prev_close < prev_hma and curr_close > curr_hma:
            signals.append({'date': data.index[i], 'type': 'BUY', 'price': curr_close, 'strength': 'STRONG' if curr_close > curr_hma * 1.02 else 'WEAK'})
        elif prev_close > prev_hma and curr_close < curr_hma:
            signals.append({'date': data.index[i], 'type': 'SELL', 'price': curr_close, 'strength': 'STRONG' if curr_close < curr_hma * 0.98 else 'WEAK'})
    return signals

# 만트라 밴드 신호 생성
def get_mantra_signals(data: pd.DataFrame) -> List[dict]:
    signals = []
    upper_band, lower_band = calculate_mantra_bands(data['Close'])
    for i in range(1, len(data)):
        curr_close = data['Close'].iloc[i]
        curr_upper = upper_band.iloc[i]
        curr_lower = lower_band.iloc[i]
        if curr_close > curr_upper:
            signals.append({'date': data.index[i], 'type': 'OVERBOUGHT', 'price': curr_close, 'strength': 'STRONG' if curr_close > curr_upper * 1.02 else 'WEAK'})
        elif curr_close < curr_lower:
            signals.append({'date': data.index[i], 'type': 'OVERSOLD', 'price': curr_close, 'strength': 'STRONG' if curr_close < curr_lower * 0.98 else 'WEAK'})
    return signals

# HMA & 만트라 밴드 시각화
def plot_hma_mantra(data: pd.DataFrame, ticker: str, save_path: str = None) -> None:
    hma = calculate_hma(data['Close'])
    upper_band, lower_band = calculate_mantra_bands(data['Close'])
    hma_signals = get_hma_signals(data)
    plt.figure(figsize=(15, 8))
    plt.plot(data.index, data['Close'], label='Close', color='black', alpha=0.5)
    plt.plot(data.index, hma, label='HMA', color='blue', linewidth=2)
    plt.plot(data.index, upper_band, label='Upper Band', color='red', linestyle='--')
    plt.plot(data.index, lower_band, label='Lower Band', color='green', linestyle='--')
    price_range = data['High'].max() - data['Low'].min()
    offset = price_range * 0.03
    for signal in hma_signals:
        color = 'green' if signal['type'] == 'BUY' else 'red'
        marker = '^' if signal['type'] == 'BUY' else 'v'
        plt.scatter(signal['date'], signal['price'], color=color, marker=marker, s=50, label='Buy Signal' if (signal['type']=='BUY' and signal==hma_signals[0]) else ('Sell Signal' if (signal['type']=='SELL' and signal==hma_signals[0]) else ""))
    plt.title(f'{ticker} - HMA & Mantra Bands')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# 볼린저 밴드와 HMA/만트라 밴드 비교 시각화
def plot_comparison(data: pd.DataFrame, ticker: str, save_path: str = None) -> None:
    hma = calculate_hma(data['Close'])
    upper_mantra, lower_mantra = calculate_mantra_bands(data['Close'])
    sma = data['Close'].rolling(window=20).mean()
    std = data['Close'].rolling(window=20).std()
    upper_bollinger = sma + (2 * std)
    lower_bollinger = sma - (2 * std)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    ax1.plot(data.index, data['Close'], label='Close', color='black', alpha=0.5)
    ax1.plot(data.index, hma, label='HMA', color='blue', linewidth=2)
    ax1.plot(data.index, upper_mantra, label='Mantra Upper', color='red', linestyle='--')
    ax1.plot(data.index, lower_mantra, label='Mantra Lower', color='green', linestyle='--')
    ax1.set_title(f'{ticker} - HMA & Mantra Bands')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(data.index, data['Close'], label='Close', color='black', alpha=0.5)
    ax2.plot(data.index, sma, label='SMA', color='blue', linewidth=2)
    ax2.plot(data.index, upper_bollinger, label='Bollinger Upper', color='red', linestyle='--')
    ax2.plot(data.index, lower_bollinger, label='Bollinger Lower', color='green', linestyle='--')
    ax2.set_title(f'{ticker} - Bollinger Bands')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# 이동평균선 계산
def calculate_ma(data: pd.Series, period: int) -> pd.Series:
    return data.rolling(window=period).mean()

# MACD 계산
def calculate_macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    exp1 = data.ewm(span=fast_period, adjust=False).mean()
    exp2 = data.ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

# RSI 계산
def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# 시그널 강도 시각화 (간단 버전)
def plot_signals_with_strength(data: pd.DataFrame, ticker: str, save_path: str = None) -> None:
    # 기본 지표 계산
    hma = calculate_hma(data['Close'])
    upper_mantra, lower_mantra = calculate_mantra_bands(data['Close'])
    hma_signals = get_hma_signals(data)
    
    # 이동평균선 계산
    ma5 = calculate_ma(data['Close'], 5)
    ma20 = calculate_ma(data['Close'], 20)
    ma60 = calculate_ma(data['Close'], 60)
    ma160 = calculate_ma(data['Close'], 160)
    ma200 = calculate_ma(data['Close'], 200)
    
    # MACD 계산
    macd, signal, hist = calculate_macd(data['Close'])
    
    # RSI 계산
    rsi = calculate_rsi(data['Close'])
    
    # VIX 데이터 가져오기
    vix = yf.download('^VIX', start=data.index[0], end=data.index[-1])['Close']
    
    # 서브플롯 생성
    fig = plt.figure(figsize=(15, 20))
    gs = fig.add_gridspec(6, 1, height_ratios=[3, 1, 1, 1, 1, 1])
    
    # 메인 차트 (캔들차트 + HMA + 만트라 밴드 + 이동평균선)
    ax1 = fig.add_subplot(gs[0])
    
    # 캔들차트 데이터 준비
    ohlc = data[['Open', 'High', 'Low', 'Close']].copy().reset_index()
    if hasattr(ohlc['Date'].dt, 'tz') and ohlc['Date'].dt.tz is not None:
        ohlc['Date'] = ohlc['Date'].dt.tz_localize(None)
    else:
        ohlc['Date'] = pd.to_datetime(ohlc['Date'])
    ohlc['Date_ordinal'] = mdates.date2num(ohlc['Date'])
    ohlc_values = ohlc[['Date_ordinal', 'Open', 'High', 'Low', 'Close']].values
    
    # 캔들차트 그리기
    candlestick_ohlc(ax1, ohlc_values, width=0.6, colorup='g', colordown='r', alpha=0.7)
    
    # 가격선 추가
    ordinal_index = mdates.date2num(data.index.to_pydatetime())
    ax1.plot(ordinal_index, data['Close'], label='Close', color='black', linewidth=0.6, alpha=0.7)
    
    # HMA, 만트라 밴드 등 겹쳐서 표시
    ax1.plot(ordinal_index, hma, label='HMA', color='blue', linewidth=1.4)
    ax1.plot(ordinal_index, upper_mantra, label='Mantra Upper', color='red', linestyle='--', linewidth=0.7)
    ax1.plot(ordinal_index, lower_mantra, label='Mantra Lower', color='green', linestyle='--', linewidth=0.7)
    
    # 이동평균선 추가
    ax1.plot(ordinal_index, ma5, label='MA5', color='purple', linewidth=0.7)
    ax1.plot(ordinal_index, ma20, label='MA20', color='orange', linewidth=0.7)
    ax1.plot(ordinal_index, ma60, label='MA60', color='brown', linewidth=0.7)
    ax1.plot(ordinal_index, ma160, label='MA160', color='pink', linewidth=0.7)
    ax1.plot(ordinal_index, ma200, label='MA200', color='gray', linewidth=0.7)
    
    # 신호 마커 추가
    price_range = data['High'].max() - data['Low'].min()
    offset = price_range * 0.03
    buy_plotted = False
    sell_plotted = False
    
    for signal in hma_signals:
        color = 'green' if signal['type'] == 'BUY' else 'red'
        ax1.axvline(signal['date'], color=color, linestyle='--', linewidth=0.35, alpha=0.7)
        marker = '^' if signal['type'] == 'BUY' else 'v'
        label = ''
        
        if signal['type'] == 'BUY':
            band_val = lower_mantra.loc[signal['date']] if signal['date'] in lower_mantra.index else signal['price']
            y = band_val - offset
            if not buy_plotted:
                label = 'Buy Signal'
                buy_plotted = True
            va = 'top'
        else:
            band_val = upper_mantra.loc[signal['date']] if signal['date'] in upper_mantra.index else signal['price']
            y = band_val + offset
            if not sell_plotted:
                label = 'Sell Signal'
                sell_plotted = True
            va = 'bottom'
            
        ax1.scatter(mdates.date2num(signal['date']), y, color=color, marker=marker, s=50, label=label)
        ax1.text(
            mdates.date2num(signal['date']), y, signal['date'].strftime('%Y-%m-%d'),
            rotation=45, fontsize=5, color=color, ha='center', va=va,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.1')
        )
    
    ax1.set_title(f'{ticker} - Technical Analysis')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax1.grid(True, alpha=0.3)
    
    # 거래량 차트
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.bar(ordinal_index, data['Volume'], color='gray', alpha=0.5, width=0.6)
    ax2.set_ylabel('Volume')
    ax2.grid(True, alpha=0.3)
    
    # VIX 차트
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(ordinal_index, vix, color='purple', linewidth=0.7)
    ax3.set_ylabel('VIX')
    ax3.grid(True, alpha=0.3)
    
    # MACD 차트
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(ordinal_index, macd, label='MACD', color='blue', linewidth=0.7)
    ax4.plot(ordinal_index, signal, label='Signal', color='red', linewidth=0.7)
    ax4.bar(ordinal_index, hist, color='gray', alpha=0.5, width=0.6)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_ylabel('MACD')
    ax4.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax4.grid(True, alpha=0.3)
    
    # RSI 차트
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    ax5.plot(ordinal_index, rsi, color='blue', linewidth=0.7)
    ax5.axhline(y=70, color='red', linestyle='--', linewidth=0.5)
    ax5.axhline(y=30, color='green', linestyle='--', linewidth=0.5)
    ax5.set_ylabel('RSI')
    ax5.set_ylim(0, 100)
    ax5.grid(True, alpha=0.3)
    
    # x축 설정
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
