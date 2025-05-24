import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from datetime import datetime
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
import mplfinance as mpf
import matplotlib
import platform
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import yfinance as yf
from matplotlib.patches import Rectangle, Patch

if platform.system() == 'Darwin':  # macOS
    matplotlib.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    matplotlib.rc('font', family='Malgun Gothic')
else:  # Linux
    matplotlib.rc('font', family='NanumGothic')
matplotlib.rcParams['axes.unicode_minus'] = False

def calculate_hma(data: pd.Series, period: int = 20) -> pd.Series:
    """
    Hull Moving Average (HMA) 계산
    
    Args:
        data (pd.Series): 가격 데이터
        period (int): 기간 (기본값: 20)
    
    Returns:
        pd.Series: HMA 값
    """
    # WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    # WMA(n/2)
    wma1 = data.ewm(span=half_period, adjust=False).mean()
    
    # WMA(n)
    wma2 = data.ewm(span=period, adjust=False).mean()
    
    # 2*WMA(n/2) - WMA(n)
    raw_hma = 2 * wma1 - wma2
    
    # WMA(sqrt(n))
    hma = raw_hma.ewm(span=sqrt_period, adjust=False).mean()
    
    return hma

def calculate_mantra_bands(data: pd.Series, period: int = 20, multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series]:
    """
    만트라 밴드 계산
    
    Args:
        data (pd.Series): 가격 데이터
        period (int): 기간 (기본값: 20)
        multiplier (float): 표준편차 승수 (기본값: 2.0)
    
    Returns:
        Tuple[pd.Series, pd.Series]: (상단 밴드, 하단 밴드)
    """
    # 중간선 (HMA)
    middle_line = calculate_hma(data, period)
    
    # 표준편차 계산
    std = data.rolling(window=period).std()
    
    # 상단/하단 밴드
    upper_band = middle_line + (multiplier * std)
    lower_band = middle_line - (multiplier * std)
    
    return upper_band, lower_band

def plot_hma_mantra(data: pd.DataFrame, ticker: str, save_path: str = None) -> None:
    """
    HMA와 만트라 밴드 시각화
    
    Args:
        data (pd.DataFrame): 가격 데이터 (Date, Close 컬럼 필요)
        ticker (str): 종목 코드
        save_path (str, optional): 저장 경로
    """
    # HMA와 만트라 밴드 계산
    hma = calculate_hma(data['Close'])
    upper_band, lower_band = calculate_mantra_bands(data['Close'])
    
    # HMA 신호 계산
    hma_signals = get_hma_signals(data)
    
    # 그래프 설정
    fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    # HMA 신호 표시
    for signal in hma_signals:
        if signal['type'] == 'BUY':
            ax2.scatter(signal['date'], signal['price'], 
                       color='green', marker='^', s=100, 
                       label='Buy Signal' if signal == hma_signals[0] else "")
        else:  # SELL
            ax2.scatter(signal['date'], signal['price'], 
                       color='red', marker='v', s=100, 
                       label='Sell Signal' if signal == hma_signals[0] else "")
    
    # 그래프 스타일 설정
    ax2.title.set_text(f'{ticker} - HMA & Mantra Bands')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 저장 또는 표시
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def get_hma_signals(data: pd.DataFrame) -> List[dict]:
    """
    HMA 기반 매매 신호 생성
    
    Args:
        data (pd.DataFrame): 가격 데이터 (Date, Close 컬럼 필요)
    
    Returns:
        List[dict]: 매매 신호 목록
    """
    signals = []
    hma = calculate_hma(data['Close'])
    
    for i in range(1, len(data)):
        prev_close = data['Close'].iloc[i-1]
        curr_close = data['Close'].iloc[i]
        prev_hma = hma.iloc[i-1]
        curr_hma = hma.iloc[i]
        
        # 매수 신호: 가격이 HMA를 상향 돌파
        if prev_close < prev_hma and curr_close > curr_hma:
            signals.append({
                'date': data.index[i],
                'type': 'BUY',
                'price': curr_close,
                'strength': 'STRONG' if curr_close > curr_hma * 1.02 else 'WEAK'
            })
        
        # 매도 신호: 가격이 HMA를 하향 돌파
        elif prev_close > prev_hma and curr_close < curr_hma:
            signals.append({
                'date': data.index[i],
                'type': 'SELL',
                'price': curr_close,
                'strength': 'STRONG' if curr_close < curr_hma * 0.98 else 'WEAK'
            })
    
    return signals

def get_mantra_signals(data: pd.DataFrame) -> List[dict]:
    """
    만트라 밴드 기반 매매 신호 생성
    
    Args:
        data (pd.DataFrame): 가격 데이터 (Date, Close 컬럼 필요)
    
    Returns:
        List[dict]: 매매 신호 목록
    """
    signals = []
    upper_band, lower_band = calculate_mantra_bands(data['Close'])
    
    for i in range(1, len(data)):
        curr_close = data['Close'].iloc[i]
        curr_upper = upper_band.iloc[i]
        curr_lower = lower_band.iloc[i]
        
        # 과매수 신호
        if curr_close > curr_upper:
            signals.append({
                'date': data.index[i],
                'type': 'OVERBOUGHT',
                'price': curr_close,
                'strength': 'STRONG' if curr_close > curr_upper * 1.02 else 'WEAK'
            })
        
        # 과매도 신호
        elif curr_close < curr_lower:
            signals.append({
                'date': data.index[i],
                'type': 'OVERSOLD',
                'price': curr_close,
                'strength': 'STRONG' if curr_close < curr_lower * 0.98 else 'WEAK'
            })
    
    return signals

def plot_comparison(data: pd.DataFrame, ticker: str, save_path: str = None) -> None:
    """
    볼린저 밴드와 HMA/만트라 밴드 비교 시각화
    
    Args:
        data (pd.DataFrame): 가격 데이터 (Date, Close 컬럼 필요)
        ticker (str): 종목 코드
        save_path (str, optional): 저장 경로
    """
    # HMA와 만트라 밴드 계산
    hma = calculate_hma(data['Close'])
    upper_mantra, lower_mantra = calculate_mantra_bands(data['Close'])
    
    # 볼린저 밴드 계산
    sma = data['Close'].rolling(window=20).mean()
    std = data['Close'].rolling(window=20).std()
    upper_bollinger = sma + (2 * std)
    lower_bollinger = sma - (2 * std)
    
    # 그래프 설정
    fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    # HMA 신호 표시
    for signal in get_hma_signals(data):
        if signal['type'] == 'BUY':
            ax2.scatter(signal['date'], signal['price'], 
                       color='green', marker='^', s=100, 
                       label='Buy Signal' if signal == get_hma_signals(data)[0] else "")
        else:  # SELL
            ax2.scatter(signal['date'], signal['price'], 
                       color='red', marker='v', s=100, 
                       label='Sell Signal' if signal == get_hma_signals(data)[0] else "")
    
    # HMA 신호 표시
    ax2.plot(data.index, hma, label='HMA', color='blue', linewidth=2)
    ax2.plot(data.index, upper_mantra, label='Upper Band', color='red', linestyle='--')
    ax2.plot(data.index, lower_mantra, label='Lower Band', color='green', linestyle='--')
    
    # 그래프 스타일 설정
    ax2.title.set_text(f'{ticker} - HMA & Mantra Bands')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 볼린저 밴드 영역 색상 채우기 (NaN 마스킹)
    valid_mask = (~upper_bollinger.isna()) & (~lower_bollinger.isna()) & (~sma.isna())
    ax2.fill_between(data.index[valid_mask], upper_bollinger[valid_mask], lower_bollinger[valid_mask], color='#FFB6C1', alpha=0.08, label='Bollinger Band Area')
    ax2.fill_between(data.index[valid_mask], upper_bollinger[valid_mask], sma[valid_mask], color='#D8BFD8', alpha=0.10, label='Bollinger Upper-Mid')
    ax2.fill_between(data.index[valid_mask], sma[valid_mask], lower_bollinger[valid_mask], color='#B0E0E6', alpha=0.10, label='Bollinger Mid-Lower')
    # plot도 마찬가지로 NaN 제외
    ax2.plot(data.index[valid_mask], upper_bollinger[valid_mask], label='Bollinger Upper', color='#FF69B4', linestyle='--', linewidth=0.8)
    ax2.plot(data.index[valid_mask], lower_bollinger[valid_mask], label='Bollinger Lower', color='#1E90FF', linestyle='--', linewidth=0.8)
    ax2.plot(data.index[valid_mask], sma[valid_mask], label='Bollinger Middle', color='#8B008B', linestyle='--', linewidth=0.8)
    # SMA 60, 200도 NaN 제외
    valid_sma60 = ~sma.rolling(window=60).mean().isna()
    valid_sma200 = ~sma.rolling(window=200).mean().isna()
    ax2.plot(data.index[valid_sma60], sma.rolling(window=60).mean()[valid_sma60], label='SMA(60)', color='orange', linewidth=1.5, linestyle='-')
    ax2.plot(data.index[valid_sma200], sma.rolling(window=200).mean()[valid_sma200], label='SMA(200)', color='brown', linewidth=1.5, linestyle='-')
    
    # 그래프 간격 조정
    plt.tight_layout()
    
    # 저장 또는 표시
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def calculate_signal_strength(data: pd.DataFrame) -> Dict[str, List[Dict]]:
    """
    매수/매도 신호 강도 계산 (순차적 신호 발생 고려)
    
    Args:
        data (pd.DataFrame): 가격 데이터
    
    Returns:
        Dict[str, List[Dict]]: 매수/매도 신호 정보
    """
    signals = {
        'buy': [],
        'sell': []
    }
    
    # HMA와 만트라 밴드 계산
    hma = calculate_hma(data['Close'])
    upper_mantra, lower_mantra = calculate_mantra_bands(data['Close'])
    
    # 신호 발생 여부를 저장할 배열
    mantra_buy_signals = np.zeros(len(data))
    mantra_sell_signals = np.zeros(len(data))
    hma_buy_signals = np.zeros(len(data))
    hma_sell_signals = np.zeros(len(data))
    
    # 만트라 밴드 신호 기록
    for i in range(1, len(data)):
        curr_close = data['Close'].iloc[i]
        curr_upper_mantra = upper_mantra.iloc[i]
        curr_lower_mantra = lower_mantra.iloc[i]
        
        # 만트라 밴드 하단 돌파 (매수 신호)
        if curr_close < curr_lower_mantra:
            mantra_buy_signals[i] = 1
        
        # 만트라 밴드 상단 돌파 (매도 신호)
        if curr_close > curr_upper_mantra:
            mantra_sell_signals[i] = 1
    
    # HMA 신호 기록
    for i in range(1, len(data)):
        curr_close = data['Close'].iloc[i]
        prev_close = data['Close'].iloc[i-1]
        curr_hma = hma.iloc[i]
        prev_hma = hma.iloc[i-1]
        
        # HMA 상향 돌파 (매수 신호)
        if prev_close < prev_hma and curr_close > curr_hma:
            hma_buy_signals[i] = 1
        
        # HMA 하향 돌파 (매도 신호)
        if prev_close > prev_hma and curr_close < curr_hma:
            hma_sell_signals[i] = 1
    
    # 순차적 신호 분석 (2일 이내)
    for i in range(2, len(data)):
        # 매수 신호 분석
        buy_strength = 0
        buy_reasons = []
        
        # 만트라 밴드 하단 돌파 확인
        if mantra_buy_signals[i] == 1:
            buy_strength += 35
            buy_reasons.append("만트라 밴드 하단돌파 (가격이 만트라 밴드 하단을 하회)")
        
        # HMA 상향 돌파 확인 (현재 또는 최근 2일 이내)
        hma_buy_recent = any(hma_buy_signals[i-2:i+1] == 1)
        if hma_buy_recent:
            buy_strength += 30
            buy_reasons.append("HMA 상향돌파 (가격이 HMA를 상향 돌파)")
            
            # 강도 추가
            curr_close = data['Close'].iloc[i]
            curr_hma = hma.iloc[i]
            if curr_close > curr_hma * 1.02:
                buy_strength += 10
                buy_reasons.append("HMA 강한 상향돌파 (가격이 HMA보다 2% 이상 상회)")
            elif curr_close > curr_hma * 1.01:
                buy_strength += 5
                buy_reasons.append("HMA 중간 상향돌파 (가격이 HMA보다 1% 이상 상회)")
        
        # 매도 신호 분석
        sell_strength = 0
        sell_reasons = []
        
        # 만트라 밴드 상단 돌파 확인
        if mantra_sell_signals[i] == 1:
            sell_strength += 35
            sell_reasons.append("만트라 밴드 상단돌파 (가격이 만트라 밴드 상단을 상회)")
        
        # HMA 하향 돌파 확인 (현재 또는 최근 2일 이내)
        hma_sell_recent = any(hma_sell_signals[i-2:i+1] == 1)
        if hma_sell_recent:
            sell_strength += 30
            sell_reasons.append("HMA 하향돌파 (가격이 HMA를 하향 돌파)")
            
            # 강도 추가
            curr_close = data['Close'].iloc[i]
            curr_hma = hma.iloc[i]
            if curr_close < curr_hma * 0.98:
                sell_strength += 10
                sell_reasons.append("HMA 강한 하향돌파 (가격이 HMA보다 2% 이상 하락)")
            elif curr_close < curr_hma * 0.99:
                sell_strength += 5
                sell_reasons.append("HMA 중간 하향돌파 (가격이 HMA보다 1% 이상 하락)")
        
        # 신호 저장
        if buy_strength > 0:
            signals['buy'].append({
                'date': data.index[i],
                'strength': buy_strength,
                'price': data['Close'].iloc[i],
                'reasons': buy_reasons
            })
        
        if sell_strength > 0:
            signals['sell'].append({
                'date': data.index[i],
                'strength': sell_strength,
                'price': data['Close'].iloc[i],
                'reasons': sell_reasons
            })
    
    return signals

def safe_addplot(series, **kwargs):
    if series is not None and not series.dropna().empty:
        return mpf.make_addplot(series, **kwargs)
    return None

def plot_signals_with_strength(data: pd.DataFrame, ticker: str, save_path: str = None) -> None:
    """
    매수/매도 신호와 강도를 시각화 (만트라 밴드 신호를 강도별 vertical line으로 표시, 범례 구분)
    """
    signals = calculate_signal_strength(data)
    hma_signals = get_hma_signals(data)
    fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(15, 28), sharex=True,
        gridspec_kw={'height_ratios': [4, 1]})
    
    # HMA 서브플롯 (ax2)
    hma = calculate_hma(data['Close'])
    upper_mantra, lower_mantra = calculate_mantra_bands(data['Close'])

    # HMA 플롯
    ax2.plot(data.index, data['Close'], label='가격', color='black', linewidth=1.5, alpha=0.7)
    ax2.plot(data.index, hma, label='HMA', color='#4169E1', linewidth=1.5)
    ax2.plot(data.index, upper_mantra, label='상단밴드', color='#FF69B4', linewidth=1.5)
    ax2.plot(data.index, lower_mantra, label='하단밴드', color='#32CD32', linewidth=1.5)

    # HMA 영역 채우기
    ax2.fill_between(data.index, upper_mantra, lower_mantra, color='#FFB6C1', alpha=0.1, label='밴드영역')
    ax2.fill_between(data.index, upper_mantra, hma, color='#D8BFD8', alpha=0.1, label='상단영역')
    ax2.fill_between(data.index, hma, lower_mantra, color='#B0E0E6', alpha=0.1, label='하단영역')

    # HMA 매수/매도 신호
    hma_buy_dates = []
    hma_sell_dates = []
    
    for i in range(1, len(data)):
        # 매수 신호: 가격이 HMA를 상향돌파
        if data['Close'].iloc[i-1] <= hma.iloc[i-1] and data['Close'].iloc[i] > hma.iloc[i]:
            hma_buy_dates.append(data.index[i])
            ax2.scatter(data.index[i], data['Close'].iloc[i], color='#006400', marker='^', s=80,
                          edgecolor='black', linewidth=1.5, label='HMA 매수' if i == 1 else "")
        
        # 매도 신호: 가격이 HMA를 하향돌파
        elif data['Close'].iloc[i-1] >= hma.iloc[i-1] and data['Close'].iloc[i] < hma.iloc[i]:
            hma_sell_dates.append(data.index[i])
            ax2.scatter(data.index[i], data['Close'].iloc[i], color='#8B0000', marker='v', s=80,
                          edgecolor='black', linewidth=1.5, label='HMA 매도' if i == 1 else "")

    # HMA 범례 마커 생성
    hma_buy_marker = mlines.Line2D([], [], color='w', marker='^', markerfacecolor='#006400',
                                  markeredgecolor='black', markersize=10, label='HMA 매수')
    hma_sell_marker = mlines.Line2D([], [], color='w', marker='v', markerfacecolor='#8B0000',
                                   markeredgecolor='black', markersize=10, label='HMA 매도')

    # HMA 범례 설정
    handles, labels = ax2.get_legend_handles_labels()
    handles = [h for h, l in zip(handles, labels) if l not in ['HMA 매수', 'HMA 매도']]
    labels = [l for l in labels if l not in ['HMA 매수', 'HMA 매도']]
    
    # 범례를 두 개로 분리하여 표시
    first_legend = ax2.legend(handles, labels, loc='upper left', fontsize=8, ncol=2)
    ax2.add_artist(first_legend)
    ax2.legend([hma_buy_marker, hma_sell_marker], ['HMA 매수', 'HMA 매도'], loc='upper right', fontsize=8)

    ax2.set_ylabel('HMA')
    ax2.grid(True, alpha=0.3)

    # HMA 신호 시점에 수직선 추가
    for date in hma_buy_dates:
        ax2.axvline(x=date, color='#006400', linestyle='--', alpha=0.3, linewidth=0.5)
    for date in hma_sell_dates:
        ax2.axvline(x=date, color='#8B0000', linestyle='--', alpha=0.3, linewidth=0.5)

    # VIX subplot (ax3)
    vix_data = yf.Ticker('^VIX').history(period=f'{len(data)}d', interval='1d')
    vix_data = vix_data.reindex(data.index, method='ffill')  # 날짜 맞추기
    ax3.plot(data.index, vix_data['Close'], color='purple', label='VIX', linewidth=1.5)
    ax3.scatter(data.index, vix_data['Close'], color='purple', s=15, marker='o', label='VIX 값', zorder=3)
    ax3.set_ylabel('VIX')
    ax3.grid(True, alpha=0.3)
    # VIX 전략별 배경색 및 전략 가이드
    vix_ranges = [12, 16, 20, 30, 100]
    vix_colors = ['#90caf9', '#a5d6a7', '#ffb3b3', '#b39ddb']
    vix_patches = []
    for i in range(len(vix_ranges)-1):
        ax3.axhspan(vix_ranges[i], vix_ranges[i+1], color=vix_colors[i], alpha=0.3)
        vix_patches.append(Patch(facecolor=vix_colors[i], edgecolor='none', alpha=0.5, label=f'{vix_ranges[i]}~{vix_ranges[i+1]}: {vix_guides[i]}'))
    handles, labels = ax3.get_legend_handles_labels()
    handles = [handles[0]] + vix_patches
    labels = [labels[0]] + [patch.get_label() for patch in vix_patches]
    ax3.legend(handles, labels, loc='upper left', fontsize=10)
    vix_max = vix_data['Close'].max()
    ax3.set_ylim(0, vix_max * 1.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# MACD, RSI 계산 함수
def calc_macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

vix_guides = [
    '저위험: 주식 비중 확대',
    '경계: 분할매수/분할매도',
    '위험: 현금 비중 확대',
    '극단적 공포: 분할매수 기회'
]

def plot_hma_mantra_with_candles(data: pd.DataFrame, ticker: str, save_path: str = None) -> None:
    """
    HMA와 만트라 밴드 및 캔들차트 시각화
    
    Args:
        data (pd.DataFrame): 가격 데이터 (Date, Close 컬럼 필요)
        ticker (str): 종목 코드
        save_path (str, optional): 저장 경로
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True, gridspec_kw={'height_ratios': [4, 1, 1]})

    # 1. 메인 차트(ax1): 캔들차트, HMA, 만트라 밴드, 볼린저 밴드, 여러 MA, 신호 마커, 수직선, 범례 등
    # (예시) candlestick_ohlc(ax1, ...), ax1.plot(...), ax1.scatter(...), ax1.axvline(...), ax1.legend(...)

    # 2. VIX 차트(ax2): VIX, 배경색, 신호 수직선, 범례 등

    # 3. 거래량 차트(ax3): 거래량, 거래량 MA, 컬러 바, 범례 등

    # x축은 datetime 인덱스, sharex=True
    # 신호 마커, 수직선, 범례 등도 메인 차트에 추가
    # HMA/볼린저/MA 단독 subplot 등은 모두 제거

    # 캔들차트 그리기
    ohlc = data[['Open', 'High', 'Low', 'Close']].copy().reset_index()
    if hasattr(ohlc['Date'].dt, 'tz') and ohlc['Date'].dt.tz is not None:
        ohlc['Date'] = ohlc['Date'].dt.tz_localize(None)
    else:
        ohlc['Date'] = pd.to_datetime(ohlc['Date'])
    ohlc['Date_ordinal'] = mdates.date2num(ohlc['Date'])
    ohlc_values = ohlc[['Date_ordinal', 'Open', 'High', 'Low', 'Close']].values
    candlestick_ohlc(ax1, ohlc_values, width=0.6, colorup='g', colordown='r', alpha=0.7)

    hma = calculate_hma(data['Close'])
    upper_mantra, lower_mantra = calculate_mantra_bands(data['Close'])
    ordinal_index = mdates.date2num(data.index.to_pydatetime())

    ax1.plot(ordinal_index, hma, label='HMA', color='#4169E1', linewidth=1.5)
    ax1.plot(ordinal_index, upper_mantra, label='만트라 상단', color='#FF69B4', linewidth=1.2)
    ax1.plot(ordinal_index, lower_mantra, label='만트라 하단', color='#32CD32', linewidth=1.2)
    ax1.fill_between(ordinal_index, upper_mantra, lower_mantra, color='#FFB6C1', alpha=0.10, label='만트라 밴드 영역')
    ax1.fill_between(ordinal_index, upper_mantra, hma, color='#D8BFD8', alpha=0.10, label='상단영역')
    ax1.fill_between(ordinal_index, hma, lower_mantra, color='#B0E0E6', alpha=0.10, label='하단영역')

    ax1.set_ylabel('가격')
    ax1.set_title(f'{ticker} 만트라 밴드 & 캔들차트')
    ax1.grid(True, alpha=0.3)
    ax1.autoscale_view()
    ax1.set_xlim(ordinal_index[0], ordinal_index[-1])
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.legend(loc='upper left', fontsize=8, ncol=2)

    # HMA 신호 계산
    hma_signals = get_hma_signals(data)
    
    # HMA 신호 표시
    for signal in hma_signals:
        if signal['type'] == 'BUY':
            ax1.scatter(signal['date'], signal['price'], 
                       color='green', marker='^', s=100, 
                       label='Buy Signal' if signal == hma_signals[0] else "")
        else:  # SELL
            ax1.scatter(signal['date'], signal['price'], 
                       color='red', marker='v', s=100, 
                       label='Sell Signal' if signal == hma_signals[0] else "")
    
    # 그래프 스타일 설정
    ax1.title.set_text(f'{ticker} - HMA & Mantra Bands')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 저장 또는 표시
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # VIX subplot (ax2)
    vix_data = yf.Ticker('^VIX').history(period=f'{len(data)}d', interval='1d')
    vix_data = vix_data.reindex(data.index, method='ffill')  # 날짜 맞추기
    ax2.plot(data.index, vix_data['Close'], color='purple', label='VIX', linewidth=1.5)
    ax2.scatter(data.index, vix_data['Close'], color='purple', s=15, marker='o', label='VIX 값', zorder=3)
    ax2.set_ylabel('VIX')
    ax2.grid(True, alpha=0.3)
    # VIX 전략별 배경색 및 전략 가이드
    vix_ranges = [12, 16, 20, 30, 100]
    vix_colors = ['#90caf9', '#a5d6a7', '#ffb3b3', '#b39ddb']
    vix_patches = []
    for i in range(len(vix_ranges)-1):
        ax2.axhspan(vix_ranges[i], vix_ranges[i+1], color=vix_colors[i], alpha=0.3)
        vix_patches.append(Patch(facecolor=vix_colors[i], edgecolor='none', alpha=0.5, label=f'{vix_ranges[i]}~{vix_ranges[i+1]}: {vix_guides[i]}'))
    handles, labels = ax2.get_legend_handles_labels()
    handles = [handles[0]] + vix_patches
    labels = [labels[0]] + [patch.get_label() for patch in vix_patches]
    ax2.legend(handles, labels, loc='upper left', fontsize=10)
    vix_max = vix_data['Close'].max()
    ax2.set_ylim(0, vix_max * 1.2)

    # SMA 50, 200 그리기
    sma50 = data['Close'].rolling(window=50).mean()
    sma200 = data['Close'].rolling(window=200).mean()
    ax3.plot(data.index, sma50, label='SMA(50)', color='orange', linewidth=1.5, linestyle='-')
    ax3.plot(data.index, sma200, label='SMA(200)', color='brown', linewidth=1.5, linestyle='-')
    ax3.set_ylabel('SMA')
    ax3.grid(True, alpha=0.3)

    # 거래량 그리기
    volume = data['Volume']
    ax3.bar(data.index, volume, color='gray', alpha=0.5)
    ax3.set_ylabel('거래량')
    ax3.grid(True, alpha=0.3)

    # MACD 그리기
    macd, signal, hist = calc_macd(data['Close'])
    ax3.plot(data.index, macd, label='MACD', color='blue', linewidth=1.5)
    ax3.plot(data.index, signal, label='Signal', color='red', linewidth=1.5)
    ax3.bar(data.index, hist, color='gray', alpha=0.5)
    ax3.set_ylabel('MACD')
    ax3.grid(True, alpha=0.3)

    # RSI 그리기
    rsi = calc_rsi(data['Close'])
    ax3.plot(data.index, rsi, label='RSI', color='green', linewidth=1.5)
    ax3.set_ylabel('RSI')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show() 