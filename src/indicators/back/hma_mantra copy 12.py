import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from datetime import datetime
import matplotlib.dates as mdates
import yfinance as yf
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.lines as mlines
import traceback
import sys
import platform
import warnings
import requests
from io import BytesIO
import zipfile
import xml.etree.ElementTree as ET
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore", category=FutureWarning)

# 한글 폰트 설정 (운영체제별)
import matplotlib
if platform.system() == 'Darwin':  # macOS
    matplotlib.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    matplotlib.rc('font', family='Malgun Gothic')
else:  # Linux
    matplotlib.rc('font', family='NanumGothic')
matplotlib.rcParams['axes.unicode_minus'] = False

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

def safe_in_index(index, key):
    """인덱스에 key가 있는지 안전하게 확인합니다."""
    try:
        return key in index
    except TypeError as e:
        print(f"[경고] hashable 타입이 아닌 객체가 인덱스에 사용됨: {type(key)}")
        print(f"key 값: {key}")
        print(f"index 타입: {type(index)}")
        print(f"예외 메시지: {str(e)}")
        print("스택 트레이스:")
        traceback.print_exc()
        return False

# HMA 신호 생성
def get_hma_signals(data: pd.DataFrame) -> List[dict]:
    """HMA 신호를 생성합니다."""
    try:
        signals = []
        hma = calculate_hma(data['Close'])
        for i in range(1, len(data)):
            try:
                prev_close = float(data['Close'].iloc[i-1].item())
                curr_close = float(data['Close'].iloc[i].item())
                prev_hma = float(hma.iloc[i-1].item())
                curr_hma = float(hma.iloc[i].item())
                # NaN 체크
                if pd.isna(prev_close) or pd.isna(curr_close) or pd.isna(prev_hma) or pd.isna(curr_hma):
                    continue
                # 매수 신호
                if prev_close < prev_hma and curr_close > curr_hma:
                    strength = 'STRONG' if curr_close > curr_hma * 1.02 else 'WEAK'
                    signals.append({
                        'date': data.index[i],
                        'type': 'BUY',
                        'price': curr_close,
                        'strength': strength
                    })
                # 매도 신호
                elif prev_close > prev_hma and curr_close < curr_hma:
                    strength = 'STRONG' if curr_close < curr_hma * 0.98 else 'WEAK'
                    signals.append({
                        'date': data.index[i],
                        'type': 'SELL',
                        'price': curr_close,
                        'strength': strength
                    })
            except Exception as e:
                print(f"신호 생성 중 오류 발생 (인덱스 {i}):")
                print(f"예외 메시지: {str(e)}")
                print("스택 트레이스:")
                traceback.print_exc()
                continue
        return signals
    except Exception as e:
        print("HMA 신호 생성 중 오류 발생:")
        print(f"예외 메시지: {str(e)}")
        print("스택 트레이스:")
        traceback.print_exc()
        return []

# 만트라 밴드 신호 생성
def get_mantra_signals(data: pd.DataFrame) -> List[dict]:
    """만트라 밴드 신호를 생성합니다."""
    try:
        signals = []
        upper_band, lower_band = calculate_mantra_bands(data['Close'])
        for i in range(1, len(data)):
            try:
                curr_close = float(data['Close'].iloc[i].item())
                curr_upper = float(upper_band.iloc[i].item())
                curr_lower = float(lower_band.iloc[i].item())
                # NaN 체크
                if pd.isna(curr_close) or pd.isna(curr_upper) or pd.isna(curr_lower):
                    continue
                # 과매수 신호
                if curr_close > curr_upper:
                    strength = 'STRONG' if curr_close > curr_upper * 1.02 else 'WEAK'
                    signals.append({
                        'date': data.index[i],
                        'type': 'OVERBOUGHT',
                        'price': curr_close,
                        'strength': strength
                    })
                # 과매도 신호
                elif curr_close < curr_lower:
                    strength = 'STRONG' if curr_close < curr_lower * 0.98 else 'WEAK'
                    signals.append({
                        'date': data.index[i],
                        'type': 'OVERSOLD',
                        'price': curr_close,
                        'strength': strength
                    })
            except Exception as e:
                print(f"만트라 신호 생성 중 오류 발생 (인덱스 {i}):")
                print(f"예외 메시지: {str(e)}")
                print("스택 트레이스:")
                traceback.print_exc()
                continue
        return signals
    except Exception as e:
        print("만트라 신호 생성 중 오류 발생:")
        print(f"예외 메시지: {str(e)}")
        print("스택 트레이스:")
        traceback.print_exc()
        return []

# HMA & 만트라 밴드 시각화
def plot_hma_mantra(data: pd.DataFrame, ticker: str, save_path: str = None) -> None:
    """HMA & 만트라 밴드를 시각화합니다."""
    try:
        hma = calculate_hma(data['Close'])
        upper_band, lower_band = calculate_mantra_bands(data['Close'])
        hma_signals = get_hma_signals(data)
        price_range = to_float(data['High'].max()) - to_float(data['Low'].min())
        offset = price_range * 0.03
        plt.figure(figsize=(15, 8))
        ax1 = plt.gca()
        plt.plot(data.index, data['Close'], label='Close', color='black', alpha=0.5)
        plt.plot(data.index, hma, label='HMA', color='blue', linewidth=2)
        plt.plot(data.index, upper_band, label='Upper Band', color='red', linestyle='--')
        plt.plot(data.index, lower_band, label='Lower Band', color='green', linestyle='--')
        if hma_signals:
            # 신호 마커 및 배경색 추가
            for i in range(len(hma_signals)):
                current_signal = hma_signals[i]
                current_date = current_signal['date']
                
                # 다음 신호의 날짜 찾기 (마지막 신호가 아닌 경우)
                if i < len(hma_signals) - 1:
                    next_date = hma_signals[i + 1]['date']
                else:
                    next_date = data.index[-1]
                
                # 배경색 추가 (현재 신호부터 다음 신호 전까지)
                if current_signal['type'] == 'BUY':
                    ax1.axvspan(current_date, next_date, color='green', alpha=0.065)
                else:
                    ax1.axvspan(current_date, next_date, color='red', alpha=0.065)
            
            # 신호 마커 및 텍스트 추가
            for idx, signal in enumerate(hma_signals):
                try:
                    # 색상 구분 (마커 색상)
                    if signal['type'] == 'BUY':
                        color = 'blue'  # 매수 마커 색상 통일
                    else:
                        color = 'red'  # 매도 마커 색상 통일
                    
                    price = signal['price']
                    offset = (abs(upper_band.max() - lower_band.min()) * 0.02)
                    if signal['type'] == 'BUY':
                        if signal['date'] in lower_band.index:
                            y = lower_band.loc[signal['date']] - offset
                        else:
                            y = price - offset
                        va = 'top'
                        text_y = y - offset * 1.2
                    else:
                        if signal['date'] in upper_band.index:
                            y = upper_band.loc[signal['date']] + offset
                        else:
                            y = price + offset
                        va = 'bottom'
                        text_y = y + offset * 1.2
                    
                    # 마커
                    ax1.plot(signal['date'], y, marker='^' if signal['type']=='BUY' else 'v', 
                            color=color, markersize=10, markeredgecolor='black', 
                            linestyle='None', label='매수 신호' if signal['type']=='BUY' else '매도 신호', 
                            zorder=10)
                    
                    # 텍스트
                    ax1.text(signal['date'], text_y, f"{signal['date'].strftime('%Y-%m-%d')}\n{signal.get('reason_short', '')}",
                            rotation=45, fontsize=7, color='black', ha='center', va=va,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', 
                                    boxstyle='round,pad=0.1'), zorder=11)
                    
                    # 수직선은 메인 차트에만 표시
                    ax1.axvline(x=signal['date'], color=color, linestyle='--', linewidth=0.3, alpha=0.8)
                    
                except Exception as e:
                    print(f"신호 마커 추가 중 오류 발생 (인덱스 {idx}):")
                    print(f"signal: {signal}")
                    print(f"예외 메시지: {str(e)}")
                    print("스택 트레이스:")
                    traceback.print_exc()
                    continue
            # 범례용 더미 객체
            buy_marker = mlines.Line2D([], [], color='green', marker='^', linestyle='None', markersize=3.6, label='Buy Signal')
            sell_marker = mlines.Line2D([], [], color='red', marker='v', linestyle='None', markersize=3.6, label='Sell Signal')
            plt.legend(handles=[buy_marker, sell_marker])
            plt.title(f'{ticker} - HMA & Mantra Bands')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(True, alpha=0.3)
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
    except Exception as e:
        print("HMA & 만트라 밴드 시각화 중 오류 발생:")
        print(f"예외 메시지: {str(e)}")
        print("스택 트레이스:")
        traceback.print_exc()

# 볼린저 밴드와 HMA/만트라 밴드 비교 시각화
def plot_comparison(data: pd.DataFrame, ticker: str, save_path: str = None) -> None:
    """볼린저 밴드와 HMA/만트라 밴드를 비교 시각화합니다."""
    try:
        hma = calculate_hma(data['Close'])
        upper_mantra, lower_band = calculate_mantra_bands(data['Close'])
        sma = data['Close'].rolling(window=20).mean()
        std = data['Close'].rolling(window=20).std()
        upper_bollinger = sma + (2 * std)
        lower_bollinger = sma - (2 * std)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
        ax1.plot(data.index, data['Close'], label='Close', color='black', alpha=0.5)
        ax1.plot(data.index, hma, label='HMA', color='blue', linewidth=2)
        ax1.plot(data.index, upper_mantra, label='Mantra Upper', color='red', linestyle='--')
        ax1.plot(data.index, lower_band, label='Mantra Lower', color='green', linestyle='--')
        hma_signals = get_hma_signals(data)
        price_range = data['High'].max() - data['Low'].min()
        offset = price_range * 0.03
        if hma_signals:
            for idx, signal in enumerate(hma_signals):
                try:
                    color = 'green' if signal['type'] == 'BUY' else 'red'
                    marker = '^' if signal['type'] == 'BUY' else 'v'
                    date = signal['date']
                    if signal['type'] == 'BUY':
                        band_val = lower_band.loc[date] if safe_in_index(lower_band.index, date) else signal['price']
                        y = band_val - offset
                        ax1.scatter(date, y, color=color, marker=marker, s=30, label=None)
                    else:
                        band_val = upper_mantra.loc[date] if safe_in_index(upper_mantra.index, date) else signal['price']
                        y = band_val + offset
                        ax1.scatter(date, y, color=color, marker=marker, s=30, label=None)
                except Exception as e:
                    print(f"신호 마커 추가 중 오류 발생 (인덱스 {idx}):")
                    print(f"signal: {signal}")
                    print(f"예외 메시지: {str(e)}")
                    print("스택 트레이스:")
                    traceback.print_exc()
                    continue
        # 범례용 더미 객체
        buy_marker = mlines.Line2D([], [], color='green', marker='^', linestyle='None', markersize=3.6, label='Buy Signal')
        sell_marker = mlines.Line2D([], [], color='red', marker='v', linestyle='None', markersize=3.6, label='Sell Signal')
        ax1.legend(handles=[buy_marker, sell_marker])
        ax1.set_title(f'{ticker} - HMA & Mantra Bands')
        ax1.set_ylabel('Price')
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
    except Exception as e:
        print("볼린저 밴드와 HMA/만트라 밴드 비교 시각화 중 오류 발생:")
        print(f"예외 메시지: {str(e)}")
        print("스택 트레이스:")
        traceback.print_exc()

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

# 안전한 float 변환 함수
def to_float(val):
    """안전한 float 변환 함수"""
    if hasattr(val, 'item'):
        return val.item()
    elif isinstance(val, (np.generic, np.ndarray)) and getattr(val, 'size', 1) == 1:
        return float(val)
    else:
        return float(val)

# 시그널 강도 시각화 (간단 버전)
def plot_signals_with_strength(data: pd.DataFrame, ticker: str, save_path: str = None) -> None:
    """신호와 강도를 함께 표시하는 차트를 그립니다."""
    try:
        # 데이터 인덱스를 datetime으로 통일
        data.index = pd.to_datetime(data.index)
        
        # 기본 지표 계산
        hma = calculate_hma(data['Close'])
        upper_mantra, lower_mantra = calculate_mantra_bands(data['Close'])
        hma_signals = get_hma_signals(data)
        
        # MACD 계산
        macd, signal_line, hist = calculate_macd(data['Close'])
        # RSI 계산
        rsi = calculate_rsi(data['Close'])
        # VIX 데이터 가져오기 및 인덱스 맞추기 (필요시)
        try:
            vix_data = yf.download('^VIX', start=data.index[0], end=data.index[-1])['Close']
            vix_data.index = pd.to_datetime(vix_data.index)
            vix = vix_data.reindex(data.index).ffill().bfill()
        except Exception as e:
            print(f"VIX 데이터 다운로드 실패: {e}")
            vix = pd.Series(index=data.index, data=np.nan)
        
        # 메인 차트만 생성
        fig, ax1 = plt.subplots(figsize=(15, 8))
        # ordinal_index를 numpy 1차원 배열로 변환
        ordinal_index = np.array(mdates.date2num(data.index.to_pydatetime())).flatten()
        
        # 캔들바 데이터 준비
        ohlc = [
            [mdates.date2num(date), float(row['Open'].iloc[0]), float(row['High'].iloc[0]), float(row['Low'].iloc[0]), float(row['Close'].iloc[0])]
            for date, row in data.iterrows()
        ]
        # 캔들바 추가
        candlestick_ohlc(ax1, ohlc, width=0.6, colorup='green', colordown='red', alpha=0.6)
        
        # 가격선, HMA, 만트라 밴드만 표시 (MA 선 제거)
        ax1.plot(ordinal_index, data['Close'], label='Close', color='black', linewidth=0.6, alpha=0.7)
        ax1.plot(ordinal_index, hma, label='HMA', color='blue', linewidth=2)
        ax1.plot(ordinal_index, upper_mantra, label='Mantra Upper', color='red', linestyle='--', linewidth=0.7)
        ax1.plot(ordinal_index, lower_mantra, label='Mantra Lower', color='green', linestyle='--', linewidth=0.7)
        
        # 볼린저 밴드 상단-중간, 하단-중간 영역 색상 채우기
        ax1.fill_between(data.index, upper_mantra, lower_mantra, where=(upper_mantra >= lower_mantra), color='orange', alpha=0.13, label='Mantra Zone')
        
        # 신호 마커 및 버티컬 라인 추가
        price_range = data['High'].max() - data['Low'].min()
        offset = price_range * 0.03
        if hma_signals:
            # 신호 마커 및 배경색 추가
            for i in range(len(hma_signals)):
                current_signal = hma_signals[i]
                current_date = current_signal['date']
                
                # 다음 신호의 날짜 찾기 (마지막 신호가 아닌 경우)
                if i < len(hma_signals) - 1:
                    next_date = hma_signals[i + 1]['date']
                else:
                    next_date = data.index[-1]
                
                # 배경색 추가 (현재 신호부터 다음 신호 전까지)
                if current_signal['type'] == 'BUY':
                    ax1.axvspan(current_date, next_date, color='green', alpha=0.065)
                else:
                    ax1.axvspan(current_date, next_date, color='red', alpha=0.065)
            
            # 신호 마커 및 텍스트 추가
            for idx, signal in enumerate(hma_signals):
                try:
                    # 색상 구분 (마커 색상)
                    if signal['type'] == 'BUY':
                        color = 'blue'  # 매수 마커 색상 통일
                    else:
                        color = 'red'  # 매도 마커 색상 통일
                    
                    price = signal['price']
                    offset = (abs(upper_mantra.max() - lower_mantra.min()) * 0.02)
                    if signal['type'] == 'BUY':
                        if signal['date'] in lower_mantra.index:
                            y = lower_mantra.loc[signal['date']] - offset
                        else:
                            y = price - offset
                        va = 'top'
                        text_y = y - offset * 1.2
                    else:
                        if signal['date'] in upper_mantra.index:
                            y = upper_mantra.loc[signal['date']] + offset
                        else:
                            y = price + offset
                        va = 'bottom'
                        text_y = y + offset * 1.2
                    
                    # 마커
                    ax1.plot(signal['date'], y, marker='^' if signal['type']=='BUY' else 'v', 
                            color=color, markersize=10, markeredgecolor='black', 
                            linestyle='None', label='매수 신호' if signal['type']=='BUY' else '매도 신호', 
                            zorder=10)
                    
                    # 텍스트
                    ax1.text(signal['date'], text_y, f"{signal['date'].strftime('%Y-%m-%d')}\n{signal.get('reason_short', '')}",
                            rotation=45, fontsize=7, color='black', ha='center', va=va,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', 
                                    boxstyle='round,pad=0.1'), zorder=11)
                    
                    # 수직선은 메인 차트에만 표시
                    ax1.axvline(x=signal['date'], color=color, linestyle='--', linewidth=0.3, alpha=0.8)
                    
                except Exception as e:
                    print(f"신호 마커 추가 중 오류 발생 (인덱스 {idx}):")
                    print(f"signal: {signal}")
                    print(f"예외 메시지: {str(e)}")
                    print("스택 트레이스:")
                    traceback.print_exc()
                    continue
        
        # 주요 라인 범례 (왼쪽 상단)
        line_handles = [
            mlines.Line2D([], [], color='blue', linewidth=2, label='HMA'),
            mlines.Line2D([], [], color='red', linestyle='--', linewidth=1, label='Mantra Upper'),
            mlines.Line2D([], [], color='green', linestyle='--', linewidth=1, label='Mantra Lower'),
            mpatches.Patch(color='green', alpha=0.065, label='매수구간'),
            mpatches.Patch(color='red', alpha=0.065, label='매도구간')
        ]
        legend1 = ax1.legend(handles=line_handles, loc='upper left', bbox_to_anchor=(0.02, 0.98),
                            fontsize=8, title='주요 라인', frameon=True, fancybox=True, borderpad=1)
        ax1.add_artist(legend1)
        
        # 신호 마커 범례 (왼쪽 중간)
        signal_handles = [
            mlines.Line2D([], [], color='blue', marker='^', linestyle='None', markersize=10,
                         markeredgecolor='black', label='매수 신호(▲)'),
            mlines.Line2D([], [], color='red', marker='v', linestyle='None', markersize=10,
                         markeredgecolor='black', label='매도 신호(▼)')
        ]
        legend2 = ax1.legend(handles=signal_handles, loc='upper left', bbox_to_anchor=(0.02, 0.75),
                            fontsize=8, title='신호 구분', frameon=True, fancybox=True, borderpad=1)
        ax1.add_artist(legend2)

        ax1.set_title(f'{ticker} - Technical Analysis')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        ax1.tick_params(axis='x', which='major', labelsize=9, rotation=45)
        ax1.tick_params(axis='x', which='minor', labelsize=7, rotation=45)
        
        # 차트 해석 텍스트 추가
        explanation = (
            '※ HMA/만트라 밴드 해석 및 실전 활용 주의사항\n'
            '- HMA가 평평하거나 하락일 때는 롱 포지션 진입 주의\n'
            '- 만트라 밴드는 스퀴즈(좁은 밴드)일 때 신호 신뢰도 낮아짐\n'
            '- 추세장에서는 만트라 밴드만으로는 부족 → 반드시 HMA와 병합'
        )
        fig.text(0.5, 0.96, explanation, ha='center', va='top', fontsize=10, color='dimgray', wrap=True)
        
        # 실제 데이터(캔들, 신호 마커 등)의 최소/최대 계산
        y_data_min = min(to_float(data['Low'].min()), to_float(lower_mantra.min()))
        y_data_max = max(to_float(data['High'].max()), to_float(upper_mantra.max()))
        y_data_range = y_data_max - y_data_min
        y_min = y_data_min - y_data_range * 0.25  # 하단 25%는 범례 영역
        y_max = y_data_max + y_data_range * 0.10  # 상단 10% 여유
        ax1.set_ylim(y_min, y_max)
        
        # 신호 마커 및 범례 추가
        for signal in hma_signals:
            marker = '^' if signal['type'] == 'BUY' else 'v'
            color = 'blue' if signal['type'] == 'BUY' else 'red'
            date = signal['date']
            price = signal['price']
            reason = signal.get('reason', '')
            offset = (abs(upper_mantra.max() - lower_mantra.min()) * 0.02)
            if signal['type'] == 'BUY':
                if date in lower_mantra.index:
                    y = lower_mantra.loc[date] - offset
                else:
                    y = price - offset
                va = 'top'
                text_y = y - offset * 1.2
            else:
                if date in upper_mantra.index:
                    y = upper_mantra.loc[date] + offset
                else:
                    y = price + offset
                va = 'bottom'
                text_y = y + offset * 1.2
            # 삼각형/역삼각형 마커
            ax1.plot(date, y, marker=marker, color=color, markersize=10, markeredgecolor='black', linestyle='None', label=reason, zorder=10)
            # 텍스트(근거 번호만)
            ax1.text(date, text_y, f"{date.strftime('%Y-%m-%d')}\n{reason}",
                rotation=45, fontsize=7, color='black', ha='center', va=va,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'), zorder=11)

        # 근거별 범례(중복 제거)
        legend_handles = [
            mlines.Line2D([], [], color='blue', marker='^', linestyle='None', markersize=10, markeredgecolor='black', label='B1: 종가 HMA 상향 돌파'),
            mlines.Line2D([], [], color='green', marker='^', linestyle='None', markersize=10, markeredgecolor='black', label='B2: 밴드 하단 상향 돌파'),
            mlines.Line2D([], [], color='red', marker='v', linestyle='None', markersize=10, markeredgecolor='black', label='T1: 종가 HMA 하향 돌파'),
            mlines.Line2D([], [], color='orange', marker='v', linestyle='None', markersize=10, markeredgecolor='black', label='T2: 상단밴드 하향 돌파'),
        ]
        ax1.legend(handles=legend_handles, loc='upper left', fontsize=8, title='주요 신호(근거별)')
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        fig = plt.gcf()
        fig.autofmt_xdate()
        ax1.tick_params(axis='x', which='both', labelbottom=True)
        # Y축 오른쪽만 표시(최종)
        ax1.yaxis.set_label_position('right')
        ax1.yaxis.tick_right()
        ax1.set_ylabel('Price', labelpad=10)
        ax1.yaxis.set_ticks_position('right')
        ax1.yaxis.set_tick_params(labelleft=False, labelright=True)
        # 메인 차트 x축: 일자별 수직선, y축: 가격별 수평선 추가 (tight_layout 이후, date2num 변환 적용)
        for x in data.index:
            ax1.axvline(x=mdates.date2num(x), color='lightgray', linestyle='-', linewidth=0.7, alpha=0.7, zorder=0)
        for y in ax1.get_yticks():
            ax1.axhline(y=y, color='lightgray', linestyle='-', linewidth=0.7, alpha=0.7, zorder=0)

        # 현재가 수평선(점선) 추가
        current_price = data['Close'].iloc[-1]
        ax1.axhline(current_price, color='black', linestyle='--', linewidth=1.2, alpha=0.8, zorder=1)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    except Exception as e:
        print("신호와 강도 시각화 중 오류 발생:")
        print(f"예외 메시지: {str(e)}")
        print("스택 트레이스:")
        traceback.print_exc()

def get_pixel_dy(ax, date_prev, hma_prev, date_now, hma_now):
    """두 데이터 좌표의 y축 픽셀 변화량을 반환합니다."""
    x1 = mdates.date2num(date_prev)
    x2 = mdates.date2num(date_now)
    pixel_xy1 = ax.transData.transform((x1, hma_prev))
    pixel_xy2 = ax.transData.transform((x2, hma_now))
    pixel_dy = pixel_xy2[1] - pixel_xy1[1]
    return pixel_dy

def get_hma_mantra_md_signals(data: pd.DataFrame, ticker: str = None) -> list:
    """
    매수/매도 신호 생성 (매도 신호 조건 사용자 정의 반영, 불필요한 변수 및 코드 정리)
    """
    signals = []
    
    # 데이터 프레임 구조 확인 및 처리
    if isinstance(data.columns, pd.MultiIndex):
        close_data = data[('Close', ticker)]
    else:
        close_data = data['Close']
    
    hma = calculate_hma(close_data)
    upper_band, lower_band = calculate_mantra_bands(close_data)
    rsi3 = calculate_rsi(close_data, period=3)
    rsi14 = calculate_rsi(close_data, period=14)
    # 기존 매수 신호에 필요한 변수만 선언
    macd, macd_signal, _ = calculate_macd(close_data)
    rsi = calculate_rsi(close_data)
    
    for i in range(1, len(data)):
        close = float(close_data.iloc[i])
        prev_close = float(close_data.iloc[i-1])
        hma_now = float(hma.iloc[i])
        hma_prev = float(hma.iloc[i-1])
        upper = float(upper_band.iloc[i])
        upper_prev = float(upper_band.iloc[i-1])
        rsi3_now = float(rsi3.iloc[i])
        rsi14_now = float(rsi14.iloc[i])
        
        # --- 매도 신호 조건 ---
        if (prev_close > hma_prev) and (close < hma_now) and (rsi3_now < rsi14_now):
            signals.append({
                'date': data.index[i],
                'type': 'SELL',
                'price': close,
                'reason': '종가 HMA 하향 돌파 + 타지마할 RSI 매도구간(rsi3 < rsi14)',
                'reason_short': 'T1'
            })
            continue
        
        if (prev_close > upper_prev) and (close < upper) and (rsi3_now < rsi14_now):
            signals.append({
                'date': data.index[i],
                'type': 'SELL',
                'price': close,
                'reason': '상단밴드 하향 돌파 + 타지마할 RSI 매도구간(rsi3 < rsi14)',
                'reason_short': 'T2'
            })
            continue
        
        # --- 매수 신호 조건(변경) ---
        # 1) 종가 HMA 상향 돌파 + 타지마할 RSI 매수 구간
        if (prev_close < hma_prev) and (close >= hma_now) and (rsi3_now >= rsi14_now):
            signals.append({
                'date': data.index[i],
                'type': 'BUY',
                'price': close,
                'reason': '종가 HMA 상향 돌파 + 타지마할 RSI 매수구간(rsi3 >= rsi14)',
                'reason_short': 'B1'
            })
            continue
        
        # 2) 전일 종가 < 밴드 하단, 당일 종가 > 밴드 하단 + 타지마할 RSI 매수 구간
        lower = float(lower_band.iloc[i])
        lower_prev = float(lower_band.iloc[i-1])
        if (prev_close < lower_prev) and (close >= lower) and (rsi3_now >= rsi14_now):
            signals.append({
                'date': data.index[i],
                'type': 'BUY',
                'price': close,
                'reason': '밴드 하단 상향 돌파 + 타지마할 RSI 매수구간(rsi3 >= rsi14)',
                'reason_short': 'B2'
            })
            continue
        
        # 3) 종가 HMA 상향 돌파 + 타지마할 RSI 매수구간 아님 + MACD 매수 구간
        if (prev_close < hma_prev) and (close >= hma_now) and (rsi3_now < rsi14_now) and (macd.iloc[i] > macd_signal.iloc[i]):
            signals.append({
                'date': data.index[i],
                'type': 'BUY',
                'price': close,
                'reason': '종가 HMA 상향 돌파 + 타지마할 RSI 매수구간 아님 + MACD 매수 구간',
                'reason_short': 'B3'
            })
            continue
    
    return signals

def get_signal_facecolor(signal):
    """신호 타입과 원인에 따른 마커 색상을 반환합니다."""
    if signal['type'] == 'BUY':
        color_map = {
            'C': '#FFD700',  # 금색
            'H': '#1E90FF',  # 도저블루
            'M': '#9370DB',  # 미디엄 퍼플
            'R': '#32CD32',  # 라임 그린
        }
    else:  # SELL
        color_map = {
            'C': '#FFA500',  # 주황색
            'H': '#000080',  # 네이비
            'M': '#DC143C',  # 크림슨
            'R': '#8B4513',  # 갈색
        }
    # reason_short가 있으면 우선 사용, 없으면 기존 reason에서 첫 글자 추출
    key = signal.get('reason_short') or (signal.get('reason','C')[0] if signal.get('reason') else 'C')
    return color_map.get(key, '#FFA500')  # 기본값은 주황색

def get_krx_foreign_trading(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """KRX에서 외국인 거래량 데이터를 가져옵니다."""
    try:
        # KRX OpenAPI URL
        url = "http://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService/getStockPriceInfo"
        
        # API 키 (실제 사용시 발급받은 키로 교체 필요)
        service_key = "YOUR_SERVICE_KEY"  # 실제 서비스 키로 교체 필요
        
        # 요청 파라미터
        params = {
            "serviceKey": service_key,
            "resultType": "json",
            "beginBasDt": start_date,
            "endBasDt": end_date,
            "itmsNm": ticker
        }
        
        # 데이터 요청
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if 'response' in data and 'body' in data['response']:
                items = data['response']['body']['items']['item']
                
                # 데이터 추출
                foreign_data = []
                for item in items:
                    foreign_data.append({
                        'date': item['basDt'],
                        'foreign_buy': float(item['frgnBuyAmt'].replace(',', '')),
                        'foreign_sell': float(item['frgnSellAmt'].replace(',', '')),
                        'foreign_net': float(item['frgnNetAmt'].replace(',', ''))
                    })
                
                # DataFrame 생성
                df = pd.DataFrame(foreign_data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                return df
            else:
                print("데이터 형식이 예상과 다릅니다.")
                return pd.DataFrame()
        else:
            print(f"KRX 데이터 요청 실패: {response.status_code}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"외국인 거래량 데이터 가져오기 실패: {str(e)}")
        return pd.DataFrame()

def plot_hma_mantra_md_signals(data: pd.DataFrame, ticker: str, save_path: str = None) -> None:
    try:
        # 데이터 인덱스를 datetime으로 통일
        data.index = pd.to_datetime(data.index)
        
        # 데이터 프레임 구조 확인 및 처리
        if isinstance(data.columns, pd.MultiIndex):
            close_data = data[('Close', ticker)]
            open_data = data[('Open', ticker)]
            high_data = data[('High', ticker)]
            low_data = data[('Low', ticker)]
            volume_data = data[('Volume', ticker)]
        else:
            close_data = data['Close']
            open_data = data['Open']
            high_data = data['High']
            low_data = data['Low']
            volume_data = data['Volume']
        
        # 20일 박스권 계산
        last_20_high = high_data.rolling(window=20).max()
        last_20_low = low_data.rolling(window=20).min()
        current_resistance = last_20_high.iloc[-1]
        current_support = last_20_low.iloc[-1]
        current_price = close_data.iloc[-1]
        
        # 기본 지표 계산
        hma = calculate_hma(close_data)
        upper_mantra, lower_mantra = calculate_mantra_bands(close_data)
        hma_signals = get_hma_mantra_md_signals(data, ticker)
        
        # 볼린저 밴드 계산
        sma20 = calculate_ma(close_data, 20)
        bb_std = close_data.rolling(window=20).std()
        bb_upper = sma20 + (bb_std * 2)
        bb_lower = sma20 - (bb_std * 2)
        
        # MACD 계산
        macd, signal_line, hist = calculate_macd(close_data)
        
        # RSI 계산 (타지마할 RSI: 3, 14, 50일)
        rsi3 = calculate_rsi(close_data, period=3)
        rsi14 = calculate_rsi(close_data, period=14)
        rsi50 = calculate_rsi(close_data, period=50)
        
        # 추가 지표 데이터 다운로드
        try:
            # VIX 데이터
            vix_data = yf.download('^VIX', start=data.index[0], end=data.index[-1])['Close']
            vix_data = vix_data.reindex(data.index).ffill()
            
            # 10년물 금리
            tnx_data = yf.download('^TNX', start=data.index[0], end=data.index[-1])['Close']
            tnx_data = tnx_data.reindex(data.index).ffill()
            
            # 달러인덱스
            dxy_data = yf.download('DX-Y.NYB', start=data.index[0], end=data.index[-1])['Close']
            dxy_data = dxy_data.reindex(data.index).ffill()
            
            # 금가격
            gold_data = yf.download('GC=F', start=data.index[0], end=data.index[-1])['Close']
            gold_data = gold_data.reindex(data.index).ffill()
        except Exception as e:
            print(f"추가 지표 데이터 다운로드 실패: {e}")
            vix_data = pd.Series(index=data.index, data=np.nan)
            tnx_data = pd.Series(index=data.index, data=np.nan)
            dxy_data = pd.Series(index=data.index, data=np.nan)
            gold_data = pd.Series(index=data.index, data=np.nan)
        
        # 차트 생성 (8개의 서브플롯)
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(8, 1, figsize=(15, 20), 
            height_ratios=[3, 1, 1, 1, 1, 1, 1, 1], sharex=True)
        
        # 배경색 설정
        fig.patch.set_facecolor('white')
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
            ax.set_facecolor('white')
        
        # 캔들바 데이터 준비
        ohlc = []
        for date, row in data.iterrows():
            if isinstance(data.columns, pd.MultiIndex):
                ohlc.append([
                    mdates.date2num(date),
                    float(row[('Open', ticker)]),
                    float(row[('High', ticker)]),
                    float(row[('Low', ticker)]),
                    float(row[('Close', ticker)])
                ])
            else:
                ohlc.append([
                    mdates.date2num(date),
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close'])
                ])
        
        # 메인 차트 (캔들차트)
        candlestick_ohlc(ax1, ohlc, width=0.6, colorup='green', colordown='red', alpha=0.6)
        
        # 20일 박스권 지지선/저항선 추가
        ax1.axhline(y=current_resistance, color='red', linestyle='--', linewidth=1, alpha=0.8, 
                   label=f'저항선: {current_resistance:.2f}')
        ax1.axhline(y=current_support, color='green', linestyle='--', linewidth=1, alpha=0.8,
                   label=f'지지선: {current_support:.2f}')
        
        # 저항선/지지선 가격 텍스트 추가
        ax1.text(data.index[-1], current_resistance, f'{current_resistance:.2f}', 
                color='red', fontsize=6, rotation=45, ha='left', va='bottom',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))
        ax1.text(data.index[-1], current_support, f'{current_support:.2f}', 
                color='green', fontsize=6, rotation=45, ha='left', va='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))

        # 현재가 수평선 및 텍스트 추가
        ax1.axhline(y=current_price, color='black', linestyle='-', linewidth=0.8, alpha=0.8)
        ax1.text(data.index[-1], current_price, f'{current_price:.2f}', 
                color='black', fontsize=6, rotation=45, ha='left', va='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))
        
        # 주요 선 추가
        ax1.plot(data.index, close_data, label='Close', color='black', linewidth=0.6, alpha=0.7)
        ax1.plot(data.index, hma, label='HMA', color='blue', linewidth=2)
        ax1.plot(data.index, upper_mantra, label='Mantra Upper', color='red', linestyle='--', linewidth=0.7)
        ax1.plot(data.index, lower_mantra, label='Mantra Lower', color='green', linestyle='--', linewidth=0.7)
        ax1.plot(data.index, bb_upper, label='BB Upper', color='orange', linestyle=':', linewidth=0.7)
        ax1.plot(data.index, bb_lower, label='BB Lower', color='orange', linestyle=':', linewidth=0.7)
        ax1.plot(data.index, sma20, label='SMA20', color='purple', linestyle=':', linewidth=0.7)
        
        # 만트라 밴드 영역 색상 채우기
        ax1.fill_between(data.index, upper_mantra, lower_mantra, 
                        where=(upper_mantra >= lower_mantra), 
                        color='orange', alpha=0.1)
        
        # 신호 마커 및 버티컬 라인 추가
        price_range = high_data.max() - low_data.min()
        offset = price_range * 0.03
        if hma_signals:
            # 신호 마커 및 배경색 추가
            for i in range(len(hma_signals)):
                current_signal = hma_signals[i]
                current_date = current_signal['date']
                
                # 다음 신호의 날짜 찾기 (마지막 신호가 아닌 경우)
                if i < len(hma_signals) - 1:
                    next_date = hma_signals[i + 1]['date']
                else:
                    next_date = data.index[-1]
                
                # 배경색 추가 (현재 신호부터 다음 신호 전까지) - 모든 서브플롯에 적용
                if current_signal['type'] == 'BUY':
                    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
                        ax.axvspan(current_date, next_date, color='green', alpha=0.065)
                else:
                    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
                        ax.axvspan(current_date, next_date, color='red', alpha=0.065)
            
            # 신호 마커 및 텍스트 추가
            for idx, signal in enumerate(hma_signals):
                try:
                    # 색상 구분 (마커 색상)
                    if signal['type'] == 'BUY':
                        color = 'blue'  # 매수 마커 색상 통일
                    else:
                        color = 'red'  # 매도 마커 색상 통일
                    
                    price = signal['price']
                    offset = (abs(upper_mantra.max() - lower_mantra.min()) * 0.02)
                    if signal['type'] == 'BUY':
                        if signal['date'] in lower_mantra.index:
                            y = lower_mantra.loc[signal['date']] - offset
                        else:
                            y = price - offset
                        va = 'top'
                        text_y = y - offset * 1.2
                        
                        # RSI와 MACD 매수 신호 확인
                        date_idx = data.index.get_loc(signal['date'])
                        rsi_buy = rsi3.iloc[date_idx] >= rsi14.iloc[date_idx]
                        macd_buy = macd.iloc[date_idx] > signal_line.iloc[date_idx]
                        
                        # 매수 마커 표시 (RSI와 MACD 모두 매수일 때 별표 추가)
                        ax1.plot(signal['date'], y, marker='^', 
                                color=color, markersize=10, markeredgecolor='black', 
                                linestyle='None', label='매수 신호', zorder=10)
                        
                        if rsi_buy and macd_buy:
                            # 별표 추가
                            star_y = y + offset * 1.5
                            ax1.plot(signal['date'], star_y, marker='*', 
                                    color='gold', markersize=12, markeredgecolor='black',
                                    linestyle='None', zorder=11)
                    else:
                        if signal['date'] in upper_mantra.index:
                            y = upper_mantra.loc[signal['date']] + offset
                        else:
                            y = price + offset
                        va = 'bottom'
                        text_y = y + offset * 1.2
                        
                        # 매도 마커 표시
                        ax1.plot(signal['date'], y, marker='v', 
                                color=color, markersize=10, markeredgecolor='black', 
                                linestyle='None', label='매도 신호', zorder=10)
                    
                    # 텍스트 표시
                    ax1.text(signal['date'], text_y, f"{signal['date'].strftime('%Y-%m-%d')}\n{signal.get('reason_short', '')}",
                            rotation=45, fontsize=7, color='black', ha='center', va=va,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', 
                                    boxstyle='round,pad=0.1'), zorder=11)
                    
                    # 수직선은 메인 차트에만 표시
                    ax1.axvline(x=signal['date'], color=color, linestyle='--', linewidth=0.3, alpha=0.8)
                    
                except Exception as e:
                    print(f"신호 마커 추가 중 오류 발생 (인덱스 {idx}):")
                    print(f"signal: {signal}")
                    print(f"예외 메시지: {str(e)}")
                    print("스택 트레이스:")
                    traceback.print_exc()
                    continue
        
        # 타지마할 RSI 서브플롯
        ax2.plot(data.index, rsi3, label='RSI(3)', color='blue', linewidth=1)
        ax2.plot(data.index, rsi14, label='RSI(14)', color='red', linewidth=1)
        ax2.plot(data.index, rsi50, label='RSI(50)', color='green', linewidth=1)
        ax2.axhline(y=70, color='red', linestyle='--', linewidth=0.5)
        ax2.axhline(y=30, color='green', linestyle='--', linewidth=0.5)

        # RSI 매수/매도 신호 표시
        for i in range(1, len(data)):
            if rsi3.iloc[i] >= rsi14.iloc[i] and rsi3.iloc[i-1] < rsi14.iloc[i-1]:
                ax2.plot(data.index[i], rsi3.iloc[i], '^', color='blue', markersize=8, 
                        label='RSI 매수신호' if i == 1 else '')
            elif rsi3.iloc[i] < rsi14.iloc[i] and rsi3.iloc[i-1] >= rsi14.iloc[i-1]:
                ax2.plot(data.index[i], rsi3.iloc[i], 'v', color='red', markersize=8, 
                        label='RSI 매도신호' if i == 1 else '')

        # RSI 범례 추가
        rsi_handles = [
            mlines.Line2D([], [], color='blue', label='RSI(3)'),
            mlines.Line2D([], [], color='red', label='RSI(14)'),
            mlines.Line2D([], [], color='green', label='RSI(50)'),
            mlines.Line2D([], [], color='blue', marker='^', linestyle='None', 
                         markersize=8, label='RSI 매수신호'),
            mlines.Line2D([], [], color='red', marker='v', linestyle='None', 
                         markersize=8, label='RSI 매도신호')
        ]
        ax2.legend(handles=rsi_handles, loc='upper left', fontsize=8)

        ax2.set_ylabel('타지마할 RSI')
        ax2.grid(True, alpha=0.3)
        
        # MACD 서브플롯
        ax3.plot(data.index, macd, label='MACD', color='blue', linewidth=1)
        ax3.plot(data.index, signal_line, label='Signal', color='red', linewidth=1)
        ax3.bar(data.index, hist, label='Histogram', 
                color=['red' if h < 0 else 'green' for h in hist], alpha=0.5)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # MACD 매수/매도 신호 표시
        for i in range(1, len(data)):
            if macd.iloc[i] > signal_line.iloc[i] and macd.iloc[i-1] <= signal_line.iloc[i-1]:
                ax3.plot(data.index[i], macd.iloc[i], '^', color='blue', markersize=8,
                        label='MACD 매수신호' if i == 1 else '')
            elif macd.iloc[i] < signal_line.iloc[i] and macd.iloc[i-1] >= signal_line.iloc[i-1]:
                ax3.plot(data.index[i], macd.iloc[i], 'v', color='red', markersize=8,
                        label='MACD 매도신호' if i == 1 else '')

        # MACD 범례 추가
        macd_handles = [
            mlines.Line2D([], [], color='blue', label='MACD'),
            mlines.Line2D([], [], color='red', label='Signal'),
            mpatches.Patch(color='green', alpha=0.5, label='Histogram(+)'),
            mpatches.Patch(color='red', alpha=0.5, label='Histogram(-)'),
            mlines.Line2D([], [], color='blue', marker='^', linestyle='None',
                         markersize=8, label='MACD 매수신호'),
            mlines.Line2D([], [], color='red', marker='v', linestyle='None',
                         markersize=8, label='MACD 매도신호')
        ]
        ax3.legend(handles=macd_handles, loc='upper left', fontsize=8)

        ax3.set_ylabel('MACD')
        ax3.grid(True, alpha=0.3)
        
        # 거래량 서브플롯
        ax4.bar(data.index, volume_data, 
                color=['red' if c < o else 'green' for c, o in zip(close_data, open_data)],
                alpha=0.5)

        # 1일 평균 거래량 계산 및 수평선 추가
        avg_volume = volume_data.mean()
        ax4.axhline(y=avg_volume, color='blue', linestyle='--', linewidth=1.5, label=f'1일 평균 거래량: {format(int(avg_volume), ",")}')

        # 거래량 범례 추가
        volume_handles = [
            mpatches.Patch(color='green', alpha=0.5, label='상승 거래량'),
            mpatches.Patch(color='red', alpha=0.5, label='하락 거래량'),
            mlines.Line2D([], [], color='blue', linestyle='--', linewidth=1.5, 
                         label=f'1일 평균 거래량: {format(int(avg_volume), ",")}')
        ]
        ax4.legend(handles=volume_handles, loc='upper left', fontsize=8)

        ax4.set_ylabel('거래량')
        ax4.grid(True, alpha=0.3)
        
        # VIX 서브플롯
        ax5.plot(data.index, vix_data, label='VIX', color='purple', linewidth=1)

        # 현재 VIX 값 계산 및 수평선 추가 (실선으로 변경, 두께 0.5로 감소)
        current_vix = float(vix_data.iloc[-1])
        ax5.axhline(y=current_vix, color='black', linestyle='-', linewidth=0.5, alpha=0.8)

        # 현재 VIX 위치에 점 표시
        ax5.plot(data.index[-1], current_vix, 'o', color='black', markersize=6, 
                 markeredgecolor='white', markeredgewidth=1, zorder=5)

        # 현재 VIX 값 텍스트 표시 (교차점 위에 45도로 표시)
        ax5.text(data.index[-1], current_vix, 
                 f'{current_vix:.2f}', 
                 color='black', fontsize=6,
                 ha='left', va='bottom',
                 rotation=45,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5),
                 zorder=6)

        # VIX 전략 기준선 추가
        ax5.axhline(y=30, color='green', linestyle='--', linewidth=1, alpha=0.8)
        ax5.axhline(y=20, color='yellow', linestyle='--', linewidth=1, alpha=0.8)
        ax5.axhline(y=12, color='red', linestyle='--', linewidth=1, alpha=0.8)

        # VIX 구간별 배경색 표시
        ax5.fill_between(data.index, 30, ax5.get_ylim()[1], color='green', alpha=0.1, label='과도한 공포 구간 (매수기회)')
        ax5.fill_between(data.index, 20, 30, color='yellow', alpha=0.1, label='공포 구간')
        ax5.fill_between(data.index, 0, 12, color='red', alpha=0.1, label='과도한 낙관 구간 (매도기회)')

        # VIX 범례 추가
        vix_handles = [
            mlines.Line2D([], [], color='purple', linewidth=1, label='VIX'),
            mlines.Line2D([], [], color='black', linestyle='-', linewidth=0.5, 
                         label=f'현재 VIX: {current_vix:.2f}'),
            mpatches.Patch(color='green', alpha=0.1, label='과도한 공포 구간 (VIX > 30, 매수기회)'),
            mpatches.Patch(color='yellow', alpha=0.1, label='공포 구간 (20 < VIX < 30)'),
            mpatches.Patch(color='red', alpha=0.1, label='과도한 낙관 구간 (VIX < 12, 매도기회)')
        ]
        ax5.legend(handles=vix_handles, loc='upper left', fontsize=8)

        ax5.set_ylabel('VIX')
        ax5.grid(True, alpha=0.3)
        
        # 10년물 금리 서브플롯
        ax6.plot(data.index, tnx_data, label='10Y Treasury', color='green', linewidth=1)

        # 현재 금리 값 계산 및 수평선 추가
        current_tnx = float(tnx_data.iloc[-1])
        ax6.axhline(y=current_tnx, color='black', linestyle='-', linewidth=0.5, alpha=0.8)

        # 현재 금리 위치에 점 표시
        ax6.plot(data.index[-1], current_tnx, 'o', color='black', markersize=6, 
                 markeredgecolor='white', markeredgewidth=1, zorder=5)

        # 현재 금리 값 텍스트 표시
        ax6.text(data.index[-1], current_tnx, 
                 f'{current_tnx:.2f}', 
                 color='black', fontsize=6,
                 ha='left', va='bottom',
                 rotation=45,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5),
                 zorder=6)

        # 범례 추가
        tnx_handles = [
            mlines.Line2D([], [], color='green', linewidth=1, label='10Y Treasury'),
            mlines.Line2D([], [], color='black', linestyle='-', linewidth=0.5, 
                         label=f'현재 금리: {current_tnx:.2f}%')
        ]
        ax6.legend(handles=tnx_handles, loc='upper left', fontsize=8)

        ax6.set_ylabel('10년물 금리')
        ax6.grid(True, alpha=0.3)
        
        # 달러인덱스 서브플롯
        ax7.plot(data.index, dxy_data, label='Dollar Index', color='blue', linewidth=1)

        # 현재 달러인덱스 값 계산 및 수평선 추가
        current_dxy = float(dxy_data.iloc[-1])
        ax7.axhline(y=current_dxy, color='black', linestyle='-', linewidth=0.5, alpha=0.8)

        # 현재 달러인덱스 위치에 점 표시
        ax7.plot(data.index[-1], current_dxy, 'o', color='black', markersize=6, 
                 markeredgecolor='white', markeredgewidth=1, zorder=5)

        # 현재 달러인덱스 값 텍스트 표시
        ax7.text(data.index[-1], current_dxy, 
                 f'{current_dxy:.2f}', 
                 color='black', fontsize=6,
                 ha='left', va='bottom',
                 rotation=45,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5),
                 zorder=6)

        # 범례 추가
        dxy_handles = [
            mlines.Line2D([], [], color='blue', linewidth=1, label='Dollar Index'),
            mlines.Line2D([], [], color='black', linestyle='-', linewidth=0.5, 
                         label=f'현재 달러인덱스: {current_dxy:.2f}')
        ]
        ax7.legend(handles=dxy_handles, loc='upper left', fontsize=8)

        ax7.set_ylabel('달러인덱스')
        ax7.grid(True, alpha=0.3)
        
        # 금가격 서브플롯
        ax8.plot(data.index, gold_data, label='Gold', color='gold', linewidth=1)

        # 현재 금가격 값 계산 및 수평선 추가
        current_gold = float(gold_data.iloc[-1])
        ax8.axhline(y=current_gold, color='black', linestyle='-', linewidth=0.5, alpha=0.8)

        # 현재 금가격 위치에 점 표시
        ax8.plot(data.index[-1], current_gold, 'o', color='black', markersize=6, 
                 markeredgecolor='white', markeredgewidth=1, zorder=5)

        # 현재 금가격 값 텍스트 표시
        ax8.text(data.index[-1], current_gold, 
                 f'{current_gold:.2f}', 
                 color='black', fontsize=6,
                 ha='left', va='bottom',
                 rotation=45,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5),
                 zorder=6)

        # 범례 추가
        gold_handles = [
            mlines.Line2D([], [], color='gold', linewidth=1, label='Gold'),
            mlines.Line2D([], [], color='black', linestyle='-', linewidth=0.5, 
                         label=f'현재 금가격: {current_gold:.2f}')
        ]
        ax8.legend(handles=gold_handles, loc='upper left', fontsize=8)

        ax8.set_ylabel('금가격')
        ax8.grid(True, alpha=0.3)
        
        # 주요 라인 범례 (왼쪽 상단)
        line_handles = [
            mlines.Line2D([], [], color='blue', linewidth=2, label='HMA'),
            mlines.Line2D([], [], color='red', linestyle='--', linewidth=1, label='Mantra Upper'),
            mlines.Line2D([], [], color='green', linestyle='--', linewidth=1, label='Mantra Lower'),
            mlines.Line2D([], [], color='red', linestyle='--', linewidth=1, label=f'저항선: {current_resistance:.2f}'),
            mlines.Line2D([], [], color='green', linestyle='--', linewidth=1, label=f'지지선: {current_support:.2f}'),
            mlines.Line2D([], [], color='black', linestyle='-', linewidth=0.8, label=f'현재가: {current_price:.2f}'),
            mpatches.Patch(color='green', alpha=0.065, label='매수구간'),
            mpatches.Patch(color='red', alpha=0.065, label='매도구간')
        ]
        legend1 = ax1.legend(handles=line_handles, loc='upper left', bbox_to_anchor=(0.02, 0.98),
                            fontsize=8, title='주요 라인', frameon=True, fancybox=True, borderpad=1)
        ax1.add_artist(legend1)
        
        # 신호 마커 범례 (왼쪽 중간)
        signal_handles = [
            mlines.Line2D([], [], color='blue', marker='^', linestyle='None', markersize=10,
                         markeredgecolor='black', label='매수 신호(▲)'),
            mlines.Line2D([], [], color='red', marker='v', linestyle='None', markersize=10,
                         markeredgecolor='black', label='매도 신호(▼)')
        ]
        legend2 = ax1.legend(handles=signal_handles, loc='upper left', bbox_to_anchor=(0.02, 0.75),
                            fontsize=8, title='신호 구분', frameon=True, fancybox=True, borderpad=1)
        ax1.add_artist(legend2)
        
        # 신호 근거 범례 (좌측 하단)
        signal_reason_handles = [
            mlines.Line2D([], [], color='blue', marker='^', linestyle='None', markersize=10,
                         markeredgecolor='black', label='B1: 종가 HMA 상향 돌파 + 타지마할 RSI 매수구간'),
            mlines.Line2D([], [], color='green', marker='^', linestyle='None', markersize=10,
                         markeredgecolor='black', label='B2: 밴드 하단 상향 돌파 + 타지마할 RSI 매수구간'),
            mlines.Line2D([], [], color='deepskyblue', marker='^', linestyle='None', markersize=10,
                         markeredgecolor='black', label='B3: 종가 HMA 상향 돌파 + 타지마할 RSI 매수구간 아님 + MACD 매수 구간'),
            mlines.Line2D([], [], color='red', marker='v', linestyle='None', markersize=10,
                         markeredgecolor='black', label='T1: 종가 HMA 하향 돌파 + 타지마할 RSI 매도구간'),
            mlines.Line2D([], [], color='orange', marker='v', linestyle='None', markersize=10,
                         markeredgecolor='black', label='T2: 상단밴드 하향 돌파 + 타지마할 RSI 매도구간'),
        ]
        ax1.legend(handles=signal_reason_handles, loc='lower left', bbox_to_anchor=(0.02, 0.02),
                            fontsize=8, title='신호 근거', frameon=True, fancybox=True, borderpad=1)
        
        # 차트 해설 텍스트 추가
        explanation = (
            '※ HMA/만트라 밴드 해석 및 실전 활용 주의사항\n'
            '- HMA가 평평하거나 하락일 때는 롱 포지션 진입 주의\n'
            '- 만트라 밴드는 스퀴즈(좁은 밴드)일 때 신호 신뢰도 낮아짐\n'
            '- 추세장에서는 만트라 밴드만으로는 부족 → 반드시 HMA와 병합\n'
            '- RSI(3)가 RSI(14)를 상향돌파: 매수 시그널 강화\n'
            '- RSI(3)가 RSI(14)를 하향돌파: 매도 시그널 강화\n'
            '- VIX, 금리, 달러, 금가격 변동성 참고'
        )
        
        # 차트 제목 설정 (타이틀 위치 조정)
        title = f'{ticker} - Technical Analysis ({data.index[0].strftime("%Y-%m-%d")} ~ {data.index[-1].strftime("%Y-%m-%d")})'
        fig.suptitle(title, y=0.92, fontsize=12)
        
        # 차트 레이아웃 설정
        plt.tight_layout(rect=[0, 0.02, 1, 0.90])
        
        # x축 날짜 포맷 설정 (마지막 subplot에만 적용)
        ax8.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax8.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax8.tick_params(axis='x', rotation=45, labelsize=8)

        # 다른 subplot들의 x축 레이블 숨기기
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
            ax.tick_params(axis='x', labelbottom=False)

        # 모든 subplot의 격자선 스타일 설정
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
            ax.grid(True, color='lightgray', alpha=0.3, linestyle='-', linewidth=0.5)
            ax.yaxis.grid(True, color='lightgray', alpha=0.3, linestyle='-', linewidth=0.5)
        
        # x축 레이블 간격 조정
        plt.gcf().autofmt_xdate(rotation=45, ha='right')

        # subplot 간격 조정
        plt.subplots_adjust(hspace=0.3)  # subplot 간 간격 조정
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        else:
            plt.show()
    except Exception as e:
        print("HMA & 만트라 밴드 시각화 중 오류 발생:")
        print(f"예외 메시지: {str(e)}")
        print("스택 트레이스:")
        traceback.print_exc()

# --- 매수형 캔들 패턴 탐지 및 마킹 ---
def to_series(val, index):
    if isinstance(val, pd.Series):
        return val
    elif isinstance(val, pd.DataFrame):
        # DataFrame이면 첫 번째 컬럼만 Series로 변환
        return val.iloc[:, 0]
    elif np.isscalar(val):
        return pd.Series(np.full(len(index), val), index=index)
    elif isinstance(val, (np.ndarray, list)):
        arr = np.asarray(val).ravel()
        if arr.size == len(index):
            return pd.Series(arr, index=index)
        elif arr.size == 1:
            return pd.Series(np.full(len(index), arr[0]), index=index)
        else:
            raise ValueError(f"Cannot broadcast array to Series of length {len(index)}")
    else:
        raise ValueError(f"Unsupported type for to_series: {type(val)}")

def detect_hammer(df, ticker=None):
    open_ = df['Open'] if 'Open' in df else df[('Open', ticker)]
    close_ = df['Close'] if 'Close' in df else df[('Close', ticker)]
    low_ = df['Low'] if 'Low' in df else df[('Low', ticker)]
    high_ = df['High'] if 'High' in df else df[('High', ticker)]
    open_ = to_series(open_, df.index)
    close_ = to_series(close_, df.index)
    low_ = to_series(low_, df.index)
    high_ = to_series(high_, df.index)
    if len(df) < 2:
        return pd.Series([False]*len(df), index=df.index)
    body = abs(close_ - open_)
    min_oc = pd.DataFrame({'open': open_, 'close': close_}).min(axis=1)
    max_oc = pd.DataFrame({'open': open_, 'close': close_}).max(axis=1)
    lower_shadow = min_oc - low_
    upper_shadow = high_ - max_oc
    return (lower_shadow > 2 * body) & (upper_shadow < body)

def detect_bullish_engulfing(df, ticker=None):
    open_ = df['Open'] if 'Open' in df else df[('Open', ticker)]
    close_ = df['Close'] if 'Close' in df else df[('Close', ticker)]
    prev_open = open_.shift(1)
    prev_close = close_.shift(1)
    return (prev_close < prev_open) & (close_ > open_) & (open_ < prev_close) & (close_ > prev_open)

def detect_bullish_harami(df, ticker=None):
    open_ = df['Open'] if 'Open' in df else df[('Open', ticker)]
    close_ = df['Close'] if 'Close' in df else df[('Close', ticker)]
    prev_open = open_.shift(1)
    prev_close = close_.shift(1)
    return (prev_close > prev_open) & (open_ > close_) & (open_ > prev_close) & (close_ < prev_open)

def detect_piercing_line(df, ticker=None):
    open_ = df['Open'] if 'Open' in df else df[('Open', ticker)]
    close_ = df['Close'] if 'Close' in df else df[('Close', ticker)]
    prev_open = open_.shift(1)
    prev_close = close_.shift(1)
    return (prev_close < prev_open) & (open_ < prev_close) & (close_ > (prev_open + prev_close) / 2) & (close_ < prev_open)

def detect_morning_star(df, ticker=None):
    open_ = df['Open'] if 'Open' in df else df[('Open', ticker)]
    close_ = df['Close'] if 'Close' in df else df[('Close', ticker)]
    prev_open = open_.shift(2)
    prev_close = close_.shift(2)
    star_open = open_.shift(1)
    star_close = close_.shift(1)
    return (
        (prev_close < prev_open) &
        (star_close < prev_close) & (star_open < prev_close) &
        (close_ > star_open) & (close_ > prev_open)
    )

def to_bool(val):
    if isinstance(val, pd.Series):
        return bool(val.iloc[0])
    return bool(val)
