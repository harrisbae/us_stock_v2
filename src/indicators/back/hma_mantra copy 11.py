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
            for idx, signal in enumerate(hma_signals):
                try:
                    color = 'green' if signal['type'] == 'BUY' else 'red'
                    marker = '^' if signal['type'] == 'BUY' else 'v'
                    date = signal['date']
                    # 모든 subplot에 더 두껍고 진한 버티컬 라인 추가
                    for ax in [ax1]:
                        ax.axvline(x=date, color=color, linestyle='--', linewidth=0.3, alpha=0.8, zorder=1)
                    if signal['type'] == 'BUY':
                        band_val = lower_band.loc[date] if safe_in_index(lower_band.index, date) else signal['price']
                        y = band_val - offset
                        ax1.scatter(date, to_float(y), color=color, marker=marker, s=30, label=None)
                    else:
                        band_val = upper_band.loc[date] if safe_in_index(upper_band.index, date) else signal['price']
                        y = band_val + offset
                        ax1.scatter(date, to_float(y), color=color, marker=marker, s=30, label=None)
                    ax1.text(
                        to_float(mdates.date2num(date)), to_float(y), date.strftime('%Y-%m-%d'),
                        rotation=45, fontsize=5, color=color, ha='center', va='top' if signal['type']=='BUY' else 'bottom',
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.1')
                    )
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
        ax1.plot(ordinal_index, hma, label='HMA', color='blue', linewidth=1.4)
        ax1.plot(ordinal_index, upper_mantra, label='Mantra Upper', color='red', linestyle='--', linewidth=0.7)
        ax1.plot(ordinal_index, lower_mantra, label='Mantra Lower', color='green', linestyle='--', linewidth=0.7)
        
        # 볼린저 밴드 상단-중간, 하단-중간 영역 색상 채우기
        ax1.fill_between(data.index, upper_mantra, lower_mantra, where=(upper_mantra >= lower_mantra), color='orange', alpha=0.13, label='Mantra Zone')
        
        # 신호 마커 및 버티컬 라인 추가
        price_range = data['High'].max() - data['Low'].min()
        offset = price_range * 0.03
        if hma_signals:
            for idx, signal in enumerate(hma_signals):
                try:
                    color = 'green' if signal['type'] == 'BUY' else 'red'
                    marker = '^' if signal['type'] == 'BUY' else 'v'
                    date = signal['date']
                    ax1.axvline(x=date, color=color, linestyle='--', linewidth=0.3, alpha=0.8, zorder=1)
                    if signal['type'] == 'BUY':
                        band_val = lower_band.loc[date] if safe_in_index(lower_band.index, date) else signal['price']
                        y = band_val - offset
                        ax1.scatter(date, to_float(y), color=color, marker=marker, s=30, label=None)
                    else:
                        band_val = upper_band.loc[date] if safe_in_index(upper_band.index, date) else signal['price']
                        y = band_val + offset
                        ax1.scatter(date, to_float(y), color=color, marker=marker, s=30, label=None)
                    ax1.text(
                        to_float(mdates.date2num(date)), to_float(y),
                        f"{date.strftime('%Y-%m-%d')}\n{signal.get('reason','')}" if signal.get('reason') else date.strftime('%Y-%m-%d'),
                        rotation=45, fontsize=6, color='black',
                        ha='center', va='top' if signal['type']=='BUY' else 'bottom',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'),
                        zorder=6
                    )
                except Exception as e:
                    print(f"신호 마커 추가 중 오류 발생 (인덱스 {idx}):")
                    print(f"signal: {signal}")
                    print(f"예외 메시지: {str(e)}")
                    print("스택 트레이스:")
                    traceback.print_exc()
                    continue
        
        # 범례 추가 (HMA, 만트라밴드, 볼린저밴드)
        line_labels = [
            ('HMA', 'blue', 2, '-'),
            ('Upper Band', 'red', 1, '--'),
            ('Lower Band', 'green', 1, '--'),
            ('Bollinger Upper', 'orange', 0.8, '-'),
            ('Bollinger Lower', 'orange', 0.8, '-'),
            ('Bollinger Middle (SMA20)', 'deepskyblue', 0.8, '--')
        ]
        line_handles = [
            mlines.Line2D([], [], color=color, linewidth=lw, linestyle=ls, label=label)
            for label, color, lw, ls in line_labels
        ]
        legend_lines = ax1.legend(line_handles, [l[0] for l in line_labels], loc='upper right', fontsize=8, title='주요 라인')
        ax1.add_artist(legend_lines)
        
        ax1.set_title(f'{ticker} - Technical Analysis')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax1.xaxis.set_minor_locator(mdates.DayLocator())
        ax1.tick_params(axis='x', which='major', labelsize=9, rotation=45)
        ax1.tick_params(axis='x', which='minor', labelsize=7, rotation=45)
        
        # 차트 해설 텍스트 추가
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
            color = 'gold' if signal['type'] == 'BUY' else 'orange'
            ax1.plot(signal['date'], signal['price'], marker=marker, color=color, markersize=10, markeredgecolor='black', linestyle='None', label='매수 신호(▲)' if signal['type']=='BUY' else '매도 신호(▼)')

        # 대표 범례(중복 제거)
        buy_marker = mlines.Line2D([], [], color='gold', marker='^', linestyle='None', markersize=10, markeredgecolor='black', label='매수 신호(▲)')
        sell_marker = mlines.Line2D([], [], color='orange', marker='v', linestyle='None', markersize=10, markeredgecolor='black', label='매도 신호(▼)')
        handles, labels = ax1.get_legend_handles_labels()
        handles = [buy_marker, sell_marker] + [h for h, l in zip(handles, labels) if l not in ['매수 신호(▲)', '매도 신호(▼)']]
        labels = ['매수 신호(▲)', '매도 신호(▼)'] + [l for l in labels if l not in ['매수 신호(▲)', '매도 신호(▼)']]
        ax1.legend(handles, labels, loc='upper left', fontsize=8, title='주요 라인 및 신호')
        
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

def get_hma_mantra_md_signals(data: pd.DataFrame, flat_threshold=0.6) -> list:
    """
    매수/매도 신호 생성 (매도 신호 조건 사용자 정의 반영, 불필요한 변수 및 코드 정리)
    """
    signals = []
    hma = calculate_hma(data['Close'])
    upper_band, lower_band = calculate_mantra_bands(data['Close'])
    rsi3 = calculate_rsi(data['Close'], period=3)
    rsi14 = calculate_rsi(data['Close'], period=14)
    # 기존 매수 신호에 필요한 변수만 선언
    macd, macd_signal, _ = calculate_macd(data['Close'])
    rsi = calculate_rsi(data['Close'])
    for i in range(1, len(data)):
        close = data['Close'].iloc[i].item()
        prev_close = data['Close'].iloc[i-1].item()
        hma_now = hma.iloc[i].item()
        hma_prev = hma.iloc[i-1].item()
        upper = upper_band.iloc[i].item()
        upper_prev = upper_band.iloc[i-1].item()
        rsi3_now = rsi3.iloc[i].item()
        rsi14_now = rsi14.iloc[i].item()
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
        if (prev_close < hma_prev) and (close > hma_now) and (rsi3_now > rsi14_now):
            signals.append({
                'date': data.index[i],
                'type': 'BUY',
                'price': close,
                'reason': '종가 HMA 상향 돌파 + 타지마할 RSI 매수구간(rsi3 > rsi14)',
                'reason_short': 'B1'
            })
            continue
        # 2) 전일 종가 < 밴드 하단, 당일 종가 > 밴드 하단 + 타지마할 RSI 매수 구간
        lower = lower_band.iloc[i].item()
        lower_prev = lower_band.iloc[i-1].item()
        if (prev_close < lower_prev) and (close > lower) and (rsi3_now > rsi14_now):
            signals.append({
                'date': data.index[i],
                'type': 'BUY',
                'price': close,
                'reason': '밴드 하단 상향 돌파 + 타지마할 RSI 매수구간(rsi3 > rsi14)',
                'reason_short': 'B2'
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

def plot_hma_mantra_md_signals(data, ticker, save_path=None, flat_threshold=0.6, show_box_range=False, box_period=60):
    """HMA와 만트라 밴드 기반 매매 신호를 시각화합니다."""
    hma = calculate_hma(data['Close'])
    upper_band, lower_band = calculate_mantra_bands(data['Close'])
    signals = get_hma_mantra_md_signals(data, flat_threshold=flat_threshold)
    fig, axes = plt.subplots(9, 1, figsize=(15, 30), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1, 1, 1, 1, 1, 1]})
    ax_main, ax_tajmahal_rsi, ax_macd, ax_rsi, ax_volume, ax_vix, ax_tnx, ax_dxy, ax_gold = axes

    # 메인 차트(캔들, HMA, 만트라 밴드, 볼린저 밴드, 신호 등)를 ax_main에 그리기
    # 캔들바 데이터 준비
    ohlc = [
        [mdates.date2num(date), float(row[('Open', ticker)]), float(row[('High', ticker)]), float(row[('Low', ticker)]), float(row[('Close', ticker)])]
        for date, row in data.iterrows()
    ]
    candlestick_ohlc(ax_main, ohlc, width=0.6, colorup='green', colordown='red', alpha=0.6)

    # HMA, 만트라 밴드, 볼린저 밴드
    hma = calculate_hma(data[('Close', ticker)])
    upper_band, lower_band = calculate_mantra_bands(data[('Close', ticker)])
    sma = data[('Close', ticker)].rolling(window=20).mean()
    std = data[('Close', ticker)].rolling(window=20).std()
    upper_bollinger = sma + (2 * std)
    lower_bollinger = sma - (2 * std)
    hma_5avg_series = hma.rolling(window=5).mean()

    ax_main.plot(data.index, data[('Close', ticker)], label='Close', color='black', alpha=0.5, linewidth=1.2)
    ax_main.plot(data.index, hma, label='HMA', color='blue', linewidth=2)
    ax_main.plot(data.index, upper_band, label='Upper Band', color='red', linestyle='--')
    ax_main.plot(data.index, lower_band, label='Lower Band', color='green', linestyle='--')
    ax_main.plot(data.index, upper_bollinger, label='Bollinger Upper', color='orange', linestyle='-', linewidth=0.8)
    ax_main.plot(data.index, lower_bollinger, label='Bollinger Lower', color='orange', linestyle='-', linewidth=0.8)
    ax_main.plot(data.index, sma, label='Bollinger Middle (SMA20)', color='deepskyblue', linestyle='--', linewidth=0.8)

    # 볼린저 밴드 상단-중간, 하단-중간 영역 색상 채우기
    ax_main.fill_between(data.index, upper_bollinger, sma, where=(upper_bollinger >= sma), color='orange', alpha=0.13, label='Bollinger Upper-Mid Zone')
    ax_main.fill_between(data.index, lower_bollinger, sma, where=(lower_bollinger <= sma), color='deepskyblue', alpha=0.13, label='Bollinger Lower-Mid Zone')

    # 메인차트 제목에 종목과 기간 표시
    start_date = data.index[0].strftime('%Y-%m-%d')
    end_date = data.index[-1].strftime('%Y-%m-%d')
    ax_main.set_title(f'{ticker} - Technical Analysis ({start_date} ~ {end_date})', pad=20, fontsize=12)

    # 신호 마커 및 텍스트(일자+근거) 추가 (밴드 하단/상단, 겹치지 않게, font color=black)
    for signal in signals:
        marker = '^' if signal['type'] == 'BUY' else 'v'
        color = 'gold' if signal['type'] == 'BUY' else 'orange'
        date = signal['date']
        price = signal['price']
        # 매수: 밴드 하단 아래, 매도: 밴드 상단 위
        if signal['type'] == 'BUY':
            if date in lower_band.index:
                y = lower_band.loc[date] - (abs(upper_band.max() - lower_band.min()) * 0.02)
            else:
                y = price
            ax_main.plot(date, y, marker=marker, color=color, markersize=10, markeredgecolor='black', linestyle='None', label=None)
            # 텍스트(일자+근거, font color=black)
            ax_main.text(date, y - (abs(upper_band.max() - lower_band.min()) * 0.03),
                f"{date.strftime('%Y-%m-%d')}\n{signal.get('reason','')}" if signal.get('reason') else date.strftime('%Y-%m-%d'),
                rotation=45, fontsize=5, color='black', ha='center', va='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'), zorder=6)
        else:
            if date in upper_band.index:
                y = upper_band.loc[date] + (abs(upper_band.max() - lower_band.min()) * 0.02)
            else:
                y = price
            ax_main.plot(date, y, marker=marker, color=color, markersize=10, markeredgecolor='black', linestyle='None', label=None)
            # 텍스트(일자+근거, font color=black)
            ax_main.text(date, y + (abs(upper_band.max() - lower_band.min()) * 0.03),
                f"{date.strftime('%Y-%m-%d')}\n{signal.get('reason','')}" if signal.get('reason') else date.strftime('%Y-%m-%d'),
                rotation=45, fontsize=5, color='black', ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'), zorder=6)

    # 대표 범례(중복 제거)
    buy_marker = mlines.Line2D([], [], color='gold', marker='^', linestyle='None', markersize=10, markeredgecolor='black', label='매수 신호(▲)')
    sell_marker = mlines.Line2D([], [], color='orange', marker='v', linestyle='None', markersize=10, markeredgecolor='black', label='매도 신호(▼)')
    handles, labels = ax_main.get_legend_handles_labels()
    handles = [buy_marker, sell_marker] + [h for h, l in zip(handles, labels) if l not in ['매수 신호(▲)', '매도 신호(▼)']]
    labels = ['매수 신호(▲)', '매도 신호(▼)'] + [l for l in labels if l not in ['매수 신호(▲)', '매도 신호(▼)']]
    ax_main.legend(handles, labels, loc='upper left', fontsize=8, title='주요 라인 및 신호')

    # MACD
    macd, signal, hist = calculate_macd(data[('Close', ticker)])
    ax_macd.plot(data.index, macd, label='MACD', color='blue')
    ax_macd.plot(data.index, signal, label='Signal', color='red')
    ax_macd.bar(data.index, hist, color=['red' if h < 0 else 'green' for h in hist], alpha=0.6)
    ax_macd.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax_macd.yaxis.set_label_position('left')
    ax_macd.yaxis.tick_left()
    ax_macd.yaxis.set_label_position('right')
    ax_macd.yaxis.tick_right()
    ax_macd.set_ylabel('MACD', labelpad=10)
    ax_macd.legend(loc='upper left', fontsize=8)
    ax_macd.grid(True, alpha=0.3)

    # MACD 매수/매도 신호 마커 및 범례 추가
    macd_buy_dates = []
    macd_sell_dates = []
    for i in range(1, len(macd)):
        # 골든크로스: MACD가 Signal을 아래에서 위로 돌파
        if macd[i-1] < signal[i-1] and macd[i] > signal[i]:
            macd_buy_dates.append(data.index[i])
        # 데드크로스: MACD가 Signal을 위에서 아래로 돌파
        if macd[i-1] > signal[i-1] and macd[i] < signal[i]:
            macd_sell_dates.append(data.index[i])
    # 마커 표시
    ax_macd.plot(macd_buy_dates, [macd[i] for i, d in enumerate(data.index) if d in macd_buy_dates], '^', color='lime', markersize=8, markeredgecolor='black', label='MACD 매수신호(↑)')
    ax_macd.plot(macd_sell_dates, [macd[i] for i, d in enumerate(data.index) if d in macd_sell_dates], 'v', color='red', markersize=8, markeredgecolor='black', label='MACD 매도신호(↓)')
    # 범례 갱신
    handles, labels = ax_macd.get_legend_handles_labels()
    ax_macd.legend(handles, labels, loc='upper left', fontsize=8, title='MACD 신호')

    # RSI
    rsi = calculate_rsi(data[('Close', ticker)])
    ax_rsi.plot(data.index, rsi, color='purple')
    ax_rsi.axhline(y=70, color='red', linestyle='--', alpha=0.3)
    ax_rsi.axhline(y=30, color='green', linestyle='--', alpha=0.3)
    ax_rsi.yaxis.set_label_position('left')
    ax_rsi.yaxis.tick_left()
    ax_rsi.yaxis.set_label_position('right')
    ax_rsi.yaxis.tick_right()
    ax_rsi.set_ylabel('RSI', labelpad=10)
    ax_rsi.grid(True, alpha=0.3)

    # RSI 매수/매도 신호 마커 및 범례 추가
    rsi_buy_dates = []
    rsi_sell_dates = []
    for i in range(1, len(rsi)):
        # 매수: 30 하향 돌파(과매도 진입)
        if rsi[i-1] > 30 and rsi[i] <= 30:
            rsi_buy_dates.append(data.index[i])
        # 매도: 70 상향 돌파(과매수 진입)
        if rsi[i-1] < 70 and rsi[i] >= 70:
            rsi_sell_dates.append(data.index[i])
    # 마커 표시
    ax_rsi.plot(rsi_buy_dates, [rsi[i] for i, d in enumerate(data.index) if d in rsi_buy_dates], '^', color='blue', markersize=8, markeredgecolor='black', label='RSI 매수신호(↑)')
    ax_rsi.plot(rsi_sell_dates, [rsi[i] for i, d in enumerate(data.index) if d in rsi_sell_dates], 'v', color='red', markersize=8, markeredgecolor='black', label='RSI 매도신호(↓)')
    # 범례 갱신 (fill_between 핸들 포함)
    handles, labels = ax_rsi.get_legend_handles_labels()
    # fill_between 핸들 추가
    buy_mask = (rsi > 30)
    sell_mask = (rsi < 70)
    buy_fill = ax_rsi.fill_between(data.index, 0, 100, where=buy_mask, color='lime', alpha=0.08, label='매수 구간')
    sell_fill = ax_rsi.fill_between(data.index, 0, 100, where=sell_mask, color='red', alpha=0.08, label='매도 구간')
    handles = [buy_fill, sell_fill] + [h for h, l in zip(handles, labels) if l not in ['매수 구간', '매도 구간']]
    labels = ['매수 구간', '매도 구간'] + [l for l in labels if l not in ['매수 구간', '매도 구간']]
    ax_rsi.legend(handles, labels, loc='upper left', fontsize=8)

    # --- 타지마할 RSI subplot 정상 출력 ---
    rsi3 = calculate_rsi(data[('Close', ticker)], period=3)
    rsi14 = calculate_rsi(data[('Close', ticker)], period=14)
    rsi50 = calculate_rsi(data[('Close', ticker)], period=50)
    ax_tajmahal_rsi.plot(data.index, rsi3, label='RSI 3', color='blue')
    ax_tajmahal_rsi.plot(data.index, rsi14, label='RSI 14', color='purple')
    ax_tajmahal_rsi.plot(data.index, rsi50, label='RSI 50', color='gray')
    ax_tajmahal_rsi.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax_tajmahal_rsi.axhline(50, color='black', linestyle='--', alpha=0.5)
    ax_tajmahal_rsi.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax_tajmahal_rsi.fill_between(data.index, 30, 70, color='orange', alpha=0.08)
    ax_tajmahal_rsi.set_ylim(0, 100)
    ax_tajmahal_rsi.set_ylabel('타지마할 RSI', labelpad=10)
    ax_tajmahal_rsi.set_title('타지마할 RSI (3, 14, 50)', fontsize=10)
    ax_tajmahal_rsi.legend(loc='upper left', fontsize=8)
    ax_tajmahal_rsi.grid(True, alpha=0.3)
    ax_tajmahal_rsi.yaxis.set_label_position('right')
    ax_tajmahal_rsi.yaxis.tick_right()
    # 매수/매도 구간 색상 채우기
    buy_mask = (rsi3 > rsi14)
    sell_mask = (rsi3 < rsi14)
    ax_tajmahal_rsi.fill_between(data.index, 0, 100, where=buy_mask, color='lime', alpha=0.08, label='매수 구간')
    ax_tajmahal_rsi.fill_between(data.index, 0, 100, where=sell_mask, color='red', alpha=0.08, label='매도 구간')
    # 신호 마커
    buy_idx = (rsi3.shift(1) < rsi14.shift(1)) & (rsi3 > rsi14)
    sell_idx = (rsi3.shift(1) > rsi14.shift(1)) & (rsi3 < rsi14)
    converge_idx = ((abs(rsi3 - rsi14) < 2) & (abs(rsi14 - rsi50) < 2) & (abs(rsi3 - rsi50) < 2))
    ax_tajmahal_rsi.plot(data.index[buy_idx], rsi3[buy_idx], '^', color='blue', markersize=8, label='매수 신호(▲)')
    ax_tajmahal_rsi.plot(data.index[sell_idx], rsi3[sell_idx], 'v', color='red', markersize=8, label='매도 신호(▼)')
    ax_tajmahal_rsi.plot(data.index[converge_idx], rsi3[converge_idx], 'o', color='black', markersize=7, label='추세전환(●)')
    handles, labels = ax_tajmahal_rsi.get_legend_handles_labels()
    ax_tajmahal_rsi.legend(handles, labels, loc='upper left', fontsize=8)

    # --- 신호별 버티컬 라인 subplot 전체에 추가 ---
    for signal in signals:
        if signal['type'] == 'BUY':
            color = '#008000'  # 진한 초록 또는 gold
            style = '--'
        else:
            color = '#B22222'  # 진한 빨강 또는 orange
            style = '--'
        for ax in [ax_main, ax_tajmahal_rsi, ax_macd, ax_rsi, ax_volume, ax_vix, ax_tnx, ax_dxy, ax_gold]:
            ax.axvline(x=signal['date'], color=color, linestyle=style, linewidth=1.2, alpha=0.7, zorder=0)

    # 최근 30일 박스권 표시
    box_period = 30
    recent_high = data[('High', ticker)].iloc[-box_period:].max()
    recent_low = data[('Low', ticker)].iloc[-box_period:].min()
    start_date = data.index[-box_period]
    end_date = data.index[-1]
    # 박스권 영역 그리기 (메인 차트)
    ax_main.axhline(recent_high, color='brown', linestyle='--', linewidth=1, alpha=0.7, label=f'박스권 상단({recent_high:.2f})')
    ax_main.axhline(recent_low, color='brown', linestyle='--', linewidth=1, alpha=0.7, label=f'박스권 하단({recent_low:.2f})')
    ax_main.fill_betweenx([recent_low, recent_high], start_date, end_date, color='sandybrown', alpha=0.12, label='박스권')
    # 박스권 상단/하단/현재가 텍스트 표시
    current_price = data[('Close', ticker)].iloc[-1]
    # 상단(저항선)
    ax_main.text(end_date, recent_high, f"{recent_high:.2f} (저항선)", color='red', fontsize=8, ha='left', va='bottom',
                 rotation=45,
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', boxstyle='round,pad=0.2'))
    # 하단(지지선)
    ax_main.text(end_date, recent_low, f"{recent_low:.2f} (지지선)", color='green', fontsize=8, ha='left', va='top',
                 rotation=45,
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='green', boxstyle='round,pad=0.2'))
    # 현재가
    ax_main.text(end_date, current_price, f"{current_price:.2f} (현재가)", color='black', fontsize=8, ha='left', va='center',
                 rotation=45,
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.2'))

    # 현재가 수평선(점선) 추가
    ax_main.axhline(current_price, color='black', linestyle='--', linewidth=1.2, alpha=0.8, zorder=1)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig = plt.gcf()
    fig.autofmt_xdate()
    ax_main.tick_params(axis='x', which='both', labelbottom=True)
    # Y축 오른쪽만 표시(최종)
    ax_main.yaxis.set_label_position('right')
    ax_main.yaxis.tick_right()
    ax_main.set_ylabel('Price', labelpad=10)
    ax_main.yaxis.set_ticks_position('right')
    ax_main.yaxis.set_tick_params(labelleft=False, labelright=True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
