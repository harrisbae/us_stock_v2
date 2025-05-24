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
                        band_val = lower_mantra.loc[date] if safe_in_index(lower_mantra.index, date) else signal['price']
                        y = band_val - offset
                        ax1.scatter(date, to_float(y), color=color, marker=marker, s=30, label=None)
                    else:
                        band_val = upper_mantra.loc[date] if safe_in_index(upper_mantra.index, date) else signal['price']
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
            ('Bollinger Middle (SMA20)', 'deepskyblue', 0.8, '--'),
            ('HMA5 5일 평균', 'purple', 1.5, '-'),
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
        plt.xticks(rotation=45)
        
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
        
        plt.tight_layout()
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
    매수 신호 필수: HMA 우상향
      1) 종가가 HMA 상향 돌파 + MACD>Signal
      2) 종가가 HMA 상향 돌파 + RSI<40
      3) 종가가 HMA 상향 돌파 + (MACD<=Signal and RSI>=40)
      4) 종가<하단밴드 + MACD>Signal
      5) 종가<하단밴드 + RSI<40
      6) 종가<하단밴드 + (MACD<=Signal and RSI>=40)
    매도 신호 필수: HMA 평탄/하락
      1) HMA 하향 돌파 + MACD<Signal
      2) HMA 하향 돌파 + RSI>70
      3) HMA 하향 돌파 + (MACD>=Signal and RSI<=70)
      4) 종가>상단밴드 + MACD<Signal
      5) 종가>상단밴드 + RSI>70
      6) 종가>상단밴드 + (MACD>=Signal and RSI<=70)
    """
    signals = []
    hma = calculate_hma(data['Close'])
    upper_band, lower_band = calculate_mantra_bands(data['Close'])
    macd, macd_signal, _ = calculate_macd(data['Close'])
    rsi = calculate_rsi(data['Close'])
    for i in range(5, len(data)):
        close = data['Close'].iloc[i].item()
        lower = lower_band.iloc[i].item()
        upper = upper_band.iloc[i].item()
        hma_now = hma.iloc[i].item()
        hma_prev = hma.iloc[i-1].item()
        macd_now = macd.iloc[i].item()
        macd_signal_now = macd_signal.iloc[i].item()
        rsi_now = rsi.iloc[i].item()
        # HMA5 평균값 계산 (직전 5일)
        hma_5avg_prev = np.mean([hma.iloc[i-1-j].item() for j in range(1, 6)])
        # flat_threshold 인자 사용
        is_hma5_up = (hma_now > hma_5avg_prev) or (abs(hma_now - hma_prev) < flat_threshold)
        # 매수 조건: HMA5 상승(필수)
        if is_hma5_up:
            # 1) 종가 HMA 상향돌파 + MACD > Signal
            if (data['Close'].iloc[i-1].item() < hma.iloc[i-1].item()) and (close > hma_now) and (macd_now > macd_signal_now):
                signals.append({
                    'date': data.index[i],
                    'type': 'BUY',
                    'price': close,
                    'reason': '(HMA5 상승) 종가 HMA 상향돌파, MACD>Signal',
                    'reason_short': 'C',
                    'buy_idx': 1
                })
                continue
            # 2) 종가 HMA 상향돌파 + RSI < 40
            if (data['Close'].iloc[i-1].item() < hma.iloc[i-1].item()) and (close > hma_now) and (rsi_now < 40):
                signals.append({
                    'date': data.index[i],
                    'type': 'BUY',
                    'price': close,
                    'reason': '(HMA5 상승) 종가 HMA 상향돌파, RSI<40',
                    'reason_short': 'R',
                    'buy_idx': 2
                })
                continue
            # 3) 종가 HMA 상향돌파 + (MACD ≤ Signal and RSI ≥ 40)
            if (data['Close'].iloc[i-1].item() < hma.iloc[i-1].item()) and (close > hma_now) and (macd_now <= macd_signal_now) and (rsi_now >= 40):
                signals.append({
                    'date': data.index[i],
                    'type': 'BUY',
                    'price': close,
                    'reason': '(HMA5 상승) 종가 HMA 상향돌파, MACD<=Signal, RSI>=40',
                    'reason_short': 'M',
                    'buy_idx': 3
                })
                continue
            # 4) 종가 < 하단밴드 + MACD > Signal
            if (close < lower) and (macd_now > macd_signal_now):
                signals.append({
                    'date': data.index[i],
                    'type': 'BUY',
                    'price': close,
                    'reason': '(HMA5 상승) 종가<하단밴드, MACD>Signal',
                    'reason_short': 'C',
                    'buy_idx': 4
                })
                continue
            # 5) 종가 < 하단밴드 + RSI < 40
            if (close < lower) and (rsi_now < 40):
                signals.append({
                    'date': data.index[i],
                    'type': 'BUY',
                    'price': close,
                    'reason': '(HMA5 상승) 종가<하단밴드, RSI<40',
                    'reason_short': 'R',
                    'buy_idx': 5
                })
                continue
            # 6) 종가 < 하단밴드 + (MACD ≤ Signal and RSI ≥ 40)
            if (close < lower) and (macd_now <= macd_signal_now) and (rsi_now >= 40):
                signals.append({
                    'date': data.index[i],
                    'type': 'BUY',
                    'price': close,
                    'reason': '(HMA5 상승) 종가<하단밴드, MACD<=Signal, RSI>=40',
                    'reason_short': 'M',
                    'buy_idx': 6
                })
                continue
        # 매도 조건: HMA5 평탄/하락(필수)
        if (hma_now <= hma_5avg_prev) or (abs(hma_now - hma_prev) < 0.6):
            # 1) HMA 하향 돌파 + MACD < Signal
            if (hma_now > close) and (hma_prev < data['Close'].iloc[i-1].item()) and (macd_now < macd_signal_now):
                signals.append({
                    'date': data.index[i],
                    'type': 'SELL',
                    'price': close,
                    'reason': 'HMA5 평탄/하락, HMA 하향돌파, MACD<Signal',
                    'reason_short': 'C'
                })
                continue
            # 2) HMA 하향 돌파 + RSI > 70
            if (hma_now > close) and (hma_prev < data['Close'].iloc[i-1].item()) and (rsi_now > 70):
                signals.append({
                    'date': data.index[i],
                    'type': 'SELL',
                    'price': close,
                    'reason': 'HMA5 평탄/하락, HMA 하향돌파, RSI>70',
                    'reason_short': 'R'
                })
                continue
            # 3) HMA 하향돌파 + (MACD ≥ Signal and RSI ≤ 70)
            if (hma_now > close) and (hma_prev < data['Close'].iloc[i-1].item()) and (macd_now >= macd_signal_now) and (rsi_now <= 70):
                signals.append({
                    'date': data.index[i],
                    'type': 'SELL',
                    'price': close,
                    'reason': 'HMA5 평탄/하락, HMA 하향돌파, MACD>=Signal, RSI<=70',
                    'reason_short': 'M'
                })
                continue
            # 4) 종가 > 상단밴드 + MACD < Signal
            if (close > upper) and (macd_now < macd_signal_now):
                signals.append({
                    'date': data.index[i],
                    'type': 'SELL',
                    'price': close,
                    'reason': 'HMA5 평탄/하락, 종가>상단밴드, MACD<Signal',
                    'reason_short': 'C'
                })
                continue
            # 5) 종가 > 상단밴드 + RSI > 70
            if (close > upper) and (rsi_now > 70):
                signals.append({
                    'date': data.index[i],
                    'type': 'SELL',
                    'price': close,
                    'reason': 'HMA5 평탄/하락, 종가>상단밴드, RSI>70',
                    'reason_short': 'R'
                })
                continue
            # 6) 종가 > 상단밴드 + (MACD ≥ Signal and RSI ≤ 70)
            if (close > upper) and (macd_now >= macd_signal_now) and (rsi_now <= 70):
                signals.append({
                    'date': data.index[i],
                    'type': 'SELL',
                    'price': close,
                    'reason': 'HMA5 평탄/하락, 종가>상단밴드, MACD>=Signal, RSI<=70',
                    'reason_short': 'M'
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

def plot_hma_mantra_md_signals(data, ticker, save_path=None, flat_threshold=0.6):
    """HMA와 만트라 밴드 기반 매매 신호를 시각화합니다."""
    hma = calculate_hma(data['Close'])
    upper_band, lower_band = calculate_mantra_bands(data['Close'])
    signals = get_hma_mantra_md_signals(data, flat_threshold=flat_threshold)
    fig, axes = plt.subplots(5, 1, figsize=(15, 18), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1, 1]})
    ax_main, ax_macd, ax_rsi, ax_volume, ax_vix = axes

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
    ax_main.plot(data.index, hma_5avg_series, label='HMA5 5일 평균', color='purple', linestyle='-', linewidth=1.5)
    ax_main.plot(data.index, upper_bollinger, label='Bollinger Upper', color='orange', linestyle='-', linewidth=0.8)
    ax_main.plot(data.index, lower_bollinger, label='Bollinger Lower', color='orange', linestyle='-', linewidth=0.8)
    ax_main.plot(data.index, sma, label='Bollinger Middle (SMA20)', color='deepskyblue', linestyle='--', linewidth=0.8)

    # 볼린저 밴드 상단-중간, 하단-중간 영역 색상 채우기
    ax_main.fill_between(data.index, upper_bollinger, sma, where=(upper_bollinger >= sma), color='orange', alpha=0.13, label='Bollinger Upper-Mid Zone')
    ax_main.fill_between(data.index, lower_bollinger, sma, where=(lower_bollinger <= sma), color='deepskyblue', alpha=0.13, label='Bollinger Lower-Mid Zone')

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
    ax_macd.set_ylabel('MACD')
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
    ax_rsi.set_ylabel('RSI')
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
    # 범례 갱신
    handles, labels = ax_rsi.get_legend_handles_labels()
    ax_rsi.legend(handles, labels, loc='upper left', fontsize=8, title='RSI 신호')

    # 거래량
    valid_data = data.dropna(subset=[('Volume', ticker), ('Open', ticker), ('Close', ticker)])
    volume_colors = ['red' if float(row[('Close', ticker)]) < float(row[('Open', ticker)]) else 'green' for _, row in valid_data.iterrows()]
    ax_volume.bar(valid_data.index, valid_data[('Volume', ticker)], color=volume_colors, alpha=0.6)
    # HA 5일 거래량(이동평균) 추가
    ha5_vol = valid_data[('Volume', ticker)].rolling(window=5).mean()
    ax_volume.plot(valid_data.index, ha5_vol, color='blue', linewidth=2, label='HA 5일 거래량')
    ax_volume.set_ylabel('Volume')
    ax_volume.grid(True, alpha=0.3)
    # 범례
    handles, labels = ax_volume.get_legend_handles_labels()
    ax_volume.legend(handles, labels, loc='upper left', fontsize=8, title='거래량')

    # VIX
    try:
        vix_data = yf.download('^VIX', start=data.index[0], end=data.index[-1])['Close']
        vix_data.index = pd.to_datetime(vix_data.index)
        vix = vix_data.reindex(data.index).ffill().bfill()
        ax_vix.plot(data.index, vix, color='purple')
        ax_vix.set_ylabel('VIX')
        ax_vix.grid(True, alpha=0.3)
        # VIX 범위별 투자가이드 색상영역
        ax_vix.axhspan(0, 15, color='green', alpha=0.10, label='VIX≤15: 저위험')
        ax_vix.axhspan(15, 25, color='yellow', alpha=0.10, label='15<VIX≤25: 중립')
        ax_vix.axhspan(25, 40, color='orange', alpha=0.10, label='25<VIX≤40: 고위험')
        ax_vix.axhspan(40, 100, color='red', alpha=0.10, label='VIX>40: 극단적 변동성')
        # 범례
        handles, labels = ax_vix.get_legend_handles_labels()
        ax_vix.legend(handles, labels, loc='upper left', fontsize=8, title='VIX 투자가이드')
        ax_vix.set_ylim(0, 80)
    except Exception as e:
        print(f"VIX 데이터 다운로드 실패: {e}")

    # 모든 subplot에 신호 버티컬 라인 추가
    for signal in signals:
        color = 'gold' if signal['type'] == 'BUY' else 'orange'
        for ax in [ax_main, ax_macd, ax_rsi, ax_volume, ax_vix]:
            ax.axvline(x=signal['date'], color=color, linestyle='--', linewidth=1, alpha=0.5, zorder=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
