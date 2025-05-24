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
        price_range = data['High'].max() - data['Low'].min()
        offset = price_range * 0.03
        if hma_signals:
            plt.figure(figsize=(15, 8))
            plt.plot(data.index, data['Close'], label='Close', color='black', alpha=0.5)
            plt.plot(data.index, hma, label='HMA', color='blue', linewidth=2)
            plt.plot(data.index, upper_band, label='Upper Band', color='red', linestyle='--')
            plt.plot(data.index, lower_band, label='Lower Band', color='green', linestyle='--')
            for idx, signal in enumerate(hma_signals):
                try:
                    color = 'green' if signal['type'] == 'BUY' else 'red'
                    marker = '^' if signal['type'] == 'BUY' else 'v'
                    date = signal['date']
                    if signal['type'] == 'BUY':
                        band_val = lower_band.loc[date] if safe_in_index(lower_band.index, date) else signal['price']
                        y = band_val - offset
                        plt.scatter(date, y, color=color, marker=marker, s=50, label=None)
                    else:
                        band_val = upper_band.loc[date] if safe_in_index(upper_band.index, date) else signal['price']
                        y = band_val + offset
                        plt.scatter(date, y, color=color, marker=marker, s=50, label=None)
                except Exception as e:
                    print(f"신호 마커 추가 중 오류 발생 (인덱스 {idx}):")
                    print(f"signal: {signal}")
                    print(f"예외 메시지: {str(e)}")
                    print("스택 트레이스:")
                    traceback.print_exc()
                    continue
            # 범례용 더미 객체
            buy_marker = mlines.Line2D([], [], color='green', marker='^', linestyle='None', markersize=8, label='Buy Signal')
            sell_marker = mlines.Line2D([], [], color='red', marker='v', linestyle='None', markersize=8, label='Sell Signal')
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
                        band_val = lower_mantra.loc[date] if safe_in_index(lower_mantra.index, date) else signal['price']
                        y = band_val - offset
                        ax1.scatter(date, y, color=color, marker=marker, s=50, label=None)
                    else:
                        band_val = upper_mantra.loc[date] if safe_in_index(upper_mantra.index, date) else signal['price']
                        y = band_val + offset
                        ax1.scatter(date, y, color=color, marker=marker, s=50, label=None)
                except Exception as e:
                    print(f"신호 마커 추가 중 오류 발생 (인덱스 {idx}):")
                    print(f"signal: {signal}")
                    print(f"예외 메시지: {str(e)}")
                    print("스택 트레이스:")
                    traceback.print_exc()
                    continue
        # 범례용 더미 객체
        buy_marker = mlines.Line2D([], [], color='green', marker='^', linestyle='None', markersize=8, label='Buy Signal')
        sell_marker = mlines.Line2D([], [], color='red', marker='v', linestyle='None', markersize=8, label='Sell Signal')
        ax1.legend(handles=[buy_marker, sell_marker])
        ax1.set_title(f'{ticker} - HMA & Mantra Bands')
        ax1.set_ylabel('Price')
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
        
        # 이동평균선 계산
        ma5 = calculate_ma(data['Close'], 5)
        ma20 = calculate_ma(data['Close'], 20)
        ma60 = calculate_ma(data['Close'], 60)
        ma160 = calculate_ma(data['Close'], 160)
        ma200 = calculate_ma(data['Close'], 200)
        
        # MACD 계산
        macd, signal_line, hist = calculate_macd(data['Close'])  # signal 변수명을 signal_line으로 변경
        
        # RSI 계산
        rsi = calculate_rsi(data['Close'])
        
        # VIX 데이터 가져오기 및 인덱스 맞추기
        try:
            vix_data = yf.download('^VIX', start=data.index[0], end=data.index[-1])['Close']
            vix_data.index = pd.to_datetime(vix_data.index)
            vix = vix_data.reindex(data.index).ffill().bfill()
        except Exception as e:
            print(f"VIX 데이터 다운로드 실패: {e}")
            vix = pd.Series(index=data.index, data=np.nan)
        
        # 서브플롯 생성
        fig = plt.figure(figsize=(15, 20))
        gs = fig.add_gridspec(5, 1, height_ratios=[3, 1, 1, 1, 1])
        
        # 메인 차트
        ax1 = fig.add_subplot(gs[0])
        
        # 캔들차트 데이터 준비
        ohlc = data[['Open', 'High', 'Low', 'Close']].copy()
        ohlc = ohlc.reset_index()
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
        if hma_signals:
            for idx, signal in enumerate(hma_signals):
                try:
                    color = 'green' if signal['type'] == 'BUY' else 'red'
                    marker = '^' if signal['type'] == 'BUY' else 'v'
                    date = signal['date']
                    if signal['type'] == 'BUY':
                        band_val = lower_mantra.loc[date] if safe_in_index(lower_mantra.index, date) else signal['price']
                        y = band_val - offset
                        ax1.scatter(date, y, color=color, marker=marker, s=50, label=None)
                    else:
                        band_val = upper_mantra.loc[date] if safe_in_index(upper_mantra.index, date) else signal['price']
                        y = band_val + offset
                        ax1.scatter(date, y, color=color, marker=marker, s=50, label=None)
                    # y값이 Series면 .item(), 아니면 float(y)
                    y_val = float(y.item()) if hasattr(y, 'item') else float(y)
                    ax1.text(
                        float(mdates.date2num(date)), y_val, date.strftime('%Y-%m-%d'),
                        rotation=45, fontsize=5, color=color, ha='center', va='top',
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
        buy_marker = mlines.Line2D([], [], color='green', marker='^', linestyle='None', markersize=8, label='Buy Signal')
        sell_marker = mlines.Line2D([], [], color='red', marker='v', linestyle='None', markersize=8, label='Sell Signal')
        ax1.legend(handles=[buy_marker, sell_marker], loc='upper left', bbox_to_anchor=(1.02, 1))
        
        ax1.set_title(f'{ticker} - Technical Analysis')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        
        # 거래량 차트
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        volume = np.asarray(data['Volume'].fillna(0)).ravel()
        ord_idx = np.asarray(ordinal_index).ravel()
        min_len = min(len(ord_idx), len(volume))
        # 전체 거래량을 단일 회색 bar로만 표시
        ax2.bar(ord_idx[:min_len], volume[:min_len], color='gray', alpha=0.5, width=0.6, label='Volume')
        # 거래량 이동평균선
        ma20_volume = data['Volume'].rolling(window=20).mean()
        ax2.plot(ord_idx, ma20_volume, color='blue', linewidth=1.2, label='MA20 Vol')
        # Price 이동평균선(좌측 y축)
        ax2_price = ax2.twinx()
        ma5_price = data['Close'].rolling(window=5).mean()
        ma20_price = data['Close'].rolling(window=20).mean()
        ma60_price = data['Close'].rolling(window=60).mean()
        ma120_price = data['Close'].rolling(window=120).mean()
        ma200_price = data['Close'].rolling(window=200).mean()
        ax2_price.plot(ord_idx, ma5_price, color='orange', linestyle='--', linewidth=1.2, label='MA5 Price')
        ax2_price.plot(ord_idx, ma20_price, color='blue', linestyle='--', linewidth=1.2, label='MA20 Price')
        ax2_price.plot(ord_idx, ma60_price, color='green', linestyle='--', linewidth=1.2, label='MA60 Price')
        ax2_price.plot(ord_idx, ma120_price, color='purple', linestyle='--', linewidth=1.2, label='MA120 Price')
        ax2_price.plot(ord_idx, ma200_price, color='red', linestyle='--', linewidth=1.2, label='MA200 Price')
        # 실제 종가(Close) 가격선 추가
        ax2_price.plot(ord_idx, data['Close'], color='black', linewidth=1.2, label='Close Price')
        ax2_price.set_ylabel('Price', color='blue')
        ax2_price.yaxis.set_label_position('left')
        ax2_price.yaxis.tick_left()
        ax2.set_ylabel('Volume', color='gray')
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')
        # MA5/MA20 골든/데드크로스 시그널 마커 및 범례
        ma_buy_idx = []
        ma_sell_idx = []
        for i in range(1, len(ma5_price)):
            prev_ma5 = ma5_price.iloc[i-1].item() if hasattr(ma5_price.iloc[i-1], 'item') else float(ma5_price.iloc[i-1])
            prev_ma20 = ma20_price.iloc[i-1].item() if hasattr(ma20_price.iloc[i-1], 'item') else float(ma20_price.iloc[i-1])
            curr_ma5 = ma5_price.iloc[i].item() if hasattr(ma5_price.iloc[i], 'item') else float(ma5_price.iloc[i])
            curr_ma20 = ma20_price.iloc[i].item() if hasattr(ma20_price.iloc[i], 'item') else float(ma20_price.iloc[i])
            if prev_ma5 < prev_ma20 and curr_ma5 > curr_ma20:
                ma_buy_idx.append(i)
            elif prev_ma5 > prev_ma20 and curr_ma5 < curr_ma20:
                ma_sell_idx.append(i)
        ax2_price.scatter(ord_idx[ma_buy_idx], ma5_price.iloc[ma_buy_idx], color='green', marker='^', s=40, label='MA 골든크로스 매수')
        ax2_price.scatter(ord_idx[ma_sell_idx], ma5_price.iloc[ma_sell_idx], color='red', marker='v', s=40, label='MA 데드크로스 매도')
        # 범례 합치기
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_price.get_legend_handles_labels()
        legend_dict = dict(zip(labels1 + labels2, lines1 + lines2))
        ax2.legend(legend_dict.values(), legend_dict.keys(), loc='upper left', fontsize=8)
        
        # VIX 차트
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.plot(ordinal_index, vix, color='purple', linewidth=0.7, label='VIX')
        ax3.set_ylabel('VIX')
        ax3.grid(True, alpha=0.3)
        # VIX 투자전략별 구간 배경색 및 범례 추가
        vix_ranges = [12, 16, 20, 30, 100]
        vix_colors = ['#90caf9', '#a5d6a7', '#ffb3b3', '#b39ddb']
        vix_guides = [
            '저위험: 주식 비중 확대',
            '경계: 분할매수/분할매도',
            '위험: 현금 비중 확대',
            '극단적 공포: 분할매수 기회'
        ]
        vix_patches = []
        for i in range(len(vix_ranges)-1):
            ax3.axhspan(vix_ranges[i], vix_ranges[i+1], color=vix_colors[i], alpha=0.2)
            from matplotlib.patches import Patch
            vix_patches.append(Patch(facecolor=vix_colors[i], edgecolor='none', alpha=0.5, label=f'{vix_ranges[i]}~{vix_ranges[i+1]}: {vix_guides[i]}'))
        handles, labels = ax3.get_legend_handles_labels()
        handles = [handles[0]] + vix_patches if handles else vix_patches
        labels = [labels[0]] + [patch.get_label() for patch in vix_patches] if labels else [patch.get_label() for patch in vix_patches]
        ax3.legend(handles, labels, loc='upper left', fontsize=9)
        
        # MACD 차트
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        ax4.plot(ordinal_index, macd, label='MACD', color='blue', linewidth=0.7)
        ax4.plot(ordinal_index, signal_line, label='Signal', color='red', linewidth=0.7)  # signal을 signal_line으로 변경
        hist_arr = np.asarray(hist).ravel()
        ord_idx_macd = np.asarray(ordinal_index).ravel()
        min_len_macd = min(len(ord_idx_macd), len(hist_arr))
        ax4.bar(ord_idx_macd[:min_len_macd], hist_arr[:min_len_macd], color='gray', alpha=0.5, width=0.6)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_ylabel('MACD')
        # MACD 매수/매도 시그널 마커 표시 및 범례 추가
        # MACD 골든크로스(매수): macd가 signal_line을 아래에서 위로 돌파
        # MACD 데드크로스(매도): macd가 signal_line을 위에서 아래로 돌파
        macd_buy_idx = []
        macd_sell_idx = []
        for i in range(1, len(macd)):
            prev_macd = macd.iloc[i-1].item() if hasattr(macd.iloc[i-1], 'item') else float(macd.iloc[i-1])
            prev_signal = signal_line.iloc[i-1].item() if hasattr(signal_line.iloc[i-1], 'item') else float(signal_line.iloc[i-1])
            curr_macd = macd.iloc[i].item() if hasattr(macd.iloc[i], 'item') else float(macd.iloc[i])
            curr_signal = signal_line.iloc[i].item() if hasattr(signal_line.iloc[i], 'item') else float(signal_line.iloc[i])
            if prev_macd < prev_signal and curr_macd > curr_signal:
                macd_buy_idx.append(i)
            elif prev_macd > prev_signal and curr_macd < curr_signal:
                macd_sell_idx.append(i)
        # 마커 표시
        ax4.scatter(ordinal_index[macd_buy_idx], macd.iloc[macd_buy_idx], color='green', marker='^', s=40, label='MACD 매수')
        ax4.scatter(ordinal_index[macd_sell_idx], macd.iloc[macd_sell_idx], color='red', marker='v', s=40, label='MACD 매도')
        # 범례
        handles, labels = ax4.get_legend_handles_labels()
        # 중복 제거
        legend_dict = dict(zip(labels, handles))
        ax4.legend(legend_dict.values(), legend_dict.keys(), loc='upper left', bbox_to_anchor=(1.02, 1))
        
        # RSI 차트
        ax5 = fig.add_subplot(gs[4], sharex=ax1)
        ax5.plot(ordinal_index, rsi, color='blue', linewidth=0.7)
        ax5.axhline(y=70, color='red', linestyle='--', linewidth=0.5)
        ax5.axhline(y=30, color='green', linestyle='--', linewidth=0.5)
        ax5.set_ylabel('RSI')
        ax5.set_ylim(0, 100)
        ax5.grid(True, alpha=0.3)
        # RSI 매수/매도 시그널 마커 및 범례 추가
        rsi_buy_idx = []
        rsi_sell_idx = []
        for i in range(1, len(rsi)):
            prev_rsi = rsi.iloc[i-1].item() if hasattr(rsi.iloc[i-1], 'item') else float(rsi.iloc[i-1])
            curr_rsi = rsi.iloc[i].item() if hasattr(rsi.iloc[i], 'item') else float(rsi.iloc[i])
            if prev_rsi < 30 and curr_rsi >= 30:
                rsi_buy_idx.append(i)
            elif prev_rsi > 70 and curr_rsi <= 70:
                rsi_sell_idx.append(i)
        ax5.scatter(ordinal_index[rsi_buy_idx], rsi.iloc[rsi_buy_idx], color='green', marker='^', s=40, label='RSI 매수')
        ax5.scatter(ordinal_index[rsi_sell_idx], rsi.iloc[rsi_sell_idx], color='red', marker='v', s=40, label='RSI 매도')
        handles, labels = ax5.get_legend_handles_labels()
        legend_dict = dict(zip(labels, handles))
        ax5.legend(legend_dict.values(), legend_dict.keys(), loc='upper left', fontsize=9)
        
        # x축 설정
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # 레이아웃 조정
        plt.subplots_adjust(hspace=0.1)
        
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
