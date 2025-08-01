"""
고급 시각화 모듈
"""

import mplfinance as mpf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
from matplotlib.legend_handler import HandlerTuple
from ..core import calculate_hma, calculate_mantra_bands, calculate_rsi, calculate_macd
from ..signals import get_hma_mantra_md_signals
from ..utils import get_available_font
import matplotlib.patches as mpatches
import pandas_datareader.data as web

def get_market_data(start_date, end_date):
    """시장 데이터(VIX, TNX, DXY)를 가져옵니다."""
    import yfinance as yf
    from datetime import datetime, timedelta
    
    # 오늘 날짜 가져오기
    today = datetime.now().date()
    
    # VIX 지수
    vix = yf.download('^VIX', start=start_date, end=today)['Close']
    print(f"VIX 마지막 데이터 날짜: {vix.index[-1].strftime('%Y-%m-%d')}")
    print(f"오늘 날짜: {today.strftime('%Y-%m-%d')}")
    
    # 미국채 10년물 금리
    tnx = yf.download('^TNX', start=start_date, end=today)['Close']
    print(f"TNX 마지막 데이터 날짜: {tnx.index[-1].strftime('%Y-%m-%d')}")
    
    # 달러 인덱스
    dxy = yf.download('DX-Y.NYB', start=start_date, end=today)['Close']
    print(f"DXY 마지막 데이터 날짜: {dxy.index[-1].strftime('%Y-%m-%d')}")
    
    return vix, tnx, dxy

def calculate_bollinger_bands(data, window=20, num_std=2):
    """볼린저 밴드 계산"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return rolling_mean, upper_band, lower_band

def calculate_support_resistance(data, window=20):
    """20일 박스권 기준 지지선과 저항선 계산"""
    rolling_min = data['Low'].rolling(window=window).min()
    rolling_max = data['High'].rolling(window=window).max()
    current_support = rolling_min.iloc[-1]
    current_resistance = rolling_max.iloc[-1]
    return current_support, current_resistance

def plot_hma_mantra_md_signals(data: pd.DataFrame, ticker: str = None, save_path: str = None, current_price: float = None):
    """HMA + 만트라 밴드 매수/매도 신호 차트"""
    # 폰트 설정
    plt.rcParams['font.family'] = 'AppleGothic'  # macOS용 한글 폰트
    plt.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지
    
    if isinstance(data.columns, pd.MultiIndex):
        ohlcv_data = data.xs(ticker, axis=1, level=1)
    else:
        ohlcv_data = data

    # 시장 데이터 가져오기
    start_date = ohlcv_data.index[0]
    end_date = ohlcv_data.index[-1]
    vix, tnx, dxy = get_market_data(start_date, end_date)

    # 기술적 지표 계산
    hma = calculate_hma(ohlcv_data['Close'])
    upper_band, lower_band = calculate_mantra_bands(ohlcv_data['Close'])
    rsi3 = calculate_rsi(ohlcv_data['Close'], period=3)
    rsi14 = calculate_rsi(ohlcv_data['Close'], period=14)
    rsi50 = calculate_rsi(ohlcv_data['Close'], period=50)
    macd, macd_signal, hist = calculate_macd(ohlcv_data['Close'])
    
    # 볼린저 밴드 계산 (plot 전에 위치)
    volume_ma = ohlcv_data['Volume'].rolling(window=20).mean()
    volume_std = ohlcv_data['Volume'].rolling(window=20).std()
    volume_upper = volume_ma + (volume_std * 2)
    
    # 볼린저 밴드 계산
    bb_ma, bb_upper, bb_lower = calculate_bollinger_bands(ohlcv_data['Close'])

    # 신호 생성
    trade_signals = get_hma_mantra_md_signals(ohlcv_data, ticker)

    # 차트 생성
    fig = plt.figure(figsize=(15, 34))
    # 메인 차트에 범례를 위한 여유 공간 확보
    gs = GridSpec(8, 1, height_ratios=[4, 1, 1, 1, 1, 1, 1, 1], figure=fig)
    
    # 메인 차트
    ax_main = fig.add_subplot(gs[0])
    ax_rsi = fig.add_subplot(gs[1], sharex=ax_main)
    ax_macd = fig.add_subplot(gs[2], sharex=ax_main)
    ax_volume = fig.add_subplot(gs[3], sharex=ax_main)
    ax_vix = fig.add_subplot(gs[4], sharex=ax_main)
    ax_tnx = fig.add_subplot(gs[5], sharex=ax_main)
    ax_dxy = fig.add_subplot(gs[6], sharex=ax_main)
    ax_hyspread = fig.add_subplot(gs[7], sharex=ax_main)

    # 메인 차트 설정
    candlestick_ohlc(ax_main, 
                     [[mdates.date2num(date), o, h, l, c] for date, (o, h, l, c) in 
                      zip(ohlcv_data.index, ohlcv_data[['Open', 'High', 'Low', 'Close']].values)],
                     width=0.6, colorup='green', colordown='red', alpha=0.7)

    # 캔들 차트 범례를 위한 더미 플롯
    green_candle = plt.Rectangle((0, 0), 1, 1, fc='green', alpha=0.7)
    red_candle = plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.7)

    # 신호 마커 범례를 위한 더미 플롯
    strong_buy = ax_main.plot([], [], '^', color='blue', markersize=10, markeredgecolor='black', label='강한 매수')[0]
    normal_buy = ax_main.plot([], [], '^', color=(0, 0, 1, 0.5), markersize=10, markeredgecolor='black', label='일반 매수')[0]
    strong_sell = ax_main.plot([], [], 'v', color='red', markersize=10, markeredgecolor='black', label='강한 매도')[0]
    normal_sell = ax_main.plot([], [], 'v', color=(1, 0, 0, 0.5), markersize=10, markeredgecolor='black', label='일반 매도')[0]

    # 신호 저장을 위한 딕셔너리
    buy_signals = {
        'B1_strong': None,  # HMA 상향돌파 + RSI 매수 + MACD > Signal
        'B1_normal': None,  # HMA 상향돌파 + RSI 매수 + MACD ≤ Signal
        'B2_strong': None,  # 밴드 하단 상향돌파 + RSI 매수 + MACD > Signal
        'B2_normal': None,  # 밴드 하단 상향돌파 + RSI 매수 + MACD ≤ Signal
        'B3': None,         # HMA 상향돌파 + RSI 비매수 + MACD > Signal
    }
    
    sell_signals = {
        'T1_strong': None,  # HMA 하향돌파 + RSI 매도 + MACD < Signal
        'T1_normal': None,  # HMA 하향돌파 + RSI 매도 + MACD ≥ Signal
        'T2_strong': None,  # 밴드 상단 하향돌파 + RSI 매도 + MACD < Signal
        'T2_normal': None,  # 밴드 상단 하향돌파 + RSI 매도 + MACD ≥ Signal
    }

    # HMA와 만트라 밴드 (주요 라인)
    line1 = ax_main.plot(ohlcv_data.index, hma, color='blue', linewidth=1.5, label='HMA')[0]
    line2 = ax_main.plot(ohlcv_data.index, upper_band, color='red', linewidth=1, linestyle='--', label='Upper Mantra')[0]
    line3 = ax_main.plot(ohlcv_data.index, lower_band, color='green', linewidth=1, linestyle='--', label='Lower Mantra')[0]
    
    # 가격 라인 추가
    price_line = ax_main.plot(ohlcv_data.index, ohlcv_data['Close'], color='black', linewidth=0.8, alpha=0.7, label='Price')[0]
    
    # 만트라 밴드 영역 채우기
    ax_main.fill_between(ohlcv_data.index, hma, upper_band, color='red', alpha=0.1, label='상단 밴드 영역')
    ax_main.fill_between(ohlcv_data.index, lower_band, hma, color='green', alpha=0.1, label='하단 밴드 영역')
    
    # 볼린저 밴드 추가
    bb_line1 = ax_main.plot(ohlcv_data.index, bb_ma, color='purple', linewidth=0.5, label='BB MA(20)')[0]
    bb_line2 = ax_main.plot(ohlcv_data.index, bb_upper, color='purple', linewidth=0.5, linestyle=':', label='BB Upper')[0]
    bb_line3 = ax_main.plot(ohlcv_data.index, bb_lower, color='purple', linewidth=0.5, linestyle=':', label='BB Lower')[0]

    # 현재가, 지지선, 저항선 계산
    current_price = ohlcv_data['Close'].iloc[-1]
    support, resistance = calculate_support_resistance(ohlcv_data)
    
    # 매수 신호 발생일에서 20일 박스권 표시
    for signal in trade_signals:
        if signal['type'] == 'BUY':
            signal_date = signal['date']
            # 신호 발생일로부터 20일 데이터 확인
            signal_idx = ohlcv_data.index.get_loc(signal_date)
            if signal_idx >= 19:  # 최소 20일 데이터가 있는지 확인
                box_start = signal_idx - 19
                box_end = signal_idx + 1
                box_high = ohlcv_data['High'].iloc[box_start:box_end].max()
                box_low = ohlcv_data['Low'].iloc[box_start:box_end].min()
                # 박스권 영역 표시
                ax_main.fill_between(ohlcv_data.index[box_start:box_end], 
                                   box_low, box_high, 
                                   color='red', alpha=0.12, zorder=0)
    
    # 수평선 및 가격 표시 추가
    # 현재가 라인
    ax_main.axhline(y=current_price, color='black', linestyle=':', linewidth=0.8, alpha=0.5)
    ax_main.text(ohlcv_data.index[-1], current_price, 
                f'현재가: {current_price:.2f}\n{ohlcv_data.index[-1].strftime("%Y-%m-%d")}', 
                rotation=45, fontsize=6, ha='left', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # 지지선
    ax_main.axhline(y=support, color='green', linestyle=':', linewidth=0.8, alpha=0.5)
    ax_main.text(ohlcv_data.index[-1], support, f'지지선: {support:.2f}', 
                rotation=45, fontsize=6, ha='left', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # 저항선
    ax_main.axhline(y=resistance, color='red', linestyle=':', linewidth=0.8, alpha=0.5)
    ax_main.text(ohlcv_data.index[-1], resistance, f'저항선: {resistance:.2f}', 
                rotation=45, fontsize=6, ha='left', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # 매수/매도 구간 배경색 설정
    def add_background_colors(ax, signals):
        buy_periods = []
        current_buy = None
        
        # 신호들을 시간순으로 정렬
        sorted_signals = sorted(signals, key=lambda x: x['date'])
        
        for signal in sorted_signals:
            if signal['type'] == 'BUY' and current_buy is None:
                current_buy = signal['date']
            elif signal['type'] == 'SELL' and current_buy is not None:
                buy_periods.append((current_buy, signal['date']))
                current_buy = None
        
        # 마지막 매수 신호가 있고 매도가 없는 경우, 현재까지를 매수 구간으로 처리
        if current_buy is not None:
            buy_periods.append((current_buy, ohlcv_data.index[-1]))
        
        # 배경색 적용
        for start, end in buy_periods:
            ax.axvspan(start, end, color='lightgreen', alpha=0.08)
            
        # 매도 구간 (매수 구간이 아닌 구간)
        sell_periods = []
        last_end = ohlcv_data.index[0]
        
        for start, end in buy_periods:
            if last_end < start:
                sell_periods.append((last_end, start))
            last_end = end
            
        if last_end < ohlcv_data.index[-1]:
            sell_periods.append((last_end, ohlcv_data.index[-1]))
            
        for start, end in sell_periods:
            ax.axvspan(start, end, color='lightpink', alpha=0.08)

    # 모든 차트에 배경색 적용
    for ax in [ax_main, ax_rsi, ax_macd, ax_volume, ax_vix, ax_tnx, ax_dxy]:
        add_background_colors(ax, trade_signals)

    # 매수/매도 신호 표시
    for signal in trade_signals:
        is_buy = signal['type'] == 'BUY'
        is_strong = False
        signal_key = None
        
        # 신호 강도 및 유형 결정
        if 'reason' in signal and 'macd_state' in signal:
            is_strong = (is_buy and signal['macd_state'] > 0) or (not is_buy and signal['macd_state'] < 0)
            
            if is_buy:
                if 'HMA 상향돌파' in signal['reason']:
                    if 'RSI 매수' in signal['reason']:
                        signal_key = 'B1_strong' if is_strong else 'B1_normal'
                    else:
                        signal_key = 'B3'
                elif '하단밴드' in signal['reason']:
                    signal_key = 'B2_strong' if is_strong else 'B2_normal'
            else:
                if 'HMA 하향돌파' in signal['reason']:
                    signal_key = 'T1_strong' if is_strong else 'T1_normal'
                elif '상단밴드' in signal['reason']:
                    signal_key = 'T2_strong' if is_strong else 'T2_normal'

        # 신호 마커 설정
        color = 'blue' if is_buy else 'red'
        if not is_strong:
            color = (1, 0, 0, 0.5) if color == 'red' else (0, 0, 1, 0.5)  # 반투명 빨강/파랑
        marker = '^' if is_buy else 'v'
        
        # 신호 위치 설정
        if is_buy:
            y = lower_band[signal['date']] * 0.99  # 하단 밴드보다 1% 아래
            # 매수 신호 번호 결정
            if 'reason' in signal:
                if 'HMA 상향돌파' in signal['reason'] and 'RSI 매수' in signal['reason']:
                    signal_num = 'B1'
                elif '하단밴드' in signal['reason']:
                    signal_num = 'B2'
                else:
                    signal_num = 'B3'
            
            # 매수 신호일 때 캔들바와 HMA 위치 비교
            signal_date = signal['date']
            candle_open = ohlcv_data.loc[signal_date, 'Open']
            candle_close = ohlcv_data.loc[signal_date, 'Close']
            hma_value = hma[signal_date]
            
            # 날짜와 신호 번호 표시 (매수 신호)
            date_str = signal['date'].strftime('%m/%d')
            ax_main.text(signal['date'], y * 0.99, f"{date_str}\n{signal_num}", 
                        rotation=45, fontsize=6, ha='right', va='top',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
            # 매수 신호 플롯
            plot = ax_main.plot(signal['date'], y, marker=marker, color=color, 
                              markersize=10, markeredgecolor='black')[0]
            
            # 캔들바 몸체가 HMA보다 위에 있을 때 별표 추가
            if min(candle_open, candle_close) > hma_value:  # 시가와 종가 중 작은 값이 HMA보다 크면
                ax_main.plot(signal_date, y * 0.98, marker='*', color=color,
                           markersize=7.5, markeredgecolor='black', zorder=10)
        else:
            y = upper_band[signal['date']] * 1.01  # 상단 밴드보다 1% 위
            # 매도 신호 번호 결정
            if 'reason' in signal:
                if 'HMA 하향돌파' in signal['reason'] and 'RSI 매도' in signal['reason']:
                    signal_num = 'T1'
                elif '상단밴드' in signal['reason']:
                    signal_num = 'T2'
                else:
                    signal_num = 'T3'
            
            # 매도 신호일 때 캔들바와 HMA 위치 비교
            signal_date = signal['date']
            candle_open = ohlcv_data.loc[signal_date, 'Open']
            candle_close = ohlcv_data.loc[signal_date, 'Close']
            hma_value = hma[signal_date]
            
            # 날짜와 신호 번호 표시 (매도 신호)
            date_str = signal_date.strftime('%m/%d')
            ax_main.text(signal_date, y * 1.01, f"{date_str}\n{signal_num}", 
                        rotation=45, fontsize=6, ha='right', va='bottom',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
            # 매도 신호 플롯
            plot = ax_main.plot(signal_date, y, marker=marker, color=color, 
                              markersize=10, markeredgecolor='black')[0]
            
            # 캔들바 몸체가 HMA보다 아래에 있을 때 별표 추가
            if max(candle_open, candle_close) < hma_value:  # 시가와 종가 중 큰 값이 HMA보다 작으면
                ax_main.plot(signal_date, y * 1.02, marker='*', color=color,
                           markersize=7.5, markeredgecolor='black', zorder=10)
        
        # 범례용 신호 저장
        if signal_key:
            if is_buy and buy_signals[signal_key] is None:
                buy_signals[signal_key] = plot
            elif not is_buy and sell_signals[signal_key] is None:
                sell_signals[signal_key] = plot

    # 메인 차트 범례 구성
    legend_elements = []
    legend_labels = []

    # 1. 캔들 차트 설명
    legend_elements.append((green_candle, red_candle))
    legend_labels.append('캔들 (상승/하락)')
    
    # 가격 라인 범례 추가
    legend_elements.append((price_line,))
    legend_labels.append('가격선')

    # 2. 기술적 지표
    legend_elements.append((line1, line2, line3))
    legend_labels.append('기술선 (HMA/상단밴드/하단밴드)')
    
    # 만트라 밴드 영역 범례 추가
    red_patch = plt.Rectangle((0, 0), 1, 1, fc='red', alpha=0.1)
    green_patch = plt.Rectangle((0, 0), 1, 1, fc='green', alpha=0.1)
    legend_elements.append((red_patch, green_patch))
    legend_labels.append('만트라 밴드 영역 (상단/하단)')
    
    # 볼린저 밴드 범례 추가
    legend_elements.append((bb_line1, bb_line2, bb_line3))
    legend_labels.append('볼린저 밴드 (MA/상단/하단)')

    # 3. 매수 신호 설명
    buy_markers = []
    if buy_signals['B1_strong']: 
        legend_elements.append((buy_signals['B1_strong'],))
        legend_elements.append((buy_signals['B1_normal'],))
        legend_labels.append('B1 강한매수: HMA돌파+RSI매수')
    if buy_signals['B2_strong']: 
        legend_elements.append((buy_signals['B2_strong'],))
        legend_labels.append('B2 강한매수: 밴드하단돌파')
    if buy_signals['B2_normal']: 
        legend_elements.append((buy_signals['B2_normal'],))
        legend_labels.append('B2 일반매수: 밴드하단돌파')
    if buy_signals['B3']: 
        legend_elements.append((buy_signals['B3'],))
        legend_labels.append('B3 매수: MACD > Signal')

    # 4. 매도 신호 설명
    sell_markers = []
    if sell_signals['T1_strong']: 
        legend_elements.append((sell_signals['T1_strong'],))
        legend_labels.append('T1 강한매도: HMA돌파+RSI매도')
    if sell_signals['T1_normal']: 
        legend_elements.append((sell_signals['T1_normal'],))
        legend_labels.append('T1 일반매도: HMA돌파+RSI매도')
    if sell_signals['T2_strong']: 
        legend_elements.append((sell_signals['T2_strong'],))
        legend_labels.append('T2 강한매도: 밴드상단돌파')
    if sell_signals['T2_normal']: 
        legend_elements.append((sell_signals['T2_normal'],))
        legend_labels.append('T2 일반매도: 밴드상단돌파')

    # 5. 신호 강도 설명
    legend_elements.append((strong_buy, normal_buy))
    legend_elements.append((strong_sell, normal_sell))
    legend_labels.append('매수 강도 (강/약)')
    legend_labels.append('매도 강도 (강/약)')

    # 범례 표시 (왼쪽 상단에 위치)
    ax_main.legend(
        [tuple(group) if isinstance(group, (list, tuple)) else group for group in legend_elements],
        legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc='upper left',
        bbox_to_anchor=(0.01, 0.99),
        fontsize=8,
        frameon=True,
        fancybox=True,
        shadow=True,
        title='차트 구성 요소',
        title_fontsize=10,
        framealpha=0.8,  # 범례 배경 투명도
        ncol=1  # 범례를 1열로 표시
    )

    # RSI 플롯
    ax_rsi.plot(ohlcv_data.index, rsi3, color='blue', linewidth=1, label='RSI(3)')
    ax_rsi.plot(ohlcv_data.index, rsi14, color='purple', linewidth=2, label='RSI(14)')
    ax_rsi.plot(ohlcv_data.index, rsi50, color='green', linewidth=1, label='RSI(50)')
    ax_rsi.axhline(y=70, color='red', linestyle='--', alpha=0.5)
    ax_rsi.axhline(y=30, color='green', linestyle='--', alpha=0.5)
    ax_rsi.axhline(y=50, color='black', linestyle='--', alpha=0.5)  # 50 수평선 추가
    ax_rsi.set_ylim([0, 100])
    ax_rsi.set_ylabel('RSI')
    ax_rsi.grid(True, alpha=0.3)

    # RSI 매수/매도 신호 표시
    for signal in trade_signals:
        if signal['type'] == 'BUY':
            # 매수 신호일 때 RSI(3)가 30을 상향 돌파하는 시점
            if 'RSI 매수' in signal.get('reason', ''):
                signal_date = signal['date']
                rsi3_value = rsi3[signal_date]
                # 수직선 표시
                ax_rsi.axvline(x=signal_date, color='blue', linestyle=':', alpha=0.5)
                # 매수 신호 마커 표시
                ax_rsi.plot(signal_date, rsi3_value, '^', color='blue', markersize=8, markeredgecolor='white')
                # 날짜와 RSI 값 표시
                ax_rsi.text(signal_date, rsi3_value, 
                           f' {signal_date.strftime("%m/%d")}\n RSI: {rsi3_value:.1f}', 
                           fontsize=6, ha='left', va='bottom',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        else:
            # 매도 신호일 때 RSI(3)가 70을 하향 돌파하는 시점
            if 'RSI 매도' in signal.get('reason', ''):
                signal_date = signal['date']
                rsi3_value = rsi3[signal_date]
                # 수직선 표시
                ax_rsi.axvline(x=signal_date, color='red', linestyle=':', alpha=0.5)
                # 매도 신호 마커 표시
                ax_rsi.plot(signal_date, rsi3_value, 'v', color='red', markersize=8, markeredgecolor='white')
                # 날짜와 RSI 값 표시
                ax_rsi.text(signal_date, rsi3_value, 
                           f' {signal_date.strftime("%m/%d")}\n RSI: {rsi3_value:.1f}', 
                           fontsize=6, ha='left', va='top',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # RSI 범례 추가
    handles, labels = ax_rsi.get_legend_handles_labels()
    ax_rsi.legend(handles, labels, loc='upper left', bbox_to_anchor=(0, 1), fontsize=8, title='RSI 신호')

    # MACD 차트
    ax_macd.plot(ohlcv_data.index, macd, color='blue', label='MACD', linewidth=2)
    ax_macd.plot(ohlcv_data.index, macd_signal, color='red', label='Signal', linewidth=1)
    ax_macd.bar(ohlcv_data.index, hist, color=['red' if h > 0 else 'blue' for h in hist], label='Histogram')
    
    # MACD 매수/매도 신호 찾기
    macd_signals = []
    prev_diff = 0
    
    for i in range(1, len(ohlcv_data.index)):
        curr_diff = macd.iloc[i] - macd_signal.iloc[i]
        
        # 크로스 발생 여부 확인
        if (curr_diff * prev_diff <= 0) and (curr_diff != prev_diff):
            cross_date = ohlcv_data.index[i]
            
            # MACD와 Signal의 값
            macd_value = macd.iloc[i]
            signal_value = macd_signal.iloc[i]
            
            # 정확한 크로스 지점의 값 계산
            cross_value = (macd_value + signal_value) / 2
            
            # 골드크로스/데드크로스 판단
            # MACD가 Signal선을 상향돌파(골드크로스 - 매수)
            if (prev_diff < 0 and curr_diff >= 0) or (prev_diff == 0 and curr_diff > 0):
                is_golden = True
            # MACD가 Signal선을 하향돌파(데드크로스 - 매도)
            elif (prev_diff > 0 and curr_diff <= 0) or (prev_diff == 0 and curr_diff < 0):
                is_golden = False
            else:
                continue  # 크로스가 아닌 경우 스킵
                
            macd_signals.append((cross_date, cross_value, is_golden))
        prev_diff = curr_diff
    
    # MACD 매수/매도 신호 표시
    for date, value, is_golden in macd_signals:
        if is_golden:  # 골드크로스 (매수)
            ax_macd.plot(date, value, '^', color='blue', markersize=8, markeredgecolor='white')
        else:  # 데드크로스 (매도)
            ax_macd.plot(date, value, 'v', color='red', markersize=8, markeredgecolor='white')
    
    # MACD 신호 더미 마커 (범례용)
    macd_buy = ax_macd.plot([], [], '^', color='blue', markersize=8, markeredgecolor='white', label='MACD 매수 (MACD>Signal)')[0]
    macd_sell = ax_macd.plot([], [], 'v', color='red', markersize=8, markeredgecolor='white', label='MACD 매도 (MACD<Signal)')[0]
    
    ax_macd.set_ylabel('MACD')
    ax_macd.grid(True, alpha=0.3)
    
    # MACD 범례 업데이트
    ax_macd.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=8)

    # 거래량 차트
    volume_colors = ['green' if close > open else 'red' for open, close in zip(ohlcv_data['Open'], ohlcv_data['Close'])]
    ax_volume.bar(ohlcv_data.index, ohlcv_data['Volume'], color=volume_colors, alpha=0.6)
    
    # Volume BB Upper(볼린저밴드 상단) 선 추가
    ax_volume.plot(ohlcv_data.index, volume_upper, color='purple', linestyle='-', linewidth=1.5, label='Volume BB Upper')
    
    # 거래량 범례용 패치 추가
    green_patch = mpatches.Patch(color='green', label='상승 거래량')
    red_patch = mpatches.Patch(color='red', label='하락 거래량')
    handles, labels = ax_volume.get_legend_handles_labels()
    ax_volume.legend(handles + [green_patch, red_patch], labels + ['상승 거래량', '하락 거래량'], loc='upper left', fontsize=8, title='거래량 색상')
    
    # 우측 Y축: 주가 볼린저밴드 상단만
    ax_volume_right = ax_volume.twinx()
    ax_volume_right.plot(ohlcv_data.index, ohlcv_data['Close'], color='blue', linewidth=0.8, alpha=0.7, label='Price')
    ax_volume_right.plot(ohlcv_data.index, bb_upper, color='red', linestyle='-', linewidth=1.5, label='Price BB Upper')
    ax_volume_right.set_ylabel('Price & Price BOL')
    ax_volume_right.legend(loc='upper right', fontsize=8)

    # VIX 이동평균선 계산
    vix_ma10 = vix.rolling(window=10).mean()
    vix_ma20 = vix.rolling(window=20).mean()
    vix_ma30 = vix.rolling(window=30).mean()
    vix_ma60 = vix.rolling(window=60).mean()

    # VIX 차트
    vix_line = ax_vix.plot(vix.index, vix.values, color='black', label='VIX', linewidth=1)[0]
    ma10_line = ax_vix.plot(vix.index, vix_ma10, color='blue', linestyle='-', linewidth=1, label='MA10')[0]
    ma20_line = ax_vix.plot(vix.index, vix_ma20, color='orange', linestyle='-', linewidth=1, label='MA20')[0]
    ma30_line = ax_vix.plot(vix.index, vix_ma30, color='green', linestyle='-', linewidth=1, label='MA30')[0]
    ma60_line = ax_vix.plot(vix.index, vix_ma60, color='purple', linestyle='-', linewidth=1, label='MA60')[0]
    # VIX 구간별 배경색 설정
    vix_max = float(vix.values.max()) * 1.1
    stable_patch = plt.Rectangle((0,0),1,1,fc='green',alpha=0.1,label='안정')
    warning_patch = plt.Rectangle((0,0),1,1,fc='yellow',alpha=0.1,label='주의')
    danger_patch = plt.Rectangle((0,0),1,1,fc='red',alpha=0.1,label='위험')
    ax_vix.axhspan(0, 20, color='green', alpha=0.1)
    ax_vix.axhspan(20, 30, color='yellow', alpha=0.1)
    ax_vix.axhspan(30, vix_max, color='red', alpha=0.1)
    # 범례 추가 (좌측 상단)
    ax_vix.legend([vix_line, ma10_line, ma20_line, ma30_line, ma60_line, stable_patch, warning_patch, danger_patch],
                 ['VIX', 'MA10', 'MA20', 'MA30', 'MA60', '안정', '주의', '위험'],
                 loc='upper left', fontsize=8, ncol=2, frameon=True)
    
    # 현재일자 VIX dot 및 값 표시
    current_vix = float(vix.iloc[-1].item()) if hasattr(vix.iloc[-1], 'item') else float(vix.iloc[-1])
    current_date = vix.index[-1]
    ax_vix.plot(current_date, current_vix, 'o', color='black', markersize=5, zorder=10)
    ax_vix.text(current_date, current_vix, f'{current_vix:.2f}', fontsize=5, rotation=45, ha='left', va='bottom',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5), zorder=11)
    
    # VIX 20일 박스권 계산 및 표시
    if len(vix) >= 20:
        vix_box_high = float(vix[-20:].max().item())
        vix_box_low = float(vix[-20:].min().item())
        vix_box_cur = float(vix.iloc[-1].item())
        # 박스권 영역을 투명한 빨간색으로 표시
        ax_vix.fill_between(vix.index[-20:], vix_box_low, vix_box_high, color='red', alpha=0.12, zorder=0)
        # 수평선(점선) 모두 검은색
        ax_vix.axhline(vix_box_high, color='black', linestyle=':', linewidth=1, alpha=0.7)
        ax_vix.axhline(vix_box_low, color='black', linestyle=':', linewidth=1, alpha=0.7)
        ax_vix.axhline(vix_box_cur, color='black', linestyle=':', linewidth=1, alpha=0.7)
        # 텍스트 위치 우측으로 약간 이동 (x좌표를 vix.index[-1]에서 +1로 이동)
        from matplotlib.dates import date2num, num2date
        last_date = vix.index[-1]
        # x축이 datetime일 경우, 1일 뒤로 이동
        if isinstance(last_date, pd.Timestamp):
            text_x = last_date + pd.Timedelta(days=1)
        else:
            text_x = last_date
        ax_vix.text(text_x, vix_box_high, f'저항선: {vix_box_high:.2f}', fontsize=6, color='black', ha='left', va='bottom', rotation=0, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))
        ax_vix.text(text_x, vix_box_low, f'지지선: {vix_box_low:.2f}', fontsize=6, color='black', ha='left', va='bottom', rotation=0, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))
        ax_vix.text(text_x, vix_box_cur, f'현재값: {vix_box_cur:.2f}', fontsize=6, color='black', ha='left', va='bottom', rotation=0, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))
    
    # TNX 차트 (미국채 10년물 금리)
    ax_tnx.plot(tnx.index, tnx.values, color='green', label='US 10Y Treasury', linewidth=1)
    
    # TNX 수준별 투자 전략 배경색 및 범례(사용자 정의)
    tnx_max = float(tnx.max().item())
    ax_tnx.axhspan(5.0, tnx_max, color='red', alpha=0.18, label='고금리 부담 구간 (≥5.0%)\n- 주식시장에 부정적, 채권 매도 압력')
    ax_tnx.axhspan(4.5, 5.0, color='red', alpha=0.10, label='고금리 압박 본격화 (4.5~5.0%)\n- 고금리 부담 지속, 리스크 확대')
    ax_tnx.axhspan(4.0, 4.5, color='limegreen', alpha=0.15, label='증시 선호 구간 (골디락스, 4.0~4.5%)\n- 증시에 긍정적, 주식-채권 균형 가능')
    ax_tnx.axhspan(0, 4.0, color='royalblue', alpha=0.13, label='침체 우려 확대 구간 (<4.0%)\n- 안전자산 선호, 성장 둔화 반영')
    
    # 현재 TNX 값 표시
    current_tnx = float(tnx.iloc[-1].item())
    current_date = tnx.index[-1]
    
    # 현재 TNX 점 찍기
    ax_tnx.plot(current_date, current_tnx, 'o', color='green', markersize=2,
                markeredgecolor='black', markeredgewidth=0.5, label=f'현재 TNX: {current_tnx:.2f}',
                zorder=10)
    
    # 현재 TNX 값과 날짜 표시
    ax_tnx.text(current_date, current_tnx,
                f' {current_date.strftime("%Y-%m-%d")}\n TNX: {current_tnx:.2f}',
                fontsize=6, ha='left', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1),
                zorder=10)
    
    ax_tnx.set_ylabel('TNX')
    ax_tnx.grid(True, alpha=0.3)
    
    # TNX 범례 (중복 제거)
    handles, labels = ax_tnx.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_tnx.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=8)

    # DXY 차트 (달러 인덱스)
    ax_dxy.plot(dxy.index, dxy.values, color='blue', label='Dollar Index', linewidth=1)
    
    # 현재 DXY 값 표시
    current_dxy = float(dxy.iloc[-1].item())
    current_date = dxy.index[-1]
    
    # 현재 DXY 점 찍기
    ax_dxy.plot(current_date, current_dxy, 'o', color='blue', markersize=2,
                markeredgecolor='black', markeredgewidth=0.5, label=f'현재 DXY: {current_dxy:.2f}',
                zorder=10)
    
    # 현재 DXY 값과 날짜 표시
    ax_dxy.text(current_date, current_dxy,
                f' {current_date.strftime("%Y-%m-%d")}\n DXY: {current_dxy:.2f}',
                fontsize=6, ha='left', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1),
                zorder=10)
    
    ax_dxy.set_ylabel('DXY')
    ax_dxy.grid(True, alpha=0.3)
    ax_dxy.legend(loc='upper right', fontsize=8)

    # High Yield Spread 데이터 불러오기 (FRED)
    try:
        hy_spread = web.DataReader('BAMLH0A0HYM2', 'fred', start=start_date, end=end_date)
    except Exception as e:
        print(f"High Yield Spread 데이터 로드 실패: {e}")
        hy_spread = None
    # High Yield Spread subplot
    if hy_spread is not None:
        ax_hyspread.plot(hy_spread.index, hy_spread['BAMLH0A0HYM2'], color='purple', label='High Yield Spread')
        # 구간별 배경색 및 투자전략
        ax_hyspread.axhspan(0, 3, color='lime', alpha=0.15, 
            label='Risk-On (≤3%)\n- 시장 신뢰 높음, 위험자산 선호\n- 주식·하이일드채권 비중 확대, 공격적 투자')
        ax_hyspread.axhspan(3, 5, color='gold', alpha=0.15, 
            label='Neutral (3~5%)\n- 신용위험 다소 증가, 중립적 시장\n- 분산투자, 리스크 관리, 점진적 비중 조절')
        ax_hyspread.axhspan(5, hy_spread['BAMLH0A0HYM2'].max(), color='pink', alpha=0.15, 
            label='Risk-Off (≥5%)\n- 시장 불안, 신용위험 급등\n- 안전자산 비중 확대, 위험자산 축소')
        # 신호 발생일 세로선
        if 'trade_signals' in locals():
            for signal in trade_signals:
                if 'date' in signal:
                    ax_hyspread.axvline(signal['date'], color='red', linestyle='--', alpha=0.3)
        ax_hyspread.set_title('High Yield Spread (BAMLH0A0HYM2)')
        ax_hyspread.set_ylabel('Spread (%)')
        
        # 현재 값 우측 상단에 표시
        current_value = hy_spread['BAMLH0A0HYM2'].iloc[-1]
        ax_hyspread.text(0.98, 0.98, f'현재: {current_value:.2f}%', 
                        transform=ax_hyspread.transAxes, fontsize=10, 
                        ha='right', va='top', 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='black'))
        
        # 범례(중복 제거) - 좌측 상단으로 위치 변경
        handles, labels = ax_hyspread.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_hyspread.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=8)
    else:
        ax_hyspread.set_title('High Yield Spread (데이터 없음)')

    # 차트 제목 및 레이아웃
    start_date = ohlcv_data.index[0].strftime('%Y-%m-%d')
    end_date = ohlcv_data.index[-1].strftime('%Y-%m-%d')
    ax_main.set_title(f'{ticker} - Technical Analysis ({start_date} ~ {end_date})')

    # x축 날짜 포맷 설정
    for ax in [ax_main, ax_rsi, ax_macd, ax_volume, ax_vix, ax_tnx, ax_dxy]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.grid(True, alpha=0.3)

    # 레이아웃 조정
    plt.tight_layout()
    # 범례를 위한 추가 여백 제거 (이미 차트 내부에 있으므로)
    plt.subplots_adjust(right=0.95)

    # --- 조건부 신호 표시 (사용자 요청) ---
    # 볼륨 볼린저밴드 상단 계산
    volume_ma = ohlcv_data['Volume'].rolling(window=20).mean()
    volume_std = ohlcv_data['Volume'].rolling(window=20).std()
    volume_upper = volume_ma + (volume_std * 2)

    # MACD 골든크로스 구간 찾기
    macd_above_signal = (macd > macd_signal)
    macd_cross = (macd.shift(1) <= macd_signal.shift(1)) & (macd > macd_signal)  # 골든크로스 발생일
    macd_golden_zone = macd_above_signal.cumsum()  # 골든크로스 이후 구간 마스킹용
    macd_golden_mask = macd_golden_zone > 0

    # 조건에 맞는 날짜 찾기
    cond = (
        (ohlcv_data['Volume'] > volume_upper) &
        macd_golden_mask &
        (rsi14 >= 50)
    )
    cond_dates = ohlcv_data.index[cond]

    # 모든 subplot 리스트
    all_axes = [ax_main, ax_rsi, ax_macd, ax_volume, ax_vix, ax_tnx, ax_dxy, ax_hyspread]

    for dt in cond_dates:
        # 모든 subplot에 수직선
        for ax in all_axes:
            ax.axvline(dt, color='magenta', linestyle='--', alpha=0.7, linewidth=1.2, zorder=20)
            # 상단 텍스트 제거, 하단만 남김
            if ax is ax_main:
                # 하단에 일자 텍스트 표시 (45도 기울임)
                ax.text(dt, ax.get_ylim()[0], dt.strftime('%Y-%m-%d'), fontsize=6, color='magenta',
                        ha='center', va='top', rotation=45,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5), zorder=25)
        # 메인차트에 종가 기준 수평선 (양봉/음봉 구분)
        close = ohlcv_data.loc[dt, 'Close']
        open_ = ohlcv_data.loc[dt, 'Open']
        if close >= open_:
            ax_main.axhline(close, color='lime', linestyle='-', linewidth=0.6, alpha=0.8, xmin=0, xmax=1, zorder=21)
            ax_main.text(ohlcv_data.index[-1], close, f'{close:.2f}', fontsize=7, color='black', ha='left', va='center',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5), zorder=22)
        else:
            ax_main.axhline(close, color='red', linestyle='-', linewidth=0.6, alpha=0.8, xmin=0, xmax=1, zorder=21)
            ax_main.text(ohlcv_data.index[-1], close, f'{close:.2f}', fontsize=7, color='black', ha='left', va='center',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5), zorder=22)

    # 저장 또는 표시
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show() 