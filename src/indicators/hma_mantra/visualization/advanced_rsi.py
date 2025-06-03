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
    try:
        # 한글 폰트 설정
        plt.rcParams['font.family'] = get_available_font()
        plt.rcParams['axes.unicode_minus'] = False
        
        # 단일 인덱스 데이터프레임으로 변환
        if isinstance(data.columns, pd.MultiIndex):
            ohlcv_data = data.xs(ticker, axis=1, level=1)
        else:
            ohlcv_data = data

        # 기술적 지표 계산
        hma = calculate_hma(ohlcv_data['Close'])
        upper_mantra, lower_mantra = calculate_mantra_bands(ohlcv_data['Close'])
        macd, macd_signal, macd_hist = calculate_macd(ohlcv_data['Close'])
        rsi = calculate_rsi(ohlcv_data['Close'])
        
        # MACD 디버깅 출력
        print('MACD:', macd.head(10))
        print('MACD Signal:', macd_signal.head(10))
        print('MACD Hist:', macd_hist.head(10))
        print('MACD shape:', macd.shape, macd_signal.shape, macd_hist.shape)
        
        # 신호 생성
        hma_signals = get_hma_mantra_md_signals(ohlcv_data, ticker)
        
        # 그래프 생성
        fig, (ax_main, ax_tajmahal_rsi, ax_macd, ax_rsi, ax_volume) = plt.subplots(
            5, 1, figsize=(15, 20), sharex=True,
            gridspec_kw={'height_ratios': [3, 1, 1, 1, 1]}
        )
        
        # 캔들스틱 데이터 준비
        ohlc = np.array([
            [mdates.date2num(date), 
             float(ohlcv_data.loc[date, 'Open']), 
             float(ohlcv_data.loc[date, 'High']), 
             float(ohlcv_data.loc[date, 'Low']), 
             float(ohlcv_data.loc[date, 'Close'])] 
            for date in ohlcv_data.index
        ])
        
        # 캔들스틱 차트
        candlestick_ohlc(ax_main, ohlc, width=0.6, colorup='green', colordown='red', alpha=0.6)
        
        # HMA, 만트라 밴드 플롯
        ax_main.plot(ohlcv_data.index, ohlcv_data['Close'], color='black', alpha=0.5, linewidth=1.2)
        ax_main.plot(ohlcv_data.index, hma, color='blue', linewidth=2)
        ax_main.plot(ohlcv_data.index, upper_mantra, color='red', linestyle='--')
        ax_main.plot(ohlcv_data.index, lower_mantra, color='green', linestyle='--')
        
        # 타지마할 RSI 플롯 (범례 없이)
        ax_tajmahal_rsi.plot(ohlcv_data.index, rsi, color='purple', linewidth=1.5)
        ax_tajmahal_rsi.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax_tajmahal_rsi.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax_tajmahal_rsi.set_ylim([0, 100])
        ax_tajmahal_rsi.set_ylabel('RSI')
        ax_tajmahal_rsi.grid(True, alpha=0.3)
        
        # MACD 플롯
        ax_macd.plot(ohlcv_data.index, macd, color='blue', label='MACD', linewidth=1, zorder=3)
        ax_macd.plot(ohlcv_data.index, macd_signal, color='red', label='Signal', linewidth=1, zorder=3)
        # Histogram 색상: 기본값(변경 없음)
        hist_dummy = ax_macd.bar([0], [0], label='Histogram')[0]
        
        ax_macd.set_facecolor('white')
        
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
                ax_macd.plot(date, value, '^', color='blue', markersize=8, markeredgecolor='white', zorder=4)
            else:  # 데드크로스 (매도)
                ax_macd.plot(date, value, 'v', color='red', markersize=8, markeredgecolor='white', zorder=4)
        
        # MACD 신호 더미 마커 (범례용)
        macd_buy = ax_macd.plot([], [], '^', color='blue', markersize=8, markeredgecolor='white', label='MACD 매수 (골든크로스)')[0]
        macd_sell = ax_macd.plot([], [], 'v', color='red', markersize=8, markeredgecolor='white', label='MACD 매도 (데드크로스)')[0]
        hist_dummy = ax_macd.bar([0], [0], label='Histogram')[0]
        macd_line = ax_macd.plot([], [], color='blue', label='MACD')[0]
        signal_line = ax_macd.plot([], [], color='red', label='Signal')[0]
        handles = [macd_line, signal_line, hist_dummy, macd_buy, macd_sell]
        labels = ['MACD', 'Signal', 'Histogram', 'MACD 매수 (골든크로스)', 'MACD 매도 (데드크로스)']
        ax_macd.legend(handles, labels, loc='upper left', fontsize=8)
        
        ax_macd.set_ylabel('MACD', loc='top', rotation=0, labelpad=30, fontsize=10)
        ax_macd.grid(True, alpha=0.3)
        
        # RSI 플롯
        ax_rsi.plot(ohlcv_data.index, rsi, color='purple', linewidth=1)
        ax_rsi.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax_rsi.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax_rsi.set_ylim([0, 100])
        ax_rsi.set_ylabel('RSI')
        ax_rsi.grid(True, alpha=0.3)
        
        # 거래량 플롯
        ax_volume.bar(ohlcv_data.index, ohlcv_data['Volume'], color='gray', alpha=0.3)
        ax_volume.set_ylabel('Volume')
        ax_volume.grid(True, alpha=0.3)
        
        # 신호 표시
        for signal in hma_signals:
            color = 'blue' if signal['type'] == 'BUY' else 'red'
            marker = '^' if signal['type'] == 'BUY' else 'v'
            
            # 신호 위치 설정
            if signal['type'] == 'BUY':
                y = lower_mantra[signal['date']] * 0.99
                date_str = signal['date'].strftime('%m/%d')
                ax_main.text(signal['date'], y * 0.99, f"{date_str}", 
                            rotation=45, fontsize=6, ha='right', va='top',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                ax_main.plot(signal['date'], y, marker=marker, color=color, 
                           markersize=10, markeredgecolor='black')
            else:
                y = upper_mantra[signal['date']] * 1.01
                date_str = signal['date'].strftime('%m/%d')
                ax_main.text(signal['date'], y * 1.01, f"{date_str}", 
                            rotation=45, fontsize=6, ha='right', va='bottom',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                ax_main.plot(signal['date'], y, marker=marker, color=color, 
                           markersize=10, markeredgecolor='black')
        
        # 현재가 표시
        if current_price:
            ax_main.axhline(y=current_price, color='blue', linestyle='--', alpha=0.5)
            ax_main.text(ohlcv_data.index[-1], current_price, f'현재가: {current_price:.2f}',
                        color='blue', ha='right', va='bottom')
        
        # 그래프 저장 또는 표시
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

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
    
    # 볼린저 밴드 계산
    bb_ma, bb_upper, bb_lower = calculate_bollinger_bands(ohlcv_data['Close'])

    # 신호 생성
    trade_signals = get_hma_mantra_md_signals(ohlcv_data, ticker)

    # 차트 생성
    fig = plt.figure(figsize=(15, 30))
    # 메인 차트에 범례를 위한 여유 공간 확보
    gs = GridSpec(7, 1, height_ratios=[4, 1, 1, 1, 1, 1, 1], figure=fig)
    
    # 메인 차트
    ax_main = fig.add_subplot(gs[0])
    ax_rsi = fig.add_subplot(gs[1], sharex=ax_main)
    ax_macd = fig.add_subplot(gs[2], sharex=ax_main)
    ax_volume = fig.add_subplot(gs[3], sharex=ax_main)
    ax_vix = fig.add_subplot(gs[4], sharex=ax_main)
    ax_tnx = fig.add_subplot(gs[5], sharex=ax_main)
    ax_dxy = fig.add_subplot(gs[6], sharex=ax_main)

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
            ax.axvspan(start, end, color='lightgreen', alpha=0.2)
            
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
            ax.axvspan(start, end, color='lightpink', alpha=0.2)

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

    # RSI 차트
    rsi3_line = ax_rsi.plot(ohlcv_data.index, rsi3, color='red', label='RSI(3)', linewidth=1)[0]
    rsi14_line = ax_rsi.plot(ohlcv_data.index, rsi14, color='purple', label='RSI(14)', linewidth=1)[0]
    rsi50_line = ax_rsi.plot(ohlcv_data.index, rsi50, color='green', label='RSI(50)', linewidth=1)[0]
    
    # 매수/매도 구간 라인
    sell_line = ax_rsi.axhline(y=70, color='r', linestyle='--', alpha=0.3, label='매도 구간 (70)')
    buy_line = ax_rsi.axhline(y=30, color='g', linestyle='--', alpha=0.3, label='매수 구간 (30)')
    
    # 신호 마커 저장용 리스트
    buy_markers_b0 = []
    buy_markers_b1 = []
    buy_markers_b2 = []
    sell_markers_s0 = []
    sell_markers_s1 = []
    sell_markers_s2 = []
    prev_diff = 0
    for i in range(1, len(ohlcv_data.index)):
        curr_diff = rsi3.iloc[i] - rsi14.iloc[i]
        date = ohlcv_data.index[i]
        rsi3_value = rsi3.iloc[i]
        rsi14_value = rsi14.iloc[i]
        rsi50_value = rsi50.iloc[i]
        # 골든/데드크로스 판별
        is_golden = False
        is_cross = False
        if (curr_diff * prev_diff <= 0) and (curr_diff != prev_diff):
            is_cross = True
            if (prev_diff < 0 and curr_diff >= 0) or (prev_diff == 0 and curr_diff > 0):
                is_golden = True
            elif (prev_diff > 0 and curr_diff <= 0) or (prev_diff == 0 and curr_diff < 0):
                is_golden = False
            else:
                continue
        prev_diff = curr_diff
        if not is_cross:
            continue
        # 매수 신호
        if is_golden:
            marker_b0 = ax_rsi.plot(date, rsi3_value, '^', color='gray', markersize=8, markeredgecolor='white', label='약한 매수 신호')[0]
            ax_rsi.axvline(x=date, color='gray', linestyle='--', alpha=0.3)
            buy_markers_b0.append(marker_b0)
            if rsi3_value > 30 and rsi14_value > 30 and (rsi3.iloc[i-1] <= 30 or rsi14.iloc[i-1] <= 30):
                marker_b1 = ax_rsi.plot(date, rsi3_value, '^', color='blue', markersize=8, markeredgecolor='white', label='매수 신호')[0]
                ax_rsi.axvline(x=date, color='blue', linestyle='--', alpha=0.3)
                buy_markers_b1.append(marker_b1)
                if rsi50_value > 50 and rsi50.iloc[i-1] <= 50:
                    marker_b2 = ax_rsi.plot(date, rsi3_value, '^', color='black', markersize=8, markeredgecolor='white', label='강한 매수 신호')[0]
                    ax_rsi.axvline(x=date, color='black', linestyle='--', alpha=0.3)
                    buy_markers_b2.append(marker_b2)
        # 매도 신호
        if not is_golden:
            marker_s0 = ax_rsi.plot(date, rsi3_value, 'v', color='orange', markersize=8, markeredgecolor='white', label='약한 매도 신호')[0]
            ax_rsi.axvline(x=date, color='orange', linestyle='--', alpha=0.3)
            sell_markers_s0.append(marker_s0)
            if rsi3_value < 70 and rsi14_value < 70 and (rsi3.iloc[i-1] >= 70 or rsi14.iloc[i-1] >= 70):
                marker_s1 = ax_rsi.plot(date, rsi3_value, 'v', color='red', markersize=8, markeredgecolor='white', label='매도 신호')[0]
                ax_rsi.axvline(x=date, color='red', linestyle='--', alpha=0.3)
                sell_markers_s1.append(marker_s1)
                if rsi50_value < 50 and rsi50.iloc[i-1] >= 50:
                    marker_s2 = ax_rsi.plot(date, rsi3_value, 'v', color='#8B0000', markersize=8, markeredgecolor='white', label='강한 매도 신호')[0]
                    ax_rsi.axvline(x=date, color='#8B0000', linestyle='--', alpha=0.3)
                    sell_markers_s2.append(marker_s2)
    
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel('타지마할 RSI')
    ax_rsi.grid(True, alpha=0.3)
    
    # 범례 구성
    handles = [rsi3_line, rsi14_line, rsi50_line, buy_line, sell_line]
    labels = ['RSI(3)', 'RSI(14)', 'RSI(50)', '매수 구간 (30)', '매도 구간 (70)']
    # 매수 신호 마커(더미) 추가
    if buy_markers_b0:
        handles.append(buy_markers_b0[0])
        labels.append('약한 매수 신호')
    if buy_markers_b1:
        handles.append(buy_markers_b1[0])
        labels.append('매수 신호')
    if buy_markers_b2:
        handles.append(buy_markers_b2[0])
        labels.append('강한 매수 신호')
    # 매도 신호 마커(더미) 추가
    if sell_markers_s0:
        handles.append(sell_markers_s0[0])
        labels.append('약한 매도 신호')
    if sell_markers_s1:
        handles.append(sell_markers_s1[0])
        labels.append('매도 신호')
    if sell_markers_s2:
        handles.append(sell_markers_s2[0])
        labels.append('강한 매도 신호')
    ax_rsi.legend(handles, labels, loc='upper left', fontsize=8)

    # 거래량 차트
    volume_colors = ['red' if c >= o else 'blue' for o, c in zip(ohlcv_data['Open'], ohlcv_data['Close'])]
    ax_volume.bar(ohlcv_data.index, ohlcv_data['Volume'], color=volume_colors, alpha=0.7)
    
    # 전체 기간 평균 거래량 계산
    avg_volume = ohlcv_data['Volume'].mean()
    
    # 평균 거래량 선 추가
    ax_volume.axhline(y=avg_volume, color='green', linestyle='--', linewidth=1, label=f'평균 거래량: {avg_volume:,.0f}')
    
    ax_volume.set_ylabel('Volume')
    ax_volume.grid(True, alpha=0.3)
    ax_volume.legend(loc='upper right', fontsize=8)

    # VIX 차트
    ax_vix.plot(vix.index, vix.values, color='black', label='VIX', linewidth=1)
    
    # VIX 구간별 배경색 설정
    vix_max = float(vix.values.max()) * 1.1
    ax_vix.axhspan(0, 20, color='green', alpha=0.1)  # 안전구간
    ax_vix.axhspan(20, 30, color='yellow', alpha=0.1)  # 주의구간
    ax_vix.axhspan(30, vix_max, color='red', alpha=0.1)  # 위험구간
    
    # VIX 기준선
    vix_safe = ax_vix.axhline(y=20, color='g', linestyle='--', alpha=0.5, label='안전구간 경계선 (VIX=20)')
    vix_danger = ax_vix.axhline(y=30, color='r', linestyle='--', alpha=0.5, label='위험구간 경계선 (VIX=30)')
    
    # 현재 VIX 값 표시
    current_vix = float(vix.iloc[-1].item())  # Series 값을 float로 변환
    current_date = vix.index[-1]
    
    # VIX 값에 따른 투자전략 색상 및 메시지 결정
    if current_vix < 20:
        dot_color = 'green'
        strategy = '적극적 매수구간'
        strategy_detail = '성장주/레버리지 ETF 투자 고려'
    elif current_vix <= 30:
        dot_color = 'yellow'
        strategy = '중립적 관망구간'
        strategy_detail = '분할 매수/리스크 관리'
    else:
        dot_color = 'red'
        strategy = '보수적 관망구간'
        strategy_detail = '현금 보유/안전자산 선호'
    
    # 현재 VIX 점 찍기
    ax_vix.plot(current_date, current_vix, 'o', color=dot_color, markersize=2, 
                markeredgecolor='black', markeredgewidth=0.5, label=f'현재 VIX: {current_vix:.2f}',
                zorder=10)  # 높은 zorder 값으로 최상위 레이어에 표시
    
    # 현재 VIX 값과 날짜만 표시 (우측에)
    text_x = current_date  # x 좌표는 현재 날짜
    text_y = current_vix   # y 좌표는 현재 VIX 값
    ax_vix.text(text_x, text_y, 
                f' {current_date.strftime("%Y-%m-%d")}\n VIX: {current_vix:.2f}', 
                fontsize=6, ha='left', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1),
                zorder=10)  # 텍스트도 최상위 레이어에 표시
    
    ax_vix.set_ylabel('VIX')
    ax_vix.grid(True, alpha=0.3)
    
    # VIX 투자전략 범례
    strategy_elements = [
        plt.Rectangle((0,0), 1, 1, fc='green', alpha=0.3, label='적극적 매수구간 (VIX < 20)\n- 시장 안정/낮은 변동성\n- 성장주/레버리지 ETF 고려'),
        plt.Rectangle((0,0), 1, 1, fc='yellow', alpha=0.3, label='중립적 관망구간 (20 ≤ VIX ≤ 30)\n- 불확실성 증가\n- 분할 매수/리스크 관리'),
        plt.Rectangle((0,0), 1, 1, fc='red', alpha=0.3, label='보수적 관망구간 (VIX > 30)\n- 높은 변동성/공포감\n- 현금 보유/안전자산 선호')
    ]
    
    # 기존 라인과 새로운 전략 범례 결합
    handles, labels = ax_vix.get_legend_handles_labels()
    all_handles = handles + strategy_elements
    all_labels = labels + [
        '적극적 매수구간 (VIX < 20)\n- 시장 안정/낮은 변동성\n- 성장주/레버리지 ETF 고려',
        '중립적 관망구간 (20 ≤ VIX ≤ 30)\n- 불확실성 증가\n- 분할 매수/리스크 관리',
        '보수적 관망구간 (VIX > 30)\n- 높은 변동성/공포감\n- 현금 보유/안전자산 선호'
    ]
    
    # 범례 위치 및 스타일 설정
    ax_vix.legend(all_handles, all_labels, 
                 loc='upper left', 
                 fontsize=8,
                 bbox_to_anchor=(0.01, 1.0))

    # TNX 차트 (미국채 10년물 금리)
    ax_tnx.plot(tnx.index, tnx.values, color='green', label='US 10Y Treasury', linewidth=1)
    
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
    ax_tnx.legend(loc='upper right', fontsize=8)

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

    # 저장 또는 표시
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show() 