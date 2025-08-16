"""
Volume Profile을 메인차트에 오버레이로 표시하는 시각화 모듈
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

def analyze_rsi_divergence_patterns(data, rsi_period=14, pattern_range=(9, 11)):
    """
    RSI 14 기준으로 9~11일 패턴의 다이버전스를 분석합니다.
    
    Args:
        data: OHLCV 데이터
        rsi_period: RSI 계산 기간 (기본값: 14)
        pattern_range: 패턴 분석 기간 범위 (최소, 최대) - 기본값: (9, 11)
    
    Returns:
        list: 감지된 다이버전스 패턴들
    """
    try:
        # RSI 계산
        rsi = calculate_rsi(data['Close'], rsi_period)
        
        # 피벗 포인트 찾기 (고점/저점)
        price_highs, price_lows = _find_pivots(data['Close'], left=3, right=3)
        rsi_highs, rsi_lows = _find_pivots(rsi, left=3, right=3)
        
        divergences = []
        min_days, max_days = pattern_range
        
        # 일반 상승다이버전스: 가격 LL, RSI HL (저점 기준)
        for i in range(len(price_lows) - 1):
            for j in range(i + 1, len(price_lows)):
                try:
                    days_diff = price_lows[j][0] - price_lows[i][0]
                    if min_days <= days_diff <= max_days:
                        # 가격이 더 낮아졌는데 RSI가 더 높아진 경우
                        if (price_lows[j][1] < price_lows[i][1] and 
                            any(rsi_lows[k][1] > rsi_lows[i][1] for k in range(len(rsi_lows)) if rsi_lows[k][0] >= price_lows[j][0])):
                            
                            # 신뢰도 계산
                            price_change = abs(price_lows[j][1] - price_lows[i][1]) / price_lows[i][1]
                            confidence = min(price_change * 10, 1.0)  # 가격 변화율 기반 신뢰도
                            
                            if confidence >= 0.7:  # 70% 이상 신뢰도
                                # 인덱스 범위 확인
                                if (0 <= price_lows[i][0] < len(data.index) and 
                                    0 <= price_lows[j][0] < len(data.index)):
                                    # 강도 계산 (5단계)
                                    if confidence >= 0.95:
                                        strength = '매우 강함'
                                    elif confidence >= 0.90:
                                        strength = '강함'
                                    elif confidence >= 0.80:
                                        strength = '보통'
                                    else:
                                        strength = '약함'
                                    
                                    # 가격 변화율 계산
                                    price_change_pct = abs(price_lows[j][1] - price_lows[i][1]) / price_lows[i][1] * 100
                                    
                                    divergences.append({
                                        'type': 'REGULAR_BULLISH',
                                        'start_date': data.index[price_lows[i][0]],
                                        'end_date': data.index[price_lows[j][0]],
                                        'price1': price_lows[i][1],
                                        'price2': price_lows[j][1],
                                        'pattern_days': days_diff,
                                        'confidence': confidence,
                                        'strength': strength,
                                        'price_change_pct': price_change_pct
                                    })
                except Exception as e:
                    continue
        
        # 일반 하락다이버전스: 가격 HH, RSI LH (고점 기준)
        for i in range(len(price_highs) - 1):
            for j in range(i + 1, len(price_highs)):
                try:
                    days_diff = price_highs[j][0] - price_highs[i][0]
                    if min_days <= days_diff <= max_days:
                        # 가격이 더 높아졌는데 RSI가 더 낮아진 경우
                        if (price_highs[j][1] > price_highs[i][1] and 
                            any(rsi_highs[k][1] < rsi_highs[i][1] for k in range(len(rsi_highs)) if rsi_highs[k][0] >= price_highs[j][0])):
                            
                            # 신뢰도 계산
                            price_change = abs(price_highs[j][1] - price_highs[i][1]) / price_highs[i][1]
                            confidence = min(price_change * 10, 1.0)
                            
                            if confidence >= 0.7:
                                # 인덱스 범위 확인
                                if (0 <= price_highs[i][0] < len(data.index) and 
                                    0 <= price_highs[j][0] < len(data.index)):
                                    # 강도 계산 (5단계)
                                    if confidence >= 0.95:
                                        strength = '매우 강함'
                                    elif confidence >= 0.90:
                                        strength = '보통'
                                    elif confidence >= 0.80:
                                        strength = '보통'
                                    else:
                                        strength = '약함'
                                    
                                    # 가격 변화율 계산
                                    price_change_pct = abs(price_highs[j][1] - price_highs[i][1]) / price_highs[i][1] * 100
                                    
                                    divergences.append({
                                        'type': 'REGULAR_BEARISH',
                                        'start_date': data.index[price_highs[i][0]],
                                        'end_date': data.index[price_highs[j][0]],
                                        'price1': price_highs[i][1],
                                        'price2': price_highs[j][1],
                                        'pattern_days': days_diff,
                                        'confidence': confidence,
                                        'strength': strength,
                                        'price_change_pct': price_change_pct
                                    })
                except Exception as e:
                    continue
        
        # 히든 상승다이버전스: 가격 HL, RSI LL (저점 기준)
        for i in range(len(price_lows) - 1):
            for j in range(i + 1, len(price_lows)):
                try:
                    days_diff = price_lows[j][0] - price_lows[i][0]
                    if min_days <= days_diff <= max_days:
                        # 가격이 더 높아졌는데 RSI가 더 낮아진 경우 (상승 추세에서)
                        if (price_lows[j][1] > price_lows[i][1] and 
                            any(rsi_lows[k][1] < rsi_lows[i][1] for k in range(len(rsi_lows)) if rsi_lows[k][0] >= price_lows[j][0])):
                            
                            price_change = abs(price_lows[j][1] - price_lows[i][1]) / price_lows[i][1]
                            confidence = min(price_change * 8, 1.0)  # 히든은 약간 낮은 신뢰도
                            
                            if confidence >= 0.7:
                                # 인덱스 범위 확인
                                if (0 <= price_lows[i][0] < len(data.index) and 
                                    0 <= price_lows[j][0] < len(data.index)):
                                    # 강도 계산 (5단계)
                                    if confidence >= 0.95:
                                        strength = '매우 강함'
                                    elif confidence >= 0.90:
                                        strength = '강함'
                                    elif confidence >= 0.80:
                                        strength = '보통'
                                    else:
                                        strength = '약함'
                                    
                                    # 가격 변화율 계산
                                    price_change_pct = abs(price_lows[j][1] - price_lows[i][1]) / price_lows[i][1] * 100
                                    
                                    divergences.append({
                                        'type': 'HIDDEN_BULLISH',
                                        'start_date': data.index[price_lows[i][0]],
                                        'end_date': data.index[price_lows[j][0]],
                                        'price1': price_lows[i][1],
                                        'price2': price_lows[j][1],
                                        'pattern_days': days_diff,
                                        'confidence': confidence,
                                        'strength': strength,
                                        'price_change_pct': price_change_pct
                                    })
                except Exception as e:
                    continue
        
        # 히든 하락다이버전스: 가격 LH, RSI HH (고점 기준)
        for i in range(len(price_highs) - 1):
            for j in range(i + 1, len(price_highs)):
                try:
                    days_diff = price_highs[j][0] - price_highs[i][0]
                    if min_days <= days_diff <= max_days:
                        # 가격이 더 낮아졌는데 RSI가 더 높아진 경우 (하락 추세에서)
                        if (price_highs[j][1] < price_highs[i][1] and 
                            any(rsi_highs[k][1] > rsi_highs[i][1] for k in range(len(rsi_highs)) if rsi_highs[k][0] >= price_highs[j][0])):
                            
                            price_change = abs(price_highs[j][1] - price_highs[i][1]) / price_highs[i][1]
                            confidence = min(price_change * 8, 1.0)
                            
                            if confidence >= 0.7:
                                # 인덱스 범위 확인
                                if (0 <= price_highs[i][0] < len(data.index) and 
                                    0 <= price_highs[j][0] < len(data.index)):
                                    # 강도 계산 (5단계)
                                    if confidence >= 0.95:
                                        strength = '매우 강함'
                                    elif confidence >= 0.90:
                                        strength = '강함'
                                    elif confidence >= 0.80:
                                        strength = '보통'
                                    else:
                                        strength = '약함'
                                    
                                    # 가격 변화율 계산
                                    price_change_pct = abs(price_highs[j][1] - price_highs[i][1]) / price_highs[i][1] * 100
                                    
                                    divergences.append({
                                        'type': 'HIDDEN_BEARISH',
                                        'start_date': data.index[price_highs[i][0]],
                                        'end_date': data.index[price_highs[j][0]],
                                        'price1': price_highs[i][1],
                                        'price2': price_highs[j][1],
                                        'pattern_days': days_diff,
                                        'confidence': confidence,
                                        'strength': strength,
                                        'price_change_pct': price_change_pct
                                    })
                except Exception as e:
                    continue
        
        return divergences
        
    except Exception as e:
        print(f"RSI 다이버전스 패턴 분석 중 오류: {e}")
        return []

def plot_rsi_divergence_patterns(ax_main, divergences, data):
    """
    메인차트에 RSI 다이버전스 패턴을 별도로 표시합니다.
    
    Args:
        ax_main: 메인 차트 축
        divergences: 감지된 다이버전스들
        data: OHLCV 데이터
    """
    if not divergences:
        return
    
    # 색상 및 라벨 정의
    colors = {
        'REGULAR_BULLISH': 'darkgreen',
        'REGULAR_BEARISH': 'darkred',
        'HIDDEN_BULLISH': 'green',
        'HIDDEN_BEARISH': 'red'
    }
    
    labels = {
        'REGULAR_BULLISH': '일반 상승다이버전스',
        'REGULAR_BEARISH': '일반 하락다이버전스',
        'HIDDEN_BULLISH': '히든 상승다이버전스',
        'HIDDEN_BEARISH': '히든 하락다이버전스'
    }
    
    for div in divergences:
        try:
            # 시작점과 끝점 날짜 (실제 날짜 사용)
            start_date = div['start_date']
            end_date = div['end_date']
            
            # 가격선 연결 (굵은 점선) - 실제 날짜 사용
            ax_main.plot([start_date, end_date], [div['price1'], div['price2']], 
                        color=colors[div['type']], linestyle='--', linewidth=3, alpha=0.8,
                        zorder=1001)
            
            # 다이버전스 라벨
            strength_text = " (강함)" if div['strength'] == 'strong' else " (보통)"
            confidence_text = f"신뢰도: {div['confidence']:.1%}"
            period_text = f"기간: {div['pattern_days']}일"
            
            label = f"{labels[div['type']]}{strength_text}\n{confidence_text}\n{period_text}"
            
            # 라벨 위치 계산 (가격선 위 또는 아래)
            y_offset = 3 if div['type'] in ['REGULAR_BULLISH', 'HIDDEN_BULLISH'] else -3
            
            ax_main.annotate(label, 
                             xy=(end_date, div['price2']),
                             xytext=(end_date + pd.Timedelta(days=8), div['price2'] + y_offset),
                             fontsize=9,
                             color=colors[div['type']],
                             fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.4', 
                                      facecolor='white', 
                                      edgecolor=colors[div['type']],
                                      linewidth=2,
                                      alpha=0.95),
                             arrowprops=dict(arrowstyle='->', 
                                           color=colors[div['type']],
                                           linewidth=2,
                                           alpha=0.9),
                             zorder=1002)
            
            # 시작점과 끝점에 마커 표시
            ax_main.scatter([start_date, end_date], [div['price1'], div['price2']], 
                           color=colors[div['type']], s=40, alpha=0.9, zorder=1003,
                           edgecolors='black', linewidth=1)
            
        except Exception as e:
            # 개별 다이버전스 처리 중 오류 발생 시 건너뛰기
            continue

def plot_rsi_divergence_on_main_chart(ax_main, divergences, data):
    """
    메인 차트에 RSI 다이버전스 패턴을 오버레이로 표시합니다.
    
    Args:
        ax_main: 메인 차트 축
        divergences: 감지된 다이버전스들
        data: OHLCV 데이터
    """
    if not divergences:
        return []
    
    # 색상 및 라벨 정의 (메인 차트용 - 더 작고 깔끔하게)
    colors = {
        'REGULAR_BULLISH': 'darkgreen',
        'REGULAR_BEARISH': 'darkred',
        'HIDDEN_BULLISH': 'green',
        'HIDDEN_BEARISH': 'red'
    }
    
    # 강도별 선 스타일 정의
    line_styles = {
        '매우 강함': {'linestyle': ':', 'linewidth': 2.5, 'alpha': 0.9},
        '강함': {'linestyle': ':', 'linewidth': 2.0, 'alpha': 0.8},
        '보통': {'linestyle': ':', 'linewidth': 1.5, 'alpha': 0.7},
        '약함': {'linestyle': ':', 'linewidth': 1.0, 'alpha': 0.6}
    }
    
    # 범례용 라벨 (축약)
    legend_labels = {
        'REGULAR_BULLISH': 'RSI 일반 상승',
        'REGULAR_BEARISH': 'RSI 일반 하락',
        'HIDDEN_BULLISH': 'RSI 히든 상승',
        'HIDDEN_BEARISH': 'RSI 히든 하락'
    }
    
    # 강도별 범례 라벨
    strength_labels = {
        '매우 강함': '매우 강함',
        '강함': '강함',
        '보통': '보통',
        '약함': '약함'
    }
    
    # 범례용 라인을 저장할 리스트
    legend_elements = []
    
    for div in divergences:
        try:
            # 시작점과 끝점 날짜 (실제 날짜 사용)
            start_date = div['start_date']
            end_date = div['end_date']
            
            # 가격선 연결 (강도별 스타일 적용)
            style = line_styles.get(div['strength'], line_styles['보통'])
            line, = ax_main.plot([start_date, end_date], [div['price1'], div['price2']], 
                                color=colors[div['type']], 
                                linestyle=style['linestyle'], 
                                linewidth=style['linewidth'], 
                                alpha=style['alpha'],
                                zorder=1001)
            
            # 범례용 요소 추가 (타입 + 강도별로 구분)
            legend_label = f"{legend_labels[div['type']]} ({div['strength']})"
            if not any(elem.get_label() == legend_label for elem in legend_elements):
                legend_elements.append(line)
                line.set_label(legend_label)
            
            # 신뢰도와 기간 정보를 라인 위에 직접 표시 (매우 작게)
            mid_date = start_date + (end_date - start_date) / 2
            mid_price = (div['price1'] + div['price2']) / 2
            
            # 신뢰도, 기간, 강도 정보 (라인 위에 하이라이트)
            info_text = f"{div['confidence']:.0%}|{div['pattern_days']}일|{div['strength']}"
            
            # 라인 위에 정보 표시 (매우 작게)
            y_offset = 1 if div['type'] in ['REGULAR_BULLISH', 'HIDDEN_BULLISH'] else -1
            
            ax_main.annotate(info_text, 
                             xy=(mid_date, mid_price + y_offset),
                             fontsize=6,
                             color=colors[div['type']],
                             fontweight='bold',
                             ha='center',
                             va='bottom' if div['type'] in ['REGULAR_BULLISH', 'HIDDEN_BULLISH'] else 'top',
                             bbox=dict(boxstyle='round,pad=0.1', 
                                      facecolor='white', 
                                      edgecolor=colors[div['type']],
                                      linewidth=0.5,
                                      alpha=0.8),
                             zorder=1002)
            
            # 시작점과 끝점에 작은 마커 표시
            ax_main.scatter([start_date, end_date], [div['price1'], div['price2']], 
                           color=colors[div['type']], s=15, alpha=0.8, zorder=1003,
                           edgecolors='black', linewidth=0.5)
            
        except Exception as e:
            continue
    
    return legend_elements

def create_rsi_divergence_chart(data, divergences, ticker, start_date, end_date, save_path=None):
    """
    RSI 다이버전스 패턴만을 위한 별도 차트를 생성합니다.
    
    Args:
        data: OHLCV 데이터
        divergences: 감지된 다이버전스들
        ticker: 종목 심볼
        start_date: 시작 날짜
        end_date: 종료 날짜
        save_path: 저장 경로
    """
    if not divergences:
        print("RSI 다이버전스 패턴이 없습니다.")
        return
    
    # 차트 생성 (메인 차트와 동일한 해상도)
    fig, (ax_main, ax_rsi) = plt.subplots(2, 1, figsize=(20, 14), 
                                          gridspec_kw={'height_ratios': [3, 1]})
    
    # 메인 차트 (캔들바 + 다이버전스)
    # 캔들바 데이터 준비
    ohlc_data = data[['Open', 'High', 'Low', 'Close']].copy()
    
    # 캔들바 플롯
    width = 0.6
    width2 = width * 0.8
    
    # 캔들바 몸통 (Open-Close)
    up = ohlc_data['Close'] > ohlc_data['Open']
    down = ohlc_data['Close'] < ohlc_data['Open']
    
    # 상승 캔들 (녹색)
    ax_main.bar(ohlc_data.index[up], ohlc_data['Close'][up] - ohlc_data['Open'][up], 
                width, bottom=ohlc_data['Open'][up], color='green', alpha=0.7, label='상승')
    
    # 하락 캔들 (빨간색)
    ax_main.bar(ohlc_data.index[down], ohlc_data['Open'][down] - ohlc_data['Close'][down], 
                width, bottom=ohlc_data['Close'][down], color='red', alpha=0.7, label='하락')
    
    # 고가-저가 선 (심지)
    for i in range(len(ohlc_data)):
        ax_main.plot([ohlc_data.index[i], ohlc_data.index[i]], 
                    [ohlc_data['Low'].iloc[i], ohlc_data['High'].iloc[i]], 
                    color='black', linewidth=1, alpha=0.8)
    
    # 종가 라인 (얇은 선으로 오버레이)
    ax_main.plot(data.index, data['Close'], color='black', linewidth=0.5, alpha=0.5, label='Close')
    
    # RSI 차트
    rsi = calculate_rsi(data['Close'], 14)
    ax_rsi.plot(data.index, rsi, color='blue', linewidth=1, label='RSI(14)')
    ax_rsi.axhline(y=70, color='red', linestyle='--', alpha=0.5)
    ax_rsi.axhline(y=30, color='green', linestyle='--', alpha=0.5)
    ax_rsi.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel('RSI')
    ax_rsi.grid(True, alpha=0.3)
    ax_rsi.legend()
    
    # 다이버전스 패턴 표시 (축약된 방식)
    colors = {
        'REGULAR_BULLISH': 'darkgreen',
        'REGULAR_BEARISH': 'darkred',
        'HIDDEN_BULLISH': 'green',
        'HIDDEN_BEARISH': 'red'
    }
    
    # 범례용 라벨 (축약)
    legend_labels = {
        'REGULAR_BULLISH': '일반 상승',
        'REGULAR_BEARISH': '일반 하락',
        'HIDDEN_BULLISH': '히든 상승',
        'HIDDEN_BEARISH': '히든 하락'
    }
    
    # 범례용 라인과 라벨을 저장할 리스트
    legend_elements = []
    
    for div in divergences:
        try:
            # 시작점과 끝점 날짜 (실제 날짜 사용)
            start_date = div['start_date']
            end_date = div['end_date']
            
            # 가격선 연결 (굵은 점선) - 실제 날짜 사용
            line, = ax_main.plot([start_date, end_date], [div['price1'], div['price2']], 
                                color=colors[div['type']], linestyle='--', linewidth=3, alpha=0.8,
                                zorder=1001)
            
            # 범례용 요소 추가 (타입 + 강도별로 구분)
            legend_label = f"{legend_labels[div['type']]} ({div['strength']})"
            if not any(elem.get_label() == legend_label for elem in legend_elements):
                legend_elements.append(line)
                line.set_label(legend_label)
            
            # 신뢰도와 기간 정보를 라인 위에 직접 표시
            mid_date = start_date + (end_date - start_date) / 2
            mid_price = (div['price1'] + div['price2']) / 2
            
            # 신뢰도와 기간 정보 (라인 위에 하이라이트)
            info_text = f"{div['confidence']:.0%} | {div['pattern_days']}일"
            
            # 라인 위에 정보 표시
            y_offset = 2 if div['type'] in ['REGULAR_BULLISH', 'HIDDEN_BULLISH'] else -2
            
            ax_main.annotate(info_text, 
                             xy=(mid_date, mid_price + y_offset),
                             fontsize=4,
                             color=colors[div['type']],
                             fontweight='bold',
                             ha='center',
                             va='bottom' if div['type'] in ['REGULAR_BULLISH', 'HIDDEN_BULLISH'] else 'top',
                             bbox=dict(boxstyle='round,pad=0.2', 
                                      facecolor='white', 
                                      edgecolor=colors[div['type']],
                                      linewidth=1.0,
                                      alpha=0.9),
                             zorder=1002)
            
            # 시작점과 끝점에 마커 표시
            ax_main.scatter([start_date, end_date], [div['price1'], div['price2']], 
                           color=colors[div['type']], s=30, alpha=0.9, zorder=1003,
                           edgecolors='black', linewidth=1)
            
        except Exception as e:
            continue
    
    # 차트 제목 및 설정
    ax_main.set_title(f'{ticker} - RSI 다이버전스 패턴 분석\n'
                      f'분석 기간: {start_date.strftime("%Y-%m-%d")} ~ {end_date.strftime("%Y-%m-%d")}\n'
                      f'감지된 패턴: {len(divergences)}개', fontsize=14, fontweight='bold')
    ax_main.set_ylabel('Price')
    ax_main.grid(True, alpha=0.3)
    
    # 범례 표시 (캔들바 + 다이버전스 타입별)
    # 캔들바 범례
    from matplotlib.patches import Patch
    candle_legend = [
        Patch(color='green', alpha=0.7, label='상승 캔들'),
        Patch(color='red', alpha=0.7, label='하락 캔들')
    ]
    
    # 다이버전스 범례
    if legend_elements:
        # 범례 위치 조정 (캔들바와 겹치지 않도록)
        ax_main.legend(handles=candle_legend + legend_elements, 
                      loc='upper left', fontsize=8, 
                      title='차트 범례', title_fontsize=9,
                      ncol=2)
    
    # x축 날짜 포맷 설정
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_main.tick_params(axis='x', rotation=45)
    ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_rsi.tick_params(axis='x', rotation=45)
    
    # 레이아웃 조정
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.06, hspace=0.15)
    
    # 저장 또는 표시
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"RSI 다이버전스 차트 저장 완료: {save_path}")
    else:
        plt.show()

def _find_pivots(series: pd.Series, left: int = 3, right: int = 3):
    """좌우 span을 기준으로 피벗 고점/저점을 탐지합니다.
    반환: (pivot_highs, pivot_lows) 각 리스트는 (index_pos, value) 튜플로 구성
    """
    values = series.values
    pivot_highs = []
    pivot_lows = []
    n = len(values)
    for i in range(left, n - right):
        window = values[i - left:i + right + 1]
        center = values[i]
        if center == np.max(window):
            pivot_highs.append((i, center))
        if center == np.min(window):
            pivot_lows.append((i, center))
    return pivot_highs, pivot_lows

def _detect_rsi_divergences(ohlcv_data: pd.DataFrame,
                            rsi_series: pd.Series,
                            pivot_span_left: int = 5,      # 3 → 5로 확장
                            pivot_span_right: int = 5,     # 3 → 5로 확장
                            valid_window_bars: int = 60,   # 20 → 60으로 확장
                            include_hidden: bool = True):
    """RSI 기반 다이버전스(일반/히든) 탐지.
    반환: 리스트[dict] with keys: kind('BULL'|'BEAR'), subtype('regular'|'hidden'),
          i1,i2 (index positions), date1,date2, price1,price2
    """
    close = ohlcv_data['Close']
    pivot_highs_price, pivot_lows_price = _find_pivots(close, pivot_span_left, pivot_span_right)
    pivot_highs_rsi, pivot_lows_rsi = _find_pivots(rsi_series, pivot_span_left, pivot_span_right)

    # index position -> pivot value lookup for RSI
    highs_rsi_map = {i: v for i, v in pivot_highs_rsi}
    lows_rsi_map = {i: v for i, v in pivot_lows_rsi}

    divergences = []
    last_idx = len(close) - 1

    # 정배(regular) 상승: 가격 LL, RSI HL (저점 기준)
    for (i1, p1), (i2, p2) in zip(pivot_lows_price[:-1], pivot_lows_price[1:]):
        if i2 <= i1:
            continue
        if last_idx - i2 > valid_window_bars:
            continue
        # 대응하는 RSI 저점이 존재하는지 확인 (근접 인덱스 사용)
        if i1 in lows_rsi_map and i2 in lows_rsi_map:
            r1 = lows_rsi_map[i1]
            r2 = lows_rsi_map[i2]
            if p2 < p1 and r2 > r1:  # 가격 LL, RSI HL
                divergences.append({
                    'kind': 'BULL', 'subtype': 'regular',
                    'i1': i1, 'i2': i2,
                    'date1': ohlcv_data.index[i1], 'date2': ohlcv_data.index[i2],
                    'price1': p1, 'price2': p2
                })

    # 정배(regular) 하락: 가격 HH, RSI LH (고점 기준)
    for (i1, p1), (i2, p2) in zip(pivot_highs_price[:-1], pivot_highs_price[1:]):
        if i2 <= i1:
            continue
        if last_idx - i2 > valid_window_bars:
            continue
        if i1 in highs_rsi_map and i2 in highs_rsi_map:
            r1 = highs_rsi_map[i1]
            r2 = highs_rsi_map[i2]
            if p2 > p1 and r2 < r1:  # 가격 HH, RSI LH
                # 다이버전스 강도 필터링 추가
                price_change_ratio = abs(p2 - p1) / p1
                rsi_change_ratio = abs(r2 - r1) / 100  # RSI는 0-100 범위
                
                # 최소 강도 기준: 가격 변화 3% 이상, RSI 변화 5% 이상
                if price_change_ratio >= 0.03 and rsi_change_ratio >= 0.05:
                    divergences.append({
                        'kind': 'BEAR', 'subtype': 'regular',
                        'i1': i1, 'i2': i2,
                        'date1': ohlcv_data.index[i1], 'date2': ohlcv_data.index[i2],
                        'price1': p1, 'price2': p2,
                        'strength': 'strong' if price_change_ratio >= 0.05 else 'moderate'
                    })

    if include_hidden:
        # 히든 상승: 가격 HL, RSI LL (저점 기준)
        for (i1, p1), (i2, p2) in zip(pivot_lows_price[:-1], pivot_lows_price[1:]):
            if i2 <= i1:
                continue
            if last_idx - i2 > valid_window_bars:
                continue
            if i1 in lows_rsi_map and i2 in lows_rsi_map:
                r1 = lows_rsi_map[i1]
                r2 = lows_rsi_map[i2]
                if p2 > p1 and r2 < r1:  # 가격 HL, RSI LL
                    divergences.append({
                        'kind': 'BULL', 'subtype': 'hidden',
                        'i1': i1, 'i2': i2,
                        'date1': ohlcv_data.index[i1], 'date2': ohlcv_data.index[i2],
                        'price1': p1, 'price2': p2
                    })

        # 히든 하락: 가격 LH, RSI HH (고점 기준)
        for (i1, p1), (i2, p2) in zip(pivot_highs_price[:-1], pivot_highs_price[1:]):
            if i2 <= i1:
                continue
            if last_idx - i2 > valid_window_bars:
                continue
            if i1 in highs_rsi_map and i2 in highs_rsi_map:
                r1 = highs_rsi_map[i1]
                r2 = highs_rsi_map[i2]
                if p2 < p1 and r2 > r1:  # 가격 LH, RSI HH
                    divergences.append({
                        'kind': 'BEAR', 'subtype': 'hidden',
                        'i1': i1, 'i2': i2,
                        'date1': ohlcv_data.index[i1], 'date2': ohlcv_data.index[i2],
                        'price1': p1, 'price2': p2
                    })

    return divergences

def _get_market_sentiment_summary(vix_value, naaim_value, pcr_value):
    """VIX, NAIIM, PCR을 종합하여 시장 심리 요약을 생성합니다."""
    if vix_value is None and naaim_value is None and pcr_value is None:
        return "시장 데이터 부족"
    
    summary_parts = []
    
    # VIX 해석
    if vix_value is not None:
        if vix_value < 20:
            vix_sentiment = "VIX:[G](안정)"
        elif vix_value < 30:
            vix_sentiment = "VIX:[O](중간)"
        else:
            vix_sentiment = "VIX:[R](불안)"
        summary_parts.append(vix_sentiment)
    
    # NAIIM 해석
    if naaim_value is not None:
        if naaim_value < 30:
            naaim_sentiment = "NAIIM:[R](보수)"
        elif naaim_value < 50:
            naaim_sentiment = "NAIIM:[O](중간)"
        else:
            naaim_sentiment = "NAIIM:[B](적극)"
        summary_parts.append(naaim_sentiment)
    
    # PCR 해석 (5단계 세밀 구분으로 통일)
    if pcr_value is not None:
        if pcr_value > 1.5:
            pcr_sentiment = "PCR:[G](과매도)"
        elif pcr_value > 1.0:
            pcr_sentiment = "PCR:[O](비관)"
        elif pcr_value > 0.7:
            pcr_sentiment = "PCR:[O](균형)"
        elif pcr_value > 0.4:
            pcr_sentiment = "PCR:[G](낙관)"
        else:
            pcr_sentiment = "PCR:[R](과열)"
        summary_parts.append(pcr_sentiment)
    
    # 종합 판단
    if len(summary_parts) >= 2:
        bullish_count = sum(1 for part in summary_parts if '[G]' in part or '[B]' in part)
        bearish_count = sum(1 for part in summary_parts if '[R]' in part)
        
        if bullish_count > bearish_count:
            overall_sentiment = " -> [UP] 매수우세"
        elif bearish_count > bullish_count:
            overall_sentiment = " -> [DOWN] 매도우세"
        else:
            overall_sentiment = " -> [EQ] 중립"
        
        summary_parts.append(overall_sentiment)
    
    return " | ".join(summary_parts)

def _calculate_investor_sentiment(data, current_index):
    """현재가격 기준으로 투자심리도를 계산합니다 (당일 이전 거래일 10일 대상)"""
    if current_index < 10:
        return None
    
    # 현재일 이전 10일간의 데이터
    start_idx = current_index - 10
    end_idx = current_index
    
    up_days = 0
    for i in range(start_idx + 1, end_idx + 1):
        # 당일 종가가 전일 종가보다 높은 경우
        if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
            up_days += 1
    
    # 10일 중 상승일 비율
    sentiment = (up_days / 10) * 100
    return sentiment

def _get_rsi_sentiment_strategy(rsi_value, sentiment_value):
    """RSI와 투자심리도를 종합하여 투자액션과 전략을 생성합니다."""
    if rsi_value is None or sentiment_value is None:
        return None, None
    
    # RSI 구간별 분류
    if rsi_value < 30:
        rsi_category = "과매도"
    elif rsi_value < 50:
        rsi_category = "약세"
    elif rsi_value < 70:
        rsi_category = "중립"
    else:
        rsi_category = "과매수"
    
    # 투자심리도 구간별 분류
    if sentiment_value < 40:
        sentiment_category = "약세"
    elif sentiment_value < 60:
        sentiment_category = "중립"
    else:
        sentiment_category = "강세"
    
    # 투자액션 결정
    if rsi_value < 30 and sentiment_value < 40:
        action = "강력매수"
        strategy = "과매도 + 약세심리 = 반등 기대"
    elif rsi_value < 30 and sentiment_value >= 40:
        action = "매수"
        strategy = "과매도 + 중립/강세심리 = 반등 신호"
    elif rsi_value < 50 and sentiment_value < 40:
        action = "관망"
        strategy = "약세 + 약세심리 = 하락 지속"
    elif rsi_value < 50 and sentiment_value >= 60:
        action = "부분매수"
        strategy = "약세 + 강세심리 = 반전 기대"
    elif rsi_value < 70 and sentiment_value < 40:
        action = "관망"
        strategy = "중립 + 약세심리 = 방향성 불분명"
    elif rsi_value < 70 and sentiment_value >= 60:
        action = "보유/추가매수"
        strategy = "중립 + 강세심리 = 상승 지속"
    elif rsi_value >= 70 and sentiment_value < 40:
        action = "부분매도"
        strategy = "과매수 + 약세심리 = 조정 기대"
    elif rsi_value >= 70 and sentiment_value >= 60:
        action = "매도"
        strategy = "과매수 + 강세심리 = 고점 신호"
    else:
        action = "관망"
        strategy = "중립 + 중립심리 = 방향성 불분명"
    
    return action, strategy

def _get_strategy_guide(vix_value, naaim_value, pcr_value):
    """VIX, NAIIM, PCR 구간별 전략 가이드를 생성합니다."""
    strategy_parts = []
    
    # VIX 전략
    if vix_value is not None:
        if vix_value < 20:
            vix_strategy = "VIX<20: 매수기회(안정)"
        elif vix_value < 30:
            vix_strategy = "VIX 20-30: 관망(중간)"
        else:
            vix_strategy = "VIX>30: 방어(불안)"
        strategy_parts.append(vix_strategy)
    
    # NAIIM 전략
    if naaim_value is not None:
        if naaim_value < 30:
            naaim_strategy = "NAIIM<30: 매수(보수)"
        elif naaim_value < 50:
            naaim_strategy = "NAIIM 30-50: 중립(균형)"
        else:
            naaim_strategy = "NAIIM>50: 주의(적극)"
        strategy_parts.append(naaim_strategy)
    
    # PCR 전략 (5단계 세밀 구분)
    if pcr_value is not None:
        if pcr_value > 1.5:
            pcr_strategy = "PCR>1.5: 매수(과매도)"
        elif pcr_value > 1.0:
            pcr_strategy = "PCR 1.0-1.5: 신중(비관)"
        elif pcr_value > 0.7:
            pcr_strategy = "PCR 0.7-1.0: 중립(균형)"
        elif pcr_value > 0.4:
            pcr_strategy = "PCR 0.4-0.7: 낙관(상승)"
        else:
            pcr_strategy = "PCR<0.4: 매도(과열)"
        strategy_parts.append(pcr_strategy)
    
    return " | ".join(strategy_parts)

def _detect_macd_divergences(ohlcv_data: pd.DataFrame,
                             macd_hist: pd.Series,
                             pivot_span_left: int = 5,      # 3 → 5로 확장
                             pivot_span_right: int = 5,     # 3 → 5로 확장
                             valid_window_bars: int = 60,   # 20 → 60으로 확장
                             include_hidden: bool = True):
    """MACD 히스토그램 기반 다이버전스(일반/히든) 탐지.
    반환 구조는 _detect_rsi_divergences와 동일.
    """
    close = ohlcv_data['Close']
    pivot_highs_price, pivot_lows_price = _find_pivots(close, pivot_span_left, pivot_span_right)
    pivot_highs_hist, pivot_lows_hist = _find_pivots(macd_hist, pivot_span_left, pivot_span_right)

    highs_hist_map = {i: v for i, v in pivot_highs_hist}
    lows_hist_map = {i: v for i, v in pivot_lows_hist}

    divergences = []
    last_idx = len(close) - 1

    # 정배(regular) 상승: 가격 LL, 히스토그램 HL (저점 기준)
    for (i1, p1), (i2, p2) in zip(pivot_lows_price[:-1], pivot_lows_price[1:]):
        if i2 <= i1 or last_idx - i2 > valid_window_bars:
            continue
        if i1 in lows_hist_map and i2 in lows_hist_map:
            h1 = lows_hist_map[i1]
            h2 = lows_hist_map[i2]
            if p2 < p1 and h2 > h1:
                divergences.append({
                    'kind': 'BULL', 'subtype': 'regular',
                    'i1': i1, 'i2': i2,
                    'date1': ohlcv_data.index[i1], 'date2': ohlcv_data.index[i2],
                    'price1': p1, 'price2': p2
                })

    # 정배(regular) 하락: 가격 HH, 히스토그램 LH (고점 기준)
    for (i1, p1), (i2, p2) in zip(pivot_highs_price[:-1], pivot_highs_price[1:]):
        if i2 <= i1 or last_idx - i2 > valid_window_bars:
            continue
        if i1 in highs_hist_map and i2 in highs_hist_map:
            h1 = highs_hist_map[i1]
            h2 = highs_hist_map[i2]
            if p2 > p1 and h2 < h1:
                # 다이버전스 강도 필터링 추가
                price_change_ratio = abs(p2 - p1) / p1
                macd_change_ratio = abs(h2 - h1) / abs(h1) if h1 != 0 else 0
                
                # 최소 강도 기준: 가격 변화 3% 이상, MACD 변화 10% 이상
                if price_change_ratio >= 0.03 and macd_change_ratio >= 0.10:
                    divergences.append({
                        'kind': 'BEAR', 'subtype': 'regular',
                        'i1': i1, 'i2': i2,
                        'date1': ohlcv_data.index[i1], 'date2': ohlcv_data.index[i2],
                        'price1': p1, 'price2': p2,
                        'strength': 'strong' if price_change_ratio >= 0.05 else 'moderate'
                    })

    if include_hidden:
        # 히든 상승: 가격 HL, 히스토그램 LL (저점 기준)
        for (i1, p1), (i2, p2) in zip(pivot_lows_price[:-1], pivot_lows_price[1:]):
            if i2 <= i1 or last_idx - i2 > valid_window_bars:
                continue
            if i1 in lows_hist_map and i2 in lows_hist_map:
                h1 = lows_hist_map[i1]
                h2 = lows_hist_map[i2]
                if p2 > p1 and h2 < h1:
                    divergences.append({
                        'kind': 'BULL', 'subtype': 'hidden',
                        'i1': i1, 'i2': i2,
                        'date1': ohlcv_data.index[i1], 'date2': ohlcv_data.index[i2],
                        'price1': p1, 'price2': p2
                    })

        # 히든 하락: 가격 LH, 히스토그램 HH (고점 기준)
        for (i1, p1), (i2, p2) in zip(pivot_highs_price[:-1], pivot_highs_price[1:]):
            if i2 <= i1 or last_idx - i2 > valid_window_bars:
                continue
            if i1 in highs_hist_map and i2 in highs_hist_map:
                h1 = highs_hist_map[i1]
                h2 = highs_hist_map[i2]
                if p2 < p1 and h2 > h1:
                    divergences.append({
                        'kind': 'BEAR', 'subtype': 'hidden',
                        'i1': i1, 'i2': i2,
                        'date1': ohlcv_data.index[i1], 'date2': ohlcv_data.index[i2],
                        'price1': p1, 'price2': p2
                    })

    return divergences

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

def analyze_poc_price_range(current_price, c_poc_price, t_poc_price):
    """POC와 현재 가격의 범위를 분석합니다."""
    
    def get_price_range(price1, price2):
        """두 가격 간의 범위를 계산합니다."""
        if price1 == 0 or price2 == 0:
            return "N/A"
        
        diff_percent = abs(price1 - price2) / price2 * 100
        
        if diff_percent <= 5:
            return "±5%"
        elif diff_percent <= 10:
            return "±10%"
        elif diff_percent <= 20:
            return "±20%"
        elif diff_percent <= 30:
            return "±30%"
        else:
            return "±30%+"
    
    def get_poc_strength(price1, price2):
        """POC 강도를 계산합니다."""
        if price1 == 0 or price2 == 0:
            return "N/A"
        
        diff_percent = abs(price1 - price2) / price2 * 100
        
        if diff_percent <= 5:
            return "매우 강함"  # 현재가가 POC에 매우 가까움
        elif diff_percent <= 10:
            return "강함"        # 현재가가 POC에 가까움
        elif diff_percent <= 20:
            return "보통"        # 현재가가 POC와 적당한 거리
        elif diff_percent <= 30:
            return "약함"        # 현재가가 POC와 멀음
        else:
            return "매우 약함"   # 현재가가 POC와 매우 멀음
    
    # C-POC 범위 분석
    c_poc_range = get_price_range(current_price, c_poc_price)
    c_poc_strength = get_poc_strength(current_price, c_poc_price)
    
    # T-POC 범위 분석
    t_poc_range = get_price_range(current_price, t_poc_price)
    t_poc_strength = get_poc_strength(current_price, t_poc_price)
    
    return {
        'c_poc_range': c_poc_range,
        'c_poc_strength': c_poc_strength,
        't_poc_range': t_poc_range,
        't_poc_strength': t_poc_strength
    }

def calculate_buffett_indicator():
    """버핏 지수를 계산합니다. (시뮬레이션 데이터)"""
    try:
        # 실제 구현에서는 yfinance나 다른 API를 사용하여 Wilshire 5000과 GDP 데이터를 가져와야 함
        # 현재는 시뮬레이션 데이터 사용
        
        # Wilshire 5000 시가총액 (시뮬레이션)
        wilshire_market_cap = 45.2  # 조 달러
        
        # US GDP (시뮬레이션)
        us_gdp = 27.4  # 조 달러
        
        # 버핏 지수 계산
        buffett_indicator = (wilshire_market_cap / us_gdp) * 100
        
        return round(buffett_indicator, 1)
    except Exception as e:
        print(f"버핏 지수 계산 오류: {e}")
        return None

def get_buffett_sentiment(buffett_value):
    """버핏 지수 해석 및 투자 가이드를 제공합니다."""
    if buffett_value is None:
        return "N/A", "gray", "데이터 부족"
    
    if buffett_value <= 50:
        sentiment = "극도 과매도"
        color = "darkgreen"
        guide = "강력한 매수 기회 - 주식 비중 80%+ 고려"
    elif buffett_value <= 75:
        sentiment = "과매도"
        color = "green"
        guide = "매수 기회 - 주식 비중 70-80% 고려"
    elif buffett_value <= 90:
        sentiment = "균형"
        color = "orange"
        guide = "균형적 배분 - 주식 비중 50-60% 유지"
    elif buffett_value <= 115:
        sentiment = "과열"
        color = "red"
        guide = "주의 필요 - 주식 비중 30-40% 고려"
    else:
        sentiment = "극도 과열"
        color = "darkred"
        guide = "강력한 매도 신호 - 주식 비중 20% 이하 고려"
    
    return sentiment, color, guide

def calculate_volume_profile(ohlcv_data, num_bins=50):
    """Fixed Range Volume Profile 계산 (Net Volume 포함)"""
    # 가격 범위 설정
    price_min = ohlcv_data['Low'].min()
    price_max = ohlcv_data['High'].max()
    
    # 가격을 N개 구간으로 나누기
    price_bins = np.linspace(price_min, price_max, num_bins + 1)
    
    # 거래량 분포 계산
    volume_profile = []
    net_volume_profile = []  # Net Volume 추가
    volume_ratios = []  # 비율 추가
    
    for i in range(len(price_bins) - 1):
        bin_low = price_bins[i]
        bin_high = price_bins[i + 1]
        
        # 해당 가격 구간에 속하는 거래량 합계
        mask = (ohlcv_data['Low'] <= bin_high) & (ohlcv_data['High'] >= bin_low)
        total_volume = ohlcv_data.loc[mask, 'Volume'].sum()
        
        # Net Volume 계산 (상승일과 하락일 구분)
        up_mask = mask & (ohlcv_data['Close'] > ohlcv_data['Open'])
        down_mask = mask & (ohlcv_data['Close'] < ohlcv_data['Open'])
        
        up_volume = ohlcv_data.loc[up_mask, 'Volume'].sum()
        down_volume = ohlcv_data.loc[down_mask, 'Volume'].sum()
        net_volume = up_volume - down_volume
        
        volume_profile.append(total_volume)
        net_volume_profile.append(net_volume)
    
    # 전체 거래량 대비 비율 계산
    total_volume = sum(volume_profile)
    volume_ratios = [vol / total_volume * 100 for vol in volume_profile]
    
    # POC (Point of Control) 계산
    poc_idx = np.argmax(volume_profile)
    poc_price = price_bins[poc_idx]
    
    # Value Area 계산 (거래량 70% 구간)
    target_volume = total_volume * 0.7
    
    # POC를 중심으로 Value Area 확장
    value_area_prices = [poc_price]
    current_volume = volume_profile[poc_idx]
    
    left_idx = poc_idx - 1
    right_idx = poc_idx + 1
    
    while current_volume < target_volume and (left_idx >= 0 or right_idx < len(volume_profile)):
        left_vol = volume_profile[left_idx] if left_idx >= 0 else 0
        right_vol = volume_profile[right_idx] if right_idx < len(volume_profile) else 0
        
        if left_vol >= right_vol and left_idx >= 0:
            value_area_prices.append(price_bins[left_idx])
            current_volume += left_vol
            left_idx -= 1
        elif right_idx < len(volume_profile):
            value_area_prices.append(price_bins[right_idx])
            current_volume += right_vol
            right_idx += 1
        else:
            break
    
    value_area_min = min(value_area_prices)
    value_area_max = max(value_area_prices)
    
    return price_bins, volume_profile, net_volume_profile, volume_ratios, poc_price, value_area_min, value_area_max

 

def get_stock_info(symbol):
    """종목의 전체 이름과 GICS 섹터/서브분류 정보를 가져옵니다."""
    try:
        # yfinance를 사용하여 종목 정보 가져오기
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # 종목 이름 (긴 이름)
        long_name = info.get('longName', symbol)
        if not long_name or long_name == symbol:
            long_name = info.get('shortName', symbol)
        
        # GICS 섹터 및 서브분류
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        
        return long_name, sector, industry
    except Exception as e:
        print(f"종목 정보 가져오기 실패: {e}")
        return symbol, 'N/A', 'N/A'

def get_local_datetime(utc_datetime):
    """UTC 시간을 한국 시간(KST)으로 변환합니다."""
    try:
        import pytz
        from datetime import datetime
        
        # UTC 시간대 설정
        utc_tz = pytz.UTC
        # 한국 시간대 설정
        kst_tz = pytz.timezone('Asia/Seoul')
        
        # UTC 시간을 한국 시간으로 변환
        if utc_datetime.tzinfo is None:
            # timezone 정보가 없는 경우 UTC로 가정
            utc_datetime = utc_tz.localize(utc_datetime)
        
        local_datetime = utc_datetime.astimezone(kst_tz)
        return local_datetime
    except ImportError:
        # pytz가 없는 경우 기본 시간 사용
        return utc_datetime
    except Exception as e:
        print(f"시간대 변환 실패: {e}")
        return utc_datetime

def get_current_stock_price(symbol):
    """현재 주가를 실시간으로 가져옵니다."""
    try:
        import yfinance as yf
        from datetime import datetime
        import pytz
        
        # yfinance로 현재 주가 정보 가져오기
        ticker = yf.Ticker(symbol)
        current_info = ticker.history(period='1d', interval='1m')
        
        if not current_info.empty:
            # 가장 최근 데이터
            latest_data = current_info.iloc[-1]
            current_price = latest_data['Close']
            current_time = latest_data.name
            
            # 현재 시간과 비교하여 데이터 신선도 확인 (시간대 일치)
            kst_tz = pytz.timezone('Asia/Seoul')
            now = datetime.now(kst_tz)
            
            # current_time이 timezone-aware인지 확인
            if current_time.tzinfo is None:
                # timezone 정보가 없으면 UTC로 가정
                current_time = pytz.UTC.localize(current_time)
            
            # 한국 시간으로 변환
            current_time_kst = current_time.astimezone(kst_tz)
            
            # 시간 차이 계산
            time_diff = (now - current_time_kst).total_seconds() / 60  # 분 단위
            
            if time_diff <= 15:  # 15분 이내 데이터
                return current_price, current_time, 'realtime'
            elif time_diff <= 60:  # 1시간 이내 데이터
                return current_price, current_time, 'recent'
            else:  # 오래된 데이터
                return current_price, current_time, 'stale'
        else:
            return None, None, 'no_data'
            
    except Exception as e:
        print(f"현재 주가 가져오기 실패: {e}")
        return None, None, 'error'

def calculate_daily_change_percentage(symbol):
    """전일대비 등락률을 계산합니다."""
    try:
        import yfinance as yf
        from datetime import datetime, timedelta
        import pytz
        
        # yfinance로 전일 종가와 현재가 가져오기
        ticker = yf.Ticker(symbol)
        
        # 전일 종가 가져오기 (2일 데이터로 전일 확인)
        hist_data = ticker.history(period='2d')
        
        if len(hist_data) >= 2:
            # 전일 종가
            prev_close = hist_data.iloc[-2]['Close']
            # 현재가 (가장 최근)
            current_price = hist_data.iloc[-1]['Close']
            
            # 등락률 계산
            change_amount = current_price - prev_close
            change_percentage = (change_amount / prev_close) * 100
            
            # 등락 방향과 색상 결정
            if change_amount > 0:
                direction = "▲"
                color = "green"
            elif change_amount < 0:
                direction = "▼"
                color = "red"
            else:
                direction = "─"
                color = "gray"
            
            return {
                'prev_close': prev_close,
                'current_price': current_price,
                'change_amount': change_amount,
                'change_percentage': change_percentage,
                'direction': direction,
                'color': color
            }
        else:
            return None
            
    except Exception as e:
        print(f"전일대비 등락률 계산 실패: {e}")
        return None

def plot_main_chart_with_volume_profile_overlay(
    data: pd.DataFrame,
    ticker: str = None,
    save_path: str = None,
    current_price: float = None,
    rsi_divergence_window: int = 60,  # 20 → 60으로 확장 (약 3개월)
    rsi_pivot_span: int = 5,          # 3 → 5로 확장 (노이즈 감소)
    include_hidden_divergence: bool = True,
):
    """Volume Profile이 메인차트에 오버레이된 차트"""
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
    
    # NAIIM 데이터 가져오기 (실제 데이터 우선, 없으면 시뮬레이션)
    try:
        from pathlib import Path
        
        naaim_file = Path("data_cache/naaim_data.csv")
        if naaim_file.exists():
            naaim_df = pd.read_csv(naaim_file)
            naaim_df['Date'] = pd.to_datetime(naaim_df['Date'])
            naaim_df.set_index('Date', inplace=True)
            naaim = naaim_df.loc[start_date:end_date]
            print(f"실제 NAIIM 데이터 로드 완료: {len(naaim)}개 데이터")
        else:
            naaim = None
            print("실제 NAIIM 데이터 파일이 없습니다.")
    except Exception as e:
        print(f"NAIIM 데이터 로드 오류: {e}")
        naaim = None
    
    # PCR 데이터 계산 (옵션이 있는 종목만)
    pcr = None
    if ticker and not ticker.endswith('.KS'):  # 한국 종목 제외
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            options = stock.options
            if options:
                nearest_expiry = options[0]
                calls = stock.option_chain(nearest_expiry).calls
                puts = stock.option_chain(nearest_expiry).puts
                
                total_call_volume = calls['volume'].sum()
                total_put_volume = puts['volume'].sum()
                
                if total_call_volume > 0:
                    pcr = total_put_volume / total_call_volume
                    print(f"{ticker} PCR 계산 완료: {pcr:.3f}")
        except Exception as e:
            print(f"{ticker} PCR 계산 오류: {e}")

    # 기술적 지표 계산
    hma = calculate_hma(ohlcv_data['Close'])
    upper_band, lower_band = calculate_mantra_bands(ohlcv_data['Close'])
    rsi3 = calculate_rsi(ohlcv_data['Close'], period=3)
    rsi14 = calculate_rsi(ohlcv_data['Close'], period=14)
    rsi50 = calculate_rsi(ohlcv_data['Close'], period=50)
    macd, macd_signal, hist = calculate_macd(ohlcv_data['Close'])
    
    # 볼린저 밴드 계산
    volume_ma = ohlcv_data['Volume'].rolling(window=20).mean()
    volume_std = ohlcv_data['Volume'].rolling(window=20).std()
    volume_upper = volume_ma + (volume_std * 2)
    
        # 볼린저 밴드 계산
    bb_ma, bb_upper, bb_lower = calculate_bollinger_bands(ohlcv_data['Close'])
    
    # SMA200일 이동평균선 계산
    sma200 = ohlcv_data['Close'].rolling(window=200).mean()
    
    # 신호 생성
    trade_signals = get_hma_mantra_md_signals(ohlcv_data, ticker)

    # Volume Profile 계산
    price_bins, volume_profile, net_volume_profile, volume_ratios, poc_price, value_area_min, value_area_max = calculate_volume_profile(ohlcv_data)
    
    # 분석 기간이 1년 이상인 경우 최근 6개월 Volume Profile 추가 계산
    analysis_period_days = (end_date - start_date).days
    recent_6mo_data = None
    recent_price_bins = None
    recent_volume_profile = None
    recent_net_volume_profile = None
    recent_volume_ratios = None
    recent_poc_price = None
    recent_value_area_min = None
    recent_value_area_max = None
    
    if analysis_period_days >= 365:  # 1년 이상
        # 최근 6개월 데이터 추출
        recent_6mo_start = end_date - pd.Timedelta(days=180)
        recent_6mo_data = ohlcv_data[recent_6mo_start:end_date]
        
        # 최근 6개월 데이터가 충분한지 확인 (최소 30일 이상)
        if len(recent_6mo_data) >= 30:
            # 최근 6개월 Volume Profile 계산
            recent_price_bins, recent_volume_profile, recent_net_volume_profile, \
            recent_volume_ratios, recent_poc_price, recent_value_area_min, recent_value_area_max = \
                calculate_volume_profile(recent_6mo_data)

    # 시장 심리 종합해석 생성
    vix_final = None
    naaim_final = None
    
    if not vix.empty:
        vix_end_date = vix.index[vix.index <= end_date][-1] if len(vix.index[vix.index <= end_date]) > 0 else vix.index[-1]
        vix_final = float(vix.loc[vix_end_date].iloc[0])
        print(f"VIX 최종값: {vix_final}")
    
    if naaim is not None and not naaim.empty:
        naaim_end_date = naaim.index[naaim.index <= end_date][-1] if len(naaim.index[naaim.index <= end_date]) > 0 else naaim.index[-1]
        naaim_final = float(naaim.loc[naaim_end_date].iloc[0])
        print(f"NAIIM 최종값: {naaim_final}")
    
    print(f"PCR 값: {pcr}")
    market_summary = _get_market_sentiment_summary(vix_final, naaim_final, pcr)
    strategy_guide = _get_strategy_guide(vix_final, naaim_final, pcr)
    
    print(f"시장 심리 요약: {market_summary}")
    print(f"전략 가이드: {strategy_guide}")
    
    # 차트 생성 (4x1 레이아웃: 메인차트 + 거래량 + RSI + MACD)
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 1, height_ratios=[3, 1, 1, 1], figure=fig, hspace=0.15)  # 간격 증가
    
    # 메인 차트 (상단)
    ax_main = fig.add_subplot(gs[0, 0])
    
    # 종목 정보 가져오기
    long_name, sector, industry = get_stock_info(ticker)
    
    # 차트 타이틀에 시장 심리 종합해석 추가
    title_text = f"{ticker} HMA Mantra Analysis"
    if market_summary and market_summary != "시장 데이터 부족":
        title_text += f" | {market_summary}"
    
    print(f"설정할 타이틀: {title_text}")
    
    # 메인 차트 타이틀에 종목 정보 추가
    main_title = f"{ticker} - {long_name}\n"
    main_title += f"섹터: {sector} | 산업: {industry}"
    ax_main.set_title(main_title, fontsize=12, fontweight='bold', pad=15)
    
    # 전체 차트 상단에 시장 심리 종합해석과 전략 가이드를 별도로 표시
    if market_summary and market_summary != "시장 데이터 부족":
        # 기본 타이틀 제거
        fig.suptitle("", fontsize=1, y=0.99)
        
            # 시장 심리 요약을 상단 중앙에 박스 형태로 표시 (zorder 최고값 설정)
    ax_summary = fig.add_axes([0.1, 0.95, 0.8, 0.03])
    ax_summary.set_facecolor('lightblue')
    ax_summary.set_xlim(0, 1)
    ax_summary.set_ylim(0, 1)
    ax_summary.axis('off')
    ax_summary.set_zorder(1000)  # 최고 zorder 값 설정
    
    # 시장 심리 요약 텍스트를 박스 안에 표시
    ax_summary.text(0.5, 0.5, market_summary, 
                   fontsize=12, fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', 
                            edgecolor='navy', linewidth=2),
                   zorder=1001)  # 텍스트도 최고 zorder 값 설정
    
    # 버핏 지수 계산 (표시는 통합 박스에서 처리)
    buffett_value = calculate_buffett_indicator()
    if buffett_value:
        buffett_sentiment, buffett_color, buffett_guide = get_buffett_sentiment(buffett_value)
        
        # 전략 가이드를 시장 심리 요약 아래에 표시 (zorder 최고값 설정)
        if strategy_guide:
            ax_strategy = fig.add_axes([0.1, 0.91, 0.8, 0.03])
            ax_strategy.set_facecolor('lightyellow')
            ax_strategy.set_xlim(0, 1)
            ax_strategy.set_ylim(0, 1)
            ax_strategy.axis('off')
            ax_strategy.set_zorder(1000)  # 최고 zorder 값 설정
            
            # 전략 가이드 텍스트를 박스 안에 표시 (폰트 크기 30% 감소)
            ax_strategy.text(0.5, 0.5, strategy_guide, 
                           fontsize=7.7, fontweight='bold', ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.4", facecolor='lightyellow', 
                                    edgecolor='orange', linewidth=2),
                           zorder=1001)  # 텍스트도 최고 zorder 값 설정
    
    # 거래량 차트
    ax_volume = fig.add_subplot(gs[1, 0], sharex=ax_main)
    # RSI 차트
    ax_rsi = fig.add_subplot(gs[2, 0], sharex=ax_main)
    # MACD 차트
    ax_macd = fig.add_subplot(gs[3, 0], sharex=ax_main)

    # 메인 차트 설정 (투명도 높임)
    candlestick_ohlc(ax_main, 
                     [[mdates.date2num(date), o, h, l, c] for date, (o, h, l, c) in 
                      zip(ohlcv_data.index, ohlcv_data[['Open', 'High', 'Low', 'Close']].values)],
                     width=0.6, colorup='green', colordown='red', alpha=0.9)
    
    # 시장 지표들을 수평으로 우측 상단에 표시
    indicators = []
    
    # VIX 값 추가
    if not vix.empty:
        end_date = ohlcv_data.index[-1]
        vix_end_date = vix.index[vix.index <= end_date][-1] if len(vix.index[vix.index <= end_date]) > 0 else vix.index[-1]
        vix_value = float(vix.loc[vix_end_date].iloc[0])
        
        if vix_value < 20:
            vix_color = 'green'
        elif vix_value < 30:
            vix_color = 'orange'
        else:
            vix_color = 'red'
        
        indicators.append({
            'text': f'VIX: {vix_value:.2f}',
            'color': vix_color,
            'date': vix_end_date.strftime("%Y-%m-%d")
        })
    
    # NAIIM 값 추가
    if naaim is not None and not naaim.empty:
        try:
            naaim_end_date = naaim.index[naaim.index <= end_date][-1] if len(naaim.index[naaim.index <= end_date]) > 0 else naaim.index[-1]
            naaim_value = float(naaim.loc[naaim_end_date].iloc[0])
            
            if naaim_value < 30:
                naaim_color = 'red'
            elif naaim_value < 50:
                naaim_color = 'orange'
            else:
                naaim_color = 'blue'
            
            indicators.append({
                'text': f'NAIIM: {naaim_value:.1f}',
                'color': naaim_color,
                'date': naaim_end_date.strftime("%Y-%m-%d")
            })
        except Exception as e:
            print(f"NAIIM 표시 오류: {e}")
    
    # PCR 값 추가
    if pcr is not None:
        try:
            if pcr < 0.7:
                pcr_color = 'green'
            elif pcr < 1.0:
                pcr_color = 'orange'
            else:
                pcr_color = 'red'
            
            indicators.append({
                'text': f'PCR: {pcr:.3f}',
                'color': pcr_color,
                'date': None
            })
        except Exception as e:
            print(f"PCR 표시 오류: {e}")
    
    # 세로로 지표들 표시 (전략 가이드와 겹치지 않도록 위치 조정)
    if indicators:
        # 지표들을 세로로 배치 (폰트 크기 6, 비율에 맞는 수직 간격)
        y_positions = [0.97, 0.94, 0.91]  # VIX, NAIIM, PCR 순서로 비율에 맞게 조정
        
        for i, indicator in enumerate(indicators):
            x_pos = 0.99  # 우측 끝에 고정
            y_pos = y_positions[i] if i < len(y_positions) else 0.91  # 비율에 맞는 y 위치 사용
            
            # NAIIM 사이즈와 동일한 width로 박스 크기 통일 (pad를 2로 조정)
            ax_main.text(x_pos, y_pos, indicator['text'], 
                        transform=ax_main.transAxes, fontsize=6, ha='right', va='top',
                        bbox=dict(facecolor=indicator['color'], alpha=0.8, edgecolor='black', pad=2, 
                                 boxstyle='round,pad=0.3'),
                        color='white', fontweight='bold',
                        zorder=1000)  # 최고 zorder 값으로 레이어 최상단에 표시

    # HMA와 만트라 밴드
    ax_main.plot(ohlcv_data.index, hma, color='blue', linewidth=1.5, label='HMA')
    ax_main.plot(ohlcv_data.index, upper_band, color='red', linewidth=1, linestyle='--', label='Upper Mantra')
    ax_main.plot(ohlcv_data.index, lower_band, color='green', linewidth=1, linestyle='--', label='Lower Mantra')
    
    # 가격 라인 추가
    ax_main.plot(ohlcv_data.index, ohlcv_data['Close'], color='black', linewidth=0.8, alpha=0.7, label='Price')
    
    # 만트라 밴드 영역 채우기
    ax_main.fill_between(ohlcv_data.index, hma, upper_band, color='red', alpha=0.1, label='상단 밴드 영역')
    ax_main.fill_between(ohlcv_data.index, lower_band, hma, color='green', alpha=0.1, label='하단 밴드 영역')
    
    # 볼린저 밴드 추가
    ax_main.plot(ohlcv_data.index, bb_ma, color='purple', linewidth=0.5, label='BB MA(20)')
    ax_main.plot(ohlcv_data.index, bb_upper, color='purple', linewidth=0.5, linestyle=':', label='BB Upper')
    ax_main.plot(ohlcv_data.index, bb_lower, color='purple', linewidth=0.5, linestyle=':', label='BB Lower')
    
    # SMA200일 이동평균선 추가
    ax_main.plot(ohlcv_data.index, sma200, color='darkblue', linewidth=1.5, linestyle='-', label='SMA200', alpha=0.8)

        # 현재 주가를 실시간으로 가져와서 연동
    realtime_price, realtime_time, data_freshness = get_current_stock_price(ticker)
    
    if realtime_price is not None:
        print(f"실시간 주가: ${realtime_price:.2f} ({data_freshness})")
        # 실시간 주가가 있으면 이를 사용, 없으면 기존 데이터 사용
        current_price = realtime_price
        price_source = "실시간"
    else:
        print(f"실시간 주가 가져오기 실패, 기존 데이터 사용")
        current_price = ohlcv_data['Close'].iloc[-1]
        price_source = "기존데이터"
    
    # 전일대비 등락률 계산
    daily_change = calculate_daily_change_percentage(ticker)
    if daily_change:
        print(f"전일대비: {daily_change['direction']} {daily_change['change_percentage']:.2f}% (${daily_change['change_amount']:.2f})")
    else:
        print("전일대비 등락률 계산 실패")
    
    support, resistance = calculate_support_resistance(ohlcv_data)
    
    # 현재가 기준 투자심리도와 RSI 계산
    current_index = len(ohlcv_data) - 1
    investor_sentiment = _calculate_investor_sentiment(ohlcv_data, current_index)
    current_rsi = rsi14.iloc[-1] if not rsi14.empty else None
    
    # RSI + 투자심리도 투자액션 및 전략 계산
    investment_action, investment_strategy = _get_rsi_sentiment_strategy(current_rsi, investor_sentiment)
    
    # 수평선 및 가격 표시 (메인차트 중앙에 표시) - 두께 50% 감소
    ax_main.axhline(y=current_price, color='black', linestyle=':', linewidth=0.4, alpha=0.5)
    # 메인차트 중앙에 텍스트 배치 (폰트 크기 30% 감소)
    center_x = ohlcv_data.index[0] + (ohlcv_data.index[-1] - ohlcv_data.index[0]) * 0.5
    ax_main.text(center_x, current_price, 
                f'현재가: {current_price:.2f}', 
                fontsize=6.3, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', pad=2, boxstyle='round,pad=0.2'),
                color='black', fontweight='bold')
    
    ax_main.axhline(y=support, color='green', linestyle=':', linewidth=0.4, alpha=0.5)
    ax_main.text(center_x, support, f'지지선: {support:.2f}', 
                fontsize=6.3, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='green', pad=2, boxstyle='round,pad=0.2'),
                color='green', fontweight='bold')
    
    ax_main.axhline(y=resistance, color='red', linestyle=':', linewidth=0.4, alpha=0.5)
    ax_main.text(center_x, resistance, f'저항선: {resistance:.2f}', 
                fontsize=6.3, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='red', pad=2, boxstyle='round,pad=0.2'),
                color='red', fontweight='bold')
    
    # 현재가 캔들바 아래에 투자심리도와 RSI 표시
    if investor_sentiment is not None and current_rsi is not None:
        # 투자심리도에 따른 색상 설정
        if investor_sentiment < 40:
            sentiment_color = 'red'  # 약세
        elif investor_sentiment < 60:
            sentiment_color = 'orange'  # 중립
        else:
            sentiment_color = 'green'  # 강세
        
        # RSI에 따른 색상 설정
        if current_rsi < 30:
            rsi_color = 'green'  # 과매도
        elif current_rsi > 70:
            rsi_color = 'red'  # 과매수
        else:
            rsi_color = 'blue'  # 중립
        
        # 메인차트 하단에 투자심리도와 RSI 표시
        current_date = ohlcv_data.index[-1]
        sentiment_text = f'투자심리도: {investor_sentiment:.1f}%'
        rsi_text = f'RSI: {current_rsi:.1f}'
        
        # 메인차트 하단 y 위치 계산 (가격 범위의 하단 10% 지점)
        price_range = ohlcv_data['High'].max() - ohlcv_data['Low'].min()
        bottom_y = ohlcv_data['Low'].min() + price_range * 0.1
        
        # 개별 RSI, 투자심리도 하이라이트 박스 제거 - 통합 박스로 대체
        
        # 밴드 하단에 통합 정보 하이라이트 박스 생성
        if investment_action and investment_strategy:
            # 투자액션에 따른 색상 설정
            if "매수" in investment_action:
                action_color = 'green'
            elif "매도" in investment_action:
                action_color = 'red'
            elif "관망" in investment_action:
                action_color = 'gray'
            else:
                action_color = 'blue'
            
                    # 밴드 하단 위치 계산 (subplot과 겹치지 않도록 더 아래쪽으로 조정)
        band_bottom_y = bottom_y - price_range * 0.30
            
                    # SMA200과 현재가 비교 정보 추가
        sma200_current = sma200.iloc[-1] if not sma200.empty else None
        if sma200_current is not None:
            if current_price > sma200_current:
                sma_status = f'SMA200: {sma200_current:.2f} (현재가 상단)'
                sma_color = 'green'
            else:
                sma_status = f'SMA200: {sma200_current:.2f} (현재가 하단)'
                sma_color = 'red'
        else:
            sma_status = 'SMA200: 데이터 부족'
            sma_color = 'gray'
        
        # 통합 정보를 하나의 박스에 표시 (현재 일시, 현재가, 투자전략, 버핏지수 포함)
        # 실시간 주가 시간을 우선으로 사용하여 일시 표시
        if realtime_price is not None and realtime_time is not None:
            # 실시간 주가 시간을 한국 시간으로 변환하여 일시 표시
            realtime_local = get_local_datetime(realtime_time)
            current_datetime = realtime_local.strftime('%Y-%m-%d %H:%M')
            
            # 전일대비 등락률 정보 추가
            if daily_change:
                change_info = f' {daily_change["direction"]} {daily_change["change_percentage"]:.2f}%'
                price_info = f'[실시간가] ${realtime_price:.2f} ({realtime_local.strftime("%H:%M")}){change_info}'
            else:
                price_info = f'[실시간가] ${realtime_price:.2f} ({realtime_local.strftime("%H:%M")})'
        else:
            # 실시간 주가가 없는 경우 기존 데이터 시간 사용
            utc_datetime = ohlcv_data.index[-1]
            local_datetime = get_local_datetime(utc_datetime)
            current_datetime = local_datetime.strftime('%Y-%m-%d %H:%M')
            
            # 전일대비 등락률 정보 추가
            if daily_change:
                change_info = f' {daily_change["direction"]} {daily_change["change_percentage"]:.2f}%'
                price_info = f'[현재가] ${current_price:.2f}{change_info}'
            else:
                price_info = f'[현재가] ${current_price:.2f}'
        
        info_text = f'[일시] {current_datetime} (KST)\n{price_info}\n[데이터] {price_source}\n\n투자전략: {investment_strategy}\n투자액션: {investment_action}\nRSI: {current_rsi:.1f}\n투자심리도: {investor_sentiment:.1f}%\n{sma_status}'
        
        # 버핏지수 정보 추가
        if buffett_value:
            info_text += f'\n\n버핏지수: {buffett_value}% ({buffett_sentiment})\n시장가이드: {buffett_guide}'
        
        # 메인차트 내부에 투자전략 정보 박스 생성 (현재가 캔들바 아래의 적절한 위치)
        # 메인차트의 y축 범위를 고려하여 적절한 위치 계산
        main_chart_bottom = ax_main.get_ylim()[0]  # 메인차트 하단 y값
        main_chart_top = ax_main.get_ylim()[1]     # 메인차트 상단 y값
        price_range = main_chart_top - main_chart_bottom
        
        # 현재가 캔들바 아래의 적절한 위치 (메인차트 하단에서 약간 위)
        strategy_y = main_chart_bottom + price_range * 0.25  # 하단에서 25% 위로 조정 (subplot과 완전히 겹치지 않도록)
        
        # 투자전략 정보를 메인차트 내부에 직접 표시 (투명도 더 높임)
        ax_main.text(current_date, strategy_y, info_text,
                    fontsize=7, ha='center', va='top',  # 폰트 크기 약간 축소, valign을 top으로 설정
                    bbox=dict(boxstyle="round,pad=0.8",  # 패딩 증가
                             facecolor='white', 
                             edgecolor='darkorange',  # 테두리색을 더 진하게
                             linewidth=2.5,  # 테두리 두께 증가
                             alpha=0.5),  # 투명도를 0.5로 더 낮춤 (더 투명하게)
                    color='black', fontweight='bold',
                    zorder=1000)  # 최고 zorder 값으로 레이어 최상단에 표시
        
        # 현재 캔들바에서 투자전략 정보 박스까지 수직선 연결 (박스 테두리색과 동일)
        ax_main.axvline(x=current_date, ymin=0, ymax=1, color='darkorange', linestyle='--', 
                       linewidth=2.0, alpha=0.8, zorder=999)  # 색상과 투명도 조정
        
        print(f"현재가 투자심리도: {investor_sentiment:.1f}%, RSI: {current_rsi:.1f}")
        if investment_action and investment_strategy:
            print(f"투자액션: {investment_action}, 투자전략: {investment_strategy}")
    
    # POC 범위 분석
    poc_analysis = analyze_poc_price_range(current_price, recent_poc_price or poc_price, poc_price)
    
    # 신호 정보를 파일로 저장 (PCR 전략과 투자액션 포함)
    # 실시간 주가 시간을 우선으로 사용하여 날짜 설정
    if realtime_price is not None and realtime_time is not None:
        realtime_local = get_local_datetime(realtime_time)
        signal_date = realtime_local.strftime('%Y-%m-%d')
    else:
        # 실시간 주가가 없는 경우 기존 데이터 시간 사용
        utc_datetime = ohlcv_data.index[-1]
        local_datetime = get_local_datetime(utc_datetime)
        signal_date = local_datetime.strftime('%Y-%m-%d')
    
    signal_info = {
        'ticker': ticker,
        'date': signal_date,
        'realtime_price': realtime_price,
        'realtime_time': realtime_time,
        'data_freshness': data_freshness,
        'price_source': price_source,
        'daily_change': daily_change,
        'signal': 'HOLD',  # 기본값
        'pcr_strategy': None,
        'investment_action': investment_action,
        'current_price': current_price,
        'rsi': current_rsi,
        'investor_sentiment': investor_sentiment,
        'vix': vix_final,
        'naaim': naaim_final,
        'pcr': pcr,
        'c_poc_price': recent_poc_price or poc_price,
        't_poc_price': poc_price,
        'c_poc_range': poc_analysis['c_poc_range'],
        'c_poc_strength': poc_analysis['c_poc_strength'],
        't_poc_range': poc_analysis['t_poc_range'],
        't_poc_strength': poc_analysis['t_poc_strength'],
        'buffett_indicator': buffett_value,
        'buffett_sentiment': buffett_sentiment if buffett_value else None,
        'buffett_guide': buffett_guide if buffett_value else None
    }
    
    # 실제 매수/매도 신호를 기반으로 신호 결정
    if trade_signals:
        latest_signal = trade_signals[-1]
        if latest_signal['type'] == 'BUY':
            # BUY 신호의 강도 결정
            if 'HMA 상향돌파' in latest_signal.get('reason', ''):
                signal_info['signal'] = 'BUY_STRONG'  # HMA 돌파는 강한 신호
            else:
                signal_info['signal'] = 'BUY'  # 일반 매수 신호
        elif latest_signal['type'] == 'SELL':
            signal_info['signal'] = 'SELL'  # 매도 신호
        else:
            signal_info['signal'] = 'HOLD'
    else:
        signal_info['signal'] = 'HOLD'
    
    # PCR 전략 가이드 생성
    if pcr is not None:
        if pcr > 1.5:
            signal_info['pcr_strategy'] = "PCR>1.5: 매수(과매도)"
        elif pcr > 1.0:
            signal_info['pcr_strategy'] = "PCR 1.0-1.5: 신중(비관)"
        elif pcr > 0.7:
            signal_info['pcr_strategy'] = "PCR 0.7-1.0: 중립(균형)"
        elif pcr > 0.4:
            signal_info['pcr_strategy'] = "PCR 0.4-0.7: 낙관(상승)"
        else:
            signal_info['pcr_strategy'] = "PCR<0.4: 매도(과열)"
    else:
        # PCR 값이 None일 때 기본값 설정
        signal_info['pcr_strategy'] = "PCR: 데이터 부족"
    
    # 신호 파일 저장
    signal_file_path = save_path.replace('_chart.png', '_signal.txt')
    with open(signal_file_path, 'w', encoding='utf-8') as f:
        f.write(f"=== {ticker} 신호 분석 ===\n")
        # 실시간 주가 시간을 우선으로 사용하여 분석 일시 표시
        if realtime_price is not None and realtime_time is not None:
            realtime_local = get_local_datetime(realtime_time)
            analysis_datetime = realtime_local.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"분석 일시: {analysis_datetime} (KST)\n")
        else:
            # 실시간 주가가 없는 경우 기존 데이터 시간 사용
            utc_datetime = ohlcv_data.index[-1]
            local_datetime = get_local_datetime(utc_datetime)
            analysis_datetime = local_datetime.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"분석 일시: {analysis_datetime} (KST)\n")
        
        f.write(f"현재가: ${signal_info['current_price']:.2f}\n")
        
        # 전일대비 등락률 정보 추가
        if daily_change:
            f.write(f"전일종가: ${daily_change['prev_close']:.2f}\n")
            f.write(f"등락률: {daily_change['direction']} {daily_change['change_percentage']:.2f}% (${daily_change['change_amount']:.2f})\n")
        else:
            f.write(f"전일종가: N/A\n")
            f.write(f"등락률: N/A\n")
        
        # 실시간 주가 정보 추가
        if realtime_price is not None and realtime_time is not None:
            realtime_local = get_local_datetime(realtime_time)
            realtime_datetime = realtime_local.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"실시간가: ${realtime_price:.2f} ({realtime_datetime} KST)\n")
            f.write(f"데이터신선도: {data_freshness}\n")
        else:
            f.write(f"실시간가: N/A\n")
            f.write(f"데이터신선도: N/A\n")
        f.write(f"RSI: {signal_info['rsi']:.1f}\n")
        # 투자심리도 값이 None일 때 안전하게 처리
        if signal_info['investor_sentiment'] is not None:
            f.write(f"투자심리도: {signal_info['investor_sentiment']:.1f}%\n")
        else:
            f.write(f"투자심리도: N/A\n")
        f.write(f"투자액션(RSI+ISI): {signal_info['investment_action']}\n")
        f.write(f"투자전략: {investment_strategy}\n")
        f.write(f"VIX: {signal_info['vix']:.2f}\n")
        # NAIIM 값이 None일 때 안전하게 처리
        if signal_info['naaim'] is not None:
            f.write(f"NAIIM: {signal_info['naaim']:.1f}\n")
        else:
            f.write(f"NAIIM: N/A\n")
        # PCR 값이 None일 때 안전하게 처리
        if signal_info['pcr'] is not None:
            f.write(f"PCR: {signal_info['pcr']:.3f}\n")
        else:
            f.write(f"PCR: N/A\n")
        # PCR 전략이 None일 때 안전하게 처리
        if signal_info['pcr_strategy'] is not None:
            f.write(f"PCR 전략: {signal_info['pcr_strategy']}\n")
        else:
            f.write(f"PCR 전략: N/A\n")
        f.write(f"C-POC: {signal_info['c_poc_price']:.2f} ({signal_info['c_poc_range']}, 강도: {signal_info['c_poc_strength']})\n")
        f.write(f"T-POC: {signal_info['t_poc_price']:.2f} ({signal_info['t_poc_range']}, 강도: {signal_info['t_poc_strength']})\n")
        f.write(f"버핏지수: {signal_info['buffett_indicator']}% ({signal_info['buffett_sentiment']})\n")
        f.write(f"버핏가이드: {signal_info['buffett_guide']}\n")
        f.write(f"=== 신호 요약 ===\n")
        f.write(f"{signal_info['signal']}\n")
    
    print(f"신호 파일 저장 완료: {signal_file_path}")

    # 매수/매도 신호 표시
    for signal in trade_signals:
        color = 'blue' if signal['type'] == 'BUY' else 'red'
        marker = '^' if signal['type'] == 'BUY' else 'v'
        is_buy = signal['type'] == 'BUY'
        
        # 신호 위치 설정
        if is_buy:
            y = lower_band[signal['date']] * 0.99
            signal_num = 'B1' if 'HMA 상향돌파' in signal.get('reason', '') else 'B2'
        else:
            y = upper_band[signal['date']] * 1.01
            signal_num = 'T1' if 'HMA 하향돌파' in signal.get('reason', '') else 'T2'
        
        # 신호 마커 표시
        ax_main.plot(signal['date'], y, marker=marker, color=color, 
                    markersize=10, markeredgecolor='black')
        
        # 날짜와 신호 번호 표시
        date_str = signal['date'].strftime('%m/%d')
        ax_main.text(signal['date'], y, f"{date_str}\n{signal_num}", 
                    rotation=45, fontsize=6, ha='right', va='top' if is_buy else 'bottom',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # 조건부 수직선/수평선 표시
    # MACD 골든크로스 구간 찾기
    macd_above_signal = (macd > macd_signal)
    macd_golden_zone = macd_above_signal.cumsum()
    macd_golden_mask = macd_golden_zone > 0

    # 조건에 맞는 날짜 찾기
    cond = (
        (ohlcv_data['Volume'] > volume_upper) &
        macd_golden_mask &
        (rsi14 >= 50)
    )
    cond_dates = ohlcv_data.index[cond]

    # 거래량 분위수 계산 (3분위)
    volume_33 = ohlcv_data['Volume'].quantile(0.33)
    volume_67 = ohlcv_data['Volume'].quantile(0.67)
    
    # 교차점을 저장할 리스트
    intersection_points = []
    
    for dt in cond_dates:
        # 수직선
        ax_main.axvline(dt, color='magenta', linestyle='--', alpha=0.7, linewidth=1.2, zorder=20)
        # 하단에 일자 텍스트 표시 (45도 기울임)
        ax_main.text(dt, ax_main.get_ylim()[0], dt.strftime('%Y-%m-%d'), fontsize=6, color='magenta',
                    ha='center', va='top', rotation=45,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5), zorder=25)
        
        # 서브플롯에도 동일 수직선 표시 (라벨 없음)
        ax_volume.axvline(dt, color='magenta', linestyle='--', alpha=0.7, linewidth=1.0, zorder=10)
        ax_rsi.axvline(dt, color='magenta', linestyle='--', alpha=0.7, linewidth=1.0, zorder=10)
        ax_macd.axvline(dt, color='magenta', linestyle='--', alpha=0.7, linewidth=1.0, zorder=10)
        
        # 종가 기준 수평선 (거래량에 따른 두께 적용)
        close = ohlcv_data.loc[dt, 'Close']
        open_ = ohlcv_data.loc[dt, 'Open']
        volume = ohlcv_data.loc[dt, 'Volume']
        
        # 거래량에 따른 두께 결정 (50% 감소)
        if volume >= volume_67:
            linewidth = 0.8  # 상위 (높은 거래량) - 기존 1.6에서 50% 감소
        elif volume >= volume_33:
            linewidth = 0.6  # 중위 (보통 거래량) - 기존 1.2에서 50% 감소
        else:
            linewidth = 0.4  # 하위 (낮은 거래량) - 기존 0.8에서 50% 감소
        
        if close >= open_:
            ax_main.axhline(close, color='lime', linestyle='-', linewidth=linewidth, alpha=0.8, xmin=0, xmax=1, zorder=21)
            ax_main.text(ohlcv_data.index[-1], close, f'{close:.2f}', fontsize=7, color='black', ha='left', va='center',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5), zorder=22)
        else:
            ax_main.axhline(close, color='red', linestyle='-', linewidth=linewidth, alpha=0.8, xmin=0, xmax=1, zorder=21)
            ax_main.text(ohlcv_data.index[-1], close, f'{close:.2f}', fontsize=7, color='black', ha='left', va='center',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5), zorder=22)
        
        # 교차점 찾기: 분홍색 수직선과 녹색/빨간 수평선의 교차점
        # 분홍색 수직선은 이미 그려져 있고, 수평선도 그려져 있음
        # 교차점은 분홍색 수직선(dt)과 해당 날짜의 종가 수평선(close)의 교차점
        
        if close >= open_:
            # 녹색 수평선과 분홍색 수직선의 교차점
            intersection_points.append({
                'date': dt,
                'price': close,  # 해당 날짜의 종가
                'type': 'green_vertical'
            })
        else:
            # 빨간 수평선과 분홍색 수직선의 교차점
            intersection_points.append({
                'date': dt,
                'price': close,  # 해당 날짜의 종가
                'type': 'red_vertical'
            })
    
    # 교차점에 X 마커 표시
    for point in intersection_points:
        if point['type'] == 'green_vertical':
            # 녹색 수평선과 분홍색 수직선의 교차점
            ax_main.plot(point['date'], point['price'], marker='x', color='black', 
                        markersize=5, markeredgewidth=2, zorder=30)
        elif point['type'] == 'red_vertical':
            # 빨간 수평선과 분홍색 수직선의 교차점
            ax_main.plot(point['date'], point['price'], marker='x', color='black', 
                        markersize=5, markeredgewidth=2, zorder=30)

    # Volume Profile 오버레이 (메인차트 우측에 반투명하게)
    # 메인차트의 X축 범위 가져오기
    main_xlim = ax_main.get_xlim()
    main_ylim = ax_main.get_ylim()
    
    # Volume Profile을 메인차트 좌측에 오버레이
    # Volume Profile의 너비를 메인차트의 10%로 설정
    overlay_width = (main_xlim[1] - main_xlim[0]) * 0.10
    
    # 좌측에 Volume Profile 배치
    overlay_start = main_xlim[0] + (main_xlim[1] - main_xlim[0]) * 0.05  # 좌측에서 약간 떨어진 위치
    
    # Volume Profile 정규화 (0~1 범위로)
    max_volume = max(volume_profile)
    normalized_volume = [v / max_volume for v in volume_profile]
    
    # Net Volume Profile 정규화
    max_net_volume = max(abs(min(net_volume_profile)), abs(max(net_volume_profile))) if net_volume_profile else 1
    normalized_net_volume = [v / max_net_volume for v in net_volume_profile]
    
    # Volume Profile 막대 그리기 (Net Volume 색상 적용)
    bin_heights = price_bins[1] - price_bins[0]
    for i, (price, vol, net_vol, ratio) in enumerate(zip(price_bins[:-1], normalized_volume, normalized_net_volume, volume_ratios)):
        bar_width = vol * overlay_width * 0.8  # 막대 너비를 거래량에 비례하게
        
        # Net Volume에 따른 색상 설정
        if net_vol > 0:
            color = 'green'  # 상승 압력
        elif net_vol < 0:
            color = 'red'    # 하락 압력
        else:
            color = 'gray'   # 중립
        
        # Volume Profile 막대 그리기 (투명도 낮춤)
        ax_main.barh(price, bar_width, height=bin_heights, left=overlay_start, 
                    alpha=0.4, color=color, zorder=10)
        
        # 비율 텍스트 표시 (주요 구간만)
        if ratio > 2.0:  # 2% 이상인 구간만 표시
            ax_main.text(overlay_start + bar_width + overlay_width * 0.05, price, 
                        f'{ratio:.1f}%', fontsize=6, color='black', ha='left', va='center',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1), zorder=11)
    
    # POC (Point of Control) 표시 (최신일자까지 연장) - 두께 50% 감소
    poc_xmin = (overlay_start - main_xlim[0]) / (main_xlim[1] - main_xlim[0])
    poc_xmax = 1.0  # 최신일자까지 연장
    ax_main.axhline(poc_price, color='red', linestyle='--', alpha=0.8, linewidth=1, 
                   xmin=poc_xmin, xmax=poc_xmax, zorder=15, label=f'POC: {poc_price:.2f}')
    
    # POC 가격 텍스트 표시
    ax_main.text(overlay_start + overlay_width * 0.5, poc_price, f'{poc_price:.2f}', 
                fontsize=8, color='red', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', pad=2), zorder=16)
    
    # Value Area 표시 (Volume Profile 영역에만)
    value_xmax = (overlay_start + overlay_width - main_xlim[0]) / (main_xlim[1] - main_xlim[0])
    ax_main.axhspan(value_area_min, value_area_max, alpha=0.1, color='green', 
                   xmin=poc_xmin, xmax=value_xmax, zorder=5, label=f'Value Area: {value_area_min:.2f}-{value_area_max:.2f}')
    
    

    # 최근 6개월 Volume Profile 표시 (1년 이상인 경우)
    if analysis_period_days >= 365 and recent_6mo_data is not None and len(recent_6mo_data) >= 30:
        # 중앙에 최근 6개월 Volume Profile 배치
        center_overlay_start = main_xlim[0] + (main_xlim[1] - main_xlim[0]) * 0.45  # 중앙
        center_overlay_width = (main_xlim[1] - main_xlim[0]) * 0.10  # 10% 너비
        
        # 최근 6개월 Volume Profile 정규화
        max_recent_volume = max(recent_volume_profile)
        normalized_recent_volume = [v / max_recent_volume for v in recent_volume_profile]
        
        # 최근 6개월 Net Volume Profile 정규화
        max_recent_net_volume = max(abs(min(recent_net_volume_profile)), abs(max(recent_net_volume_profile))) if recent_net_volume_profile else 1
        normalized_recent_net_volume = [v / max_recent_net_volume for v in recent_net_volume_profile]
        
        # 최근 6개월 Volume Profile 막대 그리기
        recent_bin_heights = recent_price_bins[1] - recent_price_bins[0]
        for i, (price, vol, net_vol, ratio) in enumerate(zip(recent_price_bins[:-1], 
                                                            normalized_recent_volume, 
                                                            normalized_recent_net_volume, 
                                                            recent_volume_ratios)):
            bar_width = vol * center_overlay_width * 0.8  # 막대 너비를 거래량에 비례하게
            
            # 색상 설정 (연한 색상으로 구분)
            if net_vol > 0:
                color = 'lightgreen'  # 연한 녹색
            elif net_vol < 0:
                color = 'lightcoral'  # 연한 빨강
            else:
                color = 'lightgray'   # 연한 회색
            
            # 중앙에 막대 그리기 (투명도 낮춤)
            ax_main.barh(price, bar_width, height=recent_bin_heights, left=center_overlay_start, 
                        alpha=0.3, color=color, zorder=8)
        
        # 최근 6개월 POC (검은색 점선) - 우측 끝까지 연장 - 두께 50% 감소
        recent_poc_xmin = (center_overlay_start - main_xlim[0]) / (main_xlim[1] - main_xlim[0])
        recent_poc_xmax = 1.0  # 우측 끝까지 연장
        ax_main.axhline(recent_poc_price, color='black', linestyle=':', alpha=0.8, linewidth=1, 
                       xmin=recent_poc_xmin, xmax=recent_poc_xmax, zorder=14, 
                       label=f'최근 6개월 POC: {recent_poc_price:.2f}')
        
        # 최근 6개월 POC 가격 텍스트 표시
        ax_main.text(center_overlay_start + center_overlay_width * 0.5, recent_poc_price, 
                    f'{recent_poc_price:.2f}', fontsize=8, color='black', ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', pad=2), zorder=16)
        
        # 최근 6개월 Value Area 표시 (Volume Profile 영역에만)
        recent_value_xmax = (center_overlay_start + center_overlay_width - main_xlim[0]) / (main_xlim[1] - main_xlim[0])
        ax_main.axhspan(recent_value_area_min, recent_value_area_max, alpha=0.05, color='blue', 
                       xmin=recent_poc_xmin, xmax=recent_value_xmax, zorder=4, 
                       label=f'최근 6개월 Value Area: {recent_value_area_min:.2f}-{recent_value_area_max:.2f}')
    
    # Net Volume Profile 범례 추가 (투명도 조정)
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.4, label='전체 기간 상승 압력'),
        plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.4, label='전체 기간 하락 압력'),
        plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.4, label='전체 기간 중립')
    ]
    
    # 최근 6개월 범례 추가 (1년 이상인 경우)
    if analysis_period_days >= 365 and recent_6mo_data is not None and len(recent_6mo_data) >= 30:
        legend_elements.extend([
            plt.Rectangle((0, 0), 1, 1, facecolor='lightgreen', alpha=0.3, label='최근 6개월 상승 압력'),
            plt.Rectangle((0, 0), 1, 1, facecolor='lightcoral', alpha=0.3, label='최근 6개월 하락 압력'),
            plt.Line2D([], [], color='black', linestyle=':', linewidth=2, label='최근 6개월 POC')
        ])
    
    # 다이버전스 범례 추가 (RSI & MACD)
    legend_elements.extend([
        # RSI Regular/Hidden
        plt.Line2D([], [], color='green', linestyle='-', linewidth=1.5, marker='^', label='Bull Div (RSI)'),
        plt.Line2D([], [], color='red', linestyle='-', linewidth=1.5, marker='v', label='Bear Div (RSI)'),
        plt.Line2D([], [], color='green', linestyle=':', linewidth=1.5, marker='^', label='Bull Div (RSI, Hidden)'),
        plt.Line2D([], [], color='red', linestyle=':', linewidth=1.5, marker='v', label='Bear Div (RSI, Hidden)'),
        # MACD Regular/Hidden
        plt.Line2D([], [], color='forestgreen', linestyle='-', linewidth=1.2, marker='^', label='Bull Div (MACD)'),
        plt.Line2D([], [], color='darkred', linestyle='-', linewidth=1.2, marker='v', label='Bear Div (MACD)'),
        plt.Line2D([], [], color='forestgreen', linestyle=':', linewidth=1.2, marker='^', label='Bull Div (MACD, Hidden)'),
        plt.Line2D([], [], color='darkred', linestyle=':', linewidth=1.2, marker='v', label='Bear Div (MACD, Hidden)'),
    ])
    
    ax_main.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95), fontsize=8)

    # 거래량 차트 표시
    # 거래량 막대 (양봉/음봉 구분)
    colors = ['green' if close >= open_ else 'red' for close, open_ in zip(ohlcv_data['Close'], ohlcv_data['Open'])]
    ax_volume.bar(ohlcv_data.index, ohlcv_data['Volume'], color=colors, alpha=0.7, width=0.8)
    
    # 거래량 이동평균선
    volume_ma_5 = ohlcv_data['Volume'].rolling(window=5).mean()
    volume_ma_20 = ohlcv_data['Volume'].rolling(window=20).mean()
    ax_volume.plot(ohlcv_data.index, volume_ma_5, color='orange', linewidth=1, label='Volume MA(5)')
    ax_volume.plot(ohlcv_data.index, volume_ma_20, color='blue', linewidth=1, label='Volume MA(20)')
    
    # 볼륨 볼린저 밴드
    ax_volume.plot(ohlcv_data.index, volume_upper, color='red', linewidth=1, linestyle='--', label='Volume BB Upper')
    
    # 거래량 차트 설정
    ax_volume.set_title('Volume')
    ax_volume.set_ylabel('Volume')
    ax_volume.legend(fontsize=8)
    ax_volume.grid(True, alpha=0.3)
    
    # RSI 차트 표시
    ax_rsi.plot(ohlcv_data.index, rsi14, color='tab:blue', linewidth=1.2, label='RSI(14)')
    # 보조로 RSI(3) 얇게 표시
    ax_rsi.plot(ohlcv_data.index, rsi3, color='tab:orange', linewidth=0.8, alpha=0.6, label='RSI(3)')
    ax_rsi.axhline(70, color='red', linestyle='--', linewidth=0.4, alpha=0.6)
    ax_rsi.axhline(50, color='gray', linestyle=':', linewidth=0.4, alpha=0.6)
    ax_rsi.axhline(30, color='green', linestyle='--', linewidth=0.4, alpha=0.6)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_title('RSI')
    ax_rsi.set_ylabel('RSI')
    ax_rsi.legend(fontsize=8, loc='upper left')
    ax_rsi.grid(True, alpha=0.3)

    # MACD 차트 표시
    macd_colors = ['green' if v >= 0 else 'red' for v in hist]
    ax_macd.bar(ohlcv_data.index, hist, color=macd_colors, alpha=0.5, width=0.8, label='Histogram')
    ax_macd.plot(ohlcv_data.index, macd, color='tab:blue', linewidth=1.2, label='MACD')
    ax_macd.plot(ohlcv_data.index, macd_signal, color='tab:orange', linewidth=1.0, label='Signal')
    ax_macd.axhline(0, color='black', linewidth=0.4, alpha=0.6)
    ax_macd.set_title('MACD (12,26,9)')
    ax_macd.set_ylabel('MACD')
    ax_macd.legend(fontsize=8, loc='upper left')
    ax_macd.grid(True, alpha=0.3)

    # =====================
    # RSI 다이버전스 탐지/표시 (메인차트에만)
    # =====================
    try:
        rsi_divs = _detect_rsi_divergences(
            ohlcv_data,
            rsi14,
            pivot_span_left=rsi_pivot_span,
            pivot_span_right=rsi_pivot_span,
            valid_window_bars=rsi_divergence_window,
            include_hidden=include_hidden_divergence,
        )

        for d in rsi_divs:
            date1, date2 = d['date1'], d['date2']
            p1, p2 = d['price1'], d['price2']
            if d['kind'] == 'BULL':
                color = 'green'
                marker = '^'
                y_text_offset = -0.01
                label = 'Bull Div (RSI)' if d['subtype'] == 'regular' else 'Bull Div (RSI, Hidden)'
                linestyle = '-' if d['subtype'] == 'regular' else ':'
            else:
                color = 'red'
                marker = 'v'
                y_text_offset = 0.01
                label = 'Bear Div (RSI)' if d['subtype'] == 'regular' else 'Bear Div (RSI, Hidden)'
                linestyle = '-' if d['subtype'] == 'regular' else ':'

            # 가격 피벗을 선으로 연결
            ax_main.plot([date1, date2], [p1, p2], color=color, linestyle=linestyle, linewidth=1.5, alpha=0.9, zorder=35)
            # 시그널 마커 (두 번째 피벗 위치)
            ax_main.plot(date2, p2, marker=marker, color=color, markersize=9, markeredgecolor='black', zorder=36)
            # 라벨(두 번째 피벗 상/하단에 배치)
            ylim = ax_main.get_ylim()
            y_offset = (ylim[1] - ylim[0]) * y_text_offset
            
            # 다이버전스 강도 표시 추가
            strength_text = ""
            if 'strength' in d:
                if d['strength'] == 'strong':
                    strength_text = " (강함)"
                    label_color = 'darkred' if d['kind'] == 'BEAR' else 'darkgreen'
                else:
                    strength_text = " (보통)"
                    label_color = color
            else:
                label_color = color
            
            full_label = label + strength_text
            ax_main.text(date2, p2 + y_offset, full_label, fontsize=7, color=label_color,
                         ha='left', va='bottom' if d['kind'] == 'BULL' else 'top',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1), zorder=37)
    except Exception:
        pass



    # =====================
    # RSI 다이버전스 패턴 분석 (9-11일, 신뢰도 70% 이상) - 별도 차트 생성
    # =====================
    try:
        rsi_pattern_divergences = analyze_rsi_divergence_patterns(
            ohlcv_data, 
            rsi_period=14, 
            pattern_range=(9, 11)
        )
        
        # RSI 다이버전스 패턴을 별도 차트로 생성
        if rsi_pattern_divergences:
            # 별도 차트 파일 경로 생성
            divergence_chart_path = save_path.replace('.png', '_rsi_divergence.png')
            
            # 별도 차트 생성 및 저장
            create_rsi_divergence_chart(
                ohlcv_data, 
                rsi_pattern_divergences, 
                ticker, 
                start_date, 
                end_date, 
                divergence_chart_path
            )
            
            print(f"RSI 다이버전스 패턴 감지: {len(rsi_pattern_divergences)}개")
            for div in rsi_pattern_divergences:
                print(f"  - {div['type']}: {div['start_date'].strftime('%Y-%m-%d')} ~ {div['end_date'].strftime('%Y-%m-%d')} (신뢰도: {div['confidence']:.1%}, 기간: {div['pattern_days']}일)")
        else:
            print("RSI 다이버전스 패턴이 감지되지 않았습니다.")
    except Exception as e:
        print(f"RSI 다이버전스 패턴 분석 실패: {e}")
        pass

    # =====================
    # MACD 히스토그램 다이버전스 탐지/표시 (메인차트에만)
    # =====================
    try:
        macd_divs = _detect_macd_divergences(
            ohlcv_data,
            hist,
            pivot_span_left=rsi_pivot_span,
            pivot_span_right=rsi_pivot_span,
            valid_window_bars=rsi_divergence_window,
            include_hidden=include_hidden_divergence,
        )

        for d in macd_divs:
            date1, date2 = d['date1'], d['date2']
            p1, p2 = d['price1'], d['price2']
            if d['kind'] == 'BULL':
                color = 'forestgreen'
                marker = '^'
                y_text_offset = -0.015
                label = 'Bull Div (MACD)' if d['subtype'] == 'regular' else 'Bull Div (MACD, Hidden)'
                linestyle = '-' if d['subtype'] == 'regular' else ':'
            else:
                color = 'darkred'
                marker = 'v'
                y_text_offset = 0.015
                label = 'Bear Div (MACD)' if d['subtype'] == 'regular' else 'Bear Div (MACD, Hidden)'
                linestyle = '-' if d['subtype'] == 'regular' else ':'

            ax_main.plot([date1, date2], [p1, p2], color=color, linestyle=linestyle, linewidth=1.2, alpha=0.85, zorder=34)
            ax_main.plot(date2, p2, marker=marker, color=color, markersize=8, markeredgecolor='black', zorder=35)
            ylim = ax_main.get_ylim()
            y_offset = (ylim[1] - ylim[0]) * y_text_offset
            ax_main.text(date2, p2 + y_offset, label, fontsize=7, color=color,
                         ha='left', va='bottom' if d['kind'] == 'BULL' else 'top',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1), zorder=36)
    except Exception:
        pass

    # =====================
    # RSI 다이버전스 패턴을 메인 차트에 오버레이로 표시
    # =====================
    try:
        if rsi_pattern_divergences:
            # 메인 차트에 RSI 다이버전스 오버레이
            rsi_legend_elements = plot_rsi_divergence_on_main_chart(ax_main, rsi_pattern_divergences, ohlcv_data)
            
            # 기존 범례에 RSI 다이버전스 범례 추가
            if rsi_legend_elements:
                # 기존 범례 가져오기
                existing_legend = ax_main.get_legend()
                if existing_legend:
                    # 기존 범례와 새로운 범례를 합쳐서 표시
                    try:
                        all_handles = list(existing_legend.legendHandles) + rsi_legend_elements
                        ax_main.legend(handles=all_handles, loc='upper left', fontsize=8)
                    except AttributeError:
                        # legendHandles 속성이 없는 경우 새로운 범례만 표시
                        ax_main.legend(handles=rsi_legend_elements, loc='upper left', fontsize=8)
                else:
                    # RSI 다이버전스 범례만 표시
                    ax_main.legend(handles=rsi_legend_elements, loc='upper left', fontsize=8)
    except Exception as e:
        print(f"메인 차트에 RSI 다이버전스 표시 실패: {e}")
        pass

    # x축 날짜 포맷 설정
    ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_volume.tick_params(axis='x', rotation=45)

    # 메인차트 설정 (종목 정보 포함)
    title = f'{ticker} - {long_name}\n'
    title += f'섹터: {sector} | 산업: {industry}\n'
    title += f'전체 기간: {start_date.strftime("%Y-%m-%d")} ~ {end_date.strftime("%Y-%m-%d")}'
    if analysis_period_days >= 365 and recent_6mo_data is not None and len(recent_6mo_data) >= 30:
        recent_6mo_start = end_date - pd.Timedelta(days=180)
        title += f'\n최근 6개월: {recent_6mo_start.strftime("%Y-%m-%d")} ~ {end_date.strftime("%Y-%m-%d")}'
    
    ax_main.set_title(title)
    ax_main.set_ylabel('Price')
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc='upper left', fontsize=8)
    
    # x축 날짜 포맷 설정
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_main.tick_params(axis='x', rotation=45)

    # 레이아웃 조정
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.06, hspace=0.12)

    # 저장 또는 표시
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show() 