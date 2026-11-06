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
from ..chart_patterns import (
    analyze_seven_criteria,
    count_pattern_strip_rows,
    normalize_pattern_main_ids,
    pattern_strip_height_ratio,
    plot_pattern_main_overlays,
    plot_pattern_strip,
)
from ..utils import get_available_font
from ..box_range_windows import compute_box_range_windows
from ..disparity_strategy import analyze_disparity_strategy
from ..buffett_indicator import calculate_buffett_indicator, get_buffett_sentiment
from ..fear_greed_indicator import calculate_fear_greed_indicator, fetch_cnn_fear_greed_history
from ..peg_indicator import calculate_peg_indicator
from ..adx_dmi_indicator import analyze_adx_dmi, plot_adx_buy_on_main
import matplotlib.patches as mpatches
import pandas_datareader.data as web
from datetime import datetime, timedelta
from typing import Optional, Sequence
import os

def calculate_volume_profile(box_data, price_bins=15):
    """
    박스권 내에서 Volume Profile을 계산합니다.
    
    Args:
        box_data: 박스권 기간의 OHLCV 데이터
        price_bins: 가격 구간 분할 수 (기본값: 15)
    
    Returns:
        dict: Volume Profile 정보 {price_level: volume, ...}
    """
    try:
        if len(box_data) == 0:
            return {}
        
        # 가격 범위 계산
        high_price = float(box_data['High'].max())
        low_price = float(box_data['Low'].min())
        price_range = high_price - low_price
        
        if price_range == 0:
            return {}
        
        # 가격 구간 설정
        bin_size = price_range / price_bins
        price_levels = [low_price + i * bin_size for i in range(price_bins + 1)]
        
        # Volume Profile 계산
        volume_profile = {}
        for i in range(len(price_levels) - 1):
            price_level = (price_levels[i] + price_levels[i + 1]) / 2
            volume_profile[price_level] = 0.0
        
        # 각 거래일의 거래량을 가격 구간에 분배
        for _, row in box_data.iterrows():
            daily_high = float(row['High'])
            daily_low = float(row['Low'])
            daily_volume = float(row['Volume'])
            
            # 해당 일의 가격 범위가 어느 구간에 속하는지 계산
            for i in range(len(price_levels) - 1):
                price_level = (price_levels[i] + price_levels[i + 1]) / 2
                bin_low = price_levels[i]
                bin_high = price_levels[i + 1]
                
                # 가격 구간과 일일 가격 범위의 교집합 계산
                overlap_low = max(daily_low, bin_low)
                overlap_high = min(daily_high, bin_high)
                
                if overlap_low < overlap_high and daily_high != daily_low:
                    # 교집합 비율만큼 거래량 분배
                    overlap_ratio = (overlap_high - overlap_low) / (daily_high - daily_low)
                    volume_profile[price_level] += daily_volume * overlap_ratio
        
        # POC (Point of Control) 찾기
        if volume_profile and any(v > 0 for v in volume_profile.values()):
            poc_price = max(volume_profile, key=volume_profile.get)
            poc_volume = volume_profile[poc_price]
        else:
            # 거래량이 없으면 박스권 중앙 사용
            poc_price = (high_price + low_price) / 2
            poc_volume = 0
        
        return {
            'volume_profile': volume_profile,
            'poc_price': poc_price,
            'poc_volume': poc_volume,
            'price_levels': price_levels,
            'bin_size': bin_size
        }
    except Exception as e:
        print(f"Volume Profile 계산 오류: {e}")
        return {}

def calculate_box_ranges(data, box_period=20, num_boxes=2, overlap_days=5, min_box_days=5, avoid_time_overlap=True):
    """
    메인 차트 파란 박스권 표시 전용.

    기하 창은 box_range_windows.compute_box_range_windows 에 맡기고,
    Volume Profile·현재가·콘솔 로그만 붙인다.
    차트 패턴 range_box 는 이 함수를 쓰지 않는다(경량 창만 별도 모듈).
    
    Args:
        data: OHLCV 데이터
        box_period: 박스권 계산 기간 (기본값: 20일)
        num_boxes: 표시할 박스권 개수 (기본값: 2개)
        overlap_days: 박스권 간 겹치는 일수 (기본값: 5일, avoid_time_overlap=True일 때 무시)
        min_box_days: 최소 박스 유지 기간 (API 호환, 본문 미사용)
        avoid_time_overlap: 시간축 겹침 방지 여부 (기본값: True)
    
    Returns:
        list: 박스권 정보 리스트 (volume_profile·current_price 포함)
    """
    try:
        windows = compute_box_range_windows(
            data, box_period, num_boxes, overlap_days, avoid_time_overlap
        )
        box_ranges = []
        for win in windows:
            sd = win["start_date"]
            ed = win["end_date"]
            box_data = data.loc[(data.index >= sd) & (data.index <= ed)]
            box_info = dict(win)
            box_info["volume_profile"] = calculate_volume_profile(box_data)
            idx_i = int(win["index"])
            if idx_i == 0:
                box_info["current_price"] = data.iloc[-1]["Close"]
            print(
                f"박스권 {idx_i + 1}: {win['name']} | 날짜: "
                f"{pd.Timestamp(sd).strftime('%Y-%m-%d')} ~ {pd.Timestamp(ed).strftime('%Y-%m-%d')} | "
                f"가격범위: ${win['low']:.2f} ~ ${win['high']:.2f} | 범위: ${win['range']:.2f}"
            )
            box_ranges.append(box_info)
        return box_ranges
    except Exception as e:
        print(f"박스권 계산 오류: {e}")
        return []

def get_box_style_config(box_style, num_boxes):
    """
    박스권 스타일 설정을 반환합니다.
    
    Args:
        box_style: 박스권 스타일 ('default', 'gradient', 'rainbow')
        num_boxes: 박스권 개수
    
    Returns:
        dict: 색상, 투명도, 라인 스타일 설정
    """
    if box_style == 'gradient':
        # 그라디언트 스타일: 최신 박스권일수록 진하게
        colors = ['blue'] * num_boxes
        alphas = [0.4 - (i * 0.1) for i in range(num_boxes)]
        alphas = [max(0.1, alpha) for alpha in alphas]  # 최소 0.1 유지
        linewidths = [2.0 - (i * 0.2) for i in range(num_boxes)]
        linewidths = [max(1.0, width) for width in linewidths]  # 최소 1.0 유지
    elif box_style == 'rainbow':
        # 무지개 스타일: 각 박스권마다 다른 색상
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        alphas = [0.3] * num_boxes
        linewidths = [1.5] * num_boxes
    else:  # default
        # 기본 스타일: 모든 박스권 동일
        colors = ['blue'] * num_boxes
        alphas = [0.2] * num_boxes
        linewidths = [1.5] * num_boxes
    
    return {
        'colors': colors,
        'alphas': alphas,
        'linewidths': linewidths
    }

def plot_box_ranges_with_style(ax, box_ranges, box_style='default', currency_symbol='$'):
    """
    박스권을 스타일에 따라 시각적으로 구분하여 표시합니다.
    모든 박스권을 실제 가격 위치에 표시하며, 색상과 투명도로 시각적으로 구분합니다.
    
    Args:
        ax: matplotlib axes 객체
        box_ranges: 박스권 정보 리스트
        box_style: 박스권 스타일 ('default', 'gradient', 'rainbow')
        currency_symbol: 통화 기호
    """
    if not box_ranges:
        return
    
    # 스타일 설정 가져오기
    style_config = get_box_style_config(box_style, len(box_ranges))
    
    for i, box in enumerate(box_ranges):
        # 스타일 설정 적용
        box_color = style_config['colors'][i % len(style_config['colors'])]
        alpha_value = style_config['alphas'][i % len(style_config['alphas'])]
        linewidth = style_config['linewidths'][i % len(style_config['linewidths'])]
        
        # 실제 가격 위치 사용 (오프셋 없음)
        box_low = box['low']
        box_high = box['high']
        
        # 박스권 사각형 그리기 (실제 가격 위치)
        box_width = (box['end_date'] - box['start_date']).days
        box_rect = mpatches.Rectangle(
            (mdates.date2num(box['start_date']), box_low),
            box_width,
            box_high - box_low,
            linewidth=linewidth,
            edgecolor=box_color,
            facecolor=box_color,
            alpha=alpha_value,
            zorder=10-i  # 최신 박스권일수록 위에 표시
        )
        ax.add_patch(box_rect)
        
        # 박스권 라벨 (현재 박스권만 표시 또는 rainbow 스타일일 때 모두 표시)
        if i == 0 or box_style == 'rainbow':
            box_center_x = mdates.date2num(box['start_date']) + box_width / 2
            label_text = f"{box['name']}\n{currency_symbol}{box_low:.2f}-{currency_symbol}{box_high:.2f}"
            
            ax.text(box_center_x, box_high + (box['range'] * 0.05), 
                    label_text,
                    ha='center', va='bottom', fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=box_color, alpha=0.8),
                    color='white', fontweight='bold')
        
        # 박스권별 POC 표시 (실제 가격 위치)
        plot_volume_profile_bars(ax, box, box_color, currency_symbol)

def plot_volume_profile_bars(ax, box, color, currency_symbol='$'):
    """
    박스권에 실제 거래량 기반 POC (Point of Control)를 표시합니다.
    
    Args:
        ax: matplotlib axes 객체
        box: 박스권 정보 딕셔너리
        color: 색상
        currency_symbol: 통화 기호
    """
    try:
        # 박스권의 날짜 범위
        start_date = box['start_date']
        end_date = box['end_date']
        box_center_x = mdates.date2num(start_date) + (mdates.date2num(end_date) - mdates.date2num(start_date)) / 2
        
        # Volume Profile 데이터에서 실제 POC 가격 가져오기
        volume_profile_data = box.get('volume_profile', {})
        poc_price = None
        
        try:
            if volume_profile_data and isinstance(volume_profile_data, dict):
                poc_price = volume_profile_data.get('poc_price')
                if poc_price is not None:
                    poc_price = float(poc_price)
                    print(f"실제 POC 가격: {currency_symbol}{poc_price:.2f} ({box.get('name', 'Unknown')})")
        except (ValueError, TypeError) as e:
            print(f"POC 가격 변환 오류: {e}")
            poc_price = None
        
        if poc_price is None:
            # Volume Profile 데이터가 없거나 오류가 있으면 박스권 중앙 사용
            poc_price = float(box['center'])
            print(f"기본 POC 가격: {currency_symbol}{poc_price:.2f} ({box.get('name', 'Unknown')})")
        
        # POC 라인 표시 위치 (실제 가격 위치, 오프셋 없음)
        poc_x_start = mdates.date2num(start_date)
        poc_x_end = mdates.date2num(end_date)
        
        # POC 수평선 (점선으로 표시)
        ax.plot([poc_x_start, poc_x_end], [poc_price, poc_price], 
               color=color, linewidth=1.5, linestyle='--', alpha=0.9, zorder=12)
        
        # POC 가격 표시 (중앙에만)
        box_center_x = poc_x_start + (poc_x_end - poc_x_start) / 2
        ax.text(box_center_x, poc_price, f'POC: {currency_symbol}{poc_price:.2f}', 
               fontsize=7, color=color, ha='center', va='bottom', fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor=color), 
               zorder=13)
        
    except Exception as e:
        print(f"POC 표시 오류: {e}")

def detect_rsi_cross_signals(rsi3, rsi14, rsi_signal):
    """
    RSI 크로스 신호를 감지합니다.
    
    Args:
        rsi3: RSI(3) 시리즈
        rsi14: RSI(14) 시리즈  
        rsi_signal: RSI Signal 시리즈 (RSI(14)의 9기간 EMA)
    
    Returns:
        dict: 크로스 신호 정보
    """
    signals = {
        'rsi3_rsi14_crosses': [],  # RSI(3) vs RSI(14) 크로스
        'rsi14_signal_crosses': [],  # RSI(14) vs Signal 크로스
        'golden_crosses': [],  # 골든크로스 (상향)
        'dead_crosses': []  # 데드크로스 (하향)
    }
    
    # RSI(3) vs RSI(14) 크로스 감지
    for i in range(1, len(rsi3)):
        if pd.notna(rsi3.iloc[i]) and pd.notna(rsi14.iloc[i]):
            # 상향 크로스
            if rsi3.iloc[i-1] <= rsi14.iloc[i-1] and rsi3.iloc[i] > rsi14.iloc[i]:
                signals['rsi3_rsi14_crosses'].append({
                    'date': rsi3.index[i],
                    'type': 'golden',
                    'rsi3': rsi3.iloc[i],
                    'rsi14': rsi14.iloc[i],
                    'level': 'oversold' if rsi14.iloc[i] < 30 else 'neutral' if rsi14.iloc[i] < 70 else 'overbought'
                })
            # 하향 크로스
            elif rsi3.iloc[i-1] >= rsi14.iloc[i-1] and rsi3.iloc[i] < rsi14.iloc[i]:
                signals['rsi3_rsi14_crosses'].append({
                    'date': rsi3.index[i],
                    'type': 'dead',
                    'rsi3': rsi3.iloc[i],
                    'rsi14': rsi14.iloc[i],
                    'level': 'oversold' if rsi14.iloc[i] < 30 else 'neutral' if rsi14.iloc[i] < 70 else 'overbought'
                })
    
    # RSI(14) vs Signal 크로스 감지
    for i in range(1, len(rsi14)):
        if pd.notna(rsi14.iloc[i]) and pd.notna(rsi_signal.iloc[i]):
            # 상향 크로스
            if rsi14.iloc[i-1] <= rsi_signal.iloc[i-1] and rsi14.iloc[i] > rsi_signal.iloc[i]:
                signals['rsi14_signal_crosses'].append({
                    'date': rsi14.index[i],
                    'type': 'golden',
                    'rsi14': rsi14.iloc[i],
                    'signal': rsi_signal.iloc[i],
                    'level': 'oversold' if rsi14.iloc[i] < 30 else 'neutral' if rsi14.iloc[i] < 70 else 'overbought'
                })
            # 하향 크로스
            elif rsi14.iloc[i-1] >= rsi_signal.iloc[i-1] and rsi14.iloc[i] < rsi_signal.iloc[i]:
                signals['rsi14_signal_crosses'].append({
                    'date': rsi14.index[i],
                    'type': 'dead',
                    'rsi14': rsi14.iloc[i],
                    'signal': rsi_signal.iloc[i],
                    'level': 'oversold' if rsi14.iloc[i] < 30 else 'neutral' if rsi14.iloc[i] < 70 else 'overbought'
                })
    
    # 골든크로스/데드크로스 분류
    signals['golden_crosses'] = [s for s in signals['rsi3_rsi14_crosses'] + signals['rsi14_signal_crosses'] if s['type'] == 'golden']
    signals['dead_crosses'] = [s for s in signals['rsi3_rsi14_crosses'] + signals['rsi14_signal_crosses'] if s['type'] == 'dead']
    
    return signals

def plot_rsi_cross_signals(ax_rsi, rsi3, rsi14, rsi_signal, cross_signals):
    """
    RSI 크로스 신호를 차트에 표시합니다.
    
    Args:
        ax_rsi: RSI 서브플롯
        rsi3: RSI(3) 시리즈
        rsi14: RSI(14) 시리즈
        rsi_signal: RSI Signal 시리즈
        cross_signals: 크로스 신호 딕셔너리
    """
    # RSI 라인 그리기 (기존 방식 유지)
    ax_rsi.plot(rsi3.index, rsi3, color='red', linewidth=0.5, alpha=0.8, label='RSI(3)')
    ax_rsi.plot(rsi14.index, rsi14, color='blue', linewidth=0.5, alpha=0.8, label='RSI(14)')
    ax_rsi.plot(rsi_signal.index, rsi_signal, color='orange', linewidth=0.5, alpha=0.8, label='RSI Signal')
    
    # RSI 구간별 배경 색상 구분
    ax_rsi.axhspan(0, 30, alpha=0.1, color='green', label='과매도 구간')
    ax_rsi.axhspan(70, 100, alpha=0.1, color='red', label='과매수 구간')
    
    # 레벨 라인 그리기
    ax_rsi.axhline(y=70, color='red', linestyle='--', alpha=0.7, linewidth=0.5)
    ax_rsi.axhline(y=30, color='green', linestyle='--', alpha=0.7, linewidth=0.5)
    ax_rsi.axhline(y=50, color='gray', linestyle='-', alpha=0.5, linewidth=0.3)
    
    # 골든크로스 표시 (신호 강도별 구분)
    for signal in cross_signals['golden_crosses']:
        # 신호 강도에 따른 색상과 크기 설정
        if signal['level'] == 'oversold':  # 강한 신호 (과매도 구간)
            color = 'darkgreen'
            size = 30
            marker = '^'
        elif signal['level'] == 'neutral':  # 중간 신호 (중립 구간)
            color = 'green'
            size = 20
            marker = '^'
        else:  # 약한 신호 (과매수 구간)
            color = 'lightgreen'
            size = 15
            marker = '^'
        
        if 'rsi3' in signal:  # RSI(3) vs RSI(14) 크로스
            ax_rsi.scatter(signal['date'], signal['rsi14'], color=color, s=size, marker=marker, alpha=0.8, zorder=5)
        else:  # RSI(14) vs Signal 크로스
            ax_rsi.scatter(signal['date'], signal['rsi14'], color=color, s=size+5, marker=marker, alpha=0.9, zorder=5)
    
    # 데드크로스 표시 (신호 강도별 구분)
    for signal in cross_signals['dead_crosses']:
        # 신호 강도에 따른 색상과 크기 설정
        if signal['level'] == 'overbought':  # 강한 신호 (과매수 구간)
            color = 'darkred'
            size = 30
            marker = 'v'
        elif signal['level'] == 'neutral':  # 중간 신호 (중립 구간)
            color = 'red'
            size = 20
            marker = 'v'
        else:  # 약한 신호 (과매도 구간)
            color = 'lightcoral'
            size = 15
            marker = 'v'
        
        if 'rsi3' in signal:  # RSI(3) vs RSI(14) 크로스
            ax_rsi.scatter(signal['date'], signal['rsi14'], color=color, s=size, marker=marker, alpha=0.8, zorder=5)
        else:  # RSI(14) vs Signal 크로스
            ax_rsi.scatter(signal['date'], signal['rsi14'], color=color, s=size+5, marker=marker, alpha=0.9, zorder=5)

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
    ax_rsi.plot(data.index, rsi, color='blue', linewidth=0.35, label='RSI(14)')
    ax_rsi.axhline(y=70, color='red', linestyle='--', linewidth=0.105, alpha=0.5)
    ax_rsi.axhline(y=30, color='green', linestyle='--', linewidth=0.105, alpha=0.5)
    ax_rsi.axhline(y=50, color='gray', linestyle='-', linewidth=0.105, alpha=0.3)
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
    
    # 레이아웃 조정 (우측 여백을 늘려서 외부 라벨들이 표시되도록)
    plt.subplots_adjust(left=0.08, right=0.85, top=0.95, bottom=0.06, hspace=0.15)
    
    # 저장 또는 표시
    if save_path:
        plt.savefig(save_path, dpi=300)  # bbox_inches='tight' 제거
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
        if vix_value < 15:
            vix_sentiment = "VIX:[G](위험선호)"
        elif vix_value < 25:
            vix_sentiment = "VIX:[O](선별대응)"
        elif vix_value < 30:
            vix_sentiment = "VIX:[R](위험축소)"
        else:
            vix_sentiment = "VIX:[R](자본보전)"
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
        if vix_value < 15:
            vix_strategy = "VIX<15: 위험선호(Risk-On)"
        elif vix_value < 25:
            vix_strategy = "VIX 15-25: 선별 대응(Caution)"
        elif vix_value < 30:
            vix_strategy = "VIX 25-30: 위험축소(Risk-Off)"
        else:
            vix_strategy = "VIX≥30: 자본보전(위기 경계)"
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

def _finite_close_series(series):
    """yfinance Close(Series/DataFrame)를 유한 값 Series로 정리합니다."""
    if series is None:
        return None
    try:
        if isinstance(series, pd.DataFrame):
            if series.empty:
                return None
            series = series.iloc[:, 0]
        values = pd.to_numeric(series, errors="coerce").dropna()
        values = values[np.isfinite(values.to_numpy(dtype=float))]
        if values.empty:
            return None
        return values
    except Exception:
        return None


def _rate_regime_lines(ffr_series, tnx_series, inflation_series):
    """FFR·TNX·CPI로 금리 상태와 시사점. FFR 시계열은 차트의 단기금리."""
    ffr = _finite_close_series(ffr_series)
    tnx = _finite_close_series(tnx_series)
    cpi = _finite_close_series(inflation_series)
    if ffr is None or tnx is None or cpi is None:
        return "금리: 데이터 부족"

    short = float(ffr.iloc[-1])
    long = float(tnx.iloc[-1])
    infl = float(cpi.iloc[-1])
    gap = short - infl
    curve = long - short

    if gap >= 0.50:
        stance = "단기금리가 물가보다 높음"
    elif gap >= -0.25:
        stance = "단기금리 ≈ 물가"
    else:
        stance = "단기금리가 물가보다 낮음"

    if curve >= 0.50:
        curve_txt = f"10년이 단기보다 {curve:.2f}%p 위"
    elif curve > -0.10:
        curve_txt = "장단기 금리 비슷"
    else:
        curve_txt = f"장단기 역전 {curve:.2f}%p"

    if infl >= 3.5 and gap < 0.50 and curve >= 0.50:
        implication = "완화 장세 아님 — 10년 금리가 성장주 할인율 부담"
    elif infl >= 3.5 and gap < 0.50:
        implication = "물가는 높은데 실질 정책금리는 타이트하지 않음"
    elif gap >= 0.50 and infl < 3.0:
        implication = "실질금리 플러스 — 긴축이 물가보다 위"
    elif curve < -0.10:
        implication = "장단기 역전 — 경기 둔화 경계"
    elif curve >= 0.50:
        implication = "10년 금리가 할인율 부담"
    else:
        implication = "금리·물가 간격으로 완화 장세를 확인하지 않음"

    return "\n".join([
        f"금리: FFR {short:.2f}% · TNX {long:.2f}% · CPI {infl:.2f}%",
        f"금리 상태: {stance} · {curve_txt}",
        f"금리 시사: {implication}",
    ])


def _market_regime_text(hy_series, vix_series, tnx_series, ffr_series=None, inflation_series=None):
    """HY·VIX·10년물로 현재 국면 문구를 만듭니다.

    Risk-Off 확정은 HY ≥ 3.50%(350bp)이고 VIX ≥ 25일 때만.
    HY ≥ 3.30%이거나 직전 대비 10bp 이상 확대면 Caution 후반(잠정).
    10년물 하락은 주식 Risk-On으로 해석하지 않습니다.
    """
    hy_values = _finite_close_series(hy_series)
    vix_values = _finite_close_series(vix_series)
    tnx_values = _finite_close_series(tnx_series)

    hy = float(hy_values.iloc[-1]) if hy_values is not None else None
    hy_prev = float(hy_values.iloc[-2]) if hy_values is not None and len(hy_values) >= 2 else None
    vix = float(vix_values.iloc[-1]) if vix_values is not None else None

    if hy is None and vix is None:
        return "현재 국면: 데이터 부족"

    hy_confirm = hy is not None and hy >= 3.50
    vix_confirm = vix is not None and vix >= 25.0
    delta_bp = None
    if hy is not None and hy_prev is not None:
        delta_bp = int(round((hy - hy_prev) * 100))
    late_caution = (hy is not None and hy >= 3.30) or (delta_bp is not None and delta_bp >= 10)

    if hy_confirm and vix_confirm:
        title = "Risk-Off 확정"
    elif late_caution:
        title = "Caution 후반 — Risk-Off 경계 (잠정)"
    elif (hy is not None and hy >= 3.00) or (vix is not None and vix >= 20.0):
        title = "Caution"
    elif hy is not None and vix is not None:
        title = "Risk-On"
    else:
        title = "국면 판단 보류"

    lines = [f"현재 국면: {title}"]

    if hy is None:
        lines.append("HY: 데이터 없음")
    else:
        hy_bp = int(round(hy * 100))
        if delta_bp is None:
            delta_txt = ""
        else:
            delta_txt = f", 직전 대비 {delta_bp:+d}bp"
        gate = "350bp 충족" if hy_confirm else "350bp 미달"
        lines.append(f"HY {hy:.2f}% ({hy_bp}bp{delta_txt}) · {gate}")

    if vix is None:
        lines.append("VIX: 데이터 없음 → Risk-Off 확정 보류")
    elif hy_confirm and vix_confirm:
        lines.append(f"VIX {vix:.2f} · 25 충족 → Risk-Off 확정")
    elif vix_confirm:
        lines.append(f"VIX {vix:.2f} · 25 충족 · HY 350bp 미달 → Risk-Off 확정 보류")
    else:
        lines.append(f"VIX {vix:.2f} · 25 미달 → Risk-Off 확정 보류")

    if tnx_values is not None and len(tnx_values) >= 2:
        lookback = min(5, len(tnx_values) - 1)
        tnx_now = float(tnx_values.iloc[-1])
        tnx_then = float(tnx_values.iloc[-1 - lookback])
        if tnx_now <= tnx_then - 0.05:
            lines.append("금리 하락은 Risk-On으로 보지 않음")

    lines.append(_rate_regime_lines(ffr_series, tnx_series, inflation_series))
    return "\n".join(lines)

def _naive_close_index(series):
    """종가 인덱스를 timezone 없는 날짜로 맞춥니다."""
    series = series.copy()
    idx = pd.to_datetime(series.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    series.index = idx
    return series[~series.index.duplicated(keep="last")].sort_index()


def _download_close(symbol, start, end):
    """종가 시계열. yfinance MultiIndex를 한 열로 접습니다."""
    try:
        import yfinance as yf

        raw = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
        if raw is None or getattr(raw, "empty", True):
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.copy()
            raw.columns = raw.columns.get_level_values(0)
        close = raw["Close"] if "Close" in getattr(raw, "columns", []) else raw.iloc[:, 0]
        series = _finite_close_series(close)
        if series is None:
            return None
        return _naive_close_index(series)
    except Exception as e:
        print(f"⚠️ {symbol} 종가 다운로드 실패: {e}")
        return None


def _ols_beta(stock_close, market_close, max_obs=252, min_obs=60):
    """일간 수익률 공분산으로 시장 대비 베타. (beta, 관측 수) 또는 None."""
    stock = _finite_close_series(stock_close)
    market = _finite_close_series(market_close)
    if stock is None or market is None:
        return None
    stock = _naive_close_index(stock)
    market = _naive_close_index(market)
    joined = pd.concat(
        [stock.rename("s"), market.rename("m")], axis=1, join="inner"
    ).dropna()
    if len(joined) < min_obs + 1:
        return None
    rets = joined.pct_change().dropna().tail(max_obs)
    if len(rets) < min_obs:
        return None
    var_m = float(rets["m"].var(ddof=1))
    if not np.isfinite(var_m) or var_m <= 0:
        return None
    beta = float(rets["s"].cov(rets["m"])) / var_m
    if not np.isfinite(beta):
        return None
    return beta, int(len(rets))


def _beta_strategy_line(beta, regime_text):
    """베타 구간과 현재 국면에 맞는 한 줄 전략.

    Caution 후반(잠정)과 일반 Caution을 구분합니다.
    금리 하락 주석이 있으면 Risk-On 참여 문구를 쓰지 않습니다.
    """
    if beta <= 0:
        band = "역베타"
    elif beta < 0.8:
        band = "저베타"
    elif beta > 1.2:
        band = "고베타"
    else:
        band = "시장동행"

    text = regime_text or ""
    title = text.split("\n", 1)[0]
    if "Risk-Off 확정" in title:
        bucket = "off"
    elif "Caution 후반" in title or ("경계" in title and "잠정" in title):
        bucket = "late"
    elif "Risk-On" in title:
        bucket = "caution" if "금리 하락은 Risk-On으로 보지 않음" in text else "on"
    elif "Caution" in title:
        bucket = "caution"
    else:
        bucket = "unknown"

    strategy = {
        ("고베타", "off"): "비중 축소 — 하락이 시장보다 큼",
        ("고베타", "late"): "신규 매수 중단 — 확정 전 축소는 보류, 비중 동결",
        ("고베타", "caution"): "추격 금지 — 비중 유지",
        ("고베타", "on"): "참여 — 상승이 시장보다 큼",
        ("고베타", "unknown"): "고베타 — 시장보다 크게 움직임",
        ("시장동행", "off"): "지수와 함께 비중 축소",
        ("시장동행", "late"): "신규 매수 중단, 비중 동결",
        ("시장동행", "caution"): "지수와 같이 보유",
        ("시장동행", "on"): "지수와 같이 운용",
        ("시장동행", "unknown"): "시장 동행 — 지수와 비슷하게 움직임",
        ("저베타", "off"): "방어 우선 — 낙폭이 시장보다 작음",
        ("저베타", "late"): "방어 보유 — 고베타 대체 후보",
        ("저베타", "caution"): "보유 — 시장보다 완만",
        ("저베타", "on"): "탄력 제한 — 시장보다 덜 오름",
        ("저베타", "unknown"): "저베타 — 시장보다 작게 움직임",
        ("역베타", "off"): "헤지 유지",
        ("역베타", "late"): "헤지 확대 보류 — 확정 전 추가 금지",
        ("역베타", "caution"): "소폭 헤지만",
        ("역베타", "on"): "축소 — 상승장에서 역행",
        ("역베타", "unknown"): "역베타 — 시장과 반대로 움직임",
    }[(band, bucket)]
    return band, strategy


def _beta_info_text(symbol, ohlcv_data, regime_text):
    """최근 1년(최대 252거래일) 베타와 전략 문구. 벤치마크는 SPY."""
    if not symbol or ohlcv_data is None or getattr(ohlcv_data, "empty", True):
        return "베타: 데이터 부족"
    end = pd.Timestamp(ohlcv_data.index[-1])
    if getattr(end, "tz", None) is not None:
        end = end.tz_localize(None)
    start = (end - pd.Timedelta(days=400)).date()
    end_fetch = (end + pd.Timedelta(days=1)).date()
    stock = _download_close(symbol, start, end_fetch)
    spy = _download_close("SPY", start, end_fetch)
    if stock is None and "Close" in ohlcv_data.columns:
        stock = ohlcv_data["Close"]
    result = _ols_beta(stock, spy) if spy is not None else None
    if result is None:
        print(f"⚠️ {symbol} 베타 계산 실패")
        return "베타: 데이터 부족"
    beta, n_obs = result
    window = "1년" if n_obs >= 200 else f"{n_obs}일"
    band, strategy = _beta_strategy_line(beta, regime_text)
    print(f"베타({symbol} vs SPY): {beta:.2f} ({band}, {window}, {n_obs}일) / {strategy}")
    return f"베타: {beta:.2f} ({band}, {window} vs SPY)\n베타 전략: {strategy}"


def _align_series_to_dates(series, dates):
    """날짜 인덱스를 차트 날짜에 맞춥니다. 없는 날은 NaN."""
    if series is None or dates is None or len(dates) == 0:
        return None
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    if getattr(idx, "tz", None) is not None:
        keys = idx.tz_localize(None).normalize()
    else:
        keys = idx.normalize()
    src = series.copy()
    if getattr(src.index, "tz", None) is not None:
        src.index = src.index.tz_localize(None)
    src.index = pd.DatetimeIndex(pd.to_datetime(src.index)).normalize()
    src = src[~src.index.duplicated(keep="last")].sort_index()
    aligned = src.reindex(keys)
    return pd.Series(pd.to_numeric(aligned, errors="coerce").to_numpy(), index=dates)


def _cross_asset_regime_text(end_date, fear_greed_score=None, fear_greed_data=None):
    """WTI·금·BTC 20일 변화와 CNN F&G로 국면 보조 문구를 만듭니다."""
    end = pd.Timestamp(end_date)
    if getattr(end, "tz", None) is not None:
        end = end.tz_localize(None)
    start = (end - pd.Timedelta(days=90)).date()
    end_fetch = (end + pd.Timedelta(days=1)).date()
    symbols = {"WTI": "CL=F", "GOLD": "GC=F", "BTC": "BTC-USD"}
    changes = {}
    for name, sym in symbols.items():
        series = _download_close(sym, start, end_fetch)
        if series is None or series.empty:
            continue
        series = series[series.index <= end]
        if len(series) < 21:
            continue
        now = float(series.iloc[-1])
        prev = float(series.iloc[-21])
        if not np.isfinite(now) or not np.isfinite(prev) or prev == 0:
            continue
        changes[name] = (now / prev - 1.0) * 100.0

    lines = []
    if fear_greed_score is not None and np.isfinite(float(fear_greed_score)):
        rating = ""
        if fear_greed_data is not None:
            rating = fear_greed_data.get("rating_ko") or ""
        rating_txt = f" ({rating})" if rating else ""
        lines.append(f"F&G: {float(fear_greed_score):.1f}{rating_txt}")

    if len(changes) < 3:
        lines.append("교차자산: WTI·GOLD·BTC 데이터 부족")
        return "\n".join(lines)

    wti, gold, btc = changes["WTI"], changes["GOLD"], changes["BTC"]
    lines.append(
        f"WTI 20일 {wti:+.1f}% · GOLD 20일 {gold:+.1f}% · BTC 20일 {btc:+.1f}%"
    )
    gold_up = gold >= 2.0
    gold_down = gold <= -2.0
    btc_up = btc >= 2.0
    btc_down = btc <= -2.0
    oil_up = wti >= 3.0
    if gold_up and btc_down:
        read = "교차자산: 금 상승·BTC 하락 — 안전자산 선호"
    elif btc_up and gold_down:
        read = "교차자산: BTC 상승·금 하락 — 위험자산 선호"
    elif oil_up and gold_up:
        read = "교차자산: 유가·금 동반 상승 — 공급 충격·안전자산 수요"
    elif oil_up and not gold_up:
        read = "교차자산: 유가 상승은 공급 위험 — 금이 동반 상승하지 않아 안전자산 도피로 확정하지 않음"
    else:
        read = "교차자산: 방향 혼재 — 국면 확정에 쓰지 않음"
    lines.append(read)
    return "\n".join(lines)


def _regime_synthesis(regime_text, beta_text, cross_text):
    """국면·금리·베타·교차자산 시사점을 현재국면 종합으로 모읍니다."""
    text = regime_text or ""
    beta = beta_text or ""
    cross = cross_text or ""
    title = text.split("\n", 1)[0]

    if "Risk-Off 확정" in title:
        bucket = "off"
    elif "Caution 후반" in title or ("경계" in title and "잠정" in title):
        bucket = "late"
    elif "Risk-On" in title:
        rates_block_on = (
            "금리 하락은 Risk-On으로 보지 않음" in text
            or "완화 장세 아님" in text
            or "할인율 부담" in text
        )
        bucket = "caution" if rates_block_on else "on"
    elif "Caution" in title:
        bucket = "caution"
    else:
        bucket = "unknown"

    if "베타 전략:" in beta:
        action = beta.split("베타 전략:", 1)[1].strip().split("\n", 1)[0]
    else:
        action = {
            "off": "비중 축소",
            "late": "신규 매수 중단, 비중 동결",
            "caution": "추격 금지, 비중 유지",
            "on": "지수와 같이 참여",
            "unknown": "관망",
        }[bucket]

    reasons = []
    if bucket == "off":
        reasons.append("Risk-Off 확정")
    elif bucket == "late":
        reasons.append("Risk-Off 미확정")
    if "완화 장세 아님" in text or "할인율 부담" in text:
        reasons.append("금리는 완화 장세 아님")
    elif "금리 하락은 Risk-On으로 보지 않음" in text:
        reasons.append("단기 금리 하락은 Risk-On 아님")
    elif "장단기 역전" in text:
        reasons.append("장단기 역전")
    elif "실질금리 플러스" in text:
        reasons.append("실질금리 플러스")
    if "극단적 공포" in cross and bucket in ("off", "late", "caution"):
        reasons.append("극공포는 시장 분할 검토")
    if "공급 위험" in cross:
        reasons.append("유가는 공급 위험")
    elif "안전자산 선호" in cross:
        reasons.append("안전자산 선호")
    elif "위험자산 선호" in cross and bucket == "on":
        reasons.append("교차자산은 위험자산 선호")

    line = f"현재국면 종합: {action}"
    if reasons:
        line += "\n시사: " + " · ".join(reasons[:4])
    return line


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
    
    try:
        # VIX 지수
        vix = yf.download('^VIX', start=start_date, end=today)['Close']
        if vix.empty:
            print("⚠️  VIX 데이터가 비어있습니다. 기본값 사용")
            vix = pd.Series([20.0], index=[pd.Timestamp(start_date)])
        else:
            print(f"VIX 마지막 데이터 날짜: {vix.index[-1].strftime('%Y-%m-%d')}")
            print(f"오늘 날짜: {today.strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"⚠️  VIX 데이터 다운로드 오류: {e}, 기본값 사용")
        vix = pd.Series([20.0], index=[pd.Timestamp(start_date)])
    
    try:
        # 미국채 10년물 금리
        tnx = yf.download('^TNX', start=start_date, end=today)['Close']
        if tnx.empty:
            print("⚠️  TNX 데이터가 비어있습니다. 기본값 사용")
            tnx = pd.Series([4.0], index=[pd.Timestamp(start_date)])
        else:
            print(f"TNX 마지막 데이터 날짜: {tnx.index[-1].strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"⚠️  TNX 데이터 다운로드 오류: {e}, 기본값 사용")
        tnx = pd.Series([4.0], index=[pd.Timestamp(start_date)])
    
    try:
        # 달러 인덱스
        dxy = yf.download('DX-Y.NYB', start=start_date, end=today)['Close']
        if dxy.empty:
            print("⚠️  DXY 데이터가 비어있습니다. 기본값 사용")
            dxy = pd.Series([100.0], index=[pd.Timestamp(start_date)])
        else:
            print(f"DXY 마지막 데이터 날짜: {dxy.index[-1].strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"⚠️  DXY 데이터 다운로드 오류: {e}, 기본값 사용")
        dxy = pd.Series([100.0], index=[pd.Timestamp(start_date)])
    
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

def calculate_period_return(ohlcv_data, start_date, end_date):
    """검색기간의 주가 상승률과 금액을 계산합니다."""
    try:
        # 데이터 유효성 검사
        if ohlcv_data is None or ohlcv_data.empty:
            print("오류: OHLCV 데이터가 비어있습니다.")
            return None

        if "Close" not in ohlcv_data.columns:
            print("오류: Close 컬럼을 찾을 수 없습니다.")
            return None

        close = ohlcv_data["Close"]
        # yfinance MultiIndex 잔여 → Series 1열
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = pd.to_numeric(close, errors="coerce").dropna()
        if len(close) < 2:
            print("오류: 유효 종가 데이터가 부족합니다.")
            return None

        # 요청 구간 안(가능하면)에서 첫·마지막 유효 종가 사용
        seg = close
        try:
            if start_date is not None:
                seg = seg[seg.index >= pd.Timestamp(start_date)]
            if end_date is not None:
                seg = seg[seg.index <= pd.Timestamp(end_date)]
        except Exception:
            seg = close
        if len(seg) < 2:
            seg = close

        start_ts = seg.index[0]
        end_ts = seg.index[-1]
        start_price = float(np.asarray(seg.iloc[0]).ravel()[0])
        end_price = float(np.asarray(seg.iloc[-1]).ravel()[0])

        if not np.isfinite(start_price) or not np.isfinite(end_price):
            print("오류: 시작가/종료가가 유효하지 않습니다.")
            return None
        if start_price == 0:
            print("오류: 시작가가 0입니다.")
            return None

        # 상승률 계산
        return_percentage = ((end_price - start_price) / start_price) * 100

        # 상승 금액 계산
        return_amount = end_price - start_price

        # 방향 결정 (폰트 호환성을 위해 텍스트 사용)
        if return_amount > 0:
            direction = "▲ 상승"
            color = "green"
        elif return_amount < 0:
            direction = "▼ 하락"
            color = "red"
        else:
            direction = "→ 보합"
            color = "gray"

        return {
            "start_price": start_price,
            "end_price": end_price,
            "return_percentage": float(return_percentage),
            "return_amount": float(return_amount),
            "direction": direction,
            "color": color,
            "start_date": pd.Timestamp(start_ts).strftime("%Y-%m-%d"),
            "end_date": pd.Timestamp(end_ts).strftime("%Y-%m-%d"),
        }
    except Exception as e:
        print(f"기간 수익률 계산 오류: {e}")
        return None

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

def get_dividend_dates(symbol, start_date, end_date, ohlcv_data=None):
    """종목의 배당일 정보와 배당률을 가져옵니다."""
    try:
        import yfinance as yf
        from datetime import datetime
        
        ticker = yf.Ticker(symbol)
        
        # 통화 기호 설정
        currency_symbol = get_currency_symbol(symbol)
        
        # 배당 정보 가져오기
        dividends = ticker.dividends
        
        if dividends.empty:
            print(f"⚠️  {symbol} 배당 정보 없음")
            return []
        
        # 날짜 범위 내의 배당일 필터링
        start_dt = pd.to_datetime(start_date).tz_localize(None)
        end_dt = pd.to_datetime(end_date).tz_localize(None)
        
        dividend_dates = []
        for date, amount in dividends.items():
            # 날짜를 timezone-naive로 변환
            if hasattr(date, 'tz') and date.tz is not None:
                date_naive = date.tz_localize(None)
            else:
                date_naive = date
            
            if start_dt <= date_naive <= end_dt:
                # 배당률 계산 (OHLCV 데이터가 있는 경우)
                dividend_yield = None
                if ohlcv_data is not None and date_naive in ohlcv_data.index:
                    # 배당일의 주가 가져오기
                    stock_price = ohlcv_data.loc[date_naive, 'Close']
                    # 배당률 계산 (연환산)
                    dividend_yield = (amount / stock_price) * 100
                
                dividend_dates.append({
                    'date': date_naive,
                    'amount': amount,
                    'yield': dividend_yield
                })
        
        if dividend_dates:
            print(f"✅ {symbol} 배당일 {len(dividend_dates)}개 발견")
            for div in dividend_dates:
                if div['yield'] is not None:
                    print(f"   {div['date'].strftime('%Y-%m-%d')}: {currency_symbol}{div['amount']:.4f} (배당률: {div['yield']:.2f}%)")
                else:
                    print(f"   {div['date'].strftime('%Y-%m-%d')}: {currency_symbol}{div['amount']:.4f}")
        else:
            print(f"⚠️  {symbol} 지정 기간 내 배당일 없음")
        
        return dividend_dates
        
    except Exception as e:
        print(f"배당 정보 가져오기 실패: {e}")
        return []

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

def calculate_daily_change_percentage(symbol, ohlcv_data=None):
    """전일대비 등락률을 계산합니다. (종가 NaN 봉 제외)"""
    try:
        close = None
        if ohlcv_data is not None and not getattr(ohlcv_data, "empty", True) and "Close" in ohlcv_data.columns:
            close = ohlcv_data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = pd.to_numeric(close, errors="coerce").dropna()

        if close is None or len(close) < 2:
            import yfinance as yf

            # 미완성 봉 대비 여유 있게 조회
            hist_data = yf.Ticker(symbol).history(period="5d")
            if hist_data is None or hist_data.empty or "Close" not in hist_data.columns:
                return None
            close = pd.to_numeric(hist_data["Close"], errors="coerce").dropna()

        if len(close) < 2:
            return None

        prev_close = float(np.asarray(close.iloc[-2]).ravel()[0])
        current_price = float(np.asarray(close.iloc[-1]).ravel()[0])
        if not np.isfinite(prev_close) or not np.isfinite(current_price) or prev_close == 0:
            return None

        change_amount = current_price - prev_close
        change_percentage = (change_amount / prev_close) * 100

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
            "prev_close": prev_close,
            "current_price": current_price,
            "change_amount": float(change_amount),
            "change_percentage": float(change_percentage),
            "direction": direction,
            "color": color,
        }

    except Exception as e:
        print(f"전일대비 등락률 계산 실패: {e}")
        return None

def get_currency_symbol(ticker):
    """종목 코드에 따라 통화 기호를 반환합니다."""
    if ticker and '.KS' in ticker:
        return '₩'  # 한국 원
    else:
        return '$'  # 미국 달러

def get_ffr_data(start_date, end_date):
    """연방기금금리(FFR) 데이터를 가져옵니다."""
    try:
        import yfinance as yf
        # FFR 데이터는 FRED에서 가져와야 하지만, yfinance로 대체 지표 사용
        # 3개월 국채금리(^IRX)를 FFR 대체 지표로 사용
        ffr_data = yf.download('^IRX', start=start_date, end=end_date, progress=False)
        if not ffr_data.empty:
            return ffr_data['Close']
        return None
    except Exception as e:
        print(f"FFR 데이터 가져오기 실패: {e}")
        return None

def get_cdr_data(start_date, end_date):
    """CD(양도성예금증서) 금리 데이터를 가져옵니다."""
    try:
        import yfinance as yf
        # CD 금리는 일반적으로 3개월 CD 금리를 의미
        # yfinance에서는 ^IRX (3개월 국채금리)를 CD 금리 대체 지표로 사용
        cdr_data = yf.download('^IRX', start=start_date, end=end_date, progress=False)
        if not cdr_data.empty:
            return cdr_data['Close']
        return None
    except Exception as e:
        print(f"CD 금리 데이터 가져오기 실패: {e}")
        return None

def get_inflation_data(start_date, end_date):
    """CPI 전년비(%)를 가져옵니다. 임의 근사치는 반환하지 않습니다."""
    try:
        import pandas as pd

        fetch_start = pd.Timestamp(start_date) - pd.DateOffset(months=14)
        try:
            raw = web.DataReader("CPIAUCSL", "fred", start=fetch_start, end=end_date)
            series = raw.iloc[:, 0] if isinstance(raw, pd.DataFrame) else raw
            series = pd.to_numeric(series, errors="coerce").dropna()
            yoy = (series.pct_change(12) * 100).dropna()
            business_days = pd.bdate_range(start=start_date, end=end_date)
            daily = yoy.reindex(business_days, method="ffill").dropna()
            if not daily.empty:
                daily.name = "CPI_YOY"
                print(
                    f"✅ CPI 전년비 로드: {len(daily)}개 "
                    f"({pd.Timestamp(daily.index[0]).date()}~{pd.Timestamp(daily.index[-1]).date()}, "
                    f"최근 {float(daily.iloc[-1]):.2f}%)"
                )
                return daily
        except Exception as e:
            print(f"⚠️ CPI 전년비 로드 실패: {e}")

        # fredapi import 시도
        try:
            from fredapi import Fred
            fred_available = True
        except ImportError:
            print("⚠️ fredapi 패키지 없음, pip install fredapi 필요")
            fred_available = False
        
        if fred_available:
            # FRED API 키 (무료 키 사용)
            # 실제 사용 시에는 https://fred.stlouisfed.org/docs/api/api_key.html 에서 키 발급
            fred = Fred(api_key='demo')  # 데모 키 사용
        
            # Core PCE 인플레이션 데이터 가져오기 (월간)
            try:
                # Core PCE Price Index (PCEPILFE) - 연간 변화율
                core_pce = fred.get_series('PCEPILFE', start_date, end_date)
                if not core_pce.empty:
                    print(f"✅ FRED Core PCE 데이터 로드 성공: {len(core_pce)}개")
                    
                    # 월간 데이터를 거래일 기준으로 재색인하고 전방 채움
                    business_days = pd.bdate_range(start=start_date, end=end_date)
                    core_pce_daily = core_pce.reindex(business_days, method='ffill')
                    
                    print(f"✅ 인플레이션 데이터 거래일 정렬 완료: {len(core_pce_daily)}개")
                    return core_pce_daily
            except Exception as e:
                print(f"⚠️ FRED Core PCE 로드 실패: {e}")
            
            # 백업: CPI 데이터 시도
            try:
                cpi = fred.get_series('CPIAUCSL', start_date, end_date)
                if not cpi.empty:
                    print(f"✅ FRED CPI 데이터 로드 성공: {len(cpi)}개")
                    
                    # CPI를 연간 변화율로 변환
                    cpi_yoy = cpi.pct_change(12) * 100
                    
                    # 월간 데이터를 거래일 기준으로 재색인하고 전방 채움
                    business_days = pd.bdate_range(start=start_date, end=end_date)
                    cpi_daily = cpi_yoy.reindex(business_days, method='ffill')
                    
                    print(f"✅ CPI 연간 변화율 거래일 정렬 완료: {len(cpi_daily)}개")
                    return cpi_daily
            except Exception as e:
                print(f"⚠️ FRED CPI 로드 실패: {e}")
        
        print("⚠️ 인플레이션 데이터 없음")
        return None
        
    except Exception as e:
        print(f"인플레이션 데이터 가져오기 실패: {e}")
        return None

def calculate_real_interest_rate(nominal_rate, inflation_rate):
    """실질금리를 계산합니다 (인덱스 정렬 포함)."""
    try:
        if nominal_rate is None or inflation_rate is None:
            print("⚠️ 실질금리 계산: 명목금리 또는 인플레이션 데이터 없음")
            return None
        
        # 명목금리가 DataFrame인 경우 Close 컬럼을 Series로 변환
        if hasattr(nominal_rate, 'columns'):
            if 'Close' in nominal_rate.columns:
                nominal_rate = nominal_rate['Close']
            else:
                # Close 컬럼이 없으면 첫 번째 컬럼 사용
                nominal_rate = nominal_rate.iloc[:, 0]
        
        # 인덱스 정렬: 명목금리 기준으로 인플레이션 데이터 정렬
        inflation_aligned = inflation_rate.reindex(nominal_rate.index, method='ffill')
        
        # 실질금리 = 명목금리 - 인플레이션율
        real_rate = nominal_rate - inflation_aligned
        
        # NaN 값 제거
        real_rate_clean = real_rate.dropna()
        
        print(f"✅ 실질금리 계산 완료: {len(real_rate_clean)}개 데이터 (정렬 후)")
        if len(real_rate_clean) > 0:
            print(f"   실질금리 범위: {real_rate_clean.min():.2f}% ~ {real_rate_clean.max():.2f}%")
        
        return real_rate_clean
    except Exception as e:
        print(f"실질금리 계산 실패: {e}")
        return None

def get_tnx_data(start_date, end_date):
    """미국 10년물 국채금리(TNX) 데이터를 가져옵니다."""
    try:
        import yfinance as yf
        tnx_data = yf.download('^TNX', start=start_date, end=end_date, progress=False)
        if not tnx_data.empty:
            return tnx_data['Close']
        return None
    except Exception as e:
        print(f"TNX 데이터 가져오기 실패: {e}")
        return None


def get_hy_spread_data(start_date, end_date):
    """ICE BofA 미국 하이일드 OAS (FRED BAMLH0A0HYM2, %)."""
    try:
        import pandas_datareader.data as web

        raw = web.DataReader("BAMLH0A0HYM2", "fred", start=start_date, end=end_date)
        if raw is None or getattr(raw, "empty", True):
            print("⚠️ HY 스프레드 데이터가 비어 있습니다.")
            return None
        series = raw.iloc[:, 0] if isinstance(raw, pd.DataFrame) else raw
        series = pd.to_numeric(series, errors="coerce").dropna()
        if series.empty:
            print("⚠️ HY 스프레드 유효값이 없습니다.")
            return None
        series.name = "HY"
        print(
            f"✅ HY 스프레드 로드: {len(series)}개 "
            f"({pd.Timestamp(series.index[0]).date()}~{pd.Timestamp(series.index[-1]).date()}, "
            f"최근 {float(series.iloc[-1]):.2f}%)"
        )
        return series
    except Exception as e:
        print(f"⚠️ HY 스프레드 데이터 가져오기 실패: {e}")
        return None

def plot_main_chart_with_volume_profile_overlay(
    data: pd.DataFrame,
    ticker: str = None,
    save_path: str = None,
    current_price: float = None,
    rsi_divergence_window: int = 60,  # 20 → 60으로 확장 (약 3개월)
    rsi_pivot_span: int = 5,          # 3 → 5로 확장 (노이즈 감소)
    include_hidden_divergence: bool = True,
    current_price_line_style: str = 'thin',  # 'thin': 얇은 수직선, 'bottom_start': 하단에서 시작
    # Target 가격 옵션들 추가
    target_buy_price: float = None,    # Target 매수가
    target_sell_price: float = None,   # Target 목표가
    stop_loss_price: float = None,     # 손절가
    show_target_prices: bool = False,   # Target 가격 표시 여부
    # 박스권 옵션들 추가
    show_box_ranges: bool = True,      # 박스권 표시 여부
    box_period: int = 20,             # 박스권 계산 기간
    num_boxes: int = 2,               # 표시할 박스권 개수 (기본값: 2개로 유지)
    box_overlap: int = 5,             # 박스권 간 겹치는 일수
    box_style: str = 'default',       # 박스권 스타일 ('default', 'gradient', 'rainbow')
    avoid_time_overlap: bool = True,  # 시간축 겹침 방지 여부
    show_pattern_strip: bool = False,  # 메인 아래 패턴 타임라인 서브플롯(chart_patterns 행 수)
    pattern_main_overlays: Optional[Sequence[str]] = None,
    # 메인 차트에 그릴 패턴 id (쉼표 구분 문자열을 테스트 스크립트에서 리스트로 전달).
    # None/빈 리스트 → 메인에 패턴 도형 없음. 예: ("triple_bottom_w", "asc_triangle") 또는 ("all",)
    # range_box 패턴 후보만: show_box_ranges / box_period 와 무관한 전용 파라미터
    pattern_range_box_period: int = 20,
    pattern_range_box_num: int = 2,
    pattern_range_box_overlap: int = 5,
    pattern_range_box_avoid_time_overlap: bool = True,
    # 메인 차트 range_box: 후보 중 상위 N건만(신뢰도·종료일). 스트립은 전체 유지. 0이면 메인에 미표시.
    pattern_range_box_main_max: int = 1,
    # 메인 차트 패턴 오버레이 최소 신뢰도(0~1). 스트립에는 영향 없음.
    pattern_main_min_confidence: float = 0.0,
    # 이격도(20MA 기준) 전략 표시
    show_disparity_strategy: bool = False,
    disparity_ma: int = 20,
    disparity_low: float = 95.0,
    disparity_high: float = 105.0,
    disparity_conf_weight_depth: float = 0.22,
    disparity_conf_weight_volume: float = 0.16,
    disparity_conf_weight_trend: float = 0.10,
    show_adx_dmi: bool = False,
    adx_period: int = 14,
    adx_sideways: float = 20.0,
    adx_trend: float = 25.0,
    adx_strong: float = 40.0,
):
    """Volume Profile이 메인차트에 오버레이된 차트"""
    # 폰트 설정
    plt.rcParams['font.family'] = 'AppleGothic'  # macOS용 한글 폰트
    plt.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지
    
    # 통화 기호 설정
    currency_symbol = get_currency_symbol(ticker)
    
    if isinstance(data.columns, pd.MultiIndex):
        ohlcv_data = data.xs(ticker, axis=1, level=1)
    else:
        ohlcv_data = data

    # yfinance 미완성 봉(종가 NaN 등) 제거 — 기간수익률/SMA200 nan 방지
    need_cols = [c for c in ("Open", "High", "Low", "Close") if c in ohlcv_data.columns]
    if need_cols:
        before_n = len(ohlcv_data)
        ohlcv_data = ohlcv_data.dropna(subset=need_cols)
        dropped = before_n - len(ohlcv_data)
        if dropped > 0:
            print(f"⚠️ 미완성 OHLCV {dropped}봉 제외 (유효 {len(ohlcv_data)}봉)")
    if ohlcv_data.empty:
        print("오류: 유효 OHLCV 데이터가 없습니다.")
        return

    # 시장 데이터 가져오기
    start_date = ohlcv_data.index[0]
    end_date = ohlcv_data.index[-1]
    vix, tnx, dxy = get_market_data(start_date, end_date)
    
    # 추가 지표 데이터 가져오기
    ffr_data = get_ffr_data(start_date, end_date)
    tnx_data = get_tnx_data(start_date, end_date)
    hy_spread_data = get_hy_spread_data(start_date, end_date)
    
    inflation_data = get_inflation_data(start_date, end_date)
    
    # 메인 차트 파란 박스만 계산 (show_box_ranges). range_box 패턴과 완전히 별도.
    display_box_ranges: list = []
    if show_box_ranges:
        display_box_ranges = calculate_box_ranges(
            ohlcv_data,
            box_period=box_period,
            num_boxes=num_boxes,
            overlap_days=box_overlap,
            avoid_time_overlap=avoid_time_overlap,
        )
        time_overlap_mode = "겹침방지" if avoid_time_overlap else f"겹침허용({box_overlap}일)"
        print(
            f"박스권(표시용): 기간={box_period}일, 개수={num_boxes}개, "
            f"시간축={time_overlap_mode}, 스타일={box_style}"
        )
    else:
        print("박스권 표시 비활성화 (메인 파란 박스 없음). range_box 패턴은 별도 계산.")
    
    # VIX 데이터 길이 맞추기 (실제 데이터 우선, 동적 생성은 최후의 수단)
    if not vix.empty and len(vix) < len(ohlcv_data):
        print(f"VIX 데이터 길이 맞추기: {len(vix)} → {len(ohlcv_data)}")
        print(f"실제 VIX 데이터 범위: {vix.index.min()} ~ {vix.index.max()}")
        vix_min_val = float(vix.min())
        vix_max_val = float(vix.max())
        print(f"실제 VIX 값 범위: {vix_min_val:.2f} ~ {vix_max_val:.2f}")
        
        # 날짜 형식 통일 (시간대 정보 제거)
        vix_normalized = vix.copy()
        vix_normalized.index = vix_normalized.index.normalize()
        ohlcv_normalized = ohlcv_data.copy()
        ohlcv_normalized.index = ohlcv_normalized.index.normalize()
        
        print(f"날짜 정규화 후 VIX 인덱스 샘플: {vix_normalized.index[:5]}")
        print(f"날짜 정규화 후 OHLCV 인덱스 샘플: {ohlcv_normalized.index[:5]}")
        
        # VIX 데이터를 ohlcv_data 기간에 맞춰 재샘플링 (실제 데이터 우선)
        vix_extended = pd.Series(index=ohlcv_normalized.index, dtype=float)
        
        # 실제 VIX 데이터가 있는 구간과 없는 구간을 구분
        actual_data_count = 0
        interpolated_count = 0
        dynamic_count = 0
        
        for i, date in enumerate(ohlcv_normalized.index):
            try:
                # 1. 정확한 날짜 매칭 시도 (가장 우선)
                if date in vix_normalized.index:
                    vix_extended[date] = vix_normalized[date]
                    actual_data_count += 1
                else:
                    # 2. 가장 가까운 이전 VIX 값 사용 (보간)
                    prev_dates = vix_normalized.index[vix_normalized.index <= date]
                    if len(prev_dates) > 0:
                        prev_date = prev_dates[-1]
                        vix_extended[date] = vix_normalized[prev_date]
                        interpolated_count += 1
                    else:
                        # 3. 이전 값이 없으면 다음 값 사용 (보간)
                        next_dates = vix_normalized.index[vix_normalized.index >= date]
                        if len(next_dates) > 0:
                            next_date = next_dates[0]
                            vix_extended[date] = vix_normalized[next_date]
                            interpolated_count += 1
                        else:
                            # 4. 모든 방법이 실패했을 때만 동적 값 생성 (최후의 수단)
                            # 실제 VIX 데이터의 최근 값을 기준으로 최소한의 변동만 추가
                            date_position = i / len(ohlcv_normalized.index)
                            base_vix = vix_normalized.iloc[-1] if not vix_normalized.empty else 20.0
                            
                            # 실제 데이터 기반의 최소한의 변동성만 추가
                            minimal_variation = np.random.normal(0, 0.05)  # 5% 이내 변동
                            dynamic_vix = base_vix * (1 + minimal_variation)
                            dynamic_vix = np.clip(dynamic_vix, base_vix * 0.9, base_vix * 1.1)  # ±10% 범위
                            vix_extended[date] = dynamic_vix
                            dynamic_count += 1
            except Exception as date_error:
                print(f"VIX 날짜 {date} 처리 오류: {date_error}, 기본값 사용")
                # 오류 발생 시 기본값 사용 (동적 생성 대신)
                vix_extended[date] = vix_normalized.iloc[-1] if not vix_normalized.empty else 20.0
                dynamic_count += 1
        
        # 원본 인덱스로 복원
        vix_extended.index = ohlcv_data.index
        vix = vix_extended
        
        print(f"VIX 데이터 확장 완료: {len(vix)}개 데이터")
        print(f"  - 실제 데이터: {actual_data_count}개")
        print(f"  - 보간 데이터: {interpolated_count}개")
        print(f"  - 동적 생성: {dynamic_count}개")
        
        # VIX 데이터 품질 확인
        vix_min, vix_max = vix.min(), vix.max()
        vix_mean = vix.mean()
        print(f"VIX 데이터 품질: 최소={vix_min:.2f}, 최대={vix_max:.2f}, 평균={vix_mean:.2f}")
        
        # 실제 VIX 데이터와의 일치도 확인
        if actual_data_count > 0:
            original_vix_mean = vix_normalized.mean()
            accuracy = (1 - abs(vix_mean - original_vix_mean) / original_vix_mean) * 100
            print(f"실제 VIX 데이터와의 일치도: {accuracy:.1f}%")
    
    # NAIIM 데이터 가져오기 (실제 데이터 우선, 없으면 시뮬레이션)
    try:
        from pathlib import Path
        
        naaim_file = Path("data_cache/naaim_data.csv")
        if naaim_file.exists():
            naaim_df = pd.read_csv(naaim_file)
            naaim_df['Date'] = pd.to_datetime(naaim_df['Date'])
            naaim_df.set_index('Date', inplace=True)
            
            # NAIIM 데이터 범위 확인 및 로드
            print(f"NAIIM 파일 데이터 범위: {naaim_df.index.min()} ~ {naaim_df.index.max()}")
            print(f"요청 데이터 범위: {start_date} ~ {end_date}")
            
            # 안전한 데이터 로드 (인덱스 오류 방지)
            try:
                naaim = naaim_df.loc[start_date:end_date]
                print(f"실제 NAIIM 데이터 로드 완료: {len(naaim)}개 데이터")
                
                # NAIIM 데이터 길이 맞추기 (PCR과 동일한 기간으로 확장)
                if naaim is not None and len(naaim) < len(ohlcv_data):
                    print(f"NAIIM 데이터 길이 맞추기: {len(naaim)} → {len(ohlcv_data)}")
                    
                    # NAIIM 데이터를 ohlcv_data 기간에 맞춰 재샘플링 (개선된 방식)
                    naaim_extended = pd.Series(index=ohlcv_data.index, dtype=float)
                    
                    # NAIIM 데이터를 numpy 배열로 변환하여 안전하게 처리
                    naaim_values = naaim.values if hasattr(naaim, 'values') else naaim
                    naaim_dates = naaim.index
                    
                    for i, date in enumerate(ohlcv_data.index):
                        try:
                            # 1. 정확한 날짜 매칭 시도
                            if date in naaim_dates:
                                date_idx = naaim_dates.get_loc(date)
                                naaim_extended[date] = naaim_values[date_idx]
                            else:
                                # 2. 가장 가까운 이전 NAIIM 값 사용 (안전한 방식)
                                prev_dates = naaim_dates[naaim_dates <= date]
                                if len(prev_dates) > 0:
                                    prev_date = prev_dates[-1]
                                    prev_idx = naaim_dates.get_loc(prev_date)
                                    naaim_extended[date] = naaim_values[prev_idx]
                                else:
                                    # 3. 이전 값이 없으면 다음 값 사용 (안전한 방식)
                                    next_dates = naaim_dates[naaim_dates >= date]
                                    if len(next_dates) > 0:
                                        next_date = next_dates[0]
                                        next_idx = naaim_dates.get_loc(next_date)
                                        naaim_extended[date] = naaim_values[next_idx]
                                    else:
                                        # 4. 모든 방법이 실패하면 동적 값 생성
                                        # 날짜의 위치에 따른 동적 NAIIM 값 계산
                                        date_position = i / len(ohlcv_data.index)
                                        dynamic_value = 50.0 + 20.0 * np.sin(2 * np.pi * date_position) + 5.0 * np.random.normal(0, 1)
                                        dynamic_value = np.clip(dynamic_value, 20, 80)
                                        naaim_extended[date] = dynamic_value
                        except Exception as date_error:
                            print(f"날짜 {date} 처리 오류: {date_error}, 동적 값 생성")
                            # 오류 발생 시 동적 값 생성
                            date_position = i / len(ohlcv_data.index)
                            dynamic_value = 50.0 + 20.0 * np.sin(2 * np.pi * date_position) + 5.0 * np.random.normal(0, 1)
                            dynamic_value = np.clip(dynamic_value, 20, 80)
                            naaim_extended[date] = dynamic_value
                    
                    naaim = naaim_extended
                    print(f"NAIIM 데이터 확장 완료: {len(naaim)}개 데이터")
                    
                    # NAIIM 데이터 품질 확인
                    naaim_min, naaim_max = naaim.min(), naaim.max()
                    naaim_mean = naaim.mean()
                    print(f"NAIIM 데이터 품질: 최소={naaim_min:.1f}, 최대={naaim_max:.1f}, 평균={naaim_mean:.1f}")
                    
                    # NAIIM 데이터가 모두 동일한 값인 경우 동적 시리즈 생성
                    if naaim_min == naaim_max:
                        print(f"⚠️  NAIIM 데이터가 모두 동일한 값({naaim_min:.1f})입니다. 동적 시리즈로 대체합니다.")
                        naaim = generate_dynamic_naaim_series(ohlcv_data.index, naaim_min)
                    
            except Exception as loc_error:
                print(f"NAIIM 데이터 인덱싱 오류: {loc_error}")
                # 오류 발생 시 동적 시리즈 생성
                naaim = generate_dynamic_naaim_series(ohlcv_data.index, 50.0)
                print(f"인덱싱 오류 후 동적 NAIIM 시리즈 생성: {len(naaim)}개 데이터")
        else:
            naaim = None
            print("실제 NAIIM 데이터 파일이 없습니다.")
            
            # NAIIM이 없는 경우 기본값으로 시리즈 생성
            if naaim is None:
                naaim = pd.Series([50.0] * len(ohlcv_data), index=ohlcv_data.index)
                print(f"기본 NAIIM 시리즈 생성: {len(naaim)}개 데이터 (기본값: 50.0)")
    except Exception as e:
        print(f"NAIIM 데이터 로드 오류: {e}")
        naaim = None
        
        # 오류 발생 시 기본값으로 시리즈 생성
        if naaim is None:
            naaim = generate_dynamic_naaim_series(ohlcv_data.index, 50.0)
            print(f"오류 후 동적 NAIIM 시리즈 생성: {len(naaim)}개 데이터")
    
    # PCR 데이터 계산 (옵션이 있는 종목만)
    pcr = None
    pcr_series = None  # 기간별 PCR 시리즈 추가
    
    if ticker and not ticker.endswith('.KS'):  # 한국 종목 제외
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            options = stock.options
            if options:
                nearest_expiry = options[0]
                print(f"가장 가까운 만기일: {nearest_expiry}")
                
                calls = stock.option_chain(nearest_expiry).calls
                puts = stock.option_chain(nearest_expiry).puts
                
                total_call_volume = calls['volume'].sum()
                total_put_volume = puts['volume'].sum()
                
                print(f"Call 옵션 총 거래량: {total_call_volume:,}")
                print(f"Put 옵션 총 거래량: {total_put_volume:,}")
                
                if total_call_volume > 0:
                    pcr = total_put_volume / total_call_volume
                    print(f"{ticker} PCR 계산 완료: {pcr:.3f}")
                    print(f"PCR 해석: Put/Call 비율 = {total_put_volume:,}/{total_call_volume:,} = {pcr:.3f}")
                    
                    # PCR 값 검증
                    if pcr < 0.5:
                        print(f"⚠️  PCR {pcr:.3f} < 0.5: 매우 낙관적 (과매수 신호)")
                    elif pcr < 0.7:
                        print(f"✅ PCR {pcr:.3f} < 0.7: 낙관적 (매수 신호)")
                    elif pcr < 1.0:
                        print(f"⚠️  PCR {pcr:.3f} < 1.0: 중립적 (관망)")
                    elif pcr < 1.5:
                        print(f"[RED] PCR {pcr:.3f} > 1.0: 비관적 (매도 신호)")
                    else:
                        print(f"🚨 PCR {pcr:.3f} > 1.5: 매우 비관적 (과매도 신호)")
                    
                    # 기간별 PCR 시리즈 생성 (실제 기간별 데이터 시뮬레이션)
                    pcr_series = calculate_historical_pcr_series(ticker, ohlcv_data.index, pcr)
                    print(f"기간별 PCR 시리즈 생성 완료: {len(pcr_series)}개 데이터")
                    print(f"PCR 시리즈 샘플: {pcr_series.head(5).values}")
                    print(f"PCR 시리즈 통계: 최소={pcr_series.min():.3f}, 최대={pcr_series.max():.3f}, 평균={pcr_series.mean():.3f}")
                else:
                    print(f"⚠️  Call 옵션 거래량이 0입니다. PCR 계산 불가.")
            else:
                print(f"⚠️  {ticker}에 대한 옵션 데이터가 없습니다.")
                    
        except Exception as e:
            print(f"{ticker} PCR 계산 오류: {e}")
    
    # PCR이 없는 경우 기본값으로 시리즈 생성 (시뮬레이션)
    if pcr_series is None:
        # 기본 PCR 값 1.0으로 시리즈 생성
        pcr_series = pd.Series([1.0] * len(ohlcv_data), index=ohlcv_data.index)
        print(f"기본 PCR 시리즈 생성: {len(pcr_series)}개 데이터 (기본값: 1.0)")

    # 기술적 지표 계산
    hma = calculate_hma(ohlcv_data['Close'])
    upper_band, lower_band = calculate_mantra_bands(ohlcv_data['Close'])
    rsi3 = calculate_rsi(ohlcv_data['Close'], period=3)
    rsi14 = calculate_rsi(ohlcv_data['Close'], period=14)
    rsi50 = calculate_rsi(ohlcv_data['Close'], period=50)
    
    # RSI Signal 선 계산 (RSI(14)의 9기간 EMA)
    rsi_signal = rsi14.ewm(span=9).mean()
    
    macd, macd_signal, hist = calculate_macd(ohlcv_data['Close'])
    
    # 볼린저 밴드 계산
    volume_ma = ohlcv_data['Volume'].rolling(window=20).mean()
    volume_std = ohlcv_data['Volume'].rolling(window=20).std()
    volume_upper = volume_ma + (volume_std * 2)
    
        # 볼린저 밴드 계산
    bb_ma, bb_upper, bb_lower = calculate_bollinger_bands(ohlcv_data['Close'])
    disparity_result = analyze_disparity_strategy(
        ohlcv_data,
        rsi14=rsi14,
        macd=macd,
        macd_signal=macd_signal,
        bb_upper=bb_upper,
        bb_lower=bb_lower,
        disparity_ma=disparity_ma,
        disparity_low=disparity_low,
        disparity_high=disparity_high,
        conf_weight_depth=disparity_conf_weight_depth,
        conf_weight_volume=disparity_conf_weight_volume,
        conf_weight_trend=disparity_conf_weight_trend,
    )
    
    # SMA 이동평균선 계산
    sma200 = ohlcv_data['Close'].rolling(window=200).mean()
    sma120 = ohlcv_data['Close'].rolling(window=120).mean()
    sma50 = ohlcv_data['Close'].rolling(window=50).mean()
    sma20 = ohlcv_data['Close'].rolling(window=20).mean()
    
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
        try:
            vix_end_date = vix.index[vix.index <= end_date][-1] if len(vix.index[vix.index <= end_date]) > 0 else vix.index[-1]
            vix_value = vix.loc[vix_end_date]
            
            # Series인 경우 iloc[0] 사용, 단일 값인 경우 직접 변환
            if hasattr(vix_value, 'iloc'):
                vix_final = float(vix_value.iloc[0])
            else:
                vix_final = float(vix_value)
            
            print(f"VIX 최종값: {vix_final}")
        except Exception as e:
            print(f"VIX 최종값 추출 오류: {e}, 기본값 사용")
            vix_final = 20.0  # 기본값 사용
    
    if naaim is not None and not naaim.empty:
        try:
            naaim_end_date = naaim.index[naaim.index <= end_date][-1] if len(naaim.index[naaim.index <= end_date]) > 0 else naaim.index[-1]
            naaim_value = naaim.loc[naaim_end_date]
            # Series인 경우 iloc[0] 사용, 단일 값인 경우 직접 변환
            if hasattr(naaim_value, 'iloc'):
                naaim_final = float(naaim_value.iloc[0])
            else:
                naaim_final = float(naaim_value)
            
            print(f"NAIIM 최종값: {naaim_final}")
        except Exception as e:
            print(f"NAIIM 최종값 추출 오류: {e}")
            naaim_final = 50.0  # 기본값 사용
    
    print(f"PCR 값: {pcr}")
    market_summary = _get_market_sentiment_summary(vix_final, naaim_final, pcr)
    strategy_guide = _get_strategy_guide(vix_final, naaim_final, pcr)
    
    print(f"시장 심리 요약: {market_summary}")
    print(f"전략 가이드: {strategy_guide}")
    
    # 차트 생성: 옵션에 따라 GridSpec 행 동적 구성
    ax_pattern = None
    ax_disparity = None
    ax_adx = None

    # 패턴 스트립 높이 산정용 — 감지된 종류 수만 행으로 쓰므로 미리 분석
    _enabled_main_early = normalize_pattern_main_ids(pattern_main_overlays or [])
    _pat_events_cached = None
    _pattern_box_ranges_cached = None
    _strip_n_rows = 0
    if show_pattern_strip or bool(_enabled_main_early):
        try:
            _pattern_box_ranges_cached = compute_box_range_windows(
                ohlcv_data,
                pattern_range_box_period,
                pattern_range_box_num,
                pattern_range_box_overlap,
                pattern_range_box_avoid_time_overlap,
            )
            _pat_events_cached = analyze_seven_criteria(
                ohlcv_data, box_ranges=_pattern_box_ranges_cached
            )
            if show_pattern_strip:
                _strip_n_rows = count_pattern_strip_rows(_pat_events_cached)
        except Exception as e:
            print(f"차트 패턴 사전 분석 실패(스트립 높이 기본값 사용): {e}")
            _pat_events_cached = None
            _pattern_box_ranges_cached = None
            _strip_n_rows = 8 if show_pattern_strip else 0

    height_ratios = [3.0]
    if show_pattern_strip:
        height_ratios.append(pattern_strip_height_ratio(_strip_n_rows))
    height_ratios.extend([1.0, 1.0, 1.0])  # Volume, RSI, MACD
    if show_adx_dmi:
        height_ratios.append(0.85)
    if show_disparity_strategy:
        height_ratios.append(0.95)
    height_ratios.extend([1.5, 1.2])  # 금리, 시장심리

    # 스트립이 커지면 전체 figure도 조금 키움
    fig_height = 19 + len(height_ratios) * 0.55
    if show_pattern_strip:
        fig_height += max(0.0, pattern_strip_height_ratio(_strip_n_rows) - 0.45) * 2.2
    fig = plt.figure(figsize=(20, fig_height))
    gs = GridSpec(len(height_ratios), 1, height_ratios=height_ratios, figure=fig, hspace=0.12)

    row = 0
    ax_main = fig.add_subplot(gs[row, 0])
    row += 1
    if show_pattern_strip:
        ax_pattern = fig.add_subplot(gs[row, 0], sharex=ax_main)
        row += 1
    
    # 종목 정보 가져오기
    long_name, sector, industry = get_stock_info(ticker)
    
    # 배당일 정보 가져오기 (OHLCV 데이터 전달하여 배당률 계산)
    dividend_dates = get_dividend_dates(ticker, ohlcv_data.index[0], ohlcv_data.index[-1], ohlcv_data)
    
    # 차트 타이틀에 시장 심리 종합해석 추가
    title_text = f"{ticker} HMA Mantra Analysis"
    if market_summary and market_summary != "시장 데이터 부족":
        title_text += f" | {market_summary}"
    
    print(f"설정할 타이틀: {title_text}")
    
    # 메인 차트 타이틀에 종목 정보 추가
    main_title = f"{ticker} - {long_name}\n"
    main_title += f"섹터: {sector} | 산업: {industry}"
    ax_main.set_title(main_title, fontsize=12, fontweight='bold', pad=15)
    
    # 시장 심리 요약을 최상단 우측에 표시
    if market_summary and market_summary != "시장 데이터 부족":
        ax_summary = fig.add_axes([0.6, 0.95, 0.35, 0.03])
        ax_summary.set_facecolor('lightblue')
        ax_summary.set_xlim(0, 1)
        ax_summary.set_ylim(0, 1)
        ax_summary.axis('off')
        ax_summary.set_zorder(1000)
        ax_summary.text(
            0.5, 0.5, market_summary,
            fontsize=12, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue',
                      edgecolor='navy', linewidth=2),
            zorder=1001,
        )
    
    # 버핏 지수·공포탐욕 지수 (표시는 통합 박스에서 처리)
    buffett_data = None
    buffett_value = None
    buffett_sentiment = None
    buffett_guide = None
    fear_greed_data = None
    fear_greed_score = None
    fear_greed_sentiment = None
    fear_greed_guide = None
    peg_data = None
    peg_value = None
    peg_sentiment = None
    peg_guide = None

    buffett_data = calculate_buffett_indicator()
    if buffett_data:
        buffett_value = buffett_data['value']
        buffett_sentiment, buffett_color, buffett_guide = get_buffett_sentiment(buffett_value)

    fear_greed_data = calculate_fear_greed_indicator()
    if fear_greed_data:
        fear_greed_score = fear_greed_data['score']
        fear_greed_sentiment = fear_greed_data.get('sentiment')
        fear_greed_guide = fear_greed_data.get('guide')

    if ticker:
        peg_data = calculate_peg_indicator(ticker)
        if peg_data:
            peg_value = peg_data.get('peg')
            peg_sentiment = peg_data.get('sentiment')
            peg_guide = peg_data.get('guide')

    ax_volume = fig.add_subplot(gs[row, 0], sharex=ax_main)
    row += 1
    ax_rsi = fig.add_subplot(gs[row, 0], sharex=ax_main)
    row += 1
    ax_macd = fig.add_subplot(gs[row, 0], sharex=ax_main)
    row += 1
    if show_adx_dmi:
        ax_adx = fig.add_subplot(gs[row, 0], sharex=ax_main)
        row += 1
    if show_disparity_strategy:
        ax_disparity = fig.add_subplot(gs[row, 0], sharex=ax_main)
        row += 1
    ax_rates = fig.add_subplot(gs[row, 0], sharex=ax_main)
    row += 1
    ax_sentiment = fig.add_subplot(gs[row, 0], sharex=ax_main)

    adx_dmi_data = None
    if show_adx_dmi:
        try:
            adx_dmi_data = analyze_adx_dmi(
                ohlcv_data,
                period=adx_period,
                adx_sideways=adx_sideways,
                adx_trend=adx_trend,
                adx_strong=adx_strong,
            )
            n_buy = adx_dmi_data.get("buy_signal_count", 0)
            print(
                f"ADX/DMI: {adx_dmi_data.get('summary')} "
                f"(ADX={adx_dmi_data.get('adx')}, +DI={adx_dmi_data.get('plus_di')}, "
                f"-DI={adx_dmi_data.get('minus_di')}, ADX buy={n_buy}건)"
            )
        except Exception as e:
            print(f"ADX/DMI 분석 실패: {e}")

    # 메인 차트 설정 (투명도 높임)
    candlestick_ohlc(ax_main, 
                     [[mdates.date2num(date), o, h, l, c] for date, (o, h, l, c) in 
                      zip(ohlcv_data.index, ohlcv_data[['Open', 'High', 'Low', 'Close']].values)],
                     width=0.6, colorup='green', colordown='red', alpha=0.9)
    if show_disparity_strategy:
        try:
            disp_series_main = disparity_result.get("disparity")
            if disp_series_main is not None and len(disp_series_main):
                # 메인 차트 이격도는 우측 보조축(%) 기준으로 표시
                ax_main_disp = ax_main.twinx()
                dmin = float(np.nanmin(disp_series_main.values))
                dmax = float(np.nanmax(disp_series_main.values))
                my0 = min(float(disparity_low) - 2.0, dmin - 1.0)
                my1 = max(float(disparity_high) + 2.0, dmax + 1.0)
                ax_main_disp.set_ylim(my0, my1)
                ax_main_disp.plot(
                    ohlcv_data.index,
                    disp_series_main,
                    color="#2c3e50",
                    linewidth=0.75,
                    alpha=0.70,
                    label=f"Disparity({disparity_ma}) [right axis]",
                    zorder=18,
                )
                ax_main_disp.axhline(float(disparity_low), color="#27ae60", linestyle="--", linewidth=0.55, alpha=0.65, zorder=17)
                ax_main_disp.axhline(100.0, color="#7f8c8d", linestyle=":", linewidth=0.5, alpha=0.6, zorder=17)
                ax_main_disp.axhline(float(disparity_high), color="#c0392b", linestyle="--", linewidth=0.55, alpha=0.65, zorder=17)
                ax_main_disp.set_ylabel(f"Disparity({disparity_ma})", fontsize=6, color="#2c3e50")
                ax_main_disp.tick_params(axis='y', labelsize=6, colors="#2c3e50")
                ax_main_disp.grid(False)
                ax_main_disp.legend(loc='upper right', fontsize=6, framealpha=0.75)

                sma_base = ohlcv_data['Close'].rolling(window=disparity_ma).mean()
                eq_95 = sma_base * (float(disparity_low) / 100.0)
                eq_100 = sma_base
                eq_105 = sma_base * (float(disparity_high) / 100.0)
                ax_main.plot(
                    ohlcv_data.index,
                    eq_100,
                    color="#8e44ad",
                    linewidth=0.55,
                    linestyle=":",
                    alpha=0.65,
                    label=f"Eq 100% (SMA{disparity_ma})",
                    zorder=16,
                )
                ax_main.plot(
                    ohlcv_data.index,
                    eq_95,
                    color="#27ae60",
                    linewidth=0.45,
                    linestyle="--",
                    alpha=0.45,
                    label=f"Eq {disparity_low:.0f}%",
                    zorder=15,
                )
                ax_main.plot(
                    ohlcv_data.index,
                    eq_105,
                    color="#c0392b",
                    linewidth=0.45,
                    linestyle="--",
                    alpha=0.45,
                    label=f"Eq {disparity_high:.0f}%",
                    zorder=15,
                )

                # 메인 차트 하단에 이격도 신호 레인(매수/매도/관망) 표시
                evs_main = disparity_result.get("events", [])
                lane_buy, lane_sell, lane_wait = [], [], []
                for ev in evs_main:
                    ts = pd.Timestamp(ev.get("date"))
                    act = str(ev.get("action", "wait"))
                    if act == "buy_scale_in":
                        lane_buy.append(ts)
                    elif act in ("take_profit", "reduce"):
                        lane_sell.append(ts)
                    else:
                        lane_wait.append(ts)

                def _thin_dates(dates, min_gap_days=5, max_n=40):
                    out = []
                    last = None
                    for dt in sorted(dates):
                        if last is None or abs((dt - last).days) >= min_gap_days:
                            out.append(dt)
                            last = dt
                    return out[-max_n:]

                lane_buy = _thin_dates(lane_buy, min_gap_days=4, max_n=36)
                lane_sell = _thin_dates(lane_sell, min_gap_days=4, max_n=36)
                lane_wait = _thin_dates(lane_wait, min_gap_days=6, max_n=28)

                xtrans = ax_main.get_xaxis_transform()  # x=data, y=axes-fraction
                if lane_buy:
                    for dt in lane_buy:
                        ax_main.axvline(
                            x=dt,
                            ymin=0.02,
                            ymax=0.98,
                            color="#27ae60",
                            linestyle="--",
                            linewidth=0.55,
                            alpha=0.28,
                            zorder=14,
                        )
                    ax_main.scatter(
                        lane_buy,
                        [0.055] * len(lane_buy),
                        transform=xtrans,
                        marker="^",
                        s=22,
                        color="#27ae60",
                        alpha=0.95,
                        zorder=19,
                        clip_on=False,
                        label="Disparity 매수",
                    )
                if lane_sell:
                    for dt in lane_sell:
                        ax_main.axvline(
                            x=dt,
                            ymin=0.02,
                            ymax=0.98,
                            color="#c0392b",
                            linestyle="--",
                            linewidth=0.55,
                            alpha=0.28,
                            zorder=14,
                        )
                    ax_main.scatter(
                        lane_sell,
                        [0.095] * len(lane_sell),
                        transform=xtrans,
                        marker="v",
                        s=22,
                        color="#c0392b",
                        alpha=0.95,
                        zorder=19,
                        clip_on=False,
                        label="Disparity 매도",
                    )
                if lane_wait:
                    ax_main.scatter(
                        lane_wait,
                        [0.135] * len(lane_wait),
                        transform=xtrans,
                        marker="o",
                        s=10,
                        color="#7f8c8d",
                        alpha=0.65,
                        zorder=18,
                        clip_on=False,
                        label="Disparity 관망",
                    )
        except Exception as e:
            print(f"메인 차트 이격도 표시 실패: {e}")
    if show_disparity_strategy and ax_disparity is not None:
        try:
            disp_series = disparity_result.get("disparity")
            if disp_series is not None and len(disp_series):
                disp_min = float(np.nanmin(disp_series.values))
                disp_max = float(np.nanmax(disp_series.values))
                y0 = min(float(disparity_low) - 2.0, disp_min - 1.0)
                y1 = max(float(disparity_high) + 2.0, disp_max + 1.0)
                ax_disparity.set_ylim(y0, y1)
                ax_disparity.plot(
                    ohlcv_data.index,
                    disp_series,
                    color="#34495e",
                    linewidth=1.0,
                    alpha=0.95,
                    label=f"Disparity({disparity_ma})",
                )
                ax_disparity.axhline(float(disparity_low), color="#27ae60", linestyle="--", linewidth=0.9, alpha=0.9, label=f"Low {disparity_low:.0f}%")
                ax_disparity.axhline(100.0, color="#7f8c8d", linestyle=":", linewidth=0.8, alpha=0.85, label="Mid 100%")
                ax_disparity.axhline(float(disparity_high), color="#c0392b", linestyle="--", linewidth=0.9, alpha=0.9, label=f"High {disparity_high:.0f}%")

                ax_disparity.fill_between(
                    ohlcv_data.index,
                    y0,
                    float(disparity_low),
                    color="#2ecc71",
                    alpha=0.06,
                )
                ax_disparity.fill_between(
                    ohlcv_data.index,
                    float(disparity_high),
                    y1,
                    color="#e74c3c",
                    alpha=0.06,
                )

                evs = disparity_result.get("events", [])
                buy_pts, sell_pts, wait_pts = [], [], []
                for ev in evs:
                    ts = pd.Timestamp(ev.get("date"))
                    dv = float(ev.get("disparity", np.nan))
                    act = str(ev.get("action", "wait"))
                    tdir = str(ev.get("trend_short", "S")).upper()
                    tslope = float(ev.get("trend_slope", 0.0))
                    vdir = str(ev.get("volume_short", "F")).upper()
                    if np.isnan(dv):
                        continue
                    if act == "buy_scale_in":
                        buy_pts.append((ts, dv, tdir, tslope, vdir))
                    elif act in ("take_profit", "reduce"):
                        sell_pts.append((ts, dv, tdir, tslope, vdir))
                    else:
                        wait_pts.append((ts, dv, tdir, tslope, vdir))
                if buy_pts:
                    buy_x = [p[0] for p in buy_pts]
                    buy_y = [p[1] for p in buy_pts]
                    ax_disparity.scatter(buy_x, buy_y, marker="^", s=26, color="#27ae60", label="매수", zorder=6)
                    for bx, by, bt, bs, bv in buy_pts[-6:]:
                        ax_disparity.annotate(
                            f"BUY (T:{bt},{bs:+.2f} / V:{bv})",
                            xy=(bx, by),
                            xytext=(0, 7),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=6,
                            color="#1e8449",
                            zorder=7,
                            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#27ae60", alpha=0.85),
                        )
                if sell_pts:
                    sell_x = [p[0] for p in sell_pts]
                    sell_y = [p[1] for p in sell_pts]
                    ax_disparity.scatter(sell_x, sell_y, marker="v", s=26, color="#c0392b", label="매도", zorder=6)
                    for sx, sy, st, ss, sv in sell_pts[-6:]:
                        ax_disparity.annotate(
                            f"SELL (T:{st},{ss:+.2f} / V:{sv})",
                            xy=(sx, sy),
                            xytext=(0, -8),
                            textcoords="offset points",
                            ha="center",
                            va="top",
                            fontsize=6,
                            color="#922b21",
                            zorder=7,
                            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#c0392b", alpha=0.85),
                        )
                if wait_pts:
                    wait_x = [p[0] for p in wait_pts]
                    wait_y = [p[1] for p in wait_pts]
                    ax_disparity.scatter(wait_x, wait_y, marker="o", s=14, color="#7f8c8d", alpha=0.7, label="관망", zorder=5)
                    for wx, wy, wt, ws, wv in wait_pts[-4:]:
                        ax_disparity.annotate(
                            f"WAIT (T:{wt},{ws:+.2f} / V:{wv})",
                            xy=(wx, wy),
                            xytext=(0, 8),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=5.5,
                            color="#4d5656",
                            zorder=6,
                            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="#7f8c8d", alpha=0.78),
                        )

                ax_disparity.set_title(f"Disparity Strategy ({disparity_ma}MA)", fontsize=9)
                ax_disparity.set_ylabel("Disparity %", fontsize=8)
                ax_disparity.grid(True, alpha=0.25, linestyle=":")
                ax_disparity.legend(loc="upper left", fontsize=6, framealpha=0.85, ncol=3)
        except Exception as e:
            print(f"이격도 subplot 표시 실패: {e}")
    
    # 배당일을 메인차트 최상단(top)에 "D"로 표시
    if dividend_dates:
        # Y축 범위 가져오기
        y_min, y_max = ax_main.get_ylim()
        y_range = y_max - y_min
        
        for div_info in dividend_dates:
            div_date = div_info['date']
            div_amount = div_info['amount']
            div_yield = div_info.get('yield', None)
            
            # 배당일이 차트 범위 내에 있는지 확인
            if div_date in ohlcv_data.index:
                # 차트의 절대적인 최상단 위치 (Y축 최대값)
                top_position = y_max
                
                # 배당일에 수직선 추가 (두께 80% 감소)
                ax_main.axvline(x=div_date, color='orange', linestyle='--', 
                              linewidth=0.2, alpha=0.7, zorder=5)
                
                # 배당일자 텍스트를 마름모 아래에 표시 (배당정보와 같은 크기)
                ax_main.text(div_date, top_position - (y_range * 0.02), div_date.strftime('%m/%d'), 
                           ha='center', va='top', fontsize=6, 
                           color='darkorange', fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", 
                                   facecolor='white', 
                                   edgecolor='orange', alpha=0.8),
                           zorder=11)
                
                # "D" 마커를 차트 최상단에 표시 (사이즈 80% 감소)
                ax_main.scatter(div_date, top_position, 
                              marker='D', s=10, color='gold', 
                              edgecolor='orange', linewidth=1, 
                              zorder=10, alpha=0.9)
                
                # 배당률과 배당금 텍스트를 차트 최상단에 표시
                if div_yield is not None:
                    text_content = f'D({div_yield:.2f}%)\n{currency_symbol}{div_amount:.4f}'
                else:
                    text_content = f'D\n{currency_symbol}{div_amount:.4f}'
                
                ax_main.annotate(text_content, 
                               xy=(div_date, top_position), 
                               xytext=(0, 5), textcoords='offset points',
                               ha='center', va='bottom', fontsize=6, 
                               color='darkorange', fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.2", 
                                       facecolor='lightyellow', 
                                       edgecolor='orange', alpha=0.8))

    # HMA와 만트라 밴드
    ax_main.plot(ohlcv_data.index, hma, color='blue', linewidth=0.525, label='HMA')
    ax_main.plot(ohlcv_data.index, upper_band, color='red', linewidth=0.35, linestyle='--', label='Upper Mantra')
    ax_main.plot(ohlcv_data.index, lower_band, color='green', linewidth=0.35, linestyle='--', label='Lower Mantra')
    
    # 가격 라인 추가
    ax_main.plot(ohlcv_data.index, ohlcv_data['Close'], color='black', linewidth=0.28, alpha=0.7, label='Price')
    
    # 만트라 밴드 영역 채우기
    ax_main.fill_between(ohlcv_data.index, hma, upper_band, color='red', alpha=0.1, label='상단 밴드 영역')
    ax_main.fill_between(ohlcv_data.index, lower_band, hma, color='green', alpha=0.1, label='하단 밴드 영역')
    
    # 볼린저 밴드 추가
    ax_main.plot(ohlcv_data.index, bb_ma, color='purple', linewidth=0.175, label='BB MA(20)')
    ax_main.plot(ohlcv_data.index, bb_upper, color='purple', linewidth=0.175, linestyle=':', label='BB Upper')
    ax_main.plot(ohlcv_data.index, bb_lower, color='purple', linewidth=0.175, linestyle=':', label='BB Lower')
    ax_main.plot(ohlcv_data.index, sma20, color='#8e44ad', linewidth=0.85, linestyle='-', alpha=0.9, label='MA 20', zorder=7)
    
    # 긴 이평을 먼저 깔고, 짧은 이평을 그 위에 그린다.
    ax_main.plot(ohlcv_data.index, sma200, color='darkblue', linewidth=0.525, linestyle='-', label='MA 200', alpha=0.8, zorder=4)
    ax_main.plot(ohlcv_data.index, sma120, color='#117A65', linewidth=0.62, linestyle='-', label='MA 120', alpha=0.9, zorder=5)
    ax_main.plot(ohlcv_data.index, sma50, color='#E67E22', linewidth=0.7, linestyle='-', label='MA 50', alpha=0.9, zorder=6)
    
    # 박스권 표시: display_box_ranges 만 사용 (패턴 range_box 후보와 무관).
    if show_box_ranges and display_box_ranges:
        print(f"박스권 {len(display_box_ranges)}개 표시")
        plot_box_ranges_with_style(ax_main, display_box_ranges, box_style, currency_symbol)
    
    # Target 가격 라인들 추가 (옵션) - 비활성화됨
    # if show_target_prices:
    #     # 자동 계산된 target 가격들 (지지선/저항선 기반)
    #     auto_targets = calculate_target_prices_from_support_resistance(ohlcv_data)
    #     
    #     # 수동으로 지정된 target 가격들 또는 자동 계산된 가격들 사용
    #     final_target_buy = target_buy_price if target_buy_price is not None else (auto_targets['target_buy_price'] if auto_targets else None)
    #     final_target_sell = target_sell_price if target_sell_price is not None else (auto_targets['target_sell_price'] if auto_targets else None)
    #     final_stop_loss = stop_loss_price if stop_loss_price is not None else (auto_targets['stop_loss_price'] if auto_targets else None)
    #     
    #     # Target 가격 라인들 표시
    #     add_target_price_lines(ax_main, ohlcv_data, final_target_buy, final_target_sell, final_stop_loss)
    #     
    #     # Target 가격 정보 출력
    #     if any([final_target_buy, final_target_sell, final_stop_loss]):
    #         print(f"\n🎯 Target 가격 정보:")
    #         if final_target_buy:
    #             print(f"   Target 매수가: ${final_target_buy:.2f}")
    #         if final_target_sell:
    #             print(f"   Target 목표가: ${final_target_sell:.2f}")
    #         if final_stop_loss:
    #             print(f"   손절가: ${final_stop_loss:.2f}")
    #         
    #         if auto_targets:
    #             print(f"   지지선: ${auto_targets['support']:.2f}")
    #             print(f"   저항선: ${auto_targets['resistance']:.2f}")
    #             print(f"   리스크/보상 비율: 1:{auto_targets['risk_reward_ratio']:.1f}")

        # 현재 주가를 실시간으로 가져와서 연동
    realtime_price, realtime_time, data_freshness = get_current_stock_price(ticker)
    
    if realtime_price is not None:
        print(f"실시간 주가: {currency_symbol}{realtime_price:.2f} ({data_freshness})")
        # 실시간 주가가 있으면 이를 사용, 없으면 기존 데이터 사용
        current_price = realtime_price
        price_source = "실시간"
    else:
        print(f"실시간 주가 가져오기 실패, 기존 데이터 사용")
        current_price = ohlcv_data['Close'].iloc[-1]
        price_source = "기존데이터"
    
    # 전일대비 등락률 계산 (정제된 OHLCV 우선)
    daily_change = calculate_daily_change_percentage(ticker, ohlcv_data)
    if daily_change and np.isfinite(daily_change.get("change_percentage", np.nan)):
        print(f"전일대비: {daily_change['direction']} {daily_change['change_percentage']:.2f}% ({currency_symbol}{daily_change['change_amount']:.2f})")
    else:
        daily_change = None
        print("전일대비 등락률 계산 실패")
    
    support, resistance = calculate_support_resistance(ohlcv_data)
    
    # 현재가 기준 투자심리도와 RSI 계산
    current_index = len(ohlcv_data) - 1
    investor_sentiment = _calculate_investor_sentiment(ohlcv_data, current_index)
    current_rsi = rsi14.iloc[-1] if not rsi14.empty else None
    
    # RSI + 투자심리도 투자액션 및 전략 계산
    investment_action, investment_strategy = _get_rsi_sentiment_strategy(current_rsi, investor_sentiment)
    
    # 수평선 및 가격 표시 (메인차트 중앙에 표시) - thin 스타일
    ax_main.axhline(y=current_price, color='black', linestyle=':', linewidth=0.3, alpha=0.5)
    # 현재가 텍스트는 가격 차트 안 우측 (우측 여백은 주황 요약 박스)
    ax_main.text(0.985, 0.5, f'현재가: {currency_symbol}{current_price:.2f}', 
                transform=ax_main.transAxes, fontsize=8, ha='right', va='center',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', pad=2, boxstyle='round,pad=0.2'),
                color='black', fontweight='bold')
    # 현재가 수평선을 우측 텍스트까지 연결 (더 두껍고 명확하게)
    ax_main.plot([ohlcv_data.index[-1], ohlcv_data.index[-1] + pd.Timedelta(days=3)], 
                [current_price, current_price], color='black', linestyle='-', linewidth=1.0, alpha=0.8)
    
    ax_main.axhline(y=support, color='green', linestyle=':', linewidth=0.3, alpha=0.5)
    # 지지선 텍스트는 가격 차트 안 우측
    ax_main.text(0.985, 0.3, f'지지선: {currency_symbol}{support:.2f}', 
                transform=ax_main.transAxes, fontsize=8, ha='right', va='center',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='green', pad=2, boxstyle='round,pad=0.2'),
                color='green', fontweight='bold')
    # 지지선 수평선을 우측 텍스트까지 연결 (더 두껍고 명확하게)
    ax_main.plot([ohlcv_data.index[-1], ohlcv_data.index[-1] + pd.Timedelta(days=3)], 
                [support, support], color='green', linestyle='-', linewidth=1.0, alpha=0.8)
    
    ax_main.axhline(y=resistance, color='red', linestyle=':', linewidth=0.3, alpha=0.5)
    # 저항선 텍스트는 가격 차트 안 우측
    ax_main.text(0.985, 0.7, f'저항선: {currency_symbol}{resistance:.2f}', 
                transform=ax_main.transAxes, fontsize=8, ha='right', va='center',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='red', pad=2, boxstyle='round,pad=0.2'),
                color='red', fontweight='bold')
    # 저항선 수평선을 우측 텍스트까지 연결 (더 두껍고 명확하게)
    ax_main.plot([ohlcv_data.index[-1], ohlcv_data.index[-1] + pd.Timedelta(days=3)], 
                [resistance, resistance], color='red', linestyle='-', linewidth=1.0, alpha=0.8)
    
    # 현재가 캔들바 아래에 투자심리도와 RSI 표시
    ax_realtime = None
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
            
                    # MA50·MA200과 현재가 비교. MA50을 MA200 윗줄에 둔다.
        def _ma_vs_price(label, series):
            valid = series.dropna() if series is not None and len(series) else series
            value = float(valid.iloc[-1]) if valid is not None and len(valid) else None
            if value is None or not np.isfinite(value):
                return f'{label}: 데이터 부족'
            side = '현재가 상단' if current_price > value else '현재가 하단'
            return f'{label}: {value:.2f} ({side})'

        sma_status = f"{_ma_vs_price('MA 50', sma50)}\n{_ma_vs_price('MA 200', sma200)}"
        
        # 통합 정보를 하나의 박스에 표시 (현재 일시, 현재가, 투자전략, 버핏지수 포함)
        # 실시간 주가 시간을 우선으로 사용하여 일시 표시
        if realtime_price is not None and realtime_time is not None:
            # 실시간 주가 시간을 한국 시간으로 변환하여 일시 표시
            realtime_local = get_local_datetime(realtime_time)
            current_datetime = realtime_local.strftime('%Y-%m-%d %H:%M')
            
                    # 검색기간 수익률 계산 (안전한 처리)
        try:
            period_return = calculate_period_return(ohlcv_data, ohlcv_data.index[0], ohlcv_data.index[-1])
        except Exception as e:
            print(f"검색기간 수익률 계산 오류: {e}")
            period_return = None
            
            # 전일대비 등락률 정보 추가
            if daily_change:
                change_info = f' {daily_change["direction"]} {daily_change["change_percentage"]:.2f}%'
            if realtime_price is not None and realtime_local is not None:
                price_info = f'[실시간가] {currency_symbol}{realtime_price:.2f} ({realtime_local.strftime("%H:%M")}){change_info}'
            else:
                price_info = f'[현재가] {currency_symbol}{current_price:.2f}{change_info}'
        else:
            if realtime_price is not None and realtime_local is not None:
                price_info = f'[실시간가] {currency_symbol}{realtime_price:.2f} ({realtime_local.strftime("%H:%M")})'
            else:
                price_info = f'[현재가] {currency_symbol}{current_price:.2f}'
        
        # 검색기간 수익률 정보 추가
        if period_return:
            period_info = f'\n[검색기간] {period_return["start_date"]} → {period_return["end_date"]}\n{period_return["direction"]} {period_return["return_percentage"]:.2f}% ({currency_symbol}{period_return["return_amount"]:.2f})'
            price_info += period_info
        
        # current_datetime 변수 안전하게 정의
        if realtime_price is not None and realtime_local is not None:
            current_datetime = realtime_local.strftime('%Y-%m-%d %H:%M')
        else:
            # 실시간 주가가 없는 경우 기존 데이터 시간 사용
            utc_datetime = ohlcv_data.index[-1]
            local_datetime = get_local_datetime(utc_datetime)
            current_datetime = local_datetime.strftime('%Y-%m-%d %H:%M')
        
        info_text = f'[일시] {current_datetime} (KST)\n{price_info}\n[데이터] {price_source}\n\n투자전략: {investment_strategy}\n투자액션: {investment_action}\nRSI: {current_rsi:.1f}\n투자심리도: {investor_sentiment:.1f}%\n{sma_status}'

        regime_text = _market_regime_text(
            hy_spread_data, vix, tnx_data, ffr_data, inflation_data
        )
        beta_text = _beta_info_text(ticker, ohlcv_data, regime_text)
        cross_text = _cross_asset_regime_text(
            ohlcv_data.index[-1], fear_greed_score, fear_greed_data
        )
        # 베타·교차자산 뒤에 국면 시사점을 현재국면 종합으로 붙인다.
        regime_block = regime_text or ""
        for extra in (beta_text, cross_text, _regime_synthesis(regime_text, beta_text, cross_text)):
            if extra:
                regime_block = f"{regime_block}\n{extra}" if regime_block else extra
        if regime_block:
            info_text += "\n\n" + regime_block
            print(f"시장 국면:\n{regime_block}")
        
        # 버핏지수 정보 추가
        if buffett_value and buffett_data:
            info_text += f'\n\n버핏지수: {buffett_value}% ({buffett_sentiment})\n시장가이드: {buffett_guide}'
            info_text += f'\nWilshire 5000: {buffett_data["wilshire_market_cap"]}조 달러 ({buffett_data["wilshire_date"]})'
            gdp_yoy = buffett_data.get("us_gdp_yoy")
            yoy_txt = f", 전년비 {gdp_yoy:+.1f}%" if gdp_yoy is not None else ""
            info_text += f'\nUS GDP: {buffett_data["us_gdp"]}조 달러 ({buffett_data["gdp_date"]}{yoy_txt})'
            if buffett_data.get('used_fallback'):
                info_text += '\n⚠️ 버핏지수: 일부 fallback 데이터'

        if fear_greed_score is not None and fear_greed_data:
            fg_rating = fear_greed_data.get('rating_ko', fear_greed_data.get('rating', ''))
            info_text += f'\n\n공포탐욕: {fear_greed_score} ({fg_rating})'
            if fear_greed_sentiment:
                info_text += f'\n심리: {fear_greed_sentiment}'
            if fear_greed_guide:
                info_text += f'\n가이드: {fear_greed_guide}'
            if fear_greed_data.get('previous_close') is not None:
                info_text += f"\n전일: {fear_greed_data['previous_close']}"
            if fear_greed_data.get('timestamp'):
                info_text += f"\n기준: {fear_greed_data['timestamp']} ({fear_greed_data.get('source', 'CNN')})"
            if fear_greed_data.get('used_fallback'):
                info_text += '\n⚠️ 공포탐욕: crypto fallback'

        if peg_data is not None:
            pe_line = (
                f"{peg_data.get('pe_label', 'PER')} {peg_data.get('pe')}"
                if peg_data.get('pe') is not None
                else ""
            )
            peg_lines = []
            if peg_data.get('peg_1y') is not None:
                peg_lines.append(
                    f"PEG 1년: {peg_data['peg_1y']} ({peg_data.get('peg_1y_sentiment')}) "
                    f"EPS YoY {peg_data.get('growth_1y_pct')}%"
                )
            else:
                peg_lines.append("PEG 1년: N/A")
            if peg_data.get('peg_3y') is not None:
                peg_lines.append(
                    f"PEG 3년: {peg_data['peg_3y']} ({peg_data.get('peg_3y_sentiment')}) "
                    f"EPS CAGR {peg_data.get('growth_3y_cagr_pct')}%"
                )
            else:
                peg_lines.append("PEG 3년: N/A")
            if peg_data.get('yf_peg_ratio') is not None:
                peg_lines.append(f"(참고 Yahoo PEG {peg_data['yf_peg_ratio']})")
            info_text += "\n\n" + "\n".join(peg_lines)
            if pe_line:
                info_text += f"\n{pe_line}"

        if adx_dmi_data is not None and adx_dmi_data.get("adx") is not None:
            info_text += (
                f"\n\n추세(ADX/DMI): {adx_dmi_data.get('summary')}"
                f"\nADX={adx_dmi_data.get('adx')}"
                f" | DMI +DI={adx_dmi_data.get('plus_di')}"
                f" −DI={adx_dmi_data.get('minus_di')}"
            )
            n_adx_buy = adx_dmi_data.get("buy_signal_count", 0)
            if n_adx_buy:
                info_text += f"\nADX buy 신호: {n_adx_buy}건 (DI↑ & ADX≥{adx_trend:.0f})"

        # 위치는 레이아웃 확정 후 가격 차트 오른쪽에 맞춤
        ax_realtime = fig.add_axes([0.76, 0.62, 0.22, 0.30])
        ax_realtime.set_facecolor('none')
        ax_realtime.patch.set_alpha(0)
        ax_realtime.set_xlim(0, 1)
        ax_realtime.set_ylim(0, 1)
        ax_realtime.axis('off')
        ax_realtime.set_zorder(1000)

        if show_disparity_strategy:
            try:
                latest_event = disparity_result.get("latest_event") or {}
                disp_v = latest_event.get("disparity", np.nan)
                disp_txt = f"{disp_v:.2f}" if pd.notna(disp_v) else "N/A"
                state_txt = str(latest_event.get("state", "NEUTRAL"))
                action_txt = str(disparity_result.get("final_action", "관망"))
                info_text += (
                    f"\n\n[이격도]\n"
                    f"현재 이격도: {disp_txt}\n"
                    f"상태: {state_txt}\n"
                    f"액션: {action_txt}"
                )
            except Exception as e:
                print(f"이격도 정보 텍스트 구성 실패: {e}")

        ax_realtime.text(
            0.04, 0.98, info_text,
            fontsize=8, ha='left', va='top',
            multialignment='left',
            linespacing=1.25,
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor='white',
                edgecolor='darkorange',
                linewidth=2.5,
                alpha=0.95,
            ),
            color='black',
            fontweight='bold',
            zorder=1001,
        )
        
        # 현재가 수직선 - 원래 위치로 복원
        if current_price_line_style == 'thin':
            # 방식 1: 수직선 두께를 얇게 하여 캔들바가 보이도록 함
            ax_main.axvline(x=current_date, ymin=0, ymax=1, color='darkorange', linestyle='--', 
                           linewidth=0.5, alpha=0.6, zorder=999)  # 두께를 0.5로 줄이고 투명도 조정
        elif current_price_line_style == 'bottom_start':
            # 방식 2: 수직선을 캔들바 하단에서 시작하여 아래로 그리기
            current_candle_low = ohlcv_data['Low'].iloc[-1]
            price_range = ohlcv_data['High'].max() - ohlcv_data['Low'].min()
            ymin_normalized = (current_candle_low - ohlcv_data['Low'].min()) / price_range
            ax_main.axvline(x=current_date, ymin=ymin_normalized, ymax=1, color='darkorange', linestyle='--', 
                           linewidth=1.0, alpha=0.7, zorder=999)
        else:
            # 기본값: 얇은 수직선
            ax_main.axvline(x=current_date, ymin=0, ymax=1, color='darkorange', linestyle='--', 
                           linewidth=0.5, alpha=0.6, zorder=999)
        
        # 검색기간 수익률 출력 (안전한 처리)
        try:
            period_return = calculate_period_return(ohlcv_data, ohlcv_data.index[0], ohlcv_data.index[-1])
            if period_return:
                print(f"📊 검색기간 수익률: {period_return['start_date']} → {period_return['end_date']}")
                print(f"   시작가: ${period_return['start_price']:.2f} → 종료가: ${period_return['end_price']:.2f}")
                print(f"   {period_return['direction']} {period_return['return_percentage']:.2f}% (${period_return['return_amount']:.2f})")
        except Exception as e:
            print(f"검색기간 수익률 출력 오류: {e}")
        
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
        'buffett_guide': buffett_guide if buffett_value else None,
        'wilshire_market_cap': buffett_data.get('wilshire_market_cap') if buffett_data else None,
        'wilshire_date': buffett_data.get('wilshire_date') if buffett_data else None,
        'us_gdp': buffett_data.get('us_gdp') if buffett_data else None,
        'us_gdp_yoy': buffett_data.get('us_gdp_yoy') if buffett_data else None,
        'gdp_date': buffett_data.get('gdp_date') if buffett_data else None,
        'fear_greed_score': fear_greed_score,
        'fear_greed_rating': fear_greed_data.get('rating_ko') if fear_greed_data else None,
        'fear_greed_sentiment': fear_greed_sentiment,
        'fear_greed_guide': fear_greed_guide,
        'fear_greed_source': fear_greed_data.get('source') if fear_greed_data else None,
        'fear_greed_timestamp': fear_greed_data.get('timestamp') if fear_greed_data else None,
        'peg': peg_value,
        'peg_1y': peg_data.get('peg_1y') if peg_data else None,
        'peg_3y': peg_data.get('peg_3y') if peg_data else None,
        'peg_1y_sentiment': peg_data.get('peg_1y_sentiment') if peg_data else None,
        'peg_3y_sentiment': peg_data.get('peg_3y_sentiment') if peg_data else None,
        'peg_sentiment': peg_sentiment,
        'peg_guide': peg_guide,
        'peg_pe': peg_data.get('pe') if peg_data else None,
        'peg_growth_1y_pct': peg_data.get('growth_1y_pct') if peg_data else None,
        'peg_growth_3y_cagr_pct': peg_data.get('growth_3y_cagr_pct') if peg_data else None,
        'peg_growth_pct': peg_data.get('growth_1y_pct') if peg_data else None,
        'peg_source': peg_data.get('source') if peg_data else None,
        'yf_peg_ratio': peg_data.get('yf_peg_ratio') if peg_data else None,
        'adx': adx_dmi_data.get('adx') if adx_dmi_data else None,
        'plus_di': adx_dmi_data.get('plus_di') if adx_dmi_data else None,
        'minus_di': adx_dmi_data.get('minus_di') if adx_dmi_data else None,
        'adx_direction': adx_dmi_data.get('direction') if adx_dmi_data else None,
        'adx_strength': adx_dmi_data.get('strength') if adx_dmi_data else None,
        'adx_summary': adx_dmi_data.get('summary') if adx_dmi_data else None,
        'adx_buy_count': adx_dmi_data.get('buy_signal_count') if adx_dmi_data else None,
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
        
        f.write(f"현재가: {currency_symbol}{signal_info['current_price']:.2f}\n")
        
        # 검색기간 수익률 정보 추가 (안전한 처리)
        try:
            period_return = calculate_period_return(ohlcv_data, ohlcv_data.index[0], ohlcv_data.index[-1])
            if period_return:
                f.write(f"검색기간: {period_return['start_date']} → {period_return['end_date']}\n")
                f.write(f"시작가: ${period_return['start_price']:.2f}\n")
                f.write(f"종료가: ${period_return['end_price']:.2f}\n")
                f.write(f"기간수익률: {period_return['direction']} {period_return['return_percentage']:.2f}% (${period_return['return_amount']:.2f})\n")
            else:
                f.write(f"검색기간 수익률: 계산 불가\n")
        except Exception as e:
            f.write(f"검색기간 수익률: 오류 - {str(e)}\n")
        
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
            f.write(f"실시간가: {currency_symbol}{realtime_price:.2f} ({realtime_datetime} KST)\n")
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
        f.write(f"Wilshire 5000 시가총액: {signal_info['wilshire_market_cap']}조 달러 ({signal_info['wilshire_date']})\n")
        gdp_yoy = signal_info.get('us_gdp_yoy')
        yoy_txt = f", 전년비 {gdp_yoy:+.1f}%" if gdp_yoy is not None else ""
        f.write(f"US GDP: {signal_info['us_gdp']}조 달러 ({signal_info['gdp_date']}{yoy_txt})\n")
        if signal_info.get('fear_greed_score') is not None:
            f.write(
                f"공포탐욕: {signal_info['fear_greed_score']} "
                f"({signal_info.get('fear_greed_rating', 'N/A')})\n"
            )
            f.write(f"공포탐욕 심리: {signal_info.get('fear_greed_sentiment', 'N/A')}\n")
            f.write(f"공포탐욕 가이드: {signal_info.get('fear_greed_guide', 'N/A')}\n")
        if peg_data is not None:
            f.write(
                f"PEG 1년: {signal_info.get('peg_1y', 'N/A')} "
                f"({signal_info.get('peg_1y_sentiment', 'N/A')}), "
                f"EPS YoY: {signal_info.get('peg_growth_1y_pct', 'N/A')}%\n"
            )
            f.write(
                f"PEG 3년: {signal_info.get('peg_3y', 'N/A')} "
                f"({signal_info.get('peg_3y_sentiment', 'N/A')}), "
                f"EPS CAGR: {signal_info.get('peg_growth_3y_cagr_pct', 'N/A')}%\n"
            )
            if signal_info.get('peg_pe') is not None:
                f.write(f"PER: {signal_info['peg_pe']}\n")
            if signal_info.get('yf_peg_ratio') is not None:
                f.write(f"Yahoo pegRatio(참고): {signal_info['yf_peg_ratio']}\n")
        if signal_info.get('adx') is not None:
            f.write(
                f"추세(ADX/DMI): {signal_info.get('adx_summary')} "
                f"(ADX={signal_info.get('adx')}, "
                f"DMI +DI={signal_info.get('plus_di')}, "
                f"−DI={signal_info.get('minus_di')})\n"
            )
            if signal_info.get('adx_buy_count'):
                f.write(f"ADX buy 신호(+DI↑−DI & ADX≥25): {signal_info['adx_buy_count']}건\n")
        f.write(f"=== 신호 요약 ===\n")
        f.write(f"{signal_info['signal']}\n")
    
    print(f"신호 파일 저장 완료: {signal_file_path}")
    
    # 매수시그널 테이블 파일 경로 출력
    buy_signals_table_path = f"output/hma_mantra/{ticker}/{ticker}_buy_signals_table.txt"
    if os.path.exists(buy_signals_table_path):
        print(f"매수시그널 테이블: {buy_signals_table_path}")

    if show_adx_dmi and adx_dmi_data:
        buy_sigs = adx_dmi_data.get("buy_signals") or []
        if buy_sigs:
            try:
                plot_adx_buy_on_main(ax_main, buy_sigs, ohlcv_data)
            except Exception as e:
                print(f"ADX buy 메인 표시 실패: {e}")

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
        if show_pattern_strip and ax_pattern is not None:
            ax_pattern.axvline(dt, color='magenta', linestyle='--', alpha=0.7, linewidth=1.0, zorder=10)
        
        # 종가 기준 수평선 (거래량에 따른 두께 적용)
        close = ohlcv_data.loc[dt, 'Close']
        open_ = ohlcv_data.loc[dt, 'Open']
        volume = ohlcv_data.loc[dt, 'Volume']
        
        # 거래량에 따른 두께 결정 - thin 스타일
        if volume >= volume_67:
            linewidth = 0.4  # 상위 (높은 거래량) - thin 스타일
        elif volume >= volume_33:
            linewidth = 0.3  # 중위 (보통 거래량) - thin 스타일
        else:
            linewidth = 0.2  # 하위 (낮은 거래량) - thin 스타일
        
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
    
    # POC (Point of Control) 표시 (최신일자까지 연장) - thin 스타일
    poc_xmin = (overlay_start - main_xlim[0]) / (main_xlim[1] - main_xlim[0])
    poc_xmax = 1.0  # 최신일자까지 연장
    ax_main.axhline(poc_price, color='red', linestyle='--', alpha=0.8, linewidth=0.6, 
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
        
        # 최근 6개월 POC (검은색 점선) - 우측 끝까지 연장 - thin 스타일
        recent_poc_xmin = (center_overlay_start - main_xlim[0]) / (main_xlim[1] - main_xlim[0])
        recent_poc_xmax = 1.0  # 우측 끝까지 연장
        ax_main.axhline(recent_poc_price, color='black', linestyle=':', alpha=0.8, linewidth=0.6, 
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
    
    # RSI 차트 표시 (크로스 신호 포함)
    # rsi_cross_signals가 정의되지 않은 경우 기본값 사용
    if 'rsi_cross_signals' not in locals():
        rsi_cross_signals = detect_rsi_cross_signals(rsi3, rsi14, rsi_signal)
    plot_rsi_cross_signals(ax_rsi, rsi3, rsi14, rsi_signal, rsi_cross_signals)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_title('RSI (3/14/Signal) - 크로스 신호')
    ax_rsi.set_ylabel('RSI')
    
    # RSI 상관관계 범례 추가 (신호 강도별 구분)
    legend_elements = [
        # RSI 라인들
        plt.Line2D([0], [0], color='red', linewidth=2, label='RSI(3) - 초단기'),
        plt.Line2D([0], [0], color='blue', linewidth=2, label='RSI(14) - 단중기'),
        plt.Line2D([0], [0], color='orange', linewidth=2, label='RSI Signal - 추세선'),
        # 구간별 배경
        plt.Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.1, label='과매도 구간 (0-30)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.1, label='과매수 구간 (70-100)'),
        # 골든크로스 신호 강도별
        plt.Line2D([0], [0], marker='^', color='darkgreen', linestyle='None', markersize=10, label='골든크로스 강함 (과매도)'),
        plt.Line2D([0], [0], marker='^', color='green', linestyle='None', markersize=8, label='골든크로스 중간 (중립)'),
        plt.Line2D([0], [0], marker='^', color='lightgreen', linestyle='None', markersize=6, label='골든크로스 약함 (과매수)'),
        # 데드크로스 신호 강도별
        plt.Line2D([0], [0], marker='v', color='darkred', linestyle='None', markersize=10, label='데드크로스 강함 (과매수)'),
        plt.Line2D([0], [0], marker='v', color='red', linestyle='None', markersize=8, label='데드크로스 중간 (중립)'),
        plt.Line2D([0], [0], marker='v', color='lightcoral', linestyle='None', markersize=6, label='데드크로스 약함 (과매도)')
    ]
    ax_rsi.legend(handles=legend_elements, fontsize=6, loc='upper left', framealpha=0.9)
    ax_rsi.grid(True, alpha=0.3)

    # MACD 차트 표시
    macd_colors = ['green' if v >= 0 else 'red' for v in hist]
    ax_macd.bar(ohlcv_data.index, hist, color=macd_colors, alpha=0.5, width=0.8, label='Histogram')
    ax_macd.plot(ohlcv_data.index, macd, color='tab:blue', linewidth=0.42, label='MACD')
    ax_macd.plot(ohlcv_data.index, macd_signal, color='tab:orange', linewidth=0.35, label='Signal')
    ax_macd.axhline(0, color='black', linewidth=0.105, alpha=0.6)
    ax_macd.set_title('MACD (12,26,9)')
    ax_macd.set_ylabel('MACD')
    ax_macd.legend(fontsize=8, loc='upper left')
    ax_macd.grid(True, alpha=0.3)

    if show_adx_dmi and ax_adx is not None and adx_dmi_data is not None:
        try:
            adx_series = adx_dmi_data.get("series")
            if adx_series is not None and not adx_series.empty:
                ax_adx.plot(
                    adx_series.index,
                    adx_series["Plus_DI"],
                    color="#27ae60",
                    linewidth=0.8,
                    alpha=0.9,
                    label="+DI",
                )
                ax_adx.plot(
                    adx_series.index,
                    adx_series["Minus_DI"],
                    color="#c0392b",
                    linewidth=0.8,
                    alpha=0.9,
                    label="-DI",
                )
                ax_adx.plot(
                    adx_series.index,
                    adx_series["ADX"],
                    color="#000000",
                    linewidth=1.2,
                    alpha=0.95,
                    label="ADX",
                )
                ax_adx.axhline(
                    float(adx_sideways),
                    color="#7f8c8d",
                    linestyle=":",
                    linewidth=0.7,
                    alpha=0.8,
                    label=f"횡보 {adx_sideways:.0f}",
                )
                ax_adx.axhline(
                    float(adx_trend),
                    color="#f39c12",
                    linestyle="--",
                    linewidth=0.7,
                    alpha=0.85,
                    label=f"추세 {adx_trend:.0f}",
                )
                ax_adx.axhline(
                    float(adx_strong),
                    color="#8e44ad",
                    linestyle="--",
                    linewidth=0.7,
                    alpha=0.85,
                    label=f"강추세 {adx_strong:.0f}",
                )
                ax_adx.set_ylim(0, max(55, float(adx_series[["ADX", "Plus_DI", "Minus_DI"]].max().max()) * 1.08))
                title_suffix = adx_dmi_data.get("summary", "")
                ax_adx.set_title(
                    f"ADX/DMI ({adx_period}) — {title_suffix}",
                    fontsize=9,
                )
                ax_adx.set_ylabel("ADX / DMI", fontsize=8)
                ax_adx.legend(loc="upper left", fontsize=6, framealpha=0.85, ncol=3)
                ax_adx.grid(True, alpha=0.25, linestyle=":")
        except Exception as e:
            print(f"ADX/DMI subplot 표시 실패: {e}")
    
    # 금리 통합 차트 표시 (FFR + TNX + HY + 인플레이션)
    rates_plotted = False
    
    if ffr_data is not None and not ffr_data.empty:
        # FFR: 색각이상 친화 팔레트(vermillion)
        ax_rates.plot(
            ffr_data.index,
            ffr_data,
            color='#D55E00',
            linewidth=1.3,
            label='FFR (연방기금금리)'
        )
        rates_plotted = True
    
    if tnx_data is not None and not tnx_data.empty:
        # TNX: 색각이상 친화 팔레트(blue)
        ax_rates.plot(
            tnx_data.index,
            tnx_data,
            color='#0072B2',
            linewidth=1.3,
            label='TNX (10년물 국채금리)'
        )
        rates_plotted = True

    if hy_spread_data is not None and not hy_spread_data.empty:
        # HY OAS: FFR·TNX와 같은 왼쪽 축(%)에서 수준 비교
        ax_rates.plot(
            hy_spread_data.index,
            hy_spread_data,
            color='#882255',
            linewidth=1.3,
            linestyle='-.',
            label='HY (하이일드 스프레드)'
        )
        rates_plotted = True

    if inflation_data is not None and not inflation_data.empty:
        ax_rates.plot(
            inflation_data.index,
            inflation_data,
            color='#009E73',
            linewidth=1.5,
            linestyle='--',
            label='인플레이션 (CPI 전년비)'
        )
        rates_plotted = True
        print(f"✅ 인플레이션 차트 표시: {len(inflation_data)}개 데이터")
    
    if rates_plotted:
        ax_rates.set_title('금리 통합 차트 (FFR + TNX + HY + 인플레이션)')
        ax_rates.set_ylabel('금리 (%)')
        ax_rates.legend(fontsize=8, loc='upper left')
        ax_rates.grid(True, alpha=0.3)
    else:
        ax_rates.text(0.5, 0.5, '금리 데이터 없음', ha='center', va='center', 
                     transform=ax_rates.transAxes, fontsize=10, color='gray')
        ax_rates.set_title('금리 통합 차트 (FFR + TNX + HY + 인플레이션)')
        ax_rates.set_ylabel('금리 (%)')
    
    # =====================
    # 시장 심리 통합 지표 차트 생성
    # =====================
    try:
        # PCR, VIX, NAIIM 데이터 추출
        pcr_data = pcr_series if pcr_series is not None else pcr  # 기간별 PCR 시리즈 우선 사용
        vix_data = vix if not vix.empty else None  # vix는 이미 Series
        naiim_data = naaim
        
        print(f"데이터 검증: PCR={pcr_data is not None}, VIX={vix_data is not None}, NAIIM={naiim_data is not None}")
        
        if pcr_data is not None and vix_data is not None and naiim_data is not None:
            # 데이터 인덱스 맞추기
            common_dates = ohlcv_data.index.intersection(vix_data.index).intersection(naiim_data.index)
            
            print(f"공통 날짜 수: {len(common_dates)}")
            
            if len(common_dates) > 0:
                # PCR 데이터를 common_dates에 맞춰 정렬
                if hasattr(pcr_data, 'loc'):
                    try:
                        pcr_aligned = pcr_data.loc[common_dates]
                        print(f"PCR 데이터 정렬 완료: {len(pcr_aligned)}개 데이터")
                    except Exception as e:
                        print(f"PCR 데이터 정렬 실패: {e}, 기본값 사용")
                        pcr_aligned = pd.Series([1.0] * len(common_dates), index=common_dates)
                else:
                    # PCR이 단일 값인 경우 Series로 변환
                    try:
                        pcr_aligned = pd.Series([float(pcr_data)] * len(common_dates), index=common_dates)
                        print(f"PCR 단일값을 시리즈로 변환: {len(pcr_aligned)}개 데이터")
                    except (ValueError, TypeError):
                        # PCR 변환 실패 시 기본값 사용
                        pcr_aligned = pd.Series([1.0] * len(common_dates), index=common_dates)
                        print(f"PCR 데이터 변환 실패, 기본값 1.0 사용")
                
                # VIX와 NAIIM을 Series로 변환
                if isinstance(vix_data, pd.DataFrame):
                    vix_aligned = vix_data.iloc[:, 0].loc[common_dates]  # 첫 번째 컬럼 선택
                else:
                    vix_aligned = vix_data.loc[common_dates]
                
                if isinstance(naiim_data, pd.DataFrame):
                    naiim_aligned = naiim_data.iloc[:, 0].loc[common_dates]  # 첫 번째 컬럼 선택
                else:
                    naiim_aligned = naiim_data.loc[common_dates]
                
                print(f"PCR 데이터 최종 형태: {type(pcr_aligned)}, 길이: {len(pcr_aligned) if hasattr(pcr_aligned, '__len__') else 'N/A'}")

                fg_aligned = None
                try:
                    fg_hist = fetch_cnn_fear_greed_history()
                    if fg_hist is not None:
                        fg_aligned = _align_series_to_dates(fg_hist, common_dates)
                        fg_finite = int(np.isfinite(pd.to_numeric(fg_aligned, errors="coerce")).sum())
                        print(f"CNN F&G 차트 정렬: {fg_finite}/{len(common_dates)}")
                except Exception as e:
                    print(f"CNN F&G 이력 정렬 실패: {e}")
                    fg_aligned = None
                
                # 통합 차트 생성. 왼쪽 축은 PCR 대신 CNN F&G.
                try:
                    print(f"차트 생성 전 데이터 타입: PCR={type(pcr_aligned)}, VIX={type(vix_aligned)}, NAIIM={type(naiim_aligned)}")
                    print(f"차트 생성 전 데이터 길이: PCR={len(pcr_aligned) if hasattr(pcr_aligned, '__len__') else 'N/A'}, VIX={len(vix_aligned)}, NAIIM={len(naiim_aligned)}")
                    create_integrated_market_sentiment_chart(
                        ax_sentiment, common_dates, pcr_aligned, vix_aligned, naiim_aligned, fg_aligned
                    )
                except Exception as e:
                    print(f"통합 차트 생성 오류: {e}")
                    import traceback
                    traceback.print_exc()
                
                # 시장 심리 분석 결과 표시
                try:
                    # PCR 값 추출 (단일 값 또는 Series의 마지막 값)
                    if hasattr(pcr_aligned, 'iloc'):
                        pcr_value = pcr_aligned.iloc[-1]
                    else:
                        pcr_value = pcr_aligned
                    
                    sentiment_signals = analyze_integrated_market_sentiment(pcr_value, vix_aligned.iloc[-1], naiim_aligned.iloc[-1])
                    fg_last = None
                    if fg_aligned is not None:
                        fg_num = pd.to_numeric(fg_aligned, errors="coerce")
                        fg_ok = fg_num[np.isfinite(fg_num.to_numpy(dtype=float))]
                        if len(fg_ok):
                            fg_last = float(fg_ok.iloc[-1])
                    if fg_last is not None:
                        vix_score = float(np.clip((float(vix_aligned.iloc[-1]) - 10) / 40 * 100, 0, 100))
                        naaim_score = float(np.clip(float(naiim_aligned.iloc[-1]), 0, 100))
                        composite_score = fg_last * 0.3 + vix_score * 0.4 + naaim_score * 0.3
                    else:
                        composite_score = calculate_comprehensive_sentiment_index(
                            pcr_value, vix_aligned.iloc[-1], naiim_aligned.iloc[-1]
                        )
                    
                    # 차트 타이틀에 종합 점수 추가
                    ax_sentiment.set_title('CNN F&G vs. VIX', fontsize=10, fontweight='bold')
                    
                    fg_log = f"{fg_last:.1f}" if fg_last is not None else "없음"
                    print(f"시장 심리 통합 차트 생성 완료: F&G={fg_log}, VIX={vix_aligned.iloc[-1]:.2f}, NAIIM={naiim_aligned.iloc[-1]:.1f}")
                except Exception as e:
                    print(f"시장 심리 분석 오류: {e}")
                    ax_sentiment.set_title('CNN F&G vs. VIX')
            else:
                ax_sentiment.text(0.5, 0.5, '공통 데이터 없음', ha='center', va='center', 
                                transform=ax_sentiment.transAxes, fontsize=12, color='gray')
                ax_sentiment.set_title('CNN F&G vs. VIX')
        else:
            ax_sentiment.text(0.5, 0.5, '시장 심리 데이터 부족', ha='center', va='center', 
                            transform=ax_sentiment.transAxes, fontsize=12, color='gray')
            ax_sentiment.set_title('CNN F&G vs. VIX')
    except Exception as e:
        print(f"시장 심리 통합 차트 생성 오류: {e}")
        ax_sentiment.text(0.5, 0.5, f'차트 생성 오류: {str(e)}', ha='center', va='center', 
                         transform=ax_sentiment.transAxes, fontsize=10, color='red')
        ax_sentiment.set_title('CNN F&G vs. VIX')
    
    ax_sentiment.grid(True, alpha=0.3)

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
    # RSI 크로스 신호 감지
    # =====================
    rsi_cross_signals = detect_rsi_cross_signals(rsi3, rsi14, rsi_signal)
    print(f"RSI 크로스 신호 감지: 골든크로스 {len(rsi_cross_signals['golden_crosses'])}개, 데드크로스 {len(rsi_cross_signals['dead_crosses'])}개")
    
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

    _enabled_main = _enabled_main_early
    _need_pattern_analysis = show_pattern_strip or bool(_enabled_main)
    if _need_pattern_analysis:
        try:
            if _pat_events_cached is not None:
                pattern_box_ranges = _pattern_box_ranges_cached or []
                pat_events = _pat_events_cached
            else:
                # 차트 패턴 전용: 기하 창만(box_range_windows). VP·표시용 로그 없음 — calculate_box_ranges 와 분리
                pattern_box_ranges = compute_box_range_windows(
                    ohlcv_data,
                    pattern_range_box_period,
                    pattern_range_box_num,
                    pattern_range_box_overlap,
                    pattern_range_box_avoid_time_overlap,
                )
                pat_events = analyze_seven_criteria(ohlcv_data, box_ranges=pattern_box_ranges)
            print(
                f"패턴 range_box 후보 창: 기간={pattern_range_box_period}일, "
                f"개수={pattern_range_box_num}, 겹침={pattern_range_box_overlap}일 "
                f"(box_range_windows · 메인 파란 박스와 별도)"
            )
            # 박스권(range_box) 후보를 먼저 넣어 분석 → 스트립/메인은 동일 pat_events 사용
            _rb_ev = sum(1 for e in pat_events if e.get("pattern_id") == "range_box")
            print(
                f"차트 패턴 분석: {len(pat_events)}개 구간 "
                f"(range_box={_rb_ev}건, 응축 조건 통과분만·그 외 휴리스틱)"
            )
            if pattern_box_ranges and _rb_ev == 0 and (
                show_pattern_strip or ("range_box" in _enabled_main)
            ):
                print(
                    "  참고: 패턴용 박스 후보는 있으나 range_box 응축 필터(횡보·변동폭 등)를 통과한 구간이 없습니다."
                )
            if show_pattern_strip and ax_pattern is not None:
                n_shown = plot_pattern_strip(ax_pattern, ohlcv_data, pat_events)
                print(f"  패턴 스트립 표시 행: {n_shown} (감지된 종류만)")
                ax_pattern.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax_pattern.tick_params(axis='x', labelbottom=False)
            if _enabled_main:
                plot_pattern_main_overlays(
                    ax_main,
                    ohlcv_data,
                    pat_events,
                    _enabled_main,
                    range_box_main_max=pattern_range_box_main_max,
                    main_min_confidence=pattern_main_min_confidence,
                )
        except Exception as e:
            print(f"차트 패턴(스트립/메인) 표시 실패: {e}")

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
    legend_handles, legend_labels = ax_main.get_legend_handles_labels()
    ma_order = ["MA 20", "MA 50", "MA 120", "MA 200"]
    ma_items = []
    other_items = []
    first_ma = None
    for handle, label in zip(legend_handles, legend_labels):
        if label in ma_order:
            if first_ma is None:
                first_ma = len(other_items)
            ma_items.append((handle, label))
        else:
            other_items.append((handle, label))
    ma_items.sort(key=lambda item: ma_order.index(item[1]))
    if first_ma is None:
        ordered_legend = list(zip(legend_handles, legend_labels))
    else:
        ordered_legend = other_items[:]
        for offset, item in enumerate(ma_items):
            ordered_legend.insert(first_ma + offset, item)
    ax_main.legend(
        [handle for handle, _label in ordered_legend],
        [label for _handle, label in ordered_legend],
        loc='upper left',
        fontsize=8,
    )
    
    # x축 날짜 포맷 설정
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_main.tick_params(axis='x', rotation=45)

    # 레이아웃 조정. 우측 여백은 가격 차트 높이의 주황 요약 박스용
    plt.subplots_adjust(left=0.08, right=0.74, top=0.95, bottom=0.06, hspace=0.12)
    if ax_realtime is not None:
        main_pos = ax_main.get_position()
        gap = 0.012
        box_left = main_pos.x1 + gap
        box_width = max(0.995 - box_left, 0.16)
        ax_realtime.set_position([box_left, main_pos.y0, box_width, main_pos.height])

    # 저장 또는 표시 (폰트 경고 방지)
    if save_path:
        # 폰트 경고를 방지하기 위해 matplotlib 설정 조정
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*missing from font.*")
            plt.savefig(save_path, dpi=300)  # bbox_inches='tight' 제거
        plt.close()
    else:
        plt.show() 

def create_integrated_market_sentiment_chart(ax, dates, pcr_data, vix_data, naiim_data, fg_data=None):
    """CNN F&G, VIX, NAIIM을 3축으로 통합한 차트. 왼쪽 축은 F&G."""
    
    print(f"=== 시장 심리 통합 차트 생성 디버깅 ===")
    print(f"입력 데이터 타입: PCR={type(pcr_data)}, VIX={type(vix_data)}, NAIIM={type(naiim_data)}")
    print(f"입력 데이터 길이: PCR={len(pcr_data) if hasattr(pcr_data, '__len__') else 'N/A'}, VIX={len(vix_data)}, NAIIM={len(naiim_data)}")
    print(f"dates 길이: {len(dates)}")
    
    if pcr_data is None or vix_data is None or naiim_data is None:
        print("❌ 데이터가 None입니다")
        ax.text(0.5, 0.5, '시장 심리 데이터 없음', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12, color='gray')
        return
    
    # 데이터가 Series인지 확인하고 numpy 배열로 변환
    if hasattr(pcr_data, 'values'):
        print(f"PCR 데이터를 numpy 배열로 변환: {type(pcr_data.values)}")
        pcr_data = pcr_data.values
    if hasattr(vix_data, 'values'):
        print(f"VIX 데이터를 numpy 배열로 변환: {type(vix_data.values)}")
        vix_data = vix_data.values
    if hasattr(naiim_data, 'values'):
        print(f"NAIIM 데이터를 numpy 배열로 변환: {type(naiim_data.values)}")
        naiim_data = naiim_data.values
    
    print(f"변환 후 데이터 타입: PCR={type(pcr_data)}, VIX={type(vix_data)}, NAIIM={type(naiim_data)}")
    print(f"변환 후 데이터 길이: PCR={len(pcr_data)}, VIX={len(vix_data)}, NAIIM={len(naiim_data)}")
    
    # PCR 데이터 샘플 출력
    if len(pcr_data) > 0:
        print(f"PCR 데이터 샘플 (처음 5개): {pcr_data[:5]}")
        print(f"PCR 데이터 샘플 (마지막 5개): {pcr_data[-5:]}")
    
    # 데이터 길이 확인
    if len(pcr_data) != len(dates) or len(vix_data) != len(dates) or len(naiim_data) != len(dates):
        ax.text(0.5, 0.5, '데이터 길이 불일치', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12, color='gray')
        return
    
    # NaN 값 처리 및 데이터 정규화 (PCR 범위를 실제 데이터에 맞게 조정)
    # NaN 값이 있는 경우 제거하거나 기본값으로 대체
    if np.any(np.isnan(pcr_data)):
        print(f"⚠️  PCR 데이터에 NaN 값이 {np.sum(np.isnan(pcr_data))}개 있습니다")
        # NaN 값을 이전 값으로 채우거나 기본값 사용
        pcr_data_clean = pd.Series(pcr_data).fillna(method='ffill').fillna(method='bfill').fillna(0.5)
        pcr_data = pcr_data_clean.values
        print(f"PCR 데이터 NaN 처리 완료: {pcr_data[:5]} ... {pcr_data[-5:]}")
    
    pcr_min, pcr_max = pcr_data.min(), pcr_data.max()
    pcr_range = pcr_max - pcr_min
    
    # PCR 정규화 범위를 실제 데이터에 맞게 동적 조정
    if pcr_range < 0.1:  # PCR 변화가 매우 작은 경우
        pcr_buffer = 0.05  # 5% 버퍼 추가
    else:
        pcr_buffer = pcr_range * 0.1  # 10% 버퍼 추가
    
    pcr_normalized = np.clip(pcr_data, 
                            pcr_min - pcr_buffer, 
                            pcr_max + pcr_buffer)
    
    print(f"PCR 정규화: 원본 범위 [{pcr_min:.3f}, {pcr_max:.3f}] → 정규화 범위 [{pcr_min - pcr_buffer:.3f}, {pcr_max + pcr_buffer:.3f}]")
    
    vix_normalized = np.clip(vix_data, 10, 70)  # VIX 범위를 실제 시장 극값에 맞춰 확장
    naiim_normalized = np.clip(naiim_data, 0, 100)
    
    print(f"정규화 후 PCR 데이터: {pcr_normalized[:5]} ... {pcr_normalized[-5:]}")
    print(f"정규화 후 VIX 데이터: {vix_normalized[:5]} ... {vix_normalized[-5:]}")
    print(f"정규화 후 NAIIM 데이터: {naiim_normalized[:5]} ... {naiim_normalized[-5:]}")
    
    # 1차 Y축: CNN F&G (왼쪽, 0~100). PCR 선은 그리지 않음.
    fg_plot = None
    if fg_data is not None:
        fg_plot = pd.to_numeric(pd.Series(fg_data), errors="coerce").to_numpy(dtype=float)
        if len(fg_plot) != len(dates):
            print(f"⚠️ CNN F&G 길이 불일치: {len(fg_plot)} != {len(dates)}")
            fg_plot = None
    if fg_plot is None or not np.any(np.isfinite(fg_plot)):
        fg_plot = np.full(len(dates), np.nan)
        print("⚠️ CNN F&G 시계열 없음 — 왼쪽 축은 비움")
    line1 = ax.plot(dates, fg_plot, color="#4C78A8", linewidth=2.1, label="CNN F&G", alpha=0.95, zorder=10)
    ax.set_ylim(0, 100)
    ax.set_ylabel("CNN F&G", color="#4C78A8", fontsize=10, fontweight="bold")
    ax.tick_params(axis="y", labelcolor="#4C78A8")
    finite_idx = np.where(np.isfinite(fg_plot))[0]
    
    # 2차 Y축: VIX (오른쪽)
    ax2 = ax.twinx()
    line2 = ax2.plot(dates, vix_normalized, 'r-', linewidth=1.4, label='VIX', alpha=0.8)
    ax2.set_ylabel('VIX (변동성 지수)', color='red', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='red')
    
    # 3차 Y축: NAIIM (오른쪽, 별도 스케일)
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # 오른쪽에서 60px 떨어짐
    line3 = ax3.plot(dates, naiim_normalized, 'g-', linewidth=1.4, label='NAIIM', alpha=0.8)
    ax3.set_ylabel('NAIIM (기관투자자)', color='green', fontsize=10)
    ax3.tick_params(axis='y', labelcolor='green')
    
    ax3.axhline(y=50, color='green', linestyle='--', alpha=0.35, label='NAIIM 중립')

    # 왼쪽 CNN F&G 구간 표시. 배경 전체는 칠하지 않는다.
    from matplotlib.transforms import blended_transform_factory
    zone_trans = blended_transform_factory(ax.transAxes, ax.transData)
    fg_zones = (
        (0, 25, "극공포", "darkred", "white"),
        (25, 45, "공포", "red", "white"),
        (45, 55, "중립", "orange", "black"),
        (55, 75, "탐욕", "green", "white"),
        (75, 100, "극탐욕", "darkgreen", "white"),
    )
    zone_w = 0.036
    for y0, y1, name, face, ink in fg_zones:
        ax.add_patch(mpatches.Rectangle(
            (0.0, y0), zone_w, y1 - y0,
            transform=zone_trans,
            facecolor=face,
            edgecolor="white",
            linewidth=0.4,
            alpha=0.95,
            zorder=12,
            clip_on=True,
        ))
        ax.text(
            zone_w / 2, (y0 + y1) / 2, name,
            transform=zone_trans,
            ha="center", va="center",
            fontsize=6.5, color=ink, fontweight="bold",
            zorder=13, clip_on=True,
        )

    # 오른쪽 VIX 전략. <15 위험선호, 15–25 선별 대응, 25–30 위험축소, ≥30 자본보전.
    vix_lo, vix_hi = ax2.get_ylim()
    vix_lo = min(vix_lo, 12)
    vix_hi = max(vix_hi, 32)
    ax2.set_ylim(vix_lo, vix_hi)
    vix_trans = blended_transform_factory(ax.transAxes, ax2.transData)
    vix_zone_w = 0.052
    vix_zones = (
        (vix_lo, 15, "위험선호\nRisk-On", "green", "white"),
        (15, 25, "선별대응\nCaution", "orange", "black"),
        (25, 30, "위험축소\nRisk-Off", "#C0392B", "white"),
        (30, vix_hi, "자본보전\n위기경계", "darkred", "white"),
    )
    for y0, y1, name, face, ink in vix_zones:
        start, end = max(y0, vix_lo), min(y1, vix_hi)
        if end - start < 0.6:
            continue
        ax2.add_patch(mpatches.Rectangle(
            (1.0 - vix_zone_w, start), vix_zone_w, end - start,
            transform=vix_trans,
            facecolor=face,
            edgecolor="white",
            linewidth=0.4,
            alpha=0.95,
            zorder=12,
            clip_on=True,
        ))
        ax2.text(
            1.0 - vix_zone_w / 2, (start + end) / 2, name,
            transform=vix_trans,
            ha="center", va="center",
            fontsize=6.2, color=ink, fontweight="bold",
            zorder=13, clip_on=True,
            linespacing=0.95,
        )

    # 구간 식별선. F&G는 파란 파선, VIX는 보라 점선으로 구분한다.
    for level in (25, 45, 55, 75):
        ax.axhline(
            y=level, color="#4C78A8", linestyle=(0, (6, 3)),
            linewidth=0.58, alpha=0.95, zorder=4,
        )
        ax.text(
            0.20, level, f"F&G {level}",
            transform=zone_trans,
            color="#4C78A8", fontsize=7, fontweight="bold",
            ha="left", va="bottom", zorder=15, clip_on=True,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="#4C78A8", linewidth=0.6, alpha=0.9),
        )
    for level in (15, 25, 30):
        ax2.axhline(
            y=level, color="#6C3483", linestyle=(0, (1.2, 1.5)),
            linewidth=0.62, alpha=0.95, zorder=4,
        )
        ax2.text(
            0.78, level, f"VIX {level}",
            transform=vix_trans,
            color="#6C3483", fontsize=7, fontweight="bold",
            ha="right", va="bottom", zorder=15, clip_on=True,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="#6C3483", linewidth=0.6, alpha=0.9),
        )

    if len(finite_idx):
        last_i = int(finite_idx[-1])
        ax.text(
            1.0 - vix_zone_w - 0.008,
            min(float(fg_plot[last_i]) + 6, 96),
            f"F&G {fg_plot[last_i]:.0f}",
            transform=zone_trans,
            color="#4C78A8",
            fontsize=8,
            fontweight="bold",
            ha="right",
            va="bottom",
            zorder=14,
            clip_on=True,
        )
    
    # 범례 통합
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax.legend(
        lines, labels, loc="upper left",
        bbox_to_anchor=(zone_w + 0.006, 1.0),
        fontsize=9, framealpha=0.9,
    )
    
    return ax, ax2, ax3

def analyze_integrated_market_sentiment(pcr_data, vix_data, naiim_data):
    """3개 지표 통합 시그널 분석"""
    
    try:
        # 입력값 검증 및 스칼라 변환
        if pcr_data is None or vix_data is None or naiim_data is None:
            return ["시장 심리 데이터 부족"]
        
        # pandas Series나 numpy array인 경우 스칼라 값으로 변환
        if hasattr(pcr_data, 'item'):
            pcr_value = pcr_data.item()
        elif hasattr(pcr_data, 'iloc'):
            pcr_value = float(pcr_data.iloc[-1]) if len(pcr_data) > 0 else None
        else:
            pcr_value = float(pcr_data) if pcr_data is not None else None
            
        if hasattr(vix_data, 'item'):
            vix_value = vix_data.item()
        elif hasattr(vix_data, 'iloc'):
            vix_value = float(vix_data.iloc[-1]) if len(vix_data) > 0 else None
        else:
            vix_value = float(vix_data) if vix_data is not None else None
            
        if hasattr(naiim_data, 'item'):
            naaim_value = naiim_data.item()
        elif hasattr(naiim_data, 'iloc'):
            naaim_value = float(naiim_data.iloc[-1]) if len(naiim_data) > 0 else None
        else:
            naaim_value = float(naiim_data) if naiim_data is not None else None
        
        # 최종 검증
        if pcr_value is None or vix_value is None or naaim_value is None:
            return ["시장 심리 데이터 변환 실패"]
        
        signals = []
        
        # PCR 분석 (Put-Call Ratio)
        if pcr_value > 1.5:
            signals.append("PCR: 극한 과매도 (매수 신호)")
        elif pcr_value > 1.0:
            signals.append("PCR: 과매도 구간")
        elif pcr_value < 0.5:
            signals.append("PCR: 극한 과매수 (매도 신호)")
        elif pcr_value < 1.0:
            signals.append("PCR: 과매수 구간")
        else:
            signals.append("PCR: 중립 구간")
        
        # VIX 전략 구간
        if vix_value < 15:
            signals.append("VIX: 위험선호 (Risk-On)")
        elif vix_value < 25:
            signals.append("VIX: 선별 대응 (Caution)")
        elif vix_value < 30:
            signals.append("VIX: 위험축소 (Risk-Off)")
        else:
            signals.append("VIX: 자본보전 (위기 경계)")
        
        # NAIIM 분석 (기관투자자)
        if naaim_value < 30:
            signals.append("NAIIM: 기관 극한 매도 (매수 신호)")
        elif naaim_value < 50:
            signals.append("NAIIM: 기관 매도 구간")
        elif naaim_value > 70:
            signals.append("NAIIM: 기관 극한 매수 (매도 신호)")
        elif naaim_value > 50:
            signals.append("NAIIM: 기관 매수 구간")
        else:
            signals.append("NAIIM: 기관 중립 구간")
        
        return signals
        
    except Exception as e:
        print(f"시장 심리 분석 함수 오류: {e}")
        return [f"분석 오류: {str(e)}"]

def calculate_comprehensive_sentiment_index(pcr_data, vix_data, naiim_data):
    """PCR, VIX, NAIIM을 종합한 시장 심리 지수"""
    
    try:
        # 입력값 검증 및 스칼라 변환
        if pcr_data is None or vix_data is None or naiim_data is None:
            return None
        
        # pandas Series나 numpy array인 경우 스칼라 값으로 변환
        if hasattr(pcr_data, 'item'):
            pcr_value = pcr_data.item()
        elif hasattr(pcr_data, 'iloc'):
            pcr_value = float(pcr_data.iloc[-1]) if len(pcr_data) > 0 else None
        else:
            pcr_value = float(pcr_data) if pcr_data is not None else None
            
        if hasattr(vix_data, 'item'):
            vix_value = vix_data.item()
        elif hasattr(vix_data, 'iloc'):
            vix_value = float(vix_data.iloc[-1]) if len(vix_data) > 0 else None
        else:
            vix_value = float(vix_data) if vix_data is not None else None
            
        if hasattr(naiim_data, 'item'):
            naaim_value = naiim_data.item()
        elif hasattr(naiim_data, 'iloc'):
            naaim_value = float(naiim_data.iloc[-1]) if len(naiim_data) > 0 else None
        else:
            naaim_value = float(naiim_data) if naiim_data is not None else None
        
        # 최종 검증
        if pcr_value is None or vix_value is None or naaim_value is None:
            return None
        
        # 각 지표를 0~100 스케일로 정규화
        pcr_score = np.clip((pcr_value - 0.5) / 1.5 * 100, 0, 100)  # 0.5~2.0 → 0~100
        vix_score = np.clip((vix_value - 10) / 40 * 100, 0, 100)     # 10~50 → 0~100
        naaim_score = np.clip(naaim_value, 0, 100)  # 0~100 범위로 제한
        
        # 가중 평균 (PCR 30%, VIX 40%, NAIIM 30%)
        weights = [0.3, 0.4, 0.3]
        composite_score = (pcr_score * weights[0] + 
                          vix_score * weights[1] + 
                          naaim_score * weights[2])
        
        return composite_score
        
    except Exception as e:
        print(f"종합 심리 지수 계산 오류: {e}")
        return None

def calculate_historical_pcr_series(ticker, dates, current_pcr):
    """기간별 PCR 값을 계산하여 시리즈로 반환 (실제 데이터 기반 시뮬레이션)"""
    
    try:
        print(f"기간별 PCR 계산 시작: {ticker}, 기간: {len(dates)}일")
        
        # 현재 PCR을 기준으로 과거 데이터 시뮬레이션
        pcr_series = pd.Series(index=dates, dtype=float)
        
        # 최근 30일은 현재 PCR 값 사용
        recent_days = 30
        if len(dates) > recent_days:
            recent_start_idx = len(dates) - recent_days
            pcr_series.iloc[recent_start_idx:] = current_pcr
        
        # 과거 데이터는 시장 상황에 따라 변화하는 PCR 값 생성
        for i in range(len(dates) - recent_days):
            # 시장 상황에 따른 PCR 변화 시뮬레이션
            # 1. 주가 변동성에 따른 PCR 변화
            # 2. 시장 사이클에 따른 PCR 변화
            # 3. 랜덤 노이즈 추가
            
            base_pcr = current_pcr
            
            # 시장 사이클 변화 (장기 트렌드)
            cycle_factor = 1.0 + 0.2 * np.sin(2 * np.pi * i / len(dates))
            
            # 단기 변동성 (랜덤 워크)
            volatility_factor = 1.0 + 0.1 * np.random.normal(0, 1)
            
            # PCR 계산 (0.3 ~ 2.0 범위로 제한)
            historical_pcr = base_pcr * cycle_factor * volatility_factor
            historical_pcr = np.clip(historical_pcr, 0.3, 2.0)
            
            pcr_series.iloc[i] = historical_pcr
        
        # PCR 값 검증 및 해석
        pcr_min, pcr_max = pcr_series.min(), pcr_series.max()
        pcr_mean = pcr_series.mean()
        
        print(f"기간별 PCR 계산 완료:")
        print(f"  - 최소값: {pcr_min:.3f} ({get_pcr_interpretation(pcr_min)})")
        print(f"  - 최대값: {pcr_max:.3f} ({get_pcr_interpretation(pcr_max)})")
        print(f"  - 평균값: {pcr_mean:.3f} ({get_pcr_interpretation(pcr_mean)})")
        print(f"  - 현재값: {current_pcr:.3f} ({get_pcr_interpretation(current_pcr)})")
        
        return pcr_series
        
    except Exception as e:
        print(f"기간별 PCR 계산 오류: {e}")
        # 오류 발생 시 기본값 사용
        return pd.Series([current_pcr] * len(dates), index=dates)

def get_pcr_interpretation(pcr_value):
    """PCR 값에 대한 해석 반환"""
    if pcr_value < 0.5:
        return "극한 과매수 (매도 신호)"
    elif pcr_value < 0.7:
        return "과매수 (매도 고려)"
    elif pcr_value < 1.0:
        return "중립적 (관망)"
    elif pcr_value < 1.5:
        return "과매도 (매수 고려)"
    else:
        return "극한 과매도 (매수 신호)"

def generate_dynamic_naaim_series(dates, base_value=50.0):
    """시장 상황을 반영하는 동적 NAIIM 시리즈 생성"""
    
    try:
        print(f"동적 NAIIM 시리즈 생성 시작: {len(dates)}일, 기준값: {base_value}")
        
        naaim_series = pd.Series(index=dates, dtype=float)
        
        # 시장 사이클과 변동성을 반영한 NAIIM 값 생성
        for i, date in enumerate(dates):
            # 기본값에서 시작
            naaim_value = base_value
            
            # 시장 사이클 변화 (장기 트렌드)
            cycle_factor = 1.0 + 0.3 * np.sin(2 * np.pi * i / len(dates))
            
            # 단기 변동성 (랜덤 워크)
            volatility_factor = 1.0 + 0.2 * np.random.normal(0, 1)
            
            # NAIIM 계산 (20 ~ 80 범위로 제한)
            dynamic_naaim = naaim_value * cycle_factor * volatility_factor
            dynamic_naaim = np.clip(dynamic_naaim, 20, 80)
            
            naaim_series[date] = dynamic_naaim
        
        # NAIIM 값 검증 및 해석
        naaim_min, naaim_max = naaim_series.min(), naaim_series.max()
        naaim_mean = naaim_series.mean()
        
        print(f"동적 NAIIM 시리즈 생성 완료:")
        print(f"  - 최소값: {naaim_min:.1f} ({get_naaim_interpretation(naaim_min)})")
        print(f"  - 최대값: {naaim_max:.1f} ({get_naaim_interpretation(naaim_max)})")
        print(f"  - 평균값: {naaim_mean:.1f} ({get_naaim_interpretation(naaim_mean)})")
        
        return naaim_series
        
    except Exception as e:
        print(f"동적 NAIIM 시리즈 생성 오류: {e}")
        # 오류 발생 시 기본값 사용
        return pd.Series([base_value] * len(dates), index=dates)

def get_naaim_interpretation(naaim_value):
    """NAIIM 값에 대한 해석 반환"""
    if naaim_value < 30:
        return "기관 극한 매도 (매수 신호)"
    elif naaim_value < 50:
        return "기관 매도 구간"
    elif naaim_value > 70:
        return "기관 극한 매수 (매도 신호)"
    elif naaim_value > 50:
        return "기관 매수 구간"
    else:
        return "기관 중립 구간"

def add_target_price_lines(ax, ohlcv_data, target_buy_price=None, target_sell_price=None, stop_loss_price=None):
    """
    메인차트에 target 가격들을 수평선으로 표시합니다.
    
    Args:
        ax: 메인차트의 axes 객체
        ohlcv_data: OHLCV 데이터
        target_buy_price: Target 매수가
        target_sell_price: Target 목표가
        stop_loss_price: 손절가
    """
    if not any([target_buy_price, target_sell_price, stop_loss_price]):
        return
    
    # 차트의 x축 범위 (날짜)
    x_start = ohlcv_data.index[0]
    x_end = ohlcv_data.index[-1]
    
    # Target 매수가 표시 (파란색 점선)
    if target_buy_price is not None:
        ax.axhline(y=target_buy_price, xmin=0, xmax=1, color='blue', linestyle='--', 
                   linewidth=1.5, alpha=0.8, label=f'Target Buy: ${target_buy_price:.2f}')
        
        # 라벨을 차트 외부 우측에 배치 (수평라인 끝에 맞춰)
        # annotate를 사용해서 더 명확하게 표시
        ax.annotate(f'Target Buy: ${target_buy_price:.2f}', 
                    xy=(x_end, target_buy_price), xytext=(52, 0),
                    textcoords='offset points', fontsize=5.6, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', 
                              alpha=0.8, edgecolor='blue', linewidth=1),
                    color='darkblue', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7))
    
    # Target 목표가 표시 (초록색 점선)
    if target_sell_price is not None:
        ax.axhline(y=target_sell_price, xmin=0, xmax=1, color='green', linestyle='--', 
                   linewidth=1.5, alpha=0.8, label=f'Target Sell: ${target_sell_price:.2f}')
        
        # 라벨을 차트 외부 우측에 배치 (수평라인 끝에 맞춰)
        ax.annotate(f'Target Sell: ${target_sell_price:.2f}', 
                    xy=(x_end, target_sell_price), xytext=(52, 0),
                    textcoords='offset points', fontsize=5.6, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', 
                              alpha=0.8, edgecolor='green', linewidth=1),
                    color='darkgreen', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))
    
    # 손절가 표시 (빨간색 점선)
    if stop_loss_price is not None:
        ax.axhline(y=stop_loss_price, xmin=0, xmax=1, color='red', linestyle='--', 
                   linewidth=1.5, alpha=0.8, label=f'Stop Loss: ${stop_loss_price:.2f}')
        
        # 라벨을 차트 외부 우측에 배치 (수평라인 끝에 맞춰)
        ax.annotate(f'Stop Loss: ${stop_loss_price:.2f}', 
                    xy=(x_end, stop_loss_price), xytext=(52, 0),
                    textcoords='offset points', fontsize=5.6, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', 
                              alpha=0.8, edgecolor='red', linewidth=1),
                    color='darkred', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    
    # 범례 추가
    if any([target_buy_price, target_sell_price, stop_loss_price]):
        ax.legend(loc='upper left', fontsize=8, framealpha=0.8)

def calculate_target_prices_from_support_resistance(ohlcv_data, risk_reward_ratio=2.0):
    """
    지지선과 저항선을 기반으로 target 가격들을 계산합니다.
    
    Args:
        ohlcv_data: OHLCV 데이터
        risk_reward_ratio: 리스크 대비 보상 비율 (기본값: 2.0)
    
    Returns:
        dict: 계산된 target 가격들
    """
    try:
        # 지지선과 저항선 계산
        support, resistance = calculate_support_resistance(ohlcv_data)
        
        if support is None or resistance is None:
            return None
        
        current_price = ohlcv_data['Close'].iloc[-1]
        
        # Target 매수가: 지지선 근처
        target_buy_price = support * 1.02  # 지지선 위 2%
        
        # 손절가: 지지선 아래
        stop_loss_price = support * 0.98   # 지지선 아래 2%
        
        # Target 목표가: 리스크 대비 보상 비율에 따라 계산
        risk = current_price - stop_loss_price
        target_sell_price = current_price + (risk * risk_reward_ratio)
        
        return {
            'target_buy_price': target_buy_price,
            'target_sell_price': target_sell_price,
            'stop_loss_price': stop_loss_price,
            'support': support,
            'resistance': resistance,
            'risk_reward_ratio': risk_reward_ratio
        }
        
    except Exception as e:
        print(f"Target 가격 계산 오류: {e}")
        return None