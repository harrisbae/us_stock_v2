"""
엘리엇 파동 시각화 - 간단하고 명확한 버전

matplotlib 기반으로 파동을 명확하게 표시합니다.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import os
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from ..utils import get_available_font


def plot_elliott_wave_simple(data: pd.DataFrame, analysis_result: Dict,
                             ticker: str = "", save_path: Optional[str] = None,
                             figsize: tuple = (20, 12)):
    """
    엘리엇 파동 차트 생성 (간단하고 명확한 버전)
    
    Args:
        data: OHLCV 데이터프레임
        analysis_result: 엘리엇 파동 분석 결과
        ticker: 종목 코드
        save_path: 저장 경로
        figsize: 그림 크기
    """
    # 데이터 정규화
    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 한글 폰트 설정 - macOS 전용 강화
    import platform
    import matplotlib.font_manager as fm
    
    if platform.system() == 'Darwin':  # macOS
        # AppleSDGothicNeo 직접 로드
        font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
        if not os.path.exists(font_path):
            font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
        
        if os.path.exists(font_path):
            from matplotlib import font_manager
            font_manager.fontManager.addfont(font_path)
            prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = prop.get_name()
        else:
            plt.rcParams['font.family'] = 'AppleGothic'
    else:
        font_name = get_available_font()
        plt.rcParams['font.family'] = font_name
    
    plt.rcParams['axes.unicode_minus'] = False
    
    # Figure 생성
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                    height_ratios=[3, 1], 
                                    sharex=True)
    fig.patch.set_facecolor('white')
    
    # 1. 캔들스틱 차트
    _plot_candlestick(ax1, df)
    
    # 2. 엘리엇 파동 표시
    if analysis_result.get('success'):
        _plot_zigzag(ax1, df, analysis_result)
        _plot_wave_labels(ax1, df, analysis_result)
        _plot_fibonacci_levels(ax1, df, analysis_result)
        _plot_current_price(ax1, analysis_result)
    
    # 3. 거래량
    _plot_volume(ax2, df)
    
    # 4. 제목 및 정보
    title = f'{ticker} 엘리엇 파동 분석'
    if analysis_result.get('success'):
        confidence = analysis_result.get('confidence', 0)
        title += f' (신뢰도: {confidence:.1f}%)'
    
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.995)
    
    # 5. 정보 박스
    _add_info_text(ax1, analysis_result, ticker)
    
    # 6. 범례 추가
    _add_legend(ax1, analysis_result)
    
    # 7. 그리드 및 레이아웃
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    # 저장
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✅ 차트 저장 완료: {save_path}")
        plt.close()
    else:
        plt.show()
    
    return fig


def _plot_candlestick(ax, df):
    """캔들스틱 차트 그리기 - 양봉 녹색, 음봉 빨간색"""
    width = 0.6
    width2 = 0.05
    
    up = df['Close'] >= df['Open']
    down = df['Close'] < df['Open']
    
    # 상승 캔들 (녹색) - 양봉
    ax.bar(df.index[up], df['Close'][up] - df['Open'][up], 
           width, bottom=df['Open'][up], color='green', alpha=0.8)
    ax.bar(df.index[up], df['High'][up] - df['Close'][up], 
           width2, bottom=df['Close'][up], color='green', alpha=0.8)
    ax.bar(df.index[up], df['Open'][up] - df['Low'][up], 
           width2, bottom=df['Low'][up], color='green', alpha=0.8)
    
    # 하락 캔들 (빨간색) - 음봉
    ax.bar(df.index[down], df['Open'][down] - df['Close'][down], 
           width, bottom=df['Close'][down], color='red', alpha=0.8)
    ax.bar(df.index[down], df['High'][down] - df['Open'][down], 
           width2, bottom=df['Open'][down], color='red', alpha=0.8)
    ax.bar(df.index[down], df['Close'][down] - df['Low'][down], 
           width2, bottom=df['Low'][down], color='red', alpha=0.8)
    
    ax.set_ylabel('가격 ($)', fontsize=12, fontweight='bold')


def _plot_zigzag(ax, df, result):
    """ZigZag 라인 그리기"""
    pivots = result.get('pivots', [])
    
    if len(pivots) < 2:
        return
    
    dates = [p['date'] for p in pivots]
    prices = [float(p['price']) for p in pivots]
    
    ax.plot(dates, prices, color='gray', linestyle='--', linewidth=2, 
            marker='o', markersize=8, markerfacecolor='gray', 
            markeredgecolor='white', markeredgewidth=2,
            label='ZigZag', zorder=5, alpha=0.7)


def _plot_wave_labels(ax, df, result):
    """파동 레이블 그리기 - 겹침 방지"""
    waves = result.get('waves', [])
    
    if not waves:
        return
    
    # 모든 레이블 위치 수집
    all_labels = []
    
    # 충격파 레이블 수집
    for wave in waves:
        if wave['type'] == 'impulse':
            impulse_labels = _collect_impulse_labels(wave)
            all_labels.extend(impulse_labels)
    
    # 조정파 레이블 수집
    for wave in waves:
        if wave['type'] == 'corrective':
            corrective_labels = _collect_corrective_labels(wave)
            all_labels.extend(corrective_labels)
    
    # 레이블 위치 조정
    adjusted_labels = _adjust_label_positions(all_labels)
    
    # 조정된 위치로 레이블 그리기
    for label_info in adjusted_labels:
        _draw_wave_label(ax, label_info)
    
    # 파동 연결선 그리기
    for wave in waves:
        if wave['type'] == 'impulse':
            _plot_impulse_wave_lines(ax, wave)
        else:
            _plot_corrective_wave_lines(ax, wave)


def _collect_impulse_labels(wave):
    """충격파 레이블 정보 수집"""
    waves_data = wave['waves']
    color = '#10B981'
    
    wave_labels = [
        ('①', waves_data['wave_1']),
        ('②', waves_data['wave_2']),
        ('③', waves_data['wave_3']),
        ('④', waves_data['wave_4']),
        ('⑤', waves_data['wave_5']),
    ]
    
    labels = []
    for label, wave_point in wave_labels:
        date = pd.to_datetime(wave_point['date'])
        price = float(wave_point['price'])
        wave_type = wave_point['type']
        
        labels.append({
            'label': label,
            'date': date,
            'price': price,
            'wave_type': wave_type,
            'color': color,
            'shape': 'circle',
            'type': 'impulse'
        })
    
    return labels


def _collect_corrective_labels(wave):
    """조정파 레이블 정보 수집 - 패턴별 하위 파동 표시"""
    waves_data = wave['waves']
    pattern = wave.get('pattern', 'Unknown')
    color = '#F77F00'
    
    labels = []
    
    # 패턴별 하위 파동 표시
    if pattern == 'Zigzag':
        # Zigzag: 5-3-5 구조 (A는 5파, B는 3파, C는 5파)
        labels.extend(_create_zigzag_labels(waves_data, color))
    elif pattern == 'Flat':
        # Flat: 3-3-5 구조 (A는 3파, B는 3파, C는 5파)
        labels.extend(_create_flat_labels(waves_data, color))
    elif pattern == 'Triangle':
        # Triangle: 3-3-3-3-3 구조
        labels.extend(_create_triangle_labels(waves_data, color))
    else:
        # Complex: 기본 A-B-C 표시
        labels.extend(_create_basic_abc_labels(waves_data, color))
    
    return labels


def _create_zigzag_labels(waves_data, color):
    """Zigzag 패턴 하위 파동 레이블 생성"""
    labels = []
    
    # A파 (5파 구조)
    wave_A = waves_data['wave_A']
    labels.append({
        'label': 'A(5)',
        'date': pd.to_datetime(wave_A['date']),
        'price': float(wave_A['price']),
        'wave_type': wave_A['type'],
        'color': color,
        'shape': 'square',
        'type': 'corrective',
        'sub_wave': 'A',
        'structure': '5'
    })
    
    # B파 (3파 구조)
    wave_B = waves_data['wave_B']
    labels.append({
        'label': 'B(3)',
        'date': pd.to_datetime(wave_B['date']),
        'price': float(wave_B['price']),
        'wave_type': wave_B['type'],
        'color': color,
        'shape': 'square',
        'type': 'corrective',
        'sub_wave': 'B',
        'structure': '3'
    })
    
    # C파 (5파 구조)
    wave_C = waves_data['wave_C']
    labels.append({
        'label': 'C(5)',
        'date': pd.to_datetime(wave_C['date']),
        'price': float(wave_C['price']),
        'wave_type': wave_C['type'],
        'color': color,
        'shape': 'square',
        'type': 'corrective',
        'sub_wave': 'C',
        'structure': '5'
    })
    
    return labels


def _create_flat_labels(waves_data, color):
    """Flat 패턴 하위 파동 레이블 생성"""
    labels = []
    
    # A파 (3파 구조)
    wave_A = waves_data['wave_A']
    labels.append({
        'label': 'A(3)',
        'date': pd.to_datetime(wave_A['date']),
        'price': float(wave_A['price']),
        'wave_type': wave_A['type'],
        'color': color,
        'shape': 'square',
        'type': 'corrective',
        'sub_wave': 'A',
        'structure': '3'
    })
    
    # B파 (3파 구조)
    wave_B = waves_data['wave_B']
    labels.append({
        'label': 'B(3)',
        'date': pd.to_datetime(wave_B['date']),
        'price': float(wave_B['price']),
        'wave_type': wave_B['type'],
        'color': color,
        'shape': 'square',
        'type': 'corrective',
        'sub_wave': 'B',
        'structure': '3'
    })
    
    # C파 (5파 구조)
    wave_C = waves_data['wave_C']
    labels.append({
        'label': 'C(5)',
        'date': pd.to_datetime(wave_C['date']),
        'price': float(wave_C['price']),
        'wave_type': wave_C['type'],
        'color': color,
        'shape': 'square',
        'type': 'corrective',
        'sub_wave': 'C',
        'structure': '5'
    })
    
    return labels


def _create_triangle_labels(waves_data, color):
    """Triangle 패턴 하위 파동 레이블 생성"""
    labels = []
    
    # A파 (3파 구조)
    wave_A = waves_data['wave_A']
    labels.append({
        'label': 'A(3)',
        'date': pd.to_datetime(wave_A['date']),
        'price': float(wave_A['price']),
        'wave_type': wave_A['type'],
        'color': color,
        'shape': 'square',
        'type': 'corrective',
        'sub_wave': 'A',
        'structure': '3'
    })
    
    # B파 (3파 구조)
    wave_B = waves_data['wave_B']
    labels.append({
        'label': 'B(3)',
        'date': pd.to_datetime(wave_B['date']),
        'price': float(wave_B['price']),
        'wave_type': wave_B['type'],
        'color': color,
        'shape': 'square',
        'type': 'corrective',
        'sub_wave': 'B',
        'structure': '3'
    })
    
    # C파 (3파 구조)
    wave_C = waves_data['wave_C']
    labels.append({
        'label': 'C(3)',
        'date': pd.to_datetime(wave_C['date']),
        'price': float(wave_C['price']),
        'wave_type': wave_C['type'],
        'color': color,
        'shape': 'square',
        'type': 'corrective',
        'sub_wave': 'C',
        'structure': '3'
    })
    
    return labels


def _create_basic_abc_labels(waves_data, color):
    """기본 A-B-C 레이블 생성"""
    labels = []
    
    # A파
    wave_A = waves_data['wave_A']
    labels.append({
        'label': 'A',
        'date': pd.to_datetime(wave_A['date']),
        'price': float(wave_A['price']),
        'wave_type': wave_A['type'],
        'color': color,
        'shape': 'square',
        'type': 'corrective',
        'sub_wave': 'A'
    })
    
    # B파
    wave_B = waves_data['wave_B']
    labels.append({
        'label': 'B',
        'date': pd.to_datetime(wave_B['date']),
        'price': float(wave_B['price']),
        'wave_type': wave_B['type'],
        'color': color,
        'shape': 'square',
        'type': 'corrective',
        'sub_wave': 'B'
    })
    
    # C파
    wave_C = waves_data['wave_C']
    labels.append({
        'label': 'C',
        'date': pd.to_datetime(wave_C['date']),
        'price': float(wave_C['price']),
        'wave_type': wave_C['type'],
        'color': color,
        'shape': 'square',
        'type': 'corrective',
        'sub_wave': 'C'
    })
    
    return labels


def _adjust_label_positions(labels):
    """레이블 위치 조정 - 겹침 방지"""
    if len(labels) <= 1:
        return labels
    
    # 날짜순으로 정렬
    labels.sort(key=lambda x: x['date'])
    
    # 겹치는 레이블 찾기 및 위치 조정
    for i in range(len(labels)):
        current = labels[i]
        
        # 기본 위치 계산
        if current['wave_type'] == 'high':
            current['y_offset'] = current['price'] * 1.02
            current['va'] = 'bottom'
        else:
            current['y_offset'] = current['price'] * 0.98
            current['va'] = 'top'
        
        # 통합 A-B-C 레이블의 경우 위치 조정
        if current.get('is_combined', False):
            current['y_offset'] = current['price'] * 1.05  # 조금 더 위로
            current['va'] = 'bottom'
        
        # 이전 레이블과 겹치는지 확인
        for j in range(i):
            prev = labels[j]
            
            # 날짜가 가까우면 (7일 이내) 위치 조정
            if abs((current['date'] - prev['date']).days) <= 7:
                # 같은 방향이면 오프셋 조정
                if current['va'] == prev['va']:
                    if current['va'] == 'bottom':
                        current['y_offset'] = max(current['y_offset'], prev['y_offset'] + current['price'] * 0.05)
                    else:
                        current['y_offset'] = min(current['y_offset'], prev['y_offset'] - current['price'] * 0.05)
    
    return labels


def _draw_wave_label(ax, label_info):
    """개별 파동 레이블 그리기"""
    label = label_info['label']
    date = label_info['date']
    price = label_info['price']
    color = label_info['color']
    shape = label_info['shape']
    va = label_info['va']
    y_offset = label_info['y_offset']
    
    # 레이블 그리기
    if shape == 'circle':
        bbox_style = 'circle,pad=0.28'  # 30% 감소 (0.4 → 0.28)
    else:
        bbox_style = 'square,pad=0.35'  # 30% 감소 (0.5 → 0.35)
    
    ax.text(date, y_offset, f' {label} ', 
            fontsize=11, fontweight='bold', color='white',  # 폰트 크기도 30% 감소 (16 → 11)
            ha='center', va=va,
            bbox=dict(boxstyle=bbox_style, facecolor=color, 
                     edgecolor='white', linewidth=2),
            zorder=10)
    
    # 가격 표시
    if va == 'bottom':
        price_y = price * 1.05
    else:
        price_y = price * 0.95
    
    ax.text(date, price_y, f'${price:.1f}', 
            fontsize=10, color=color, ha='center', va=va,
            fontweight='bold', zorder=10)


def _plot_impulse_wave_lines(ax, wave):
    """충격파 연결선 그리기"""
    waves_data = wave['waves']
    color = '#10B981'
    
    wave_labels = [
        ('①', waves_data['wave_1']),
        ('②', waves_data['wave_2']),
        ('③', waves_data['wave_3']),
        ('④', waves_data['wave_4']),
        ('⑤', waves_data['wave_5']),
    ]
    
    # 파동 연결선 그리기
    wave_points = [waves_data['wave_0']] + [wave_point for _, wave_point in wave_labels]
    dates = [pd.to_datetime(point['date']) for point in wave_points]
    prices = [float(point['price']) for point in wave_points]
    
    # 충격파 연결선 - 녹색 실선 (두께 50% 감소)
    ax.plot(dates, prices, color=color, linewidth=1.5, alpha=0.8, zorder=5)


def _plot_corrective_wave_lines(ax, wave):
    """조정파 연결선 그리기"""
    waves_data = wave['waves']
    color = '#F77F00'
    
    wave_labels = [
        ('A', waves_data['wave_A']),
        ('B', waves_data['wave_B']),
        ('C', waves_data['wave_C']),
    ]
    
    # 파동 연결선 그리기
    wave_points = [waves_data['wave_0']] + [wave_point for _, wave_point in wave_labels]
    dates = [pd.to_datetime(point['date']) for point in wave_points]
    prices = [float(point['price']) for point in wave_points]
    
    # 조정파 연결선 - 주황색 실선 (두께 50% 감소)
    ax.plot(dates, prices, color=color, linewidth=1.5, alpha=0.8, zorder=5)


# 기존 함수들은 새로운 스마트 레이블링 시스템으로 대체됨


def _plot_fibonacci_levels(ax, df, result):
    """피보나치 레벨 그리기"""
    fib_levels = result.get('fibonacci_levels', {})
    
    if not fib_levels:
        return
    
    retracements = fib_levels.get('retracement', {})
    key_levels = ['38.2%', '50.0%', '61.8%']
    
    for level in key_levels:
        if level in retracements:
            price = float(retracements[level])
            if pd.notna(price):
                ax.axhline(y=price, color='#10B981', linestyle=':', 
                          linewidth=1.5, alpha=0.5, zorder=3)
                ax.text(df.index[-1], price, f' Fib {level}  ', 
                       fontsize=9, color='#10B981', va='center', ha='right',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='white', alpha=0.7))


def _plot_current_price(ax, result):
    """현재가 라인 그리기"""
    current_pos = result.get('current_position')
    
    if not current_pos:
        return
    
    current_price = float(current_pos['current_price'])
    
    ax.axhline(y=current_price, color='#8B5CF6', linestyle='-', 
              linewidth=2.5, alpha=0.8, zorder=4, label='현재가')
    
    # 현재가 레이블
    ax.text(ax.get_xlim()[1], current_price, f' ${current_price:.2f} ', 
           fontsize=11, fontweight='bold', color='white', 
           va='center', ha='right',
           bbox=dict(boxstyle='round,pad=0.5', 
                    facecolor='#8B5CF6', alpha=0.9))


def _plot_volume(ax, df):
    """거래량 차트 그리기 - 양봉 녹색, 음봉 빨간색"""
    colors = ['green' if c >= o else 'red' 
             for c, o in zip(df['Close'], df['Open'])]
    
    ax.bar(df.index, df['Volume'], color=colors, alpha=0.5, width=0.6)
    ax.set_ylabel('거래량', fontsize=12, fontweight='bold')
    ax.set_xlabel('날짜', fontsize=12, fontweight='bold')


def _add_info_text(ax, result, ticker):
    """정보 박스 추가"""
    if not result.get('success'):
        info_text = f"⚠️  {result.get('message', '분석 실패')}"
    else:
        confidence = result.get('confidence', 0)
        waves = result.get('waves', [])
        impulse_count = len([w for w in waves if w['type'] == 'impulse'])
        corrective_count = len([w for w in waves if w['type'] == 'corrective'])
        current_pos = result.get('current_position')
        
        info_lines = [
            f"📊 {ticker} 엘리엇 파동",
            "━━━━━━━━━━━━━━━━",
            f"신뢰도: {confidence:.1f}%",
            f"충격파: {impulse_count}개",
            f"조정파: {corrective_count}개",
        ]
        
        if current_pos:
            info_lines.append("")
            info_lines.append(f"현재: {current_pos['message']}")
            info_lines.append(f"가격: ${float(current_pos['current_price']):.2f}")
        
        info_text = '\n'.join(info_lines)
    
    # 정보 박스 추가 - 한글 폰트 명시적 사용
    # 현재 설정된 폰트를 사용 (monospace 대신)
    current_font = plt.rcParams['font.family']
    
    ax.text(0.02, 0.98, info_text,
           transform=ax.transAxes,
           fontsize=11,
           verticalalignment='top',
           bbox=dict(boxstyle='round,pad=1', 
                    facecolor='lightyellow', 
                    edgecolor='gray', 
                    linewidth=2,
                    alpha=0.95),
           # fontfamily 제거 - 기본 폰트 사용
           zorder=15)


def _add_legend(ax, result):
    """범례 추가 - 충격파와 조정파 구분"""
    if not result.get('success'):
        return
    
    # 범례 항목들
    legend_elements = []
    
    # 충격파 범례
    legend_elements.append(plt.Line2D([0], [0], color='#10B981', linewidth=1.5, 
                                     label='충격파 (①②③④⑤)'))
    
    # 조정파 범례 - 패턴별 하위 파동 설명
    legend_elements.append(plt.Line2D([0], [0], color='#F77F00', linewidth=1.5, 
                                     label='조정파 (A-B-C 구조)'))
    
    # 패턴별 하위 파동 범례
    legend_elements.append(plt.Line2D([0], [0], color='#F77F00', linestyle='-', 
                                     linewidth=0, label='  Zigzag: A(5)-B(3)-C(5)'))
    legend_elements.append(plt.Line2D([0], [0], color='#F77F00', linestyle='-', 
                                     linewidth=0, label='  Flat: A(3)-B(3)-C(5)'))
    legend_elements.append(plt.Line2D([0], [0], color='#F77F00', linestyle='-', 
                                     linewidth=0, label='  Triangle: A(3)-B(3)-C(3)'))
    
    # ZigZag 범례
    legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle='--', 
                                     linewidth=2, label='ZigZag'))
    
    # 피보나치 범례
    legend_elements.append(plt.Line2D([0], [0], color='#10B981', linestyle=':', 
                                     linewidth=1.5, label='피보나치'))
    
    # 현재가 범례
    legend_elements.append(plt.Line2D([0], [0], color='#8B5CF6', linewidth=2.5, 
                                     label='현재가'))
    
    # 범례 추가 (중앙 위쪽)
    ax.legend(handles=legend_elements, loc='upper center', 
              fontsize=10, framealpha=0.9, 
              bbox_to_anchor=(0.5, 0.98))

