"""
엘리엇 파동 차트 시각화 모듈

이 모듈은 엘리엇 파동 분석 결과를 시각화합니다.

주요 기능:
1. 캔들스틱 차트 + 엘리엇 파동 오버레이
2. ZigZag 라인 표시
3. 파동 번호 레이블 (1, 2, 3, 4, 5, A, B, C)
4. 피보나치 되돌림/확장 레벨
5. 현재 위치 및 목표가 표시
6. 신뢰도 정보 박스

Phase 1: 기본 차트 시각화
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import mplfinance as mpf
import pandas as pd
import numpy as np
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from ..utils import get_available_font


class ElliottWaveChart:
    """엘리엇 파동 차트 클래스"""
    
    # 색상 팔레트
    COLORS = {
        'impulse_wave': '#2E86AB',      # 파란색 (충격파)
        'corrective_wave': '#F77F00',   # 주황색 (조정파)
        'zigzag_line': '#9CA3AF',       # 회색 (ZigZag)
        'support': '#10B981',           # 녹색 (지지선)
        'resistance': '#EF4444',        # 빨간색 (저항선)
        'neutral': '#6B7280',           # 회색 (중립)
        'current_price': '#8B5CF6',     # 보라색 (현재가)
        'target': '#F59E0B',            # 노란색 (목표가)
    }
    
    def __init__(self, data: pd.DataFrame, analysis_result: Dict, ticker: str = ""):
        """
        초기화
        
        Args:
            data: OHLCV 데이터프레임
            analysis_result: 엘리엇 파동 분석 결과
            ticker: 종목 코드
        """
        # 데이터 정규화 (MultiIndex 처리)
        self.data = self._normalize_data(data)
        self.result = analysis_result
        self.ticker = ticker
        self.fig = None
        self.ax_main = None
    
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        데이터 정규화 - MultiIndex 컬럼을 단순 컬럼으로 변환
        
        Args:
            data: 원본 데이터프레임
            
        Returns:
            정규화된 데이터프레임
        """
        df = data.copy()
        
        # MultiIndex 컬럼인 경우 첫 번째 레벨만 사용
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # 필요한 컬럼만 선택
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"필수 컬럼 '{col}'이(가) 데이터에 없습니다.")
        
        return df[required_cols]
        
    def plot(self, save_path: Optional[str] = None, figsize: tuple = (16, 10)):
        """
        엘리엇 파동 차트 생성
        
        Args:
            save_path: 저장 경로 (None이면 화면에 표시)
            figsize: 그림 크기
        """
        # 한글 폰트 설정
        font_name = get_available_font()
        plt.rcParams['font.family'] = font_name
        plt.rcParams['axes.unicode_minus'] = False
        
        # Figure 생성
        self.fig = plt.figure(figsize=figsize)
        
        # GridSpec으로 레이아웃 설정
        gs = self.fig.add_gridspec(4, 1, height_ratios=[3, 1, 0.8, 0.3], hspace=0.3)
        
        # 메인 차트 (캔들스틱 + 엘리엇 파동)
        self.ax_main = self.fig.add_subplot(gs[0, 0])
        self._plot_candlestick()
        
        # 엘리엇 파동이 감지된 경우
        if self.result.get('success'):
            self._plot_elliott_waves()
            self._plot_fibonacci_levels()
            self._plot_current_position()
        
        # 거래량 차트
        ax_volume = self.fig.add_subplot(gs[1, 0], sharex=self.ax_main)
        self._plot_volume(ax_volume)
        
        # 정보 박스
        ax_info = self.fig.add_subplot(gs[2, 0])
        self._plot_info_box(ax_info)
        
        # 범례
        ax_legend = self.fig.add_subplot(gs[3, 0])
        self._plot_legend(ax_legend)
        
        # 제목
        title = f'{self.ticker} 엘리엇 파동 분석'
        if self.result.get('success'):
            confidence = self.result.get('confidence', 0)
            title += f' (신뢰도: {confidence:.1f}%)'
        else:
            title += f' ({self.result.get("message", "")})'
        
        self.fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        # 저장 또는 표시
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"차트 저장 완료: {save_path}")
        else:
            plt.show()
        
        return self.fig
    
    def _plot_candlestick(self):
        """캔들스틱 차트 그리기"""
        # 날짜를 숫자 인덱스로 변환
        date_indices = np.arange(len(self.data))
        
        # 상승/하락 구분
        up = (self.data['Close'].values >= self.data['Open'].values)
        down = (self.data['Close'].values < self.data['Open'].values)
        
        # 상승 캔들 (빨간색)
        self.ax_main.bar(date_indices[up], 
                        self.data['Close'].values[up] - self.data['Open'].values[up],
                        bottom=self.data['Open'].values[up], 
                        color='red', width=0.6, alpha=0.8)
        self.ax_main.bar(date_indices[up], 
                        self.data['High'].values[up] - self.data['Close'].values[up],
                        bottom=self.data['Close'].values[up], 
                        color='red', width=0.1)
        self.ax_main.bar(date_indices[up], 
                        self.data['Open'].values[up] - self.data['Low'].values[up],
                        bottom=self.data['Low'].values[up], 
                        color='red', width=0.1)
        
        # 하락 캔들 (파란색)
        self.ax_main.bar(date_indices[down], 
                        self.data['Open'].values[down] - self.data['Close'].values[down],
                        bottom=self.data['Close'].values[down], 
                        color='blue', width=0.6, alpha=0.8)
        self.ax_main.bar(date_indices[down], 
                        self.data['High'].values[down] - self.data['Open'].values[down],
                        bottom=self.data['Open'].values[down], 
                        color='blue', width=0.1)
        self.ax_main.bar(date_indices[down], 
                        self.data['Close'].values[down] - self.data['Low'].values[down],
                        bottom=self.data['Low'].values[down], 
                        color='blue', width=0.1)
        
        # X축 날짜 레이블 설정
        num_ticks = min(10, len(self.data))
        tick_indices = np.linspace(0, len(self.data)-1, num_ticks, dtype=int)
        tick_labels = [self.data.index[i].strftime('%Y-%m-%d') for i in tick_indices]
        
        self.ax_main.set_xticks(tick_indices)
        self.ax_main.set_xticklabels(tick_labels, rotation=45, ha='right')
        
        self.ax_main.set_ylabel('가격 (Price)', fontsize=11)
        self.ax_main.grid(True, alpha=0.3, linestyle='--')
    
    def _plot_elliott_waves(self):
        """엘리엇 파동 오버레이"""
        pivots = self.result.get('pivots', [])
        waves = self.result.get('waves', [])
        
        if not pivots:
            return
        
        # 1. ZigZag 라인 그리기
        self._plot_zigzag_line(pivots)
        
        # 2. 파동 레이블 그리기
        for wave in waves:
            if wave['type'] == 'impulse':
                self._plot_impulse_labels(wave)
            elif wave['type'] == 'corrective':
                self._plot_corrective_labels(wave)
    
    def _plot_zigzag_line(self, pivots):
        """ZigZag 라인 그리기"""
        if len(pivots) < 2:
            return
        
        pivot_indices = []
        pivot_prices = []
        
        for pivot in pivots:
            # 날짜를 인덱스로 변환
            try:
                idx = self.data.index.get_loc(pivot['date'])
                pivot_indices.append(idx)
                pivot_prices.append(pivot['price'])
            except KeyError:
                continue
        
        if len(pivot_indices) >= 2:
            self.ax_main.plot(pivot_indices, pivot_prices,
                            color=self.COLORS['zigzag_line'],
                            linestyle='--', linewidth=1.5, alpha=0.6,
                            label='ZigZag', zorder=5)
            
            # 피봇 포인트 마커
            self.ax_main.scatter(pivot_indices, pivot_prices,
                               color=self.COLORS['zigzag_line'],
                               s=50, alpha=0.7, zorder=6)
    
    def _plot_impulse_labels(self, wave: Dict):
        """충격파 레이블 (1, 2, 3, 4, 5)"""
        waves_data = wave['waves']
        
        # 파동 번호와 위치
        wave_labels = [
            ('0', waves_data['wave_0']),
            ('①', waves_data['wave_1']),
            ('②', waves_data['wave_2']),
            ('③', waves_data['wave_3']),
            ('④', waves_data['wave_4']),
            ('⑤', waves_data['wave_5']),
        ]
        
        for label, wave_point in wave_labels:
            try:
                idx = self.data.index.get_loc(wave_point['date'])
                # numpy/pandas 타입을 float로 변환
                price = float(wave_point['price']) if hasattr(wave_point['price'], '__float__') else wave_point['price']
                
                # 고점/저점에 따라 레이블 위치 조정
                y_offset = 0.02 if wave_point['type'] == 'high' else -0.02
                
                # 레이블 표시
                self.ax_main.annotate(
                    label,
                    xy=(idx, price),
                    xytext=(0, 20 if wave_point['type'] == 'high' else -20),
                    textcoords='offset points',
                    fontsize=14,
                    fontweight='bold',
                    color=self.COLORS['impulse_wave'],
                    bbox=dict(boxstyle='circle,pad=0.3',
                             facecolor='white',
                             edgecolor=self.COLORS['impulse_wave'],
                             linewidth=2),
                    ha='center',
                    zorder=10
                )
                
                # 가격 표시
                price_text = f'${price:.2f}'
                self.ax_main.annotate(
                    price_text,
                    xy=(idx, price),
                    xytext=(0, 40 if wave_point['type'] == 'high' else -40),
                    textcoords='offset points',
                    fontsize=9,
                    color=self.COLORS['impulse_wave'],
                    ha='center',
                    zorder=10
                )
                
            except (KeyError, Exception):
                continue
    
    def _plot_corrective_labels(self, wave: Dict):
        """조정파 레이블 (A, B, C)"""
        waves_data = wave['waves']
        
        # 파동 레이블
        wave_labels = [
            ('A', waves_data['wave_A']),
            ('B', waves_data['wave_B']),
            ('C', waves_data['wave_C']),
        ]
        
        for label, wave_point in wave_labels:
            try:
                idx = self.data.index.get_loc(wave_point['date'])
                # numpy/pandas 타입을 float로 변환
                price = float(wave_point['price']) if hasattr(wave_point['price'], '__float__') else wave_point['price']
                
                # 레이블 표시
                self.ax_main.annotate(
                    label,
                    xy=(idx, price),
                    xytext=(0, 20 if wave_point['type'] == 'high' else -20),
                    textcoords='offset points',
                    fontsize=14,
                    fontweight='bold',
                    color=self.COLORS['corrective_wave'],
                    bbox=dict(boxstyle='square,pad=0.4',
                             facecolor='white',
                             edgecolor=self.COLORS['corrective_wave'],
                             linewidth=2),
                    ha='center',
                    zorder=10
                )
                
            except (KeyError, Exception):
                continue
    
    def _plot_fibonacci_levels(self):
        """피보나치 되돌림/확장 레벨"""
        fib_levels = self.result.get('fibonacci_levels', {})
        
        if not fib_levels:
            return
        
        # 되돌림 레벨 (지지선)
        retracements = fib_levels.get('retracement', {})
        for level_name, price in retracements.items():
            if pd.notna(price):
                # numpy/pandas 타입을 float로 변환
                price_val = float(price) if hasattr(price, '__float__') else price
                
                self.ax_main.axhline(y=price_val,
                                    color=self.COLORS['support'],
                                    linestyle=':',
                                    linewidth=1,
                                    alpha=0.5,
                                    zorder=3)
                
                # 레벨 레이블
                self.ax_main.text(len(self.data) * 0.02, price_val,
                                 f'  Fib {level_name}',
                                 fontsize=8,
                                 color=self.COLORS['support'],
                                 va='center',
                                 alpha=0.7)
    
    def _plot_current_position(self):
        """현재 위치 및 목표가 표시"""
        current_pos = self.result.get('current_position')
        
        if not current_pos:
            return
        
        current_price = float(current_pos['current_price']) if hasattr(current_pos['current_price'], '__float__') else current_pos['current_price']
        
        # 현재가 수평선
        self.ax_main.axhline(y=current_price,
                            color=self.COLORS['current_price'],
                            linestyle='-',
                            linewidth=2,
                            alpha=0.8,
                            label='현재가',
                            zorder=8)
        
        # 현재가 레이블
        self.ax_main.text(len(self.data) * 0.98, current_price,
                         f' ${current_price:.2f} ',
                         fontsize=10,
                         fontweight='bold',
                         color='white',
                         bbox=dict(boxstyle='round,pad=0.5',
                                  facecolor=self.COLORS['current_price'],
                                  alpha=0.9),
                         va='center',
                         ha='right',
                         zorder=9)
        
        # 목표가 표시
        targets = self.result.get('targets', {})
        if 'retracement' in targets:
            # 조정 목표가 (38.2%, 61.8%)
            key_levels = ['38.2%', '61.8%']
            for level in key_levels:
                if level in targets['retracement']:
                    target_price = targets['retracement'][level]
                    if pd.notna(target_price):
                        # numpy/pandas 타입을 float로 변환
                        target_price_val = float(target_price) if hasattr(target_price, '__float__') else target_price
                        
                        self.ax_main.axhline(y=target_price_val,
                                            color=self.COLORS['target'],
                                            linestyle='--',
                                            linewidth=1.5,
                                            alpha=0.6,
                                            zorder=4)
    
    def _plot_volume(self, ax):
        """거래량 차트"""
        date_indices = np.arange(len(self.data))
        
        # 상승/하락 구분
        colors = ['red' if self.data['Close'].values[i] >= self.data['Open'].values[i]
                 else 'blue' for i in range(len(self.data))]
        
        ax.bar(date_indices, self.data['Volume'].values, color=colors, alpha=0.5, width=0.6)
        ax.set_ylabel('거래량 (Volume)', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(self.ax_main.get_xlim())
    
    def _plot_info_box(self, ax):
        """정보 박스"""
        ax.axis('off')
        
        if not self.result.get('success'):
            info_text = f"⚠️  {self.result.get('message', '분석 실패')}"
            ax.text(0.5, 0.5, info_text, fontsize=12, ha='center', va='center',
                   transform=ax.transAxes)
            return
        
        # 정보 텍스트 생성
        current_pos = self.result.get('current_position')
        waves = self.result.get('waves', [])
        confidence = self.result.get('confidence', 0)
        
        info_lines = []
        info_lines.append(f"📊 분석 정보")
        info_lines.append(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        if current_pos:
            info_lines.append(f"현재 위치: {current_pos['message']}")
            info_lines.append(f"현재가: ${current_pos['current_price']:.2f}")
        
        if waves:
            info_lines.append(f"감지된 파동: {len(waves)}개")
            for i, wave in enumerate(waves[-2:], 1):  # 최근 2개만
                wave_type = '충격파' if wave['type'] == 'impulse' else '조정파'
                if wave['type'] == 'corrective':
                    pattern = wave.get('pattern', 'Unknown')
                    info_lines.append(f"  • {wave_type} ({pattern})")
                else:
                    gain = wave.get('gain_pct', 0)
                    info_lines.append(f"  • {wave_type} ({gain:+.1f}%)")
        
        info_lines.append(f"신뢰도: {confidence:.1f}%")
        
        # 목표가 정보
        targets = self.result.get('targets', {})
        if 'retracement' in targets:
            info_lines.append(f"\n조정 목표가:")
            info_lines.append(f"  38.2%: ${targets['retracement']['38.2%']:.2f}")
            info_lines.append(f"  61.8%: ${targets['retracement']['61.8%']:.2f}")
        elif 'extension' in targets:
            info_lines.append(f"\n상승 목표가:")
            info_lines.append(f"  161.8%: ${targets['extension']['161.8%']:.2f}")
        
        info_text = '\n'.join(info_lines)
        
        ax.text(0.02, 0.95, info_text,
               fontsize=9,
               fontfamily='monospace',
               verticalalignment='top',
               transform=ax.transAxes,
               bbox=dict(boxstyle='round,pad=0.8',
                        facecolor='lightyellow',
                        edgecolor='gray',
                        alpha=0.8))
    
    def _plot_legend(self, ax):
        """범례"""
        ax.axis('off')
        
        if not self.result.get('success'):
            return
        
        legend_elements = [
            mpatches.Patch(color=self.COLORS['impulse_wave'],
                          label='충격파 (1-2-3-4-5)'),
            mpatches.Patch(color=self.COLORS['corrective_wave'],
                          label='조정파 (A-B-C)'),
            mpatches.Patch(color=self.COLORS['zigzag_line'],
                          label='ZigZag 라인'),
            mpatches.Patch(color=self.COLORS['support'],
                          label='피보나치 지지선'),
            mpatches.Patch(color=self.COLORS['current_price'],
                          label='현재가'),
            mpatches.Patch(color=self.COLORS['target'],
                          label='목표가'),
        ]
        
        ax.legend(handles=legend_elements,
                 loc='center',
                 ncol=6,
                 fontsize=9,
                 frameon=False)


# 편의 함수
def plot_elliott_wave_chart(data: pd.DataFrame, analysis_result: Dict,
                            ticker: str = "", save_path: Optional[str] = None):
    """
    엘리엇 파동 차트 생성 (편의 함수)
    
    Args:
        data: OHLCV 데이터프레임
        analysis_result: 엘리엇 파동 분석 결과
        ticker: 종목 코드
        save_path: 저장 경로
    """
    chart = ElliottWaveChart(data, analysis_result, ticker)
    return chart.plot(save_path=save_path)

