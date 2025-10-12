"""
엘리엇 파동 Plotly 인터랙티브 차트 시각화 모듈 (Phase 3 - 개선)

이 모듈은 Plotly를 사용하여 엘리엇 파동 분석 결과를 인터랙티브하게 시각화합니다.

Phase 3 개선 버전:
- 캔들스틱 차트 개선
- 파동 레이블 명확하게 표시
- 자동 범위 조정
- 더 나은 레이아웃
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime


class ElliottWavePlotly:
    """엘리엇 파동 Plotly 인터랙티브 차트 클래스"""
    
    # 색상 팔레트
    COLORS = {
        'impulse_wave': '#2E86AB',      # 파란색 (충격파)
        'corrective_wave': '#F77F00',   # 주황색 (조정파)
        'zigzag_line': '#9CA3AF',       # 회색 (ZigZag)
        'support': '#10B981',           # 녹색 (지지선)
        'resistance': '#EF4444',        # 빨간색 (저항선)
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
        # 데이터 정규화
        self.data = self._normalize_data(data)
        self.result = analysis_result
        self.ticker = ticker
        self.fig = None
        
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터 정규화"""
        df = data.copy()
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"필수 컬럼 '{col}'이(가) 데이터에 없습니다.")
        
        return df[required_cols]
    
    def create_chart(self, save_html: Optional[str] = None, 
                    save_png: Optional[str] = None,
                    height: int = 1000,
                    show_chart: bool = False) -> go.Figure:
        """
        인터랙티브 차트 생성
        
        Args:
            save_html: HTML 파일 저장 경로
            save_png: PNG 파일 저장 경로
            height: 차트 높이 (픽셀)
            show_chart: 브라우저에서 차트 표시 여부
            
        Returns:
            Plotly Figure 객체
        """
        # 서브플롯 생성
        self.fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            specs=[[{"type": "candlestick"}],
                   [{"type": "bar"}]]
        )
        
        # 1. 캔들스틱 차트
        self._add_candlestick()
        
        # 2. 엘리엇 파동 (성공한 경우만)
        if self.result.get('success'):
            self._add_zigzag_line()
            self._add_wave_annotations()
            self._add_fibonacci_levels()
            self._add_current_price()
        
        # 3. 거래량
        self._add_volume()
        
        # 4. 레이아웃 설정
        self._configure_layout(height)
        
        # 5. 저장
        if save_html:
            self.fig.write_html(save_html)
            print(f"✅ HTML 저장 완료: {save_html}")
        
        if save_png:
            try:
                self.fig.write_image(save_png, width=1600, height=height)
                print(f"✅ PNG 저장 완료: {save_png}")
            except Exception as e:
                print(f"⚠️  PNG 저장 오류: {e}")
                print(f"💡 kaleido가 설치되어 있는지 확인하세요: pip install kaleido")
        
        if show_chart:
            self.fig.show()
        
        return self.fig
    
    def _add_candlestick(self):
        """캔들스틱 차트 추가"""
        self.fig.add_trace(
            go.Candlestick(
                x=self.data.index,
                open=self.data['Open'],
                high=self.data['High'],
                low=self.data['Low'],
                close=self.data['Close'],
                name='캔들',
                increasing_line_color='#EF4444',
                decreasing_line_color='#3B82F6',
                increasing_fillcolor='#EF4444',
                decreasing_fillcolor='#3B82F6'
            ),
            row=1, col=1
        )
    
    def _add_zigzag_line(self):
        """ZigZag 라인 추가"""
        pivots = self.result.get('pivots', [])
        
        if len(pivots) < 2:
            return
        
        dates = [p['date'] for p in pivots]
        prices = [float(p['price']) for p in pivots]
        
        self.fig.add_trace(
            go.Scatter(
                x=dates,
                y=prices,
                mode='lines+markers',
                name='ZigZag',
                line=dict(color=self.COLORS['zigzag_line'], width=2, dash='dash'),
                marker=dict(size=10, color=self.COLORS['zigzag_line']),
                hovertemplate='<b>%{x}</b><br>가격: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    def _add_wave_annotations(self):
        """파동 주석 추가 - 간소화 버전"""
        waves = self.result.get('waves', [])
        
        # 충격파만 표시 (조정파는 너무 많아서 제외)
        impulse_waves = [w for w in waves if w['type'] == 'impulse']
        
        for i, wave in enumerate(impulse_waves, 1):
            waves_data = wave['waves']
            
            # 주요 파동만 표시 (1, 3, 5)
            key_waves = [
                ('①', waves_data['wave_1']),
                ('③', waves_data['wave_3']),
                ('⑤', waves_data['wave_5']),
            ]
            
            for label, wave_point in key_waves:
                price = float(wave_point['price'])
                date = wave_point['date']
                wave_type = wave_point['type']
                
                # Y축 offset (고점은 위, 저점은 아래)
                y_shift = 30 if wave_type == 'high' else -30
                
                self.fig.add_annotation(
                    x=date,
                    y=price,
                    text=f"<b>{label}</b><br>${price:.1f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=self.COLORS['impulse_wave'],
                    ax=0,
                    ay=y_shift,
                    bgcolor='white',
                    bordercolor=self.COLORS['impulse_wave'],
                    borderwidth=2,
                    borderpad=6,
                    font=dict(size=11, color=self.COLORS['impulse_wave']),
                    xref='x', yref='y',
                    row=1, col=1
                )
    
    def _add_fibonacci_levels(self):
        """피보나치 레벨 추가 - 주요 레벨만"""
        fib_levels = self.result.get('fibonacci_levels', {})
        
        if not fib_levels:
            return
        
        # 주요 되돌림 레벨만 표시
        retracements = fib_levels.get('retracement', {})
        key_levels = ['38.2%', '50.0%', '61.8%']
        
        for level in key_levels:
            if level in retracements:
                price = retracements[level]
                if pd.notna(price):
                    price_val = float(price)
                    
                    self.fig.add_hline(
                        y=price_val,
                        line_dash="dot",
                        line_color=self.COLORS['support'],
                        line_width=1,
                        opacity=0.4,
                        annotation_text=f"Fib {level}",
                        annotation_position="right",
                        annotation_font_size=9,
                        annotation_font_color=self.COLORS['support'],
                        row=1, col=1
                    )
    
    def _add_current_price(self):
        """현재가 라인 추가"""
        current_pos = self.result.get('current_position')
        
        if not current_pos:
            return
        
        current_price = float(current_pos['current_price'])
        
        self.fig.add_hline(
            y=current_price,
            line_dash="solid",
            line_color=self.COLORS['current_price'],
            line_width=2.5,
            opacity=0.8,
            annotation_text=f"현재가: ${current_price:.2f}",
            annotation_position="right",
            annotation_font_size=11,
            annotation_font_color='white',
            annotation_bgcolor=self.COLORS['current_price'],
            row=1, col=1
        )
    
    def _add_volume(self):
        """거래량 차트 추가"""
        colors = ['#EF4444' if c >= o else '#3B82F6'
                 for c, o in zip(self.data['Close'], self.data['Open'])]
        
        self.fig.add_trace(
            go.Bar(
                x=self.data.index,
                y=self.data['Volume'],
                name='거래량',
                marker_color=colors,
                opacity=0.5,
                showlegend=False
            ),
            row=2, col=1
        )
    
    def _configure_layout(self, height: int):
        """레이아웃 설정"""
        # 제목
        title_text = f"{self.ticker} 엘리엇 파동 분석"
        if self.result.get('success'):
            confidence = self.result.get('confidence', 0)
            title_text += f" (신뢰도: {confidence:.1f}%)"
        
        # 정보 텍스트 생성
        info_text = self._create_info_text()
        
        self.fig.update_layout(
            title={
                'text': title_text,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'family': 'Arial Black'}
            },
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            height=height,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='gray',
                borderwidth=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Arial')
        )
        
        # X축 설정
        self.fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='#E5E7EB',
            row=1, col=1
        )
        
        self.fig.update_xaxes(
            title_text="날짜",
            showgrid=True,
            gridwidth=1,
            gridcolor='#E5E7EB',
            row=2, col=1
        )
        
        # Y축 설정
        self.fig.update_yaxes(
            title_text="가격 ($)",
            showgrid=True,
            gridwidth=1,
            gridcolor='#E5E7EB',
            row=1, col=1
        )
        
        self.fig.update_yaxes(
            title_text="거래량",
            showgrid=True,
            gridwidth=1,
            gridcolor='#E5E7EB',
            row=2, col=1
        )
        
        # 정보 박스 추가
        if info_text:
            self.fig.add_annotation(
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                xanchor='left', yanchor='top',
                text=info_text,
                showarrow=False,
                font=dict(size=10, family='Courier New'),
                align='left',
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor='#6B7280',
                borderwidth=2,
                borderpad=10
            )
    
    def _create_info_text(self) -> str:
        """정보 박스 텍스트 생성"""
        if not self.result.get('success'):
            return f"⚠️  {self.result.get('message', '분석 실패')}"
        
        lines = []
        lines.append(f"<b>📊 {self.ticker} 엘리엇 파동</b>")
        lines.append("━━━━━━━━━━━━━━━━")
        
        confidence = self.result.get('confidence', 0)
        lines.append(f"<b>신뢰도:</b> {confidence:.1f}%")
        
        waves = self.result.get('waves', [])
        impulse_count = len([w for w in waves if w['type'] == 'impulse'])
        corrective_count = len([w for w in waves if w['type'] == 'corrective'])
        lines.append(f"<b>충격파:</b> {impulse_count}개")
        lines.append(f"<b>조정파:</b> {corrective_count}개")
        
        current_pos = self.result.get('current_position')
        if current_pos:
            lines.append("")
            lines.append(f"<b>현재 위치:</b>")
            lines.append(f"{current_pos['message']}")
            lines.append(f"<b>현재가:</b> ${float(current_pos['current_price']):.2f}")
        
        targets = self.result.get('targets', {})
        if 'retracement' in targets and '61.8%' in targets['retracement']:
            target_price = float(targets['retracement']['61.8%'])
            lines.append("")
            lines.append(f"<b>목표(61.8%):</b> ${target_price:.2f}")
        elif 'extension' in targets and '161.8%' in targets['extension']:
            target_price = float(targets['extension']['161.8%'])
            lines.append("")
            lines.append(f"<b>목표(161.8%):</b> ${target_price:.2f}")
        
        return "<br>".join(lines)


# 편의 함수
def create_elliott_wave_chart(data: pd.DataFrame, analysis_result: Dict,
                              ticker: str = "",
                              save_html: Optional[str] = None,
                              save_png: Optional[str] = None,
                              height: int = 1000,
                              show_chart: bool = False) -> go.Figure:
    """
    엘리엇 파동 Plotly 차트 생성 (편의 함수)
    
    Args:
        data: OHLCV 데이터프레임
        analysis_result: 엘리엇 파동 분석 결과
        ticker: 종목 코드
        save_html: HTML 파일 저장 경로
        save_png: PNG 파일 저장 경로
        height: 차트 높이 (픽셀)
        show_chart: 브라우저에서 차트 표시 여부
        
    Returns:
        Plotly Figure 객체
    """
    chart = ElliottWavePlotly(data, analysis_result, ticker)
    return chart.create_chart(
        save_html=save_html,
        save_png=save_png,
        height=height,
        show_chart=show_chart
    )
