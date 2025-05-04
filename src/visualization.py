import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import pandas as pd
import os
from datetime import datetime
import numpy as np
import platform
from matplotlib.patches import Patch

# 한글 폰트 설정
def set_korean_font():
    system = platform.system()
    
    if system == 'Darwin':  # macOS
        font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
        if not os.path.exists(font_path):
            font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
    elif system == 'Windows':
        font_path = 'C:/Windows/Fonts/malgun.ttf'
    else:  # Linux 등 기타 시스템
        font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
        if not os.path.exists(font_path):
            # 나눔 폰트가 없으면 기본 sans-serif 폰트 사용
            font_path = None
    
    if font_path and os.path.exists(font_path):
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rc('font', family=font_name)
    else:
        # 적절한 폰트를 찾지 못한 경우 sans-serif 폰트 사용
        plt.rc('font', family='sans-serif')
    
    # 마이너스 기호 깨짐 방지
    plt.rc('axes', unicode_minus=False)

# 클래스 정의 전에 한글 폰트 설정 함수 호출
set_korean_font()

class StockVisualizer:
    def __init__(self, data=None, ticker=None):
        """
        주식 시각화 클래스 초기화
        
        Args:
            data (pd.DataFrame): 기술적 지표가 포함된 주가 데이터
            ticker (str): 종목 코드
        """
        self.data = data
        self.ticker = ticker
        self.fig = None
        self.axes = None
        self._is_sell_chart = False
        self.consolidation_ranges = []  # 박스권 정보를 저장할 리스트
        
    def set_data(self, data, ticker=None):
        """데이터 및 종목코드 설정"""
        self.data = data
        if ticker:
            self.ticker = ticker
            
    def plot_all(self, figsize=(14, 28), dpi=100):
        """모든 차트 그리기"""
        if self.data is None:
            print("ERROR: 시각화할 데이터가 없습니다.")
            return None
            
        # 매도 차트 플래그 설정
        self._is_sell_chart = False
        
        # 차트 레이아웃 설정
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = gridspec.GridSpec(7, 1, height_ratios=[3, 1.5, 1, 1, 1, 1, 1])
        
        # 메인 차트 (캔들스틱 + 볼린저 밴드)
        self.axes = []
        self.axes.append(plt.subplot(gs[0]))
        
        # 이동평균선 차트 (독립적인 서브플롯으로 분리)
        self.axes.append(plt.subplot(gs[1], sharex=self.axes[0]))
        
        # 거래량 차트
        self.axes.append(plt.subplot(gs[2], sharex=self.axes[0]))
        
        # RSI 차트 (독립적인 서브플롯으로 분리)
        self.axes.append(plt.subplot(gs[3], sharex=self.axes[0]))
        
        # MACD 차트
        self.axes.append(plt.subplot(gs[4], sharex=self.axes[0]))
        
        # MFI 차트 (독립적인 서브플롯으로 분리)
        self.axes.append(plt.subplot(gs[5], sharex=self.axes[0]))
        
        # 보조지표 차트 (볼린저 밴드 %B, OBV)
        self.axes.append(plt.subplot(gs[6], sharex=self.axes[0]))
        
        # 각 차트 그리기
        self._plot_candlestick_with_bb()
        self._plot_moving_averages()
        self._plot_volume()
        self._plot_rsi()
        self._plot_macd()
        self._plot_mfi()
        self._plot_momentum()

        # 매수 신호가 있는 경우 모든 서브플롯에 수직선 추가
        if 'Buy_Score' in self.data.columns:
            self._add_buy_signal_markers()
        
        # x축 날짜 형식 설정
        for ax in self.axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            ax.grid(True, alpha=0.3)
            # 각 서브플롯의 x축 날짜 표시를 45도 각도로 설정
            plt.setp(ax.get_xticklabels(), rotation=45)
            
        # 전체 레이아웃 조정
        plt.tight_layout()
        
        # 제목 설정
        if self.ticker:
            period = f"{self.data.index[0].strftime('%Y-%m-%d')} ~ {self.data.index[-1].strftime('%Y-%m-%d')}"
            self.fig.suptitle(f"{self.ticker} 기술적 분석 차트 ({period})", fontsize=16)
            plt.subplots_adjust(top=0.95)
            
        return self.fig
    
    def plot_all_sell_signals(self, figsize=(14, 28), dpi=100):
        """매도 신호를 강조한 모든 차트 그리기"""
        if self.data is None:
            print("ERROR: 시각화할 데이터가 없습니다.")
            return None
            
        # 매도 차트 플래그 설정
        self._is_sell_chart = True
        
        # 차트 레이아웃 설정
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = gridspec.GridSpec(7, 1, height_ratios=[3, 1.5, 1, 1, 1, 1, 1])
        
        # 메인 차트 (캔들스틱 + 볼린저 밴드)
        self.axes = []
        self.axes.append(plt.subplot(gs[0]))
        
        # 이동평균선 차트 (독립적인 서브플롯으로 분리)
        self.axes.append(plt.subplot(gs[1], sharex=self.axes[0]))
        
        # 거래량 차트
        self.axes.append(plt.subplot(gs[2], sharex=self.axes[0]))
        
        # RSI 차트 (독립적인 서브플롯으로 분리)
        self.axes.append(plt.subplot(gs[3], sharex=self.axes[0]))
        
        # MACD 차트
        self.axes.append(plt.subplot(gs[4], sharex=self.axes[0]))
        
        # MFI 차트 (독립적인 서브플롯으로 분리)
        self.axes.append(plt.subplot(gs[5], sharex=self.axes[0]))
        
        # 보조지표 차트 (볼린저 밴드 %B, OBV)
        self.axes.append(plt.subplot(gs[6], sharex=self.axes[0]))
        
        # 각 차트 그리기
        self._plot_candlestick_with_bb()
        self._plot_moving_averages()
        self._plot_volume()
        self._plot_rsi()
        self._plot_macd()
        self._plot_mfi()
        self._plot_momentum()

        # 매도 신호가 있는 경우 모든 서브플롯에 수직선 추가
        if 'Sell_Score' in self.data.columns:
            self._add_sell_signal_markers()
        
        # x축 날짜 형식 설정
        for ax in self.axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            ax.grid(True, alpha=0.3)
            # 각 서브플롯의 x축 날짜 표시를 45도 각도로 설정
            plt.setp(ax.get_xticklabels(), rotation=45)
            
        # 전체 레이아웃 조정
        plt.tight_layout()
        
        # 제목 설정
        if self.ticker:
            period = f"{self.data.index[0].strftime('%Y-%m-%d')} ~ {self.data.index[-1].strftime('%Y-%m-%d')}"
            self.fig.suptitle(f"{self.ticker} 매도 신호 분석 차트 ({period})", fontsize=16, color='red')
            plt.subplots_adjust(top=0.95)
            
        return self.fig
        
    def _plot_candlestick_with_bb(self):
        """캔들스틱 + 볼린저 밴드 그리기"""
        ax = self.axes[0]
        
        # 캔들스틱
        for i in range(len(self.data)):
            # 시가, 종가, 고가, 저가
            open_price = self.data['Open'].iloc[i]
            close_price = self.data['Close'].iloc[i]
            high_price = self.data['High'].iloc[i]
            low_price = self.data['Low'].iloc[i]
            date = self.data.index[i]
            
            # 양봉(빨간색), 음봉(파란색)
            color = 'red' if close_price >= open_price else 'blue'
            
            # 몸통 그리기
            rect = plt.Rectangle(
                (mdates.date2num(date) - 0.25, min(open_price, close_price)),
                0.5, abs(close_price - open_price),
                fill=True, color=color, alpha=0.5
            )
            ax.add_patch(rect)
            
            # 꼬리 그리기
            ax.plot(
                [mdates.date2num(date), mdates.date2num(date)],
                [low_price, min(open_price, close_price)],
                color=color, linewidth=1
            )
            ax.plot(
                [mdates.date2num(date), mdates.date2num(date)],
                [max(open_price, close_price), high_price],
                color=color, linewidth=1
            )
            
            # 매수 신호 화살표 표시 (매도 차트가 아닌 경우에만)
            if 'Buy_Score' in self.data.columns and (not hasattr(self, '_is_sell_chart') or not self._is_sell_chart):
                buy_score = self.data['Buy_Score'].iloc[i]
                
                # 매수 점수가 50점 이상인 경우 화살표로 표시
                if buy_score >= 70:  # 매우 강한 신호
                    ax.annotate('⬆',
                        xy=(mdates.date2num(date), low_price * 0.99),
                        xytext=(mdates.date2num(date), low_price * 0.97),
                        fontsize=15, color='magenta',
                        arrowprops=dict(facecolor='magenta', shrink=0.05),
                        horizontalalignment='center'
                    )
                elif buy_score >= 50:  # 강한 신호
                    ax.annotate('⬆',
                        xy=(mdates.date2num(date), low_price * 0.99),
                        xytext=(mdates.date2num(date), low_price * 0.97),
                        fontsize=12, color='green',
                        arrowprops=dict(facecolor='green', shrink=0.05),
                        horizontalalignment='center'
                    )
            
            # 매도 신호 화살표 표시 (매도 차트인 경우에만)
            if 'Sell_Score' in self.data.columns and hasattr(self, '_is_sell_chart') and self._is_sell_chart:
                sell_score = self.data['Sell_Score'].iloc[i]
                
                # 매도 점수가 50점 이상인 경우 화살표로 표시
                if sell_score >= 70:  # 매우 강한 신호
                    ax.annotate('⬇',
                        xy=(mdates.date2num(date), high_price * 1.01),
                        xytext=(mdates.date2num(date), high_price * 1.03),
                        fontsize=15, color='red',
                        arrowprops=dict(facecolor='red', shrink=0.05),
                        horizontalalignment='center'
                    )
                elif sell_score >= 50:  # 강한 신호
                    ax.annotate('⬇',
                        xy=(mdates.date2num(date), high_price * 1.01),
                        xytext=(mdates.date2num(date), high_price * 1.03),
                        fontsize=12, color='crimson',
                        arrowprops=dict(facecolor='crimson', shrink=0.05),
                        horizontalalignment='center'
                    )
        
        # 박스권(횡보구간) 표시
        if self.consolidation_ranges:
            for box in self.consolidation_ranges:
                # 박스권 영역 색상 설정 (반투명한 회색)
                rect = plt.Rectangle(
                    (mdates.date2num(box['start_date']), box['lower_price']),
                    mdates.date2num(box['end_date']) - mdates.date2num(box['start_date']),
                    box['upper_price'] - box['lower_price'],
                    fill=True,
                    facecolor='darkgoldenrod', alpha=0.15,
                    linestyle='--', linewidth=2, edgecolor='goldenrod'
                )
                ax.add_patch(rect)
                
                # 박스권 레이블 추가
                range_percent = box['range_percent'] * 100
                label_text = f"박스권: {range_percent:.1f}%"
                
                # 박스권 중간 위치에 텍스트 표시
                mid_date = box['start_date'] + (box['end_date'] - box['start_date']) / 2
                mid_price = (box['upper_price'] + box['lower_price']) / 2
                
                ax.text(
                    mdates.date2num(mid_date), mid_price,
                    label_text, fontsize=8, color='darkgoldenrod',
                    ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='goldenrod', boxstyle='round,pad=0.3')
                )
        
        # 볼린저 밴드
        if 'BB_Upper' in self.data.columns and 'BB_Lower' in self.data.columns:
            ax.plot(self.data.index, self.data['BB_Upper'], 'k--', alpha=0.3, label='BB 상단')
            ax.plot(self.data.index, self.data['BB_Middle'], 'k-', alpha=0.3, label='BB 중간')
            ax.plot(self.data.index, self.data['BB_Lower'], 'k--', alpha=0.3, label='BB 하단')
            ax.fill_between(self.data.index, self.data['BB_Upper'], self.data['BB_Lower'], color='gray', alpha=0.1)
            
        # 캔들스틱 범례를 위한 패치 추가
        red_patch = Patch(color='red', alpha=0.5, label='양봉')
        blue_patch = Patch(color='blue', alpha=0.5, label='음봉')
        handles, labels = ax.get_legend_handles_labels()
        handles.extend([red_patch, blue_patch])
        
        # 박스권 범례 추가
        if self.consolidation_ranges:
            box_patch = Patch(color='darkgoldenrod', alpha=0.15, label='박스권')
            handles.append(box_patch)
        
        # 매수 또는 매도 신호 범례 추가
        if self.fig and hasattr(self, '_is_sell_chart') and self._is_sell_chart:
            # 매도 차트인 경우 매도 신호 범례 추가
            red_arrow = plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='crimson', markersize=10, label='매도 신호')
            dark_red_arrow = plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='강력 매도 신호')
            handles.extend([red_arrow, dark_red_arrow])
        elif 'Buy_Score' in self.data.columns:
            # 매수 차트인 경우 매수 신호 범례 추가
            green_arrow = plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, label='매수 신호')
            magenta_arrow = plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='magenta', markersize=10, label='강력 매수 신호')
            handles.extend([green_arrow, magenta_arrow])
        
        ax.set_ylabel('가격')
        ax.legend(handles=handles, loc='upper left')
        
    def _plot_moving_averages(self):
        """이동평균선 차트 그리기 (독립적인 서브플롯)"""
        ax = self.axes[1]
        
        # 종가 라인
        ax.plot(self.data.index, self.data['Close'], color='black', linewidth=1, label='종가')
        
        # 이동평균선
        if 'MA5' in self.data.columns:
            ax.plot(self.data.index, self.data['MA5'], color='red', linewidth=1, label='MA5')
        if 'MA20' in self.data.columns:
            ax.plot(self.data.index, self.data['MA20'], color='orange', linewidth=1, label='MA20')
        if 'MA60' in self.data.columns:
            ax.plot(self.data.index, self.data['MA60'], color='green', linewidth=1, label='MA60')
        if 'MA120' in self.data.columns:
            ax.plot(self.data.index, self.data['MA120'], color='blue', linewidth=1, label='MA120')
        if 'MA200' in self.data.columns:
            ax.plot(self.data.index, self.data['MA200'], color='purple', linewidth=1, label='MA200')
            
        ax.set_ylabel('이동평균선')
        ax.legend(loc='upper left')
        
    def _plot_volume(self):
        """거래량 차트 그리기"""
        ax = self.axes[2]
        
        # 기본 거래량 바 그리기
        colors = ['red' if close >= open else 'blue' for open, close in zip(self.data['Open'], self.data['Close'])]
        ax.bar(self.data.index, self.data['Volume'], color=colors, alpha=0.7, width=0.8)
        
        # 거래량 이동평균
        if 'Volume_MA20' in self.data.columns:
            ax.plot(self.data.index, self.data['Volume_MA20'], color='darkgoldenrod', linewidth=1, label='거래량 MA20')
        
        # 외국인 매수 정보가 있는 경우 표시
        if 'Foreign_Buy' in self.data.columns:
            # 외국인 순매수 변환 (단위 조정)
            foreign_buy_millions = self.data['Foreign_Buy'] / 1000000  # 백만 단위로 변환
            
            # 외국인 순매수 플롯 추가 (양수는 매수, 음수는 매도)
            twin_ax = ax.twinx()  # 보조 y축 생성
            twin_ax.plot(self.data.index, foreign_buy_millions, color='purple', linewidth=1, 
                        linestyle='-', marker='.', alpha=0.6, label='외국인순매수(백만)')
            twin_ax.axhline(y=0, color='purple', linestyle='--', alpha=0.3)
            twin_ax.set_ylabel('외국인순매수(백만원)', color='purple')
            twin_ax.grid(False)
            
            # 외국인 매수 표식 추가
            buy_signal_days = self.data[self.data['Foreign_Buy_Signal'] == 1].index
            if len(buy_signal_days) > 0:
                y_vals = [self.data.loc[day, 'Volume'] * 0.5 for day in buy_signal_days]
                twin_ax.scatter(buy_signal_days, [0] * len(buy_signal_days), 
                               marker='^', color='purple', alpha=0.7, s=30)
            
            # 범례 추가
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = twin_ax.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc='upper left')
        else:
            ax.legend(loc='upper left')
        
        ax.set_ylabel('거래량')
        
    def _plot_rsi(self):
        """RSI 지표 그리기 (독립적인 서브플롯)"""
        ax = self.axes[3]
        
        # RSI
        if 'RSI' in self.data.columns:
            ax.plot(self.data.index, self.data['RSI'], color='purple', linewidth=1, label='RSI')
            
            # 과매수/과매도 영역
            ax.axhline(y=70, color='r', linestyle='-', alpha=0.3)
            ax.axhline(y=30, color='g', linestyle='-', alpha=0.3)
            ax.fill_between(self.data.index, 70, 100, color='red', alpha=0.1)
            ax.fill_between(self.data.index, 0, 30, color='green', alpha=0.1)
            
        ax.set_ylabel('RSI')
        ax.set_ylim(0, 100)
        ax.legend(loc='upper left')
        
    def _plot_macd(self):
        """MACD 지표 그리기"""
        ax = self.axes[4]
        
        # MACD
        if 'MACD' in self.data.columns and 'MACD_Signal' in self.data.columns:
            ax.plot(self.data.index, self.data['MACD'], color='blue', linewidth=1, label='MACD')
            ax.plot(self.data.index, self.data['MACD_Signal'], color='red', linewidth=1, label='Signal')
            
            # MACD 히스토그램
            if 'MACD_Hist' in self.data.columns:
                for i in range(len(self.data)):
                    if not np.isnan(self.data['MACD_Hist'].iloc[i]):
                        date = self.data.index[i]
                        hist = self.data['MACD_Hist'].iloc[i]
                        color = 'red' if hist >= 0 else 'blue'
                        
                        rect = plt.Rectangle(
                            (mdates.date2num(date) - 0.25, 0 if hist >= 0 else hist),
                            0.5, abs(hist),
                            fill=True, color=color, alpha=0.3
                        )
                        ax.add_patch(rect)
                        
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_ylabel('MACD')
        ax.legend(loc='upper left')

    def _plot_mfi(self):
        """MFI 지표 그리기 (독립적인 서브플롯)"""
        ax = self.axes[5]
        
        # MFI
        if 'MFI' in self.data.columns:
            ax.plot(self.data.index, self.data['MFI'], color='orange', linewidth=1, label='MFI')
            
            # 과매수/과매도 영역
            ax.axhline(y=80, color='r', linestyle='-', alpha=0.3)
            ax.axhline(y=20, color='g', linestyle='-', alpha=0.3)
            ax.fill_between(self.data.index, 80, 100, color='red', alpha=0.1)
            ax.fill_between(self.data.index, 0, 20, color='green', alpha=0.1)
            
        ax.set_ylabel('MFI')
        ax.set_ylim(0, 100)
        ax.legend(loc='upper left')
        
    def _plot_momentum(self):
        """볼린저 밴드 %B, OBV 지표 그리기"""
        ax = self.axes[6]
        
        # 볼린저 밴드 %B
        if 'BB_PB' in self.data.columns:
            ax.plot(self.data.index, self.data['BB_PB'], color='green', linewidth=1, label='BB %B')
            ax.axhline(y=1, color='r', linestyle='-', alpha=0.3)
            ax.axhline(y=0, color='g', linestyle='-', alpha=0.3)
            ax.fill_between(self.data.index, 1, 1.5, color='red', alpha=0.1)
            ax.fill_between(self.data.index, 0, -0.5, color='green', alpha=0.1)
            
        # OBV (정규화)
        if 'OBV' in self.data.columns:
            obv_norm = (self.data['OBV'] - self.data['OBV'].min()) / (self.data['OBV'].max() - self.data['OBV'].min())
            ax.plot(self.data.index, obv_norm, color='blue', linewidth=1, label='OBV (Norm)')
            
        ax.set_ylabel('보조지표')
        ax.set_ylim(-0.1, 1.5)
        ax.legend(loc='upper left')
        
    def _add_buy_signal_markers(self):
        """모든 서브플롯에 매수 신호에 대한 수직선과 점수 표시"""
        # 매수 점수가 15점 이상인 신호만 필터링 (임계값 낮춤)
        buy_signals = self.data[self.data['Buy_Score'] >= 15].copy()
        
        if len(buy_signals) == 0:
            return
            
        # 모든 서브플롯에 매수 신호 수직선 추가
        for i, ax in enumerate(self.axes):
            y_min, y_max = ax.get_ylim()
            
            for idx, row in buy_signals.iterrows():
                score = row['Buy_Score']
                color = 'magenta' if score >= 70 else ('green' if score >= 50 else ('orange' if score >= 30 else 'gray'))
                linestyle = '-' if score >= 50 else ('--' if score >= 30 else ':')
                linewidth = 1.5 if score >= 50 else 1
                
                # 수직선 추가
                ax.axvline(x=idx, color=color, alpha=0.5, linestyle=linestyle, linewidth=linewidth)
                
                # 첫 번째 서브플롯(캔들스틱 차트)에 점수와 매수 근거 표시
                if i == 0:
                    # 매수 근거 텍스트 생성
                    buy_reasons = []
                    if 'MA_Cross' in self.data.columns and row['MA_Cross'] == 1:
                        buy_reasons.append("골든크로스")
                    if 'MA_Trend_Signal' in self.data.columns and row['MA_Trend_Signal'] == 1:
                        buy_reasons.append("MA상승")
                    if 'RSI_Signal' in self.data.columns and row['RSI_Signal'] == 1:
                        buy_reasons.append("RSI반등")
                    if 'RSI_Up_Signal' in self.data.columns and row['RSI_Up_Signal'] == 1:
                        buy_reasons.append("RSI상승")
                    if 'MACD_Cross' in self.data.columns and row['MACD_Cross'] == 1:
                        buy_reasons.append("MACD돌파")
                    if 'MFI_Signal' in self.data.columns and row['MFI_Signal'] == 1:
                        buy_reasons.append("MFI반등")
                    if 'BB_Signal' in self.data.columns and row['BB_Signal'] == 1:
                        buy_reasons.append("BB반등")
                    if 'OBV_Signal' in self.data.columns and row['OBV_Signal'] == 1:
                        buy_reasons.append("OBV상승")
                    if 'Volume_Signal' in self.data.columns and row['Volume_Signal'] == 1:
                        buy_reasons.append("거래량급증")
                    if 'Foreign_Buy_Signal' in self.data.columns and row['Foreign_Buy_Signal'] == 1:
                        buy_reasons.append("외국인매수")
                    
                    # 캔들 위에 매수 점수 표시
                    score_text = f"{score:.0f}"
                    ax.text(idx, y_max * 0.98, score_text, 
                           color=color, fontweight='bold', ha='center', va='top', fontsize=5,
                           bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.1'))
                    
                    # 매수 근거 텍스트 표시 - 45도 각도로 설정
                    if buy_reasons:
                        reason_text = " | ".join(buy_reasons)
                        text_y_position = y_max * 0.9
                        # 긴 텍스트일 경우 위치 조정
                        if len(buy_reasons) > 2:
                            text_y_position = y_max * 0.88
                        
                        ax.text(idx, text_y_position, reason_text, 
                               color=color, fontsize=7, ha='right', va='top', rotation=45,
                               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
                    
                # MACD 차트에 매수 신호 레이블 표시 (MACD 크로스 발생 여부)
                if i == 4 and 'MACD_Cross' in self.data.columns and row['MACD_Cross'] == 1:
                    ax.text(idx, y_max * 0.9, 'MACD돌파', color=color, fontsize=8, fontweight='bold', 
                           ha='right', rotation=45, va='top',
                           bbox=dict(facecolor='white', alpha=0.8))
                           
                # RSI 차트에 매수 신호 레이블 표시 (RSI 반등 여부)
                if i == 3:
                    if 'RSI_Signal' in self.data.columns and row['RSI_Signal'] == 1:
                        ax.text(idx, y_max * 0.9, 'RSI반등', color=color, fontsize=8, fontweight='bold', 
                               ha='right', rotation=45, va='top',
                               bbox=dict(facecolor='white', alpha=0.8))
                    elif 'RSI_Up_Signal' in self.data.columns and row['RSI_Up_Signal'] == 1:
                        ax.text(idx, y_max * 0.8, 'RSI상승', color=color, fontsize=8, fontweight='bold', 
                               ha='right', rotation=45, va='top',
                               bbox=dict(facecolor='white', alpha=0.8))
                           
                # MFI 차트에 매수 신호 레이블 표시 (MFI 반등 여부)
                if i == 5 and 'MFI_Signal' in self.data.columns and row['MFI_Signal'] == 1:
                    ax.text(idx, y_max * 0.9, 'MFI반등', color=color, fontsize=8, fontweight='bold', 
                           ha='right', rotation=45, va='top',
                           bbox=dict(facecolor='white', alpha=0.8))
                           
                # 볼린저 밴드 %B 차트에 매수 신호 레이블 표시 (BB 반등 여부)
                if i == 6 and 'BB_Signal' in self.data.columns and row['BB_Signal'] == 1:
                    ax.text(idx, y_max * 0.9, 'BB반등', color=color, fontsize=8, fontweight='bold', 
                           ha='right', rotation=45, va='top',
                           bbox=dict(facecolor='white', alpha=0.8))
                
                # 이동평균선 차트에 매수 신호 레이블 표시
                if i == 1:
                    if 'MA_Cross' in self.data.columns and row['MA_Cross'] == 1:
                        ax.text(idx, y_max * 0.9, '골든크로스', color=color, fontsize=8, fontweight='bold', 
                               ha='right', rotation=45, va='top',
                               bbox=dict(facecolor='white', alpha=0.8))
                    elif 'MA_Trend_Signal' in self.data.columns and row['MA_Trend_Signal'] == 1:
                        ax.text(idx, y_max * 0.85, 'MA상승', color=color, fontsize=8, fontweight='bold', 
                               ha='right', rotation=45, va='top',
                               bbox=dict(facecolor='white', alpha=0.8))
                
                # 거래량 차트에 매수 신호 레이블 표시
                if i == 2 and 'Volume_Signal' in self.data.columns and row['Volume_Signal'] == 1:
                    ax.text(idx, y_max * 0.9, '거래량급증', color=color, fontsize=8, fontweight='bold', 
                           ha='right', rotation=45, va='top',
                           bbox=dict(facecolor='white', alpha=0.8))
        
        # 각 서브플롯에 해당 지표의 매수 신호 평가 내용 추가
        self._add_subplot_evaluations()
    
    def _add_sell_signal_markers(self):
        """모든 서브플롯에 매도 신호에 대한 수직선과 점수 표시"""
        # 매도 점수가 15점 이상인 신호만 필터링 (임계값 낮춤)
        sell_signals = self.data[self.data['Sell_Score'] >= 15].copy()
        
        if len(sell_signals) == 0:
            return
            
        # 모든 서브플롯에 매도 신호 수직선 추가
        for i, ax in enumerate(self.axes):
            y_min, y_max = ax.get_ylim()
            
            for idx, row in sell_signals.iterrows():
                score = row['Sell_Score']
                color = 'red' if score >= 70 else ('crimson' if score >= 50 else ('tomato' if score >= 30 else 'lightcoral'))
                linestyle = '-' if score >= 50 else ('--' if score >= 30 else ':')
                linewidth = 1.5 if score >= 50 else 1
                
                # 수직선 추가
                ax.axvline(x=idx, color=color, alpha=0.5, linestyle=linestyle, linewidth=linewidth)
                
                # 첫 번째 서브플롯(캔들스틱 차트)에 점수와 매도 근거 표시
                if i == 0:
                    # 매도 근거 텍스트 생성
                    sell_reasons = []
                    if 'MA_Death_Cross' in self.data.columns and row['MA_Death_Cross'] == 1:
                        sell_reasons.append("데드크로스")
                    if 'MA_Down_Trend_Signal' in self.data.columns and row['MA_Down_Trend_Signal'] == 1:
                        sell_reasons.append("MA하락")
                    if 'RSI_Sell_Signal' in self.data.columns and row['RSI_Sell_Signal'] == 1:
                        sell_reasons.append("RSI고점")
                    if 'RSI_Down_Signal' in self.data.columns and row['RSI_Down_Signal'] == 1:
                        sell_reasons.append("RSI하락")
                    if 'MACD_Death_Cross' in self.data.columns and row['MACD_Death_Cross'] == 1:
                        sell_reasons.append("MACD하락")
                    if 'MFI_Sell_Signal' in self.data.columns and row['MFI_Sell_Signal'] == 1:
                        sell_reasons.append("MFI고점")
                    if 'BB_Sell_Signal' in self.data.columns and row['BB_Sell_Signal'] == 1:
                        sell_reasons.append("BB고점")
                    if 'OBV_Sell_Signal' in self.data.columns and row['OBV_Sell_Signal'] == 1:
                        sell_reasons.append("OBV하락")
                    if 'Volume_Sell_Signal' in self.data.columns and row['Volume_Sell_Signal'] == 1:
                        sell_reasons.append("매도거래급증")
                    if 'Foreign_Sell_Signal' in self.data.columns and row['Foreign_Sell_Signal'] == 1:
                        sell_reasons.append("외국인매도")
                    
                    # 캔들 위에 매도 점수 표시
                    score_text = f"{score:.0f}"
                    ax.text(idx, y_max * 0.98, score_text, 
                           color=color, fontweight='bold', ha='center', va='top', fontsize=5,
                           bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.1'))
                    
                    # 매도 근거 텍스트 표시 - 45도 각도로 설정
                    if sell_reasons:
                        reason_text = " | ".join(sell_reasons)
                        text_y_position = y_max * 0.9
                        # 긴 텍스트일 경우 위치 조정
                        if len(sell_reasons) > 2:
                            text_y_position = y_max * 0.88
                        
                        ax.text(idx, text_y_position, reason_text, 
                               color=color, fontsize=7, ha='right', va='top', rotation=45,
                               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
                    
                # MACD 차트에 매도 신호 레이블 표시 (MACD 크로스 발생 여부)
                if i == 4 and 'MACD_Death_Cross' in self.data.columns and row['MACD_Death_Cross'] == 1:
                    ax.text(idx, y_max * 0.9, 'MACD하락', color=color, fontsize=8, fontweight='bold', 
                           ha='right', rotation=45, va='top',
                           bbox=dict(facecolor='white', alpha=0.8))
                           
                # RSI 차트에 매도 신호 레이블 표시 (RSI 고점 여부)
                if i == 3:
                    if 'RSI_Sell_Signal' in self.data.columns and row['RSI_Sell_Signal'] == 1:
                        ax.text(idx, y_max * 0.9, 'RSI고점', color=color, fontsize=8, fontweight='bold', 
                               ha='right', rotation=45, va='top',
                               bbox=dict(facecolor='white', alpha=0.8))
                    elif 'RSI_Down_Signal' in self.data.columns and row['RSI_Down_Signal'] == 1:
                        ax.text(idx, y_max * 0.8, 'RSI하락', color=color, fontsize=8, fontweight='bold', 
                               ha='right', rotation=45, va='top',
                               bbox=dict(facecolor='white', alpha=0.8))
                           
                # MFI 차트에 매도 신호 레이블 표시 (MFI 고점 여부)
                if i == 5 and 'MFI_Sell_Signal' in self.data.columns and row['MFI_Sell_Signal'] == 1:
                    ax.text(idx, y_max * 0.9, 'MFI고점', color=color, fontsize=8, fontweight='bold', 
                           ha='right', rotation=45, va='top',
                           bbox=dict(facecolor='white', alpha=0.8))
                           
                # 볼린저 밴드 %B 차트에 매도 신호 레이블 표시 (BB 고점 여부)
                if i == 6 and 'BB_Sell_Signal' in self.data.columns and row['BB_Sell_Signal'] == 1:
                    ax.text(idx, y_max * 0.9, 'BB고점', color=color, fontsize=8, fontweight='bold', 
                           ha='right', rotation=45, va='top',
                           bbox=dict(facecolor='white', alpha=0.8))
                
                # 이동평균선 차트에 매도 신호 레이블 표시
                if i == 1:
                    if 'MA_Death_Cross' in self.data.columns and row['MA_Death_Cross'] == 1:
                        ax.text(idx, y_max * 0.9, '데드크로스', color=color, fontsize=8, fontweight='bold', 
                               ha='right', rotation=45, va='top',
                               bbox=dict(facecolor='white', alpha=0.8))
                    elif 'MA_Down_Trend_Signal' in self.data.columns and row['MA_Down_Trend_Signal'] == 1:
                        ax.text(idx, y_max * 0.85, 'MA하락', color=color, fontsize=8, fontweight='bold', 
                               ha='right', rotation=45, va='top',
                               bbox=dict(facecolor='white', alpha=0.8))
                
                # 거래량 차트에 매도 신호 레이블 표시
                if i == 2 and 'Volume_Sell_Signal' in self.data.columns and row['Volume_Sell_Signal'] == 1:
                    ax.text(idx, y_max * 0.9, '매도거래급증', color=color, fontsize=8, fontweight='bold', 
                           ha='right', rotation=45, va='top',
                           bbox=dict(facecolor='white', alpha=0.8))
        
        # 각 서브플롯에 해당 지표의 매도 신호 평가 내용 추가
        # 매도 평가를 위한 별도 평가 함수 추가가 필요할 수 있으나,
        # 현재는 기존 평가 함수를 재사용
        self._add_subplot_evaluations()
    
    def _add_subplot_evaluations(self):
        """각 서브플롯에 해당 기술적 지표의 매수 신호 평가 내용 추가"""
        # 최신 데이터 가져오기
        latest = self.data.iloc[-1]
        
        # 1. 캔들스틱 차트 - 추세 평가
        ax = self.axes[0]
        trend_text = ""
        
        if 'Short_Trend' in latest and 'Mid_Trend' in latest and 'Long_Trend' in latest:
            short_trend = "상승" if latest['Short_Trend'] > 0 else "하락"
            mid_trend = "상승" if latest['Mid_Trend'] > 0 else "하락"
            long_trend = "상승" if latest['Long_Trend'] > 0 else "하락"
            
            trend_score = 0
            if latest['Short_Trend'] > 0: trend_score += 1
            if latest['Mid_Trend'] > 0: trend_score += 1
            if latest['Long_Trend'] > 0: trend_score += 1
            
            trend_eval = "매우 강세" if trend_score == 3 else ("강세" if trend_score == 2 else ("약세" if trend_score == 1 else "매우 약세"))
            trend_text = f"추세: {trend_eval} (단기: {short_trend}, 중기: {mid_trend}, 장기: {long_trend})"
            
        ax.text(0.02, 0.02, trend_text, transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
        
        # 2. 이동평균선 차트 - 골든크로스/데드크로스 평가
        ax = self.axes[1]
        ma_text = ""
        
        if 'MA5' in latest and 'MA20' in latest:
            ma5 = latest['MA5']
            ma20 = latest['MA20']
            
            if 'MA_Cross' in latest and latest['MA_Cross'] == 1:
                ma_text = "매수 신호: 골든크로스 발생 (MA5 > MA20)"
            elif ma5 > ma20:
                ma_text = "추세: 상승 중 (MA5 > MA20)"
            elif ma5 < ma20:
                ma_text = "추세: 하락 중 (MA5 < MA20)"
        
        ax.text(0.02, 0.02, ma_text, transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
        
        # 3. 거래량 차트 - 거래량 평가
        ax = self.axes[2]
        volume_text = ""
        
        if 'Volume' in latest and 'Volume_MA20' in latest:
            volume = latest['Volume']
            volume_ma20 = latest['Volume_MA20']
            
            if volume > volume_ma20 * 1.5:
                volume_text = "매수 신호: 거래량 급증 (20일 평균 대비 +50% 이상)"
            elif volume > volume_ma20:
                volume_text = "거래량: 20일 평균 이상"
            else:
                volume_text = "거래량: 20일 평균 이하"
        
        # 외국인 매수 정보 추가
        if 'Foreign_Buy' in latest:
            foreign_buy = latest['Foreign_Buy']
            foreign_buy_millions = foreign_buy / 1000000  # 백만 단위로 변환
            
            if foreign_buy > 0:
                volume_text += f" | 외국인: {foreign_buy_millions:.1f}백만원 순매수"
            else:
                volume_text += f" | 외국인: {abs(foreign_buy_millions):.1f}백만원 순매도"
        
        ax.text(0.02, 0.02, volume_text, transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
        
        # 4. RSI 차트 - RSI 평가
        ax = self.axes[3]
        rsi_text = ""
        
        if 'RSI' in latest:
            rsi = latest['RSI']
            
            if rsi > 70:
                rsi_text = f"과매수: RSI {rsi:.2f} (매도 고려)"
            elif rsi < 30:
                rsi_text = f"과매도: RSI {rsi:.2f} (매수 기회)"
            elif rsi >= 50 and rsi <= 70:
                rsi_text = f"강세: RSI {rsi:.2f} (중립 구간)"
            elif rsi >= 30 and rsi < 50:
                rsi_text = f"약세: RSI {rsi:.2f} (중립 구간)"
                
            if 'RSI_Signal' in latest and latest['RSI_Signal'] == 1:
                rsi_text += " - 매수 신호: 과매도 반등"
        
        ax.text(0.02, 0.02, rsi_text, transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
        
        # 5. MACD 차트 - MACD 평가
        ax = self.axes[4]
        macd_text = ""
        
        if 'MACD' in latest and 'MACD_Signal' in latest:
            macd = latest['MACD']
            signal = latest['MACD_Signal']
            
            if macd > signal:
                if 'MACD_Cross' in latest and latest['MACD_Cross'] == 1:
                    macd_text = f"매수 신호: MACD({macd:.2f}) > 시그널({signal:.2f}) 상향돌파"
                else:
                    macd_text = f"강세: MACD({macd:.2f}) > 시그널({signal:.2f})"
            else:
                macd_text = f"약세: MACD({macd:.2f}) < 시그널({signal:.2f})"
        
        ax.text(0.02, 0.02, macd_text, transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
        
        # 6. MFI 차트 - MFI 평가
        ax = self.axes[5]
        mfi_text = ""
        
        if 'MFI' in latest:
            mfi = latest['MFI']
            
            if mfi > 80:
                mfi_text = f"과매수: MFI {mfi:.2f} (매도 고려)"
            elif mfi < 20:
                mfi_text = f"과매도: MFI {mfi:.2f} (매수 기회)"
            elif mfi >= 50 and mfi <= 80:
                mfi_text = f"강세: MFI {mfi:.2f} (중립 구간)"
            elif mfi >= 20 and mfi < 50:
                mfi_text = f"약세: MFI {mfi:.2f} (중립 구간)"
                
            if 'MFI_Signal' in latest and latest['MFI_Signal'] == 1:
                mfi_text += " - 매수 신호: 과매도 반등"
        
        ax.text(0.02, 0.02, mfi_text, transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
        
        # 7. 보조지표 차트 - 볼린저 밴드 %B 평가
        ax = self.axes[6]
        bb_text = ""
        
        if 'BB_PB' in latest:
            bb_pb = latest['BB_PB']
            
            if bb_pb > 1:
                bb_text = f"과매수: %B {bb_pb:.2f} (밴드 상단 돌파)"
            elif bb_pb < 0:
                bb_text = f"과매도: %B {bb_pb:.2f} (밴드 하단 돌파)"
            elif bb_pb >= 0.8 and bb_pb <= 1:
                bb_text = f"강세: %B {bb_pb:.2f} (상단 근접)"
            elif bb_pb >= 0 and bb_pb <= 0.2:
                bb_text = f"약세: %B {bb_pb:.2f} (하단 근접)"
            else:
                bb_text = f"중립: %B {bb_pb:.2f}"
                
            if 'BB_Signal' in latest and latest['BB_Signal'] == 1:
                bb_text += " - 매수 신호: 과매도 반등"
                
        # OBV 평가 추가
        if 'OBV' in latest and 'OBV_Signal' in latest and latest['OBV_Signal'] == 1:
            bb_text += " | OBV: 상승 추세 (매수 신호)"
        
        ax.text(0.02, 0.02, bb_text, transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
        
    def save_chart(self, output_dir=None, dpi=300):
        """차트 저장"""
        if self.fig is None:
            print("ERROR: 저장할 차트가 없습니다.")
            return
        
        if output_dir is None:
            output_dir = os.path.join('output', datetime.now().strftime('%Y%m%d'))
        
        # 디렉토리가 없으면 생성
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if self.ticker:
            output_path = os.path.join(output_dir, f"{self.ticker}_technical_analysis.png")
            self.fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            print(f"차트가 {output_path}에 저장되었습니다.")
            return output_path
        else:
            print("WARNING: 종목코드가 설정되지 않아 차트를 저장할 수 없습니다.")
            return None
            
    def save_sell_chart(self, output_dir=None, dpi=300):
        """매도 신호 차트 저장"""
        if self.fig is None:
            print("ERROR: 저장할 차트가 없습니다.")
            return
        
        if output_dir is None:
            output_dir = os.path.join('output', datetime.now().strftime('%Y%m%d'))
        
        # 디렉토리가 없으면 생성
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if self.ticker:
            output_path = os.path.join(output_dir, f"{self.ticker}_sell_signals.png")
            self.fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            print(f"매도 신호 차트가 {output_path}에 저장되었습니다.")
            return output_path
        else:
            print("WARNING: 종목코드가 설정되지 않아 차트를 저장할 수 없습니다.")
            return None

    def set_consolidation_ranges(self, ranges):
        """박스권 정보 설정"""
        self.consolidation_ranges = ranges
        return self 