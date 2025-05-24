"""
차트 시각화 모듈
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import platform
import mplfinance as mpf

# 한글 폰트 설정
if platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':  # Windows
    plt.rc('font', family='Malgun Gothic')
else:  # Linux
    plt.rc('font', family='NanumGothic')
    
plt.rc('axes', unicode_minus=False)  # 마이너스 기호 깨짐 방지

class ChartAnalysis:
    def __init__(self, symbol):
        self.symbol = symbol
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=365)  # 1년치 데이터
        self.data = None
        
    def get_stock_data(self):
        """주가 데이터 수집"""
        # Yahoo Finance에서 주가 데이터 수집
        self.data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        
        # 기술적 지표 계산
        self.calculate_indicators()

    def calculate_indicators(self):
        """기술적 지표 계산"""
        try:
            # 이동평균선
            self.data['MA5'] = self.data['Close'].rolling(window=5).mean()
            self.data['MA20'] = self.data['Close'].rolling(window=20).mean()
            self.data['MA60'] = self.data['Close'].rolling(window=60).mean()
            self.data['MA120'] = self.data['Close'].rolling(window=120).mean()
            
            # 볼린저 밴드 (20일, 2표준편차)
            bb_middle = self.data['Close'].rolling(window=20).mean()
            bb_std = self.data['Close'].rolling(window=20).std()
            self.data['BB_Middle'] = bb_middle
            self.data['BB_Upper'] = bb_middle + 2 * bb_std
            self.data['BB_Lower'] = bb_middle - 2 * bb_std
            
            # %B 지표 계산
            self.data['BB_PB'] = (self.data['Close'] - self.data['BB_Lower']) / (self.data['BB_Upper'] - self.data['BB_Lower'])
            
            # RSI (14일)
            delta = self.data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            self.data['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD (12, 26, 9)
            exp1 = self.data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = self.data['Close'].ewm(span=26, adjust=False).mean()
            self.data['MACD'] = exp1 - exp2
            self.data['Signal'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
            self.data['MACD_hist'] = self.data['MACD'] - self.data['Signal']
            
            return True
        except Exception as e:
            print(f"지표 계산 중 오류 발생: {str(e)}")
            return False

    def create_chart_dashboard(self, save_path=None):
        """차트 분석 대시보드 생성"""
        # 데이터 수집
        self.get_stock_data()
        
        # 차트 스타일 설정
        style = mpf.make_mpf_style(base_mpf_style='charles', rc={'font.family': plt.rcParams['font.family']})
        
        # 서브플롯 설정
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'{self.symbol} 차트 분석', fontsize=16)
        
        # 1. 캔들스틱 차트
        ax1 = fig.add_subplot(221)
        self._plot_candlestick(ax1)
        
        # 2. 볼린저 밴드
        ax2 = fig.add_subplot(222)
        self._plot_bollinger_bands(ax2)
        
        # 3. RSI
        ax3 = fig.add_subplot(223)
        self._plot_rsi(ax3)
        
        # 4. MACD
        ax4 = fig.add_subplot(224)
        self._plot_macd(ax4)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _plot_candlestick(self, ax):
        """캔들스틱 차트"""
        # 캔들스틱
        mpf.plot(self.data, type='candle', style='charles', ax=ax)
        
        # 이동평균선
        ax.plot(self.data.index, self.data['MA5'], label='MA5', alpha=0.7)
        ax.plot(self.data.index, self.data['MA20'], label='MA20', alpha=0.7)
        ax.plot(self.data.index, self.data['MA60'], label='MA60', alpha=0.7)
        ax.plot(self.data.index, self.data['MA120'], label='MA120', alpha=0.7)
        
        ax.set_title('캔들스틱 차트')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    def _plot_bollinger_bands(self, ax):
        """볼린저 밴드"""
        ax.plot(self.data.index, self.data['Close'], label='종가', alpha=0.7)
        ax.plot(self.data.index, self.data['BB_Upper'], label='상단', linestyle='--', alpha=0.7)
        ax.plot(self.data.index, self.data['BB_Middle'], label='중앙', alpha=0.7)
        ax.plot(self.data.index, self.data['BB_Lower'], label='하단', linestyle='--', alpha=0.7)
        
        ax.fill_between(self.data.index, self.data['BB_Upper'], self.data['BB_Lower'],
                       alpha=0.1)
        
        ax.set_title('볼린저 밴드')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    def _plot_rsi(self, ax):
        """RSI"""
        ax.plot(self.data.index, self.data['RSI'], label='RSI', color='purple', alpha=0.7)
        
        # 과매수/과매도 구간
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.3)
        ax.axhline(y=30, color='green', linestyle='--', alpha=0.3)
        ax.fill_between(self.data.index, 70, 100, color='red', alpha=0.1)
        ax.fill_between(self.data.index, 0, 30, color='green', alpha=0.1)
        
        ax.set_title('RSI (14)')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    def _plot_macd(self, ax):
        """MACD"""
        # MACD 라인과 시그널 라인
        ax.plot(self.data.index, self.data['MACD'], label='MACD', alpha=0.7)
        ax.plot(self.data.index, self.data['Signal'], label='Signal', alpha=0.7)
        
        # MACD 히스토그램
        colors = ['red' if x >= 0 else 'blue' for x in self.data['MACD_hist']]
        ax.bar(self.data.index, self.data['MACD_hist'], color=colors, alpha=0.5,
               label='Histogram')
        
        ax.set_title('MACD (12, 26, 9)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

def main(symbol, save_path=None):
    """메인 함수"""
    try:
        analyzer = ChartAnalysis(symbol)
        analyzer.create_chart_dashboard(save_path)
        print(f"차트 분석 완료: {save_path if save_path else '차트 표시'}")
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    return True

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python chart_analysis.py <symbol> [save_path]")
        sys.exit(1)
    
    symbol = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else None
    main(symbol, save_path) 