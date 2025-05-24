"""
뉴스 분석 모듈
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import platform

# 한글 폰트 설정
if platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':  # Windows
    plt.rc('font', family='Malgun Gothic')
else:  # Linux
    plt.rc('font', family='NanumGothic')
    
plt.rc('axes', unicode_minus=False)  # 마이너스 기호 깨짐 방지

class NewsAnalysis:
    def __init__(self, symbol):
        self.symbol = symbol
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=30)  # 30일치 뉴스
        self.news_data = []
        
    def get_news_data(self):
        """뉴스 데이터 수집"""
        # Yahoo Finance에서 뉴스 데이터 수집
        stock = yf.Ticker(self.symbol)
        news = stock.news
        
        for item in news:
            self.news_data.append({
                'date': datetime.fromtimestamp(item.get('timestamp', 0)),
                'title': item.get('title', ''),
                'link': item.get('link', ''),
                'source': item.get('source', '')
            })
        
        # 데이터프레임으로 변환
        self.news_df = pd.DataFrame(self.news_data)
        if not self.news_df.empty:
            self.news_df.set_index('date', inplace=True)
            self.news_df.sort_index(inplace=True)

    def create_news_dashboard(self, save_path=None):
        """뉴스 분석 대시보드 생성"""
        # 데이터 수집
        self.get_news_data()
        
        if self.news_df.empty:
            print(f"경고: {self.symbol}에 대한 뉴스 데이터가 없습니다.")
            return
        
        # 차트 생성
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'{self.symbol} 뉴스 분석', fontsize=16)
        
        # 1. 일별 뉴스 건수
        ax1 = fig.add_subplot(221)
        self._plot_daily_news_count(ax1)
        
        # 2. 시간대별 뉴스 분포
        ax2 = fig.add_subplot(222)
        self._plot_hourly_distribution(ax2)
        
        # 3. 주요 뉴스 소스
        ax3 = fig.add_subplot(223)
        self._plot_news_sources(ax3)
        
        # 4. 최근 주요 뉴스
        ax4 = fig.add_subplot(224)
        self._plot_recent_news(ax4)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _plot_daily_news_count(self, ax):
        """일별 뉴스 건수"""
        daily_counts = self.news_df.resample('D').size()
        
        ax.bar(daily_counts.index, daily_counts.values, color='skyblue', alpha=0.7)
        ax.set_title('일별 뉴스 건수')
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('뉴스 건수')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    def _plot_hourly_distribution(self, ax):
        """시간대별 뉴스 분포"""
        hourly_counts = self.news_df.groupby(self.news_df.index.hour).size()
        
        ax.bar(hourly_counts.index, hourly_counts.values, color='lightcoral', alpha=0.7)
        ax.set_title('시간대별 뉴스 분포')
        ax.set_xlabel('시간')
        ax.set_ylabel('뉴스 건수')
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, alpha=0.3)

    def _plot_news_sources(self, ax):
        """주요 뉴스 소스"""
        source_counts = self.news_df['source'].value_counts().head(10)
        
        ax.barh(source_counts.index, source_counts.values, color='purple', alpha=0.6)
        ax.set_title('주요 뉴스 소스 (상위 10개)')
        ax.set_xlabel('뉴스 건수')
        ax.grid(True, alpha=0.3)

    def _plot_recent_news(self, ax):
        """최근 주요 뉴스"""
        # 최근 10개 뉴스 선택
        recent_news = self.news_df.head(10)
        
        headlines = []
        headlines.append("최근 주요 뉴스:")
        for _, row in recent_news.iterrows():
            date_str = row.name.strftime('%m/%d %H:%M')
            headlines.append(f"• [{date_str}] {row['title'][:50]}...")
        
        # 텍스트 표시
        ax.text(0.05, 0.95, "최근 뉴스 헤드라인", fontsize=12, fontweight='bold',
                transform=ax.transAxes, va='top')
        
        headline_text = "\n".join(headlines)
        ax.text(0.05, 0.85, headline_text, transform=ax.transAxes,
                va='top', fontsize=10, linespacing=1.5)
        
        ax.axis('off')

def main(symbol, save_path=None):
    """메인 함수"""
    try:
        analyzer = NewsAnalysis(symbol)
        analyzer.create_news_dashboard(save_path)
        print(f"뉴스 분석 완료: {save_path if save_path else '차트 표시'}")
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    return True

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python news_analysis.py <symbol> [save_path]")
        sys.exit(1)
    
    symbol = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else None
    main(symbol, save_path) 