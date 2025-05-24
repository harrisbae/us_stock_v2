"""
GICS 섹터 분석 모듈
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import platform
import itertools

# 한글 폰트 설정
if platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':  # Windows
    plt.rc('font', family='Malgun Gothic')
else:  # Linux
    plt.rc('font', family='NanumGothic')
    
plt.rc('axes', unicode_minus=False)  # 마이너스 기호 깨짐 방지

# GICS 섹터 ETF 매핑
SECTOR_ETFS = {
    'XLK': '정보기술',
    'XLF': '금융',
    'XLV': '헬스케어',
    'XLE': '에너지',
    'XLY': '임의소비재',
    'XLP': '필수소비재',
    'XLI': '산업재',
    'XLB': '소재',
    'XLU': '유틸리티',
    'XLRE': '부동산',
    'XLC': '커뮤니케이션'
}

class SectorAnalysis:
    def __init__(self, symbol):
        self.symbol = symbol
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=365)  # 1년치 데이터
        
    def get_sector_data(self):
        """섹터 ETF 데이터 수집"""
        # 대상 종목 데이터
        self.stock = yf.download(self.symbol, start=self.start_date, end=self.end_date)['Close']
        
        # 섹터 ETF 데이터
        self.sector_data = {}
        for etf in SECTOR_ETFS.keys():
            data = yf.download(etf, start=self.start_date, end=self.end_date)['Close']
            self.sector_data[etf] = data

    def calculate_correlations(self):
        """섹터별 상관관계 분석"""
        # 모든 데이터를 데이터프레임으로 결합
        all_data = pd.DataFrame(index=self.stock.index)
        all_data[self.symbol] = self.stock
        
        for etf, name in SECTOR_ETFS.items():
            all_data[name] = self.sector_data[etf].reindex(self.stock.index)
        
        # NA 값 처리
        all_data = all_data.ffill().bfill()
        
        # 상관관계 계산
        return all_data.corr()

    def create_sector_dashboard(self, save_path=None):
        """섹터 분석 대시보드 생성"""
        # 데이터 수집
        self.get_sector_data()
        correlations = self.calculate_correlations()
        
        # 차트 생성
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'{self.symbol} GICS 섹터 분석', fontsize=16)
        
        # 1. 섹터별 상대 성과 비교
        ax1 = fig.add_subplot(231)
        self._plot_sector_performance(ax1, correlations)
        
        # 2. 섹터 상관관계 히트맵
        ax2 = fig.add_subplot(232)
        self._plot_correlation_heatmap(ax2, correlations)
        
        # 3. 섹터별 모멘텀 분석
        ax3 = fig.add_subplot(233)
        self._plot_sector_momentum(ax3)
        
        # 4. 섹터 로테이션 분석
        ax4 = fig.add_subplot(234)
        self._plot_sector_rotation(ax4)
        
        # 5. 섹터별 변동성 분석
        ax5 = fig.add_subplot(235)
        self._plot_sector_volatility(ax5)
        
        # 6. 투자 전략 제안
        ax6 = fig.add_subplot(236)
        self._plot_investment_strategy(ax6)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _plot_sector_performance(self, ax, correlations):
        """섹터별 상대 성과 비교"""
        # 색상/마커 리스트
        color_list = ['royalblue', 'forestgreen', 'orange', 'crimson', 'purple', 'teal', 'gold']
        marker_list = ['o', '^', 's', 'D', 'P', 'X', '*']
        color_cycle = itertools.cycle(color_list)
        marker_cycle = itertools.cycle(marker_list)
        
        # 데이터 정규화
        normalized_data = pd.DataFrame(index=self.stock.index)
        normalized_data[self.symbol] = self.stock / self.stock.iloc[0] * 100
        
        for etf, name in SECTOR_ETFS.items():
            data = self.sector_data[etf]
            normalized_data[name] = data / data.iloc[0] * 100
        
        # NA 값 처리
        normalized_data = normalized_data.ffill().bfill()
        
        # 상관계수 기준 설정
        corr_threshold = 0.65
        high_corr_sectors = []
        high_corr_colors = {}
        high_corr_markers = {}
        
        # 상관관계 높은 섹터에 색상/마커 미리 할당
        for col in normalized_data.columns:
            if col != self.symbol:
                corr = correlations.loc[self.symbol, col] if col in correlations.columns else 0
                if corr >= corr_threshold:
                    color = next(color_cycle)
                    marker = next(marker_cycle)
                    high_corr_sectors.append(col)
                    high_corr_colors[col] = color
                    high_corr_markers[col] = marker
        
        # 차트 그리기
        for col in normalized_data.columns:
            if col == self.symbol:
                # 해당 종목은 검정색, 두께 0.75로 표시
                ax.plot(normalized_data.index, normalized_data[col], 
                       label=col, linewidth=0.75, color='black', alpha=1.0,
                       zorder=10)
            elif col in high_corr_sectors:
                color = high_corr_colors[col]
                marker = high_corr_markers[col]
                # 라인
                ax.plot(normalized_data.index, normalized_data[col], 
                        label=col, linewidth=0.7, color=color, alpha=1.0, linestyle='-')
                # 마커(10개 간격마다, 크기 15)
                marker_idx = range(0, len(normalized_data.index), max(1, len(normalized_data.index)//10))
                ax.scatter(normalized_data.index[marker_idx], normalized_data[col].iloc[marker_idx], 
                           color=color, marker=marker, s=15, zorder=11)
            else:
                ax.plot(normalized_data.index, normalized_data[col], 
                        label=col, linewidth=0.5, color='gray', alpha=0.3, linestyle='-')
        
        ax.set_title('섹터별 상대 성과 비교 (기준: 100)')
        
        # 범례 위치와 스타일 조정 (폰트 사이즈 8 고정)
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                         fontsize=8, frameon=True, 
                         facecolor='white', edgecolor='gray',
                         title='종목 및 섹터')
        
        # 해당 종목과 상관관계 높은 섹터의 범례 항목을 굵게/색상 맞춤 표시
        for text in legend.get_texts():
            if text.get_text() == self.symbol:
                text.set_weight('bold')
                text.set_color('black')
            elif text.get_text() in high_corr_sectors:
                text.set_weight('bold')
                text.set_color(high_corr_colors[text.get_text()])
        
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('상대 가격')

    def _plot_correlation_heatmap(self, ax, correlations):
        """상관관계 히트맵"""
        im = ax.imshow(correlations, cmap='RdYlBu', aspect='auto')
        
        # 축 레이블 설정 (폰트 크기 8)
        ax.set_xticks(np.arange(len(correlations.columns)))
        ax.set_yticks(np.arange(len(correlations.columns)))
        ax.set_xticklabels(correlations.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(correlations.columns, fontsize=8)
        
        # 상관계수 표시 (폰트 크기 8)
        for i in range(len(correlations.columns)):
            for j in range(len(correlations.columns)):
                text = ax.text(j, i, f'{correlations.iloc[i, j]:.2f}',
                             ha='center', va='center', fontsize=8)
        
        ax.set_title('섹터간 상관관계')
        plt.colorbar(im, ax=ax)

    def _plot_sector_momentum(self, ax):
        """섹터별 모멘텀 분석"""
        # 20일, 60일 모멘텀 계산
        momentum_data = []
        
        for etf, name in SECTOR_ETFS.items():
            data = self.sector_data[etf]
            mom20 = (data.iloc[-1] / data.iloc[-20] - 1) * 100
            mom60 = (data.iloc[-1] / data.iloc[-60] - 1) * 100
            momentum_data.append({
                '섹터': name,
                '20일 모멘텀': mom20.iloc[0],
                '60일 모멘텀': mom60.iloc[0]
            })
        
        momentum_df = pd.DataFrame(momentum_data)
        momentum_df.set_index('섹터', inplace=True)
        
        # 차트 그리기
        x = np.arange(len(momentum_df.index))
        width = 0.35
        
        ax.bar(x - width/2, momentum_df['20일 모멘텀'], width, label='20일 모멘텀', color='skyblue')
        ax.bar(x + width/2, momentum_df['60일 모멘텀'], width, label='60일 모멘텀', color='lightcoral')
        
        ax.set_title('섹터별 모멘텀')
        ax.set_xticks(x)
        ax.set_xticklabels(momentum_df.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('수익률 (%)')

    def _plot_sector_rotation(self, ax):
        """섹터 로테이션 분석"""
        # 20일 변화율 계산
        rotation_data = []
        
        for etf, name in SECTOR_ETFS.items():
            data = self.sector_data[etf]
            change = (data.iloc[-1] / data.iloc[-20] - 1) * 100
            rotation_data.append({
                '섹터': name,
                '변화율': change.iloc[0]
            })
        
        rotation_df = pd.DataFrame(rotation_data)
        rotation_df.set_index('섹터', inplace=True)
        
        # 상위/하위 섹터 구분
        rotation_df = rotation_df.sort_values('변화율', ascending=False)
        colors = ['g' if x >= 0 else 'r' for x in rotation_df['변화율']]
        
        # 차트 그리기
        ax.barh(rotation_df.index, rotation_df['변화율'], color=colors)
        ax.set_title('섹터 로테이션 (20일 변화율)')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('변화율 (%)')

    def _plot_sector_volatility(self, ax):
        """섹터별 변동성 분석"""
        # 20일 변동성 계산
        volatility_data = []
        
        for etf, name in SECTOR_ETFS.items():
            data = self.sector_data[etf]
            returns = data.pct_change().dropna()  # NA 값 제거
            vol = (returns.std() * np.sqrt(252) * 100).iloc[0]  # Series에서 값 추출
            volatility_data.append({
                '섹터': name,
                '변동성': vol
            })
        
        volatility_df = pd.DataFrame(volatility_data)
        volatility_df.set_index('섹터', inplace=True)
        
        # 차트 그리기
        volatility_df = volatility_df.sort_values('변동성', ascending=True)
        ax.barh(volatility_df.index, volatility_df['변동성'], color='purple', alpha=0.6)
        ax.set_title('섹터별 변동성 (연간화)')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('변동성 (%)')

    def _plot_investment_strategy(self, ax):
        """투자 전략 제안"""
        # 섹터 분석 기반 전략 도출
        strategies = []
        
        # 모멘텀 상위 3개 섹터 선정
        momentum_data = []
        for etf, name in SECTOR_ETFS.items():
            data = self.sector_data[etf]
            mom20 = ((data.iloc[-1] / data.iloc[-20] - 1) * 100).iloc[0]  # Series에서 값 추출
            momentum_data.append({
                '섹터': name,
                '모멘텀': mom20
            })
        
        momentum_df = pd.DataFrame(momentum_data)
        momentum_df.set_index('섹터', inplace=True)
        
        top_sectors = momentum_df.sort_values('모멘텀', ascending=False).head(3)
        bottom_sectors = momentum_df.sort_values('모멘텀', ascending=True).head(3)
        
        # 전략 텍스트 생성
        strategies.append("• 강세 섹터 (매수 고려)")
        for sector, row in top_sectors.iterrows():
            strategies.append(f"  - {sector}: {row['모멘텀']:.1f}% 상승")
        
        strategies.append("\n• 약세 섹터 (비중 축소 고려)")
        for sector, row in bottom_sectors.iterrows():
            strategies.append(f"  - {sector}: {row['모멘텀']:.1f}% 하락")
        
        # 변동성 기반 리스크 관리
        vol_data = []
        for etf, name in SECTOR_ETFS.items():
            data = self.sector_data[etf]
            returns = data.pct_change().dropna()  # NA 값 제거
            vol = (returns.std() * np.sqrt(252) * 100).iloc[0]  # Series에서 값 추출
            vol_data.append({
                '섹터': name,
                '변동성': vol
            })
        
        vol_df = pd.DataFrame(vol_data)
        vol_df.set_index('섹터', inplace=True)
        
        high_vol_sectors = vol_df.sort_values('변동성', ascending=False).head(3)
        strategies.append("\n• 고변동성 섹터 (리스크 관리 필요)")
        for sector, row in high_vol_sectors.iterrows():
            strategies.append(f"  - {sector}: 변동성 {row['변동성']:.1f}%")
        
        # 전략 표시
        ax.text(0.05, 0.95, "섹터 투자 전략", fontsize=12, fontweight='bold',
                transform=ax.transAxes, va='top')
        
        strategy_text = "\n".join(strategies)
        ax.text(0.05, 0.85, strategy_text, transform=ax.transAxes,
                va='top', fontsize=10, linespacing=1.5)
        
        ax.axis('off')

def main(symbol, save_path=None):
    """메인 함수"""
    try:
        analyzer = SectorAnalysis(symbol)
        analyzer.create_sector_dashboard(save_path)
        print(f"섹터 분석 완료: {save_path if save_path else '차트 표시'}")
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    return True

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python gics_analysis.py <symbol> [save_path]")
        sys.exit(1)
    
    symbol = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else None
    main(symbol, save_path) 