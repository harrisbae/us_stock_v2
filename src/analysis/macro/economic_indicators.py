"""
거시경제 지표 분석 모듈
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import platform
import pandas_datareader.data as web
import requests
import re
from bs4 import BeautifulSoup

# 한글 폰트 설정
if platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':  # Windows
    plt.rc('font', family='Malgun Gothic')
else:  # Linux
    plt.rc('font', family='NanumGothic')
    
plt.rc('axes', unicode_minus=False)  # 마이너스 기호 깨짐 방지

class MacroAnalysis:
    def __init__(self, symbol):
        self.symbol = symbol
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=365)  # 1년치 데이터
        
        # GICS 섹터 매핑
        self.sector_mapping = {
            'AAPL': '정보기술 (Information Technology)',
            'MSFT': '정보기술 (Information Technology)',
            'GOOGL': '커뮤니케이션서비스 (Communication Services)',
            'AMZN': '임의소비재 (Consumer Discretionary)',
            'META': '커뮤니케이션서비스 (Communication Services)',
            'NVDA': '정보기술 (Information Technology)',
            'TSLA': '임의소비재 (Consumer Discretionary)',
            'JPM': '금융 (Financials)',
            'V': '정보기술 (Information Technology)',
            'JNJ': '헬스케어 (Health Care)',
            'TEM': '헬스케어/AI (Health Care/AI)',
            'PLTR': '정보기술/AI (Information Technology/AI)',
            'GLD': '원자재 (Materials)',
            'BAC': '금융 (Financials)',
            'XLP': '필수소비재 (Consumer Staples)',
            'XLU': '유틸리티 (Utilities)',
            'XLE': '에너지 (Energy)',
            'TQQQ': '레버리지 기술주 (Leveraged Technology)',
            'RCAT': '기술 (Technology)',
            'COIN': '기술 (Technology)'
        }
        
        # GICS 서브섹터 매핑
        self.subsector_mapping = {
            'AAPL': '기술 하드웨어 및 장비',
            'MSFT': '소프트웨어',
            'GOOGL': '인터랙티브 미디어 및 서비스',
            'AMZN': '인터넷 및 직접 마케팅 소매',
            'META': '인터랙티브 미디어 및 서비스',
            'NVDA': '반도체 및 반도체 장비',
            'TSLA': '자동차',
            'JPM': '은행',
            'V': '정보기술 서비스',
            'JNJ': '제약',
            'TEM': 'AI 기반 의료 서비스',
            'PLTR': 'AI/빅데이터 분석 소프트웨어',
            'GLD': '귀금속 ETF',
            'BAC': '대형 상업은행',
            'XLP': '필수소비재 섹터 ETF',
            'XLU': '유틸리티 섹터 ETF',
            'XLE': '에너지 섹터 ETF',
            'TQQQ': '나스닥100 3X 레버리지 ETF',
            'RCAT': '컴퓨터 하드웨어 (Computer Hardware)',
            'COIN': '암호화폐 거래소 (Cryptocurrency Exchange)'
        }
        
        # 섹터별 적정 PER 범위
        self.sector_per_ranges = {
            '정보기술': (15, 30),
            '커뮤니케이션서비스': (12, 25),
            '임의소비재': (10, 20),
            '금융': (8, 15),
            '헬스케어': (15, 25),
            '원자재': (8, 15),
            '필수소비재': (12, 20),
            '유틸리티': (10, 18),
            '에너지': (8, 15),
            '레버리지': (5, 10)
        }
        
        # 섹터별 적정 PBR 범위
        self.sector_pbr_ranges = {
            '정보기술': (2.0, 4.0),
            '커뮤니케이션서비스': (1.5, 3.0),
            '임의소비재': (1.0, 2.5),
            '금융': (0.8, 1.5),
            '헬스케어': (2.0, 3.5),
            '원자재': (0.8, 1.8),
            '필수소비재': (1.5, 2.5),
            '유틸리티': (1.0, 2.0),
            '에너지': (0.8, 1.5),
            '레버리지': (0.5, 1.0)
        }
        
    def get_economic_data(self):
        """주요 경제 지표 데이터 수집"""
        # VIX - 변동성 지수
        self.vix = yf.download('^VIX', start=self.start_date, end=self.end_date)['Close']
        
        # 금리 관련
        self.tnx = yf.download('^TNX', start=self.start_date, end=self.end_date)['Close']  # 10년물 금리
        self.tyx = yf.download('^TYX', start=self.start_date, end=self.end_date)['Close']  # 30년물 금리
        
        # 달러 인덱스
        self.dxy = yf.download('DX-Y.NYB', start=self.start_date, end=self.end_date)['Close']
        
        # 주요 지수
        self.sp500 = yf.download('^GSPC', start=self.start_date, end=self.end_date)['Close']
        self.nasdaq = yf.download('^IXIC', start=self.start_date, end=self.end_date)['Close']
        
        # 대상 종목
        self.stock = yf.download(self.symbol, start=self.start_date, end=self.end_date)['Close']

    def calculate_correlations(self):
        """상관관계 분석"""
        # 모든 데이터를 동일한 인덱스로 리샘플링
        all_data = pd.DataFrame(index=self.stock.index)
        all_data['VIX'] = self.vix.reindex(self.stock.index)
        all_data['TNX'] = self.tnx.reindex(self.stock.index)
        all_data['TYX'] = self.tyx.reindex(self.stock.index)
        all_data['DXY'] = self.dxy.reindex(self.stock.index)
        all_data['S&P500'] = self.sp500.reindex(self.stock.index)
        all_data['NASDAQ'] = self.nasdaq.reindex(self.stock.index)
        all_data[self.symbol] = self.stock
        
        # NA 값 제거
        all_data = all_data.ffill().bfill()
        
        # 상관관계 계산
        return all_data.corr()

    def create_macro_dashboard(self, save_path=None):
        """거시경제 대시보드 생성"""
        # 데이터 수집
        self.get_economic_data()
        correlations = self.calculate_correlations()
        
        # 차트 생성
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'{self.symbol} 거시경제 지표 분석', fontsize=16)
        
        # 1. 주가 및 주요 지수 비교
        ax1 = fig.add_subplot(231)
        self._plot_price_comparison(ax1)
        
        # 2. VIX와의 관계
        ax2 = fig.add_subplot(232)
        self._plot_vix_analysis(ax2)
        
        # 3. 금리와의 관계
        ax3 = fig.add_subplot(233)
        self._plot_interest_rate_analysis(ax3)
        
        # 4. 달러 인덱스와의 관계
        ax4 = fig.add_subplot(234)
        self._plot_dxy_analysis(ax4)
        
        # 5. 상관관계 히트맵
        ax5 = fig.add_subplot(235)
        self._plot_correlation_heatmap(ax5, correlations)
        
        # 6. 투자 전략 제안
        ax6 = fig.add_subplot(236)
        self._plot_investment_strategy(ax6)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _plot_price_comparison(self, ax):
        """주가 및 주요 지수 비교 차트"""
        # 데이터 정규화
        normalized_data = pd.DataFrame(index=self.stock.index)
        normalized_data[self.symbol] = self.stock / self.stock.iloc[0] * 100
        normalized_data['S&P500'] = self.sp500 / self.sp500.iloc[0] * 100
        normalized_data['NASDAQ'] = self.nasdaq / self.nasdaq.iloc[0] * 100
        
        # NA 값 처리
        normalized_data = normalized_data.ffill().bfill()
        
        # 차트 그리기
        for col in normalized_data.columns:
            ax.plot(normalized_data.index, normalized_data[col], label=col)
        
        ax.set_title('상대 성과 비교 (기준: 100)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('상대 가격')

    def _plot_vix_analysis(self, ax):
        """VIX 분석 차트"""
        # VIX 이동평균선 계산
        vix_ma5 = self.vix.rolling(window=5).mean()
        vix_ma20 = self.vix.rolling(window=20).mean()
        vix_ma60 = self.vix.rolling(window=60).mean()
        # 기존 VIX 선(더 굵게)
        ax.plot(self.vix.index, self.vix, color='red', label='VIX', alpha=0.7, linewidth=2.5)
        # 이동평균선 추가 (범례 표기 없이)
        ax.plot(self.vix.index, vix_ma5, color='blue', linewidth=1.2)
        ax.plot(self.vix.index, vix_ma20, color='orange', linewidth=1.2)
        ax.plot(self.vix.index, vix_ma60, color='green', linewidth=1.2)
        # VIX 구간 표시
        ax.axhline(y=20, color='g', linestyle='--', alpha=0.3)
        ax.axhline(y=30, color='r', linestyle='--', alpha=0.3)
        vix_max = self.vix.max().iloc[0]
        ax.fill_between(self.vix.index, 0, 20, color='green', alpha=0.1, label='안정')
        ax.fill_between(self.vix.index, 20, 30, color='yellow', alpha=0.1, label='주의')
        ax.fill_between(self.vix.index, 30, vix_max*1.1, color='red', alpha=0.1, label='위험')
        # 범례를 VIX와 구간만 남기고 정리
        handles, labels = ax.get_legend_handles_labels()
        compact_labels = []
        compact_handles = []
        for h, l in zip(handles, labels):
            if l in ['VIX', '안정', '주의', '위험']:
                compact_labels.append(l)
                compact_handles.append(h)
        ax.legend(compact_handles, compact_labels, loc='upper left', fontsize=9, ncol=2, frameon=True)
        ax.set_title('VIX (변동성) 분석')
        ax.grid(True, alpha=0.3)

    def _plot_interest_rate_analysis(self, ax):
        """금리 분석 차트"""
        # 10년물, 30년물
        ax.plot(self.tnx.index, self.tnx, label='10년물', color='blue')
        ax.plot(self.tyx.index, self.tyx, label='30년물', color='red')
        # 기준금리(FEDFUNDS)
        try:
            fedfunds = web.DataReader('FEDFUNDS', 'fred', self.start_date, self.end_date)
            fedfunds = fedfunds.reindex(self.tnx.index).ffill()
            ax.plot(fedfunds.index, fedfunds['FEDFUNDS'], label='기준금리', color='purple', linestyle='--')
            current_fed = fedfunds['FEDFUNDS'].iloc[-1]
            if hasattr(current_fed, 'item'):
                current_fed = current_fed.item()
            ax.text(self.tyx.index[-1], current_fed, f'기준금리: {current_fed:.2f}%', color='purple', fontsize=8, verticalalignment='bottom')
        except Exception as e:
            print(f"FEDFUNDS 데이터 불러오기 실패: {e}")
        # 3개월물(TB3MS)
        try:
            tb3ms = web.DataReader('TB3MS', 'fred', self.start_date, self.end_date)
            tb3ms = tb3ms.reindex(self.tnx.index).ffill()
            ax.plot(tb3ms.index, tb3ms['TB3MS'], label='3개월물', color='green', linestyle=':')
            current_tb3 = tb3ms['TB3MS'].iloc[-1]
            if hasattr(current_tb3, 'item'):
                current_tb3 = current_tb3.item()
            ax.text(self.tyx.index[-1], current_tb3, f'3개월물: {current_tb3:.2f}%', color='green', fontsize=8, verticalalignment='top')
        except Exception as e:
            print(f"TB3MS 데이터 불러오기 실패: {e}")
        # 현재 금리 표시
        current_tnx = self.tnx.iloc[-1]
        current_tyx = self.tyx.iloc[-1]
        if hasattr(current_tnx, 'item'):
            current_tnx = current_tnx.item()
        if hasattr(current_tyx, 'item'):
            current_tyx = current_tyx.item()
        ax.text(self.tnx.index[-1], current_tnx, f'10년물: {current_tnx:.2f}%', verticalalignment='bottom', color='blue', fontsize=8)
        ax.text(self.tyx.index[-1], current_tyx, f'30년물: {current_tyx:.2f}%', verticalalignment='top', color='red', fontsize=8)
        ax.set_title('금리 동향')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('금리 (%)')

    def _plot_dxy_analysis(self, ax):
        """달러 인덱스 + GDP 성장률 추이 차트"""
        # DXY
        ax.plot(self.dxy.index, self.dxy, color='green', label='DXY')
        # 이동평균선 추가
        ma50 = self.dxy.rolling(window=50).mean()
        ma200 = self.dxy.rolling(window=200).mean()
        ax.plot(ma50.index, ma50, '--', color='orange', label='50일 평균')
        ax.plot(ma200.index, ma200, '--', color='red', label='200일 평균')
        # DXY 범례(좌측 y축)
        ax.legend(loc='upper left', fontsize=9)
        # GDP 성장률(분기별, 전년동기대비)
        # 성장/물가 공식 경제지표 변수 미리 선언
        self.gdp_growth = None
        self.gdp_growth_series = None
        self.cpi_infl = None
        self.cpi_infl_series = None
        self.pce_infl = None
        self.pce_infl_series = None
        try:
            gdp = web.DataReader('GDP', 'fred', self.start_date - pd.DateOffset(years=2), self.end_date)
            gdp = gdp.resample('QE').last()
            self.gdp_growth_series = (gdp['GDP'].pct_change(4) * 100).dropna()  # 시계열(도표용)
            self.gdp_growth = ((gdp['GDP'].iloc[-1] - gdp['GDP'].iloc[-5]) / gdp['GDP'].iloc[-5]) * 100  # 스칼라(리포트용)
        except Exception as e:
            print(f"GDP 데이터 불러오기 실패: {e}")
        try:
            cpi = web.DataReader('CPIAUCSL', 'fred', self.start_date - pd.DateOffset(years=2), self.end_date)
            cpi = cpi.resample('ME').last()
            self.cpi_infl_series = (cpi['CPIAUCSL'].pct_change(12) * 100).dropna()
            if not self.cpi_infl_series.empty:
                self.cpi_infl = self.cpi_infl_series.values[-1]
        except Exception as e:
            print(f"CPI 데이터 불러오기 실패: {e}")
        try:
            pce = web.DataReader('PCEPI', 'fred', self.start_date - pd.DateOffset(years=2), self.end_date)
            pce = pce.resample('ME').last()
            self.pce_infl_series = (pce['PCEPI'].pct_change(12) * 100).dropna()
            if not self.pce_infl_series.empty:
                self.pce_infl = self.pce_infl_series.values[-1]
        except Exception as e:
            print(f"PCE 데이터 불러오기 실패: {e}")
        ax2 = ax.twinx()
        # GDP 성장률(한 번만 plot)
        if isinstance(self.gdp_growth_series, pd.Series) and not self.gdp_growth_series.empty:
            ax2.plot(self.gdp_growth_series.index, self.gdp_growth_series.values, color='blue', marker='o', label='GDP 성장률(전년동기대비)')
        # CPI, PCE 도표
        if isinstance(self.cpi_infl_series, pd.Series) and not self.cpi_infl_series.empty:
            ax2.plot(self.cpi_infl_series.index, self.cpi_infl_series.values, color='red', label='CPI 상승률(전년동기대비)')
        if isinstance(self.pce_infl_series, pd.Series) and not self.pce_infl_series.empty:
            ax2.plot(self.pce_infl_series.index, self.pce_infl_series.values, color='orange', label='PCE 상승률(전년동기대비)')
        # 최근 값 텍스트/강조 등 기존 코드 유지
        if isinstance(self.gdp_growth_series, pd.Series) and not self.gdp_growth_series.empty:
            last_gdp_value = self.gdp_growth_series.values[-1]
            ax2.plot([self.gdp_growth_series.index[-1], self.end_date], [last_gdp_value, last_gdp_value], color='blue', linestyle='--', alpha=0.5)
            ax2.scatter(self.end_date, last_gdp_value, color='blue', marker='o')
            ax2.text(
                self.end_date, last_gdp_value, f'{last_gdp_value:.2f}%',
                color='blue', fontsize=8, va='bottom', rotation=45,
                bbox=dict(facecolor='white', alpha=0.45, edgecolor='none')
            )
            # 우측 상단에 현재 GDP 성장률 강조
            ax2.text(
                0.98, 0.95,
                f"현재 GDP 성장률: {last_gdp_value:.2f}%",
                transform=ax2.transAxes,
                ha='right', va='top',
                color='blue', fontsize=10, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )
        # CPI, PCE 최신값도 GDP와 동일하게 처리
        if isinstance(self.cpi_infl_series, pd.Series) and not self.cpi_infl_series.empty:
            last_cpi_value = self.cpi_infl_series.values[-1]
            ax2.plot([self.cpi_infl_series.index[-1], self.end_date], [last_cpi_value, last_cpi_value], color='red', linestyle='--', alpha=0.5)
            ax2.scatter(self.end_date, last_cpi_value, color='red', marker='o')
            ax2.text(
                self.end_date, last_cpi_value, f'{last_cpi_value:.2f}%',
                color='red', fontsize=8, va='bottom', rotation=45,
                bbox=dict(facecolor='white', alpha=0.45, edgecolor='none')
            )
        if isinstance(self.pce_infl_series, pd.Series) and not self.pce_infl_series.empty:
            last_pce_value = self.pce_infl_series.values[-1]
            ax2.plot([self.pce_infl_series.index[-1], self.end_date], [last_pce_value, last_pce_value], color='orange', linestyle='--', alpha=0.5)
            ax2.scatter(self.end_date, last_pce_value, color='orange', marker='o')
            ax2.text(
                self.end_date, last_pce_value, f'{last_pce_value:.2f}%',
                color='orange', fontsize=8, va='bottom', rotation=45,
                bbox=dict(facecolor='white', alpha=0.45, edgecolor='none')
            )
        # GDPNow 실시간 추정치 표시
        gdpnow = self.get_gdpnow_latest()
        self.gdpnow = gdpnow
        if gdpnow is not None:
            ax2.text(
                0.98, 0.85,
                f"실시간 GDPNow 추정치: {gdpnow:.2f}%",
                transform=ax2.transAxes,
                ha='right', va='top',
                color='purple', fontsize=10, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )
        else:
            ax2.text(
                0.98, 0.85,
                f"GDPNow 데이터 없음",
                transform=ax2.transAxes,
                ha='right', va='top',
                color='gray', fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
            )
        # GDP/CPI/PCE 범례(우측 상단)
        ax2.legend(loc='upper right', fontsize=9)
        ax.set_title('GDP & 달러인덱스 추이')
        ax.legend(loc='upper left')
        ax.set_ylabel('DXY')
        # X축 날짜 레이블 45도 회전
        plt.setp(ax.get_xticklabels(), rotation=45)
        # X축 기간을 DXY와 동일하게 맞춤
        ax.set_xlim(self.dxy.index[0], self.dxy.index[-1])
        if 'ax2' in locals():
            ax2.set_xlim(self.dxy.index[0], self.dxy.index[-1])

    def get_gdpnow_latest(self):
        url = "https://www.atlantafed.org/cqer/research/gdpnow"
        try:
            r = requests.get(url, timeout=5)
            # 1차: 정규식
            matches = re.findall(r"Latest estimate: ([\\d\\.]+) percent", r.text)
            if matches:
                return float(matches[0])
            # 2차: BeautifulSoup로 파싱
            soup = BeautifulSoup(r.text, 'html.parser')
            for tag in soup.find_all(string=re.compile(r"Latest estimate:")):
                m = re.search(r"Latest estimate: ([\\d\\.]+) percent", tag)
                if m:
                    return float(m.group(1))
        except Exception as e:
            print(f"GDPNow 파싱 오류: {e}")
        return None

    def _plot_correlation_heatmap(self, ax, correlations):
        """상관관계 히트맵"""
        im = ax.imshow(correlations, cmap='RdYlBu', aspect='auto')
        
        # 축 레이블 설정
        ax.set_xticks(np.arange(len(correlations.columns)))
        ax.set_yticks(np.arange(len(correlations.columns)))
        ax.set_xticklabels(correlations.columns, rotation=45, ha='right')
        ax.set_yticklabels(correlations.columns)
        
        # 상관계수 표시
        for i in range(len(correlations.columns)):
            for j in range(len(correlations.columns)):
                text = ax.text(j, i, f'{correlations.iloc[i, j]:.2f}',
                             ha='center', va='center')
        
        ax.set_title('지표간 상관관계')
        plt.colorbar(im, ax=ax)

    def _plot_investment_strategy(self, ax):
        """투자 전략 제안 + 성장/물가 국면 맵핑 표 + 자동 리포트"""
        report_lines = []
        # 최근 데이터 기반 투자 전략 도출
        current_vix = self.vix.iloc[-1]
        if hasattr(current_vix, 'item'):
            current_vix = current_vix.item()
        current_tnx = self.tnx.iloc[-1]
        if hasattr(current_tnx, 'item'):
            current_tnx = current_tnx.item()
        dxy_last = self.dxy.iloc[-1]
        dxy_20ago = self.dxy.iloc[-20]
        if hasattr(dxy_last, 'item'):
            dxy_last = dxy_last.item()
        if hasattr(dxy_20ago, 'item'):
            dxy_20ago = dxy_20ago.item()
        dxy_trend = ((dxy_last - dxy_20ago) / dxy_20ago * 100)
        sp500_last = self.sp500.iloc[-1]
        sp500_20ago = self.sp500.iloc[-20]
        nasdaq_last = self.nasdaq.iloc[-1]
        nasdaq_20ago = self.nasdaq.iloc[-20]
        if hasattr(sp500_last, 'item'):
            sp500_last = sp500_last.item()
        if hasattr(sp500_20ago, 'item'):
            sp500_20ago = sp500_20ago.item()
        if hasattr(nasdaq_last, 'item'):
            nasdaq_last = nasdaq_last.item()
        if hasattr(nasdaq_20ago, 'item'):
            nasdaq_20ago = nasdaq_20ago.item()
        sp500_trend = ((sp500_last - sp500_20ago) / sp500_20ago) * 100
        nasdaq_trend = ((nasdaq_last - nasdaq_20ago) / nasdaq_20ago) * 100
        # GDPNow 실시간 추정치 가져오기
        gdpnow = getattr(self, 'gdpnow', None)
        # 성장률(growth) 계산: GDPNow > 공식 GDP > (없으면) 주가지수 수익률
        growth = None
        if gdpnow is not None:
            growth = gdpnow
        elif getattr(self, 'gdp_growth', None) is not None:
            growth = self.gdp_growth
        else:
            # S&P500, NASDAQ 등 주가지수 수익률 평균(이전 로직)
            growth = (sp500_trend + nasdaq_trend) / 2
        # 물가상승률(inflation) 계산: CPI > (없으면) 금리, DXY 변화율
        inflation = None
        if getattr(self, 'cpi_infl', None) is not None:
            inflation = self.cpi_infl
        else:
            inflation = (current_tnx + dxy_trend) / 2
        # 성장/물가 국면 판별
        if growth > 0 and inflation > 0:
            regime = '성장↑, 물가↑ (리플레이션)'
            sector_rec = '에너지, 소재, 산업재'
            strat = '가치주, 인플레 방어주 비중 확대'
        elif growth > 0 and inflation <= 0:
            regime = '성장↑, 물가↓ (골디락스)'
            sector_rec = 'IT, 임의소비재, 헬스케어'
            strat = '성장주, 기술주 비중 확대'
        elif growth <= 0 and inflation > 0:
            regime = '성장↓, 물가↑ (스태그플레이션)'
            sector_rec = '필수소비재, 유틸리티, 헬스케어'
            strat = '방어주, 고배당주, 현금성 자산'
        else:
            regime = '성장↓, 물가↓ (디플레이션/침체)'
            sector_rec = '헬스케어, 필수소비재, 부동산'
            strat = '채권, 현금, 방어주 비중 확대'
        # GDPNow 실시간 전망 해설
        gdp_comment = None
        if gdpnow is not None and getattr(self, 'gdp_growth', None) is not None:
            if gdpnow > self.gdp_growth:
                gdp_comment = f"실시간 GDPNow 추정치는 {gdpnow:.2f}%로, 공식 발표치 대비 상승 전망입니다."
            elif gdpnow < self.gdp_growth:
                gdp_comment = f"실시간 GDPNow 추정치는 {gdpnow:.2f}%로, 공식 발표치 대비 하락 전망입니다."
            else:
                gdp_comment = f"실시간 GDPNow 추정치는 {gdpnow:.2f}%로, 공식 발표치와 유사한 수준입니다."
        elif gdpnow is not None:
            gdp_comment = f"실시간 GDPNow 추정치는 {gdpnow:.2f}%입니다."
        # 1. 분석 기준일
        report_date = self.end_date.strftime('%Y-%m-%d')
        report_lines.append(f"[분석 기준일: {report_date}]")
        # 2. 종목
        report_lines.append(f"[종목: {self.symbol}]")
        # 3. 섹터/서브섹터
        report_lines.append(f"[섹터: {self.get_gics_sector()}]")
        report_lines.append(f"[서브섹터: {self.get_gics_subsector()}]")
        # 4~5. 밸류에이션 정보
        valuation = self.get_valuation_analysis()
        if valuation:
            diff_str = f"적정주가 대비 괴리율: {valuation['price_diff']:+.1f}%"
            report_lines.append(f"[현재주가: ${valuation['current_price']:.2f}]  [Fair Value: ${valuation['fair_value']:.2f}]")
            report_lines.append(f"[{diff_str}]")
            report_lines.append(f"[밸류에이션: {valuation['valuation_status']}]")
            # 디버깅용: 상태와 괴리율 출력
            print(f"DEBUG: {self.symbol} | valuation_status={valuation['valuation_status']} | price_diff={valuation['price_diff']}")
            if valuation['growth'] is not None:
                report_lines.append(f"[성장률: {valuation['growth']:.1f}%]")
            if valuation['dividend_yield'] is not None:
                report_lines.append(f"[배당수익률: {valuation['dividend_yield']:.1f}%]")
            # 밸류에이션 해설 추가
            if valuation['valuation_status'] == '매우 저평가':
                report_lines.append("[해설] 현재 주가는 적정주가 대비 20% 이상 저평가되어 있습니다. 성장성, 수익성, 배당 등을 고려할 때 매수 매력이 높습니다.")
            elif valuation['valuation_status'] == '저평가':
                report_lines.append("[해설] 현재 주가는 적정주가 대비 10% 이상 저평가 구간입니다. 추가 상승 여력이 있습니다.")
            elif valuation['valuation_status'] == '매우 고평가':
                report_lines.append("[해설] 현재 주가는 적정주가 대비 20% 이상 고평가되어 있습니다. 신중한 접근이 필요합니다.")
            elif valuation['valuation_status'] == '고평가':
                report_lines.append("[해설] 현재 주가는 적정주가 대비 10% 이상 고평가 구간입니다. 단기 변동성에 유의하세요.")
            else:
                report_lines.append("[해설] 현재 주가는 적정주가와 유사한 수준입니다. 시장 평균 수준의 밸류에이션입니다.")
        # 6. 현재 국면
        report_lines.append(f"[현재 국면: {regime if 'regime' in locals() and regime else 'N/A'}]")
        report_lines.append(f"• 추천 섹터: {sector_rec if 'sector_rec' in locals() and sector_rec else 'N/A'}")
        report_lines.append(f"• 전략: {strat if 'strat' in locals() and strat else 'N/A'}")
        if gdp_comment:
            report_lines.append(f"[경제 해설] {gdp_comment}")
        # 경고 메시지(필요시)
        if len(report_lines) <= 4:
            report_lines.append("[경고] 데이터 부족")
        # PER/PBR/ROE 표 (오른쪽 상단, 간격 넓힘, 폰트 8)
        if valuation:
            table_data = [["PER", "PBR", "ROE(%)"],
                          [f"{valuation['per']:.1f}" if valuation['per'] is not None else '-',
                           f"{valuation['pbr']:.2f}" if valuation['pbr'] is not None else '-',
                           f"{valuation['roe']:.1f}" if valuation['roe'] is not None else '-']]
            table = ax.table(cellText=table_data, cellLoc='center', loc='center', bbox=[0.70, 0.68, 0.30, 0.08])
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            for (row, col), cell in table.get_celld().items():
                cell.set_linewidth(1)
                if row == 0:
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#e0e0e0')
        # [해설] 내용 줄바꿈 및 줄간격 넓힘
        y_base = 0.95
        y_gap = 0.035
        for i, line in enumerate(report_lines):
            if line.startswith('[해설]'):
                import textwrap
                wrapped = textwrap.fill(line, width=40)
                y = y_base - i * y_gap
                ax.text(0.05, y, wrapped, transform=ax.transAxes, va='top', fontsize=8, family='AppleGothic', linespacing=1.3)
            elif line.startswith('[현재 국면:') or line.startswith('• 추천 섹터:') or line.startswith('• 전략:'):
                y = y_base - (i + 2) * y_gap
                if line.startswith('[현재 국면:'):
                    ax.text(0.05, y, line, transform=ax.transAxes, va='top', fontsize=8, family='AppleGothic', bbox=dict(facecolor='red', alpha=0.15, edgecolor='none'))
                else:
                    ax.text(0.05, y, line, transform=ax.transAxes, va='top', fontsize=8, family='AppleGothic', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            elif line.startswith('[종목:'):
                y = y_base - i * y_gap
                ax.text(0.05, y, line, transform=ax.transAxes, va='top', fontsize=8, family='AppleGothic', bbox=dict(facecolor='red', alpha=0.15, edgecolor='none'))
            else:
                y = y_base - i * y_gap
                ax.text(0.05, y, line, transform=ax.transAxes, va='top', fontsize=8, family='AppleGothic', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        # 성장/물가 국면 맵핑 테이블 (크게, 아래쪽, 테두리 제거)
        mapping_data = [
            ["성장↑, 물가↑", "에너지, 소재, 산업재", "가치주, 인플레 방어"],
            ["성장↑, 물가↓", "IT, 임의소비재, 헬스케어", "성장주, 기술주"],
            ["성장↓, 물가↑", "필수소비재, 유틸리티, 헬스케어", "방어주, 고배당"],
            ["성장↓, 물가↓", "헬스케어, 필수소비재, 부동산", "채권, 현금, 방어주"]
        ]
        col_labels = ["성장/물가 국면", "추천 섹터", "전략 요약"]
        table2 = ax.table(cellText=mapping_data, colLabels=col_labels, cellLoc='center', loc='lower left', bbox=[0.05, 0.05, 0.85, 0.28])
        table2.set_fontsize(13)
        for (row, col), cell in table2.get_celld().items():
            cell.set_linewidth(0)
            if row == 0:
                cell.set_fontsize(15)
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#e0e0e0')
        # X, Y축 값(눈금, 라벨 등) 모두 제거
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis('off')
        # VIX/금리/달러 전략 가이드 (리포트 전략 라인 바로 아래에 텍스트로 출력, 2줄 아래로)
        strategies = []
        if current_vix < 20:
            strategies.append("• VIX < 20: 시장 안정구간  - 적극적 매수 고려, 레버리지 ETF 고려")
        elif current_vix < 30:
            strategies.append("• 20 ≤ VIX < 30: 주의구간  - 분할 매수 전략, 위험 관리 강화")
        else:
            strategies.append("• VIX ≥ 30: 위험구간  - 현금 비중 확대, 안전자산 선호")
        if current_tnx > 4:
            strategies.append("• 금리 > 4%: 고금리 구간  - 채권 비중 확대 고려, 배당주 주목")
        else:
            strategies.append("• 금리 < 4%: 저금리 구간  - 성장주 주목, 기술주 선호")
        if dxy_trend > 2:
            strategies.append("• 달러 강세  - 수출주 주의, 원자재 약세 예상")
        elif dxy_trend < -2:
            strategies.append("• 달러 약세  - 신흥국 투자 고려, 원자재 강세 예상")
        strategy_text = "\n".join(strategies)
        # 전략 라인 바로 아래에 출력 (리포트 y_base, y_gap 기준)
        strat_idx = None
        for i, line in enumerate(report_lines):
            if line.startswith("• 전략:"):
                strat_idx = i
                break
        if strat_idx is not None:
            y_strat = y_base - (strat_idx + 4) * y_gap
            ax.text(0.05, y_strat, strategy_text, transform=ax.transAxes, va='top', fontsize=8, family='AppleGothic')

    def get_gics_sector(self):
        """GICS 섹터 정보 조회"""
        return self.sector_mapping.get(self.symbol, '섹터 정보 없음')
        
    def get_gics_subsector(self):
        """GICS 서브섹터 정보 조회"""
        return self.subsector_mapping.get(self.symbol, '서브섹터 정보 없음')

    def get_valuation_data(self):
        """종목의 밸류에이션 데이터 수집"""
        try:
            # yfinance를 통해 재무 데이터 수집
            stock = yf.Ticker(self.symbol)
            info = stock.info
            # 현재 주가
            current_price = info.get('currentPrice', None)
            if current_price is None:
                current_price = info.get('regularMarketPrice', None)
            # PER
            per = info.get('trailingPE', None)
            # PBR
            pbr = info.get('priceToBook', None)
            # ROE
            roe = info.get('returnOnEquity', None)
            if roe is not None:
                roe = roe * 100  # 퍼센트로 변환
            # 성장률 (3년 평균)
            growth = info.get('earningsGrowth', None)
            if growth is not None:
                growth = growth * 100  # 퍼센트로 변환
            # 배당수익률
            dividend_yield = info.get('dividendYield', None)
            if dividend_yield is not None:
                dividend_yield = dividend_yield * 100  # 퍼센트로 변환
            return {
                'current_price': current_price,
                'per': per,
                'pbr': pbr,
                'roe': roe,
                'growth': growth,
                'dividend_yield': dividend_yield
            }
        except Exception as e:
            print(f"밸류에이션 데이터 수집 실패: {e}")
            return None
            
    def calculate_fair_value(self, valuation_data):
        """적정주가 산출"""
        if not valuation_data:
            return None
            
        # 섹터 정보 가져오기
        sector = self.get_gics_sector().split()[0]  # 한글 섹터명만 추출
        
        # 섹터별 적정 PER, PBR 범위
        per_range = self.sector_per_ranges.get(sector, (10, 20))
        pbr_range = self.sector_pbr_ranges.get(sector, (1.0, 2.0))
        
        fair_values = []
        
        # PER 기반 적정주가
        if valuation_data['per'] is not None and valuation_data['growth'] is not None:
            # 성장률을 고려한 적정 PER 계산
            growth_adjusted_per = min(per_range[1], max(per_range[0], valuation_data['growth']))
            fair_value_per = valuation_data['current_price'] * (growth_adjusted_per / valuation_data['per'])
            fair_values.append(fair_value_per)
        
        # PBR 기반 적정주가
        if valuation_data['pbr'] is not None and valuation_data['roe'] is not None:
            # ROE를 고려한 적정 PBR 계산
            roe_adjusted_pbr = min(pbr_range[1], max(pbr_range[0], valuation_data['roe'] / 10))
            fair_value_pbr = valuation_data['current_price'] * (roe_adjusted_pbr / valuation_data['pbr'])
            fair_values.append(fair_value_pbr)
        
        # 배당수익률 기반 적정주가
        if valuation_data['dividend_yield'] is not None:
            # 섹터별 적정 배당수익률 가정 (3~5%)
            target_yield = 4.0
            fair_value_div = valuation_data['current_price'] * (valuation_data['dividend_yield'] / target_yield)
            fair_values.append(fair_value_div)
        
        if fair_values:
            # 적정주가의 평균값 계산
            fair_value = sum(fair_values) / len(fair_values)
            return fair_value
        return None
        
    def get_valuation_analysis(self, premium=False):
        """밸류에이션 분석 결과 (premium: 미래 성장 프리미엄 반영)"""
        valuation_data = self.get_valuation_data()
        if not valuation_data:
            return None
        fair_value = self.calculate_fair_value(valuation_data)
        fair_value_premium = None
        premium_applied = False
        # PLTR 등 AI/빅데이터/고성장주는 프리미엄 적용
        ai_growth_stocks = ['PLTR', 'NVDA', 'TEM']
        if self.symbol in ai_growth_stocks or premium:
            premium_applied = True
            # 미래 성장률(5년 평균 30%)과 프리미엄 PER(50~100) 적용
            future_growth = 30.0  # %
            premium_per = 70.0    # 프리미엄 PER(가변)
            if valuation_data['per'] is not None:
                fair_value_premium = valuation_data['current_price'] * (premium_per / valuation_data['per'])
        if fair_value is None:
            return None
        current_price = valuation_data['current_price']
        price_diff = ((fair_value - current_price) / current_price) * 100
        # 밸류에이션 상태 판단
        if price_diff > 20:
            valuation_status = "매우 저평가"
        elif price_diff > 10:
            valuation_status = "저평가"
        elif price_diff < -20:
            valuation_status = "매우 고평가"
        elif price_diff < -10:
            valuation_status = "고평가"
        else:
            valuation_status = "적정가"
        result = {
            'current_price': current_price,
            'fair_value': fair_value,
            'price_diff': price_diff,
            'valuation_status': valuation_status,
            'per': valuation_data['per'],
            'pbr': valuation_data['pbr'],
            'roe': valuation_data['roe'],
            'growth': valuation_data['growth'],
            'dividend_yield': valuation_data['dividend_yield'],
            'premium_applied': premium_applied,
            'fair_value_premium': fair_value_premium
        }
        return result

def main(symbol, save_path=None):
    """메인 함수"""
    try:
        analyzer = MacroAnalysis(symbol)
        analyzer.create_macro_dashboard(save_path)
        print(f"거시경제 분석 완료: {save_path if save_path else '차트 표시'}")
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    return True

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python economic_indicators.py <symbol> [save_path]")
        sys.exit(1)
    
    symbol = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) > 2 else None
    main(symbol, save_path) 