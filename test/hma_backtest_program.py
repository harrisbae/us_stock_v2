#!/usr/bin/env python3
"""
HMA Mantra 매수 시그널 백테스트 프로그램
목적: HMA Mantra 전략의 매수 시그널이 실제로 유효한지 검증
조건: 매수 신호 발생 시 USD 1,000씩 고정 매수하여 성과 분석
"""

import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class HMAMantraBacktest:
    def __init__(self, symbol, period="24mo", initial_capital=1000):
        """
        HMA Mantra 백테스트 초기화
        
        Args:
            symbol (str): 종목 심볼
            period (str): 백테스트 기간 (예: "24mo" = 2년)
            initial_capital (float): 매수당 투자 금액 (USD)
        """
        self.symbol = symbol
        self.period = period
        self.initial_capital = initial_capital
        self.data = None
        self.signals = []
        self.portfolio = []
        self.benchmark_data = None
        
    def load_data(self):
        """주식 데이터, 벤치마크 데이터, VIX 데이터 로드"""
        print(f"데이터 로드 중: {self.symbol}")
        
        try:
            # 주식 데이터 로드 (yf.download 사용 - 더 안정적)
            self.data = yf.download(self.symbol, period=self.period, interval="1d", progress=False)
            
            if self.data.empty:
                # 기간을 줄여서 다시 시도
                print(f"2년 데이터 없음, 1년 데이터로 시도...")
                self.data = yf.download(self.symbol, period="12mo", interval="1d", progress=False)
                
                if self.data.empty:
                    print(f"1년 데이터도 없음, 6개월 데이터로 시도...")
                    self.data = yf.download(self.symbol, period="6mo", interval="1d", progress=False)
                    
                    if self.data.empty:
                        raise ValueError(f"데이터를 가져올 수 없습니다: {self.symbol}")
                    else:
                        print(f"6개월 데이터 로드 완료")
                else:
                    print(f"1년 데이터 로드 완료")
            else:
                print(f"2년 데이터 로드 완료")
            
            # 벤치마크 데이터 로드 (S&P 500)
            self.benchmark_data = yf.download("^GSPC", period=self.period, interval="1d", progress=False)
            if self.benchmark_data.empty:
                self.benchmark_data = yf.download("^GSPC", period="12mo", interval="1d", progress=False)
            
            # VIX 데이터 로드
            print("VIX 데이터 로드 중...")
            self.vix_data = yf.download("^VIX", period=self.period, interval="1d", progress=False)
            if self.vix_data.empty:
                self.vix_data = yf.download("^VIX", period="12mo", interval="1d", progress=False)
                if self.vix_data.empty:
                    print("VIX 데이터 로드 실패, 기본값 사용")
                    self.vix_data = None
                else:
                    print("VIX 데이터 로드 완료 (1년)")
            else:
                print("VIX 데이터 로드 완료 (2년)")
            
            print(f"데이터 로드 완료: {len(self.data)} 개 데이터 포인트")
            
        except Exception as e:
            print(f"데이터 로드 중 오류: {e}")
            raise ValueError(f"데이터를 가져올 수 없습니다: {self.symbol}")
        
    def calculate_rsi(self, data, window=14):
        """RSI 계산"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self):
        """hma.sh와 동일한 복합 신호 로직(HMA, 만트라 밴드, RSI, MACD) 적용"""
        print("매수 신호 생성 중...")
        from src.indicators.hma_mantra.signals import get_hma_mantra_md_signals
        self.signals = [
            s for s in get_hma_mantra_md_signals(self.data, self.symbol)
            if s['type'] == 'BUY'
        ]
        print(f"매수 신호 생성 완료: {len(self.signals)} 개 신호")
        
    def calculate_returns(self, buy_date, buy_price, hold_periods=[30, 90, 180]):
        """매수 후 수익률 계산 (현재까지 보유 수익률 포함)"""
        returns = {}
        
        for period in hold_periods:
            # 매수 후 n일 후 날짜 계산
            sell_date = buy_date + timedelta(days=period)
            
            # 매도 날짜의 가격 찾기
            future_data = self.data[buy_date:sell_date]
            if len(future_data) > 0:
                sell_price = future_data['Close'].iloc[-1]
                period_return = (sell_price - buy_price) / buy_price * 100
                returns[f'{period}d'] = period_return
            else:
                returns[f'{period}d'] = None
        # 현재까지 보유 수익률
        last_price = self.data['Close'].iloc[-1]
        returns['to_now'] = (last_price - buy_price) / buy_price * 100
        return returns
    
    def calculate_benchmark_returns(self, buy_date, hold_periods=[30, 90, 180]):
        """벤치마크 수익률 계산 (매수일이 벤치마크 데이터에 없으면 가장 가까운 과거 거래일 사용)"""
        benchmark_returns = {}
        
        for period in hold_periods:
            sell_date = buy_date + timedelta(days=period)
            future_benchmark = self.benchmark_data[buy_date:sell_date]
            if len(future_benchmark) > 0:
                # 매수일의 벤치마크 데이터가 있는지 확인
                if buy_date in self.benchmark_data.index:
                    buy_benchmark = self.benchmark_data.loc[buy_date, 'Close']
                else:
                    # 가장 가까운 과거 거래일 찾기
                    prev_dates = self.benchmark_data.index[self.benchmark_data.index < buy_date]
                    if len(prev_dates) > 0:
                        nearest_date = prev_dates[-1]
                        buy_benchmark = self.benchmark_data.loc[nearest_date, 'Close']
                    else:
                        buy_benchmark = None
                sell_benchmark = future_benchmark['Close'].iloc[-1]
                if buy_benchmark is not None:
                    benchmark_return = (sell_benchmark - buy_benchmark) / buy_benchmark * 100
                    benchmark_returns[f'{period}d'] = benchmark_return
                else:
                    benchmark_returns[f'{period}d'] = None
            else:
                benchmark_returns[f'{period}d'] = None
        return benchmark_returns
    
    def run_backtest(self):
        """백테스트 실행 (매수 수량, 수익금 포함, VIX 지수 추가)"""
        print("백테스트 실행 중...")
        
        results = []
        
        for signal in self.signals:
            buy_date = signal['date']
            buy_price = signal['price']
            
            # 매수 수량 계산
            buy_amount = self.initial_capital
            buy_qty = round(buy_amount / buy_price, 4) if buy_price > 0 else 0
            
            # 수익률 계산
            returns = self.calculate_returns(buy_date, buy_price)
            benchmark_returns = self.calculate_benchmark_returns(buy_date)
            
            # VIX 지수값 가져오기
            vix_value = 'N/A'
            if self.vix_data is not None and buy_date in self.vix_data.index:
                vix_value = round(self.vix_data.loc[buy_date, 'Close'], 2)
            
            # 수익금 계산
            def calc_profit(r):
                if r is None:
                    return 'N/A'
                return round(buy_qty * buy_price * (r / 100), 2)
            profit_1m = calc_profit(returns.get('30d'))
            profit_3m = calc_profit(returns.get('90d'))
            profit_6m = calc_profit(returns.get('180d'))
            profit_now = calc_profit(returns.get('to_now'))
            
            # 신호 타당성 평가
            signal_evaluation = self.evaluate_signal(returns, benchmark_returns)
            
            result = {
                '매수 신호 발생일': buy_date.strftime('%Y-%m-%d'),
                '매수 가격': round(buy_price, 2),
                '매수 금액': buy_amount,
                '매수 수량': buy_qty,
                'VIX 지수': vix_value,
                '1개월 수익률(%)': round(returns.get('30d', 0), 2) if returns.get('30d') is not None else 'N/A',
                '1개월 수익금': profit_1m,
                '3개월 수익률(%)': round(returns.get('90d', 0), 2) if returns.get('90d') is not None else 'N/A',
                '3개월 수익금': profit_3m,
                '6개월 수익률(%)': round(returns.get('180d', 0), 2) if returns.get('180d') is not None else 'N/A',
                '6개월 수익금': profit_6m,
                '현재까지 수익률(%)': round(returns.get('to_now', 0), 2) if returns.get('to_now') is not None else 'N/A',
                '현재까지 수익금': profit_now,
                '벤치마크 1개월(%)': round(benchmark_returns.get('30d', 0), 2) if benchmark_returns.get('30d') is not None else 'N/A',
                '벤치마크 3개월(%)': round(benchmark_returns.get('90d', 0), 2) if benchmark_returns.get('90d') is not None else 'N/A',
                '벤치마크 6개월(%)': round(benchmark_returns.get('180d', 0), 2) if benchmark_returns.get('180d') is not None else 'N/A',
                '신호 타당성 평가': signal_evaluation
            }
            
            results.append(result)
        
        return results
    
    def evaluate_signal(self, returns, benchmark_returns):
        """신호 타당성 평가"""
        # 6개월 수익률 기준으로 평가
        def get_scalar(val):
            if isinstance(val, pd.Series):
                return val.values[0]
            return val
        if returns.get('180d') is not None and benchmark_returns.get('180d') is not None:
            stock_return = get_scalar(returns['180d'])
            benchmark_return = get_scalar(benchmark_returns['180d'])
            
            if stock_return > benchmark_return + 5:  # 벤치마크 대비 5% 이상 우수
                return "우수"
            elif stock_return > benchmark_return:  # 벤치마크 대비 우수
                return "양호"
            elif stock_return > 0:  # 양의 수익률
                return "보통"
            else:  # 손실
                return "불량"
        else:
            return "평가 불가"
    
    def generate_summary_statistics(self, results):
        """요약 통계 생성 (현재까지 수익률, 수익금 포함)"""
        if not results:
            return {}

        def get_scalar(val):
            if isinstance(val, pd.Series):
                return val.values[0]
            return val

        def is_valid(val):
            v = get_scalar(val)
            if isinstance(v, str):
                return v != 'N/A'
            if isinstance(v, (float, int, np.floating, np.integer)):
                return not pd.isna(v)
            return False

        returns_1m = [get_scalar(r['1개월 수익률(%)']) for r in results if is_valid(r['1개월 수익률(%)'])]
        returns_3m = [get_scalar(r['3개월 수익률(%)']) for r in results if is_valid(r['3개월 수익률(%)'])]
        returns_6m = [get_scalar(r['6개월 수익률(%)']) for r in results if is_valid(r['6개월 수익률(%)'])]
        returns_now = [get_scalar(r['현재까지 수익률(%)']) for r in results if is_valid(r['현재까지 수익률(%)'])]
        profits_1m = [get_scalar(r['1개월 수익금']) for r in results if is_valid(r['1개월 수익금'])]
        profits_3m = [get_scalar(r['3개월 수익금']) for r in results if is_valid(r['3개월 수익금'])]
        profits_6m = [get_scalar(r['6개월 수익금']) for r in results if is_valid(r['6개월 수익금'])]
        profits_now = [get_scalar(r['현재까지 수익금']) for r in results if is_valid(r['현재까지 수익금'])]
        
        # VIX 통계 계산
        vix_values = [get_scalar(r['VIX 지수']) for r in results if is_valid(r['VIX 지수'])]
        avg_vix = round(np.mean(vix_values), 2) if vix_values else 0
        min_vix = round(np.min(vix_values), 2) if vix_values else 0
        max_vix = round(np.max(vix_values), 2) if vix_values else 0
        
        summary = {
            '총 신호 수': len(results),
            '평균 VIX 지수': avg_vix,
            '최소 VIX 지수': min_vix,
            '최대 VIX 지수': max_vix,
            '평균 1개월 수익률(%)': round(np.mean(returns_1m), 2) if returns_1m else 0,
            '평균 3개월 수익률(%)': round(np.mean(returns_3m), 2) if returns_3m else 0,
            '평균 6개월 수익률(%)': round(np.mean(returns_6m), 2) if returns_6m else 0,
            '평균 현재까지 수익률(%)': round(np.mean(returns_now), 2) if returns_now else 0,
            '평균 1개월 수익금': round(np.mean(profits_1m), 2) if profits_1m else 0,
            '평균 3개월 수익금': round(np.mean(profits_3m), 2) if profits_3m else 0,
            '평균 6개월 수익금': round(np.mean(profits_6m), 2) if profits_6m else 0,
            '평균 현재까지 수익금': round(np.mean(profits_now), 2) if profits_now else 0,
            '총 1개월 수익금': round(np.sum(profits_1m), 2) if profits_1m else 0,
            '총 3개월 수익금': round(np.sum(profits_3m), 2) if profits_3m else 0,
            '총 6개월 수익금': round(np.sum(profits_6m), 2) if profits_6m else 0,
            '총 현재까지 수익금': round(np.sum(profits_now), 2) if profits_now else 0,
            '승률 1개월(%)': round(len([r for r in returns_1m if r > 0]) / len(returns_1m) * 100, 2) if returns_1m else 0,
            '승률 3개월(%)': round(len([r for r in returns_3m if r > 0]) / len(returns_3m) * 100, 2) if returns_3m else 0,
            '승률 6개월(%)': round(len([r for r in returns_6m if r > 0]) / len(returns_6m) * 100, 2) if returns_6m else 0,
            '현재까지 승률(%)': round(len([r for r in returns_now if r > 0]) / len(returns_now) * 100, 2) if returns_now else 0,
        }
        
        return summary
    
    def save_results(self, results, summary):
        """결과 저장 (수익금 통계 포함, 투자기간별 요약 표 추가)"""
        output_dir = Path(f"test/backtest_results/{self.symbol}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 모든 value가 Series면 float(스칼라)로 변환
        for r in results:
            for k, v in r.items():
                if isinstance(v, pd.Series):
                    r[k] = v.values[0]

        # 상세 결과를 CSV로 저장
        df_results = pd.DataFrame(results)
        csv_path = output_dir / f'{self.symbol}_backtest_detailed.csv'
        df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # 투자기간별 총 투자금액, 총 수익금, 투자수익률 계산
        n_signals = summary['총 신호 수']
        invest_1m = n_signals * self.initial_capital
        invest_3m = n_signals * self.initial_capital
        invest_6m = n_signals * self.initial_capital
        invest_now = n_signals * self.initial_capital
        profit_1m = summary['총 1개월 수익금']
        profit_3m = summary['총 3개월 수익금']
        profit_6m = summary['총 6개월 수익금']
        profit_now = summary['총 현재까지 수익금']
        roi_1m = round((profit_1m / invest_1m * 100) if invest_1m else 0, 2)
        roi_3m = round((profit_3m / invest_3m * 100) if invest_3m else 0, 2)
        roi_6m = round((profit_6m / invest_6m * 100) if invest_6m else 0, 2)
        roi_now = round((profit_now / invest_now * 100) if invest_now else 0, 2)

        # 요약 결과를 마크다운으로 저장
        md_path = output_dir / f'{self.symbol}_backtest_summary.md'

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# {self.symbol} HMA Mantra 백테스트 결과\n\n")
            f.write(f"**백테스트 기간**: {self.period}\n")
            f.write(f"**매수 금액**: USD {self.initial_capital:,}\n")
            f.write(f"**총 신호 수**: {summary['총 신호 수']}개\n\n")
            
            f.write("## 요약 통계\n\n")
            f.write("| 지표 | 값 |\n")
            f.write("|------|----|\n")
            f.write(f"| 평균 VIX 지수 | {summary['평균 VIX 지수']} |\n")
            f.write(f"| 최소 VIX 지수 | {summary['최소 VIX 지수']} |\n")
            f.write(f"| 최대 VIX 지수 | {summary['최대 VIX 지수']} |\n")
            f.write(f"| 평균 1개월 수익률 | {summary['평균 1개월 수익률(%)']}% |\n")
            f.write(f"| 평균 3개월 수익률 | {summary['평균 3개월 수익률(%)']}% |\n")
            f.write(f"| 평균 6개월 수익률 | {summary['평균 6개월 수익률(%)']}% |\n")
            f.write(f"| 평균 현재까지 수익률 | {summary['평균 현재까지 수익률(%)']}% |\n")
            f.write(f"| 평균 1개월 수익금 | {summary['평균 1개월 수익금']}$ |\n")
            f.write(f"| 평균 3개월 수익금 | {summary['평균 3개월 수익금']}$ |\n")
            f.write(f"| 평균 6개월 수익금 | {summary['평균 6개월 수익금']}$ |\n")
            f.write(f"| 평균 현재까지 수익금 | {summary['평균 현재까지 수익금']}$ |\n")
            f.write(f"| 총 1개월 수익금 | {summary['총 1개월 수익금']}$ |\n")
            f.write(f"| 총 3개월 수익금 | {summary['총 3개월 수익금']}$ |\n")
            f.write(f"| 총 6개월 수익금 | {summary['총 6개월 수익금']}$ |\n")
            f.write(f"| 총 현재까지 수익금 | {summary['총 현재까지 수익금']}$ |\n")
            f.write(f"| 1개월 승률 | {summary['승률 1개월(%)']}% |\n")
            f.write(f"| 3개월 승률 | {summary['승률 3개월(%)']}% |\n")
            f.write(f"| 6개월 승률 | {summary['승률 6개월(%)']}% |\n")
            f.write(f"| 현재까지 승률 | {summary['현재까지 승률(%)']}% |\n\n")

            # 투자기간별 요약 표
            f.write("## 투자기간별 총 투자금액, 총 수익금, 투자수익률\n\n")
            f.write("| 구분 | 총 투자금액($) | 총 수익금($) | 투자수익률(%) |\n")
            f.write("|------|:-------------:|:------------:|:-------------:|\n")
            f.write(f"| 1개월 | {invest_1m:,} | {profit_1m:,} | {roi_1m} |\n")
            f.write(f"| 3개월 | {invest_3m:,} | {profit_3m:,} | {roi_3m} |\n")
            f.write(f"| 6개월 | {invest_6m:,} | {profit_6m:,} | {roi_6m} |\n")
            f.write(f"| 현재까지 | {invest_now:,} | {profit_now:,} | {roi_now} |\n\n")
            
            f.write("## 상세 결과\n\n")
            f.write(df_results.to_markdown(index=False))
        
        print(f"결과 저장 완료:")
        print(f"  - 상세 결과: {csv_path}")
        print(f"  - 요약 결과: {md_path}")

    def create_visualization(self, results):
        """시각화 생성 (수익금 분포 추가)"""
        import matplotlib.pyplot as plt
        # 한글 폰트 설정 (macOS: AppleGothic)
        plt.rcParams['font.family'] = 'AppleGothic'
        plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지
        if not results:
            print("시각화할 데이터가 없습니다.")
            return
        
        # 결과 디렉토리 생성
        output_dir = Path(f"test/backtest_results/{self.symbol}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 수익률/수익금/VIX 분포 히스토그램
        fig, axes = plt.subplots(3, 4, figsize=(24, 18))
        fig.suptitle(f'{self.symbol} HMA Mantra 백테스트 결과', fontsize=16)
        
        # 1개월 수익률
        returns_1m = [r['1개월 수익률(%)'] for r in results if r['1개월 수익률(%)'] != 'N/A']
        if returns_1m:
            axes[0, 0].hist(returns_1m, bins=10, alpha=0.7, color='skyblue')
            axes[0, 0].axvline(np.mean(returns_1m), color='red', linestyle='--', label=f'평균: {np.mean(returns_1m):.2f}%')
            axes[0, 0].set_title('1개월 수익률 분포')
            axes[0, 0].set_xlabel('수익률 (%)')
            axes[0, 0].set_ylabel('빈도')
            axes[0, 0].legend()
        # 1개월 수익금
        profits_1m = [r['1개월 수익금'] for r in results if r['1개월 수익금'] != 'N/A']
        if profits_1m:
            axes[1, 0].hist(profits_1m, bins=10, alpha=0.7, color='deepskyblue')
            axes[1, 0].axvline(np.mean(profits_1m), color='red', linestyle='--', label=f'평균: {np.mean(profits_1m):.2f}$')
            axes[1, 0].set_title('1개월 수익금 분포')
            axes[1, 0].set_xlabel('수익금 ($)')
            axes[1, 0].set_ylabel('빈도')
            axes[1, 0].legend()
        # 3개월 수익률
        returns_3m = [r['3개월 수익률(%)'] for r in results if r['3개월 수익률(%)'] != 'N/A']
        if returns_3m:
            axes[0, 1].hist(returns_3m, bins=10, alpha=0.7, color='lightgreen')
            axes[0, 1].axvline(np.mean(returns_3m), color='red', linestyle='--', label=f'평균: {np.mean(returns_3m):.2f}%')
            axes[0, 1].set_title('3개월 수익률 분포')
            axes[0, 1].set_xlabel('수익률 (%)')
            axes[0, 1].set_ylabel('빈도')
            axes[0, 1].legend()
        # 3개월 수익금
        profits_3m = [r['3개월 수익금'] for r in results if r['3개월 수익금'] != 'N/A']
        if profits_3m:
            axes[1, 1].hist(profits_3m, bins=10, alpha=0.7, color='limegreen')
            axes[1, 1].axvline(np.mean(profits_3m), color='red', linestyle='--', label=f'평균: {np.mean(profits_3m):.2f}$')
            axes[1, 1].set_title('3개월 수익금 분포')
            axes[1, 1].set_xlabel('수익금 ($)')
            axes[1, 1].set_ylabel('빈도')
            axes[1, 1].legend()
        # 6개월 수익률
        returns_6m = [r['6개월 수익률(%)'] for r in results if r['6개월 수익률(%)'] != 'N/A']
        if returns_6m:
            axes[0, 2].hist(returns_6m, bins=10, alpha=0.7, color='orange')
            axes[0, 2].axvline(np.mean(returns_6m), color='red', linestyle='--', label=f'평균: {np.mean(returns_6m):.2f}%')
            axes[0, 2].set_title('6개월 수익률 분포')
            axes[0, 2].set_xlabel('수익률 (%)')
            axes[0, 2].set_ylabel('빈도')
            axes[0, 2].legend()
        # 6개월 수익금
        profits_6m = [r['6개월 수익금'] for r in results if r['6개월 수익금'] != 'N/A']
        if profits_6m:
            axes[1, 2].hist(profits_6m, bins=10, alpha=0.7, color='darkorange')
            axes[1, 2].axvline(np.mean(profits_6m), color='red', linestyle='--', label=f'평균: {np.mean(profits_6m):.2f}$')
            axes[1, 2].set_title('6개월 수익금 분포')
            axes[1, 2].set_xlabel('수익금 ($)')
            axes[1, 2].set_ylabel('빈도')
            axes[1, 2].legend()
        # 현재까지 수익률
        returns_now = [r['현재까지 수익률(%)'] for r in results if r['현재까지 수익률(%)'] != 'N/A']
        if returns_now:
            axes[0, 3].hist(returns_now, bins=10, alpha=0.7, color='purple')
            axes[0, 3].axvline(np.mean(returns_now), color='red', linestyle='--', label=f'평균: {np.mean(returns_now):.2f}%')
            axes[0, 3].set_title('현재까지 수익률 분포')
            axes[0, 3].set_xlabel('수익률 (%)')
            axes[0, 3].set_ylabel('빈도')
            axes[0, 3].legend()
        # 현재까지 수익금
        profits_now = [r['현재까지 수익금'] for r in results if r['현재까지 수익금'] != 'N/A']
        if profits_now:
            axes[1, 3].hist(profits_now, bins=10, alpha=0.7, color='indigo')
            axes[1, 3].axvline(np.mean(profits_now), color='red', linestyle='--', label=f'평균: {np.mean(profits_now):.2f}$')
            axes[1, 3].set_title('현재까지 수익금 분포')
            axes[1, 3].set_xlabel('수익금 ($)')
            axes[1, 3].set_ylabel('빈도')
            axes[1, 3].legend()
        
        # VIX 지수 분포
        vix_values = [r['VIX 지수'] for r in results if r['VIX 지수'] != 'N/A']
        if vix_values:
            axes[2, 0].hist(vix_values, bins=10, alpha=0.7, color='darkred')
            axes[2, 0].axvline(np.mean(vix_values), color='red', linestyle='--', label=f'평균: {np.mean(vix_values):.2f}')
            axes[2, 0].set_title('VIX 지수 분포')
            axes[2, 0].set_xlabel('VIX 지수')
            axes[2, 0].set_ylabel('빈도')
            axes[2, 0].legend()
        
        # VIX vs 수익률 산점도 (6개월)
        vix_6m_data = [(r['VIX 지수'], r['6개월 수익률(%)']) for r in results 
                       if r['VIX 지수'] != 'N/A' and r['6개월 수익률(%)'] != 'N/A']
        if vix_6m_data:
            vix_x = [x[0] for x in vix_6m_data]
            returns_y = [x[1] for x in vix_6m_data]
            axes[2, 1].scatter(vix_x, returns_y, alpha=0.6, color='purple')
            axes[2, 1].set_title('VIX vs 6개월 수익률')
            axes[2, 1].set_xlabel('VIX 지수')
            axes[2, 1].set_ylabel('6개월 수익률 (%)')
            
            # 상관계수 계산
            correlation = np.corrcoef(vix_x, returns_y)[0, 1]
            axes[2, 1].text(0.05, 0.95, f'상관계수: {correlation:.3f}', 
                           transform=axes[2, 1].transAxes, fontsize=10,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # VIX vs 수익률 산점도 (현재까지)
        vix_now_data = [(r['VIX 지수'], r['현재까지 수익률(%)']) for r in results 
                        if r['VIX 지수'] != 'N/A' and r['현재까지 수익률(%)'] != 'N/A']
        if vix_now_data:
            vix_x = [x[0] for x in vix_now_data]
            returns_y = [x[1] for x in vix_now_data]
            axes[2, 2].scatter(vix_x, returns_y, alpha=0.6, color='orange')
            axes[2, 2].set_title('VIX vs 현재까지 수익률')
            axes[2, 2].set_xlabel('VIX 지수')
            axes[2, 2].set_ylabel('현재까지 수익률 (%)')
            
            # 상관계수 계산
            correlation = np.corrcoef(vix_x, returns_y)[0, 1]
            axes[2, 2].text(0.05, 0.95, f'상관계수: {correlation:.3f}', 
                           transform=axes[2, 2].transAxes, fontsize=10,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 신호 타당성 평가 분포
        evaluations = [r['신호 타당성 평가'] for r in results]
        evaluation_counts = pd.Series(evaluations).value_counts()
        if len(evaluation_counts) > 0:
            axes[2, 3].bar(evaluation_counts.index, evaluation_counts.values, alpha=0.7, color=['green', 'lightgreen', 'yellow', 'red'])
            axes[2, 3].set_title('신호 타당성 평가 분포')
            axes[2, 3].set_xlabel('평가 등급')
            axes[2, 3].set_ylabel('신호 수')
            axes[2, 3].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{self.symbol}_backtest_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"시각화 저장 완료: {output_dir / f'{self.symbol}_backtest_results.png'}")
    
    def run(self):
        """전체 백테스트 실행"""
        print(f"=== {self.symbol} HMA Mantra 백테스트 시작 ===")
        
        # 1. 데이터 로드
        self.load_data()
        
        # 2. 신호 생성
        self.generate_signals()
        
        if not self.signals:
            print("매수 신호가 없습니다.")
            return [], {}
        
        # 3. 백테스트 실행
        results = self.run_backtest()
        
        # 4. 요약 통계 생성
        summary = self.generate_summary_statistics(results)
        
        # 5. 결과 저장
        self.save_results(results, summary)
        
        # 6. 시각화 생성
        self.create_visualization(results)
        
        # 7. 결과 출력
        print("\n=== 백테스트 완료 ===")
        print(f"총 신호 수: {summary['총 신호 수']}개")
        print(f"평균 6개월 수익률: {summary['평균 6개월 수익률(%)']}%")
        print(f"6개월 승률: {summary['승률 6개월(%)']}%")
        
        return results, summary

def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print("Usage: python hma_backtest_program.py <symbol> [period] [initial_capital]")
        print("Example: python hma_backtest_program.py SMR 24mo 1000")
        sys.exit(1)
    
    symbol = sys.argv[1]
    period = sys.argv[2] if len(sys.argv) > 2 else "24mo"
    initial_capital = float(sys.argv[3]) if len(sys.argv) > 3 else 1000
    
    try:
        backtest = HMAMantraBacktest(symbol, period, initial_capital)
        results, summary = backtest.run()
        
    except Exception as e:
        print(f"백테스트 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 