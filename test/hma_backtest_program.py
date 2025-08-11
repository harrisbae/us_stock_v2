#!/usr/bin/env python3
"""
HMA Mantra 매수 시그널 백테스트 프로그램
목적: HMA Mantra 전략의 매수 시그널이 실제로 유효한지 검증
조건: 매수 신호 발생 시 USD 1,000씩 고정 매수하여 성과 분석
VIX 대역별 매수비용 지원 추가
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
import argparse
warnings.filterwarnings('ignore')
import pandas_datareader.data as web

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class HMAMantraBacktest:
    def __init__(self, symbol, period="24mo", initial_capital=1000, vix_low=0, vix_high=999, vix_bands=None, result_dir=None):
        """
        HMA Mantra 백테스트 초기화
        
        Args:
            symbol (str): 종목 심볼
            period (str): 백테스트 기간 (예: "24mo" = 2년)
            initial_capital (float): 기본 매수당 투자 금액 (USD)
            vix_low (float): VIX 하한값
            vix_high (float): VIX 상한값
            vix_bands (dict): VIX 대역별 매수비용 설정
            result_dir (str): 결과 저장 디렉토리
        """
        self.symbol = symbol
        self.period = period
        self.initial_capital = initial_capital
        self.vix_low = vix_low
        self.vix_high = vix_high
        self.vix_bands = vix_bands or {}
        self.result_dir = result_dir
        self.data = None
        self.signals = []
        self.portfolio = []
        self.benchmark_data = None
        self.vix_data = None
        self.sp500_data = None
        self.hyg_data = None
        self.tlt_data = None
        
    def parse_vix_bands(self, vix_bands_str):
        """VIX 대역 문자열을 파싱하여 딕셔너리로 변환"""
        if not vix_bands_str:
            return {}
            
        bands = {}
        for band_str in vix_bands_str.split(','):
            if ':' in band_str:
                band_name, cost_str = band_str.strip().split(':')
                try:
                    cost = int(cost_str)
                    bands[band_name.strip()] = cost
                except ValueError:
                    print(f"경고: 잘못된 비용 값 '{cost_str}' 무시")
                    
        return bands
    
    def get_vix_band_cost(self, vix_value):
        """VIX 값에 따른 매수비용 반환"""
        if not self.vix_bands:
            return self.initial_capital
            
        # VIX 대역별 비용 결정
        if vix_value < 20:
            return self.vix_bands.get('low', self.vix_bands.get('0-20', self.initial_capital))
        elif vix_value <= 25:
            return self.vix_bands.get('mid', self.vix_bands.get('20-25', self.initial_capital))
        else:
            return self.vix_bands.get('high', self.vix_bands.get('25+', self.initial_capital))
    
    def get_vix_value(self, date):
        """특정 날짜의 VIX 값 반환"""
        if self.vix_data is None:
            return 20  # 기본값
            
        # 해당 날짜 또는 가장 가까운 이전 날짜의 VIX 값
        try:
            vix_value = self.vix_data.loc[date, 'Close']
            return float(vix_value)
        except KeyError:
            # 해당 날짜가 없으면 가장 가까운 이전 날짜 찾기
            try:
                prev_dates = self.vix_data.index[self.vix_data.index <= date]
                if len(prev_dates) > 0:
                    vix_value = self.vix_data.loc[prev_dates[-1], 'Close']
                    return float(vix_value)
                else:
                    return 20  # 기본값
            except:
                return 20  # 기본값
    
    def check_vix_condition(self, date):
        """VIX 조건 확인"""
        vix_value = self.get_vix_value(date)
        return self.vix_low <= vix_value <= self.vix_high
    

    
    def load_data(self):
        """주식 데이터, 벤치마크 데이터, VIX 데이터, S&P 500, HYG, TLT 데이터 로드"""
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
            
            # S&P 500 데이터 로드 (NAIIM 계산용)
            print("S&P 500 데이터 로드 중...")
            self.sp500_data = yf.download("^GSPC", period=self.period, interval="1d", progress=False)
            if self.sp500_data.empty:
                self.sp500_data = yf.download("^GSPC", period="12mo", interval="1d", progress=False)
            print("S&P 500 데이터 로드 완료")
            
            # HYG 데이터 로드 (High Yield Spread 계산용)
            print("HYG 데이터 로드 중...")
            self.hyg_data = yf.download("HYG", period=self.period, interval="1d", progress=False)
            if self.hyg_data.empty:
                self.hyg_data = yf.download("HYG", period="12mo", interval="1d", progress=False)
            print("HYG 데이터 로드 완료")
            
            # TLT 데이터 로드 (High Yield Spread 계산용)
            print("TLT 데이터 로드 중...")
            self.tlt_data = yf.download("TLT", period=self.period, interval="1d", progress=False)
            if self.tlt_data.empty:
                self.tlt_data = yf.download("TLT", period="12mo", interval="1d", progress=False)
            print("TLT 데이터 로드 완료")
            
            # VIX 데이터 로드 (VIX 필터 사용 시 더 긴 기간 필요)
            print("VIX 데이터 로드 중...")
            vix_period = self.period
            if self.vix_low > 0 or self.vix_high < 999 or self.vix_bands:
                # VIX 필터 사용 시 더 긴 기간으로 VIX 데이터 로드
                if self.period == "6mo":
                    vix_period = "12mo"
                elif self.period == "12mo":
                    vix_period = "24mo"
                elif self.period == "24mo":
                    vix_period = "60mo"
                print(f"VIX 필터 사용으로 인해 VIX 데이터를 {vix_period} 기간으로 로드")
            
            self.vix_data = yf.download("^VIX", period=vix_period, interval="1d", progress=False)
            if self.vix_data.empty:
                # 더 짧은 기간으로 재시도
                fallback_periods = ["12mo", "6mo", "3mo"]
                for fallback_period in fallback_periods:
                    self.vix_data = yf.download("^VIX", period=fallback_period, interval="1d", progress=False)
                    if not self.vix_data.empty:
                        print(f"VIX 데이터 로드 완료 ({fallback_period})")
                        break
                
                if self.vix_data.empty:
                    print("VIX 데이터 로드 실패, 기본값 사용")
                    self.vix_data = None
            else:
                print(f"VIX 데이터 로드 완료 ({vix_period})")
            
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
        
        # 모든 매수 신호 가져오기
        all_signals = get_hma_mantra_md_signals(self.data, self.symbol)
        buy_signals = [s for s in all_signals if s['type'] == 'BUY']
        
        # VIX 조건 필터링
        if self.vix_low > 0 or self.vix_high < 999:
            filtered_signals = []
            for signal in buy_signals:
                if self.check_vix_condition(signal['date']):
                    filtered_signals.append(signal)
                else:
                    print(f"VIX 조건 미충족으로 신호 제외: {signal['date'].strftime('%Y-%m-%d')} (VIX: {self.get_vix_value(signal['date']):.2f})")
            self.signals = filtered_signals
            print(f"VIX 조건 필터링 후: {len(self.signals)} 개 신호 (원래 {len(buy_signals)} 개)")
        else:
            self.signals = buy_signals
            
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
                # Ticker 객체가 아닌 스칼라 값 반환
                if isinstance(period_return, pd.Series):
                    period_return = period_return.values[0]
                returns[f'{period}d'] = period_return
            else:
                returns[f'{period}d'] = None
        # 현재까지 보유 수익률
        last_price = self.data['Close'].iloc[-1]
        returns['to_now'] = (last_price - buy_price) / buy_price * 100
        # Ticker 객체가 아닌 스칼라 값 반환
        if isinstance(returns['to_now'], pd.Series):
            returns['to_now'] = returns['to_now'].values[0]
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
                    # Ticker 객체가 아닌 스칼라 값 반환
                    if isinstance(benchmark_return, pd.Series):
                        benchmark_return = benchmark_return.values[0]
                    benchmark_returns[f'{period}d'] = benchmark_return
                else:
                    benchmark_returns[f'{period}d'] = None
            else:
                benchmark_returns[f'{period}d'] = None
        return benchmark_returns
    
    def run_backtest(self):
        """백테스트 실행 (매수 수량, 수익금 포함, VIX 지수, NAIIM, HY Spread)"""
        print("백테스트 실행 중...")
        
        results = []
        
        for signal in self.signals:
            buy_date = signal['date']
            buy_price = signal['price']
            
            # VIX 대역별 매수비용 계산
            vix_value = self.get_vix_value(buy_date)
            buy_amount = self.get_vix_band_cost(vix_value)
            buy_qty = round(buy_amount / buy_price, 4) if buy_price > 0 else 0
            
            # 수익률 계산
            returns = self.calculate_returns(buy_date, buy_price)
            benchmark_returns = self.calculate_benchmark_returns(buy_date)
            
            # VIX 지수값 가져오기
            vix_value = 'N/A'
            if self.vix_data is not None and buy_date in self.vix_data.index:
                vix_value = round(self.vix_data.loc[buy_date, 'Close'], 2)
            
            # NAIIM 값 가져오기
            naiim_value = self.get_naiim_value(buy_date)
            
            # HY Spread 값 가져오기
            hy_spread_value = self.get_hy_spread_value(buy_date)
            
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
                'NAIIM': naiim_value,
                'HY Spread (%)': hy_spread_value,
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
        """요약 통계 생성 (현재까지 수익률, 수익금 포함, MDD, CAGR 추가)"""
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
        
        # VIX 대역별 매수 횟수 계산
        vix_low_count = 0
        vix_mid_count = 0
        vix_high_count = 0
        
        for r in results:
            vix_val = get_scalar(r['VIX 지수'])
            if is_valid(vix_val):
                if vix_val < 20:
                    vix_low_count += 1
                elif vix_val <= 25:
                    vix_mid_count += 1
                else:
                    vix_high_count += 1
        
        # NAIIM 통계 계산
        naiim_values = [get_scalar(r['NAIIM']) for r in results if is_valid(r['NAIIM'])]
        avg_naiim = round(np.mean(naiim_values), 2) if naiim_values else 50
        min_naiim = round(np.min(naiim_values), 2) if naiim_values else 50
        max_naiim = round(np.max(naiim_values), 2) if naiim_values else 50
        
        # HY Spread 통계 계산
        hy_spread_values = [get_scalar(r['HY Spread (%)']) for r in results if is_valid(r['HY Spread (%)'])]
        avg_hy_spread = round(np.mean(hy_spread_values), 2) if hy_spread_values else 3.0
        min_hy_spread = round(np.min(hy_spread_values), 2) if hy_spread_values else 3.0
        max_hy_spread = round(np.max(hy_spread_values), 2) if hy_spread_values else 3.0
        
        # MDD 및 CAGR 계산
        equity_curve = self.calculate_equity_curve(results)
        mdd, mdd_start, mdd_end = self.calculate_mdd(equity_curve)
        
        # 투자 기간 계산 (첫 거래일부터 마지막 거래일까지)
        if results:
            first_date = min(r['매수 신호 발생일'] for r in results)
            last_date = max(r.get('매도일', pd.Timestamp.now().date()) for r in results)
            
            # 날짜 타입 통일
            if isinstance(first_date, str):
                first_date = pd.to_datetime(first_date).date()
            elif isinstance(first_date, pd.Timestamp):
                first_date = first_date.date()
                
            if isinstance(last_date, str):
                last_date = pd.to_datetime(last_date).date()
            elif isinstance(last_date, pd.Timestamp):
                last_date = last_date.date()
                
            investment_years = (last_date - first_date).days / 365.25
        else:
            investment_years = 0
        
        # 초기 투자금액과 최종 포트폴리오 가치 계산
        initial_investment = sum(r['매수 금액'] for r in results if isinstance(r['매수 금액'], (int, float)))
        total_profit_now = round(np.sum(profits_now), 2) if profits_now else 0
        final_value = initial_investment + total_profit_now
        
        cagr = self.calculate_cagr(initial_investment, final_value, investment_years)
        
        summary = {
            '총 신호 수': len(results),
            '평균 VIX 지수': avg_vix,
            '최소 VIX 지수': min_vix,
            '최대 VIX 지수': max_vix,
            'VIX Low (<20) 매수 횟수': vix_low_count,
            'VIX Mid (20-25) 매수 횟수': vix_mid_count,
            'VIX High (>25) 매수 횟수': vix_high_count,
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
            # MDD 및 CAGR 추가
            'Maximum Drawdown (%)': round(mdd, 2),
            'MDD 시작일': mdd_start.strftime('%Y-%m-%d') if mdd_start else 'N/A',
            'MDD 종료일': mdd_end.strftime('%Y-%m-%d') if mdd_end else 'N/A',
            '투자 기간 (년)': round(investment_years, 2),
            'CAGR (%)': round(cagr, 2),
            '평균 NAIIM': avg_naiim,
            '최소 NAIIM': min_naiim,
            '최대 NAIIM': max_naiim,
            '평균 HY Spread (%)': avg_hy_spread,
            '최소 HY Spread (%)': min_hy_spread,
            '최대 HY Spread (%)': max_hy_spread,
        }
        
        return summary
    
    def save_results(self, results, summary):
        """결과 저장 (수익금 통계 포함, 투자기간별 요약 표 추가)"""
        output_dir = Path(self.result_dir) if hasattr(self, 'result_dir') and self.result_dir else Path(f"test/backtest_results/{self.symbol}")
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
        
        # 요약 통계를 CSV로 저장
        summary_df = pd.DataFrame([summary])
        summary_csv_path = output_dir / f'{self.symbol}_backtest_summary.csv'
        summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')

        # 투자기간별 총 투자금액, 총 수익금, 투자수익률 계산
        n_signals = summary['총 신호 수']
        
        # 실제 투자금액 계산 (VIX 대역별 매수비용 고려)
        total_investment = sum(r['매수 금액'] for r in results if isinstance(r['매수 금액'], (int, float)))
        
        invest_1m = total_investment
        invest_3m = total_investment
        invest_6m = total_investment
        invest_now = total_investment
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
            
            # 매수 금액 표시 (VIX 대역별 설정 고려)
            if self.vix_bands:
                f.write(f"**매수 금액**: VIX 대역별 설정\n")
                f.write(f"  - VIX < 20 (low): USD {self.vix_bands.get('low', self.initial_capital):,}\n")
                f.write(f"  - 20 ≤ VIX ≤ 25 (mid): USD {self.vix_bands.get('mid', self.initial_capital):,}\n")
                f.write(f"  - VIX > 25 (high): USD {self.vix_bands.get('high', self.initial_capital):,}\n")
                f.write(f"**총 투자금액**: USD {total_investment:,}\n")
            else:
                f.write(f"**매수 금액**: USD {self.initial_capital:,}\n")
            
            f.write(f"**총 신호 수**: {summary['총 신호 수']}개\n")
            f.write(f"**투자 기간**: {summary.get('투자 기간 (년)', 0):.2f}년\n\n")
            
            f.write("## 요약 통계\n\n")
            f.write("| 지표 | 값 |\n")
            f.write("|------|----|\n")
            f.write(f"| 평균 VIX 지수 | {summary['평균 VIX 지수']} |\n")
            f.write(f"| 최소 VIX 지수 | {summary['최소 VIX 지수']} |\n")
            f.write(f"| 최대 VIX 지수 | {summary['최대 VIX 지수']} |\n")
            f.write(f"| VIX Low (<20) 매수 횟수 | {summary['VIX Low (<20) 매수 횟수']}회 |\n")
            f.write(f"| VIX Mid (20-25) 매수 횟수 | {summary['VIX Mid (20-25) 매수 횟수']}회 |\n")
            f.write(f"| VIX High (>25) 매수 횟수 | {summary['VIX High (>25) 매수 횟수']}회 |\n")
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
            f.write(f"| 현재까지 승률 | {summary['현재까지 승률(%)']}% |\n")
            f.write(f"| Maximum Drawdown | {summary.get('Maximum Drawdown (%)', 0):.2f}% |\n")
            f.write(f"| MDD 기간 | {summary.get('MDD 시작일', 'N/A')} ~ {summary.get('MDD 종료일', 'N/A')} |\n")
            f.write(f"| CAGR | {summary.get('CAGR (%)', 0):.2f}% |\n")
            f.write(f"| 평균 NAIIM | {summary.get('평균 NAIIM', 50):.2f} |\n")
            f.write(f"| 최소 NAIIM | {summary.get('최소 NAIIM', 50):.2f} |\n")
            f.write(f"| 최대 NAIIM | {summary.get('최대 NAIIM', 50):.2f} |\n")
            f.write(f"| 평균 HY Spread (%) | {summary.get('평균 HY Spread (%)', 3.0):.2f}% |\n")
            f.write(f"| 최소 HY Spread (%) | {summary.get('최소 HY Spread (%)', 3.0):.2f}% |\n")
            f.write(f"| 최대 HY Spread (%) | {summary.get('최대 HY Spread (%)', 3.0):.2f}% |\n\n")

            # MDD 상세 설명 추가
            f.write("## Maximum Drawdown (MDD) 상세 분석\n\n")
            
            mdd = summary.get('Maximum Drawdown (%)', 0)
            mdd_start = summary.get('MDD 시작일', 'N/A')
            mdd_end = summary.get('MDD 종료일', 'N/A')
            
            f.write(f"**MDD 값**: {mdd:.2f}%\n")
            f.write(f"**MDD 기간**: {mdd_start} ~ {mdd_end}\n\n")
            
            if mdd <= -100:
                f.write("### ⚠️ MDD -100%의 의미\n\n")
                f.write("**MDD -100%는 포트폴리오 가치가 완전히 사라졌음을 의미합니다.**\n\n")
                f.write("**발생 가능한 케이스들**:\n\n")
                f.write("#### 케이스 1: 개별 종목 폭락\n")
                f.write("- 특정 종목이 급격히 하락하여 투자금액을 초과하는 손실 발생\n")
                f.write("- 예시: $1,000 투자 → 종목 가격 50% 하락 → $500 손실\n")
                f.write("- 하지만 이 경우 MDD는 -50%가 되어야 함\n\n")
                
                f.write("#### 케이스 2: 레버리지/마진 거래\n")
                f.write("- 레버리지나 마진 거래로 인한 추가 손실\n")
                f.write("- 예시: $1,000 투자 + $1,000 마진 → $2,000 손실 → MDD -100%\n\n")
                
                f.write("#### 케이스 3: 계산 로직 문제\n")
                f.write("- 백테스트 계산 과정에서 포트폴리오 가치가 잘못 계산됨\n")
                f.write("- 현재가 기준 재평가 시점의 문제\n")
                f.write("- 거래 데이터 누락 또는 중복 계산\n\n")
                
                f.write("#### 케이스 4: 극단적 시장 상황\n")
                f.write("- 급격한 시장 하락으로 모든 포지션이 동시에 손실\n")
                f.write("- 하지만 이는 매우 드문 경우\n\n")
                
                f.write("**현재 상황 분석**:\n")
                f.write("- AAPL은 비교적 안정적인 종목\n")
                f.write("- 6개월 기간 동안 극단적 하락 없음\n")
                f.write("- **MDD -100%는 계산 로직상의 문제일 가능성이 높음**\n\n")
                
                f.write("**권장 조치**:\n")
                f.write("1. 백테스트 계산 로직 점검\n")
                f.write("2. 포트폴리오 가치 계산 과정 검증\n")
                f.write("3. 실제 MDD 재계산 필요\n\n")
            else:
                f.write(f"### MDD {mdd:.2f}% 분석\n\n")
                if mdd > -20:
                    f.write("**우수한 리스크 관리**: MDD가 -20% 이하로 잘 관리되고 있습니다.\n")
                elif mdd > -50:
                    f.write("**적정한 리스크 수준**: MDD가 -50% 이하로 관리되고 있습니다.\n")
                else:
                    f.write("**높은 리스크 수준**: MDD가 -50%를 초과하여 리스크 관리가 필요합니다.\n")
                f.write("\n")

            # CAGR vs MDD 전략 분류 및 분석
            f.write("## 전략 분석 (CAGR vs MDD)\n\n")
            
            # 전략 분류 결정
            cagr = summary.get('CAGR (%)', 0)
            mdd = summary.get('Maximum Drawdown (%)', 0)
            
            if cagr > 15 and mdd > -20:
                strategy_type = "🟢 CAGR↑ & MDD↓ = 이상적인 전략"
                strategy_desc = "높은 수익률과 낮은 리스크를 모두 달성한 우수한 전략입니다."
                risk_level = "낮음"
                recommendation = "현재 전략을 유지하고 지속적으로 모니터링하세요."
            elif cagr > 15 and mdd <= -20:
                strategy_type = "🟡 CAGR↑ & MDD↑ = 고수익·고위험 전략"
                strategy_desc = "높은 수익률을 달성했지만 리스크도 높은 전략입니다."
                risk_level = "높음"
                recommendation = "수익률은 우수하지만 MDD를 줄이는 것이 중요합니다."
            elif cagr <= 15 and mdd > -20:
                strategy_type = "🔵 CAGR↓ & MDD↓ = 안정적이지만 수익 낮음"
                strategy_desc = "리스크는 낮지만 수익률이 낮은 안정적인 전략입니다."
                risk_level = "낮음"
                recommendation = "안정성을 유지하면서 수익률 개선을 모색하세요."
            else:
                strategy_type = "🔴 CAGR↓ & MDD↑ = 피해야 할 전략"
                strategy_desc = "낮은 수익률과 높은 리스크를 가진 전략입니다."
                risk_level = "매우 높음"
                recommendation = "전략을 근본적으로 재검토하고 개선이 필요합니다."
            
            f.write(f"**전략 분류**: {strategy_type}\n\n")
            f.write(f"**특징**: {strategy_desc}\n")
            f.write(f"**리스크 수준**: {risk_level}\n")
            f.write(f"**권장사항**: {recommendation}\n\n")
            
            # 전략 개선 방향
            f.write("### 전략 개선 방향\n\n")
            if cagr > 15 and mdd <= -20:
                f.write("**1순위: MDD 감소 (즉시 적용)**\n")
                f.write("- 손절 기준 설정 (-5% ~ -10%)\n")
                f.write("- 포지션 사이징 최적화\n")
                f.write("- 진입 조건 세분화 (RSI, MACD 등 추가)\n\n")
                f.write("**2순위: CAGR 유지 (중기)**\n")
                f.write("- 진입 조건 개선\n")
                f.write("- 시장 환경별 전략 조정\n\n")
                f.write("**3순위: 전략 최적화 (장기)**\n")
                f.write("- 다양한 시장 환경에서 검증\n")
                f.write("- 다른 전략과의 조합 검토\n\n")
            elif cagr <= 15 and mdd > -20:
                f.write("**1순위: 수익률 개선 (즉시 적용)**\n")
                f.write("- 진입 조건 최적화\n")
                f.write("- 홀딩 기간 조정\n")
                f.write("- 시장 환경별 전략 차별화\n\n")
            elif cagr <= 15 and mdd <= -20:
                f.write("**1순위: 전략 근본 재검토 (즉시 적용)**\n")
                f.write("- 백테스팅 파라미터 재설정\n")
                f.write("- 다른 전략 모델 검토\n")
                f.write("- 시장 환경 분석 및 대응\n\n")
            
            # 투자자별 권장사항
            f.write("### 투자자별 권장사항\n\n")
            if cagr > 15 and mdd > -20:
                f.write("**🟢 모든 투자자에게 적합**: 현재 전략을 유지하고 지속적으로 모니터링하세요.\n\n")
            elif cagr > 15 and mdd <= -20:
                f.write("**🟡 적극적 투자자**: 현재 전략이 적합하지만 리스크 관리 강화가 필요합니다.\n")
                f.write("**🟡 중립적 투자자**: 리스크 관리 개선 후 점진적 투자를 고려하세요.\n")
                f.write("**🔴 보수적 투자자**: MDD가 -20% 이하로 개선된 후 고려하세요.\n\n")
            elif cagr <= 15 and mdd > -20:
                f.write("**🟡 보수적 투자자**: 안정성은 우수하지만 수익률 개선이 필요합니다.\n")
                f.write("**🔴 적극적 투자자**: 더 높은 수익률을 추구하는 전략을 고려하세요.\n\n")
            else:
                f.write("**🔴 모든 투자자에게 부적합**: 전략을 근본적으로 재검토하고 개선이 필요합니다.\n\n")

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
        print(f"  - 요약 통계: {summary_csv_path}")
        print(f"  - 요약 결과: {md_path}")

    def create_visualization(self, results, summary):
        """시각화 생성 (수익금 분포 추가, MDD/CAGR 정보 포함, NAIIM 차트 추가)"""
        # 한글 폰트 설정 (macOS: AppleGothic)
        plt.rcParams['font.family'] = 'AppleGothic'
        plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지
        if not results:
            print("시각화할 데이터가 없습니다.")
            return
        
        # 결과 디렉토리 생성
        output_dir = Path(self.result_dir) if hasattr(self, 'result_dir') and self.result_dir else Path(f"test/backtest_results/{self.symbol}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # High Yield Spread 데이터 불러오기 (FRED)
        try:
            hy_spread = web.DataReader('BAMLH0A0HYM2', 'fred', start=self.data.index.min(), end=self.data.index.max())
        except Exception as e:
            print(f"High Yield Spread 데이터 로드 실패: {e}")
            hy_spread = None
        
        # 1. 수익률/수익금/VIX 분포 히스토그램 + High Yield Spread subplot + MDD/CAGR + NAIIM
        fig, axes = plt.subplots(4, 4, figsize=(24, 24))
        fig.suptitle(f'{self.symbol} HMA Mantra 백테스트 결과', fontsize=16)
        
        # MDD 및 CAGR 정보 표시 (summary에서 가져오기)
        mdd = summary.get('Maximum Drawdown (%)', 0)
        mdd_start = summary.get('MDD 시작일', 'N/A')
        mdd_end = summary.get('MDD 종료일', 'N/A')
        
        # 투자 기간 계산
        if results:
            first_date = min(r['매수 신호 발생일'] for r in results)
            last_date = max(r.get('매도일', pd.Timestamp.now().date()) for r in results)
            
            # 날짜 타입 통일
            if isinstance(first_date, str):
                first_date = pd.to_datetime(first_date).date()
            elif isinstance(first_date, pd.Timestamp):
                first_date = first_date.date()
                
            if isinstance(last_date, str):
                last_date = pd.to_datetime(last_date).date()
            elif isinstance(last_date, pd.Timestamp):
                last_date = last_date.date()
                
            investment_years = (last_date - first_date).days / 365.25
        else:
            investment_years = 0
        
        # 초기 투자금액과 최종 포트폴리오 가치
        initial_investment = sum(r['매수 금액'] for r in results if isinstance(r['매수 금액'], (int, float)))
        final_value = initial_investment + summary.get('총 현재까지 수익금', 0) if 'summary' in locals() else initial_investment
        
        cagr = self.calculate_cagr(initial_investment, final_value, investment_years)
        
        # MDD/CAGR 정보를 제목에 추가
        fig.suptitle(f'{self.symbol} HMA Mantra 백테스트 결과\nMDD: {mdd:.2f}%, CAGR: {cagr:.2f}%', fontsize=16)
        
        # 매수일자 목록 추출 (수직선 표시용)
        buy_dates = []
        for r in results:
            if isinstance(r['매수 신호 발생일'], str):
                buy_date = pd.to_datetime(r['매수 신호 발생일'])
            else:
                buy_date = r['매수 신호 발생일']
            buy_dates.append(buy_date)
        
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
            axes[1, 0].set_xlabel('수익금 ($)')
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
        
        # 4행 1열: High Yield Spread 시계열
        if hy_spread is not None:
            axes[3, 0].plot(hy_spread.index, hy_spread.values, color='purple', linewidth=2)
            axes[3, 0].set_title('High Yield Spread (BAMLH0A0HYM2)')
            axes[3, 0].set_xlabel('날짜')
            axes[3, 0].set_ylabel('High Yield Spread (%)')
            axes[3, 0].grid(True, alpha=0.3)
            
            # 매수일자별 수직선 추가
            for buy_date in buy_dates:
                if buy_date in hy_spread.index:
                    axes[3, 0].axvline(x=buy_date, color='red', linestyle='--', alpha=0.7, linewidth=1)
        else:
            axes[3, 0].text(0.5, 0.5, 'High Yield Spread\n데이터 없음', 
                           ha='center', va='center', transform=axes[3, 0].transAxes)
            axes[3, 0].set_title('High Yield Spread')
        
        # 4행 2열: NAIIM 시계열
        if hasattr(self, 'sp500_data') and self.sp500_data is not None:
            # NAIIM 데이터 계산
            naiim_data = []
            naiim_dates = []
            
            for date in self.sp500_data.index:
                naiim_value = self.get_naiim_value(date)
                naiim_data.append(naiim_value)
                naiim_dates.append(date)
            
            axes[3, 1].plot(naiim_dates, naiim_data, color='blue', linewidth=2)
            axes[3, 1].set_title('NAIIM (S&P 500 모멘텀 기반)')
            axes[3, 1].set_xlabel('날짜')
            axes[3, 1].set_ylabel('NAIIM')
            axes[3, 1].grid(True, alpha=0.3)
            axes[3, 1].set_ylim(0, 100)
            
            # 매수일자별 수직선 추가
            for buy_date in buy_dates:
                if buy_date in self.sp500_data.index:
                    axes[3, 1].axvline(x=buy_date, color='red', linestyle='--', alpha=0.7, linewidth=1)
        else:
            axes[3, 1].text(0.5, 0.5, 'NAIIM\n데이터 없음', 
                           ha='center', va='center', transform=axes[3, 1].transAxes)
            axes[3, 1].set_title('NAIIM')
        
        # 4행 3열: 빈 차트 (투자심리도 제거)
        axes[3, 2].text(0.5, 0.5, '투자심리도\n차트 제거됨', 
                       ha='center', va='center', transform=axes[3, 2].transAxes)
        axes[3, 2].set_title('투자심리도 (제거됨)')
        
        # 4행 4열: HY Spread (HYG/TLT 기반)
        if hasattr(self, 'hyg_data') and hasattr(self, 'tlt_data') and self.hyg_data is not None and self.tlt_data is not None:
            # HY Spread 데이터 계산
            hy_spread_data = []
            hy_spread_dates = []
            
            for date in self.hyg_data.index:
                if date in self.tlt_data.index:
                    hy_spread_value = self.get_hy_spread_value(date)
                    hy_spread_data.append(hy_spread_value)
                    hy_spread_dates.append(date)
            
            axes[3, 3].plot(hy_spread_dates, hy_spread_data, color='orange', linewidth=2)
            axes[3, 3].set_title('HY Spread (HYG/TLT 기반)')
            axes[3, 3].set_xlabel('날짜')
            axes[3, 3].set_ylabel('HY Spread (%)')
            axes[3, 3].grid(True, alpha=0.3)
            
            # 매수일자별 수직선 추가
            for buy_date in buy_dates:
                if buy_date in self.hyg_data.index:
                    axes[3, 3].axvline(x=buy_date, color='red', linestyle='--', alpha=0.7, linewidth=1)
        else:
            axes[3, 3].text(0.5, 0.5, 'HY Spread\n데이터 없음', 
                           ha='center', va='center', transform=axes[3, 3].transAxes)
            axes[3, 3].set_title('HY Spread')
        
        # 레이아웃 조정
        plt.tight_layout()
        
        # 결과 저장
        output_path = output_dir / f'{self.symbol}_backtest_results.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"시각화 결과 저장: {output_path}")
        
        plt.close()
    
    def calculate_mdd(self, equity_curve):
        """
        Maximum Drawdown (MDD) 계산 (개선된 버전)
        
        Args:
            equity_curve (pd.Series): 일별 포트폴리오 가치 변화
            
        Returns:
            tuple: (MDD, MDD 시작일, MDD 종료일)
        """
        if equity_curve.empty or len(equity_curve) < 2:
            return 0, None, None
            
        # 누적 최고점 계산
        running_max = equity_curve.expanding().max()
        
        # Drawdown 계산 (백분율)
        drawdown = (equity_curve - running_max) / running_max * 100
        
        # MDD 찾기 (가장 큰 손실)
        mdd = drawdown.min()
        mdd_end_idx = drawdown.idxmin()
        
        # MDD 시작일 찾기 (MDD 종료일 이전의 최고점)
        if mdd_end_idx is not None:
            # MDD 종료일 이전의 최고점 찾기
            before_mdd = equity_curve.loc[:mdd_end_idx]
            if len(before_mdd) > 0:
                mdd_start_idx = before_mdd.idxmax()
            else:
                mdd_start_idx = mdd_end_idx
        else:
            mdd_start_idx = None
            
        # 디버깅 정보 출력
        print(f"=== MDD 계산 디버깅 ===")
        print(f"포트폴리오 가치 범위: ${equity_curve.min():.2f} ~ ${equity_curve.max():.2f}")
        print(f"계산된 MDD: {mdd:.2f}%")
        print(f"MDD 시작일: {mdd_start_idx}")
        print(f"MDD 종료일: {mdd_end_idx}")
        
        if mdd_start_idx and mdd_end_idx:
            start_value = equity_curve.loc[mdd_start_idx]
            end_value = equity_curve.loc[mdd_end_idx]
            print(f"MDD 시작 시 가치: ${start_value:.2f}")
            print(f"MDD 종료 시 가치: ${end_value:.2f}")
            print(f"실제 손실: ${end_value - start_value:.2f}")
            print(f"실제 손실률: {((end_value - start_value) / start_value * 100):.2f}%")
        print("========================\n")
        
        return mdd, mdd_start_idx, mdd_end_idx
    
    def calculate_cagr(self, initial_value, final_value, years):
        """
        Compound Annual Growth Rate (CAGR) 계산
        
        Args:
            initial_value (float): 초기 가치
            final_value (float): 최종 가치
            years (float): 투자 기간 (년)
            
        Returns:
            float: CAGR (%)
        """
        if years <= 0 or initial_value <= 0:
            return 0
            
        if final_value <= 0:
            return -100
            
        cagr = (pow(final_value / initial_value, 1 / years) - 1) * 100
        return cagr
    
    def calculate_equity_curve(self, results):
        """
        일별 포트폴리오 가치 변화 계산 (완전히 새로 작성된 버전)
        
        Args:
            results (list): 백테스트 결과 리스트
            
        Returns:
            pd.Series: 일별 포트폴리오 가치
        """
        if not results:
            return pd.Series()
            
        # 모든 매수 신호를 시간순으로 정렬
        positions = []
        for r in results:
            buy_date = r['매수 신호 발생일']
            # buy_date가 문자열인 경우 datetime으로 변환
            if isinstance(buy_date, str):
                buy_date = pd.to_datetime(buy_date).date()
            elif isinstance(buy_date, pd.Timestamp):
                buy_date = buy_date.date()
                
            buy_price = r['매수 가격']
            buy_amount = r['매수 금액']
            shares = buy_amount / buy_price
            
            # 각 기간별 수익률을 이용해 포트폴리오 가치 변화 추적
            returns_1m = r.get('1개월 수익률(%)', 0)
            returns_3m = r.get('3개월 수익률(%)', 0)
            returns_6m = r.get('6개월 수익률(%)', 0)
            returns_now = r.get('현재까지 수익률(%)', 0)
            
            # 수익률이 문자열인 경우 처리
            if isinstance(returns_1m, str) and returns_1m != 'N/A':
                returns_1m = float(returns_1m)
            if isinstance(returns_3m, str) and returns_3m != 'N/A':
                returns_3m = float(returns_3m)
            if isinstance(returns_6m, str) and returns_6m != 'N/A':
                returns_6m = float(returns_6m)
            if isinstance(returns_now, str) and returns_now != 'N/A':
                returns_now = float(returns_now)
                
            positions.append({
                'buy_date': buy_date,
                'shares': shares,
                'buy_price': buy_price,
                'buy_amount': buy_amount,
                'returns_1m': returns_1m,
                'returns_3m': returns_3m,
                'returns_6m': returns_6m,
                'returns_now': returns_now
            })
        
        if not positions:
            return pd.Series()
        
        # 일별 포트폴리오 가치 계산
        dates = []
        values = []
        
        # 백테스트 기간 동안의 일별 가치 계산
        start_date = min(pos['buy_date'] for pos in positions)
        end_date = pd.Timestamp.now().date()
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        for date in date_range:
            date_obj = date.date()
            total_value = 0
            
            for position in positions:
                if date_obj >= position['buy_date']:
                    # 매수일로부터 경과일수 계산
                    days_since_buy = (date_obj - position['buy_date']).days
                    
                    # 기간별 수익률 적용 (더 정확한 손실 반영)
                    if days_since_buy <= 30:
                        # 1개월 이내 - 1개월 수익률 적용
                        if isinstance(position['returns_1m'], (int, float)) and position['returns_1m'] != 'N/A':
                            current_price = position['buy_price'] * (1 + position['returns_1m'] / 100)
                        else:
                            current_price = position['buy_price']
                    elif days_since_buy <= 90:
                        # 3개월 이내 - 3개월 수익률 적용
                        if isinstance(position['returns_3m'], (int, float)) and position['returns_3m'] != 'N/A':
                            current_price = position['buy_price'] * (1 + position['returns_3m'] / 100)
                        else:
                            current_price = position['buy_price']
                    elif days_since_buy <= 180:
                        # 6개월 이내 - 6개월 수익률 적용
                        if isinstance(position['returns_6m'], (int, float)) and position['returns_6m'] != 'N/A':
                            current_price = position['buy_price'] * (1 + position['returns_6m'] / 100)
                        else:
                            current_price = position['buy_price']
                    else:
                        # 6개월 이후 - 현재까지 수익률 적용
                        if isinstance(position['returns_now'], (int, float)) and position['returns_now'] != 'N/A':
                            current_price = position['buy_price'] * (1 + position['returns_now'] / 100)
                        else:
                            current_price = position['buy_price']
                    
                    total_value += position['shares'] * current_price
            
            dates.append(date_obj)
            values.append(total_value)
        
        # 디버깅: 포트폴리오 가치 변화 확인
        print(f"=== 포트폴리오 가치 변화 디버깅 ===")
        print(f"총 포지션 수: {len(positions)}")
        for i, pos in enumerate(positions):
            print(f"포지션 {i+1}: {pos['buy_date']} - 주식수: {pos['shares']:.4f} - 매수가: ${pos['buy_price']:.2f}")
            print(f"  수익률: 1M={pos['returns_1m']}%, 3M={pos['returns_3m']}%, 6M={pos['returns_6m']}%, 현재={pos['returns_now']}%")
        
        if values:
            print(f"초기 포트폴리오 가치: ${values[0]:.2f}")
            print(f"최종 포트폴리오 가치: ${values[-1]:.2f}")
            print(f"최소 포트폴리오 가치: ${min(values):.2f}")
            print(f"최대 포트폴리오 가치: ${max(values):.2f}")
            print(f"포트폴리오 가치 변화: {((values[-1] - values[0]) / values[0] * 100):.2f}%")
            
            # MDD 계산을 위한 추가 정보
            running_max = pd.Series(values).expanding().max()
            drawdown = (pd.Series(values) - running_max) / running_max * 100
            mdd = drawdown.min()
            print(f"계산된 MDD: {mdd:.2f}%")
        print("==========================================\n")
        
        return pd.Series(values, index=dates)

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
        self.create_visualization(results, summary)
        
        # 7. 결과 출력
        print("\n=== 백테스트 완료 ===")
        print(f"총 신호 수: {summary['총 신호 수']}개")
        print(f"평균 6개월 수익률: {summary['평균 6개월 수익률(%)']}%")
        print(f"6개월 승률: {summary['승률 6개월(%)']}%")
        
        return results, summary

    def get_naiim_value(self, date):
        """특정 날짜의 NAIIM 값 반환 (S&P 500 모멘텀 기반)"""
        try:
            # S&P 500 데이터가 있는지 확인
            if hasattr(self, 'sp500_data') and self.sp500_data is not None:
                # 해당 날짜 또는 가장 가까운 이전 날짜의 S&P 500 값
                if date in self.sp500_data.index:
                    current_price = self.sp500_data.loc[date, 'Close']
                else:
                    # 가장 가까운 이전 날짜 찾기
                    prev_dates = self.sp500_data.index[self.sp500_data.index <= date]
                    if len(prev_dates) > 0:
                        current_price = self.sp500_data.loc[prev_dates[-1], 'Close']
                    else:
                        return 50  # 기본값
                
                # 20일 전 가격과 비교하여 모멘텀 계산
                if date in self.sp500_data.index:
                    start_date = date - timedelta(days=20)
                    if start_date in self.sp500_data.index:
                        start_price = self.sp500_data.loc[start_date, 'Close']
                        momentum = ((current_price - start_price) / start_price) * 100
                        # NAIIM은 0-100 범위로 정규화
                        naiim = max(0, min(100, 50 + momentum * 2))
                        return round(naiim, 2)
                
                return 50  # 기본값
            else:
                return 50  # 기본값
        except:
            return 50  # 기본값
    
    def get_hy_spread_value(self, date):
        """특정 날짜의 High Yield Spread 값 반환 (HYG/TLT 기반)"""
        try:
            # HYG와 TLT 데이터가 있는지 확인
            if hasattr(self, 'hyg_data') and hasattr(self, 'tlt_data') and self.hyg_data is not None and self.tlt_data is not None:
                # 해당 날짜 또는 가장 가까운 이전 날짜의 값들
                if date in self.hyg_data.index and date in self.tlt_data.index:
                    hyg_price = self.hyg_data.loc[date, 'Close']
                    tlt_price = self.tlt_data.loc[date, 'Close']
                else:
                    # 가장 가까운 이전 날짜 찾기
                    prev_dates_hyg = self.hyg_data.index[self.hyg_data.index <= date]
                    prev_dates_tlt = self.tlt_data.index[self.tlt_data.index <= date]
                    
                    if len(prev_dates_hyg) > 0 and len(prev_dates_tlt) > 0:
                        hyg_price = self.hyg_data.loc[prev_dates_hyg[-1], 'Close']
                        tlt_price = self.tlt_data.loc[prev_dates_tlt[-1], 'Close']
                    else:
                        return 3.0  # 기본값
                
                # High Yield Spread 계산 (HYG 대비 TLT 수익률 차이)
                # 일반적으로 3-8% 범위
                spread = ((hyg_price / tlt_price - 1) * 100) + 3.0
                return round(max(0, min(10, spread)), 2)
            else:
                return 3.0  # 기본값
        except:
            return 3.0  # 기본값
    


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='HMA Mantra 백테스트 프로그램')
    parser.add_argument('symbol', help='종목 심볼 (예: BAC, AAPL)')
    parser.add_argument('period', nargs='?', default='24mo', help='분석 기간 (기본값: 24mo)')
    parser.add_argument('initial_capital', nargs='?', type=float, default=1000, help='기본 매수 금액 (기본값: 1000)')
    parser.add_argument('--vix-low', type=float, default=0, help='VIX 하한값 (기본값: 0)')
    parser.add_argument('--vix-high', type=float, default=999, help='VIX 상한값 (기본값: 999)')
    parser.add_argument('--vix-bands', help='VIX 대역별 매수비용 (예: low:1000,mid:800,high:500)')
    parser.add_argument('--result-dir', help='결과 저장 디렉토리')
    
    args = parser.parse_args()
    
    # VIX 대역 파싱
    vix_bands = {}
    if args.vix_bands:
        backtest_temp = HMAMantraBacktest(args.symbol)
        vix_bands = backtest_temp.parse_vix_bands(args.vix_bands)
        print(f"VIX 대역 설정: {vix_bands}")
    
    try:
        backtest = HMAMantraBacktest(
            symbol=args.symbol,
            period=args.period,
            initial_capital=args.initial_capital,
            vix_low=args.vix_low,
            vix_high=args.vix_high,
            vix_bands=vix_bands,
            result_dir=args.result_dir
        )
        results, summary = backtest.run()
        
    except Exception as e:
        print(f"백테스트 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 