#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
매크로 경제 지표 분석 모듈
VIX, High Yield Spread, NAIIM 등을 분석하여 매수 신호를 개선합니다.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MacroAnalysis:
    def __init__(self):
        """매크로 분석 클래스 초기화"""
        self.vix_data = None
        self.hy_spread_data = None
        self.naiim_data = None
        self.stock_data = None
        
    def fetch_vix_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        VIX 데이터 수집
        
        Args:
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)
            
        Returns:
            VIX 데이터프레임
        """
        try:
            # VIX 데이터 수집 (^VIX)
            vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
            
            if vix.empty:
                return None
                
            # 일일 VIX 종가만 사용
            vix_data = vix[['Close']].copy()
            vix_data.columns = ['VIX']
            vix_data.index.name = 'Date'
            
            # VIX 이동평균 추가
            vix_data['VIX_MA20'] = vix_data['VIX'].rolling(window=20).mean()
            vix_data['VIX_MA50'] = vix_data['VIX'].rolling(window=50).mean()
            
            # VIX 상태 분류
            vix_data['VIX_Status'] = vix_data['VIX'].apply(self._classify_vix_status)
            
            self.vix_data = vix_data
            return vix_data
            
        except Exception as e:
            return None
    
    def fetch_high_yield_spread_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        High Yield Spread 데이터 수집
        
        Args:
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)
            
        Returns:
            HY Spread 데이터프레임
        """
        try:
            # 고수익 채권 ETF (HYG)와 국채 ETF (TLT) 데이터 수집
            hyg = yf.download('HYG', start=start_date, end=end_date, progress=False)
            tlt = yf.download('TLT', start=start_date, end=end_date, progress=False)
            
            if hyg.empty or tlt.empty:
                return None
            
            # 수익률 계산
            hyg_returns = hyg['Close'].pct_change()
            tlt_returns = tlt['Close'].pct_change()
            
            # 스프레드 계산 (고수익 채권 수익률 - 국채 수익률)
            hy_spread = hyg_returns - tlt_returns
            
            # 데이터프레임 생성
            spread_data = pd.DataFrame({
                'HYG_Return': hyg_returns,
                'TLT_Return': tlt_returns,
                'HY_Spread': hy_spread
            }, index=hyg.index)
            
            # NaN 값 제거
            spread_data = spread_data.dropna()
            
            # 이동평균 추가
            spread_data['HY_Spread_MA20'] = spread_data['HY_Spread'].rolling(window=20, min_periods=1).mean()
            spread_data['HY_Spread_MA50'] = spread_data['HY_Spread'].rolling(window=50, min_periods=1).mean()
            
            # 스프레드 상태 분류
            spread_data['Spread_Status'] = spread_data['HY_Spread'].apply(self._classify_spread_status)
            
            self.hy_spread_data = spread_data
            return spread_data
            
        except Exception as e:
            return None
    
    def fetch_naiim_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        NAIIM 데이터 수집 (투자 매니저 전망 지수)
        
        Args:
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)
            
        Returns:
            NAIIM 데이터프레임
        """
        try:
            # NAIIM 데이터는 월간 데이터이므로 주기적으로 업데이트 필요
            # 임시로 시장 심리 지표로 사용할 수 있는 다른 지표 사용
            
            # CNN Fear & Greed Index 대체 사용
            # 또는 시장의 전반적인 모멘텀 지표 사용
            
            # 여기서는 간단한 시장 모멘텀 지표 생성
            if self.stock_data is not None:
                # S&P 500의 모멘텀 지표 생성
                sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
                
                if not sp500.empty:
                    # 모멘텀 지표 계산
                    sp500['Returns'] = sp500['Close'].pct_change()
                    sp500['Momentum_20'] = sp500['Returns'].rolling(window=20).sum()
                    sp500['Momentum_50'] = sp500['Returns'].rolling(window=50).sum()
                    
                    # 시장 심리 지수 (0-100)
                    sp500['Market_Sentiment'] = 50 + (sp500['Momentum_20'] * 100)
                    sp500['Market_Sentiment'] = sp500['Market_Sentiment'].clip(0, 100)
                    
                    # NAIIM과 유사한 지표로 사용
                    naiim_data = pd.DataFrame({
                        'NAIIM': sp500['Market_Sentiment'],
                        'Momentum_20': sp500['Momentum_20'],
                        'Momentum_50': sp500['Momentum_50']
                    }, index=sp500.index)
                    
                    # NAIIM 상태 분류
                    naiim_data['NAIIM_Status'] = naiim_data['NAIIM'].apply(self._classify_naiim_status)
                    
                    self.naiim_data = naiim_data
                    return naiim_data
            
            return None
            
        except Exception as e:
            return None
    
    def _classify_vix_status(self, vix_value: float) -> str:
        """VIX 값 상태 분류"""
        if pd.isna(vix_value):
            return 'Unknown'
        elif vix_value < 15:
            return 'Low_Fear'      # 낮은 공포 (과매수 가능성)
        elif vix_value < 25:
            return 'Normal'         # 정상
        elif vix_value < 35:
            return 'High_Fear'      # 높은 공포 (매수 기회)
        else:
            return 'Extreme_Fear'   # 극도의 공포 (강력한 매수 신호)
    
    def _classify_spread_status(self, spread_value: float) -> str:
        """High Yield Spread 상태 분류"""
        if pd.isna(spread_value):
            return 'Unknown'
        elif spread_value < -0.01:
            return 'Low_Risk'       # 낮은 리스크 (과매수 가능성)
        elif spread_value < 0.01:
            return 'Normal'          # 정상
        elif spread_value < 0.03:
            return 'High_Risk'       # 높은 리스크 (매수 기회)
        else:
            return 'Extreme_Risk'    # 극도의 리스크 (강력한 매수 신호)
    
    def _classify_naiim_status(self, naiim_value: float) -> str:
        """NAIIM 상태 분류"""
        if pd.isna(naiim_value):
            return 'Unknown'
        elif naiim_value > 70:
            return 'Bullish'        # 강세 전망 (과매수 가능성)
        elif naiim_value > 50:
            return 'Neutral'         # 중립
        elif naiim_value > 30:
            return 'Bearish'         # 약세 전망 (매수 기회)
        else:
            return 'Extreme_Bearish' # 극도의 약세 (강력한 매수 신호)
    
    def calculate_macro_score(self, date: str) -> Dict:
        """
        특정 날짜의 매크로 종합 점수 계산
        
        Args:
            date: 날짜 (YYYY-MM-DD)
            
        Returns:
            매크로 점수 딕셔너리
        """
        try:
            target_date = pd.to_datetime(date)
            
            # 각 지표의 점수 계산
            vix_score = self._calculate_vix_score(target_date)
            spread_score = self._calculate_spread_score(target_date)
            naiim_score = self._calculate_naiim_score(target_date)
            
            # 종합 점수 계산 (가중 평균)
            macro_score = (vix_score * 0.4 + spread_score * 0.3 + naiim_score * 0.3)
            
            return {
                'date': date,
                'vix_score': vix_score,
                'spread_score': spread_score,
                'naiim_score': naiim_score,
                'macro_score': macro_score,
                'recommendation': self._get_macro_recommendation(macro_score)
            }
            
        except Exception as e:
            return None
    
    def _calculate_vix_score(self, date: pd.Timestamp) -> float:
        """VIX 점수 계산 (0-100, 높을수록 매수 유리)"""
        if self.vix_data is None or date not in self.vix_data.index:
            return 50  # 기본값
        
        vix_value = self.vix_data.loc[date, 'VIX']
        vix_ma20 = self.vix_data.loc[date, 'VIX_MA20']
        
        # VIX가 이동평균보다 높을수록 높은 점수
        if pd.isna(vix_value) or pd.isna(vix_ma20):
            return 50
        
        # VIX 상대적 위치에 따른 점수
        if vix_value > vix_ma20 * 1.5:
            return 90  # 극도의 공포
        elif vix_value > vix_ma20 * 1.2:
            return 80  # 높은 공포
        elif vix_value > vix_ma20:
            return 70  # 공포
        elif vix_value > vix_ma20 * 0.8:
            return 50  # 중립
        else:
            return 30  # 낮은 공포
    
    def _calculate_spread_score(self, date: pd.Timestamp) -> float:
        """High Yield Spread 점수 계산 (0-100, 높을수록 매수 유리)"""
        if self.hy_spread_data is None or date not in self.hy_spread_data.index:
            return 50  # 기본값
        
        spread_value = self.hy_spread_data.loc[date, 'HY_Spread']
        spread_ma20 = self.hy_spread_data.loc[date, 'HY_Spread_MA20']
        
        if pd.isna(spread_value) or pd.isna(spread_ma20):
            return 50
        
        # 스프레드가 좁아질수록 높은 점수 (리스크 선호도 증가)
        if spread_value < spread_ma20 * 0.5:
            return 90  # 극도로 좁은 스프레드
        elif spread_value < spread_ma20 * 0.8:
            return 80  # 좁은 스프레드
        elif spread_value < spread_ma20:
            return 70  # 정상보다 좁음
        elif spread_value < spread_ma20 * 1.2:
            return 50  # 정상
        else:
            return 30  # 넓은 스프레드
    
    def _calculate_naiim_score(self, date: pd.Timestamp) -> float:
        """NAIIM 점수 계산 (0-100, 낮을수록 매수 유리)"""
        if self.naiim_data is None or date not in self.naiim_data.index:
            return 50  # 기본값
        
        naiim_value = self.naiim_data.loc[date, 'NAIIM']
        
        if pd.isna(naiim_value):
            return 50
        
        # NAIIM이 낮을수록 높은 점수 (반대 심리)
        if naiim_value < 20:
            return 90  # 극도의 약세
        elif naiim_value < 35:
            return 80  # 약세
        elif naiim_value < 50:
            return 70  # 약간 약세
        elif naiim_value < 65:
            return 50  # 중립
        else:
            return 30  # 강세
    
    def _get_macro_recommendation(self, macro_score: float) -> str:
        """매크로 점수에 따른 매수 권장사항"""
        if macro_score >= 80:
            return "강력 매수 추천"
        elif macro_score >= 70:
            return "매수 추천"
        elif macro_score >= 60:
            return "약한 매수"
        elif macro_score >= 40:
            return "관망"
        else:
            return "매수 비추천"
    
    def analyze_macro_correlation(self, stock_signals: pd.DataFrame) -> pd.DataFrame:
        """
        주식 매수 신호와 매크로 지표의 상관관계 분석
        
        Args:
            stock_signals: 주식 매수 신호 데이터프레임
            
        Returns:
            상관관계 분석 결과
        """
        try:
            if self.vix_data is None or self.hy_spread_data is None or self.naiim_data is None:
                return None
            
            # 매수 신호가 발생한 날짜들 추출
            buy_dates = stock_signals[stock_signals['Buy_Signal'] == True].index
            
            if len(buy_dates) == 0:
                return None
            
            # 매수 신호 발생일의 매크로 지표 값들 수집
            macro_analysis = []
            
            for date in buy_dates:
                if date in self.vix_data.index:
                    vix_value = self.vix_data.loc[date, 'VIX']
                    vix_status = self.vix_data.loc[date, 'VIX_Status']
                else:
                    vix_value = np.nan
                    vix_status = 'Unknown'
                
                if date in self.hy_spread_data.index:
                    spread_value = self.hy_spread_data.loc[date, 'HY_Spread']
                    spread_status = self.hy_spread_data.loc[date, 'Spread_Status']
                else:
                    spread_value = np.nan
                    spread_status = 'Unknown'
                
                if date in self.naiim_data.index:
                    naiim_value = self.naiim_data.loc[date, 'NAIIM']
                    naiim_status = self.naiim_data.loc[date, 'NAIIM_Status']
                else:
                    naiim_value = np.nan
                    naiim_status = 'Unknown'
                
                # 매크로 점수 계산
                macro_score = self.calculate_macro_score(date.strftime('%Y-%m-%d'))
                
                macro_analysis.append({
                    'Date': date,
                    'VIX': vix_value,
                    'VIX_Status': vix_status,
                    'HY_Spread': spread_value,
                    'Spread_Status': spread_status,
                    'NAIIM': naiim_value,
                    'NAIIM_Status': naiim_status,
                    'Macro_Score': macro_score['macro_score'] if macro_score else np.nan,
                    'Recommendation': macro_score['recommendation'] if macro_score else 'Unknown'
                })
            
            # 결과를 데이터프레임으로 변환
            result_df = pd.DataFrame(macro_analysis)
            
            # 상관관계 분석
            correlation_analysis = self._calculate_correlations(result_df)
            
            return result_df, correlation_analysis
            
        except Exception as e:
            return None
    
    def _calculate_correlations(self, macro_df: pd.DataFrame) -> Dict:
        """상관관계 계산"""
        try:
            # 수치형 컬럼만 선택
            numeric_cols = ['VIX', 'HY_Spread', 'NAIIM', 'Macro_Score']
            numeric_df = macro_df[numeric_cols].dropna()
            
            if len(numeric_df) < 2:
                return {"error": "상관관계 계산을 위한 충분한 데이터가 없습니다."}
            
            # 상관계수 계산
            correlations = numeric_df.corr()
            
            # 매크로 점수와의 상관관계
            macro_correlations = correlations['Macro_Score'].drop('Macro_Score')
            
            return {
                'correlation_matrix': correlations,
                'macro_score_correlations': macro_correlations,
                'sample_size': len(numeric_df)
            }
            
        except Exception as e:
            return {"error": f"상관관계 계산 중 오류: {e}"}
    
    def plot_macro_analysis(self, macro_df: pd.DataFrame, save_path: str = None):
        """매크로 분석 결과 시각화"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('매크로 지표 분석 결과', fontsize=16)
            
            # 1. VIX 분포
            if 'VIX' in macro_df.columns:
                axes[0, 0].hist(macro_df['VIX'].dropna(), bins=20, alpha=0.7, color='red')
                axes[0, 0].set_title('매수 신호 발생일 VIX 분포')
                axes[0, 0].set_xlabel('VIX')
                axes[0, 0].set_ylabel('빈도')
                axes[0, 0].grid(True)
            
            # 2. High Yield Spread 분포
            if 'HY_Spread' in macro_df.columns:
                axes[0, 1].hist(macro_df['HY_Spread'].dropna(), bins=20, alpha=0.7, color='blue')
                axes[0, 1].set_title('매수 신호 발생일 HY Spread 분포')
                axes[0, 1].set_xlabel('HY Spread')
                axes[0, 1].set_ylabel('빈도')
                axes[0, 1].grid(True)
            
            # 3. NAIIM 분포
            if 'NAIIM' in macro_df.columns:
                axes[1, 0].hist(macro_df['NAIIM'].dropna(), bins=20, alpha=0.7, color='green')
                axes[1, 0].set_title('매수 신호 발생일 NAIIM 분포')
                axes[1, 0].set_xlabel('NAIIM')
                axes[1, 0].set_ylabel('빈도')
                axes[1, 0].grid(True)
            
            # 4. 매크로 점수 분포
            if 'Macro_Score' in macro_df.columns:
                axes[1, 1].hist(macro_df['Macro_Score'].dropna(), bins=20, alpha=0.7, color='purple')
                axes[1, 1].set_title('매크로 종합 점수 분포')
                axes[1, 1].set_xlabel('Macro Score')
                axes[1, 1].set_ylabel('빈도')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
            return fig
            
        except Exception as e:
            print(f"시각화 중 오류: {e}")
            return None
