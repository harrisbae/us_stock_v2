import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 상위 디렉토리를 파이썬 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.technical_analysis import TechnicalAnalysis

class TestTechnicalAnalysis(unittest.TestCase):
    """기술적 분석 테스트 클래스"""
    
    def setUp(self):
        """테스트 준비"""
        # 테스트용 데이터 생성
        # 100일 간의 가상 주가 데이터 생성
        dates = [datetime.now() - timedelta(days=i) for i in range(100)]
        dates.reverse()  # 날짜 오름차순으로 정렬
        
        np.random.seed(42)  # 결과 재현성을 위한 시드 설정
        
        # 시작 가격
        start_price = 100
        
        # 주가 데이터 생성 (랜덤 워크 모델)
        prices = [start_price]
        for i in range(1, 100):
            # 전날 가격에 랜덤한 변동 추가 (-3% ~ +3%)
            change = np.random.normal(0.001, 0.02)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # 데이터프레임 생성
        close_prices = np.array(prices)
        open_prices = close_prices * (1 + np.random.normal(0, 0.01, size=100))
        high_prices = np.maximum(close_prices, open_prices) * (1 + np.random.uniform(0, 0.01, size=100))
        low_prices = np.minimum(close_prices, open_prices) * (1 - np.random.uniform(0, 0.01, size=100))
        volumes = np.random.randint(1000, 10000, size=100)
        
        self.data = pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volumes
        }, index=pd.DatetimeIndex(dates))
        
        # 기술적 분석 객체 생성
        self.analyzer = TechnicalAnalysis(self.data)
        
    def test_moving_averages(self):
        """이동평균선 계산 테스트"""
        # 이동평균선 계산
        self.analyzer.add_moving_averages()
        
        # 결과 확인
        self.assertIn('MA5', self.data.columns)
        self.assertIn('MA20', self.data.columns)
        self.assertIn('MA60', self.data.columns)
        self.assertIn('MA120', self.data.columns)
        self.assertIn('MA200', self.data.columns)
        
        # MA5 계산 값 확인 (20번째 값)
        expected_ma5 = self.data['Close'].iloc[15:20].mean()
        self.assertAlmostEqual(self.data['MA5'].iloc[19], expected_ma5)
        
    def test_bollinger_bands(self):
        """볼린저 밴드 계산 테스트"""
        # 볼린저 밴드 계산
        self.analyzer.add_bollinger_bands()
        
        # 결과 확인
        self.assertIn('BB_Upper', self.data.columns)
        self.assertIn('BB_Middle', self.data.columns)
        self.assertIn('BB_Lower', self.data.columns)
        self.assertIn('BB_PB', self.data.columns)
        
        # 볼린저 밴드 값 계산 확인 (30번째 값)
        idx = 30
        expected_middle = self.data['Close'].iloc[idx-20+1:idx+1].mean()
        expected_std = self.data['Close'].iloc[idx-20+1:idx+1].std()
        
        self.assertAlmostEqual(self.data['BB_Middle'].iloc[idx], expected_middle)
        self.assertAlmostEqual(self.data['BB_Upper'].iloc[idx], expected_middle + 2 * expected_std)
        self.assertAlmostEqual(self.data['BB_Lower'].iloc[idx], expected_middle - 2 * expected_std)
        
    def test_rsi(self):
        """RSI 계산 테스트"""
        # RSI 계산
        self.analyzer.add_rsi()
        
        # 결과 확인
        self.assertIn('RSI', self.data.columns)
        
        # RSI 범위 확인 (0-100 사이)
        valid_rsi = self.data['RSI'].dropna()
        self.assertTrue(all(0 <= rsi <= 100 for rsi in valid_rsi))
        
    def test_macd(self):
        """MACD 계산 테스트"""
        # MACD 계산
        self.analyzer.add_macd()
        
        # 결과 확인
        self.assertIn('MACD', self.data.columns)
        self.assertIn('MACD_Signal', self.data.columns)
        self.assertIn('MACD_Hist', self.data.columns)
        
        # MACD 히스토그램 계산 확인
        idx = 50
        self.assertAlmostEqual(
            self.data['MACD_Hist'].iloc[idx],
            self.data['MACD'].iloc[idx] - self.data['MACD_Signal'].iloc[idx]
        )
        
    def test_calculate_all_indicators(self):
        """모든 지표 계산 테스트"""
        # 모든 지표 계산
        self.analyzer.calculate_all_indicators()
        
        # 필수 컬럼 확인
        essential_columns = [
            'MA5', 'MA20', 'MA60', 'MA120', 'Volume_MA20',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_PB',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'OBV', 'MFI'
        ]
        
        for col in essential_columns:
            self.assertIn(col, self.data.columns)

if __name__ == '__main__':
    unittest.main() 