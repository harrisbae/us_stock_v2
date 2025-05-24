import unittest
import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# 상위 디렉토리를 파이썬 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_fetcher import DataFetcher

class TestDataFetcher(unittest.TestCase):
    """데이터 수집 테스트 클래스"""
    
    def setUp(self):
        """테스트 준비"""
        self.fetcher = DataFetcher()
        self.ticker = "AAPL"  # 애플을 테스트 종목으로 사용
        
    def test_fetch_data(self):
        """데이터 수집 테스트"""
        # 데이터 가져오기 (5일간의 일봉)
        data = self.fetcher.fetch_data(self.ticker, period="5d", interval="1d")
        
        # 데이터가 존재하는지 확인
        self.assertIsNotNone(data)
        self.assertFalse(data.empty)
        
        # 필요한 컬럼이 있는지 확인
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_columns:
            self.assertIn(col, data.columns)
            
        # 데이터 크기 확인 (5일 데이터이므로 5개 이하의 행이 있어야 함)
        self.assertLessEqual(len(data), 5)
        
    def test_save_data(self):
        """데이터 저장 테스트"""
        # 먼저 데이터 가져오기
        data = self.fetcher.fetch_data(self.ticker, period="5d", interval="1d")
        
        # 임시 출력 디렉토리
        test_output_dir = os.path.join("test", "test_output")
        os.makedirs(test_output_dir, exist_ok=True)
        
        # 파일 저장
        output_file = self.fetcher.save_data(test_output_dir)
        
        # 파일이 존재하는지 확인
        self.assertIsNotNone(output_file)
        self.assertTrue(os.path.exists(output_file))
        
        # 저장된 데이터 읽어서 원본과 비교
        saved_data = pd.read_csv(output_file, index_col=0, parse_dates=True)
        self.assertEqual(len(data), len(saved_data))
        
        # 테스트 후 파일 삭제
        try:
            os.remove(output_file)
        except:
            pass

if __name__ == '__main__':
    unittest.main() 