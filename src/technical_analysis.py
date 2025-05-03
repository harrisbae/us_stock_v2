import pandas as pd
import numpy as np
from datetime import datetime

class TechnicalAnalysis:
    def __init__(self, data=None):
        """
        기술적 분석 클래스 초기화
        
        Args:
            data (pd.DataFrame): OHLCV 데이터 (Open, High, Low, Close, Volume)
        """
        self.data = data
        
    def set_data(self, data):
        """데이터 설정"""
        self.data = data
        
    def calculate_all_indicators(self):
        """모든 기술적 지표 계산"""
        if self.data is None:
            print("ERROR: 데이터가 없습니다.")
            return None
            
        # 이동평균선
        self.add_moving_averages()
        
        # 볼린저 밴드
        self.add_bollinger_bands()
        
        # RSI
        self.add_rsi()
        
        # MACD
        self.add_macd()
        
        # OBV
        self.add_obv()
        
        # MFI
        self.add_mfi()
        
        return self.data
        
    def add_moving_averages(self, periods=[5, 20, 60, 120, 200]):
        """이동평균선 추가"""
        for period in periods:
            self.data[f'MA{period}'] = self.data['Close'].rolling(window=period).mean()
            
        # 거래량 이동평균
        self.data['Volume_MA20'] = self.data['Volume'].rolling(window=20).mean()
        
        return self.data
        
    def add_bollinger_bands(self, period=20, std_dev=2):
        """볼린저 밴드 추가"""
        # 중간 밴드 (20일 이동평균)
        middle_band = self.data['Close'].rolling(window=period).mean()
        
        # 표준편차
        rolling_std = self.data['Close'].rolling(window=period).std()
        
        # 상단 밴드
        self.data['BB_Upper'] = middle_band + (rolling_std * std_dev)
        
        # 중간 밴드
        self.data['BB_Middle'] = middle_band
        
        # 하단 밴드
        self.data['BB_Lower'] = middle_band - (rolling_std * std_dev)
        
        # %B 지표
        self.data['BB_PB'] = (self.data['Close'] - self.data['BB_Lower']) / (self.data['BB_Upper'] - self.data['BB_Lower'])
        
        return self.data
        
    def add_rsi(self, period=14):
        """RSI(Relative Strength Index) 추가"""
        delta = self.data['Close'].diff()
        
        # 상승과 하락 분리
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 평균 계산
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # RS 계산
        rs = avg_gain / avg_loss
        
        # RSI 계산
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        return self.data
        
    def add_macd(self, fast=12, slow=26, signal=9):
        """MACD(Moving Average Convergence Divergence) 추가"""
        # 빠른 지수이동평균
        ema_fast = self.data['Close'].ewm(span=fast, min_periods=fast).mean()
        
        # 느린 지수이동평균
        ema_slow = self.data['Close'].ewm(span=slow, min_periods=slow).mean()
        
        # MACD 라인
        self.data['MACD'] = ema_fast - ema_slow
        
        # 시그널 라인
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=signal, min_periods=signal).mean()
        
        # 히스토그램
        self.data['MACD_Hist'] = self.data['MACD'] - self.data['MACD_Signal']
        
        return self.data
        
    def add_obv(self):
        """OBV(On-Balance Volume) 추가"""
        obv = np.zeros(len(self.data))
        
        for i in range(1, len(self.data)):
            if self.data['Close'].iloc[i] > self.data['Close'].iloc[i-1]:
                obv[i] = obv[i-1] + self.data['Volume'].iloc[i]
            elif self.data['Close'].iloc[i] < self.data['Close'].iloc[i-1]:
                obv[i] = obv[i-1] - self.data['Volume'].iloc[i]
            else:
                obv[i] = obv[i-1]
                
        self.data['OBV'] = obv
        
        return self.data
        
    def add_mfi(self, period=14):
        """MFI(Money Flow Index) 추가"""
        # 임시 데이터프레임 생성
        df = self.data.copy()
        
        # 전형적 가격 (Typical Price)
        df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
        
        # 자금 흐름 (Money Flow)
        df['MF'] = df['TP'] * df['Volume']
        
        # 자금 흐름 방향 (긍정적/부정적)
        df['Direction'] = 0  # 기본값
        df.loc[df['TP'] > df['TP'].shift(1), 'Direction'] = 1  # 상승
        df.loc[df['TP'] < df['TP'].shift(1), 'Direction'] = -1  # 하락
        
        # 긍정적 자금 흐름과 부정적 자금 흐름 분리
        df['Positive_MF'] = 0.0
        df['Negative_MF'] = 0.0
        
        df.loc[df['Direction'] > 0, 'Positive_MF'] = df['MF']
        df.loc[df['Direction'] < 0, 'Negative_MF'] = df['MF']
        
        # n기간 동안의 긍정적/부정적 자금 흐름 합계
        df['Positive_MF_Sum'] = df['Positive_MF'].rolling(window=period).sum()
        df['Negative_MF_Sum'] = df['Negative_MF'].rolling(window=period).sum()
        
        # 자금 비율 (Money Ratio)
        # 0으로 나누는 문제 방지
        df['Money_Ratio'] = df.apply(
            lambda x: x['Positive_MF_Sum'] / x['Negative_MF_Sum'] 
            if x['Negative_MF_Sum'] > 0 else 
            (100 if x['Positive_MF_Sum'] > 0 else 50),  # 분모가 0이면 특별 처리
            axis=1
        )
        
        # MFI 계산
        df['MFI'] = 100 - (100 / (1 + df['Money_Ratio']))
        
        # MFI 값을 원본 데이터프레임에 복사
        self.data['MFI'] = df['MFI']
        
        return self.data
    
    def add_trend_analysis(self):
        """가격 추세 분석 추가"""
        # 단기(20일), 중기(60일), 장기(120일) 추세
        self.data['Short_Trend'] = np.where(self.data['Close'] > self.data['MA20'], 1, -1)
        self.data['Mid_Trend'] = np.where(self.data['Close'] > self.data['MA60'], 1, -1)
        self.data['Long_Trend'] = np.where(self.data['Close'] > self.data['MA120'], 1, -1)
        
        # 종합 추세 점수 (-3 ~ 3)
        self.data['Trend_Score'] = self.data['Short_Trend'] + self.data['Mid_Trend'] + self.data['Long_Trend']
        
        return self.data

    def find_buy_signals(self, window=10):
        """
        모든 기술적 지표를 종합적으로 고려하여 매수 신호 생성
        
        Args:
            window (int): 신호 탐색 기간 (최근 n일)
            
        Returns:
            pd.DataFrame: 매수 신호가 포함된 데이터프레임
        """
        if self.data is None:
            print("ERROR: 데이터가 없습니다.")
            return None
            
        # 매수 점수 컬럼 추가 (0~100)
        self.data['Buy_Score'] = 0
        
        # 1. 이동평균선 기반 점수 (최대 20점)
        # - MA5 > MA20 (황금 십자가 패턴)
        if 'MA5' in self.data.columns and 'MA20' in self.data.columns:
            self.data['MA_Cross'] = np.where(
                (self.data['MA5'] > self.data['MA20']) & 
                (self.data['MA5'].shift(1) <= self.data['MA20'].shift(1)), 
                1, 0
            )
            # 최근 골든크로스 발생 시 점수 부여
            self.data.loc[self.data['MA_Cross'] == 1, 'Buy_Score'] += 20
            
            # 단기 이동평균선이 중기선 위에 있는 경우 점수 추가
            self.data.loc[self.data['MA5'] > self.data['MA20'], 'Buy_Score'] += 5
            
            # 추세 신호 추가 (단기 상승 추세)
            self.data['MA_Trend_Signal'] = np.where(
                (self.data['MA5'] > self.data['MA5'].shift(3)) & 
                (self.data['MA20'] > self.data['MA20'].shift(5)), 
                1, 0
            )
            self.data.loc[self.data['MA_Trend_Signal'] == 1, 'Buy_Score'] += 5
            
        # 2. RSI 기반 점수 (최대 15점)
        if 'RSI' in self.data.columns:
            # RSI가 과매도 구간(30)에서 회복하는 경우
            self.data['RSI_Signal'] = np.where(
                (self.data['RSI'] > 30) & 
                (self.data['RSI'].shift(1) <= 30), 
                1, 0
            )
            self.data.loc[self.data['RSI_Signal'] == 1, 'Buy_Score'] += 15
            
            # RSI가 30~50 구간에 있는 경우
            self.data.loc[(self.data['RSI'] > 30) & (self.data['RSI'] < 50), 'Buy_Score'] += 5
            
            # RSI 상승 추세 신호
            self.data['RSI_Up_Signal'] = np.where(
                (self.data['RSI'] > self.data['RSI'].shift(3)) &
                (self.data['RSI'] < 70), 
                1, 0
            )
            self.data.loc[self.data['RSI_Up_Signal'] == 1, 'Buy_Score'] += 5
            
        # 3. MACD 기반 점수 (최대 15점)
        if 'MACD' in self.data.columns and 'MACD_Signal' in self.data.columns:
            # MACD가 시그널 라인을 상향돌파
            self.data['MACD_Cross'] = np.where(
                (self.data['MACD'] > self.data['MACD_Signal']) & 
                (self.data['MACD'].shift(1) <= self.data['MACD_Signal'].shift(1)), 
                1, 0
            )
            self.data.loc[self.data['MACD_Cross'] == 1, 'Buy_Score'] += 15
            
            # MACD가 0선 아래에서 상승 중인 경우
            self.data.loc[(self.data['MACD'] < 0) & (self.data['MACD'] > self.data['MACD'].shift(1)), 'Buy_Score'] += 5
            
        # 4. MFI 기반 점수 (최대 15점)
        if 'MFI' in self.data.columns:
            # MFI가 과매도 구간(20)에서 회복하는 경우
            self.data['MFI_Signal'] = np.where(
                (self.data['MFI'] > 20) & 
                (self.data['MFI'].shift(1) <= 20), 
                1, 0
            )
            self.data.loc[self.data['MFI_Signal'] == 1, 'Buy_Score'] += 15
            
            # MFI가 20~40 구간에 있는 경우
            self.data.loc[(self.data['MFI'] > 20) & (self.data['MFI'] < 40), 'Buy_Score'] += 5
            
        # 5. 볼린저 밴드 %B 기반 점수 (최대 15점)
        if 'BB_PB' in self.data.columns:
            # %B가 0.2 이하에서 반등하는 경우
            self.data['BB_Signal'] = np.where(
                (self.data['BB_PB'] > 0.2) & 
                (self.data['BB_PB'].shift(1) <= 0.2), 
                1, 0
            )
            self.data.loc[self.data['BB_Signal'] == 1, 'Buy_Score'] += 15
            
            # %B가 0.2~0.5 구간에 있는 경우
            self.data.loc[(self.data['BB_PB'] > 0.2) & (self.data['BB_PB'] < 0.5), 'Buy_Score'] += 5
            
        # 6. OBV 기반 점수 (최대 10점)
        if 'OBV' in self.data.columns:
            # OBV가 상승 추세인 경우
            self.data['OBV_Signal'] = np.where(
                self.data['OBV'] > self.data['OBV'].rolling(window=5).mean(), 
                1, 0
            )
            self.data.loc[self.data['OBV_Signal'] == 1, 'Buy_Score'] += 10
            
        # 7. 거래량 기반 점수 (최대 10점)
        if 'Volume' in self.data.columns and 'Volume_MA20' in self.data.columns:
            # 거래량이 20일 평균보다 높은 경우
            self.data['Volume_Signal'] = np.where(
                self.data['Volume'] > self.data['Volume_MA20'] * 1.5,
                1, 0
            )
            self.data.loc[self.data['Volume_Signal'] == 1, 'Buy_Score'] += 10
            
        # 8. 외국인 거래량 기반 점수 (최대 10점) - 추가된 부분
        if 'Foreign_Buy_Signal' in self.data.columns:
            # 외국인 순매수 신호가 있는 경우
            self.data.loc[self.data['Foreign_Buy_Signal'] == 1, 'Buy_Score'] += 5
            
            # 외국인 순매수 비율이 높은 경우 (전체 거래의 5% 이상)
            if 'Foreign_Buy_Ratio' in self.data.columns:
                self.data.loc[self.data['Foreign_Buy_Ratio'] > 0.05, 'Buy_Score'] += 5
            
        # 최근 window 기간의 데이터만 반환
        recent_data = self.data.iloc[-window:].copy()
        
        # 매수 신호 강도에 따라 분류
        recent_data['Buy_Signal'] = '약'  # 기본값
        recent_data.loc[recent_data['Buy_Score'] >= 30, 'Buy_Signal'] = '중'
        recent_data.loc[recent_data['Buy_Score'] >= 50, 'Buy_Signal'] = '강'
        recent_data.loc[recent_data['Buy_Score'] >= 70, 'Buy_Signal'] = '매우 강'
        
        return recent_data 