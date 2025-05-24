import numpy as np

def calculate_hma(data, period=20):
    """
    Hull Moving Average (HMA) 계산
    """
    # WMA 계산 함수
    def wma(data, period):
        weights = np.arange(1, period + 1)
        return data.rolling(period).apply(lambda x: np.sum(weights * x) / weights.sum(), raw=True)
    
    # HMA 계산
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    wma1 = wma(data, half_period)
    wma2 = wma(data, period)
    hma = wma(2 * wma1 - wma2, sqrt_period)
    
    return hma

def calculate_mantra_bands(data, period=20, multiplier=2.0):
    """
    만트라 밴드 계산
    """
    # 기본 이동평균
    ma = data.rolling(window=period).mean()
    
    # 표준편차
    std = data.rolling(window=period).std()
    
    # 상단/하단 밴드
    upper_band = ma + (multiplier * std)
    lower_band = ma - (multiplier * std)
    
    return upper_band, ma, lower_band 