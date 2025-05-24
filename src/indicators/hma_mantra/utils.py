"""
Utility functions module
"""

import numpy as np
import pandas as pd
import matplotlib.font_manager as fm

def get_available_font():
    """사용 가능한 폰트 반환"""
    preferred_fonts = ['AppleGothic', 'Malgun Gothic', 'NanumGothic', 'Arial Unicode MS']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in preferred_fonts:
        if font in available_fonts:
            return font
    
    return 'DejaVu Sans'

def to_float(val):
    """안전한 float 변환"""
    if hasattr(val, 'item'):
        return val.item()
    elif isinstance(val, (np.generic, np.ndarray)) and getattr(val, 'size', 1) == 1:
        return float(val)
    else:
        return float(val)

def to_bool(val):
    """안전한 bool 변환"""
    if isinstance(val, pd.Series):
        return bool(val.iloc[0])
    return bool(val)

def to_series(val, index):
    """다양한 타입의 데이터를 pandas Series로 변환"""
    if isinstance(val, pd.Series):
        return val
    elif isinstance(val, pd.DataFrame):
        return val.iloc[:, 0]
    elif np.isscalar(val):
        return pd.Series(np.full(len(index), val), index=index)
    elif isinstance(val, (np.ndarray, list)):
        arr = np.asarray(val).ravel()
        if arr.size == len(index):
            return pd.Series(arr, index=index)
        elif arr.size == 1:
            return pd.Series(np.full(len(index), arr[0]), index=index)
        else:
            raise ValueError(f"Cannot broadcast array to Series of length {len(index)}")
    else:
        raise ValueError(f"Unsupported type for to_series: {type(val)}")

def get_gics_sector(ticker):
    """GICS 섹터 분류"""
    sectors = {
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
        'TQQQ': '레버리지 기술주 (Leveraged Technology)'
    }
    return sectors.get(ticker, '섹터 정보 없음')

def get_gics_subsector(ticker):
    """GICS 서브섹터 분류"""
    subsectors = {
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
        'TQQQ': '나스닥100 3X 레버리지 ETF'
    }
    return subsectors.get(ticker, '서브섹터 정보 없음')

def get_sector_outlook(ticker, vix, tnx, dxy):
    """섹터 전망"""
    sector = get_gics_sector(ticker).split()[0]
    outlook = ''
    
    if sector == '정보기술':
        if vix < 20 and tnx < 4.0:
            outlook = '긍정적 (금리 하향 안정화로 성장주 선호 가능)'
        else:
            outlook = '중립적 (금리 변동성 주시 필요)'
    elif sector == '커뮤니케이션서비스':
        if vix < 20:
            outlook = '긍정적 (광고 시장 회복 기대)'
        else:
            outlook = '중립적 (경기 민감도 높음)'
    elif sector == '임의소비재':
        if dxy < 104 and tnx < 4.0:
            outlook = '긍정적 (소비 심리 개선 기대)'
        else:
            outlook = '중립적 (금리와 환율 영향 주시)'
    elif sector == '금융':
        if tnx > 4.0:
            outlook = '긍정적 (순이자마진 개선)'
        else:
            outlook = '중립적 (금리 정책 방향성 주시)'
    elif sector == '헬스케어':
        outlook = '긍정적 (디펜시브 성격, 고령화 수혜)'
    else:
        outlook = '중립적 (개별 종목 분석 필요)'
    
    return outlook

def get_trading_strategy(hma, upper_mantra, lower_mantra, rsi14, macd, signal_line):
    """투자전략 도출"""
    hma_trend = '상승' if hma.iloc[-1] > hma.iloc[-2] else '하락'
    
    band_width = upper_mantra.iloc[-1] - lower_mantra.iloc[-1]
    prev_band_width = upper_mantra.iloc[-2] - lower_mantra.iloc[-2]
    band_state = '확장' if band_width > prev_band_width else '수축'
    
    rsi_state = '과매수' if rsi14.iloc[-1] > 70 else '과매도' if rsi14.iloc[-1] < 30 else '중립'
    
    macd_state = '상승' if macd.iloc[-1] > signal_line.iloc[-1] else '하락'
    
    if hma_trend == '상승' and band_state == '확장' and rsi_state != '과매수' and macd_state == '상승':
        return '적극적 매수 (추세 상승 + 변동성 확대)'
    elif hma_trend == '하락' and band_state == '확장' and rsi_state != '과매도' and macd_state == '하락':
        return '적극적 매도 (추세 하락 + 변동성 확대)'
    elif band_state == '수축':
        return '중립 (변동성 수축구간, 브레이크아웃 대기)'
    elif rsi_state == '과매수':
        return '부분 차익실현 고려 (과매수 구간)'
    elif rsi_state == '과매도':
        return '분할 매수 고려 (과매도 구간)'
    else:
        return '중립 (추가 시그널 확인 필요)' 