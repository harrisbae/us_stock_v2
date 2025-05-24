import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from datetime import datetime
import matplotlib.dates as mdates
import yfinance as yf
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.lines as mlines
import traceback
import sys
import platform
import warnings
import requests
from io import BytesIO
import zipfile
import xml.etree.ElementTree as ET
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore", category=FutureWarning)

# 한글 폰트 설정 (운영체제별)
import matplotlib
import matplotlib.font_manager as fm

def get_available_font():
    # 선호하는 폰트 목록
    preferred_fonts = ['AppleGothic', 'Malgun Gothic', 'NanumGothic', 'Arial Unicode MS']
    
    # 시스템에 설치된 폰트 목록
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 선호하는 폰트 중 사용 가능한 첫 번째 폰트 반환
    for font in preferred_fonts:
        if font in available_fonts:
            return font
    
    # 기본 폰트 반환
    return 'DejaVu Sans'

# 사용 가능한 폰트 설정
font_name = get_available_font()
matplotlib.rc('font', family=font_name)
matplotlib.rcParams['axes.unicode_minus'] = False

# HMA 계산
def calculate_hma(data: pd.Series, period: int = 20) -> pd.Series:
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    wma1 = data.ewm(span=half_period, adjust=False).mean()
    wma2 = data.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=sqrt_period, adjust=False).mean()
    return hma

# GICS 섹터 분류 함수
def get_gics_sector(ticker):
    # 여기에 실제 GICS 섹터 분류 로직 구현 필요
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

# GICS 서브섹터 분류 함수
def get_gics_subsector(ticker):
    # 여기에 실제 GICS 서브섹터 분류 로직 구현 필요
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

# 섹터 전망 함수
def get_sector_outlook(ticker, vix, tnx, dxy):
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

# 투자전략 도출 함수
def get_trading_strategy(hma, upper_mantra, lower_mantra, rsi14, macd, signal_line):
    # HMA 추세
    hma_trend = '상승' if hma.iloc[-1] > hma.iloc[-2] else '하락'
    
    # 만트라밴드 상태
    band_width = upper_mantra.iloc[-1] - lower_mantra.iloc[-1]
    prev_band_width = upper_mantra.iloc[-2] - lower_mantra.iloc[-2]
    band_state = '확장' if band_width > prev_band_width else '수축'
    
    # RSI 상태
    rsi_state = '과매수' if rsi14.iloc[-1] > 70 else '과매도' if rsi14.iloc[-1] < 30 else '중립'
    
    # MACD 상태
    macd_state = '상승' if macd.iloc[-1] > signal_line.iloc[-1] else '하락'
    
    # 전략 도출
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

# 만트라 밴드 계산
def calculate_mantra_bands(data: pd.Series, period: int = 20, multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series]:
    middle_line = calculate_hma(data, period)
    std = data.rolling(window=period).std()
    upper_band = middle_line + (multiplier * std)
    lower_band = middle_line - (multiplier * std)
    return upper_band, lower_band

def safe_in_index(index, key):
    """인덱스에 key가 있는지 안전하게 확인합니다."""
    try:
        return key in index
    except TypeError as e:
        print(f"[경고] hashable 타입이 아닌 객체가 인덱스에 사용됨: {type(key)}")
        print(f"key 값: {key}")
        print(f"index 타입: {type(index)}")
        print(f"예외 메시지: {str(e)}")
        print("스택 트레이스:")
        traceback.print_exc()
        return False

# HMA 신호 생성
def get_hma_signals(data: pd.DataFrame) -> List[dict]:
    """HMA 신호를 생성합니다."""
    try:
        signals = []
        hma = calculate_hma(data['Close'])
        for i in range(1, len(data)):
            try:
                prev_close = float(data['Close'].iloc[i-1].item())
                curr_close = float(data['Close'].iloc[i].item())
                prev_hma = float(hma.iloc[i-1].item())
                curr_hma = float(hma.iloc[i].item())
                # NaN 체크
                if pd.isna(prev_close) or pd.isna(curr_close) or pd.isna(prev_hma) or pd.isna(curr_hma):
                    continue
                # 매수 신호
                if prev_close < prev_hma and curr_close > curr_hma:
                    strength = 'STRONG' if curr_close > curr_hma * 1.02 else 'WEAK'
                    signals.append({
                        'date': data.index[i],
                        'type': 'BUY',
                        'price': curr_close,
                        'strength': strength
                    })
                # 매도 신호
                elif prev_close > prev_hma and curr_close < curr_hma:
                    strength = 'STRONG' if curr_close < curr_hma * 0.98 else 'WEAK'
                    signals.append({
                        'date': data.index[i],
                        'type': 'SELL',
                        'price': curr_close,
                        'strength': strength
                    })
            except Exception as e:
                print(f"신호 생성 중 오류 발생 (인덱스 {i}):")
                print(f"예외 메시지: {str(e)}")
                print("스택 트레이스:")
                traceback.print_exc()
                continue
        return signals
    except Exception as e:
        print("HMA 신호 생성 중 오류 발생:")
        print(f"예외 메시지: {str(e)}")
        print("스택 트레이스:")
        traceback.print_exc()
        return []

# 만트라 밴드 신호 생성
def get_mantra_signals(data: pd.DataFrame) -> List[dict]:
    """만트라 밴드 신호를 생성합니다."""
    try:
        signals = []
        upper_band, lower_band = calculate_mantra_bands(data['Close'])
        for i in range(1, len(data)):
            try:
                curr_close = float(data['Close'].iloc[i].item())
                curr_upper = float(upper_band.iloc[i].item())
                curr_lower = float(lower_band.iloc[i].item())
                # NaN 체크
                if pd.isna(curr_close) or pd.isna(curr_upper) or pd.isna(curr_lower):
                    continue
                # 과매수 신호
                if curr_close > curr_upper:
                    strength = 'STRONG' if curr_close > curr_upper * 1.02 else 'WEAK'
                    signals.append({
                        'date': data.index[i],
                        'type': 'OVERBOUGHT',
                        'price': curr_close,
                        'strength': strength
                    })
                # 과매도 신호
                elif curr_close < curr_lower:
                    strength = 'STRONG' if curr_close < curr_lower * 0.98 else 'WEAK'
                    signals.append({
                        'date': data.index[i],
                        'type': 'OVERSOLD',
                        'price': curr_close,
                        'strength': strength
                    })
            except Exception as e:
                print(f"만트라 신호 생성 중 오류 발생 (인덱스 {i}):")
                print(f"예외 메시지: {str(e)}")
                print("스택 트레이스:")
                traceback.print_exc()
                continue
        return signals
    except Exception as e:
        print("만트라 신호 생성 중 오류 발생:")
        print(f"예외 메시지: {str(e)}")
        print("스택 트레이스:")
        traceback.print_exc()
        return []

# HMA & 만트라 밴드 시각화
def plot_hma_mantra(data: pd.DataFrame, ticker: str, save_path: str = None) -> None:
    """HMA & 만트라 밴드를 시각화합니다."""
    try:
        hma = calculate_hma(data['Close'])
        upper_band, lower_band = calculate_mantra_bands(data['Close'])
        hma_signals = get_hma_signals(data)
        price_range = to_float(data['High'].max()) - to_float(data['Low'].min())
        offset = price_range * 0.03
        plt.figure(figsize=(15, 8))
        ax1 = plt.gca()
        plt.plot(data.index, data['Close'], label='Close', color='black', alpha=0.5)
        plt.plot(data.index, hma, label='HMA', color='blue', linewidth=2)
        plt.plot(data.index, upper_band, label='Upper Band', color='red', linestyle='--')
        plt.plot(data.index, lower_band, label='Lower Band', color='green', linestyle='--')
        if hma_signals:
            # 신호 마커 및 배경색 추가
            for i in range(len(hma_signals)):
                current_signal = hma_signals[i]
                current_date = current_signal['date']
                
                # 다음 신호의 날짜 찾기 (마지막 신호가 아닌 경우)
                if i < len(hma_signals) - 1:
                    next_date = hma_signals[i + 1]['date']
                else:
                    next_date = data.index[-1]
                
                # 배경색 추가 (현재 신호부터 다음 신호 전까지)
                if current_signal['type'] == 'BUY':
                    ax1.axvspan(current_date, next_date, color='green', alpha=0.065)
                else:
                    ax1.axvspan(current_date, next_date, color='red', alpha=0.065)
            
            # 신호 마커 및 텍스트 추가
            for idx, signal in enumerate(hma_signals):
                try:
                    # 색상 구분 (마커 색상)
                    if signal['type'] == 'BUY':
                        color = 'blue'  # 매수 마커 색상 통일
                    else:
                        color = 'red'  # 매도 마커 색상 통일
                    
                    price = signal['price']
                    offset = (abs(upper_band.max() - lower_band.min()) * 0.02)
                    if signal['type'] == 'BUY':
                        if signal['date'] in lower_band.index:
                            y = lower_band.loc[signal['date']] - offset
                        else:
                            y = price - offset
                        va = 'top'
                        text_y = y - offset * 1.2
                    else:
                        if signal['date'] in upper_band.index:
                            y = upper_band.loc[signal['date']] + offset
                        else:
                            y = price + offset
                        va = 'bottom'
                        text_y = y + offset * 1.2
                    
                    # 마커
                    ax1.plot(signal['date'], y, marker='^' if signal['type']=='BUY' else 'v', 
                            color=color, markersize=10, markeredgecolor='black', 
                            linestyle='None', label='매수 신호' if signal['type']=='BUY' else '매도 신호', 
                            zorder=10)
                    
                    # 텍스트
                    ax1.text(signal['date'], text_y, f"{signal['date'].strftime('%Y-%m-%d')}\n{signal.get('reason_short', '')}",
                            rotation=45, fontsize=7, color='black', ha='center', va=va,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', 
                                    boxstyle='round,pad=0.1'), zorder=11)
                    
                    # 수직선은 메인 차트에만 표시
                    ax1.axvline(x=signal['date'], color=color, linestyle='--', linewidth=0.3, alpha=0.8)
                    
                except Exception as e:
                    print(f"신호 마커 추가 중 오류 발생 (인덱스 {idx}):")
                    print(f"signal: {signal}")
                    print(f"예외 메시지: {str(e)}")
                    print("스택 트레이스:")
                    traceback.print_exc()
                    continue
            # 범례용 더미 객체
            buy_marker = mlines.Line2D([], [], color='green', marker='^', linestyle='None', markersize=3.6, label='Buy Signal')
            sell_marker = mlines.Line2D([], [], color='red', marker='v', linestyle='None', markersize=3.6, label='Sell Signal')
            plt.legend(handles=[buy_marker, sell_marker])
            plt.title(f'{ticker} - HMA & Mantra Bands')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(True, alpha=0.3)
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
    except Exception as e:
        print("HMA & 만트라 밴드 시각화 중 오류 발생:")
        print(f"예외 메시지: {str(e)}")
        print("스택 트레이스:")
        traceback.print_exc()

# 볼린저 밴드와 HMA/만트라 밴드 비교 시각화
def plot_comparison(data: pd.DataFrame, ticker: str, save_path: str = None) -> None:
    """볼린저 밴드와 HMA/만트라 밴드를 비교 시각화합니다."""
    try:
        hma = calculate_hma(data['Close'])
        upper_mantra, lower_band = calculate_mantra_bands(data['Close'])
        sma = data['Close'].rolling(window=20).mean()
        std = data['Close'].rolling(window=20).std()
        upper_bollinger = sma + (2 * std)
        lower_bollinger = sma - (2 * std)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
        ax1.plot(data.index, data['Close'], label='Close', color='black', alpha=0.5)
        ax1.plot(data.index, hma, label='HMA', color='blue', linewidth=2)
        ax1.plot(data.index, upper_mantra, label='Mantra Upper', color='red', linestyle='--')
        ax1.plot(data.index, lower_band, label='Mantra Lower', color='green', linestyle='--')
        hma_signals = get_hma_signals(data)
        price_range = data['High'].max() - data['Low'].min()
        offset = price_range * 0.03
        if hma_signals:
            for idx, signal in enumerate(hma_signals):
                try:
                    color = 'green' if signal['type'] == 'BUY' else 'red'
                    marker = '^' if signal['type'] == 'BUY' else 'v'
                    date = signal['date']
                    if signal['type'] == 'BUY':
                        band_val = lower_band.loc[date] if safe_in_index(lower_band.index, date) else signal['price']
                        y = band_val - offset
                        ax1.scatter(date, y, color=color, marker=marker, s=30, label=None)
                    else:
                        band_val = upper_mantra.loc[date] if safe_in_index(upper_mantra.index, date) else signal['price']
                        y = band_val + offset
                        ax1.scatter(date, y, color=color, marker=marker, s=30, label=None)
                except Exception as e:
                    print(f"신호 마커 추가 중 오류 발생 (인덱스 {idx}):")
                    print(f"signal: {signal}")
                    print(f"예외 메시지: {str(e)}")
                    print("스택 트레이스:")
                    traceback.print_exc()
                    continue
        # 범례용 더미 객체
        buy_marker = mlines.Line2D([], [], color='green', marker='^', linestyle='None', markersize=3.6, label='Buy Signal')
        sell_marker = mlines.Line2D([], [], color='red', marker='v', linestyle='None', markersize=3.6, label='Sell Signal')
        ax1.legend(handles=[buy_marker, sell_marker])
        ax1.set_title(f'{ticker} - HMA & Mantra Bands')
        ax1.set_ylabel('Price')
        ax2.plot(data.index, data['Close'], label='Close', color='black', alpha=0.5)
        ax2.plot(data.index, sma, label='SMA', color='blue', linewidth=2)
        ax2.plot(data.index, upper_bollinger, label='Bollinger Upper', color='red', linestyle='--')
        ax2.plot(data.index, lower_bollinger, label='Bollinger Lower', color='green', linestyle='--')
        ax2.set_title(f'{ticker} - Bollinger Bands')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    except Exception as e:
        print("볼린저 밴드와 HMA/만트라 밴드 비교 시각화 중 오류 발생:")
        print(f"예외 메시지: {str(e)}")
        print("스택 트레이스:")
        traceback.print_exc()

# 이동평균선 계산
def calculate_ma(data: pd.Series, period: int) -> pd.Series:
    return data.rolling(window=period).mean()

# MACD 계산
def calculate_macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    exp1 = data.ewm(span=fast_period, adjust=False).mean()
    exp2 = data.ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

# RSI 계산
def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# 안전한 float 변환 함수
def to_float(val):
    """안전한 float 변환 함수"""
    if hasattr(val, 'item'):
        return val.item()
    elif isinstance(val, (np.generic, np.ndarray)) and getattr(val, 'size', 1) == 1:
        return float(val)
    else:
        return float(val)

# 시그널 강도 시각화 (간단 버전)
def plot_signals_with_strength(data: pd.DataFrame, ticker: str, save_path: str = None) -> None:
    """신호와 강도를 함께 표시하는 차트를 그립니다."""
    try:
        # 데이터 인덱스를 datetime으로 통일
        data.index = pd.to_datetime(data.index)
        
        # 기본 지표 계산
        hma = calculate_hma(data['Close'])
        upper_mantra, lower_mantra = calculate_mantra_bands(data['Close'])
        hma_signals = get_hma_signals(data)
        
        # MACD 계산
        macd, signal_line, hist = calculate_macd(data['Close'])
        # RSI 계산
        rsi = calculate_rsi(data['Close'])
        # VIX 데이터 가져오기 및 인덱스 맞추기 (필요시)
        try:
            vix_data = yf.download('^VIX', start=data.index[0], end=data.index[-1])['Close']
            vix_data.index = pd.to_datetime(vix_data.index)
            vix = vix_data.reindex(data.index).ffill().bfill()
        except Exception as e:
            print(f"VIX 데이터 다운로드 실패: {e}")
            vix = pd.Series(index=data.index, data=np.nan)
        
        # 메인 차트만 생성
        fig, ax1 = plt.subplots(figsize=(15, 8))
        # ordinal_index를 numpy 1차원 배열로 변환
        ordinal_index = np.array(mdates.date2num(data.index.to_pydatetime())).flatten()
        
        # 캔들바 데이터 준비
        ohlc = [
            [mdates.date2num(date), float(row['Open'].iloc[0]), float(row['High'].iloc[0]), float(row['Low'].iloc[0]), float(row['Close'].iloc[0])]
            for date, row in data.iterrows()
        ]
        # 캔들바 추가
        candlestick_ohlc(ax1, ohlc, width=0.6, colorup='green', colordown='red', alpha=0.6)
        
        # 오른쪽 Y축 추가
        ax1_right = ax1.twinx()
        ax1_right.set_ylim(ax1.get_ylim())
        ax1_right.tick_params(axis='y', labelright=True, labelleft=False)
        ax1.tick_params(axis='y', labelright=False, labelleft=True)
        
        # 가격선, HMA, 만트라 밴드만 표시 (MA 선 제거)
        ax1.plot(ordinal_index, data['Close'], label='Close', color='black', linewidth=0.6, alpha=0.7)
        ax1.plot(ordinal_index, hma, label='HMA', color='blue', linewidth=2)
        ax1.plot(ordinal_index, upper_mantra, label='Mantra Upper', color='red', linestyle='--', linewidth=0.7)
        ax1.plot(ordinal_index, lower_mantra, label='Mantra Lower', color='green', linestyle='--', linewidth=0.7)
        
        # 볼린저 밴드 상단-중간, 하단-중간 영역 색상 채우기
        ax1.fill_between(data.index, upper_mantra, lower_mantra, where=(upper_mantra >= lower_mantra), color='orange', alpha=0.13, label='Mantra Zone')
        
        # 신호 마커 및 버티컬 라인 추가
        price_range = data['High'].max() - data['Low'].min()
        offset = price_range * 0.03
        if hma_signals:
            # 신호 마커 및 배경색 추가
            for i in range(len(hma_signals)):
                current_signal = hma_signals[i]
                current_date = current_signal['date']
                
                # 다음 신호의 날짜 찾기 (마지막 신호가 아닌 경우)
                if i < len(hma_signals) - 1:
                    next_date = hma_signals[i + 1]['date']
                else:
                    next_date = data.index[-1]
                
                # 배경색 추가 (현재 신호부터 다음 신호 전까지)
                if current_signal['type'] == 'BUY':
                    ax1.axvspan(current_date, next_date, color='green', alpha=0.065)
                else:
                    ax1.axvspan(current_date, next_date, color='red', alpha=0.065)
            
            # 신호 마커 및 텍스트 추가
            for idx, signal in enumerate(hma_signals):
                try:
                    # 색상 구분 (마커 색상)
                    if signal['type'] == 'BUY':
                        color = 'blue'  # 매수 마커 색상 통일
                    else:
                        color = 'red'  # 매도 마커 색상 통일
                    
                    price = signal['price']
                    offset = (abs(upper_mantra.max() - lower_mantra.min()) * 0.02)
                    if signal['type'] == 'BUY':
                        if signal['date'] in lower_mantra.index:
                            y = lower_mantra.loc[signal['date']] - offset
                        else:
                            y = price - offset
                        va = 'top'
                        text_y = y - offset * 1.2
                    else:
                        if signal['date'] in upper_mantra.index:
                            y = upper_mantra.loc[signal['date']] + offset
                        else:
                            y = price + offset
                        va = 'bottom'
                        text_y = y + offset * 1.2
                    
                    # 마커
                    ax1.plot(signal['date'], y, marker='^' if signal['type']=='BUY' else 'v', 
                            color=color, markersize=10, markeredgecolor='black', 
                            linestyle='None', label='매수 신호' if signal['type']=='BUY' else '매도 신호', 
                            zorder=10)
                    
                    # 텍스트
                    ax1.text(signal['date'], text_y, f"{signal['date'].strftime('%Y-%m-%d')}\n{signal.get('reason_short', '')}",
                            rotation=45, fontsize=7, color='black', ha='center', va=va,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', 
                                    boxstyle='round,pad=0.1'), zorder=11)
                    
                    # 수직선은 메인 차트에만 표시
                    ax1.axvline(x=signal['date'], color=color, linestyle='--', linewidth=0.3, alpha=0.8)
                    
                except Exception as e:
                    print(f"신호 마커 추가 중 오류 발생 (인덱스 {idx}):")
                    print(f"signal: {signal}")
                    print(f"예외 메시지: {str(e)}")
                    print("스택 트레이스:")
                    traceback.print_exc()
                    continue
            # 범례용 더미 객체
            buy_marker = mlines.Line2D([], [], color='green', marker='^', linestyle='None', markersize=3.6, label='Buy Signal')
            sell_marker = mlines.Line2D([], [], color='red', marker='v', linestyle='None', markersize=3.6, label='Sell Signal')
            plt.legend(handles=[buy_marker, sell_marker])
            plt.title(f'{ticker} - HMA & Mantra Bands')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(True, alpha=0.3)
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
    except Exception as e:
        print("신호와 강도 시각화 중 오류 발생:")
        print(f"예외 메시지: {str(e)}")
        print("스택 트레이스:")
        traceback.print_exc()

def get_pixel_dy(ax, date_prev, hma_prev, date_now, hma_now):
    """두 데이터 좌표의 y축 픽셀 변화량을 반환합니다."""
    x1 = mdates.date2num(date_prev)
    x2 = mdates.date2num(date_now)
    pixel_xy1 = ax.transData.transform((x1, hma_prev))
    pixel_xy2 = ax.transData.transform((x2, hma_now))
    pixel_dy = pixel_xy2[1] - pixel_xy1[1]
    return pixel_dy

def get_hma_mantra_md_signals(data: pd.DataFrame, ticker: str = None) -> list:
    """
    매수/매도 신호 생성 (매도 신호 조건 사용자 정의 반영, 불필요한 변수 및 코드 정리)
    """
    signals = []
    
    # 데이터 프레임 구조 확인 및 처리
    if isinstance(data.columns, pd.MultiIndex):
        close_data = data[('Close', ticker)]
    else:
        close_data = data['Close']
    
    hma = calculate_hma(close_data)
    upper_band, lower_band = calculate_mantra_bands(close_data)
    rsi3 = calculate_rsi(close_data, period=3)
    rsi14 = calculate_rsi(close_data, period=14)
    # 기존 매수 신호에 필요한 변수만 선언
    macd, macd_signal, _ = calculate_macd(close_data)
    rsi = calculate_rsi(close_data)
    
    for i in range(1, len(data)):
        close = float(close_data.iloc[i])
        prev_close = float(close_data.iloc[i-1])
        hma_now = float(hma.iloc[i])
        hma_prev = float(hma.iloc[i-1])
        upper = float(upper_band.iloc[i])
        upper_prev = float(upper_band.iloc[i-1])
        rsi3_now = float(rsi3.iloc[i])
        rsi14_now = float(rsi14.iloc[i])
        
        # --- 매도 신호 조건 ---
        if (prev_close > hma_prev) and (close < hma_now) and (rsi3_now < rsi14_now):
            signals.append({
                'date': data.index[i],
                'type': 'SELL',
                'price': close,
                'reason': '종가 HMA 하향 돌파 + 타지마할 RSI 매도구간(rsi3 < rsi14)',
                'reason_short': 'T1'
            })
            continue
        
        if (prev_close > upper_prev) and (close < upper) and (rsi3_now < rsi14_now):
            signals.append({
                'date': data.index[i],
                'type': 'SELL',
                'price': close,
                'reason': '상단밴드 하향 돌파 + 타지마할 RSI 매도구간(rsi3 < rsi14)',
                'reason_short': 'T2'
            })
            continue
        
        # --- 매수 신호 조건(변경) ---
        # 1) 종가 HMA 상향 돌파 + 타지마할 RSI 매수 구간
        if (prev_close < hma_prev) and (close >= hma_now) and (rsi3_now >= rsi14_now):
            signals.append({
                'date': data.index[i],
                'type': 'BUY',
                'price': close,
                'reason': '종가 HMA 상향 돌파 + 타지마할 RSI 매수구간(rsi3 >= rsi14)',
                'reason_short': 'B1'
            })
            continue
        
        # 2) 전일 종가 < 밴드 하단, 당일 종가 > 밴드 하단 + 타지마할 RSI 매수 구간
        lower = float(lower_band.iloc[i])
        lower_prev = float(lower_band.iloc[i-1])
        if (prev_close < lower_prev) and (close >= lower) and (rsi3_now >= rsi14_now):
            signals.append({
                'date': data.index[i],
                'type': 'BUY',
                'price': close,
                'reason': '밴드 하단 상향 돌파 + 타지마할 RSI 매수구간(rsi3 >= rsi14)',
                'reason_short': 'B2'
            })
            continue
        
        # 3) 종가 HMA 상향 돌파 + 타지마할 RSI 매수구간 아님 + MACD 매수 구간
        if (prev_close < hma_prev) and (close >= hma_now) and (rsi3_now < rsi14_now) and (macd.iloc[i] > macd_signal.iloc[i]):
            signals.append({
                'date': data.index[i],
                'type': 'BUY',
                'price': close,
                'reason': '종가 HMA 상향 돌파 + 타지마할 RSI 매수구간 아님 + MACD 매수 구간',
                'reason_short': 'B3'
            })
            continue
    
    return signals

def get_signal_facecolor(signal):
    """신호 타입과 원인에 따른 마커 색상을 반환합니다."""
    if signal['type'] == 'BUY':
        color_map = {
            'C': '#FFD700',  # 금색
            'H': '#1E90FF',  # 도저블루
            'M': '#9370DB',  # 미디엄 퍼플
            'R': '#32CD32',  # 라임 그린
        }
    else:  # SELL
        color_map = {
            'C': '#FFA500',  # 주황색
            'H': '#000080',  # 네이비
            'M': '#DC143C',  # 크림슨
            'R': '#8B4513',  # 갈색
        }
    # reason_short가 있으면 우선 사용, 없으면 기존 reason에서 첫 글자 추출
    key = signal.get('reason_short') or (signal.get('reason','C')[0] if signal.get('reason') else 'C')
    return color_map.get(key, '#FFA500')  # 기본값은 주황색

def get_krx_foreign_trading(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """KRX에서 외국인 거래량 데이터를 가져옵니다."""
    try:
        # KRX OpenAPI URL
        url = "http://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService/getStockPriceInfo"
        
        # API 키 (실제 사용시 발급받은 키로 교체 필요)
        service_key = "YOUR_SERVICE_KEY"  # 실제 서비스 키로 교체 필요
        
        # 요청 파라미터
        params = {
            "serviceKey": service_key,
            "resultType": "json",
            "beginBasDt": start_date,
            "endBasDt": end_date,
            "itmsNm": ticker
        }
        
        # 데이터 요청
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if 'response' in data and 'body' in data['response']:
                items = data['response']['body']['items']['item']
                
                # 데이터 추출
                foreign_data = []
                for item in items:
                    foreign_data.append({
                        'date': item['basDt'],
                        'foreign_buy': float(item['frgnBuyAmt'].replace(',', '')),
                        'foreign_sell': float(item['frgnSellAmt'].replace(',', '')),
                        'foreign_net': float(item['frgnNetAmt'].replace(',', ''))
                    })
                
                # DataFrame 생성
                df = pd.DataFrame(foreign_data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                return df
            else:
                print("데이터 형식이 예상과 다릅니다.")
                return pd.DataFrame()
        else:
            print(f"KRX 데이터 요청 실패: {response.status_code}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"외국인 거래량 데이터 가져오기 실패: {str(e)}")
        return pd.DataFrame()

def plot_hma_mantra_md_signals(data: pd.DataFrame, ticker: str, save_path: str = None, current_price: float = None) -> None:
    try:
        # 데이터 인덱스를 datetime으로 통일
        data.index = pd.to_datetime(data.index)
        
        # 데이터 프레임 구조 확인 및 처리
        if isinstance(data.columns, pd.MultiIndex):
            close_data = data[('Close', ticker)]
            open_data = data[('Open', ticker)]
            high_data = data[('High', ticker)]
            low_data = data[('Low', ticker)]
            volume_data = data[('Volume', ticker)]
        else:
            close_data = data['Close']
            open_data = data['Open']
            high_data = data['High']
            low_data = data['Low']
            volume_data = data['Volume']
        
        # 현재가 설정 (입력값이 없으면 마지막 종가 사용)
        if current_price is not None:
            display_price = float(current_price)
            price_label = f'현재가(실시간): ${display_price:,.2f}'
            price_color = 'blue'  # 실시간 가격은 파란색으로 표시
        else:
            display_price = float(close_data.iloc[-1])
            price_label = f'종가: ${display_price:,.2f}'
            price_color = 'black'  # 종가는 검은색으로 표시
        
        # 20일 박스권 계산
        last_20_high = high_data.rolling(window=20).max()
        last_20_low = low_data.rolling(window=20).min()

        # 데이터가 충분한지 확인
        if len(data) >= 20 and not last_20_high.iloc[-1:].isna().any() and not last_20_low.iloc[-1:].isna().any():
            current_resistance = float(last_20_high.iloc[-1])
            current_support = float(last_20_low.iloc[-1])
        else:
            # 데이터가 부족한 경우 전체 기간의 최고/최저값 사용
            current_resistance = float(high_data.max())
            current_support = float(low_data.min())
        
        # 기본 지표 계산
        hma = calculate_hma(close_data)
        upper_mantra, lower_mantra = calculate_mantra_bands(close_data)
        hma_signals = get_hma_mantra_md_signals(data, ticker)
        
        # 볼린저 밴드 계산
        sma20 = calculate_ma(close_data, 20)
        bb_std = close_data.rolling(window=20).std()
        bb_upper = sma20 + (bb_std * 2)
        bb_lower = sma20 - (bb_std * 2)
        
        # MACD 계산
        macd, signal_line, hist = calculate_macd(close_data)
        
        # RSI 계산 (타지마할 RSI: 3, 14, 50일)
        rsi3 = calculate_rsi(close_data, period=3)
        rsi14 = calculate_rsi(close_data, period=14)
        rsi50 = calculate_rsi(close_data, period=50)
        
        # 추가 지표 데이터 다운로드
        try:
            # VIX 데이터
            vix_data = yf.download('^VIX', start=data.index[0], end=data.index[-1])['Close']
            vix_data = vix_data.reindex(data.index).ffill()
            current_vix = float(vix_data.iloc[-1]) if not vix_data.empty else 0.0
            
            # 10년물 금리
            tnx_data = yf.download('^TNX', start=data.index[0], end=data.index[-1])['Close']
            tnx_data = tnx_data.reindex(data.index).ffill()
            current_tnx = float(tnx_data.iloc[-1]) if not tnx_data.empty else 0.0
            
            # 달러인덱스
            dxy_data = yf.download('DX-Y.NYB', start=data.index[0], end=data.index[-1])['Close']
            dxy_data = dxy_data.reindex(data.index).ffill()
            current_dxy = float(dxy_data.iloc[-1]) if not dxy_data.empty else 0.0
            
            # 금가격
            gold_data = yf.download('GC=F', start=data.index[0], end=data.index[-1])['Close']
            gold_data = gold_data.reindex(data.index).ffill()
            current_gold = float(gold_data.iloc[-1]) if not gold_data.empty else 0.0
        except Exception as e:
            print(f"추가 지표 데이터 다운로드 실패: {e}")
            # 데이터가 없는 경우 기본값 설정
            vix_data = pd.Series(20.0, index=data.index)  # 보통의 VIX 수준
            tnx_data = pd.Series(4.0, index=data.index)   # 보통의 금리 수준
            dxy_data = pd.Series(100.0, index=data.index) # 보통의 달러인덱스 수준
            gold_data = pd.Series(2000.0, index=data.index) # 보통의 금가격 수준
            current_vix = 20.0
            current_tnx = 4.0
            current_dxy = 100.0
            current_gold = 2000.0
        
        # 현재 시장 상태 계산
        current_vix = float(vix_data.iloc[-1])
        current_tnx = float(tnx_data.iloc[-1])
        current_dxy = float(dxy_data.iloc[-1])
        current_gold = float(gold_data.iloc[-1])
        
        # 차트 생성 (8개의 서브플롯)
        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(8, 1, figsize=(15, 24), 
            height_ratios=[7.2, 1, 1, 1, 1, 1, 1, 1], sharex=True)
        
        # 배경색 설정
        fig.patch.set_facecolor('white')
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
            ax.set_facecolor('white')
        
        # 캔들바 데이터 준비
        ohlc = []
        for date, o, h, l, c in zip(data.index, open_data, high_data, low_data, close_data):
            ohlc.append([
                mdates.date2num(date),
                float(o),
                float(h),
                float(l),
                float(c)
            ])
        
        # 메인 차트 (캔들차트)
        candlestick_ohlc(ax1, ohlc, width=0.6, colorup='green', colordown='red', alpha=0.6)
        
        # Y축 포맷 설정
        def price_formatter(x, p):
            return f'${x:,.2f}'

        ax1.yaxis.set_major_formatter(plt.FuncFormatter(price_formatter))

        # 오른쪽 Y축 추가 및 포맷 설정
        ax1_right = ax1.twinx()
        ax1_right.set_ylim(ax1.get_ylim())
        ax1_right.yaxis.set_major_formatter(plt.FuncFormatter(price_formatter))
        ax1_right.tick_params(axis='y', labelright=True, labelleft=False)
        ax1.tick_params(axis='y', labelright=False, labelleft=True)

        # Y축 범위 설정 (현재가 고려)
        y_min = min(float(low_data.min()), display_price)
        y_max = max(float(high_data.max()), display_price)
        y_margin = (y_max - y_min) * 0.1
        ax1.set_ylim(y_min - y_margin, y_max + y_margin)
        ax1_right.set_ylim(ax1.get_ylim())
        
        # 20일 박스권 지지선/저항선 추가
        ax1.axhline(y=current_resistance, color='red', linestyle='--', linewidth=1, alpha=0.8, 
                   label=f'저항선: ${current_resistance:,.2f}')
        ax1.axhline(y=current_support, color='green', linestyle='--', linewidth=1, alpha=0.8,
                   label=f'지지선: ${current_support:,.2f}')
        
        # 저항선/지지선 가격 텍스트 추가
        ax1.text(data.index[-1], current_resistance, f'{current_resistance:.2f}', 
                color='red', fontsize=6, rotation=45, ha='left', va='bottom',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))
        ax1.text(data.index[-1], current_support, f'{current_support:.2f}', 
                color='green', fontsize=6, rotation=45, ha='left', va='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))

        # 현재가 수평선 및 텍스트 추가 (수정된 부분)
        ax1.axhline(y=display_price, color=price_color, linestyle='-', linewidth=1.2, alpha=0.8)
        ax1.text(data.index[-1], display_price, price_label,
                color=price_color, fontsize=5, rotation=45, ha='left', va='bottom',
                bbox=dict(facecolor='white', edgecolor=price_color, alpha=0.7, pad=0.5))
        
        # 주요 선 추가
        ax1.plot(data.index, close_data, label='Close', color='black', linewidth=0.6, alpha=0.7)
        ax1.plot(data.index, hma, label='HMA', color='blue', linewidth=2)
        ax1.plot(data.index, upper_mantra, label='Mantra Upper', color='red', linestyle='--', linewidth=0.7)
        ax1.plot(data.index, lower_mantra, label='Mantra Lower', color='green', linestyle='--', linewidth=0.7)
        ax1.plot(data.index, bb_upper, label='BB Upper', color='orange', linestyle=':', linewidth=0.7)
        ax1.plot(data.index, bb_lower, label='BB Lower', color='orange', linestyle=':', linewidth=0.7)
        ax1.plot(data.index, sma20, label='SMA20', color='purple', linestyle=':', linewidth=0.7)
        
        # 만트라 밴드 영역 색상 채우기
        ax1.fill_between(data.index, upper_mantra, lower_mantra, 
                        where=(upper_mantra >= lower_mantra), 
                        color='orange', alpha=0.1)
        
        # 신호 마커 및 버티컬 라인 추가
        price_range = high_data.max() - low_data.min()
        offset = price_range * 0.03
        if hma_signals:
            # 신호 마커 및 배경색 추가
            for i in range(len(hma_signals)):
                current_signal = hma_signals[i]
                current_date = current_signal['date']
                
                # 다음 신호의 날짜 찾기 (마지막 신호가 아닌 경우)
                if i < len(hma_signals) - 1:
                    next_date = hma_signals[i + 1]['date']
                else:
                    next_date = data.index[-1]
                
                # 배경색 추가 (현재 신호부터 다음 신호 전까지)
                if current_signal['type'] == 'BUY':
                    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
                        ax.axvspan(current_date, next_date, color='green', alpha=0.065)
                else:
                    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
                        ax.axvspan(current_date, next_date, color='red', alpha=0.065)
            
            # 신호 마커 및 텍스트 추가
            for idx, signal in enumerate(hma_signals):
                try:
                    # 색상 구분 (마커 색상)
                    if signal['type'] == 'BUY':
                        color = 'blue'  # 매수 마커 색상 통일
                    else:
                        color = 'red'  # 매도 마커 색상 통일
                    
                    price = signal['price']
                    date_idx = data.index.get_loc(signal['date'])
                    hma_value = hma.iloc[date_idx]
                    price_diff_ratio = abs(price - hma_value) / hma_value
                    
                    # 캔들바가 HMA를 걸치는지 확인
                    high_price = high_data.iloc[date_idx]
                    low_price = low_data.iloc[date_idx]
                    is_not_crossing_hma = (low_price > hma_value) or (high_price < hma_value)
                    
                    offset = (abs(upper_mantra.max() - lower_mantra.min()) * 0.02)
                    if signal['type'] == 'BUY':
                        if signal['date'] in lower_mantra.index:
                            y = lower_mantra.loc[signal['date']] - offset
                        else:
                            y = price - offset
                        va = 'top'
                        text_y = y - offset * 1.2
                        
                        # 매수 마커 표시
                        ax1.plot(signal['date'], y, marker='^', 
                                color=color, markersize=10, markeredgecolor='black', 
                                linestyle='None', label='매수 신호', zorder=10)
                        
                        # 가격이 HMA보다 90% 이상 높거나, HMA를 걸치지 않고 완전히 위에 있는 경우 태양 마크 추가
                        if (price > hma_value and price_diff_ratio >= 0.9) or (is_not_crossing_hma and low_price > hma_value):
                            ax1.plot(signal['date'], y - offset * 1.5, marker='*', 
                                    color='gold', markersize=12, markeredgecolor='black',
                                    linestyle='None', zorder=11)
                    else:
                        if signal['date'] in upper_mantra.index:
                            y = upper_mantra.loc[signal['date']] + offset
                        else:
                            y = price + offset
                        va = 'bottom'
                        text_y = y + offset * 1.2
                        
                        # 매도 마커 표시
                        ax1.plot(signal['date'], y, marker='v', 
                                color=color, markersize=10, markeredgecolor='black', 
                                linestyle='None', label='매도 신호', zorder=10)
                        
                        # 가격이 HMA보다 90% 이상 낮거나, HMA를 걸치지 않고 완전히 아래에 있는 경우 태양 마크 추가
                        if (price < hma_value and price_diff_ratio >= 0.9) or (is_not_crossing_hma and high_price < hma_value):
                            ax1.plot(signal['date'], y + offset * 1.5, marker='*', 
                                    color='gold', markersize=12, markeredgecolor='black',
                                    linestyle='None', zorder=11)
                    
                    # 텍스트 표시
                    ax1.text(signal['date'], text_y, f"{signal['date'].strftime('%Y-%m-%d')}\n{signal.get('reason_short', '')}",
                            rotation=45, fontsize=7, color='black', ha='center', va=va,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', 
                                    boxstyle='round,pad=0.1'), zorder=11)
                    
                    # 수직선은 메인 차트에만 표시
                    ax1.axvline(x=signal['date'], color=color, linestyle='--', linewidth=0.3, alpha=0.8)
                    
                except Exception as e:
                    print(f"신호 마커 추가 중 오류 발생 (인덱스 {idx}):")
                    print(f"signal: {signal}")
                    print(f"예외 메시지: {str(e)}")
                    print("스택 트레이스:")
                    traceback.print_exc()
                    continue
        
        # 타지마할 RSI 서브플롯
        ax2.plot(data.index, rsi3, label='RSI(3)', color='blue', linewidth=1)
        ax2.plot(data.index, rsi14, label='RSI(14)', color='red', linewidth=1)
        ax2.plot(data.index, rsi50, label='RSI(50)', color='green', linewidth=1)
        ax2.axhline(y=70, color='red', linestyle='--', linewidth=0.5)
        ax2.axhline(y=30, color='green', linestyle='--', linewidth=0.5)

        # RSI 매수/매도 신호 표시
        for i in range(1, len(data)):
            if rsi3.iloc[i] >= rsi14.iloc[i] and rsi3.iloc[i-1] < rsi14.iloc[i-1]:
                ax2.plot(data.index[i], rsi3.iloc[i], '^', color='blue', markersize=8, 
                        label='RSI 매수신호' if i == 1 else '')
            elif rsi3.iloc[i] < rsi14.iloc[i] and rsi3.iloc[i-1] >= rsi14.iloc[i-1]:
                ax2.plot(data.index[i], rsi3.iloc[i], 'v', color='red', markersize=8, 
                        label='RSI 매도신호' if i == 1 else '')

        # RSI 범례 추가
        rsi_handles = [
            mlines.Line2D([], [], color='blue', label='RSI(3)'),
            mlines.Line2D([], [], color='red', label='RSI(14)'),
            mlines.Line2D([], [], color='green', label='RSI(50)'),
            mlines.Line2D([], [], color='blue', marker='^', linestyle='None', 
                         markersize=8, label='RSI 매수신호'),
            mlines.Line2D([], [], color='red', marker='v', linestyle='None', 
                         markersize=8, label='RSI 매도신호')
        ]
        ax2.legend(handles=rsi_handles, loc='upper left', fontsize=8)

        ax2.set_ylabel('타지마할 RSI')
        ax2.grid(True, alpha=0.3)
        
        # MACD 서브플롯
        ax3.plot(data.index, macd, label='MACD', color='blue', linewidth=1)
        ax3.plot(data.index, signal_line, label='Signal', color='red', linewidth=1)
        ax3.bar(data.index, hist, label='Histogram', 
                color=['red' if h < 0 else 'green' for h in hist], alpha=0.5)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # MACD 매수/매도 신호 표시
        for i in range(1, len(data)):
            if macd.iloc[i] > signal_line.iloc[i] and macd.iloc[i-1] <= signal_line.iloc[i-1]:
                ax3.plot(data.index[i], macd.iloc[i], '^', color='blue', markersize=8,
                        label='MACD 매수신호' if i == 1 else '')
            elif macd.iloc[i] < signal_line.iloc[i] and macd.iloc[i-1] >= signal_line.iloc[i-1]:
                ax3.plot(data.index[i], macd.iloc[i], 'v', color='red', markersize=8,
                        label='MACD 매도신호' if i == 1 else '')

        # MACD 범례 추가
        macd_handles = [
            mlines.Line2D([], [], color='blue', label='MACD'),
            mlines.Line2D([], [], color='red', label='Signal'),
            mpatches.Patch(color='green', alpha=0.5, label='Histogram(+)'),
            mpatches.Patch(color='red', alpha=0.5, label='Histogram(-)'),
            mlines.Line2D([], [], color='blue', marker='^', linestyle='None',
                         markersize=8, label='MACD 매수신호'),
            mlines.Line2D([], [], color='red', marker='v', linestyle='None',
                         markersize=8, label='MACD 매도신호')
        ]
        ax3.legend(handles=macd_handles, loc='upper left', fontsize=8)

        ax3.set_ylabel('MACD')
        ax3.grid(True, alpha=0.3)
        
        # 거래량 서브플롯
        ax4.bar(data.index, volume_data, 
                color=['red' if c < o else 'green' for c, o in zip(close_data, open_data)],
                alpha=0.5)

        # 1일 평균 거래량 계산 및 수평선 추가
        avg_volume = volume_data.mean()
        ax4.axhline(y=avg_volume, color='blue', linestyle='--', linewidth=1.5, label=f'1일 평균 거래량: {format(int(avg_volume), ",")}')

        # 거래량 범례 추가
        volume_handles = [
            mpatches.Patch(color='green', alpha=0.5, label='상승 거래량'),
            mpatches.Patch(color='red', alpha=0.5, label='하락 거래량'),
            mlines.Line2D([], [], color='blue', linestyle='--', linewidth=1.5, 
                         label=f'1일 평균 거래량: {format(int(avg_volume), ",")}')
        ]
        ax4.legend(handles=volume_handles, loc='upper left', fontsize=8)

        ax4.set_ylabel('거래량')
        ax4.grid(True, alpha=0.3)
        
        # VIX 서브플롯
        ax5.plot(data.index, vix_data, label='VIX', color='purple', linewidth=1)

        # 현재 VIX 값 계산 및 수평선 추가 (실선으로 변경, 두께 0.5로 감소)
        current_vix = float(vix_data.iloc[-1])
        ax5.axhline(y=current_vix, color='black', linestyle='-', linewidth=0.5, alpha=0.8)

        # 현재 VIX 위치에 점 표시
        ax5.plot(data.index[-1], current_vix, 'o', color='black', markersize=6, 
                 markeredgecolor='white', markeredgewidth=1, zorder=5)

        # 현재 VIX 값 텍스트 표시 (교차점 위에 45도로 표시)
        ax5.text(data.index[-1], current_vix, 
                 f'{current_vix:.2f}', 
                 color='black', fontsize=6,
                 ha='left', va='bottom',
                 rotation=45,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5),
                 zorder=6)

        # VIX 전략 기준선 추가
        ax5.axhline(y=30, color='green', linestyle='--', linewidth=1, alpha=0.8)
        ax5.axhline(y=20, color='yellow', linestyle='--', linewidth=1, alpha=0.8)
        ax5.axhline(y=12, color='red', linestyle='--', linewidth=1, alpha=0.8)

        # VIX 구간별 배경색 표시
        ax5.fill_between(data.index, 30, ax5.get_ylim()[1], color='green', alpha=0.1, label='과도한 공포 구간 (매수기회)')
        ax5.fill_between(data.index, 20, 30, color='yellow', alpha=0.1, label='공포 구간')
        ax5.fill_between(data.index, 0, 12, color='red', alpha=0.1, label='과도한 낙관 구간 (매도기회)')

        # VIX 범례 추가
        vix_handles = [
            mlines.Line2D([], [], color='purple', linewidth=1, label='VIX'),
            mlines.Line2D([], [], color='black', linestyle='-', linewidth=0.5, 
                         label=f'현재 VIX: {current_vix:.2f}'),
            mpatches.Patch(color='green', alpha=0.1, label='과도한 공포 구간 (VIX > 30, 매수기회)'),
            mpatches.Patch(color='yellow', alpha=0.1, label='공포 구간 (20 < VIX < 30)'),
            mpatches.Patch(color='red', alpha=0.1, label='과도한 낙관 구간 (VIX < 12, 매도기회)')
        ]
        ax5.legend(handles=vix_handles, loc='upper left', fontsize=8)

        ax5.set_ylabel('VIX')
        ax5.grid(True, alpha=0.3)
        
        # 10년물 금리 서브플롯
        ax6.plot(data.index, tnx_data, label='10Y Treasury', color='green', linewidth=1)

        # 현재 금리 값 계산 및 수평선 추가
        current_tnx = float(tnx_data.iloc[-1])
        ax6.axhline(y=current_tnx, color='black', linestyle='-', linewidth=0.5, alpha=0.8)

        # 현재 금리 위치에 점 표시
        ax6.plot(data.index[-1], current_tnx, 'o', color='black', markersize=6, 
                 markeredgecolor='white', markeredgewidth=1, zorder=5)

        # 현재 금리 값 텍스트 표시
        ax6.text(data.index[-1], current_tnx, 
                 f'{current_tnx:.2f}', 
                 color='black', fontsize=6,
                 ha='left', va='bottom',
                 rotation=45,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5),
                 zorder=6)

        # 범례 추가
        tnx_handles = [
            mlines.Line2D([], [], color='green', linewidth=1, label='10Y Treasury'),
            mlines.Line2D([], [], color='black', linestyle='-', linewidth=0.5, 
                         label=f'현재 금리: {current_tnx:.2f}%')
        ]
        ax6.legend(handles=tnx_handles, loc='upper left', fontsize=8)

        ax6.set_ylabel('10년물 금리')
        ax6.grid(True, alpha=0.3)
        
        # 달러인덱스 서브플롯
        ax7.plot(data.index, dxy_data, label='Dollar Index', color='blue', linewidth=1)

        # 현재 달러인덱스 값 계산 및 수평선 추가
        current_dxy = float(dxy_data.iloc[-1])
        ax7.axhline(y=current_dxy, color='black', linestyle='-', linewidth=0.5, alpha=0.8)

        # 현재 달러인덱스 위치에 점 표시
        ax7.plot(data.index[-1], current_dxy, 'o', color='black', markersize=6, 
                 markeredgecolor='white', markeredgewidth=1, zorder=5)

        # 현재 달러인덱스 값 텍스트 표시
        ax7.text(data.index[-1], current_dxy, 
                 f'{current_dxy:.2f}', 
                 color='black', fontsize=6,
                 ha='left', va='bottom',
                 rotation=45,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5),
                 zorder=6)

        # 범례 추가
        dxy_handles = [
            mlines.Line2D([], [], color='blue', linewidth=1, label='Dollar Index'),
            mlines.Line2D([], [], color='black', linestyle='-', linewidth=0.5, 
                         label=f'현재 달러인덱스: {current_dxy:.2f}')
        ]
        ax7.legend(handles=dxy_handles, loc='upper left', fontsize=8)

        ax7.set_ylabel('달러인덱스')
        ax7.grid(True, alpha=0.3)
        
        # 금가격 서브플롯
        ax8.plot(data.index, gold_data, label='Gold', color='gold', linewidth=1)

        # 현재 금가격 값 계산 및 수평선 추가
        current_gold = float(gold_data.iloc[-1])
        ax8.axhline(y=current_gold, color='black', linestyle='-', linewidth=0.5, alpha=0.8)

        # 현재 금가격 위치에 점 표시
        ax8.plot(data.index[-1], current_gold, 'o', color='black', markersize=6, 
                 markeredgecolor='white', markeredgewidth=1, zorder=5)

        # 현재 금가격 값 텍스트 표시
        ax8.text(data.index[-1], current_gold, 
                 f'{current_gold:.2f}', 
                 color='black', fontsize=6,
                 ha='left', va='bottom',
                 rotation=45,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5),
                 zorder=6)

        # 범례 추가
        gold_handles = [
            mlines.Line2D([], [], color='gold', linewidth=1, label='Gold'),
            mlines.Line2D([], [], color='black', linestyle='-', linewidth=0.5, 
                         label=f'현재 금가격: {current_gold:.2f}')
        ]
        ax8.legend(handles=gold_handles, loc='upper left', fontsize=8)

        ax8.set_ylabel('금가격')
        ax8.grid(True, alpha=0.3)
        
        # 주요 라인 범례 (왼쪽 상단)
        line_handles = [
            mlines.Line2D([], [], color='blue', linewidth=2, label='HMA'),
            mlines.Line2D([], [], color='red', linestyle='--', linewidth=1, label='Mantra Upper'),
            mlines.Line2D([], [], color='green', linestyle='--', linewidth=1, label='Mantra Lower'),
            mlines.Line2D([], [], color='red', linestyle='--', linewidth=1, label=f'저항선: {current_resistance:.2f}'),
            mlines.Line2D([], [], color='green', linestyle='--', linewidth=1, label=f'지지선: {current_support:.2f}'),
            mlines.Line2D([], [], color=price_color, linestyle='-', linewidth=0.8, label=price_label),
            mpatches.Patch(color='green', alpha=0.065, label='매수구간'),
            mpatches.Patch(color='red', alpha=0.065, label='매도구간')
        ]
        legend1 = ax1.legend(handles=line_handles, loc='upper left', bbox_to_anchor=(0.02, 0.98),
                            fontsize=8, title='주요 라인', frameon=True, fancybox=True, borderpad=1)
        ax1.add_artist(legend1)
        
        # 신호 마커 범례 (왼쪽 중간)
        signal_handles = [
            mlines.Line2D([], [], color='blue', marker='^', linestyle='None', markersize=10,
                         markeredgecolor='black', label='매수 신호(▲)'),
            mlines.Line2D([], [], color='red', marker='v', linestyle='None', markersize=10,
                         markeredgecolor='black', label='매도 신호(▼)')
        ]
        legend2 = ax1.legend(handles=signal_handles, loc='upper left', bbox_to_anchor=(0.02, 0.45),
                            fontsize=8, title='신호 구분', frameon=True, fancybox=True, borderpad=1)
        ax1.add_artist(legend2)

        # 근거별 범례 (왼쪽 하단)
        legend_handles = [
            mlines.Line2D([], [], color='blue', marker='^', linestyle='None', markersize=10, markeredgecolor='black', label='B1: 종가 HMA 상향 돌파'),
            mlines.Line2D([], [], color='green', marker='^', linestyle='None', markersize=10, markeredgecolor='black', label='B2: 밴드 하단 상향 돌파'),
            mlines.Line2D([], [], color='red', marker='v', linestyle='None', markersize=10, markeredgecolor='black', label='T1: 종가 HMA 하향 돌파'),
            mlines.Line2D([], [], color='orange', marker='v', linestyle='None', markersize=10, markeredgecolor='black', label='T2: 상단밴드 하향 돌파'),
        ]
        legend3 = ax1.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.02, 0.15),
                            fontsize=8, title='주요 신호(근거별)', frameon=True, fancybox=True, borderpad=1)
        ax1.add_artist(legend3)
        
        # 차트 해설 텍스트 추가
        explanation = (
            '※ HMA/만트라 밴드 해석 및 실전 활용 주의사항\n'
            '- HMA가 평평하거나 하락일 때는 롱 포지션 진입 주의\n'
            '- 만트라 밴드는 스퀴즈(좁은 밴드)일 때 신호 신뢰도 낮아짐\n'
            '- 추세장에서는 만트라 밴드만으로는 부족 → 반드시 HMA와 병합\n'
            '- RSI(3)가 RSI(14)를 상향돌파: 매수 시그널 강화\n'
            '- RSI(3)가 RSI(14)를 하향돌파: 매도 시그널 강화\n'
            '- VIX, 금리, 달러, 금가격 변동성 참고'
        )
        
        # 차트 제목 설정 (타이틀 위치 조정)
        title = f'{ticker} - Technical Analysis ({data.index[0].strftime("%Y-%m-%d")} ~ {data.index[-1].strftime("%Y-%m-%d")})'
        fig.suptitle(title, y=0.98, fontsize=12)

        # 거시경제 분석 및 투자 전략 요약 텍스트 추가
        macro_analysis = (
            '[ 거시경제 환경 포지셔닝 ]\n'
            f'시장지표  VIX: {current_vix:.1f} │ 금리: {current_tnx:.1f}% │ DXY: {current_dxy:.1f} │ GOLD: ${current_gold:.1f}\n'
            '시장성향  ' + ('■ Risk-on  ' if current_vix < 20 else '□ Risk-off ') + 
            ('■ 긴축↑   ' if current_tnx > 4.0 else '□ 긴축↓   ') + 
            ('■ USD강세' if current_dxy > 104 else '□ USD약세') + '\n'
            '투자환경  ' + (
                '▶ 위험자산 선호 가능 (VIX↓ + 금리↓ + USD↓)' if current_vix < 20 and current_tnx < 4.0 and current_dxy < 104
                else '◀ 보수적 접근 필요 (VIX↑ + 금리↑ + USD↑)'
            )
        )

        # GICS 분류 및 섹터 전망
        gics_info = (
            '[ GICS 섹터 분석 ]\n'
            '섹터분류  ' + get_gics_sector(ticker) + '\n'
            '서브섹터  ' + get_gics_subsector(ticker) + '\n'
            '섹터전망  ' + ('▲ 긍정적 (성장주 선호 환경)' if current_vix < 20 and current_tnx < 4.0 
                        else '■ 중립적 (변동성 주시)')
        )

        # 기술적 분석 요약
        technical_summary = (
            '[ 기술적 분석 요약 ]\n'
            '추세분석  ' + ('↗ HMA 상승' if hma.iloc[-1] > hma.iloc[-2] else '↘ HMA 하락') + ' │ ' +
            ('▲ 밴드 확장' if (upper_mantra.iloc[-1] - lower_mantra.iloc[-1]) > (upper_mantra.iloc[-2] - lower_mantra.iloc[-2]) 
             else '● 밴드 수축') + '\n'
            '기술지표  ' + ('▲ 과매수' if rsi14.iloc[-1] > 70 else '▼ 과매도' if rsi14.iloc[-1] < 30 else '━ RSI 중립') + ' │ ' +
            ('↗ MACD 상승' if macd.iloc[-1] > signal_line.iloc[-1] else '↘ MACD 하락') + '\n'
            '투자전략  ' + (
                '★ 적극적 매수 (상승추세 + 변동성 확대)' 
                if (hma.iloc[-1] > hma.iloc[-2] and 
                    (upper_mantra.iloc[-1] - lower_mantra.iloc[-1]) > (upper_mantra.iloc[-2] - lower_mantra.iloc[-2]) and 
                    rsi14.iloc[-1] <= 70 and 
                    macd.iloc[-1] > signal_line.iloc[-1])
                else '★ 적극적 매도 (하락추세 + 변동성 확대)'
                if (hma.iloc[-1] < hma.iloc[-2] and 
                    (upper_mantra.iloc[-1] - lower_mantra.iloc[-1]) > (upper_mantra.iloc[-2] - lower_mantra.iloc[-2]) and 
                    rsi14.iloc[-1] >= 30 and 
                    macd.iloc[-1] < signal_line.iloc[-1])
                else '◆ 중립 대기 (변동성 수축)'
                if (upper_mantra.iloc[-1] - lower_mantra.iloc[-1]) <= (upper_mantra.iloc[-2] - lower_mantra.iloc[-2])
                else '▼ 부분 차익실현'
                if rsi14.iloc[-1] > 70
                else '▲ 분할 매수'
                if rsi14.iloc[-1] < 30
                else '■ 관망 (추가 시그널 대기)'
            )
        )

        # 분석 텍스트를 차트 상단에 추가 (위치 조정)
        # 상단 여백 확보를 위해 subplot 조정
        plt.subplots_adjust(top=0.91)  # 상단 여백 더욱 축소
        
        # 타이틀과 분석 텍스트를 위한 배경 박스 추가 (높이 20% 증가)
        fig.patches.extend([
            plt.Rectangle((0.02, 0.94), 0.96, 0.06,  # 높이 20% 증가 (0.05 -> 0.06)
                        facecolor='white', edgecolor='none',  # edgecolor를 'lightgray'에서 'none'으로 변경
                        alpha=0.95, transform=fig.transFigure)
        ])
        
        # 차트 메인 타이틀 (최상단, 위치 상향 조정)
        title = f'{ticker} - Technical Analysis ({data.index[0].strftime("%Y-%m-%d")} ~ {data.index[-1].strftime("%Y-%m-%d")})'
        fig.suptitle(title, y=0.985, fontsize=11, weight='bold')
        
        # 세 개의 서브 영역 박스 생성 (높이 20% 증가)
        box_height = 0.0384  # 높이 20% 증가 (0.032 -> 0.0384)
        base_y = 0.942  # 기준 y 좌표 하향 조정
        
        # 박스와 화살표를 위한 zorder 설정
        box_zorder = 1
        arrow_zorder = 2
        chart_zorder = 3
        text_zorder = 4

        # 왼쪽 박스 (거시경제 분석)
        left_box = plt.Rectangle((0.03, base_y), 0.30, box_height,
                               facecolor='white', edgecolor='none',  # edgecolor를 'lightgray'에서 'none'으로 변경
                               alpha=0.9, transform=fig.transFigure,
                               zorder=box_zorder)
        
        # 중앙 박스 (GICS 섹터 분석)
        center_box = plt.Rectangle((0.35, base_y), 0.30, box_height,
                                 facecolor='white', edgecolor='none',  # edgecolor를 'lightgray'에서 'none'으로 변경
                                 alpha=0.9, transform=fig.transFigure,
                                 zorder=box_zorder)
        
        # 오른쪽 박스 (기술적 분석)
        right_box = plt.Rectangle((0.67, base_y), 0.30, box_height,
                                facecolor='white', edgecolor='none',  # edgecolor를 'lightgray'에서 'none'으로 변경
                                alpha=0.9, transform=fig.transFigure,
                                zorder=box_zorder)
        
        fig.patches.extend([left_box, center_box, right_box])
        
        # 화살표 추가 (높이에 맞게 조정)
        arrow_y = base_y + (box_height * 0.6)  # 박스 높이에 비례하여 조정
        arrow1 = plt.Rectangle((0.33, arrow_y), 0.02, 0.012,
                             facecolor='black', alpha=0.3, transform=fig.transFigure,
                             zorder=arrow_zorder)
        arrow2 = plt.Rectangle((0.65, arrow_y), 0.02, 0.012,
                             facecolor='black', alpha=0.3, transform=fig.transFigure,
                             zorder=arrow_zorder)
        
        # 분석 정보 표시 (세 영역으로 분리)
        text_y = base_y + (box_height/2)  # 텍스트 세로 위치 조정
        
        # 왼쪽 (거시경제 분석) - 성장/물가 도표 추가
        # 성장/물가 도표를 위한 Axes 생성 (30% 크기 증가)
        chart_width, chart_height = 0.091, 0.0364  # 도표 크기 30% 증가
        chart_x = 0.03  # 도표 x 위치 왼쪽으로 이동
        chart_y = base_y + 0.008  # 도표 y 위치 상단으로 약간 이동
        
        macro_ax = fig.add_axes([chart_x, chart_y, chart_width, chart_height], zorder=chart_zorder)
        macro_ax.set_xticks([])
        macro_ax.set_yticks([])
        macro_ax.set_facecolor('white')  # 배경색을 흰색으로 설정
        
        # 도표 배경 설정 - 격자 추가
        macro_ax.grid(True, color='gray', linestyle=':', alpha=0.3)
        macro_ax.axvline(x=0.5, color='black', linewidth=0.5, alpha=0.3)
        macro_ax.axhline(y=0.5, color='black', linewidth=0.5, alpha=0.3)
        
        # 사분면 배경색 설정 (투명도 증가)
        macro_ax.fill_between([0, 0.5], 0.5, 1, color='red', alpha=0.15)    # 스태그플레이션
        macro_ax.fill_between([0.5, 1], 0.5, 1, color='yellow', alpha=0.15) # 인플레이션
        macro_ax.fill_between([0, 0.5], 0, 0.5, color='gray', alpha=0.15)   # 침체
        macro_ax.fill_between([0.5, 1], 0, 0.5, color='green', alpha=0.15)  # 이상적
        
        # 테두리 추가
        macro_ax.spines['top'].set_visible(True)
        macro_ax.spines['right'].set_visible(True)
        macro_ax.spines['bottom'].set_visible(True)
        macro_ax.spines['left'].set_visible(True)
        for spine in macro_ax.spines.values():
            spine.set_color('gray')
            spine.set_linewidth(0.5)
            spine.set_alpha(0.5)
        
        # 현재 위치 표시 (VIX와 금리 기반으로 판단)
        growth_x = 0.7 if current_vix < 20 else 0.3  # VIX가 낮으면 성장 긍정적
        inflation_y = 0.7 if current_tnx > 4.0 else 0.3  # 금리가 높으면 물가 상승
        
        # 현재 위치에 점 표시 (테두리가 있는 빨간 점, 반투명)
        macro_ax.plot(growth_x, inflation_y, 'o', color='red', markersize=6, alpha=0.5,
                     markeredgecolor='white', markeredgewidth=1, zorder=text_zorder-2)
        
        # 현재 위치에 십자선 추가
        macro_ax.axvline(x=growth_x, color='black', linestyle=':', linewidth=0.5, alpha=0.3, zorder=text_zorder-3)
        macro_ax.axhline(y=inflation_y, color='black', linestyle=':', linewidth=0.5, alpha=0.3, zorder=text_zorder-3)
        
        # 축 레이블 (도표에 더 가깝게 조정)
        macro_ax.text(0.5, -0.08, '성장', ha='center', va='top', fontsize=6, 
                     fontweight='bold', transform=macro_ax.transAxes, zorder=text_zorder+2)
        macro_ax.text(-0.02, 0.5, '물가', ha='right', va='center', fontsize=6, 
                     fontweight='bold', rotation=90, transform=macro_ax.transAxes, zorder=text_zorder+2)
        
        # 사분면 레이블 추가 (배경 추가)
        def add_text_with_background(ax, x, y, text, fontsize=4, alpha=0.7):
            text_obj = ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                             alpha=alpha, zorder=text_zorder+1)
            bbox = dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.8)
            text_obj.set_bbox(bbox)
        
        # 각 사분면에 경제 상황 표시
        add_text_with_background(macro_ax, 0.25, 0.75, '스태그\n플레이션', fontsize=4)
        add_text_with_background(macro_ax, 0.75, 0.75, '인플레이션', fontsize=4)
        add_text_with_background(macro_ax, 0.25, 0.25, '침체', fontsize=4)
        add_text_with_background(macro_ax, 0.75, 0.25, '이상적', fontsize=4)
        
        # 자산 종류별 마커 추가
        marker_size = 3
        
        # 스태그플레이션 영역 (방어주 + 원자재)
        macro_ax.plot(0.2, 0.8, 'o', color='blue', markersize=marker_size, label='필수소비재', zorder=text_zorder+3)
        macro_ax.plot(0.3, 0.8, '^', color='green', markersize=marker_size, label='헬스케어', zorder=text_zorder+3)
        macro_ax.plot(0.25, 0.85, 's', color='red', markersize=marker_size, label='에너지', zorder=text_zorder+3)
        
        # 인플레이션 영역 (가치주 + 실물)
        macro_ax.plot(0.7, 0.8, 's', color='brown', markersize=marker_size, label='금융', zorder=text_zorder+3)
        macro_ax.plot(0.8, 0.8, 'D', color='gold', markersize=marker_size, label='원자재', zorder=text_zorder+3)
        macro_ax.plot(0.75, 0.85, 'h', color='purple', markersize=marker_size, label='부동산', zorder=text_zorder+3)
        
        # 침체 영역 (안전자산)
        macro_ax.plot(0.2, 0.2, 'v', color='gray', markersize=marker_size, label='유틸리티', zorder=text_zorder+3)
        macro_ax.plot(0.3, 0.2, 'p', color='darkblue', markersize=marker_size, label='통신서비스', zorder=text_zorder+3)
        
        # 이상적 영역 (성장주)
        macro_ax.plot(0.7, 0.2, '*', color='orange', markersize=marker_size*1.2, label='IT', zorder=text_zorder+3)
        macro_ax.plot(0.8, 0.2, 'h', color='red', markersize=marker_size, label='임의소비재', zorder=text_zorder+3)
        macro_ax.plot(0.75, 0.25, '^', color='purple', markersize=marker_size, label='산업재', zorder=text_zorder+3)
        
        # 범례 추가 (우측에 세로로 위치, 도표와 가깝게, 그룹별 구분)
        # 그룹별 핸들과 라벨 생성
        handles = []
        labels = []
        
        # 스태그플레이션 영역 (방어주 + 원자재)
        handles.append(mlines.Line2D([], [], linestyle='none', marker='', color='none'))
        labels.append('■ 스태그플레이션 선호')
        handles.extend([
            mlines.Line2D([], [], color='blue', marker='o', linestyle='none', markersize=marker_size, label='필수소비재'),
            mlines.Line2D([], [], color='green', marker='^', linestyle='none', markersize=marker_size, label='헬스케어'),
            mlines.Line2D([], [], color='red', marker='s', linestyle='none', markersize=marker_size, label='에너지')
        ])
        labels.extend(['필수소비재', '헬스케어', '에너지'])
        
        # 인플레이션 영역 (가치주 + 실물)
        handles.append(mlines.Line2D([], [], linestyle='none', marker='', color='none'))
        labels.append('■ 인플레이션 선호')
        handles.extend([
            mlines.Line2D([], [], color='brown', marker='s', linestyle='none', markersize=marker_size, label='금융'),
            mlines.Line2D([], [], color='gold', marker='D', linestyle='none', markersize=marker_size, label='원자재'),
            mlines.Line2D([], [], color='purple', marker='h', linestyle='none', markersize=marker_size, label='부동산')
        ])
        labels.extend(['금융', '원자재', '부동산'])
        
        # 침체 영역 (안전자산)
        handles.append(mlines.Line2D([], [], linestyle='none', marker='', color='none'))
        labels.append('■ 침체기 선호')
        handles.extend([
            mlines.Line2D([], [], color='gray', marker='v', linestyle='none', markersize=marker_size, label='유틸리티'),
            mlines.Line2D([], [], color='darkblue', marker='p', linestyle='none', markersize=marker_size, label='통신서비스')
        ])
        labels.extend(['유틸리티', '통신서비스'])
        
        # 이상적 영역 (성장주)
        handles.append(mlines.Line2D([], [], linestyle='none', marker='', color='none'))
        labels.append('■ 이상적 국면 선호')
        handles.extend([
            mlines.Line2D([], [], color='orange', marker='*', linestyle='none', markersize=marker_size*1.2, label='IT'),
            mlines.Line2D([], [], color='red', marker='h', linestyle='none', markersize=marker_size, label='임의소비재'),
            mlines.Line2D([], [], color='purple', marker='^', linestyle='none', markersize=marker_size, label='산업재')
        ])
        labels.extend(['IT', '임의소비재', '산업재'])
        
        # 범례 생성 (도표 아래에 가로로 배치)
        legend = macro_ax.legend(handles, labels, bbox_to_anchor=(1.05, 1.3), loc='upper left',
                               ncol=1, fontsize=5, frameon=True,
                               columnspacing=0.35, handletextpad=0.2)
        
        # 사분면 구분선
        macro_ax.set_xlim(0, 1)
        macro_ax.set_ylim(0, 1)
        
        # 거시경제 분석 텍스트 (위치 조정)
        # 현재 경제 상황 판단
        econ_status = (
            "스태그플레이션" if growth_x < 0.5 and inflation_y > 0.5 else
            "인플레이션" if growth_x > 0.5 and inflation_y > 0.5 else
            "침체" if growth_x < 0.5 and inflation_y < 0.5 else
            "이상적"
        )
        
        # ASCII 화살표 사용
        def get_market_symbol(condition):
            return '↑' if condition else '↓'
        
        # 시장 상태에 따른 색상 코드
        def get_status_color(status):
            color_map = {
                "스태그플레이션": "red",
                "인플레이션": "orange",
                "침체": "gray",
                "이상적": "green"
            }
            return color_map.get(status, "black")
        
        # 시장 상태 아이콘
        def get_recommendation_symbol(is_positive):
            return '+' if is_positive else '-'
        
        # 투자 추천 상태
        is_favorable = current_vix < 20 and current_tnx < 4.0 and current_dxy < 104
        
        macro_text = (
            f"시장지표  VIX: {current_vix:.1f} | 금리: {current_tnx:.1f}% | DXY: {current_dxy:.1f}\n"
            '시장성향  ' + (f"{get_market_symbol(current_vix < 20)} Risk-on  " if current_vix < 20 else f"{get_market_symbol(current_vix < 20)} Risk-off ") + 
            (f"{get_market_symbol(current_tnx > 4.0)} 긴축   " if current_tnx > 4.0 else f"{get_market_symbol(current_tnx > 4.0)} 완화   ") + 
            (f"{get_market_symbol(current_dxy > 104)} USD강세" if current_dxy > 104 else f"{get_market_symbol(current_dxy > 104)} USD약세") + '\n'
            f'경제국면  {econ_status} | ' + (
                f"{get_recommendation_symbol(is_favorable)} 매수 검토" if is_favorable
                else f"{get_recommendation_symbol(is_favorable)} 관망 권고"
            )
        )
        
        fig.text(0.20, text_y, macro_text, ha='left', va='center', fontsize=8, color='black',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.0, pad=0.2))
        
        # 중앙 (GICS 섹터 분석)
        fig.text(0.5, text_y, gics_info, ha='center', va='center', fontsize=8, color='black',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.0, pad=0.2))
        
        # 오른쪽 (기술적 분석)
        fig.text(0.82, text_y, technical_summary, ha='center', va='center', fontsize=8, color='black',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.0, pad=0.2))
        
        # 차트 영역 시작 위치 조정
        fig.text(0.5, 0.922, "▼ 차트 분석 영역 ▼", ha='center', va='bottom', fontsize=8, color='black',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=0.2))

        # 모든 subplot의 격자선 스타일 설정
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
            ax.grid(True, color='lightgray', alpha=0.3, linestyle='-', linewidth=0.5)
            ax.yaxis.grid(True, color='lightgray', alpha=0.3, linestyle='-', linewidth=0.5)

        # x축 날짜 포맷 설정 (마지막 subplot에만 적용)
        ax8.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax8.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax8.tick_params(axis='x', rotation=45, labelsize=8)

        # 다른 subplot들의 x축 레이블 숨기기
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
            ax.tick_params(axis='x', labelbottom=False)
        
        # subplot 간격 및 레이아웃 조정 (상단 여백 증가)
        plt.subplots_adjust(
            left=0.1,    # 왼쪽 여백
            right=0.9,   # 오른쪽 여백
            bottom=0.05, # 아래쪽 여백
            top=0.92,    # 위쪽 여백 증가
            hspace=0.4   # subplot 간 간격 증가
        )
        
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='white')
            plt.close()
        else:
            plt.show()
    except Exception as e:
        print("HMA & 만트라 밴드 시각화 중 오류 발생:")
        print(f"예외 메시지: {str(e)}")
        print("스택 트레이스:")
        traceback.print_exc()

# --- 매수형 캔들 패턴 탐지 및 마킹 ---
def to_series(val, index):
    if isinstance(val, pd.Series):
        return val
    elif isinstance(val, pd.DataFrame):
        # DataFrame이면 첫 번째 컬럼만 Series로 변환
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

def detect_hammer(df, ticker=None):
    open_ = df['Open'] if 'Open' in df else df[('Open', ticker)]
    close_ = df['Close'] if 'Close' in df else df[('Close', ticker)]
    low_ = df['Low'] if 'Low' in df else df[('Low', ticker)]
    high_ = df['High'] if 'High' in df else df[('High', ticker)]
    open_ = to_series(open_, df.index)
    close_ = to_series(close_, df.index)
    low_ = to_series(low_, df.index)
    high_ = to_series(high_, df.index)
    if len(df) < 2:
        return pd.Series([False]*len(df), index=df.index)
    body = abs(close_ - open_)
    min_oc = pd.DataFrame({'open': open_, 'close': close_}).min(axis=1)
    max_oc = pd.DataFrame({'open': open_, 'close': close_}).max(axis=1)
    lower_shadow = min_oc - low_
    upper_shadow = high_ - max_oc
    return (lower_shadow > 2 * body) & (upper_shadow < body)

def detect_bullish_engulfing(df, ticker=None):
    open_ = df['Open'] if 'Open' in df else df[('Open', ticker)]
    close_ = df['Close'] if 'Close' in df else df[('Close', ticker)]
    prev_open = open_.shift(1)
    prev_close = close_.shift(1)
    return (prev_close < prev_open) & (close_ > open_) & (open_ < prev_close) & (close_ > prev_open)

def detect_bullish_harami(df, ticker=None):
    open_ = df['Open'] if 'Open' in df else df[('Open', ticker)]
    close_ = df['Close'] if 'Close' in df else df[('Close', ticker)]
    prev_open = open_.shift(1)
    prev_close = close_.shift(1)
    return (prev_close > prev_open) & (open_ > close_) & (open_ > prev_close) & (close_ < prev_open)

def detect_piercing_line(df, ticker=None):
    open_ = df['Open'] if 'Open' in df else df[('Open', ticker)]
    close_ = df['Close'] if 'Close' in df else df[('Close', ticker)]
    prev_open = open_.shift(1)
    prev_close = close_.shift(1)
    return (prev_close < prev_open) & (open_ < prev_close) & (close_ > (prev_open + prev_close) / 2) & (close_ < prev_open)

def detect_morning_star(df, ticker=None):
    open_ = df['Open'] if 'Open' in df else df[('Open', ticker)]
    close_ = df['Close'] if 'Close' in df else df[('Close', ticker)]
    prev_open = open_.shift(2)
    prev_close = close_.shift(2)
    star_open = open_.shift(1)
    star_close = close_.shift(1)
    return (
        (prev_close < prev_open) &
        (star_close < prev_close) & (star_open < prev_close) &
        (close_ > star_open) & (close_ > prev_open)
    )

def to_bool(val):
    if isinstance(val, pd.Series):
        return bool(val.iloc[0])
    return bool(val)
