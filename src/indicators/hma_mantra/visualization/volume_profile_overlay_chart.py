"""
Volume Profile을 메인차트에 오버레이로 표시하는 시각화 모듈
"""

import mplfinance as mpf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
from matplotlib.legend_handler import HandlerTuple
from ..core import calculate_hma, calculate_mantra_bands, calculate_rsi, calculate_macd
from ..signals import get_hma_mantra_md_signals
from ..utils import get_available_font
import matplotlib.patches as mpatches
import pandas_datareader.data as web

def _find_pivots(series: pd.Series, left: int = 3, right: int = 3):
    """좌우 span을 기준으로 피벗 고점/저점을 탐지합니다.
    반환: (pivot_highs, pivot_lows) 각 리스트는 (index_pos, value) 튜플로 구성
    """
    values = series.values
    pivot_highs = []
    pivot_lows = []
    n = len(values)
    for i in range(left, n - right):
        window = values[i - left:i + right + 1]
        center = values[i]
        if center == np.max(window):
            pivot_highs.append((i, center))
        if center == np.min(window):
            pivot_lows.append((i, center))
    return pivot_highs, pivot_lows

def _detect_rsi_divergences(ohlcv_data: pd.DataFrame,
                            rsi_series: pd.Series,
                            pivot_span_left: int = 3,
                            pivot_span_right: int = 3,
                            valid_window_bars: int = 20,
                            include_hidden: bool = True):
    """RSI 기반 다이버전스(일반/히든) 탐지.
    반환: 리스트[dict] with keys: kind('BULL'|'BEAR'), subtype('regular'|'hidden'),
          i1,i2 (index positions), date1,date2, price1,price2
    """
    close = ohlcv_data['Close']
    pivot_highs_price, pivot_lows_price = _find_pivots(close, pivot_span_left, pivot_span_right)
    pivot_highs_rsi, pivot_lows_rsi = _find_pivots(rsi_series, pivot_span_left, pivot_span_right)

    # index position -> pivot value lookup for RSI
    highs_rsi_map = {i: v for i, v in pivot_highs_rsi}
    lows_rsi_map = {i: v for i, v in pivot_lows_rsi}

    divergences = []
    last_idx = len(close) - 1

    # 정배(regular) 상승: 가격 LL, RSI HL (저점 기준)
    for (i1, p1), (i2, p2) in zip(pivot_lows_price[:-1], pivot_lows_price[1:]):
        if i2 <= i1:
            continue
        if last_idx - i2 > valid_window_bars:
            continue
        # 대응하는 RSI 저점이 존재하는지 확인 (근접 인덱스 사용)
        if i1 in lows_rsi_map and i2 in lows_rsi_map:
            r1 = lows_rsi_map[i1]
            r2 = lows_rsi_map[i2]
            if p2 < p1 and r2 > r1:  # 가격 LL, RSI HL
                divergences.append({
                    'kind': 'BULL', 'subtype': 'regular',
                    'i1': i1, 'i2': i2,
                    'date1': ohlcv_data.index[i1], 'date2': ohlcv_data.index[i2],
                    'price1': p1, 'price2': p2
                })

    # 정배(regular) 하락: 가격 HH, RSI LH (고점 기준)
    for (i1, p1), (i2, p2) in zip(pivot_highs_price[:-1], pivot_highs_price[1:]):
        if i2 <= i1:
            continue
        if last_idx - i2 > valid_window_bars:
            continue
        if i1 in highs_rsi_map and i2 in highs_rsi_map:
            r1 = highs_rsi_map[i1]
            r2 = highs_rsi_map[i2]
            if p2 > p1 and r2 < r1:  # 가격 HH, RSI LH
                divergences.append({
                    'kind': 'BEAR', 'subtype': 'regular',
                    'i1': i1, 'i2': i2,
                    'date1': ohlcv_data.index[i1], 'date2': ohlcv_data.index[i2],
                    'price1': p1, 'price2': p2
                })

    if include_hidden:
        # 히든 상승: 가격 HL, RSI LL (저점 기준)
        for (i1, p1), (i2, p2) in zip(pivot_lows_price[:-1], pivot_lows_price[1:]):
            if i2 <= i1:
                continue
            if last_idx - i2 > valid_window_bars:
                continue
            if i1 in lows_rsi_map and i2 in lows_rsi_map:
                r1 = lows_rsi_map[i1]
                r2 = lows_rsi_map[i2]
                if p2 > p1 and r2 < r1:  # 가격 HL, RSI LL
                    divergences.append({
                        'kind': 'BULL', 'subtype': 'hidden',
                        'i1': i1, 'i2': i2,
                        'date1': ohlcv_data.index[i1], 'date2': ohlcv_data.index[i2],
                        'price1': p1, 'price2': p2
                    })

        # 히든 하락: 가격 LH, RSI HH (고점 기준)
        for (i1, p1), (i2, p2) in zip(pivot_highs_price[:-1], pivot_highs_price[1:]):
            if i2 <= i1:
                continue
            if last_idx - i2 > valid_window_bars:
                continue
            if i1 in highs_rsi_map and i2 in highs_rsi_map:
                r1 = highs_rsi_map[i1]
                r2 = highs_rsi_map[i2]
                if p2 < p1 and r2 > r1:  # 가격 LH, RSI HH
                    divergences.append({
                        'kind': 'BEAR', 'subtype': 'hidden',
                        'i1': i1, 'i2': i2,
                        'date1': ohlcv_data.index[i1], 'date2': ohlcv_data.index[i2],
                        'price1': p1, 'price2': p2
                    })

    return divergences

def _get_market_sentiment_summary(vix_value, naaim_value, pcr_value):
    """VIX, NAIIM, PCR을 종합하여 시장 심리 요약을 생성합니다."""
    if vix_value is None and naaim_value is None and pcr_value is None:
        return "시장 데이터 부족"
    
    summary_parts = []
    
    # VIX 해석
    if vix_value is not None:
        if vix_value < 20:
            vix_sentiment = "VIX:[G](안정)"
        elif vix_value < 30:
            vix_sentiment = "VIX:[O](중간)"
        else:
            vix_sentiment = "VIX:[R](불안)"
        summary_parts.append(vix_sentiment)
    
    # NAIIM 해석
    if naaim_value is not None:
        if naaim_value < 30:
            naaim_sentiment = "NAIIM:[R](보수)"
        elif naaim_value < 50:
            naaim_sentiment = "NAIIM:[O](중간)"
        else:
            naaim_sentiment = "NAIIM:[B](적극)"
        summary_parts.append(naaim_sentiment)
    
    # PCR 해석
    if pcr_value is not None:
        if pcr_value < 0.7:
            pcr_sentiment = "PCR:[G](매수)"
        elif pcr_value < 1.0:
            pcr_sentiment = "PCR:[O](균형)"
        else:
            pcr_sentiment = "PCR:[R](매도)"
        summary_parts.append(pcr_sentiment)
    
    # 종합 판단
    if len(summary_parts) >= 2:
        bullish_count = sum(1 for part in summary_parts if '[G]' in part or '[B]' in part)
        bearish_count = sum(1 for part in summary_parts if '[R]' in part)
        
        if bullish_count > bearish_count:
            overall_sentiment = " -> [UP] 매수우세"
        elif bearish_count > bullish_count:
            overall_sentiment = " -> [DOWN] 매도우세"
        else:
            overall_sentiment = " -> [EQ] 중립"
        
        summary_parts.append(overall_sentiment)
    
    return " | ".join(summary_parts)

def _get_strategy_guide(vix_value, naaim_value, pcr_value):
    """VIX, NAIIM, PCR 구간별 전략 가이드를 생성합니다."""
    strategy_parts = []
    
    # VIX 전략
    if vix_value is not None:
        if vix_value < 20:
            vix_strategy = "VIX<20: 매수기회(안정)"
        elif vix_value < 30:
            vix_strategy = "VIX 20-30: 관망(중간)"
        else:
            vix_strategy = "VIX>30: 방어(불안)"
        strategy_parts.append(vix_strategy)
    
    # NAIIM 전략
    if naaim_value is not None:
        if naaim_value < 30:
            naaim_strategy = "NAIIM<30: 매수(보수)"
        elif naaim_value < 50:
            naaim_strategy = "NAIIM 30-50: 중립(균형)"
        else:
            naaim_strategy = "NAIIM>50: 주의(적극)"
        strategy_parts.append(naaim_strategy)
    
    # PCR 전략
    if pcr_value is not None:
        if pcr_value < 0.7:
            pcr_strategy = "PCR<0.7: 매수(콜우세)"
        elif pcr_value < 1.0:
            pcr_strategy = "PCR 0.7-1.0: 중립(균형)"
        else:
            pcr_strategy = "PCR>1.0: 매도(풋우세)"
        strategy_parts.append(pcr_strategy)
    
    return " | ".join(strategy_parts)

def _detect_macd_divergences(ohlcv_data: pd.DataFrame,
                             macd_hist: pd.Series,
                             pivot_span_left: int = 3,
                             pivot_span_right: int = 3,
                             valid_window_bars: int = 20,
                             include_hidden: bool = True):
    """MACD 히스토그램 기반 다이버전스(일반/히든) 탐지.
    반환 구조는 _detect_rsi_divergences와 동일.
    """
    close = ohlcv_data['Close']
    pivot_highs_price, pivot_lows_price = _find_pivots(close, pivot_span_left, pivot_span_right)
    pivot_highs_hist, pivot_lows_hist = _find_pivots(macd_hist, pivot_span_left, pivot_span_right)

    highs_hist_map = {i: v for i, v in pivot_highs_hist}
    lows_hist_map = {i: v for i, v in pivot_lows_hist}

    divergences = []
    last_idx = len(close) - 1

    # 정배(regular) 상승: 가격 LL, 히스토그램 HL (저점 기준)
    for (i1, p1), (i2, p2) in zip(pivot_lows_price[:-1], pivot_lows_price[1:]):
        if i2 <= i1 or last_idx - i2 > valid_window_bars:
            continue
        if i1 in lows_hist_map and i2 in lows_hist_map:
            h1 = lows_hist_map[i1]
            h2 = lows_hist_map[i2]
            if p2 < p1 and h2 > h1:
                divergences.append({
                    'kind': 'BULL', 'subtype': 'regular',
                    'i1': i1, 'i2': i2,
                    'date1': ohlcv_data.index[i1], 'date2': ohlcv_data.index[i2],
                    'price1': p1, 'price2': p2
                })

    # 정배(regular) 하락: 가격 HH, 히스토그램 LH (고점 기준)
    for (i1, p1), (i2, p2) in zip(pivot_highs_price[:-1], pivot_highs_price[1:]):
        if i2 <= i1 or last_idx - i2 > valid_window_bars:
            continue
        if i1 in highs_hist_map and i2 in highs_hist_map:
            h1 = highs_hist_map[i1]
            h2 = highs_hist_map[i2]
            if p2 > p1 and h2 < h1:
                divergences.append({
                    'kind': 'BEAR', 'subtype': 'regular',
                    'i1': i1, 'i2': i2,
                    'date1': ohlcv_data.index[i1], 'date2': ohlcv_data.index[i2],
                    'price1': p1, 'price2': p2
                })

    if include_hidden:
        # 히든 상승: 가격 HL, 히스토그램 LL (저점 기준)
        for (i1, p1), (i2, p2) in zip(pivot_lows_price[:-1], pivot_lows_price[1:]):
            if i2 <= i1 or last_idx - i2 > valid_window_bars:
                continue
            if i1 in lows_hist_map and i2 in lows_hist_map:
                h1 = lows_hist_map[i1]
                h2 = lows_hist_map[i2]
                if p2 > p1 and h2 < h1:
                    divergences.append({
                        'kind': 'BULL', 'subtype': 'hidden',
                        'i1': i1, 'i2': i2,
                        'date1': ohlcv_data.index[i1], 'date2': ohlcv_data.index[i2],
                        'price1': p1, 'price2': p2
                    })

        # 히든 하락: 가격 LH, 히스토그램 HH (고점 기준)
        for (i1, p1), (i2, p2) in zip(pivot_highs_price[:-1], pivot_highs_price[1:]):
            if i2 <= i1 or last_idx - i2 > valid_window_bars:
                continue
            if i1 in highs_hist_map and i2 in highs_hist_map:
                h1 = highs_hist_map[i1]
                h2 = highs_hist_map[i2]
                if p2 < p1 and h2 > h1:
                    divergences.append({
                        'kind': 'BEAR', 'subtype': 'hidden',
                        'i1': i1, 'i2': i2,
                        'date1': ohlcv_data.index[i1], 'date2': ohlcv_data.index[i2],
                        'price1': p1, 'price2': p2
                    })

    return divergences

def get_market_data(start_date, end_date):
    """시장 데이터(VIX, TNX, DXY)를 가져옵니다."""
    import yfinance as yf
    from datetime import datetime, timedelta
    
    # 오늘 날짜 가져오기
    today = datetime.now().date()
    
    # VIX 지수
    vix = yf.download('^VIX', start=start_date, end=today)['Close']
    print(f"VIX 마지막 데이터 날짜: {vix.index[-1].strftime('%Y-%m-%d')}")
    print(f"오늘 날짜: {today.strftime('%Y-%m-%d')}")
    
    # 미국채 10년물 금리
    tnx = yf.download('^TNX', start=start_date, end=today)['Close']
    print(f"TNX 마지막 데이터 날짜: {tnx.index[-1].strftime('%Y-%m-%d')}")
    
    # 달러 인덱스
    dxy = yf.download('DX-Y.NYB', start=start_date, end=today)['Close']
    print(f"DXY 마지막 데이터 날짜: {dxy.index[-1].strftime('%Y-%m-%d')}")
    
    return vix, tnx, dxy

def calculate_bollinger_bands(data, window=20, num_std=2):
    """볼린저 밴드 계산"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return rolling_mean, upper_band, lower_band

def calculate_support_resistance(data, window=20):
    """20일 박스권 기준 지지선과 저항선 계산"""
    rolling_min = data['Low'].rolling(window=window).min()
    rolling_max = data['High'].rolling(window=window).max()
    current_support = rolling_min.iloc[-1]
    current_resistance = rolling_max.iloc[-1]
    return current_support, current_resistance

def calculate_volume_profile(ohlcv_data, num_bins=50):
    """Fixed Range Volume Profile 계산 (Net Volume 포함)"""
    # 가격 범위 설정
    price_min = ohlcv_data['Low'].min()
    price_max = ohlcv_data['High'].max()
    
    # 가격을 N개 구간으로 나누기
    price_bins = np.linspace(price_min, price_max, num_bins + 1)
    
    # 거래량 분포 계산
    volume_profile = []
    net_volume_profile = []  # Net Volume 추가
    volume_ratios = []  # 비율 추가
    
    for i in range(len(price_bins) - 1):
        bin_low = price_bins[i]
        bin_high = price_bins[i + 1]
        
        # 해당 가격 구간에 속하는 거래량 합계
        mask = (ohlcv_data['Low'] <= bin_high) & (ohlcv_data['High'] >= bin_low)
        total_volume = ohlcv_data.loc[mask, 'Volume'].sum()
        
        # Net Volume 계산 (상승일과 하락일 구분)
        up_mask = mask & (ohlcv_data['Close'] > ohlcv_data['Open'])
        down_mask = mask & (ohlcv_data['Close'] < ohlcv_data['Open'])
        
        up_volume = ohlcv_data.loc[up_mask, 'Volume'].sum()
        down_volume = ohlcv_data.loc[down_mask, 'Volume'].sum()
        net_volume = up_volume - down_volume
        
        volume_profile.append(total_volume)
        net_volume_profile.append(net_volume)
    
    # 전체 거래량 대비 비율 계산
    total_volume = sum(volume_profile)
    volume_ratios = [vol / total_volume * 100 for vol in volume_profile]
    
    # POC (Point of Control) 계산
    poc_idx = np.argmax(volume_profile)
    poc_price = price_bins[poc_idx]
    
    # Value Area 계산 (거래량 70% 구간)
    target_volume = total_volume * 0.7
    
    # POC를 중심으로 Value Area 확장
    value_area_prices = [poc_price]
    current_volume = volume_profile[poc_idx]
    
    left_idx = poc_idx - 1
    right_idx = poc_idx + 1
    
    while current_volume < target_volume and (left_idx >= 0 or right_idx < len(volume_profile)):
        left_vol = volume_profile[left_idx] if left_idx >= 0 else 0
        right_vol = volume_profile[right_idx] if right_idx < len(volume_profile) else 0
        
        if left_vol >= right_vol and left_idx >= 0:
            value_area_prices.append(price_bins[left_idx])
            current_volume += left_vol
            left_idx -= 1
        elif right_idx < len(volume_profile):
            value_area_prices.append(price_bins[right_idx])
            current_volume += right_vol
            right_idx += 1
        else:
            break
    
    value_area_min = min(value_area_prices)
    value_area_max = max(value_area_prices)
    
    return price_bins, volume_profile, net_volume_profile, volume_ratios, poc_price, value_area_min, value_area_max

 

def plot_main_chart_with_volume_profile_overlay(
    data: pd.DataFrame,
    ticker: str = None,
    save_path: str = None,
    current_price: float = None,
    rsi_divergence_window: int = 20,
    rsi_pivot_span: int = 3,
    include_hidden_divergence: bool = True,
):
    """Volume Profile이 메인차트에 오버레이된 차트"""
    # 폰트 설정
    plt.rcParams['font.family'] = 'AppleGothic'  # macOS용 한글 폰트
    plt.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지
    
    if isinstance(data.columns, pd.MultiIndex):
        ohlcv_data = data.xs(ticker, axis=1, level=1)
    else:
        ohlcv_data = data

    # 시장 데이터 가져오기
    start_date = ohlcv_data.index[0]
    end_date = ohlcv_data.index[-1]
    vix, tnx, dxy = get_market_data(start_date, end_date)
    
    # NAIIM 데이터 가져오기 (실제 데이터 우선, 없으면 시뮬레이션)
    try:
        from pathlib import Path
        
        naaim_file = Path("data_cache/naaim_data.csv")
        if naaim_file.exists():
            naaim_df = pd.read_csv(naaim_file)
            naaim_df['Date'] = pd.to_datetime(naaim_df['Date'])
            naaim_df.set_index('Date', inplace=True)
            naaim = naaim_df.loc[start_date:end_date]
            print(f"실제 NAIIM 데이터 로드 완료: {len(naaim)}개 데이터")
        else:
            naaim = None
            print("실제 NAIIM 데이터 파일이 없습니다.")
    except Exception as e:
        print(f"NAIIM 데이터 로드 오류: {e}")
        naaim = None
    
    # PCR 데이터 계산 (옵션이 있는 종목만)
    pcr = None
    if ticker and not ticker.endswith('.KS'):  # 한국 종목 제외
        try:
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            options = stock.options
            if options:
                nearest_expiry = options[0]
                calls = stock.option_chain(nearest_expiry).calls
                puts = stock.option_chain(nearest_expiry).puts
                
                total_call_volume = calls['volume'].sum()
                total_put_volume = puts['volume'].sum()
                
                if total_call_volume > 0:
                    pcr = total_put_volume / total_call_volume
                    print(f"{ticker} PCR 계산 완료: {pcr:.3f}")
        except Exception as e:
            print(f"{ticker} PCR 계산 오류: {e}")

    # 기술적 지표 계산
    hma = calculate_hma(ohlcv_data['Close'])
    upper_band, lower_band = calculate_mantra_bands(ohlcv_data['Close'])
    rsi3 = calculate_rsi(ohlcv_data['Close'], period=3)
    rsi14 = calculate_rsi(ohlcv_data['Close'], period=14)
    rsi50 = calculate_rsi(ohlcv_data['Close'], period=50)
    macd, macd_signal, hist = calculate_macd(ohlcv_data['Close'])
    
    # 볼린저 밴드 계산
    volume_ma = ohlcv_data['Volume'].rolling(window=20).mean()
    volume_std = ohlcv_data['Volume'].rolling(window=20).std()
    volume_upper = volume_ma + (volume_std * 2)
    
    # 볼린저 밴드 계산
    bb_ma, bb_upper, bb_lower = calculate_bollinger_bands(ohlcv_data['Close'])

    # 신호 생성
    trade_signals = get_hma_mantra_md_signals(ohlcv_data, ticker)

    # Volume Profile 계산
    price_bins, volume_profile, net_volume_profile, volume_ratios, poc_price, value_area_min, value_area_max = calculate_volume_profile(ohlcv_data)
    
    # 분석 기간이 1년 이상인 경우 최근 6개월 Volume Profile 추가 계산
    analysis_period_days = (end_date - start_date).days
    recent_6mo_data = None
    recent_price_bins = None
    recent_volume_profile = None
    recent_net_volume_profile = None
    recent_volume_ratios = None
    recent_poc_price = None
    recent_value_area_min = None
    recent_value_area_max = None
    
    if analysis_period_days >= 365:  # 1년 이상
        # 최근 6개월 데이터 추출
        recent_6mo_start = end_date - pd.Timedelta(days=180)
        recent_6mo_data = ohlcv_data[recent_6mo_start:end_date]
        
        # 최근 6개월 데이터가 충분한지 확인 (최소 30일 이상)
        if len(recent_6mo_data) >= 30:
            # 최근 6개월 Volume Profile 계산
            recent_price_bins, recent_volume_profile, recent_net_volume_profile, \
            recent_volume_ratios, recent_poc_price, recent_value_area_min, recent_value_area_max = \
                calculate_volume_profile(recent_6mo_data)

    # 시장 심리 종합해석 생성
    vix_final = None
    naaim_final = None
    
    if not vix.empty:
        vix_end_date = vix.index[vix.index <= end_date][-1] if len(vix.index[vix.index <= end_date]) > 0 else vix.index[-1]
        vix_final = float(vix.loc[vix_end_date].iloc[0])
        print(f"VIX 최종값: {vix_final}")
    
    if naaim is not None and not naaim.empty:
        naaim_end_date = naaim.index[naaim.index <= end_date][-1] if len(naaim.index[naaim.index <= end_date]) > 0 else naaim.index[-1]
        naaim_final = float(naaim.loc[naaim_end_date].iloc[0])
        print(f"NAIIM 최종값: {naaim_final}")
    
    print(f"PCR 값: {pcr}")
    market_summary = _get_market_sentiment_summary(vix_final, naaim_final, pcr)
    strategy_guide = _get_strategy_guide(vix_final, naaim_final, pcr)
    
    print(f"시장 심리 요약: {market_summary}")
    print(f"전략 가이드: {strategy_guide}")
    
    # 차트 생성 (4x1 레이아웃: 메인차트 + 거래량 + RSI + MACD)
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 1, height_ratios=[3, 1, 1, 1], figure=fig, hspace=0.1)
    
    # 메인 차트 (상단)
    ax_main = fig.add_subplot(gs[0, 0])
    
    # 차트 타이틀에 시장 심리 종합해석 추가
    title_text = f"{ticker} HMA Mantra Analysis"
    if market_summary and market_summary != "시장 데이터 부족":
        title_text += f" | {market_summary}"
    
    print(f"설정할 타이틀: {title_text}")
    
    # 메인 차트 타이틀은 간단하게 설정
    ax_main.set_title(f"{ticker} HMA Mantra Analysis", fontsize=14, fontweight='bold', pad=15)
    
    # 전체 차트 상단에 시장 심리 종합해석과 전략 가이드를 별도로 표시
    if market_summary and market_summary != "시장 데이터 부족":
        # 기본 타이틀 제거
        fig.suptitle("", fontsize=1, y=0.99)
        
        # 시장 심리 요약을 상단 중앙에 박스 형태로 표시
        ax_summary = fig.add_axes([0.1, 0.95, 0.8, 0.03])
        ax_summary.set_facecolor('lightblue')
        ax_summary.set_xlim(0, 1)
        ax_summary.set_ylim(0, 1)
        ax_summary.axis('off')
        
        # 시장 심리 요약 텍스트를 박스 안에 표시
        ax_summary.text(0.5, 0.5, market_summary, 
                       fontsize=12, fontweight='bold', ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', 
                                edgecolor='navy', linewidth=2))
        
        # 전략 가이드를 시장 심리 요약 아래에 표시
        if strategy_guide:
            ax_strategy = fig.add_axes([0.1, 0.91, 0.8, 0.03])
            ax_strategy.set_facecolor('lightyellow')
            ax_strategy.set_xlim(0, 1)
            ax_strategy.set_ylim(0, 1)
            ax_strategy.axis('off')
            
            # 전략 가이드 텍스트를 박스 안에 표시
            ax_strategy.text(0.5, 0.5, strategy_guide, 
                           fontsize=11, fontweight='bold', ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', 
                                    edgecolor='orange', linewidth=2))
    
    # 거래량 차트
    ax_volume = fig.add_subplot(gs[1, 0], sharex=ax_main)
    # RSI 차트
    ax_rsi = fig.add_subplot(gs[2, 0], sharex=ax_main)
    # MACD 차트
    ax_macd = fig.add_subplot(gs[3, 0], sharex=ax_main)

    # 메인 차트 설정 (투명도 높임)
    candlestick_ohlc(ax_main, 
                     [[mdates.date2num(date), o, h, l, c] for date, (o, h, l, c) in 
                      zip(ohlcv_data.index, ohlcv_data[['Open', 'High', 'Low', 'Close']].values)],
                     width=0.6, colorup='green', colordown='red', alpha=0.9)
    
    # 시장 지표들을 수평으로 우측 상단에 표시
    indicators = []
    
    # VIX 값 추가
    if not vix.empty:
        end_date = ohlcv_data.index[-1]
        vix_end_date = vix.index[vix.index <= end_date][-1] if len(vix.index[vix.index <= end_date]) > 0 else vix.index[-1]
        vix_value = float(vix.loc[vix_end_date].iloc[0])
        
        if vix_value < 20:
            vix_color = 'green'
        elif vix_value < 30:
            vix_color = 'orange'
        else:
            vix_color = 'red'
        
        indicators.append({
            'text': f'VIX: {vix_value:.2f}',
            'color': vix_color,
            'date': vix_end_date.strftime("%Y-%m-%d")
        })
    
    # NAIIM 값 추가
    if naaim is not None and not naaim.empty:
        try:
            naaim_end_date = naaim.index[naaim.index <= end_date][-1] if len(naaim.index[naaim.index <= end_date]) > 0 else naaim.index[-1]
            naaim_value = float(naaim.loc[naaim_end_date].iloc[0])
            
            if naaim_value < 30:
                naaim_color = 'red'
            elif naaim_value < 50:
                naaim_color = 'orange'
            else:
                naaim_color = 'blue'
            
            indicators.append({
                'text': f'NAIIM: {naaim_value:.1f}',
                'color': naaim_color,
                'date': naaim_end_date.strftime("%Y-%m-%d")
            })
        except Exception as e:
            print(f"NAIIM 표시 오류: {e}")
    
    # PCR 값 추가
    if pcr is not None:
        try:
            if pcr < 0.7:
                pcr_color = 'green'
            elif pcr < 1.0:
                pcr_color = 'orange'
            else:
                pcr_color = 'red'
            
            indicators.append({
                'text': f'PCR: {pcr:.3f}',
                'color': pcr_color,
                'date': None
            })
        except Exception as e:
            print(f"PCR 표시 오류: {e}")
    
    # 수평으로 지표들 표시 (전략 가이드와 겹치지 않도록 위치 조정)
    if indicators:
        # 지표들을 수평으로 배치 (간격을 0.15에서 0.12로 줄이고, 더 우측 상단에 배치)
        for i, indicator in enumerate(indicators):
            x_pos = 0.99 - (len(indicators) - 1 - i) * 0.12  # 더 우측에 붙이고 간격 좁힘
            y_pos = 0.88  # 전략 가이드와 겹치지 않도록 조정
            
            ax_main.text(x_pos, y_pos, indicator['text'], 
                        transform=ax_main.transAxes, fontsize=11, ha='right', va='top',
                        bbox=dict(facecolor=indicator['color'], alpha=0.8, edgecolor='black', pad=4),
                        color='white', fontweight='bold')

    # HMA와 만트라 밴드
    ax_main.plot(ohlcv_data.index, hma, color='blue', linewidth=1.5, label='HMA')
    ax_main.plot(ohlcv_data.index, upper_band, color='red', linewidth=1, linestyle='--', label='Upper Mantra')
    ax_main.plot(ohlcv_data.index, lower_band, color='green', linewidth=1, linestyle='--', label='Lower Mantra')
    
    # 가격 라인 추가
    ax_main.plot(ohlcv_data.index, ohlcv_data['Close'], color='black', linewidth=0.8, alpha=0.7, label='Price')
    
    # 만트라 밴드 영역 채우기
    ax_main.fill_between(ohlcv_data.index, hma, upper_band, color='red', alpha=0.1, label='상단 밴드 영역')
    ax_main.fill_between(ohlcv_data.index, lower_band, hma, color='green', alpha=0.1, label='하단 밴드 영역')
    
    # 볼린저 밴드 추가
    ax_main.plot(ohlcv_data.index, bb_ma, color='purple', linewidth=0.5, label='BB MA(20)')
    ax_main.plot(ohlcv_data.index, bb_upper, color='purple', linewidth=0.5, linestyle=':', label='BB Upper')
    ax_main.plot(ohlcv_data.index, bb_lower, color='purple', linewidth=0.5, linestyle=':', label='BB Lower')

    # 현재가, 지지선, 저항선 계산
    current_price = ohlcv_data['Close'].iloc[-1]
    support, resistance = calculate_support_resistance(ohlcv_data)
    
    # 수평선 및 가격 표시 (Volume Profile보다 더 왼쪽으로 조정)
    ax_main.axhline(y=current_price, color='black', linestyle=':', linewidth=0.8, alpha=0.5)
    # Volume Profile보다 더 왼쪽에 텍스트 배치
    text_x = ohlcv_data.index[0] - (ohlcv_data.index[-1] - ohlcv_data.index[0]) * 0.02
    ax_main.text(text_x, current_price, 
                f'현재가: {current_price:.2f}', 
                fontsize=8, ha='right', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', pad=2))
    
    ax_main.axhline(y=support, color='green', linestyle=':', linewidth=0.8, alpha=0.5)
    ax_main.text(text_x, support, f'지지선: {support:.2f}', 
                fontsize=8, ha='right', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='green', pad=2))
    
    ax_main.axhline(y=resistance, color='red', linestyle=':', linewidth=0.8, alpha=0.5)
    ax_main.text(text_x, resistance, f'저항선: {resistance:.2f}', 
                fontsize=8, ha='right', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', pad=2))

    # 매수/매도 신호 표시
    for signal in trade_signals:
        color = 'blue' if signal['type'] == 'BUY' else 'red'
        marker = '^' if signal['type'] == 'BUY' else 'v'
        is_buy = signal['type'] == 'BUY'
        
        # 신호 위치 설정
        if is_buy:
            y = lower_band[signal['date']] * 0.99
            signal_num = 'B1' if 'HMA 상향돌파' in signal.get('reason', '') else 'B2'
        else:
            y = upper_band[signal['date']] * 1.01
            signal_num = 'T1' if 'HMA 하향돌파' in signal.get('reason', '') else 'T2'
        
        # 신호 마커 표시
        ax_main.plot(signal['date'], y, marker=marker, color=color, 
                    markersize=10, markeredgecolor='black')
        
        # 날짜와 신호 번호 표시
        date_str = signal['date'].strftime('%m/%d')
        ax_main.text(signal['date'], y, f"{date_str}\n{signal_num}", 
                    rotation=45, fontsize=6, ha='right', va='top' if is_buy else 'bottom',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # 조건부 수직선/수평선 표시
    # MACD 골든크로스 구간 찾기
    macd_above_signal = (macd > macd_signal)
    macd_golden_zone = macd_above_signal.cumsum()
    macd_golden_mask = macd_golden_zone > 0

    # 조건에 맞는 날짜 찾기
    cond = (
        (ohlcv_data['Volume'] > volume_upper) &
        macd_golden_mask &
        (rsi14 >= 50)
    )
    cond_dates = ohlcv_data.index[cond]

    # 거래량 분위수 계산 (3분위)
    volume_33 = ohlcv_data['Volume'].quantile(0.33)
    volume_67 = ohlcv_data['Volume'].quantile(0.67)
    
    # 교차점을 저장할 리스트
    intersection_points = []
    
    for dt in cond_dates:
        # 수직선
        ax_main.axvline(dt, color='magenta', linestyle='--', alpha=0.7, linewidth=1.2, zorder=20)
        # 하단에 일자 텍스트 표시 (45도 기울임)
        ax_main.text(dt, ax_main.get_ylim()[0], dt.strftime('%Y-%m-%d'), fontsize=6, color='magenta',
                    ha='center', va='top', rotation=45,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5), zorder=25)
        
        # 서브플롯에도 동일 수직선 표시 (라벨 없음)
        ax_volume.axvline(dt, color='magenta', linestyle='--', alpha=0.7, linewidth=1.0, zorder=10)
        ax_rsi.axvline(dt, color='magenta', linestyle='--', alpha=0.7, linewidth=1.0, zorder=10)
        ax_macd.axvline(dt, color='magenta', linestyle='--', alpha=0.7, linewidth=1.0, zorder=10)
        
        # 종가 기준 수평선 (거래량에 따른 두께 적용)
        close = ohlcv_data.loc[dt, 'Close']
        open_ = ohlcv_data.loc[dt, 'Open']
        volume = ohlcv_data.loc[dt, 'Volume']
        
        # 거래량에 따른 두께 결정
        if volume >= volume_67:
            linewidth = 1.6  # 상위 (높은 거래량)
        elif volume >= volume_33:
            linewidth = 1.2  # 중위 (보통 거래량)
        else:
            linewidth = 0.8  # 하위 (낮은 거래량)
        
        if close >= open_:
            ax_main.axhline(close, color='lime', linestyle='-', linewidth=linewidth, alpha=0.8, xmin=0, xmax=1, zorder=21)
            ax_main.text(ohlcv_data.index[-1], close, f'{close:.2f}', fontsize=7, color='black', ha='left', va='center',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5), zorder=22)
        else:
            ax_main.axhline(close, color='red', linestyle='-', linewidth=linewidth, alpha=0.8, xmin=0, xmax=1, zorder=21)
            ax_main.text(ohlcv_data.index[-1], close, f'{close:.2f}', fontsize=7, color='black', ha='left', va='center',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5), zorder=22)
        
        # 교차점 찾기: 분홍색 수직선과 녹색/빨간 수평선의 교차점
        # 분홍색 수직선은 이미 그려져 있고, 수평선도 그려져 있음
        # 교차점은 분홍색 수직선(dt)과 해당 날짜의 종가 수평선(close)의 교차점
        
        if close >= open_:
            # 녹색 수평선과 분홍색 수직선의 교차점
            intersection_points.append({
                'date': dt,
                'price': close,  # 해당 날짜의 종가
                'type': 'green_vertical'
            })
        else:
            # 빨간 수평선과 분홍색 수직선의 교차점
            intersection_points.append({
                'date': dt,
                'price': close,  # 해당 날짜의 종가
                'type': 'red_vertical'
            })
    
    # 교차점에 X 마커 표시
    for point in intersection_points:
        if point['type'] == 'green_vertical':
            # 녹색 수평선과 분홍색 수직선의 교차점
            ax_main.plot(point['date'], point['price'], marker='x', color='black', 
                        markersize=5, markeredgewidth=2, zorder=30)
        elif point['type'] == 'red_vertical':
            # 빨간 수평선과 분홍색 수직선의 교차점
            ax_main.plot(point['date'], point['price'], marker='x', color='black', 
                        markersize=5, markeredgewidth=2, zorder=30)

    # Volume Profile 오버레이 (메인차트 우측에 반투명하게)
    # 메인차트의 X축 범위 가져오기
    main_xlim = ax_main.get_xlim()
    main_ylim = ax_main.get_ylim()
    
    # Volume Profile을 메인차트 좌측에 오버레이
    # Volume Profile의 너비를 메인차트의 10%로 설정
    overlay_width = (main_xlim[1] - main_xlim[0]) * 0.10
    
    # 좌측에 Volume Profile 배치
    overlay_start = main_xlim[0] + (main_xlim[1] - main_xlim[0]) * 0.05  # 좌측에서 약간 떨어진 위치
    
    # Volume Profile 정규화 (0~1 범위로)
    max_volume = max(volume_profile)
    normalized_volume = [v / max_volume for v in volume_profile]
    
    # Net Volume Profile 정규화
    max_net_volume = max(abs(min(net_volume_profile)), abs(max(net_volume_profile))) if net_volume_profile else 1
    normalized_net_volume = [v / max_net_volume for v in net_volume_profile]
    
    # Volume Profile 막대 그리기 (Net Volume 색상 적용)
    bin_heights = price_bins[1] - price_bins[0]
    for i, (price, vol, net_vol, ratio) in enumerate(zip(price_bins[:-1], normalized_volume, normalized_net_volume, volume_ratios)):
        bar_width = vol * overlay_width * 0.8  # 막대 너비를 거래량에 비례하게
        
        # Net Volume에 따른 색상 설정
        if net_vol > 0:
            color = 'green'  # 상승 압력
        elif net_vol < 0:
            color = 'red'    # 하락 압력
        else:
            color = 'gray'   # 중립
        
        # Volume Profile 막대 그리기 (투명도 낮춤)
        ax_main.barh(price, bar_width, height=bin_heights, left=overlay_start, 
                    alpha=0.4, color=color, zorder=10)
        
        # 비율 텍스트 표시 (주요 구간만)
        if ratio > 2.0:  # 2% 이상인 구간만 표시
            ax_main.text(overlay_start + bar_width + overlay_width * 0.05, price, 
                        f'{ratio:.1f}%', fontsize=6, color='black', ha='left', va='center',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1), zorder=11)
    
    # POC (Point of Control) 표시 (최신일자까지 연장)
    poc_xmin = (overlay_start - main_xlim[0]) / (main_xlim[1] - main_xlim[0])
    poc_xmax = 1.0  # 최신일자까지 연장
    ax_main.axhline(poc_price, color='red', linestyle='--', alpha=0.8, linewidth=2, 
                   xmin=poc_xmin, xmax=poc_xmax, zorder=15, label=f'POC: {poc_price:.2f}')
    
    # POC 가격 텍스트 표시
    ax_main.text(overlay_start + overlay_width * 0.5, poc_price, f'{poc_price:.2f}', 
                fontsize=8, color='red', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', pad=2), zorder=16)
    
    # Value Area 표시 (Volume Profile 영역에만)
    value_xmax = (overlay_start + overlay_width - main_xlim[0]) / (main_xlim[1] - main_xlim[0])
    ax_main.axhspan(value_area_min, value_area_max, alpha=0.1, color='green', 
                   xmin=poc_xmin, xmax=value_xmax, zorder=5, label=f'Value Area: {value_area_min:.2f}-{value_area_max:.2f}')
    
    

    # 최근 6개월 Volume Profile 표시 (1년 이상인 경우)
    if analysis_period_days >= 365 and recent_6mo_data is not None and len(recent_6mo_data) >= 30:
        # 중앙에 최근 6개월 Volume Profile 배치
        center_overlay_start = main_xlim[0] + (main_xlim[1] - main_xlim[0]) * 0.45  # 중앙
        center_overlay_width = (main_xlim[1] - main_xlim[0]) * 0.10  # 10% 너비
        
        # 최근 6개월 Volume Profile 정규화
        max_recent_volume = max(recent_volume_profile)
        normalized_recent_volume = [v / max_recent_volume for v in recent_volume_profile]
        
        # 최근 6개월 Net Volume Profile 정규화
        max_recent_net_volume = max(abs(min(recent_net_volume_profile)), abs(max(recent_net_volume_profile))) if recent_net_volume_profile else 1
        normalized_recent_net_volume = [v / max_recent_net_volume for v in recent_net_volume_profile]
        
        # 최근 6개월 Volume Profile 막대 그리기
        recent_bin_heights = recent_price_bins[1] - recent_price_bins[0]
        for i, (price, vol, net_vol, ratio) in enumerate(zip(recent_price_bins[:-1], 
                                                            normalized_recent_volume, 
                                                            normalized_recent_net_volume, 
                                                            recent_volume_ratios)):
            bar_width = vol * center_overlay_width * 0.8  # 막대 너비를 거래량에 비례하게
            
            # 색상 설정 (연한 색상으로 구분)
            if net_vol > 0:
                color = 'lightgreen'  # 연한 녹색
            elif net_vol < 0:
                color = 'lightcoral'  # 연한 빨강
            else:
                color = 'lightgray'   # 연한 회색
            
            # 중앙에 막대 그리기 (투명도 낮춤)
            ax_main.barh(price, bar_width, height=recent_bin_heights, left=center_overlay_start, 
                        alpha=0.3, color=color, zorder=8)
        
        # 최근 6개월 POC (검은색 점선) - 우측 끝까지 연장
        recent_poc_xmin = (center_overlay_start - main_xlim[0]) / (main_xlim[1] - main_xlim[0])
        recent_poc_xmax = 1.0  # 우측 끝까지 연장
        ax_main.axhline(recent_poc_price, color='black', linestyle=':', alpha=0.8, linewidth=2, 
                       xmin=recent_poc_xmin, xmax=recent_poc_xmax, zorder=14, 
                       label=f'최근 6개월 POC: {recent_poc_price:.2f}')
        
        # 최근 6개월 POC 가격 텍스트 표시
        ax_main.text(center_overlay_start + center_overlay_width * 0.5, recent_poc_price, 
                    f'{recent_poc_price:.2f}', fontsize=8, color='black', ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', pad=2), zorder=16)
        
        # 최근 6개월 Value Area 표시 (Volume Profile 영역에만)
        recent_value_xmax = (center_overlay_start + center_overlay_width - main_xlim[0]) / (main_xlim[1] - main_xlim[0])
        ax_main.axhspan(recent_value_area_min, recent_value_area_max, alpha=0.05, color='blue', 
                       xmin=recent_poc_xmin, xmax=recent_value_xmax, zorder=4, 
                       label=f'최근 6개월 Value Area: {recent_value_area_min:.2f}-{recent_value_area_max:.2f}')
    
    # Net Volume Profile 범례 추가 (투명도 조정)
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.4, label='전체 기간 상승 압력'),
        plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.4, label='전체 기간 하락 압력'),
        plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.4, label='전체 기간 중립')
    ]
    
    # 최근 6개월 범례 추가 (1년 이상인 경우)
    if analysis_period_days >= 365 and recent_6mo_data is not None and len(recent_6mo_data) >= 30:
        legend_elements.extend([
            plt.Rectangle((0, 0), 1, 1, facecolor='lightgreen', alpha=0.3, label='최근 6개월 상승 압력'),
            plt.Rectangle((0, 0), 1, 1, facecolor='lightcoral', alpha=0.3, label='최근 6개월 하락 압력'),
            plt.Line2D([], [], color='black', linestyle=':', linewidth=2, label='최근 6개월 POC')
        ])
    
    # 다이버전스 범례 추가 (RSI & MACD)
    legend_elements.extend([
        # RSI Regular/Hidden
        plt.Line2D([], [], color='green', linestyle='-', linewidth=1.5, marker='^', label='Bull Div (RSI)'),
        plt.Line2D([], [], color='red', linestyle='-', linewidth=1.5, marker='v', label='Bear Div (RSI)'),
        plt.Line2D([], [], color='green', linestyle=':', linewidth=1.5, marker='^', label='Bull Div (RSI, Hidden)'),
        plt.Line2D([], [], color='red', linestyle=':', linewidth=1.5, marker='v', label='Bear Div (RSI, Hidden)'),
        # MACD Regular/Hidden
        plt.Line2D([], [], color='forestgreen', linestyle='-', linewidth=1.2, marker='^', label='Bull Div (MACD)'),
        plt.Line2D([], [], color='darkred', linestyle='-', linewidth=1.2, marker='v', label='Bear Div (MACD)'),
        plt.Line2D([], [], color='forestgreen', linestyle=':', linewidth=1.2, marker='^', label='Bull Div (MACD, Hidden)'),
        plt.Line2D([], [], color='darkred', linestyle=':', linewidth=1.2, marker='v', label='Bear Div (MACD, Hidden)'),
    ])
    
    ax_main.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95), fontsize=8)

    # 거래량 차트 표시
    # 거래량 막대 (양봉/음봉 구분)
    colors = ['green' if close >= open_ else 'red' for close, open_ in zip(ohlcv_data['Close'], ohlcv_data['Open'])]
    ax_volume.bar(ohlcv_data.index, ohlcv_data['Volume'], color=colors, alpha=0.7, width=0.8)
    
    # 거래량 이동평균선
    volume_ma_5 = ohlcv_data['Volume'].rolling(window=5).mean()
    volume_ma_20 = ohlcv_data['Volume'].rolling(window=20).mean()
    ax_volume.plot(ohlcv_data.index, volume_ma_5, color='orange', linewidth=1, label='Volume MA(5)')
    ax_volume.plot(ohlcv_data.index, volume_ma_20, color='blue', linewidth=1, label='Volume MA(20)')
    
    # 볼륨 볼린저 밴드
    ax_volume.plot(ohlcv_data.index, volume_upper, color='red', linewidth=1, linestyle='--', label='Volume BB Upper')
    
    # 거래량 차트 설정
    ax_volume.set_title('Volume')
    ax_volume.set_ylabel('Volume')
    ax_volume.legend(fontsize=8)
    ax_volume.grid(True, alpha=0.3)
    
    # RSI 차트 표시
    ax_rsi.plot(ohlcv_data.index, rsi14, color='tab:blue', linewidth=1.2, label='RSI(14)')
    # 보조로 RSI(3) 얇게 표시
    ax_rsi.plot(ohlcv_data.index, rsi3, color='tab:orange', linewidth=0.8, alpha=0.6, label='RSI(3)')
    ax_rsi.axhline(70, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
    ax_rsi.axhline(50, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    ax_rsi.axhline(30, color='green', linestyle='--', linewidth=0.8, alpha=0.6)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_title('RSI')
    ax_rsi.set_ylabel('RSI')
    ax_rsi.legend(fontsize=8, loc='upper left')
    ax_rsi.grid(True, alpha=0.3)

    # MACD 차트 표시
    macd_colors = ['green' if v >= 0 else 'red' for v in hist]
    ax_macd.bar(ohlcv_data.index, hist, color=macd_colors, alpha=0.5, width=0.8, label='Histogram')
    ax_macd.plot(ohlcv_data.index, macd, color='tab:blue', linewidth=1.2, label='MACD')
    ax_macd.plot(ohlcv_data.index, macd_signal, color='tab:orange', linewidth=1.0, label='Signal')
    ax_macd.axhline(0, color='black', linewidth=0.8, alpha=0.6)
    ax_macd.set_title('MACD (12,26,9)')
    ax_macd.set_ylabel('MACD')
    ax_macd.legend(fontsize=8, loc='upper left')
    ax_macd.grid(True, alpha=0.3)

    # =====================
    # RSI 다이버전스 탐지/표시 (메인차트에만)
    # =====================
    try:
        rsi_divs = _detect_rsi_divergences(
            ohlcv_data,
            rsi14,
            pivot_span_left=rsi_pivot_span,
            pivot_span_right=rsi_pivot_span,
            valid_window_bars=rsi_divergence_window,
            include_hidden=include_hidden_divergence,
        )

        for d in rsi_divs:
            date1, date2 = d['date1'], d['date2']
            p1, p2 = d['price1'], d['price2']
            if d['kind'] == 'BULL':
                color = 'green'
                marker = '^'
                y_text_offset = -0.01
                label = 'Bull Div (RSI)' if d['subtype'] == 'regular' else 'Bull Div (RSI, Hidden)'
                linestyle = '-' if d['subtype'] == 'regular' else ':'
            else:
                color = 'red'
                marker = 'v'
                y_text_offset = 0.01
                label = 'Bear Div (RSI)' if d['subtype'] == 'regular' else 'Bear Div (RSI, Hidden)'
                linestyle = '-' if d['subtype'] == 'regular' else ':'

            # 가격 피벗을 선으로 연결
            ax_main.plot([date1, date2], [p1, p2], color=color, linestyle=linestyle, linewidth=1.5, alpha=0.9, zorder=35)
            # 시그널 마커 (두 번째 피벗 위치)
            ax_main.plot(date2, p2, marker=marker, color=color, markersize=9, markeredgecolor='black', zorder=36)
            # 라벨(두 번째 피벗 상/하단에 배치)
            ylim = ax_main.get_ylim()
            y_offset = (ylim[1] - ylim[0]) * y_text_offset
            ax_main.text(date2, p2 + y_offset, label, fontsize=7, color=color,
                         ha='left', va='bottom' if d['kind'] == 'BULL' else 'top',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1), zorder=37)
    except Exception:
        pass

    # =====================
    # MACD 히스토그램 다이버전스 탐지/표시 (메인차트에만)
    # =====================
    try:
        macd_divs = _detect_macd_divergences(
            ohlcv_data,
            hist,
            pivot_span_left=rsi_pivot_span,
            pivot_span_right=rsi_pivot_span,
            valid_window_bars=rsi_divergence_window,
            include_hidden=include_hidden_divergence,
        )

        for d in macd_divs:
            date1, date2 = d['date1'], d['date2']
            p1, p2 = d['price1'], d['price2']
            if d['kind'] == 'BULL':
                color = 'forestgreen'
                marker = '^'
                y_text_offset = -0.015
                label = 'Bull Div (MACD)' if d['subtype'] == 'regular' else 'Bull Div (MACD, Hidden)'
                linestyle = '-' if d['subtype'] == 'regular' else ':'
            else:
                color = 'darkred'
                marker = 'v'
                y_text_offset = 0.015
                label = 'Bear Div (MACD)' if d['subtype'] == 'regular' else 'Bear Div (MACD, Hidden)'
                linestyle = '-' if d['subtype'] == 'regular' else ':'

            ax_main.plot([date1, date2], [p1, p2], color=color, linestyle=linestyle, linewidth=1.2, alpha=0.85, zorder=34)
            ax_main.plot(date2, p2, marker=marker, color=color, markersize=8, markeredgecolor='black', zorder=35)
            ylim = ax_main.get_ylim()
            y_offset = (ylim[1] - ylim[0]) * y_text_offset
            ax_main.text(date2, p2 + y_offset, label, fontsize=7, color=color,
                         ha='left', va='bottom' if d['kind'] == 'BULL' else 'top',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1), zorder=36)
    except Exception:
        pass

    # x축 날짜 포맷 설정
    ax_volume.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_volume.tick_params(axis='x', rotation=45)

    # 메인차트 설정
    title = f'{ticker} - Main Chart with Volume Profile Overlay\n'
    title += f'전체 기간: {start_date.strftime("%Y-%m-%d")} ~ {end_date.strftime("%Y-%m-%d")}'
    if analysis_period_days >= 365 and recent_6mo_data is not None and len(recent_6mo_data) >= 30:
        recent_6mo_start = end_date - pd.Timedelta(days=180)
        title += f'\n최근 6개월: {recent_6mo_start.strftime("%Y-%m-%d")} ~ {end_date.strftime("%Y-%m-%d")}'
    
    ax_main.set_title(title)
    ax_main.set_ylabel('Price')
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc='upper left', fontsize=8)
    
    # x축 날짜 포맷 설정
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_main.tick_params(axis='x', rotation=45)

    # 레이아웃 조정
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.06, hspace=0.12)

    # 저장 또는 표시
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show() 