import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import mplfinance as mpf
from datetime import datetime, timedelta
import os

# 한글 폰트 설정 (Mac: AppleGothic, Windows: Malgun Gothic, Linux: NanumGothic)
import matplotlib
import platform
if platform.system() == 'Darwin':
    matplotlib.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    matplotlib.rc('font', family='Malgun Gothic')
else:
    matplotlib.rc('font', family='NanumGothic')
matplotlib.rcParams['axes.unicode_minus'] = False

def get_financial_data(ticker, period="1y"):
    """
    기업의 재무 데이터를 가져옵니다.
    
    Args:
        ticker (str): 주식 심볼
        period (str): 데이터 기간 (1y, 2y, 5y 등)
    
    Returns:
        dict: 재무 지표 데이터
    """
    try:
        # yfinance Ticker 객체 생성
        stock = yf.Ticker(ticker)
        
        # 주가 데이터 가져오기
        price_data = stock.history(period=period)
        
        if price_data.empty:
            return {}, price_data
        
        # 재무제표 데이터 가져오기
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        
        # 재무 지표 계산
        financial_metrics = {}
        
        # 기본 정보 가져오기
        shares_outstanding = stock.info.get('sharesOutstanding', 0)
        
        # EPS (Earnings Per Share) - 분기별
        if not financials.empty and 'Net Income' in financials.index and shares_outstanding > 0:
            net_income = financials.loc['Net Income']
            eps = net_income / shares_outstanding
            financial_metrics['EPS'] = eps
        
        # BPS (Book Value Per Share) - 분기별
        # 가능한 주당순자산 관련 컬럼들
        equity_columns = ['Total Stockholder Equity', 'Total Equity', 'Stockholders Equity', 'Shareholders Equity']
        equity_column = None
        for col in equity_columns:
            if col in balance_sheet.index:
                equity_column = col
                break
        
        if not balance_sheet.empty and equity_column and shares_outstanding > 0:
            book_value = balance_sheet.loc[equity_column]
            bps = book_value / shares_outstanding
            financial_metrics['BPS'] = bps
        

        
        return financial_metrics, price_data
        
    except Exception as e:
        print(f"재무 데이터 가져오기 오류: {e}")
        return {}, pd.DataFrame()

def plot_main_chart_with_financial_analysis(data, financial_metrics, ticker, save_path):
    """
    메인차트와 재무 분석 subplot을 함께 출력합니다.
    
    Args:
        data (pd.DataFrame): 주가 데이터
        financial_metrics (dict): 재무 지표 데이터
        ticker (str): 주식 심볼
        save_path (str): 저장 경로
    """
    # 차트 생성 (EPS만 표시)
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(1, 1, figure=fig, hspace=0.15, wspace=0.1)
    
    # EPS 차트
    ax_eps = fig.add_subplot(gs[0, 0])
    
    if 'EPS' in financial_metrics and not financial_metrics['EPS'].empty:
        # 데이터 정리 (NaN 제거)
        clean_data = financial_metrics['EPS'].dropna()
        
        if not clean_data.empty:
            # EPS 데이터를 왼쪽 Y축에 표시
            ax_eps.plot(range(len(clean_data)), clean_data.values, color='red', linewidth=2, marker='o', label='EPS')
            ax_eps.set_ylabel('EPS ($)', color='red', fontsize=12)
            ax_eps.tick_params(axis='y', labelcolor='red')
            ax_eps.grid(True, alpha=0.3)
            
            # 주가 데이터를 오른쪽 Y축에 표시
            ax_price = ax_eps.twinx()
            
            # EPS 데이터와 매칭되는 주가 데이터 찾기
            price_data_matched = []
            for eps_date in clean_data.index:
                # 타임존 문제 해결: EPS 날짜를 타임존이 없는 형태로 변환
                eps_date_naive = eps_date.tz_localize(None) if eps_date.tz is not None else eps_date
                
                # 주가 데이터의 인덱스도 타임존 제거
                data_naive = data.copy()
                data_naive.index = data_naive.index.tz_localize(None)
                
                # EPS 날짜 이전의 가장 가까운 주가 찾기
                price_before = data_naive[data_naive.index <= eps_date_naive]
                if not price_before.empty:
                    price_data_matched.append(price_before['Close'].iloc[-1])
                else:
                    price_data_matched.append(np.nan)
            
            ax_price.plot(range(len(clean_data)), price_data_matched, color='blue', linewidth=2, marker='s', label='주가')
            ax_price.set_ylabel('주가 ($)', color='blue', fontsize=12)
            ax_price.tick_params(axis='y', labelcolor='blue')
            
            # BPS 데이터 추가 (같은 Y축에 다른 색상으로)
            if 'BPS' in financial_metrics and not financial_metrics['BPS'].empty:
                bps_data = financial_metrics['BPS'].dropna()
                # EPS와 같은 날짜 범위에 맞춰 BPS 데이터 필터링
                bps_matched = []
                for eps_date in clean_data.index:
                    bps_before = bps_data[bps_data.index <= eps_date]
                    if not bps_before.empty:
                        bps_matched.append(bps_before.iloc[-1])
                    else:
                        bps_matched.append(np.nan)
                
                ax_eps.plot(range(len(clean_data)), bps_matched, color='green', linewidth=2, marker='^', label='BPS')
            
            # 제목 설정
            ax_eps.set_title(f'{ticker} EPS, BPS 및 주가 추이', fontsize=14)
            ax_eps.set_xlabel('분기', fontsize=12)
            
            # X축에 분기 표시
            if len(clean_data) > 0:
                quarters = [d.strftime('%Y-Q%q') if hasattr(d, 'strftime') else str(d)[:7] for d in clean_data.index]
                ax_eps.set_xticks(range(len(clean_data)))
                ax_eps.set_xticklabels(quarters, rotation=45, fontsize=10)
            
            # 범례 추가
            lines1, labels1 = ax_eps.get_legend_handles_labels()
            lines2, labels2 = ax_price.get_legend_handles_labels()
            ax_eps.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
        else:
            ax_eps.text(0.5, 0.5, 'EPS\n데이터 없음', ha='center', va='center', 
                       transform=ax_eps.transAxes, fontsize=12)
            ax_eps.set_title('EPS', fontsize=12)
    else:
        ax_eps.text(0.5, 0.5, 'EPS\n데이터 없음', ha='center', va='center', 
                   transform=ax_eps.transAxes, fontsize=12)
        ax_eps.set_title('EPS', fontsize=12)
    
    # 레이아웃 조정
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.08, hspace=0.15, wspace=0.1)
    
    # 차트 저장
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"재무 분석 차트 저장 완료: {save_path}")

def calculate_financial_correlation(price_data, financial_metrics):
    """
    주가와 재무 지표 간의 상관관계를 계산합니다.
    
    Args:
        price_data (pd.DataFrame): 주가 데이터
        financial_metrics (dict): 재무 지표 데이터
    
    Returns:
        dict: 상관관계 결과
    """
    correlations = {}
    price_returns = price_data['Close'].pct_change().dropna()
    
    for indicator, data in financial_metrics.items():
        if not data.empty:
            # 재무 데이터와 주가 데이터의 공통 기간 찾기
            common_dates = data.index.intersection(price_returns.index)
            if len(common_dates) > 10:  # 최소 10개 데이터 포인트 필요
                indicator_data = data.loc[common_dates]
                price_data_common = price_returns.loc[common_dates]
                
                # 상관관계 계산
                correlation = indicator_data.corr(price_data_common)
                correlations[indicator] = correlation
    
    return correlations 