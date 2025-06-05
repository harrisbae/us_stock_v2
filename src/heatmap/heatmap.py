import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import squarify
import os
import matplotlib
import platform
import matplotlib as mpl
from datetime import datetime
import plotly.express as px
import sys
import plotly.graph_objects as go
import numpy as np

# 한글 폰트 설정
if platform.system() == 'Darwin':  # macOS
    matplotlib.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':  # Windows
    matplotlib.rc('font', family='Malgun Gothic')
else:  # Linux
    matplotlib.rc('font', family='NanumGothic')
matplotlib.rc('axes', unicode_minus=False)

def load_portfolio(file_path):
    """
    CSV 파일에서 보유 주식 목록을 불러옵니다.
    컬럼 예시: ticker,수량,매입가
    """
    return pd.read_csv(file_path, encoding='utf-8', comment='#')

def fetch_market_data(tickers, period='day'):
    """
    yfinance로 실시간 가격, 시가총액, 등락률 등 수집
    period: 'day', 'week', 'month' 중 하나
    """
    data = []
    for ticker in tickers:
        try:
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info
            price = info.get('regularMarketPrice')
            # 등락률 계산
            if period == 'day':
                change = info.get('regularMarketChangePercent')
            else:
                # 기간별 등락률 계산
                if period == 'week':
                    hist = yf_ticker.history(period='7d')
                elif period == 'month':
                    hist = yf_ticker.history(period='1mo')
                else:
                    hist = yf_ticker.history(period='1d')
                if len(hist) > 1:
                    start_price = hist['Close'][0]
                    end_price = hist['Close'][-1]
                    change = ((end_price - start_price) / start_price) * 100
                else:
                    change = None
            data.append({
                'ticker': ticker,
                'price': price,
                'change': change,
                'marketCap': info.get('marketCap'),
                'sector': info.get('sector', 'N/A')
            })
        except Exception as e:
            print(f"{ticker} 데이터 수집 실패: {e}")
            data.append({
                'ticker': ticker,
                'price': None,
                'change': None,
                'marketCap': None,
                'sector': 'N/A'
            })
    return pd.DataFrame(data)

def make_heatmap(portfolio, market_data, save_path=None, period_kr='당일'):
    """
    트리맵(히트맵) 시각화
    """
    df = pd.merge(portfolio, market_data, on='ticker')
    df['eval_value'] = df['수량'] * df['price']
    # 0 이하 영역 제거 (squarify 오류 방지)
    df = df[df['eval_value'] > 0]
    sizes = df['eval_value'].fillna(0)
    labels = [
        f"{row['ticker']}\n{row['change']:+.2f}%" if pd.notnull(row['change']) else f"{row['ticker']}\nN/A"
        for _, row in df.iterrows()
    ]
    # 등락률이 모두 None이면 0으로 처리
    if df['change'].isnull().all():
        norm_change = [0.5] * len(df)
        vmin, vmax = 0, 1
    else:
        vmin, vmax = df['change'].min(), df['change'].max()
        norm_change = (df['change'].fillna(0) - vmin) / (vmax - vmin + 1e-6)
    colors = plt.cm.RdYlGn(norm_change)
    fig = plt.figure(figsize=(19.2, 10.8))  # 1920x1080 px at dpi=100
    ax = plt.gca()
    # 히트맵을 위에서 20% 아래로 이동
    fig.subplots_adjust(top=0.8)
    squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.8, ax=ax, linewidth=2, edgecolor='black')
    plt.axis('off')
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ax.set_title(f'포트폴리오 히트맵 ({period_kr} 등락률, {now_str})')
    # 컬러바(범례) 추가
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=norm)
    sm.set_array([])
    fig = plt.gcf()
    cbar = fig.colorbar(sm, orientation='horizontal', pad=0.05, aspect=40, ax=ax)
    cbar.set_label('등락률(%)')
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"히트맵 저장 완료: {save_path}")
    else:
        plt.show()

def make_heatmap_plotly(portfolio, market_data, save_path=None, period_kr='당일'):
    # 모든 주요 섹터 리스트 정의 (필요시 추가/수정 가능)
    all_sectors = [
        'Technology', 'Financial', 'Healthcare', 'Consumer Cyclical', 'Consumer Defensive',
        'Communication Services', 'Industrials', 'Utilities', 'Energy', 'Basic Materials', 'Real Estate'
    ]
    df = pd.merge(portfolio, market_data, on='ticker')
    df['eval_value'] = df['수량'] * df['price']
    df['등락률'] = df['change']
    df['등락률_str'] = df['등락률'].apply(lambda x: f"{x:+.2f}%" if pd.notnull(x) else "N/A")
    # 없는 섹터도 빈 ticker로 추가
    existing_sectors = set(df['sector'].dropna())
    for sector in all_sectors:
        if sector not in existing_sectors:
            df = pd.concat([df, pd.DataFrame([{
                'sector': sector, 'ticker': None, 'eval_value': 0, '등락률': None, '등락률_str': ''
            }])], ignore_index=True)
    # sector가 N/A인 경우 ETC로 표기
    df['sector'] = df['sector'].replace('N/A', 'ETC')
    fig = px.treemap(
        df,
        path=['sector', 'ticker'],
        values='eval_value',
        color='등락률',
        color_continuous_scale='RdYlGn',
        title='포트폴리오 히트맵 (섹터별 그룹)',
        custom_data=['등락률_str']
    )
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fig.update_traces(
        texttemplate='%{label}<br>%{value:,.0f}<br>%{customdata[0]}',
        textfont_size=36,
        textfont_color='black',
        textposition='middle center',
        marker_line_width=2,
        marker_line_color='black',
        selector=dict(type='treemap')
    )
    fig.update_layout(
        title={
            'text': f'포트폴리오 히트맵 (섹터별 그룹, {period_kr} 등락률) - {now_str}',
            'y': 0.99,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {
                'size': 62,
                'color': 'black'
            }
        },
        margin=dict(t=210, l=100, r=100, b=50)
    )
    if save_path:
        fig.write_image(save_path, width=3840, height=2160)
        print(f"plotly 히트맵 저장 완료: {save_path}")
    else:
        fig.show()

def main():
    # 예시 입력 파일 경로
    portfolio_path = os.path.join(os.path.dirname(__file__), 'stock_portfolio.csv')
    portfolio = load_portfolio(portfolio_path)
    # 기간 인자 받기
    if len(sys.argv) > 1:
        period = sys.argv[1]
    else:
        period = 'day'
    period_map = {'day': '당일', 'week': '주간', 'month': '월간'}
    period_kr = period_map.get(period, '당일')
    market_data = fetch_market_data(portfolio['ticker'].tolist(), period=period)
    # 결과 저장 폴더 및 파일 경로
    output_dir = os.path.join(os.path.dirname(__file__), '../../output/heatmap')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'portfolio_heatmap.png')
    make_heatmap(portfolio, market_data, save_path=save_path, period_kr=period_kr)
    # plotly 버전도 함께 저장
    plotly_save_path = os.path.join(output_dir, 'portfolio_heatmap_plotly.png')
    make_heatmap_plotly(portfolio, market_data, save_path=plotly_save_path, period_kr=period_kr)

if __name__ == '__main__':
    main() 