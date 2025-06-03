import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import squarify
import os
import matplotlib
import platform
import matplotlib as mpl
from datetime import datetime

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
    return pd.read_csv(file_path, encoding='utf-8')

def fetch_market_data(tickers):
    """
    yfinance로 실시간 가격, 시가총액, 등락률 등 수집
    """
    data = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            data.append({
                'ticker': ticker,
                'price': info.get('regularMarketPrice'),
                'change': info.get('regularMarketChangePercent'),
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

def make_heatmap(portfolio, market_data, save_path=None):
    """
    트리맵(히트맵) 시각화
    """
    df = pd.merge(portfolio, market_data, on='ticker')
    df['eval_value'] = df['수량'] * df['price']
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
    plt.figure(figsize=(16, 9))
    ax = plt.gca()
    squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.8, ax=ax)
    plt.axis('off')
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ax.set_title(f'포트폴리오 히트맵 ({now_str})')

    # 컬러바(범례) 추가
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=norm)
    sm.set_array([])
    fig = plt.gcf()
    cbar = fig.colorbar(sm, orientation='horizontal', pad=0.05, aspect=40, ax=ax)
    cbar.set_label('등락률(%)')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"히트맵 저장 완료: {save_path}")
    else:
        plt.show()

def main():
    # 예시 입력 파일 경로
    portfolio_path = os.path.join(os.path.dirname(__file__), 'stock_portfolio.csv')
    portfolio = load_portfolio(portfolio_path)
    market_data = fetch_market_data(portfolio['ticker'].tolist())
    # 결과 저장 폴더 및 파일 경로
    output_dir = os.path.join(os.path.dirname(__file__), '../../output/heatmap')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'portfolio_heatmap.png')
    make_heatmap(portfolio, market_data, save_path=save_path)

if __name__ == '__main__':
    main() 