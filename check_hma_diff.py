import yfinance as yf
from src.indicators.hma_mantra import calculate_hma

df = yf.download('XLE', period='130d')
hma = calculate_hma(df['Close'])

target_dates = ['2025-04-30', '2025-05-05', '2025-05-13']
for target_date in target_dates:
    # 실제 존재하는 날짜로 변환 (거래일만)
    if target_date not in df.index.strftime('%Y-%m-%d'):
        print(f"{target_date} is not a trading day in the data.")
        continue
    idx = df.index.get_indexer([target_date])[0]
    prev_idx = idx - 1
    prev_date = df.index[prev_idx]
    hma_now = float(hma.loc[target_date])
    hma_prev = float(hma.loc[prev_date])
    print(f'{target_date} | HMA: {hma_now:.4f} | prev HMA({prev_date.date()}): {hma_prev:.4f} | abs diff: {abs(hma_now-hma_prev):.4f}') 