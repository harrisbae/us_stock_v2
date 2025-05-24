import pandas as pd
from src.indicators.hma_mantra import calculate_hma, calculate_mantra_bands, calculate_rsi

df = pd.read_csv('data/TSLA.csv', index_col=0, parse_dates=True)
hma = calculate_hma(df['Close'])
upper, lower = calculate_mantra_bands(df['Close'])
rsi3 = calculate_rsi(df['Close'], 3)
rsi14 = calculate_rsi(df['Close'], 14)

d = '2025-05-07'
d_prev = df.index[df.index.get_loc(d)-1].strftime('%Y-%m-%d')

print(f'[{d_prev}]')
print('  prev_close:', df.loc[d_prev, 'Close'])
print('  prev_HMA:', hma.loc[d_prev])
print('  prev_LowerBand:', lower.loc[d_prev])

print(f'[{d}]')
print('  close:', df.loc[d, 'Close'])
print('  HMA:', hma.loc[d])
print('  LowerBand:', lower.loc[d])
print('  rsi3:', rsi3.loc[d])
print('  rsi14:', rsi14.loc[d])

# B1 조건
b1 = (df.loc[d_prev, 'Close'] < hma.loc[d_prev]) and (df.loc[d, 'Close'] >= hma.loc[d]) and (rsi3.loc[d] >= rsi14.loc[d])
print('\nB1 조건:', b1)
print('  (prev_close < prev_HMA):', df.loc[d_prev, 'Close'] < hma.loc[d_prev])
print('  (close >= HMA):', df.loc[d, 'Close'] >= hma.loc[d])
print('  (rsi3 >= rsi14):', rsi3.loc[d] >= rsi14.loc[d])

# B2 조건
b2 = (df.loc[d_prev, 'Close'] < lower.loc[d_prev]) and (df.loc[d, 'Close'] >= lower.loc[d]) and (rsi3.loc[d] >= rsi14.loc[d])
print('\nB2 조건:', b2)
print('  (prev_close < prev_LowerBand):', df.loc[d_prev, 'Close'] < lower.loc[d_prev])
print('  (close >= LowerBand):', df.loc[d, 'Close'] >= lower.loc[d])
print('  (rsi3 >= rsi14):', rsi3.loc[d] >= rsi14.loc[d]) 