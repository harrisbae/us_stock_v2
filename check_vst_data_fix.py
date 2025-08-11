import yfinance as yf
import pandas as pd
from datetime import timedelta

# VST 데이터 다운로드
print("VST 데이터 다운로드 중...")
vst = yf.download('VST', start='2024-01-01', end='2024-02-01')

# 2024-01-19 매수신호 기준으로 확인
buy_date = pd.to_datetime('2024-01-19')
start_date = buy_date - timedelta(days=20)
end_date = buy_date - timedelta(days=1)

period_data = vst.loc[start_date:end_date]
last_10 = period_data.tail(10)

print(f'\n=== 2024-01-19 매수신호 기준 당일 이전 10개 거래일 ===')
print(f'기간: {last_10.index[0].strftime("%Y-%m-%d")} ~ {last_10.index[-1].strftime("%Y-%m-%d")}')
print()

# 방법 1: 직접 비교
up_days_method1 = 0
for i, (date, row) in enumerate(last_10.iterrows()):
    open_price = row['Open'].item()
    close_price = row['Close'].item()
    is_up = '양봉' if close_price > open_price else '음봉'
    if close_price > open_price:
        up_days_method1 += 1
    print(f'{i+1:2d}. {date.strftime("%Y-%m-%d")}: 시가 ${open_price:.2f}, 종가 ${close_price:.2f} ({is_up})')

print(f'\n=== 방법별 계산 결과 ===')
print(f'방법 1 (직접 비교): 양봉 {up_days_method1}/10 = {up_days_method1/10*100:.1f}%')

# 방법 2: pandas boolean indexing
up_days_method2 = len(last_10[last_10['Close'] > last_10['Open']])
print(f'방법 2 (pandas): 양봉 {up_days_method2}/10 = {up_days_method2/10*100:.1f}%')

# 방법 3: numpy 비교
import numpy as np
up_days_method3 = np.sum(last_10['Close'].values > last_10['Open'].values)
print(f'방법 3 (numpy): 양봉 {up_days_method3}/10 = {up_days_method3/10*100:.1f}%')

print(f'\n=== 문제 분석 ===')
print(f'pandas boolean indexing 결과: {up_days_method2}')
print(f'직접 계산 결과: {up_days_method1}')
print(f'numpy 계산 결과: {up_days_method3}')
print(f'어떤 방법이 정확한가요?')
