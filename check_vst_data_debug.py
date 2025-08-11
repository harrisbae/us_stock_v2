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

print(f"\n=== 디버깅 정보 ===")
print(f"매수신호일: {buy_date.strftime('%Y-%m-%d')}")
print(f"시작일: {start_date.strftime('%Y-%m-%d')}")
print(f"종료일: {end_date.strftime('%Y-%m-%d')}")

period_data = vst.loc[start_date:end_date]
print(f"필터링된 전체 데이터: {len(period_data)}개")

if len(period_data) >= 10:
    last_10 = period_data.tail(10)
    print(f"마지막 10개 데이터: {len(last_10)}개")
    
    print(f'\n=== 실제 사용된 10개 거래일 ===')
    print(f'기간: {last_10.index[0].strftime("%Y-%m-%d")} ~ {last_10.index[-1].strftime("%Y-%m-%d")}')
    print()
    
    for i, (date, row) in enumerate(last_10.iterrows()):
        open_price = row['Open'].item()
        close_price = row['Close'].item()
        is_up = '양봉' if close_price > open_price else '음봉'
        print(f'{i+1:2d}. {date.strftime("%Y-%m-%d")}: 시가 ${open_price:.2f}, 종가 ${close_price:.2f} ({is_up})')

    up_days = len(last_10[last_10['Close'] > last_10['Open']])
    print(f'\n양봉 개수: {up_days}/10 = {up_days/10*100:.1f}%')
else:
    print(f"데이터 부족: {len(period_data)}개 (10개 필요)")
