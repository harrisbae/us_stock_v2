import pandas as pd
import os
import matplotlib.pyplot as plt
import platform

# 한글 폰트 설정
if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rc('axes', unicode_minus=False)

# 데이터 파일 경로
csv_path = os.path.join('output', '20250507', 'VST', 'VST_data.csv')
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"{csv_path} 파일이 존재하지 않습니다.")

df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')

# 파라미터
initial_cash = 100000
cash = initial_cash
position = 0  # 보유 주식 수
trade_log = []
fee = 0.0  # 수수료 비율

# 신호 기준
BUY_SCORE = 50  # 매수 신호 점수 기준
SELL_SCORE = 30  # 매도 신호 점수 기준

# 매수/매도 비율
BUY_RATIO = 0.2  # 매수 20%
SELL_RATIO = 0.1  # 매도 10%

for date, row in df.iterrows():
    price = row['Close']
    buy_score = row.get('Buy_Score', 0)
    sell_score = row.get('Sell_Score', 0)
    action = None
    # 매수 신호: 현금의 20%만 매수
    if buy_score >= BUY_SCORE and cash > 0:
        buy_amount = cash * BUY_RATIO
        buy_qty = buy_amount * (1 - fee) / price
        position += buy_qty
        cash -= buy_amount
        action = f"매수(20%) @ {price:.2f}"
    # 매도 신호: 보유 주식의 10%만 매도
    elif sell_score >= SELL_SCORE and position > 0:
        sell_qty = position * SELL_RATIO
        sell_amount = sell_qty * price * (1 - fee)
        position -= sell_qty
        cash += sell_amount
        action = f"매도(10%) @ {price:.2f}"
    if action:
        trade_log.append((date, action, cash + position * price))

# 마지막 날 평가
final_value = cash + position * df['Close'].iloc[-1]
ret = (final_value - initial_cash) / initial_cash * 100

print(f"\n[단순 룰 기반 누적 수익률 백테스트 결과]")
print(f"초기 자본: {initial_cash:,.0f}")
print(f"최종 평가금액: {final_value:,.2f}")
print(f"누적 수익률: {ret:.2f}%\n")
print("거래 내역:")
for date, action, value in trade_log:
    print(f"{date.strftime('%Y-%m-%d')}: {action} (총 평가금액: {value:,.2f})")

# 누적 평가금액 시계열 생성
portfolio_values = []
current_cash = initial_cash
current_position = 0
for date, row in df.iterrows():
    price = row['Close']
    buy_score = row.get('Buy_Score', 0)
    sell_score = row.get('Sell_Score', 0)
    # 매수
    if buy_score >= BUY_SCORE and current_cash > 0:
        buy_amount = current_cash * BUY_RATIO
        buy_qty = buy_amount * (1 - fee) / price
        current_position += buy_qty
        current_cash -= buy_amount
    # 매도
    elif sell_score >= SELL_SCORE and current_position > 0:
        sell_qty = current_position * SELL_RATIO
        sell_amount = sell_qty * price * (1 - fee)
        current_position -= sell_qty
        current_cash += sell_amount
    portfolio_values.append(current_cash + current_position * price)

# 그래프 시각화
plt.figure(figsize=(12,6))
plt.plot(df.index, portfolio_values, label='누적 평가금액(포트폴리오)', color='blue', linewidth=2)
plt.plot(df.index, df['Close'] / df['Close'].iloc[0] * initial_cash, label='TQQQ 종가(비교)', color='gray', linestyle='--', alpha=0.7)
plt.title('TQQQ 단순 룰 기반 백테스트 누적 수익률')
plt.xlabel('날짜')
plt.ylabel('금액')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show() 