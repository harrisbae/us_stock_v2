import subprocess

symbols = ['AAPL', 'TSLA', 'COIN', 'GLD', 'NVDA', 'PLTR', 'TEM', 'RCAT']
 
for symbol in symbols:
    print(f"\n===== {symbol} =====")
    subprocess.run(['python', 'src/analysis/macro/economic_indicators.py', symbol]) 