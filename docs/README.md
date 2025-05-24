# 주식 기술적 분석 시스템 문서

이 디렉토리는 주식 기술적 분석 시스템에 대한 문서를 포함하고 있습니다.

## 목차

1. [시스템 개요](overview.md)
2. [기술 지표 설명](indicators.md)
3. [백테스트 방법](backtest.md)
4. [API 문서](api.md)
5. [개발 가이드](development.md)

## 빠른 시작

### 설치

```bash
git clone https://github.com/yourusername/stock_tech_v1.git
cd stock_tech_v1
pip install -r requirements.txt
```

### 사용 예

```bash
# 애플 주식 1년 일봉 데이터로 기술적 분석 및 시각화
python main.py --ticker AAPL --period 1y --interval 1d --visualize

# 삼성전자 주식 3개월 일봉 데이터로 기술적 분석
python main.py --ticker 005930.KS --period 3mo --interval 1d

# 테슬라 주식 데이터로 이동평균 교차 전략 백테스트
python backtest/simple_backtest.py --ticker TSLA --period 5y --strategy ma_crossover
```

## 참고 자료

- [Yahoo Finance API 문서](https://pypi.org/project/yfinance/)
- [Pandas 문서](https://pandas.pydata.org/docs/)
- [Matplotlib 문서](https://matplotlib.org/stable/contents.html)
- [NumPy 문서](https://numpy.org/doc/stable/) 