# main.sh 사용 가이드

`main.sh`는 주식 기술적 분석 시스템을 쉽게 실행할 수 있는 쉘 스크립트입니다. 다양한 옵션을 통해 종목 분석을 유연하게 수행할 수 있습니다.

## 기본 사용법

```bash
./main.sh [옵션] [종목코드1] [종목코드2] ...
```

종목코드를 지정하지 않으면 기본 종목(AAPL, TSLA, AMZN)을 분석합니다.

## 옵션 설명

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `-h, --help` | 도움말 표시 | - |
| `-p, --period PERIOD` | 데이터 기간 지정 | `1mo` |
| `-i, --interval INTERVAL` | 봉 간격 지정 | `1d` |
| `-s, --start-date DATE` | 시작 날짜 (YYYY-MM-DD 형식) | - |
| `-e, --end-date DATE` | 종료 날짜 (YYYY-MM-DD 형식) | 현재 날짜 |
| `-o, --output DIR` | 출력 디렉토리 지정 | `output/YYYYMMDD/TICKER` |
| `-nv, --no-visualize` | 차트 시각화 생략 | 시각화 실행 |

### 주요 기간(PERIOD) 옵션
- `1d`: 1일
- `5d`: 5일
- `1mo`: 1개월
- `3mo`: 3개월
- `6mo`: 6개월
- `1y`: 1년
- `2y`: 2년
- `5y`: 5년
- `10y`: 10년
- `max`: 최대 가능 기간

### 주요 간격(INTERVAL) 옵션
- `1m`: 1분봉
- `5m`: 5분봉
- `15m`: 15분봉
- `30m`: 30분봉
- `60m`: 60분봉
- `1h`: 1시간봉
- `1d`: 일봉
- `1wk`: 주봉
- `1mo`: 월봉

## 사용 예제

### 예제 1: 기본 사용법
애플(AAPL) 주식에 대해 기본 분석 수행 (1개월 데이터, 일봉 기준)

```bash
./main.sh AAPL
```

### 예제 2: 여러 종목 동시 분석
애플, 테슬라, 아마존 주식을 3개월 데이터로 분석

```bash
./main.sh -p 3mo AAPL TSLA AMZN
```

### 예제 3: 특정 기간 분석
테슬라 주식을 2025년 1월부터 5월까지의 데이터로 분석

```bash
./main.sh -s 2025-01-01 -e 2025-05-01 TSLA
```

### 예제 4: 시각화 없이 빠른 분석
시각화 없이 여러 종목의 매수 신호만 빠르게 확인

```bash
./main.sh -nv AAPL TSLA AMZN NVDA MSFT
```

### 예제 5: 짧은 기간 고밀도 분석
최근 5일간의 데이터를 5분봉으로 상세 분석

```bash
./main.sh -p 5d -i 5m AAPL
```

### 예제 6: 한국 주식 분석 (외국인 거래량 포함)
삼성전자와 하나금융지주 한국 주식을 분석 (외국인 거래량 정보 포함)

```bash
./main.sh -p 3mo 005930.KS 086790.KS
```

### 예제 7: 커스텀 출력 디렉토리 지정
분석 결과를 특정 디렉토리에 저장

```bash
./main.sh -o ./analysis_results AAPL
```

### 예제 8: 장기 추세 분석
2년 데이터를 주봉 기준으로 분석하여 장기 추세 파악

```bash
./main.sh -p 2y -i 1wk AMZN
```

## 결과 확인

분석 결과는 기본적으로 `output/YYYYMMDD/TICKER/` 디렉토리에 저장됩니다:

- `TICKER_data.csv`: 모든 기술적 지표가 포함된 데이터 파일
- `TICKER_technical_analysis.png`: 종합 차트 이미지 (시각화 옵션 활성화 시)

## 참고사항

- 외국인 거래량 정보는 한국 주식(예: 005930.KS)에만 제공됩니다.
- 분석 결과는 투자 결정을 위한 참고용으로만 사용하시기 바랍니다.
- 모든 결과는 과거 데이터를 기반으로 하며, 미래 주가를 정확히 예측하지 못할 수 있습니다. 