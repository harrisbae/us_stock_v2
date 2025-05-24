# HMA Mantra 분석 실행 가이드

## 1. 기본 실행 방법

### 1.1 Python 스크립트 직접 실행
```bash
python src/indicators/hma_mantra_example.py <종목코드> <기간(일)> <flat_threshold>
```

예시:
```bash
# SPY 종목, 120일 기간, flat_threshold 0.6으로 분석
python src/indicators/hma_mantra_example.py SPY 120 0.6
```

### 1.2 Shell 스크립트 사용
```bash
./hma.sh -s <종목코드> -p <기간> -t <flat_threshold>
```

예시:
```bash
# SPY 종목, 120일 기간, flat_threshold 0.6으로 분석
./hma.sh -s SPY -p 120 -t 0.6
```

## 2. 파라미터 설명

### 2.1 필수 파라미터
- `종목코드`: 분석할 주식 종목 코드 (예: SPY, AAPL, QQQ 등)
- `기간`: 분석할 기간 (일 단위)
- `flat_threshold`: HMA 평탄화 판단 기준값 (기본값: 0.6)

### 2.2 기간 설정 예시
- 단기 분석: 12일
- 중기 분석: 120일
- 장기 분석: 365일

## 3. 출력 결과

### 3.1 저장 위치
분석 결과는 다음 경로에 저장됩니다:
```
output/hma_mantra/<종목코드>/<종목코드>_hma_mantra_md_signals.png
```

### 3.2 차트 구성
- 매수 신호 (^):
  - Close 기반 (금색)
  - HMA 기반 (파란색)
  - MACD 기반 (보라색)
  - RSI 기반 (초록색)
- 매도 신호 (v):
  - Close 기반 (주황색)
  - HMA 기반 (진한 파란색)
  - MACD 기반 (빨간색)
  - RSI 기반 (갈색)

## 4. 실행 예시

### 4.1 단기 분석
```bash
# 12일 기간 분석
./hma.sh -s SPY -p 12 -t 0.6
```

### 4.2 중기 분석
```bash
# 120일 기간 분석
./hma.sh -s SPY -p 120 -t 0.6
```

### 4.3 장기 분석
```bash
# 365일 기간 분석
./hma.sh -s SPY -p 365 -t 0.6
```

### 4.4 여러 종목 동시 분석
```bash
# SPY와 QQQ 종목 분석
./hma.sh -s "SPY,QQQ" -p 120 -t 0.6
```

## 5. 주의사항
- flat_threshold 값은 시장 상황과 분석 목적에 따라 조정이 필요할 수 있습니다.
- 기간이 길수록 더 많은 신호와 트렌드를 확인할 수 있습니다.
- 결과 차트는 PNG 형식으로 저장되며, 차트 상단에 분석 기간과 설정값이 표시됩니다. 