# HMA+만트라 자동매매 신호 시스템 개발/테스트 내역

## 1. 주요 보조지표 및 신호 조건

- **HMA (Hull Moving Average)**
- **만트라 밴드 (Mantra Bands)**
- **MACD, RSI 등 추가 보조지표**
- **매수/매도 신호의 시각화 및 커스터마이징**

---

## 2. 매수/매도 신호의 필수 조건

### 매수 신호 필수 조건
- 아래 둘 중 하나만 만족하면 필수 조건 충족:
  - 오늘 HMA > 전일 기준 5일 HMA 평균
  - 오늘 HMA와 전일 HMA의 차이의 절대값이 `flat_threshold` 미만 (기본값: 0.6, 커맨드라인 인자로 조정 가능)

  ```python
  is_hma5_up = (hma_now > hma_5avg_prev) or (abs(hma_now - hma_prev) < flat_threshold)
  ```

### 매도 신호 필수 조건
- 아래 둘 중 하나만 만족하면 필수 조건 충족:
  - 오늘 HMA ≤ 전일 기준 5일 HMA 평균
  - 오늘 HMA와 전일 HMA의 차이의 절대값이 `flat_threshold` 미만

  ```python
  is_hma5_flat_or_down = (hma_now <= hma_5avg_prev) or (abs(hma_now - hma_prev) < flat_threshold)
  ```

---

## 3. 매수/매도 신호의 세부 조건

### 매수 신호 (필수 조건 + 아래 6가지 중 하나)
1. 종가 HMA 상향돌파 + MACD > Signal
2. 종가 HMA 상향돌파 + RSI < 40
3. 종가 HMA 상향돌파 + (MACD ≤ Signal and RSI ≥ 40)
4. 종가 < 하단밴드 + MACD > Signal
5. 종가 < 하단밴드 + RSI < 40
6. 종가 < 하단밴드 + (MACD ≤ Signal and RSI ≥ 40)

### 매도 신호 (필수 조건 + 아래 6가지 중 하나)
1. HMA 하향 돌파 + MACD < Signal
2. HMA 하향 돌파 + RSI > 70
3. HMA 하향 돌파 + (MACD ≥ Signal and RSI ≤ 70)
4. 종가 > 상단밴드 + MACD < Signal
5. 종가 > 상단밴드 + RSI > 70
6. 종가 > 상단밴드 + (MACD ≥ Signal and RSI ≤ 70)

---

## 4. 평탄 기준값(flat_threshold) 커스터마이징

- flat_threshold 값은 커맨드라인 인자 또는 함수 인자로 입력 가능
- 예시:
  ```bash
  python src/indicators/hma_mantra_example.py SPY 120 0.3
  ```
- plot 함수와 신호 생성 함수 모두 flat_threshold를 연동

---

## 5. 시각화 및 디버깅

- 신호 마커, 내부 숫자, 범례, 텍스트 위치 등 세부 시각화 커스터마이징
- 각 날짜별 HMA, 종가, 신호 조건, 기울기 등 디버깅 코드로 검증
- 차트 해상도(픽셀) 기준 기울기 계산 함수도 예시로 구현

---

## 6. 코드 구조 및 주요 함수

- `get_hma_mantra_md_signals(data, flat_threshold=0.6)`: 신호 생성, 평탄 기준값 인자화
- `plot_hma_mantra_md_signals(data, ticker, save_path=None, flat_threshold=0.6)`: 시각화, flat_threshold 연동
- `hma_mantra_example.py`: 커맨드라인에서 flat_threshold 입력 가능

---

## 7. 기타

- 각종 조건, 시각화, 디버깅, threshold 조정 등 반복적으로 실험 및 개선
- 모든 주요 변경점은 코드와 차트에 즉시 반영 및 테스트

--- 