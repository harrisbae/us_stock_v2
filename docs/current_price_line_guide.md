# 현재가 수직선 스타일 가이드

## 개요
현재가를 표시하는 수직선의 스타일을 두 가지 방식으로 선택할 수 있습니다.

## 스타일 옵션

### 1. 얇은 수직선 (`thin`) - 기본값
- **설명**: 수직선의 두께를 얇게 하여 캔들바가 잘 보이도록 함
- **특징**: 
  - `linewidth=0.5`로 설정
  - `alpha=0.6`으로 투명도 조정
  - 전체 차트 높이에 걸쳐 수직선 표시
- **사용법**: `current_price_line_style='thin'`

### 2. 하단 시작 수직선 (`bottom_start`)
- **설명**: 수직선을 현재 캔들바의 하단에서 시작하여 아래로 그리기
- **특징**:
  - `linewidth=1.0`으로 설정
  - `alpha=0.7`로 투명도 조정
  - 현재 캔들바의 Low 가격에서 시작하여 차트 하단까지 표시
- **사용법**: `current_price_line_style='bottom_start'`

## 사용 예시

### 기본 사용 (얇은 수직선)
```python
from src.indicators.hma_mantra.visualization.volume_profile_overlay_chart import plot_main_chart_with_volume_profile_overlay

# 기본값으로 얇은 수직선 사용
plot_main_chart_with_volume_profile_overlay(
    data=stock_data,
    ticker='AAPL',
    save_path='output/chart.png'
)
```

### 하단 시작 수직선 사용
```python
# 하단에서 시작하는 수직선 사용
plot_main_chart_with_volume_profile_overlay(
    data=stock_data,
    ticker='AAPL',
    save_path='output/chart.png',
    current_price_line_style='bottom_start'
)
```

### 명시적으로 얇은 수직선 지정
```python
# 명시적으로 얇은 수직선 지정
plot_main_chart_with_volume_profile_overlay(
    data=stock_data,
    ticker='AAPL',
    save_path='output/chart.png',
    current_price_line_style='thin'
)
```

## 시각적 효과

### 얇은 수직선 (`thin`)
- 캔들바와 겹쳐도 캔들바가 잘 보임
- 전체 차트에 걸쳐 현재가 위치를 명확하게 표시
- 투명도가 낮아 다른 요소와 겹치지 않음

### 하단 시작 수직선 (`bottom_start`)
- 현재 캔들바의 하단에서 시작하여 자연스러운 연결
- 캔들바와 겹치지 않아 가독성 향상
- 차트 하단의 정보와 연결되어 시각적 일관성 제공

## 권장 사용법

1. **일반적인 분석**: `thin` 스타일 사용 (기본값)
2. **캔들바 가독성 중시**: `thin` 스타일 사용
3. **시각적 연결성 중시**: `bottom_start` 스타일 사용
4. **프레젠테이션용**: `bottom_start` 스타일 사용

## 기술적 세부사항

### 수직선 좌표 계산
- **thin**: `ymin=0, ymax=1` (전체 차트 높이)
- **bottom_start**: `ymin=normalized_low_price, ymax=1` (캔들바 하단부터 차트 하단까지)

### 정규화된 좌표
- `ymin_normalized = (current_candle_low - min_price) / price_range`
- 가격 범위를 0-1 사이로 정규화하여 matplotlib의 `ymin` 파라미터에 맞춤

### zorder 설정
- 수직선: `zorder=999`
- 투자전략 정보 박스: `zorder=1000`
- 적절한 레이어 순서로 겹침 방지
