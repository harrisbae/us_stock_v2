# Volume Profile 기능 가이드

## 📊 개요

Volume Profile은 특정 기간 동안 각 가격대별로 거래된 총 거래량을 시각화하는 고급 기술적 분석 도구입니다. 이 기능을 통해 시장 참여자들이 어떤 가격대에서 가장 활발하게 거래했는지 파악할 수 있습니다.

## 🎯 주요 개념

### 1. POC (Point of Control)
- **정의**: 거래량이 가장 많은 가격대
- **의미**: 시장 참여자들이 가장 많이 거래한 가격 레벨
- **표시**: 빨간 점선으로 표시

### 2. Value Area
- **정의**: 전체 거래량의 70%가 발생한 가격 구간
- **의미**: 주요 지지/저항 구간으로 활용 가능
- **표시**: 연한 초록색 영역으로 표시

### 3. Volume Distribution
- **정의**: 각 가격대별 거래량 분포
- **표시**: 수평 막대그래프로 시각화

## 🚀 사용 방법

### 명령행 옵션

```bash
./hma.sh -s SYMBOL -v VOLUME_PROFILE_TYPE
```

#### 옵션 설명
- `-s SYMBOL`: 분석할 종목 코드
- `-v TYPE`: Volume Profile 유형
  - `none`: Volume Profile 없음 (기본값)
  - `separate`: 별도 영역에 Volume Profile
  - `overlay`: 메인차트에 Volume Profile 오버레이

### 사용 예시

#### 1. Volume Profile 오버레이
```bash
./hma.sh -s TSLA -v overlay
```
- 메인차트에 Volume Profile 오버레이
- 공간 효율적, 직관적 분석

#### 2. Volume Profile 별도 영역
```bash
./hma.sh -s TSLA -v separate
```
- 메인차트 + 거래량 차트 (좌측)
- Volume Profile (우측 별도 영역)

#### 3. 기본 기술적 분석
```bash
./hma.sh -s TSLA -v none
# 또는
./hma.sh -s TSLA
```

#### 4. 종목 파일과 함께 사용
```bash
./hma.sh -f symbols.txt -v overlay
```

## 📈 차트 구성

### 1. 분리형 레이아웃 (separate)

#### 메인차트 (좌상단 75%)
- 캔들스틱 차트
- HMA, 만트라 밴드, 볼린저 밴드
- 매수/매도 신호 (B1/B2/T1/T2)
- 조건부 수직선/수평선
- 지지선/저항선/현재가

#### 거래량 차트 (좌하단 25%)
- 거래량 막대 (양봉/음봉 구분)
- Volume MA(5), Volume MA(20)
- Volume BB Upper

#### Volume Profile (우측 전체)
- 거래량 분포 (수평 막대그래프)
- POC (Point of Control) - 빨간 점선 + 가격 표시
- Value Area (70% 거래량 구간)

### 2. 오버레이형 레이아웃 (overlay)

#### 메인차트 (상단 75%)
- 캔들스틱 차트 + 모든 기술적 지표
- **Volume Profile 오버레이** (우측 20% 영역)
  - 반투명 거래량 막대
  - POC (빨간 점선)
  - Value Area (연한 초록 영역)

#### 거래량 차트 (하단 25%)
- 거래량 막대 및 이동평균선

## 🔧 기술적 구현

### 1. Volume Profile 계산 로직

```python
def calculate_volume_profile(ohlcv_data, num_bins=50):
    # 가격 범위 설정
    price_min = ohlcv_data['Low'].min()
    price_max = ohlcv_data['High'].max()
    
    # 가격을 50개 구간으로 나누기
    price_bins = np.linspace(price_min, price_max, num_bins + 1)
    
    # 거래량 분포 계산
    volume_profile = []
    for i in range(len(price_bins) - 1):
        bin_low = price_bins[i]
        bin_high = price_bins[i + 1]
        
        # 해당 가격 구간에 속하는 거래량 합계
        mask = (ohlcv_data['Low'] <= bin_high) & (ohlcv_data['High'] >= bin_low)
        total_volume = ohlcv_data.loc[mask, 'Volume'].sum()
        volume_profile.append(total_volume)
    
    # POC 계산
    poc_idx = np.argmax(volume_profile)
    poc_price = price_bins[poc_idx]
    
    # Value Area 계산 (거래량 70% 구간)
    # POC를 중심으로 양쪽으로 확장하여 70% 달성
    
    return price_bins, volume_profile, poc_price, value_area_min, value_area_max
```

### 2. 오버레이 구현

```python
# Volume Profile 오버레이 (메인차트 우측에 반투명하게)
overlay_width = (main_xlim[1] - main_xlim[0]) * 0.2
overlay_start = main_xlim[1] - overlay_width

# Volume Profile 정규화
max_volume = max(volume_profile)
normalized_volume = [v / max_volume for v in volume_profile]

# 막대 그리기
for i, (price, vol) in enumerate(zip(price_bins[:-1], normalized_volume)):
    bar_width = vol * overlay_width * 0.8
    ax_main.barh(price, bar_width, height=bin_heights, left=overlay_start, 
                alpha=0.3, color='blue', zorder=10)
```

## 📊 분석 활용법

### 1. 지지/저항 분석
- **POC 활용**: 거래량이 가장 많은 가격대는 강한 지지/저항 역할
- **Value Area**: 70% 거래량 구간은 주요 거래 구간으로 활용

### 2. 매수/매도 타이밍
- **POC 돌파**: 거래량 집중 구간 돌파 시 강한 신호
- **Value Area 내 거래**: 주요 거래 구간 내에서의 움직임 분석

### 3. 시장 구조 분석
- **거래량 분포**: 가격대별 거래량 패턴으로 시장 구조 파악
- **시간대별 변화**: 기간별 Volume Profile 변화로 시장 진화 분석

## 📁 파일 구조

```
src/indicators/hma_mantra/visualization/
├── volume_profile_chart.py          # 분리형 Volume Profile
└── volume_profile_overlay_chart.py  # 오버레이형 Volume Profile

test/
├── volume_profile_test.py           # 분리형 테스트
└── volume_profile_overlay_test.py   # 오버레이형 테스트

output/hma_mantra/SYMBOL/
├── SYMBOL_volume_profile_chart.png      # 분리형 결과
└── SYMBOL_volume_profile_overlay_chart.png  # 오버레이형 결과
```

## 🎨 시각적 요소

### 색상 체계
- **POC**: 빨간색 점선 (`#FF0000`)
- **Value Area**: 연한 초록색 영역 (`#90EE90`, alpha=0.2)
- **Volume Profile 막대**: 파란색 (`#0000FF`, alpha=0.3/0.6)
- **거래량 막대**: 양봉(초록)/음봉(빨강)

### 레이아웃 설정
- **분리형**: 2x2 그리드 (width_ratios=[4,1], height_ratios=[3,1])
- **오버레이형**: 2x1 그리드 (height_ratios=[3,1])
- **간격**: hspace=0.1, wspace=0.05

## 🔍 고급 분석 팁

### 1. 다중 기간 분석
- 단기/중기/장기 Volume Profile 비교
- 시장 구조 변화 추적

### 2. 상대적 분석
- 동일 섹터 종목들의 Volume Profile 비교
- 시장 전체 Volume Profile과 개별 종목 비교

### 3. 거래량 패턴 분석
- 거래량 집중 구간과 가격 움직임의 상관관계
- 거래량 분산과 시장 불확실성

## 🚨 주의사항

### 1. 데이터 품질
- 충분한 거래 데이터 필요 (최소 30일 이상 권장)
- 거래량이 적은 종목은 신뢰도 저하

### 2. 해석 주의
- Volume Profile은 과거 데이터 기반
- 현재 시장 상황과 차이 가능성
- 다른 기술적 지표와 함께 사용 권장

### 3. 시장 환경
- 급격한 시장 변화 시 Volume Profile 의미 변화
- 뉴스나 이벤트로 인한 거래량 급증 시 해석 주의

## 📈 향후 개선 계획

### 1. 기능 확장
- [ ] 실시간 Volume Profile 업데이트
- [ ] 다중 기간 Volume Profile 비교
- [ ] Volume Profile 기반 자동 신호 생성

### 2. 시각화 개선
- [ ] 3D Volume Profile 시각화
- [ ] 인터랙티브 Volume Profile 차트
- [ ] 커스텀 색상 테마 지원

### 3. 분석 도구
- [ ] Volume Profile 기반 백테스팅
- [ ] 자동 지지/저항선 생성
- [ ] Volume Profile 기반 포트폴리오 최적화

---

## 📞 지원 및 문의

Volume Profile 기능 사용 중 문제가 발생하거나 개선 제안이 있으시면 언제든 연락주세요.

**마지막 업데이트**: 2025-07-26
**버전**: 1.0.0 