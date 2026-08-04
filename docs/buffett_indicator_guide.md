# 버핏 지수 (Buffett Indicator) 가이드

## 개요
버핏 지수는 미국 주식시장의 전체 시가총액을 GDP로 나눈 값으로, 워렌 버핏이 "주식시장의 가치를 측정하는 가장 좋은 단일 지표"라고 평가한 지표입니다.

## 📊 계산 공식

```
버핏 지수 = (Wilshire 5000 시가총액 / US GDP) × 100
```

### 구성 요소
- **Wilshire 5000 시가총액**: 미국 전체 주식시장의 시가총액 (조 달러)
- **US GDP**: 미국 국내총생산 (조 달러)
- **단위**: 퍼센트 (%)

## 🎯 해석 기준

| 버핏 지수 | 심리도 | 투자 가이드 | 주식 비중 권장 |
|-----------|--------|-------------|----------------|
| ≤ 50% | 극도 과매도 | 강력한 매수 기회 | 80%+ |
| 50-75% | 과매도 | 매수 기회 | 70-80% |
| 75-90% | 균형 | 균형적 배분 | 50-60% |
| 90-115% | 과열 | 주의 필요 | 30-40% |
| > 115% | 극도 과열 | 강력한 매도 신호 | 20% 이하 |

## 📈 데이터 소스

### 1. Wilshire 5000 시가총액
- **지수**: `^W5000` (Yahoo Finance)
- **달러 환산**: 검증된 캘리브레이션 `(지수 / 기준지수) × 기준시총(조$)`
- **캘리브 기준**: `src/indicators/hma_mantra/buffett_indicator.py` 상수 (buffettindicator.org 등과 교차검증 후 갱신)
- **기준일**: 최신 거래일

### 2. US GDP
- **데이터 제공**: FRED `GDP` (pandas_datareader 또는 `FRED_API_KEY` + fredapi)
- **단위**: 10억 달러 → 조 달러 (`/ 1000`)
- **업데이트**: 분기별
- **실패 시**: 최근 FRED 관측치 fallback

## 🔧 구현 방법

### 기본 사용법
```python
from src.indicators.hma_mantra.buffett_indicator import calculate_buffett_indicator

# 버핏 지수 계산
buffett_data = calculate_buffett_indicator()

if buffett_data:
    print(f"버핏 지수: {buffett_data['value']}%")
    print(f"Wilshire 5000: {buffett_data['wilshire_market_cap']}조 달러")
    print(f"US GDP: {buffett_data['us_gdp']}조 달러")
```

### 반환 데이터 구조
```python
{
    'value': 196.6,                    # 버핏 지수 값 (%) - 2025년 8월 최신
    'wilshire_market_cap': 57.2,       # Wilshire 5000 시가총액 (조 달러)
    'wilshire_date': '2025-08-22',    # 시가총액 기준일
    'us_gdp': 29.1,                   # US GDP (조 달러)
    'gdp_date': '2025-Q2'             # GDP 기준일
}
```

## 📊 투자 심리도 분석

### 심리도 계산 함수
```python
from src.indicators.hma_mantra.visualization.volume_profile_overlay_chart import get_buffett_sentiment

sentiment, color, guide = get_buffett_sentiment(buffett_value)
```

### 심리도별 색상
- **극도 과매도**: `darkgreen` (진한 초록)
- **과매도**: `green` (초록)
- **균형**: `orange` (주황)
- **과열**: `red` (빨강)
- **극도 과열**: `darkred` (진한 빨강)

## 🚀 실제 사용 예시

### 1. 차트에 표시
```python
# 통합 정보 박스에 버핏 지수 표시
if buffett_value and buffett_data:
    info_text += f'\n\n버핏지수: {buffett_value}% ({buffett_sentiment})'
    info_text += f'\n시장가이드: {buffett_guide}'
    info_text += f'\nWilshire 5000: {buffett_data["wilshire_market_cap"]}조 달러 ({buffett_data["wilshire_date"]})'
    info_text += f'\nUS GDP: {buffett_data["us_gdp"]}조 달러 ({buffett_data["gdp_date"]})'
```

### 2. 신호 파일에 저장
```python
# 신호 분석 결과에 버핏 지수 포함
signal_info = {
    'buffett_indicator': buffett_value,
    'wilshire_market_cap': buffett_data.get('wilshire_market_cap'),
    'wilshire_date': buffett_data.get('wilshire_date'),
    'us_gdp': buffett_data.get('us_gdp'),
    'gdp_date': buffett_data.get('gdp_date')
}
```

## 📋 테스트 실행

### 테스트 스크립트 실행
```bash
python test_buffett_indicator.py          # 자동 검증
python test_buffett_indicator.py --no-external  # 외부 참조 비교 생략
```

### 테스트 내용
1. **버핏 지수 계산**: 실제 데이터를 사용한 계산
2. **투자 심리도 분석**: 계산된 지수에 따른 심리도 판단
3. **계산 공식 검증**: 수동 계산과 자동 계산 결과 비교
4. **데이터 소스 확인**: Wilshire 5000과 GDP 데이터 검증

## ⚠️ 주의사항

### 1. 데이터 한계
- **GDP 데이터**: 분기별 업데이트로 인한 지연
- **시가총액**: 실시간 변동으로 인한 불안정성
- **계산 오차**: 근사값 계산으로 인한 오차 가능성

### 2. 투자 판단
- **참고용 지표**: 단일 지표에 의존하지 말 것
- **시장 상황**: 글로벌 경제 상황과 연계 분석 필요
- **개별 종목**: 시장 전체 지표이므로 개별 종목 판단과는 별개

### 3. 기술적 제약
- **API 제한**: Yahoo Finance API 사용량 제한
- **네트워크**: 인터넷 연결 상태에 따른 데이터 로드 실패 가능
- **데이터 품질**: 외부 데이터의 정확성 보장 어려움

## 🔮 향후 개선 방향

### 1. 데이터 소스 확장
- **FRED API**: 공식 GDP 데이터 연동
- **실시간 업데이트**: 자동 데이터 갱신 시스템
- **역사적 데이터**: 장기 트렌드 분석 기능

### 2. 분석 기능 강화
- **시계열 분석**: 과거 데이터 기반 패턴 분석
- **예측 모델**: 머신러닝 기반 미래 예측
- **국제 비교**: 다른 국가와의 비교 분석

### 3. 사용자 경험 개선
- **웹 대시보드**: 실시간 모니터링 인터페이스
- **알림 시스템**: 임계값 도달 시 자동 알림
- **모바일 앱**: 스마트폰에서의 편리한 접근

## 📚 참고 자료

- **워렌 버핏**: Berkshire Hathaway CEO
- **Wilshire 5000**: 미국 전체 주식시장 지수
- **FRED**: Federal Reserve Economic Data
- **GDP**: Gross Domestic Product (국내총생산)
- **시가총액**: Market Capitalization

## 📞 지원 및 문의

버핏 지수 관련 문의사항이나 개선 제안이 있으시면 프로젝트 이슈를 통해 연락해 주세요.
