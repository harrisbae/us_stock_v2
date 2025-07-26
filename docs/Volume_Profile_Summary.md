# Volume Profile 기능 요약

## 🎯 기능 개요

Volume Profile은 가격대별 거래량 분포를 시각화하여 시장의 거래 패턴을 분석하는 고급 기술적 분석 도구입니다.

## 📊 주요 구성 요소

### 1. POC (Point of Control)
- 거래량이 가장 많은 가격대
- 강한 지지/저항 역할
- 빨간 점선으로 표시

### 2. Value Area
- 전체 거래량의 70% 구간
- 주요 거래 구간
- 연한 초록색 영역으로 표시

### 3. Volume Distribution
- 가격대별 거래량 분포
- 수평 막대그래프로 시각화

## 🚀 사용법

### 기본 명령어
```bash
./hma.sh -s SYMBOL -v VOLUME_PROFILE_TYPE
```

### 옵션
- `-v none`: Volume Profile 없음 (기본)
- `-v separate`: 별도 영역에 Volume Profile
- `-v overlay`: 메인차트에 Volume Profile 오버레이

### 사용 예시
```bash
# 오버레이 버전
./hma.sh -s TSLA -v overlay

# 별도 영역 버전
./hma.sh -s TSLA -v separate

# 기본 분석
./hma.sh -s TSLA
```

## 📈 차트 유형

### 1. 분리형 레이아웃 (separate)
- 메인차트 + 거래량 차트 (좌측)
- Volume Profile (우측 별도 영역)
- **장점**: Volume Profile을 독립적으로 자세히 분석 가능

### 2. 오버레이형 레이아웃 (overlay)
- 메인차트에 Volume Profile 오버레이
- 거래량 차트 (하단)
- **장점**: 공간 효율적, 직관적 분석, 실시간 매칭

## 🔧 기술적 특징

### 계산 로직
1. 가격 범위를 50개 구간으로 분할
2. 각 구간별 거래량 합계 계산
3. POC (최대 거래량 구간) 식별
4. Value Area (70% 거래량 구간) 계산

### 시각화
- **POC**: 빨간 점선 + 가격 텍스트
- **Value Area**: 연한 초록색 영역
- **Volume Profile**: 반투명 파란색 막대

## 📁 생성 파일

### 분리형
- `SYMBOL_volume_profile_chart.png`

### 오버레이형
- `SYMBOL_volume_profile_overlay_chart.png`

## 💡 분석 활용

### 1. 지지/저항 분석
- POC를 강한 지지/저항선으로 활용
- Value Area를 주요 거래 구간으로 활용

### 2. 매수/매도 타이밍
- POC 돌파 시 강한 신호
- Value Area 내 거래 패턴 분석

### 3. 시장 구조 분석
- 거래량 분포로 시장 구조 파악
- 시간대별 Volume Profile 변화 추적

## ⚠️ 주의사항

### 데이터 품질
- 충분한 거래 데이터 필요 (30일 이상 권장)
- 거래량이 적은 종목은 신뢰도 저하

### 해석 주의
- 과거 데이터 기반이므로 현재 상황과 차이 가능
- 다른 기술적 지표와 함께 사용 권장

## 🔄 향후 계획

### 기능 확장
- [ ] 실시간 업데이트
- [ ] 다중 기간 비교
- [ ] 자동 신호 생성

### 시각화 개선
- [ ] 3D 시각화
- [ ] 인터랙티브 차트
- [ ] 커스텀 테마

---

**버전**: 1.0.0  
**마지막 업데이트**: 2025-07-26  
**상세 가이드**: [Volume_Profile_Guide.md](Volume_Profile_Guide.md) 