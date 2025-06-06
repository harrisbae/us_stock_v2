# 📘 Mantra Band + HMA 종합 기술적 분석 전략 가이드

## 1. 전략 개요

**만트라 밴드**는 이격도 기반의 평균회귀 밴드이며,  
**HMA(Hull Moving Average)**는 빠르고 부드러운 중심 추세선을 제공합니다.  
이 둘을 결합하면 **방향성과 진입 타이밍을 동시에 포착**할 수 있습니다.

---

## 2. 구성 요소 및 역할

| 요소 | 설명 |
|------|------|
| `HMA(21)` | 중심 추세선. SMA보다 빠르고 부드러움 |
| `만트라 밴드` | 평균회귀 밴드. 기준선(HMA)을 중심으로 ±α × 평균 이격도 |
| `α (알파)` | 밴드 민감도 계수 (보통 1.2 ~ 1.6) |
| `보조지표` | RSI, MACD, ADX 등으로 진입 신호 정밀화 |

---

## 3. 신호 구조

### 📥 매수 조건

- 종가 < 하단 밴드
- HMA 우상향
- MACD > Signal
- (선택) RSI < 40

→ ✅ **위 조건 중 2~3개 이상 충족 시 진입**

### 📤 매도 조건

- 종가 > 상단 밴드
- HMA 평탄 또는 하락
- MACD < Signal
- (선택) RSI > 70

→ ✅ **위 조건 중 2~3개 이상 충족 시 익절/청산**