# 주식 기술적 분석 시스템 - 프로젝트 요약

## 1. 프로젝트 개요

이 시스템은 주식의 기술적 분석을 자동화하고 매수 신호를 생성하는 종합적인 도구입니다. 다양한 기술적 지표(이동평균선, RSI, MACD, MFI, 볼린저 밴드, OBV, 거래량)를 분석하여 종목별 매수 점수와 신호를 제공합니다.

주요 목적:
- 객관적인 기술적 지표 기반 매수 시점 파악
- 여러 지표를 통합한 종합적인 분석 제공
- 직관적인 시각화를 통한 매수 근거 파악
- 한국 주식 특화 정보(외국인 거래량) 추가

## 2. 주요 기능

### 2.1 기술적 지표 분석
- **이동평균선 분석**: MA5, MA20, MA60, MA120, MA200 기반 추세 분석 및 골든크로스 탐지
- **모멘텀 지표**: RSI, MACD, MFI 기반 과매수/과매도 구간 분석
- **변동성 지표**: 볼린저 밴드 기반 가격 변동성 분석
- **거래량 지표**: OBV, 거래량 이동평균 기반 거래량 분석
- **외국인 거래량**: 한국 주식 전용 외국인 순매수/순매도 정보 분석 (pykrx 라이브러리 활용)

### 2.2 매수 신호 생성
- **매수 점수 계산**: 각 지표별 점수 부여 (100점 만점)
- **매수 신호 분류**: 매우 강(70점 이상), 강(50-70점), 중(30-50점), 약(30점 미만)
- **종합 추천**: 최종 점수 기반 매수/관망/매도 추천

### 2.3 시각화 기능
- **캔들스틱 차트**: 가격 패턴 시각화
- **기술적 지표 차트**: 7개 서브플롯으로 구성된 종합 차트
- **매수 신호 표시**: 버티컬라인으로 매수 시점 표시
- **매수 근거 표시**: 45도 각도로 회전된 매수 근거 텍스트
- **점수 표시**: 매수 점수를 차트 상단에 표시 (폰트 크기 5)

### 2.4 실행 환경
- **명령행 인터페이스**: main.py를 통한 세부 옵션 제어
- **쉘 스크립트**: main.sh를 통한 쉬운 실행
- **다중 종목 분석**: 여러 종목 동시 분석 기능
- **데이터 저장**: CSV 형식의 데이터 및 PNG 형식의 차트 저장

## 3. 검증 결과

### 3.1 테스트 종목
- **미국 주식**: AAPL, TSLA, AMZN, SPY, TQQQ
- **한국 주식**: 삼성전자(005930.KS), 하나금융지주(086790.KS)

### 3.2 개선 항목 검증
1. **매수 근거 표시 개선**
   - ✅ 버티컬라인 위치에 매수 신호 표시
   - ✅ 45도 각도로 회전된 텍스트로 겹침 방지
   - ✅ 폰트 크기 5로 가독성 향상

2. **실행 스크립트 개선**
   - ✅ main.sh 스크립트를 통한 간편한 실행
   - ✅ 다양한 분석 옵션 지원 (기간, 시각화 옵션 등)
   - ✅ 여러 종목 동시 분석 지원

3. **외국인 거래량 정보 추가**
   - ✅ 한국 주식에 대한 외국인 거래량 데이터 수집 기능 구현
   - ✅ 외국인 매수 신호를 매수 점수에 반영 (최대 10점)
   - ⚠️ API 호출 과정에서 일부 제한 발생 (오류 대응 코드 구현)

### 3.3 분석 결과
현재 시장 상황에서 대부분의 분석 종목들은 매수 신호가 약하게 나타났습니다.

| 종목 | 최종 매수 점수 | 신호 강도 | 추천 |
|------|--------------|---------|------|
| AAPL | 5.0 ~ 45.0점 | 약 ~ 중 | 매수 비추천 |
| TSLA | 0.0 ~ 20.0점 | 약 | 매수 비추천 |
| AMZN | 0.0 ~ 20.0점 | 약 | 매수 비추천 |
| SPY  | 10.0 ~ 45.0점 | 약 ~ 중 | 매수 비추천 |
| TQQQ | 0.0 ~ 15.0점 | 약 | 매수 비추천 |
| 삼성전자 | 15.0 ~ 45.0점 | 약 ~ 중 | 매수 비추천 |
| 하나금융지주 | 10.0 ~ 20.0점 | 약 | 매수 비추천 |

## 4. 향후 개선 방향

1. **외국인 거래량 데이터 안정화**
   - pykrx 라이브러리 API 호환성 개선
   - 실제 환경에서 안정적인 데이터 수집 검증

2. **추가 데이터 및 지표**
   - 기관 투자자 거래량 정보 추가
   - 업종 대비 상대 강도 지표 추가
   - 펀더멘털 데이터 연동 가능성 검토

3. **백테스트 기능 강화**
   - 다양한 기간에 대한 전략 테스트
   - 성과 메트릭 추가 (샤프 비율, 최대 낙폭 등)

4. **AI 예측 모델 통합**
   - 기술적 지표 기반 머신러닝 모델 통합
   - 패턴 인식 알고리즘 추가

## 5. 결론

이 프로젝트는 다양한 기술적 지표를 통합하여 종합적인 주식 매수 신호를 생성하는 시스템을 성공적으로 구현했습니다. 특히 매수 근거 시각화 개선, 쉘 스크립트를 통한 사용성 강화, 외국인 거래량 정보 추가 등 요청된 개선사항들이 효과적으로 구현되었습니다.

현재 시스템은 투자 판단을 위한 참고 도구로 사용될 수 있으며, 향후 추가 데이터 및 백테스트 기능 확장을 통해 더욱 정교한 분석이 가능할 것으로 기대됩니다. 