#!/usr/bin/env python3
"""
HMA Mantra 매수신호 분석 - 종목별 분석 도구
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import warnings
warnings.filterwarnings('ignore')

def calculate_rsi(data, window=14):
    """RSI 계산"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_scalar_value(value):
    """pandas Series를 스칼라 값으로 변환"""
    if hasattr(value, 'item'):
        return value.item()
    elif hasattr(value, 'values'):
        return value.values[0]
    else:
        return value

def calculate_investment_strategy(rsi, sentiment):
    """RSI + 투자심리도 결합 투자전략 (6구간 분류 + 매매 액션)"""
    try:
        # RSI가 NaN인 경우
        if pd.isna(rsi):
            if sentiment >= 70:
                return "📈 보유 지속, 눌림목 매수 (심리도 기반)", "관망·부분 매수"
            elif sentiment >= 50:
                return "📊 관망, 추가 지표 확인 (심리도 기반)", "관망"
            elif sentiment >= 30:
                return "🤔 관망 또는 포지션 축소 (심리도 기반)", "관망·부분 매도"
            else:
                return "🛒 분할 매수, 역발상 진입 (심리도 기반)", "매수"
        
        # ① 과열 구간: RSI ≥ 70 & 투자심리도 ≥ 70
        if rsi >= 70 and sentiment >= 70:
            return "🚨 분할 매도·차익 실현", "매도"
        
        # ② 강세 추세 지속: RSI 50~70 & 투자심리도 ≥ 70
        elif 50 <= rsi < 70 and sentiment >= 70:
            return "📈 보유 지속, 눌림목 매수", "관망·부분 매수"
        
        # ③ 단기 급등 가능성: RSI ≥ 70 & 투자심리도 ≤ 50
        elif rsi >= 70 and sentiment <= 50:
            return "⚡ 단기 트레이딩, 손절 엄격히", "단기 매수·단기 청산"
        
        # ④ 과매도 구간: RSI ≤ 30 & 투자심리도 ≤ 30
        elif rsi <= 30 and sentiment <= 30:
            return "🛒 분할 매수, 역발상 진입", "매수"
        
        # ⑤ 저점 매집 가능성: RSI 30~50 & 투자심리도 ≤ 30
        elif 30 < rsi < 50 and sentiment <= 30:
            return "🏦 저점 매집, 추세 전환 관찰", "매수"
        
        # ⑥ 추세 피로 가능성: RSI ≤ 30 & 투자심리도 ≥ 50
        elif rsi <= 30 and sentiment >= 50:
            return "🤔 관망 또는 포지션 축소", "관망·부분 매도"
        
        # 기타 중립 구간
        else:
            return "📊 관망, 추가 지표 확인", "관망"
                
    except Exception as e:
        return "전략 계산 오류", "오류"

def calculate_investor_sentiment(data, date):
    """특정 날짜 기준 당일 이전 10일간 주가 상승일수 비율 계산 (투자심리도)"""
    try:
        # 해당 날짜 또는 가장 가까운 이전 날짜 찾기
        if date in data.index:
            current_date = date
        else:
            prev_dates = data.index[data.index <= date]
            if len(prev_dates) > 0:
                current_date = prev_dates[-1]
            else:
                return 50  # 기본값
        
        # 당일 이전 거래일 기준 10개를 찾기 위해 충분히 이전 날짜부터 시작
        start_date = current_date - timedelta(days=20)  # 충분한 여유를 두고 시작
        end_date = current_date - timedelta(days=1)  # 당일 제외
        
        # 해당 기간의 모든 거래일 데이터 추출
        period_data = data.loc[start_date:end_date]
        
        if len(period_data) < 10:
            print(f"    ⚠️  경고: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} 기간에 {len(period_data)}일 데이터 (10일 필요)")
            return 50  # 기본값
        
        # 당일 이전 거래일 기준 10개 선택 (가장 최근 10개)
        last_10_trading_days = period_data.tail(10)
        
        # 상승일 여부 계산 (당일 종가 > 전일 종가) - numpy 사용
        import numpy as np
        # numpy 배열로 변환하여 계산
        close_prices = last_10_trading_days['Close'].values
        up_days = 0
        for i in range(1, len(close_prices)):
            if close_prices[i] > close_prices[i-1]:
                up_days += 1
        
        # 투자심리도: 상승일 수 / 10 * 100 (10일은 고정)
        sentiment = (up_days / 10) * 100
        return round(sentiment, 2)
    except Exception as e:
        print(f"투자심리도 계산 오류: {e}")
        return 50  # 기본값

def analyze_stock_signals(symbol, start_date='2024-01-01', custom_signals=None):
    """지정된 종목의 HMA Mantra 매수신호 분석"""
    
    print(f"{symbol} 데이터 다운로드 중...")
    stock = yf.download(symbol, start=start_date, end=None)
    
    if stock.empty:
        print("데이터 다운로드 실패")
        return
    
    print(f"데이터 다운로드 완료: {len(stock)} 개 데이터")
    print(f"데이터 기간: {stock.index[0].strftime('%Y-%m-%d')} ~ {stock.index[-1].strftime('%Y-%m-%d')}")
    
    # RSI 계산
    stock['RSI'] = calculate_rsi(stock)
    
    # HMA Mantra 매수신호 발생일들
    if custom_signals:
        buy_signals = custom_signals
    else:
        # 기본 매수신호 날짜들 (VST 기준, 실제 사용시 수정 필요)
        buy_signals = [
            '2024-01-08', '2024-01-19', '2024-02-13', '2024-03-12', '2024-03-19',
            '2024-04-01', '2024-04-05', '2024-04-24', '2024-05-24', '2024-06-24',
            '2024-07-31', '2024-08-29', '2024-09-11', '2024-10-16', '2024-11-05',
            '2024-11-26', '2024-12-24', '2025-01-30', '2025-03-12', '2025-04-01',
            '2025-04-09', '2024-04-22', '2025-05-12', '2025-05-23', '2025-06-02',
            '2025-06-16', '2025-07-07', '2025-07-18'
        ]
    
    print(f"\n총 {len(buy_signals)}개의 매수신호 분석:")
    print("=" * 80)
    
    results = []
    
    for signal_date in buy_signals:
        try:
            # 날짜 파싱
            buy_date = pd.to_datetime(signal_date)
            
            # 해당 날짜의 데이터 확인
            if buy_date in stock.index:
                # 가격 정보 (스칼라 값으로 변환)
                close_price = get_scalar_value(stock.loc[buy_date, 'Close'])
                rsi_value = get_scalar_value(stock.loc[buy_date, 'RSI'])
                
                # 10일 전 가격
                start_date = buy_date - timedelta(days=10)
                if start_date in stock.index:
                    start_price = get_scalar_value(stock.loc[start_date, 'Close'])
                    price_change_10d = ((close_price - start_price) / start_price) * 100
                else:
                    price_change_10d = None
                
                # 투자심리도 계산 (10일간 상승일수 비율 기반)
                sentiment = calculate_investor_sentiment(stock, buy_date)
                
                # 디버깅: 투자심리도 계산 과정 출력
                start_date_debug = buy_date - timedelta(days=20)
                end_date_debug = buy_date - timedelta(days=1)
                period_data_debug = stock.loc[start_date_debug:end_date_debug]
                if len(period_data_debug) >= 10:
                    last_10_debug = period_data_debug.tail(10)
                    close_prices_debug = last_10_debug['Close'].values
                    up_days_debug = 0
                    for i in range(1, len(close_prices_debug)):
                        if close_prices_debug[i] > close_prices_debug[i-1]:
                            up_days_debug += 1
                    print(f"    🔍 투자심리도 계산: {last_10_debug.index[0].strftime('%Y-%m-%d')} ~ {last_10_debug.index[-1].strftime('%Y-%m-%d')} (거래일 10개), 상승일: {up_days_debug}/10 = {sentiment:.1f}%")
                
                # RSI 상태 분석
                if pd.isna(rsi_value):
                    rsi_status = "계산불가"
                elif rsi_value < 30:
                    rsi_status = "과매도"
                elif rsi_value > 70:
                    rsi_status = "과매수"
                else:
                    rsi_status = "중립"
                
                # 투자심리도 상태 분석 (상승일 비율 기반)
                if sentiment < 30:
                    sentiment_status = "매우 부정적"
                elif sentiment < 40:
                    sentiment_status = "부정적"
                elif sentiment < 60:
                    sentiment_status = "중립"
                elif sentiment < 70:
                    sentiment_status = "긍정적"
                else:
                    sentiment_status = "매우 긍정적"
                
                # RSI + 투자심리도 결합 투자전략 분석
                investment_strategy, trading_action = calculate_investment_strategy(rsi_value, sentiment)
                
                result = {
                    '매수신호일': signal_date,
                    '종가': f"${close_price:.2f}",
                    'RSI': f"{rsi_value:.1f}" if not pd.isna(rsi_value) else 'N/A',
                    'RSI_상태': rsi_status,
                    '투자심리도': f"{sentiment:.1f}",
                    '심리도_상태': sentiment_status,
                    '투자전략': investment_strategy,
                    '매매액션': trading_action,
                    '10일간_가격변화': f"{price_change_10d:.2f}%" if price_change_10d is not None else 'N/A'
                }
                
                results.append(result)
                
                rsi_display = f"{rsi_value:.1f}" if not pd.isna(rsi_value) else 'N/A'
                print(f"📅 {signal_date}: 종가 ${close_price:.2f} | RSI: {rsi_display} ({rsi_status}) | 투자심리도: {sentiment:.1f} ({sentiment_status}) | 투자전략: {investment_strategy} | 매매액션: {trading_action}")
                
            else:
                print(f"⚠️  {signal_date}: 데이터 없음")
                
        except Exception as e:
            print(f"❌ {signal_date}: 오류 발생 - {str(e)}")
    
    # 결과를 DataFrame으로 변환하여 CSV로 저장
    if results:
        df_results = pd.DataFrame(results)
        
        # CSV 저장
        output_file = f'{symbol.lower()}_signal_analysis_results.csv'
        df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n📊 분석 결과가 '{output_file}'에 저장되었습니다.")
        
        # 요약 통계
        print("\n" + "=" * 80)
        print("📈 요약 통계")
        print("=" * 80)
        
        # RSI 통계
        rsi_values = []
        for r in results:
            if r['RSI'] != 'N/A':
                try:
                    rsi_values.append(float(r['RSI']))
                except:
                    pass
        
        if rsi_values:
            print(f"RSI 평균: {np.mean(rsi_values):.1f}")
            print(f"RSI 최소: {np.min(rsi_values):.1f}")
            print(f"RSI 최대: {np.max(rsi_values):.1f}")
            print(f"과매도 신호 (RSI < 30): {len([r for r in results if r['RSI_상태'] == '과매도'])}회")
            print(f"과매수 신호 (RSI > 70): {len([r for r in results if r['RSI_상태'] == '과매수'])}회")
            print(f"중립 신호 (30 ≤ RSI ≤ 70): {len([r for r in results if r['RSI_상태'] == '중립'])}회")
        
        # 투자심리도 통계
        sentiment_values = []
        for r in results:
            if r['투자심리도'] != 'N/A':
                try:
                    sentiment_values.append(float(r['투자심리도']))
                except:
                    pass
        
        if sentiment_values:
            print(f"\n투자심리도 평균: {np.mean(sentiment_values):.1f}")
            print(f"투자심리도 최소: {np.min(sentiment_values):.1f}")
            print(f"투자심리도 최대: {np.max(sentiment_values):.1f}")
            print(f"매우 부정적 (심리도 < 30): {len([r for r in results if r['심리도_상태'] == '매우 부정적'])}회")
            print(f"부정적 (30 ≤ 심리도 < 40): {len([r for r in results if r['심리도_상태'] == '부정적'])}회")
            print(f"중립 (40 ≤ 심리도 < 60): {len([r for r in results if r['심리도_상태'] == '중립'])}회")
            print(f"긍정적 (60 ≤ 심리도 < 70): {len([r for r in results if r['심리도_상태'] == '긍정적'])}회")
            print(f"매우 긍정적 (심리도 ≥ 70): {len([r for r in results if r['심리도_상태'] == '매우 긍정적'])}회")
        
        # 투자전략 통계 (6구간 분류 + 매매 액션)
        if results:
            print(f"\n📊 투자전략 분포 (6구간 분류):")
            
            # 구간별 분류
            strategy_categories = {
                "🚨 과열 구간 (분할 매도)": 0,
                "📈 강세 추세 (보유 지속)": 0,
                "⚡ 단기 급등 (트레이딩)": 0,
                "🛒 과매도 (역발상 매수)": 0,
                "🏦 저점 매집": 0,
                "🤔 추세 피로 (관망)": 0,
                "📊 기타 (관망)": 0
            }
            
            for r in results:
                strategy = r['투자전략']
                if "분할 매도" in strategy:
                    strategy_categories["🚨 과열 구간 (분할 매도)"] += 1
                elif "보유 지속" in strategy:
                    strategy_categories["📈 강세 추세 (보유 지속)"] += 1
                elif "단기 트레이딩" in strategy:
                    strategy_categories["⚡ 단기 급등 (트레이딩)"] += 1
                elif "분할 매수" in strategy:
                    strategy_categories["🛒 과매도 (역발상 매수)"] += 1
                elif "저점 매집" in strategy:
                    strategy_categories["🏦 저점 매집"] += 1
                elif "관망 또는 포지션 축소" in strategy:
                    strategy_categories["🤔 추세 피로 (관망)"] += 1
                else:
                    strategy_categories["📊 기타 (관망)"] += 1
            
            for category, count in strategy_categories.items():
                if count > 0:
                    print(f"  {category}: {count}회")
            
            # 매매 액션별 분류
            print(f"\n🎯 매매 액션별 분포:")
            action_counts = {}
            for r in results:
                action = r['매매액션']
                action_counts[action] = action_counts.get(action, 0) + 1
            
            for action, count in sorted(action_counts.items()):
                print(f"  {action}: {count}회")
        
        # 10일간 가격변화 통계
        price_changes = []
        for r in results:
            if r['10일간_가격변화'] != 'N/A':
                try:
                    price_changes.append(float(r['10일간_가격변화'].replace('%', '')))
                except:
                    pass
        
        if price_changes:
            print(f"\n10일간 가격변화 통계:")
            print(f"  평균: {np.mean(price_changes):.2f}%")
            print(f"  최소: {np.min(price_changes):.2f}%")
            print(f"  최대: {np.max(price_changes):.2f}%")
            print(f"  상승: {len([p for p in price_changes if p > 0])}회")
            print(f"  하락: {len([p for p in price_changes if p < 0])}회")
        
        # 상세 결과 테이블 출력
        print(f"\n" + "=" * 80)
        print("📋 상세 결과")
        print("=" * 80)
        print(f"{'매수신호일':<12} {'종가':<10} {'RSI':<8} {'RSI상태':<8} {'투자심리도':<10} {'심리도상태':<10} {'투자전략':<25} {'매매액션':<15} {'10일변화':<10}")
        print("-" * 120)
        
        for r in results:
            print(f"{r['매수신호일']:<12} {r['종가']:<10} {r['RSI']:<8} {r['RSI_상태']:<8} {r['투자심리도']:<10} {r['심리도_상태']:<10} {r['투자전략']:<25} {r['매매액션']:<15} {r['10일간_가격변화']:<10}")
    
    else:
        print("분석할 결과가 없습니다.")

if __name__ == "__main__":
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser(description='HMA Mantra 매수신호 분석 도구')
    parser.add_argument('symbol', help='분석할 종목 심볼 (예: AAPL, TSLA, VST)')
    parser.add_argument('--start', default='2024-01-01', help='분석 시작 날짜 (기본값: 2024-01-01)')
    parser.add_argument('--signals', help='매수신호 날짜들을 쉼표로 구분하여 입력 (예: 2024-01-08,2024-01-19)')
    
    args = parser.parse_args()
    
    print(f"{args.symbol} HMA Mantra 매수신호 분석 - 투자심리도 및 RSI 계산")
    print("=" * 80)
    
    # 매수신호 날짜가 입력된 경우 사용, 아니면 기본값 사용
    if args.signals:
        # 사용자 입력 매수신호 날짜 파싱
        buy_signals = [date.strip() for date in args.signals.split(',')]
        print(f"사용자 입력 매수신호 날짜: {buy_signals}")
    else:
        print("기본 매수신호 날짜 사용 (실제 사용시 --signals 옵션으로 수정 필요)")
    
    analyze_stock_signals(args.symbol, args.start, buy_signals if args.signals else None)
