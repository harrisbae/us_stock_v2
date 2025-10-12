#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
엘리엇 파동 분석 테스트 스크립트

사용법:
    python test/elliott_wave_test.py AAPL 120d
    python test/elliott_wave_test.py TSLA --from 2024-01-01 --to 2025-01-01
    python test/elliott_wave_test.py SPY 1y --min-wave-size 3 --zigzag-threshold 5
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse

from src.indicators.hma_mantra.elliott_wave import ElliottWaveAnalyzer, analyze_elliott_wave
from src.indicators.hma_mantra.visualization.elliott_wave_simple import plot_elliott_wave_simple
from src.indicators.hma_mantra.visualization.elliott_wave_plotly import create_elliott_wave_chart


def parse_period(period_str: str) -> tuple:
    """
    기간 문자열 파싱
    
    Args:
        period_str: 기간 문자열 (예: '120d', '6mo', '1y', '2024-01-01_2025-01-01')
        
    Returns:
        (start_date, end_date) 튜플
    """
    # 날짜 범위 형식인 경우
    if '_' in period_str:
        dates = period_str.split('_')
        if len(dates) == 2:
            start_date = datetime.strptime(dates[0], '%Y-%m-%d')
            end_date = datetime.strptime(dates[1], '%Y-%m-%d')
            return start_date, end_date
    
    # 상대 기간 형식인 경우
    end_date = datetime.now()
    
    if period_str.endswith('d'):
        days = int(period_str[:-1])
        start_date = end_date - timedelta(days=days)
    elif period_str.endswith('mo'):
        months = int(period_str[:-2])
        start_date = end_date - timedelta(days=months * 30)
    elif period_str.endswith('y'):
        years = int(period_str[:-1])
        start_date = end_date - timedelta(days=years * 365)
    else:
        # 기본값: 120일
        start_date = end_date - timedelta(days=120)
    
    return start_date, end_date


def print_analysis_report(result: dict, ticker: str):
    """
    분석 결과 리포트 출력
    
    Args:
        result: 분석 결과
        ticker: 종목 코드
    """
    print("\n" + "=" * 60)
    print(f"  엘리엇 파동 분석: {ticker}")
    print("=" * 60)
    
    if not result.get('success'):
        print(f"\n⚠️  {result.get('message', '분석 실패')}")
        print("=" * 60)
        return
    
    # 기본 정보
    print(f"\n✅ 분석 성공!")
    print(f"신뢰도: {result['confidence']:.1f}%")
    print(f"감지된 피봇 포인트: {len(result['pivots'])}개")
    print(f"감지된 파동: {len(result['waves'])}개")
    
    # 파동 상세 정보
    print("\n" + "-" * 60)
    print("📊 감지된 파동:")
    print("-" * 60)
    
    for i, wave in enumerate(result['waves'], 1):
        print(f"\n파동 #{i}:")
        print(f"  유형: {'충격파 (Impulse)' if wave['type'] == 'impulse' else '조정파 (Corrective)'}")
        
        if wave['type'] == 'impulse':
            print(f"  방향: {'상승' if wave['direction'] == 'up' else '하락'}")
            gain_pct = float(wave['gain_pct']) if hasattr(wave['gain_pct'], '__float__') else wave['gain_pct']
            print(f"  변동폭: {gain_pct:+.2f}%")
            print(f"  기간: {wave['start_date'].strftime('%Y-%m-%d')} ~ {wave['end_date'].strftime('%Y-%m-%d')}")
            start_price = float(wave['start_price']) if hasattr(wave['start_price'], '__float__') else wave['start_price']
            end_price = float(wave['end_price']) if hasattr(wave['end_price'], '__float__') else wave['end_price']
            print(f"  가격: ${start_price:.2f} → ${end_price:.2f}")
            
            # 파동별 상세 정보
            waves_data = wave['waves']
            print(f"\n  파동 상세:")
            
            # 각 가격을 float로 변환
            w0 = float(waves_data['wave_0']['price']) if hasattr(waves_data['wave_0']['price'], '__float__') else waves_data['wave_0']['price']
            w1 = float(waves_data['wave_1']['price']) if hasattr(waves_data['wave_1']['price'], '__float__') else waves_data['wave_1']['price']
            w2 = float(waves_data['wave_2']['price']) if hasattr(waves_data['wave_2']['price'], '__float__') else waves_data['wave_2']['price']
            w3 = float(waves_data['wave_3']['price']) if hasattr(waves_data['wave_3']['price'], '__float__') else waves_data['wave_3']['price']
            w4 = float(waves_data['wave_4']['price']) if hasattr(waves_data['wave_4']['price'], '__float__') else waves_data['wave_4']['price']
            w5 = float(waves_data['wave_5']['price']) if hasattr(waves_data['wave_5']['price'], '__float__') else waves_data['wave_5']['price']
            
            print(f"    파동 1: ${w0:.2f} → ${w1:.2f}")
            print(f"    파동 2: ${w1:.2f} → ${w2:.2f}")
            print(f"    파동 3: ${w2:.2f} → ${w3:.2f}")
            print(f"    파동 4: ${w3:.2f} → ${w4:.2f}")
            print(f"    파동 5: ${w4:.2f} → ${w5:.2f}")
            
            # 피보나치 비율
            validation = wave.get('validation', {})
            fib_ratios = validation.get('fibonacci_ratios', {})
            if fib_ratios:
                print(f"\n  피보나치 비율:")
                w2_ret = fib_ratios.get('wave_2_retracement', 0)
                w2_ret_val = float(w2_ret) if hasattr(w2_ret, '__float__') else w2_ret
                print(f"    파동 2 되돌림: {w2_ret_val:.2%}")
                
                w3_ext = fib_ratios.get('wave_3_extension', 0)
                w3_ext_val = float(w3_ext) if hasattr(w3_ext, '__float__') else w3_ext
                print(f"    파동 3 확장: {w3_ext_val:.2f}x")
                
                w4_ret = fib_ratios.get('wave_4_retracement', 0)
                w4_ret_val = float(w4_ret) if hasattr(w4_ret, '__float__') else w4_ret
                print(f"    파동 4 되돌림: {w4_ret_val:.2%}")
        
        else:  # corrective
            print(f"  패턴: {wave['pattern']}")
            retracement_pct = float(wave['retracement_pct']) if hasattr(wave['retracement_pct'], '__float__') else wave['retracement_pct']
            print(f"  되돌림: {retracement_pct:.2f}%")
            print(f"  기간: {wave['start_date'].strftime('%Y-%m-%d')} ~ {wave['end_date'].strftime('%Y-%m-%d')}")
            start_price = float(wave['start_price']) if hasattr(wave['start_price'], '__float__') else wave['start_price']
            end_price = float(wave['end_price']) if hasattr(wave['end_price'], '__float__') else wave['end_price']
            print(f"  가격: ${start_price:.2f} → ${end_price:.2f}")
    
    # 현재 위치
    current_pos = result.get('current_position')
    if current_pos:
        print("\n" + "-" * 60)
        print("🎯 현재 상태:")
        print("-" * 60)
        print(f"현재 위치: {current_pos['message']}")
        current_price = float(current_pos['current_price']) if hasattr(current_pos['current_price'], '__float__') else current_pos['current_price']
        print(f"현재가: ${current_price:.2f}")
        print(f"날짜: {current_pos['current_date'].strftime('%Y-%m-%d')}")
    
    # 목표가
    targets = result.get('targets', {})
    if targets:
        print("\n" + "-" * 60)
        print("📈 목표가:")
        print("-" * 60)
        
        if 'retracement' in targets:
            print("조정 목표가 (되돌림):")
            for level, price in targets['retracement'].items():
                price_val = float(price) if hasattr(price, '__float__') else price
                print(f"  {level}: ${price_val:.2f}")
        
        if 'extension' in targets:
            print("상승 목표가 (확장):")
            for level, price in targets['extension'].items():
                price_val = float(price) if hasattr(price, '__float__') else price
                print(f"  {level}: ${price_val:.2f}")
    
    # 추천 사항
    print("\n" + "-" * 60)
    print("💡 투자 참고사항:")
    print("-" * 60)
    
    confidence = result['confidence']
    if confidence >= 70:
        print("✅ 높은 신뢰도 - 엘리엇 파동 패턴이 명확하게 감지되었습니다.")
    elif confidence >= 50:
        print("⚠️  중간 신뢰도 - 패턴이 감지되었으나 추가 검증이 필요합니다.")
    else:
        print("❌ 낮은 신뢰도 - 패턴이 불명확합니다. 다른 지표와 함께 확인하세요.")
    
    if current_pos:
        position = current_pos.get('position', '')
        if position == 'wave_5':
            print("⚠️  파동 5 진행 중 - 종료 신호를 주의 깊게 관찰하세요.")
            print("   (거래량 감소, RSI 과매수, 다이버전스 등)")
        elif position == 'after_wave_5':
            print("📉 조정파 예상 - 매도 타이밍을 고려하거나 조정 후 재진입을 계획하세요.")
        elif position == 'corrective':
            print("📊 조정파 진행 중 - 지지선에서 매수 기회를 찾을 수 있습니다.")
    
    print("\n⚠️  면책 조항: 이 분석은 참고용이며, 투자 결정의 유일한 근거로 사용하지 마세요.")
    print("=" * 60)


def main():
    """메인 함수"""
    # 인자 파싱
    parser = argparse.ArgumentParser(description='엘리엇 파동 분석')
    parser.add_argument('ticker', type=str, help='종목 코드 (예: AAPL, TSLA)')
    parser.add_argument('period', type=str, nargs='?', default='120d',
                       help='기간 (예: 120d, 6mo, 1y)')
    parser.add_argument('--from', dest='from_date', type=str,
                       help='시작 날짜 (YYYY-MM-DD)')
    parser.add_argument('--to', dest='to_date', type=str,
                       help='종료 날짜 (YYYY-MM-DD)')
    parser.add_argument('--min-wave-size', type=float, default=3.0,
                       help='최소 파동 크기 (%%, 기본값: 3.0)')
    parser.add_argument('--zigzag-threshold', type=float, default=5.0,
                       help='ZigZag 임계값 (%%, 기본값: 5.0)')
    parser.add_argument('--no-chart', action='store_true',
                       help='차트 생성 안함 (분석만)')
    parser.add_argument('--auto_adjust', type=str, default='false',
                       help='가격 조정 여부 (true/false)')
    
    args = parser.parse_args()
    
    ticker = args.ticker.upper()
    
    print(f"\n엘리엇 파동 분석 시작: {ticker}")
    print("-" * 60)
    
    # 날짜 범위 계산
    if args.from_date and args.to_date:
        start_date = datetime.strptime(args.from_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.to_date, '%Y-%m-%d')
        print(f"기간: {args.from_date} ~ {args.to_date}")
    elif args.from_date:
        start_date = datetime.strptime(args.from_date, '%Y-%m-%d')
        end_date = datetime.now()
        print(f"기간: {args.from_date} ~ 현재")
    else:
        start_date, end_date = parse_period(args.period)
        print(f"기간: {args.period}")
    
    print(f"최소 파동 크기: {args.min_wave_size}%")
    print(f"ZigZag 임계값: {args.zigzag_threshold}%")
    print("-" * 60)
    
    # 데이터 다운로드
    print("\n📥 데이터 다운로드 중...")
    try:
        auto_adjust = args.auto_adjust.lower() == 'true'
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=auto_adjust)
        
        if data.empty:
            print(f"❌ 오류: {ticker} 데이터를 가져올 수 없습니다.")
            sys.exit(1)
        
        print(f"✅ {len(data)}개의 데이터 포인트 다운로드 완료")
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        sys.exit(1)
    
    # 엘리엇 파동 분석
    print("\n🔍 엘리엇 파동 분석 중...")
    try:
        analyzer = ElliottWaveAnalyzer(
            data,
            min_wave_size=args.min_wave_size,
            zigzag_threshold=args.zigzag_threshold
        )
        result = analyzer.analyze()
        
    except Exception as e:
        print(f"❌ 분석 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 결과 출력
    print_analysis_report(result, ticker)
    
    # 출력 디렉토리 생성
    output_dir = f"output/hma_mantra/{ticker}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 결과 저장 (JSON)
    import json
    result_file = os.path.join(output_dir, f"{ticker}_elliott_wave_analysis.json")
    
    # 날짜를 문자열로 변환 (JSON 직렬화를 위해)
    def convert_dates(obj):
        if isinstance(obj, dict):
            return {k: convert_dates(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_dates(item) for item in obj]
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d')
        elif isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d')
        elif isinstance(obj, np.ndarray):
            # numpy 배열을 리스트로 변환
            return obj.tolist()
        elif hasattr(obj, '__float__'):
            # numpy 스칼라를 Python float로 변환
            return float(obj)
        elif hasattr(obj, '__int__'):
            # numpy 스칼라를 Python int로 변환
            return int(obj)
        else:
            return obj
    
    result_serializable = convert_dates(result)
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 분석 결과 저장: {result_file}")
    
    # 차트 생성 (matplotlib + Plotly)
    if not args.no_chart:
        period_str = f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
        
        # 1. matplotlib 차트 (명확하고 간단)
        print("\n📊 차트 생성 중 (matplotlib)...")
        try:
            png_path = os.path.join(output_dir,
                                   f"{ticker}_elliott_wave_{period_str}_chart.png")
            
            plot_elliott_wave_simple(
                data=data,
                analysis_result=result,
                ticker=ticker,
                save_path=png_path
            )
            
            print(f"✅ PNG 차트 저장 완료: {png_path}")
            
        except Exception as e:
            print(f"⚠️  PNG 차트 생성 오류: {e}")
            import traceback
            traceback.print_exc()
        
        # 2. Plotly 인터랙티브 차트 (HTML)
        print("\n📊 인터랙티브 차트 생성 중 (Plotly)...")
        try:
            html_path = os.path.join(output_dir,
                                    f"{ticker}_elliott_wave_{period_str}_interactive.html")
            
            create_elliott_wave_chart(
                data=data,
                analysis_result=result,
                ticker=ticker,
                save_html=html_path,
                save_png=None,  # matplotlib에서 이미 PNG 생성
                height=1000,
                show_chart=False
            )
            
            print(f"✅ HTML 차트 저장 완료: {html_path}")
            print(f"💡 HTML 파일을 브라우저로 열면 인터랙티브 기능을 사용할 수 있습니다!")
            
        except Exception as e:
            print(f"⚠️  HTML 차트 생성 오류: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ 엘리엇 파동 분석 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()

