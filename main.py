#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
주식 기술적 분석 시스템
"""

import sys
from src.data_fetcher import DataFetcher
from src.technical_analysis import TechnicalAnalysis
from src.visualization import StockVisualizer
from src.utils import setup_argparser, create_output_dir, print_analysis_summary, is_valid_ticker

def main():
    """메인 함수"""
    # 커맨드라인 인자 파싱
    parser = setup_argparser()
    args = parser.parse_args()
    
    # 종목코드 검증
    if not is_valid_ticker(args.ticker):
        print(f"ERROR: 유효하지 않은 종목코드입니다: {args.ticker}")
        sys.exit(1)
    
    # 상태 메시지
    print(f"분석 중: {args.ticker} ({args.period}, {args.interval})")
    
    # 1. 데이터 수집
    print("1. 데이터 수집 중...")
    fetcher = DataFetcher()
    
    # 날짜 범위가 설정된 경우 해당 범위로 데이터 수집
    if args.start_date and args.end_date:
        print(f"날짜 범위: {args.start_date} ~ {args.end_date}")
        data = fetcher.fetch_data(args.ticker, args.period, args.interval, 
                                 start_date=args.start_date, end_date=args.end_date)
    else:
        data = fetcher.fetch_data(args.ticker, args.period, args.interval)
    
    if data is None or data.empty:
        print("ERROR: 데이터를 가져올 수 없습니다.")
        sys.exit(1)
    
    print(f"  - {len(data)}개의 데이터 포인트 수집됨")
    
    # 2. 기술적 지표 계산
    print("2. 기술적 지표 계산 중...")
    analyzer = TechnicalAnalysis(data)
    data = analyzer.calculate_all_indicators()
    
    # 추세 분석 추가
    analyzer.add_trend_analysis()
    
    # 매수 신호 분석
    buy_signals = analyzer.find_buy_signals(window=10)
    
    # 매도 신호 분석 추가
    sell_signals = analyzer.find_sell_signals(window=10)
    
    # 3. 결과 저장
    output_dir = args.output if args.output else create_output_dir(args.ticker)
    print(f"3. 데이터 저장 중... ({output_dir})")
    fetcher.save_data(output_dir)
    
    # 4. 결과 출력
    print("4. 분석 결과 요약:")
    print_analysis_summary(data, args.ticker)
    
    # 매수 신호 출력 (매도 전용 모드가 아닌 경우에만)
    if not args.sell_only:
        print("\n5. 매수 신호 분석 결과:")
        print("==================================================")
        print(f"종목: {args.ticker} 매수 신호 분석")
        print("==================================================")
        print(f"최근 10일간 매수 신호 점수:")
        
        for idx, row in buy_signals.iterrows():
            date_str = idx.strftime('%Y-%m-%d')
            score = row['Buy_Score']
            signal = row['Buy_Signal']
            print(f"- {date_str}: {score:.1f}점 ({signal})")
        
        # 최근 매수 추천
        latest_score = buy_signals['Buy_Score'].iloc[-1]
        latest_signal = buy_signals['Buy_Signal'].iloc[-1]
        
        print("\n최종 추천:")
        if latest_score >= 70:
            print("✅ 강력 매수 추천")
        elif latest_score >= 50:
            print("✅ 매수 추천")
        elif latest_score >= 30:
            print("⚠️ 관망 추천")
        else:
            print("❌ 매수 비추천")
        
        print("==================================================")
        print("* 참고: 이 분석은 단순 참고용이며, 투자 결정은 추가적인 연구가 필요합니다.")
        print("==================================================")
    
    # 매도 신호 출력 (매수 전용 모드가 아닌 경우에만)
    if not args.buy_only:
        print("\n6. 매도 신호 분석 결과:")
        print("==================================================")
        print(f"종목: {args.ticker} 매도 신호 분석")
        print("==================================================")
        print(f"최근 10일간 매도 신호 점수:")
        
        for idx, row in sell_signals.iterrows():
            date_str = idx.strftime('%Y-%m-%d')
            score = row['Sell_Score']
            signal = row['Sell_Signal']
            print(f"- {date_str}: {score:.1f}점 ({signal})")
        
        # 최근 매도 추천
        latest_sell_score = sell_signals['Sell_Score'].iloc[-1]
        latest_sell_signal = sell_signals['Sell_Signal'].iloc[-1]
        
        print("\n최종 매도 추천:")
        if latest_sell_score >= 70:
            print("⛔ 강력 매도 추천")
        elif latest_sell_score >= 50:
            print("⛔ 매도 추천")
        elif latest_sell_score >= 30:
            print("⚠️ 일부 매도 검토")
        else:
            print("✅ 매도 비추천")
        
        print("==================================================")
        print("* 참고: 이 분석은 단순 참고용이며, 투자 결정은 추가적인 연구가 필요합니다.")
        print("==================================================")
    
    # 시각화 (필요한 경우)
    if args.visualize:
        print("7. 차트 생성 중...")
        
        # 박스권(횡보구간) 분석
        consolidation_ranges = analyzer.find_consolidation_ranges(
            window=args.box_window, 
            threshold=args.box_threshold
        )
        if consolidation_ranges:
            print(f"총 {len(consolidation_ranges)}개의 박스권 구간이 발견되었습니다.")
            for box in consolidation_ranges:
                start_date = box['start_date'].strftime('%Y-%m-%d')
                end_date = box['end_date'].strftime('%Y-%m-%d')
                range_percent = box['range_percent'] * 100
                print(f"  - {start_date} ~ {end_date}: 변동폭 {range_percent:.1f}% (상단: {box['upper_price']:.2f}, 하단: {box['lower_price']:.2f})")
        
        # 매도 신호 분석 모드인 경우 매도 차트만 생성
        if args.sell_only:
            visualizer = StockVisualizer(data, args.ticker)
            visualizer.set_consolidation_ranges(consolidation_ranges)
            visualizer.plot_all_sell_signals()
            chart_path = visualizer.save_sell_chart(output_dir)
            print(f"매도 신호 차트가 {output_dir} 디렉토리에 저장되었습니다.")
        # 매수 신호 분석 모드인 경우 매수 차트만 생성
        elif args.buy_only:
            visualizer = StockVisualizer(data, args.ticker)
            visualizer.set_consolidation_ranges(consolidation_ranges)
            visualizer.plot_all()
            chart_path = visualizer.save_chart(output_dir)
            print(f"매수 신호 차트가 {output_dir} 디렉토리에 저장되었습니다.")
        # 일반 분석 모드인 경우 매수/매도 차트 모두 생성
        else:
            # 매수 신호 차트
            visualizer = StockVisualizer(data, args.ticker)
            visualizer.set_consolidation_ranges(consolidation_ranges)
            visualizer.plot_all()
            visualizer.save_chart(output_dir)
            
            # 매도 신호 차트 (별도 파일로 저장)
            visualizer = StockVisualizer(data, args.ticker)
            visualizer.set_consolidation_ranges(consolidation_ranges)
            visualizer.plot_all_sell_signals()
            visualizer.save_sell_chart(output_dir)
    
    print("\n완료!")
    print(f"결과는 {output_dir} 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main() 