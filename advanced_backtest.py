#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
고급 백테스팅 시스템
다양한 전략으로 백테스팅을 수행하고 결과를 분석합니다.
"""

import sys
import argparse
from src.data_fetcher import DataFetcher
from src.technical_analysis import TechnicalAnalysis
from src.backtest_engine import AdvancedBacktestEngine
from src.utils import setup_argparser, create_output_dir
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def setup_backtest_parser():
    """백테스팅용 인자 파서 설정"""
    parser = argparse.ArgumentParser(description='고급 백테스팅 시스템')
    
    parser.add_argument('ticker', help='종목 코드 (예: AAPL, 005930.KS)')
    parser.add_argument('--strategy', '-s', 
                       choices=['momentum', 'mean_reversion', 'breakout', 'multi_factor'],
                       default='multi_factor',
                       help='백테스팅 전략 선택')
    parser.add_argument('--period', '-p', default='1y', help='데이터 기간')
    parser.add_argument('--interval', '-i', default='1d', help='데이터 간격')
    parser.add_argument('--capital', '-c', type=float, default=100000,
                       help='초기 자본금 (기본값: 100,000)')
    parser.add_argument('--output', '-o', help='출력 디렉토리')
    parser.add_argument('--visualize', '-v', action='store_true', help='차트 생성')
    parser.add_argument('--start-date', help='시작 날짜 (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='종료 날짜 (YYYY-MM-DD)')
    
    # 전략별 매개변수
    parser.add_argument('--rsi-period', type=int, default=14, help='RSI 기간')
    parser.add_argument('--rsi-oversold', type=float, default=30, help='RSI 과매도 기준')
    parser.add_argument('--rsi-overbought', type=float, default=70, help='RSI 과매수 기준')
    parser.add_argument('--ma-short', type=int, default=20, help='단기 이동평균 기간')
    parser.add_argument('--ma-long', type=int, default=50, help='장기 이동평균 기간')
    parser.add_argument('--stop-loss', type=float, default=0.05, help='손절 비율')
    parser.add_argument('--take-profit', type=float, default=0.15, help='익절 비율')
    
    return parser

def run_backtest(args):
    """백테스팅 실행"""
    print(f"=== {args.ticker} 고급 백테스팅 시작 ===")
    print(f"전략: {args.strategy}")
    print(f"초기 자본: ${args.capital:,.0f}")
    print(f"기간: {args.period}")
    print("=" * 50)
    
    # 1. 데이터 수집
    print("1. 데이터 수집 중...")
    fetcher = DataFetcher()
    
    if args.start_date and args.end_date:
        print(f"날짜 범위: {args.start_date} ~ {args.end_date}")
        data = fetcher.fetch_data(args.ticker, args.period, args.interval,
                                 start_date=args.start_date, end_date=args.end_date)
    else:
        data = fetcher.fetch_data(args.ticker, args.period, args.interval)
    
    if data is None or data.empty:
        print("ERROR: 데이터를 가져올 수 없습니다.")
        return None
    
    print(f"  - {len(data)}개의 데이터 포인트 수집됨")
    print(f"  - 기간: {data.index[0].strftime('%Y-%m-%d')} ~ {data.index[-1].strftime('%Y-%m-%d')}")
    
    # 2. 기술적 지표 계산
    print("2. 기술적 지표 계산 중...")
    analyzer = TechnicalAnalysis(data)
    data = analyzer.calculate_all_indicators()
    analyzer.add_advanced_indicators()
    
    print("  - 기본 지표 계산 완료")
    print("  - 고급 지표 계산 완료")
    
    # 3. 백테스팅 엔진 초기화
    print("3. 백테스팅 엔진 초기화...")
    backtest_engine = AdvancedBacktestEngine(data, args.capital)
    
    # 4. 백테스팅 실행
    print(f"4. {args.strategy} 전략 백테스팅 실행 중...")
    
    # 전략별 매개변수 설정
    strategy_params = {
        'rsi_period': args.rsi_period,
        'rsi_oversold': args.rsi_oversold,
        'rsi_overbought': args.rsi_overbought,
        'ma_short': args.ma_short,
        'ma_long': args.ma_long,
        'stop_loss': args.stop_loss,
        'take_profit': args.take_profit
    }
    
    try:
        results = backtest_engine.run_strategy(args.strategy, **strategy_params)
        print("  - 백테스팅 완료")
    except Exception as e:
        print(f"ERROR: 백테스팅 실행 중 오류 발생: {e}")
        return None
    
    # 5. 결과 출력
    print("\n5. 백테스팅 결과:")
    print("=" * 50)
    print(f"초기 자본: ${results['initial_capital']:,.0f}")
    print(f"최종 자본: ${results['final_capital']:,.0f}")
    print(f"총 수익률: {results['total_return']:.2%}")
    print(f"연간 수익률: {results['annualized_return']:.2%}")
    print(f"샤프 비율: {results['sharpe_ratio']:.2f}")
    print(f"최대 낙폭: {results['max_drawdown']:.2%}")
    print(f"총 거래: {results['total_trades']}")
    print(f"승률: {results['win_rate']:.2%}")
    print("=" * 50)
    
    # 6. 결과 저장
    output_dir = args.output if args.output else create_output_dir(f"{args.ticker}_backtest")
    print(f"\n6. 결과 저장 중... ({output_dir})")
    
    # 거래 기록 저장
    if results['trades']:
        trades_df = pd.DataFrame(results['trades'])
        trades_df.to_csv(f"{output_dir}/trades.csv", index=False)
        print(f"  - 거래 기록: {output_dir}/trades.csv")
    
    # 포트폴리오 가치 저장
    portfolio_df = pd.DataFrame({
        'Date': data.index,
        'Portfolio_Value': results['portfolio_values']
    })
    portfolio_df.to_csv(f"{output_dir}/portfolio_values.csv", index=False)
    print(f"  - 포트폴리오 가치: {output_dir}/portfolio_values.csv")
    
    # 7. 시각화
    if args.visualize:
        print("7. 차트 생성 중...")
        try:
            fig = backtest_engine.plot_results(results, f"{output_dir}/backtest_results.png")
            print(f"  - 백테스트 결과 차트: {output_dir}/backtest_results.png")
        except Exception as e:
            print(f"  - 차트 생성 중 오류: {e}")
    
    # 8. 상세 분석 리포트 생성
    print("8. 상세 분석 리포트 생성 중...")
    generate_backtest_report(results, args, output_dir)
    
    print("\n=== 백테스팅 완료 ===")
    print(f"결과는 {output_dir} 디렉토리에 저장되었습니다.")
    
    return results

def generate_backtest_report(results, args, output_dir):
    """백테스트 상세 리포트 생성"""
    report_content = f"""# {args.ticker} 백테스트 결과 리포트

## 기본 정보
- **종목 코드**: {args.ticker}
- **전략**: {args.strategy}
- **백테스트 기간**: {args.period}
- **초기 자본**: ${results['initial_capital']:,.0f}

## 성과 요약
- **최종 자본**: ${results['final_capital']:,.0f}
- **총 수익률**: {results['total_return']:.2%}
- **연간 수익률**: {results['annualized_return']:.2%}
- **샤프 비율**: {results['sharpe_ratio']:.2f}
- **최대 낙폭**: {results['max_drawdown']:.2%}

## 거래 통계
- **총 거래 수**: {results['total_trades']}
- **승리 거래**: {results['winning_trades']}
- **승률**: {results['win_rate']:.2%}

## 전략 매개변수
- **RSI 기간**: {args.rsi_period}
- **RSI 과매도**: {args.rsi_oversold}
- **RSI 과매수**: {args.rsi_overbought}
- **단기 이동평균**: {args.ma_short}
- **장기 이동평균**: {args.ma_long}
- **손절 비율**: {args.stop_loss:.1%}
- **익절 비율**: {args.take_profit:.1%}

## 분석 및 권장사항
이 백테스트 결과는 과거 데이터를 기반으로 한 것으로, 미래 성과를 보장하지 않습니다.
투자 결정 시 추가적인 분석과 리스크 관리가 필요합니다.

---
*생성일시: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(f"{output_dir}/backtest_report.md", 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"  - 상세 리포트: {output_dir}/backtest_report.md")

def main():
    """메인 함수"""
    parser = setup_backtest_parser()
    args = parser.parse_args()
    
    # 백테스팅 실행
    results = run_backtest(args)
    
    if results is None:
        sys.exit(1)

if __name__ == "__main__":
    main()
