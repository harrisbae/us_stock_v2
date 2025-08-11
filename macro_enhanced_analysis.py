#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
매크로 지표 기반 매수 신호 개선 시스템
VIX, High Yield Spread, NAIIM을 활용하여 매수 오류를 줄입니다.
"""

import sys
import argparse
from src.data_fetcher import DataFetcher
from src.technical_analysis import TechnicalAnalysis
from src.analysis.macro.macro_analysis import MacroAnalysis
from src.utils import setup_argparser, create_output_dir
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def setup_macro_parser():
    """매크로 분석용 인자 파서 설정"""
    parser = argparse.ArgumentParser(description='매크로 지표 기반 매수 신호 개선 시스템')
    
    parser.add_argument('ticker', help='종목 코드 (예: AAPL, 005930.KS)')
    parser.add_argument('--period', '-p', default='1y', help='데이터 기간')
    parser.add_argument('--interval', '-i', default='1d', help='데이터 간격')
    parser.add_argument('--output', '-o', help='출력 디렉토리')
    parser.add_argument('--visualize', '-v', action='store_true', help='차트 생성')
    parser.add_argument('--start-date', help='시작 날짜 (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='종료 날짜 (YYYY-MM-DD)')
    
    # 매크로 분석 옵션
    parser.add_argument('--macro-weight', type=float, default=0.3, 
                       help='매크로 점수의 가중치 (0.0-1.0, 기본값: 0.3)')
    parser.add_argument('--technical-weight', type=float, default=0.7,
                       help='기술적 점수의 가중치 (0.0-1.0, 기본값: 0.7)')
    parser.add_argument('--vix-threshold', type=float, default=25.0,
                       help='VIX 매수 임계값 (기본값: 25.0)')
    parser.add_argument('--spread-threshold', type=float, default=0.02,
                       help='HY Spread 매수 임계값 (기본값: 0.02)')
    parser.add_argument('--naiim-threshold', type=float, default=40.0,
                       help='NAIIM 매수 임계값 (기본값: 40.0)')
    
    return parser

def run_macro_enhanced_analysis(args):
    """매크로 지표 기반 매수 신호 개선 분석 실행"""
    print(f"=== {args.ticker} 매크로 지표 기반 분석 시작 ===")
    print(f"매크로 가중치: {args.macro_weight:.1%}, 기술적 가중치: {args.technical_weight:.1%}")
    print("=" * 50)
    
    # 1. 주식 데이터 수집
    fetcher = DataFetcher()
    
    if args.start_date and args.end_date:
        stock_data = fetcher.fetch_data(args.ticker, args.period, args.interval,
                                       start_date=args.start_date, end_date=args.end_date)
    else:
        stock_data = fetcher.fetch_data(args.ticker, args.period, args.interval)
    
    if stock_data is None or stock_data.empty:
        print("ERROR: 주식 데이터를 가져올 수 없습니다.")
        return None
    
    print(f"데이터 수집 완료: {len(stock_data)}개 포인트")
    
    # 2. 기술적 지표 계산
    analyzer = TechnicalAnalysis(stock_data)
    stock_data = analyzer.calculate_all_indicators()
    analyzer.add_advanced_indicators()
    
    # 매수 신호 찾기
    buy_signals = analyzer.find_buy_signals(window=10)
    
    print(f"기술적 분석 완료: {len(buy_signals)}개 매수 신호")
    
    # 3. 매크로 지표 데이터 수집
    macro_analyzer = MacroAnalysis()
    macro_analyzer.stock_data = stock_data
    
    start_date = args.start_date if args.start_date else (stock_data.index[0] - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = args.end_date if args.end_date else stock_data.index[-1].strftime('%Y-%m-%d')
    
    # VIX 데이터 수집
    vix_data = macro_analyzer.fetch_vix_data(start_date, end_date)
    spread_data = macro_analyzer.fetch_high_yield_spread_data(start_date, end_date)
    naiim_data = macro_analyzer.fetch_naiim_data(start_date, end_date)
    
    if vix_data is not None and spread_data is not None and naiim_data is not None:
        print("매크로 지표 수집 완료")
        
        # 매수 신호가 발생한 날짜들 추출
        buy_signal_dates = buy_signals[buy_signals['Buy_Signal'] == True].index
        
        if len(buy_signal_dates) > 0:
            macro_analysis_result = macro_analyzer.analyze_macro_correlation(buy_signals)
            
            if macro_analysis_result is not None:
                macro_df, correlation_analysis = macro_analysis_result
                
                # 상관관계 결과 간단 출력
                if 'error' not in correlation_analysis:
                    print(f"상관관계 분석 완료 (샘플: {correlation_analysis['sample_size']}개)")
                else:
                    print("상관관계 분석 실패")
            else:
                print("매크로 상관관계 분석 실패")
        else:
            print("분석할 매수 신호가 없습니다.")
    else:
        print("매크로 데이터 수집 실패")
    
    # 5. 개선된 매수 신호 생성
    enhanced_signals = create_enhanced_signals(
        stock_data, buy_signals, macro_analyzer, 
        args.macro_weight, args.technical_weight,
        args.vix_threshold, args.spread_threshold, args.naiim_threshold
    )
    
    if enhanced_signals is not None:
        print(f"개선된 매수 신호 생성 완료: {len(enhanced_signals)}개")
    
    # 6. 결과 저장
    output_dir = args.output if args.output else create_output_dir(f"{args.ticker}_macro_enhanced")
    
    # 개선된 매수 신호 저장
    if enhanced_signals is not None:
        enhanced_signals.to_csv(f"{output_dir}/enhanced_buy_signals.csv", encoding='utf-8-sig')
    
    # 매크로 분석 결과 저장
    if 'macro_df' in locals() and macro_df is not None:
        macro_df.to_csv(f"{output_dir}/macro_analysis.csv", encoding='utf-8-sig')
    
    # 7. 시각화 (조용히)
    if args.visualize:
        try:
            # 매크로 분석 차트
            if 'macro_df' in locals() and macro_df is not None:
                macro_analyzer.plot_macro_analysis(macro_df, f"{output_dir}/macro_analysis.png")
            
            # 개선된 매수 신호 차트
            if enhanced_signals is not None:
                plot_enhanced_signals(stock_data, enhanced_signals, args.ticker, output_dir)
                
        except Exception as e:
            pass  # 시각화 오류는 조용히 무시
    
    # 8. 상세 분석 리포트 생성 (조용히)
    try:
        generate_macro_analysis_report(
            args, enhanced_signals, macro_df if 'macro_df' in locals() else None, 
            correlation_analysis if 'correlation_analysis' in locals() else None, 
            output_dir
        )
    except Exception as e:
        pass  # 리포트 생성 오류는 조용히 무시
    
    print(f"\n=== 분석 완료 === 결과 저장: {output_dir}")
    
    return enhanced_signals

def create_enhanced_signals(stock_data, buy_signals, macro_analyzer, 
                           macro_weight, technical_weight,
                           vix_threshold, spread_threshold, naiim_threshold):
    """매크로 지표를 반영한 개선된 매수 신호 생성"""
    try:
        enhanced_signals = buy_signals.copy()
        
        # 매크로 점수 컬럼 추가
        enhanced_signals['Macro_Score'] = 0.0
        enhanced_signals['Enhanced_Score'] = 0.0
        enhanced_signals['Macro_Recommendation'] = ''
        enhanced_signals['Signal_Strength'] = 'Unknown'  # 기본값 설정
        
        for idx, row in enhanced_signals.iterrows():
            if row['Buy_Signal']:
                # 해당 날짜의 매크로 점수 계산
                macro_score = macro_analyzer.calculate_macro_score(idx.strftime('%Y-%m-%d'))
                
                if macro_score:
                    enhanced_signals.loc[idx, 'Macro_Score'] = macro_score['macro_score']
                    enhanced_signals.loc[idx, 'Macro_Recommendation'] = macro_score['recommendation']
                    
                    # 개선된 점수 계산 (기술적 점수 + 매크로 점수)
                    technical_score = row['Buy_Score']
                    enhanced_score = (technical_score * technical_weight + 
                                    macro_score['macro_score'] * macro_weight)
                    
                    enhanced_signals.loc[idx, 'Enhanced_Score'] = enhanced_score
                    
                    # 매크로 조건 검증
                    vix_ok = check_vix_condition(idx, macro_analyzer, vix_threshold)
                    spread_ok = check_spread_condition(idx, macro_analyzer, spread_threshold)
                    naiim_ok = check_naiim_condition(idx, macro_analyzer, naiim_threshold)
                    
                    # 모든 매크로 조건을 만족하는 경우만 강화된 신호
                    if vix_ok and spread_ok and naiim_ok:
                        enhanced_signals.loc[idx, 'Signal_Strength'] = 'Strong'
                    else:
                        enhanced_signals.loc[idx, 'Signal_Strength'] = 'Weak'
                else:
                    enhanced_signals.loc[idx, 'Macro_Score'] = 50  # 기본값
                    enhanced_signals.loc[idx, 'Enhanced_Score'] = row['Buy_Score']
                    enhanced_signals.loc[idx, 'Signal_Strength'] = 'Unknown'
        
        return enhanced_signals
        
    except Exception as e:
        print(f"개선된 매수 신호 생성 중 오류: {e}")
        return None

def check_vix_condition(date, macro_analyzer, threshold):
    """VIX 조건 검증"""
    try:
        if macro_analyzer.vix_data is not None and date in macro_analyzer.vix_data.index:
            vix_value = macro_analyzer.vix_data.loc[date, 'VIX']
            return vix_value >= threshold
        return False
    except:
        return False

def check_spread_condition(date, macro_analyzer, threshold):
    """High Yield Spread 조건 검증"""
    try:
        if macro_analyzer.hy_spread_data is not None and date in macro_analyzer.hy_spread_data.index:
            spread_value = macro_analyzer.hy_spread_data.loc[date, 'HY_Spread']
            return spread_value <= threshold
        return False
    except:
        return False

def check_naiim_condition(date, macro_analyzer, threshold):
    """NAIIM 조건 검증"""
    try:
        if macro_analyzer.naiim_data is not None and date in macro_analyzer.naiim_data.index:
            naiim_value = macro_analyzer.naiim_data.loc[date, 'NAIIM']
            return naiim_value <= threshold
        return False
    except:
        return False

def plot_enhanced_signals(stock_data, enhanced_signals, ticker, output_dir):
    """개선된 매수 신호 시각화"""
    try:
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(f'{ticker} 개선된 매수 신호 분석', fontsize=16)
        
        # 1. 주가 차트와 매수 신호
        axes[0].plot(stock_data.index, stock_data['Close'], 'k-', alpha=0.7, label='주가')
        
        # 강한 신호와 약한 신호 구분
        strong_signals = enhanced_signals[enhanced_signals['Signal_Strength'] == 'Strong']
        weak_signals = enhanced_signals[enhanced_signals['Signal_Strength'] == 'Weak']
        
        if not strong_signals.empty:
            strong_dates = strong_signals.index
            strong_prices = stock_data.loc[strong_dates, 'Close']
            axes[0].scatter(strong_dates, strong_prices, color='green', marker='^', 
                           s=100, label='강한 매수 신호', zorder=5)
        
        if not weak_signals.empty:
            weak_dates = weak_signals.index
            weak_prices = stock_data.loc[weak_dates, 'Close']
            axes[0].scatter(weak_dates, weak_prices, color='orange', marker='o', 
                           s=80, label='약한 매수 신호', zorder=5)
        
        axes[0].set_title('주가와 매수 신호')
        axes[0].set_ylabel('주가')
        axes[0].legend()
        axes[0].grid(True)
        
        # 2. 개선된 점수 비교
        if not enhanced_signals.empty:
            buy_dates = enhanced_signals[enhanced_signals['Buy_Signal'] == True].index
            
            if len(buy_dates) > 0:
                technical_scores = enhanced_signals.loc[buy_dates, 'Buy_Score']
                macro_scores = enhanced_signals.loc[buy_dates, 'Macro_Score']
                enhanced_scores = enhanced_signals.loc[buy_dates, 'Enhanced_Score']
                
                x_pos = range(len(buy_dates))
                width = 0.25
                
                axes[1].bar([x - width for x in x_pos], technical_scores, width, 
                           label='기술적 점수', alpha=0.7)
                axes[1].bar(x_pos, macro_scores, width, label='매크로 점수', alpha=0.7)
                axes[1].bar([x + width for x in x_pos], enhanced_scores, width, 
                           label='개선된 점수', alpha=0.7)
                
                axes[1].set_title('점수 비교')
                axes[1].set_xlabel('매수 신호 순서')
                axes[1].set_ylabel('점수')
                axes[1].legend()
                axes[1].grid(True)
                axes[1].set_xticks(x_pos)
                axes[1].set_xticklabels([d.strftime('%m-%d') for d in buy_dates], rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/enhanced_signals.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
        
    except Exception as e:
        print(f"시각화 중 오류: {e}")
        return None

def generate_macro_analysis_report(args, enhanced_signals, macro_df, correlation_analysis, output_dir):
    """매크로 분석 상세 리포트 생성"""
    try:
        report_content = f"""# {args.ticker} 매크로 지표 기반 매수 신호 개선 분석 리포트

## 분석 설정
- **종목 코드**: {args.ticker}
- **분석 기간**: {args.period}
- **매크로 가중치**: {args.macro_weight:.1%}
- **기술적 가중치**: {args.technical_weight:.1%}
- **VIX 임계값**: {args.vix_threshold}
- **HY Spread 임계값**: {args.spread_threshold}
- **NAIIM 임계값**: {args.naiim_threshold}

## 분석 결과 요약
"""
        
        if enhanced_signals is not None:
            total_signals = len(enhanced_signals[enhanced_signals['Buy_Signal'] == True])
            strong_signals = len(enhanced_signals[enhanced_signals['Signal_Strength'] == 'Strong'])
            weak_signals = len(enhanced_signals[enhanced_signals['Signal_Strength'] == 'Weak'])
            
            # 0으로 나누기 방지
            signal_strength_ratio = (strong_signals / total_signals * 100) if total_signals > 0 else 0
            
            report_content += f"""
- **총 매수 신호**: {total_signals}
- **강한 신호**: {strong_signals}
- **약한 신호**: {weak_signals}
- **신호 강화율**: {signal_strength_ratio:.1f}% (강한 신호 비율)
"""
        
        if macro_df is not None:
            report_content += f"""
## 매크로 지표 분석
- **분석된 매수 신호**: {len(macro_df)}개
- **평균 VIX**: {macro_df['VIX'].mean():.2f}
- **평균 HY Spread**: {macro_df['HY_Spread'].mean():.4f}
- **평균 NAIIM**: {macro_df['NAIIM'].mean():.2f}
- **평균 매크로 점수**: {macro_df['Macro_Score'].mean():.2f}
"""
        
        if correlation_analysis and 'error' not in correlation_analysis:
            report_content += f"""
## 상관관계 분석
- **샘플 크기**: {correlation_analysis['sample_size']}
- **VIX와 매크로 점수 상관관계**: {correlation_analysis['macro_score_correlations'].get('VIX', 'N/A'):.3f}
- **HY Spread와 매크로 점수 상관관계**: {correlation_analysis['macro_score_correlations'].get('HY_Spread', 'N/A'):.3f}
- **NAIIM과 매크로 점수 상관관계**: {correlation_analysis['macro_score_correlations'].get('NAIIM', 'N/A'):.3f}
"""
        
        report_content += f"""
## 분석 및 권장사항
이 분석은 기술적 지표와 매크로 경제 지표를 종합하여 매수 신호의 품질을 개선합니다.

### 매크로 지표 해석
- **VIX**: 높을수록 시장 공포, 매수 기회
- **HY Spread**: 좁을수록 리스크 선호도 증가, 매수 기회
- **NAIIM**: 낮을수록 투자자 비관, 반대 심리 매수 기회

### 신호 강화 기준
- **강한 신호**: 모든 매크로 조건 만족
- **약한 신호**: 일부 매크로 조건 불만족
- **신호 강화율**: 전체 신호 중 강한 신호의 비율

---
*생성일시: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(f"{output_dir}/macro_analysis_report.md", 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"  - 상세 리포트: {output_dir}/macro_analysis_report.md")
        
    except Exception as e:
        print(f"리포트 생성 중 오류: {e}")

def main():
    """메인 함수"""
    parser = setup_macro_parser()
    args = parser.parse_args()
    
    # 가중치 검증
    if abs(args.macro_weight + args.technical_weight - 1.0) > 0.01:
        print("ERROR: 매크로 가중치와 기술적 가중치의 합이 1.0이어야 합니다.")
        sys.exit(1)
    
    # 매크로 지표 기반 분석 실행
    results = run_macro_enhanced_analysis(args)
    
    if results is None:
        sys.exit(1)

if __name__ == "__main__":
    main()
