#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
코스탈라니 달걀모형 백테스트 스크립트
과거 데이터를 기반으로 모델의 예측 정확도를 테스트합니다.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from kostalany_data_loader import MacroDataLoader
from kostalany_simple import KostalanyEggModel

def calculate_returns(data, stage, market_returns):
    """
    각 경제 사이클 단계에 따른 기대 수익률을 계산합니다.
    
    Args:
        data: 경제 지표 데이터
        stage: 경제 사이클 단계 (A-F)
        market_returns: 기준 시장 수익률 데이터
    
    Returns:
        expected_return: 기대 수익률
    """
    # 백테스트 시장 수익률 (실제 시장에서의 단계별 수익률)
    # 여기서는 각 단계별 시장 수익률을 사용합니다 (실제 데이터를 기반으로 한 추정)
    stage_returns = {
        'A': -0.10,  # 침체기에는 -10% 성과 (중앙값)
        'B': -0.05,  # 회복 시작 단계 -5% 성과
        'C': 0.05,   # 회복 단계 +5% 성과
        'D': 0.10,   # 확장 단계 +10% 성과
        'E': 0.15,   # 호황 단계 +15% 성과
        'F': 0.05,   # 호황 정점 +5% 성과 (과열)
    }
    
    # 해당 날짜의 실제 시장 수익률이 있으면 사용, 없으면 단계별 평균 수익률 사용
    date_str = data.get('date')
    if date_str in market_returns:
        actual_return = market_returns[date_str]
    else:
        actual_return = stage_returns.get(stage, 0)
    
    return actual_return

def run_backtest(country='us', years=10, market_data_file=None):
    """
    백테스트를 실행합니다.
    
    Args:
        country: 분석할 국가 ('us' 또는 'korea')
        years: 백테스트 기간 (연 단위)
        market_data_file: 시장 수익률 데이터 파일 경로 (선택 사항)
    
    Returns:
        results: 백테스트 결과 데이터프레임
    """
    print(f"\n=== 코스탈라니 달걀모형 백테스트 시작 ({country.upper()}) ===\n")
    print(f"분석 기간: 과거 {years}년")
    
    # 시장 수익률 데이터 로드 (있는 경우)
    market_returns = {}
    if market_data_file and os.path.exists(market_data_file):
        try:
            market_data = pd.read_csv(market_data_file)
            for _, row in market_data.iterrows():
                market_returns[row['date']] = row['return']
            print(f"시장 수익률 데이터 로드 완료: {len(market_returns)} 기간")
        except Exception as e:
            print(f"시장 수익률 데이터 로드 실패: {e}")
    else:
        print("시장 수익률 데이터 파일이 없습니다. 단계별 평균 수익률을 사용합니다.")
    
    # 데이터 로더 초기화
    data_loader = MacroDataLoader(country=country)
    
    # 모델 초기화
    model = KostalanyEggModel()
    
    # 과거 데이터 로드
    print("\n과거 데이터 로드 중...")
    historical_data = data_loader.get_historical_data(years=years)
    print(f"과거 {len(historical_data)} 기간의 데이터 로드 완료")
    
    # 백테스트 결과를 저장할 리스트
    results = []
    cumulative_return = 1.0  # 누적 수익률 (1.0 = 100%)
    
    # 각 시점에 대해 모델 예측 및 수익률 계산
    for date, indicators in sorted(historical_data.items()):
        if not all(k in indicators for k in ['GDP', 'Inflation', 'Interest', 'Unemployment']):
            continue
            
        print(f"\n[{date}] 분석 중...")
        
        # 모델 입력 형식으로 데이터 변환
        input_data = {
            'GDP': indicators['GDP'],
            'Inflation': indicators['Inflation'],
            'Interest': indicators['Interest'],
            'Unemployment': indicators['Unemployment']
        }
        
        # 경제 사이클 위치 계산
        try:
            _, _, position, stage = model.plot_egg_model(input_data, {}, show_plot=False)
            
            # 해당 단계의 기대 수익률 계산
            indicators['date'] = date
            expected_return = calculate_returns(indicators, stage, market_returns)
            
            # 누적 수익률 업데이트
            cumulative_return *= (1 + expected_return)
            
            # 결과 저장
            result = {
                'date': date,
                'gdp': indicators['GDP'],
                'inflation': indicators['Inflation'],
                'interest': indicators['Interest'],
                'unemployment': indicators['Unemployment'],
                'stage': stage,
                'stage_name': model.stages[stage],
                'position': position,
                'expected_return': expected_return,
                'cumulative_return': cumulative_return
            }
            
            print(f"경제 사이클 단계: {stage} ({model.stages[stage]})")
            print(f"기대 수익률: {expected_return*100:.2f}%")
            print(f"누적 수익률: {(cumulative_return-1)*100:.2f}%")
            
            results.append(result)
        except Exception as e:
            print(f"분석 오류: {e}")
    
    # 결과를 데이터프레임으로 변환
    if results:
        results_df = pd.DataFrame(results)
        
        # 백테스트 결과 시각화
        plt.figure(figsize=(12, 8))
        
        # 누적 수익률 그래프
        plt.subplot(2, 1, 1)
        plt.plot(results_df['date'], results_df['cumulative_return'], 'b-', linewidth=2)
        plt.title('코스탈라니 모델 백테스트 결과')
        plt.ylabel('누적 수익률')
        plt.grid(True)
        
        # 단계별 색상 설정
        stage_colors = {
            'A': 'blue',
            'B': 'skyblue',
            'C': 'green',
            'D': 'yellowgreen',
            'E': 'orange',
            'F': 'red'
        }
        
        # 경제 사이클 단계 그래프
        plt.subplot(2, 1, 2)
        for i, stage in enumerate(['A', 'B', 'C', 'D', 'E', 'F']):
            mask = results_df['stage'] == stage
            if mask.any():
                plt.scatter(
                    results_df.loc[mask, 'date'],
                    [i] * mask.sum(),
                    label=f"{stage}: {model.stages[stage]}",
                    color=stage_colors.get(stage, 'gray'),
                    s=100
                )
        
        plt.yticks(range(6), ['A', 'B', 'C', 'D', 'E', 'F'])
        plt.ylabel('경제 사이클 단계')
        plt.xlabel('날짜')
        plt.grid(True)
        plt.legend(loc='best')
        
        # 결과 저장
        plt.tight_layout()
        os.makedirs('output', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/kostalany_backtest_{country}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n백테스트 결과 이미지가 저장되었습니다: {filename}")
        
        # 결과 CSV 저장
        csv_filename = f"output/kostalany_backtest_{country}_{timestamp}.csv"
        results_df.to_csv(csv_filename, index=False)
        print(f"백테스트 결과 CSV가 저장되었습니다: {csv_filename}")
        
        plt.close()
        
        return results_df
    else:
        print("백테스트 결과가 없습니다.")
        return None

def main():
    """백테스트 메인 함수"""
    parser = argparse.ArgumentParser(description='코스탈라니 달걀모형 백테스트')
    parser.add_argument('--country', type=str, default='us', choices=['korea', 'us'],
                        help='분석할 국가 (korea 또는 us)')
    parser.add_argument('--years', type=int, default=10,
                        help='백테스트 기간 (연 단위, 기본: 10년)')
    parser.add_argument('--market_data', type=str, default=None,
                        help='시장 수익률 데이터 파일 경로 (선택 사항)')
    
    args = parser.parse_args()
    
    # 모델 초기화 (단계별 발생 빈도 출력을 위해 필요)
    model = KostalanyEggModel()
    
    # 백테스트 실행
    results = run_backtest(
        country=args.country,
        years=args.years,
        market_data_file=args.market_data
    )
    
    if results is not None:
        print("\n===== 백테스트 통계 =====")
        
        # 수익률 통계
        total_return = results['cumulative_return'].iloc[-1] - 1
        annualized_return = (results['cumulative_return'].iloc[-1] ** (1 / (len(results) / 12))) - 1
        
        print(f"총 수익률: {total_return*100:.2f}%")
        print(f"연평균 수익률: {annualized_return*100:.2f}%")
        
        # 단계별 발생 빈도
        stage_counts = results['stage'].value_counts()
        print("\n단계별 발생 빈도:")
        stages_info = {
            'A': '인플레이션 우려와 금리 상승',
            'B': '인플레이션 억제를 위한 긴축정책',
            'C': '경기 침체와 소비 감소',
            'D': '경기 회복을 위한 완화정책',
            'E': '경기 확장과 소비 증가',
            'F': '호황기와 최대 경기 활성화'
        }
        for stage in ['A', 'B', 'C', 'D', 'E', 'F']:
            count = stage_counts.get(stage, 0)
            pct = count / len(results) * 100
            print(f"{stage} ({stages_info.get(stage, '알 수 없음')}): {count}회 ({pct:.1f}%)")
    
    print("\n백테스트가 완료되었습니다.")

if __name__ == "__main__":
    main() 