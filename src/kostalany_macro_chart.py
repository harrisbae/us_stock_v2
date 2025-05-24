#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
코스탈라니 달걀모형 - 거시경제 지표 차트 생성기
여러 거시경제 지표의 변동 추이를 시각화합니다.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import argparse
from matplotlib.colors import LinearSegmentedColormap
from kostalany_data_loader import MacroDataLoader
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # macOS 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지

def create_macro_charts(country='us', years=3, output_path=None, interval='month', start_date=None, end_date=None):
    """
    거시경제 지표 차트를 생성합니다.
    
    Args:
        country (str): 분석할 국가 ('us' 또는 'korea')
        years (int): 데이터 조회 기간 (년)
        output_path (str): 출력 파일 경로 (기본값: None, 자동 생성)
        interval (str): 'month' 또는 'day' (기본값: 'month')
        start_date (str): 시작일 (YYYY-MM-DD)
        end_date (str): 종료일 (YYYY-MM-DD)
    Returns:
        str: 저장된 이미지 파일 경로
    """
    # output 디렉토리 생성
    os.makedirs('output', exist_ok=True)
    
    # 데이터 로더 초기화
    data_loader = MacroDataLoader(country=country)
    
    # 현재 지표 데이터 로드
    current_indicators, details = data_loader.get_current_indicators()
    
    # 과거 데이터 로드
    if start_date and end_date:
        print(f"\n{start_date} ~ {end_date} 기간 데이터 로드 중...")
        historical_data = data_loader.get_historical_data(start_date=start_date, end_date=end_date, interval=interval)
    else:
        print(f"\n{years}년 과거 데이터 로드 중...")
        historical_data = data_loader.get_historical_data(years=years, interval=interval)
    print(f"과거 {len(historical_data)}개 시점의 데이터 로드 완료")
    
    # 데이터프레임 변환
    data_list = []
    for date_str, data in historical_data.items():
        # 날짜가 'YYYY-MM' 형식인 경우, 'YYYY-MM-01'로 변환하여 정렬 가능하게 함
        if len(date_str) == 7:  # 'YYYY-MM' 형식
            date_str = f"{date_str}-01"
        
        row = data.copy()
        row['date'] = date_str
        data_list.append(row)
    
    if data_list:
        df = pd.DataFrame(data_list)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # start_date, end_date로 필터링
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        # 데이터프레임 열 이름 출력하여 확인
        print("\n데이터프레임 열 이름:")
        print(df.columns.tolist())
        # 연도 컬럼 생성 후 연도별 카운트 출력
        df['year'] = df['date'].dt.year
        print("date min:", df['date'].min(), "date max:", df['date'].max())
        print("연도별 데이터 개수:", df['year'].value_counts().sort_index().to_dict())
        
        # 성장/인플레이션 점수 계산 (간단 예시)
        # 표준편차가 0이거나 NaN이면 1로 대체
        gdp_std = df['GDP'].std() if df['GDP'].std() != 0 and not np.isnan(df['GDP'].std()) else 1
        unemp_std = df['Unemployment'].std() if df['Unemployment'].std() != 0 and not np.isnan(df['Unemployment'].std()) else 1
        infl_std = df['Inflation'].std() if df['Inflation'].std() != 0 and not np.isnan(df['Inflation'].std()) else 1
        int_std = df['Interest'].std() if df['Interest'].std() != 0 and not np.isnan(df['Interest'].std()) else 1
        gdp_norm = (df['GDP'] - df['GDP'].mean()) / gdp_std
        unemp_norm = (df['Unemployment'] - df['Unemployment'].mean()) / unemp_std
        infl_norm = (df['Inflation'] - df['Inflation'].mean()) / infl_std
        int_norm = (df['Interest'] - df['Interest'].mean()) / int_std
        df['성장점수'] = gdp_norm - unemp_norm
        df['인플레이션점수'] = infl_norm + int_norm
        
        # 연도별 데이터 분리
        year_list = sorted(df['year'].unique())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 연도별 색상
        markers = ['s', 'o', 'D']
        
        # 배경색 영역 정의 (더 진한 색상)
        background_zones = [
            {'label': '성장↑+인플레↑ (경기과열)', 'color': '#ff9999', 'x': (2023, 2023.5)},  # 진한 빨강
            {'label': '성장↑+인플레↓ (골디락스)', 'color': '#66ff66', 'x': (2023.5, 2024)},  # 진한 초록
            {'label': '성장↓+인플레↑ (스태그플레이션)', 'color': '#ffe066', 'x': (2024, 2024.5)},  # 진한 노랑
            {'label': '성장↓+인플레↓ (경기침체)', 'color': '#6699ff', 'x': (2024.5, 2025)}   # 진한 파랑
        ]
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1]})
        ax1, ax2 = axes
        
        # 상단: 성장/인플레이션 점수 추이 (전체 데이터 연속 plot)
        growth_color = '#1f77b4'
        infl_color = '#ff7f0e'
        ax1.plot(df['date'], df['성장점수'], label='성장 점수', color=growth_color, marker='o', linestyle='-', linewidth=2)
        ax1.plot(df['date'], df['인플레이션점수'], label='인플레이션 점수', color=infl_color, marker='o', linestyle='--', linewidth=2)
        ax1.axhline(0, color='gray', linestyle='dashed', linewidth=1)
        # 배경색
        for zone in background_zones:
            ax1.axvspan(pd.Timestamp(f'{int(zone['x'][0])}-01-01'), pd.Timestamp(f'{int(zone['x'][1])}-01-01'), color=zone['color'], alpha=0.4, label=zone['label'])
        # 커스텀 컬러박스 범례
        growth_patch = mpatches.Patch(color=growth_color, label='성장 점수')
        infl_patch = mpatches.Patch(color=infl_color, label='인플레이션 점수')
        custom_handles = [growth_patch, infl_patch]
        custom_labels = ['성장 점수', '인플레이션 점수']
        zone_patches = [mpatches.Patch(color=zone['color'], label=zone['label'], alpha=0.4) for zone in background_zones]
        custom_handles += zone_patches
        custom_labels += [zone['label'] for zone in background_zones]
        ax1.legend(custom_handles, custom_labels, loc='upper left', fontsize=10, frameon=True)
        ax1.set_ylabel('점수 (정규화)', fontsize=12)
        # 전체 연도 범위로 제목 표시
        ax1.set_title(f"{year_list[0]}~{year_list[-1]}년 성장-인플레이션 점수 추이", fontsize=15)
        # x축 연도별 포맷터 추가
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1.xaxis.set_minor_locator(mdates.MonthLocator())
        ax1.tick_params(axis='x', which='major', labelsize=11)
        # x축 범위 명시적으로 전체 데이터로 지정
        ax1.set_xlim(df['date'].min(), df['date'].max())
        
        # 하단: 금리 및 VIX, 기타 주요 지표 추이 (전체 데이터 연속 plot)
        ax2.plot(df['date'], df['Interest'], label='기준금리', color=colors[0], marker=markers[0], linestyle='-', linewidth=2)
        if 'CD' in df.columns:
            ax2.plot(df['date'], df['CD'], label='CD금리', color='#8c564b', marker='o', linestyle='-', linewidth=2)
        if 'US10Y' in df.columns:
            ax2.plot(df['date'], df['US10Y'], label='미국10년물국채', color='#bcbd22', marker='s', linestyle='-', linewidth=2)
        ax2.plot(df['date'], df['GDP'], label='GDP성장률', color='#17becf', marker='^', linestyle='--', linewidth=2)
        ax2.plot(df['date'], df['Inflation'], label='소비자물가지수', color='#d62728', marker='v', linestyle='--', linewidth=2)
        ax2.set_ylabel('금리 및 지표 (%)', fontsize=12)
        ax2.set_xlabel('날짜', fontsize=12)
        ax2.set_xlim(df['date'].min(), df['date'].max())
        # x축 월별 레이블 추가
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.tick_params(axis='x', which='major', labelsize=11)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.set_xticks(df['date'])  # 데이터가 있는 날짜만 x축에 표시
        # VIX, 달러지수 (보조 y축)
        ax2b = ax2.twinx()
        if 'VIX' in df.columns:
            ax2b.plot(df['date'], df['VIX'], label='VIX', color='#9467bd', marker='D', linestyle=':', linewidth=2)
        if 'USD' in df.columns:
            ax2b.plot(df['date'], df['USD'], label='달러지수', color='#7f7f7f', marker='x', linestyle=':', linewidth=2)
        ax2b.set_ylabel('VIX / 달러지수', fontsize=12)
        # 우측 축 범위를 달러 지수 범위로 조정 (0~115)
        ax2b.set_ylim(0, 115)
        
        # 하단 배경색
        for zone in background_zones:
            ax2.axvspan(pd.Timestamp(f'{int(zone['x'][0])}-01-01'), pd.Timestamp(f'{int(zone['x'][1])}-01-01'), color=zone['color'], alpha=0.2)
        
        # 범례 (단순화)
        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = ax2b.get_legend_handles_labels()
        ax2.legend(h1+h2, l1+l2, loc='upper left', fontsize=10)
        
        # VIX 실시간 값 표시
        if 'VIX' in df.columns:
            vix_latest = df['VIX'].iloc[-1]
            ax2.text(1.02, 0.95, f'VIX: {vix_latest:.1f}', transform=ax2.transAxes, 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'),
                     color='purple', fontsize=10, verticalalignment='top')
        
        # 달러 지수 실시간 값 표시
        if 'USD' in df.columns:
            usd_latest = df['USD'].iloc[-1]
            ax2.text(1.02, 0.90, f'USD: {usd_latest:.1f}', transform=ax2.transAxes, 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'),
                     color='gray', fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        # 그래프 저장
        if output_path is None:
            output_path = f'output/macro_chart_{country}_{int(years)}y_{interval}.png'
        df.to_csv(output_path.replace('.png', '.csv'), index=False)  # 데이터프레임을 CSV로 저장
        plt.savefig(output_path)
        plt.close()
        
        return output_path

if __name__ == "__main__":
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='거시경제 지표 차트 생성')
    parser.add_argument('--country', type=str, default='us', choices=['us', 'korea'],
                      help='분석할 국가 (기본값: us)')
    parser.add_argument('--years', type=int, default=3,
                      help='데이터 조회 기간 (년) (기본값: 3)')
    parser.add_argument('--output', type=str, default=None,
                      help='출력 파일 경로 (기본값: 자동 생성)')
    parser.add_argument('--interval', type=str, default='month', choices=['month', 'day'],
                      help='데이터 단위 (month 또는 day, 기본값: month)')
    parser.add_argument('--start_date', type=str, default=None, help='시작일 (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None, help='종료일 (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # 차트 생성
    output_path = create_macro_charts(
        country=args.country,
        years=args.years,
        output_path=args.output,
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    print(f"\n차트가 생성되었습니다: {output_path}")