#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
코스탈라니 달걀모형 - 자동 데이터 연동 버전
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import matplotlib.patches as patches
from matplotlib.path import Path
import platform
import os
import argparse
from kostalany_data_loader import MacroDataLoader
from kostalany_simple import KostalanyEggModel
from dotenv import load_dotenv

def main():
    """
    메인 함수 - 거시경제 지표 데이터를 자동으로 로딩하여 달걀모형 분석 실행
    """
    # .env 파일 자동 로드
    load_dotenv()
    fred_api_key = os.getenv('FRED_API_KEY')
    
    parser = argparse.ArgumentParser(description='코스탈라니 달걀모형 경제 사이클 분석')
    parser.add_argument('--country', type=str, default='korea', choices=['korea', 'us'],
                        help='분석할 국가 (korea 또는 us)')
    parser.add_argument('--manual', action='store_true',
                        help='수동으로 데이터 입력 (자동 데이터 로딩 대신)')
    parser.add_argument('--history', action='store_true',
                        help='과거 데이터도 함께 표시')
    parser.add_argument('--history_years', type=int, default=5,
                        help='표시할 과거 데이터 연수 (기본: 5년)')
    
    args = parser.parse_args()
    
    model = KostalanyEggModel()
    
    print(f"\n=== 코스탈라니 달걀모형 경제 사이클 분석 ({args.country.upper()}) ===\n")
    
    indicators = {}
    historical_data = {}
    details = {}
    
    if args.manual:
        # 수동 데이터 입력 모드
        print("현재 거시경제 지표를 입력해주세요:\n")
        
        try:
            gdp = float(input("GDP 성장률(%): "))
            inflation = float(input("인플레이션(%): "))
            interest = float(input("기준금리(%): "))
            unemployment = float(input("실업률(%): "))
            
            indicators = {
                'GDP': gdp,
                'Inflation': inflation,
                'Interest': interest,
                'Unemployment': unemployment
            }
            
            # 과거 데이터 입력 요청
            if args.history:
                print("\n과거 데이터를 입력하시겠습니까? (y/n): ", end="")
                use_history = input().lower() == 'y'
                
                if use_history:
                    while True:
                        print("\n과거 데이터 입력 (종료하려면 날짜에 'q' 입력):")
                        date = input("날짜 (YYYY-MM): ")
                        if date.lower() == 'q':
                            break
                        
                        hist_gdp = float(input(f"{date} GDP 성장률(%): "))
                        hist_inf = float(input(f"{date} 인플레이션(%): "))
                        hist_int = float(input(f"{date} 기준금리(%): "))
                        hist_unemp = float(input(f"{date} 실업률(%): "))
                        
                        historical_data[date] = {
                            'GDP': hist_gdp,
                            'Inflation': hist_inf,
                            'Interest': hist_int,
                            'Unemployment': hist_unemp
                        }
        except ValueError:
            print("숫자 형식으로 입력해주세요.")
            return
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
            return
    else:
        # 자동 데이터 로딩 모드
        print("거시경제 데이터를 자동으로 로딩중...\n")
        
        try:
            # FRED_API_KEY를 명시적으로 전달
            data_loader = MacroDataLoader(country=args.country, fred_api_key=fred_api_key)
            loaded_indicators, details = data_loader.get_current_indicators()
            
            # 대소문자 표준화를 위해 데이터 키 변환
            indicators = {
                'GDP': loaded_indicators.get('Gdp', 0),
                'Inflation': loaded_indicators.get('Inflation', 0),
                'Interest': loaded_indicators.get('Interest', 0),
                'Unemployment': loaded_indicators.get('Unemployment', 0),
                'Vix': loaded_indicators.get('Vix', 0),
                'Dxy': loaded_indicators.get('Dxy', 0)
            }
            
            print("현재 거시경제 지표:")
            for key, value in loaded_indicators.items():
                detail = details[key]
                print(f"{key}: {value:.2f}% (출처: {detail['source']}, 업데이트: {detail['last_update']})")
            
            # 과거 데이터 로딩
            if args.history:
                print("\n과거 데이터를 로딩중...")
                loaded_historical_data = data_loader.get_historical_data(years=args.history_years)
                
                # 과거 데이터도 대소문자 표준화 적용
                for date, values in loaded_historical_data.items():
                    historical_data[date] = {
                        'GDP': values.get('GDP', 0),
                        'Inflation': values.get('Inflation', 0),
                        'Interest': values.get('Interest', 0),
                        'Unemployment': values.get('Unemployment', 0)
                    }
                
                print(f"과거 {args.history_years}년 데이터 로딩 완료")
        except Exception as e:
            print(f"데이터 로딩 중 오류가 발생했습니다: {e}")
            return
    
    # 분석 직전 데이터 점검 로그 추가
    print('\n==== 분석 입력 데이터 점검 ===')
    print('indicators:', indicators)
    if historical_data:
        print('historical_data 샘플:', list(historical_data.items())[:2])
    else:
        print('historical_data: 없음')
    
    # 달걀모형 분석 실행
    try:
        fig, ax, position, stage = model.plot_egg_model(indicators, historical_data)
        
        # 자세한 분석 결과 출력
        print("\n=== 분석 결과 ===\n")
        print(f"현재 경제 사이클 단계: {stage} ({model.stages[stage]})")
        print(f"투자 전략 제안: {model.strategies[stage]}")
        
        print("\n거시경제 지표 평가:")
        
        # GDP 성장률 평가 분석
        gdp = indicators['GDP']
        gdp_eval = ""
        if gdp >= 3.0:
            gdp_eval = "3% 이상 (호황기)"
        elif gdp >= 1.0:
            gdp_eval = "1~3% 사이로 상승 (경기 회복)"
        elif gdp > 0:
            gdp_eval = "0~1% 사이로 회복 시작"
        elif gdp > -1.0:
            gdp_eval = "0% 이하 경미한 마이너스 성장"
        else:
            gdp_eval = "심각한 마이너스 성장 (경기 침체)"
        print(f"GDP 성장률 {gdp:.2f}%: {gdp_eval}")
        
        # 인플레이션 평가 분석
        inflation = indicators['Inflation']
        inflation_eval = ""
        if inflation >= 5.0:
            inflation_eval = "5% 이상 (인플레이션 정점)"
        elif inflation >= 4.0:
            inflation_eval = "4% 이상 (경기 과열로 인한 물가 상승)"
        elif inflation >= 2.0:
            inflation_eval = "2~4% 사이 (경기 활성화)"
        elif inflation >= 1.0:
            inflation_eval = "1~2% 사이 (적정 인플레이션)"
        else:
            inflation_eval = "1% 이하 (경기 침체로 인한 낮은 물가)"
        print(f"인플레이션 {inflation:.2f}%: {inflation_eval}")
        
        # 기준금리 평가 분석
        interest = indicators['Interest']
        interest_eval = ""
        if interest >= 10.0:
            interest_eval = "10% 이상 (버블기)"
        elif interest >= 8.0:
            interest_eval = "8~10% (긴축정책 정점)"
        elif interest >= 6.0:
            interest_eval = "6~8% (인플레이션 억제)"
        elif interest >= 5.0:
            interest_eval = "5~6% (경기 과열 방지)"
        elif interest >= 3.0:
            interest_eval = "3~5% (완만한 금리 상승, 회복기)"
        else:
            interest_eval = "3% 미만 (경기 부양을 위한 저금리)"
        print(f"기준금리 {interest:.2f}%: {interest_eval}")
        
        # 실업률 평가 분석
        unemployment = indicators['Unemployment']
        unemployment_eval = ""
        if unemployment >= 8.0:
            unemployment_eval = "7~10% 이상 (경기 침체로 인한 높은 실업률)"
        elif unemployment >= 6.0:
            unemployment_eval = "6~8% (실업률 정점)"
        elif unemployment >= 5.0:
            unemployment_eval = "5~7% (긴축정책으로 인한 고용 감소)"
        elif unemployment >= 4.0:
            unemployment_eval = "4~6% (고용 회복 시작)"
        else:
            unemployment_eval = "2~4% (완전 고용에 가까운 낮은 실업률)"
        print(f"실업률 {unemployment:.2f}%: {unemployment_eval}")
        
        # VIX 지수 평가 분석 (추가됨)
        if 'Vix' in indicators:
            vix = indicators['Vix']
            vix_eval = ""
            vix_trend = details.get('Vix', {}).get('trend', "")
            trend_str = f", 추세: {vix_trend}" if vix_trend else ""
            
            if vix >= 40.0:
                vix_eval = "40 이상 (극심한 시장 공포, C단계 신호)"
            elif vix >= 30.0:
                vix_eval = "30~40 (높은 시장 불안, B단계 신호)"
            elif vix >= 20.0:
                vix_eval = "20~30 (보통 수준의 변동성, A 또는 E단계)"
            else:
                vix_eval = "20 이하 (낮은 변동성, 시장 안정, E-F단계)"
            
            print(f"VIX 지수 {vix:.2f}{trend_str}: {vix_eval}")
        
        # 달러지수 평가 분석 (추가됨)
        if 'Dxy' in indicators:
            dxy = indicators['Dxy']
            dxy_eval = ""
            dxy_trend = details.get('Dxy', {}).get('trend', "")
            trend_str = f", 추세: {dxy_trend}" if dxy_trend else ""
            
            if dxy >= 105:
                dxy_eval = "105 이상 (강한 달러, A-B단계 신호)"
            elif dxy >= 100:
                dxy_eval = "100~105 (달러 강세, B-C단계 가능성)"
            elif dxy >= 95:
                dxy_eval = "95~100 (중립적 달러)"
            elif dxy >= 90:
                dxy_eval = "90~95 (달러 약세, D-E단계 가능성)"
            else:
                dxy_eval = "90 이하 (약한 달러, E-F단계 신호)"
            
            print(f"달러지수 {dxy:.2f}{trend_str}: {dxy_eval}")
        
        # 결과 저장
        os.makedirs('output/kostalany', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        country_suffix = args.country.lower()
        filename = f"output/kostalany/kostalany_model_{country_suffix}_{timestamp}.png"
        md_filename = f"output/kostalany/kostalany_model_{country_suffix}_{timestamp}.md"
        try:
            if fig is not None:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"\n분석 결과 이미지가 저장되었습니다: {filename}")
            else:
                print("[오류] fig 객체가 None입니다. 차트가 생성되지 않았습니다.")
        except Exception as e:
            print(f"[오류] 분석 결과 이미지 저장 실패: {e}")
        
        # 분석 결과를 md 파일로 저장
        try:
            with open(md_filename, 'w', encoding='utf-8') as f:
                f.write(f"# 코스탈라니 달걀모형 경제 사이클 분석 결과\n\n")
                f.write(f"- 분석일시: {timestamp}\n")
                f.write(f"- 국가: {args.country.upper()}\n\n")
                f.write(f"## 주요 거시경제 지표\n")
                for key, value in indicators.items():
                    f.write(f"- {key}: {value}\n")
                f.write(f"\n## 현재 경제 사이클 단계\n")
                f.write(f"- 단계: {stage} ({model.stages[stage]})\n")
                f.write(f"- 투자 전략: {model.strategies[stage]}\n\n")
                f.write(f"## 거시경제 지표 평가\n")
                # GDP 평가
                gdp = indicators['GDP']
                if gdp >= 3.0:
                    gdp_eval = "3% 이상 (호황기)"
                elif gdp >= 1.0:
                    gdp_eval = "1~3% 사이로 상승 (경기 회복)"
                elif gdp > 0:
                    gdp_eval = "0~1% 사이로 회복 시작"
                elif gdp > -1.0:
                    gdp_eval = "0% 이하 경미한 마이너스 성장"
                else:
                    gdp_eval = "심각한 마이너스 성장 (경기 침체)"
                f.write(f"- GDP 성장률 {gdp:.2f}%: {gdp_eval}\n")
                # 인플레이션 평가
                inflation = indicators['Inflation']
                if inflation >= 5.0:
                    inflation_eval = "5% 이상 (인플레이션 정점)"
                elif inflation >= 4.0:
                    inflation_eval = "4% 이상 (경기 과열로 인한 물가 상승)"
                elif inflation >= 2.0:
                    inflation_eval = "2~4% 사이 (경기 활성화)"
                elif inflation >= 1.0:
                    inflation_eval = "1~2% 사이 (적정 인플레이션)"
                else:
                    inflation_eval = "1% 이하 (경기 침체로 인한 낮은 물가)"
                f.write(f"- 인플레이션 {inflation:.2f}%: {inflation_eval}\n")
                # 기준금리 평가
                interest = indicators['Interest']
                if interest >= 10.0:
                    interest_eval = "10% 이상 (버블기)"
                elif interest >= 8.0:
                    interest_eval = "8~10% (긴축정책 정점)"
                elif interest >= 6.0:
                    interest_eval = "6~8% (인플레이션 억제)"
                elif interest >= 5.0:
                    interest_eval = "5~6% (경기 과열 방지)"
                elif interest >= 3.0:
                    interest_eval = "3~5% (완만한 금리 상승, 회복기)"
                else:
                    interest_eval = "3% 미만 (경기 부양을 위한 저금리)"
                f.write(f"- 기준금리 {interest:.2f}%: {interest_eval}\n")
                # 실업률 평가
                unemployment = indicators['Unemployment']
                if unemployment >= 8.0:
                    unemployment_eval = "7~10% 이상 (경기 침체로 인한 높은 실업률)"
                elif unemployment >= 6.0:
                    unemployment_eval = "6~8% (실업률 정점)"
                elif unemployment >= 5.0:
                    unemployment_eval = "5~7% (긴축정책으로 인한 고용 감소)"
                elif unemployment >= 4.0:
                    unemployment_eval = "4~6% (고용 회복 시작)"
                else:
                    unemployment_eval = "2~4% (완전 고용에 가까운 낮은 실업률)"
                f.write(f"- 실업률 {unemployment:.2f}%: {unemployment_eval}\n")
                # VIX 평가
                if 'Vix' in indicators:
                    vix = indicators['Vix']
                    if vix >= 40.0:
                        vix_eval = "40 이상 (극심한 시장 공포, C단계 신호)"
                    elif vix >= 30.0:
                        vix_eval = "30~40 (높은 시장 불안, B단계 신호)"
                    elif vix >= 20.0:
                        vix_eval = "20~30 (보통 수준의 변동성, A 또는 E단계)"
                    else:
                        vix_eval = "20 이하 (낮은 변동성, 시장 안정, E-F단계)"
                    f.write(f"- VIX 지수 {vix:.2f}: {vix_eval}\n")
                # DXY 평가
                if 'Dxy' in indicators:
                    dxy = indicators['Dxy']
                    if dxy >= 105:
                        dxy_eval = "105 이상 (강한 달러, A-B단계 신호)"
                    elif dxy >= 100:
                        dxy_eval = "100~105 (달러 강세, B-C단계 가능성)"
                    elif dxy >= 95:
                        dxy_eval = "95~100 (중립적 달러)"
                    elif dxy >= 90:
                        dxy_eval = "90~95 (달러 약세, D-E단계 가능성)"
                    else:
                        dxy_eval = "90 이하 (약한 달러, E-F단계 신호)"
                    f.write(f"- 달러지수 {dxy:.2f}: {dxy_eval}\n")
                f.write(f"\n---\n분석 이미지: {filename}\n")
            print(f"분석 결과 md 파일이 저장되었습니다: {md_filename}")
        except Exception as e:
            print(f"[오류] 분석 결과 md 파일 저장 실패: {e}")
        
        # 결과 표시 대신 그래프 닫기
        plt.close(fig)
        
        # (예시) 분석 또는 차트 생성 직전 데이터 점검 로그 추가
        if 'df' in locals():
            print('==== 데이터프레임 정보 ===')
            print('컬럼:', df.columns)
            print('shape:', df.shape)
            print('head:', df.head())
            print('tail:', df.tail())
            if 'date' in df.columns:
                print('date min:', df['date'].min(), 'date max:', df['date'].max())
        # 만약 df가 아니라 dict 등이라면 keys와 일부 값 출력
        if 'historical_data' in locals():
            print('==== 히스토리컬 데이터 샘플 ===')
            print('keys:', list(historical_data.keys())[:5])
            print('sample:', list(historical_data.items())[:2])
        
    except Exception as e:
        print(f"분석 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main() 