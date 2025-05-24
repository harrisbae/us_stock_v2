#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
코스탈라니 달걀모형 간단 구현
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import matplotlib.patches as patches
from matplotlib.path import Path
import platform
import os
import random
import sys
import matplotlib
matplotlib.use('Agg')

class KostalanyEggModel:
    def __init__(self):
        # 경제 사이클 단계 정의
        self.stages = {
            'A': '인플레이션 우려와 금리 상승',
            'B': '인플레이션 억제를 위한 긴축정책',
            'C': '경기 침체와 소비 감소',
            'D': '경기 회복을 위한 완화정책',
            'E': '경기 확장과 소비 증가',
            'F': '호황기와 최대 경기 활성화'
        }
        
        # 지표 평가 기준 설정
        self.gdp_criteria = {
            'A': '3% 이하로 감소 시작',
            'B': '0~1% 사이로 급격히 감소',
            'C': '0% 이하 (마이너스 성장)',
            'D': '0~1% 사이로 회복 시작',
            'E': '1~3% 사이로 상승',
            'F': '3% 이상 (호황기)'
        }
        
        self.inflation_criteria = {
            'A': '4% 이상 (경기 과열)',
            'B': '5% 이상 (인플레이션 정점)',
            'C': '2~3% 사이로 하락',
            'D': '1% 이하 (경기 침체)',
            'E': '1~2% 사이 (적정 인플레이션)',
            'F': '2~4% 사이 (경기 활성화)'
        }
        
        self.interest_criteria = {
            'A': '6~10% 이상 (인플레이션 억제, 버블 단계)',
            'B': '8~10% 이상 (긴축정책 정점)',
            'C': '5~8% (침체기) 또는 3% 미만 (위기기)',
            'D': '3% 미만 (경기 부양을 위한 저금리)',
            'E': '3~5% (완만한 금리 상승, 회복기)',
            'F': '5~8% (경기 과열 방지) 또는 10% 이상 (버블기)'
        }
        
        self.unemployment_criteria = {
            'A': '4~5% (상승 시작)',
            'B': '5~7% (고용 감소)',
            'C': '7~10% 이상 (높은 실업률)',
            'D': '6~8% (실업률 정점)',
            'E': '4~6% (고용 회복 시작)',
            'F': '2~4% (완전 고용)'
        }
        
        # 투자 전략
        self.strategies = {
            'A': '주식시장 어려움, 안전자산 비중 확대',
            'B': '소비재, 필수품 관련 주식 중심 투자',
            'C': '채권과 배당주 비중 확대, 저평가 주식 매수 시작',
            'D': '경기 민감주와 성장주 관심 확대',
            'E': '경기 민감주와 성장주 투자 유리',
            'F': '고점 접근, 위험자산 비중 점진적 축소 검토'
        }
        
        # 상세 자산배분 전략
        self.asset_allocation = {
            'A': {
                '현금': '30%', 
                '채권': '40%', 
                '주식': '20%', 
                '금/원자재': '10%',
                '설명': '인플레이션 우려와 금리 상승 국면에서는 안전자산 비중을 높이고, 단기 채권과 물가연동채권에 투자하는 것이 유리합니다. 주식은 경기방어주와 가치주 중심으로 배분하세요.'
            },
            'B': {
                '현금': '25%', 
                '채권': '30%', 
                '주식': '35%', 
                '금/원자재': '10%',
                '설명': '긴축정책 국면에서는 필수소비재, 유틸리티, 헬스케어 등 경기방어 섹터와 배당주에 집중하세요. 채권은 금리 상승에 대비해 단기물 위주로 구성하는 것이 안전합니다.'
            },
            'C': {
                '현금': '20%', 
                '채권': '50%', 
                '주식': '25%', 
                '금/원자재': '5%',
                '설명': '경기 침체기에는 장기 국채와 우량 회사채의 비중을 높게 유지하세요. 주식은 저평가된 가치주와 고배당주를 매수할 좋은 기회입니다. 경기 회복 신호가 보이면 서서히 경기민감주 비중을 높이세요.'
            },
            'D': {
                '현금': '15%', 
                '채권': '35%', 
                '주식': '45%', 
                '금/원자재': '5%',
                '설명': '경기 회복 국면에서는 주식 비중을 점진적으로 확대하고, 금융, 경기소비재, IT 등 경기민감 섹터에 투자하세요. 채권은 중장기물로 포트폴리오를 재조정하는 것이 유리합니다.'
            },
            'E': {
                '현금': '10%', 
                '채권': '20%', 
                '주식': '60%', 
                '금/원자재': '10%',
                '설명': '경기 확장기에는 주식, 특히 성장주와 경기민감주 비중을 높게 유지하세요. 원자재와 부동산 관련 투자도 수익성이 좋습니다. 채권 비중은 낮추고 듀레이션을 짧게 유지하는 것이 좋습니다.'
            },
            'F': {
                '현금': '20%', 
                '채권': '30%', 
                '주식': '40%', 
                '금/원자재': '10%',
                '설명': '호황기 말기에는 점진적으로 위험자산 비중을 줄이고 안전자산 비중을 늘리세요. 주식은 경기방어주 비중을 높이고, 채권은 단기물 위주로 구성하는 것이 좋습니다. 고평가된 성장주는 이익실현을 고려하세요.'
            }
        }
    
    def determine_position(self, indicators):
        """경제 지표를 바탕으로 현재 경제 사이클 위치 결정"""
        stage_scores = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0}
        
        # GDP 성장률 평가
        if 'GDP' in indicators:
            gdp = indicators['GDP']
            # GDP가 마이너스인 경우 C나 D 단계에 더 적합 (점수 로직 수정)
            if gdp < 0:
                if gdp > -1.0:
                    # 경미한 마이너스 성장은 D와 C 쪽에 더 가중치
                    stage_scores['C'] += 2
                    stage_scores['D'] += 2
                    # E 점수 감소
                    stage_scores['E'] += 0
                else:
                    # 심각한 마이너스 성장은 B와 C 단계
                    stage_scores['B'] += 1
                    stage_scores['C'] += 3
            else:  # 기존 양수 GDP 로직
                if gdp >= 3.0:
                    stage_scores['F'] += 2
                    stage_scores['E'] += 1
                elif gdp >= 1.0:
                    stage_scores['E'] += 2
                    stage_scores['F'] += 1
                else:  # 0~1% 사이
                    stage_scores['D'] += 2
                    stage_scores['E'] += 1
        
        # 인플레이션 평가
        if 'Inflation' in indicators:
            inf = indicators['Inflation']
            if inf >= 5.0:
                stage_scores['B'] += 2
                stage_scores['A'] += 1
            elif inf >= 4.0:
                stage_scores['A'] += 2
                stage_scores['F'] += 1
            elif inf >= 2.0:
                # 2~4% 인플레이션은 경기 활성화 (E와 F에 더 적합)
                stage_scores['F'] += 1
                stage_scores['E'] += 1
            elif inf >= 1.0:
                stage_scores['E'] += 1
                stage_scores['D'] += 1
            else:
                stage_scores['D'] += 2
                stage_scores['C'] += 1
        
        # 금리 평가
        if 'Interest' in indicators:
            rate = indicators['Interest']
            if rate >= 10.0:  # 금리 10% 이상 - F-A-B 구간 (버블기)
                stage_scores['A'] += 3
                stage_scores['B'] += 2
                stage_scores['F'] += 1
            elif rate >= 8.0:  # 금리 8~10% - B 구간 (긴축정책 정점)
                stage_scores['B'] += 3
                stage_scores['A'] += 1
                stage_scores['C'] += 1  
            elif rate >= 6.0:  # 금리 6~8% - A 또는 C 구간
                stage_scores['A'] += 2
                stage_scores['C'] += 1
                stage_scores['F'] += 1
            elif rate >= 5.0:  # 금리 5~6% - F 또는 C 구간
                stage_scores['F'] += 2
                stage_scores['C'] += 1
            elif rate >= 3.0:  # 금리 3~5% - E 구간 (완만한 금리 상승)
                stage_scores['E'] += 2  # 가중치 조정
                stage_scores['F'] += 1
            else:  # 금리 3% 미만 - C-D-E 구간 (회복 및 위기기)
                stage_scores['D'] += 3
                stage_scores['C'] += 1
                stage_scores['E'] += 1
        
        # 실업률 평가
        if 'Unemployment' in indicators:
            unemp = indicators['Unemployment']
            if unemp >= 8.0:
                stage_scores['C'] += 2
                stage_scores['D'] += 1
            elif unemp >= 6.0:
                stage_scores['D'] += 2
                stage_scores['B'] += 1
            elif unemp >= 5.0:
                stage_scores['B'] += 2
                stage_scores['A'] += 1
            elif unemp >= 4.0:
                stage_scores['E'] += 1  # 가중치 조정
                stage_scores['D'] += 1
            else:
                stage_scores['F'] += 2
                stage_scores['E'] += 1
        
        # VIX 지수 평가
        if 'Vix' in indicators:
            vix = indicators['Vix']
            if vix >= 40.0:  # 40 이상 - 공포 국면 (C단계 가능성 높음)
                stage_scores['C'] += 3
                stage_scores['B'] += 1
            elif vix >= 30.0:  # 30-40 - 불안감 (B단계 신호)
                stage_scores['B'] += 2
                stage_scores['A'] += 1
                stage_scores['C'] += 1
            elif vix >= 20.0:  # 20-30 - 정상 변동성 (A 또는 E 단계)
                stage_scores['A'] += 1
                stage_scores['E'] += 1
            else:  # 20 이하 - 시장 안정 (E-F 단계)
                stage_scores['E'] += 1
                stage_scores['F'] += 2
        
        # 달러지수(DXY) 평가
        if 'Dxy' in indicators:
            dxy = indicators['Dxy']
            dxy_trend = None
            
            # 상세 정보에서 추세 데이터 가져오기 
            if hasattr(indicators, 'get') and indicators.get('Dxy_trend'):
                dxy_trend = indicators.get('Dxy_trend')
                
            # 추세 기반 평가
            if dxy_trend:
                if dxy_trend == "강한_상승":
                    stage_scores['A'] += 3
                    stage_scores['B'] += 2
                elif dxy_trend == "상승":
                    stage_scores['A'] += 2
                    stage_scores['B'] += 1
                elif dxy_trend == "강한_하락":
                    stage_scores['D'] += 2
                    stage_scores['E'] += 2
                elif dxy_trend == "하락":
                    stage_scores['D'] += 1
                    stage_scores['E'] += 1
                # 안정 상태는 추가 점수 없음
            else:
                # 추세 정보가 없는 경우 절대값 기준으로 평가
                # 달러지수 범위는 일반적으로 90-105 사이에서 움직임
                if dxy >= 105:  # 강한 달러 - A, B 단계 가능성
                    stage_scores['A'] += 1
                    stage_scores['B'] += 2
                elif dxy >= 100:  # 달러 강세 - B, C 단계
                    stage_scores['B'] += 1
                    stage_scores['C'] += 1
                elif dxy <= 90:  # 약한 달러 - E, F 단계
                    stage_scores['E'] += 1
                    stage_scores['F'] += 1
        
        # 가장 높은 점수를 가진 단계 결정
        max_stage = max(stage_scores, key=stage_scores.get)
        max_score = stage_scores[max_stage]
        
        # 결과 반환
        stage_order = ['A', 'B', 'C', 'D', 'E', 'F']
        position = stage_order.index(max_stage) / len(stage_order)
        
        return position, max_stage, stage_scores
    
    def plot_egg_model(self, indicators=None, historical_data=None, show_plot=True):
        """
        코스탈라니 달걀모형 시각화
        
        Args:
            indicators (dict): 현재 거시경제 지표 딕셔너리
            historical_data (dict): 과거 데이터 딕셔너리
            show_plot (bool): 그래프를 화면에 표시할지 여부
            
        Returns:
            tuple: (fig, ax, position, stage) - 그래프, 축, 위치 좌표, 단계
        """
        # 한글 폰트 설정
        system = platform.system()
        if system == 'Darwin':  # macOS
            plt.rcParams['font.family'] = 'AppleGothic'
        elif system == 'Windows':
            plt.rcParams['font.family'] = 'Malgun Gothic'
        else:  # Linux 등
            plt.rcParams['font.family'] = 'DejaVu Sans'
        
        # 유니코드 문제 해결
        plt.rcParams['axes.unicode_minus'] = False
        
        # 빈 지표 처리
        if indicators is None:
            indicators = {}
        
        # 기본값 설정 (None 값 처리)
        if 'GDP' not in indicators or indicators['GDP'] is None:
            indicators['GDP'] = 2.0
        if 'Inflation' not in indicators or indicators['Inflation'] is None:
            indicators['Inflation'] = 2.0
        if 'Interest' not in indicators or indicators['Interest'] is None:
            indicators['Interest'] = 3.0
        if 'Unemployment' not in indicators or indicators['Unemployment'] is None:
            indicators['Unemployment'] = 4.0
        
        # 현재 위치 결정
        position, stage, scores = self.determine_position(indicators)
        # 디버깅: 각 단계별 점수 확인
        print("\n경제 사이클 단계별 점수:")
        for stage_key, score in scores.items():
            print(f"{stage_key}: {score}")
        
        # 디버깅: 주요 값 확인
        print(f"[DEBUG] indicators: {indicators}")
        position, stage, scores = self.determine_position(indicators)
        print(f"[DEBUG] position: {position}, stage: {stage}")
        if position is None or stage is None:
            print("[경고] position 또는 stage가 None입니다. 시각화가 제대로 표시되지 않을 수 있습니다.")
        
        # 그림 설정 - 여백 조정하여 위쪽으로 이동
        fig, ax = plt.subplots(figsize=(14, 12))
        plt.subplots_adjust(top=0.9, bottom=0.1)  # 상하 여백 조정
        
        # 축 범위 조정 - y축 범위를 위로 이동
        ax.set_xlim(-6, 6)
        ax.set_ylim(-9, 9)  # 상단 여백 확보
        
        # 배경 설정 - 미묘한 그라데이션 배경 추가
        rect = patches.Rectangle((-6, -9), 12, 18, facecolor='#f8f9fa', alpha=0.3)
        ax.add_patch(rect)
        
        # 타원 그리기 - 위치를 위로 이동
        ellipse = patches.Ellipse((0, 1), 8, 10, fill=False, color='black', linewidth=2)
        ax.add_patch(ellipse)
        
        # 중앙선 그리기
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # 경제 사이클 단계 표시 - 위치를 위로 이동
        positions = {
            'A': (0, 6),
            'B': (4, 3.5),
            'C': (4, -1.5),
            'D': (0, -4),
            'E': (-4, -1.5),
            'F': (-4, 3.5)
        }
        
        # 단계 레이블 폰트 사이즈 조정
        stage_fontsize = 14
        description_fontsize = 7
        
        for key, pos in positions.items():
            # 주요 위치 점 표시
            ax.plot(pos[0], pos[1], 'ko', markersize=12)
            
            # 단계 레이블 위치 조정
            if key == 'A':
                ax.text(pos[0]-0.3, pos[1]+0.5, key, fontsize=stage_fontsize, fontweight='bold')
                ax.text(pos[0], pos[1]-1.0, self.stages[key], fontsize=description_fontsize, ha='center')
            elif key == 'B':
                ax.text(pos[0]+0.5, pos[1]+0.5, key, fontsize=stage_fontsize, fontweight='bold')
                ax.text(pos[0]+0.8, pos[1], self.stages[key], fontsize=description_fontsize, ha='left')
            elif key == 'C':
                ax.text(pos[0]+0.5, pos[1]-0.5, key, fontsize=stage_fontsize, fontweight='bold')
                ax.text(pos[0]+0.8, pos[1], self.stages[key], fontsize=description_fontsize, ha='left')
            elif key == 'D':
                ax.text(pos[0]-0.3, pos[1]-0.8, key, fontsize=stage_fontsize, fontweight='bold')
                ax.text(pos[0], pos[1]-1.2, self.stages[key], fontsize=description_fontsize, ha='center')
            elif key == 'E':
                ax.text(pos[0]-0.5, pos[1]-0.5, key, fontsize=stage_fontsize, fontweight='bold')
                ax.text(pos[0]-0.8, pos[1], self.stages[key], fontsize=description_fontsize, ha='right')
            elif key == 'F':
                ax.text(pos[0]-0.5, pos[1]+0.5, key, fontsize=stage_fontsize, fontweight='bold')
                ax.text(pos[0]-0.8, pos[1], self.stages[key], fontsize=description_fontsize, ha='right')
        
        # 호황/불황 영역 표시
        left_semi = patches.Wedge((0, 1), 4.5, 90, 270, width=9, 
                                 fc='#ffcccc', alpha=0.2, label='호황기')
        right_semi = patches.Wedge((0, 1), 4.5, 270, 90, width=9, 
                                  fc='#ccccff', alpha=0.2, label='불황기')
        ax.add_patch(left_semi)
        ax.add_patch(right_semi)
        
        # 투자 전략 표시 - 위치와 폰트 크기 조정
        text_positions = {
            'B': (5.2, 2.5, '예금인출/채권투자'),
            'C': (5.2, -2.5, '채권매도/부동산투자'),
            'E': (-5.2, -2.5, '주식투자/부동산매도'),
            'F': (-5.2, 2.5, '예금입금/주식매도')
        }
        
        strategy_fontsize = 8
        for key, (x, y, text) in text_positions.items():
            ax.text(x, y, text, fontsize=strategy_fontsize, ha='center', 
                   bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray', boxstyle='round,pad=0.2'))
        
        # 금리선 표시 - 위치 조정 및 더 명확하게 표시
        ax.plot([0, 4], [6, 3.5], 'r--', alpha=0.5, linewidth=1.5)  # A-B 금리 10% 이상 라인
        ax.plot([-4, 0], [3.5, 6], 'r--', alpha=0.5, linewidth=1.5)  # F-A 금리 10% 이상 라인
        
        ax.plot([0, -4], [-4, -1.5], 'g--', alpha=0.5, linewidth=1.5)  # D-E 금리 3% 미만 라인
        ax.plot([4, 0], [-1.5, -4], 'g--', alpha=0.5, linewidth=1.5)  # C-D 금리 3% 미만 라인

        # 금리 레이블 위치 조정
        ax.text(0, 4.5, '금리 10%', fontsize=9, ha='center', color='red',
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        ax.text(0, 1, '금리 5%', fontsize=9, ha='center', color='red',
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        ax.text(0, -2.5, '금리 3%', fontsize=9, ha='center', color='red',
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        
        # 영역 설명 추가 - 위치 조정 및 더 명확하게 표시
        ax.text(-2, 5, '[호황기]', fontsize=11, fontweight='bold', 
               bbox=dict(facecolor='#ffcccc', alpha=0.3, boxstyle='round,pad=0.2'))
        ax.text(3, 5, '[불황기]', fontsize=11, fontweight='bold',
               bbox=dict(facecolor='#ccccff', alpha=0.3, boxstyle='round,pad=0.2'))
        
        # 각 구간별 설명 추가 - 위치 조정
        ax.text(0, 5.3, '버블(F-A-B)', fontsize=8, color='darkred', ha='center',
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        ax.text(3, 1, '침체(B-C)', fontsize=8, color='darkblue', ha='center',
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        ax.text(0, -3.3, '위기/회복(C-D-E)', fontsize=8, color='darkgreen', ha='center',
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        ax.text(-3, 1, '호황(E-F)', fontsize=8, color='darkred', ha='center',
               bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        
        # 현재 위치 표시
        if position is not None:
            # 수정: position은 0부터 1 사이의 값으로 변환 (stage_order 인덱스를 0~1 사이로 변환)
            # 수정: 올바른 각도로 변환
            position_angle = position * 2 * np.pi
            
            # 타원 상의 위치 계산
            stage_order = ['A', 'B', 'C', 'D', 'E', 'F']
            stage_idx = stage_order.index(stage)
            
            # 각 단계별 좌표 사용 (타원 위에 정확히 표시)
            pos_coords = {'A': (0, 6), 'B': (4, 3.5), 'C': (4, -1.5), 
                          'D': (0, -4), 'E': (-4, -1.5), 'F': (-4, 3.5)}
            
            # 주변에 랜덤한 오프셋 추가하여 포인트가 정확히 단계 위치와 겹치지 않도록 함
            x_offset = random.uniform(-0.2, 0.2)
            y_offset = random.uniform(-0.2, 0.2)
            
            x = pos_coords[stage][0] + x_offset
            y = pos_coords[stage][1] + y_offset
            
            # 현재 날짜 가져오기
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # 빨간색 큰 원으로 현재 위치 표시
            ax.plot(x, y, 'ro', markersize=15, alpha=0.7)
            
            # 현재 위치 레이블에 일자 추가
            position_text = ax.text(x, y+0.7, f'현재 위치 ({current_date})', fontsize=10, ha='center', color='red',
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', boxstyle='round,pad=0.2'))
            
            # 현재 경제 사이클 및 지표 값 표시 - 컴팩트하게 수정
            # 박스 제거하고 투명 배경으로 변경
            indicator_text = f"현재 경제 사이클: {stage} 단계\n"
            indicator_text += f"({self.stages[stage]})\n\n"
            indicator_text += "현재 거시경제 지표:\n"
            
            # 각 지표를 두 줄로 표시하여 가독성 향상
            indicator_line1 = f"GDP: {indicators['GDP']:.1f}%   Inflation: {indicators['Inflation']:.1f}%\n"
            indicator_line2 = f"Interest: {indicators['Interest']:.1f}%   Unemployment: {indicators['Unemployment']:.1f}%"
            indicator_text += indicator_line1 + indicator_line2
            
            # 지표값 정보 - 우측 상단으로 위치 이동, 투명 배경 적용
            ax.text(5.8, 6.0, indicator_text, fontsize=8, ha='right',
                   color='black', bbox=dict(facecolor='white', alpha=0.7, 
                                           edgecolor='gray', boxstyle='round,pad=0.3'))
            
            # 투자 전략 표시 - 우측 하단에 강조하여 표시
            strategy_text = f"투자 전략: {self.strategies[stage]}"
            ax.text(5.8, 8.0, strategy_text, fontsize=10, ha='right', color='blue',
                   bbox=dict(facecolor='#e6f2ff', alpha=0.9, edgecolor='blue', boxstyle='round,pad=0.5'))
        
        # 히스토리 데이터 표시 (날짜 순서대로)
        if historical_data is not None:
            dates = sorted(historical_data.keys())
            prev_x, prev_y = None, None
            
            for i, date in enumerate(dates):
                data = historical_data[date]
                hist_position, hist_stage, _ = self.determine_position(data)
                
                # 위치 계산
                angle = hist_position * (2 * np.pi)
                x = 4 * np.cos(angle)
                y = 5 * np.sin(angle)
                
                # 점 표시
                ax.plot(x, y, 'bo', markersize=6, alpha=0.6)
                
                # 날짜 레이블 표시 - 겹치지 않도록 위치 조정
                if len(dates) <= 10:  # 날짜가 10개 이하일 때만 모든 날짜 표시
                    # 각 날짜마다 다른 위치에 텍스트 배치
                    offset_x = 0.3 * np.cos(angle + np.pi/2)
                    offset_y = 0.3 * np.sin(angle + np.pi/2)
                    
                    # 첫 번째와 마지막 날짜는 특별히 표시
                    if i == 0 or i == len(dates) - 1:
                        if i == 0:  # 첫 번째 날짜
                            label_text = f"{date} (시작)"
                            bbox_props = dict(facecolor='lightgreen', alpha=0.7, edgecolor='green', boxstyle='round,pad=0.2')
                        else:  # 마지막 날짜
                            label_text = f"{date} (최근)"
                            bbox_props = dict(facecolor='lightyellow', alpha=0.7, edgecolor='orange', boxstyle='round,pad=0.2')
                            
                        ax.text(x + offset_x, y + offset_y, label_text, fontsize=8, ha='center', va='center',
                               bbox=bbox_props)
                    else:
                        ax.text(x + offset_x, y + offset_y, date, fontsize=7, ha='center', va='center',
                               bbox=dict(facecolor='white', alpha=0.6, edgecolor='blue', boxstyle='round,pad=0.1'))
                
                # 이전 점과 연결선 그리기
                if prev_x is not None and prev_y is not None:
                    ax.arrow(prev_x, prev_y, (x-prev_x)*0.9, (y-prev_y)*0.9, 
                            head_width=0.2, head_length=0.3, fc='blue', ec='blue', alpha=0.5)
                
                prev_x, prev_y = x, y
                
            # 히스토리 데이터가 있는 경우 범례 추가
            if len(dates) > 0:
                ax.plot([], [], 'bo-', markersize=6, alpha=0.6, label='과거 데이터')
                if position is not None:
                    # 현재 날짜 가져오기
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    ax.plot([], [], 'ro', markersize=10, alpha=0.7, label=f'현재 위치 ({current_date})')
                ax.legend(loc='upper right', fontsize=8, framealpha=0.7)
        
        # 축 설정
        ax.set_xlim(-6, 6)
        ax.set_ylim(-9, 9)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 경제 사이클 단계 설명을 하단에 배치 (범례 박스)
        # 중첩 방지를 위해 크기와 위치 조정
        legend_box = patches.FancyBboxPatch(
            (-5.9, -8.1), 11.8, 1.4, boxstyle=patches.BoxStyle("Round", pad=0.3),
            facecolor='white', alpha=0.7, edgecolor='gray'
        )
        ax.add_patch(legend_box)
        
        # 범례 제목 - 위치 조정
        ax.text(-5.5, -6.0, "경제 사이클 단계 설명:", fontsize=10, fontweight='bold')
        
        # 각 단계별 설명
        stages_description = [
            ('A', '인플레이션 우려와 금리 상승', '주식시장 어려움, 안전자산 비중 확대'),
            ('B', '인플레이션 억제를 위한 긴축정책', '소비재, 필수품 관련 주식 중심 투자'),
            ('C', '경기 침체와 소비 감소', '채권과 배당주 비중 확대, 저평가 주식 매수 시작'),
            ('D', '경기 회복을 위한 완화정책', '경기 민감주와 성장주 관심 확대'),
            ('E', '경기 확장과 소비 증가', '경기 민감주와 성장주 투자 유리'),
            ('F', '호황기와 최대 경기 활성화', '고점 접근, 위험자산 비중 점진적 축소 검토')
        ]
        
        # 현재 단계는 강조표시
        colors = {'A': 'black', 'B': 'black', 'C': 'black', 'D': 'black', 'E': 'black', 'F': 'black'}
        if position is not None:
            colors[stage] = 'red'  # 현재 단계는 빨간색으로 강조
        
        # 3열 2행 배치로 변경하여 중첩 방지
        x_positions = [-5.5, -1.0, 3.5]  # 왼쪽, 중앙, 오른쪽 열 x 위치
        y_start = -7.2  # 시작 y 위치 - 위치 조정
        y_gap = -0.6  # 행 간격 (더 넓게 조정)
        
        for i, (stage_key, desc, strategy) in enumerate(stages_description):
            col = i % 3  # 0, 1, 2 (왼쪽, 중간, 오른쪽 열)
            row = i // 3  # 0 또는 1 (윗줄, 아랫줄)
            x_pos = x_positions[col]
            y_pos = y_start + (row * y_gap)
            
            # 원 마커와 단계 표시
            color = colors[stage_key]
            if color == 'red':  # 현재 단계 강조
                circle = patches.Circle((x_pos, y_pos), radius=0.15, facecolor='red', alpha=0.8)
                weight = 'bold'
                box_color = '#ffeeee'  # 연한 빨간색 배경
            else:
                circle = patches.Circle((x_pos, y_pos), radius=0.15, facecolor='darkblue', alpha=0.6)
                weight = 'normal'
                box_color = 'white'
                
            ax.add_patch(circle)
            
            # 단계 점수 표시 (괄호 안에 점수 표시)
            score_text = ''
            if scores is not None:
                score = scores[stage_key]
                score_text = f" ({score}점"
                
                # 점수 내역 표시
                score_details = []
                if stage_key == 'A' and 'GDP' in indicators and indicators['GDP'] < 3.0:
                    score_details.append("GDP↓")
                if stage_key == 'A' and 'Inflation' in indicators and indicators['Inflation'] >= 4.0:
                    score_details.append("인플↑")
                if stage_key == 'A' and 'Interest' in indicators and indicators['Interest'] >= 6.0:
                    score_details.append("금리↑")
                    
                if stage_key == 'B' and 'Inflation' in indicators and indicators['Inflation'] >= 5.0:
                    score_details.append("인플↑↑")
                if stage_key == 'B' and 'Interest' in indicators and indicators['Interest'] >= 8.0:
                    score_details.append("금리↑↑")
                if stage_key == 'B' and 'Unemployment' in indicators and indicators['Unemployment'] >= 5.0:
                    score_details.append("실업↑")
                    
                if stage_key == 'C' and 'GDP' in indicators and indicators['GDP'] < 0:
                    score_details.append("GDP↓↓")
                if stage_key == 'C' and 'Inflation' in indicators and indicators['Inflation'] <= 3.0:
                    score_details.append("인플↓")
                if stage_key == 'C' and 'Unemployment' in indicators and indicators['Unemployment'] >= 7.0:
                    score_details.append("실업↑↑")
                    
                if stage_key == 'D' and 'GDP' in indicators and indicators['GDP'] >= 0 and indicators['GDP'] < 1.0:
                    score_details.append("GDP→")
                if stage_key == 'D' and 'Inflation' in indicators and indicators['Inflation'] <= 1.0:
                    score_details.append("인플↓↓")
                if stage_key == 'D' and 'Interest' in indicators and indicators['Interest'] < 3.0:
                    score_details.append("금리↓↓")
                    
                if stage_key == 'E' and 'GDP' in indicators and indicators['GDP'] >= 1.0 and indicators['GDP'] < 3.0:
                    score_details.append("GDP↑")
                if stage_key == 'E' and 'Inflation' in indicators and indicators['Inflation'] >= 1.0 and indicators['Inflation'] < 2.0:
                    score_details.append("인플→")
                if stage_key == 'E' and 'Interest' in indicators and indicators['Interest'] >= 3.0 and indicators['Interest'] < 5.0:
                    score_details.append("금리→")
                    
                if stage_key == 'F' and 'GDP' in indicators and indicators['GDP'] >= 3.0:
                    score_details.append("GDP↑↑")
                if stage_key == 'F' and 'Inflation' in indicators and indicators['Inflation'] >= 2.0 and indicators['Inflation'] < 4.0:
                    score_details.append("인플↑")
                if stage_key == 'F' and 'Unemployment' in indicators and indicators['Unemployment'] < 4.0:
                    score_details.append("실업↓↓")
                
                if score_details:
                    score_text += f": {', '.join(score_details)}"
                score_text += ")"
            
            # 단계 표시에 점수 추가
            ax.text(x_pos, y_pos, f"{stage_key}", color='white', fontsize=9, ha='center', va='center', fontweight='bold')
            
            # 설명 텍스트 - 중첩을 방지하기 위해 텍스트 줄바꿈 및 위치 조정
            desc_text = f"{desc}{score_text}\n{strategy}"
                
            # 텍스트 수직 중앙 정렬을 위해 va='center' 추가
            ax.text(x_pos + 0.3, y_pos, desc_text, fontsize=7.5, color=color, fontweight=weight,
                  bbox=dict(facecolor=box_color, alpha=0.7, boxstyle='round,pad=0.2'), ha='left', va='center')
        
        # 전체 그래프에 날짜 및 제목 표시
        current_date = datetime.now().strftime("%Y-%m-%d")
        plt.title(f'앙드레 코스탈라니의 달걀모형 경제 사이클 분석 ({current_date})', fontsize=15, pad=20)
        
        # 워터마크 추가 - 겹치지 않도록 위치 조정
        ax.text(0, -8.7, "© 코스탈라니 달걀모형 분석 - 경제 사이클 예측 시스템", 
                fontsize=8, color='gray', alpha=0.7, ha='center')
        
        # 현재 위치에 점수 표시
        if position is not None and stage is not None and scores is not None:
            max_score = scores[stage]
            # 현재 단계에 대한 점수 표시를 위치 근처에 추가
            ax.text(x + 0.7, y + 0.5, f"점수: {max_score}", fontsize=9, ha='center', 
                   color='red', fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', boxstyle='round,pad=0.2'))
        
        if show_plot:
            plt.show()
        
        return fig, ax, position, stage

def main():
    """
    메인 함수 - 사용자 입력을 받아 달걀모형 분석 실행
    """
    model = KostalanyEggModel()
    
    print("\n=== 코스탈라니 달걀모형 경제 사이클 분석 ===\n")
    print("현재 거시경제 지표를 입력해주세요 (취소하려면 'q' 입력):\n")
    
    try:
        # GDP 입력
        while True:
            gdp_input = input("GDP 성장률(%): ").strip()
            if not gdp_input:
                print("입력이 없습니다. 다시 시도해주세요.")
                continue
            if gdp_input.lower() == 'q':
                print("분석이 취소되었습니다.")
                return
            
            try:
                gdp = float(gdp_input.replace(',', '.'))
                break
            except ValueError:
                print("올바른 숫자 형식으로 입력해주세요. (예: 1.5)")
        
        # 인플레이션 입력
        while True:
            inflation_input = input("인플레이션(%): ").strip()
            if not inflation_input:
                print("입력이 없습니다. 다시 시도해주세요.")
                continue
            if inflation_input.lower() == 'q':
                print("분석이 취소되었습니다.")
                return
            
            try:
                inflation = float(inflation_input.replace(',', '.'))
                break
            except ValueError:
                print("올바른 숫자 형식으로 입력해주세요. (예: 1.5)")
        
        # 금리 입력
        while True:
            interest_input = input("기준금리(%): ").strip()
            if not interest_input:
                print("입력이 없습니다. 다시 시도해주세요.")
                continue
            if interest_input.lower() == 'q':
                print("분석이 취소되었습니다.")
                return
            
            try:
                interest = float(interest_input.replace(',', '.'))
                break
            except ValueError:
                print("올바른 숫자 형식으로 입력해주세요. (예: 1.5)")
        
        # 실업률 입력
        while True:
            unemployment_input = input("실업률(%): ").strip()
            if not unemployment_input:
                print("입력이 없습니다. 다시 시도해주세요.")
                continue
            if unemployment_input.lower() == 'q':
                print("분석이 취소되었습니다.")
                return
            
            try:
                unemployment = float(unemployment_input.replace(',', '.'))
                break
            except ValueError:
                print("올바른 숫자 형식으로 입력해주세요. (예: 1.5)")
        
        indicators = {
            'GDP': gdp,
            'Inflation': inflation,
            'Interest': interest,
            'Unemployment': unemployment
        }
        
        # 히스토리 데이터 입력 여부 확인
        use_history = None
        while use_history is None:
            history_input = input("\n히스토리 데이터를 입력하시겠습니까? (y/n): ").lower().strip()
            if history_input == 'y' or history_input == '예':
                use_history = True
            elif history_input == 'n' or history_input == '아니오':
                use_history = False
            elif history_input.startswith('ㅇ'):  # 'ㅇ', 'ㅇㅇ' 등을 '예'로 인식
                use_history = True
                print("'예'로 인식했습니다.")
            elif history_input.startswith('ㄴ') or history_input.startswith('ㅁ') or history_input.startswith('ㅈ'):
                # 'ㄴ', 'ㅁ', 'ㅈ' 등을 '아니오'로 인식
                use_history = False
                print("'아니오'로 인식했습니다.")
            else:
                print("'y' 또는 'n'으로 입력해주세요.")
        
        historical_data = {}
        
        if use_history:
            while True:
                print("\n과거 데이터 입력 (종료하려면 날짜에 'q' 입력):")
                date = input("날짜 (YYYY-MM): ")
                if date.lower() == 'q':
                    break
                
                try:
                    # GDP 입력
                    while True:
                        hist_gdp_input = input(f"{date} GDP 성장률(%): ").strip()
                        if hist_gdp_input.lower() == 'q':
                            break
                        
                        try:
                            hist_gdp = float(hist_gdp_input.replace(',', '.'))
                            break
                        except ValueError:
                            print("올바른 숫자 형식으로 입력해주세요. (예: 1.5)")
                    
                    if hist_gdp_input.lower() == 'q':
                        break
                    
                    # 인플레이션 입력
                    while True:
                        hist_inf_input = input(f"{date} 인플레이션(%): ").strip()
                        if hist_inf_input.lower() == 'q':
                            break
                        
                        try:
                            hist_inf = float(hist_inf_input.replace(',', '.'))
                            break
                        except ValueError:
                            print("올바른 숫자 형식으로 입력해주세요. (예: 1.5)")
                    
                    if hist_inf_input.lower() == 'q':
                        break
                    
                    # 금리 입력
                    while True:
                        hist_int_input = input(f"{date} 기준금리(%): ").strip()
                        if hist_int_input.lower() == 'q':
                            break
                        
                        try:
                            hist_int = float(hist_int_input.replace(',', '.'))
                            break
                        except ValueError:
                            print("올바른 숫자 형식으로 입력해주세요. (예: 1.5)")
                    
                    if hist_int_input.lower() == 'q':
                        break
                    
                    # 실업률 입력
                    while True:
                        hist_unemp_input = input(f"{date} 실업률(%): ").strip()
                        if hist_unemp_input.lower() == 'q':
                            break
                        
                        try:
                            hist_unemp = float(hist_unemp_input.replace(',', '.'))
                            break
                        except ValueError:
                            print("올바른 숫자 형식으로 입력해주세요. (예: 1.5)")
                    
                    if hist_unemp_input.lower() == 'q':
                        break
                    
                    historical_data[date] = {
                        'GDP': hist_gdp,
                        'Inflation': hist_inf,
                        'Interest': hist_int,
                        'Unemployment': hist_unemp
                    }
                except Exception as e:
                    print(f"입력 처리 중 오류가 발생했습니다: {e}")
                    print("다시 시도해주세요.")
        
        # 달걀모형 분석 실행
        fig, ax, position, stage = model.plot_egg_model(indicators, historical_data)
        
        # 자세한 분석 결과 출력
        print("\n=== 분석 결과 ===\n")
        print(f"현재 경제 사이클 단계: {stage} ({model.stages[stage]})")
        print(f"투자 전략 제안: {model.strategies[stage]}")
        
        # 자산배분 전략 출력
        print("\n=== 현재 단계의 자산배분 전략 ===")
        print(f"[{stage}단계: {model.stages[stage]}]")
        for asset, allocation in model.asset_allocation[stage].items():
            if asset != '설명':
                print(f"- {asset}: {allocation}")
        print(f"\n{model.asset_allocation[stage]['설명']}")
        
        # 모든 단계의 자산배분 전략 보기 옵션
        print("\n모든 단계의 자산배분 전략을 보시겠습니까? (y/n): ", end="")
        view_all = input().lower()
        if view_all == 'y':
            print("\n=== 경제 사이클 단계별 자산배분 전략 ===")
            for key in ['A', 'B', 'C', 'D', 'E', 'F']:
                print(f"\n[{key}단계: {model.stages[key]}]")
                print(f"투자 전략: {model.strategies[key]}")
                for asset, allocation in model.asset_allocation[key].items():
                    if asset != '설명':
                        print(f"- {asset}: {allocation}")
                print(f"설명: {model.asset_allocation[key]['설명']}")
        
        print("\n거시경제 지표 평가:")
        
        # GDP 성장률 평가 분석
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
        print(f"GDP 성장률 {gdp}%: {gdp_eval}")
        
        # 인플레이션 평가 분석
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
        print(f"인플레이션 {inflation}%: {inflation_eval}")
        
        # 기준금리 평가 분석
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
        print(f"기준금리 {interest}%: {interest_eval}")
        
        # 실업률 평가 분석
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
        print(f"실업률 {unemployment}%: {unemployment_eval}")
        
        # 각 단계별 점수 근거 설명
        print("\n경제 사이클 단계별 점수 근거:")
        _, _, scores = model.determine_position(indicators)
        
        # GDP 기반 점수 설명
        print("\nGDP 성장률 기반 점수:")
        if gdp < 0:
            if gdp > -1.0:
                print("- 경미한 마이너스 성장: C 단계 +2점, D 단계 +2점")
            else:
                print("- 심각한 마이너스 성장: B 단계 +1점, C 단계 +3점")
        else:
            if gdp >= 3.0:
                print("- 높은 GDP 성장률: F 단계 +2점, E 단계 +1점")
            elif gdp >= 1.0:
                print("- 양호한 GDP 성장률: E 단계 +2점, F 단계 +1점")
            else:
                print("- 낮은 양의 GDP 성장률: D 단계 +2점, E 단계 +1점")
        
        # 인플레이션 기반 점수 설명
        print("\n인플레이션 기반 점수:")
        if inflation >= 5.0:
            print("- 높은 인플레이션: B 단계 +2점, A 단계 +1점")
        elif inflation >= 4.0:
            print("- 상당한 인플레이션: A 단계 +2점, F 단계 +1점")
        elif inflation >= 2.0:
            print("- 적정 인플레이션: F 단계 +1점, E 단계 +1점")
        elif inflation >= 1.0:
            print("- 낮은 인플레이션: E 단계 +1점, D 단계 +1점")
        else:
            print("- 매우 낮은 인플레이션: D 단계 +2점, C 단계 +1점")
        
        # 금리 기반 점수 설명
        print("\n기준금리 기반 점수:")
        if interest >= 10.0:
            print("- 매우 높은 금리: A 단계 +3점, B 단계 +2점, F 단계 +1점")
        elif interest >= 8.0:
            print("- 높은 금리: B 단계 +3점, A 단계 +1점, C 단계 +1점")
        elif interest >= 6.0:
            print("- 상당한 금리: A 단계 +2점, C 단계 +1점, F 단계 +1점")
        elif interest >= 5.0:
            print("- 중간 금리: F 단계 +2점, C 단계 +1점")
        elif interest >= 3.0:
            print("- 적정 금리: E 단계 +2점, F 단계 +1점")
        else:
            print("- 낮은 금리: D 단계 +3점, C 단계 +1점, E 단계 +1점")
        
        # 실업률 기반 점수 설명
        print("\n실업률 기반 점수:")
        if unemployment >= 8.0:
            print("- 높은 실업률: C 단계 +2점, D 단계 +1점")
        elif unemployment >= 6.0:
            print("- 상당한 실업률: D 단계 +2점, B 단계 +1점")
        elif unemployment >= 5.0:
            print("- 중간 실업률: B 단계 +2점, A 단계 +1점")
        elif unemployment >= 4.0:
            print("- 적정 실업률: E 단계 +1점, D 단계 +1점")
        else:
            print("- 낮은 실업률: F 단계 +2점, E 단계 +1점")
        
        print("\n총 점수:")
        for stage_key, score in scores.items():
            print(f"{stage_key} 단계: {score}점")
        
        # 결과 저장
        os.makedirs('output', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/kostalany_model_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n분석 결과 이미지가 저장되었습니다: {filename}")
        
    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main() 