import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.font_manager as fm
import platform
import os

class CostalanyEggModel:
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
        
        # 각 단계별 주요 거시경제 지표 특성
        self.macro_indicators = {
            'A': {'GDP': '감소 시작', '인플레이션': '높음', '금리': '상승', '실업률': '낮음 유지'},
            'B': {'GDP': '감소', '인플레이션': '정점', '금리': '정점', '실업률': '상승 시작'},
            'C': {'GDP': '저점', '인플레이션': '하락', '금리': '하락', '실업률': '높음'},
            'D': {'GDP': '회복 시작', '인플레이션': '낮음', '금리': '저점', '실업률': '정점'},
            'E': {'GDP': '상승', '인플레이션': '상승 시작', '금리': '상승 시작', '실업률': '하락'},
            'F': {'GDP': '정점', '인플레이션': '상승', '금리': '상승', '실업률': '낮음'}
        }
        
        # 각 지표별 평가 기준과 근거 (상세설명)
        self.indicator_criteria = {
            'GDP 성장률': {
                'A': '3% 이하로 감소 시작 (호황기 이후 경기 과열 냉각)',
                'B': '0~1% 사이로 급격히 감소 (긴축정책 효과)',
                'C': '0% 이하 (마이너스 성장, 경기 침체)',
                'D': '0~1% 사이로 회복 시작 (경기 부양책 효과)',
                'E': '1~3% 사이로 상승 (경기 회복 본격화)',
                'F': '3% 이상 (경제 활성화, 호황기)'
            },
            '인플레이션': {
                'A': '4% 이상 (경기 과열로 인한 물가 상승)',
                'B': '5% 이상 (인플레이션 정점)',
                'C': '2~3% 사이로 하락 (수요 감소로 인한 물가 안정)',
                'D': '1% 이하 (경기 침체로 인한 낮은 물가)',
                'E': '1~2% 사이 (적정 인플레이션)',
                'F': '2~4% 사이 (경기 활성화로 인한 물가 상승)'
            },
            '기준금리': {
                'A': '6~10% (인플레이션 억제를 위한 금리 상승)',
                'B': '8~10% 이상 (긴축정책 정점)',
                'C': '5~8% (금리 하락 시작)',
                'D': '1~3% (경기 부양을 위한 저금리)',
                'E': '3~5% (완만한 금리 상승)',
                'F': '5~8% (경기 과열 방지를 위한 금리 정상화)'
            },
            '실업률': {
                'A': '4~5% (아직 낮지만 상승 시작)',
                'B': '5~7% (긴축정책으로 인한 고용 감소)',
                'C': '7~10% 이상 (경기 침체로 인한 높은 실업률)',
                'D': '6~8% (실업률 정점)',
                'E': '4~6% (고용 회복 시작)',
                'F': '2~4% (완전 고용에 가까운 낮은 실업률)'
            },
            '경제 구간': {
                'D-E': '회복기 (금리 3% 미만, 거래량 적음, 주식 소유자 수 적음)',
                'E-F': '호황기 (금리 3~10%, 거래량과 주식 소유자 수 증가)',
                'F-A': '버블 상승기 (금리 10% 이상, 거래량 폭증, 주식 소유자 수 급증)',
                'A-B': '버블 하락기 (금리 10% 이상, 거래량 감소, 주식 소유자 수 감소)',
                'B-C': '침체기 (금리 3~10%, 거래량 증가, 주식 소유자 수 감소)',
                'C-D': '위기기 (금리 3% 미만, 거래량 폭증, 주식 소유자 수 최저)'
            }
        }
        
        # 각 단계의 금리 구간 정의
        self.interest_rate_ranges = {
            'D-E': '3% 미만',         # 회복기
            'E-F': '3% 이상 10% 미만', # 호황기
            'F-A': '10% 이상',        # 버블기(상승)
            'A-B': '10% 이상',        # 버블기(하락)
            'B-C': '3% 이상 10% 미만', # 침체기
            'C-D': '3% 미만'          # 위기기
        }
        
    def plot_egg_model(self, current_position=None, indicators=None, historical_positions=None):
        """코스탈라니 달걀모형 시각화
        
        Args:
            current_position: 현재 위치 (0~1 사이 값)
            indicators: 현재 거시경제 지표 딕셔너리
            historical_positions: {날짜: (위치값, 단계)} 형태의 과거 위치 데이터
        """
        # 한글 폰트 설정 - 시스템별 기본 한글 폰트 사용
        system = platform.system()
        if system == 'Darwin':  # macOS
            plt.rcParams['font.family'] = 'AppleGothic'
        elif system == 'Windows':
            plt.rcParams['font.family'] = 'Malgun Gothic'
        else:  # Linux 등
            plt.rcParams['font.family'] = 'DejaVu Sans'
        
        # 유니코드 문제 해결
        plt.rcParams['axes.unicode_minus'] = False
            
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 타원 그리기
        ellipse = patches.Ellipse((0, 0), 8, 10, fill=False, color='black', linewidth=2)
        ax.add_patch(ellipse)
        
        # 중앙선 그리기
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        
        # 경제 사이클 단계 표시
        positions = {
            'A': (0, 5),
            'B': (4, 2.5),
            'C': (4, -2.5),
            'D': (0, -5),
            'E': (-4, -2.5),
            'F': (-4, 2.5)
        }
        
        for stage, pos in positions.items():
            ax.plot(pos[0], pos[1], 'ko', markersize=12)
            ax.text(pos[0]+0.3, pos[1]+0.3, stage, fontsize=15, fontweight='bold')
            
        # 호황/불황 구분 색상 채우기
        # 왼쪽 반원 (호황기)
        left_semi = patches.Wedge((0, 0), 4.5, 50, 270, width=9, 
                                  fc='#ffcccc', alpha=0.3, label='호황기')
        # 오른쪽 반원 (불황기)
        right_semi = patches.Wedge((0, 0), 4.5, 270, 90, width=9, 
                                   fc='#ccccff', alpha=0.3, label='불황기')
        ax.add_patch(left_semi)
        ax.add_patch(right_semi)
        
        # 구간별 특성 텍스트 추가
        text_positions = {
            'A': (0, 6, ''),
            'B': (5, 2.5, '채권투자'),
            'C': (5, -2.5, '부동산투자'),
            'D': (0, -6, ''),
            'E': (-5, -2.5, '부동산매도'),
            'F': (-5, 2.5, '주식매도'),
            'a1': (0, 3.5, '금리 10%'),
            'a2': (0, 0, '금리 5%'),
            'a3': (0, -3.5, '금리 3%')
        }
        
        for stage, (x, y, text) in text_positions.items():
            if stage.startswith('a'):  # 금리 표시는 빨간색으로
                ax.text(x, y, text, fontsize=12, ha='center', color='red')
            else:
                ax.text(x, y, text, fontsize=12, ha='center')
        
        # 금리선 그리기 (F-A-B 10% 이상, B-C-E-F 3~10%, D-C,D-E 3% 미만)
        ax.plot([0, 4], [5, 2.5], 'r--', alpha=0.5)  # A-B 금리 10% 이상 라인
        ax.plot([-4, 0], [2.5, 5], 'r--', alpha=0.5)  # F-A 금리 10% 이상 라인
        
        ax.plot([4, 4], [2.5, -2.5], 'r:', alpha=0.5)  # B-C 금리 3~10% 라인
        ax.plot([4, 0], [-2.5, -5], 'g--', alpha=0.5)  # C-D 금리 3% 미만 라인
        ax.plot([0, -4], [-5, -2.5], 'g--', alpha=0.5)  # D-E 금리 3% 미만 라인
        ax.plot([-4, -4], [-2.5, 2.5], 'r:', alpha=0.5)  # E-F 금리 3~10% 라인
        
        # 화살표 그리기 (시계 반대 방향)
        arrow_path = [
            (positions['A'][0], positions['A'][1]),
            (positions['B'][0]-1, positions['B'][1]+1),
            (positions['B'][0], positions['B'][1]),
            (positions['C'][0], positions['C'][1]),
            (positions['D'][0], positions['D'][1]),
            (positions['E'][0]+1, positions['E'][1]-1),
            (positions['E'][0], positions['E'][1]),
            (positions['F'][0], positions['F'][1]),
            (positions['A'][0], positions['A'][1])
        ]
        
        # codes 배열 수정: vertices와 같은 길이인 9개로 맞춤
        codes = [Path.MOVETO] + [Path.CURVE3, Path.CURVE3] * 3 + [Path.CURVE3, Path.CURVE3]
        
        path = Path(arrow_path, codes)
        arrow_patch = patches.PathPatch(path, facecolor='none', 
                                        edgecolor='blue', linewidth=2, alpha=0.7)
        ax.add_patch(arrow_patch)
        
        # 영역 설명 추가
        ax.text(-3, 4, '[호황기]', fontsize=14, fontweight='bold')
        ax.text(3, 4, '[불황기]', fontsize=14, fontweight='bold')
        
        # 각 구간별 설명 추가
        ax.text(0, 4, '버블(F-A-B)', fontsize=10, color='darkred', ha='center')
        ax.text(3, 0, '침체(B-C)', fontsize=10, color='darkblue', ha='center')
        ax.text(0, -4, '위기/회복(C-D-E)', fontsize=10, color='darkgreen', ha='center')
        ax.text(-3, 0, '호황(E-F)', fontsize=10, color='darkred', ha='center')
        
        # 과거 위치 데이터는 main.py에서 처리하므로 여기서는 기본 표시만 수행
        if historical_positions:
            # 날짜 순으로 정렬
            sorted_dates = sorted(historical_positions.keys())
            historical_x = []
            historical_y = []
            
            for date in sorted_dates:
                position, stage = historical_positions[date]
                position_angle = position * (2 * np.pi)
                x = 4 * np.cos(position_angle)
                y = 5 * np.sin(position_angle)
                historical_x.append(x)
                historical_y.append(y)
            
            # 기본 경로 표시 (세부 표시는 main.py에서 담당)
            ax.plot(historical_x, historical_y, 'b-', alpha=0.3, linewidth=1)
        
        # 현재 포지션 그리기
        if current_position is not None:
            position_angle = current_position * (2 * np.pi)
            x = 4 * np.cos(position_angle)
            y = 5 * np.sin(position_angle)
            ax.plot(x, y, 'ro', markersize=15, alpha=0.7)
        
        # 거시경제 지표 정보 표시 (하단에 더 깔끔하게 표시)
        if indicators:
            indicator_text = "현재 거시경제 지표:\n"
            for key, value in indicators.items():
                if key != '날짜':  # 날짜는 별도로 표시
                    indicator_text += f"{key}: {value:.2f}%\n"
            if '날짜' in indicators:
                indicator_text += f"날짜: {indicators['날짜']}"
                
            # 배경 색이 있는 텍스트 박스로 표시
            ax.text(0, -8, indicator_text, fontsize=12, ha='center', 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'))
        
        # 축 설정
        ax.set_xlim(-6, 6)
        ax.set_ylim(-8, 8)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 기본 제목 설정 (main.py에서 필요시 변경)
        plt.title('앙드레 코스탈라니의 달걀모형과 경제 사이클', fontsize=16, pad=20)
        
        return fig, ax
    
    def determine_position(self, indicators):
        """
        거시경제 지표를 바탕으로 현재 경제 사이클 위치 추정
        indicators: 딕셔너리 형태의 거시경제 지표 데이터
        return: 0~1 사이의 값으로 현재 위치 (0: A 위치, 0.167: B 위치, ... 0.833: F 위치)
        """
        # 각 단계별 점수 계산
        stage_scores = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0}
        
        # GDP 성장률
        if 'GDP 성장률' in indicators:
            gdp = indicators['GDP 성장률']
            if gdp >= 3.0:
                stage_scores['F'] += 2
                stage_scores['E'] += 1
            elif gdp >= 1.0:
                stage_scores['E'] += 2
                stage_scores['F'] += 1
            elif gdp > 0:
                stage_scores['D'] += 2
                stage_scores['E'] += 1
            elif gdp > -1.0:
                stage_scores['C'] += 1
                stage_scores['D'] += 2
            else:
                stage_scores['B'] += 1
                stage_scores['C'] += 2
        
        # 인플레이션
        if '인플레이션' in indicators:
            inf = indicators['인플레이션']
            if inf >= 5.0:
                stage_scores['B'] += 2
                stage_scores['A'] += 1
            elif inf >= 3.0:
                stage_scores['A'] += 2
                stage_scores['F'] += 1
            elif inf >= 2.0:
                stage_scores['F'] += 1
                stage_scores['E'] += 1
            elif inf >= 1.0:
                stage_scores['E'] += 1
                stage_scores['D'] += 1
            else:
                stage_scores['D'] += 2
                stage_scores['C'] += 1
        
        # 금리 - README 기준 적용
        if '기준금리' in indicators:
            rate = indicators['기준금리']
            if rate >= 10.0:  # 10% 이상 - F-A, A-B 구간
                stage_scores['A'] += 3
                stage_scores['B'] += 2
                stage_scores['F'] += 1
            elif rate >= 3.0:  # 3~10% - B-C, E-F 구간
                if '인플레이션' in indicators and indicators['인플레이션'] > 4.0:
                    # 높은 인플레이션(4% 초과)이면 B-C 쪽으로
                    stage_scores['B'] += 2
                    stage_scores['C'] += 2
                    stage_scores['F'] += 1
                elif '인플레이션' in indicators and indicators['인플레이션'] > 3.0 and rate >= 6.0:
                    # 인플레이션 3% 초과 & 금리 6% 이상이면 버블 초기(A) 지역
                    stage_scores['A'] += 2
                    stage_scores['F'] += 2
                    stage_scores['B'] += 1
                else:
                    # 인플레이션이 적당하면 E-F 쪽으로 (호황기)
                    stage_scores['F'] += 3
                    stage_scores['E'] += 2
                    stage_scores['A'] += 1
            else:  # 3% 미만 - C-D, D-E 구간
                if 'GDP 성장률' in indicators and indicators['GDP 성장률'] < 0:
                    # GDP 성장률이 마이너스면 C-D 쪽으로
                    stage_scores['C'] += 3
                    stage_scores['D'] += 2
                else:
                    # GDP 성장률이 플러스면 D-E 쪽으로
                    stage_scores['D'] += 2
                    stage_scores['E'] += 3
        
        # 실업률
        if '실업률' in indicators:
            unemp = indicators['실업률']
            if unemp >= 8.0:
                stage_scores['D'] += 2
                stage_scores['C'] += 1
            elif unemp >= 6.0:
                stage_scores['C'] += 2
                stage_scores['B'] += 1
            elif unemp >= 4.0:
                stage_scores['B'] += 1
                stage_scores['A'] += 1
            else:
                stage_scores['F'] += 2
                stage_scores['E'] += 1
        
        # 가장 높은 점수를 가진 단계 결정
        max_stage = max(stage_scores, key=stage_scores.get)
        stage_order = ['A', 'B', 'C', 'D', 'E', 'F']
        position = stage_order.index(max_stage) / len(stage_order)
        
        return position, max_stage
        
    def get_evaluation_details(self, indicators):
        """
        거시경제 지표에 따른 단계별 평가 근거를 상세히 설명
        """
        details = []
        stage_scores = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0}
        
        # GDP 성장률 평가
        if 'GDP 성장률' in indicators:
            gdp = indicators['GDP 성장률']
            gdp_eval = f"GDP 성장률 {gdp}%: "
            
            if gdp >= 3.0:
                gdp_eval += f"경제 활성화 단계로 F단계(+2점), E단계(+1점)에 가깝습니다. (기준: {self.indicator_criteria['GDP 성장률']['F']})"
                stage_scores['F'] += 2
                stage_scores['E'] += 1
            elif gdp >= 1.0:
                gdp_eval += f"경기 회복 본격화 단계로 E단계(+2점), F단계(+1점)에 가깝습니다. (기준: {self.indicator_criteria['GDP 성장률']['E']})"
                stage_scores['E'] += 2
                stage_scores['F'] += 1
            elif gdp > 0:
                gdp_eval += f"경기 회복 시작 단계로 D단계(+2점), E단계(+1점)에 가깝습니다. (기준: {self.indicator_criteria['GDP 성장률']['D']})"
                stage_scores['D'] += 2
                stage_scores['E'] += 1
            elif gdp > -1.0:
                gdp_eval += f"경기 침체 완화 단계로 C단계(+1점), D단계(+2점)에 가깝습니다. (기준: {self.indicator_criteria['GDP 성장률']['C']})"
                stage_scores['C'] += 1
                stage_scores['D'] += 2
            else:
                gdp_eval += f"심각한 경기 침체 단계로 B단계(+1점), C단계(+2점)에 가깝습니다. (기준: {self.indicator_criteria['GDP 성장률']['B']})"
                stage_scores['B'] += 1
                stage_scores['C'] += 2
                
            details.append(gdp_eval)
        
        # 인플레이션 평가
        if '인플레이션' in indicators:
            inf = indicators['인플레이션']
            inf_eval = f"인플레이션 {inf}%: "
            
            if inf >= 5.0:
                inf_eval += f"높은 인플레이션으로 B단계(+2점), A단계(+1점)에 가깝습니다. (기준: {self.indicator_criteria['인플레이션']['B']})"
                stage_scores['B'] += 2
                stage_scores['A'] += 1
            elif inf >= 3.0:
                inf_eval += f"경기 과열 징후로 A단계(+2점), F단계(+1점)에 가깝습니다. (기준: {self.indicator_criteria['인플레이션']['A']})"
                stage_scores['A'] += 2
                stage_scores['F'] += 1
            elif inf >= 2.0:
                inf_eval += f"경기 활성화에 따른 물가 상승으로 F단계(+1점), E단계(+1점)에 가깝습니다. (기준: {self.indicator_criteria['인플레이션']['F']})"
                stage_scores['F'] += 1
                stage_scores['E'] += 1
            elif inf >= 1.0:
                inf_eval += f"적정 인플레이션으로 E단계(+1점), D단계(+1점)에 가깝습니다. (기준: {self.indicator_criteria['인플레이션']['E']})"
                stage_scores['E'] += 1
                stage_scores['D'] += 1
            else:
                inf_eval += f"낮은 인플레이션으로 D단계(+2점), C단계(+1점)에 가깝습니다. (기준: {self.indicator_criteria['인플레이션']['D']})"
                stage_scores['D'] += 2
                stage_scores['C'] += 1
                
            details.append(inf_eval)
        
        # 금리 평가
        if '기준금리' in indicators:
            rate = indicators['기준금리']
            rate_eval = f"기준금리 {rate}%: "
            
            if rate >= 10.0:
                rate_eval += f"매우 높은 금리로 A-B 구간(버블기)에 해당합니다. A단계(+3점), B단계(+2점), F단계(+1점)에 가깝습니다. (기준: {self.interest_rate_ranges['F-A']}, {self.interest_rate_ranges['A-B']})"
                stage_scores['A'] += 3
                stage_scores['B'] += 2
                stage_scores['F'] += 1
            elif rate >= 3.0:
                if '인플레이션' in indicators and indicators['인플레이션'] > 4.0:
                    rate_eval += f"높은 금리와 매우 높은 인플레이션(4% 초과)으로 B-C 구간(침체기)에 해당합니다. B단계(+2점), C단계(+2점), F단계(+1점)에 가깝습니다. (기준: {self.interest_rate_ranges['B-C']})"
                    stage_scores['B'] += 2
                    stage_scores['C'] += 2
                    stage_scores['F'] += 1
                elif '인플레이션' in indicators and indicators['인플레이션'] > 3.0 and rate >= 6.0:
                    rate_eval += f"높은 금리(6% 이상)와 높은 인플레이션(3% 초과)으로 버블 초기 A단계(+2점), F단계(+2점), B단계(+1점)에 가깝습니다."
                    stage_scores['A'] += 2
                    stage_scores['F'] += 2
                    stage_scores['B'] += 1
                else:
                    rate_eval += f"중간~높은 금리이지만 적절한 인플레이션으로 E-F 구간(호황기)에 해당합니다. F단계(+3점), E단계(+2점), A단계(+1점)에 가깝습니다. (기준: {self.interest_rate_ranges['E-F']})"
                    stage_scores['F'] += 3
                    stage_scores['E'] += 2
                    stage_scores['A'] += 1
            else:
                if 'GDP 성장률' in indicators and indicators['GDP 성장률'] < 0:
                    rate_eval += f"낮은 금리와 마이너스 성장률로 C-D 구간(위기기)에 해당합니다. C단계(+3점), D단계(+2점)에 가깝습니다. (기준: {self.interest_rate_ranges['C-D']})"
                    stage_scores['C'] += 3
                    stage_scores['D'] += 2
                else:
                    rate_eval += f"낮은 금리와 플러스 성장률로 D-E 구간(회복기)에 해당합니다. D단계(+2점), E단계(+3점)에 가깝습니다. (기준: {self.interest_rate_ranges['D-E']})"
                    stage_scores['D'] += 2
                    stage_scores['E'] += 3
                    
            details.append(rate_eval)
        
        # 실업률 평가
        if '실업률' in indicators:
            unemp = indicators['실업률']
            unemp_eval = f"실업률 {unemp}%: "
            
            if unemp >= 8.0:
                unemp_eval += f"매우 높은 실업률로 D단계(+2점), C단계(+1점)에 가깝습니다. (기준: {self.indicator_criteria['실업률']['D']})"
                stage_scores['D'] += 2
                stage_scores['C'] += 1
            elif unemp >= 6.0:
                unemp_eval += f"높은 실업률로 C단계(+2점), B단계(+1점)에 가깝습니다. (기준: {self.indicator_criteria['실업률']['C']})"
                stage_scores['C'] += 2
                stage_scores['B'] += 1
            elif unemp >= 4.0:
                unemp_eval += f"적정 수준의 실업률로 B단계(+1점), A단계(+1점)에 가깝습니다. (기준: {self.indicator_criteria['실업률']['B']})"
                stage_scores['B'] += 1
                stage_scores['A'] += 1
            else:
                unemp_eval += f"낮은 실업률로 F단계(+2점), E단계(+1점)에 가깝습니다. (기준: {self.indicator_criteria['실업률']['F']})"
                stage_scores['F'] += 2
                stage_scores['E'] += 1
                
            details.append(unemp_eval)
        
        # 최종 평가 결과
        max_stage = max(stage_scores, key=stage_scores.get)
        max_score = stage_scores[max_stage]
        
        final_eval = f"종합 평가: {max_stage} 단계 ({self.stages[max_stage]}, 점수: {max_score}점)\n"
        final_eval += "각 단계별 점수: "
        for stage, score in stage_scores.items():
            final_eval += f"{stage}({score}점) "
        
        details.append(final_eval)
        
        return details
    
    def get_investment_strategy(self, position_stage):
        """경제 사이클 위치에 따른 투자 전략 제안"""
        strategies = {
            'A': '금리 상승기에 접어들었습니다. 주식 시장은 어려워질 수 있으며, 안전자산 비중을 높이는 것이 좋습니다.',
            'B': '인플레이션이 높고 긴축 정책이 시행되는 시기입니다. 주식은 어려울 수 있으나 소비재와 필수품 관련 주식 중심으로 투자하세요.',
            'C': '경기 침체기입니다. 채권과 배당주에 비중을 두고, 경기 회복을 대비한 저평가 주식을 서서히 매수할 때입니다.',
            'D': '경기 회복을 위한 정책이 시행되는 시기입니다. 경기 민감주와 성장주에 관심을 가질 때입니다.',
            'E': '경기가 회복되고 있습니다. 경기 민감주와 성장주에 투자하는 것이 유리합니다.',
            'F': '호황기로, 기업 실적이 좋고 시장이 낙관적입니다. 그러나 고점에 가까워지고 있으므로 점진적으로 위험자산 비중을 줄이는 것을 고려하세요.'
        }
        return strategies.get(position_stage, '현재 경제 상황을 더 자세히 분석해야 합니다.') 

def plot_kostalany_egg_advanced(
    gdp, inflation, interest, unemployment, vix, dxy, today=None, position_label="F~E", current_xy=(-0.7, 0.3), invest_strategy="경기 민감주와 성장주 투자 유리"
):
    if today is None:
        today = datetime.now().strftime('%Y-%m-%d')

    # 한글 폰트 설정
    if platform.system() == 'Darwin':
        plt.rc('font', family='AppleGothic')
    elif platform.system() == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    else:
        plt.rc('font', family='NanumGothic')
    plt.rc('axes', unicode_minus=False)

    fig, ax = plt.subplots(figsize=(12, 13))
    # 타원 좌표
    theta = np.linspace(0, 2*np.pi, 400)
    x_egg = np.cos(theta)
    y_egg = np.sin(theta) * 0.8

    # 호황/불황 음영
    ax.fill_between(x_egg, y_egg, where=(x_egg<0), color='#ffe6e6', alpha=0.25, label='호황기')
    ax.fill_between(x_egg, y_egg, where=(x_egg>0), color='#e6e6ff', alpha=0.25, label='불황기')
    ax.plot(x_egg, y_egg, 'k-', lw=2, alpha=0.7)

    # 단계별 위치/라벨
    stages = {
        'A': (0, 1),
        'B': (1, 0.5),
        'C': (1, -0.5),
        'D': (0, -1),
        'E': (-1, -0.5),
        'F': (-1, 0.5)
    }
    stage_labels = {
        'A': '호황기 최대 경기 상승\n(예금금리/주식매도)',
        'B': '인플레이션 억제 긴축정책\n(예금금리/채권투자)',
        'C': '경기 침체 소비 감소\n(채권/부동산투자)',
        'D': '경기 최저 완화정책',
        'E': '경기 확장 소비 증가\n(주식투자/부동산매도)',
        'F': '호황기 최대 경기 회복\n(예금금리/주식매도)'
    }
    for k, (x, y) in stages.items():
        ax.plot(x, y, 'ko', ms=12, zorder=5)
        ax.text(x, y+0.09, k, fontsize=18, ha='center', va='bottom', weight='bold')
        ax.text(x, y-0.13, stage_labels[k], fontsize=11, ha='center', va='top', color='dimgray')

    # 중앙 버티컬 라인과 금리 정보
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    interest_info = [
        ("금리 10%", 0.5, "#FFB347"),
        ("금리 5%", 0.0, "#FFE66D"),
        ("금리 3% 미만", -0.5, "#4ECDC4")
    ]
    
    for text, y_pos, color in interest_info:
        ax.text(0, y_pos, text, fontsize=11, ha='center', va='center', 
                color='black', bbox=dict(facecolor=color, alpha=0.3, boxstyle='round,pad=0.3'))

    # 현재 위치 마킹 (반투명)
    ax.plot(*current_xy, 'ro', ms=18, zorder=10, alpha=0.4)
    ax.text(current_xy[0], current_xy[1]+0.13, f'현재 위치 ({today})\n{position_label}', color='red', fontsize=13, ha='center', va='bottom', weight='bold', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.4'), alpha=0.7)

    # 상단 타이틀/날짜
    ax.text(0, 1.32, "앙드레 코스탈라니의 달걀모형 경제 사이클 분석", fontsize=18, ha='center', va='bottom', weight='bold')
    ax.text(0, 1.26, f"분석일: {today}", fontsize=12, ha='center', va='bottom', color='gray')

    # 주요 경제지표 박스 (GDP 포함, 우측 상단, 전체 영역만 반투명 컬러, 개별 항목은 컬러 없음)
    indicator_values = [
        ("GDP", gdp),
        ("Inflation", inflation),
        ("Interest", interest),
        ("Unemployment", unemployment),
        ("VIX", vix),
        ("DXY", dxy)
    ]
    # 박스 위치 및 크기 설정 (분석일 Y 좌표에 맞춤)
    line_gap = 0.06 * 0.85
    box_width = 0.38 * 0.85
    box_x = 0.65 - box_width/2
    box_y = 1.26  # 분석일 텍스트와 동일한 Y 좌표
    box_height = (line_gap * (len(indicator_values)+1) + 0.04) * 0.9 * 0.85
    # 전체 반투명 컬러 영역
    ax.add_patch(
        patches.FancyBboxPatch((box_x, box_y - box_height), box_width, box_height, 
            boxstyle="round,pad=0.10", linewidth=0, edgecolor=None, facecolor="#B5B2C2", alpha=0.25, transform=ax.transData, zorder=9)
    )
    # 타이틀
    ax.text(box_x + box_width/2, box_y - 0.01, "경제지표", fontsize=12, ha='center', va='top', weight='bold', color='#333', zorder=20)
    # 각 지표별 텍스트 (컬러 박스 없음)
    for i, (label, value) in enumerate(indicator_values):
        ax.text(box_x + 0.03, box_y - (i+1)*line_gap - 0.01, f"{label}: {value:.2f}%" if label!="VIX" and label!="DXY" else f"{label}: {value:.2f}",
                fontsize=10, ha='left', va='top', color='black', zorder=20)

    # 평가 근거와 투자 전략 설명 (차트 하단)
    if hasattr(plot_kostalany_egg_advanced, 'evaluation_details') and hasattr(plot_kostalany_egg_advanced, 'invest_strategy'):
        eval_text = '\n'.join(plot_kostalany_egg_advanced.evaluation_details)
        strategy_text = plot_kostalany_egg_advanced.invest_strategy
        ax.text(0, -1.22, f"[평가 근거]\n{eval_text}\n\n[추천 투자 전략]\n{strategy_text}",
                fontsize=10, ha='center', va='top', color='black',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'), zorder=30)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.25, 1.38)
    ax.axis('off')
    plt.tight_layout()

    # 결과 저장
    output_dir = "output/kostalany"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"kostalany_model_advanced.png")
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n분석 결과 이미지가 저장되었습니다: {filename}")

    plt.show() 

if __name__ == "__main__":
    # 샘플 데이터로 차트 생성
    plot_kostalany_egg_advanced(
        gdp=2.5,              # GDP 성장률
        inflation=3.2,        # 인플레이션
        interest=5.5,         # 기준금리
        unemployment=4.1,     # 실업률
        vix=15.5,            # VIX 지수
        dxy=102.5,           # 달러 인덱스
        today="2024-03-19",  # 분석일
        position_label="F~E", # 현재 위치
        current_xy=(-0.7, 0.3), # 현재 위치 좌표
        invest_strategy="경기 민감주와 성장주 투자 유리" # 투자 전략
    ) 
