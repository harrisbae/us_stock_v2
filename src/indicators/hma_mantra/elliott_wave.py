"""
엘리엇 파동 분석 모듈 (Elliott Wave Theory Analysis)

이 모듈은 주식 차트에서 엘리엇 파동 패턴을 자동으로 감지하고 분석합니다.

주요 기능:
1. ZigZag 알고리즘을 사용한 주요 전환점(Pivot) 감지
2. 5파 충격파(Impulse Wave) 감지 (1-2-3-4-5)
3. 3파 조정파(Corrective Wave) 감지 (A-B-C)
4. 피보나치 비율 검증
5. 엘리엇 파동 규칙 검증
6. 신뢰도 점수 계산
7. 다음 목표가 예측

Phase 1: 기본 기능 구현
- ZigZag 패턴 생성
- 파동 감지 및 레이블링
- 기본 규칙 검증
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')


class ElliottWaveAnalyzer:
    """엘리엇 파동 분석 클래스"""
    
    # 피보나치 비율 상수
    FIBONACCI_RATIOS = {
        'retracement': [0.236, 0.382, 0.5, 0.618, 0.786],
        'extension': [1.0, 1.272, 1.618, 2.0, 2.618]
    }
    
    def __init__(self, data: pd.DataFrame, min_wave_size: float = 3.0, 
                 zigzag_threshold: float = 5.0):
        """
        엘리엇 파동 분석기 초기화
        
        Args:
            data: OHLCV 데이터프레임
            min_wave_size: 최소 파동 크기 (%) - 작은 노이즈 필터링
            zigzag_threshold: ZigZag 감지 임계값 (%) - 전환점 감지 민감도
        """
        self.data = data.copy()
        self.min_wave_size = min_wave_size
        self.zigzag_threshold = zigzag_threshold
        self.pivots = []
        self.waves = []
        self.current_wave_position = None
        self.confidence_score = 0.0
        
    def analyze(self) -> Dict:
        """
        전체 엘리엇 파동 분석 실행
        
        Returns:
            분석 결과 딕셔너리
        """
        # Step 1: ZigZag 패턴 생성 (주요 전환점 찾기)
        self.pivots = self._find_zigzag_pivots()
        
        if len(self.pivots) < 5:
            return {
                'success': False,
                'message': '충분한 피봇 포인트가 없습니다 (최소 5개 필요)',
                'pivots': self.pivots,
                'waves': [],
                'confidence': 0.0
            }
        
        # Step 2: 파동 패턴 매칭 (5-3 파동 찾기)
        self.waves = self._detect_wave_patterns()
        
        if not self.waves:
            return {
                'success': False,
                'message': '엘리엇 파동 패턴을 찾을 수 없습니다',
                'pivots': self.pivots,
                'waves': [],
                'confidence': 0.0
            }
        
        # Step 3: 신뢰도 점수 계산
        self.confidence_score = self._calculate_confidence()
        
        # Step 4: 현재 파동 위치 파악
        self.current_wave_position = self._identify_current_position()
        
        # Step 5: 다음 목표가 계산
        targets = self._calculate_targets()
        
        return {
            'success': True,
            'message': '엘리엇 파동 분석 완료',
            'pivots': self.pivots,
            'waves': self.waves,
            'current_position': self.current_wave_position,
            'confidence': self.confidence_score,
            'targets': targets,
            'fibonacci_levels': self._calculate_fibonacci_levels()
        }
    
    def _find_zigzag_pivots(self) -> List[Dict]:
        """
        ZigZag 알고리즘으로 주요 전환점(Pivot) 찾기
        
        Returns:
            피봇 포인트 리스트 [{date, price, type, index}, ...]
        """
        high = self.data['High'].values
        low = self.data['Low'].values
        close = self.data['Close'].values
        
        pivots = []
        
        # 지역 최대값과 최소값 찾기
        order = max(5, int(len(self.data) * 0.02))  # 동적 윈도우 크기
        
        # 고점 찾기
        high_peaks = argrelextrema(high, np.greater, order=order)[0]
        
        # 저점 찾기
        low_peaks = argrelextrema(low, np.less, order=order)[0]
        
        # 피봇 포인트 병합 및 정렬
        for idx in high_peaks:
            if idx >= len(self.data):
                continue
            pivots.append({
                'date': self.data.index[idx],
                'price': high[idx],
                'type': 'high',
                'index': idx
            })
        
        for idx in low_peaks:
            if idx >= len(self.data):
                continue
            pivots.append({
                'date': self.data.index[idx],
                'price': low[idx],
                'type': 'low',
                'index': idx
            })
        
        # 날짜 순으로 정렬
        pivots.sort(key=lambda x: x['index'])
        
        # 노이즈 제거: 최소 변동폭 이하의 피봇 필터링
        filtered_pivots = self._filter_small_pivots(pivots)
        
        return filtered_pivots
    
    def _filter_small_pivots(self, pivots: List[Dict]) -> List[Dict]:
        """
        작은 변동폭의 피봇 포인트 제거
        
        Args:
            pivots: 피봇 포인트 리스트
            
        Returns:
            필터링된 피봇 포인트 리스트
        """
        if len(pivots) < 2:
            return pivots
        
        filtered = [pivots[0]]
        
        for i in range(1, len(pivots)):
            prev_pivot = filtered[-1]
            curr_pivot = pivots[i]
            
            # 가격 변동률 계산
            price_change = abs(curr_pivot['price'] - prev_pivot['price'])
            price_change_pct = (price_change / prev_pivot['price']) * 100
            
            # 최소 변동폭 이상인 경우에만 추가
            if price_change_pct >= self.min_wave_size:
                # 타입이 교차하는지 확인 (high-low-high 또는 low-high-low)
                if curr_pivot['type'] != prev_pivot['type']:
                    filtered.append(curr_pivot)
                else:
                    # 같은 타입이면 더 극단적인 값으로 대체
                    if curr_pivot['type'] == 'high':
                        if curr_pivot['price'] > prev_pivot['price']:
                            filtered[-1] = curr_pivot
                    else:  # low
                        if curr_pivot['price'] < prev_pivot['price']:
                            filtered[-1] = curr_pivot
        
        return filtered
    
    def _detect_wave_patterns(self) -> List[Dict]:
        """
        5-3 파동 패턴 감지
        
        Returns:
            감지된 파동 리스트
        """
        waves = []
        
        # 충격파 (Impulse Wave) 감지: 5개의 연속된 피봇
        impulse_waves = self._find_impulse_waves()
        if impulse_waves:
            waves.extend(impulse_waves)
        
        # 조정파 (Corrective Wave) 감지: 3개의 연속된 피봇
        corrective_waves = self._find_corrective_waves()
        if corrective_waves:
            waves.extend(corrective_waves)
        
        return waves
    
    def _find_impulse_waves(self) -> List[Dict]:
        """
        5파 충격파 감지
        
        Returns:
            충격파 리스트
        """
        impulse_waves = []
        
        # 최소 6개의 피봇 필요 (5파 충격파)
        if len(self.pivots) < 6:
            return impulse_waves
        
        # 슬라이딩 윈도우로 5파 패턴 찾기
        for i in range(len(self.pivots) - 5):
            # 5파 충격파: 6개의 피봇 필요 (시작점 + 5개 파동 끝점)
            wave_candidates = self.pivots[i:i+6]
            
            # 파동 1-2-3-4-5 추출 (올바른 매핑)
            waves_data = {
                'wave_0': wave_candidates[0],  # 시작점
                'wave_1': wave_candidates[1],  # 파동 1 끝
                'wave_2': wave_candidates[2],  # 파동 2 끝
                'wave_3': wave_candidates[3],  # 파동 3 끝
                'wave_4': wave_candidates[4],  # 파동 4 끝
                'wave_5': wave_candidates[5],  # 파동 5 끝
            }
            
            # 엘리엇 파동 규칙 검증
            if self._validate_impulse_wave(waves_data):
                impulse_wave = {
                    'type': 'impulse',
                    'direction': 'up' if waves_data['wave_5']['price'] > waves_data['wave_0']['price'] else 'down',
                    'waves': waves_data,
                    'start_date': waves_data['wave_0']['date'],
                    'end_date': waves_data['wave_5']['date'],
                    'start_price': waves_data['wave_0']['price'],
                    'end_price': waves_data['wave_5']['price'],
                    'gain_pct': ((waves_data['wave_5']['price'] - waves_data['wave_0']['price']) / 
                                waves_data['wave_0']['price'] * 100),
                    'validation': self._get_wave_validation_details(waves_data)
                }
                impulse_waves.append(impulse_wave)
        
        return impulse_waves
    
    def _find_corrective_waves(self) -> List[Dict]:
        """
        3파 조정파 감지 (A-B-C)
        
        Returns:
            조정파 리스트
        """
        corrective_waves = []
        
        # 최소 5개의 피봇 필요 (3파 조정파)
        if len(self.pivots) < 5:
            return corrective_waves
        
        # 슬라이딩 윈도우로 3파 패턴 찾기
        for i in range(len(self.pivots) - 4):
            wave_candidates = self.pivots[i:i+5]
            
            waves_data = {
                'wave_0': wave_candidates[0],  # 시작점
                'wave_A': wave_candidates[1],  # 파동 A 끝
                'wave_B': wave_candidates[2],  # 파동 B 끝
                'wave_C': wave_candidates[3],  # 파동 C 끝
            }
            
            # 조정파 패턴 확인
            if self._validate_corrective_wave(waves_data):
                corrective_wave = {
                    'type': 'corrective',
                    'pattern': self._identify_corrective_pattern(waves_data),
                    'waves': waves_data,
                    'start_date': waves_data['wave_0']['date'],
                    'end_date': waves_data['wave_C']['date'],
                    'start_price': waves_data['wave_0']['price'],
                    'end_price': waves_data['wave_C']['price'],
                    'retracement_pct': ((waves_data['wave_C']['price'] - waves_data['wave_0']['price']) / 
                                       waves_data['wave_0']['price'] * 100)
                }
                corrective_waves.append(corrective_wave)
        
        return corrective_waves
    
    def _validate_impulse_wave(self, waves: Dict) -> bool:
        """
        엘리엇 파동 충격파 규칙 검증
        
        규칙:
        1. 파동 2는 파동 1의 시작점 아래로 내려가지 않음
        2. 파동 3은 절대 가장 짧은 파동이 아님
        3. 파동 4는 파동 1의 고점 아래로 침범하지 않음
        4. 올바른 상승-하락-상승-하락-상승 패턴
        
        Args:
            waves: 파동 데이터
            
        Returns:
            검증 통과 여부
        """
        try:
            w0 = waves['wave_0']['price']
            w1 = waves['wave_1']['price']
            w2 = waves['wave_2']['price']
            w3 = waves['wave_3']['price']
            w4 = waves['wave_4']['price']
            w5 = waves['wave_5']['price']
            
            # 상승 충격파인 경우
            if w5 > w0:
                # 패턴 검증: 상승-하락-상승-하락-상승
                if not (w1 > w0 and w2 < w1 and w3 > w2 and w4 < w3 and w5 > w4):
                    return False
                
                # 규칙 1: 파동 2는 파동 1 시작점 아래로 가지 않음
                if w2 < w0:
                    return False
                
                # 규칙 2: 파동 3은 가장 짧은 파동이 아님
                wave1_len = abs(w1 - w0)
                wave3_len = abs(w3 - w2)
                wave5_len = abs(w5 - w4)
                
                if wave3_len <= wave1_len and wave3_len <= wave5_len:
                    return False
                
                # 규칙 3: 파동 4는 파동 1의 고점 아래로 침범하지 않음
                if w4 < w1:
                    return False
                
                return True
            
            # 하락 충격파인 경우
            else:
                # 패턴 검증: 하락-상승-하락-상승-하락
                if not (w1 < w0 and w2 > w1 and w3 < w2 and w4 > w3 and w5 < w4):
                    return False
                
                # 규칙 1: 파동 2는 파동 1 시작점 위로 가지 않음
                if w2 > w0:
                    return False
                
                # 규칙 2: 파동 3은 가장 짧은 파동이 아님
                wave1_len = abs(w1 - w0)
                wave3_len = abs(w3 - w2)
                wave5_len = abs(w5 - w4)
                
                if wave3_len <= wave1_len and wave3_len <= wave5_len:
                    return False
                
                # 규칙 3: 파동 4는 파동 1의 저점 위로 침범하지 않음
                if w4 > w1:
                    return False
                
                return True
                
        except Exception as e:
            return False
    
    def _validate_corrective_wave(self, waves: Dict) -> bool:
        """
        조정파 패턴 검증
        
        Args:
            waves: 파동 데이터
            
        Returns:
            검증 통과 여부
        """
        try:
            w0 = waves['wave_0']['price']
            wA = waves['wave_A']['price']
            wB = waves['wave_B']['price']
            wC = waves['wave_C']['price']
            
            # 기본 조정파 조건: A-B-C가 명확한 3파 구조
            # A파와 C파는 같은 방향, B파는 반대 방향
            
            # 하락 조정파 (w0 > wC)
            if w0 > wC:
                # A: 하락, B: 반등, C: 하락
                if wA < w0 and wB > wA and wC < wB:
                    return True
            
            # 상승 조정파 (w0 < wC)
            else:
                # A: 상승, B: 조정, C: 상승
                if wA > w0 and wB < wA and wC > wB:
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _identify_corrective_pattern(self, waves: Dict) -> str:
        """
        조정파 패턴 식별 (Zigzag, Flat, Triangle 등)
        
        Args:
            waves: 파동 데이터
            
        Returns:
            패턴 이름
        """
        try:
            w0 = waves['wave_0']['price']
            wA = waves['wave_A']['price']
            wB = waves['wave_B']['price']
            wC = waves['wave_C']['price']
            
            # A파 크기
            a_move = abs(wA - w0)
            # B파 되돌림 비율
            b_retrace = abs(wB - wA) / a_move if a_move > 0 else 0
            # C파 크기 대비 A파
            c_move = abs(wC - wB)
            c_to_a_ratio = c_move / a_move if a_move > 0 else 0
            
            # Zigzag: 5-3-5 구조, C파가 A파와 비슷하거나 더 큼
            if b_retrace < 0.62 and c_to_a_ratio > 0.8:
                return 'Zigzag'
            
            # Flat: 3-3-5 구조, B파가 A파 시작점 근처까지 되돌림
            elif b_retrace > 0.8:
                return 'Flat'
            
            # Triangle: 수렴 패턴
            elif 0.5 < b_retrace < 0.8 and 0.5 < c_to_a_ratio < 0.9:
                return 'Triangle'
            
            else:
                return 'Complex'
                
        except Exception:
            return 'Unknown'
    
    def _get_wave_validation_details(self, waves: Dict) -> Dict:
        """
        파동 검증 상세 정보 반환
        
        Args:
            waves: 파동 데이터
            
        Returns:
            검증 상세 정보
        """
        w0 = waves['wave_0']['price']
        w1 = waves['wave_1']['price']
        w2 = waves['wave_2']['price']
        w3 = waves['wave_3']['price']
        w4 = waves['wave_4']['price']
        w5 = waves['wave_5']['price']
        
        # 파동 길이 계산
        wave1_len = abs(w1 - w0)
        wave2_len = abs(w2 - w1)
        wave3_len = abs(w3 - w2)
        wave4_len = abs(w4 - w3)
        wave5_len = abs(w5 - w4)
        
        # 피보나치 비율 계산
        wave2_retrace = wave2_len / wave1_len if wave1_len > 0 else 0
        wave3_extension = wave3_len / wave1_len if wave1_len > 0 else 0
        wave4_retrace = wave4_len / wave3_len if wave3_len > 0 else 0
        wave5_extension = wave5_len / wave1_len if wave1_len > 0 else 0
        
        return {
            'wave_lengths': {
                'wave_1': wave1_len,
                'wave_2': wave2_len,
                'wave_3': wave3_len,
                'wave_4': wave4_len,
                'wave_5': wave5_len
            },
            'fibonacci_ratios': {
                'wave_2_retracement': wave2_retrace,
                'wave_3_extension': wave3_extension,
                'wave_4_retracement': wave4_retrace,
                'wave_5_vs_wave_1': wave5_extension
            }
        }
    
    def _calculate_confidence(self) -> float:
        """
        엘리엇 파동 분석 신뢰도 계산
        
        Returns:
            신뢰도 점수 (0-100)
        """
        if not self.waves:
            return 0.0
        
        total_score = 0.0
        max_score = 0.0
        
        for wave in self.waves:
            if wave['type'] == 'impulse':
                score = self._calculate_impulse_confidence(wave)
                total_score += score
                max_score += 100.0
            elif wave['type'] == 'corrective':
                score = self._calculate_corrective_confidence(wave)
                total_score += score
                max_score += 100.0
        
        confidence = (total_score / max_score * 100) if max_score > 0 else 0.0
        return min(100.0, confidence)
    
    def _calculate_impulse_confidence(self, wave: Dict) -> float:
        """
        충격파 신뢰도 계산
        
        Args:
            wave: 충격파 데이터
            
        Returns:
            신뢰도 점수 (0-100)
        """
        score = 0.0
        
        validation = wave.get('validation', {})
        fib_ratios = validation.get('fibonacci_ratios', {})
        
        # 파동 2 되돌림 비율 (61.8% 근처면 높은 점수)
        wave2_ret = fib_ratios.get('wave_2_retracement', 0)
        if 0.5 <= wave2_ret <= 0.65:
            score += 25.0
        elif 0.4 <= wave2_ret <= 0.78:
            score += 15.0
        
        # 파동 3 확장 비율 (1.618배 근처면 높은 점수)
        wave3_ext = fib_ratios.get('wave_3_extension', 0)
        if 1.5 <= wave3_ext <= 2.0:
            score += 30.0
        elif 1.2 <= wave3_ext <= 2.5:
            score += 20.0
        
        # 파동 4 되돌림 비율 (38.2% 근처면 높은 점수)
        wave4_ret = fib_ratios.get('wave_4_retracement', 0)
        if 0.3 <= wave4_ret <= 0.5:
            score += 25.0
        elif 0.2 <= wave4_ret <= 0.6:
            score += 15.0
        
        # 파동 5와 파동 1 비율
        wave5_vs_1 = fib_ratios.get('wave_5_vs_wave_1', 0)
        if 0.8 <= wave5_vs_1 <= 1.2:
            score += 20.0
        elif 0.6 <= wave5_vs_1 <= 1.5:
            score += 10.0
        
        return score
    
    def _calculate_corrective_confidence(self, wave: Dict) -> float:
        """
        조정파 신뢰도 계산
        
        Args:
            wave: 조정파 데이터
            
        Returns:
            신뢰도 점수 (0-100)
        """
        score = 50.0  # 기본 점수
        
        pattern = wave.get('pattern', 'Unknown')
        
        # 알려진 패턴이면 높은 점수
        if pattern in ['Zigzag', 'Flat', 'Triangle']:
            score += 30.0
        
        # 되돌림 비율이 합리적이면 점수 추가
        retracement = abs(wave.get('retracement_pct', 0))
        if 30 <= retracement <= 70:
            score += 20.0
        
        return min(100.0, score)
    
    def _identify_current_position(self) -> Optional[Dict]:
        """
        현재 파동 위치 식별
        
        Returns:
            현재 위치 정보
        """
        if not self.waves:
            return None
        
        # 가장 최근의 파동 선택
        latest_wave = self.waves[-1]
        
        # 현재가
        current_price = self.data['Close'].iloc[-1]
        current_date = self.data.index[-1]
        
        if latest_wave['type'] == 'impulse':
            # 파동 5가 진행 중인지 확인
            w5_price = latest_wave['waves']['wave_5']['price']
            w5_date = latest_wave['waves']['wave_5']['date']
            
            if current_date > w5_date:
                position = 'after_wave_5'
                message = '파동 5 종료 후, 조정파 예상'
            else:
                position = 'wave_5'
                message = '파동 5 진행 중'
        else:
            position = 'corrective'
            message = f'{latest_wave["pattern"]} 조정파 진행 중'
        
        return {
            'position': position,
            'message': message,
            'latest_wave': latest_wave,
            'current_price': current_price,
            'current_date': current_date
        }
    
    def _calculate_targets(self) -> Dict:
        """
        다음 목표가 계산 (피보나치 확장/되돌림 기반)
        
        Returns:
            목표가 정보
        """
        if not self.waves or not self.current_wave_position:
            return {}
        
        latest_wave = self.waves[-1]
        targets = {}
        
        if latest_wave['type'] == 'impulse':
            # 충격파 후 조정 목표가
            w0 = latest_wave['waves']['wave_0']['price']
            w5 = latest_wave['waves']['wave_5']['price']
            
            impulse_range = w5 - w0
            
            # 피보나치 되돌림 레벨
            targets['retracement'] = {
                '23.6%': w5 - (impulse_range * 0.236),
                '38.2%': w5 - (impulse_range * 0.382),
                '50.0%': w5 - (impulse_range * 0.5),
                '61.8%': w5 - (impulse_range * 0.618),
                '78.6%': w5 - (impulse_range * 0.786)
            }
            
        elif latest_wave['type'] == 'corrective':
            # 조정파 후 다음 충격파 목표가
            wC = latest_wave['waves']['wave_C']['price']
            w0 = latest_wave['waves']['wave_0']['price']
            
            corrective_range = abs(wC - w0)
            
            # 피보나치 확장 레벨
            if wC < w0:  # 하락 조정 후 상승 예상
                targets['extension'] = {
                    '100%': wC + corrective_range,
                    '127.2%': wC + (corrective_range * 1.272),
                    '161.8%': wC + (corrective_range * 1.618),
                    '200%': wC + (corrective_range * 2.0)
                }
            else:  # 상승 조정 후 하락 예상
                targets['extension'] = {
                    '100%': wC - corrective_range,
                    '127.2%': wC - (corrective_range * 1.272),
                    '161.8%': wC - (corrective_range * 1.618),
                    '200%': wC - (corrective_range * 2.0)
                }
        
        return targets
    
    def _calculate_fibonacci_levels(self) -> Dict:
        """
        피보나치 되돌림/확장 레벨 계산
        
        Returns:
            피보나치 레벨 딕셔너리
        """
        if not self.waves:
            return {}
        
        latest_wave = self.waves[-1]
        
        if latest_wave['type'] == 'impulse':
            start = latest_wave['start_price']
            end = latest_wave['end_price']
        else:
            start = latest_wave['waves']['wave_0']['price']
            end = latest_wave['waves']['wave_C']['price']
        
        range_val = end - start
        
        levels = {
            'retracement': {},
            'extension': {}
        }
        
        # 되돌림 레벨
        for ratio in self.FIBONACCI_RATIOS['retracement']:
            levels['retracement'][f'{ratio*100:.1f}%'] = end - (range_val * ratio)
        
        # 확장 레벨
        for ratio in self.FIBONACCI_RATIOS['extension']:
            levels['extension'][f'{ratio*100:.1f}%'] = start + (range_val * ratio)
        
        return levels


# 편의 함수
def analyze_elliott_wave(data: pd.DataFrame, min_wave_size: float = 3.0,
                         zigzag_threshold: float = 5.0) -> Dict:
    """
    엘리엇 파동 분석 실행 (편의 함수)
    
    Args:
        data: OHLCV 데이터프레임
        min_wave_size: 최소 파동 크기 (%)
        zigzag_threshold: ZigZag 감지 임계값 (%)
        
    Returns:
        분석 결과 딕셔너리
    """
    analyzer = ElliottWaveAnalyzer(data, min_wave_size, zigzag_threshold)
    return analyzer.analyze()

