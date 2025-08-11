#!/usr/bin/env python3
"""
시장 지표 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.indicators.hma_mantra.visualization.advanced import (
    get_market_data, get_naaim_data, get_real_naaim_data, 
    calculate_pcr, add_market_indicators
)
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

def test_market_indicators():
    """시장 지표 함수들을 테스트합니다."""
    
    print("=== 시장 지표 테스트 ===")
    
    # 테스트 기간 설정
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    print(f"테스트 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    
    # 1. VIX, TNX, DXY 데이터 테스트
    print("\n1. VIX, TNX, DXY 데이터 테스트")
    try:
        vix, tnx, dxy = get_market_data(start_date, end_date)
        if not vix.empty:
            vix_value = float(vix.iloc[-1])
            print(f"VIX: {vix_value:.2f}")
        else:
            print("VIX: 데이터 없음")
            
        if not tnx.empty:
            tnx_value = float(tnx.iloc[-1])
            print(f"TNX: {tnx_value:.2f}")
        else:
            print("TNX: 데이터 없음")
            
        if not dxy.empty:
            dxy_value = float(dxy.iloc[-1])
            print(f"DXY: {dxy_value:.2f}")
        else:
            print("DXY: 데이터 없음")
    except Exception as e:
        print(f"시장 데이터 테스트 오류: {e}")
    
    # 2. NAIIM 데이터 테스트
    print("\n2. NAIIM 데이터 테스트")
    try:
        naaim = get_naaim_data(start_date, end_date)
        if naaim is not None and not naaim.empty:
            naaim_value = float(naaim.iloc[-1])
            print(f"NAIIM: {naaim_value:.2f}")
        else:
            print("NAIIM: 데이터 없음")
    except Exception as e:
        print(f"NAIIM 테스트 오류: {e}")
    
    # 3. 실제 NAIIM 데이터 테스트
    print("\n3. 실제 NAIIM 데이터 테스트")
    try:
        real_naaim = get_real_naaim_data()
        if real_naaim is not None:
            print(f"실제 NAIIM: {real_naaim.iloc[-1]:.2f}")
        else:
            print("실제 NAIIM 데이터 없음")
    except Exception as e:
        print(f"실제 NAIIM 테스트 오류: {e}")
    
    # 4. PCR 계산 테스트
    print("\n4. PCR 계산 테스트")
    test_symbols = ['AAPL', 'SPY', 'QQQ']
    for symbol in test_symbols:
        try:
            pcr = calculate_pcr(symbol)
            print(f"{symbol} PCR: {pcr:.3f}" if pcr is not None else f"{symbol} PCR: 계산 불가")
        except Exception as e:
            print(f"{symbol} PCR 테스트 오류: {e}")
    
    # 5. 시장 지표 차트 테스트
    print("\n5. 시장 지표 차트 테스트")
    try:
        # 간단한 차트 생성
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 더미 데이터로 차트 생성
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        prices = [100 + i * 0.1 for i in range(len(dates))]
        
        ax.plot(dates, prices, 'b-', linewidth=2)
        ax.set_title('시장 지표 테스트 차트')
        ax.set_xlabel('날짜')
        ax.set_ylabel('가격')
        
        # 시장 지표들 추가
        current_vix = float(vix.iloc[-1]) if not vix.empty else 25.0
        current_naaim = float(naaim.iloc[-1]) if naaim is not None and not naaim.empty else 50.0
        test_pcr = 0.85
        
        add_market_indicators(ax, current_naaim, test_pcr, current_vix)
        
        # 차트 저장
        plt.tight_layout()
        plt.savefig('test_market_indicators.png', dpi=300, bbox_inches='tight')
        print("테스트 차트가 'test_market_indicators.png'로 저장되었습니다.")
        
        plt.show()
        
    except Exception as e:
        print(f"차트 테스트 오류: {e}")

if __name__ == "__main__":
    test_market_indicators()
