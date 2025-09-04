#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCR, VIX, NAIIM 통합 시장 심리 지표 차트 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def add_sentiment_zone_legend(ax):
    """시장 심리 구간별 설명 범례 추가 (종합점수 가이드 중심, 좌측 하단 배치)"""
    
    # 종합점수 해석 가이드 (좌측 하단, 구간별 색상 배경)
    
    # 강력 매수 구간 (0-30) - 진한 파란색 배경
    strong_buy_text = """🔵 0-30
강력 매수"""
    
    ax.text(0.02, 0.05, strong_buy_text, transform=ax.transAxes, fontsize=6.4,
            ha='left', va='bottom', bbox=dict(boxstyle="round,pad=0.5", facecolor='darkblue', 
                                          alpha=0.9, edgecolor='navy', linewidth=1),
            color='white')
    
    # 매수 관심 구간 (30-50) - 연한 파란색 배경
    buy_interest_text = """🔵 30-50
매수 관심"""
    
    ax.text(0.02, 0.15, buy_interest_text, transform=ax.transAxes, fontsize=6.4,
            ha='left', va='bottom', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', 
                                          alpha=0.9, edgecolor='blue', linewidth=1),
            color='black')
    
    # 중립 구간 (50) - 회색 배경
    neutral_text = """⚪ 50
중립"""
    
    ax.text(0.02, 0.25, neutral_text, transform=ax.transAxes, fontsize=6.4,
            ha='left', va='bottom', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', 
                                          alpha=0.9, edgecolor='gray', linewidth=1),
            color='black')
    
    # 매도 관심 구간 (50-70) - 연한 빨간색 배경
    sell_interest_text = """🔴 50-70
매도 관심"""
    
    ax.text(0.02, 0.35, sell_interest_text, transform=ax.transAxes, fontsize=6.4,
            ha='left', va='bottom', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral', 
                                          alpha=0.9, edgecolor='red', linewidth=1),
            color='black')
    
    # 강력 매도 구간 (70-100) - 진한 빨간색 배경
    strong_sell_text = """🔴 70-100
강력 매도"""
    
    ax.text(0.02, 0.45, strong_sell_text, transform=ax.transAxes, fontsize=6.4,
            ha='left', va='bottom', bbox=dict(boxstyle="round,pad=0.5", facecolor='darkred', 
                                          alpha=0.9, edgecolor='maroon', linewidth=1),
            color='white')

def test_integrated_sentiment_chart():
    """통합 시장 심리 지표 차트 테스트"""
    
    print("=== PCR, VIX, NAIIM 통합 시장 심리 지표 차트 테스트 ===")
    
    # 테스트 데이터 생성
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    # AAPL 데이터 다운로드
    try:
        print("AAPL 데이터 다운로드 중...")
        aapl = yf.download('AAPL', start=start_date, end=end_date, progress=False)
        print(f"AAPL 데이터 로드 완료: {len(aapl)}개")
    except Exception as e:
        print(f"AAPL 데이터 로드 실패: {e}")
        return
    
    # VIX 데이터 다운로드
    try:
        print("VIX 데이터 다운로드 중...")
        vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
        print(f"VIX 데이터 로드 완료: {len(vix)}개")
    except Exception as e:
        print(f"VIX 데이터 로드 실패: {e}")
        return
    
    # PCR 데이터 시뮬레이션 (VIX 기반)
    vix_normalized = (vix['Close'] - 10) / 40  # 10~50 → 0~1
    pcr = 0.5 + vix_normalized * 1.5  # 0.5~2.0 범위
    print(f"PCR 데이터 시뮬레이션 완료: {len(pcr)}개")
    
    # NAIIM 데이터 시뮬레이션 (VIX 기반)
    naaim = 100 - vix_normalized * 60  # 100~40 범위
    print(f"NAIIM 데이터 시뮬레이션 완료: {len(naaim)}개")
    
    # 데이터 정렬
    common_dates = aapl.index.intersection(vix.index).intersection(pcr.index).intersection(naaim.index)
    print(f"공통 데이터 기간: {len(common_dates)}개")
    
    if len(common_dates) == 0:
        print("공통 데이터가 없습니다.")
        return
    
    # 데이터 정규화
    pcr_normalized = np.clip(pcr.loc[common_dates], 0.5, 2.0)
    vix_normalized = np.clip(vix.loc[common_dates, 'Close'], 10, 50)
    naaim_normalized = np.clip(naaim.loc[common_dates], 0, 100)
    
    # 차트 생성
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 1, height_ratios=[2, 1], figure=fig, hspace=0.3)
    
    # 메인 차트 (AAPL)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_main.plot(common_dates, aapl.loc[common_dates, 'Close'], 'b-', linewidth=2, label='AAPL')
    ax_main.set_title('AAPL 주가 + 시장 심리 지표', fontsize=14, fontweight='bold')
    ax_main.set_ylabel('주가 ($)', color='blue')
    ax_main.tick_params(axis='y', labelcolor='blue')
    ax_main.legend(loc='upper left')
    ax_main.grid(True, alpha=0.3)
    
    # 시장 심리 통합 지표 차트
    ax_sentiment = fig.add_subplot(gs[1, 0])
    
    # 1차 Y축: PCR (왼쪽)
    line1 = ax_sentiment.plot(common_dates, pcr_normalized, 'b-', linewidth=2, label='PCR', alpha=0.8)
    ax_sentiment.set_ylabel('PCR (Put-Call Ratio)', color='blue', fontsize=10)
    ax_sentiment.tick_params(axis='y', labelcolor='blue')
    
    # 2차 Y축: VIX (오른쪽)
    ax2 = ax_sentiment.twinx()
    line2 = ax2.plot(common_dates, vix_normalized, 'r-', linewidth=2, label='VIX', alpha=0.8)
    ax2.set_ylabel('VIX (변동성 지수)', color='red', fontsize=10)
    ax2.tick_params(axis='y', labelcolor='red')
    
    # 3차 Y축: NAIIM (오른쪽, 별도 스케일)
    ax3 = ax_sentiment.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # 오른쪽에서 60px 떨어짐
    line3 = ax3.plot(common_dates, naaim_normalized, 'g-', linewidth=2, label='NAIIM', alpha=0.8)
    ax3.set_ylabel('NAIIM (기관투자자)', color='green', fontsize=10)
    ax3.tick_params(axis='y', labelcolor='green')
    
    # 기준선 추가
    ax_sentiment.axhline(y=1.0, color='blue', linestyle='--', alpha=0.5, label='PCR 중립')
    ax2.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='VIX 탐욕')
    ax2.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='VIX 중립')
    ax2.axhline(y=40, color='red', linestyle='--', alpha=0.5, label='VIX 공포')
    ax3.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='NAIIM 중립')
    
    # 범례 통합
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax_sentiment.legend(lines, labels, loc='upper left', fontsize=9)
    
    # 구간별 배경 하이라이트
    # PCR 극한 과매도 구간 (매수 신호)
    pcr_buy_zones = pcr_normalized > 1.5
    if np.any(pcr_buy_zones):
        buy_dates = common_dates[pcr_buy_zones]
        if len(buy_dates) > 0:
            ax_sentiment.axvspan(buy_dates.min(), buy_dates.max(), 
                               alpha=0.2, color='lightblue', label='PCR 매수 신호')
    
    # VIX 극한 공포 구간 (매수 신호)
    vix_buy_zones = vix_normalized > 40
    if np.any(vix_buy_zones):
        buy_dates = common_dates[vix_buy_zones]
        if len(buy_dates) > 0:
            ax_sentiment.axvspan(buy_dates.min(), buy_dates.max(), 
                               alpha=0.2, color='lightcoral', label='VIX 매수 신호')
    
    # NAIIM 극한 매도 구간 (매수 신호)
    naaim_buy_zones = naaim_normalized < 30
    if np.any(naaim_buy_zones):
        buy_dates = common_dates[naaim_buy_zones]
        if len(buy_dates) > 0:
            ax_sentiment.axvspan(buy_dates.min(), buy_dates.max(), 
                               alpha=0.2, color='lightgreen', label='NAIIM 매수 신호')
    
    ax_sentiment.set_title('시장 심리 통합 지표 (PCR + VIX + NAIIM)', fontsize=12, fontweight='bold')
    ax_sentiment.grid(True, alpha=0.3)
    
    # 구간별 설명 범례 추가 (좌측 하단)
    add_sentiment_zone_legend(ax_sentiment)
    
    # 시장 심리 분석
    print("\n=== 시장 심리 분석 결과 ===")
    
    # PCR 분석
    pcr_current = float(pcr_normalized.iloc[-1])
    if pcr_current > 1.5:
        pcr_signal = "PCR: 극한 과매도 (매수 신호)"
    elif pcr_current > 1.0:
        pcr_signal = "PCR: 과매도 구간"
    elif pcr_current < 0.5:
        pcr_signal = "PCR: 극한 과매수 (매도 신호)"
    elif pcr_current < 1.0:
        pcr_signal = "PCR: 과매수 구간"
    else:
        pcr_signal = "PCR: 중립 구간"
    
    # VIX 분석
    vix_current = float(vix_normalized.iloc[-1])
    if vix_current > 40:
        vix_signal = "VIX: 극한 공포 (매수 신호)"
    elif vix_current > 30:
        vix_signal = "VIX: 공포 구간"
    elif vix_current < 20:
        vix_signal = "VIX: 탐욕 구간"
    else:
        vix_signal = "VIX: 중립 구간"
    
    # NAIIM 분석
    naaim_current = float(naaim_normalized.iloc[-1])
    if naaim_current < 30:
        naaim_signal = "NAIIM: 기관 극한 매도 (매수 신호)"
    elif naaim_current < 50:
        naaim_signal = "NAIIM: 기관 매도 구간"
    elif naaim_current > 70:
        naaim_signal = "NAIIM: 기관 극한 매수 (매도 신호)"
    elif naaim_current > 50:
        naaim_signal = "NAIIM: 기관 매수 구간"
    else:
        naaim_signal = "NAIIM: 기관 중립 구간"
    
    # 종합 점수 계산
    pcr_score = np.clip((pcr_current - 0.5) / 1.5 * 100, 0, 100)
    vix_score = np.clip((vix_current - 10) / 40 * 100, 0, 100)
    naaim_score = naaim_current
    
    weights = [0.3, 0.4, 0.3]
    composite_score = (pcr_score * weights[0] + vix_score * weights[1] + naaim_score * weights[2])
    
    print(f"PCR: {pcr_current:.3f} - {pcr_signal}")
    print(f"VIX: {vix_current:.2f} - {vix_signal}")
    print(f"NAIIM: {naaim_current:.1f} - {naaim_signal}")
    print(f"종합 시장 심리 점수: {composite_score:.1f}/100")
    
    # 차트 표시
    plt.tight_layout()
    plt.show()
    
    print("\n=== 테스트 완료 ===")

if __name__ == "__main__":
    test_integrated_sentiment_chart()
