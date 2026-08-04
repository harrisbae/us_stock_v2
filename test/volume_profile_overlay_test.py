#!/usr/bin/env python3
"""
Volume Profile 오버레이 차트 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from src.indicators.hma_mantra.visualization.volume_profile_overlay_chart import plot_main_chart_with_volume_profile_overlay

def main():
    import sys
    import argparse
    
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='Volume Profile 오버레이 차트 생성기')
    parser.add_argument('ticker', nargs='?', default='TSLA', help='종목코드 (기본값: TSLA)')
    parser.add_argument('period', nargs='?', default='6mo', help='데이터 기간 (기본값: 6mo)')
    parser.add_argument('rsi_window', nargs='?', type=int, default=20, help='RSI 다이버전스 윈도우 (기본값: 20)')
    parser.add_argument('rsi_pivot', nargs='?', type=int, default=3, help='RSI 피벗 스팬 (기본값: 3)')
    parser.add_argument('target_prices', nargs='?', default='', help='Target 가격들 (형식: buy:sell:stop)')
    parser.add_argument('show_targets', nargs='?', default='true', help='Target 가격 표시 여부 (기본값: true)')
    parser.add_argument('--auto_adjust', default='false', help='가격 조정 여부 (기본값: false)')
    # 박스권 옵션들 추가
    parser.add_argument('--show-box-ranges', default='true', help='박스권 표시 여부 (기본값: true)')
    parser.add_argument('--box-period', type=int, default=20, help='박스권 계산 기간 (기본값: 20일)')
    parser.add_argument('--num-boxes', type=int, default=2, help='표시할 박스권 개수 (기본값: 2)')
    parser.add_argument('--box-overlap', type=int, default=5, help='박스권 간 겹치는 일수 (기본값: 5)')
    parser.add_argument('--box-style', default='default', help='박스권 스타일 (default/gradient/rainbow, 기본값: default)')
    parser.add_argument('--avoid-time-overlap', default='true', help='시간축 겹침 방지 여부 (기본값: true)')
    parser.add_argument(
        '--show-pattern-strip',
        action='store_true',
        help='메인 차트 아래 차트 패턴 타임라인 서브플롯 표시(행 수=CHART_PATTERN_CRITERIA)',
    )
    parser.add_argument(
        '--pattern-main',
        default='',
        help=(
            '메인 차트에 그릴 패턴 id (쉼표 구분). all=전 id. '
            '예: post_box_bull,fvg_gap,triple_bottom_w,range_box,...'
        ),
    )
    parser.add_argument(
        '--pattern-range-box-main-max',
        type=int,
        default=1,
        metavar='N',
        help='메인 차트 range_box 최대 표시 개수(신뢰도·종료일 선별). 스트립은 전체. 0=메인 미표시 (기본 1)',
    )
    parser.add_argument(
        '--pattern-main-min-confidence',
        type=float,
        default=0.0,
        metavar='F',
        help='메인 차트 패턴 오버레이 최소 신뢰도(0~1, 기본 0.0). 스트립에는 영향 없음',
    )
    parser.add_argument('--show-disparity-strategy', action='store_true', help='이격도 95/105 전략 마커/요약 표시')
    parser.add_argument('--disparity-ma', type=int, default=20, help='이격도 기준 이동평균 기간 (기본 20)')
    parser.add_argument('--disparity-low', type=float, default=95.0, help='이격도 저평가 임계값 (기본 95)')
    parser.add_argument('--disparity-high', type=float, default=105.0, help='이격도 고평가 임계값 (기본 105)')
    parser.add_argument('--disparity-conf-weight-depth', type=float, default=0.22, help='이격도 conf 가중치(깊이, 기본 0.22)')
    parser.add_argument('--disparity-conf-weight-volume', type=float, default=0.16, help='이격도 conf 가중치(거래량, 기본 0.16)')
    parser.add_argument('--disparity-conf-weight-trend', type=float, default=0.10, help='이격도 conf 가중치(추세, 기본 0.10)')
    parser.add_argument(
        '--disparity-conf-preset',
        default='',
        help='이격도 conf 프리셋(growth|balanced|defensive). 지정 시 개별 weight보다 우선',
    )
    parser.add_argument('--show-adx-dmi', action='store_true', help='ADX·DMI(+DI/−DI) 추세·강도 서브플롯 표시')
    parser.add_argument('--adx-period', type=int, default=14, help='ADX/DMI 기간 (기본 14)')
    parser.add_argument('--adx-sideways', type=float, default=20.0, help='ADX 횡보 임계 (기본 20)')
    parser.add_argument('--adx-trend', type=float, default=25.0, help='ADX 추세 시작 (기본 25)')
    parser.add_argument('--adx-strong', type=float, default=40.0, help='ADX 강추세 (기본 40)')
    parser.add_argument('--tech-chart', action='store_true', help='기술 분석 주석 차트(별도 PNG) 추가 생성')
    parser.add_argument('--tech-chart-only', action='store_true', help='기술 분석 주석 차트만 생성 (메인 오버레이 차트 생략)')
    parser.add_argument('--show-bb', action='store_true', help='기술 분석 주석 차트에 볼린저 밴드 선 표시 (기본은 이탈/돌파 태그만)')

    # argparse로 인자 파싱
    args = parser.parse_args()
    
    # 변수 할당
    ticker = args.ticker
    period = args.period
    rsi_window = args.rsi_window
    rsi_pivot = args.rsi_pivot
    auto_adjust = args.auto_adjust.lower() in ['true', '1', 'yes', 'y']
    
    # 박스권 옵션들
    show_box_ranges = args.show_box_ranges.lower() in ['true', '1', 'yes', 'y']
    box_period = args.box_period
    num_boxes = args.num_boxes
    box_overlap = args.box_overlap
    box_style = args.box_style
    avoid_time_overlap = args.avoid_time_overlap.lower() in ['true', '1', 'yes', 'y']
    show_pattern_strip = args.show_pattern_strip
    pm = (args.pattern_main or '').strip()
    pattern_main_overlays = None if not pm else [x.strip() for x in pm.split(',') if x.strip()]
    disparity_conf_preset = (args.disparity_conf_preset or '').strip().lower()

    # 프리셋 적용: 지정 시 개별 weight보다 우선
    preset_map = {
        'growth': (0.28, 0.10, 0.08),
        'balanced': (0.22, 0.16, 0.10),
        'defensive': (0.16, 0.14, 0.18),
    }
    if disparity_conf_preset:
        if disparity_conf_preset in preset_map:
            w_depth, w_volume, w_trend = preset_map[disparity_conf_preset]
            print(
                f"이격도 conf 프리셋 적용: {disparity_conf_preset} "
                f"(depth={w_depth}, volume={w_volume}, trend={w_trend})"
            )
        else:
            print(
                f"알 수 없는 disparity conf 프리셋: {disparity_conf_preset} "
                f"(허용: growth|balanced|defensive) → 개별 weight 사용"
            )
            w_depth = args.disparity_conf_weight_depth
            w_volume = args.disparity_conf_weight_volume
            w_trend = args.disparity_conf_weight_trend
    else:
        w_depth = args.disparity_conf_weight_depth
        w_volume = args.disparity_conf_weight_volume
        w_trend = args.disparity_conf_weight_trend

    # Target 가격 옵션들
    target_buy_price = None
    target_sell_price = None
    stop_loss_price = None
    show_target_prices = args.show_targets.lower() in ['true', '1', 'yes', 'y']
    
    # Target 가격들을 명령행 인자에서 받기 (형식: buy:sell:stop)
    if args.target_prices:
        target_prices = args.target_prices.split(':')
        if len(target_prices) >= 1 and target_prices[0]:
            try:
                target_buy_price = float(target_prices[0])
            except ValueError:
                target_buy_price = None
        
        if len(target_prices) >= 2 and target_prices[1]:
            try:
                target_sell_price = float(target_prices[1])
            except ValueError:
                target_sell_price = None
        
        if len(target_prices) >= 3 and target_prices[2]:
            try:
                stop_loss_price = float(target_prices[2])
            except ValueError:
                stop_loss_price = None
    
    print(f"데이터 다운로드 중: {ticker} (period={period})")
    
    # 날짜 범위인지 확인 (YYYY-MM-DD_YYYY-MM-DD 형식)
    if '_' in period and len(period.split('_')) == 2:
        start_date, end_date = period.split('_')
        print(f"날짜 범위: {start_date} ~ {end_date}")
        
        # yfinance는 end 날짜를 포함하지 않으므로 하루를 더해서 다운로드
        from datetime import datetime, timedelta
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        end_date_plus_one = (end_dt + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # 날짜 범위로 데이터 다운로드 (종료일 포함을 위해 하루 추가)
        data = yf.download(ticker, start=start_date, end=end_date_plus_one, progress=False, auto_adjust=auto_adjust)
    else:
        # 기존 period 방식으로 데이터 다운로드
        data = yf.download(ticker, period=period, progress=False, auto_adjust=auto_adjust)
    
    if data.empty:
        print(f"데이터를 가져올 수 없습니다: {ticker}")
        return
    
    print(f"다운로드 완료: {len(data)} 개 데이터")
    print(f"데이터 기간: {data.index[0].strftime('%Y-%m-%d')} ~ {data.index[-1].strftime('%Y-%m-%d')}")
    
    # 출력 경로 설정
    output_dir = f"output/hma_mantra/{ticker}"
    os.makedirs(output_dir, exist_ok=True)
    save_path = f"{output_dir}/{ticker}_volume_profile_overlay_{period}_chart.png"

    # 기술 분석 주석 차트 (별도 파일, 생성 일자 포함 — 일자별 보존)
    if args.tech_chart or args.tech_chart_only:
        from src.indicators.hma_mantra.visualization.tech_annotation_chart import plot_tech_annotation_chart
        from datetime import datetime as _dt
        date_tag = _dt.now().strftime('%Y%m%d')
        tech_save_path = f"{output_dir}/{ticker}_tech_annotation_{period}_{date_tag}_chart.png"
        print("기술 분석 주석 차트 생성 중...")
        plot_tech_annotation_chart(
            data=data,
            ticker=ticker,
            save_path=tech_save_path,
            adx_period=args.adx_period,
            target_buy_price=target_buy_price,
            target_sell_price=target_sell_price,
            stop_loss_price=stop_loss_price,
            show_target_prices=show_target_prices,
            show_bb=bool(args.show_bb),
        )
        if args.tech_chart_only:
            return

    print(f"Volume Profile 오버레이 차트 생성 중...")
    
    # Volume Profile 오버레이 차트 생성
    plot_main_chart_with_volume_profile_overlay(
        data=data,
        ticker=ticker,
        save_path=save_path,
        rsi_divergence_window=rsi_window,
        rsi_pivot_span=rsi_pivot,
        include_hidden_divergence=True,
        # Target 가격 옵션들 추가
        target_buy_price=target_buy_price,
        target_sell_price=target_sell_price,
        stop_loss_price=stop_loss_price,
        show_target_prices=show_target_prices,
        # 박스권 옵션들 추가
        show_box_ranges=show_box_ranges,
        box_period=box_period,
        num_boxes=num_boxes,
        box_overlap=box_overlap,
        box_style=box_style,
        avoid_time_overlap=avoid_time_overlap,
        show_pattern_strip=show_pattern_strip,
        pattern_main_overlays=pattern_main_overlays,
        pattern_range_box_main_max=args.pattern_range_box_main_max,
        pattern_main_min_confidence=args.pattern_main_min_confidence,
        show_disparity_strategy=args.show_disparity_strategy,
        disparity_ma=args.disparity_ma,
        disparity_low=args.disparity_low,
        disparity_high=args.disparity_high,
        disparity_conf_weight_depth=w_depth,
        disparity_conf_weight_volume=w_volume,
        disparity_conf_weight_trend=w_trend,
        show_adx_dmi=args.show_adx_dmi,
        adx_period=args.adx_period,
        adx_sideways=args.adx_sideways,
        adx_trend=args.adx_trend,
        adx_strong=args.adx_strong,
    )
    
    print(f"차트 저장 완료: {save_path}")

if __name__ == "__main__":
    main() 