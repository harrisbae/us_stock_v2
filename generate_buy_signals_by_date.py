#!/usr/bin/env python3
"""
여러 종목의 매수시그널을 일자별로 정리하여 테이블 생성
"""

import yfinance as yf
import pandas as pd
import sys
import os
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.indicators.hma_mantra.signals import get_hma_mantra_md_signals

def get_vix_data(start_date, end_date):
    """VIX 데이터 다운로드"""
    try:
        # 시작일을 조금 앞당겨서 데이터 확보
        start_dt = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=30)
        vix_data = yf.download('^VIX', start=start_dt.strftime('%Y-%m-%d'), end=end_date, progress=False)
        if vix_data.empty:
            return None
        # Series로 반환 (날짜별 종가)
        return vix_data['Close']
    except Exception as e:
        print(f"⚠️  VIX 데이터 로드 오류: {e}", file=sys.stderr)
        return None

def read_symbols_from_file(file_path):
    """파일에서 종목 리스트를 읽어옵니다."""
    symbols = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 주석 라인과 빈 라인 건너뛰기
                if line and not line.startswith('#'):
                    symbols.append(line)
        return symbols
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")
        return []

def analyze_stock_buy_signals(symbol, start_date='2025-01-01', end_date='2025-10-14'):
    """단일 종목의 모든 매수시그널 분석"""
    try:
        # 데이터 다운로드
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            return []
        
        # HMA Mantra 매수시그널 분석
        signals = get_hma_mantra_md_signals(data, symbol)
        
        # 매수 신호만 필터링
        buy_signals = [s for s in signals if s['type'] == 'BUY']
        sell_signals = [s for s in signals if s['type'] == 'SELL']
        
        if not buy_signals:
            return []
        
        # 현재가
        current_price = float(data['Close'].iloc[-1])
        
        # 회사명 매핑
        company_names = {
            'CEG': 'Constellation Energy',
            'TEM': 'Templeton Emerging Markets',
            'RKLB': 'Rocket Lab USA',
            'OKLO': 'Oklo Inc',
            'SMR': 'NuScale Power',
            'LEU': 'Centrus Energy',
            'IONQ': 'IonQ Inc',
            'TSLA': 'Tesla Inc',
            'PLTR': 'Palantir Technologies',
            'NVDA': 'NVIDIA Corp',
            'META': 'Meta Platforms',
            'MSFT': 'Microsoft Corp',
            'GOOGL': 'Alphabet Inc',
            'HIMS': 'Hims & Hers Health',
            'RXRX': 'Recursion Pharmaceuticals',
            'HOOD': 'Robinhood Markets'
        }
        
        company_name = company_names.get(symbol, symbol)
        
        # 모든 매수시그널에 대한 정보 생성
        results = []
        for buy_signal in buy_signals:
            price_change = current_price - buy_signal['price']
            return_pct = (price_change / buy_signal['price']) * 100
            
            results.append({
                'symbol': symbol,
                'company_name': company_name,
                'buy_date': buy_signal['date'],
                'buy_date_str': buy_signal['date'].strftime('%Y-%m-%d'),
                'buy_price': buy_signal['price'],
                'buy_reason': buy_signal['reason'],
                'macd_state': buy_signal.get('macd_state', 0),
                'current_price': current_price,
                'return_pct': return_pct,
                'total_buy_signals': len(buy_signals),
                'total_sell_signals': len(sell_signals)
            })
        
        return results
        
    except Exception as e:
        print(f"⚠️  {symbol} 분석 오류: {e}", file=sys.stderr)
        return []

def main():
    """메인 실행 함수"""
    if len(sys.argv) < 2:
        print("Usage: python generate_buy_signals_by_date.py SYMBOL_FILE [START_DATE] [END_DATE]")
        sys.exit(1)
    
    symbol_file = sys.argv[1]
    start_date = sys.argv[2] if len(sys.argv) > 2 else '2025-01-01'
    end_date = sys.argv[3] if len(sys.argv) > 3 else '2025-10-14'
    
    # 종목 리스트 읽기
    symbols = read_symbols_from_file(symbol_file)
    
    if not symbols:
        print("❌ 종목 리스트를 읽을 수 없습니다.")
        sys.exit(1)
    
    # 각 종목 분석 (stderr로 진행상황 출력)
    results = []
    for symbol in symbols:
        symbol_results = analyze_stock_buy_signals(symbol, start_date, end_date)
        results.extend(symbol_results)
    
    if not results:
        print("❌ 매수시그널이 있는 종목이 없습니다.")
        sys.exit(1)
    
    # VIX 데이터 가져오기
    vix_data = get_vix_data(start_date, end_date)
    
    # 날짜별로 그룹화
    date_groups = defaultdict(list)
    for result in results:
        date_groups[result['buy_date']].append(result)
    
    # 날짜 순으로 정렬
    sorted_dates = sorted(date_groups.keys())
    
    # 출력 파일 생성
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"output/analysis/buy_signals_by_date_{timestamp}.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== 매수시그널 일자별 종목 정리 ===\n")
        f.write(f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"분석 기간: {start_date} ~ {end_date}\n")
        f.write(f"분석 종목: {len(symbols)}개\n")
        f.write(f"매수시그널 발견: {len(results)}개\n\n")
        
        # 일자별 테이블
        for buy_date in sorted_dates:
            stocks_on_date = date_groups[buy_date]
            date_str = buy_date.strftime('%Y-%m-%d')
            weekday = buy_date.strftime('%A')
            
            # 해당 날짜의 VIX 값 가져오기
            vix_value = None
            if vix_data is not None:
                # 정확한 날짜에 VIX 값이 없으면 가장 가까운 이전 날짜 사용
                try:
                    if buy_date in vix_data.index:
                        vix_value = float(vix_data.loc[buy_date])
                    else:
                        # 해당 날짜 이전의 가장 가까운 VIX 값 찾기
                        prev_dates = vix_data.index[vix_data.index < buy_date]
                        if len(prev_dates) > 0:
                            vix_value = float(vix_data.loc[prev_dates[-1]])
                except:
                    pass
            
            vix_str = f"VIX: {vix_value:.2f}" if vix_value else "VIX: N/A"
            
            f.write(f"\n📅 {date_str} ({weekday}) - {len(stocks_on_date)}개 종목 | {vix_str}\n")
            f.write("=" * 150 + "\n")
            f.write(f"{'순번':<4} {'종목':<6} {'회사명':<20} {'매수가격':<12} {'현재가':<12} {'수익률':<12} {'매수이유':<25} {'MACD':<8} {'VIX':<8} {'신호수':<10}\n")
            f.write("-" * 150 + "\n")
            
            for i, stock in enumerate(stocks_on_date, 1):
                return_str = f"{stock['return_pct']:+.1f}%"
                signal_count = f"{stock['total_buy_signals']}B/{stock['total_sell_signals']}S"
                vix_display = f"{vix_value:.2f}" if vix_value else "N/A"
                f.write(f"{i:<4} {stock['symbol']:<6} {stock['company_name']:<20} "
                       f"${stock['buy_price']:<11.2f} ${stock['current_price']:<11.2f} {return_str:<12} "
                       f"{stock['buy_reason']:<25} {stock['macd_state']:<8.4f} {vix_display:<8} {signal_count:<10}\n")
            
            f.write("-" * 150 + "\n")
        
        # 요약 통계
        f.write(f"\n\n=== 일자별 매수시그널 요약 ===\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'날짜':<12} {'요일':<10} {'VIX':<8} {'종목수':<8} {'종목 리스트':<60}\n")
        f.write("-" * 100 + "\n")
        
        for buy_date in sorted_dates:
            stocks_on_date = date_groups[buy_date]
            date_str = buy_date.strftime('%Y-%m-%d')
            weekday = buy_date.strftime('%A')
            stock_count = len(stocks_on_date)
            stock_symbols = ', '.join([stock['symbol'] for stock in stocks_on_date])
            
            # VIX 값 가져오기
            vix_value = None
            if vix_data is not None:
                try:
                    if buy_date in vix_data.index:
                        vix_value = float(vix_data.loc[buy_date])
                    else:
                        prev_dates = vix_data.index[vix_data.index < buy_date]
                        if len(prev_dates) > 0:
                            vix_value = float(vix_data.loc[prev_dates[-1]])
                except:
                    pass
            
            vix_display = f"{vix_value:.2f}" if vix_value else "N/A"
            
            f.write(f"{date_str:<12} {weekday:<10} {vix_display:<8} {stock_count:<8} {stock_symbols:<60}\n")
        
        f.write("-" * 100 + "\n")
        f.write(f"\n총 분석 종목: {len(symbols)}개\n")
        f.write(f"매수시그널 발견: {len(results)}개\n")
        f.write(f"매수시그널 발생 일수: {len(sorted_dates)}일\n")
        
        # 가장 많은 매수시그널이 발생한 날짜
        if sorted_dates:
            max_date = max(date_groups.keys(), key=lambda x: len(date_groups[x]))
            max_count = len(date_groups[max_date])
            max_stocks = [stock['symbol'] for stock in date_groups[max_date]]
            
            f.write(f"\n🏆 최대 매수시그널 발생일: {max_date.strftime('%Y-%m-%d')} ({max_count}개 종목)\n")
            f.write(f"   종목: {', '.join(max_stocks)}\n")
    
    # 파일 경로만 stdout에 출력 (스크립트에서 사용)
    print(output_file)

if __name__ == "__main__":
    main()
