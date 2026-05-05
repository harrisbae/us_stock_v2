#!/usr/bin/env python3
"""
반등 신호, 반등 기대, 상승 지속 대상 추출 스크립트
"""

import re
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

def parse_buy_signals_file(file_path):
    """매수 신호 파일 파싱"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    signals = []
    current_date = None
    current_vix = None
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 날짜 라인 찾기
        date_match = re.search(r'📅 (\d{4}-\d{2}-\d{2})', line)
        if date_match:
            current_date_str = date_match.group(1)
            current_date = datetime.strptime(current_date_str, '%Y-%m-%d')
            # VIX 추출
            vix_match = re.search(r'VIX: ([\d.]+)', line)
            current_vix = float(vix_match.group(1)) if vix_match else None
            i += 1
            continue
        
        # 데이터 라인 찾기 (순번, 종목, 회사명, 매수가격, 현재가, 수익률, 매수이유, MACD, VIX, 신호수)
        # 형식: "1    ETHZ   ETHZ                 $17.94       $10.68       -40.5%       HMA 상향돌파 + RSI 매수         0.1803   19.08    33B/32S"
        if current_date and re.match(r'^\d+\s+\w+', line) and '$' in line:
            try:
                # 공백으로 분리
                parts = line.split()
                if len(parts) < 6:
                    i += 1
                    continue
                
                rank = int(parts[0])
                symbol = parts[1]
                
                # 회사명 찾기 (종목 다음부터 $ 전까지)
                company_parts = []
                j = 2
                while j < len(parts) and not parts[j].startswith('$'):
                    company_parts.append(parts[j])
                    j += 1
                company_name = ' '.join(company_parts) if company_parts else symbol
                
                # 가격 정보 추출
                buy_price_str = parts[j].replace('$', '') if j < len(parts) else None
                current_price_str = parts[j+1].replace('$', '') if j+1 < len(parts) else None
                return_pct_str = parts[j+2].replace('%', '') if j+2 < len(parts) else None
                
                if not all([buy_price_str, current_price_str, return_pct_str]):
                    i += 1
                    continue
                
                buy_price = float(buy_price_str)
                current_price = float(current_price_str)
                return_pct = float(return_pct_str)
                
                # 매수 이유 추출 (수익률 다음부터 MACD 전까지)
                reason_parts = []
                k = j + 3
                while k < len(parts) and not re.match(r'^-?[\d.]+$', parts[k]):
                    reason_parts.append(parts[k])
                    k += 1
                reason = ' '.join(reason_parts) if reason_parts else "알 수 없음"
                
                signals.append({
                    'buy_date': current_date,
                    'buy_date_str': current_date_str,
                    'symbol': symbol,
                    'company_name': company_name,
                    'buy_price': buy_price,
                    'current_price': current_price,
                    'return_pct': return_pct,
                    'vix': current_vix,
                    'reason': reason
                })
            except (ValueError, IndexError) as e:
                pass
        
        i += 1
    
    return signals

def categorize_signals(signals, days=30):
    """신호를 반등 신호, 반등 기대, 상승 지속으로 분류"""
    cutoff_date = datetime.now() - timedelta(days=days)
    
    recent_signals = [s for s in signals if s['buy_date'] >= cutoff_date]
    
    rebound_signals = []      # 반등 신호: 수익률 -10% 이하
    rebound_expected = []     # 반등 기대: 수익률 -10% ~ +5%
    continuation = []         # 상승 지속: 수익률 +5% 이상
    
    for signal in recent_signals:
        return_pct = signal['return_pct']
        
        if return_pct <= -10:
            rebound_signals.append(signal)
        elif return_pct < 5:
            rebound_expected.append(signal)
        else:
            continuation.append(signal)
    
    return rebound_signals, rebound_expected, continuation

def main():
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='반등 신호 / 반등 기대 / 상승 지속 대상 추출')
    parser.add_argument('--days', type=int, default=30, help='최근 N일 내 매수 신호 기준 (기본값: 30)')
    args = parser.parse_args()
    
    days = args.days
    
    # 최신 매수 신호 파일 찾기
    analysis_dir = Path('output/analysis')
    if not analysis_dir.exists():
        print("❌ output/analysis 디렉토리를 찾을 수 없습니다.")
        return
    
    # 최신 파일 찾기
    signal_files = list(analysis_dir.glob('buy_signals_by_date_*.txt'))
    if not signal_files:
        print("❌ 매수 신호 파일을 찾을 수 없습니다.")
        return
    
    latest_file = max(signal_files, key=lambda p: p.stat().st_mtime)
    print(f"📊 분석 파일: {latest_file.name}\n")
    
    # 신호 파싱
    signals = parse_buy_signals_file(latest_file)
    print(f"총 {len(signals)}개의 매수 신호 발견\n")
    
    # 분류
    rebound_signals, rebound_expected, continuation = categorize_signals(signals, days=days)
    
    # 결과 출력
    output_file = analysis_dir / f"rebound_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("반등 신호 / 반등 기대 / 상승 지속 대상 분석\n")
        f.write(f"분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"기준 파일: {latest_file.name}\n")
        f.write(f"최근 {days}일 내 매수 신호 기준\n")
        f.write("=" * 100 + "\n\n")
        
        # 1. 반등 신호 (수익률 -10% 이하)
        f.write("🔴 반등 신호 (수익률 -10% 이하)\n")
        f.write("=" * 100 + "\n")
        if rebound_signals:
            f.write(f"{'날짜':<12} {'종목':<8} {'회사명':<25} {'매수가격':<12} {'현재가':<12} {'수익률':<12} {'VIX':<8} {'매수이유':<30}\n")
            f.write("-" * 100 + "\n")
            for s in sorted(rebound_signals, key=lambda x: x['return_pct']):
                f.write(f"{s['buy_date_str']:<12} {s['symbol']:<8} {s['company_name']:<25} "
                       f"${s['buy_price']:<11.2f} ${s['current_price']:<11.2f} {s['return_pct']:+.1f}%{'':<6} "
                       f"{s['vix']:<8.2f} {s['reason']:<30}\n")
            f.write(f"\n총 {len(rebound_signals)}개 종목\n")
        else:
            f.write("반등 신호 대상 없음\n")
        f.write("\n\n")
        
        # 2. 반등 기대 (수익률 -10% ~ +5%)
        f.write("🟡 반등 기대 (수익률 -10% ~ +5%)\n")
        f.write("=" * 100 + "\n")
        if rebound_expected:
            f.write(f"{'날짜':<12} {'종목':<8} {'회사명':<25} {'매수가격':<12} {'현재가':<12} {'수익률':<12} {'VIX':<8} {'매수이유':<30}\n")
            f.write("-" * 100 + "\n")
            for s in sorted(rebound_expected, key=lambda x: x['return_pct']):
                f.write(f"{s['buy_date_str']:<12} {s['symbol']:<8} {s['company_name']:<25} "
                       f"${s['buy_price']:<11.2f} ${s['current_price']:<11.2f} {s['return_pct']:+.1f}%{'':<6} "
                       f"{s['vix']:<8.2f} {s['reason']:<30}\n")
            f.write(f"\n총 {len(rebound_expected)}개 종목\n")
        else:
            f.write("반등 기대 대상 없음\n")
        f.write("\n\n")
        
        # 3. 상승 지속 (수익률 +5% 이상)
        f.write("🟢 상승 지속 (수익률 +5% 이상)\n")
        f.write("=" * 100 + "\n")
        if continuation:
            f.write(f"{'날짜':<12} {'종목':<8} {'회사명':<25} {'매수가격':<12} {'현재가':<12} {'수익률':<12} {'VIX':<8} {'매수이유':<30}\n")
            f.write("-" * 100 + "\n")
            for s in sorted(continuation, key=lambda x: x['return_pct'], reverse=True):
                f.write(f"{s['buy_date_str']:<12} {s['symbol']:<8} {s['company_name']:<25} "
                       f"${s['buy_price']:<11.2f} ${s['current_price']:<11.2f} {s['return_pct']:+.1f}%{'':<6} "
                       f"{s['vix']:<8.2f} {s['reason']:<30}\n")
            f.write(f"\n총 {len(continuation)}개 종목\n")
        else:
            f.write("상승 지속 대상 없음\n")
        f.write("\n\n")
        
        # 요약
        f.write("=" * 100 + "\n")
        f.write("요약\n")
        f.write("=" * 100 + "\n")
        f.write(f"반등 신호: {len(rebound_signals)}개\n")
        f.write(f"반등 기대: {len(rebound_expected)}개\n")
        f.write(f"상승 지속: {len(continuation)}개\n")
        f.write(f"총계: {len(rebound_signals) + len(rebound_expected) + len(continuation)}개\n")
    
    print(f"✅ 결과 파일 저장: {output_file}")
    print(f"\n📊 분류 결과:")
    print(f"   🔴 반등 신호: {len(rebound_signals)}개")
    print(f"   🟡 반등 기대: {len(rebound_expected)}개")
    print(f"   🟢 상승 지속: {len(continuation)}개")
    
    return str(output_file)

if __name__ == '__main__':
    result_file = main()
    if result_file:
        print(f"\n📄 파일 경로: {result_file}")

