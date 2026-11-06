#!/usr/bin/env python3
"""
종목 목록 파일 또는 -s 지정 종목에서 급등 가능(준비/셋업/급등) 종목을 필터링.

예:
  python filter_surge_setup.py -f stocks/us_Harris_Wish.txt -p 6mo
  python filter_surge_setup.py -s GM,DAL,LRCX -p 6mo
  python filter_surge_setup.py -f stocks/us_Harris_Wish.txt --min-score 45 --include-active
  ./hma.sh --filter-surge-setup -s GM,DAL,LRCX -p 6mo
  ./hma.sh --filter-surge-setup -f stocks/us_Harris_Wish.txt -p 6mo
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from indicators.hma_mantra.surge_setup_screener import (  # noqa: E402
    format_candidates_table,
    parse_symbol_list,
    parse_symbols_arg,
    screen_surge_candidates,
    write_candidates_csv,
    write_symbol_list,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="급등 준비/셋업/급등 진행 종목 필터")
    parser.add_argument(
        "-f", "--file",
        default="",
        help="종목 목록 파일 (txt/csv)",
    )
    parser.add_argument(
        "-s", "--symbols",
        default="",
        help="콤마 구분 종목 (예: GM,DAL,LRCX). -f 와 함께 쓰면 합집합",
    )
    parser.add_argument("-p", "--period", default="6mo", help="시세 기간 (기본 6mo)")
    parser.add_argument("--min-score", type=int, default=40, help="최소 점수 (기본 40)")
    parser.add_argument(
        "--kinds",
        default="",
        help="허용 셋업 종류 필터 (예: A,B). 비우면 전체(준비 포함)",
    )
    parser.add_argument(
        "--include-active",
        action="store_true",
        help="이미 급등진행/급등과열인 종목도 포함",
    )
    parser.add_argument(
        "--out-dir",
        default="output/analysis",
        help="결과 저장 디렉터리 (기본 output/analysis)",
    )
    parser.add_argument(
        "--out-prefix",
        default="",
        help="출력 파일 prefix (기본 surge_setup_filtered_YYYYMMDD)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="파일 저장 생략 (콘솔만)",
    )
    args = parser.parse_args()

    tickers: list[str] = []
    src_bits: list[str] = []

    if args.symbols:
        from_s = parse_symbols_arg(args.symbols)
        if not from_s:
            print(f"오류: -s 종목을 파싱할 수 없습니다: {args.symbols!r}", file=sys.stderr)
            return 1
        tickers.extend(from_s)
        src_bits.append(f"-s {','.join(from_s)}")

    if args.file:
        src = Path(args.file)
        if not src.is_file():
            print(f"오류: 종목 파일 없음: {src}", file=sys.stderr)
            return 1
        from_f = parse_symbol_list(src)
        if not from_f:
            print(f"오류: 파싱된 종목이 없습니다: {src}", file=sys.stderr)
            return 1
        tickers.extend(from_f)
        src_bits.append(src.name)

    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        print("오류: -s SYMBOL[,SYMBOL...] 또는 -f/--file SYMBOL_FILE 이 필요합니다.", file=sys.stderr)
        return 1

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()] or None
    print(f"스캔: {len(tickers)}종목 · period={args.period} · min_score={args.min_score}")
    print(f"입력: {'; '.join(src_bits)}")
    if kinds:
        print(f"kinds={kinds}")
    if args.include_active:
        print("include: 급등진행/급등과열")

    hits, errors = screen_surge_candidates(
        tickers,
        period=args.period,
        min_score=args.min_score,
        include_active_surge=args.include_active,
        kinds=kinds,
    )

    print(f"\n통과 {len(hits)} / 입력 {len(tickers)}" + (f" · 스킵/오류 {len(errors)}" if errors else ""))
    print(format_candidates_table(hits))

    if errors and len(errors) <= 20:
        print(f"\n스킵/오류: {', '.join(errors)}")
    elif errors:
        print(f"\n스킵/오류 {len(errors)}건 (예: {', '.join(errors[:8])} …)")

    if not args.no_save:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d")
        prefix = args.out_prefix or f"surge_setup_filtered_{stamp}"
        csv_path = write_candidates_csv(out_dir / f"{prefix}.csv", hits)
        list_path = write_symbol_list(
            out_dir / f"{prefix}.txt",
            hits,
            header=f"from {' + '.join(src_bits)} · min_score={args.min_score}",
        )
        print(f"\nCSV : {csv_path}")
        print(f"LIST: {list_path}  (hma.sh --file 로 재사용 가능)")

    return 0


if __name__ == "__main__":
    # yfinance/matplotlib 노이즈 완화
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    raise SystemExit(main())
