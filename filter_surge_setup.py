#!/usr/bin/env python3
"""
종목 목록 파일에서 급등 가능(셋업/급등) 종목을 필터링.

예:
  python filter_surge_setup.py -f stocks/us_Harris_Wish.txt -p 6mo
  python filter_surge_setup.py -f stocks/us_Harris_Wish.txt --min-score 45 --include-active
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
    screen_surge_candidates,
    write_candidates_csv,
    write_symbol_list,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="급등 셋업/급등 진행 종목 필터")
    parser.add_argument("-f", "--file", required=True, help="종목 목록 파일 (txt/csv)")
    parser.add_argument("-p", "--period", default="6mo", help="시세 기간 (기본 6mo)")
    parser.add_argument("--min-score", type=int, default=40, help="최소 점수 (기본 40)")
    parser.add_argument(
        "--kinds",
        default="",
        help="허용 셋업 종류 필터 (예: A,B). 비우면 전체",
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

    src = Path(args.file)
    if not src.is_file():
        print(f"오류: 종목 파일 없음: {src}", file=sys.stderr)
        return 1

    tickers = parse_symbol_list(src)
    if not tickers:
        print(f"오류: 파싱된 종목이 없습니다: {src}", file=sys.stderr)
        return 1

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()] or None
    print(f"스캔: {len(tickers)}종목 · period={args.period} · min_score={args.min_score}")
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
            header=f"from {src.name} · min_score={args.min_score}",
        )
        print(f"\nCSV : {csv_path}")
        print(f"LIST: {list_path}  (hma.sh --file 로 재사용 가능)")

    return 0


if __name__ == "__main__":
    # yfinance/matplotlib 노이즈 완화
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    raise SystemExit(main())
