"""
종목 목록 → 급등 셋업/급등 진행 후보 필터.

detect_surge_setup / detect_surge_state 를 재사용해
셋업A·셋업B·점화임박 (및 옵션으로 이미 급등 중) 종목을 걸러낸다.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import yfinance as yf

from .adx_dmi_indicator import analyze_adx_dmi
from .visualization.tech_annotation_chart import (
    _compute_aux_trend_indicators,
    _normalize,
    compute_support_resistance_levels,
    detect_surge_setup,
    detect_surge_state,
)

SETUP_STATES = ("점화임박", "셋업A", "셋업B")
ACTIVE_SURGE_STATES = ("급등진행", "급등과열")
TIER_ORDER = {"점화임박": 0, "셋업A": 1, "셋업B": 2, "급등진행": 3, "급등과열": 4}


@dataclass
class SurgeCandidate:
    ticker: str
    tier: str
    score: int
    kind: Optional[str]
    price: float
    flip_age: Optional[int]
    flip_date: Optional[str]
    consol_net: Optional[float]
    consol_rng: Optional[float]
    dist_hi20: Optional[float]
    adx: Optional[float]
    rsi: Optional[float]
    ret_5: Optional[float]
    vol_mult: Optional[float]
    reasons: str
    surge_state: str
    setup_state: str


def parse_symbol_list(path: str | Path) -> List[str]:
    """
    종목 파일 파싱.
    - 한 줄 한 티커
    - # 주석 / 빈 줄 무시
    - CSV면 첫 컬럼(ticker/symbol) 사용
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines()]
    if not lines:
        return []

    # CSV 헤더 감지
    header = lines[0].lower().replace(" ", "")
    if "ticker" in header or header.startswith("symbol") or "," in lines[0]:
        tickers: List[str] = []
        with path.open(encoding="utf-8", errors="ignore", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
            except csv.Error:
                dialect = csv.excel
            reader = csv.DictReader(f, dialect=dialect)
            if reader.fieldnames:
                keys = {k.lower().strip(): k for k in reader.fieldnames if k}
                key = keys.get("ticker") or keys.get("symbol") or reader.fieldnames[0]
                for row in reader:
                    raw = (row.get(key) or "").strip()
                    tok = _normalize_ticker(raw)
                    if tok:
                        tickers.append(tok)
                return list(dict.fromkeys(tickers))

    tickers = []
    for line in lines:
        if not line or line.startswith("#"):
            continue
        tok = _normalize_ticker(line.replace(",", " ").split()[0])
        if tok:
            tickers.append(tok)
    return list(dict.fromkeys(tickers))


def _normalize_ticker(raw: str) -> Optional[str]:
    tok = (raw or "").strip().upper().strip("[]()\"'")
    if not tok or tok.startswith("#"):
        return None
    # 한글 섹션 헤더 등 제외
    if not all(c.isalnum() or c in (".", "-", "^") for c in tok):
        return None
    if len(tok) > 12:
        return None
    return tok


def _rsi14(close: pd.Series) -> Optional[float]:
    if close is None or len(close) < 16:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    val = (100 - 100 / (1 + rs)).iloc[-1]
    if pd.isna(val):
        return None
    return round(float(val), 1)


def evaluate_ticker(
    ticker: str,
    *,
    period: str = "6mo",
    auto_adjust: bool = True,
) -> Optional[Dict[str, Any]]:
    """단일 종목 급등/셋업 스냅샷. 실패 시 None."""
    try:
        raw = yf.download(ticker, period=period, progress=False, auto_adjust=auto_adjust)
        if raw is None or len(raw) < 40:
            return None
        if hasattr(raw.columns, "nlevels") and raw.columns.nlevels > 1:
            raw.columns = raw.columns.get_level_values(0)
        df = _normalize(raw)
        aux = _compute_aux_trend_indicators(df)
        if aux.get("st_dir_series") is None:
            return None
        adx = analyze_adx_dmi(df)
        lv = {x["label"]: x for x in compute_support_resistance_levels(df)}
        close = df["Close"].astype(float)
        rsi = _rsi14(close)
        surge = detect_surge_state(
            df,
            aux_ind=aux,
            adx_res=adx,
            rsi_val=rsi,
            lv_by_label=lv,
        )
        setup = detect_surge_setup(df, aux_ind=aux, adx_res=adx)
        return {
            "ticker": ticker,
            "price": float(close.iloc[-1]),
            "adx": adx.get("adx"),
            "rsi": rsi,
            "surge": surge,
            "setup": setup,
        }
    except Exception:
        return None


def _to_candidate(snap: Dict[str, Any], *, prefer_setup: bool) -> Optional[SurgeCandidate]:
    surge = snap["surge"]
    setup = snap["setup"]
    s_state = surge.get("state") or "해당없음"
    u_state = setup.get("state") or "해당없음"

    if s_state in ACTIVE_SURGE_STATES and not prefer_setup:
        tier = s_state
        score = int(surge.get("score") or 0)
        kind = None
        reasons = ",".join(surge.get("reasons") or [])
        flip_age = setup.get("flip_age")
        flip_date = setup.get("flip_date")
        consol_net = setup.get("consol_net")
        consol_rng = setup.get("consol_rng")
        dist = setup.get("dist_hi20")
    elif setup.get("active") and u_state in SETUP_STATES:
        tier = u_state
        score = int(setup.get("score") or 0)
        kind = setup.get("kind")
        reasons = ",".join(setup.get("reasons") or [])
        flip_age = setup.get("flip_age")
        flip_date = setup.get("flip_date")
        consol_net = setup.get("consol_net")
        consol_rng = setup.get("consol_rng")
        dist = setup.get("dist_hi20")
    else:
        return None

    return SurgeCandidate(
        ticker=snap["ticker"],
        tier=tier,
        score=score,
        kind=kind,
        price=float(snap["price"]),
        flip_age=flip_age,
        flip_date=flip_date,
        consol_net=consol_net,
        consol_rng=consol_rng,
        dist_hi20=dist,
        adx=snap.get("adx") if snap.get("adx") is not None else setup.get("adx"),
        rsi=snap.get("rsi"),
        ret_5=surge.get("ret_5"),
        vol_mult=surge.get("vol_mult"),
        reasons=reasons[:80],
        surge_state=s_state,
        setup_state=u_state,
    )


def screen_surge_candidates(
    tickers: Sequence[str],
    *,
    period: str = "6mo",
    min_score: int = 40,
    include_active_surge: bool = False,
    kinds: Optional[Sequence[str]] = None,
    auto_adjust: bool = True,
) -> Tuple[List[SurgeCandidate], List[str]]:
    """
    Returns:
      (통과 후보, 오류/스킵 티커)
    """
    allow_kinds = None
    if kinds:
        allow_kinds = {k.strip().upper() for k in kinds if k and k.strip()}

    hits: List[SurgeCandidate] = []
    errors: List[str] = []

    for t in tickers:
        snap = evaluate_ticker(t, period=period, auto_adjust=auto_adjust)
        if snap is None:
            errors.append(t)
            continue

        surge = snap["surge"]
        setup = snap["setup"]
        s_state = surge.get("state") or "해당없음"

        # 이미 급등 중
        if s_state in ACTIVE_SURGE_STATES:
            if not include_active_surge:
                continue
            cand = _to_candidate(snap, prefer_setup=False)
        elif setup.get("active"):
            cand = _to_candidate(snap, prefer_setup=True)
        else:
            continue

        if cand is None or cand.score < min_score:
            continue
        if allow_kinds and cand.kind and cand.kind.upper() not in allow_kinds:
            # 급등진행/과열은 kind=None → kinds 필터 시 제외할지: include_active면 통과
            if cand.tier not in ACTIVE_SURGE_STATES:
                continue
        hits.append(cand)

    hits.sort(key=lambda c: (TIER_ORDER.get(c.tier, 9), -c.score, c.ticker))
    return hits, errors


def write_candidates_csv(path: str | Path, rows: Iterable[SurgeCandidate]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = [f.name for f in fields(SurgeCandidate)]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    return path


def write_symbol_list(path: str | Path, rows: Iterable[SurgeCandidate], *, header: str = "") -> Path:
    """hma.sh --file 용 티커 목록."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", encoding="utf-8") as f:
        if header:
            f.write(f"# {header}\n")
        f.write(f"# generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · surge setup filter\n")
        for r in rows:
            f.write(f"{r.ticker}\n")
    return path


def format_candidates_table(rows: Sequence[SurgeCandidate]) -> str:
    if not rows:
        return "(후보 없음)"
    lines = [
        f"{'TICKER':5s} {'TIER':8s} {'SCR':>3s} {'PRICE':>10s} {'ST↑':>5s} "
        f"{'NET':>7s} {'RNG':>6s} {'HI20':>7s} {'ADX':>5s} {'RSI':>5s} REASONS"
    ]
    for r in rows:
        age = f"+{r.flip_age}d" if r.flip_age is not None else "-"
        net = f"{r.consol_net:+.1f}%" if r.consol_net is not None else "-"
        rng = f"{r.consol_rng:.1f}%" if r.consol_rng is not None else "-"
        hi = f"{r.dist_hi20:+.1f}%" if r.dist_hi20 is not None else "-"
        adx = f"{r.adx:.0f}" if r.adx is not None else "-"
        rsi = f"{r.rsi:.0f}" if r.rsi is not None else "-"
        lines.append(
            f"{r.ticker:5s} {r.tier:8s} {r.score:3d} {r.price:10.2f} {age:>5s} "
            f"{net:>7s} {rng:>6s} {hi:>7s} {adx:>5s} {rsi:>5s} {r.reasons}"
        )
    return "\n".join(lines)
