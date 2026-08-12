"""
기술 분석 주석 전용 차트 (인포그래픽 스타일).

메인 오버레이 차트와 별도 파일로 생성:
- 심플 캔들 + SMA20
- 지지/저항 수평선 + 우측 가격 라벨
- 자동 추세선 (피벗 연결)
- 캔들/차트 패턴 서클 + 한글 라벨 (신뢰도 상위 N건)
- 배당락일 D 마커 (+실제 하락 시 주석)
- 거래량 스트립 + 급증 강조
- 우측 인사이트 패널(BB·ST·SMA200 스냅샷 포함), 하단 요약 바
- SMA200·Supertrend 메인 오버레이 + ③ 저항돌파 목표 체인
- 볼린저 이탈/돌파 이벤트 태그 (+ 옵션 --show-bb 밴드 선)
- 현재가 기준 급등패턴 스코어·패널·메인 배지
- 급등 직전 셋업(A: ST전환후횡보 / B: ST전환직후 / 점화임박)
- 현재가 기준 급락패턴·급락 직전 셋업(A/B/붕괴임박) — 급등과 대칭
- PEG(1년·3년) 재무 평가 + 투자 전략 코멘트 (패널·하단바)
- 종목별 과거 급등 이력(A/B/D) · 유사조건 · 메인 마커
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter
from mplfinance.original_flavor import candlestick_ohlc

from ..adx_dmi_indicator import analyze_adx_dmi
from ..core import calculate_hma, calculate_mantra_bands
from ..chart_patterns import (
    CANDLE_PATTERN_IDS,
    CHART_PATTERN_CRITERIA,
    ID_TO_ROW,
    analyze_seven_criteria,
)
from ..disparity_strategy import analyze_disparity_strategy
from ..peg_indicator import build_peg_investment_view, calculate_peg_indicator

SUPPORT_COLOR = "#27ae60"
STRONG_SUPPORT_COLOR = "#c0392b"
RESISTANCE_COLOR = "#f39c12"
HIGH_COLOR = "#2980b9"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    return out


def _find_pivots(values: np.ndarray, span: int, mode: str) -> List[int]:
    n = len(values)
    out: List[int] = []
    for i in range(span, n - span):
        window = values[i - span: i + span + 1]
        if mode == "high" and values[i] >= window.max():
            out.append(i)
        elif mode == "low" and values[i] <= window.min():
            out.append(i)
    return out


def _cluster_levels(prices: List[float], cluster_pct: float) -> List[Tuple[float, int]]:
    """가격 클러스터링 → [(레벨 가격, 터치 수)]"""
    if not prices:
        return []
    prices = sorted(prices)
    clusters: List[List[float]] = [[prices[0]]]
    for p in prices[1:]:
        if abs(p - np.mean(clusters[-1])) / max(np.mean(clusters[-1]), 1e-9) <= cluster_pct:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [(float(np.mean(c)), len(c)) for c in clusters]


def compute_support_resistance_levels(
    ohlcv: pd.DataFrame,
    *,
    pivot_span: int = 3,
    cluster_pct: float = 0.012,
) -> List[Dict[str, Any]]:
    """
    지지/저항 레벨 산출.

    Returns:
        [{price, kind, label, color, touches}]
        (전고점/단기저항/현재지지/강력지지 + 최근지지/최근저항)

    Note:
        신저점·급락 구간에서는 피벗 저점이 모두 현재가 위에만 남아
        지지 후보가 비는 경우가 있다. 이 때는 기간 저점·롤링 저점으로 폴백한다.
        최근지지/최근저항은 lookback(20~40봉) 피벗이며, 구조선과 1.5% 이내면 병합한다.
    """
    df = _normalize(ohlcv)
    high = df["High"].astype(float).values
    low = df["Low"].astype(float).values
    close = df["Close"].astype(float)
    current = float(close.iloc[-1])
    period_high = float(high.max())
    period_low = float(low.min())
    low_s = df["Low"].astype(float)
    roll20_low = float(low_s.rolling(20, min_periods=1).min().iloc[-1])
    roll60_n = min(60, len(low_s))
    roll60_low = float(low_s.rolling(roll60_n, min_periods=1).min().iloc[-1])

    hi_idx = _find_pivots(high, pivot_span, "high")
    lo_idx = _find_pivots(low, pivot_span, "low")

    res_levels = _cluster_levels([float(high[i]) for i in hi_idx], cluster_pct)
    sup_levels = _cluster_levels([float(low[i]) for i in lo_idx], cluster_pct)

    levels: List[Dict[str, Any]] = []

    # 전고점 저항
    levels.append({
        "price": period_high,
        "kind": "high",
        "label": "전고점 저항",
        "color": HIGH_COLOR,
        "touches": 1,
    })

    # 단기 저항: 현재가 위, 전고점 미만에서 가장 가까운 클러스터
    above = [(p, t) for p, t in res_levels if current < p < period_high * 0.999]
    if above:
        p, t = min(above, key=lambda x: x[0] - current)
        levels.append({
            "price": p,
            "kind": "resistance",
            "label": "단기 저항",
            "color": RESISTANCE_COLOR,
            "touches": t,
        })

    # 현재 지지: 현재가 아래에서 가장 가까운 피벗 클러스터
    below = [(p, t) for p, t in sup_levels if p < current * 0.998]
    near_support: Optional[Tuple[float, int]] = None
    if below:
        near_support = max(below, key=lambda x: x[0])
        levels.append({
            "price": near_support[0],
            "kind": "support",
            "label": "현재 지지",
            "color": SUPPORT_COLOR,
            "touches": near_support[1],
        })
    else:
        # 폴백: 신저점 부근 — 피벗 지지가 전부 현재가 위일 때 기간/롤링 저점 사용
        fb = min(period_low, roll20_low)
        if fb > 0 and fb <= current * 1.005:
            near_support = (fb, 1)
            levels.append({
                "price": fb,
                "kind": "support",
                "label": "현재 지지",
                "color": SUPPORT_COLOR,
                "touches": 1,
            })
        # 이탈한 직전 피벗 지지 (현재가~+25% 구간, 가장 가까운 클러스터)
        above_sup = [(p, t) for p, t in sup_levels if current < p <= current * 1.25]
        if above_sup:
            p, t = min(above_sup, key=lambda x: (x[0] - current) / (1.0 + 0.25 * t))
            levels.append({
                "price": p,
                "kind": "broken_support",
                "label": "직전 지지(이탈)",
                "color": "#16a085",
                "touches": t,
            })

    # 강력 지지: 현재가 아래·현재 지지와 구분되는 레벨 (터치 수 우선)
    strong_cands = [
        (p, t) for p, t in below
        if near_support is None or abs(p - near_support[0]) / max(current, 1e-9) > cluster_pct
    ]
    # 기간/60일 저점이 현재 지지보다 충분히 깊으면 강력 지지 후보
    if near_support is not None:
        for p, t in ((period_low, 2), (roll60_low, 1)):
            if p < near_support[0] * (1.0 - cluster_pct) and p < current * 0.998:
                strong_cands.append((p, t))
    if strong_cands:
        p, t = max(strong_cands, key=lambda x: (x[1], x[0]))
        levels.append({
            "price": p,
            "kind": "strong_support",
            "label": "강력 지지",
            "color": STRONG_SUPPORT_COLOR,
            "touches": t,
        })
    elif near_support is not None and abs(near_support[0] - period_low) / max(current, 1e-9) <= cluster_pct:
        # 현재 지지=기간 저점만 있을 때: 직전 피벗 저점(이탈)을 보조 지지 참고선으로 표시하지 않고
        # 기간 저점을 강력 지지로도 강조하지 않음 — 단일 '현재 지지'로 충분
        pass

    # ---------- 최근 지지/저항 (최신성 보조 레이어) ----------
    # 최근 N봉: 20~40, 또는 기간의 1/3
    # 선택: 시간상 가장 최근 피벗 우선(구조선과 1.5% 이내면 스킵·다음 후보)
    # 폴백: 초단기(10·5봉) 고/저 — 신저점·급락에서 피벗이 구조선과 겹칠 때
    n = len(df)
    lookback = int(min(40, max(20, n // 3)))
    i0 = max(0, n - lookback)
    recent_hi = high[i0:]
    recent_lo = low[i0:]
    r_hi_local = _find_pivots(recent_hi, max(2, pivot_span), "high")
    r_lo_local = _find_pivots(recent_lo, max(2, pivot_span), "low")
    # (price, abs_index) — 최신성 = 큰 abs_index 우선
    r_hi_pts = [(float(recent_hi[i]), i0 + int(i)) for i in r_hi_local]
    r_lo_pts = [(float(recent_lo[i]), i0 + int(i)) for i in r_lo_local]

    def _too_close_to_existing(price: float, merge_pct: float = 0.015) -> bool:
        for lv in levels:
            if abs(price - float(lv["price"])) / max(current, 1e-9) <= merge_pct:
                return True
        return False

    def _pick_recent(
        pts: List[Tuple[float, int]],
        *,
        above: bool,
    ) -> Optional[Tuple[float, int]]:
        """
        최신성 우선:
          1) 시간상 가장 최근 피벗 (구조선과 1.5% 이내면 스킵)
          2) 초단기 롤링 고/저 (5·10봉, 완성봉)
          3) 더 오래된 피벗
        """
        side = [
            (p, idx) for p, idx in pts
            if (p > current * 1.002 if above else p < current * 0.998)
        ]
        side.sort(key=lambda x: x[1], reverse=True)

        # 1) 최신 피벗 1개만 우선 시도
        if side and not _too_close_to_existing(side[0][0]):
            return side[0]

        # 2) 초단기 롤링 (완성봉만: 마지막 봉 제외)
        #    구조선과 겹친 뒤엔 더 짧은 창(3→5→10)으로 가까운 되돌림 고/저 탐색
        for win in (3, 5, 10):
            j0 = max(0, n - 1 - win)
            j1 = max(j0, n - 1)
            if j1 <= j0:
                continue
            if above:
                p = float(np.max(high[j0:j1]))
                if p > current * 1.002 and not _too_close_to_existing(p):
                    return p, j1 - 1
            else:
                p = float(np.min(low[j0:j1]))
                if p < current * 0.998 and not _too_close_to_existing(p):
                    return p, j1 - 1

        # 3) 나머지 피벗 (최신→과거)
        for p, idx in side[1:]:
            if not _too_close_to_existing(p):
                return p, idx
        return None

    picked_res = _pick_recent(r_hi_pts, above=True)
    if picked_res is not None:
        rp, _ = picked_res
        levels.append({
            "price": rp,
            "kind": "recent_resistance",
            "label": "최근저항",
            "color": "#e67e22",
            "touches": 1,
            "lookback": lookback,
        })

    picked_sup = _pick_recent(r_lo_pts, above=False)
    if picked_sup is not None:
        sp, _ = picked_sup
        levels.append({
            "price": sp,
            "kind": "recent_support",
            "label": "최근지지",
            "color": "#1abc9c",
            "touches": 1,
            "lookback": lookback,
        })

    return levels


def compute_trendlines(
    ohlcv: pd.DataFrame,
    *,
    pivot_span: int = 3,
    min_pivot_gap: int = 8,
    max_break_bars: int = 3,
) -> List[Dict[str, Any]]:
    """
    터치 검증 기반 피벗 추세선.

    모든 피벗 쌍을 후보로 두고 다음 조건을 만족하는 선 중 최고점을 선택:
      - 앵커 구간(첫~둘째 피벗) 사이에서 가격이 선을 침범하지 않음
      - 둘째 피벗 이후 침범은 max_break_bars 봉까지 허용 (최근 돌파로 표시)
      - 피벗 터치 2회 이상 (터치 수 우선, 최신성 차선으로 점수화)
      - 거의 수평인 선은 수평 지지/저항과 중복이므로 제외

    Returns:
        [{x0, y0, x1, y1, kind, touches, slope, price_now, broken}]
        kind: 'resistance' | 'support'
    """
    df = _normalize(ohlcv)
    high = df["High"].astype(float).values
    low = df["Low"].astype(float).values
    n = len(df)
    xs = np.array([mdates.date2num(pd.Timestamp(d).to_pydatetime()) for d in df.index])
    tol = 0.3 * float(np.mean(high - low))

    def _best_line(vals: np.ndarray, pivots: List[int], side: str) -> Optional[Dict[str, Any]]:
        best: Optional[Dict[str, Any]] = None
        for a in range(len(pivots)):
            for b in range(a + 1, len(pivots)):
                i, j = pivots[a], pivots[b]
                if j - i < min_pivot_gap:
                    continue
                slope = (vals[j] - vals[i]) / max(xs[j] - xs[i], 1e-9)
                # 거의 수평이면 수평 S/R 라인과 중복
                if abs(slope) * (xs[-1] - xs[i]) < tol * 1.5:
                    continue
                line_vals = vals[i] + slope * (xs[i:] - xs[i])
                seg = vals[i:]
                if side == "res":
                    viol = seg > line_vals + tol
                else:
                    viol = seg < line_vals - tol
                # 앵커 구간 내 침범 불가
                if viol[: j - i + 1].any():
                    continue
                # 둘째 피벗 이후 침범은 제한적으로 허용 (돌파)
                break_bars = int(viol[j - i + 1:].sum())
                if break_bars > max_break_bars:
                    continue
                touches = sum(
                    1 for p in pivots
                    if p >= i and abs(vals[p] - (vals[i] + slope * (xs[p] - xs[i]))) <= tol
                )
                if touches < 2:
                    continue
                score = touches * 1000 + j
                if best is None or score > best["_score"]:
                    best = {
                        "x0": float(xs[i]),
                        "y0": float(vals[i]),
                        "x1": float(xs[-1]),
                        "y1": float(vals[i] + slope * (xs[-1] - xs[i])),
                        "kind": "resistance" if side == "res" else "support",
                        "touches": touches,
                        "slope": float(slope),
                        "price_now": float(vals[i] + slope * (xs[-1] - xs[i])),
                        "broken": break_bars > 0,
                        "_score": score,
                    }
        if best is not None:
            best.pop("_score", None)
        return best

    hi_idx = _find_pivots(high, pivot_span, "high")
    lo_idx = _find_pivots(low, pivot_span, "low")

    lines: List[Dict[str, Any]] = []
    res = _best_line(high, hi_idx, "res")
    if res:
        lines.append(res)
    sup = _best_line(low, lo_idx, "sup")
    if sup:
        lines.append(sup)
    return lines


def _rsi14(close: pd.Series, period: int = 14) -> Optional[float]:
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    v = rsi.iloc[-1]
    return round(float(v), 1) if pd.notna(v) else None


def _pattern_slot_key(ev: Dict[str, Any]) -> str:
    """평가·점수 저장 키. FVG는 상승/하락을 분리한다."""
    pid = ev.get("pattern_id") or ""
    if pid == "fvg_gap":
        return "fvg_gap:bear" if (ev.get("direction") or "bull") == "bear" else "fvg_gap:bull"
    return str(pid)


def _fvg_is_bear(ev: Dict[str, Any]) -> bool:
    return (ev.get("direction") or "bull") == "bear"


def _fvg_opposite_pair(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """상승 FVG ↔ 하락 FVG 쌍이면 True (시간 근접 dedup 예외)."""
    if a.get("pattern_id") != "fvg_gap" or b.get("pattern_id") != "fvg_gap":
        return False
    return _fvg_is_bear(a) != _fvg_is_bear(b)


def _fvg_rank_key(ev: Dict[str, Any], current: Optional[float]) -> Tuple:
    """
    방향별 FVG 우선순위: 미무효 > 미달성 > 최신.
    current 없으면 end 시각만 사용.
    """
    end = pd.Timestamp(ev.get("end") or ev.get("start"))
    if current is None:
        return (True, True, end)
    gl = ev.get("gap_low")
    gh = ev.get("gap_high")
    if gl is None or gh is None:
        return (False, False, end)
    gap_low, gap_high = float(gl), float(gh)
    if gap_low > gap_high:
        gap_low, gap_high = gap_high, gap_low
    height = gap_high - gap_low
    bear = _fvg_is_bear(ev)
    if not bear:
        invalidated = current < gap_low
        target = gap_high + height
        achieved = current >= target
    else:
        invalidated = current > gap_high
        target = gap_low - height
        achieved = current <= target
    return (not invalidated, not achieved, end)


def _select_top_patterns(
    events: List[Dict[str, Any]],
    *,
    min_confidence: float,
    max_patterns: int,
    dedup_days: int = 7,
    current: Optional[float] = None,
) -> List[Dict[str, Any]]:
    cands = [
        ev for ev in events
        if float(ev.get("confidence", 0)) >= min_confidence and ev.get("pattern_id") in ID_TO_ROW
    ]

    # 패턴 id별 최신 1건. FVG만 상승/하락 각 1건(미무효·미달성 우선).
    best_by_key: Dict[str, Dict[str, Any]] = {}
    for ev in cands:
        key = _pattern_slot_key(ev)
        prev = best_by_key.get(key)
        if prev is None:
            best_by_key[key] = ev
            continue
        if ev.get("pattern_id") == "fvg_gap":
            if _fvg_rank_key(ev, current) > _fvg_rank_key(prev, current):
                best_by_key[key] = ev
        else:
            end = pd.Timestamp(ev.get("end") or ev.get("start"))
            prev_end = pd.Timestamp(prev.get("end") or prev.get("start"))
            if end > prev_end:
                best_by_key[key] = ev
    cands = list(best_by_key.values())

    cands.sort(
        key=lambda ev: (
            float(ev.get("confidence", 0)),
            pd.Timestamp(ev.get("end") or ev.get("start")),
        ),
        reverse=True,
    )

    # 시간축 근접 중복 제거 (신뢰도 높은 것 우선 · 상승/하락 FVG 쌍·FVG 슬롯은 예외)
    selected: List[Dict[str, Any]] = []
    for ev in cands:
        center = pd.Timestamp(ev.get("end") or ev.get("start"))
        too_close = False
        for s in selected:
            days = abs((center - pd.Timestamp(s.get("end") or s.get("start"))).days)
            if days >= dedup_days:
                continue
            if _fvg_opposite_pair(ev, s):
                continue
            # FVG는 상승/하락 각 1건 확보 — 타 패턴과의 근접으로 탈락시키지 않음
            if ev.get("pattern_id") == "fvg_gap":
                continue
            too_close = True
            break
        if too_close:
            continue
        selected.append(ev)
        if len(selected) >= max_patterns:
            break
    return selected


BULL_PATTERN_IDS = {
    "triple_bottom_w", "bull_flag", "asc_triangle", "post_box_bull",
    "bull_hammer", "bull_engulf",
}
BEAR_PATTERN_IDS = {
    "double_top_m", "bear_flag", "bear_diamond",
    "bear_hanging_man", "bear_engulf",
}

# 메인 차트 패턴 번호 마커 ↔ 하단 미니 차트 연동용
PATTERN_NUM_MARKS = "❶❷❸❹❺"

# 패턴 id → (방향 함의, 방향 부호: +1 강세 / -1 약세 / 0 중립)
PATTERN_IMPLICATION: Dict[str, Tuple[str, int]] = {
    "double_top_m": ("하락 반전형", -1),
    "bear_flag": ("하락 지속형", -1),
    "bear_diamond": ("하락 반전형", -1),
    "bear_hanging_man": ("약세 반전형", -1),
    "bear_engulf": ("약세 반전형", -1),
    "triple_bottom_w": ("상승 반전형", 1),
    "bull_flag": ("상승 지속형", 1),
    "asc_triangle": ("상승 지속형", 1),
    "post_box_bull": ("상승 지속형", 1),
    "bull_hammer": ("강세 반전형", 1),
    "bull_engulf": ("강세 반전형", 1),
    "range_box": ("중립·횡보", 0),
    "fvg_gap": ("갭·중립", 0),
    "doji": ("중립", 0),
}


def _score_scenarios(
    *,
    current: float,
    adx_res: Dict[str, Any],
    disparity_value: Optional[float],
    rsi_val: Optional[float],
    trendlines: List[Dict[str, Any]],
    lv_by_label: Dict[str, Dict[str, Any]],
    sma20_slope: Optional[float],
    top_patterns: List[Dict[str, Any]],
    tb_eval: Optional[Dict[str, Any]] = None,
    dt_eval: Optional[Dict[str, Any]] = None,
    aux_ind: Optional[Dict[str, Any]] = None,
    surge_state: Optional[Dict[str, Any]] = None,
    surge_setup: Optional[Dict[str, Any]] = None,
    plunge_state: Optional[Dict[str, Any]] = None,
    plunge_setup: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    시나리오별 가능성 점수 (0~100, 지표 근거 기반 상대 척도 — 확률 아님).

    sc1=지지 이탈, sc2=추세 하락, sc3=저항 돌파, sc4=추세 상승,
    sc5=하락 후 반등, sc6=상승 후 조정.
    각 신호가 해당 시나리오 점수에 가중치를 더하고 근거 문구를 남긴다.
    """
    scores: Dict[str, Dict[str, Any]] = {
        k: {"score": 40.0, "reasons": []}
        for k in ("sc1", "sc2", "sc3", "sc4", "sc5", "sc6")
    }

    def add(keys: Tuple[str, ...], w: float, reason: str) -> None:
        for k in keys:
            scores[k]["score"] += w
            if w > 0:
                scores[k]["reasons"].append((w, reason))

    # 1) ADX 추세 방향·강도 (지속 ①②③④ 중심 · 반전은 약하게)
    direction = str(adx_res.get("direction") or "")
    strength = str(adx_res.get("strength") or "")
    adx_v = adx_res.get("adx")
    strong = "강" in strength
    if direction == "하락":
        w = 16 if strong else 12
        add(("sc1", "sc2"), w, f"ADX 하락 추세({adx_v})")
        add(("sc3", "sc4"), -6, "")
        add(("sc5",), 4, f"하락국면·반등대기({adx_v})")  # 캡출레이션 반등 토양
        add(("sc6",), -4, "")
    elif direction == "상승":
        w = 16 if strong else 12
        add(("sc3", "sc4"), w, f"ADX 상승 추세({adx_v})")
        add(("sc1", "sc2"), -6, "")
        add(("sc6",), 4, f"상승국면·조정대기({adx_v})")
        add(("sc5",), -4, "")
    elif direction == "횡보":
        add(("sc2", "sc4"), -6, "")  # 추세 지속 시나리오 신뢰 저하
        add(("sc5", "sc6"), 3, "횡보·평균회귀")

    # 2) DI 우세
    plus_di = adx_res.get("plus_di")
    minus_di = adx_res.get("minus_di")
    if plus_di is not None and minus_di is not None:
        gap = float(minus_di) - float(plus_di)
        if gap >= 5:
            add(("sc1", "sc2"), 6, f"-DI 우세({minus_di:.0f}>{plus_di:.0f})")
            add(("sc5",), 3, "-DI우세·반등여지")
        elif gap <= -5:
            add(("sc3", "sc4"), 6, f"+DI 우세({plus_di:.0f}>{minus_di:.0f})")
            add(("sc6",), 3, "+DI우세·조정여지")

    # 3) RSI — 극단은 반전(⑤⑥) 우선, 지속은 약하게
    if rsi_val is not None:
        if rsi_val >= 70:
            add(("sc6",), 10, f"RSI 과매수·조정({rsi_val})")
            add(("sc1", "sc2"), 3, f"RSI 과매수({rsi_val})")
            add(("sc3",), -4, "")
        elif rsi_val <= 30:
            add(("sc5",), 10, f"RSI 과매도·반등({rsi_val})")
            add(("sc3", "sc4"), 3, f"RSI 과매도({rsi_val})")
            add(("sc1",), -4, "")

    # 4) 이격도 — 동일하게 반전 우선
    if disparity_value is not None:
        if disparity_value >= 105:
            add(("sc6",), 8, f"이격도 과열·조정({disparity_value:.1f})")
            add(("sc1", "sc2"), 3, f"이격도 과열({disparity_value:.1f})")
            add(("sc3", "sc4"), -4, "")
        elif disparity_value <= 95:
            add(("sc5",), 8, f"이격도 침체·반등({disparity_value:.1f})")
            add(("sc3", "sc4"), 3, f"이격도 침체({disparity_value:.1f})")
            add(("sc1", "sc2"), -4, "")

    # 5) 추세선 상태 (기울기·역할·돌파 여부) — 지속 시나리오
    for tl in trendlines:
        touches = int(tl.get("touches", 0))
        broken = bool(tl.get("broken"))
        up = tl.get("slope", 0) > 0
        if tl["kind"] == "resistance":
            if broken:
                add(("sc3",), 6, "저항 추세선 상향 돌파")
                add(("sc5",), 4, "저항선 돌파·반등확인")
            elif up:
                add(("sc4",), 6, f"상승 저항 추세선 유효({touches}터치)")
            else:
                add(("sc2",), 8, f"하락 저항 추세선 유효({touches}터치)")
                add(("sc6",), 3, "하락저항·조정압력")
        else:  # support
            if broken:
                add(("sc1",), 6, "지지 추세선 하향 이탈")
                add(("sc6",), 3, "지지이탈·조정심화")
            elif up:
                add(("sc4",), 8, f"상승 지지 추세선 유효({touches}터치)")
                add(("sc5",), 3, "상승지지·반등받침")
            else:
                add(("sc2",), 6, f"하락 지지 추세선({touches}터치)")

    # 6) SMA20 기울기
    if sma20_slope is not None:
        if sma20_slope > 0:
            add(("sc3", "sc4"), 5, "SMA20 상승 기울기")
            add(("sc6",), 2, "SMA20상승·과열시조정")
        elif sma20_slope < 0:
            add(("sc1", "sc2"), 5, "SMA20 하락 기울기")
            add(("sc5",), 2, "SMA20하락·침체시반등")

    # 7) 지지/저항 근접도
    near_sup = lv_by_label.get("현재 지지")
    if near_sup and 0 <= (current - near_sup["price"]) / current <= 0.015:
        add(("sc1",), 5, "지지선 근접")
        add(("sc5",), 5, "지지근접·반등자리")
    near_res = lv_by_label.get("단기 저항")
    if near_res and 0 <= (near_res["price"] - current) / current <= 0.015:
        add(("sc3",), 5, "저항선 근접")
        add(("sc6",), 5, "저항근접·조정자리")
    recent_sup = lv_by_label.get("최근지지")
    if recent_sup and 0 <= (current - recent_sup["price"]) / current <= 0.015:
        add(("sc1",), 3, "최근지지 근접")
        add(("sc5",), 3, "최근지지·반등")
    recent_res = lv_by_label.get("최근저항")
    if recent_res and 0 <= (recent_res["price"] - current) / current <= 0.015:
        add(("sc3",), 3, "최근저항 근접")
        add(("sc6",), 3, "최근저항·조정")

    # 8) 최근 패턴 방향 (최신 2건)
    recent = sorted(
        top_patterns,
        key=lambda ev: pd.Timestamp(ev.get("end") or ev.get("start")),
        reverse=True,
    )[:2]
    for ev in recent:
        pid = ev.get("pattern_id")
        label = CHART_PATTERN_CRITERIA[ID_TO_ROW[pid]]["label"] if pid in ID_TO_ROW else pid
        if pid == "fvg_gap":
            if _fvg_is_bear(ev):
                add(("sc1", "sc2"), 5, "최근 하락FVG")
                add(("sc5",), 3, "하락FVG·반등갭")
            else:
                add(("sc3", "sc4"), 5, "최근 상승FVG")
                add(("sc6",), 3, "상승FVG·조정갭")
        elif pid in BULL_PATTERN_IDS:
            add(("sc3", "sc4"), 5, f"최근 강세 패턴({label})")
            add(("sc5",), 4, f"강세패턴·반등({label})")
        elif pid in BEAR_PATTERN_IDS:
            add(("sc1", "sc2"), 5, f"최근 약세 패턴({label})")
            add(("sc6",), 4, f"약세패턴·조정({label})")

    # 9) 역삼중창 상세 평가
    if tb_eval is not None:
        if tb_eval.get("breakout"):
            add(("sc3", "sc4"), 6, "역삼중창 넥라인 돌파")
            add(("sc5",), 5, "역삼중창·반등확인")
        elif tb_eval.get("vol_ok_n", 0) >= 2 and tb_eval.get("slope_cls") != "우하향":
            add(("sc3", "sc4"), 5, "역삼중창 거래량 확인")
            add(("sc5",), 3, "역삼중창 형성")

    # 10) 쌍봉(M) 상세 평가
    if dt_eval is not None:
        if dt_eval.get("breakout"):
            add(("sc1", "sc2"), 6, "쌍봉 목선 이탈")
            add(("sc6",), 5, "쌍봉·조정확인")
        elif dt_eval.get("vol_ok_n", 0) >= 1:
            add(("sc1", "sc2"), 4, "쌍봉 형성(고점 저항)")
            add(("sc6",), 4, "쌍봉·조정대기")

    # 11) BB / Supertrend / SMA200
    if aux_ind:
        bb_state = aux_ind.get("bb_state")
        if bb_state == "하단이탈":
            add(("sc5",), 8, "BB하단이탈·반등")
            add(("sc1", "sc2"), 2, "BB하단이탈")
        elif bb_state == "상단돌파":
            add(("sc6",), 8, "BB상단돌파·조정")
            add(("sc3", "sc4"), 2, "BB상단돌파")

        st_dir = aux_ind.get("st_dir")
        if st_dir == "down":
            add(("sc1", "sc2"), 6, "ST하락(저항)")
            add(("sc5",), 3, "ST하락중·반등관문")
        elif st_dir == "up":
            add(("sc3", "sc4"), 6, "ST상승(지지)")
            add(("sc6",), 3, "ST상승중·조정관문")

        sma_st = aux_ind.get("sma200_state")
        if sma_st == "하회":
            add(("sc1", "sc2"), 5, "SMA200하회")
            add(("sc5",), 2, "장기하회·반등한계")
        elif sma_st == "상회":
            add(("sc3", "sc4"), 5, "SMA200상회")
            add(("sc6",), 2, "장기상회·조정여지")

    # 12) 급등 패턴 (현재가 스냅샷)
    if surge_state:
        st_name = surge_state.get("state")
        if st_name == "급등진행":
            add(("sc3", "sc4"), 5, "급등패턴")
            add(("sc5",), 3, "급등·모멘텀")
        elif st_name == "급등과열":
            add(("sc6",), 8, "급등과열·조정")
            add(("sc3",), 2, "급등유지")
        elif st_name == "급등약함":
            add(("sc3", "sc4"), 2, "급등약함")

    # 13) 급등 직전 셋업 (A/B/점화임박)
    if surge_setup and surge_setup.get("active"):
        kind = surge_setup.get("kind")
        st_setup = surge_setup.get("state")
        if st_setup == "점화임박":
            add(("sc3", "sc4"), 8, "급등점화임박")
        elif kind == "A":
            add(("sc3", "sc4"), 6, "급등셋업A")
        elif kind == "B":
            add(("sc3", "sc4"), 5, "급등셋업B")

    # 14) 급락 패턴 (현재가 스냅샷)
    if plunge_state:
        pl_name = plunge_state.get("state")
        if pl_name == "급락진행":
            add(("sc1", "sc2"), 5, "급락패턴")
            add(("sc6",), 3, "급락·모멘텀")
        elif pl_name == "급락과매도":
            add(("sc5",), 8, "급락과매도·반등")
            add(("sc1",), 2, "급락유지")
        elif pl_name == "급락약함":
            add(("sc1", "sc2"), 2, "급락약함")

    # 15) 급락 직전 셋업 (A/B/붕괴임박)
    if plunge_setup and plunge_setup.get("active"):
        kind = plunge_setup.get("kind")
        pl_setup = plunge_setup.get("state")
        if pl_setup == "붕괴임박":
            add(("sc1", "sc2"), 8, "급락붕괴임박")
        elif kind == "A":
            add(("sc1", "sc2"), 6, "급락셋업A")
        elif kind == "B":
            add(("sc1", "sc2"), 5, "급락셋업B")

    for k, v in scores.items():
        v["score"] = int(round(min(90.0, max(5.0, v["score"]))))
        reasons = sorted((r for r in v["reasons"] if r[1]), key=lambda x: x[0], reverse=True)
        seen: List[str] = []
        for _, txt in reasons:
            if txt not in seen:
                seen.append(txt)
        v["reasons"] = seen[:3]
    return scores


def _score_gauge(score: int) -> str:
    """0~100 점수 → 5칸 게이지 문자열."""
    filled = max(0, min(5, int(round(score / 20.0))))
    return "■" * filled + "□" * (5 - filled)


def _pick_scenario_top3(entries: List[Dict[str, Any]], n: int = 3) -> List[str]:
    """점수 내림차순 TopN 키. 무효/달성(active=False)은 Top3 제외."""
    ranked = sorted(
        [e for e in entries if e.get("active", True)],
        key=lambda e: (-int(e.get("score") or 0), str(e.get("num") or ""), str(e.get("key") or "")),
    )
    return [str(e["key"]) for e in ranked[:n]]


def _build_top3_scenario_panel(entries: List[Dict[str, Any]], top3_keys: List[str]) -> List[str]:
    """시나리오 전체 목록 (인사이트 패널용 · Top3 문구 없음)."""
    if not entries:
        return []
    # top3_keys는 메인 차트 ★ 식별용으로만 유지(패널 미표시)
    _ = top3_keys
    ordered = sorted(
        entries,
        key=lambda e: (0 if e.get("active", True) else 1, str(e.get("num") or ""), str(e.get("key") or "")),
    )
    lines: List[str] = []
    for e in ordered:
        tgt = e.get("target_txt") or ""
        lines.append(
            f"  {e['num']} {e.get('title', '')} → {tgt}  "
            f"{_score_gauge(int(e['score']))} {int(e['score'])}점"
        )
        for extra in e.get("detail_lines") or []:
            lines.append(extra)
        reasons = e.get("reasons") or []
        if reasons:
            lines.append("     근거: " + " · ".join(reasons[:2]))
    return lines


def _proximity_state(current: float, level: float, *, bullish_break: bool) -> str:
    """레벨까지 거리 상태 문구."""
    if level <= 0 or current <= 0:
        return "대기"
    pct = (level / current - 1.0) * 100.0
    # 돌파 방향 기준: 상승돌파는 레벨이 위(+), 하락이탈은 레벨이 아래(-)
    dist = abs(pct)
    if bullish_break:
        if current >= level:
            return "돌파권"
        if dist <= 1.5:
            return "근접"
        if dist <= 4.0:
            return "접근"
        return "대기"
    if current <= level:
        return "이탈권"
    if dist <= 1.5:
        return "근접"
    if dist <= 4.0:
        return "접근"
    return "대기"


def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def _atr_value(df: pd.DataFrame, period: int = 14) -> float:
    atr = _atr_series(df, period).iloc[-1]
    if pd.notna(atr) and float(atr) > 0:
        return float(atr)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    return float(max((high - low).tail(period).mean(), 1e-9))


def _compute_aux_trend_indicators(
    df: pd.DataFrame,
    *,
    bb_window: int = 20,
    bb_std: float = 2.0,
    st_period: int = 10,
    st_mult: float = 3.0,
    sma_long: int = 200,
) -> Dict[str, Any]:
    """
    보조 지표 스냅샷 + 시계열 (패널/점수/메인 오버레이용).

    Returns keys:
      bb_*, st_line/st_dir/st_role, sma200*,
      st_series, st_dir_series, sma200_series (plot용, 없으면 None)
    """
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    current = float(close.iloc[-1])
    out: Dict[str, Any] = {
        "bb_mid": None,
        "bb_upper": None,
        "bb_lower": None,
        "bb_width_pct": None,
        "bb_state": None,
        "bb_mid_series": None,
        "bb_upper_series": None,
        "bb_lower_series": None,
        "st_line": None,
        "st_dir": None,
        "st_role": None,
        "st_series": None,
        "st_dir_series": None,
        "sma200": None,
        "sma200_state": None,
        "sma200_series": None,
        "sma200_window": None,
    }

    # --- Bollinger (20, 2) ---
    if len(close) >= bb_window:
        mid = close.rolling(bb_window).mean()
        std = close.rolling(bb_window).std()
        upper = mid + bb_std * std
        lower = mid - bb_std * std
        m, u, lo = float(mid.iloc[-1]), float(upper.iloc[-1]), float(lower.iloc[-1])
        if all(np.isfinite(x) for x in (m, u, lo)) and m > 0:
            out["bb_mid"] = m
            out["bb_upper"] = u
            out["bb_lower"] = lo
            out["bb_width_pct"] = (u - lo) / m * 100.0
            out["bb_mid_series"] = mid
            out["bb_upper_series"] = upper
            out["bb_lower_series"] = lower
            if current < lo:
                out["bb_state"] = "하단이탈"
            elif current > u:
                out["bb_state"] = "상단돌파"
            else:
                out["bb_state"] = "밴드내"

    # --- Supertrend (ATR10 × 3) — numpy 루프 ---
    n = len(df)
    if n >= st_period + 2:
        atr = _atr_series(df, st_period).to_numpy(dtype=float)
        c = close.to_numpy(dtype=float)
        h = high.to_numpy(dtype=float)
        l = low.to_numpy(dtype=float)
        hl2 = (h + l) * 0.5
        basic_ub = hl2 + st_mult * atr
        basic_lb = hl2 - st_mult * atr
        final_ub = basic_ub.copy()
        final_lb = basic_lb.copy()
        for i in range(1, n):
            if not (np.isfinite(basic_ub[i]) and np.isfinite(basic_lb[i])):
                continue
            prev_ub, prev_lb = final_ub[i - 1], final_lb[i - 1]
            if np.isfinite(prev_ub) and (basic_ub[i] < prev_ub or c[i - 1] > prev_ub):
                final_ub[i] = basic_ub[i]
            elif np.isfinite(prev_ub):
                final_ub[i] = prev_ub
            if np.isfinite(prev_lb) and (basic_lb[i] > prev_lb or c[i - 1] < prev_lb):
                final_lb[i] = basic_lb[i]
            elif np.isfinite(prev_lb):
                final_lb[i] = prev_lb

        st_arr = np.full(n, np.nan, dtype=float)
        dir_arr = np.ones(n, dtype=int)
        for i in range(1, n):
            if not (np.isfinite(final_ub[i]) and np.isfinite(final_lb[i])):
                dir_arr[i] = dir_arr[i - 1]
                st_arr[i] = st_arr[i - 1]
                continue
            if dir_arr[i - 1] <= 0:
                if c[i] > final_ub[i]:
                    dir_arr[i] = 1
                    st_arr[i] = final_lb[i]
                else:
                    dir_arr[i] = -1
                    st_arr[i] = final_ub[i]
            else:
                if c[i] < final_lb[i]:
                    dir_arr[i] = -1
                    st_arr[i] = final_ub[i]
                else:
                    dir_arr[i] = 1
                    st_arr[i] = final_lb[i]

        st = pd.Series(st_arr, index=df.index)
        direction = pd.Series(dir_arr, index=df.index)
        out["st_series"] = st
        out["st_dir_series"] = direction
        if np.isfinite(st_arr[-1]) and st_arr[-1] > 0:
            out["st_line"] = float(st_arr[-1])
            out["st_dir"] = "up" if dir_arr[-1] > 0 else "down"
            out["st_role"] = "지지" if dir_arr[-1] > 0 else "저항"

    # --- SMA200 ---
    if len(close) >= max(50, sma_long // 4):
        win = min(sma_long, len(close))
        sma_s = close.rolling(win, min_periods=max(50, win // 2)).mean()
        sma = float(sma_s.iloc[-1])
        if np.isfinite(sma) and sma > 0:
            out["sma200"] = sma
            out["sma200_window"] = win
            out["sma200_state"] = "상회" if current >= sma else "하회"
            out["sma200_series"] = sma_s

    return out


def _draw_aux_trend_overlays(ax, df: pd.DataFrame, aux: Dict[str, Any]) -> None:
    """2단계: SMA200·Supertrend 메인 오버레이 (BB는 그리지 않음)."""
    st = aux.get("st_series")
    st_dir = aux.get("st_dir_series")
    if st is not None and st_dir is not None:
        up = st.where(st_dir > 0)
        down = st.where(st_dir < 0)
        ax.plot(
            df.index, up, color="#1e8449", linewidth=2.2, alpha=0.92,
            label="ST↑", zorder=7,
        )
        ax.plot(
            df.index, down, color="#c0392b", linewidth=2.2, alpha=0.92,
            label="ST↓", zorder=7,
        )

    sma_s = aux.get("sma200_series")
    win = int(aux.get("sma200_window") or 0)
    # 약 6mo(≥100봉) 이상일 때만 메인에 표시 (짧으면 패널만)
    if sma_s is not None and win >= 100:
        lbl = "SMA200" if win >= 200 else f"SMA{win}"
        ax.plot(
            df.index, sma_s, color="#7f8c8d", linewidth=1.4, alpha=0.90,
            label=lbl, zorder=6,
        )


def _bb_cross_events(
    close: pd.Series,
    upper: pd.Series,
    lower: pd.Series,
    *,
    lookback: int = 40,
) -> List[Tuple[int, str]]:
    """최근 lookback 내 BB 상단돌파/하단이탈 최초 교차 인덱스. ('up'|'down')."""
    n = len(close)
    i0 = max(1, n - lookback)
    c = close.to_numpy(dtype=float)
    u = upper.to_numpy(dtype=float)
    lo = lower.to_numpy(dtype=float)
    events: List[Tuple[int, str]] = []
    for i in range(i0, n):
        if not (np.isfinite(c[i]) and np.isfinite(u[i]) and np.isfinite(lo[i])):
            continue
        prev_ok = np.isfinite(c[i - 1]) and np.isfinite(u[i - 1]) and np.isfinite(lo[i - 1])
        if c[i] < lo[i] and (not prev_ok or c[i - 1] >= lo[i - 1]):
            events.append((i, "down"))
        elif c[i] > u[i] and (not prev_ok or c[i - 1] <= u[i - 1]):
            events.append((i, "up"))
    return events


def _draw_bb_visuals(
    ax,
    df: pd.DataFrame,
    aux: Dict[str, Any],
    *,
    show_bb: bool,
    fmt_price,
) -> None:
    """
    3단계 볼린저 시각화.
    - 기본: 이탈/돌파 이벤트 태그 (+ 현재 밴드 수평 점선)
    - show_bb=True: BB 상·하단(·중간) 시계열 선
    """
    upper_s = aux.get("bb_upper_series")
    lower_s = aux.get("bb_lower_series")
    mid_s = aux.get("bb_mid_series")
    if upper_s is None or lower_s is None:
        return

    close = df["Close"].astype(float)
    bb_color = "#d35400"

    if show_bb:
        ax.plot(
            df.index, upper_s, color=bb_color, linewidth=0.75, linestyle=":",
            alpha=0.65, label="BB상단", zorder=5,
        )
        ax.plot(
            df.index, lower_s, color=bb_color, linewidth=0.75, linestyle=":",
            alpha=0.65, label="BB하단", zorder=5,
        )
        if mid_s is not None:
            ax.plot(
                df.index, mid_s, color="#e67e22", linewidth=0.55, linestyle=":",
                alpha=0.40, label="BB중간", zorder=5,
            )

    # 현재 밴드값 수평 점선 (만트라와 구분 · 얇게)
    bb_lo = aux.get("bb_lower")
    bb_up = aux.get("bb_upper")
    state = aux.get("bb_state")
    if bb_lo is not None and state == "하단이탈":
        ax.axhline(float(bb_lo), color=bb_color, linestyle=":", linewidth=1.0, alpha=0.75, zorder=8)
    if bb_up is not None and state == "상단돌파":
        ax.axhline(float(bb_up), color=bb_color, linestyle=":", linewidth=1.0, alpha=0.75, zorder=8)

    # 이벤트 태그 (최근 교차 최대 2개 + 현재 상태)
    events = _bb_cross_events(close, upper_s, lower_s, lookback=40)
    mark_idxs = {i for i, _ in events[-2:]}
    # 현재 밴드 밖이면 마지막 봉도 강조
    last_i = len(df) - 1
    if state in ("하단이탈", "상단돌파"):
        mark_idxs.add(last_i)

    low_s = df["Low"].astype(float)
    high_s = df["High"].astype(float)
    for i in sorted(mark_idxs):
        d = df.index[i]
        if state == "하단이탈" and i == last_i:
            kind = "down"
        elif state == "상단돌파" and i == last_i:
            kind = "up"
        else:
            kind = next((k for j, k in events if j == i), None)
            if kind is None:
                # 지속 이탈 구간: 종가 위치로 판별
                cv = float(close.iloc[i])
                if bb_lo is not None and cv < float(bb_lo):
                    kind = "down"
                elif bb_up is not None and cv > float(bb_up):
                    kind = "up"
                else:
                    continue
        if kind == "down":
            y = float(low_s.iloc[i])
            ax.annotate(
                "BB↓이탈",
                xy=(d, y),
                xytext=(0, -22),
                textcoords="offset points",
                fontsize=7.2,
                fontweight="bold",
                color=bb_color,
                ha="center",
                va="top",
                zorder=25,
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor=bb_color,
                    linewidth=0.7,
                    alpha=0.92,
                ),
                arrowprops=dict(arrowstyle="-", color=bb_color, lw=0.75, linestyle=":", alpha=0.7),
            )
            if i == last_i and bb_lo is not None:
                ax.annotate(
                    f"BB하단 {fmt_price(float(bb_lo))}",
                    xy=(d, float(bb_lo)),
                    xytext=(10, -4),
                    textcoords="offset points",
                    fontsize=6.8,
                    color=bb_color,
                    ha="left",
                    va="top",
                    zorder=25,
                    alpha=0.9,
                    bbox=dict(
                        boxstyle="round,pad=0.12",
                        facecolor="white",
                        edgecolor=bb_color,
                        linewidth=0.55,
                        alpha=0.88,
                    ),
                )
        else:
            y = float(high_s.iloc[i])
            ax.annotate(
                "BB↑돌파",
                xy=(d, y),
                xytext=(0, 20),
                textcoords="offset points",
                fontsize=7.2,
                fontweight="bold",
                color=bb_color,
                ha="center",
                va="bottom",
                zorder=25,
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor=bb_color,
                    linewidth=0.7,
                    alpha=0.92,
                ),
                arrowprops=dict(arrowstyle="-", color=bb_color, lw=0.75, linestyle=":", alpha=0.7),
            )
            if i == last_i and bb_up is not None:
                ax.annotate(
                    f"BB상단 {fmt_price(float(bb_up))}",
                    xy=(d, float(bb_up)),
                    xytext=(10, 4),
                    textcoords="offset points",
                    fontsize=6.8,
                    color=bb_color,
                    ha="left",
                    va="bottom",
                    zorder=25,
                    alpha=0.9,
                    bbox=dict(
                        boxstyle="round,pad=0.12",
                        facecolor="white",
                        edgecolor=bb_color,
                        linewidth=0.55,
                        alpha=0.88,
                    ),
                )


def _append_sc3_level(
    cands: List[Tuple[float, str]],
    price: Optional[float],
    label: str,
    *,
    current: float,
    merge_pct: float = 0.015,
) -> None:
    """③ 경로 후보 추가 (현재가 위 · 기존과 1.5% 이내면 스킵)."""
    if price is None:
        return
    p = float(price)
    if p <= current * 1.002:
        return
    for ep, _ in cands:
        if abs(p - ep) / max(current, 1e-9) <= merge_pct:
            return
    cands.append((p, label))


def _resolve_rebound_scenario(
    *,
    current: float,
    df: pd.DataFrame,
    lv_by_label: Dict[str, Any],
    aux_ind: Dict[str, Any],
    x_right: float,
    x_fut_mid: float,
    x_fut_end: float,
    peer_target: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    ⑤ 하락 후 반등 — 꺾인 경로: 현재 → 단기저점 → 반등목표.
    """
    if current <= 0:
        return None
    near_sup = lv_by_label.get("현재 지지")
    recent_sup = lv_by_label.get("최근지지")
    near_res = lv_by_label.get("단기 저항")
    recent_res = lv_by_label.get("최근저항")
    bb_lo = aux_ind.get("bb_lower")
    st_px = aux_ind.get("st_line")
    sma20 = float(df["Close"].astype(float).rolling(20).mean().iloc[-1])
    period_low = float(df["Low"].astype(float).min())

    dip_cands: List[Tuple[float, str]] = []
    for p, lb in (
        (bb_lo, "BB하단"),
        (recent_sup["price"] if recent_sup else None, "최근지지"),
        (near_sup["price"] if near_sup else None, "현재지지"),
        (period_low, "기간저점"),
    ):
        if p is None:
            continue
        pv = float(p)
        if pv < current * 0.998:
            dip_cands.append((pv, lb))
    if dip_cands:
        dip_px, dip_lb = max(dip_cands, key=lambda x: x[0])  # 현재 아래 최근접
    else:
        dip_px, dip_lb = current * 0.97, "가상저점"

    bounce_cands: List[Tuple[float, str]] = []
    for p, lb in (
        (recent_res["price"] if recent_res else None, "최근저항"),
        (near_res["price"] if near_res else None, "단기저항"),
        (st_px if aux_ind.get("st_dir") == "down" else None, "ST저항"),
        (sma20 if np.isfinite(sma20) else None, "SMA20"),
    ):
        if p is None:
            continue
        pv = float(p)
        if pv > current * 1.005 and pv > dip_px * 1.01:
            bounce_cands.append((pv, lb))
    if not bounce_cands:
        bounce_px, bounce_lb = current * 1.06, "반등목표"
    else:
        bounce_px, bounce_lb = min(bounce_cands, key=lambda x: x[0])

    # ③과 목표 너무 가까우면 소폭 이격
    if peer_target is not None and abs(bounce_px - float(peer_target)) / current < 0.015:
        bounce_px = float(peer_target) * 0.97
        if bounce_px <= current:
            bounce_px = current * 1.04
        bounce_lb = f"{bounce_lb}·이격"

    path_y = [current, float(dip_px), float(bounce_px)]
    path_x = [x_right, x_fut_mid, x_fut_end]
    return {
        "target": float(bounce_px),
        "path_x": path_x,
        "path_y": path_y,
        "waypoint_labels": ["현재", dip_lb, bounce_lb],
        "note": f"V경로 {dip_lb}→{bounce_lb}",
    }


def _resolve_pullback_scenario(
    *,
    current: float,
    df: pd.DataFrame,
    lv_by_label: Dict[str, Any],
    aux_ind: Dict[str, Any],
    x_right: float,
    x_fut_mid: float,
    x_fut_end: float,
    peer_target: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    ⑥ 상승 후 조정 — 꺾인 경로: 현재 → 단기고점 → 조정목표.
    """
    if current <= 0:
        return None
    near_sup = lv_by_label.get("현재 지지")
    recent_sup = lv_by_label.get("최근지지")
    near_res = lv_by_label.get("단기 저항")
    recent_res = lv_by_label.get("최근저항")
    bb_up = aux_ind.get("bb_upper")
    st_px = aux_ind.get("st_line")
    sma20 = float(df["Close"].astype(float).rolling(20).mean().iloc[-1])
    period_high = float(df["High"].astype(float).max())

    peak_cands: List[Tuple[float, str]] = []
    for p, lb in (
        (bb_up, "BB상단"),
        (recent_res["price"] if recent_res else None, "최근저항"),
        (near_res["price"] if near_res else None, "단기저항"),
        (period_high, "기간고점"),
    ):
        if p is None:
            continue
        pv = float(p)
        if pv > current * 1.002:
            peak_cands.append((pv, lb))
    if peak_cands:
        peak_px, peak_lb = min(peak_cands, key=lambda x: x[0])
    else:
        peak_px, peak_lb = current * 1.03, "가상고점"

    pull_cands: List[Tuple[float, str]] = []
    for p, lb in (
        (recent_sup["price"] if recent_sup else None, "최근지지"),
        (near_sup["price"] if near_sup else None, "현재지지"),
        (st_px if aux_ind.get("st_dir") == "up" else None, "ST지지"),
        (sma20 if np.isfinite(sma20) else None, "SMA20"),
    ):
        if p is None:
            continue
        pv = float(p)
        if pv < current * 0.995 and pv < peak_px * 0.99:
            pull_cands.append((pv, lb))
    if not pull_cands:
        pull_px, pull_lb = current * 0.94, "조정목표"
    else:
        pull_px, pull_lb = max(pull_cands, key=lambda x: x[0])

    if peer_target is not None and abs(pull_px - float(peer_target)) / current < 0.015:
        pull_px = float(peer_target) * 1.03
        if pull_px >= current:
            pull_px = current * 0.96
        pull_lb = f"{pull_lb}·이격"

    path_y = [current, float(peak_px), float(pull_px)]
    path_x = [x_right, x_fut_mid, x_fut_end]
    return {
        "target": float(pull_px),
        "path_x": path_x,
        "path_y": path_y,
        "waypoint_labels": ["현재", peak_lb, pull_lb],
        "note": f"Λ경로 {peak_lb}→{pull_lb}",
    }


def _peg_panel_lines(peg_view: Optional[Dict[str, Any]]) -> List[str]:
    """인사이트 패널용 PEG·재무 전략 줄."""
    if not peg_view:
        return []
    lines = list(peg_view.get("lines") or [])
    return lines[:6]


def _adx_insight_sentence(adx_res: Dict[str, Any]) -> str:
    """ADX/DMI 1문장 해석 (인사이트 확인용)."""
    adx = adx_res.get("adx")
    if adx is None:
        return "ADX 데이터 부족"
    try:
        adx_f = float(adx)
    except (TypeError, ValueError):
        return "ADX 데이터 부족"

    direction = str(adx_res.get("direction") or "")
    strength = str(adx_res.get("strength") or "")
    plus_di, minus_di = adx_res.get("plus_di"), adx_res.get("minus_di")
    di_bit = ""
    if plus_di is not None and minus_di is not None:
        gap = float(plus_di) - float(minus_di)
        if abs(gap) < 2:
            di_bit = "DI혼조"
        elif gap > 0:
            di_bit = f"+DI우세(갭{gap:+.0f})"
        else:
            di_bit = f"-DI우세(갭{gap:+.0f})"

    if adx_f < 20:
        base = "횡보권 · 돌파/이탈 확인 전 관망 비중"
        return f"{base} · {di_bit}" if di_bit else base
    if direction == "상승":
        base = f"상승 추세({strength}) · ③④ 가점"
        return f"{base} · {di_bit}" if di_bit else base
    if direction == "하락":
        base = f"하락 추세({strength}) · ①② 가점"
        return f"{base} · {di_bit}" if di_bit else base
    summary = adx_res.get("summary") or "국면 불명"
    return f"{summary} · {di_bit}" if di_bit else str(summary)


def _adx_panel_lines(adx_res: Optional[Dict[str, Any]]) -> List[str]:
    """인사이트 패널용 [ADX/DMI] 블록 — 수치·국면·해석 1문장."""
    if not adx_res or adx_res.get("adx") is None:
        return []
    adx = adx_res.get("adx")
    plus_di = adx_res.get("plus_di")
    minus_di = adx_res.get("minus_di")
    summary = adx_res.get("summary") or "—"

    lines: List[str] = ["[ADX/DMI]"]
    # 수치 줄
    bits = [f"ADX {adx} ({summary})"]
    if plus_di is not None and minus_di is not None:
        p, m = float(plus_di), float(minus_di)
        gap = p - m
        if p >= m:
            bits.append(f"+DI {p:.1f} > -DI {m:.1f} (갭 {gap:+.1f})")
        else:
            bits.append(f"-DI {m:.1f} > +DI {p:.1f} (갭 {gap:+.1f})")
    lines.append(" | ".join(bits))
    lines.append(f"해석: {_adx_insight_sentence(adx_res)}")
    return lines


def _aux_indicator_panel_lines(aux: Dict[str, Any], *, current: float, fmt_price) -> List[str]:
    """패널용 BB / Supertrend / SMA200 한 줄씩."""
    lines: List[str] = []
    if aux.get("bb_lower") is not None and aux.get("bb_upper") is not None:
        state = aux.get("bb_state") or ""
        width = aux.get("bb_width_pct")
        w_txt = f" · 폭{width:.1f}%" if width is not None else ""
        if state == "하단이탈":
            lines.append(
                f"BB(20,2): 하단이탈 {fmt_price(aux['bb_lower'])}{w_txt}"
            )
        elif state == "상단돌파":
            lines.append(
                f"BB(20,2): 상단돌파 {fmt_price(aux['bb_upper'])}{w_txt}"
            )
        else:
            lines.append(
                f"BB(20,2): {fmt_price(aux['bb_lower'])}~{fmt_price(aux['bb_upper'])}{w_txt}"
            )

    if aux.get("st_line") is not None and aux.get("st_dir"):
        st_dir_kr = "상승" if aux["st_dir"] == "up" else "하락"
        role = aux.get("st_role") or ""
        pct = (float(aux["st_line"]) / current - 1.0) * 100.0
        lines.append(
            f"ST(10,3): {st_dir_kr} · {role} {fmt_price(aux['st_line'])} ({pct:+.1f}%)"
        )

    if aux.get("sma200") is not None:
        pct = (float(aux["sma200"]) / current - 1.0) * 100.0
        st = aux.get("sma200_state") or ""
        win = aux.get("sma200_window") or 200
        win_txt = "" if win >= 200 else f"·{win}봉근사"
        lines.append(
            f"SMA200{win_txt}: {fmt_price(aux['sma200'])} ({pct:+.1f}%) · {st}"
        )
    return lines


def detect_surge_state(
    df: pd.DataFrame,
    *,
    aux_ind: Optional[Dict[str, Any]] = None,
    adx_res: Optional[Dict[str, Any]] = None,
    rsi_val: Optional[float] = None,
    lv_by_label: Optional[Dict[str, Any]] = None,
    vol_surge_mult: float = 2.0,
) -> Dict[str, Any]:
    """
    현재가 기준 급등주 패턴 스냅샷.

    Returns:
      score(0~100), state(급등진행|급등과열|급등약함|해당없음),
      reasons, ret_5/10/20, vol_mult, overheat
    """
    out: Dict[str, Any] = {
        "score": 0,
        "state": "해당없음",
        "reasons": [],
        "ret_5": None,
        "ret_10": None,
        "ret_20": None,
        "vol_mult": None,
        "overheat": False,
    }
    if df is None or len(df) < 25:
        return out

    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    current = float(close.iloc[-1])
    if current <= 0:
        return out

    def _ret(n: int) -> Optional[float]:
        if len(close) <= n:
            return None
        base = float(close.iloc[-1 - n])
        if base <= 0:
            return None
        return (current / base - 1.0) * 100.0

    ret5, ret10, ret20 = _ret(5), _ret(10), _ret(20)
    out["ret_5"], out["ret_10"], out["ret_20"] = ret5, ret10, ret20

    vol_ma = vol.rolling(20, min_periods=5).mean()
    v_last = float(vol.iloc[-1])
    v_ma = float(vol_ma.iloc[-1]) if pd.notna(vol_ma.iloc[-1]) else 0.0
    vol_mult = (v_last / v_ma) if v_ma > 0 else 0.0
    # 최근 3봉 중 최대 배수
    for i in range(1, min(3, len(vol))):
        vm = float(vol_ma.iloc[-1 - i]) if pd.notna(vol_ma.iloc[-1 - i]) else 0.0
        if vm > 0:
            vol_mult = max(vol_mult, float(vol.iloc[-1 - i]) / vm)
    out["vol_mult"] = vol_mult

    score = 0.0
    reasons: List[Tuple[float, str]] = []

    def bump(w: float, reason: str) -> None:
        nonlocal score
        score += w
        if w > 0:
            reasons.append((w, reason))

    # 수익률
    if ret5 is not None:
        if ret5 >= 15:
            bump(18, f"5일+{ret5:.0f}%")
        elif ret5 >= 8:
            bump(10, f"5일+{ret5:.0f}%")
        elif ret5 >= 4:
            bump(5, f"5일+{ret5:.0f}%")
        elif ret5 < -5:
            bump(-8, f"5일{ret5:.0f}%")
    if ret10 is not None:
        if ret10 >= 25:
            bump(20, f"10일+{ret10:.0f}%")
        elif ret10 >= 12:
            bump(12, f"10일+{ret10:.0f}%")
        elif ret10 >= 6:
            bump(6, f"10일+{ret10:.0f}%")
        elif ret10 < -8:
            bump(-10, f"10일{ret10:.0f}%")
    if ret20 is not None:
        if ret20 >= 40:
            bump(12, f"20일+{ret20:.0f}%")
        elif ret20 >= 20:
            bump(7, f"20일+{ret20:.0f}%")

    # 거래량
    if vol_mult >= vol_surge_mult * 1.5:
        bump(16, f"거래량×{vol_mult:.1f}")
    elif vol_mult >= vol_surge_mult:
        bump(10, f"거래량×{vol_mult:.1f}")
    elif vol_mult >= 1.4:
        bump(4, f"거래량×{vol_mult:.1f}")

    # 이평·ST
    sma20 = float(close.rolling(20, min_periods=10).mean().iloc[-1])
    if np.isfinite(sma20) and current > sma20:
        bump(8, "SMA20상회")
    elif np.isfinite(sma20) and current < sma20 * 0.98:
        bump(-6, "SMA20하회")

    aux = aux_ind or {}
    if aux.get("st_dir") == "up":
        bump(10, "ST↑")
    elif aux.get("st_dir") == "down":
        bump(-6, "ST↓")

    # ADX/DI
    adx = adx_res or {}
    if adx.get("direction") == "상승":
        bump(8, f"ADX상승({adx.get('adx')})")
    elif adx.get("direction") == "하락":
        bump(-5, f"ADX하락({adx.get('adx')})")
    plus_di, minus_di = adx.get("plus_di"), adx.get("minus_di")
    if plus_di is not None and minus_di is not None and float(plus_di) - float(minus_di) >= 5:
        bump(6, "+DI우세")

    # 저항 돌파·상회 유지
    lv = lv_by_label or {}
    near_res = lv.get("단기 저항")
    recent_res = lv.get("최근저항")
    for lab, key in (("단기저항", "단기 저항"), ("최근저항", "최근저항")):
        lv_i = lv.get(key)
        if lv_i and current > float(lv_i["price"]) * 1.002:
            bump(8, f"{lab}상회")
            break
    if near_res is None and recent_res is None:
        # 20일 고점 상회
        hi20 = float(df["High"].astype(float).tail(21).iloc[:-1].max()) if len(df) >= 22 else None
        if hi20 and current >= hi20:
            bump(6, "20일고점돌파")

    # 과열
    overheat = False
    if rsi_val is not None and rsi_val >= 70:
        overheat = True
        bump(4, f"RSI과열({rsi_val})")  # 급등 증거이기도 함
    if aux.get("bb_state") == "상단돌파":
        overheat = True
        bump(4, "BB상단")
    out["overheat"] = overheat

    score_i = int(round(min(100.0, max(0.0, score))))
    out["score"] = score_i
    reasons_sorted = [r for _, r in sorted(reasons, key=lambda x: x[0], reverse=True)]
    out["reasons"] = reasons_sorted[:4]

    # 상승 모멘텀 없으면 급등 아님
    momentum_ok = (ret5 is not None and ret5 >= 3) or (ret10 is not None and ret10 >= 6)
    if not momentum_ok or score_i < 40:
        out["state"] = "해당없음"
    elif score_i >= 55 and overheat:
        out["state"] = "급등과열"
    elif score_i >= 55:
        out["state"] = "급등진행"
    else:
        out["state"] = "급등약함"
    return out


def _surge_panel_lines(surge: Dict[str, Any]) -> List[str]:
    """인사이트용 급등 패턴 줄."""
    state = surge.get("state") or "해당없음"
    score = int(surge.get("score") or 0)
    if state == "해당없음" and score < 30:
        return []
    bits: List[str] = [f"급등패턴: {state}({score})"]
    for key, lab in (("ret_5", "5일"), ("ret_10", "10일"), ("ret_20", "20일")):
        v = surge.get(key)
        if v is not None:
            bits.append(f"{lab}{v:+.0f}%")
            break
    vm = surge.get("vol_mult")
    if vm is not None and vm >= 1.2:
        bits.append(f"거래량×{vm:.1f}")
    reasons = surge.get("reasons") or []
    # 수익률·거래량 외 핵심 근거 1개
    extra = [r for r in reasons if not r.startswith(("5일", "10일", "20일", "거래량"))]
    if extra:
        bits.append(extra[0])
    return [" · ".join(bits)]


def _draw_surge_status_caption(
    ax,
    text: str,
    *,
    fc: str,
    ec: str,
    row: int = 0,
    target_xy: Optional[Tuple[Any, float]] = None,
) -> None:
    """
    급등/셋업 상태 캡션 — 메인 차트 우상단(axes fraction) 고정.
    target_xy(마지막 봉 고가 등)가 있으면 점선으로 대상과 연결.
    """
    y = 0.972 - max(0, int(row)) * 0.052
    cap_xy = (0.795, y)
    if target_xy is not None:
        tx = mdates.date2num(pd.Timestamp(target_xy[0]).to_pydatetime())
        ty = float(target_xy[1])
        ax.add_artist(
            ConnectionPatch(
                xyA=cap_xy,
                coordsA=ax.transAxes,
                xyB=(tx, ty),
                coordsB=ax.transData,
                axesA=ax,
                axesB=ax,
                color=ec,
                lw=1.1,
                linestyle=(0, (1.0, 2.2)),
                alpha=0.78,
                zorder=31,
                clip_on=False,
            )
        )
        ax.scatter(
            [tx], [ty],
            s=22,
            facecolors="none",
            edgecolors=ec,
            linewidths=1.0,
            zorder=31,
            alpha=0.85,
        )
    ax.text(
        cap_xy[0],
        cap_xy[1],
        text,
        transform=ax.transAxes,
        fontsize=7.8,
        fontweight="bold",
        color=ec,
        ha="right",
        va="top",
        zorder=32,
        clip_on=False,
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.15,
            alpha=0.94,
        ),
    )


def _draw_surge_badge(ax, df: pd.DataFrame, surge: Dict[str, Any]) -> None:
    """급등진행/과열일 때 우상단 상태 캡션 (+대상 봉 점선)."""
    state = surge.get("state")
    if state not in ("급등진행", "급등과열"):
        return
    if df is None or df.empty:
        return
    score = int(surge.get("score") or 0)
    if state == "급등과열":
        text, fc, ec = f"급등↑과열({score})", "#fef5e7", "#e67e22"
    else:
        text, fc, ec = f"급등↑({score})", "#e8f8f5", "#16a085"
    target = (df.index[-1], float(df["High"].astype(float).iloc[-1]))
    _draw_surge_status_caption(ax, text, fc=fc, ec=ec, row=0, target_xy=target)


def detect_surge_setup(
    df: pd.DataFrame,
    *,
    aux_ind: Optional[Dict[str, Any]] = None,
    adx_res: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    급등 직전 셋업 스냅샷 (아직 급등 중이 아닐 때).

    A: ST↓→↑ 전환 후 8~35일 횡보·압축 (MSFT형)
    B: ST↓→↑ 전환 후 ≤10일 (전환 직후 돌파 감시)
    점화임박: A/B + 20일고점 근접

    Returns:
      active, state(셋업A|셋업B|점화임박|해당없음), kind(A|B|None),
      score, flip_age, flip_date, consol_net/rng, adx, vol_dry, dist_hi20, reasons
    """
    out: Dict[str, Any] = {
        "active": False,
        "state": "해당없음",
        "kind": None,
        "score": 0,
        "flip_age": None,
        "flip_date": None,
        "down_len": None,
        "consol_net": None,
        "consol_rng": None,
        "adx": None,
        "vol_dry": None,
        "dist_hi20": None,
        "reasons": [],
    }
    if df is None or len(df) < 40:
        return out

    aux = aux_ind or {}
    st_dir_s = aux.get("st_dir_series")
    if st_dir_s is None or len(st_dir_s) < 40:
        return out

    close = df["Close"].astype(float).to_numpy()
    high = df["High"].astype(float).to_numpy()
    low = df["Low"].astype(float).to_numpy()
    vol = df["Volume"].astype(float).to_numpy()
    darr = st_dir_s.reindex(df.index).to_numpy()
    n = len(close)
    if darr[-1] <= 0:
        return out

    # 최근 ST↓→↑ 플립
    flip_i = None
    for i in range(n - 1, 0, -1):
        if darr[i] > 0 and darr[i - 1] < 0:
            flip_i = i
            break
    if flip_i is None:
        return out

    flip_age = n - 1 - flip_i
    if flip_age < 1 or flip_age > 40:
        return out

    j = flip_i - 1
    while j >= 0 and darr[j] < 0:
        j -= 1
    down_len = flip_i - 1 - (j + 1) + 1
    out["flip_age"] = flip_age
    out["flip_date"] = str(pd.Timestamp(df.index[flip_i]).date())
    out["down_len"] = int(down_len)

    # 플립 이후 ST↑ 유지 비율
    up_ratio = float((darr[flip_i:] > 0).mean())
    if up_ratio < 0.80:
        return out

    c0 = float(close[flip_i])
    c1 = float(close[-1])
    if c0 <= 0:
        return out
    consol_net = (c1 / c0 - 1.0) * 100.0
    hi_seg = float(np.nanmax(high[flip_i:]))
    lo_seg = float(np.nanmin(low[flip_i:]))
    consol_rng = (hi_seg / lo_seg - 1.0) * 100.0 if lo_seg > 0 else 999.0
    out["consol_net"] = consol_net
    out["consol_rng"] = consol_rng

    # 거래량 소강: 최근 5일 / 플립 직후 전반
    half = max(3, flip_age // 2)
    v_early = float(np.nanmean(vol[flip_i: flip_i + half])) if half > 0 else 0.0
    v_late = float(np.nanmean(vol[-5:]))
    vol_dry = (v_late / v_early) if v_early > 0 else None
    out["vol_dry"] = vol_dry

    hi20 = float(np.nanmax(high[-21:-1])) if n >= 22 else float(np.nanmax(high[:-1]))
    dist_hi20 = (c1 / hi20 - 1.0) * 100.0 if hi20 > 0 else None
    out["dist_hi20"] = dist_hi20

    adx_val = None
    if adx_res and adx_res.get("adx") is not None:
        try:
            adx_val = float(adx_res["adx"])
        except (TypeError, ValueError):
            adx_val = None
    out["adx"] = adx_val

    reasons: List[Tuple[float, str]] = []
    score = 0.0

    def bump(w: float, reason: str) -> None:
        nonlocal score
        score += w
        if w > 0:
            reasons.append((w, reason))

    # 공통: 하락 구간·플립
    if down_len >= 10:
        bump(8, f"ST↓{down_len}일→↑")
    elif down_len >= 5:
        bump(5, f"ST↓{down_len}일→↑")
    else:
        bump(2, f"ST전환(+{flip_age}일)")

    bump(4, f"플립+{flip_age}일")

    tight = abs(consol_net) <= 8.0 and consol_rng <= 16.0
    soft = abs(consol_net) <= 12.0 and consol_rng <= 22.0
    near_hi = dist_hi20 is not None and dist_hi20 >= -2.0
    at_hi = dist_hi20 is not None and dist_hi20 >= -0.5

    # 유형 판정
    kind: Optional[str] = None
    if 8 <= flip_age <= 35 and soft:
        kind = "A"
        bump(18, "횡보압축")
        if tight:
            bump(10, f"타이트(폭{consol_rng:.0f}%)")
        else:
            bump(4, f"횡보(폭{consol_rng:.0f}%)")
        if abs(consol_net) <= 4:
            bump(6, f"순변{consol_net:+.0f}%")
    elif flip_age <= 10:
        kind = "B"
        bump(14, "ST전환직후")
        if soft:
            bump(6, "초기횡보")
    else:
        # 플립은 있으나 횡보·직후 조건 미충족
        return out

    if adx_val is not None:
        if adx_val <= 18:
            bump(10, f"ADX낮음({adx_val:.0f})")
        elif adx_val <= 25:
            bump(5, f"ADX약({adx_val:.0f})")
        elif adx_val >= 35:
            bump(-4, f"ADX높음({adx_val:.0f})")

    if vol_dry is not None:
        if vol_dry <= 0.85:
            bump(8, f"거래량소강(×{vol_dry:.2f})")
        elif vol_dry <= 1.05:
            bump(3, "거래량보합")
        elif vol_dry >= 1.6 and near_hi:
            bump(6, f"거래량유입(×{vol_dry:.1f})")

    if at_hi:
        bump(12, "20일고점돌파권")
    elif near_hi:
        bump(7, f"고점근접({dist_hi20:+.1f}%)")
    elif dist_hi20 is not None and dist_hi20 <= -8:
        bump(-6, f"고점이격({dist_hi20:+.0f}%)")

    score_i = int(round(min(100.0, max(0.0, score))))
    out["score"] = score_i
    out["kind"] = kind
    out["reasons"] = [r for _, r in sorted(reasons, key=lambda x: x[0], reverse=True)][:4]

    if score_i < 35:
        return out

    ignite = near_hi and score_i >= 55 and (
        at_hi or (vol_dry is not None and vol_dry >= 1.15) or (adx_val is not None and adx_val <= 20)
    )
    if ignite:
        out["state"] = "점화임박"
    elif kind == "A":
        out["state"] = "셋업A"
    else:
        out["state"] = "셋업B"
    out["active"] = True
    return out


def _surge_setup_panel_lines(setup: Dict[str, Any]) -> List[str]:
    """인사이트용 급등 셋업 줄."""
    if not setup or not setup.get("active"):
        return []
    state = setup.get("state") or "해당없음"
    score = int(setup.get("score") or 0)
    bits: List[str] = [f"급등셋업: {state}({score})"]
    age = setup.get("flip_age")
    if age is not None:
        bits.append(f"ST↑+{age}일")
    net = setup.get("consol_net")
    rng = setup.get("consol_rng")
    if net is not None and rng is not None:
        bits.append(f"횡보{net:+.0f}%/{rng:.0f}%")
    dist = setup.get("dist_hi20")
    if dist is not None:
        bits.append(f"고점{dist:+.1f}%")
    reasons = setup.get("reasons") or []
    extra = [r for r in reasons if not r.startswith(("ST", "플립", "횡보", "타이트"))]
    if extra:
        bits.append(extra[0])
    return [" · ".join(bits)]


def _draw_surge_setup_badge(
    ax,
    df: pd.DataFrame,
    setup: Dict[str, Any],
    *,
    surge_state: Optional[Dict[str, Any]] = None,
) -> None:
    """셋업A/B/점화임박 캡션 (이미 급등진행·과열이면 생략) — 우상단 고정."""
    if not setup or not setup.get("active"):
        return
    if surge_state and surge_state.get("state") in ("급등진행", "급등과열"):
        return
    if df is None or df.empty:
        return
    state = setup.get("state")
    score = int(setup.get("score") or 0)
    if state == "점화임박":
        text, fc, ec = f"급등점화({score})", "#f5eef8", "#8e44ad"
    elif state == "셋업A":
        text, fc, ec = f"급등셋업A({score})", "#eaf2f8", "#2980b9"
    elif state == "셋업B":
        text, fc, ec = f"급등셋업B({score})", "#eafaf1", "#1abc9c"
    else:
        return
    target = (df.index[-1], float(df["High"].astype(float).iloc[-1]))
    _draw_surge_status_caption(ax, text, fc=fc, ec=ec, row=0, target_xy=target)


def detect_plunge_state(
    df: pd.DataFrame,
    *,
    aux_ind: Optional[Dict[str, Any]] = None,
    adx_res: Optional[Dict[str, Any]] = None,
    rsi_val: Optional[float] = None,
    lv_by_label: Optional[Dict[str, Any]] = None,
    vol_surge_mult: float = 2.0,
) -> Dict[str, Any]:
    """
    현재가 기준 급락 패턴 스냅샷 (급등의 대칭).

    Returns:
      score(0~100), state(급락진행|급락과매도|급락약함|해당없음),
      reasons, ret_5/10/20, vol_mult, oversold
    """
    out: Dict[str, Any] = {
        "score": 0,
        "state": "해당없음",
        "reasons": [],
        "ret_5": None,
        "ret_10": None,
        "ret_20": None,
        "vol_mult": None,
        "oversold": False,
    }
    if df is None or len(df) < 25:
        return out

    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    current = float(close.iloc[-1])
    if current <= 0:
        return out

    def _ret(n: int) -> Optional[float]:
        if len(close) <= n:
            return None
        base = float(close.iloc[-1 - n])
        if base <= 0:
            return None
        return (current / base - 1.0) * 100.0

    ret5, ret10, ret20 = _ret(5), _ret(10), _ret(20)
    out["ret_5"], out["ret_10"], out["ret_20"] = ret5, ret10, ret20

    vol_ma = vol.rolling(20, min_periods=5).mean()
    v_last = float(vol.iloc[-1])
    v_ma = float(vol_ma.iloc[-1]) if pd.notna(vol_ma.iloc[-1]) else 0.0
    vol_mult = (v_last / v_ma) if v_ma > 0 else 0.0
    for i in range(1, min(3, len(vol))):
        vm = float(vol_ma.iloc[-1 - i]) if pd.notna(vol_ma.iloc[-1 - i]) else 0.0
        if vm > 0:
            vol_mult = max(vol_mult, float(vol.iloc[-1 - i]) / vm)
    out["vol_mult"] = vol_mult

    score = 0.0
    reasons: List[Tuple[float, str]] = []

    def bump(w: float, reason: str) -> None:
        nonlocal score
        score += w
        if w > 0:
            reasons.append((w, reason))

    # 수익률 (하락 가점)
    if ret5 is not None:
        if ret5 <= -15:
            bump(18, f"5일{ret5:.0f}%")
        elif ret5 <= -8:
            bump(10, f"5일{ret5:.0f}%")
        elif ret5 <= -4:
            bump(5, f"5일{ret5:.0f}%")
        elif ret5 > 5:
            bump(-8, f"5일+{ret5:.0f}%")
    if ret10 is not None:
        if ret10 <= -25:
            bump(20, f"10일{ret10:.0f}%")
        elif ret10 <= -12:
            bump(12, f"10일{ret10:.0f}%")
        elif ret10 <= -6:
            bump(6, f"10일{ret10:.0f}%")
        elif ret10 > 8:
            bump(-10, f"10일+{ret10:.0f}%")
    if ret20 is not None:
        if ret20 <= -40:
            bump(12, f"20일{ret20:.0f}%")
        elif ret20 <= -20:
            bump(7, f"20일{ret20:.0f}%")

    if vol_mult >= vol_surge_mult * 1.5:
        bump(16, f"거래량×{vol_mult:.1f}")
    elif vol_mult >= vol_surge_mult:
        bump(10, f"거래량×{vol_mult:.1f}")
    elif vol_mult >= 1.4:
        bump(4, f"거래량×{vol_mult:.1f}")

    sma20 = float(close.rolling(20, min_periods=10).mean().iloc[-1])
    if np.isfinite(sma20) and current < sma20:
        bump(8, "SMA20하회")
    elif np.isfinite(sma20) and current > sma20 * 1.02:
        bump(-6, "SMA20상회")

    aux = aux_ind or {}
    if aux.get("st_dir") == "down":
        bump(10, "ST↓")
    elif aux.get("st_dir") == "up":
        bump(-6, "ST↑")

    adx = adx_res or {}
    if adx.get("direction") == "하락":
        bump(8, f"ADX하락({adx.get('adx')})")
    elif adx.get("direction") == "상승":
        bump(-5, f"ADX상승({adx.get('adx')})")
    plus_di, minus_di = adx.get("plus_di"), adx.get("minus_di")
    if plus_di is not None and minus_di is not None and float(minus_di) - float(plus_di) >= 5:
        bump(6, "-DI우세")

    lv = lv_by_label or {}
    for lab, key in (("현재지지", "현재 지지"), ("최근지지", "최근지지"), ("강력지지", "강력 지지")):
        lv_i = lv.get(key)
        if lv_i and current < float(lv_i["price"]) * 0.998:
            bump(8, f"{lab}이탈")
            break
    else:
        if len(df) >= 22:
            lo20 = float(df["Low"].astype(float).tail(21).iloc[:-1].min())
            if lo20 > 0 and current <= lo20:
                bump(6, "20일저점이탈")

    oversold = False
    if rsi_val is not None and rsi_val <= 30:
        oversold = True
        bump(4, f"RSI과매도({rsi_val})")
    if aux.get("bb_state") == "하단이탈":
        oversold = True
        bump(4, "BB하단")
    out["oversold"] = oversold

    score_i = int(round(min(100.0, max(0.0, score))))
    out["score"] = score_i
    out["reasons"] = [r for _, r in sorted(reasons, key=lambda x: x[0], reverse=True)][:4]

    momentum_ok = (ret5 is not None and ret5 <= -3) or (ret10 is not None and ret10 <= -6)
    if not momentum_ok or score_i < 40:
        out["state"] = "해당없음"
    elif score_i >= 55 and oversold:
        out["state"] = "급락과매도"
    elif score_i >= 55:
        out["state"] = "급락진행"
    else:
        out["state"] = "급락약함"
    return out


def _plunge_panel_lines(plunge: Dict[str, Any]) -> List[str]:
    """인사이트용 급락 패턴 줄."""
    state = plunge.get("state") or "해당없음"
    score = int(plunge.get("score") or 0)
    if state == "해당없음" and score < 30:
        return []
    bits: List[str] = [f"급락패턴: {state}({score})"]
    for key, lab in (("ret_5", "5일"), ("ret_10", "10일"), ("ret_20", "20일")):
        v = plunge.get(key)
        if v is not None:
            bits.append(f"{lab}{v:+.0f}%")
            break
    vm = plunge.get("vol_mult")
    if vm is not None and vm >= 1.2:
        bits.append(f"거래량×{vm:.1f}")
    reasons = plunge.get("reasons") or []
    extra = [r for r in reasons if not r.startswith(("5일", "10일", "20일", "거래량"))]
    if extra:
        bits.append(extra[0])
    return [" · ".join(bits)]


def _draw_plunge_status_caption(
    ax,
    text: str,
    *,
    fc: str,
    ec: str,
    row: int = 0,
    target_xy: Optional[Tuple[Any, float]] = None,
) -> None:
    """
    급락/셋업 상태 캡션 — 메인 차트 우하단(axes fraction) 고정.
    target_xy(마지막 봉 저가 등)가 있으면 점선으로 대상과 연결.
    """
    y = 0.028 + max(0, int(row)) * 0.052
    cap_xy = (0.795, y)
    if target_xy is not None:
        tx = mdates.date2num(pd.Timestamp(target_xy[0]).to_pydatetime())
        ty = float(target_xy[1])
        ax.add_artist(
            ConnectionPatch(
                xyA=cap_xy,
                coordsA=ax.transAxes,
                xyB=(tx, ty),
                coordsB=ax.transData,
                axesA=ax,
                axesB=ax,
                color=ec,
                lw=1.1,
                linestyle=(0, (1.0, 2.2)),
                alpha=0.78,
                zorder=31,
                clip_on=False,
            )
        )
        ax.scatter(
            [tx], [ty],
            s=22,
            facecolors="none",
            edgecolors=ec,
            linewidths=1.0,
            zorder=31,
            alpha=0.85,
        )
    ax.text(
        cap_xy[0],
        cap_xy[1],
        text,
        transform=ax.transAxes,
        fontsize=7.8,
        fontweight="bold",
        color=ec,
        ha="right",
        va="bottom",
        zorder=32,
        clip_on=False,
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.15,
            alpha=0.94,
        ),
    )


def _draw_plunge_badge(ax, df: pd.DataFrame, plunge: Dict[str, Any]) -> None:
    """급락진행/과매도일 때 우하단 상태 캡션 (+대상 봉 점선)."""
    state = plunge.get("state")
    if state not in ("급락진행", "급락과매도"):
        return
    if df is None or df.empty:
        return
    score = int(plunge.get("score") or 0)
    if state == "급락과매도":
        text, fc, ec = f"급락↓과매도({score})", "#fdedec", "#c0392b"
    else:
        text, fc, ec = f"급락↓({score})", "#fadbd8", "#922b21"
    target = (df.index[-1], float(df["Low"].astype(float).iloc[-1]))
    _draw_plunge_status_caption(ax, text, fc=fc, ec=ec, row=0, target_xy=target)


def detect_plunge_setup(
    df: pd.DataFrame,
    *,
    aux_ind: Optional[Dict[str, Any]] = None,
    adx_res: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    급락 직전 셋업 스냅샷 (급등 셋업의 대칭).

    A: ST↑→↓ 전환 후 8~35일 횡보·압축
    B: ST↑→↓ 전환 후 ≤10일
    붕괴임박: A/B + 20일저점 근접

    Returns:
      active, state(셋업A|셋업B|붕괴임박|해당없음), kind(A|B|None),
      score, flip_age, flip_date, consol_net/rng, adx, vol_dry, dist_lo20, reasons
    """
    out: Dict[str, Any] = {
        "active": False,
        "state": "해당없음",
        "kind": None,
        "score": 0,
        "flip_age": None,
        "flip_date": None,
        "up_len": None,
        "consol_net": None,
        "consol_rng": None,
        "adx": None,
        "vol_dry": None,
        "dist_lo20": None,
        "reasons": [],
    }
    if df is None or len(df) < 40:
        return out

    aux = aux_ind or {}
    st_dir_s = aux.get("st_dir_series")
    if st_dir_s is None or len(st_dir_s) < 40:
        return out

    close = df["Close"].astype(float).to_numpy()
    high = df["High"].astype(float).to_numpy()
    low = df["Low"].astype(float).to_numpy()
    vol = df["Volume"].astype(float).to_numpy()
    darr = st_dir_s.reindex(df.index).to_numpy()
    n = len(close)
    if darr[-1] >= 0:
        return out

    # 최근 ST↑→↓ 플립
    flip_i = None
    for i in range(n - 1, 0, -1):
        if darr[i] < 0 and darr[i - 1] > 0:
            flip_i = i
            break
    if flip_i is None:
        return out

    flip_age = n - 1 - flip_i
    if flip_age < 1 or flip_age > 40:
        return out

    j = flip_i - 1
    while j >= 0 and darr[j] > 0:
        j -= 1
    up_len = flip_i - 1 - (j + 1) + 1
    out["flip_age"] = flip_age
    out["flip_date"] = str(pd.Timestamp(df.index[flip_i]).date())
    out["up_len"] = int(up_len)

    dn_ratio = float((darr[flip_i:] < 0).mean())
    if dn_ratio < 0.80:
        return out

    c0 = float(close[flip_i])
    c1 = float(close[-1])
    if c0 <= 0:
        return out
    consol_net = (c1 / c0 - 1.0) * 100.0
    hi_seg = float(np.nanmax(high[flip_i:]))
    lo_seg = float(np.nanmin(low[flip_i:]))
    consol_rng = (hi_seg / lo_seg - 1.0) * 100.0 if lo_seg > 0 else 999.0
    out["consol_net"] = consol_net
    out["consol_rng"] = consol_rng

    half = max(3, flip_age // 2)
    v_early = float(np.nanmean(vol[flip_i: flip_i + half])) if half > 0 else 0.0
    v_late = float(np.nanmean(vol[-5:]))
    vol_dry = (v_late / v_early) if v_early > 0 else None
    out["vol_dry"] = vol_dry

    lo20 = float(np.nanmin(low[-21:-1])) if n >= 22 else float(np.nanmin(low[:-1]))
    dist_lo20 = (c1 / lo20 - 1.0) * 100.0 if lo20 > 0 else None
    out["dist_lo20"] = dist_lo20

    adx_val = None
    if adx_res and adx_res.get("adx") is not None:
        try:
            adx_val = float(adx_res["adx"])
        except (TypeError, ValueError):
            adx_val = None
    out["adx"] = adx_val

    reasons: List[Tuple[float, str]] = []
    score = 0.0

    def bump(w: float, reason: str) -> None:
        nonlocal score
        score += w
        if w > 0:
            reasons.append((w, reason))

    if up_len >= 10:
        bump(8, f"ST↑{up_len}일→↓")
    elif up_len >= 5:
        bump(5, f"ST↑{up_len}일→↓")
    else:
        bump(2, f"ST전환(+{flip_age}일)")

    bump(4, f"플립+{flip_age}일")

    tight = abs(consol_net) <= 8.0 and consol_rng <= 16.0
    soft = abs(consol_net) <= 12.0 and consol_rng <= 22.0
    near_lo = dist_lo20 is not None and dist_lo20 <= 2.0
    at_lo = dist_lo20 is not None and dist_lo20 <= 0.5

    kind: Optional[str] = None
    if 8 <= flip_age <= 35 and soft:
        kind = "A"
        bump(18, "횡보압축")
        if tight:
            bump(10, f"타이트(폭{consol_rng:.0f}%)")
        else:
            bump(4, f"횡보(폭{consol_rng:.0f}%)")
        if abs(consol_net) <= 4:
            bump(6, f"순변{consol_net:+.0f}%")
    elif flip_age <= 10:
        kind = "B"
        bump(14, "ST전환직후")
        if soft:
            bump(6, "초기횡보")
    else:
        return out

    if adx_val is not None:
        if adx_val <= 18:
            bump(10, f"ADX낮음({adx_val:.0f})")
        elif adx_val <= 25:
            bump(5, f"ADX약({adx_val:.0f})")
        elif adx_val >= 35:
            bump(-4, f"ADX높음({adx_val:.0f})")

    # -DI 우세
    plus_di, minus_di = (adx_res or {}).get("plus_di"), (adx_res or {}).get("minus_di")
    if plus_di is not None and minus_di is not None and float(minus_di) - float(plus_di) >= 5:
        bump(6, "-DI우세")

    if vol_dry is not None:
        if vol_dry <= 0.85:
            bump(8, f"거래량소강(×{vol_dry:.2f})")
        elif vol_dry <= 1.05:
            bump(3, "거래량보합")
        elif vol_dry >= 1.6 and near_lo:
            bump(6, f"거래량유입(×{vol_dry:.1f})")

    if at_lo:
        bump(12, "20일저점이탈권")
    elif near_lo:
        bump(7, f"저점근접({dist_lo20:+.1f}%)")
    elif dist_lo20 is not None and dist_lo20 >= 8:
        bump(-6, f"저점이격({dist_lo20:+.0f}%)")

    score_i = int(round(min(100.0, max(0.0, score))))
    out["score"] = score_i
    out["kind"] = kind
    out["reasons"] = [r for _, r in sorted(reasons, key=lambda x: x[0], reverse=True)][:4]

    if score_i < 35:
        return out

    collapse = near_lo and score_i >= 55 and (
        at_lo or (vol_dry is not None and vol_dry >= 1.15) or (adx_val is not None and adx_val <= 20)
    )
    if collapse:
        out["state"] = "붕괴임박"
    elif kind == "A":
        out["state"] = "셋업A"
    else:
        out["state"] = "셋업B"
    out["active"] = True
    return out


def _plunge_setup_panel_lines(setup: Dict[str, Any]) -> List[str]:
    """인사이트용 급락 셋업 줄."""
    if not setup or not setup.get("active"):
        return []
    state = setup.get("state") or "해당없음"
    score = int(setup.get("score") or 0)
    bits: List[str] = [f"급락셋업: {state}({score})"]
    age = setup.get("flip_age")
    if age is not None:
        bits.append(f"ST↓+{age}일")
    net = setup.get("consol_net")
    rng = setup.get("consol_rng")
    if net is not None and rng is not None:
        bits.append(f"횡보{net:+.0f}%/{rng:.0f}%")
    dist = setup.get("dist_lo20")
    if dist is not None:
        bits.append(f"저점{dist:+.1f}%")
    reasons = setup.get("reasons") or []
    extra = [r for r in reasons if not r.startswith(("ST", "플립", "횡보", "타이트"))]
    if extra:
        bits.append(extra[0])
    return [" · ".join(bits)]


def _draw_plunge_setup_badge(
    ax,
    df: pd.DataFrame,
    setup: Dict[str, Any],
    *,
    plunge_state: Optional[Dict[str, Any]] = None,
) -> None:
    """급락 셋업A/B/붕괴임박 캡션 (이미 급락진행·과매도면 생략) — 우하단 고정."""
    if not setup or not setup.get("active"):
        return
    if plunge_state and plunge_state.get("state") in ("급락진행", "급락과매도"):
        return
    if df is None or df.empty:
        return
    state = setup.get("state")
    score = int(setup.get("score") or 0)
    if state == "붕괴임박":
        text, fc, ec = f"급락붕괴({score})", "#f5eef8", "#6c3483"
    elif state == "셋업A":
        text, fc, ec = f"급락셋업A({score})", "#fdedec", "#e74c3c"
    elif state == "셋업B":
        text, fc, ec = f"급락셋업B({score})", "#fbeee6", "#d35400"
    else:
        return
    target = (df.index[-1], float(df["Low"].astype(float).iloc[-1]))
    _draw_plunge_status_caption(ax, text, fc=fc, ec=ec, row=0, target_xy=target)


def analyze_ticker_surge_history(
    df: pd.DataFrame,
    *,
    aux_ind: Optional[Dict[str, Any]] = None,
    surge_setup: Optional[Dict[str, Any]] = None,
    min_ret5: float = 8.0,
    min_ret10: float = 12.0,
    min_vol: float = 1.5,
    min_spacing: int = 10,
    max_events: int = 8,
) -> Dict[str, Any]:
    """
    종목 구간 내 과거 급등 이벤트 탐지 → 유형(A/B/D) 프로파일 → 현재 셋업 유사도.

    A: ST↓→↑ 후 8~35일 횡보 압축 뒤 급등
    B: ST↓→↑ 직후(≤10일) 급등
    D: 이미 ST↑ 모멘텀 연속 급등
    """
    out: Dict[str, Any] = {
        "events": [],
        "n": 0,
        "kind_counts": {"A": 0, "B": 0, "D": 0},
        "profile": {},
        "conditions": [],
        "similarity": {"score": 0, "label": "해당없음", "tips": []},
        "relaxed": False,
        "adaptive_thr": None,
    }
    if df is None or len(df) < 60:
        out["conditions"] = ["급등 이력 분석: 데이터 부족"]
        return out

    aux = aux_ind or {}
    st_dir_s = aux.get("st_dir_series")
    if st_dir_s is None:
        out["conditions"] = ["급등 이력 분석: ST 시계열 없음"]
        return out

    close = df["Close"].astype(float).to_numpy()
    high = df["High"].astype(float).to_numpy()
    low = df["Low"].astype(float).to_numpy()
    vol = df["Volume"].astype(float).to_numpy()
    vol_ma = pd.Series(vol).rolling(20, min_periods=5).mean().fillna(0).to_numpy()
    darr = st_dir_s.reindex(df.index).to_numpy()
    idx = df.index
    n = len(close)

    def _collect(r5_min: float, r10_min: float, vol_min: float, need_break: bool) -> List[Dict[str, Any]]:
        events_local: List[Dict[str, Any]] = []
        last_i = -999
        for i in range(30, n):
            if darr[i] <= 0:
                continue
            c = close[i]
            if close[i - 5] <= 0 or close[i - 10] <= 0:
                continue
            r5 = (c / close[i - 5] - 1.0) * 100.0
            r10 = (c / close[i - 10] - 1.0) * 100.0
            vm = vol[i] / vol_ma[i] if vol_ma[i] > 0 else 0.0
            for k in range(1, 3):
                if i - k >= 0 and vol_ma[i - k] > 0:
                    vm = max(vm, vol[i - k] / vol_ma[i - k])
            hi20 = float(np.nanmax(high[i - 20 : i])) if i >= 20 else c
            breakout = c >= hi20 * (0.995 if need_break else 0.98)
            if not ((r5 >= r5_min or r10 >= r10_min) and vm >= vol_min and breakout):
                continue
            if i - last_i < min_spacing:
                continue
            last_i = i

            flip_age = None
            flip_i = None
            for j in range(i, max(0, i - 80), -1):
                if j >= 1 and darr[j] > 0 and darr[j - 1] < 0:
                    flip_age = i - j
                    flip_i = j
                    break

            look = 15
            a = max(0, i - look)
            b = i - 1
            pre_net = pre_rng = None
            if b > a:
                seg = close[a : b + 1]
                if seg[0] > 0:
                    pre_net = (seg[-1] / seg[0] - 1.0) * 100.0
                lo_v = float(np.nanmin(low[a : b + 1]))
                hi_v = float(np.nanmax(high[a : b + 1]))
                if lo_v > 0:
                    pre_rng = (hi_v / lo_v - 1.0) * 100.0

            soft = (
                pre_net is not None
                and pre_rng is not None
                and abs(pre_net) <= 12.0
                and pre_rng <= 22.0
            )
            if flip_age is not None and 8 <= flip_age <= 35 and soft:
                kind = "A"
            elif flip_age is not None and flip_age <= 10:
                kind = "B"
            else:
                kind = "D"

            events_local.append(
                {
                    "idx": i,
                    "date": str(pd.Timestamp(idx[i]).date()),
                    "ret5": round(r5, 1),
                    "ret10": round(r10, 1),
                    "vol": round(vm, 2),
                    "kind": kind,
                    "flip_age": flip_age,
                    "flip_i": flip_i,
                    "pre_net": round(pre_net, 1) if pre_net is not None else None,
                    "pre_rng": round(pre_rng, 1) if pre_rng is not None else None,
                    "price": float(c),
                }
            )
        return events_local

    events = _collect(min_ret5, min_ret10, min_vol, True)
    if not events:
        # 저변동 종목 완화 패스 (절대 임계)
        events = _collect(5.0, 8.0, 1.3, False)
        out["relaxed"] = bool(events)
    if not events:
        # 종목 자체 분포 기준 (REIT 등 절대%가 낮은 종목)
        all_r5 = []
        all_r10 = []
        for i in range(30, n):
            if close[i - 5] > 0:
                all_r5.append((close[i] / close[i - 5] - 1.0) * 100.0)
            if close[i - 10] > 0:
                all_r10.append((close[i] / close[i - 10] - 1.0) * 100.0)
        if all_r5 and all_r10:
            r5_thr = float(max(2.5, min(5.0, np.percentile(all_r5, 90))))
            r10_thr = float(max(4.0, min(8.0, np.percentile(all_r10, 90))))
            events = _collect(r5_thr, r10_thr, 1.15, False)
            out["relaxed"] = bool(events)
            if events:
                out["adaptive_thr"] = {"r5": round(r5_thr, 2), "r10": round(r10_thr, 2)}

    events = events[-max_events:]
    out["events"] = events
    out["n"] = len(events)
    counts = {"A": 0, "B": 0, "D": 0}
    for e in events:
        counts[e["kind"]] = counts.get(e["kind"], 0) + 1
    out["kind_counts"] = counts

    if not events:
        out["conditions"] = ["급등 이력 부족 (기간 내 급등 미검출 · 저변동/횡보 가능)"]
        return out

    def _med(vals: List[Any]) -> Optional[float]:
        v = [x for x in vals if x is not None and np.isfinite(x)]
        return float(np.median(v)) if v else None

    flip_ages = [e["flip_age"] for e in events if e.get("flip_age") is not None]
    pre_rngs = [e["pre_rng"] for e in events]
    pre_nets = [e["pre_net"] for e in events]
    vols = [e["vol"] for e in events]
    ret5s = [e["ret5"] for e in events]

    med_flip = _med(flip_ages)
    med_rng = _med(pre_rngs)
    med_net = _med(pre_nets)
    med_vol = _med(vols)
    med_r5 = _med(ret5s)

    dominant = max(counts.keys(), key=lambda k: (counts[k], {"A": 3, "B": 2, "D": 1}[k]))
    if counts[dominant] == 0:
        dominant = "D"

    profile = {
        "dominant_kind": dominant,
        "med_flip_age": round(med_flip, 0) if med_flip is not None else None,
        "med_pre_rng": round(med_rng, 1) if med_rng is not None else None,
        "med_pre_net": round(med_net, 1) if med_net is not None else None,
        "med_vol": round(med_vol, 2) if med_vol is not None else None,
        "med_ret5": round(med_r5, 1) if med_r5 is not None else None,
    }
    out["profile"] = profile

    kind_lab = {"A": "ST전환후횡보", "B": "ST전환직후", "D": "모멘텀연속"}
    if out.get("relaxed") and out.get("adaptive_thr"):
        thr = out["adaptive_thr"]
        tag = f"종목상대(p90 {thr['r5']:.1f}%/{thr['r10']:.1f}%) · "
    elif out.get("relaxed"):
        tag = "완화기준 · "
    else:
        tag = ""
    conditions = [
        f"이 종목 급등 {len(events)}회({tag}주유형 {dominant}/{kind_lab[dominant]}) "
        f"A{counts['A']}/B{counts['B']}/D{counts['D']}"
    ]
    bits = []
    if med_flip is not None:
        bits.append(f"플립+{med_flip:.0f}일")
    if med_rng is not None:
        bits.append(f"직전횡보폭~{med_rng:.0f}%")
    if med_net is not None:
        bits.append(f"직전순변~{med_net:+.0f}%")
    if med_vol is not None:
        bits.append(f"점화Vol×{med_vol:.1f}")
    if med_r5 is not None:
        bits.append(f"급등5일~+{med_r5:.0f}%")
    if bits:
        conditions.append("전형조건: " + " · ".join(bits))
    if dominant == "A":
        conditions.append("유사감시: ST↑유지 + 횡보압축 + 20일고점 돌파·거래량 유입")
    elif dominant == "B":
        conditions.append("유사감시: ST↓→↑ 직후 고점 돌파·거래량 확인")
    else:
        conditions.append("유사감시: ST↑ 모멘텀 유지 중 가속·거래량 확대")
    out["conditions"] = conditions

    sim = {"score": 0, "label": "해당없음", "tips": []}
    setup = surge_setup or {}
    if setup.get("active") and events:
        score = 30.0
        tips: List[str] = []
        sk = setup.get("kind")
        st_name = setup.get("state")
        if sk == dominant or (st_name == "점화임박" and dominant in ("A", "B")):
            score += 25
            tips.append(f"유형일치({dominant})")
        elif sk in ("A", "B") and counts.get(sk, 0) > 0:
            score += 12
            tips.append(f"유형부분일치({sk})")

        age = setup.get("flip_age")
        if age is not None and med_flip is not None:
            diff = abs(float(age) - med_flip)
            if diff <= 5:
                score += 15
                tips.append("플립시차≈전형")
            elif diff <= 12:
                score += 8
                tips.append("플립시차근접")

        rng = setup.get("consol_rng")
        if rng is not None and med_rng is not None:
            if abs(float(rng) - med_rng) <= 5:
                score += 12
                tips.append("횡보폭≈전형")
            elif float(rng) <= med_rng + 3:
                score += 6
                tips.append("횡보압축양호")

        dist = setup.get("dist_hi20")
        if dist is not None and dist >= -3:
            score += 10
            tips.append("고점근접(점화권)")
        if st_name == "점화임박":
            score += 8
            tips.append("점화임박")

        score_i = int(round(min(100.0, max(0.0, score))))
        if score_i >= 70:
            label = "유사높음"
        elif score_i >= 50:
            label = "유사보통"
        elif score_i >= 35:
            label = "유사낮음"
        else:
            label = "유사약함"
        sim = {"score": score_i, "label": label, "tips": tips[:4]}
    elif events:
        sim = {
            "score": 0,
            "label": "셋업대기",
            "tips": [f"과거주유형 {dominant} 조건 감시"],
        }
    out["similarity"] = sim
    return out


def _hist_surge_panel_lines(hist: Dict[str, Any]) -> List[str]:
    """패널용 종목 급등 이력·유사조건."""
    if not hist:
        return []
    lines: List[str] = ["[종목 급등이력]"]
    conds = hist.get("conditions") or []
    if hist.get("n", 0) <= 0:
        lines.extend(conds[:2] or ["급등 이력 부족"])
        return lines
    for c in conds[:3]:
        lines.append(c)
    sim = hist.get("similarity") or {}
    if sim.get("label") and sim.get("label") not in ("해당없음",):
        tips = " · ".join(sim.get("tips") or [])
        bit = f"유사도: {sim.get('label')}({sim.get('score', 0)})"
        if tips:
            bit += f" · {tips}"
        lines.append(bit)
    recent = hist.get("events") or []
    if recent:
        tail = recent[-3:]
        bits = [f"{e['date'][5:]}{e['kind']}+{e['ret5']:.0f}%" for e in tail]
        lines.append("최근: " + ", ".join(bits))
    return lines

def _draw_historical_surge_markers(ax, df: pd.DataFrame, hist: Dict[str, Any]) -> None:
    """
    메인 차트 과거 급등 점화 마커(A/B/D).
    ▲은 고가에 두고, 캡션은 고가 위 여백에서 Y스택·X교대로 배치해 봉·라벨과 겹침을 줄임.
    """
    events = hist.get("events") or []
    if not events or df is None or df.empty:
        return
    colors = {"A": "#16a085", "B": "#2980b9", "D": "#8e44ad"}
    high = df["High"].astype(float)
    y0, y1 = ax.get_ylim()
    span = max(float(y1) - float(y0), 1e-9)
    gap = span * 0.032
    lift = span * 0.028

    items: List[Dict[str, Any]] = []
    for e in events:
        i = int(e["idx"])
        if i < 0 or i >= len(df):
            continue
        d = df.index[i]
        y = float(high.iloc[i])
        kind = e.get("kind") or "D"
        pref = y + lift
        items.append(
            {
                "d": d,
                "y": y,
                "kind": kind,
                "col": colors.get(kind, "#7f8c8d"),
                "pref": pref,
                "xnum": mdates.date2num(pd.Timestamp(d).to_pydatetime()),
            }
        )
    if not items:
        return

    # 캡션이 차트 상단을 넘지 않도록, 고가 위 선호값을 스택
    stack_hi = float(y1) - span * 0.04
    stack_lo = float(y0) + span * 0.08
    stacked = _stack_y_positions(
        [it["pref"] for it in items],
        y_min=stack_lo,
        y_max=max(stack_lo + gap, stack_hi),
        min_gap=gap,
    )

    x_left = mdates.date2num(pd.Timestamp(df.index[0]).to_pydatetime())
    x_right = mdates.date2num(pd.Timestamp(df.index[-1]).to_pydatetime())
    x_span = max(x_right - x_left, 1.0)
    x_pad = max(1.2, x_span * 0.012)

    for k, (it, ly) in enumerate(zip(items, stacked)):
        d, y, kind, col = it["d"], it["y"], it["kind"], it["col"]
        ax.scatter(
            [d], [y],
            marker="^",
            s=36,
            color=col,
            edgecolors="white",
            linewidths=0.55,
            zorder=26,
            alpha=0.95,
        )
        # 최근 봉 구간: 우상단 상태캡션·시나리오·저항라벨과 겹침 → ▲만 표시
        i_near_end = it["xnum"] >= (x_right - x_pad * 6)
        if i_near_end:
            continue

        side = 1 if (k % 2 == 0) else -1
        if k >= 1 and abs(it["xnum"] - items[k - 1]["xnum"]) < x_pad * 3.5:
            x_off = side * x_pad * 1.8
        else:
            x_off = side * x_pad * 0.85
        tx = it["xnum"] + x_off
        if tx < x_left + x_pad:
            tx = it["xnum"] + abs(x_off)
        elif tx > x_right - x_pad * 4:
            tx = it["xnum"] - abs(x_off)

        # 캡션이 고가와 너무 붙으면 한 단 더 올림
        if ly < y + lift * 0.6:
            ly = min(stack_hi, y + lift)

        ax.annotate(
            f"급등{kind}",
            xy=(d, y),
            xytext=(tx, ly),
            textcoords="data",
            fontsize=6.4,
            fontweight="bold",
            color=col,
            ha="center",
            va="bottom",
            zorder=27,
            alpha=0.95,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor=col,
                linewidth=0.7,
                alpha=0.92,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color=col,
                lw=0.65,
                alpha=0.55,
                shrinkA=0,
                shrinkB=2,
            ),
        )


def analyze_ticker_plunge_history(
    df: pd.DataFrame,
    *,
    aux_ind: Optional[Dict[str, Any]] = None,
    plunge_setup: Optional[Dict[str, Any]] = None,
    min_ret5: float = 8.0,
    min_ret10: float = 12.0,
    min_vol: float = 1.5,
    min_spacing: int = 10,
    max_events: int = 8,
) -> Dict[str, Any]:
    """
    종목 구간 내 과거 급락 이벤트 탐지 → 유형(A/B/D) 프로파일 → 현재 셋업 유사도.

    A: ST↑→↓ 후 8~35일 횡보 압축 뒤 급락
    B: ST↑→↓ 직후(≤10일) 급락
    D: 이미 ST↓ 모멘텀 연속 급락
    """
    out: Dict[str, Any] = {
        "events": [],
        "n": 0,
        "kind_counts": {"A": 0, "B": 0, "D": 0},
        "profile": {},
        "conditions": [],
        "similarity": {"score": 0, "label": "해당없음", "tips": []},
        "relaxed": False,
        "adaptive_thr": None,
    }
    if df is None or len(df) < 60:
        out["conditions"] = ["급락 이력 분석: 데이터 부족"]
        return out

    aux = aux_ind or {}
    st_dir_s = aux.get("st_dir_series")
    if st_dir_s is None:
        out["conditions"] = ["급락 이력 분석: ST 시계열 없음"]
        return out

    close = df["Close"].astype(float).to_numpy()
    high = df["High"].astype(float).to_numpy()
    low = df["Low"].astype(float).to_numpy()
    vol = df["Volume"].astype(float).to_numpy()
    vol_ma = pd.Series(vol).rolling(20, min_periods=5).mean().fillna(0).to_numpy()
    darr = st_dir_s.reindex(df.index).to_numpy()
    idx = df.index
    n = len(close)

    def _collect(r5_min: float, r10_min: float, vol_min: float, need_break: bool) -> List[Dict[str, Any]]:
        events_local: List[Dict[str, Any]] = []
        last_i = -999
        for i in range(30, n):
            if darr[i] >= 0:
                continue
            c = close[i]
            if close[i - 5] <= 0 or close[i - 10] <= 0:
                continue
            r5 = (c / close[i - 5] - 1.0) * 100.0
            r10 = (c / close[i - 10] - 1.0) * 100.0
            vm = vol[i] / vol_ma[i] if vol_ma[i] > 0 else 0.0
            for k in range(1, 3):
                if i - k >= 0 and vol_ma[i - k] > 0:
                    vm = max(vm, vol[i - k] / vol_ma[i - k])
            lo20 = float(np.nanmin(low[i - 20 : i])) if i >= 20 else c
            breakdown = c <= lo20 * (1.005 if need_break else 1.02)
            if not ((r5 <= -r5_min or r10 <= -r10_min) and vm >= vol_min and breakdown):
                continue
            if i - last_i < min_spacing:
                continue
            last_i = i

            flip_age = None
            flip_i = None
            for j in range(i, max(0, i - 80), -1):
                if j >= 1 and darr[j] < 0 and darr[j - 1] > 0:
                    flip_age = i - j
                    flip_i = j
                    break

            look = 15
            a = max(0, i - look)
            b = i - 1
            pre_net = pre_rng = None
            if b > a:
                seg = close[a : b + 1]
                if seg[0] > 0:
                    pre_net = (seg[-1] / seg[0] - 1.0) * 100.0
                lo_v = float(np.nanmin(low[a : b + 1]))
                hi_v = float(np.nanmax(high[a : b + 1]))
                if lo_v > 0:
                    pre_rng = (hi_v / lo_v - 1.0) * 100.0

            soft = (
                pre_net is not None
                and pre_rng is not None
                and abs(pre_net) <= 12.0
                and pre_rng <= 22.0
            )
            if flip_age is not None and 8 <= flip_age <= 35 and soft:
                kind = "A"
            elif flip_age is not None and flip_age <= 10:
                kind = "B"
            else:
                kind = "D"

            events_local.append(
                {
                    "idx": i,
                    "date": str(pd.Timestamp(idx[i]).date()),
                    "ret5": round(r5, 1),
                    "ret10": round(r10, 1),
                    "vol": round(vm, 2),
                    "kind": kind,
                    "flip_age": flip_age,
                    "flip_i": flip_i,
                    "pre_net": round(pre_net, 1) if pre_net is not None else None,
                    "pre_rng": round(pre_rng, 1) if pre_rng is not None else None,
                    "price": float(c),
                }
            )
        return events_local

    events = _collect(min_ret5, min_ret10, min_vol, True)
    if not events:
        events = _collect(5.0, 8.0, 1.3, False)
        out["relaxed"] = bool(events)
    if not events:
        all_r5 = []
        all_r10 = []
        for i in range(30, n):
            if close[i - 5] > 0:
                all_r5.append((close[i] / close[i - 5] - 1.0) * 100.0)
            if close[i - 10] > 0:
                all_r10.append((close[i] / close[i - 10] - 1.0) * 100.0)
        if all_r5 and all_r10:
            # 하락 꼬리(p10): 절댓값을 임계로 사용
            r5_thr = float(max(2.5, min(5.0, abs(np.percentile(all_r5, 10)))))
            r10_thr = float(max(4.0, min(8.0, abs(np.percentile(all_r10, 10)))))
            events = _collect(r5_thr, r10_thr, 1.15, False)
            out["relaxed"] = bool(events)
            if events:
                out["adaptive_thr"] = {"r5": round(r5_thr, 2), "r10": round(r10_thr, 2)}

    events = events[-max_events:]
    out["events"] = events
    out["n"] = len(events)
    counts = {"A": 0, "B": 0, "D": 0}
    for e in events:
        counts[e["kind"]] = counts.get(e["kind"], 0) + 1
    out["kind_counts"] = counts

    if not events:
        out["conditions"] = ["급락 이력 부족 (기간 내 급락 미검출 · 저변동/횡보 가능)"]
        return out

    def _med(vals: List[Any]) -> Optional[float]:
        v = [x for x in vals if x is not None and np.isfinite(x)]
        return float(np.median(v)) if v else None

    flip_ages = [e["flip_age"] for e in events if e.get("flip_age") is not None]
    pre_rngs = [e["pre_rng"] for e in events]
    pre_nets = [e["pre_net"] for e in events]
    vols = [e["vol"] for e in events]
    ret5s = [e["ret5"] for e in events]

    med_flip = _med(flip_ages)
    med_rng = _med(pre_rngs)
    med_net = _med(pre_nets)
    med_vol = _med(vols)
    med_r5 = _med(ret5s)

    dominant = max(counts.keys(), key=lambda k: (counts[k], {"A": 3, "B": 2, "D": 1}[k]))
    if counts[dominant] == 0:
        dominant = "D"

    profile = {
        "dominant_kind": dominant,
        "med_flip_age": round(med_flip, 0) if med_flip is not None else None,
        "med_pre_rng": round(med_rng, 1) if med_rng is not None else None,
        "med_pre_net": round(med_net, 1) if med_net is not None else None,
        "med_vol": round(med_vol, 2) if med_vol is not None else None,
        "med_ret5": round(med_r5, 1) if med_r5 is not None else None,
    }
    out["profile"] = profile

    kind_lab = {"A": "ST전환후횡보", "B": "ST전환직후", "D": "모멘텀연속"}
    if out.get("relaxed") and out.get("adaptive_thr"):
        thr = out["adaptive_thr"]
        tag = f"종목상대(p10 -{thr['r5']:.1f}%/-{thr['r10']:.1f}%) · "
    elif out.get("relaxed"):
        tag = "완화기준 · "
    else:
        tag = ""
    conditions = [
        f"이 종목 급락 {len(events)}회({tag}주유형 {dominant}/{kind_lab[dominant]}) "
        f"A{counts['A']}/B{counts['B']}/D{counts['D']}"
    ]
    bits = []
    if med_flip is not None:
        bits.append(f"플립+{med_flip:.0f}일")
    if med_rng is not None:
        bits.append(f"직전횡보폭~{med_rng:.0f}%")
    if med_net is not None:
        bits.append(f"직전순변~{med_net:+.0f}%")
    if med_vol is not None:
        bits.append(f"붕괴Vol×{med_vol:.1f}")
    if med_r5 is not None:
        bits.append(f"급락5일~{med_r5:.0f}%")
    if bits:
        conditions.append("전형조건: " + " · ".join(bits))
    if dominant == "A":
        conditions.append("유사감시: ST↓유지 + 횡보압축 + 20일저점 이탈·거래량 유입")
    elif dominant == "B":
        conditions.append("유사감시: ST↑→↓ 직후 저점 이탈·거래량 확인")
    else:
        conditions.append("유사감시: ST↓ 모멘텀 유지 중 가속·거래량 확대")
    out["conditions"] = conditions

    sim = {"score": 0, "label": "해당없음", "tips": []}
    setup = plunge_setup or {}
    if setup.get("active") and events:
        score = 30.0
        tips: List[str] = []
        sk = setup.get("kind")
        st_name = setup.get("state")
        if sk == dominant or (st_name == "붕괴임박" and dominant in ("A", "B")):
            score += 25
            tips.append(f"유형일치({dominant})")
        elif sk in ("A", "B") and counts.get(sk, 0) > 0:
            score += 12
            tips.append(f"유형부분일치({sk})")

        age = setup.get("flip_age")
        if age is not None and med_flip is not None:
            diff = abs(float(age) - med_flip)
            if diff <= 5:
                score += 15
                tips.append("플립시차≈전형")
            elif diff <= 12:
                score += 8
                tips.append("플립시차근접")

        rng = setup.get("consol_rng")
        if rng is not None and med_rng is not None:
            if abs(float(rng) - med_rng) <= 5:
                score += 12
                tips.append("횡보폭≈전형")
            elif float(rng) <= med_rng + 3:
                score += 6
                tips.append("횡보압축양호")

        dist = setup.get("dist_lo20")
        if dist is not None and dist <= 3:
            score += 10
            tips.append("저점근접(붕괴권)")
        if st_name == "붕괴임박":
            score += 8
            tips.append("붕괴임박")

        score_i = int(round(min(100.0, max(0.0, score))))
        if score_i >= 70:
            label = "유사높음"
        elif score_i >= 50:
            label = "유사보통"
        elif score_i >= 35:
            label = "유사낮음"
        else:
            label = "유사약함"
        sim = {"score": score_i, "label": label, "tips": tips[:4]}
    elif events:
        sim = {
            "score": 0,
            "label": "셋업대기",
            "tips": [f"과거주유형 {dominant} 조건 감시"],
        }
    out["similarity"] = sim
    return out


def _hist_plunge_panel_lines(hist: Dict[str, Any]) -> List[str]:
    """패널용 종목 급락 이력·유사조건."""
    if not hist:
        return []
    lines: List[str] = ["[종목 급락이력]"]
    conds = hist.get("conditions") or []
    if hist.get("n", 0) <= 0:
        lines.extend(conds[:2] or ["급락 이력 부족"])
        return lines
    for c in conds[:3]:
        lines.append(c)
    sim = hist.get("similarity") or {}
    if sim.get("label") and sim.get("label") not in ("해당없음",):
        tips = " · ".join(sim.get("tips") or [])
        bit = f"유사도: {sim.get('label')}({sim.get('score', 0)})"
        if tips:
            bit += f" · {tips}"
        lines.append(bit)
    recent = hist.get("events") or []
    if recent:
        tail = recent[-3:]
        bits = [f"{e['date'][5:]}{e['kind']}{e['ret5']:+.0f}%" for e in tail]
        lines.append("최근: " + ", ".join(bits))
    return lines


def _draw_historical_plunge_markers(ax, df: pd.DataFrame, hist: Dict[str, Any]) -> None:
    """
    메인 차트 과거 급락 붕괴 마커(A/B/D).
    ▼은 저가에 두고, 캡션은 저가 아래 여백에서 Y스택·X교대로 배치해 봉·라벨과 겹침을 줄임.
    """
    events = hist.get("events") or []
    if not events or df is None or df.empty:
        return
    colors = {"A": "#c0392b", "B": "#d35400", "D": "#6c3483"}
    low = df["Low"].astype(float)
    y0, y1 = ax.get_ylim()
    span = max(float(y1) - float(y0), 1e-9)
    gap = span * 0.032
    drop = span * 0.028

    items: List[Dict[str, Any]] = []
    for e in events:
        i = int(e["idx"])
        if i < 0 or i >= len(df):
            continue
        d = df.index[i]
        y = float(low.iloc[i])
        kind = e.get("kind") or "D"
        pref = y - drop
        items.append(
            {
                "d": d,
                "y": y,
                "kind": kind,
                "col": colors.get(kind, "#7f8c8d"),
                "pref": pref,
                "xnum": mdates.date2num(pd.Timestamp(d).to_pydatetime()),
            }
        )
    if not items:
        return

    stack_lo = float(y0) + span * 0.04
    stack_hi = float(y1) - span * 0.08
    # 저가 아래 선호값을 스택 (위쪽 선호부터 → 충돌 시 아래로)
    stacked = _stack_y_positions(
        [it["pref"] for it in items],
        y_min=stack_lo,
        y_max=max(stack_lo + gap, stack_hi),
        min_gap=gap,
    )

    x_left = mdates.date2num(pd.Timestamp(df.index[0]).to_pydatetime())
    x_right = mdates.date2num(pd.Timestamp(df.index[-1]).to_pydatetime())
    x_span = max(x_right - x_left, 1.0)
    x_pad = max(1.2, x_span * 0.012)

    for k, (it, ly) in enumerate(zip(items, stacked)):
        d, y, kind, col = it["d"], it["y"], it["kind"], it["col"]
        ax.scatter(
            [d], [y],
            marker="v",
            s=36,
            color=col,
            edgecolors="white",
            linewidths=0.55,
            zorder=26,
            alpha=0.95,
        )
        # 최근 봉 구간: 우하단 상태캡션·시나리오와 겹침 → ▼만 표시
        if it["xnum"] >= (x_right - x_pad * 6):
            continue

        side = 1 if (k % 2 == 0) else -1
        if k >= 1 and abs(it["xnum"] - items[k - 1]["xnum"]) < x_pad * 3.5:
            x_off = side * x_pad * 1.8
        else:
            x_off = side * x_pad * 0.85
        tx = it["xnum"] + x_off
        if tx < x_left + x_pad:
            tx = it["xnum"] + abs(x_off)
        elif tx > x_right - x_pad * 4:
            tx = it["xnum"] - abs(x_off)

        # 캡션이 저가와 너무 붙으면 한 단 더 내림
        if ly > y - drop * 0.6:
            ly = max(stack_lo, y - drop)

        ax.annotate(
            f"급락{kind}",
            xy=(d, y),
            xytext=(tx, ly),
            textcoords="data",
            fontsize=6.4,
            fontweight="bold",
            color=col,
            ha="center",
            va="top",
            zorder=27,
            alpha=0.95,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor=col,
                linewidth=0.7,
                alpha=0.92,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color=col,
                lw=0.65,
                alpha=0.55,
                shrinkA=0,
                shrinkB=2,
            ),
        )


def _close_slope_per_day(df: pd.DataFrame, lookback: int = 30) -> Optional[float]:
    """종가 회귀 기울기 (가격 / date2num일)."""
    tail = df.tail(lookback)
    if len(tail) < 10:
        return None
    xs = np.array(
        [mdates.date2num(pd.Timestamp(d).to_pydatetime()) for d in tail.index],
        dtype=float,
    )
    ys = tail["Close"].astype(float).values
    if xs[-1] <= xs[0]:
        return None
    return float(np.polyfit(xs, ys, 1)[0])


def _pick_directional_trendline(
    trendlines: List[Dict[str, Any]],
    *,
    direction: str,
    current: float,
) -> Optional[Dict[str, Any]]:
    """
    추세 지속용 선 선택.
    down: 하락 지지 우선 → 기타 하락선
    up: 상승 저항 우선 → 기타 상승선
    """
    if direction == "down":
        cands = [t for t in trendlines if float(t.get("slope") or 0) < 0]
        prefer_kind = "support"
    else:
        cands = [t for t in trendlines if float(t.get("slope") or 0) > 0]
        prefer_kind = "resistance"
    if not cands:
        return None

    def _rank(t: Dict[str, Any]) -> Tuple:
        pn = float(t.get("price_now") or 0.0)
        gap = abs(pn - current) / max(abs(current), 1e-9)
        kind_ok = 1 if t.get("kind") == prefer_kind else 0
        side_ok = 0
        if direction == "down":
            side_ok = 1 if pn <= current * 1.02 else 0
        else:
            side_ok = 1 if pn >= current * 0.98 else 0
        near = 1 if gap <= 0.35 else 0
        alive = 0 if t.get("broken") else 1
        touches = int(t.get("touches") or 0)
        # 높을수록 우선
        return (kind_ok, side_ok, near, alive, touches, -gap)

    return max(cands, key=_rank)


def _resolve_trend_follow_scenario(
    *,
    direction: str,
    current: float,
    trendlines: List[Dict[str, Any]],
    df: pd.DataFrame,
    x_right: float,
    x_fut_end: float,
    lv_by_label: Dict[str, Any],
    peer_target: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    ② 추세하락 / ④ 추세상승 목표가 산출.
    - 추세선 연장(또는 동일 기울기) 기반
    - Low.min/High.max 캡 제거
    - 회귀는 일 단위 기울기
    - 지지이탈/저항돌파와 1.5% 이내 중복 시 이격 조정 또는 비활성
    """
    horizon = float(x_fut_end - x_right)
    if horizon <= 0 or current <= 0:
        return None
    atr = _atr_value(df)
    min_move = max(atr * 1.2, current * 0.03)
    max_move = current * 0.40
    tl = _pick_directional_trendline(trendlines, direction=direction, current=current)
    reg = _close_slope_per_day(df, 30)
    src = ""
    slope = None
    target: Optional[float] = None

    if tl is not None:
        slope = float(tl["slope"])
        pn = float(tl["price_now"])
        if direction == "down":
            if pn > current * 1.02:
                # 하락 저항선이 위 — 현재가에서 동일 기울기 지속
                target = current + slope * horizon
                src = "하락저항기울기"
            else:
                # 하락 지지선 연장
                target = pn + slope * horizon
                src = "하락지지연장"
                if target >= current:
                    target = current + slope * horizon
                    src = "하락지지기울기"
        else:
            if pn < current * 0.98:
                target = current + slope * horizon
                src = "상승지지기울기"
            else:
                target = pn + slope * horizon
                src = "상승저항연장"
                if target <= current:
                    target = current + slope * horizon
                    src = "상승저항기울기"
    elif reg is not None and ((direction == "down" and reg < 0) or (direction == "up" and reg > 0)):
        slope = float(reg)
        target = current + slope * horizon
        src = "30봉회귀"
    else:
        return None

    assert target is not None and slope is not None

    if direction == "down":
        # 의미 있는 하방 + 과도 투영 제한 (Low.min 캡 사용 안 함)
        target = min(target, current - min_move)
        floor = current - max_move
        strong = lv_by_label.get("강력 지지")
        if strong is not None:
            floor = max(floor, float(strong["price"]) * 0.90)
        target = max(target, floor)
    else:
        target = max(target, current + min_move)
        ceil = current + max_move
        high_res = lv_by_label.get("전고점 저항")
        if high_res is not None:
            ceil = min(ceil, float(high_res["price"]) * 1.05)
        target = min(target, ceil)

    active = True
    status = None
    note = ""
    if peer_target is not None and abs(target - float(peer_target)) / current < 0.015:
        # ①/③과 중복 → ATR만큼 추가 이격, 그래도 겹치면 비활성
        if direction == "down":
            adj = min(target - max(atr, current * 0.02), current - min_move * 1.5)
            floor = current - max_move
            adj = max(adj, floor)
        else:
            adj = max(target + max(atr, current * 0.02), current + min_move * 1.5)
            ceil = current + max_move
            adj = min(adj, ceil)
        if abs(adj - float(peer_target)) / current < 0.015:
            active = False
            status = "레벨시나리오와중복"
            note = "①과 목표 근접 → 추세케이스 보류" if direction == "down" else "③과 목표 근접 → 추세케이스 보류"
        else:
            target = adj
            src = f"{src}·이격조정"
            note = "레벨시나리오와 분리"

    return {
        "target": float(target),
        "slope": float(slope),
        "src": src,
        "active": active,
        "status": status,
        "note": note,
        "path_x": [x_right, x_fut_end],
        "path_y": [current, float(target)],
    }


def _build_sr_case_eval_lines(
    entries: List[Dict[str, Any]],
    *,
    current: float,
    fmt_price,
    lv_by_label: Dict[str, Any],
) -> List[str]:
    """
    현재가 기준 핵심 케이스 2개 평가:
      ▼ 하락(지지 이탈) ①
      ▲ 상승(저항 돌파) ③
    """
    by_key = {str(e["key"]): e for e in entries}
    sc1 = by_key.get("sc1")
    sc3 = by_key.get("sc3")
    if sc1 is None and sc3 is None:
        return []

    near_sup = lv_by_label.get("현재 지지")
    strong_sup = lv_by_label.get("강력 지지")
    broken_sup = lv_by_label.get("직전 지지(이탈)")
    recent_sup = lv_by_label.get("최근지지")
    near_res = lv_by_label.get("단기 저항")
    high_res = lv_by_label.get("전고점 저항")
    recent_res = lv_by_label.get("최근저항")

    lines: List[str] = ["케이스 평가 (현재가→지지/저항)"]

    if sc1 is not None:
        trig = None
        if recent_sup and near_sup and recent_sup["price"] > near_sup["price"]:
            trig = recent_sup["price"]
        elif near_sup:
            trig = near_sup["price"]
        elif recent_sup:
            trig = recent_sup["price"]
        st = _proximity_state(current, float(trig), bullish_break=False) if trig else "대기"
        lines.append(
            f"▼① 하락·지지이탈  {_score_gauge(int(sc1['score']))} {int(sc1['score'])}점 [{st}]"
        )
        lines.append(f"  목표 {sc1.get('target_txt') or '-'}")
        if trig is not None:
            lines.append(f"  트리거: {fmt_price(trig)} 종가 하향이탈")
        path_bits = ["현재"]
        if recent_sup and near_sup and recent_sup["price"] > near_sup["price"]:
            path_bits.append("최근지지")
        if near_sup:
            path_bits.append("현재지지")
        if strong_sup and (not near_sup or strong_sup["price"] < near_sup["price"]):
            path_bits.append("강력지지")
        lines.append("  경로: " + " → ".join(path_bits))
        inv = near_res["price"] if near_res else (broken_sup["price"] if broken_sup else None)
        if inv is not None:
            lines.append(f"  무효: {fmt_price(inv)} 상향 회복")
        reasons = sc1.get("reasons") or []
        if reasons:
            lines.append("  근거: " + " · ".join(reasons[:2]))

    if sc3 is not None:
        wlabels = sc3.get("waypoint_labels") or []
        trig = None
        if len(wlabels) >= 2 and len(sc3.get("path_y") or []) >= 2:
            trig = float(sc3["path_y"][1])
        elif recent_res and (near_res is None or recent_res["price"] < near_res["price"]):
            trig = recent_res["price"]
        elif near_res:
            trig = near_res["price"]
        elif high_res:
            trig = high_res["price"]
        st = _proximity_state(current, float(trig), bullish_break=True) if trig else "대기"
        lines.append(
            f"▲③ 상승·저항돌파  {_score_gauge(int(sc3['score']))} {int(sc3['score'])}점 [{st}]"
        )
        lines.append(f"  목표 {sc3.get('target_txt') or '-'}")
        if trig is not None:
            lines.append(f"  트리거: {fmt_price(trig)} 종가 상향돌파")
        if wlabels:
            lines.append("  경로: " + " → ".join(wlabels))
        else:
            path_bits = ["현재"]
            if recent_res and (near_res is None or recent_res["price"] < near_res["price"]):
                path_bits.append("최근저항")
            if near_res and near_res["price"] > current:
                path_bits.append("단기저항")
            if high_res and high_res["price"] > current:
                path_bits.append("전고점")
            lines.append("  경로: " + " → ".join(path_bits))
        inv = near_sup["price"] if near_sup else None
        if inv is not None:
            lines.append(f"  무효: {fmt_price(inv)} 하향 재이탈")
        reasons = sc3.get("reasons") or []
        if reasons:
            lines.append("  근거: " + " · ".join(reasons[:2]))

    if sc1 is not None and sc3 is not None:
        s1, s3 = int(sc1["score"]), int(sc3["score"])
        if s1 >= s3 + 8:
            verdict = f"판정: 하락케이스 우위 ({s1}>{s3})"
        elif s3 >= s1 + 8:
            verdict = f"판정: 상승케이스 우위 ({s3}>{s1})"
        else:
            verdict = f"판정: 혼조·관망 (↓{s1} / ↑{s3})"
        lines.append(verdict)

    return lines


def _annotate_scenario_waypoints(
    ax,
    path_x: List[float],
    path_y: List[float],
    labels: List[str],
    color: str,
) -> None:
    """시나리오 경로 중간 경유점(지지/저항) 마커."""
    if len(path_x) < 2 or len(path_y) != len(path_x):
        return
    for i in range(1, len(path_x)):
        ax.scatter(
            [path_x[i]], [path_y[i]],
            s=28 if i < len(path_x) - 1 else 36,
            marker="o" if i < len(path_x) - 1 else "D",
            color=color,
            edgecolors="white",
            linewidths=0.6,
            zorder=14,
            alpha=0.9,
        )
        if i < len(labels) and labels[i]:
            va = "top" if i == len(path_x) - 1 and path_y[i] < path_y[0] else "bottom"
            dy = -7 if va == "top" else 5
            ax.annotate(
                labels[i],
                xy=(path_x[i], path_y[i]),
                xytext=(0, dy),
                textcoords="offset points",
                fontsize=6.2,
                fontweight="bold",
                color=color,
                ha="center",
                va=va,
                zorder=15,
                alpha=0.9,
            )


def _stack_y_positions(
    prefs: List[float],
    *,
    y_min: float,
    y_max: float,
    min_gap: float,
) -> List[float]:
    """선호 y를 위에서부터 최소 간격으로 스택 (겹침 방지)."""
    if not prefs:
        return []
    order = sorted(range(len(prefs)), key=lambda i: -float(prefs[i]))
    out = [0.0] * len(prefs)
    placed: List[float] = []
    lo = float(y_min)
    hi = float(y_max)
    gap = max(float(min_gap), 1e-9)
    for i in order:
        y = min(max(float(prefs[i]), lo), hi)
        # 이미 배치된(더 위쪽) 라벨과 간격 확보 → 아래로 밀기
        for py in sorted(placed, reverse=True):
            if y > py - gap and y < py + gap:
                y = py - gap
        y = min(max(y, lo), hi)
        # 아래로 밀려 다른 것과 다시 겹치면 위로
        for py in sorted(placed):
            if abs(y - py) < gap:
                y = py + gap
        y = min(max(y, lo), hi)
        placed.append(y)
        out[i] = y
    return out


def _draw_left_gutter_level_captions(
    ax,
    items: List[Dict[str, Any]],
    *,
    x_gutter: float,
    x_line: float,
    y_min: float,
    y_max: float,
) -> None:
    """
    지지/저항·목표가 이름 캡션을 좌측 여백에 배치 (봉과 분리).
    x_gutter: 캡션 x, x_line: 점선이 닿는 차트 좌단(데이터 시작).
    """
    if not items:
        return
    span = max(float(y_max) - float(y_min), 1e-9)
    gap = span * 0.030
    prefs = [float(it["price"]) for it in items]
    stacked = _stack_y_positions(
        prefs,
        y_min=float(y_min) + span * 0.02,
        y_max=float(y_max) - span * 0.02,
        min_gap=gap,
    )
    for it, ly in zip(items, stacked):
        col = it.get("color") or "#34495e"
        fs = float(it.get("fontsize") or 8.4)
        label = str(it.get("label") or "")
        py = float(it["price"])
        ax.annotate(
            label,
            xy=(x_line, py),
            xytext=(x_gutter, ly),
            textcoords="data",
            fontsize=fs,
            fontweight="bold",
            color=col,
            ha="right",
            va="center",
            zorder=21,
            clip_on=False,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor=col,
                linewidth=0.75,
                alpha=0.92,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color=col,
                lw=0.75,
                linestyle=":",
                alpha=0.55,
                shrinkA=1,
                shrinkB=1,
            ),
        )


def _pick_active_trendline(
    trendlines: List[Dict[str, Any]],
    current: float,
) -> Optional[Dict[str, Any]]:
    """현재가에 가장 가까운 유효(미돌파 우선) 추세선."""
    if not trendlines:
        return None

    def _key(tl: Dict[str, Any]) -> Tuple[int, float]:
        pn = float(tl.get("price_now") or 0.0)
        dist = abs(pn - current) / max(abs(current), 1e-9)
        broken_pen = 1 if tl.get("broken") else 0
        return (broken_pen, dist)

    return min(trendlines, key=_key)


def _draw_trendlines_with_lane(
    ax,
    trendlines: List[Dict[str, Any]],
    *,
    y_min: float,
    y_max: float,
    current: Optional[float] = None,
    x_left: Optional[float] = None,
    x_right: Optional[float] = None,
    fmt_price=None,
) -> Optional[Dict[str, Any]]:
    """
    추세선 + 좌상단 레인 캡션 + 현재가 관계(방안 A+C).
    - 활성(현재가 최근접) 추세선 강조
    - 현재가 수평선 + 추세선 연장점·거리 태그
    반환: 활성 추세선(없으면 None)
    """
    active = _pick_active_trendline(trendlines, current) if current is not None else None
    if not trendlines and current is None:
        return None

    y_span = max(float(y_max) - float(y_min), 1e-9)
    items: List[Dict[str, Any]] = []

    for tl in trendlines:
        is_active = active is not None and tl is active
        col = "#e67e22" if tl["kind"] == "resistance" else "#9b59b6"
        ax.plot(
            [tl["x0"], tl["x1"]],
            [tl["y0"], tl["y1"]],
            color=col,
            linestyle="-",
            linewidth=2.35 if is_active else 1.15,
            alpha=0.95 if is_active else 0.55,
            zorder=8 if is_active else 7,
        )
        direction = "하락" if tl["slope"] < 0 else "상승"
        role = "저항" if tl["kind"] == "resistance" else "지지"
        # 화살표=기울기 방향 (역할↑↓과 혼동·시나리오 번호(터치수) 혼동 방지)
        arrow = "↘" if tl["slope"] < 0 else "↗"
        short = f"{'★' if is_active else ''}{arrow}{direction}{role}·터치{tl['touches']}"
        if tl.get("broken"):
            short += "·돌파"
        x_mid = (float(tl["x0"]) + float(tl["x1"])) / 2.0
        y_mid = (float(tl["y0"]) + float(tl["y1"])) / 2.0
        items.append({
            "col": col,
            "short": short,
            "x_mid": x_mid,
            "y_mid": y_mid,
            "y_pref": y_mid,
            "active": is_active,
        })

    # 좌상단 레인: axes fraction y를 위에서부터 스택
    n = len(items)
    if n:
        order = sorted(range(n), key=lambda i: (-int(items[i]["active"]), -items[i]["y_pref"]))
        lane_ys = [0.0] * n
        y_ax = 0.965
        step = 0.048
        for k, i in enumerate(order):
            lane_ys[i] = y_ax - k * step

        for i, it in enumerate(items):
            ax.annotate(
                it["short"],
                xy=(it["x_mid"], it["y_mid"]),
                xytext=(0.015, lane_ys[i]),
                xycoords="data",
                textcoords="axes fraction",
                fontsize=7.6 if it["active"] else 7.2,
                fontweight="bold",
                color=it["col"],
                ha="left",
                va="center",
                zorder=26,
                arrowprops=dict(
                    arrowstyle="-",
                    color=it["col"],
                    lw=1.05 if it["active"] else 0.85,
                    linestyle=":",
                    alpha=0.75 if it["active"] else 0.55,
                    shrinkA=0,
                    shrinkB=2,
                ),
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    facecolor="#fff8e7" if it["active"] else "white",
                    edgecolor=it["col"],
                    linewidth=1.15 if it["active"] else 0.85,
                    alpha=0.96,
                ),
            )

    # --- 방안 C: 현재가 수평 + 활성 추세선 교차/거리 ---
    if current is not None and x_left is not None and x_right is not None:
        ax.plot(
            [x_left, x_right],
            [current, current],
            color="#2c3e50",
            linestyle="-.",
            linewidth=1.15,
            alpha=0.70,
            zorder=9,
        )
        ax.scatter(
            [x_right], [current],
            s=42, marker="o", color="#2c3e50",
            edgecolors="white", linewidths=0.8, zorder=12,
        )
        ax.annotate(
            "현재가",
            xy=(0.0, current),
            xycoords=("axes fraction", "data"),
            xytext=(3, 0),
            textcoords="offset points",
            fontsize=7.0,
            fontweight="bold",
            color="#2c3e50",
            ha="left",
            va="center",
            zorder=20,
            clip_on=False,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#2c3e50",
                      linewidth=0.6, alpha=0.90),
        )

        if active is not None:
            pn = float(active["price_now"])
            col = "#e67e22" if active["kind"] == "resistance" else "#9b59b6"
            pct = (pn / current - 1.0) * 100.0 if current else 0.0
            # 추세선 현재 연장점
            ax.scatter(
                [x_right], [pn],
                s=48, marker="D", color=col,
                edgecolors="white", linewidths=0.7, zorder=13,
            )
            # 현재가 ↔ 추세선 수직 연결 (우측 끝)
            ax.plot(
                [x_right, x_right],
                [current, pn],
                color=col,
                linestyle=":",
                linewidth=1.2,
                alpha=0.85,
                zorder=10,
            )
            role = "활성저항선" if active["kind"] == "resistance" else "활성지지선"
            dist_txt = f"{role} {pct:+.1f}%"
            if fmt_price is not None:
                dist_txt = f"{dist_txt} ({fmt_price(pn)})"
            # 지지선 아래 / 저항선 위 + 점선 연결 (선 위 최근 구간)
            is_support = active["kind"] != "resistance"
            x_span = max(float(x_right) - float(x_left), 1.0)
            x0 = float(active.get("x0") or x_left)
            y0 = float(active.get("y0") or pn)
            slope = float(active.get("slope") or 0.0)
            # 선의 중후반(최근 65% 지점)에 앵커
            x_anchor = x0 + (float(x_right) - x0) * 0.72
            y_on_line = y0 + slope * (x_anchor - x0)
            y_off = y_span * 0.050
            y_cap = y_on_line - y_off if is_support else y_on_line + y_off
            y_cap = min(max(y_cap, float(y_min) + y_span * 0.015), float(y_max) - y_span * 0.015)
            ax.annotate(
                dist_txt,
                xy=(x_anchor, y_on_line),
                xytext=(x_anchor, y_cap),
                textcoords="data",
                fontsize=7.0,
                fontweight="bold",
                color=col,
                ha="center",
                va="top" if is_support else "bottom",
                zorder=22,
                clip_on=False,
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    facecolor="#fff8e7",
                    edgecolor=col,
                    linewidth=0.9,
                    alpha=0.95,
                ),
                arrowprops=dict(
                    arrowstyle="-",
                    color=col,
                    lw=0.85,
                    linestyle=":",
                    alpha=0.72,
                    shrinkA=1,
                    shrinkB=2,
                ),
            )

    return active


def _nudge_ys_away_from_current(
    prefs: List[float],
    *,
    current: float,
    y_min: float,
    y_max: float,
    avoid_frac: float = 0.10,
) -> List[float]:
    """현재가 근처 금지대를 피해 라벨 선호 y를 위/아래로 밀어낸다."""
    span = max(float(y_max) - float(y_min), 1e-9)
    avoid_lo = float(current) - span * avoid_frac
    avoid_hi = float(current) + span * avoid_frac
    out: List[float] = []
    for y in prefs:
        yy = float(y)
        if avoid_lo <= yy <= avoid_hi:
            # 목표가/선호가 현재가 위면 위로, 아래면 아래로
            yy = avoid_hi if yy >= float(current) else avoid_lo
        out.append(min(max(yy, float(y_min)), float(y_max)))
    return out


def _scenario_badge_label(num: str, *, status: Optional[str] = None) -> str:
    """메인용 번호 뱃지 텍스트 (★는 뱃지 옆에 별도 표시)."""
    mark = ""
    if status == "무효화":
        mark = "×"
    elif status == "달성":
        mark = "✓"
    return f"{num}{mark}"


def _draw_num_badge(
    ax,
    x: float,
    y: float,
    text: str,
    color: str,
    *,
    is_top3: bool = False,
    expired: bool = False,
    z: int = 22,
) -> None:
    """번호 뱃지. Top3는 테두리만 강조, 별은 옆(_draw_scenario_number_lane)에서 표시."""
    edge = "#7f8c8d" if expired else color
    tc = "#7f8c8d" if expired else color
    lw = 1.55 if is_top3 else (1.0 if not expired else 0.85)
    fs = 8.6 if is_top3 else 7.6
    ax.annotate(
        text,
        xy=(float(x), float(y)),
        fontsize=fs,
        fontweight="bold",
        color=tc,
        ha="center",
        va="center",
        zorder=z,
        bbox=dict(
            boxstyle="circle,pad=0.20",
            facecolor="white",
            edgecolor=edge,
            linewidth=lw,
            alpha=0.96 if not expired else 0.80,
        ),
        clip_on=False,
    )


def _scenario_strength_style(
    score: float,
    scores: List[float],
) -> Dict[str, float]:
    """
    평가점수 상대 정규화 → 메인 시나리오 선 세기 (방안 D+A).
    Top3와 무관. t∈[0,1], lw/alpha/z 반환.
    """
    vals = [float(s) for s in scores if s is not None]
    if not vals:
        t = 0.5
    else:
        lo, hi = min(vals), max(vals)
        t = 0.5 if hi <= lo + 1e-9 else (float(score) - lo) / (hi - lo)
        t = max(0.0, min(1.0, t))
    return {
        "t": t,
        "lw": 0.85 + 1.75 * t,
        "alpha": 0.32 + 0.60 * t,
        "z": 11 + int(round(10 * t)),
    }


def _draw_scenario_on_main(
    ax,
    *,
    label: str,
    color: str,
    x_end: float,
    y_end: float,
    xytext: Tuple[float, float],
    is_top3: bool,
    path_x: Optional[List[float]] = None,
    path_y: Optional[List[float]] = None,
    arrow_from: Optional[Tuple[float, float]] = None,
    linestyle: str = "--",
    label_y: Optional[float] = None,
    label_x: Optional[float] = None,
    show_label: bool = True,
    strength: Optional[Dict[str, float]] = None,
) -> None:
    """메인 차트 시나리오 경로/화살표.

    선 세기: strength(점수 상대) 우선. Top3는 라벨 ★만 (선 굵기와 분리).
    """
    if strength is not None:
        lw = float(strength.get("lw", 1.4))
        alpha = float(strength.get("alpha", 0.7))
        z = int(strength.get("z", 16))
    elif is_top3:
        lw, alpha, z = 2.2, 0.95, 24
    else:
        lw, alpha, z = 1.15, 0.42, 12
    fs = 8.2 if is_top3 else 6.8
    bw, ba = (1.35, 0.95) if is_top3 else (0.55, 0.45)
    text = f"★ {label}" if is_top3 else label

    if path_x is not None and path_y is not None and len(path_x) >= 2:
        ax.plot(path_x, path_y, color=color, linestyle=linestyle,
                linewidth=lw, alpha=alpha, zorder=z - 1)
        ax.annotate(
            "",
            xy=(path_x[-1], path_y[-1]),
            xytext=(path_x[-2], path_y[-2]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=linestyle, alpha=alpha),
            zorder=z - 1,
        )
    elif arrow_from is not None:
        ax.annotate(
            "",
            xy=(x_end, y_end),
            xytext=arrow_from,
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=linestyle, alpha=alpha),
            zorder=z - 1,
        )

    if not show_label:
        return

    lx = float(label_x) if label_x is not None else float(x_end)
    if label_y is not None:
        ax.annotate(
            text,
            xy=(x_end, y_end),
            xytext=(lx, float(label_y)),
            textcoords="data",
            fontsize=fs,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
            zorder=z,
            alpha=1.0 if is_top3 else 0.50,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.7, linestyle=":", alpha=0.50),
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="white",
                edgecolor=color,
                alpha=ba,
                linewidth=bw,
            ),
            clip_on=False,
        )
    else:
        ax.annotate(
            text,
            xy=(x_end, y_end),
            xytext=(8, xytext[1]),
            textcoords="offset points",
            fontsize=fs,
            fontweight="bold",
            color=color,
            ha="left",
            zorder=z,
            alpha=1.0 if is_top3 else 0.50,
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="white",
                edgecolor=color,
                alpha=ba,
                linewidth=bw,
            ),
            clip_on=False,
        )


def _draw_scenario_number_lane(
    ax,
    draws: List[Dict[str, Any]],
    *,
    x_right: float,
    x_lane: float,
    y_min: float,
    y_max: float,
    current: float,
) -> None:
    """
    메인: 시나리오 ①~⑥ 번호만 표시.
    - 목표가 Y에 가느다란 수평선 (마지막 봉 → 우측 끝)
    - 번호 뱃지는 우측 끝에 배치 (겹치면 Y스택)
    - Top3는 번호 옆에 ★
    """
    if not draws:
        return
    span = max(float(y_max) - float(y_min), 1e-9)
    # 뱃지는 실제 목표가 Y를 우선 — 현재가 회피 nudge는 ①↔② 경로와 번호를 엇갈리게 만듦
    prefs = [
        min(max(float(d["y"]), float(y_min)), float(y_max))
        for d in draws
    ]
    min_gap = span * max(0.032, min(0.065, 0.50 / max(len(draws), 1)))
    lane_ys = _stack_y_positions(prefs, y_min=y_min, y_max=y_max, min_gap=min_gap)

    x0 = float(x_right)
    x1 = float(x_lane)
    score_pool = [float(d.get("score") or 0) for d in draws]
    for d, ly in zip(draws, lane_ys):
        is_top = bool(d.get("is_top3"))
        expired = not bool(d.get("active", True))
        status = d.get("status")
        color = d.get("color") or "#34495e"
        y_tgt = min(max(float(d["y"]), float(y_min)), float(y_max))
        st = _scenario_strength_style(float(d.get("score") or 0), score_pool)
        # 목표가 가는 선 — 세기는 점수 상대 (Top3와 분리)
        ax.plot(
            [x0, x1], [y_tgt, y_tgt],
            color=color,
            linestyle="-",
            linewidth=0.45 + 0.55 * st["t"],
            alpha=0.35 + 0.45 * st["t"],
            zorder=12,
            solid_capstyle="round",
        )
        # 번호 Y가 목표와 다르면 짧은 수직 연결 (같은 시나리오 색으로만)
        if abs(ly - y_tgt) > span * 0.008:
            ax.plot(
                [x1, x1], [y_tgt, ly],
                color=color, linestyle=":", linewidth=0.55,
                alpha=0.40, zorder=12,
            )
        badge = _scenario_badge_label(str(d.get("num") or "?"), status=status)
        _draw_num_badge(
            ax, x1, ly, badge, color,
            is_top3=is_top, expired=expired, z=24 if is_top else 20,
        )
        # Top3: 번호 뱃지 옆에 ★ (식별만)
        if is_top:
            ax.annotate(
                "★",
                xy=(x1, ly),
                xytext=(11, 0),
                textcoords="offset points",
                fontsize=9.0,
                fontweight="bold",
                color=color,
                ha="left",
                va="center",
                zorder=25,
                clip_on=False,
            )

    ax.text(
        0.995, 0.015,
        "번호=①~⑥(우측)  ★Top3  · 선굵기=점수상대  · 시나리오명은 인사이트",
        transform=ax.transAxes,
        fontsize=6.5,
        color="#5d6d7e",
        ha="right",
        va="bottom",
        zorder=30,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#bdc3c7", linewidth=0.5, alpha=0.85),
    )


def _pattern_target_score(
    pe: Dict[str, Any],
    adx_res: Dict[str, Any],
    *,
    bullish: bool,
    break_label: str,
    base: float = 40.0,
) -> Tuple[int, List[str]]:
    """패턴 측정 목표 시나리오 공용 점수 (기준선 돌파/이탈 · 거래량 · 넥라인 기울기 · ADX)."""
    s = base
    r: List[str] = []
    word = "돌파" if bullish else "이탈"
    if pe["breakout"]:
        s += 15.0
        r.append(f"{break_label} {word} 완료")
    else:
        r.append(f"{break_label} {word} 대기 ({-pe['breakout_pct']:+.1f}%)")
    if pe["vol_ok_n"]:
        s += 5.0 * pe["vol_ok_n"]
        r.append(f"거래량 조건 {pe['vol_ok_n']}/{pe['vol_total']} 충족")
    slope_cls = pe.get("slope_cls")
    if slope_cls == "우상향":
        s += 5.0 if bullish else -8.0
        r.append("넥라인 우상향" + ("" if bullish else "(감점)"))
    elif slope_cls == "우하향":
        s += -8.0 if bullish else 5.0
        r.append("넥라인 우하향" + ("(감점)" if bullish else ""))
    fav = "상승" if bullish else "하락"
    unfav = "하락" if bullish else "상승"
    if adx_res.get("direction") == fav:
        s += 8.0
        r.append(f"ADX {fav} 추세 동반")
    elif adx_res.get("direction") == unfav:
        s -= 6.0
        r.append(f"ADX {unfav} 추세(감점)")
    return int(round(min(90.0, max(5.0, s)))), r


def _pattern_trade_guide(
    pe: Dict[str, Any],
    adx_res: Dict[str, Any],
    *,
    bullish: bool,
    break_label: str,
    score: Optional[int],
    reasons: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    패턴 상태·긍정/부정 조건·매매 가이드(매수/매도/유지) 산출.

    반환: status, positives, negatives, action, action_note, weak, progress_pct
    """
    word = "돌파" if bullish else "이탈"
    progress = pe.get("progress_pct")
    rr = pe.get("risk_reward")
    inv_pct = abs(float(pe.get("invalid_pct") or 0.0))

    # ---- 상태 ----
    if pe.get("invalidated"):
        status = "무효화"
    elif progress is not None and progress >= 100.0:
        status = "달성"
    else:
        status = "진행"

    # ---- 긍정 / 부정 조건 ----
    positives: List[str] = []
    negatives: List[str] = []
    if pe.get("breakout"):
        positives.append(f"{break_label}{word}○")
    else:
        negatives.append(f"{break_label}{word}대기")
    vol_ok = int(pe.get("vol_ok_n") or 0)
    vol_tot = int(pe.get("vol_total") or 0)
    if vol_tot:
        (positives if vol_ok >= max(1, (vol_tot + 1) // 2) else negatives).append(
            f"거래량{vol_ok}/{vol_tot}"
        )
    slope_cls = pe.get("slope_cls")
    if slope_cls == "우상향":
        (positives if bullish else negatives).append("넥우상향")
    elif slope_cls == "우하향":
        (positives if not bullish else negatives).append("넥우하향")
    fav = "상승" if bullish else "하락"
    unfav = "하락" if bullish else "상승"
    if adx_res.get("direction") == fav:
        positives.append(f"ADX{fav}")
    elif adx_res.get("direction") == unfav:
        negatives.append(f"ADX{unfav}")
    if rr is not None and rr >= 1.5:
        positives.append(_fmt_risk_reward(rr, short=True))
    elif rr is not None and rr < 1.0:
        negatives.append(f"{_fmt_risk_reward(rr, short=True)}부족")
    fill = pe.get("fill_status")
    if fill == "미채움":
        positives.append("갭미채움")
    elif fill == "완전":
        negatives.append("갭완전채움")
    elif fill == "부분":
        negatives.append("갭부분채움")
    # reasons에서 짧은 키워드 보강
    for r in (reasons or [])[:2]:
        if "감점" in r or "대기" in r:
            if r not in negatives and len(negatives) < 4:
                negatives.append(r.replace(" ", "")[:10])
        elif r not in positives and len(positives) < 4 and "대기" not in r:
            short = r.replace(" ", "").replace("조건", "")[:10]
            if short and short not in positives:
                positives.append(short)

    positives = positives[:3]
    negatives = negatives[:2]
    weak = score is not None and score < 45

    # ---- 매매 가이드 ----
    action = "유지"
    note = "관망"
    if status == "무효화":
        # 강세패턴 무효 → 매도/청산, 약세패턴 무효 → 매수 검토
        action = "매도" if bullish else "매수"
        note = "무효선이탈·전제붕괴"
    elif status == "달성":
        # 강세 목표 달성 → 익절(매도), 약세 목표 달성 → 커버/관망
        action = "매도" if bullish else "유지"
        note = "목표가소진" if bullish else "하락목표소진"
    else:
        # 진행
        if not pe.get("breakout"):
            action = "유지"
            note = f"{word}확인전관망"
        elif inv_pct < 1.5:
            action = "유지"
            note = "무효선근접"
        elif progress is not None and progress >= 50.0:
            action = "유지"
            note = "추격자제·달성중"
        elif rr is not None and rr < 1.0:
            action = "유지"
            note = "이익부족"
        elif weak:
            action = "유지"
            note = "약한신호"
        else:
            # 돌파 직후 ~ 달성률 50% 미만 + 조건 충족
            action = "매수" if bullish else "매도"
            note = f"{word}후진입" if pe.get("breakout") else note

    # ---- 매수/매도 타점 (강세: 기준선 매수·목표 매도 / 약세: 기준선 매도·목표 커버매수) ----
    neck = float(pe.get("neck_now") or 0.0)
    tgt = float(pe.get("target") or 0.0)
    inv = float(pe.get("invalid") or 0.0)
    if bullish:
        buy_px, sell_px = neck, tgt
        buy_role, sell_role = "매수타점(돌파/재테스트)", "매도타점(익절)"
    else:
        sell_px, buy_px = neck, tgt
        buy_role, sell_role = "매수타점(커버)", "매도타점(이탈)"

    action_why = _guide_action_why(
        status=status,
        action=action,
        note=note,
        bullish=bullish,
        break_label=break_label,
        word=word,
        positives=positives,
        negatives=negatives,
        progress=progress,
        rr=rr,
        weak=weak,
    )

    return {
        "status": status,
        "positives": positives,
        "negatives": negatives,
        "action": action,
        "action_note": note,
        "action_why": action_why,
        "weak": weak,
        "progress_pct": progress,
        "buy_px": buy_px,
        "sell_px": sell_px,
        "stop_px": inv,
        "buy_role": buy_role,
        "sell_role": sell_role,
        "bullish": bullish,
    }


def _guide_action_why(
    *,
    status: str,
    action: str,
    note: str,
    bullish: bool,
    break_label: str,
    word: str,
    positives: List[str],
    negatives: List[str],
    progress: Optional[float],
    rr: Optional[float],
    weak: bool,
) -> str:
    """캡션용: 가이드(매수/매도/유지) 판단 근거를 쉬운 한 줄로 설명."""
    side = "상승" if bullish else "하락"
    pos_hint = "·".join(positives[:2]) if positives else ""
    neg_hint = "·".join(negatives[:1]) if negatives else ""

    if status == "무효화":
        if bullish:
            return "근거:무효선 이탈로 상승패턴 전제 붕괴 → 매도/청산"
        return "근거:무효선 이탈로 하락패턴 전제 붕괴 → 매수 검토"

    if status == "달성":
        if bullish:
            return "근거:측정 목표가 도달·상승여력 소진 → 익절(매도)"
        return "근거:하락 목표가 소진 → 추격보다 관망"

    # 진행
    if "확인전관망" in note:
        return f"근거:{break_label}{word} 미확인 → 성급한 진입 자제·관망"
    if note == "무효선근접":
        return "근거:손절(무효선)이 너무 가까워 리스크 불리 → 관망"
    if note == "추격자제·달성중":
        pct = f"{progress:.0f}%" if progress is not None else "50%+"
        return f"근거:목표가까지 {pct} 진행 → 추격 진입 자제"
    if note == "이익부족" or note == "손익비부족":
        rr_txt = _fmt_risk_reward(rr, short=False) if rr is not None else "위험1당 이익부족"
        return f"근거:{rr_txt}로 기대이익이 위험보다 작음 → 관망"
    if note == "약한신호":
        return "근거:신뢰 낮음(신호 약함) → 관망"
    if "후진입" in note:
        why = f"근거:{break_label}{word} 확인"
        if pos_hint:
            why += f"·{pos_hint}"
        why += f" → {side}방향 {action}"
        return why
    if action == "유지":
        if neg_hint:
            return f"근거:{neg_hint} 등으로 조건 미흡 → 관망"
        return "근거:뚜렷한 진입 조건 부족 → 관망"
    if pos_hint:
        return f"근거:{pos_hint} → {action}"
    return f"근거:{note} → {action}"


def _fmt_px_plain(p: float) -> str:
    """캡션용 가격 (matplotlib $ 수식 모드 회피)."""
    return f"{float(p):,.2f}"


def _fmt_risk_reward(rr: Optional[float], *, short: bool = True) -> str:
    """
    손익비(기대이익 ÷ 위험)를 쉬운 말로 표기.
    예: rr=1.5 → '이익1.5배' / '위험1당 이익1.5배'
    """
    if rr is None:
        return "손익비—"
    try:
        v = float(rr)
    except (TypeError, ValueError):
        return "손익비—"
    if not np.isfinite(v):
        return "손익비—"
    if short:
        return f"이익{v:.1f}배"
    return f"위험1당 이익{v:.1f}배"


def _fmt_confidence_score(score: Optional[float], *, short: bool = True) -> str:
    """
    신뢰 점수(0~100)를 쉬운 말로 표기.
    높음(≥70) / 보통(45~69) / 낮음(<45)
    예: 신뢰높음(76) · 신뢰 보통 · 76/100
    """
    if score is None:
        return "신뢰—"
    try:
        # 0~1 비율로 들어오면 100점 환산
        v = float(score)
        if 0.0 <= v <= 1.0:
            v = v * 100.0
        v_i = int(round(v))
    except (TypeError, ValueError):
        return "신뢰—"
    if not np.isfinite(v_i):
        return "신뢰—"
    v_i = max(0, min(100, v_i))
    if v_i >= 70:
        level = "높음"
    elif v_i >= 45:
        level = "보통"
    else:
        level = "낮음"
    if short:
        return f"신뢰{level}({v_i})"
    return f"신뢰 {level} · {v_i}/100"


def _confidence_reference_lines() -> List[str]:
    """
    [참고] 요약 범례 — 패널 하단용 (핵심만 압축).

    카테고리: 신뢰도 · 추세 · 밴드 · 시나리오 · 급등/급락 · PEG · 주의
    """
    return [
        "[참고] 요약 (상대척도·확률 아님)",
        "신뢰: 0~100 · ≥70높음 · 45~69보통 · <45낮음 | 돌파·거래량·넥라인·ADX",
        "추세: ADX=<20횡보 · 20~25약 · 25~40보통 · ≥40강 | +DI↑매수 · -DI↑매도",
        "지표: ST↑지지/ST↓저항 · SMA200 장기기준(③상단) · BB하단과매도/상단과매수",
        "시나리오: ①지지이탈 ②추세↓ ③저항돌파 ④추세↑ · ⑤V반등 ⑥Λ조정",
        "급등: 진행·과열(배지) | 셋업A=ST↑후횡보 | B=전환직후 | 점화=고점근접",
        "급락: 진행·과매도(배지) | 셋업A=ST↓후횡보 | B=전환직후 | 붕괴=저점근접",
        "PEG: PER÷성장% · <1저평가 · 1~1.5적정 · ≥1.5부담 | 재무≠단기매매신호",
        "급등이력: A=ST↑후횡보 · B=전환직후 · D=모멘텀 · ▲마커=과거점화 · 유사도=현재셋업비교",
        "급락이력: A=ST↓후횡보 · B=전환직후 · D=모멘텀 · ▼마커=과거붕괴 · 유사도=현재셋업비교",
    ]


def _join_caption_parts(parts: List[str], sep: str = " · ", max_len: int = 52) -> str:
    """캡션 조각을 연결하고, 길면 구분자 기준으로만 줄바꿈(정보 절단 최소화)."""
    bits = [str(p).strip() for p in parts if p and str(p).strip()]
    if not bits:
        return "—"
    lines: List[str] = []
    cur = ""
    for b in bits:
        cand = f"{cur}{sep}{b}" if cur else b
        if len(cand) <= max_len:
            cur = cand
            continue
        if cur:
            lines.append(cur)
        if len(b) > max_len:
            lines.append(b[: max_len - 1] + "…")
            cur = ""
        else:
            cur = b
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _annotate_level_price(
    ax,
    x: float,
    y: float,
    text: str,
    color: str,
    *,
    dy: int,
    fontsize: float = 6.2,
) -> None:
    """투영 구간 우측 가격 라벨 (배경으로 선과 겹침 완화)."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(-1, dy),
        textcoords="offset points",
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        ha="right",
        va="center",
        zorder=15,
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                  edgecolor=color, linewidth=0.45, alpha=0.88),
    )


def _mini_legend_text(
    *,
    has_buy: bool = False,
    has_sell: bool = False,
    has_target: bool = False,
    has_invalid: bool = False,
    has_impulse: bool = False,
    has_break: bool = False,
    has_surge: bool = False,
    pattern_span: bool = False,
) -> str:
    """미니별 범례 한 줄 (해당 미니에 실제 있는 항목만)."""
    bits: List[str] = []
    if has_buy:
        bits.append("▲매수")
    if has_sell:
        bits.append("▼매도")
    if has_target:
        bits.append("--목표")
    if has_invalid:
        bits.append("..무효")
    if has_impulse:
        bits.append("■임펄스Vol")
    if has_break:
        bits.append("■돌파Vol")
    if has_surge and not has_impulse and not has_break:
        bits.append("■Vol급증")
    if pattern_span:
        bits.append("=패턴구간")
    return "  ".join(bits)


def _fmt_mini_vol(v: float, _pos=None) -> str:
    """미니 거래량 Y라벨 — 1e7/1e8 오프셋 대신 M/K 표기."""
    av = abs(float(v))
    if av >= 1e6:
        return f"{v / 1e6:.1f}M"
    if av >= 1e3:
        return f"{v / 1e3:.0f}K"
    return f"{v:.0f}"


def _kill_axis_offsets(ax) -> None:
    """ScalarFormatter 잔여 오프셋(1e8 등) 강제 제거."""
    for axis in (ax.xaxis, ax.yaxis):
        ot = axis.get_offset_text()
        ot.set_visible(False)
        ot.set_text("")
        ot.set_alpha(0.0)


def _compute_period_volume_profile(
    ohlcv: pd.DataFrame,
    *,
    num_bins: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    차트 기간 Fixed-Range 매물대(Volume Profile).
    Returns: centers, volumes, bin_height, poc, poc_share_pct, total_vol
    """
    df = _normalize(ohlcv)
    if df.empty or "Volume" not in df.columns:
        return None
    low = df["Low"].astype(float)
    high = df["High"].astype(float)
    vol = df["Volume"].astype(float)
    p_lo, p_hi = float(low.min()), float(high.max())
    if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_hi <= p_lo:
        return None
    n = num_bins if num_bins is not None else int(min(36, max(18, len(df) // 3)))
    edges = np.linspace(p_lo, p_hi, n + 1)
    volumes = np.zeros(n, dtype=float)
    for i in range(n):
        b_lo, b_hi = edges[i], edges[i + 1]
        mask = (low <= b_hi) & (high >= b_lo)
        volumes[i] = float(vol.loc[mask].sum())
    total = float(volumes.sum())
    if total <= 0:
        return None
    poc_i = int(np.argmax(volumes))
    centers = (edges[:-1] + edges[1:]) / 2.0
    return {
        "centers": centers,
        "volumes": volumes,
        "bin_height": float(edges[1] - edges[0]),
        "poc": float(centers[poc_i]),
        "poc_share_pct": float(volumes[poc_i] / total * 100.0),
        "total_vol": total,
    }


def _draw_period_volume_profile(
    ax,
    vp: Dict[str, Any],
    *,
    x_right: float,
    x_pad: float,
    fmt_price,
) -> None:
    """
    방안 A: 우측 패딩에 얇은 매물대 히스토그램 + POC 점선/태그.
    (시나리오 번호 레인 x_pad*0.92 와 겹치지 않게 폭은 패딩의 ~48%)
    """
    vols = np.asarray(vp["volumes"], dtype=float)
    centers = np.asarray(vp["centers"], dtype=float)
    h = float(vp["bin_height"])
    poc = float(vp["poc"])
    vmax = float(vols.max()) if len(vols) else 0.0
    if vmax <= 0:
        return
    left = float(x_right) + float(x_pad) * 0.02
    max_w = float(x_pad) * 0.48
    for price, v in zip(centers, vols):
        if v <= 0:
            continue
        w = (float(v) / vmax) * max_w
        # POC 구간만 조금 진하게
        is_poc = abs(float(price) - poc) < h * 0.51
        ax.barh(
            float(price), w, height=h * 0.92, left=left,
            color="#e74c3c" if is_poc else "#7f8c8d",
            alpha=0.55 if is_poc else 0.28,
            linewidth=0, zorder=3, clip_on=False,
        )
    # POC 점선 (차트 전체)
    ax.axhline(poc, color="#c0392b", linestyle="--", linewidth=0.75, alpha=0.65, zorder=9)
    ax.annotate(
        f"POC {fmt_price(poc)}",
        xy=(left + max_w * 0.55, poc),
        fontsize=7.0,
        fontweight="bold",
        color="#c0392b",
        ha="center",
        va="center",
        zorder=21,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#c0392b", linewidth=0.6, alpha=0.9),
    )


def _apply_vol_yaxis(ax, *, labelsize: float = 9.0) -> None:
    """거래량 Y축을 M/K로 고정하고 과학표기 오프셋을 끈다."""
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_mini_vol))
    ax.tick_params(axis="y", labelsize=labelsize)
    _kill_axis_offsets(ax)


def _style_shared_date_axis(ax, *, x_last: float, labelsize: float = 9.5) -> None:
    """
    메인·거래량·이격도 공유 X축 일자 표기.
    - 월 단위: YYYY-MM
    - 현재시점(마지막 봉): YYYY-MM-DD
    - 라벨은 축 안쪽(하단)에 그려 미니차트에 가려지지 않게 함
    - 데이터 마지막일 이후(우측 시나리오 패딩) 라벨은 숨김
    """
    from matplotlib.ticker import FixedLocator, FuncFormatter as _FF

    x_last = float(x_last)
    x0, _x1 = [float(v) for v in ax.get_xlim()]
    # mdates.num2date → tz-aware(UTC); 비교용으로 naive로 통일
    d0 = pd.Timestamp(mdates.num2date(x0)).tz_localize(None).normalize()
    d_last = pd.Timestamp(mdates.num2date(x_last)).tz_localize(None).normalize()

    # 월초 틱 (현재일과 5일 이상 떨어진 것만)
    month_ticks: List[float] = []
    cur = pd.Timestamp(year=d0.year, month=d0.month, day=1)
    while cur <= d_last + pd.Timedelta(days=1):
        xn = float(mdates.date2num(cur.to_pydatetime()))
        if xn >= x0 - 1.0 and abs((cur - d_last).days) >= 5:
            month_ticks.append(xn)
        if cur.month == 12:
            cur = pd.Timestamp(year=cur.year + 1, month=1, day=1)
        else:
            cur = pd.Timestamp(year=cur.year, month=cur.month + 1, day=1)

    ticks = sorted(set(month_ticks + [x_last]))

    def _fmt(x, _pos=None):
        xf = float(x)
        if xf > x_last + 0.6:
            return ""
        if abs(xf - x_last) < 0.8:
            return mdates.num2date(xf).strftime("%Y-%m-%d")
        return mdates.num2date(xf).strftime("%Y-%m")

    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(_FF(_fmt))
    # 라벨은 축 아래 — 미니차트와 간격(레이아웃)으로 가림 방지
    ax.tick_params(
        axis="x", which="major", labelsize=labelsize,
        pad=3, length=4, direction="out", labelbottom=True, labelcolor="#1a252f",
    )
    ax.tick_params(axis="x", which="minor", length=2, color="#bdc3c7", direction="out")
    for lbl in ax.get_xticklabels():
        lbl.set_verticalalignment("top")
        lbl.set_clip_on(False)
        lbl.set_zorder(20)
    _kill_axis_offsets(ax)

    # 현재시점 강조 가이드선 (라벨은 틱으로 표시)
    ax.axvline(x_last, color="#1a5276", linewidth=0.9, alpha=0.40, linestyle=":", zorder=3)


def _style_mini_vol_xaxis(ax_vol) -> None:
    """거래량축: X=날짜만, Y=M/K (과학적 표기 1e7/1e8 제거)."""
    ax_vol.tick_params(axis="x", labelsize=5.0, pad=1.5, length=2.5, labelbottom=True)
    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax_vol.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=3))
    _apply_vol_yaxis(ax_vol, labelsize=5.0)
    for lbl in ax_vol.get_xticklabels():
        lbl.set_clip_on(False)


def _draw_mini_caption_card(
    ax_cap,
    header: str,
    body: str,
    guide: str,
    *,
    action: Optional[str] = None,
    legend_line: Optional[str] = None,
    reason: Optional[str] = None,
) -> None:
    """미니 캡션: 내용을 카드 세로 중앙에 균등·타이트하게 배치."""
    ax_cap.axis("off")
    ax_cap.set_xlim(0, 1)
    ax_cap.set_ylim(0, 1)
    edge = {"매수": "#27ae60", "매도": "#c0392b", "유지": "#95a5a6"}.get(action or "", "#bdc3c7")
    ax_cap.add_patch(
        FancyBboxPatch(
            (0.02, 0.04), 0.96, 0.92,
            boxstyle="round,pad=0.012,rounding_size=0.035",
            transform=ax_cap.transAxes,
            facecolor="#f8f9fa",
            edgecolor=edge,
            linewidth=1.0,
            alpha=0.98,
            clip_on=False,
            zorder=0,
        )
    )
    # 글자색 검정 통일, 가이드만 bold
    ink = "#000000"
    rows: List[Tuple[str, float, str, str]] = []
    if legend_line:
        rows.append((legend_line, 6.5, "normal", ink))
    rows.append((header or "—", 7.5, "normal", ink))
    rows.append((body or "—", 6.8, "normal", ink))
    rows.append((guide or "—", 7.3, "bold", ink))
    if reason:
        rows.append((reason, 6.5, "normal", ink))

    n = len(rows)
    # 카드 내부(약 0.12~0.88)에 세로 중앙 정렬, 줄간격 균등
    y_top, y_bot = 0.86, 0.14
    if n == 1:
        ys = [0.5]
    else:
        span = y_top - y_bot
        ys = [y_top - span * i / (n - 1) for i in range(n)]
    for (txt, fs, weight, col), y in zip(rows, ys):
        ax_cap.text(
            0.5, y, txt,
            transform=ax_cap.transAxes,
            fontsize=fs,
            fontweight=weight,
            color=col,
            ha="center",
            va="center",
            linespacing=1.0,
            zorder=1,
        )


def _evaluate_triple_bottom_w(ev: Dict[str, Any], df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    역삼중창(W) 상세 평가: 어깨/머리 식별, 넥라인 기울기·돌파 상태, 거래량 3항목 채점.

    검출 이벤트의 pivot_dates/pivot_prices(저점 3개), neckline_dates/neckline_prices(중간 고점 2개)를 사용.
    """
    pds = ev.get("pivot_dates")
    pps = ev.get("pivot_prices")
    nds = ev.get("neckline_dates")
    nps = ev.get("neckline_prices")
    if not pds or not pps or not nds or not nps or len(pds) != 3 or len(nds) != 2:
        return None

    pds = [pd.Timestamp(t) for t in pds]
    nds = [pd.Timestamp(t) for t in nds]
    pps = [float(p) for p in pps]
    nps = [float(p) for p in nps]

    idx = df.index
    pos = [int(idx.get_indexer([t], method="pad")[0]) for t in pds]
    if any(p < 0 for p in pos):
        return None
    i, j, k = pos
    n = len(df)
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    current = float(close.iloc[-1])

    # 어깨/머리 라벨: 중앙 저점이 최저일 때만 '머리'
    mid_is_head = pps[1] <= min(pps[0], pps[2])
    labels = ["좌어깨", "머리" if mid_is_head else "중앙 저점", "우어깨"]

    # 넥라인 기울기 분류 (±1% 기준)
    slope_pct = (nps[1] / nps[0] - 1.0) * 100.0 if nps[0] else 0.0
    if slope_pct > 1.0:
        slope_cls, slope_comment = "우상향", "신뢰 보강"
    elif slope_pct < -1.0:
        slope_cls, slope_comment = "우하향", "신뢰 낮음"
    else:
        slope_cls, slope_comment = "평형", ""

    # 넥라인 연장값·돌파 상태
    x1 = mdates.date2num(nds[0].to_pydatetime())
    x2 = mdates.date2num(nds[1].to_pydatetime())
    slope = (nps[1] - nps[0]) / max(x2 - x1, 1e-9)
    x_last = mdates.date2num(pd.Timestamp(idx[-1]).to_pydatetime())
    neck_now = nps[1] + slope * (x_last - x2)
    breakout = current > neck_now
    breakout_pct = (current / neck_now - 1.0) * 100.0 if neck_now else 0.0

    # 돌파 봉 탐색 (우어깨 이후 최초 종가 돌파)
    breakout_pos = None
    for t in range(k + 1, n):
        x_t = mdates.date2num(pd.Timestamp(idx[t]).to_pydatetime())
        if float(close.iloc[t]) > nps[1] + slope * (x_t - x2):
            breakout_pos = t
            break

    # 거래량 평가 (고전 이론 3항목)
    m_lh = (i + j) // 2
    m_hr = (j + k) // 2
    seg_l = vol.iloc[max(0, i - 2): m_lh + 1]
    seg_h = vol.iloc[m_lh: m_hr + 1]
    seg_r = vol.iloc[m_hr: min(k + 3, n)]
    checks: List[Tuple[str, str]] = []

    ok_a = len(seg_r) > 0 and len(seg_h) > 0 and seg_r.mean() < seg_h.mean()
    checks.append(("우어깨 감소", "○" if ok_a else "×"))

    m = min(8, n - 1 - k)
    if m >= 2:
        rebound = vol.iloc[k + 1: k + 1 + m].mean()
        before = vol.iloc[max(j, k - m): k + 1].mean()
        ok_b = rebound > before
        checks.append(("반등 증량", "○" if ok_b else "×"))
    else:
        ok_b = False
        checks.append(("반등 증량", "판단 불가"))

    if breakout_pos is not None:
        vol_ma20 = vol.rolling(20).mean()
        ma_at = vol_ma20.iloc[breakout_pos]
        ok_c = pd.notna(ma_at) and float(vol.iloc[breakout_pos]) >= 1.5 * float(ma_at)
        checks.append(("돌파 증량", "○" if ok_c else "×"))
    else:
        ok_c = None
        checks.append(("돌파 증량", "대기"))

    vol_ok_n = sum(1 for _, v in checks if v == "○")
    vol_total = sum(1 for _, v in checks if v in ("○", "×"))

    # 측정 목표치 (measured move): 목표 = 돌파 지점 넥라인 값 + 패턴 높이(넥라인-머리)
    head_i = int(np.argmin(pps))
    head_price = pps[head_i]
    x_head = mdates.date2num(pds[head_i].to_pydatetime())
    neck_at_head = nps[0] + slope * (x_head - x1)
    height = max(neck_at_head - head_price, 0.0)
    if breakout_pos is not None:
        x_bo = mdates.date2num(pd.Timestamp(idx[breakout_pos]).to_pydatetime())
        neck_ref = nps[1] + slope * (x_bo - x2)
    else:
        neck_ref = neck_now
    target = neck_ref + height
    target_pct = (target / current - 1.0) * 100.0 if current else 0.0
    invalid = pps[2]  # 우어깨 저점 이탈 시 패턴 무효
    invalid_pct = (invalid / current - 1.0) * 100.0 if current else 0.0
    risk = current - invalid
    risk_reward = (target - current) / risk if risk > 0 and target > current else None
    progress_pct = ((current - neck_ref) / height * 100.0) if (breakout and height > 0) else None
    formation_days = (pds[2] - pds[0]).days

    return {
        "invalidated": current < invalid,
        "target": float(target),
        "target_pct": target_pct,
        "invalid": float(invalid),
        "invalid_pct": invalid_pct,
        "risk_reward": risk_reward,
        "progress_pct": progress_pct,
        "formation_days": formation_days,
        "height": float(height),
        "pivot_dates": pds,
        "pivot_prices": pps,
        "labels": labels,
        "neck_dates": nds,
        "neck_prices": nps,
        "slope_cls": slope_cls,
        "slope_pct": slope_pct,
        "slope_comment": slope_comment,
        "neck_now": float(neck_now),
        "breakout": breakout,
        "breakout_pct": breakout_pct,
        "breakout_date": pd.Timestamp(idx[breakout_pos]) if breakout_pos is not None else None,
        "vol_checks": checks,
        "vol_ok_n": vol_ok_n,
        "vol_total": vol_total,
        "vol_segments": [
            (idx[max(0, i - 2)], idx[m_lh], float(seg_l.mean()) if len(seg_l) else None, "좌"),
            (idx[m_lh], idx[m_hr], float(seg_h.mean()) if len(seg_h) else None, "머" if mid_is_head else "중"),
            (idx[m_hr], idx[min(k + 2, n - 1)], float(seg_r.mean()) if len(seg_r) else None, "우"),
        ],
        "end_str": pds[2].strftime("%m/%d"),
    }


def _evaluate_double_top_m(ev: Dict[str, Any], df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    쌍봉(M) 상세 평가: 두 고점·목선(중간 저점) 식별, 목선 이탈 상태, 거래량 채점, 측정 목표(하락).

    검출 이벤트의 peak_dates/peak_prices(고점 2개), valley_date/valley_price(중간 저점)를 사용.
    """
    pk_ds = ev.get("peak_dates")
    pk_ps = ev.get("peak_prices")
    v_d = ev.get("valley_date")
    v_p = ev.get("valley_price")
    if not pk_ds or not pk_ps or v_d is None or v_p is None or len(pk_ds) != 2:
        return None

    pk_ds = [pd.Timestamp(t) for t in pk_ds]
    pk_ps = [float(p) for p in pk_ps]
    v_d = pd.Timestamp(v_d)
    v_p = float(v_p)

    idx = df.index
    pos = [int(idx.get_indexer([t], method="pad")[0]) for t in pk_ds]
    if any(p < 0 for p in pos):
        return None
    i, k = pos
    n = len(df)
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    current = float(close.iloc[-1])

    # 목선: 두 고점 사이 저점의 수평선
    neck = v_p
    breakout = current < neck  # 하락 이탈 완료 여부
    breakout_pct = (current / neck - 1.0) * 100.0 if neck else 0.0

    # 이탈 봉 탐색 (제2봉 이후 최초 종가 이탈)
    breakout_pos = None
    for t in range(k + 1, n):
        if float(close.iloc[t]) < neck:
            breakout_pos = t
            break

    # 거래량 평가 (고전 이론: 제2봉 감소 · 이탈 증량)
    w = max(3, (k - i) // 4)
    seg_p1 = vol.iloc[max(0, i - w): i + w + 1]
    seg_p2 = vol.iloc[max(0, k - w): k + w + 1]
    checks: List[Tuple[str, str]] = []
    ok_a = len(seg_p1) > 0 and len(seg_p2) > 0 and seg_p2.mean() < seg_p1.mean()
    checks.append(("제2봉 감소", "○" if ok_a else "×"))
    if breakout_pos is not None:
        vol_ma20 = vol.rolling(20).mean()
        ma_at = vol_ma20.iloc[breakout_pos]
        ok_b = pd.notna(ma_at) and float(vol.iloc[breakout_pos]) >= 1.5 * float(ma_at)
        checks.append(("이탈 증량", "○" if ok_b else "×"))
    else:
        checks.append(("이탈 증량", "대기"))
    vol_ok_n = sum(1 for _, v in checks if v == "○")
    vol_total = sum(1 for _, v in checks if v in ("○", "×"))

    # 측정 목표치 (measured move): 목표 = 목선 − 패턴 높이(고점 평균 − 목선)
    height = max(float(np.mean(pk_ps)) - neck, 0.0)
    target = neck - height
    target_pct = (target / current - 1.0) * 100.0 if current else 0.0
    invalid = max(pk_ps)  # 고점 상향 돌파 시 패턴 무효
    invalid_pct = (invalid / current - 1.0) * 100.0 if current else 0.0
    risk = invalid - current
    risk_reward = (current - target) / risk if risk > 0 and target < current else None
    progress_pct = ((neck - current) / height * 100.0) if (breakout and height > 0) else None
    formation_days = (pk_ds[1] - pk_ds[0]).days

    return {
        "invalidated": current > invalid,
        "target": float(target),
        "target_pct": target_pct,
        "invalid": float(invalid),
        "invalid_pct": invalid_pct,
        "risk_reward": risk_reward,
        "progress_pct": progress_pct,
        "formation_days": formation_days,
        "height": float(height),
        "pivot_dates": [pk_ds[0], v_d, pk_ds[1]],
        "pivot_prices": [pk_ps[0], v_p, pk_ps[1]],
        "labels": ["제1봉", "목선 저점", "제2봉"],
        "neck_dates": [v_d, pk_ds[1]],
        "neck_prices": [neck, neck],
        "neck_now": float(neck),
        "breakout": breakout,
        "breakout_pct": breakout_pct,
        "breakout_date": pd.Timestamp(idx[breakout_pos]) if breakout_pos is not None else None,
        "vol_checks": checks,
        "vol_ok_n": vol_ok_n,
        "vol_total": vol_total,
        "end_str": pk_ds[1].strftime("%m/%d"),
    }


def _evaluate_asc_triangle(ev: Dict[str, Any], df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    상승삼각 상세 평가: 수평 저항선·상승 지지선 추정, 돌파 상태, 거래량 채점, 측정 목표(상승).

    검출 이벤트는 구간(start~end)만 제공하므로 구간 내 고가/저가로 구조를 계산.
    저항선 = 상위 5개 고가 평균(수평), 지지선 = 저가 선형회귀(우상향).
    """
    start = pd.Timestamp(ev.get("start"))
    end = pd.Timestamp(ev.get("end") or ev.get("start"))
    idx = df.index
    i0 = int(idx.get_indexer([start], method="pad")[0])
    i1 = int(idx.get_indexer([end], method="pad")[0])
    if i0 < 0 or i1 - i0 < 8:
        return None

    n = len(df)
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    current = float(close.iloc[-1])
    hi_w = df["High"].iloc[i0:i1 + 1].astype(float).values
    lo_w = df["Low"].iloc[i0:i1 + 1].astype(float).values

    resistance = float(np.mean(np.sort(hi_w)[-5:]))
    x = np.arange(len(lo_w), dtype=float)
    sup_slope, sup_icpt = np.polyfit(x, lo_w, 1)
    sup_first = float(sup_icpt)
    sup_last = float(sup_icpt + sup_slope * x[-1])

    breakout = current > resistance
    breakout_pct = (current / resistance - 1.0) * 100.0 if resistance else 0.0
    breakout_pos = None
    for t in range(i1 + 1, n):
        if float(close.iloc[t]) > resistance:
            breakout_pos = t
            break

    # 거래량 평가 (고전 이론: 형성 후반 감소 · 돌파 증량)
    half = (i1 - i0) // 2
    seg_a = vol.iloc[i0: i0 + half + 1]
    seg_b = vol.iloc[i0 + half: i1 + 1]
    checks: List[Tuple[str, str]] = []
    ok_a = len(seg_a) > 0 and len(seg_b) > 0 and seg_b.mean() < seg_a.mean()
    checks.append(("형성 후반 감소", "○" if ok_a else "×"))
    if breakout_pos is not None:
        vol_ma20 = vol.rolling(20).mean()
        ma_at = vol_ma20.iloc[breakout_pos]
        ok_b = pd.notna(ma_at) and float(vol.iloc[breakout_pos]) >= 1.5 * float(ma_at)
        checks.append(("돌파 증량", "○" if ok_b else "×"))
    else:
        checks.append(("돌파 증량", "대기"))
    vol_ok_n = sum(1 for _, v in checks if v == "○")
    vol_total = sum(1 for _, v in checks if v in ("○", "×"))

    # 측정 목표치: 목표 = 저항선 + 패턴 높이(삼각형 좌측 최대 폭)
    height = max(resistance - sup_first, 0.0)
    target = resistance + height
    target_pct = (target / current - 1.0) * 100.0 if current else 0.0
    invalid = sup_last  # 상승 지지선 이탈 시 패턴 무효
    invalid_pct = (invalid / current - 1.0) * 100.0 if current else 0.0
    risk = current - invalid
    risk_reward = (target - current) / risk if risk > 0 and target > current else None
    progress_pct = ((current - resistance) / height * 100.0) if (breakout and height > 0) else None
    formation_days = (end - start).days

    return {
        "invalidated": current < invalid,
        "target": float(target),
        "target_pct": target_pct,
        "invalid": float(invalid),
        "invalid_pct": invalid_pct,
        "risk_reward": risk_reward,
        "progress_pct": progress_pct,
        "formation_days": formation_days,
        "height": float(height),
        "pivot_dates": [],
        "pivot_prices": [],
        "labels": [],
        "neck_dates": [idx[i0], idx[i1]],
        "neck_prices": [resistance, resistance],
        "neck_now": float(resistance),
        "aux_line": ([idx[i0], idx[i1]], [sup_first, sup_last]),
        "breakout": breakout,
        "breakout_pct": breakout_pct,
        "breakout_date": pd.Timestamp(idx[breakout_pos]) if breakout_pos is not None else None,
        "vol_checks": checks,
        "vol_ok_n": vol_ok_n,
        "vol_total": vol_total,
        "end_str": end.strftime("%m/%d"),
    }


def _evaluate_candle_range(
    ev: Dict[str, Any], df: pd.DataFrame, *, bullish: bool, label: str
) -> Optional[Dict[str, Any]]:
    """
    캔들형 패턴(엥걸핑·망치형·행잉맨) 공용 평가: 패턴봉 범위를 투영한 측정 목표.

    강세: 기준선 = 패턴봉 고가(돌파), 목표 = 고가 + 범위, 무효선 = 저가.
    약세: 기준선 = 패턴봉 저가(이탈), 목표 = 저가 − 범위, 무효선 = 고가.
    """
    d = pd.Timestamp(ev.get("start"))
    idx = df.index
    i = int(idx.get_indexer([d], method="pad")[0])
    if i < 1:
        return None

    n = len(df)
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    current = float(close.iloc[-1])
    high_c = float(df["High"].iloc[i])
    low_c = float(df["Low"].iloc[i])
    rng = high_c - low_c
    if rng <= 0:
        return None

    ref = high_c if bullish else low_c  # 돌파/이탈 기준선
    if bullish:
        breakout = current > ref
        breakout_pct = (current / ref - 1.0) * 100.0 if ref else 0.0
    else:
        breakout = current < ref
        breakout_pct = (current / ref - 1.0) * 100.0 if ref else 0.0
    breakout_pos = None
    for t in range(i + 1, n):
        c_t = float(close.iloc[t])
        if (bullish and c_t > ref) or (not bullish and c_t < ref):
            breakout_pos = t
            break

    # 거래량 평가 (패턴봉 증량 · 돌파/이탈 증량)
    word = "돌파" if bullish else "이탈"
    checks: List[Tuple[str, str]] = []
    ok_a = float(vol.iloc[i]) > float(vol.iloc[i - 1])
    checks.append((f"{label} 증량", "○" if ok_a else "×"))
    if breakout_pos is not None:
        vol_ma20 = vol.rolling(20).mean()
        ma_at = vol_ma20.iloc[breakout_pos]
        ok_b = pd.notna(ma_at) and float(vol.iloc[breakout_pos]) >= 1.5 * float(ma_at)
        checks.append((f"{word} 증량", "○" if ok_b else "×"))
    else:
        checks.append((f"{word} 증량", "대기"))
    vol_ok_n = sum(1 for _, v in checks if v == "○")
    vol_total = sum(1 for _, v in checks if v in ("○", "×"))

    # 측정 목표치: 패턴봉 범위 1R 투영
    target = ref + rng if bullish else ref - rng
    target_pct = (target / current - 1.0) * 100.0 if current else 0.0
    invalid = low_c if bullish else high_c
    invalid_pct = (invalid / current - 1.0) * 100.0 if current else 0.0
    risk = (current - invalid) if bullish else (invalid - current)
    reward = (target - current) if bullish else (current - target)
    risk_reward = reward / risk if risk > 0 and reward > 0 else None
    if breakout:
        progress_pct = ((current - ref) / rng * 100.0) if bullish else ((ref - current) / rng * 100.0)
    else:
        progress_pct = None
    invalidated = current < invalid if bullish else current > invalid

    return {
        "invalidated": invalidated,
        "target": float(target),
        "target_pct": target_pct,
        "invalid": float(invalid),
        "invalid_pct": invalid_pct,
        "risk_reward": risk_reward,
        "progress_pct": progress_pct,
        "formation_days": 2,
        "height": float(rng),
        "pivot_dates": [idx[i]],
        "pivot_prices": [float(close.iloc[i])],
        "labels": [label],
        "neck_dates": [idx[i], idx[-1]],
        "neck_prices": [ref, ref],
        "neck_now": float(ref),
        "breakout": breakout,
        "breakout_pct": breakout_pct,
        "breakout_date": pd.Timestamp(idx[breakout_pos]) if breakout_pos is not None else None,
        "vol_checks": checks,
        "vol_ok_n": vol_ok_n,
        "vol_total": vol_total,
        "end_str": d.strftime("%m/%d"),
    }


def _evaluate_flag(ev: Dict[str, Any], df: pd.DataFrame, *, bullish: bool) -> Optional[Dict[str, Any]]:
    """
    깃발형 패턴(상승/하락깃발) 평가: 깃대 높이를 깃발 돌파/이탈 지점에 투영한 측정 목표.

    검출 이벤트의 start(급등/급락 종료 = 깃발 시작)~end(깃발 끝)를 사용하고,
    깃대는 깃발 시작 직전 약 18봉 구간의 종가 변화로 계산.
    """
    start = pd.Timestamp(ev.get("start"))
    end = pd.Timestamp(ev.get("end") or ev.get("start"))
    idx = df.index
    i0 = int(idx.get_indexer([start], method="pad")[0])
    i1 = int(idx.get_indexer([end], method="pad")[0])
    if i0 < 0 or i1 <= i0:
        return None

    n = len(df)
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    current = float(close.iloc[-1])
    flag_hi = float(df["High"].iloc[i0:i1 + 1].max())
    flag_lo = float(df["Low"].iloc[i0:i1 + 1].min())

    pole_start = max(0, i0 - 18)
    pole = abs(float(close.iloc[i0]) - float(close.iloc[pole_start]))
    if pole <= 0:
        return None

    ref = flag_hi if bullish else flag_lo  # 깃발 상단 돌파 / 하단 이탈 기준선
    if bullish:
        breakout = current > ref
    else:
        breakout = current < ref
    breakout_pct = (current / ref - 1.0) * 100.0 if ref else 0.0
    breakout_pos = None
    for t in range(i1 + 1, n):
        c_t = float(close.iloc[t])
        if (bullish and c_t > ref) or (not bullish and c_t < ref):
            breakout_pos = t
            break

    # 거래량 평가 (깃발 구간 감소 · 돌파/이탈 증량)
    word = "돌파" if bullish else "이탈"
    checks: List[Tuple[str, str]] = []
    seg_pole = vol.iloc[pole_start: i0 + 1]
    seg_flag = vol.iloc[i0: i1 + 1]
    ok_a = len(seg_pole) > 0 and len(seg_flag) > 0 and seg_flag.mean() < seg_pole.mean()
    checks.append(("깃발 구간 감소", "○" if ok_a else "×"))
    if breakout_pos is not None:
        vol_ma20 = vol.rolling(20).mean()
        ma_at = vol_ma20.iloc[breakout_pos]
        ok_b = pd.notna(ma_at) and float(vol.iloc[breakout_pos]) >= 1.5 * float(ma_at)
        checks.append((f"{word} 증량", "○" if ok_b else "×"))
    else:
        checks.append((f"{word} 증량", "대기"))
    vol_ok_n = sum(1 for _, v in checks if v == "○")
    vol_total = sum(1 for _, v in checks if v in ("○", "×"))

    # 측정 목표치: 깃대 높이 투영
    target = ref + pole if bullish else ref - pole
    target_pct = (target / current - 1.0) * 100.0 if current else 0.0
    invalid = flag_lo if bullish else flag_hi  # 깃발 반대편 이탈 시 무효
    invalid_pct = (invalid / current - 1.0) * 100.0 if current else 0.0
    risk = (current - invalid) if bullish else (invalid - current)
    reward = (target - current) if bullish else (current - target)
    risk_reward = reward / risk if risk > 0 and reward > 0 else None
    if breakout:
        progress_pct = ((current - ref) / pole * 100.0) if bullish else ((ref - current) / pole * 100.0)
    else:
        progress_pct = None
    invalidated = current < invalid if bullish else current > invalid

    return {
        "invalidated": invalidated,
        "target": float(target),
        "target_pct": target_pct,
        "invalid": float(invalid),
        "invalid_pct": invalid_pct,
        "risk_reward": risk_reward,
        "progress_pct": progress_pct,
        "formation_days": (end - start).days,
        "height": float(pole),
        "pivot_dates": [],
        "pivot_prices": [],
        "labels": [],
        "neck_dates": [idx[i0], idx[i1]],
        "neck_prices": [ref, ref],
        "neck_now": float(ref),
        "aux_line": ([idx[pole_start], idx[i0]], [float(close.iloc[pole_start]), float(close.iloc[i0])]),
        "breakout": breakout,
        "breakout_pct": breakout_pct,
        "breakout_date": pd.Timestamp(idx[breakout_pos]) if breakout_pos is not None else None,
        "vol_checks": checks,
        "vol_ok_n": vol_ok_n,
        "vol_total": vol_total,
        "end_str": end.strftime("%m/%d"),
    }


def _plot_double_top_detail(ax, dt: Dict[str, Any], last_ts, fmt_price) -> None:
    """쌍봉(M) 전용 렌더: 고점/저점 마커 + 수평 목선(연장) + 상태 라벨."""
    col = "#c0392b"
    neck_col = "#d35400"

    for ts, price, lbl in zip(dt["pivot_dates"], dt["pivot_prices"], dt["labels"]):
        above = price > dt["neck_now"]
        ax.scatter([ts], [price], s=60, marker="o", facecolor="none",
                   edgecolor=col, linewidth=1.7, zorder=18)
        ax.annotate(
            lbl,
            xy=(ts, price),
            xytext=(0, 8 if above else -17),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            color=col,
            ha="center",
            zorder=21,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=col, alpha=0.9, linewidth=0.6),
        )

    nds, nps = dt["neck_dates"], dt["neck_prices"]
    ax.plot(nds, nps, color=neck_col, linestyle="--", linewidth=1.7, alpha=0.95, zorder=17)
    ax.plot([nds[1], last_ts], [nps[1], dt["neck_now"]], color=neck_col,
            linestyle=":", linewidth=1.3, alpha=0.7, zorder=16)

    if dt["breakout"]:
        status = f"이탈 완료 ({dt['breakout_pct']:+.1f}%)"
    else:
        status = f"이탈 대기 (목선까지 {-dt['breakout_pct']:+.1f}%)"
    mid_x = nds[0] + (nds[1] - nds[0]) / 2
    ax.annotate(
        f"쌍봉(M) 목선 · {status}",
        xy=(mid_x, nps[0]),
        xytext=(0, -14),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold",
        color=neck_col,
        ha="center",
        zorder=21,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=neck_col, alpha=0.9, linewidth=0.7),
    )


def _plot_triple_bottom_detail(ax, tb: Dict[str, Any], last_ts, y_span: float, fmt_price) -> None:
    """역삼중창 전용 렌더: 어깨/머리 마커 + 넥라인(연장) + 상태 라벨."""
    col = "#16a085"
    neck_col = "#c0392b"

    for ts, price, lbl in zip(tb["pivot_dates"], tb["pivot_prices"], tb["labels"]):
        ax.scatter([ts], [price], s=60, marker="o", facecolor="none",
                   edgecolor=col, linewidth=1.7, zorder=18)
        ax.annotate(
            lbl,
            xy=(ts, price),
            xytext=(0, -17),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            color=col,
            ha="center",
            zorder=21,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=col, alpha=0.9, linewidth=0.6),
        )

    nds, nps = tb["neck_dates"], tb["neck_prices"]
    ax.plot(nds, nps, color=neck_col, linestyle="--", linewidth=1.7, alpha=0.95, zorder=17)
    ax.scatter(nds, nps, color=neck_col, s=26, marker="^", edgecolors="white", linewidths=0.4, zorder=18)
    ax.plot([nds[1], last_ts], [nps[1], tb["neck_now"]], color=neck_col,
            linestyle=":", linewidth=1.3, alpha=0.7, zorder=16)

    if tb["breakout"]:
        status = f"돌파 완료 ({tb['breakout_pct']:+.1f}%)"
    else:
        status = f"돌파 대기 (넥라인까지 {-tb['breakout_pct']:+.1f}%)"
    mid_x = nds[0] + (nds[1] - nds[0]) / 2
    mid_y = (nps[0] + nps[1]) / 2.0
    ax.annotate(
        f"역삼중창(W) 넥라인 · {tb['slope_cls']} · {status}",
        xy=(mid_x, mid_y),
        xytext=(0, 10),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold",
        color=neck_col,
        ha="center",
        zorder=21,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=neck_col, alpha=0.9, linewidth=0.7),
    )


def _plot_pattern_target_mini(
    ax,
    df: pd.DataFrame,
    tb: Dict[str, Any],
    current: float,
    fmt_price,
    score: Optional[int],
    *,
    col: str = "#16a085",
    neck_col: str = "#c0392b",
    title_base: str = "역삼중창(W) 패턴 목표 ⑤",
    direction: str = "up",
    guide: Optional[Dict[str, Any]] = None,
    ax_vol: Any = None,
    ax_cap: Any = None,
) -> None:
    """미니 차트: 가격(측정목표) + 거래량 스트립 + 전용 캡션 축(겹침 방지)."""
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)

    # 표시 구간: 첫 피벗(없으면 기준선 시작) 5봉 전 ~ 마지막 봉
    start_anchor = tb["pivot_dates"][0] if tb["pivot_dates"] else tb["neck_dates"][0]
    start_pos = max(0, int(df.index.get_indexer([start_anchor], method="pad")[0]) - 5)
    seg = close.iloc[start_pos:]
    vol_seg = vol.iloc[start_pos:]
    x0 = mdates.date2num(pd.Timestamp(seg.index[0]).to_pydatetime())
    x_last = mdates.date2num(pd.Timestamp(seg.index[-1]).to_pydatetime())
    x_proj = x_last + (x_last - x0) * 0.22  # 우측 투영 여백

    is_fvg = tb.get("gap_low") is not None and tb.get("gap_high") is not None
    ohlc_seg = df.iloc[start_pos:] if is_fvg else None
    if is_fvg and ohlc_seg is not None and len(ohlc_seg):
        # FVG는 3봉 구조이므로 캔들로 표시 (갭 시각 확인)
        quads = [
            [mdates.date2num(pd.Timestamp(d).to_pydatetime()), float(o), float(h), float(l), float(c)]
            for d, (o, h, l, c) in zip(
                ohlc_seg.index, ohlc_seg[["Open", "High", "Low", "Close"]].values
            )
        ]
        candlestick_ohlc(ax, quads, width=0.55, colorup="#27ae60", colordown="#c0392b", alpha=0.92)
        # 형성 3봉 구간 강조
        if len(tb.get("neck_dates", [])) >= 2:
            t_a = mdates.date2num(pd.Timestamp(tb["neck_dates"][0]).to_pydatetime())
            t_b = mdates.date2num(pd.Timestamp(tb["neck_dates"][-1]).to_pydatetime())
            if t_b < t_a:
                t_a, t_b = t_b, t_a
            ax.axvspan(t_a - 0.4, t_b + 0.4, color=col, alpha=0.08, zorder=2)
    else:
        ax.plot(seg.index, seg.values, color="#5d6d7e", linewidth=1.0, alpha=0.95, zorder=5)

    # FVG 갭 존 음영
    if is_fvg and len(tb.get("neck_dates", [])) >= 2:
        ax.axhspan(tb["gap_low"], tb["gap_high"], color=col, alpha=0.18, zorder=3)
        fvg_tag = "FVG↓" if direction == "down" else "FVG↑"
        ax.annotate(
            fvg_tag,
            xy=(tb["neck_dates"][0], (tb["gap_low"] + tb["gap_high"]) / 2.0),
            fontsize=7, color=col, fontweight="bold", ha="left", va="center", zorder=12,
        )

    # 피벗 마커 (목선 위 피벗은 라벨 위, 아래 피벗은 라벨 아래)
    for ts, price, lbl in zip(tb["pivot_dates"], tb["pivot_prices"], tb["labels"]):
        above = price > tb["neck_now"]
        ax.scatter([ts], [price], s=34, marker="o", facecolor="none",
                   edgecolor=col, linewidth=1.4, zorder=10)
        ax.annotate(lbl, xy=(ts, price), xytext=(0, 4 if above else -10), textcoords="offset points",
                    fontsize=6.2, fontweight="bold", color=col, ha="center", zorder=12)

    # 넥라인(기준선) + 연장
    nds, nps = tb["neck_dates"], tb["neck_prices"]
    ax.plot(nds, nps, color=neck_col, linestyle="--", linewidth=1.4, alpha=0.95, zorder=9)
    ax.plot([nds[1], seg.index[-1]], [nps[1], tb["neck_now"]], color=neck_col,
            linestyle=":", linewidth=1.1, alpha=0.7, zorder=8)

    # 보조선 (예: 상승삼각의 상승 지지선)
    aux = tb.get("aux_line")
    if aux:
        ax.plot(aux[0], aux[1], color=neck_col, linestyle="-.", linewidth=1.2, alpha=0.8, zorder=8)

    # 목표·무효선 + 우측 가격 라벨 (범례는 미니별, 라벨은 우측으로 분리)
    target = tb["target"]
    expired = bool(tb.get("invalidated")) or (
        tb.get("progress_pct") is not None and tb["progress_pct"] >= 100.0
    )
    ax.plot([x_last, x_proj], [target, target], color=col, linestyle="--",
            linewidth=1.3, alpha=0.85, zorder=10)
    invalid = tb["invalid"]
    ax.plot([x_last, x_proj], [invalid, invalid], color="#e74c3c", linestyle=":",
            linewidth=1.2, alpha=0.85, zorder=10)
    if not expired:
        ax.annotate(
            "",
            xy=((x_last + x_proj) / 2.0, target),
            xytext=(x_last, current),
            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2, linestyle=":"),
            zorder=10,
        )

    # 현재가 마커
    ax.scatter([seg.index[-1]], [current], s=30, marker="o", color="#2c3e50",
               edgecolors="white", linewidths=0.6, zorder=11)

    # 매수(▲) / 매도(▼) 타점 — 글자 대신 마커 + 미니별 범례
    g = guide or {}
    buy_px = g.get("buy_px")
    sell_px = g.get("sell_px")
    x_mark = x_last + (x_proj - x_last) * 0.42
    sell_near_target = (
        sell_px is not None
        and abs(float(sell_px) - float(target)) / max(abs(float(target)), 1e-9) < 0.003
    )
    if buy_px is not None:
        ax.plot([x_last, x_proj], [buy_px, buy_px], color="#27ae60", linestyle="-.",
                linewidth=1.0, alpha=0.55, zorder=11)
        ax.scatter([x_mark], [buy_px], s=55, marker="^", color="#27ae60",
                   edgecolors="white", linewidths=0.6, zorder=14)
    if sell_px is not None:
        ax.scatter([x_mark], [sell_px], s=55, marker="v", color="#c0392b",
                   edgecolors="white", linewidths=0.6, zorder=14)
        if not sell_near_target:
            ax.plot([x_last, x_proj], [sell_px, sell_px], color="#c0392b", linestyle="-.",
                    linewidth=1.0, alpha=0.55, zorder=11)

    ax.set_xlim(x0, x_proj)
    if is_fvg and ohlc_seg is not None and len(ohlc_seg):
        y_vals = [float(ohlc_seg["Low"].min()), float(ohlc_seg["High"].max()), target, invalid]
    else:
        y_vals = [float(seg.min()), float(seg.max()), target, invalid]
    if tb.get("gap_low") is not None:
        y_vals.extend([tb["gap_low"], tb["gap_high"]])
    if buy_px is not None:
        y_vals.append(float(buy_px))
    if sell_px is not None:
        y_vals.append(float(sell_px))
    y_lo, y_hi = min(y_vals), max(y_vals)
    pad = (y_hi - y_lo) * 0.14 or 1.0
    ax.set_ylim(y_lo - pad, y_hi + pad)

    # 목표/무효 가격 라벨 — 우측, 간격 좁으면 위·아래로 벌림 (좌측 범례와 분리)
    y_rng = (y_hi - y_lo) or 1.0
    tgt_dy = 5 if target >= invalid else -7
    inv_dy = -7 if target >= invalid else 5
    if abs(target - invalid) / y_rng < 0.10:
        tgt_dy = 9 if target >= invalid else -10
        inv_dy = -10 if target >= invalid else 9
    tgt_txt = f"목표 {fmt_price(target)}"
    if sell_near_target:
        tgt_txt = f"목표=▼ {fmt_price(target)}"
    _annotate_level_price(ax, x_proj, target, tgt_txt, col, dy=tgt_dy)
    _annotate_level_price(
        ax, x_proj, invalid, f"무효 {fmt_price(invalid)}", "#e74c3c", dy=inv_dy,
    )

    # 제목: 상태 + 액션 배지
    title = title_base
    status = g.get("status")
    if status == "무효화":
        title += " · 무효화"
    elif status == "달성":
        title += " · 목표 달성"
    elif tb["breakout"] and tb.get("progress_pct") is not None:
        if tb["progress_pct"] >= 0:
            title += f" · 달성률 {tb['progress_pct']:.0f}%"
        else:
            title += " · 돌파 후 되돌림" if direction == "up" else " · 이탈 후 되돌림"
    action = g.get("action")
    action_colors = {"매수": "#27ae60", "매도": "#c0392b", "유지": "#7f8c8d"}
    if action:
        title += f"  [{action}]"
    ax.set_title(title, fontsize=8.0, fontweight="bold", color=action_colors.get(action, col), pad=3)

    ax.grid(True, alpha=0.2, linestyle=":")
    ax.tick_params(axis="y", labelsize=6.5)
    # 날짜는 거래량축에만 — 가격축 라벨·틱 라벨 완전 숨김
    ax.tick_params(axis="x", labelbottom=False, length=0)
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), visible=False)

    # ---------- 거래량 스트립 (신뢰 요인 시각화) ----------
    has_vol_surge = False
    has_impulse_vol = False
    has_break_vol = False
    if ax_vol is not None and len(vol_seg):
        vol_ma = vol.rolling(20).mean()
        colors = []
        for ts, v in vol_seg.items():
            ma = vol_ma.loc[ts] if ts in vol_ma.index else np.nan
            if pd.notna(ma) and float(v) >= 1.5 * float(ma):
                colors.append("#e67e22")  # 급증
                has_vol_surge = True
            elif pd.notna(ma) and float(v) >= 1.2 * float(ma):
                colors.append("#f39c12")
            else:
                colors.append("#bdc3c7")
        ax_vol.bar(vol_seg.index, vol_seg.values / 1e6, color=colors, width=0.8, alpha=0.85, zorder=3)
        ma_seg = vol_ma.reindex(vol_seg.index)
        ax_vol.plot(ma_seg.index, ma_seg.values / 1e6, color="#34495e", linewidth=0.9, alpha=0.85, zorder=4)

        # 임펄스(패턴 중앙) · 돌파봉 강조
        if len(tb.get("neck_dates", [])) >= 2:
            mid_ts = tb["neck_dates"][0] + (tb["neck_dates"][-1] - tb["neck_dates"][0]) / 2
            mid_i = int(df.index.get_indexer([mid_ts], method="nearest")[0])
            if 0 <= mid_i < len(df):
                mts = df.index[mid_i]
                if mts in vol_seg.index:
                    ax_vol.bar([mts], [float(vol.loc[mts]) / 1e6], color=col, width=0.9, alpha=0.95, zorder=6)
                    has_impulse_vol = True
        bdt = tb.get("breakout_date")
        if bdt is not None:
            bts = pd.Timestamp(bdt)
            bi = int(df.index.get_indexer([bts], method="nearest")[0])
            if 0 <= bi < len(df) and df.index[bi] in vol_seg.index:
                bts = df.index[bi]
                ax_vol.bar([bts], [float(vol.loc[bts]) / 1e6], color="#8e44ad", width=0.9, alpha=0.95, zorder=6)
                has_break_vol = True

        # 거래량 조건 요약 배지 (임펄스=패턴색, 돌파=보라, 급증=주황)
        vok, vtot = tb.get("vol_ok_n"), tb.get("vol_total")
        badge = f"Vol {vok}/{vtot}" if vtot else "Vol"
        ax_vol.text(
            0.98, 0.92, badge,
            transform=ax_vol.transAxes, fontsize=6.0, fontweight="bold",
            color="#27ae60" if vtot and (vok or 0) >= max(1, (vtot + 1) // 2) else "#c0392b",
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="#95a5a6", linewidth=0.5, alpha=0.9),
        )
        ax_vol.set_xlim(x0, x_proj)
        ax_vol.set_ylabel("Vol(M)", fontsize=5.5, color="#7f8c8d", labelpad=1)
        ax_vol.yaxis.set_major_formatter(FuncFormatter(lambda v, _p=None: f"{v:.0f}"))
        _kill_axis_offsets(ax_vol)
        ax_vol.tick_params(axis="x", labelsize=5.0, pad=1.5, length=2.5, labelbottom=True)
        ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax_vol.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=3))
        for lbl in ax_vol.get_xticklabels():
            lbl.set_clip_on(False)
        ax_vol.grid(True, alpha=0.15, linestyle=":")
    else:
        ax.tick_params(axis="x", labelbottom=True, labelsize=5.5, pad=1.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=3))

    # ---------- 캡션(제안 A): 슬림 헤더·조건1줄·가이드 + 텍스트 범례 ----------
    legend_line = _mini_legend_text(
        has_buy=buy_px is not None,
        has_sell=sell_px is not None,
        has_target=True,
        has_invalid=True,
        has_impulse=has_impulse_vol,
        has_break=has_break_vol,
        has_surge=has_vol_surge,
    )

    st = status or "—"
    bits1 = [f"[{st}]"]
    if status == "진행" and tb.get("progress_pct") is not None and tb["progress_pct"] >= 0:
        bits1.append(f"달성{tb['progress_pct']:.0f}%")
    elif status == "달성":
        bits1.append(
            "사후○" if (
                (direction == "up" and current >= tb["target"] * 0.98)
                or (direction == "down" and current <= tb["target"] * 1.02)
            ) else "사후×"
        )
    if score is not None:
        bits1.append(_fmt_confidence_score(score, short=True))
    if tb.get("risk_reward") is not None:
        bits1.append(_fmt_risk_reward(tb["risk_reward"], short=True))
    # 목표/무효 숫자는 차트 우측 라벨만 사용 (캡션 중복 제거)
    line1 = _join_caption_parts(bits1, max_len=42)

    pos = g.get("positives") or []
    neg = g.get("negatives") or []
    checks = tb.get("vol_checks") or []
    chk_bits = [f"{n}{v}" for n, v in checks]
    if status == "무효화":
        cond_parts = (neg + chk_bits)[:2] or ["—"]
        line2 = _join_caption_parts(["부정"] + cond_parts, sep=" · ", max_len=42)
    else:
        cond_parts = (list(pos[:1]) + chk_bits)[:2] or ["—"]
        line2 = _join_caption_parts(["긍정"] + cond_parts, sep=" · ", max_len=42)
        if neg and status == "진행":
            # 주의는 같은 줄에 짧게만 (줄바꿈으로 겹침 방지)
            line2 = _join_caption_parts(
                ["긍정"] + cond_parts[:1] + [f"주의:{neg[0]}"], sep=" · ", max_len=42,
            )

    weak_tag = "(약)" if g.get("weak") and action == "유지" else ""
    act = action or "유지"
    note = g.get("action_note") or ""
    stop_px = g.get("stop_px")
    guide_bits = [f"가이드:{act}{weak_tag}"]
    if note:
        guide_bits.append(note)
    if buy_px is not None:
        guide_bits.append(f"▲{_fmt_px_plain(buy_px)}")
    if sell_px is not None:
        guide_bits.append(f"▼{_fmt_px_plain(sell_px)}")
    if stop_px is not None:
        guide_bits.append(f"손절{_fmt_px_plain(stop_px)}")
    line3 = _join_caption_parts(guide_bits, sep=" · ", max_len=42)
    why_raw = (g.get("action_why") or "").strip()
    line4 = _join_caption_parts([why_raw], max_len=46) if why_raw else ""

    if ax_cap is not None:
        _draw_mini_caption_card(
            ax_cap, line1, line2, line3,
            action=act,
            legend_line=legend_line or None,
            reason=line4 or None,
        )
    else:
        ax.text(
            0.5, -0.18, "\n".join(x for x in (legend_line, line1, line2, line3, line4) if x),
            transform=ax.transAxes, fontsize=5.8, fontweight="bold",
            color="#34495e", ha="center", va="top", linespacing=1.1, clip_on=False,
        )


def _plot_pattern_span_on_main(
    ax,
    df: pd.DataFrame,
    ev: Dict[str, Any],
    *,
    x_left: float,
    x_right: float,
    y_min_all: float,
    y_max_all: float,
    y_span: float,
    span_lane: int,
    status: Optional[str] = None,
    tag: Optional[str] = None,
) -> int:
    """
    메인 차트 패턴 식별 표기 (시나리오 번호 없음).
    - 연한 구간 음영 + 짧은 패턴명 태그 (예: 쌍봉, FVG)
    - 무효/달성은 태그 옆 ×/✓
    반환: 사용한 상단 레인 수(0 또는 1).
    """
    pid = ev["pattern_id"]
    crit = CHART_PATTERN_CRITERIA[ID_TO_ROW[pid]]
    col = crit["color"]
    raw = (tag or crit.get("label") or pid)[:6]
    mark = ""
    if status == "무효화":
        mark = "×"
    elif status == "달성":
        mark = "✓"
    label_txt = f"{raw}{mark}"
    expired = status in ("무효화", "달성")
    start = pd.Timestamp(ev.get("start"))
    end = pd.Timestamp(ev.get("end") or ev.get("start"))
    i0 = int(df.index.get_indexer([start], method="pad")[0])
    i1 = int(df.index.get_indexer([end], method="pad")[0])
    if i0 < 0 or i1 < 0:
        return 0

    t0 = mdates.date2num(start.to_pydatetime())
    t1 = mdates.date2num(end.to_pydatetime())
    if t1 < t0:
        t0, t1 = t1, t0
    sign = PATTERN_IMPLICATION.get(pid, ("", 0))[1]
    if pid == "fvg_gap":
        d = ev.get("direction")
        if d == "bull":
            sign = 1
        elif d == "bear":
            sign = -1
    short = pid in CANDLE_PATTERN_IDS or (i1 - i0) <= 1
    long_span = (t1 - t0) > (x_right - x_left) * 0.45
    label_kw = dict(
        fontsize=7.0,
        fontweight="bold",
        color="#7f8c8d" if expired else col,
        ha="center",
        zorder=18,
        alpha=0.55 if expired else 0.80,
        bbox=dict(
            boxstyle="round,pad=0.15",
            facecolor="white",
            edgecolor="#95a5a6" if expired else col,
            linewidth=0.7,
            alpha=0.85,
        ),
    )

    if short:
        price = float(df["Low"].iloc[i1]) if sign > 0 else float(df["High"].iloc[i1])
        tick = y_span * 0.045
        if sign > 0:
            y_num, va = price - tick * 1.25, "top"
        else:
            y_num, va = price + tick * 1.25, "bottom"
        ax.annotate(
            label_txt,
            xy=(t1, price),
            xytext=(t1, y_num),
            textcoords="data",
            va=va,
            arrowprops=dict(
                arrowstyle="-",
                color=col,
                lw=0.65,
                linestyle=":",
                alpha=0.55,
                shrinkA=0,
                shrinkB=1,
            ),
            **label_kw,
        )
        return 0

    ax.axvspan(start, end, color=col, alpha=0.06, zorder=2)
    t_mid = (t0 + t1) / 2.0
    seg = df.iloc[i0: i1 + 1]
    seg_hi = float(seg["High"].max())
    seg_lo = float(seg["Low"].min())
    # 음영 구간 끝점 표시(점선 브라켓) — 봉 위에 올리지 않음
    bracket_y = seg_hi + y_span * 0.012 if sign <= 0 else seg_lo - y_span * 0.012
    ax.plot([t0, t1], [bracket_y, bracket_y], color=col, linewidth=0.9, alpha=0.45, zorder=14, linestyle=":")
    for xe in (t0, t1):
        ax.plot(
            [xe, xe],
            [bracket_y - y_span * 0.008, bracket_y + y_span * 0.008],
            color=col, linewidth=0.9, alpha=0.45, zorder=14,
        )

    if long_span:
        # 상단 전용 레인: 봉·음영과 분리, 구간 중앙→캡션 점선
        y_br = y_max_all + y_span * (0.055 + 0.065 * span_lane)
        anchor_y = seg_hi
        ax.annotate(
            label_txt,
            xy=(t_mid, anchor_y),
            xytext=(t_mid, y_br),
            textcoords="data",
            va="bottom",
            arrowprops=dict(
                arrowstyle="-",
                color=col,
                lw=0.75,
                linestyle=":",
                alpha=0.60,
                shrinkA=0,
                shrinkB=2,
            ),
            **label_kw,
        )
        return 1

    # 중기 구간: 봉 위/아래 여백에 callout (음영 중앙에서 연결)
    if sign > 0:
        y_br = seg_lo - y_span * 0.065
        va = "top"
        anchor_y = seg_lo
    else:
        y_br = seg_hi + y_span * 0.065
        va = "bottom"
        anchor_y = seg_hi
    ax.annotate(
        label_txt,
        xy=(t_mid, anchor_y),
        xytext=(t_mid, y_br),
        textcoords="data",
        va=va,
        arrowprops=dict(
            arrowstyle="-",
            color=col,
            lw=0.75,
            linestyle=":",
            alpha=0.60,
            shrinkA=0,
            shrinkB=2,
        ),
        **label_kw,
    )
    return 0


def _plot_generic_pattern_mini(
    ax, df: pd.DataFrame, ev: Dict[str, Any], num: str,
    *, ax_vol: Any = None, ax_cap: Any = None,
) -> None:
    """하단 미니 차트: 일반 패턴 구간 확대 + 거래량 스트립 + 전용 캡션."""
    pid = ev["pattern_id"]
    crit = CHART_PATTERN_CRITERIA[ID_TO_ROW[pid]]
    col = crit["color"]
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)

    start = pd.Timestamp(ev.get("start"))
    end = pd.Timestamp(ev.get("end") or ev.get("start"))
    i0 = max(0, int(df.index.get_indexer([start], method="pad")[0]))
    i1 = max(i0, int(df.index.get_indexer([end], method="pad")[0]))
    a = max(0, i0 - 6)
    b = min(len(df) - 1, i1 + 12)
    seg = close.iloc[a:b + 1]
    vol_seg = vol.iloc[a:b + 1]

    ax.plot(seg.index, seg.values, color="#5d6d7e", linewidth=1.0, alpha=0.95, zorder=5)
    ax.axvspan(df.index[i0], df.index[i1], color=col, alpha=0.10, zorder=2)
    span_close = close.iloc[i0:i1 + 1]
    ax.plot(span_close.index, span_close.values, color=col, linewidth=1.7, alpha=0.95, zorder=8)
    if pid in CANDLE_PATTERN_IDS or i1 - i0 <= 1:
        ax.scatter([df.index[i1]], [float(close.iloc[i1])], s=46, marker="o",
                   facecolor="none", edgecolor=col, linewidth=1.7, zorder=10)

    conf = float(ev.get("confidence", 0))
    impl_txt, sign = PATTERN_IMPLICATION.get(pid, ("", 0))
    ax.set_title(f"{num} {crit['label']} · {end.strftime('%m/%d')}",
                 fontsize=8.0, fontweight="bold", color=col, pad=3)
    ax.grid(True, alpha=0.2, linestyle=":")
    ax.tick_params(axis="y", labelsize=6.5)
    ax.tick_params(axis="x", labelbottom=False, length=0)
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), visible=False)

    surge = False
    if ax_vol is not None and len(vol_seg):
        vol_ma = vol.rolling(20).mean()
        colors = [
            "#e67e22" if (pd.notna(vol_ma.loc[ts]) and float(v) >= 1.5 * float(vol_ma.loc[ts]))
            else ("#f39c12" if (pd.notna(vol_ma.loc[ts]) and float(v) >= 1.2 * float(vol_ma.loc[ts])) else "#bdc3c7")
            for ts, v in vol_seg.items()
        ]
        ax_vol.bar(vol_seg.index, vol_seg.values / 1e6, color=colors, width=0.8, alpha=0.85, zorder=3)
        ax_vol.plot(vol_seg.index, vol_ma.reindex(vol_seg.index).values / 1e6,
                    color="#34495e", linewidth=0.9, alpha=0.85, zorder=4)
        ax_vol.axvspan(df.index[i0], df.index[i1], color=col, alpha=0.12, zorder=2)
        pat_vol = vol.iloc[i0:i1 + 1]
        ma_at = vol_ma.iloc[i1] if i1 < len(vol_ma) else np.nan
        surge = pd.notna(ma_at) and float(pat_vol.mean()) >= 1.2 * float(ma_at)
        ax_vol.text(
            0.98, 0.92, "구간증량○" if surge else "구간증량×",
            transform=ax_vol.transAxes, fontsize=6.0, fontweight="bold",
            color="#27ae60" if surge else "#c0392b", ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="#95a5a6", linewidth=0.5, alpha=0.9),
        )
        ax_vol.set_ylabel("Vol(M)", fontsize=5.5, color="#7f8c8d", labelpad=1)
        ax_vol.yaxis.set_major_formatter(FuncFormatter(lambda v, _p=None: f"{v:.0f}"))
        _kill_axis_offsets(ax_vol)
        ax_vol.tick_params(axis="x", labelsize=5.0, pad=1.5, length=2.5, labelbottom=True)
        ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax_vol.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=3))
        for lbl in ax_vol.get_xticklabels():
            lbl.set_clip_on(False)
        ax_vol.grid(True, alpha=0.15, linestyle=":")
    else:
        ax.tick_params(axis="x", labelbottom=True, labelsize=5.5, pad=1.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=3))

    legend_line = _mini_legend_text(pattern_span=True, has_surge=bool(ax_vol is not None and surge))
    if ax_vol is not None and not surge:
        legend_line = _mini_legend_text(pattern_span=True) + "  ·Vol보통"

    if i1 >= len(df) - 1:
        after = "진행 중"
    else:
        base = float(close.iloc[i1])
        ret = (float(close.iloc[-1]) / base - 1.0) * 100.0 if base else 0.0
        chk = ""
        if sign != 0 and abs(ret) >= 1.0:
            chk = " ○" if sign * ret > 0 else " ×"
        after = f"이후 {ret:+.1f}%{chk}"
    line1 = _join_caption_parts([f"[일반] {_fmt_confidence_score(conf, short=True)}"], max_len=42)
    line2 = _join_caption_parts([f"함의: {impl_txt or '—'}"], max_len=42)
    line3 = _join_caption_parts(["가이드:유지", after], max_len=42)
    if ax_cap is not None:
        _draw_mini_caption_card(
            ax_cap, line1, line2, line3, action="유지", legend_line=legend_line or None,
        )
    else:
        ax.set_xlabel(f"{line1}\n{line2}\n{line3}", fontsize=6.0, fontweight="bold", color="#34495e")


def _evaluate_fvg_gap(ev: Dict[str, Any], df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    FVG(Fair Value Gap) 상세 평가.

    상방 갭(bull): 갭 존이 지지 후보. 목표 = 갭상단 + 갭높이, 무효선 = 갭하단(완전 채움 이탈).
    하방 갭(bear): 갭 존이 저항 후보. 목표 = 갭하단 − 갭높이, 무효선 = 갭상단.
    CE(중앙값)를 기준선(넥라인 역할)으로 표시.
    """
    gl = ev.get("gap_low")
    gh = ev.get("gap_high")
    if gl is None or gh is None:
        return None
    gap_low, gap_high = float(gl), float(gh)
    if gap_low > gap_high:
        gap_low, gap_high = gap_high, gap_low
    height = gap_high - gap_low
    if height <= 0:
        return None

    bullish = (ev.get("direction") or "bull") != "bear"
    start = pd.Timestamp(ev.get("start"))
    end = pd.Timestamp(ev.get("end") or ev.get("start"))
    idx = df.index
    i0 = int(idx.get_indexer([start], method="pad")[0])
    i1 = int(idx.get_indexer([end], method="pad")[0])
    if i0 < 0 or i1 < 0:
        return None

    n = len(df)
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    current = float(close.iloc[-1])
    mid = (gap_low + gap_high) / 2.0  # CE (consequent encroachment)

    # 돌파 = 임펄스 방향으로 갭을 벗어나 유지
    if bullish:
        breakout = current > gap_high
        breakout_pct = (current / gap_high - 1.0) * 100.0 if gap_high else 0.0
        invalidated = current < gap_low
        target = gap_high + height
        invalid = gap_low
    else:
        breakout = current < gap_low
        breakout_pct = (current / gap_low - 1.0) * 100.0 if gap_low else 0.0
        invalidated = current > gap_high
        target = gap_low - height
        invalid = gap_high

    # 돌파/이탈 봉 탐색 (갭 형성 이후)
    breakout_pos = None
    for t in range(i1 + 1, n):
        c_t = float(close.iloc[t])
        if (bullish and c_t > gap_high) or (not bullish and c_t < gap_low):
            breakout_pos = t
            break

    # 거래량: 임펄스 봉(3봉 중 중간) 증량 · 돌파 증량
    mid_i = (i0 + i1) // 2
    checks: List[Tuple[str, str]] = []
    vol_ma20 = vol.rolling(20).mean()
    ma_mid = vol_ma20.iloc[mid_i] if mid_i < n else None
    ok_a = pd.notna(ma_mid) and float(vol.iloc[mid_i]) >= 1.2 * float(ma_mid)
    checks.append(("임펄스 증량", "○" if ok_a else "×"))
    if breakout_pos is not None:
        ma_at = vol_ma20.iloc[breakout_pos]
        ok_b = pd.notna(ma_at) and float(vol.iloc[breakout_pos]) >= 1.5 * float(ma_at)
        checks.append(("돌파 증량", "○" if ok_b else "×"))
    else:
        checks.append(("돌파 증량", "대기"))
    # 채움 상태
    in_gap = gap_low <= current <= gap_high
    if invalidated:
        checks.append(("갭 미채움", "×"))
    elif in_gap:
        checks.append(("갭 미채움", "부분"))
    else:
        checks.append(("갭 미채움", "○"))
    vol_ok_n = sum(1 for _, v in checks if v == "○")
    vol_total = sum(1 for _, v in checks if v in ("○", "×"))

    target_pct = (target / current - 1.0) * 100.0 if current else 0.0
    invalid_pct = (invalid / current - 1.0) * 100.0 if current else 0.0
    risk = (current - invalid) if bullish else (invalid - current)
    reward = (target - current) if bullish else (current - target)
    risk_reward = reward / risk if risk > 0 and reward > 0 else None
    if breakout and height > 0:
        progress_pct = ((current - gap_high) / height * 100.0) if bullish else ((gap_low - current) / height * 100.0)
    elif in_gap and height > 0:
        # 부분 채움 진행률 (갭 진입 깊이, 음수)
        progress_pct = -((gap_high - current) / height * 100.0) if bullish else -((current - gap_low) / height * 100.0)
    else:
        progress_pct = None

    col = "#27ae60" if bullish else "#c0392b"
    neck_col = "#16a085" if bullish else "#e74c3c"

    return {
        "bullish": bullish,
        "col": col,
        "neck_col": neck_col,
        "gap_low": float(gap_low),
        "gap_high": float(gap_high),
        "invalidated": invalidated,
        "target": float(target),
        "target_pct": target_pct,
        "invalid": float(invalid),
        "invalid_pct": invalid_pct,
        "risk_reward": risk_reward,
        "progress_pct": progress_pct,
        "formation_days": max(1, (end - start).days),
        "height": float(height),
        "pivot_dates": [start, end],
        "pivot_prices": [gap_low, gap_high],
        "labels": ["갭하단", "갭상단"],
        "neck_dates": [start, end],
        "neck_prices": [mid, mid],
        "neck_now": float(mid),
        "aux_line": ([start, idx[-1]], [gap_high if bullish else gap_low, gap_high if bullish else gap_low]),
        "breakout": breakout,
        "breakout_pct": breakout_pct,
        "breakout_date": pd.Timestamp(idx[breakout_pos]) if breakout_pos is not None else None,
        "vol_checks": checks,
        "vol_ok_n": vol_ok_n,
        "vol_total": max(vol_total, 1),
        "end_str": end.strftime("%m/%d"),
        "fill_status": "완전" if invalidated else ("부분" if in_gap else "미채움"),
    }


def _pattern_expired(pe: Optional[Dict[str, Any]]) -> bool:
    """무효화(무효선 이탈) 또는 목표 달성 완료(달성률 100% 이상) 패턴 여부."""
    return pe is not None and (
        pe["invalidated"] or (pe.get("progress_pct") or 0.0) >= 100.0
    )


# 패턴별 측정 목표 분석 사양: 시나리오 번호·방향·기준선 라벨·미니 차트 색·평가 함수
PATTERN_TARGET_SPECS: Dict[str, Dict[str, Any]] = {
    "triple_bottom_w": dict(num="⑤", name="역삼중창(W)", bullish=True, break_label="넥라인",
                            col="#16a085", neck_col="#c0392b", base=40.0,
                            evaluate=_evaluate_triple_bottom_w),
    "double_top_m": dict(num="⑥", name="쌍봉(M)", bullish=False, break_label="목선",
                         col="#c0392b", neck_col="#d35400", base=40.0,
                         evaluate=_evaluate_double_top_m),
    "asc_triangle": dict(num="⑦", name="상승삼각", bullish=True, break_label="저항선",
                         col="#2980b9", neck_col="#d35400", base=40.0,
                         evaluate=_evaluate_asc_triangle),
    "bull_engulf": dict(num="⑧", name="강세엥걸핑", bullish=True, break_label="장악봉 고가",
                        col="#27ae60", neck_col="#2ecc71", base=35.0,
                        evaluate=lambda ev, df: _evaluate_candle_range(ev, df, bullish=True, label="장악봉")),
    "bear_engulf": dict(num="⑨", name="약세엥걸핑", bullish=False, break_label="장악봉 저가",
                        col="#c0392b", neck_col="#e74c3c", base=35.0,
                        evaluate=lambda ev, df: _evaluate_candle_range(ev, df, bullish=False, label="장악봉")),
    "bull_flag": dict(num="⑩", name="상승깃발", bullish=True, break_label="깃발 상단",
                      col="#27ae60", neck_col="#16a085", base=40.0,
                      evaluate=lambda ev, df: _evaluate_flag(ev, df, bullish=True)),
    "bear_flag": dict(num="⑪", name="하락깃발", bullish=False, break_label="깃발 하단",
                      col="#c0392b", neck_col="#e67e22", base=40.0,
                      evaluate=lambda ev, df: _evaluate_flag(ev, df, bullish=False)),
    "bull_hammer": dict(num="⑫", name="망치형", bullish=True, break_label="망치봉 고가",
                        col="#27ae60", neck_col="#2ecc71", base=33.0,
                        evaluate=lambda ev, df: _evaluate_candle_range(ev, df, bullish=True, label="망치봉")),
    "bear_hanging_man": dict(num="⑬", name="행잉맨", bullish=False, break_label="행잉맨 저가",
                             col="#c0392b", neck_col="#e74c3c", base=33.0,
                             evaluate=lambda ev, df: _evaluate_candle_range(ev, df, bullish=False, label="행잉맨")),
    "fvg_gap": dict(num="⑭", name="FVG(갭)", bullish=True, break_label="갭",
                    col="#8e44ad", neck_col="#9b59b6", base=36.0,
                    evaluate=_evaluate_fvg_gap),
}


def _load_ticker_note(ticker: str) -> Optional[str]:
    import os

    path = os.path.join("stocks", "notes", f"{ticker}.txt")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return None
    return None


def plot_tech_annotation_chart(
    data: pd.DataFrame,
    ticker: str,
    save_path: str,
    *,
    pattern_min_confidence: float = 0.55,
    max_patterns: int = 5,
    sr_pivot_span: int = 3,
    volume_surge_mult: float = 2.0,
    adx_period: int = 14,
    show_trendlines: bool = True,
    target_buy_price: Optional[float] = None,
    target_sell_price: Optional[float] = None,
    stop_loss_price: Optional[float] = None,
    show_target_prices: bool = True,
    show_bb: bool = False,
) -> Optional[str]:
    """기술 분석 주석 차트를 별도 PNG로 생성."""
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False

    df = _normalize(data)
    if df.empty or len(df) < 30:
        print("tech annotation: 데이터 부족")
        return None

    close = df["Close"].astype(float)
    current = float(close.iloc[-1])

    # 통화 기호
    # '$'는 matplotlib mathtext로 해석되므로 '\$'로 이스케이프
    if str(ticker).upper().endswith((".KS", ".KQ")):
        fmt_price = lambda p: f"₩{p:,.0f}"
        currency = "₩"
    else:
        fmt_price = lambda p: f"\\${p:.2f}"
        currency = "$"

    # ---------- 분석 ----------
    sr_levels = compute_support_resistance_levels(df, pivot_span=sr_pivot_span)
    lv_by_label = {lv["label"]: lv for lv in sr_levels}
    trendlines = compute_trendlines(df, pivot_span=sr_pivot_span) if show_trendlines else []

    try:
        pattern_events = analyze_seven_criteria(df)
    except Exception as e:
        print(f"tech annotation: 패턴 분석 실패 {e}")
        pattern_events = []
    top_patterns = _select_top_patterns(
        pattern_events,
        min_confidence=pattern_min_confidence,
        max_patterns=max_patterns,
        current=float(close.iloc[-1]),
    )

    # 패턴별 측정 목표 상세 평가 (레지스트리 기반)
    # pattern_evals_all: 미니·시나리오 패널용 (달성/무효 포함) — FVG는 방향별 키
    # pattern_evals: 진행 중만 (점수·추세 보조)
    pattern_evals_all: Dict[str, Dict[str, Any]] = {}
    pattern_evals: Dict[str, Dict[str, Any]] = {}
    for e in top_patterns:
        pid = e.get("pattern_id")
        spec = PATTERN_TARGET_SPECS.get(pid)
        if spec is None:
            continue
        try:
            pe = spec["evaluate"](e, df)
        except Exception as ex:
            print(f"tech annotation: 패턴 평가 실패 {pid} {ex}")
            pe = None
        if pe is None:
            continue
        slot = _pattern_slot_key(e)
        pattern_evals_all[slot] = pe
        if not _pattern_expired(pe):
            pattern_evals[slot] = pe
    tb_eval = pattern_evals_all.get("triple_bottom_w")
    dt_eval = pattern_evals_all.get("double_top_m")

    adx_res: Dict[str, Any] = {}
    try:
        adx_res = analyze_adx_dmi(df, period=adx_period)
    except Exception as e:
        print(f"tech annotation: ADX 분석 실패 {e}")

    disp_res: Dict[str, Any] = {}
    try:
        disp_res = analyze_disparity_strategy(df)
    except Exception as e:
        print(f"tech annotation: 이격도 분석 실패 {e}")

    rsi_val = _rsi14(close)

    dividend_dates: List[Dict[str, Any]] = []
    try:
        from .volume_profile_overlay_chart import get_dividend_dates

        dividend_dates = get_dividend_dates(ticker, df.index[0], df.index[-1], df) or []
    except Exception as e:
        print(f"tech annotation: 배당 조회 실패 {e}")

    note_text = _load_ticker_note(ticker)
    peg_data = None
    peg_view: Dict[str, Any] = {}
    try:
        peg_data = calculate_peg_indicator(ticker)
        peg_view = build_peg_investment_view(peg_data)
    except Exception as e:
        print(f"tech annotation: PEG 평가 실패 {e}")
        peg_view = build_peg_investment_view(None)

    # ---------- 레이아웃 ----------
    # 패턴 감지 시 하단에 패턴별 미니 차트 행(최대 4개)을 추가하고 그림 높이를 확장
    mini_patterns = top_patterns[:4]
    # 각 원소: (ax_price, ax_vol_strip, ax_caption) — 겹침 방지용 3단 분리
    mini_axes: List[Tuple[Any, Any, Any]] = []
    if mini_patterns:
        fig = plt.figure(figsize=(20, 14.5))
        # 이격도(하단) ↔ 미니차트 사이 일자 라벨 공간 확보
        ax_main = fig.add_axes([0.045, 0.535, 0.665, 0.385])
        ax_vol = fig.add_axes([0.045, 0.460, 0.665, 0.060], sharex=ax_main)
        ax_disp = fig.add_axes([0.045, 0.385, 0.665, 0.060], sharex=ax_main)
        ax_panel = fig.add_axes([0.775, 0.385, 0.215, 0.535])
        mini_w = 0.215
        mini_gap = (0.945 - 4 * mini_w) / 3.0
        for m in range(len(mini_patterns)):
            x = 0.045 + m * (mini_w + mini_gap)
            # 캡션 글씨 +30%에 맞춰 캡션 영역 소폭 확대
            ax_mp = fig.add_axes([x, 0.225, mini_w, 0.100])   # 가격 (top≈0.325 → 이격도와 ~0.06 간격)
            ax_mv = fig.add_axes([x, 0.180, mini_w, 0.036], sharex=ax_mp)  # 거래량(날짜)
            ax_mc = fig.add_axes([x, 0.052, mini_w, 0.110])   # 캡션(글씨 확대)
            mini_axes.append((ax_mp, ax_mv, ax_mc))
        ax_bar = fig.add_axes([0.045, 0.008, 0.945, 0.040])
    else:
        fig = plt.figure(figsize=(20, 11))
        ax_main = fig.add_axes([0.045, 0.345, 0.665, 0.555])
        ax_vol = fig.add_axes([0.045, 0.235, 0.665, 0.10], sharex=ax_main)
        ax_disp = fig.add_axes([0.045, 0.125, 0.665, 0.10], sharex=ax_main)
        ax_panel = fig.add_axes([0.775, 0.125, 0.215, 0.775])
        ax_bar = fig.add_axes([0.045, 0.025, 0.945, 0.07])
    for ax in (ax_panel, ax_bar):
        ax.axis("off")

    fig.suptitle(
        f"{ticker} 기술적 분석 요약  |  현재가 {fmt_price(current)}",
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )

    # ---------- 메인: 캔들 + SMA20 ----------
    quads = [
        [mdates.date2num(d.to_pydatetime()), o, h, l, c]
        for d, (o, h, l, c) in zip(df.index, df[["Open", "High", "Low", "Close"]].values)
    ]
    candlestick_ohlc(ax_main, quads, width=0.6, colorup="green", colordown="red", alpha=0.9)
    sma20 = close.rolling(20).mean()
    ax_main.plot(df.index, sma20, color="#8e44ad", linewidth=0.9, alpha=0.8, label="SMA20")

    # ---------- HMA + 만트라 밴드 ----------
    try:
        hma = calculate_hma(close)
        upper_band, lower_band = calculate_mantra_bands(close)
        ax_main.plot(df.index, hma, color="blue", linewidth=0.7, alpha=0.9, label="HMA")
        ax_main.plot(df.index, upper_band, color="red", linewidth=0.5, linestyle="--", alpha=0.8, label="Upper Mantra")
        ax_main.plot(df.index, lower_band, color="green", linewidth=0.5, linestyle="--", alpha=0.8, label="Lower Mantra")
        ax_main.fill_between(df.index, hma, upper_band, color="red", alpha=0.08)
        ax_main.fill_between(df.index, lower_band, hma, color="green", alpha=0.08)
    except Exception as e:
        print(f"tech annotation: 만트라 밴드 계산 실패 {e}")

    # ---------- 2단계: SMA200 + Supertrend / 3단계: BB 태그(·밴드) ----------
    aux_ind = _compute_aux_trend_indicators(df)
    try:
        _draw_aux_trend_overlays(ax_main, df, aux_ind)
    except Exception as e:
        print(f"tech annotation: ST/SMA200 표시 실패 {e}")
    try:
        _draw_bb_visuals(ax_main, df, aux_ind, show_bb=show_bb, fmt_price=fmt_price)
    except Exception as e:
        print(f"tech annotation: BB 표시 실패 {e}")

    x_left = mdates.date2num(pd.Timestamp(df.index[0]).to_pydatetime())
    x_right = mdates.date2num(pd.Timestamp(df.index[-1]).to_pydatetime())
    x_pad = (x_right - x_left) * 0.16  # 우측 여백: 시나리오 라벨(ha=left)용
    x_left_pad = (x_right - x_left) * 0.08  # 좌측 여백: 지지/저항 캡션 레인
    x_gutter = x_left - x_left_pad * 0.55
    ax_main.set_xlim(x_left - x_left_pad, x_right + x_pad)

    # ---------- 기간 매물대 (우측 얇은 VP + POC) ----------
    vp_info = _compute_period_volume_profile(df)
    if vp_info is not None:
        try:
            _draw_period_volume_profile(
                ax_main, vp_info, x_right=x_right, x_pad=x_pad, fmt_price=fmt_price,
            )
        except Exception as e:
            print(f"tech annotation: 매물대 표시 실패 {e}")
            vp_info = None

    # 우측 가격 태그 (겹침 방지: 기존 태그와 가격 차 0.8% 미만이면 생략)
    tag_prices: List[float] = []

    def _right_price_tag(price: float, color: str, *, force: bool = False) -> None:
        if not force and any(abs(price - t) / max(current, 1e-9) < 0.008 for t in tag_prices):
            return
        tag_prices.append(price)
        ax_main.annotate(
            fmt_price(price),
            xy=(1.0, price),
            xycoords=("axes fraction", "data"),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold",
            color="white",
            va="center",
            ha="left",
            zorder=30,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=color, edgecolor="none", alpha=0.95),
            annotation_clip=False,
        )

    # ---------- 지지/저항 수평선 + 우측 가격태그 + 좌측 여백 이름캡션 ----------
    left_caption_items: List[Dict[str, Any]] = []
    for lv in sr_levels:
        p = lv["price"]
        kind = lv["kind"]
        if kind in ("support", "strong_support"):
            style = "-"
            lw = 1.35
            alpha = 0.85
            fs = 8.6
        elif kind == "broken_support":
            style = ":"
            lw = 1.15
            alpha = 0.80
            fs = 8.2
        elif kind == "resistance":
            style = "--"
            lw = 1.2
            alpha = 0.85
            fs = 8.6
        elif kind in ("recent_support", "recent_resistance"):
            style = "-."
            lw = 0.95
            alpha = 0.70
            fs = 7.8
        else:
            style = "-"
            lw = 1.2
            alpha = 0.85
            fs = 8.4
        ax_main.axhline(p, color=lv["color"], linestyle=style, linewidth=lw, alpha=alpha, zorder=8)
        left_caption_items.append(
            {"price": float(p), "label": lv["label"], "color": lv["color"], "fontsize": fs}
        )
        _right_price_tag(p, lv["color"], force=not kind.startswith("recent"))

    # ---------- 목표가/손절가 ----------
    if show_target_prices:
        targets = [
            (target_buy_price, "매수 목표", "#2ecc71"),
            (target_sell_price, "매도 목표", "#3498db"),
            (stop_loss_price, "손절", "#e74c3c"),
        ]
        for tp, tlabel, tcolor in targets:
            if tp is None:
                continue
            ax_main.axhline(tp, color=tcolor, linestyle="-.", linewidth=1.3, alpha=0.9, zorder=9)
            left_caption_items.append(
                {"price": float(tp), "label": tlabel, "color": tcolor, "fontsize": 8.6}
            )
            _right_price_tag(float(tp), tcolor, force=True)

    try:
        _draw_left_gutter_level_captions(
            ax_main,
            left_caption_items,
            x_gutter=x_gutter,
            x_line=x_left,
            y_min=float(df["Low"].min()),
            y_max=float(df["High"].max()),
        )
    except Exception as e:
        print(f"tech annotation: 좌측 레벨 캡션 표시 실패 {e}")

    # ---------- 추세선 (활성 강조 + 현재가 관계 A+C) ----------
    y_data_min = float(df["Low"].min())
    y_data_max = float(df["High"].max())
    active_tl = _draw_trendlines_with_lane(
        ax_main,
        trendlines,
        y_min=y_data_min,
        y_max=y_data_max,
        current=current,
        x_left=x_left,
        x_right=x_right,
        fmt_price=fmt_price,
    )
    for tl in trendlines:
        col = "#e67e22" if tl["kind"] == "resistance" else "#9b59b6"
        # 추세선 현재 연장 가격 태그 (우측 축 — 시나리오 라벨과 역할 분리)
        if y_data_min <= tl["price_now"] <= y_data_max:
            _right_price_tag(tl["price_now"], col, force=(tl is active_tl))
    # 현재가 우측 태그
    _right_price_tag(current, "#2c3e50", force=True)

    # ---------- 시나리오 투영 (우측 여백) ----------
    x_fut_mid = x_right + x_pad * 0.40
    x_fut_end = x_right + x_pad * 0.85
    horizon_days = int(round(x_fut_end - x_right))

    # 시나리오별 가능성 점수 (지표 근거 기반)
    _disp_series_for_score = disp_res.get("disparity") if disp_res else None
    disp_last_val = None
    if _disp_series_for_score is not None and len(_disp_series_for_score) and pd.notna(_disp_series_for_score.iloc[-1]):
        disp_last_val = float(_disp_series_for_score.iloc[-1])
    sma20_slope = None
    sma20_valid = sma20.dropna()
    if len(sma20_valid) >= 6:
        sma20_slope = float(sma20_valid.iloc[-1] - sma20_valid.iloc[-6])
    surge_state = detect_surge_state(
        df,
        aux_ind=aux_ind,
        adx_res=adx_res,
        rsi_val=rsi_val,
        lv_by_label=lv_by_label,
        vol_surge_mult=volume_surge_mult,
    )
    surge_setup = detect_surge_setup(
        df,
        aux_ind=aux_ind,
        adx_res=adx_res,
    )
    hist_surge = analyze_ticker_surge_history(
        df,
        aux_ind=aux_ind,
        surge_setup=surge_setup,
    )
    plunge_state = detect_plunge_state(
        df,
        aux_ind=aux_ind,
        adx_res=adx_res,
        rsi_val=rsi_val,
        lv_by_label=lv_by_label,
        vol_surge_mult=volume_surge_mult,
    )
    plunge_setup = detect_plunge_setup(
        df,
        aux_ind=aux_ind,
        adx_res=adx_res,
    )
    hist_plunge = analyze_ticker_plunge_history(
        df,
        aux_ind=aux_ind,
        plunge_setup=plunge_setup,
    )
    scenario_scores = _score_scenarios(
        current=current,
        adx_res=adx_res,
        disparity_value=disp_last_val,
        rsi_val=rsi_val,
        trendlines=trendlines,
        lv_by_label=lv_by_label,
        sma20_slope=sma20_slope,
        top_patterns=top_patterns,
        tb_eval=tb_eval,
        dt_eval=dt_eval,
        aux_ind=aux_ind,
        surge_state=surge_state,
        surge_setup=surge_setup,
        plunge_state=plunge_state,
        plunge_setup=plunge_setup,
    )

    # 패턴 점수·가이드를 먼저 산출 (Top3 랭킹·미니 차트 공용)
    pattern_scores: Dict[str, int] = {}
    pattern_reasons: Dict[str, List[str]] = {}
    pattern_guides: Dict[str, Dict[str, Any]] = {}
    for e in top_patterns:
        pid = e.get("pattern_id")
        slot = _pattern_slot_key(e)
        pe = pattern_evals_all.get(slot)
        spec = PATTERN_TARGET_SPECS.get(pid)
        if pe is None or spec is None:
            continue
        bullish = bool(pe.get("bullish", spec["bullish"]))
        sc, rs = _pattern_target_score(
            pe, adx_res,
            bullish=bullish,
            break_label=spec["break_label"],
            base=spec["base"],
        )
        pattern_scores[slot] = sc
        pattern_reasons[slot] = rs
        pattern_guides[slot] = _pattern_trade_guide(
            pe, adx_res, bullish=bullish, break_label=spec["break_label"],
            score=sc, reasons=rs,
        )

    scenario_entries: List[Dict[str, Any]] = []

    def _add_entry(
        key: str,
        num: str,
        short: str,
        title: str,
        score: int,
        target_txt: str,
        reasons: Optional[List[str]] = None,
        detail_lines: Optional[List[str]] = None,
        *,
        active: bool = True,
        y: Optional[float] = None,
        color: Optional[str] = None,
        path_x: Optional[List[float]] = None,
        path_y: Optional[List[float]] = None,
        arrow_from: Optional[Tuple[float, float]] = None,
        linestyle: str = "--",
        status: Optional[str] = None,
    ) -> None:
        scenario_entries.append({
            "key": key,
            "num": num,
            "short": short,
            "title": title,
            "score": int(score),
            "target_txt": target_txt,
            "reasons": reasons or [],
            "detail_lines": detail_lines or [],
            "active": active,
            "y": y,
            "color": color,
            "path_x": path_x,
            "path_y": path_y,
            "arrow_from": arrow_from,
            "linestyle": linestyle,
            "status": status,
        })

    # ① 지지 이탈
    near_sup = lv_by_label.get("현재 지지")
    strong_sup = lv_by_label.get("강력 지지")
    recent_sup = lv_by_label.get("최근지지")
    sc1_path_x = sc1_path_y = None
    sc1_target = None
    sc1_wlabels: List[str] = []
    # 경로: 현재 → (최근지지) → 현재지지 → (강력지지) — 최근은 구조선과 구분될 때만
    path_prices: List[float] = [current]
    path_labs: List[str] = ["현재"]
    if (
        recent_sup
        and near_sup
        and recent_sup["price"] < current
        and recent_sup["price"] > near_sup["price"] * 1.005
    ):
        path_prices.append(float(recent_sup["price"]))
        path_labs.append("최근지지")
    if near_sup and strong_sup and strong_sup["price"] < near_sup["price"]:
        path_prices.append(float(near_sup["price"]))
        path_labs.append("현재지지")
        path_prices.append(float(strong_sup["price"]))
        path_labs.append("강력지지")
        sc1_target = strong_sup["price"]
    elif near_sup:
        path_prices.append(float(near_sup["price"]))
        path_labs.append("현재지지")
        sc1_target = near_sup["price"]
    elif recent_sup and recent_sup["price"] < current:
        path_prices.append(float(recent_sup["price"]))
        path_labs.append("최근지지")
        sc1_target = recent_sup["price"]
    if sc1_target is not None and len(path_prices) >= 2:
        # x 좌표: 균등 분할 (2점이면 끝, 3점 이상이면 mid 포함)
        if len(path_prices) == 2:
            sc1_path_x = [x_right, x_fut_end]
        elif len(path_prices) == 3:
            sc1_path_x = [x_right, x_fut_mid, x_fut_end]
        else:
            xs_pts = np.linspace(x_right, x_fut_end, len(path_prices))
            sc1_path_x = [float(x) for x in xs_pts]
        sc1_path_y = path_prices
        sc1_wlabels = path_labs
        sc1_pct = (sc1_target / current - 1.0) * 100.0
        sc1_score = scenario_scores["sc1"]["score"]
        trig_px = path_prices[1] if len(path_prices) > 1 else sc1_target
        st1 = _proximity_state(current, float(trig_px), bullish_break=False)
        _add_entry(
            "sc1", "①", "지지이탈", "지지 이탈",
            sc1_score,
            f"{fmt_price(sc1_target)} ({sc1_pct:+.1f}%)",
            scenario_scores["sc1"].get("reasons"),
            [f"     케이스▼ 트리거 {fmt_price(trig_px)} 이탈 · [{st1}]"],
            y=sc1_target, color="#c0392b",
            path_x=sc1_path_x, path_y=sc1_path_y, linestyle="--",
        )
        scenario_entries[-1]["waypoint_labels"] = sc1_wlabels

    # ② 추세 하락 (지지 추세선 연장 · Low.min 캡 제거 · ①과 이격)
    sc2_res = _resolve_trend_follow_scenario(
        direction="down",
        current=current,
        trendlines=trendlines,
        df=df,
        x_right=x_right,
        x_fut_end=x_fut_end,
        lv_by_label=lv_by_label,
        peer_target=sc1_target,
    )
    if sc2_res is not None:
        sc2_target = float(sc2_res["target"])
        sc2_pct = (sc2_target / current - 1.0) * 100.0
        sc2_score = scenario_scores["sc2"]["score"]
        details = [f"     추세▼ {sc2_res['src']}"]
        if sc2_res.get("note"):
            details.append(f"     {sc2_res['note']}")
        _add_entry(
            "sc2", "②", "추세하락", f"추세 하락({sc2_res['src']})",
            sc2_score,
            f"{fmt_price(sc2_target)} ({sc2_pct:+.1f}%)",
            scenario_scores["sc2"].get("reasons"),
            details,
            active=bool(sc2_res.get("active", True)),
            y=sc2_target, color="#7d3c98",
            path_x=sc2_res.get("path_x"), path_y=sc2_res.get("path_y"),
            arrow_from=(x_right, current), linestyle=":",
            status=sc2_res.get("status"),
        )

    # ③ 저항 돌파 (구조선 + ST·SMA200 목표 체인)
    near_res = lv_by_label.get("단기 저항")
    high_res = lv_by_label.get("전고점 저항")
    recent_res = lv_by_label.get("최근저항")
    sc3_path_x = sc3_path_y = None
    sc3_target = None
    sc3_wlabels: List[str] = []
    sc3_cands: List[Tuple[float, str]] = []
    if recent_res:
        _append_sc3_level(sc3_cands, recent_res["price"], "최근저항", current=current)
    if near_res:
        _append_sc3_level(sc3_cands, near_res["price"], "단기저항", current=current)
    # ST: 하락 중이면 저항, 상승 중이면 지지(현재가 위일 때만 돌파 경로)
    st_px = aux_ind.get("st_line")
    if st_px is not None and float(st_px) > current * 1.002:
        st_lab = "ST저항" if aux_ind.get("st_dir") == "down" else "ST"
        _append_sc3_level(sc3_cands, float(st_px), st_lab, current=current)
    sma200_px = aux_ind.get("sma200")
    if sma200_px is not None and float(sma200_px) > current * 1.002:
        _append_sc3_level(sc3_cands, float(sma200_px), "SMA200", current=current)
    if high_res:
        _append_sc3_level(sc3_cands, high_res["price"], "전고점", current=current)

    sc3_cands.sort(key=lambda x: x[0])
    if sc3_cands:
        # 중기 체인: SMA200이 있으면 거기까지 (전고점은 수평선으로만 유지)
        sma_idx = next((i for i, (_, lb) in enumerate(sc3_cands) if lb == "SMA200"), None)
        if sma_idx is not None:
            sc3_cands = sc3_cands[: sma_idx + 1]
        path_up = [current] + [p for p, _ in sc3_cands]
        labs_up = ["현재"] + [lb for _, lb in sc3_cands]
        sc3_target = float(sc3_cands[-1][0])
        if len(path_up) == 2:
            sc3_path_x = [x_right, x_fut_end]
        elif len(path_up) == 3:
            sc3_path_x = [x_right, x_fut_mid, x_fut_end]
        else:
            sc3_path_x = [float(x) for x in np.linspace(x_right, x_fut_end, len(path_up))]
        sc3_path_y = path_up
        sc3_wlabels = labs_up
        sc3_pct = (sc3_target / current - 1.0) * 100.0
        sc3_score = scenario_scores["sc3"]["score"]
        trig_px = path_up[1]
        st3 = _proximity_state(current, float(trig_px), bullish_break=True)
        _add_entry(
            "sc3", "③", "저항돌파", "저항 돌파",
            sc3_score,
            f"{fmt_price(sc3_target)} ({sc3_pct:+.1f}%)",
            scenario_scores["sc3"].get("reasons"),
            [f"     케이스▲ 트리거 {fmt_price(trig_px)} 돌파 · [{st3}]"],
            y=sc3_target, color="#1e8449",
            path_x=sc3_path_x, path_y=sc3_path_y, linestyle="--",
        )
        scenario_entries[-1]["waypoint_labels"] = sc3_wlabels

    # ④ 추세 상승 (저항 추세선 연장 · High.max 캡 완화 · ③과 이격)
    sc4_res = _resolve_trend_follow_scenario(
        direction="up",
        current=current,
        trendlines=trendlines,
        df=df,
        x_right=x_right,
        x_fut_end=x_fut_end,
        lv_by_label=lv_by_label,
        peer_target=sc3_target,
    )
    if sc4_res is not None:
        sc4_target = float(sc4_res["target"])
        sc4_pct = (sc4_target / current - 1.0) * 100.0
        sc4_score = scenario_scores["sc4"]["score"]
        details = [f"     추세▲ {sc4_res['src']}"]
        if sc4_res.get("note"):
            details.append(f"     {sc4_res['note']}")
        _add_entry(
            "sc4", "④", "추세상승", f"추세 상승({sc4_res['src']})",
            sc4_score,
            f"{fmt_price(sc4_target)} ({sc4_pct:+.1f}%)",
            scenario_scores["sc4"].get("reasons"),
            details,
            active=bool(sc4_res.get("active", True)),
            y=sc4_target, color="#2471a3",
            path_x=sc4_res.get("path_x"), path_y=sc4_res.get("path_y"),
            arrow_from=(x_right, current), linestyle=":",
            status=sc4_res.get("status"),
        )

    # ⑤ 하락 후 반등 (V 꺾인 경로)
    sc5_res = _resolve_rebound_scenario(
        current=current,
        df=df,
        lv_by_label=lv_by_label,
        aux_ind=aux_ind,
        x_right=x_right,
        x_fut_mid=x_fut_mid,
        x_fut_end=x_fut_end,
        peer_target=sc3_target,
    )
    if sc5_res is not None:
        sc5_target = float(sc5_res["target"])
        sc5_pct = (sc5_target / current - 1.0) * 100.0
        sc5_score = scenario_scores["sc5"]["score"]
        details = [f"     반등▲ {sc5_res.get('note') or 'V경로'}"]
        _add_entry(
            "sc5", "⑤", "하락후반등", "하락 후 반등",
            sc5_score,
            f"{fmt_price(sc5_target)} ({sc5_pct:+.1f}%)",
            scenario_scores["sc5"].get("reasons"),
            details,
            y=sc5_target, color="#16a085",
            path_x=sc5_res.get("path_x"), path_y=sc5_res.get("path_y"),
            linestyle="-.",
        )
        scenario_entries[-1]["waypoint_labels"] = sc5_res.get("waypoint_labels") or []

    # ⑥ 상승 후 조정 (Λ 꺾인 경로)
    sc6_res = _resolve_pullback_scenario(
        current=current,
        df=df,
        lv_by_label=lv_by_label,
        aux_ind=aux_ind,
        x_right=x_right,
        x_fut_mid=x_fut_mid,
        x_fut_end=x_fut_end,
        peer_target=sc1_target,
    )
    if sc6_res is not None:
        sc6_target = float(sc6_res["target"])
        sc6_pct = (sc6_target / current - 1.0) * 100.0
        sc6_score = scenario_scores["sc6"]["score"]
        details = [f"     조정▼ {sc6_res.get('note') or 'Λ경로'}"]
        _add_entry(
            "sc6", "⑥", "상승후조정", "상승 후 조정",
            sc6_score,
            f"{fmt_price(sc6_target)} ({sc6_pct:+.1f}%)",
            scenario_scores["sc6"].get("reasons"),
            details,
            y=sc6_target, color="#e67e22",
            path_x=sc6_res.get("path_x"), path_y=sc6_res.get("path_y"),
            linestyle="-.",
        )
        scenario_entries[-1]["waypoint_labels"] = sc6_res.get("waypoint_labels") or []

    # 패턴은 시나리오 번호에 넣지 않음. 지지/저항·반전 시나리오 ①~⑥ Top3.
    top3_keys = _pick_scenario_top3(scenario_entries, 3)
    top3_set = set(top3_keys)
    scenario_panel_lines = _build_top3_scenario_panel(scenario_entries, top3_keys)
    case_eval_lines = _build_sr_case_eval_lines(
        scenario_entries,
        current=current,
        fmt_price=fmt_price,
        lv_by_label=lv_by_label,
    )

    # 메인: 현재가→시나리오 경로선 (선 세기=점수 상대, Top3는 ★만)
    score_pool = [
        float(e.get("score") or 0)
        for e in scenario_entries
        if e.get("active", True) and e.get("y") is not None
    ]
    for e in scenario_entries:
        key = e.get("key")
        if not e.get("active", True):
            continue
        px, py = e.get("path_x"), e.get("path_y")
        if not (px and py):
            continue
        st = _scenario_strength_style(float(e.get("score") or 0), score_pool)
        _draw_scenario_on_main(
            ax_main,
            label=f"{e['num']}{e.get('short', '')}",
            color=e.get("color") or "#34495e",
            x_end=float(px[-1]),
            y_end=float(py[-1]),
            xytext=(8, 0),
            is_top3=key in top3_set,  # 라벨용(show_label=False면 미사용)
            path_x=px,
            path_y=py,
            linestyle=e.get("linestyle") or ("--" if key in ("sc1", "sc3") else ":"),
            show_label=False,
            strength=st,
        )
        if key in ("sc1", "sc3", "sc5", "sc6"):
            _annotate_scenario_waypoints(
                ax_main, list(px), list(py),
                list(e.get("waypoint_labels") or []),
                e.get("color") or "#34495e",
            )
        # 경로 끝 번호+약칭 캡션은 표시하지 않음 (우측 번호 레인 · 인사이트 목록으로 식별)

    try:
        _draw_surge_badge(ax_main, df, surge_state)
    except Exception as e:
        print(f"tech annotation: 급등 배지 표시 실패 {e}")
    try:
        _draw_surge_setup_badge(ax_main, df, surge_setup, surge_state=surge_state)
    except Exception as e:
        print(f"tech annotation: 급등셋업 배지 표시 실패 {e}")
    try:
        _draw_plunge_badge(ax_main, df, plunge_state)
    except Exception as e:
        print(f"tech annotation: 급락 배지 표시 실패 {e}")
    try:
        _draw_plunge_setup_badge(ax_main, df, plunge_setup, plunge_state=plunge_state)
    except Exception as e:
        print(f"tech annotation: 급락셋업 배지 표시 실패 {e}")

    # 메인: 시나리오 목표가 수평선 + 우측 번호 레인
    # 번호 레인 Y범위: OHLC만이 아니라 시나리오 목표가·추세선 연장가를 포함
    # (신저점 구간에서 Low.min≈현재가면 ② 목표가 레인 밖으로 클램프→①과 번호 겹침)
    sc_draws: List[Dict[str, Any]] = []
    for e in scenario_entries:
        if e.get("y") is None:
            continue
        if not e.get("active", True) and e.get("key") in ("sc2", "sc4"):
            continue  # 중복 보류 케이스는 번호 레인에서 제외
        sc_draws.append({
            "key": e["key"],
            "num": e["num"],
            "y": float(e["y"]),
            "color": e.get("color") or "#34495e",
            "is_top3": e["key"] in top3_set,
            "active": e.get("active", True),
            "status": e.get("status"),
            "score": int(e.get("score") or 0),
        })
    y_candidates = [float(df["Low"].min()), float(df["High"].max()), float(current)]
    y_candidates.extend(float(d["y"]) for d in sc_draws)
    for tl in trendlines:
        if tl.get("price_now") is not None:
            y_candidates.append(float(tl["price_now"]))
    for e in scenario_entries:
        for yy in e.get("path_y") or []:
            y_candidates.append(float(yy))
    y_lo_sc = min(y_candidates)
    y_hi_sc = max(y_candidates)
    y_sp_sc = max(y_hi_sc - y_lo_sc, 1e-9)
    # 메인 ylim도 시나리오 목표가 포함하도록 확장 (② 경로가 잘리지 않게)
    cur_lo, cur_hi = ax_main.get_ylim()
    need_lo = y_lo_sc - y_sp_sc * 0.04
    need_hi = y_hi_sc + y_sp_sc * 0.06
    if need_lo < cur_lo or need_hi > cur_hi:
        ax_main.set_ylim(min(cur_lo, need_lo), max(cur_hi, need_hi))
    # ylim 확정 후 급등/급락 이력 캡션 배치 (스케일 기준 겹침 방지)
    try:
        _draw_historical_surge_markers(ax_main, df, hist_surge)
    except Exception as e:
        print(f"tech annotation: 급등이력 마커 표시 실패 {e}")
    try:
        _draw_historical_plunge_markers(ax_main, df, hist_plunge)
    except Exception as e:
        print(f"tech annotation: 급락이력 마커 표시 실패 {e}")
    x_lane = x_right + x_pad * 0.92
    _draw_scenario_number_lane(
        ax_main,
        sc_draws,
        x_right=x_right,
        x_lane=x_lane,
        y_min=y_lo_sc + y_sp_sc * 0.03,
        y_max=y_hi_sc - y_sp_sc * 0.03,
        current=current,
    )

    # ---------- 하단 패턴 미니 차트 행 ----------
    for m, ev in enumerate(mini_patterns):
        ax_mp, ax_mv, ax_mc = mini_axes[m]
        num = PATTERN_NUM_MARKS[m]
        pid = ev.get("pattern_id")
        slot = _pattern_slot_key(ev)
        pe = pattern_evals_all.get(slot)
        spec = PATTERN_TARGET_SPECS.get(pid)
        if pe is not None and spec is not None:
            bullish = bool(pe.get("bullish", spec["bullish"]))
            direction = "up" if bullish else "down"
            name = spec["name"]
            if pid == "fvg_gap":
                name = "FVG상방" if bullish else "FVG하방"
            _plot_pattern_target_mini(
                ax_mp, df, pe, current, fmt_price, pattern_scores.get(slot),
                col=pe.get("col", spec["col"]),
                neck_col=pe.get("neck_col", spec["neck_col"]),
                title_base=f"{num} {name} 목표 {spec['num']}",
                direction=direction,
                guide=pattern_guides.get(slot),
                ax_vol=ax_mv,
                ax_cap=ax_mc,
            )
        else:
            _plot_generic_pattern_mini(ax_mp, df, ev, num, ax_vol=ax_mv, ax_cap=ax_mc)

    # ---------- 패턴 구간 식별 (시나리오 번호 없음 · 상세는 하단 미니) ----------
    y_min_all = float(df["Low"].min())
    y_max_all = float(df["High"].max())
    y_span = y_max_all - y_min_all
    span_lane = 0
    for m, ev in enumerate(mini_patterns):
        pid = ev.get("pattern_id")
        slot = _pattern_slot_key(ev)
        spec = PATTERN_TARGET_SPECS.get(pid) or {}
        guide = pattern_guides.get(slot) or {}
        st = guide.get("status")
        if st not in ("무효화", "달성"):
            pe = pattern_evals_all.get(slot)
            if pe is not None and _pattern_expired(pe):
                st = "무효화" if pe.get("invalidated") else "달성"
            else:
                st = None
        tag = spec.get("name") or CHART_PATTERN_CRITERIA[ID_TO_ROW[pid]]["label"]
        if pid == "fvg_gap":
            pe = pattern_evals_all.get(slot) or {}
            bullish = bool(pe.get("bullish", not _fvg_is_bear(ev)))
            tag = "FVG↑" if bullish else "FVG↓"
        used = _plot_pattern_span_on_main(
            ax_main, df, ev,
            x_left=x_left, x_right=x_right,
            y_min_all=y_min_all, y_max_all=y_max_all, y_span=y_span,
            span_lane=span_lane,
            status=st,
            tag=tag,
        )
        span_lane += used
    if span_lane > 0:
        y_bottom, y_top_cur = ax_main.get_ylim()
        y_need = y_max_all + y_span * (0.055 + 0.065 * span_lane + 0.05)
        if y_top_cur < y_need:
            ax_main.set_ylim(y_bottom, y_need)

    # ---------- 배당락일 ----------
    for div in dividend_dates:
        d = div.get("date")
        if d is None or d not in df.index:
            continue
        ax_main.scatter(
            d,
            y_min_all - y_span * 0.02,
            marker="D",
            s=42,
            color="gold",
            edgecolor="darkorange",
            linewidth=1.0,
            zorder=18,
            clip_on=False,
        )
        ax_main.annotate(
            "D",
            xy=(d, y_min_all - y_span * 0.02),
            fontsize=6,
            fontweight="bold",
            color="black",
            ha="center",
            va="center",
            zorder=19,
        )
        # 배당락 후 실제 하락 시 주석 (다음 거래일 종가 하락)
        loc = df.index.get_loc(d)
        if isinstance(loc, int) and loc + 1 < len(df):
            drop = float(df["Close"].iloc[loc + 1]) - float(df["Close"].iloc[loc])
            if drop < 0:
                ax_main.annotate(
                    "배당락 하락",
                    xy=(d, float(df["Low"].iloc[loc])),
                    xytext=(0, -14),
                    textcoords="offset points",
                    fontsize=7.5,
                    color="darkorange",
                    ha="center",
                )

    ax_main.set_ylabel(f"가격 ({currency})", fontsize=10.5)
    ax_main.grid(True, alpha=0.22, linestyle=":")
    ax_main.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.005),
        ncol=8,
        fontsize=8.0,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.4,
    )
    ax_main.tick_params(axis="y", labelsize=9.5)
    ax_main.tick_params(axis="x", labelbottom=False, length=0)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # ---------- 거래량 + 급증 강조 ----------
    vol = df["Volume"].astype(float)
    vol_ma20 = vol.rolling(20).mean()
    surge_mask = vol > vol_ma20 * volume_surge_mult
    colors = np.where(surge_mask, "#e67e22", "#95a5a6")
    # 백만 단위로 플롯 → ScalarFormatter 1e8 오프셋 원천 차단
    vol_m = vol / 1e6
    vol_ma_m = vol_ma20 / 1e6
    ax_vol.bar(df.index, vol_m, color=colors, alpha=0.85, width=0.7)
    ax_vol.plot(df.index, vol_ma_m, color="#2980b9", linewidth=0.8, alpha=0.8, label="VOL MA20")
    if surge_mask.any():
        top_surge = vol[surge_mask].idxmax()
        ax_vol.annotate(
            "거래량 급증: 추세 확인",
            xy=(top_surge, float(vol_m.loc[top_surge])),
            xytext=(8, -2),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            color="#e67e22",
        )
    # 역삼중창 구간별 평균 거래량 표시 (좌어깨/머리/우어깨)
    if tb_eval is not None and tb_eval.get("vol_segments"):
        for t0, t1, mean_v, seg_lbl in tb_eval["vol_segments"]:
            if mean_v is None:
                continue
            mean_m = float(mean_v) / 1e6
            ax_vol.axvspan(t0, t1, color="#16a085", alpha=0.08, zorder=1)
            ax_vol.hlines(mean_m, t0, t1, color="#16a085", linewidth=1.4, alpha=0.95, zorder=11)
            mid = t0 + (t1 - t0) / 2
            ax_vol.annotate(
                seg_lbl,
                xy=(mid, mean_m),
                xytext=(0, 4),
                textcoords="offset points",
                fontsize=7.5,
                fontweight="bold",
                color="#16a085",
                ha="center",
                zorder=12,
            )

    ax_vol.set_ylabel("Volume (M)", fontsize=9.5)
    ax_vol.grid(True, alpha=0.2, linestyle=":")
    ax_vol.legend(loc="upper left", fontsize=7.5, framealpha=0.85)
    ax_vol.tick_params(axis="x", labelbottom=False, length=0)
    ax_vol.yaxis.set_major_formatter(FuncFormatter(lambda v, _p=None: f"{v:.0f}"))
    _kill_axis_offsets(ax_vol)
    plt.setp(ax_vol.get_xticklabels(), visible=False)

    # ---------- 이격도 스트립 ----------
    disp_series = disp_res.get("disparity") if disp_res else None
    if disp_series is not None and getattr(disp_series, "notna", None) is not None and disp_series.notna().any():
        d = disp_series.astype(float)
        ax_disp.plot(df.index, d, color="#2c3e50", linewidth=1.0, alpha=0.9, label="이격도(20MA)")
        ax_disp.axhline(100, color="#7f8c8d", linewidth=0.8, alpha=0.7)
        ax_disp.axhline(105, color="#e74c3c", linewidth=0.8, linestyle="--", alpha=0.8)
        ax_disp.axhline(95, color="#27ae60", linewidth=0.8, linestyle="--", alpha=0.8)
        ax_disp.fill_between(df.index, d, 105, where=(d > 105), color="#e74c3c", alpha=0.18, interpolate=True)
        ax_disp.fill_between(df.index, d, 95, where=(d < 95), color="#27ae60", alpha=0.18, interpolate=True)
        # D95/D105 이벤트 마커 (이격도 값 위에 표시)
        for ev in (disp_res.get("events") or [])[-30:]:
            ts = pd.Timestamp(ev["date"])
            if ts in d.index and pd.notna(d.loc[ts]):
                ax_disp.scatter(
                    [ts], [float(d.loc[ts])],
                    marker=ev.get("marker", "o"), s=26,
                    color=ev.get("color", "#34495e"),
                    edgecolors="white", linewidths=0.4, zorder=10,
                )
        # 마지막 값 강조
        last_disp = float(d.iloc[-1]) if pd.notna(d.iloc[-1]) else None
        if last_disp is not None:
            ax_disp.annotate(
                f"{last_disp:.1f}%",
                xy=(df.index[-1], last_disp),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=8.5,
                fontweight="bold",
                color="#2c3e50",
                va="center",
            )
        d_valid = d.dropna()
        y_lo = min(93.0, float(d_valid.min()) - 1.0)
        y_hi = max(107.0, float(d_valid.max()) + 1.0)
        ax_disp.set_ylim(y_lo, y_hi)
        ax_disp.set_ylabel("이격도(%)", fontsize=9.5)
        ax_disp.grid(True, alpha=0.2, linestyle=":")
        ax_disp.legend(loc="upper left", fontsize=7.5, framealpha=0.85)
        ax_disp.tick_params(axis="y", labelsize=9)
    else:
        ax_disp.text(0.5, 0.5, "이격도 데이터 없음", transform=ax_disp.transAxes,
                     ha="center", va="center", fontsize=9, color="#7f8c8d")
    # 공유 X축 일자: 월(YYYY-MM) + 현재일(YYYY-MM-DD) 틱 (축 안쪽 배치)
    _style_shared_date_axis(ax_disp, x_last=x_right, labelsize=9.0)
    # sharex 잔여 오프셋/상위축 일자 라벨 정리
    for _ax in (ax_main, ax_vol):
        _ax.tick_params(axis="x", labelbottom=False, length=0)
        plt.setp(_ax.get_xticklabels(), visible=False)
        _kill_axis_offsets(_ax)
    _kill_axis_offsets(ax_disp)

    # ---------- 우측 인사이트 패널 ----------
    def _with_dist(price: float) -> str:
        """가격 + 현재가 대비 거리 %"""
        pct = (price / current - 1.0) * 100.0
        return f"{fmt_price(price)} ({pct:+.1f}%)"

    panel_lines: List[str] = [f"[{ticker} 인사이트]", ""]
    if "강력 지지" in lv_by_label:
        panel_lines.append(f"핵심 지지: {_with_dist(lv_by_label['강력 지지']['price'])} 강력")
    if "현재 지지" in lv_by_label:
        panel_lines.append(f"현재 지지: {_with_dist(lv_by_label['현재 지지']['price'])}")
    if "최근지지" in lv_by_label:
        lb = lv_by_label["최근지지"].get("lookback")
        lb_txt = f" ·{lb}봉" if lb else ""
        panel_lines.append(f"최근지지: {_with_dist(lv_by_label['최근지지']['price'])}{lb_txt}")
    if "직전 지지(이탈)" in lv_by_label:
        panel_lines.append(f"직전 지지(이탈): {_with_dist(lv_by_label['직전 지지(이탈)']['price'])}")
    if "단기 저항" in lv_by_label:
        panel_lines.append(f"단기 저항: {_with_dist(lv_by_label['단기 저항']['price'])}")
    if "최근저항" in lv_by_label:
        lb = lv_by_label["최근저항"].get("lookback")
        lb_txt = f" ·{lb}봉" if lb else ""
        panel_lines.append(f"최근저항: {_with_dist(lv_by_label['최근저항']['price'])}{lb_txt}")
    if "전고점 저항" in lv_by_label:
        panel_lines.append(f"전고점: {_with_dist(lv_by_label['전고점 저항']['price'])}")
    if vp_info is not None:
        poc = float(vp_info["poc"])
        poc_pct = (poc / current - 1.0) * 100.0
        near_sup = lv_by_label.get("현재 지지")
        extra = ""
        if near_sup is not None:
            d_sup = (poc / float(near_sup["price"]) - 1.0) * 100.0
            extra = f" · 지지대비 {d_sup:+.1f}%"
        panel_lines.append(
            f"매물 POC: {fmt_price(poc)} (현재가대비 {poc_pct:+.1f}% · 비중 {vp_info['poc_share_pct']:.1f}%{extra})"
        )
    if trendlines:
        for tl in trendlines:
            direction = "하락" if tl["slope"] < 0 else "상승"
            role = "저항" if tl["kind"] == "resistance" else "지지"
            star = "★" if active_tl is not None and tl is active_tl else ""
            line = f"{star}{direction} {role} 추세선: {_with_dist(tl['price_now'])} {tl['touches']}터치"
            if tl.get("broken"):
                line += "·돌파"
            if active_tl is not None and tl is active_tl:
                line += " ·활성"
            panel_lines.append(line)
    if show_target_prices and any(p is not None for p in (target_buy_price, target_sell_price, stop_loss_price)):
        panel_lines.append("")
        if target_buy_price is not None:
            panel_lines.append(f"매수 목표: {_with_dist(float(target_buy_price))}")
        if target_sell_price is not None:
            panel_lines.append(f"매도 목표: {_with_dist(float(target_sell_price))}")
        if stop_loss_price is not None:
            panel_lines.append(f"손절가: {_with_dist(float(stop_loss_price))}")
    panel_lines.append("")
    _adx_lines = _adx_panel_lines(adx_res)
    if _adx_lines:
        panel_lines.extend(_adx_lines)
    latest_ev = (disp_res.get("latest_event") or {}) if disp_res else {}
    disp_v = latest_ev.get("disparity")
    if disp_v is not None and pd.notna(disp_v):
        panel_lines.append(f"이격도(20MA): {float(disp_v):.1f}% → {disp_res.get('final_action', '관망')}")
    if rsi_val is not None:
        rsi_state = "과매수" if rsi_val >= 70 else ("과매도" if rsi_val <= 30 else "중립")
        panel_lines.append(f"RSI(14): {rsi_val} ({rsi_state})")
    for _aux_line in _aux_indicator_panel_lines(aux_ind, current=current, fmt_price=fmt_price):
        panel_lines.append(_aux_line)
    _peg_lines = _peg_panel_lines(peg_view)
    if _peg_lines:
        panel_lines.append("")
        panel_lines.extend(_peg_lines)
    for _surge_line in _surge_panel_lines(surge_state):
        panel_lines.append(_surge_line)
    # 이미 급등 중이면 셋업 줄은 생략 (중복·혼동 방지)
    if surge_state.get("state") not in ("급등진행", "급등과열"):
        for _setup_line in _surge_setup_panel_lines(surge_setup):
            panel_lines.append(_setup_line)
    _hist_lines = _hist_surge_panel_lines(hist_surge)
    if _hist_lines:
        panel_lines.append("")
        panel_lines.extend(_hist_lines)
    for _plunge_line in _plunge_panel_lines(plunge_state):
        panel_lines.append(_plunge_line)
    if plunge_state.get("state") not in ("급락진행", "급락과매도"):
        for _pl_setup_line in _plunge_setup_panel_lines(plunge_setup):
            panel_lines.append(_pl_setup_line)
    _hist_pl_lines = _hist_plunge_panel_lines(hist_plunge)
    if _hist_pl_lines:
        panel_lines.append("")
        panel_lines.extend(_hist_pl_lines)
    if dividend_dates:
        total_div = sum(float(d.get("amount") or 0) for d in dividend_dates)
        if total_div > 0:
            div_yield_pct = total_div / current * 100.0
            panel_lines.append(
                f"기간 배당: {fmt_price(total_div)} ({len(dividend_dates)}회, 수익률 {div_yield_pct:.1f}%)"
            )
    if scenario_panel_lines:
        panel_lines.append("")
        panel_lines.append("시나리오 가능성 (~20일, 0~100점):")
        panel_lines.append("①지지이탈 ②추세↓ ③저항돌파 ④추세↑")
        panel_lines.append("⑤하락후반등 ⑥상승후조정")
        if case_eval_lines:
            panel_lines.append("")
            panel_lines.extend(case_eval_lines)
            panel_lines.append("")
        panel_lines.extend(scenario_panel_lines)
        panel_lines.append("※ 지지/저항·추세 종합 (패턴은 가점만·확률 아님)")
    if mini_patterns:
        panel_lines.append("")
        def _pat_sign(ev: Dict[str, Any]) -> int:
            pid = ev["pattern_id"]
            if pid == "fvg_gap":
                return -1 if _fvg_is_bear(ev) else 1
            return PATTERN_IMPLICATION.get(pid, ("", 0))[1]
        bull_n = sum(1 for ev in mini_patterns if _pat_sign(ev) > 0)
        bear_n = sum(1 for ev in mini_patterns if _pat_sign(ev) < 0)
        if bull_n > bear_n:
            agg = f"강세 우위 {bull_n}:{bear_n}"
        elif bear_n > bull_n:
            agg = f"약세 우위 {bear_n}:{bull_n}"
        else:
            agg = "혼조"
        panel_lines.append(f"패턴 참고 (식별·{agg}) — 상세: 하단 미니")
        for m, ev in enumerate(mini_patterns):
            pid = ev["pattern_id"]
            slot = _pattern_slot_key(ev)
            spec = PATTERN_TARGET_SPECS.get(pid) or {}
            name = spec.get("name") or CHART_PATTERN_CRITERIA[ID_TO_ROW[pid]]["label"]
            pe = pattern_evals_all.get(slot)
            if pid == "fvg_gap":
                bullish = bool(pe.get("bullish")) if pe else (not _fvg_is_bear(ev))
                name = "FVG상방" if bullish else "FVG하방"
            guide = pattern_guides.get(slot) or {}
            st = guide.get("status")
            if pe is not None and _pattern_expired(pe) and st not in ("무효화", "달성"):
                st = "무효화" if pe.get("invalidated") else "달성"
            st_txt = f"[{st}]" if st else "[진행]"
            tgt = ""
            if pe is not None and pe.get("target") is not None:
                tgt = f" → {fmt_price(pe['target'])}"
            panel_lines.append(f"  {PATTERN_NUM_MARKS[m]} {name} {st_txt}{tgt}")
        panel_lines.append("")
        panel_lines.extend(_confidence_reference_lines())
    if note_text:
        panel_lines.append("")
        panel_lines.append("[종목 메모]")
        panel_lines.extend(note_text.splitlines()[:6])

    # 패널 높이 대비 줄 수에 맞춰 폰트 자동 조정 (기본 대비 약 10% 축소)
    panel_fs, panel_ls = 8.55, 1.40
    panel_h_in = ax_panel.get_position().height * fig.get_size_inches()[1]
    needed_in = len(panel_lines) * panel_fs * panel_ls / 72.0
    if needed_in > panel_h_in * 0.96:
        panel_ls = 1.22
        panel_fs = max(5.5, (panel_h_in * 0.96 * 72.0) / (len(panel_lines) * panel_ls))
    ax_panel.text(
        0.02, 0.99,
        "\n".join(panel_lines),
        transform=ax_panel.transAxes,
        fontsize=panel_fs,
        va="top",
        ha="left",
        linespacing=panel_ls,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#34495e", linewidth=1.4, alpha=0.95),
    )

    # ---------- 하단 요약 바 ----------
    strong = lv_by_label.get("강력 지지") or lv_by_label.get("현재 지지")
    resist = lv_by_label.get("단기 저항") or lv_by_label.get("전고점 저항")
    strategy_bits: List[str] = []
    if strong:
        strategy_bits.append(f"지지 {fmt_price(strong['price'])} 이탈 시 리스크 관리")
    if resist:
        strategy_bits.append(f"저항 {fmt_price(resist['price'])} 돌파 시 추세 전환 확인")
    fa = disp_res.get("final_action") if disp_res else None
    if fa:
        strategy_bits.append(f"이격도 액션: {fa}")
    if peg_view.get("summary"):
        strategy_bits.append(f"재무(PEG): {peg_view['summary']}")
        core = peg_view.get("core_strategy")
        if core:
            strategy_bits.append(core)

    bar_parts: List[str] = []
    if strong:
        bar_parts.append(f"핵심 지지: {fmt_price(strong['price'])}")
    if resist:
        bar_parts.append(f"단기 저항: {fmt_price(resist['price'])}")
    if adx_res.get("adx") is not None:
        bar_parts.append(f"현재 추세: {adx_res.get('summary')} (ADX {adx_res.get('adx')})")
    if peg_view.get("primary_peg") is not None:
        bar_parts.append(
            f"PEG{peg_view.get('primary_label') or ''}: "
            f"{peg_view['primary_peg']}({peg_view.get('primary_sentiment')}) "
            f"→ {peg_view.get('action')}"
        )
    if surge_state.get("state") in ("급등진행", "급등과열", "급등약함"):
        bar_parts.append(f"급등: {surge_state.get('state')}({int(surge_state.get('score') or 0)})")
    elif surge_setup.get("active"):
        bar_parts.append(f"급등셋업: {surge_setup.get('state')}({int(surge_setup.get('score') or 0)})")
    sim = (hist_surge or {}).get("similarity") or {}
    if hist_surge.get("n", 0) > 0 and sim.get("label") not in (None, "해당없음"):
        bar_parts.append(
            f"급등유사: {sim.get('label')}({int(sim.get('score') or 0)})"
            f"/주유형{(hist_surge.get('profile') or {}).get('dominant_kind', '-')}"
        )
    if plunge_state.get("state") in ("급락진행", "급락과매도", "급락약함"):
        bar_parts.append(f"급락: {plunge_state.get('state')}({int(plunge_state.get('score') or 0)})")
    elif plunge_setup.get("active"):
        bar_parts.append(f"급락셋업: {plunge_setup.get('state')}({int(plunge_setup.get('score') or 0)})")
    sim_pl = (hist_plunge or {}).get("similarity") or {}
    if hist_plunge.get("n", 0) > 0 and sim_pl.get("label") not in (None, "해당없음"):
        bar_parts.append(
            f"급락유사: {sim_pl.get('label')}({int(sim_pl.get('score') or 0)})"
            f"/주유형{(hist_plunge.get('profile') or {}).get('dominant_kind', '-')}"
        )
    bar_text = "  |  ".join(bar_parts)
    if strategy_bits:
        bar_text += "\n투자 전략: " + " · ".join(strategy_bits)

    ax_bar.text(
        0.5, 0.5,
        bar_text or "요약 데이터 부족",
        transform=ax_bar.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="center",
        linespacing=1.7,
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#eaf2f8", edgecolor="#2471a3", linewidth=1.6, alpha=0.95),
    )

    fig.text(0.5, 0.005, "주의: 본 분석은 참고용이며, 투자 결과의 책임은 본인에게 있습니다.", fontsize=8, color="#7f8c8d", ha="center")

    # draw 후 오프셋 잔여(1e8 등) 재제거 — savefig 직전 필수
    fig.canvas.draw()
    for _ax in fig.axes:
        _kill_axis_offsets(_ax)

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"기술 분석 주석 차트 저장: {save_path}")
    return save_path
