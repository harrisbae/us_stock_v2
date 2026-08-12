"""
차트 패턴 기준(CHART_PATTERN_CRITERIA 메타·감지 휴리스틱·스트립 시각화).

박스권 range_box (Rectangle / Box):
    상단 저항과 하단 지지 사이에서 수평 횡보하며 에너지를 응축하는 단계.
    구현은 box_range_windows.compute_box_range_windows 로 만든 시간 창 후보에
    휴리스틱 필터(_range_box_consolidation_ok)를 적용한다.
    메인 차트 파란 박스는 calculate_box_ranges(VP·로그 포함)만 쓰고 패턴과 코드 경로가 다르다.

나머지 패턴은 OHLC 피벗·구간 통계 기반 근사(참고용).

파이프라인: range_box 추출 → post_box_bull·fvg_gap → _detect_non_range_box_patterns
→ 병합(analyze_seven_criteria) 후 스트립/메인 표시.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from matplotlib.patches import Polygon as MplPolygon, Rectangle as MplRectangle

CHART_PATTERN_CRITERIA: List[Dict[str, Any]] = [
    {"id": "double_top_m", "label": "쌍봉(M)", "confidence": 1.0, "color": "#c0392b"},
    {"id": "bear_flag", "label": "하락깃발", "confidence": 0.8, "color": "#e74c3c"},
    {"id": "bear_diamond", "label": "하락다이아", "confidence": 0.65, "color": "#9b59b6"},
    {
        "id": "range_box",
        "label": "박스권",
        "confidence": 0.5,
        "color": "#7f8c8d",
        # 개념 정의(Rectangle/Box): 상단 저항·하단 지지 사이 수평 횡보·에너지 응축
        "definition": "Rectangle/Box — 상단 저항과 하단 지지 사이에서 수평 횡보하며 에너지를 응축하는 단계",
    },
    {"id": "triple_bottom_w", "label": "역삼중창(W)", "confidence": 0.65, "color": "#16a085"},
    {"id": "bull_flag", "label": "상승깃발", "confidence": 0.8, "color": "#27ae60"},
    {"id": "asc_triangle", "label": "상승삼각", "confidence": 1.0, "color": "#2980b9"},
    {
        "id": "post_box_bull",
        "label": "박스후장대양봉",
        "confidence": 0.62,
        "color": "#e67e22",
        "definition": "range_box 구간 직후 봉이 박스 상단을 돌파한 장대 양봉(휴리스틱)",
    },
    {
        "id": "fvg_gap",
        "label": "FVG(갭)",
        "confidence": 0.55,
        "color": "#8e44ad",
        "definition": "3봉 기준 Fair Value Gap — 상방: 3번째 저가>1번째 고가, 하방: 3번째 고가<1번째 저가",
    },
    {"id": "bull_hammer", "label": "강세해머", "confidence": 0.56, "color": "#27ae60"},
    {"id": "bull_engulf", "label": "강세엥걸핑", "confidence": 0.62, "color": "#2ecc71"},
    {"id": "bear_hanging_man", "label": "행잉맨", "confidence": 0.56, "color": "#e74c3c"},
    {"id": "bear_engulf", "label": "약세엥걸핑", "confidence": 0.62, "color": "#c0392b"},
    {"id": "doji", "label": "도지(중립)", "confidence": 0.45, "color": "#7f8c8d"},
]

ID_TO_ROW = {c["id"]: i for i, c in enumerate(CHART_PATTERN_CRITERIA)}

ALL_PATTERN_IDS: Tuple[str, ...] = tuple(c["id"] for c in CHART_PATTERN_CRITERIA)
CANDLE_PATTERN_IDS: Tuple[str, ...] = (
    "bull_hammer",
    "bull_engulf",
    "bear_hanging_man",
    "bear_engulf",
    "doji",
)
CHART_SHAPE_PATTERN_IDS: Tuple[str, ...] = tuple(pid for pid in ALL_PATTERN_IDS if pid not in CANDLE_PATTERN_IDS)
BULL_CANDLE_IDS: Tuple[str, ...] = ("bull_hammer", "bull_engulf")
BEAR_CANDLE_IDS: Tuple[str, ...] = ("bear_hanging_man", "bear_engulf")
NEUTRAL_CANDLE_IDS: Tuple[str, ...] = ("doji",)


def normalize_pattern_main_ids(arg: Optional[Sequence[str]]) -> Set[str]:
    """
    메인 차트에 그릴 패턴 id 집합. None/빈 시퀀스 → 아무 것도 안 그림.
    'all' 또는 '*' → CHART_PATTERN_CRITERIA 전 id.
    'chart_all' → 기존 차트 패턴만.
    'candle_all' → 캔들 패턴만.
    'bull_candles'/'bear_candles'/'neutral_candles' 지원.
    """
    if not arg:
        return set()
    out: Set[str] = set()
    for raw in arg:
        s = str(raw).strip().lower()
        if not s:
            continue
        if s in ("all", "*"):
            return set(ALL_PATTERN_IDS)
        if s == "chart_all":
            out.update(CHART_SHAPE_PATTERN_IDS)
            continue
        if s == "candle_all":
            out.update(CANDLE_PATTERN_IDS)
            continue
        if s == "bull_candles":
            out.update(BULL_CANDLE_IDS)
            continue
        if s == "bear_candles":
            out.update(BEAR_CANDLE_IDS)
            continue
        if s == "neutral_candles":
            out.update(NEUTRAL_CANDLE_IDS)
            continue
        if s in ID_TO_ROW:
            out.add(s)
    return out


def _normalize_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """yfinance MultiIndex 컬럼 등 단일 시계열 OHLCV로 맞춤."""
    df = ohlcv.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    need = {"Open", "High", "Low", "Close"}
    if not need.issubset(set(df.columns)):
        raise ValueError("OHLCV에 Open/High/Low/Close 필요")
    return df


def _idx_ts(df: pd.DataFrame, i: int) -> pd.Timestamp:
    return pd.Timestamp(df.index[i])


def _pivot_highs(high: pd.Series, left: int = 3, right: int = 3) -> List[Tuple[int, float]]:
    hv = high.astype(float).values
    out: List[Tuple[int, float]] = []
    for i in range(left, len(hv) - right):
        v = hv[i]
        seg = hv[i - left : i + right + 1]
        if v != float(np.max(seg)):
            continue
        if v > hv[i - 1] or v > hv[i + 1]:
            out.append((i, v))
    return out


def _pivot_lows(low: pd.Series, left: int = 3, right: int = 3) -> List[Tuple[int, float]]:
    lv = low.astype(float).values
    out: List[Tuple[int, float]] = []
    for i in range(left, len(lv) - right):
        v = lv[i]
        seg = lv[i - left : i + right + 1]
        if v != float(np.min(seg)):
            continue
        if v < lv[i - 1] or v < lv[i + 1]:
            out.append((i, v))
    return out


# range_box 패턴 인정(근사): 개념상 Rectangle/Box — 저항·지지 사이 수평 횡보·에너지 응축.
_RANGE_BOX_MAX_BAND_OVER_MID = 0.26  # (고-저)/중심가
_RANGE_BOX_MAX_NET_DRIFT_RATIO = 0.62  # |종가_끝−종가_처음|/(고-저)
_RANGE_BOX_MAX_EXPAND_VS_PRIOR = 1.28  # 직전 동일 길이 구간 대비 박스 높이 상한
_RANGE_BOX_MIN_INTERIOR_CLOSE_FRAC = 0.20  # 종가가 상·하단 근처가 아닌 '안쪽'에 있는 날 비율 하한(횡보 근사)
_RANGE_BOX_EDGE_TRIM_FRAC = 0.08  # 박스 높이 대비 상·하단 각각 제외 비율
_RANGE_BOX_MIN_BALANCE_FRAC = 0.12  # 종가가 중심 위인 날 비율이 [이값, 1-이값] 밖이면 한쪽 편향으로 제외


def select_range_box_events_for_main(
    events: Optional[List[Dict[str, Any]]],
    *,
    max_count: int = 1,
) -> List[Dict[str, Any]]:
    """
    메인 차트에 그릴 range_box 이벤트만 선별한다.
    패턴 스트립에는 전체 pat_events 가 그대로 쓰이므로 여기서 줄어든 목록만 반환한다.

    정렬: 신뢰도(confidence) 내림차순 → 동률이면 종료일(end) 최근 순.
    max_count: 1이면 대표 1구간만(기본). 큰 값이면 상위 N건까지.
    """
    if not events or max_count < 1:
        return []

    def sort_key(ev: Dict[str, Any]) -> Tuple[float, float]:
        conf = float(ev.get("confidence", 0.5))
        end = ev.get("end")
        ed = pd.Timestamp(end if end is not None else ev.get("start", pd.Timestamp.min))
        return (-conf, -ed.timestamp())

    ordered = sorted(events, key=sort_key)
    return ordered[:max_count]


def _range_box_consolidation_ok(ohlcv: pd.DataFrame, br: Dict[str, Any]) -> Tuple[bool, float]:
    """
    박스권(Rectangle/Box) 개념에 대응하는지 근사 검사.

    정의: 상단 저항과 하단 지지 사이에서 수평 횡보하며 에너지를 응축하는 단계.

    검사 요약(모두 해당 구간 OHLCV 슬라이스 sub 기준):
    - hi/lo/band/mid: sub 의 High 최대·Low 최소(후보 dict 의 가격과 혼용하지 않음).
    - 박스 대비 과대 폭, 첫·끝 종가 순방향 이동 비율.
    - 종가가 박스 상·하단 8% 트림 안쪽에 있는 날 비율(횡보).
    - 종가가 중심가 위인 날 비율이 극단이면 한쪽 편향 추세로 제외.
    - 동일 봉 수만큼 직전 구간과 변동 폭 비교(응축 보너스 신뢰도).

    반환: (통과 여부, 신뢰도 0~1 근사).
    """
    sd = br.get("start_date")
    ed = br.get("end_date")
    if sd is None or ed is None:
        return False, 0.0
    sd = pd.Timestamp(sd)
    ed = pd.Timestamp(ed)
    sub = ohlcv.loc[(ohlcv.index >= sd) & (ohlcv.index <= ed)]
    if len(sub) < 5:
        return False, 0.0

    # 고·저는 반드시 해당 구간 실제 봉 기준(후보 dict 와 혼용 시 불일치 방지)
    hi = float(sub["High"].max())
    lo = float(sub["Low"].min())
    band = hi - lo
    if band <= 0:
        return False, 0.0
    mid = (hi + lo) / 2.0
    if mid <= 0 or band / mid > _RANGE_BOX_MAX_BAND_OVER_MID:
        return False, 0.0

    close_s = sub["Close"].astype(float)
    c = close_s.values
    drift = abs(float(c[-1]) - float(c[0])) / band
    if drift > _RANGE_BOX_MAX_NET_DRIFT_RATIO:
        return False, 0.0

    # 횡보: 상·하단 근처에만 붙지 않은 종가 비율
    trim = max(band * _RANGE_BOX_EDGE_TRIM_FRAC, mid * 1e-6)
    interior_mask = (close_s >= lo + trim) & (close_s <= hi - trim)
    if interior_mask.mean() < _RANGE_BOX_MIN_INTERIOR_CLOSE_FRAC:
        return False, 0.0

    # 한쪽으로만 쏠린 구간(상단만/하단만) 제외 — 중심 위 종가 비율이 극단이면 탈락
    frac_above_mid = float((close_s > mid).mean())
    if frac_above_mid < _RANGE_BOX_MIN_BALANCE_FRAC or frac_above_mid > (1.0 - _RANGE_BOX_MIN_BALANCE_FRAC):
        return False, 0.0

    conf = 0.52
    # 직전 구간: 현재와 동일한 봉 개수(len(sub))만큼 바로 이전에 붙어 비교
    idx_arr = ohlcv.index.get_indexer([sub.index[0]], method="nearest")
    start_pos = int(idx_arr[0]) if len(idx_arr) else -1
    if start_pos < 0:
        mask_fb = (ohlcv.index >= sd) & (ohlcv.index <= ed)
        pos_fb = np.flatnonzero(np.asarray(mask_fb, dtype=bool))
        start_pos = int(pos_fb[0]) if pos_fb.size else -1
    n_bars = len(sub)
    p_prior0 = start_pos - n_bars
    if p_prior0 >= 0 and start_pos >= 0:
        prior = ohlcv.iloc[p_prior0:start_pos]
        if len(prior) >= 3:
            pb = float(prior["High"].max() - prior["Low"].min())
            if pb > 0:
                if band > pb * _RANGE_BOX_MAX_EXPAND_VS_PRIOR:
                    return False, 0.0
                if band <= pb * 0.92:
                    conf = min(0.68, conf + 0.14)

    return True, float(conf)


def _events_from_box_ranges(
    ohlcv: pd.DataFrame,
    box_ranges: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """내부용: extract_range_box_pattern_events 와 동일."""
    ev: List[Dict[str, Any]] = []
    for br in box_ranges:
        sd = br.get("start_date")
        ed = br.get("end_date")
        if sd is None or ed is None:
            continue
        ok, conf = _range_box_consolidation_ok(ohlcv, br)
        if not ok:
            continue
        ev.append(
            {
                "pattern_id": "range_box",
                "start": pd.Timestamp(sd),
                "end": pd.Timestamp(ed),
                "confidence": conf,
            }
        )
    return ev


def extract_range_box_pattern_events(
    ohlcv: pd.DataFrame,
    box_ranges: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    1단계 — 박스권(range_box)만 추출.

    box_ranges 는 box_range_windows.compute_box_range_windows 의 결과(기하만)를 넘긴다.
    메인 표시용 calculate_box_ranges 와 혼용하지 말 것.
    _range_box_consolidation_ok 로 패턴 이벤트만 만든다.
    """
    return _events_from_box_ranges(ohlcv, box_ranges)


def _detect_non_range_box_patterns(ohlcv: pd.DataFrame) -> List[Dict[str, Any]]:
    """2단계 — range_box 이외 6종(피벗·구간 통계 기반)."""
    out: List[Dict[str, Any]] = []
    high = ohlcv["High"]
    low = ohlcv["Low"]
    piv_hi = _pivot_highs(high, 3, 3)
    piv_lo = _pivot_lows(low, 3, 3)
    out.extend(_detect_double_tops(ohlcv, piv_hi))
    out.extend(_detect_triple_bottoms(ohlcv, piv_lo))
    out.extend(_detect_ascending_triangles(ohlcv))
    out.extend(_detect_flags(ohlcv))
    out.extend(_detect_diamonds(ohlcv))
    return out


def _detect_double_tops(
    ohlcv: pd.DataFrame, pivots: List[Tuple[int, float]], max_events: int = 4
) -> List[Dict[str, Any]]:
    """두 고점이 유사하고 중간에 목선 아래 저점이 있는 구간(쌍봉 근사)."""
    low = ohlcv["Low"]
    events: List[Dict[str, Any]] = []
    if len(pivots) > 35:
        pivots = pivots[-35:]
    for a in range(len(pivots)):
        i, p1 = pivots[a]
        for b in range(a + 1, len(pivots)):
            j, p2 = pivots[b]
            if j - i < 8:
                continue
            mx = max(p1, p2)
            if mx <= 0:
                continue
            if abs(p1 - p2) / mx > 0.045:
                continue
            seg = low.iloc[i : j + 1].astype(float)
            seg_low = float(seg.min())
            vi = i + int(seg.values.argmin())
            lv = float(low.iloc[vi])
            if seg_low > min(p1, p2) * 0.985:
                continue
            sim = 1.0 - abs(p1 - p2) / mx
            conf = float(np.clip(0.55 + 0.45 * sim, 0.55, 1.0))
            events.append(
                {
                    "pattern_id": "double_top_m",
                    "start": _idx_ts(ohlcv, i),
                    "end": _idx_ts(ohlcv, j),
                    "confidence": conf,
                    "peak_dates": [_idx_ts(ohlcv, i), _idx_ts(ohlcv, j)],
                    "peak_prices": [float(p1), float(p2)],
                    "valley_date": _idx_ts(ohlcv, vi),
                    "valley_price": lv,
                }
            )
            if len(events) >= max_events:
                return events
    return events


def _triple_bottom_valid(
    i: int,
    j: int,
    k: int,
    ni1: int,
    ni2: int,
    l1: float,
    l2: float,
    l3: float,
    np1: float,
    np2: float,
) -> bool:
    """세 저점(i<j<k)과 목선 고점(ni1, ni2)이 시간·가격상 한 세트로 맞는지."""
    if not (i < ni1 < j < ni2 < k):
        return False
    floor = min(l1, l2, l3)
    if floor <= 0:
        return False
    # 반등 고점은 세 저점 중 최저보다 위(목선·지지 의미)
    if min(np1, np2) <= floor:
        return False
    return True


def _triple_bottom_neckline_indices(ohlcv: pd.DataFrame, i: int, j: int, k: int) -> Tuple[int, int]:
    """
    역삼중창 목선: 1~2저점 사이·2~3저점 사이 구간에서 각각 고가(반등 고점)가 가장 큰 봉 인덱스.
    """
    hv = ohlcv["High"].astype(float).values
    n = len(hv)
    if j > i + 1:
        s1 = slice(i + 1, j)
        if s1.stop > s1.start:
            ni1 = i + 1 + int(np.argmax(hv[s1]))
        else:
            ni1 = i
    else:
        ni1 = i if hv[i] >= hv[j] else j
    if k > j + 1:
        s2 = slice(j + 1, k)
        if s2.stop > s2.start:
            ni2 = j + 1 + int(np.argmax(hv[s2]))
        else:
            ni2 = j
    else:
        ni2 = j if hv[j] >= hv[k] else k
    return ni1, ni2


def _detect_triple_bottoms(
    ohlcv: pd.DataFrame, pivots: List[Tuple[int, float]], max_events: int = 3
) -> List[Dict[str, Any]]:
    lows = pivots[-35:] if len(pivots) > 35 else pivots
    events: List[Dict[str, Any]] = []
    for a in range(len(lows)):
        for b in range(a + 1, len(lows)):
            for c in range(b + 1, len(lows)):
                i, l1 = lows[a]
                j, l2 = lows[b]
                k, l3 = lows[c]
                m = max(l1, l2, l3)
                if m <= 0:
                    continue
                if max(abs(l1 - l2), abs(l2 - l3), abs(l1 - l3)) / m > 0.04:
                    continue
                if k - i < 15:
                    continue
                ni1, ni2 = _triple_bottom_neckline_indices(ohlcv, i, j, k)
                hv = ohlcv["High"].astype(float).values
                np1, np2 = float(hv[ni1]), float(hv[ni2])
                if not _triple_bottom_valid(i, j, k, ni1, ni2, l1, l2, l3, np1, np2):
                    continue
                conf = 0.65
                events.append(
                    {
                        "pattern_id": "triple_bottom_w",
                        "start": _idx_ts(ohlcv, i),
                        "end": _idx_ts(ohlcv, k),
                        "confidence": conf,
                        "pivot_dates": [
                            _idx_ts(ohlcv, i),
                            _idx_ts(ohlcv, j),
                            _idx_ts(ohlcv, k),
                        ],
                        "pivot_prices": [float(l1), float(l2), float(l3)],
                        "neckline_dates": [
                            _idx_ts(ohlcv, ni1),
                            _idx_ts(ohlcv, ni2),
                        ],
                        "neckline_prices": [np1, np2],
                    }
                )
                if len(events) >= max_events:
                    return events
    return events


def _detect_ascending_triangles(ohlcv: pd.DataFrame, win: int = 25) -> List[Dict[str, Any]]:
    """저점 연결이 우상향·고점은 횡보에 가깝다고 볼 수 있는 구간."""
    events: List[Dict[str, Any]] = []
    close = ohlcv["Close"]
    hi = ohlcv["High"]
    lo = ohlcv["Low"]
    n = len(ohlcv)
    if n < win + 5:
        return events
    x = np.arange(win, dtype=float)
    for end in range(win, n):
        sl = slice(end - win, end)
        hc = hi.iloc[sl].values
        lc = lo.iloc[sl].values
        cc = close.iloc[sl].values
        if cc.mean() <= 0:
            continue
        if np.std(hc) / cc.mean() > 0.035:
            continue
        slope_lo, _ = np.polyfit(x, lc, 1)
        if slope_lo <= 0:
            continue
        events.append(
            {
                "pattern_id": "asc_triangle",
                "start": _idx_ts(ohlcv, end - win),
                "end": _idx_ts(ohlcv, end - 1),
                "confidence": 0.72,
            }
        )
        break
    return events


def _detect_flags(ohlcv: pd.DataFrame) -> List[Dict[str, Any]]:
    """급등/급락 후 짧은 횡보(깃발 근사)."""
    events: List[Dict[str, Any]] = []
    close = ohlcv["Close"].astype(float)
    n = len(close)
    if n < 35:
        return events
    cv = close.values
    # 상승깃발: 이전 구간 상승 후 6~14일 좁은 범위
    for end in range(30, n):
        base = end - 28
        impulse_end = end - 10
        flag_end = end
        c0 = float(cv[base])
        c1 = float(cv[impulse_end])
        if c0 <= 0:
            continue
        imp_ret = (c1 - c0) / c0
        if imp_ret < 0.07:
            continue
        fc = cv[impulse_end:flag_end]
        if len(fc) < 5:
            continue
        rng = (float(np.max(fc)) - float(np.min(fc))) / float(np.mean(fc))
        if rng > 0.06:
            continue
        events.append(
            {
                "pattern_id": "bull_flag",
                "start": _idx_ts(ohlcv, impulse_end),
                "end": _idx_ts(ohlcv, flag_end - 1),
                "confidence": 0.75,
            }
        )
        break
    # 하락깃발
    for end in range(30, n):
        base = end - 28
        impulse_end = end - 10
        flag_end = end
        c0 = float(cv[base])
        c1 = float(cv[impulse_end])
        if c0 <= 0:
            continue
        imp_ret = (c1 - c0) / c0
        if imp_ret > -0.07:
            continue
        fc = cv[impulse_end:flag_end]
        if len(fc) < 5:
            continue
        rng = (float(np.max(fc)) - float(np.min(fc))) / float(np.mean(fc))
        if rng > 0.06:
            continue
        events.append(
            {
                "pattern_id": "bear_flag",
                "start": _idx_ts(ohlcv, impulse_end),
                "end": _idx_ts(ohlcv, flag_end - 1),
                "confidence": 0.75,
            }
        )
        break
    return events


def _detect_diamonds(ohlcv: pd.DataFrame, win: int = 14) -> List[Dict[str, Any]]:
    """변동성 확대 후 축소(다이아몬드 근사)."""
    events: List[Dict[str, Any]] = []
    tr = (
        (ohlcv["High"] - ohlcv["Low"]).rolling(win, min_periods=win // 2).mean()
    )
    n = len(tr)
    if n < win * 4:
        return events
    for mid in range(win * 2, n - win):
        left = float(tr.iloc[mid - win])
        midv = float(tr.iloc[mid])
        right = float(tr.iloc[mid + win])
        if left <= 0:
            continue
        if midv > left * 1.12 and right < midv * 0.88:
            events.append(
                {
                    "pattern_id": "bear_diamond",
                    "start": _idx_ts(ohlcv, mid - win),
                    "end": _idx_ts(ohlcv, mid + win),
                    "confidence": 0.55,
                }
            )
            break
    return events


def _detect_post_box_long_bull(
    ohlcv: pd.DataFrame,
    range_box_events: Sequence[Dict[str, Any]],
    *,
    max_events: int = 6,
) -> List[Dict[str, Any]]:
    """
    range_box 로 인정된 구간 직후 첫 봉이 박스 상단을 돌파한 장대 양봉인지 검사(휴리스틱).
    """
    out: List[Dict[str, Any]] = []
    idx = ohlcv.index
    for ev in range_box_events:
        if ev.get("pattern_id") != "range_box":
            continue
        sd = pd.Timestamp(ev["start"])
        ed = pd.Timestamp(ev["end"])
        sub = ohlcv.loc[(idx >= sd) & (idx <= ed)]
        if len(sub) < 3:
            continue
        box_high = float(sub["High"].max())
        box_low = float(sub["Low"].min())
        band = box_high - box_low
        if band <= 0:
            continue
        bodies = (sub["Close"].astype(float) - sub["Open"].astype(float)).abs()
        med_body = float(bodies.median()) if len(bodies) else 0.0
        after = ohlcv.loc[idx > ed]
        if len(after) < 1:
            continue
        bar = after.iloc[0]
        ts = pd.Timestamp(after.index[0])
        o = float(bar["Open"])
        c = float(bar["Close"])
        if c <= o:
            continue
        body = c - o
        if c <= box_high:
            continue
        if body < max(med_body * 1.65, band * 0.22, 1e-9):
            continue
        conf = float(np.clip(0.55 + 0.08 * (body / max(med_body, 1e-9)), 0.55, 0.92))
        out.append(
            {
                "pattern_id": "post_box_bull",
                "start": ts,
                "end": ts,
                "confidence": conf,
                "box_high": box_high,
                "box_low": box_low,
            }
        )
        if len(out) >= max_events:
            break
    return out


def _detect_fvg_gaps(
    ohlcv: pd.DataFrame,
    *,
    max_events: int = 8,
    lookback_bars: int = 220,
) -> List[Dict[str, Any]]:
    """
    3봉 FVG(Fair Value Gap). 최근 lookback_bars 안에서만 검색.
    상방: Low[i] > High[i-2]. 하방: High[i] < Low[i-2].
    """
    out: List[Dict[str, Any]] = []
    n = len(ohlcv)
    if n < 3:
        return out
    hi = ohlcv["High"].astype(float).values
    lo = ohlcv["Low"].astype(float).values
    i0 = max(2, n - lookback_bars)
    for i in range(n - 1, i0 - 1, -1):
        h0, l0 = float(hi[i - 2]), float(lo[i - 2])
        h2, l2 = float(hi[i]), float(lo[i])
        if l2 > h0:
            out.append(
                {
                    "pattern_id": "fvg_gap",
                    "direction": "bull",
                    "gap_low": h0,
                    "gap_high": l2,
                    "start": _idx_ts(ohlcv, i - 2),
                    "end": _idx_ts(ohlcv, i),
                    "confidence": 0.56,
                }
            )
        elif h2 < l0:
            out.append(
                {
                    "pattern_id": "fvg_gap",
                    "direction": "bear",
                    "gap_low": h2,
                    "gap_high": l0,
                    "start": _idx_ts(ohlcv, i - 2),
                    "end": _idx_ts(ohlcv, i),
                    "confidence": 0.56,
                }
            )
        if len(out) >= max_events:
            break
    return out


def _detect_candle_patterns(ohlcv: pd.DataFrame) -> List[Dict[str, Any]]:
    """캔들 패턴(상승/하락/중립) 간단 휴리스틱 감지."""
    out: List[Dict[str, Any]] = []
    n = len(ohlcv)
    if n < 2:
        return out
    o = ohlcv["Open"].astype(float).values
    h = ohlcv["High"].astype(float).values
    l = ohlcv["Low"].astype(float).values
    c = ohlcv["Close"].astype(float).values

    def _body(i: int) -> float:
        return abs(c[i] - o[i])

    for i in range(n):
        rng = max(h[i] - l[i], 1e-12)
        body = _body(i)
        up_w = h[i] - max(o[i], c[i])
        lo_w = min(o[i], c[i]) - l[i]

        # doji: 몸통이 전체 레인지 대비 매우 작을 때
        if body / rng <= 0.10:
            ts = _idx_ts(ohlcv, i)
            out.append(
                {
                    "pattern_id": "doji",
                    "start": ts,
                    "end": ts,
                    "confidence": 0.45,
                    "direction": "neutral",
                }
            )

        if i < 2:
            continue
        prev_close = c[i - 1]
        min_prev = float(np.min(c[max(0, i - 5) : i]))
        max_prev = float(np.max(c[max(0, i - 5) : i]))
        near_recent_low = prev_close <= (min_prev * 1.035)
        near_recent_high = prev_close >= (max_prev * 0.985)

        # bull_hammer: 저점권 + 긴 아래꼬리 + 작은 몸통
        if near_recent_low and lo_w >= body * 2.0 and up_w <= body * 0.8 and body / rng <= 0.45:
            ts = _idx_ts(ohlcv, i)
            out.append(
                {
                    "pattern_id": "bull_hammer",
                    "start": ts,
                    "end": ts,
                    "confidence": 0.56,
                    "direction": "bull",
                }
            )

        # bear_hanging_man: 고점권 + 긴 아래꼬리 + 작은 몸통
        if near_recent_high and lo_w >= body * 2.0 and up_w <= body * 0.8 and body / rng <= 0.45:
            ts = _idx_ts(ohlcv, i)
            out.append(
                {
                    "pattern_id": "bear_hanging_man",
                    "start": ts,
                    "end": ts,
                    "confidence": 0.56,
                    "direction": "bear",
                }
            )

        # engulfing: 직전 몸통을 현재 몸통이 감싸는지
        po, pc = o[i - 1], c[i - 1]
        prev_bear = pc < po
        prev_bull = pc > po
        curr_bull = c[i] > o[i]
        curr_bear = c[i] < o[i]
        prev_lo, prev_hi = min(po, pc), max(po, pc)
        curr_lo, curr_hi = min(o[i], c[i]), max(o[i], c[i])
        engulf = curr_lo <= prev_lo and curr_hi >= prev_hi
        if prev_bear and curr_bull and engulf and near_recent_low:
            ts = _idx_ts(ohlcv, i)
            out.append(
                {
                    "pattern_id": "bull_engulf",
                    "start": ts,
                    "end": ts,
                    "confidence": 0.62,
                    "direction": "bull",
                }
            )
        if prev_bull and curr_bear and engulf and near_recent_high:
            ts = _idx_ts(ohlcv, i)
            out.append(
                {
                    "pattern_id": "bear_engulf",
                    "start": ts,
                    "end": ts,
                    "confidence": 0.62,
                    "direction": "bear",
                }
            )
    return out


def analyze_seven_criteria(
    ohlcv: pd.DataFrame,
    box_ranges: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    차트 패턴 이벤트 병합 리스트(박스권·FVG·박스후양봉 + 기존 OHLC 휴리스틱).

    처리 순서:
      1) range_box — extract_range_box_pattern_events
      2) post_box_bull — range_box 이벤트 기준 직후 장대 양봉
      3) fvg_gap — 3봉 FVG
      4) 기존 6종 — _detect_non_range_box_patterns
      5) 캔들 패턴 5종 — _detect_candle_patterns

    box_ranges: box_range_windows.compute_box_range_windows 결과.
    """
    if ohlcv is None or len(ohlcv) < 10:
        return []

    try:
        ohlcv = _normalize_ohlcv(ohlcv)
    except ValueError:
        return []

    box_ranges = box_ranges or []
    merged: List[Dict[str, Any]] = []

    rb_events = extract_range_box_pattern_events(ohlcv, box_ranges)
    merged.extend(rb_events)
    merged.extend(_detect_post_box_long_bull(ohlcv, rb_events))
    merged.extend(_detect_fvg_gaps(ohlcv))
    merged.extend(_detect_non_range_box_patterns(ohlcv))
    merged.extend(_detect_candle_patterns(ohlcv))

    return merged


def plot_pattern_strip(
    ax,
    ohlcv_data: pd.DataFrame,
    events: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """메인 차트 바로 아래 서브플롯: 행별 패턴명 + 구간(broken_barh).

    events 는 analyze_seven_criteria 결과(박스권 range_box 선추출 후 타 패턴 병합).
    """
    events = events or []
    n = len(CHART_PATTERN_CRITERIA)
    ax.set_ylim(0, n)
    ax.set_yticks([i + 0.41 for i in range(n)])
    ax.set_yticklabels(
        [f'{c["label"]} · {c["id"]}' for c in CHART_PATTERN_CRITERIA],
        fontsize=5,
    )
    n_pat = len(CHART_PATTERN_CRITERIA)
    ax.set_title(
        f"차트/캔들 패턴 ({n_pat}행, 휴리스틱) — 행: 표시명 · --pattern-main id",
        fontsize=8,
        loc="left",
        pad=2,
    )
    ax.grid(True, axis="x", alpha=0.25, linestyle=":")
    ax.tick_params(axis="y", labelsize=6)

    for ev in events:
        pid = ev.get("pattern_id") or ev.get("id")
        if pid not in ID_TO_ROW:
            continue
        row = ID_TO_ROW[pid]
        start = pd.Timestamp(ev["start"])
        end = pd.Timestamp(ev["end"])
        if end < start:
            start, end = end, start
        x0 = mdates.date2num(start.to_pydatetime())
        x1 = mdates.date2num(end.to_pydatetime())
        width = max(x1 - x0, 1.0 / 32.0)
        base_conf = CHART_PATTERN_CRITERIA[row]["confidence"]
        conf = float(ev.get("confidence", base_conf))
        alpha = min(0.2 + 0.6 * conf, 0.92)
        color = ev.get("color") or CHART_PATTERN_CRITERIA[row]["color"]
        ax.broken_barh([(x0, width)], (row, 0.82), facecolors=color, alpha=alpha, edgecolors="none")

    if not events:
        ax.text(
            0.99,
            0.5,
            "감지된 패턴 없음",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=6,
            color="gray",
            alpha=0.95,
        )


# 패턴별 짝 색: (W 저점선·마커, 목선·연장) — 같은 인덱스가 한 세트
_TRIPLE_BOTTOM_MATCHED_PAIRS: List[Tuple[str, str]] = [
    ("#16a085", "#c0392b"),
    ("#2980b9", "#d35400"),
    ("#8e44ad", "#e67e22"),
]


def plot_triple_bottom_w_on_main(
    ax,
    events: Optional[List[Dict[str, Any]]] = None,
    ohlcv: Optional[pd.DataFrame] = None,
) -> None:
    """
    역삼중창(W) + 목선을 한 세트(짝 색)로만 표시. 목선 없는 이벤트는 스킵.
    """
    events = events or []
    w_no = 0
    for ev in events:
        if ev.get("pattern_id") != "triple_bottom_w":
            continue
        pds = ev.get("pivot_dates")
        pps = ev.get("pivot_prices")
        nds = ev.get("neckline_dates")
        nps = ev.get("neckline_prices")
        if (
            not pds
            or not pps
            or len(pds) != 3
            or len(pps) != 3
            or not nds
            or not nps
            or len(nds) != 2
            or len(nps) != 2
        ):
            continue

        w_col, neck_col = _TRIPLE_BOTTOM_MATCHED_PAIRS[w_no % len(_TRIPLE_BOTTOM_MATCHED_PAIRS)]
        w_no += 1

        xs = [pd.Timestamp(d) for d in pds]
        ys = [float(p) for p in pps]
        nx = [pd.Timestamp(d) for d in nds]
        ny = [float(p) for p in nps]

        ax.plot(
            xs,
            ys,
            color=w_col,
            linewidth=2.0,
            alpha=0.92,
            linestyle="-",
            solid_capstyle="round",
            zorder=38,
        )
        ax.scatter(
            xs,
            ys,
            color=w_col,
            s=36,
            zorder=39,
            edgecolors="white",
            linewidths=0.6,
        )

        ax.plot(
            nx,
            ny,
            color=neck_col,
            linestyle="--",
            linewidth=1.7,
            alpha=0.95,
            zorder=37,
            solid_capstyle="round",
        )
        ax.scatter(
            nx,
            ny,
            color=neck_col,
            s=22,
            zorder=37,
            marker="^",
            edgecolors="white",
            linewidths=0.4,
        )
        x0n = mdates.date2num(nx[0])
        x1n = mdates.date2num(nx[1])
        if x1n != x0n and ohlcv is not None and len(ohlcv.index) > 0:
            slope = (ny[1] - ny[0]) / (x1n - x0n)
            x_end = mdates.date2num(pd.Timestamp(ohlcv.index[-1]))
            y_end = ny[1] + slope * (x_end - x1n)
            ax.plot(
                [nx[1], ohlcv.index[-1]],
                [ny[1], y_end],
                color=neck_col,
                linestyle=":",
                linewidth=1.2,
                alpha=0.65,
                zorder=36,
            )

        mid_x, mid_y = xs[1], ys[1]
        title = "역삼중창(W)" if w_no == 1 else f"역삼중창(W){w_no}"
        ax.annotate(
            f"{title} · triple_bottom_w\n목선",
            xy=(mid_x, mid_y),
            xytext=(0, -18),
            textcoords="offset points",
            fontsize=6,
            color=w_col,
            fontweight="bold",
            ha="center",
            va="top",
            zorder=40,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.9,
                edgecolor=w_col,
                linewidth=1.0,
            ),
        )


def _ohlcv_between(ohlcv: pd.DataFrame, start: Any, end: Any) -> pd.DataFrame:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if e < s:
        s, e = e, s
    m = (ohlcv.index >= s) & (ohlcv.index <= e)
    return ohlcv.loc[m]


NECKLINE_COLOR_DOUBLE_TOP = "#d35400"  # 쌍봉 목선(역삼중창 목선과 동일 계열)


def plot_double_top_m_on_main(
    ax,
    events: Optional[List[Dict[str, Any]]] = None,
    ohlcv: Optional[pd.DataFrame] = None,
) -> None:
    """쌍봉(M) 곡선 + 두 고점 사이 저점 수평 목선 + 연장 점선."""
    events = events or []
    col = CHART_PATTERN_CRITERIA[ID_TO_ROW["double_top_m"]]["color"]
    neck = NECKLINE_COLOR_DOUBLE_TOP
    for n, ev in enumerate(events):
        pds = ev.get("peak_dates")
        pps = ev.get("peak_prices")
        vd = ev.get("valley_date")
        vp = ev.get("valley_price")
        if not pds or not pps or len(pds) != 2 or vd is None or vp is None:
            continue
        t1 = pd.Timestamp(pds[0])
        t2 = pd.Timestamp(pds[1])
        xs = [t1, pd.Timestamp(vd), t2]
        ys = [float(pps[0]), float(vp), float(pps[1])]
        ax.plot(xs, ys, color=col, linewidth=2.0, alpha=0.9, zorder=38, solid_capstyle="round")
        ax.scatter(xs, ys, color=col, s=34, zorder=39, edgecolors="white", linewidths=0.5)

        x0 = mdates.date2num(t1)
        x1 = mdates.date2num(t2)
        vp = float(vp)
        ax.plot(
            [t1, t2],
            [vp, vp],
            color=neck,
            linestyle="--",
            linewidth=1.7,
            alpha=0.95,
            zorder=37,
            solid_capstyle="round",
        )
        if ohlcv is not None and len(ohlcv.index) > 0:
            t_end = pd.Timestamp(ohlcv.index[-1])
            if t_end > t2:
                ax.plot(
                    [t2, t_end],
                    [vp, vp],
                    color=neck,
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.65,
                    zorder=36,
                )
        xm = mdates.num2date((x0 + x1) / 2.0)
        ax.annotate(
            "목선 · double_top_m",
            xy=(xm, vp),
            xytext=(0, -12),
            textcoords="offset points",
            fontsize=6,
            color=neck,
            fontweight="bold",
            ha="center",
            va="top",
            zorder=40,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                alpha=0.85,
                edgecolor=neck,
                linewidth=0.6,
            ),
        )

        ax.annotate(
            "쌍봉(M) · double_top_m" if n == 0 else f"M{n+1} · double_top_m",
            xy=(xs[1], ys[1]),
            xytext=(0, 12),
            textcoords="offset points",
            fontsize=6,
            color=col,
            fontweight="bold",
            ha="center",
            zorder=40,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.88, edgecolor=col, linewidth=0.8),
        )


def plot_range_box_on_main(
    ax,
    events: Optional[List[Dict[str, Any]]] = None,
    ohlcv: Optional[pd.DataFrame] = None,
) -> None:
    events = events or []
    if ohlcv is None or len(ohlcv) == 0:
        return
    col = CHART_PATTERN_CRITERIA[ID_TO_ROW["range_box"]]["color"]
    for n, ev in enumerate(events):
        sub = _ohlcv_between(ohlcv, ev["start"], ev["end"])
        if len(sub) < 1:
            continue
        y0 = float(sub["Low"].min())
        y1 = float(sub["High"].max())
        x0 = mdates.date2num(pd.Timestamp(ev["start"]))
        x1 = mdates.date2num(pd.Timestamp(ev["end"]))
        ax.fill_betweenx([y0, y1], x0, x1, color=col, alpha=0.12, zorder=32)
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=col, linewidth=1.0, alpha=0.85, zorder=33)
        xm = mdates.num2date((x0 + x1) / 2.0)
        ym = (y0 + y1) / 2.0
        lbl = "박스권 · range_box" if n == 0 else f"박스권({n+1}) · range_box"
        ax.annotate(
            lbl,
            xy=(xm, ym),
            fontsize=6,
            color=col,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=40,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.88, edgecolor=col, linewidth=0.6),
        )


def plot_asc_triangle_on_main(
    ax,
    events: Optional[List[Dict[str, Any]]] = None,
    ohlcv: Optional[pd.DataFrame] = None,
) -> None:
    events = events or []
    if ohlcv is None:
        return
    col = CHART_PATTERN_CRITERIA[ID_TO_ROW["asc_triangle"]]["color"]
    for ev in events:
        sub = _ohlcv_between(ohlcv, ev["start"], ev["end"])
        if len(sub) < 5:
            continue
        hi = sub["High"].astype(float).values
        lo = sub["Low"].astype(float).values
        idx = sub.index
        xh = float(np.max(hi))
        tnums = mdates.date2num([pd.Timestamp(x) for x in idx])
        m, b = np.polyfit(tnums, lo, 1)
        t0, t1 = tnums[0], tnums[-1]
        y0, y1 = m * t0 + b, m * t1 + b
        ax.plot([idx[0], idx[-1]], [xh, xh], color=col, linestyle="-", linewidth=1.4, alpha=0.9, zorder=38)
        ax.plot([idx[0], idx[-1]], [y0, y1], color=col, linestyle="--", linewidth=1.4, alpha=0.9, zorder=38)
        tmid = (t0 + t1) / 2.0
        xm = mdates.num2date(tmid)
        y_support_mid = m * tmid + b
        tri_bbox = dict(
            boxstyle="round,pad=0.15",
            facecolor="white",
            alpha=0.9,
            edgecolor=col,
            linewidth=0.5,
        )
        ax.annotate(
            "저항선 · asc_triangle",
            xy=(xm, xh),
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=5,
            color=col,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=41,
            bbox=tri_bbox,
        )
        ax.annotate(
            "지지선 · asc_triangle",
            xy=(xm, y_support_mid),
            xytext=(0, -11),
            textcoords="offset points",
            fontsize=5,
            color=col,
            fontweight="bold",
            ha="center",
            va="top",
            zorder=41,
            bbox=tri_bbox,
        )
        ax.annotate(
            "상승삼각 · asc_triangle",
            xy=(idx[len(idx) // 2], (xh + (y0 + y1) / 2) / 2),
            fontsize=6,
            color=col,
            fontweight="bold",
            zorder=40,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor=col, linewidth=0.7),
        )


def plot_flag_on_main(
    ax,
    events: Optional[List[Dict[str, Any]]] = None,
    ohlcv: Optional[pd.DataFrame] = None,
    pattern_id: str = "bull_flag",
) -> None:
    events = events or []
    if ohlcv is None:
        return
    col = CHART_PATTERN_CRITERIA[ID_TO_ROW[pattern_id]]["color"]
    label = "상승깃발" if pattern_id == "bull_flag" else "하락깃발"
    for n, ev in enumerate(events):
        sub = _ohlcv_between(ohlcv, ev["start"], ev["end"])
        if len(sub) < 1:
            continue
        y0 = float(sub["Low"].min())
        y1 = float(sub["High"].max())
        x0 = mdates.date2num(pd.Timestamp(ev["start"]))
        x1 = mdates.date2num(pd.Timestamp(ev["end"]))
        ax.fill_betweenx([y0, y1], x0, x1, color=col, alpha=0.14, zorder=32)
        ax.plot([x0, x1], [y1, y1], color=col, linewidth=1.2, linestyle=":", alpha=0.9, zorder=34)
        ax.plot([x0, x1], [y0, y0], color=col, linewidth=1.2, linestyle=":", alpha=0.9, zorder=34)
        if pattern_id == "bull_flag":
            xm = mdates.num2date((x0 + x1) / 2.0)
            neck_bbox = dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                alpha=0.9,
                edgecolor=col,
                linewidth=0.5,
            )
            ax.annotate(
                "지지선 · bull_flag",
                xy=(xm, y0),
                xytext=(0, -11),
                textcoords="offset points",
                fontsize=5,
                color=col,
                fontweight="bold",
                ha="center",
                va="top",
                zorder=41,
                bbox=neck_bbox,
            )
            ax.annotate(
                "저항선 · bull_flag",
                xy=(xm, y1),
                xytext=(0, 11),
                textcoords="offset points",
                fontsize=5,
                color=col,
                fontweight="bold",
                ha="center",
                va="bottom",
                zorder=41,
                bbox=neck_bbox,
            )
        ax.annotate(
            f"{label} · {pattern_id}" if n == 0 else f"{label}{n+1} · {pattern_id}",
            xy=(pd.Timestamp(ev["end"]), (y0 + y1) / 2),
            fontsize=6,
            color=col,
            fontweight="bold",
            zorder=40,
            ha="right",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor=col, linewidth=0.6),
        )


def plot_bear_diamond_on_main(
    ax,
    events: Optional[List[Dict[str, Any]]] = None,
    ohlcv: Optional[pd.DataFrame] = None,
) -> None:
    events = events or []
    if ohlcv is None:
        return
    col = CHART_PATTERN_CRITERIA[ID_TO_ROW["bear_diamond"]]["color"]
    for ev in events:
        sub = _ohlcv_between(ohlcv, ev["start"], ev["end"])
        if len(sub) < 2:
            continue
        t0 = mdates.date2num(pd.Timestamp(ev["start"]))
        t1 = mdates.date2num(pd.Timestamp(ev["end"]))
        ytop = float(sub["High"].max())
        ybot = float(sub["Low"].min())
        ymid = (ytop + ybot) / 2.0
        tmid = (t0 + t1) / 2.0
        verts = [(t0, ymid), (tmid, ytop), (t1, ymid), (tmid, ybot)]
        poly = MplPolygon(verts, closed=True, facecolor=col, edgecolor=col, alpha=0.18, linewidth=1.2, zorder=33)
        ax.add_patch(poly)
        ax.annotate(
            "하락다이아 · bear_diamond",
            xy=(mdates.num2date(tmid), ymid),
            fontsize=6,
            color=col,
            fontweight="bold",
            zorder=40,
            ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor=col, linewidth=0.6),
        )


def _median_bar_xwidth_num(ohlcv: pd.DataFrame) -> float:
    idx = ohlcv.index
    if len(idx) < 2:
        return 0.8
    nums = mdates.date2num(pd.DatetimeIndex(idx).to_pydatetime())
    d = float(np.median(np.diff(nums)))
    return max(d * 0.4, 1e-6)


def plot_post_box_bull_on_main(
    ax,
    events: Optional[List[Dict[str, Any]]] = None,
    ohlcv: Optional[pd.DataFrame] = None,
) -> None:
    """박스 직후 장대 양봉: 돌파 봉 몸통 강조 + 박스 상단선."""
    events = events or []
    if ohlcv is None or not len(events):
        return
    col = CHART_PATTERN_CRITERIA[ID_TO_ROW["post_box_bull"]]["color"]
    half = _median_bar_xwidth_num(ohlcv) / 2.0
    for ev in events:
        if ev.get("pattern_id") != "post_box_bull":
            continue
        ts = pd.Timestamp(ev["start"])
        if ts not in ohlcv.index:
            continue
        bar = ohlcv.loc[ts]
        o = float(bar["Open"])
        c = float(bar["Close"])
        if c <= o:
            continue
        xn = mdates.date2num(ts.to_pydatetime())
        rect = MplRectangle(
            (xn - half, o),
            2 * half,
            c - o,
            facecolor=col,
            edgecolor=col,
            alpha=0.35,
            linewidth=1.0,
            zorder=34,
        )
        ax.add_patch(rect)
        bh = ev.get("box_high")
        if bh is not None:
            ax.axhline(float(bh), xmin=0, xmax=1, color=col, linestyle="--", alpha=0.5, linewidth=0.9, zorder=33)
        ax.annotate(
            "박스후장대양봉 · post_box_bull",
            xy=(ts, c),
            fontsize=6,
            color=col,
            fontweight="bold",
            zorder=40,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor=col, linewidth=0.6),
        )


def plot_fvg_gap_on_main(
    ax,
    events: Optional[List[Dict[str, Any]]] = None,
    ohlcv: Optional[pd.DataFrame] = None,
) -> None:
    """FVG 갭 구간을 가격대 박스로 표시. 상승/하락 최신 각 1개에 FVG↑/FVG↓ 라벨."""
    events = events or []
    if ohlcv is None or not len(events):
        return

    col_base = CHART_PATTERN_CRITERIA[ID_TO_ROW["fvg_gap"]]["color"]
    fvg_events: List[Dict[str, Any]] = [
        ev for ev in events if ev.get("pattern_id") == "fvg_gap"
    ]
    if not fvg_events:
        return

    fvg_events_sorted = sorted(
        fvg_events,
        key=lambda ev: pd.Timestamp(ev.get("end") or ev.get("start")),
    )
    latest_bull = None
    latest_bear = None
    for ev in fvg_events_sorted:
        if (ev.get("direction") or "bull") == "bear":
            latest_bear = ev
        else:
            latest_bull = ev
    label_set = {id(x) for x in (latest_bull, latest_bear) if x is not None}

    for ev in fvg_events_sorted:
        gl = ev.get("gap_low")
        gh = ev.get("gap_high")
        if gl is None or gh is None:
            continue
        gl, gh = float(gl), float(gh)
        if gl > gh:
            gl, gh = gh, gl
        t0 = mdates.date2num(pd.Timestamp(ev["start"]).to_pydatetime())
        t1 = mdates.date2num(pd.Timestamp(ev["end"]).to_pydatetime())
        direction = ev.get("direction") or "bull"
        col = "#27ae60" if direction == "bull" else "#c0392b"

        poly = MplPolygon(
            [(t0, gl), (t1, gl), (t1, gh), (t0, gh)],
            closed=True,
            facecolor=col,
            edgecolor=col_base,
            alpha=0.10,
            linewidth=0.7,
            zorder=32,
        )
        ax.add_patch(poly)

        if id(ev) in label_set:
            tag = "FVG↓" if direction == "bear" else "FVG↑"
            ax.annotate(
                tag,
                xy=(mdates.num2date((t0 + t1) / 2.0), (gl + gh) / 2.0),
                fontsize=6,
                color=col,
                fontweight="bold",
                zorder=40,
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.75,
                    edgecolor=col,
                    linewidth=0.5,
                ),
            )


def plot_candle_patterns_on_main(
    ax,
    events: Optional[List[Dict[str, Any]]] = None,
    ohlcv: Optional[pd.DataFrame] = None,
) -> None:
    """캔들 패턴 마커(강세 ▲ / 약세 ▼ / 중립 ●)를 메인 차트에 표시."""
    events = events or []
    if ohlcv is None or not len(events):
        return
    ohlcv_n = _normalize_ohlcv(ohlcv)
    low_s = ohlcv_n["Low"].astype(float)
    high_s = ohlcv_n["High"].astype(float)
    cmin = float(low_s.min())
    cmax = float(high_s.max())
    pad = max((cmax - cmin) * 0.022, 1e-6)

    marker_map = {
        "bull_hammer": ("^", "#27ae60", "BH", "bull"),
        "bull_engulf": ("^", "#2ecc71", "BE", "bull"),
        "bear_hanging_man": ("v", "#e74c3c", "HM", "bear"),
        "bear_engulf": ("v", "#c0392b", "SE", "bear"),
        "doji": ("o", "#7f8c8d", "DJ", "neutral"),
    }
    for ev in events:
        pid = str(ev.get("pattern_id") or "")
        if pid not in marker_map:
            continue
        ts = pd.Timestamp(ev.get("start"))
        if ts not in ohlcv_n.index:
            continue
        marker, color, short, direction = marker_map[pid]
        low_v = float(low_s.loc[ts])
        high_v = float(high_s.loc[ts])
        if direction == "bull":
            y_m = low_v - pad * 0.35
            y_t = low_v - pad * 1.15
            va = "top"
        elif direction == "bear":
            y_m = high_v + pad * 0.35
            y_t = high_v + pad * 1.15
            va = "bottom"
        else:
            y_m = (low_v + high_v) / 2.0
            y_t = y_m
            va = "center"
        ax.scatter(
            [ts], [y_m],
            marker=marker, s=26, color=color, zorder=41,
            edgecolors="white", linewidths=0.5,
        )
        if direction == "neutral":
            ax.annotate(
                short,
                xy=(ts, y_m),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=5.2,
                fontweight="bold",
                color=color,
                ha="left",
                va="center",
                zorder=42,
                bbox=dict(
                    boxstyle="round,pad=0.12",
                    facecolor="white",
                    edgecolor=color,
                    linewidth=0.55,
                    alpha=0.90,
                ),
            )
        else:
            ax.annotate(
                short,
                xy=(ts, y_m),
                xytext=(ts, y_t),
                textcoords="data",
                fontsize=5.2,
                fontweight="bold",
                color=color,
                ha="center",
                va=va,
                zorder=42,
                bbox=dict(
                    boxstyle="round,pad=0.12",
                    facecolor="white",
                    edgecolor=color,
                    linewidth=0.55,
                    alpha=0.90,
                ),
                arrowprops=dict(
                    arrowstyle="-",
                    color=color,
                    lw=0.55,
                    linestyle=":",
                    alpha=0.55,
                    shrinkA=0,
                    shrinkB=1,
                ),
            )


def plot_pattern_main_overlays(
    ax,
    ohlcv: pd.DataFrame,
    events: Optional[List[Dict[str, Any]]],
    enabled_ids: Set[str],
    *,
    range_box_main_max: int = 1,
    main_min_confidence: float = 0.0,
) -> None:
    """pattern_id별 메인 차트 오버레이. enabled_ids가 비어 있으면 아무 것도 안 함.

    range_box 는 스트립과 달리 메인에서만 상위 N건(신뢰도·종료일)으로 선별해 표시한다.
    """
    if not enabled_ids:
        return
    try:
        ohlcv_n = _normalize_ohlcv(ohlcv)
    except ValueError:
        return
    events = events or []
    by_id: Dict[str, List[Dict[str, Any]]] = {}
    for e in events:
        pid = e.get("pattern_id")
        conf = float(e.get("confidence", 0.0))
        if pid in enabled_ids and conf >= float(main_min_confidence):
            by_id.setdefault(pid, []).append(e)
    if "triple_bottom_w" in enabled_ids:
        plot_triple_bottom_w_on_main(ax, by_id.get("triple_bottom_w"), ohlcv_n)
    if "double_top_m" in enabled_ids:
        plot_double_top_m_on_main(ax, by_id.get("double_top_m"), ohlcv_n)
    if "range_box" in enabled_ids:
        rb_all = by_id.get("range_box") or []
        if range_box_main_max < 1:
            rb_show: List[Dict[str, Any]] = []
        else:
            rb_show = select_range_box_events_for_main(rb_all, max_count=range_box_main_max)
            if len(rb_all) > len(rb_show):
                print(
                    f"  메인 range_box: 전체 {len(rb_all)}건 중 선별 {len(rb_show)}건 표시 "
                    f"(신뢰도 우선, 동률 시 종료일 최근)"
                )
        plot_range_box_on_main(ax, rb_show, ohlcv_n)
    if "asc_triangle" in enabled_ids:
        plot_asc_triangle_on_main(ax, by_id.get("asc_triangle"), ohlcv_n)
    if "bull_flag" in enabled_ids:
        plot_flag_on_main(ax, by_id.get("bull_flag"), ohlcv_n, "bull_flag")
    if "bear_flag" in enabled_ids:
        plot_flag_on_main(ax, by_id.get("bear_flag"), ohlcv_n, "bear_flag")
    if "bear_diamond" in enabled_ids:
        plot_bear_diamond_on_main(ax, by_id.get("bear_diamond"), ohlcv_n)
    if "post_box_bull" in enabled_ids:
        plot_post_box_bull_on_main(ax, by_id.get("post_box_bull"), ohlcv_n)
    if "fvg_gap" in enabled_ids:
        plot_fvg_gap_on_main(ax, by_id.get("fvg_gap"), ohlcv_n)
    candle_ids = set(CANDLE_PATTERN_IDS)
    if enabled_ids & candle_ids:
        candle_events: List[Dict[str, Any]] = []
        for cid in candle_ids:
            candle_events.extend(by_id.get(cid, []))
        plot_candle_patterns_on_main(ax, candle_events, ohlcv_n)
