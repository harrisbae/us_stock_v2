from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _compute_disparity(close: pd.Series, ma_window: int) -> pd.Series:
    ma = close.rolling(window=ma_window).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        disp = (close / ma) * 100.0
    return disp.replace([np.inf, -np.inf], np.nan)


def _classify_trend(close: pd.Series, fast: int = 20, slow: int = 60) -> str:
    sma_fast = close.rolling(window=fast).mean()
    sma_slow = close.rolling(window=slow).mean()
    if len(close) < max(fast, slow) + 2:
        return "unknown"
    if pd.isna(sma_fast.iloc[-1]) or pd.isna(sma_slow.iloc[-1]):
        return "unknown"
    fast_slope = _safe_float(sma_fast.iloc[-1] - sma_fast.iloc[-2])
    if sma_fast.iloc[-1] > sma_slow.iloc[-1] and fast_slope > 0:
        return "up"
    if sma_fast.iloc[-1] < sma_slow.iloc[-1] and fast_slope < 0:
        return "down"
    return "sideways"


def _classify_volume(volume: pd.Series) -> str:
    if len(volume) < 20:
        return "unknown"
    v5 = volume.rolling(5).mean().iloc[-1]
    v20 = volume.rolling(20).mean().iloc[-1]
    if pd.isna(v5) or pd.isna(v20) or v20 <= 0:
        return "unknown"
    ratio = float(v5 / v20)
    if ratio >= 1.10:
        return "up"
    if ratio <= 0.90:
        return "down"
    return "flat"


def _rsi_state(rsi14: Optional[pd.Series]) -> str:
    if rsi14 is None or len(rsi14) == 0 or pd.isna(rsi14.iloc[-1]):
        return "unknown"
    v = float(rsi14.iloc[-1])
    if v >= 70:
        return "overbought"
    if v <= 30:
        return "oversold"
    return "neutral"


def _macd_state(macd: Optional[pd.Series], macd_signal: Optional[pd.Series]) -> str:
    if macd is None or macd_signal is None or len(macd) == 0 or len(macd_signal) == 0:
        return "unknown"
    if pd.isna(macd.iloc[-1]) or pd.isna(macd_signal.iloc[-1]):
        return "unknown"
    if float(macd.iloc[-1]) > float(macd_signal.iloc[-1]):
        return "bull"
    return "bear"


def _bb_state(close: pd.Series, bb_upper: Optional[pd.Series], bb_lower: Optional[pd.Series]) -> str:
    if bb_upper is None or bb_lower is None or len(close) == 0:
        return "unknown"
    c = close.iloc[-1]
    u = bb_upper.iloc[-1] if len(bb_upper) else np.nan
    l = bb_lower.iloc[-1] if len(bb_lower) else np.nan
    if pd.isna(c) or pd.isna(u) or pd.isna(l):
        return "unknown"
    if c >= u:
        return "upper_break"
    if c <= l:
        return "lower_break"
    return "inside"


def _monthly_confirm(close: pd.Series) -> str:
    if len(close) < 80:
        return "unknown"
    m_close = close.resample("M").last().dropna()
    if len(m_close) < 6:
        return "unknown"
    m20 = m_close.rolling(window=4).mean()
    m60 = m_close.rolling(window=6).mean()
    if pd.isna(m20.iloc[-1]) or pd.isna(m60.iloc[-1]):
        return "unknown"
    if m20.iloc[-1] > m60.iloc[-1]:
        return "up"
    if m20.iloc[-1] < m60.iloc[-1]:
        return "down"
    return "sideways"


def analyze_disparity_strategy(
    ohlcv: pd.DataFrame,
    *,
    rsi14: Optional[pd.Series] = None,
    macd: Optional[pd.Series] = None,
    macd_signal: Optional[pd.Series] = None,
    bb_upper: Optional[pd.Series] = None,
    bb_lower: Optional[pd.Series] = None,
    disparity_ma: int = 20,
    disparity_low: float = 95.0,
    disparity_high: float = 105.0,
    conf_weight_depth: float = 0.22,
    conf_weight_volume: float = 0.16,
    conf_weight_trend: float = 0.10,
) -> Dict[str, Any]:
    close = ohlcv["Close"].astype(float)
    volume = ohlcv["Volume"].astype(float)
    disparity = _compute_disparity(close, disparity_ma)
    vol5 = volume.rolling(5).mean()
    vol20 = volume.rolling(20).mean()
    sma20 = close.rolling(20).mean()
    sma60 = close.rolling(60).mean()
    trend = _classify_trend(close, fast=20, slow=60)
    vol_state = _classify_volume(volume)
    rsi_state = _rsi_state(rsi14)
    macd_state = _macd_state(macd, macd_signal)
    bb_state = _bb_state(close, bb_upper, bb_lower)
    m_confirm = _monthly_confirm(close)

    events: List[Dict[str, Any]] = []
    buy_candidate_flags: List[bool] = []
    for i in range(len(close)):
        d = disparity.iloc[i]
        if pd.isna(d):
            continue
        state = "NEUTRAL"
        action = "wait"
        color = "gray"
        marker = "o"
        short = "W"
        conf = 0.50
        # 각 시점 기준으로 추세/거래량을 재평가해야 레이블 판정이 왜곡되지 않는다.
        loc_close = close.iloc[: i + 1]
        loc_vol = volume.iloc[: i + 1]
        loc_trend = _classify_trend(loc_close, fast=20, slow=60)
        loc_vol_state = _classify_volume(loc_vol)
        loc_sma20 = loc_close.rolling(20).mean()
        slope_v = float(loc_sma20.iloc[-1] - loc_sma20.iloc[-2]) if len(loc_sma20) >= 2 and pd.notna(loc_sma20.iloc[-1]) and pd.notna(loc_sma20.iloc[-2]) else 0.0
        if loc_trend == "up":
            trend_short = "U"
        elif loc_trend == "down":
            trend_short = "D"
        else:
            trend_short = "S"
        if loc_vol_state == "up":
            vol_short = "U"
        elif loc_vol_state == "down":
            vol_short = "D"
        else:
            vol_short = "F"

        if d <= disparity_low:
            if loc_trend == "up" and loc_vol_state == "down":
                state, action, color, marker, short, conf = "D95_PULLBACK", "buy_scale_in", "#27ae60", "^", "B", 0.68
            elif loc_trend == "down":
                state, action, color, marker, short, conf = "D95_FALLING_KNIFE", "wait", "#c0392b", "v", "FK", 0.62
        elif d >= disparity_high:
            if loc_trend == "up" and loc_vol_state == "up":
                state, action, color, marker, short, conf = "D105_HOLD_MOMO", "hold", "#2980b9", "^", "H", 0.67
            elif loc_trend == "up" and loc_vol_state == "down":
                state, action, color, marker, short, conf = "D105_GREED_TAKEPROFIT", "take_profit", "#f39c12", "v", "TP", 0.70
            elif loc_trend == "down":
                state, action, color, marker, short, conf = "D105_DEADCAT", "reduce", "#8e44ad", "v", "DC", 0.71

        if state != "NEUTRAL":
            ts = pd.Timestamp(close.index[i])
            events.append(
                {
                    "date": ts,
                    "price": float(close.iloc[i]),
                    "state": state,
                    "action": action,
                    "confidence": conf,
                    "color": color,
                    "marker": marker,
                    "short": short,
                    "disparity": float(d),
                    "trend": loc_trend,
                    "volume_state": loc_vol_state,
                    "trend_short": trend_short,
                    "trend_slope": slope_v,
                    "volume_short": vol_short,
                }
            )
        buy_candidate_flags.append(state == "D95_PULLBACK")

    buy_candidate_ranges: List[Dict[str, Any]] = []
    s_idx: Optional[int] = None
    for i, flag in enumerate(buy_candidate_flags):
        if flag and s_idx is None:
            s_idx = i
        is_end = (not flag) or (i == len(buy_candidate_flags) - 1)
        if s_idx is not None and is_end:
            e_idx = i if flag and i == len(buy_candidate_flags) - 1 else (i - 1)
            if e_idx >= s_idx:
                sub = ohlcv.iloc[s_idx : e_idx + 1]
                if len(sub):
                    lo = float(sub["Low"].min())
                    hi = float(sub["High"].max())
                    mid = (lo + hi) / 2.0
                    pad = max((hi - lo) * 0.08, max(mid, 1.0) * 0.002)
                    # 구간별 동적 신뢰도(0.50~0.92)
                    sub_disp = disparity.iloc[s_idx : e_idx + 1].dropna()
                    d_avg = float(sub_disp.mean()) if len(sub_disp) else float(disparity_low)
                    depth_score = np.clip((float(disparity_low) - d_avg) / 3.0, 0.0, 1.0)

                    sub_v5 = vol5.iloc[e_idx]
                    sub_v20 = vol20.iloc[e_idx]
                    if pd.notna(sub_v5) and pd.notna(sub_v20) and float(sub_v20) > 0:
                        v_ratio = float(sub_v5 / sub_v20)
                        # 거래량이 줄수록(조정 완화) 가산점
                        vol_score = np.clip((1.0 - v_ratio) / 0.30, 0.0, 1.0)
                    else:
                        vol_score = 0.35

                    s20 = sma20.iloc[e_idx]
                    s60 = sma60.iloc[e_idx]
                    if pd.notna(s20) and pd.notna(s60) and float(s60) != 0.0:
                        trend_score = np.clip((float(s20) - float(s60)) / (abs(float(s60)) * 0.05), 0.0, 1.0)
                    else:
                        trend_score = 0.30

                    w_depth = float(max(conf_weight_depth, 0.0))
                    w_volume = float(max(conf_weight_volume, 0.0))
                    w_trend = float(max(conf_weight_trend, 0.0))
                    dyn_conf = 0.50 + (w_depth * depth_score) + (w_volume * vol_score) + (w_trend * trend_score)
                    dyn_conf = float(np.clip(dyn_conf, 0.50, 0.92))
                    buy_candidate_ranges.append(
                        {
                            "start": pd.Timestamp(ohlcv.index[s_idx]),
                            "end": pd.Timestamp(ohlcv.index[e_idx]),
                            "low": lo - pad,
                            "high": hi + pad,
                            "count": int(e_idx - s_idx + 1),
                            "confidence": dyn_conf,
                        }
                    )
            s_idx = None

    latest_disp = _safe_float(disparity.iloc[-1], default=np.nan)
    latest_event = events[-1] if events else None

    action_map = {
        "buy_scale_in": "적극 분할 매수",
        "hold": "보유",
        "take_profit": "수익 실현",
        "reduce": "비중 축소",
        "wait": "관망",
    }
    final_action = action_map.get(latest_event["action"], "관망") if latest_event else "관망"

    tech_summary = (
        f"이격도({disparity_ma}MA): {latest_disp:.2f}, 추세: {trend}, 거래량: {vol_state}, "
        f"RSI: {rsi_state}, MACD: {macd_state}, BB: {bb_state}, 월봉확인: {m_confirm}"
    )
    risk_note = (
        "안전마진 점검: 95 이하+우상향 조정이면 유리, 하락 추세의 95 이탈은 낙하 칼날 위험. "
        "시장주기 점검: 과열(105 상회)에서 거래량 둔화는 탐욕 구간 리스크."
    )
    if latest_event is not None:
        plan = f"최종 액션: {final_action} ({latest_event['state']}, 신뢰도 {latest_event['confidence']:.2f})"
    else:
        plan = f"최종 액션: {final_action} (명확한 D95/D105 이벤트 없음)"

    return {
        "disparity": disparity,
        "events": events,
        "buy_candidate_ranges": buy_candidate_ranges,
        "latest_event": latest_event,
        "final_action": final_action,
        "tech_summary": tech_summary,
        "risk_note": risk_note,
        "plan": plan,
    }


def plot_disparity_strategy_on_main(
    ax,
    events: List[Dict[str, Any]],
    *,
    max_markers: int = 30,
    min_label_gap_days: int = 4,
    max_labels: int = 10,
) -> List[Any]:
    if not events:
        return []
    show = events[-max_markers:]
    latest_by_state: Dict[str, Dict[str, Any]] = {}
    for ev in show:
        ts = pd.Timestamp(ev["date"])
        y = float(ev["price"])
        ax.scatter([ts], [y], marker=ev["marker"], s=38, color=ev["color"], edgecolors="white", linewidths=0.5, zorder=43)

        # 상태별 최신 이벤트 1건만 우측 레이블 후보로 사용
        st = str(ev.get("state", ""))
        if st and (st not in latest_by_state or pd.Timestamp(latest_by_state[st]["date"]) < ts):
            latest_by_state[st] = ev

    # 우측 레이블은 최근 구간(show)뿐 아니라 전체 이벤트 기준으로도 보장한다.
    for ev in events:
        st = str(ev.get("state", ""))
        ts = pd.Timestamp(ev.get("date"))
        if st and (st not in latest_by_state or pd.Timestamp(latest_by_state[st]["date"]) < ts):
            latest_by_state[st] = ev

    if latest_by_state:
        # 우측 충돌 방지를 위해 y를 가격 비례가 아닌 세로 고정 슬롯으로 배치한다.
        cand = sorted(
            latest_by_state.values(),
            key=lambda e: pd.Timestamp(e.get("date", pd.Timestamp.min)),
            reverse=True,
        )[:max_labels]
        y_slots = [0.92, 0.86, 0.80, 0.74, 0.68, 0.62, 0.56, 0.50]
        for idx, ev in enumerate(cand):
            if idx >= len(y_slots):
                break
            y_frac = y_slots[idx]
            label_text = f"{ev.get('short', '')} ({ev.get('disparity', np.nan):.1f})"
            ax.text(
                1.09,
                y_frac,
                label_text,
                transform=ax.transAxes,
                fontsize=6,
                color=ev.get("color", "black"),
                ha="left",
                va="center",
                zorder=44,
                clip_on=False,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.88,
                    edgecolor=ev.get("color", "black"),
                    linewidth=0.7,
                ),
            )
    elif events:
        # 상태 분류가 단일/희소해도 최신 이벤트 1건은 우측에 표시
        ev = events[-1]
        label_text = f"{ev.get('short', '')} ({ev.get('disparity', np.nan):.1f})"
        ax.text(
            1.09,
            0.92,
            label_text,
            transform=ax.transAxes,
            fontsize=6,
            color=ev.get("color", "black"),
            ha="left",
            va="center",
            zorder=44,
            clip_on=False,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                alpha=0.88,
                edgecolor=ev.get("color", "black"),
                linewidth=0.7,
            ),
        )
    return [
        Line2D([], [], marker="^", linestyle="None", color="#27ae60", label="Disparity Buy/Pullback"),
        Line2D([], [], marker="v", linestyle="None", color="#f39c12", label="Disparity TP/Reduce"),
    ]

