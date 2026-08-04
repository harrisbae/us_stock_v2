"""
ADX / DMI (+DI, -DI) 추세·강도 판별 모듈 (Wilder 방식).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def calculate_adx_dmi(
    ohlcv: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """
    OHLCV에서 ADX, +DI, -DI 시계열을 계산합니다.

    Returns:
        DataFrame columns: ADX, Plus_DI, Minus_DI
    """
    df = ohlcv.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = _wilder_smooth(tr, period)
    plus_di = 100.0 * _wilder_smooth(pd.Series(plus_dm, index=df.index), period) / atr
    minus_di = 100.0 * _wilder_smooth(pd.Series(minus_dm, index=df.index), period) / atr

    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = 100.0 * di_diff / di_sum.replace(0, np.nan)
    adx = _wilder_smooth(dx, period)

    return pd.DataFrame(
        {"ADX": adx, "Plus_DI": plus_di, "Minus_DI": minus_di},
        index=df.index,
    )


def _strength_label(
    adx: float,
    *,
    adx_sideways: float,
    adx_trend: float,
    adx_strong: float,
) -> str:
    if adx < adx_sideways:
        return "약함"
    if adx < adx_trend:
        return "약함"
    if adx < adx_strong:
        return "보통"
    if adx < 50:
        return "강함"
    return "매우강함"


def classify_adx_regime(
    adx: Optional[float],
    plus_di: Optional[float],
    minus_di: Optional[float],
    *,
    adx_sideways: float = 20.0,
    adx_trend: float = 25.0,
    adx_strong: float = 40.0,
    di_epsilon: float = 2.0,
) -> Tuple[str, str, str]:
    """
    Returns:
        (direction_ko, strength_ko, summary)  예: ("상승", "강함", "상승·강함")
    """
    if adx is None or plus_di is None or minus_di is None:
        return "N/A", "N/A", "데이터 부족"
    if np.isnan(adx) or np.isnan(plus_di) or np.isnan(minus_di):
        return "N/A", "N/A", "데이터 부족"

    strength = _strength_label(
        float(adx),
        adx_sideways=adx_sideways,
        adx_trend=adx_trend,
        adx_strong=adx_strong,
    )

    if adx < adx_sideways:
        return "횡보", strength, f"횡보·{strength}"

    di_gap = float(plus_di) - float(minus_di)
    if abs(di_gap) < di_epsilon:
        return "횡보", strength, f"횡보·{strength}"

    if di_gap > 0:
        direction = "상승"
    else:
        direction = "하락"

    return direction, strength, f"{direction}·{strength}"


def detect_adx_buy_signals(
    series: pd.DataFrame,
    *,
    adx_min: float = 25.0,
) -> List[Dict[str, Any]]:
    """
    +DI가 -DI를 상향 돌파(골든크로스)하고, 동일 봉 ADX >= adx_min 인 시점.

    Returns:
        [{date, adx, plus_di, minus_di}, ...]
    """
    if series is None or series.empty or len(series) < 2:
        return []

    plus = series["Plus_DI"].astype(float)
    minus = series["Minus_DI"].astype(float)
    adx = series["ADX"].astype(float)

    prev_plus_le_minus = plus.shift(1) <= minus.shift(1)
    curr_plus_gt_minus = plus > minus
    cross_up = prev_plus_le_minus & curr_plus_gt_minus
    strong_enough = adx >= float(adx_min)
    mask = cross_up & strong_enough

    signals: List[Dict[str, Any]] = []
    for dt in series.index[mask.fillna(False)]:
        if not (pd.notna(plus.loc[dt]) and pd.notna(minus.loc[dt]) and pd.notna(adx.loc[dt])):
            continue
        signals.append(
            {
                "date": pd.Timestamp(dt),
                "adx": round(float(adx.loc[dt]), 2),
                "plus_di": round(float(plus.loc[dt]), 2),
                "minus_di": round(float(minus.loc[dt]), 2),
            }
        )
    return signals


def plot_adx_buy_on_main(
    ax,
    signals: List[Dict[str, Any]],
    ohlcv: pd.DataFrame,
    *,
    label: str = "ADX buy",
) -> None:
    """메인 차트에 ADX buy 마커 표시."""
    if not signals or ohlcv is None or ohlcv.empty:
        return

    df = ohlcv.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    low_s = df["Low"].astype(float)

    for sig in signals:
        dt = pd.Timestamp(sig["date"])
        if dt not in low_s.index:
            nearest = low_s.index[low_s.index.get_indexer([dt], method="nearest")[0]]
            dt = nearest
        y = float(low_s.loc[dt]) * 0.985
        ax.scatter(
            dt,
            y,
            marker="^",
            s=48,
            color="#f1c40f",
            edgecolors="#1a1a1a",
            linewidth=0.7,
            zorder=48,
            clip_on=False,
        )
        ax.annotate(
            label,
            xy=(dt, y),
            xytext=(0, -10),
            textcoords="offset points",
            fontsize=6,
            fontweight="bold",
            color="#1a1a1a",
            ha="center",
            va="top",
            zorder=49,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="#fff9e6",
                edgecolor="#f39c12",
                alpha=0.9,
                linewidth=0.5,
            ),
        )


def analyze_adx_dmi(
    ohlcv: pd.DataFrame,
    *,
    period: int = 14,
    adx_sideways: float = 20.0,
    adx_trend: float = 25.0,
    adx_strong: float = 40.0,
    di_epsilon: float = 2.0,
) -> Dict[str, Any]:
    """최신 봉 기준 ADX/DMI 분석 결과."""
    series = calculate_adx_dmi(ohlcv, period=period)
    if series.empty:
        return {"direction": "N/A", "strength": "N/A", "summary": "데이터 부족"}

    last = series.iloc[-1]
    adx = float(last["ADX"]) if pd.notna(last["ADX"]) else None
    plus_di = float(last["Plus_DI"]) if pd.notna(last["Plus_DI"]) else None
    minus_di = float(last["Minus_DI"]) if pd.notna(last["Minus_DI"]) else None

    direction, strength, summary = classify_adx_regime(
        adx,
        plus_di,
        minus_di,
        adx_sideways=adx_sideways,
        adx_trend=adx_trend,
        adx_strong=adx_strong,
        di_epsilon=di_epsilon,
    )

    buy_signals = detect_adx_buy_signals(series, adx_min=adx_trend)

    return {
        "period": period,
        "adx": round(adx, 2) if adx is not None else None,
        "plus_di": round(plus_di, 2) if plus_di is not None else None,
        "minus_di": round(minus_di, 2) if minus_di is not None else None,
        "direction": direction,
        "strength": strength,
        "summary": summary,
        "series": series,
        "adx_sideways": adx_sideways,
        "adx_trend": adx_trend,
        "adx_strong": adx_strong,
        "buy_signals": buy_signals,
        "buy_signal_count": len(buy_signals),
    }
