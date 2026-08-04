"""
버핏 지수 (Market Cap / GDP) 계산 모듈.

- GDP: FRED `GDP` (명목 GDP, 10억 달러) → 조 달러 환산
- 시가총액: Yahoo `^W5000` 지수 × 검증된 캘리브레이션 계수
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# buffettindicator.org / Wilshire Full Cap 기준점 (주기적 교차검증 후 갱신)
WILSHIRE_TICKER = "^W5000"
WILSHIRE_CALIB_DATE = "2026-05-15"
WILSHIRE_CALIB_INDEX = 74518.0
WILSHIRE_CALIB_CAP_TRILLIONS = 80.7

# FRED 실패 시 최근 관측치 (조 달러)
GDP_FALLBACK_TRILLIONS = 31.856
GDP_FALLBACK_DATE = "2026-Q1"


def _yf_last_close(download_df: pd.DataFrame) -> float:
    """yfinance download 결과에서 최종 종가를 float으로 추출 (MultiIndex 대응)."""
    if download_df is None or download_df.empty:
        raise ValueError("empty price dataframe")
    close = download_df["Close"]
    if isinstance(close, pd.DataFrame):
        val = close.iloc[-1].iloc[0]
    else:
        val = close.iloc[-1]
    return float(np.asarray(val).ravel()[0])


def _quarter_label(ts: pd.Timestamp) -> str:
    q = (int(ts.month) - 1) // 3 + 1
    return f"{ts.year}-Q{q}"


def fetch_us_gdp_trillions() -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """
  Returns:
      (gdp_trillions, gdp_date_label, source_tag)
    """
    # 1) pandas_datareader / FRED
    try:
        import pandas_datareader.data as web

        gdp = web.DataReader("GDP", "fred", datetime(2020, 1, 1), datetime.now())
        if gdp is not None and not gdp.empty:
            last_billions = float(gdp["GDP"].iloc[-1])
            last_ts = gdp.index[-1]
            return last_billions / 1000.0, _quarter_label(last_ts), "FRED:GDP"
    except Exception as e:
        print(f"⚠️ FRED GDP(pandas_datareader) 실패: {e}")

    # 2) fredapi (환경 변수 FRED_API_KEY)
    try:
        import os
        from fredapi import Fred

        api_key = os.environ.get("FRED_API_KEY")
        if api_key:
            fred = Fred(api_key=api_key)
            series = fred.get_series("GDP", observation_start="2020-01-01")
            if series is not None and len(series) > 0:
                last_billions = float(series.iloc[-1])
                last_ts = series.index[-1]
                return last_billions / 1000.0, _quarter_label(last_ts), "FRED:GDP(fredapi)"
    except Exception as e:
        print(f"⚠️ FRED GDP(fredapi) 실패: {e}")

    return None, None, None


def fetch_wilshire_market_cap_trillions() -> Tuple[Optional[float], Optional[str], Optional[float], Optional[str]]:
    """
  Returns:
      (market_cap_trillions, trade_date, index_level, source_tag)
    """
    try:
        import yfinance as yf

        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=14)
        wilshire = yf.download(
            WILSHIRE_TICKER,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
        )
        index_level = _yf_last_close(wilshire)
        scale = WILSHIRE_CALIB_CAP_TRILLIONS / WILSHIRE_CALIB_INDEX
        market_cap_trillions = index_level * scale
        trade_date = wilshire.index[-1].strftime("%Y-%m-%d")
        source = f"Yahoo:{WILSHIRE_TICKER}×calib({WILSHIRE_CALIB_DATE})"
        return market_cap_trillions, trade_date, index_level, source
    except Exception as e:
        print(f"⚠️ Wilshire({WILSHIRE_TICKER}) 로드 실패: {e}")
        return None, None, None, None


def calculate_buffett_indicator() -> Optional[Dict[str, Any]]:
    """버핏 지수를 계산합니다."""
    gdp_t, gdp_date, gdp_src = fetch_us_gdp_trillions()
    cap_t, wilshire_date, index_level, cap_src = fetch_wilshire_market_cap_trillions()

    used_fallback = False
    sources = []

    if gdp_t is None:
        gdp_t = GDP_FALLBACK_TRILLIONS
        gdp_date = GDP_FALLBACK_DATE
        used_fallback = True
        sources.append("GDP:fallback")
    else:
        sources.append(gdp_src or "GDP")

    if cap_t is None:
        cap_t = WILSHIRE_CALIB_CAP_TRILLIONS
        wilshire_date = WILSHIRE_CALIB_DATE
        index_level = WILSHIRE_CALIB_INDEX
        used_fallback = True
        sources.append("Wilshire:fallback")
    else:
        sources.append(cap_src or "Wilshire")

    buffett_value = (cap_t / gdp_t) * 100.0

    result: Dict[str, Any] = {
        "value": round(buffett_value, 1),
        "wilshire_market_cap": round(cap_t, 1),
        "wilshire_date": wilshire_date,
        "wilshire_index": round(index_level, 1) if index_level is not None else None,
        "us_gdp": round(gdp_t, 2),
        "gdp_date": gdp_date,
        "data_sources": ", ".join(sources),
        "used_fallback": used_fallback,
    }

    print("🔄 버핏 지수 계산 완료:")
    if index_level is not None:
        print(f"  - Wilshire 지수({WILSHIRE_TICKER}): {index_level:.0f}")
    print(f"  - Wilshire 5000 시가총액: {result['wilshire_market_cap']}조 달러 ({result['wilshire_date']})")
    print(f"  - US GDP: {result['us_gdp']}조 달러 ({result['gdp_date']})")
    print(f"  - 버핏 지수: {result['value']}%")
    print(f"  - 데이터: {result['data_sources']}")
    if used_fallback:
        print("  ⚠️ 일부 데이터는 fallback 값을 사용했습니다.")

    return result


def get_buffett_sentiment(buffett_value: Optional[float]) -> Tuple[str, str, str]:
    """버핏 지수 해석 및 투자 가이드를 제공합니다."""
    if buffett_value is None:
        return "N/A", "gray", "데이터 부족"

    if buffett_value <= 50:
        return "극도 과매도", "darkgreen", "강력한 매수 기회 - 주식 비중 80%+ 고려"
    if buffett_value <= 75:
        return "과매도", "green", "매수 기회 - 주식 비중 70-80% 고려"
    if buffett_value <= 90:
        return "균형", "orange", "균형적 배분 - 주식 비중 50-60% 유지"
    if buffett_value <= 115:
        return "과열", "red", "주의 필요 - 주식 비중 30-40% 고려"
    return "극도 과열", "darkred", "강력한 매도 신호 - 주식 비중 20% 이하 고려"


def validate_buffett_result(
    data: Dict[str, Any],
    *,
    external_ratio_pct: Optional[float] = None,
    external_tolerance_pct: float = 15.0,
) -> Dict[str, bool]:
    """계산 결과 내부·외부 일관성 검증."""
    recalc = (data["wilshire_market_cap"] / data["us_gdp"]) * 100.0
    internal_ok = abs(recalc - data["value"]) < 0.15

    # 합리적 범위 (역사적 관측 ~50% ~ 280%)
    range_ok = 40.0 <= data["value"] <= 300.0

    # fallback이 아니고 최근 거래일이면 데이터 신선도 OK
    fresh_ok = not data.get("used_fallback", True)
    if data.get("wilshire_date"):
        try:
            w_date = datetime.strptime(data["wilshire_date"], "%Y-%m-%d").date()
            fresh_ok = fresh_ok and (datetime.now().date() - w_date).days <= 10
        except ValueError:
            fresh_ok = False

    external_ok = True
    if external_ratio_pct is not None:
        external_ok = abs(data["value"] - external_ratio_pct) <= external_tolerance_pct

    return {
        "internal_formula": internal_ok,
        "plausible_range": range_ok,
        "data_fresh": fresh_ok,
        "external_reference": external_ok,
    }
