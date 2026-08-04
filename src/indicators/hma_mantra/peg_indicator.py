"""
PEG (Price/Earnings to Growth) — 1개년·3개년 EPS 성장률 기준.

PEG = PER / EPS 성장률(%)
- 1개년: 최근 2개 회계연도 EPS YoY
- 3개년: 최근 vs 3년 전 EPS CAGR
- PER: Forward → Trailing (yfinance)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

_EPS_ROWS = ("Diluted EPS", "Basic EPS")


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        if f != f:
            return None
        return f
    except (TypeError, ValueError):
        return None


def _growth_to_percent(growth_decimal: float) -> float:
    g = float(growth_decimal)
    if abs(g) <= 1.5:
        return g * 100.0
    return g


def get_peg_sentiment(peg: Optional[float]) -> Tuple[str, str, str]:
    if peg is None:
        return "N/A", "gray", "PEG 산출 불가"
    if peg < 0:
        return "적자/역성장", "gray", "EPS 성장률 마이너스 — PEG 해석 제한"

    if peg < 1.0:
        return "저평가", "darkgreen", "성장 대비 PER 낮음"
    if peg < 1.5:
        return "적정", "green", "성장·가격 균형"
    if peg < 2.0:
        return "다소 고평가", "orange", "성장 대비 가격 부담"
    return "고평가", "darkred", "성장 대비 가격 과다"


def _bucket(peg: Optional[float]) -> str:
    """PEG 구간 키."""
    if peg is None:
        return "na"
    if peg < 0:
        return "neg"
    if peg < 1.0:
        return "cheap"
    if peg < 1.5:
        return "fair"
    if peg < 2.0:
        return "rich"
    return "expensive"


def build_peg_investment_view(peg_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    PEG 1년·3년 종합 → 재무 관점 투자 코멘트.

    Returns:
      action: 매수검토|보유|신중|관망|해당없음
      summary: 하단바용 한 줄
      lines: 패널용 여러 줄
      primary_peg / primary_label / primary_sentiment
    """
    out: Dict[str, Any] = {
        "action": "해당없음",
        "summary": "",
        "lines": [],
        "primary_peg": None,
        "primary_label": None,
        "primary_sentiment": "N/A",
    }
    if not peg_data:
        out["summary"] = "PEG 데이터 없음"
        out["lines"] = ["PEG: 데이터 없음 (ETF·적자·미제공 가능)"]
        return out

    peg_1y = peg_data.get("peg_1y")
    peg_3y = peg_data.get("peg_3y")
    g1 = peg_data.get("growth_1y_pct")
    g3 = peg_data.get("growth_3y_cagr_pct")
    pe = peg_data.get("pe")
    pe_label = peg_data.get("pe_label") or "PER"
    sent_1y = peg_data.get("peg_1y_sentiment") or "N/A"
    sent_3y = peg_data.get("peg_3y_sentiment") or "N/A"

    if peg_1y is None and peg_3y is None:
        yf_peg = peg_data.get("yf_peg_ratio")
        yf_ok = bool(peg_data.get("yf_peg_usable", True))
        yf_note = peg_data.get("yf_peg_note") or ""
        lines = ["[재무·PEG]", "자체 PEG(EPS YoY/CAGR) 산출불가 — 성장률≤0 또는 EPS 부족"]
        if pe is not None:
            lines.append(f"  {pe_label}={pe}")
        # Yahoo PEG 폴백 (예상성장 기준 · Forward 적자면 사용 안 함)
        if yf_ok and yf_peg is not None and yf_peg > 0:
            sent_yf, _, _ = get_peg_sentiment(yf_peg)
            b = _bucket(yf_peg)
            action = {
                "cheap": "매수검토",
                "fair": "보유",
                "rich": "신중",
                "expensive": "신중",
            }.get(b, "관망")
            core = {
                "cheap": "성장 대비 밸류 매력 · 기술적 지지·돌파 확인 후 분할매수",
                "fair": "밸류·성장 균형 · 추세 유효 시 보유",
                "rich": "성장 대비 가격 부담 · 신규 비중 확대 자제",
                "expensive": "고평가 구간 · 신규매수 비권고, 저항에서 분할매도 검토",
            }.get(b, "Yahoo PEG만 참고 · 실적 가시성 확인 후 판단")
            impl = None
            if pe is not None and pe > 0 and yf_peg > 0:
                impl = round(pe / yf_peg, 1)
            lines.append(f"(폴백) Yahoo PEG={yf_peg} ({sent_yf}) · 예상성장 기준")
            if impl is not None:
                lines.append(f"  암시성장≈{impl}% (≈{pe_label}/{yf_peg})")
            lines.append(f"재무액션: {action} · {core}")
            lines.append("주의: Yahoo≠과거EPS PEG · 애널리스트 장기성장 가정")
            out["action"] = action
            out["summary"] = f"Yahoo PEG {yf_peg}({sent_yf}) → {action}"
            out["primary_peg"] = yf_peg
            out["primary_label"] = "Yahoo"
            out["primary_sentiment"] = sent_yf
            out["core_strategy"] = core
            out["lines"] = lines
            return out
        out["action"] = "관망"
        out["summary"] = "PEG 산출불가 · 실적·성장 확인 전 관망"
        if yf_peg is not None and not yf_ok:
            lines.append(f"Yahoo PEG={yf_peg} 있으나 미사용" + (f" ({yf_note})" if yf_note else ""))
        out["lines"] = lines + ["재무액션: 관망 · PER만으로는 성장 대비 매력 판단 어려움"]
        return out

    # 대표 PEG: 1년 우선, 없으면 3년
    if peg_1y is not None:
        primary, primary_lab, primary_sent = peg_1y, "1년", sent_1y
    else:
        primary, primary_lab, primary_sent = peg_3y, "3년", sent_3y
    out["primary_peg"] = primary
    out["primary_label"] = primary_lab
    out["primary_sentiment"] = primary_sent

    b1, b3 = _bucket(peg_1y), _bucket(peg_3y)
    # 종합 버킷: 둘 다 있으면 더 보수적(비싼) 쪽, 하나면 그 쪽
    rank = {"na": -1, "neg": 0, "cheap": 1, "fair": 2, "rich": 3, "expensive": 4}
    if b1 != "na" and b3 != "na":
        overall = b1 if rank[b1] >= rank[b3] else b3
    else:
        overall = b1 if b1 != "na" else b3

    action_map = {
        "cheap": "매수검토",
        "fair": "보유",
        "rich": "신중",
        "expensive": "신중",
        "neg": "관망",
        "na": "관망",
    }
    action = action_map.get(overall, "관망")

    strategy_by_bucket = {
        "cheap": "성장 대비 밸류 매력 · 기술적 지지·돌파 확인 후 분할매수",
        "fair": "밸류·성장 균형 · 추세 유효 시 보유, 급등 시 기계적 익절 구간 설정",
        "rich": "성장 대비 가격 부담 · 신규 비중 확대 자제, 모멘텀 약화 시 축소",
        "expensive": "고평가 구간 · 신규매수 비권고, 보유분은 저항·목표가에서 분할매도",
        "neg": "역성장/적자 · PEG 해석 제한, 실적 턴어라운드 확인 전 관망",
        "na": "PEG 참고 불가 · 수급·섹터·기술 시그널 위주 판단",
    }
    core = strategy_by_bucket.get(overall, strategy_by_bucket["na"])

    # 1년 vs 3년 괴리
    diverge = ""
    if peg_1y is not None and peg_3y is not None and peg_1y > 0 and peg_3y > 0:
        if peg_1y <= 1.2 and peg_3y >= 1.8:
            diverge = "단기 성장 가속(1년 저PEG) · 지속 여부 실적으로 검증"
        elif peg_1y >= 1.8 and peg_3y <= 1.2:
            diverge = "단기 성장 둔화 신호 · 3년 싸보여도 눈높이 하향 경계"
        elif abs(peg_1y - peg_3y) >= 0.8:
            diverge = f"1년·3년 PEG 괴리({peg_1y:.2f} vs {peg_3y:.2f}) · 성장 지속성 점검"

    lines: List[str] = ["[재무·PEG]"]
    pe_bit = f"{pe_label}={pe}" if pe is not None else "PER=N/A"
    if peg_1y is not None:
        g1_bit = f"YoY {g1:+.1f}%" if g1 is not None else "YoY N/A"
        lines.append(f"PEG 1년: {peg_1y} ({sent_1y}) · {g1_bit} · {pe_bit}")
    else:
        lines.append(f"PEG 1년: N/A · {pe_bit}")
    if peg_3y is not None:
        g3_bit = f"CAGR {g3:+.1f}%" if g3 is not None else "CAGR N/A"
        lines.append(f"PEG 3년: {peg_3y} ({sent_3y}) · {g3_bit}")
    else:
        lines.append("PEG 3년: N/A (연간 EPS 4개년 미만 또는 CAGR≤0)")

    yf_peg = peg_data.get("yf_peg_ratio")
    if yf_peg is not None:
        if peg_data.get("yf_peg_usable", True):
            lines.append(f"(참고) Yahoo PEG={yf_peg} · 예상성장기준(액션에 미반영)")
        else:
            note = peg_data.get("yf_peg_note") or "신뢰도 낮음"
            lines.append(f"(참고) Yahoo PEG={yf_peg} 미사용 · {note}")

    lines.append(f"재무액션: {action} · {core}")
    if diverge:
        lines.append(f"주의: {diverge}")

    # 성장률 절대 수준 코멘트
    if g1 is not None and g1 >= 25 and overall in ("fair", "rich"):
        lines.append("참고: 고성장(1년≥25%)이면 PEG 다소 높아도 모멘텀 병행 가능")
    elif g1 is not None and 0 < g1 < 8 and overall == "cheap":
        lines.append("참고: 저성장·저PEG — 밸류트랩 여부(섹터 둔화) 점검")

    summary = f"PEG{primary_lab} {primary}({primary_sent}) → {action}"
    if diverge:
        summary += " · 1y/3y괴리"

    out["action"] = action
    out["summary"] = summary
    out["lines"] = lines
    out["core_strategy"] = core
    out["diverge_note"] = diverge
    return out


def _extract_annual_eps_points(ticker) -> List[Tuple[pd.Timestamp, float]]:
    """회계연도별 EPS (최신순)."""
    for attr in ("financials", "income_stmt"):
        df = getattr(ticker, attr, None)
        if df is None or getattr(df, "empty", True):
            continue
        for label in _EPS_ROWS:
            if label not in df.index:
                continue
            points: List[Tuple[pd.Timestamp, float]] = []
            for dt, raw in df.loc[label].items():
                v = _safe_float(raw)
                if v is not None and v > 0:
                    points.append((pd.Timestamp(dt), v))
            if len(points) >= 2:
                points.sort(key=lambda x: x[0], reverse=True)
                return points
    return []


def _eps_yoy_growth_pct(points: List[Tuple[pd.Timestamp, float]]) -> Optional[float]:
    """최근 1개년 EPS YoY (%)."""
    if len(points) < 2:
        return None
    latest, prev = points[0][1], points[1][1]
    if prev <= 0:
        return None
    return ((latest / prev) - 1.0) * 100.0


def _eps_3y_cagr_pct(points: List[Tuple[pd.Timestamp, float]]) -> Optional[float]:
    """3개년 EPS CAGR (%). points[0]=최신, points[3]=3년 전."""
    if len(points) < 4:
        return None
    latest, old = points[0][1], points[3][1]
    if old <= 0 or latest <= 0:
        return None
    return ((latest / old) ** (1.0 / 3.0) - 1.0) * 100.0


def _peg_from_pe_and_growth(pe: Optional[float], growth_pct: Optional[float]) -> Optional[float]:
    if pe is None or growth_pct is None or growth_pct <= 0:
        return None
    return round(pe / growth_pct, 2)


def calculate_peg_indicator(ticker: str) -> Optional[Dict[str, Any]]:
    if not ticker or not str(ticker).strip():
        return None

    symbol = str(ticker).strip().upper()

    try:
        import yfinance as yf

        yt = yf.Ticker(symbol)
        info = yt.info or {}
    except Exception as e:
        print(f"⚠️ PEG 데이터 로드 실패 ({symbol}): {e}")
        return None

    trailing_pe = _safe_float(info.get("trailingPE"))
    forward_pe = _safe_float(info.get("forwardPE"))
    pe_used = forward_pe if forward_pe is not None else trailing_pe
    pe_label = "Forward PER" if forward_pe is not None else "Trailing PER"

    eps_points = _extract_annual_eps_points(yt)
    growth_1y_pct = _eps_yoy_growth_pct(eps_points)
    growth_3y_pct = _eps_3y_cagr_pct(eps_points)
    growth_1y_source = "EPS YoY(재무제표)"
    growth_3y_source = "EPS 3년 CAGR(재무제표)"

    # 재무제표 EPS 부족 시 yfinance YoY 보조 (1개년만)
    if growth_1y_pct is None:
        eg = _safe_float(info.get("earningsGrowth"))
        if eg is not None:
            growth_1y_pct = _growth_to_percent(eg)
            growth_1y_source = "yfinance:earningsGrowth"

    peg_1y = _peg_from_pe_and_growth(pe_used, growth_1y_pct)
    peg_3y = _peg_from_pe_and_growth(pe_used, growth_3y_pct)

    sent_1y, color_1y, _ = get_peg_sentiment(peg_1y)
    sent_3y, color_3y, _ = get_peg_sentiment(peg_3y)

    # Yahoo: pegRatio ≈ trailingPegRatio (있을 때). 장기 예상성장 기반(과거 YoY와 다름).
    yf_peg = _safe_float(info.get("pegRatio"))
    yf_trailing_peg = _safe_float(info.get("trailingPegRatio"))
    if yf_peg is None and yf_trailing_peg is not None:
        yf_peg = yf_trailing_peg
    eps_forward = _safe_float(info.get("epsForward")) or _safe_float(info.get("forwardEps"))
    yf_peg_usable = True
    yf_peg_note = ""
    if yf_peg is not None and yf_peg <= 0:
        yf_peg_usable = False
        yf_peg_note = "Yahoo PEG≤0"
    elif forward_pe is not None and forward_pe < 0:
        yf_peg_usable = False
        yf_peg_note = "Forward PER<0(적자전망)"
    elif eps_forward is not None and eps_forward < 0:
        yf_peg_usable = False
        yf_peg_note = "Forward EPS<0(적자전망)"

    has_any = peg_1y is not None or peg_3y is not None
    if not has_any and pe_used is None:
        print(f"⚠️ PEG 산출 불가 ({symbol}): PER·EPS 성장률 부족")
        return {
            "ticker": symbol,
            "peg": None,
            "peg_1y": None,
            "peg_3y": None,
            "sentiment": "N/A",
            "guide": "ETF·적자·EPS 미제공 종목은 PEG 미적용",
            "source": "unavailable",
            "yf_peg_ratio": round(yf_peg, 2) if yf_peg is not None else None,
            "yf_peg_usable": False,
            "yf_peg_note": yf_peg_note or "PER 없음",
        }

    result: Dict[str, Any] = {
        "ticker": symbol,
        "pe": round(pe_used, 2) if pe_used is not None else None,
        "pe_label": pe_label,
        "trailing_pe": round(trailing_pe, 2) if trailing_pe is not None else None,
        "forward_pe": round(forward_pe, 2) if forward_pe is not None else None,
        "growth_1y_pct": round(growth_1y_pct, 2) if growth_1y_pct is not None else None,
        "growth_3y_cagr_pct": round(growth_3y_pct, 2) if growth_3y_pct is not None else None,
        "growth_1y_source": growth_1y_source if growth_1y_pct is not None else None,
        "growth_3y_source": growth_3y_source if growth_3y_pct is not None else None,
        "peg_1y": peg_1y,
        "peg_3y": peg_3y,
        "peg_1y_sentiment": sent_1y,
        "peg_3y_sentiment": sent_3y,
        "peg_1y_color": color_1y,
        "peg_3y_color": color_3y,
        "yf_peg_ratio": round(yf_peg, 2) if yf_peg is not None else None,
        "yf_trailing_peg": round(yf_trailing_peg, 2) if yf_trailing_peg is not None else None,
        "yf_peg_usable": yf_peg_usable,
        "yf_peg_note": yf_peg_note,
        "eps_years_used": len(eps_points),
        "latest_eps_year": eps_points[0][0].strftime("%Y") if eps_points else None,
        # 하위 호환
        "peg": peg_1y if peg_1y is not None else peg_3y,
        "growth_pct": round(growth_1y_pct, 2) if growth_1y_pct is not None else None,
        "growth_label": "EPS",
        "sentiment": sent_1y if peg_1y is not None else sent_3y,
        "color": color_1y if peg_1y is not None else color_3y,
        "guide": f"1년 PEG={peg_1y or 'N/A'}, 3년 PEG={peg_3y or 'N/A'}",
        "source": "EPS 1년·3년 CAGR",
    }

    view = build_peg_investment_view(result)
    result["action"] = view.get("action")
    result["strategy_summary"] = view.get("summary")
    result["strategy_lines"] = view.get("lines")
    result["core_strategy"] = view.get("core_strategy")
    result["guide"] = view.get("summary") or result["guide"]

    print(f"🔄 PEG ({symbol}) — {pe_label}={pe_used}")
    if peg_1y is not None and growth_1y_pct is not None:
        print(f"   1개년: 성장 {growth_1y_pct:.1f}% → PEG {peg_1y} ({sent_1y}) [{growth_1y_source}]")
    else:
        print("   1개년: 산출 불가 (성장률≤0 또는 데이터 부족)")
    if peg_3y is not None and growth_3y_pct is not None:
        print(f"   3개년: CAGR {growth_3y_pct:.1f}% → PEG {peg_3y} ({sent_3y}) [{growth_3y_source}]")
    else:
        print("   3개년: 산출 불가 (연간 EPS 4개년 미만 또는 CAGR≤0)")
    if yf_peg is not None:
        print(f"   (참고) yfinance pegRatio={yf_peg}")
    if view.get("summary"):
        print(f"   전략: {view['summary']}")

    return result
