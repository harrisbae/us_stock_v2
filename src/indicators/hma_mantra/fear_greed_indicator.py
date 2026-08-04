"""
CNN Fear & Greed Index (공포·탐욕 지수) 조회 모듈.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CNN_FEAR_GREED_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
CNN_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.cnn.com/markets/fear-and-greed",
}

# CNN 장애 시 참고용 (암호화폐 F&G — 미국 주식과 다를 수 있음)
CRYPTO_FNG_URL = "https://api.alternative.me/fng/?limit=1"


def _http_get_json(url: str, headers: Dict[str, str]) -> Dict[str, Any]:
    try:
        import requests

        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️ requests GET 실패 ({url}): {e}")

    req = Request(url, headers=headers)
    with urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _parse_timestamp(ts: Optional[str]) -> Optional[str]:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone().strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return str(ts)[:16]


def _rating_korean(rating: Optional[str]) -> str:
    mapping = {
        "extreme fear": "극단적 공포",
        "fear": "공포",
        "neutral": "중립",
        "greed": "탐욕",
        "extreme greed": "극단적 탐욕",
    }
    if not rating:
        return "N/A"
    return mapping.get(str(rating).strip().lower(), str(rating))


def fetch_cnn_fear_greed() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """CNN Fear & Greed Index 조회."""
    try:
        payload = _http_get_json(CNN_FEAR_GREED_URL, CNN_HEADERS)
        fg = payload.get("fear_and_greed") or {}
        score = fg.get("score")
        if score is None:
            raise ValueError("CNN 응답에 score 없음")
        score = round(float(score), 1)
        rating = str(fg.get("rating", "")).strip().lower()
        result = {
            "score": score,
            "rating": rating,
            "rating_ko": _rating_korean(rating),
            "timestamp": _parse_timestamp(fg.get("timestamp")),
            "previous_close": _safe_round(fg.get("previous_close")),
            "previous_1_week": _safe_round(fg.get("previous_1_week")),
            "previous_1_month": _safe_round(fg.get("previous_1_month")),
            "source": "CNN",
            "used_fallback": False,
        }
        return result, "CNN"
    except (HTTPError, URLError, ValueError, KeyError, TypeError) as e:
        print(f"⚠️ CNN 공포탐욕 지수 로드 실패: {e}")
        return None, None


def _safe_round(val: Any, digits: int = 1) -> Optional[float]:
    if val is None:
        return None
    try:
        return round(float(val), digits)
    except (TypeError, ValueError):
        return None


def fetch_crypto_fear_greed_fallback() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """CNN 실패 시 alternative.me (암호화폐) F&G."""
    try:
        payload = _http_get_json(CRYPTO_FNG_URL, {"User-Agent": CNN_HEADERS["User-Agent"]})
        row = (payload.get("data") or [{}])[0]
        score = round(float(row["value"]), 1)
        rating = str(row.get("value_classification", "")).strip().lower()
        ts = row.get("timestamp")
        ts_label = None
        if ts:
            try:
                ts_label = datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone().strftime(
                    "%Y-%m-%d %H:%M"
                )
            except (ValueError, TypeError, OSError):
                ts_label = None
        result = {
            "score": score,
            "rating": rating.replace(" ", "_"),
            "rating_ko": _rating_korean(rating),
            "timestamp": ts_label,
            "previous_close": None,
            "previous_1_week": None,
            "previous_1_month": None,
            "source": "alternative.me(crypto)",
            "used_fallback": True,
        }
        return result, "alternative.me"
    except Exception as e:
        print(f"⚠️ 암호화폐 F&G fallback 실패: {e}")
        return None, None


def calculate_fear_greed_indicator() -> Optional[Dict[str, Any]]:
    """공포·탐욕 지수를 조회합니다."""
    data, _ = fetch_cnn_fear_greed()
    if data is None:
        data, _ = fetch_crypto_fear_greed_fallback()

    if not data:
        return None

    sentiment, color, guide = get_fear_greed_sentiment(data["score"], data.get("rating"))

    data["sentiment"] = sentiment
    data["color"] = color
    data["guide"] = guide

    print("🔄 공포·탐욕 지수 조회 완료:")
    print(f"  - 지수: {data['score']} ({data['rating_ko']})")
    print(f"  - 소스: {data['source']}")
    if data.get("timestamp"):
        print(f"  - 기준: {data['timestamp']}")
    if data.get("previous_close") is not None:
        print(f"  - 전일: {data['previous_close']}")
    if data.get("used_fallback"):
        print("  ⚠️ CNN 대신 fallback 소스 사용 (암호화폐 지수 — 참고용)")

    return data


def get_fear_greed_sentiment(
    score: Optional[float],
    rating: Optional[str] = None,
) -> Tuple[str, str, str]:
    """점수·등급 기반 해석 (역발상: 극공포=매수, 극탐욕=주의)."""
    if score is None:
        return "N/A", "gray", "데이터 부족"

    s = float(score)
    if s <= 25:
        return "극단적 공포", "darkgreen", "역발상 매수 구간 — 분할 매수·장기 매수 검토"
    if s <= 45:
        return "공포", "green", "매수 우위 — 점진적 매수·비중 확대 검토"
    if s <= 55:
        return "중립", "orange", "균형 — 추세·종목별 시그널 우선"
    if s <= 75:
        return "탐욕", "red", "과열 주의 — 차익실현·비중 축소 검토"
    return "극단적 탐욕", "darkred", "과열 극대 — 신규 매수 자제·리스크 관리 강화"
