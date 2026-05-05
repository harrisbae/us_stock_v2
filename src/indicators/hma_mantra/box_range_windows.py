"""
시간 창별 고가·저가 박스의 기하 정보만 계산한다.

- 차트 패턴 `range_box` 후보: 이 모듈만 사용 (Volume Profile·표시 로직 없음).
- 메인 파란 박스: `volume_profile_overlay_chart.calculate_box_ranges` 가
  여기서 만든 창에 VP·현재가·로그를 덧붙인다.

두 경로는 데이터 소스가 같아도 호출 API 를 분리한다.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def compute_box_range_windows(
    data: pd.DataFrame,
    box_period: int = 20,
    num_boxes: int = 2,
    overlap_days: int = 5,
    avoid_time_overlap: bool = True,
) -> List[Dict[str, Any]]:
    """
    OHLCV 마지막 구간부터 거슬러 여러 개의 직사각형 시간 창을 만든다.

    반환 각 원소: start_date, end_date, high, low, range, center, name, period, index
    (메인 표시용 calculate_box_ranges 와 동일한 키 중 VP/ current_price 제외)

    Args:
        data: OHLCV (High, Low 필수)
        box_period: 창 길이(봉 개수)
        num_boxes: 창 개수
        overlap_days: avoid_time_overlap False 일 때만 사용
        avoid_time_overlap: True 면 창들이 시간축에서 겹치지 않음
    """
    box_ranges: List[Dict[str, Any]] = []
    data_len = len(data)
    if data_len < 2 or "High" not in data.columns or "Low" not in data.columns:
        return box_ranges

    for i in range(num_boxes):
        if i == 0:
            start_idx = data_len - box_period
            end_idx = data_len
            box_name = "현재 박스권"
        else:
            if avoid_time_overlap:
                start_idx = data_len - box_period - (i * box_period)
                end_idx = start_idx + box_period
            else:
                start_idx = data_len - box_period - (i * (box_period - overlap_days))
                end_idx = start_idx + box_period
            box_name = f"{i+1}번째 박스권"

        if start_idx < 0 or end_idx > data_len or start_idx >= end_idx:
            continue

        box_data = data.iloc[start_idx:end_idx]
        box_high = float(box_data["High"].max())
        box_low = float(box_data["Low"].min())
        box_range = box_high - box_low

        if box_range > 0:
            box_ranges.append(
                {
                    "start_date": data.index[start_idx],
                    "end_date": data.index[end_idx - 1],
                    "high": box_high,
                    "low": box_low,
                    "range": box_range,
                    "center": (box_high + box_low) / 2,
                    "name": box_name,
                    "period": box_period,
                    "index": i,
                }
            )

    return box_ranges
