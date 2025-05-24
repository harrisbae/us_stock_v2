"""
Candlestick pattern detection module
"""

import pandas as pd
import numpy as np
from .utils import to_series

def detect_hammer(df, ticker=None):
    """해머 패턴 감지"""
    open_ = df['Open'] if 'Open' in df else df[('Open', ticker)]
    close_ = df['Close'] if 'Close' in df else df[('Close', ticker)]
    low_ = df['Low'] if 'Low' in df else df[('Low', ticker)]
    high_ = df['High'] if 'High' in df else df[('High', ticker)]
    
    open_ = to_series(open_, df.index)
    close_ = to_series(close_, df.index)
    low_ = to_series(low_, df.index)
    high_ = to_series(high_, df.index)
    
    if len(df) < 2:
        return pd.Series([False]*len(df), index=df.index)
    
    body = abs(close_ - open_)
    min_oc = pd.DataFrame({'open': open_, 'close': close_}).min(axis=1)
    max_oc = pd.DataFrame({'open': open_, 'close': close_}).max(axis=1)
    lower_shadow = min_oc - low_
    upper_shadow = high_ - max_oc
    
    return (lower_shadow > 2 * body) & (upper_shadow < body)

def detect_bullish_engulfing(df, ticker=None):
    """매수 잠식형 패턴 감지"""
    open_ = df['Open'] if 'Open' in df else df[('Open', ticker)]
    close_ = df['Close'] if 'Close' in df else df[('Close', ticker)]
    prev_open = open_.shift(1)
    prev_close = close_.shift(1)
    
    return (prev_close < prev_open) & (close_ > open_) & (open_ < prev_close) & (close_ > prev_open)

def detect_bullish_harami(df, ticker=None):
    """매수 하라미 패턴 감지"""
    open_ = df['Open'] if 'Open' in df else df[('Open', ticker)]
    close_ = df['Close'] if 'Close' in df else df[('Close', ticker)]
    prev_open = open_.shift(1)
    prev_close = close_.shift(1)
    
    return (prev_close > prev_open) & (open_ > close_) & (open_ > prev_close) & (close_ < prev_open)

def detect_piercing_line(df, ticker=None):
    """관통형 패턴 감지"""
    open_ = df['Open'] if 'Open' in df else df[('Open', ticker)]
    close_ = df['Close'] if 'Close' in df else df[('Close', ticker)]
    prev_open = open_.shift(1)
    prev_close = close_.shift(1)
    
    return (prev_close < prev_open) & (open_ < prev_close) & (close_ > (prev_open + prev_close) / 2) & (close_ < prev_open)

def detect_morning_star(df, ticker=None):
    """모닝스타 패턴 감지"""
    open_ = df['Open'] if 'Open' in df else df[('Open', ticker)]
    close_ = df['Close'] if 'Close' in df else df[('Close', ticker)]
    prev_open = open_.shift(2)
    prev_close = close_.shift(2)
    star_open = open_.shift(1)
    star_close = close_.shift(1)
    
    return (
        (prev_close < prev_open) &
        (star_close < prev_close) & (star_open < prev_close) &
        (close_ > star_open) & (close_ > prev_open)
    ) 