"""
HMA + Mantra Band Technical Analysis Package
"""

from .core import (
    calculate_hma,
    calculate_mantra_bands,
    calculate_ma,
    calculate_macd,
    calculate_rsi
)

from .signals import (
    get_hma_signals,
    get_mantra_signals,
    get_hma_mantra_md_signals
)

from .patterns import (
    detect_hammer,
    detect_bullish_engulfing,
    detect_bullish_harami,
    detect_piercing_line,
    detect_morning_star
)

from .visualization import (
    plot_hma_mantra,
    plot_comparison,
    plot_signals_with_strength,
    plot_hma_mantra_md_signals
)

from .utils import (
    get_gics_sector,
    get_gics_subsector,
    get_sector_outlook,
    get_trading_strategy,
    get_available_font,
    to_float,
    to_bool,
    to_series
)

__all__ = [
    'calculate_hma',
    'calculate_mantra_bands',
    'calculate_ma',
    'calculate_macd',
    'calculate_rsi',
    'get_hma_signals',
    'get_mantra_signals',
    'get_hma_mantra_md_signals',
    'detect_hammer',
    'detect_bullish_engulfing',
    'detect_bullish_harami',
    'detect_piercing_line',
    'detect_morning_star',
    'plot_hma_mantra',
    'plot_comparison',
    'plot_signals_with_strength',
    'plot_hma_mantra_md_signals',
    'get_gics_sector',
    'get_gics_subsector',
    'get_sector_outlook',
    'get_trading_strategy',
    'get_available_font',
    'to_float',
    'to_bool',
    'to_series'
] 