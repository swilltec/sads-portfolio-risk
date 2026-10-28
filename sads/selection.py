"""Rank selection from completed forecast losses."""

import numpy as np

from . import constants as c
from .types import BoolArray, FloatArray, IntArray


def best_rank(losses: FloatArray) -> int:
    """Return the column with minimum mean loss, breaking ties by column order."""
    return int(np.argmin(np.mean(losses, axis=0)))


def regime_rank(losses: FloatArray, matching: BoolArray, min_regime: int) -> int:
    """Select within a regime when sufficiently populated, otherwise use all rows."""
    return best_rank(losses[matching] if matching.sum() >= min_regime else losses)


def adaptive_path(
    losses: FloatArray,
    regimes: BoolArray,
    calibration: int,
    lookback: int = c.SELECTION_WINDOW,
    min_regime: int = c.MIN_REGIME_OBSERVATIONS,
) -> IntArray:
    """Choose each rank from prior losses in the current stress regime.

    losses has shape (dates, candidate ranks). Selection starts at calibration;
    earlier entries are zero placeholders. The current date's loss is excluded.
    """
    return _rolling_rank_path(losses, calibration, lookback, regimes, min_regime)


def trailing_optimal_path(
    losses: FloatArray, calibration: int, lookback: int = c.RANK_LOOKBACK
) -> IntArray:
    """Choose the minimum-loss rank over prior dates, starting after calibration."""
    return _rolling_rank_path(losses, calibration, lookback)


def _rolling_rank_path(
    losses: FloatArray,
    calibration: int,
    lookback: int,
    regimes: BoolArray | None = None,
    min_regime: int = c.MIN_REGIME_OBSERVATIONS,
) -> IntArray:
    """Apply a common trailing-history boundary to both rank-selection rules."""
    selected = np.zeros(len(losses), dtype=int)
    for i in range(calibration, len(losses)):
        start = max(0, i - lookback)
        history = losses[start:i]
        selected[i] = (
            best_rank(history)
            if regimes is None
            else regime_rank(history, regimes[start:i] == regimes[i], min_regime)
        )
    return selected
