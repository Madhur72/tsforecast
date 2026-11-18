"""Simple models. They are the benchmark every other model has to beat, so they always run."""
import numpy as np
from ..base import SeriesForecaster


class Naive(SeriesForecaster):
    """Repeat the last value."""
    name = "Naive"
    def forecast(self, h, horizon, cfg): return np.full(horizon, h[-1])


class Median4(SeriesForecaster):
    """Repeat the median of the last 4 periods. Robust to a single spike."""
    name = "Median4"
    def forecast(self, h, horizon, cfg): return np.full(horizon, np.median(h[-4:]))


class Mean8(SeriesForecaster):
    """Repeat the mean of the last 8 periods."""
    name = "Mean8"
    def forecast(self, h, horizon, cfg): return np.full(horizon, h[-8:].mean())


class Hurdle(SeriesForecaster):
    """Two-part model for bursty series: (share of the last 8 periods that were active) x (typical size of an active period)."""
    name = "Hurdle"
    def forecast(self, h, horizon, cfg):
        nz = h[-12:][h[-12:] > 0]
        return np.full(horizon, float((h[-8:] > 0).mean()) * (float(np.median(nz)) if len(nz) else 0.0))


class SeasonalNaive(SeriesForecaster):
    """Repeat the last full season (a week for daily data). With no season it is the same as Naive."""
    name = "SeasonalNaive"
    def forecast(self, h, horizon, cfg):
        m = cfg.season_length
        return np.resize(h[-m:], horizon) if m > 1 and len(h) >= m else np.full(horizon, h[-1])
