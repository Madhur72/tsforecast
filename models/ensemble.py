"""Ensemble: the median of several models' forecasts (ETS, ARIMA, Theta, LightGBM when it is running, and the mean of the last 4 periods).
Medians of different model families are usually steadier than any one of them."""
import numpy as np
from ..base import BaseForecaster
from .ets import ETS, ARIMA, Theta
from .lightgbm import LightGBM


class Ensemble(BaseForecaster):
    name = "Ensemble"

    def predict(self, histories, horizon, ctx):
        members = [m().predict(histories, horizon, ctx) for m in (ETS, ARIMA, Theta)]
        if ctx.features is not None:                                   # LightGBM only if some selected model asked for the feature table
            try: members.append(LightGBM().predict(histories, horizon, ctx))
            except Exception: pass
        out = {}
        for k, h in histories.items():
            mem = [m[k] for m in members if k in m] + [np.full(horizon, np.asarray(h, float)[-4:].mean())]
            out[k] = np.median(mem, axis=0)
        return out
