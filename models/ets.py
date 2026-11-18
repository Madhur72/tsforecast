"""Statistical models from the statsforecast library (fast, one fit per series)."""
import warnings
import numpy as np, pandas as pd
from ..base import BaseForecaster

FAKE_FREQ = {"D": "D", "W": "W-MON", "M": "MS"}        # dates only label the points; the models see a plain regular series


class _StatsForecast(BaseForecaster):
    def make(self, cfg): raise NotImplementedError

    def predict(self, histories, horizon, ctx):
        warnings.filterwarnings("ignore")
        from statsforecast import StatsForecast
        cfg, keys = ctx.cfg, list(histories); freq = FAKE_FREQ[cfg.freq]
        if not keys: return {}
        df = pd.concat([pd.DataFrame({"unique_id": str(j), "ds": pd.date_range("2000-01-03", periods=len(histories[k]), freq=freq), "y": histories[k]}) for j, k in enumerate(keys)])

        def fit(dd):
            o = StatsForecast(models=[self.make(cfg)], freq=freq, n_jobs=1).forecast(df=dd, h=horizon).reset_index()
            return {keys[int(j)]: np.clip(g.sort_values("ds")[self.name].values, 0, None) for j, g in o.groupby("unique_id")}

        try: return fit(df)
        except Exception:                                  # one bad series must not kill the batch: retry series by series
            out = {}
            for j, g in df.groupby("unique_id"):
                try: out.update(fit(g))
                except Exception: pass
            return out


class ETS(_StatsForecast):
    name = "ETS"
    def make(self, cfg):
        from statsforecast.models import AutoETS
        return AutoETS(season_length=cfg.season_length, alias=self.name)


class ARIMA(_StatsForecast):
    name = "ARIMA"
    def make(self, cfg):
        from statsforecast.models import AutoARIMA
        return AutoARIMA(season_length=cfg.season_length, alias=self.name)


class Theta(_StatsForecast):
    name = "Theta"
    def make(self, cfg):
        from statsforecast.models import AutoTheta
        return AutoTheta(season_length=cfg.season_length, alias=self.name)


class TSB(_StatsForecast):
    """Built for mostly-zero (intermittent) series."""
    name = "TSB"
    def make(self, cfg):
        from statsforecast.models import TSB as _TSB
        return _TSB(alpha_d=0.3, alpha_p=0.3, alias=self.name)


class Croston(_StatsForecast):
    """Also for intermittent series."""
    name = "Croston"
    def make(self, cfg):
        from statsforecast.models import CrostonOptimized
        return CrostonOptimized(alias=self.name)
