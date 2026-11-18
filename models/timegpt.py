"""TimeGPT through the Nixtla API. Needs NIXTLA_API_KEY in the environment (.env). Without it the model is skipped.
Data leaves your machine, so only numbers and dates are sent: series ids are replaced by anonymous labels (s0, s1, ...) and never include real names."""
import os
import numpy as np, pandas as pd
from ..base import BaseForecaster

FREQ = {"D": ("D", "D"), "W": ("7D", "W-MON"), "M": ("MS", "MS")}      # (date step for the fake dates, Nixtla freq)


class TimeGPT(BaseForecaster):
    name = "TimeGPT"
    heavy = True                       # an external service: off when foundation_models=False

    def load(self):
        if not os.getenv("NIXTLA_API_KEY"): raise RuntimeError("NIXTLA_API_KEY is not set")
        from nixtla import NixtlaClient
        self._c = NixtlaClient(api_key=os.environ["NIXTLA_API_KEY"])

    def predict(self, histories, horizon, ctx):
        keys = [k for k, v in histories.items() if len(v) >= 4]
        if not keys: return {}
        step, freq = FREQ[ctx.cfg.freq]
        df = pd.concat([pd.DataFrame({"unique_id": f"s{j}", "ds": pd.date_range("2000-01-03", periods=len(histories[k]), freq=step), "y": np.asarray(histories[k], float)}) for j, k in enumerate(keys)])
        fc = self._c.forecast(df=df, h=horizon, freq=freq, model="timegpt-1")
        col = [c for c in fc.columns if c not in ("unique_id", "ds")][0]
        return {keys[int(u[1:])]: np.clip(g.sort_values("ds")[col].values, 0, None) for u, g in fc.groupby("unique_id")}
