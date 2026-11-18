"""One global LightGBM for all series, trained on the feature table. Retrained at every forecast origin on rows whose target was already known then.

It uses WHATEVER feature columns the feature builders produced (ctx.feature_cols): amount-like ones (ctx.amount_cols, in the target's unit) are divided by the series'
recent level so big and small series look alike; the rest are used as they are."""
import numpy as np, pandas as pd
from ..base import BaseForecaster


class LightGBM(BaseForecaster):
    name = "LightGBM"
    requires_features = True
    min_rows = 150

    def _X(self, df, ctx):
        sc = np.maximum(df.level8.fillna(0).values, ctx.floor)
        amounts = ["y"] + [c for c in (ctx.amount_cols or []) if c in df]
        X = pd.DataFrame({c: np.clip(df[c].values / sc, 0, 50) for c in amounts})
        for c in (ctx.feature_cols or []):
            if c not in amounts: X[c] = pd.to_numeric(df[c], errors="coerce").values
        X["log_scale"] = np.log1p(sc); X["level_code"] = df.group.map({l: j for j, l in enumerate(ctx.cfg.groups)}).values
        return X, sc

    def predict(self, histories, horizon, ctx):
        import lightgbm as lgb
        F, i = ctx.features, ctx.i
        tr = F[(F.widx <= i - 1 - horizon) & F.y_future.notna() & F.series.isin(ctx.eligible)]
        cur = F[(F.widx == i - 1) & F.series.isin(list(histories))]
        if len(tr) < self.min_rows or cur.empty: return {}
        mult = horizon if ctx.cfg.aggregate == "sum" else 1
        X, sc = self._X(tr, ctx)
        m = lgb.LGBMRegressor(objective="l1", n_estimators=200, learning_rate=0.05, num_leaves=8, min_child_samples=10, subsample=0.8, subsample_freq=1, random_state=0, verbose=-1)
        m.fit(X, np.clip(tr.y_future.values / (mult * sc), 0, 50))
        Xc, scc = self._X(cur, ctx)
        agg = np.clip(m.predict(Xc), 0, None) * mult * scc                     # the horizon aggregate; spread it so that aggregating the periods gives it back
        per = agg / horizon if ctx.cfg.aggregate == "sum" else agg
        return {n: np.full(horizon, v) for n, v in zip(cur.series, per)}
