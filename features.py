"""Stage 2 - features. Features depend on the use case, so there is NO fixed feature table: you pick builders, or write your own.

A feature builder is a class with `build(view)`, or just a function `fn(view)`. It receives a View (the data as known at that point) and returns
{feature name: DataFrame of periods x series, same shape as view.Y}. Values at period t may use data up to and including t only (the leakage check enforces it).

    class MyFeatures(FeatureBuilder):
        amount_cols = ("promo_sales",)                  # features in the target's unit: models scale them by the series' level
        def build(self, view):
            Y = view.Y
            return {"promo_sales": Y.where(view.aux["count"] > 5), "dow_peak": Y.rolling(7).max()}

    Forecaster(cfg, features=[TimeSeriesFeatures(), MyFeatures()])

The module itself adds only the reserved columns every model needs: period, series, group, widx, y (the target), level8 (the recent level), y_future (what is being forecast).
Models that use features (LightGBM) take whatever columns the builders produced. Models that do not use features never trigger any of this.
"""
from typing import Dict, List, Tuple
import numpy as np, pandas as pd
from .preprocess import parent_of_series, period_end

RESERVED = {"period", "series", "group", "widx", "y", "level8", "y_future"}


def _slope(a):
    m = np.isfinite(a)
    return np.polyfit(np.arange(len(a))[m], a[m], 1)[0] if m.sum() >= 6 else np.nan


class FeatureBuilder:
    """Base class for a feature builder. Set `amount_cols` to the names (in what build() returns) that are amounts in the target's unit."""
    amount_cols: tuple = ()

    def build(self, view) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError


# ------------------------------------------------------------------------------------------------ reusable builders (generic: they only need the target series, aux series, or a declared hierarchy)
class TimeSeriesFeatures(FeatureBuilder):
    """Computed from the target series alone, so they make sense for any series: lags, rolling statistics, trend, sparsity, spikes, month."""
    amount_cols = ("lag_1", "lag_2", "lag_3", "mean_4", "mean_8", "mean_12", "median_4", "median_12", "max_8", "std_8", "ewm_4")

    def build(self, view):
        Y = view.Y; started = Y.cumsum() > 0; Yn = Y.where(started); L = len(Y); f = {}
        for k in (1, 2, 3): f[f"lag_{k}"] = Yn.shift(k)
        for w in (4, 8, 12): f[f"mean_{w}"] = Yn.rolling(w, min_periods=2).mean()
        f["median_4"], f["median_12"] = Yn.rolling(4, min_periods=2).median(), Yn.rolling(12, min_periods=3).median()
        f["max_8"], f["std_8"], f["ewm_4"] = Yn.rolling(8, min_periods=2).max(), Yn.rolling(8, min_periods=3).std(), Yn.ewm(alpha=0.5, min_periods=2).mean()
        f["level_ratio"] = Yn / f["median_12"].replace(0, np.nan)
        f["trend_8"] = Yn.rolling(8, min_periods=6).apply(_slope, raw=True) / f["mean_8"].replace(0, np.nan)
        z = (Yn == 0).astype(float).where(Yn.notna()); f["zero_share_4"], f["zero_share_8"] = z.rolling(4, min_periods=2).mean(), z.rolling(8, min_periods=2).mean()
        idx = np.arange(L)[:, None]
        def since(mask):                                              # periods since the last True (0 = this period)
            pos = pd.DataFrame(np.where(mask.values, idx, np.nan), index=Y.index, columns=Y.columns).ffill()
            return pd.DataFrame(idx - pos.values, index=Y.index, columns=Y.columns)
        f["periods_since_active"] = since((Yn > 0).fillna(False))
        f["active_periods"] = (Yn > 0).cumsum()
        thr = 5 * Yn.where(Yn > 0).expanding(min_periods=4).median(); spike = (Yn > thr).fillna(False)
        f["is_spike"] = spike.astype(float).where(thr.notna()); f["periods_since_spike"] = since(spike)
        f["concentration_12"] = Yn.rolling(12, min_periods=4).max() / Yn.rolling(12, min_periods=4).sum().replace(0, np.nan)
        f["month"] = pd.DataFrame(np.repeat(np.asarray(Y.index.month, float)[:, None], Y.shape[1], 1), index=Y.index, columns=Y.columns)
        return f


class AuxFeatures(FeatureBuilder):
    """Every extra per-period series you declared in Config.aux (for example a count of distinct customers, or a price): the value, its recent mean, and the target per unit of it."""
    def build(self, view):
        f = {}
        for name, A in view.aux.items():
            f[name] = A; f[f"{name}_mean_4"] = A.rolling(4, min_periods=2).mean(); f[f"value_per_{name}"] = view.Y / A.where(A > 0)
        return f


class HierarchyFeatures(FeatureBuilder):
    """The parent's recent level, and this series' share of it (needs parent_of in the Config)."""
    amount_cols = ("parent_mean_4",)

    def build(self, view):
        Y = view.Y; started = Y.cumsum() > 0; m4 = Y.where(started).rolling(4, min_periods=2).mean()
        par = {n: parent_of_series(n, view.cfg) for n in Y.columns}
        pm4 = pd.DataFrame({n: (m4[p] if p in Y.columns else np.nan) for n, p in par.items()}, index=Y.index)
        return {"parent_mean_4": pm4, "parent_share_4": m4 / pm4.replace(0, np.nan)}


def default_builders(cfg) -> list:
    """A sensible starting set that needs nothing use-case specific. Replace it with your own list: Forecaster(cfg, features=[...])."""
    b = [TimeSeriesFeatures()]
    if cfg.aux: b.append(AuxFeatures())
    if cfg.parent_of: b.append(HierarchyFeatures())
    return b


# ------------------------------------------------------------------------------------------------ assembly
def build_features(prep, target=None, builders=None, i: int = None) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Run the builders on the data known after the first i periods and stack the result into one long table (one row per series x period).
    Returns (table, feature column names, the feature columns that are amounts)."""
    cfg = prep.cfg; view = prep.view(target, i); Y = view.Y; H = cfg.horizon
    builders = default_builders(cfg) if builders is None else builders
    started = Y.cumsum() > 0; Yn = Y.where(started)
    frames, amounts = {}, set()
    for b in builders:
        out = b.build(view) if hasattr(b, "build") else b(view)
        for k, fr in out.items():
            if k in RESERVED: raise ValueError(f"feature name '{k}' is reserved")
            frames[k] = fr.reindex(index=Y.index, columns=Y.columns).where(started)
        amounts |= set(getattr(b, "amount_cols", ()))
    fut = [Yn.shift(-k) for k in range(1, H + 1)]                          # TARGET: the horizon aggregate. Never an input.
    reserved = {"y": Yn, "level8": Yn.rolling(8, min_periods=2).mean(),
                "y_future": sum(fut) if cfg.aggregate == "sum" else (sum(fut) / H if cfg.aggregate == "mean" else Yn.shift(-H))}
    out = None
    for name, v in {**reserved, **frames}.items():
        m = v.rename_axis("period").reset_index().melt(id_vars="period", var_name="series", value_name=name).set_index(["period", "series"])
        out = m if out is None else out.join(m)
    out = out.reset_index(); out = out[out["y"].notna()].copy()
    out["group"] = out.series.str.split("|").str[0]; out["widx"] = out.period.map({p: j for j, p in enumerate(Y.index)})
    return out, list(frames), sorted(a for a in amounts if a in frames)


def check_leakage(prep, target=None, builders=None, back: int = 9) -> bool:
    """Features at an old period must be identical whether or not later data exists. Works for any builders, including yours. True = nothing differs."""
    full, cols, _ = build_features(prep, target, builders); n = len(prep.series[target or prep.cfg.targets[0]]) - back
    part, _, _ = build_features(prep, target, builders, i=n)
    t0 = prep.series[target or prep.cfg.targets[0]].index[n - 1]
    a = full[full.period == t0].set_index("series")[cols].sort_index(); b = part[part.period == t0].set_index("series")[cols].sort_index().reindex(a.index)
    return bool(np.isclose(a.values.astype(float), b.values.astype(float), equal_nan=True).all())
