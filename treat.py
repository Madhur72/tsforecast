"""Series treatment: what is done to ONE series' history just before a model sees it. Applied per forecast origin, using only that history (no leakage).

  fill       missing values (NaN, and zeros if `zeros_missing`) are interpolated: "linear" | "ffill" | "none"
  outliers   "clip": values beyond median +/- outlier_k robust deviations (MAD) are pulled back to that bound; "none"
  transform  "log1p" | "sqrt" | "none" | "auto" (auto = log1p when the history's skewness exceeds skew_limit); predictions are mapped back

All of it is set in Config, so a new dataset only changes settings, never code. `describe` reports skewness and outliers per series so the choice can be informed."""
from typing import Callable, Tuple
import numpy as np, pandas as pd


def skewness(x) -> float:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3 or x.std() == 0: return 0.0
    return float(((x - x.mean()) ** 3).mean() / x.std() ** 3)


def _bounds(x, k):
    med = np.nanmedian(x); mad = np.nanmedian(np.abs(x - med)) * 1.4826
    if mad == 0: mad = np.nanstd(x) if np.nanstd(x) > 0 else 0.0
    return (med - k * mad, med + k * mad) if mad > 0 else (-np.inf, np.inf)


def treat(x: np.ndarray, cfg) -> Tuple[np.ndarray, Callable]:
    """Returns (treated history, inverse function for predictions)."""
    x = np.asarray(x, float).copy(); ident = lambda a: np.asarray(a, float)
    if cfg.zeros_missing: x[x == 0] = np.nan
    if np.isnan(x).any() and cfg.fill != "none":
        s = pd.Series(x); s = s.interpolate(limit_area="inside") if cfg.fill == "linear" else s.ffill()
        x = s.bfill().ffill().values
    x = np.nan_to_num(x, nan=0.0)
    if cfg.outliers == "clip":
        lo, hi = _bounds(x, cfg.outlier_k); x = np.clip(x, max(lo, 0.0), hi)
    t = cfg.transform
    if t == "auto": t = "log1p" if skewness(x) > cfg.skew_limit and x.min() >= 0 else "none"
    if t == "log1p": return np.log1p(x), (lambda a: np.clip(np.expm1(np.asarray(a, float)), 0, None))
    if t == "sqrt": return np.sqrt(np.clip(x, 0, None)), (lambda a: np.clip(np.asarray(a, float), 0, None) ** 2)
    return x, ident


def describe(frame: pd.DataFrame, cfg) -> pd.DataFrame:
    """Per series: skewness, outlier count, share of missing/zero values. Use it to decide the Config treatment options."""
    rows = {}
    for c in frame.columns:
        s = frame[c]; nz = np.flatnonzero(s.fillna(0).values > 0)
        v = s.iloc[nz[0]:].values.astype(float) if len(nz) else np.array([])
        lo, hi = _bounds(v, cfg.outlier_k) if len(v) else (-np.inf, np.inf)
        rows[c] = dict(skew=round(skewness(v), 2), outliers=int(((v < lo) | (v > hi)).sum()) if len(v) else 0, missing_share=round(float(np.isnan(v).mean()), 3) if len(v) else 0.0,
                       zero_share_active=round(float((v == 0).mean()), 3) if len(v) else 0.0)
    return pd.DataFrame(rows).T.rename_axis("series")
