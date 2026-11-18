"""Seasonality detection. The statistical models (ETS, ARIMA, Theta, SeasonalNaive) need the length of the seasonal cycle in periods; this finds it from the data instead of assuming one.

How: for every usable series (status ok, not sparse) take log1p of its values from its first non-zero period, difference once (removes level and trend), and measure the autocorrelation at each
candidate cycle length for the frequency (day: 7, 14, 30; week: 4, 13, 26, 52; month: 12, 6, 4, 3). A candidate needs at least 3 full cycles of history. The pooled score of a length is the MEDIAN
over series, so a few lumpy series cannot create a season. A length counts only if its score is above max(0.25, 2 / sqrt(n)); of the lengths that count, the smallest one within 10% of the best is
used (so 14 is not chosen when the real cycle is 7). If none counts, the old default applies (7 for days, 1 otherwise: no seasonality).

Config: `season="auto"` (default) detects; `season="manual"` uses `season_length` as given; `season="off"` uses the default. The report (`detect(...)[1]`) shows every candidate's score."""
import numpy as np, pandas as pd

CANDIDATES = {"D": (7, 14, 30), "W": (4, 13, 26, 52), "M": (12, 6, 4, 3)}
MIN_SCORE, MIN_CYCLES, NEAR_BEST = 0.25, 3, 0.9


def default_length(cfg) -> int: return 7 if cfg.freq == "D" else 1


def _acf(d: np.ndarray, lag: int) -> float:
    d = d - d.mean(); den = float((d * d).sum())
    return float((d[lag:] * d[:-lag]).sum() / den) if den > 0 and len(d) > lag else 0.0


def scores(frame: pd.DataFrame, cfg, status: dict) -> pd.DataFrame:
    """One row per (series, candidate length): the autocorrelation of the differenced log series at that lag."""
    rows = []
    for c in frame.columns:
        if status.get(c) != "ok": continue
        v = frame[c].values.astype(float); nz = np.flatnonzero(v > 0)
        if not len(nz): continue
        x = np.log1p(np.clip(v[nz[0]:], 0, None)); d = np.diff(x)
        for m in CANDIDATES[cfg.freq]:
            if len(x) >= MIN_CYCLES * m + 1: rows.append(dict(series=c, length=m, acf=_acf(d, m), n=len(d)))
    return pd.DataFrame(rows, columns=["series", "length", "acf", "n"])


def detect(frame: pd.DataFrame, cfg, status: dict):
    """-> (season length in periods, report per candidate length: pooled score, threshold, series counted, passed)."""
    default = default_length(cfg)
    if getattr(cfg, "season", "auto") != "auto": return cfg.season_length, pd.DataFrame()
    sc = scores(frame, cfg, status)
    if sc.empty: return default, pd.DataFrame()
    rep = sc.groupby("length").agg(score=("acf", "median"), n=("n", "median"), series=("series", "nunique"))
    rep["threshold"] = np.maximum(MIN_SCORE, 2 / np.sqrt(rep.n)); rep["passed"] = rep.score >= rep.threshold
    ok = rep[rep.passed]
    if ok.empty: return default, rep
    best = ok.score.max(); return int(ok[ok.score >= NEAR_BEST * best].index.min()), rep
