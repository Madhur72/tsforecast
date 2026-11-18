"""Stage 1 - raw rows -> clean, regular series (one column per series) and a profile of each series. Nothing here assumes what the data means.

    prep = prepare(df, cfg, steps=[...])        # steps are optional, use-case specific extras (see steps.py)
    prep.series[target]    periods x series (complete periods only)
    prep.aux[name]         extra per-period numbers you asked for in Config.aux
    prep.profile           history, sparsity and pattern per series
"""
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np, pandas as pd

RULE = {"D": "D", "W": "W-MON", "M": "MS"}                       # pandas rule for the start of each period
TOP = "all"                                                      # the group made of every row together; its single series is called "all|ALL"


def period_start(day: pd.Series, freq: str) -> pd.Series:
    if freq == "D": return day
    if freq == "W": return day - pd.to_timedelta(day.dt.dayofweek, unit="D")
    return day.dt.to_period("M").dt.start_time


def period_end(t, freq: str) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t if freq == "D" else t + pd.Timedelta(days=6) if freq == "W" else t + pd.offsets.MonthEnd(0)


def next_period(t, freq: str) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t + (pd.Timedelta(days=1) if freq == "D" else pd.Timedelta(days=7) if freq == "W" else pd.DateOffset(months=1))


def group_of(series: str) -> str: return series.split("|", 1)[0]


def parent_of_series(series: str, cfg):
    """'store|north/s1' -> the series of its parent group, if Config.parent_of declares one; otherwise the top series 'all|ALL'."""
    g, rest = series.split("|", 1)
    if g == TOP: return None
    pg = (cfg.parent_of or {}).get(g)
    return pg + "|" + "/".join(rest.split("/")[:len(cfg.groups[pg])]) if pg else f"{TOP}|ALL"


# ------------------------------------------------------------------------------------------------ steps (optional, use-case specific)
class Step:
    """Optional use-case logic. Override what you need; all three hooks are optional. See steps.py for examples."""
    def before(self, d: pd.DataFrame, cfg) -> pd.DataFrame:
        """Runs on the cleaned rows, before series are built (e.g. rename tiny groups). Must return the rows."""
        return d

    def after(self, prep: "Prepared") -> None:
        """Runs once the series are built (e.g. compute flags and store them in prep.extras)."""

    def cut(self, prep: "Prepared", i: int) -> dict:
        """What this step knows after the first i periods only (i=None: everything). Becomes view.extras, so features cannot peek at the future."""
        return {}


# ------------------------------------------------------------------------------------------------ cleaning
def clean(df: pd.DataFrame, cfg):
    """Fix what is wrong with the rows and count every fix, so nothing disappears silently."""
    if not cfg.targets or not cfg.time_col: raise ValueError("Config needs target and time_col")
    main, rep, d = cfg.targets[0], {"rows_in": len(df)}, df.copy()
    ts = pd.to_datetime(d[cfg.time_col], utc=True, errors="coerce")
    rep["bad_timestamps"] = int(ts.isna().sum()); d["_ts"] = ts; d = d.dropna(subset=["_ts"])
    if cfg.id_col and cfg.id_col in d:
        n = len(d); d = d.drop_duplicates(cfg.id_col); rep["duplicate_ids"] = n - len(d)
    for t in cfg.targets: d[t] = pd.to_numeric(d[t], errors="coerce")
    rep["null_target"] = int(d[main].isna().sum()); rep["negative_target"] = int((d[main] < 0).sum())
    d = d[d[main].notna() & (d[main] >= 0)]
    for t in cfg.targets[1:]: d[t] = d[t].fillna(0.0).clip(lower=0.0)        # an extra target never removes a row
    cols = sorted({c for ks in cfg.groups.values() for c in ks} | {c for c, how in (cfg.aux or {}).values() if c and how == "nunique"})
    rep["missing_keys_filled"] = int(d[cols].isna().sum().sum()) if cols else 0
    d[cols] = d[cols].fillna("UNKNOWN").astype(str)
    for c, how in (cfg.aux or {}).values():
        if c and how in ("sum", "mean"): d[c] = pd.to_numeric(d[c], errors="coerce")
    d["_day"] = d["_ts"].dt.tz_convert(cfg.tz).dt.tz_localize(None).dt.normalize()
    d["_period"] = period_start(d["_day"], cfg.freq)
    rep["rows_out"] = len(d)
    return d, rep


# ------------------------------------------------------------------------------------------------ series
def _wide(x, group, keys, idx, tcol, how, col):
    g = x.groupby([tcol] + keys)
    s = {"sum": lambda: g[col].sum(), "mean": lambda: g[col].mean(), "nunique": lambda: g[col].nunique(), "size": lambda: g.size()}[how]()
    if keys:
        w = s.unstack(keys).reindex(idx)
        w.columns = [group + "|" + "/".join(map(str, c if isinstance(c, tuple) else (c,))) for c in w.columns]
    else:
        w = s.reindex(idx).to_frame(f"{TOP}|ALL")
    return w.ffill() if how == "mean" else w.fillna(0.0)            # a level that was not observed keeps its last value; an amount is 0


def build_frames(d: pd.DataFrame, cfg):
    """(daily frame per target, aux frames per period): one column per series."""
    days = pd.date_range(d["_day"].min(), d["_day"].max(), freq="D")
    daily = {t: pd.concat([_wide(d, g, ks, days, "_day", cfg.row_agg, t) for g, ks in cfg.groups.items()], axis=1) for t in cfg.targets}
    periods = pd.date_range(d["_period"].min(), d["_period"].max(), freq=RULE[cfg.freq])
    aux = {name: pd.concat([_wide(d, g, ks, periods, "_period", how, col or cfg.targets[0]) for g, ks in cfg.groups.items()], axis=1) for name, (col, how) in (cfg.aux or {}).items()}
    return daily, aux


def to_periods(daily: pd.DataFrame, cfg) -> pd.DataFrame:
    """Daily -> periods, keeping only COMPLETE periods (an unfinished last period would look like a collapse)."""
    if cfg.freq == "D": return daily
    agg = "sum" if cfg.row_agg == "sum" else "mean"
    p = (daily.resample("W-MON", closed="left", label="left") if cfg.freq == "W" else daily.resample("MS")).agg(agg)
    last = daily.index[-1]
    return p[[period_end(t, cfg.freq) <= last for t in p.index]]


# ------------------------------------------------------------------------------------------------ profile
def profile(frame: pd.DataFrame, cfg) -> pd.DataFrame:
    """History, sparsity and demand pattern per series. status: ok | sparse (mostly zeros) | short_history | no_data."""
    rows = []
    for c in frame.columns:
        s = frame[c]; nz = s[s > 0]
        if nz.empty:
            rows.append(dict(series=c, first_active=pd.NaT, history=0, active_periods=0, zero_share=1.0, ADI=np.inf, CV2=np.nan, pattern="no_data", status="no_data")); continue
        s = s.loc[nz.index[0]:]; n_nz = int((s > 0).sum()); adi = len(s) / n_nz; cv2 = float((nz.std(ddof=0) / nz.mean()) ** 2)
        pattern = ("smooth" if cv2 < .49 else "erratic") if adi < 1.32 else ("intermittent" if cv2 < .49 else "lumpy")
        zs = float((s == 0).mean())
        status = "short_history" if len(s) < cfg.min_history else ("sparse" if zs > .6 else "ok")
        rows.append(dict(series=c, first_active=s.index[0], history=len(s), active_periods=n_nz, zero_share=round(zs, 2), ADI=round(adi, 2), CV2=round(cv2, 2), pattern=pattern, status=status))
    return pd.DataFrame(rows).set_index("series")


@dataclass
class View:
    """The data as known after the first `i` periods. Feature builders and history filters receive a View and must only use what is in it."""
    cfg: object
    target: str
    Y: pd.DataFrame                                 # the target: periods x series
    series: Dict[str, pd.DataFrame]                 # every target
    aux: Dict[str, pd.DataFrame]                    # the extra per-period numbers from Config.aux
    extras: dict = field(default_factory=dict)      # whatever the steps added (as known at that time)


@dataclass
class Prepared:
    cfg: object
    series: Dict[str, pd.DataFrame]                 # target -> periods x series
    aux: Dict[str, pd.DataFrame]                    # name -> periods x series
    profile: pd.DataFrame
    daily: Dict[str, pd.DataFrame]                  # target -> days x series
    steps: List[Step] = field(default_factory=list)
    extras: dict = field(default_factory=dict)      # filled by steps
    report: dict = field(default_factory=dict)

    @property
    def status(self) -> dict: return self.profile.status.to_dict()

    @property
    def eligible(self) -> list: return [n for n, s in self.status.items() if s in ("ok", "sparse")]

    def view(self, target=None, i: int = None) -> View:
        """The data for one target, cut to the first i periods (None = all periods)."""
        t = target or self.cfg.targets[0]; n = len(self.series[t]) if i is None else i
        extras = dict(self.extras) if i is None else {}
        for s in self.steps: extras.update(s.cut(self, i) or {})
        return View(self.cfg, t, self.series[t].iloc[:n], {k: v.iloc[:n] for k, v in self.series.items()}, {k: v.iloc[:n] for k, v in self.aux.items()}, extras)


def _describe(frame, cfg):
    from .treat import describe
    return describe(frame, cfg)


def prepare(df: pd.DataFrame, cfg, steps=None) -> Prepared:
    steps = list(steps or [])
    d, rep = clean(df, cfg)
    for s in steps: d = s.before(d, cfg)
    daily, aux = build_frames(d, cfg)
    series = {t: to_periods(daily[t], cfg) for t in cfg.targets}
    idx, cols = series[cfg.targets[0]].index, series[cfg.targets[0]].columns
    aux = {k: v.reindex(index=idx, columns=cols).fillna(0.0) for k, v in aux.items()}
    prep = Prepared(cfg, series, aux, profile(series[cfg.targets[0]], cfg).join(_describe(series[cfg.targets[0]], cfg)), daily, steps, {}, rep)
    for s in steps: s.after(prep)
    return prep
