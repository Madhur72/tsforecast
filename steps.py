"""Optional, USE-CASE SPECIFIC helpers. The core (preprocess, evaluate, forecast, models) knows nothing about them: use one only when it fits your data, or write your own
the same way (subclass Step; features: subclass FeatureBuilder; history filter: a function fn(view) -> True for periods to leave out).

    Forecaster(cfg, steps=[FoldSmallGroups(min_active=10), FlagOutages()], features=[TimeSeriesFeatures(), OutageFeatures()], history_filter=drop_outages())

FoldSmallGroups   tiny series are merged into one OTHER series under their parent, so they do not each need a model (event logs with many small users / teams)
FlagOutages       flags periods where EVERYTHING was zero for several weekdays in a row: suspected missing data, not real zeros (event logs that can stop being recorded)
OutageFeatures    the flags as a feature for models that use features
drop_outages      a history filter that leaves flagged periods out of every model's history
drop_ranges       a history filter for KNOWN closures (company leave, shutdowns): periods mostly inside the given date ranges are left out
combine           several history filters as one
"""
import numpy as np, pandas as pd
from .preprocess import Step, TOP, RULE, period_end
from .features import FeatureBuilder

OUTAGE_DAYS_PER_PERIOD = {"D": 1, "W": 3, "M": 10}      # flagged days that make a whole period "suspected missing data"


class FoldSmallGroups(Step):
    """For each group listed in Config.parent_of: series with fewer than `min_active` active periods are renamed OTHER under their parent.
    Note: the groups' own columns (those that are not part of the parent) should be different columns, e.g. team_id for one group and agent_id for another."""

    def __init__(self, min_active: int = 10): self.min_active = min_active

    def before(self, d, cfg):
        main, x = cfg.targets[0], d.copy()
        for g, ks in cfg.groups.items():
            if not ks or g not in (cfg.parent_of or {}): continue
            pg = cfg.parent_of[g]; pk = cfg.groups[pg] if pg else []; child = [k for k in ks if k not in pk]
            act = d[d[main] > 0].groupby(ks)["_period"].nunique().rename("aw").reset_index()
            m = d[ks].drop_duplicates().merge(act, on=ks, how="left").fillna({"aw": 0})
            small = m[m.aw < self.min_active][ks].assign(_small=True)
            flag = d[ks].merge(small, on=ks, how="left")["_small"].eq(True).values
            x.loc[flag, child] = "OTHER"
        return x


def suspected_days(total: pd.Series, weekdays: int) -> pd.Series:
    """Zero for a run covering at least `weekdays` weekdays, strictly between the first and the last activity. Weekends alone never count."""
    s = total.fillna(0.0)
    if not (s > 0).any(): return pd.Series(False, index=s.index)
    first, last = s[s > 0].index[0], s[s > 0].index[-1]
    z = s == 0; runs = (z != z.shift()).cumsum()
    wd = (z & (s.index.dayofweek < 5)).groupby(runs).transform("sum")
    return z & (wd >= weekdays) & (s.index > first) & (s.index < last)


def outage_periods(total_daily: pd.Series, index: pd.Index, cfg, weekdays: int) -> pd.Series:
    """Number of suspected-missing days in each period of `index`."""
    flags = suspected_days(total_daily, weekdays).astype(float)
    if cfg.freq != "D": flags = flags.resample(RULE[cfg.freq], **({"closed": "left", "label": "left"} if cfg.freq == "W" else {})).sum()
    return flags.reindex(index).fillna(0.0)


class FlagOutages(Step):
    """Adds `outage_days` (suspected-missing days per period) to prep.extras, and to every view as known at that time."""

    def __init__(self, weekdays: int = 3): self.weekdays = weekdays

    def _days(self, prep, i):
        t = prep.cfg.targets[0]; Y = prep.series[t]; tot = prep.daily[t][f"{TOP}|ALL"]
        if i is None: return outage_periods(tot, Y.index, prep.cfg, self.weekdays)
        return outage_periods(tot.loc[:period_end(Y.index[i - 1], prep.cfg.freq)], Y.index[:i], prep.cfg, self.weekdays)     # only what was known then

    def after(self, prep): prep.extras["outage_days"] = self._days(prep, None)

    def cut(self, prep, i): return {"outage_days": self._days(prep, i)}


class OutageFeatures(FeatureBuilder):
    """The suspected-missing days of the period, as a feature."""
    def build(self, view):
        Y = view.Y; o = view.extras.get("outage_days")
        if o is None: return {}
        return {"outage_days": pd.DataFrame(np.repeat(o.reindex(Y.index).fillna(0).values[:, None], Y.shape[1], 1), index=Y.index, columns=Y.columns)}


def drop_outages(min_days: int = None):
    """A history filter: leave periods with at least `min_days` suspected-missing days (default depends on the period length) out of every model's history."""
    def fn(view):
        o = view.extras["outage_days"]
        return o >= (min_days or OUTAGE_DAYS_PER_PERIOD[view.cfg.freq])
    return fn


def drop_ranges(ranges, min_share: float = 0.5):
    """A history filter for KNOWN closures (company leave, shutdowns, public-holiday breaks): `ranges` = [(first_day, last_day), ...].
    A period is left out of every model's history when at least `min_share` of its days fall inside a closure. Weekends need no entry: daily models learn them."""
    spans = [(pd.Timestamp(a).normalize(), pd.Timestamp(b).normalize()) for a, b in ranges]

    def fn(view):
        idx = view.Y.index; freq = view.cfg.freq; share = pd.Series(0.0, index=idx)
        for t in idx:
            a, b = t, period_end(t, freq); days = (b - a).days + 1
            inside = sum(max(0, (min(b, hi) - max(a, lo)).days + 1) for lo, hi in spans)
            share[t] = min(inside, days) / days
        return share >= min_share
    return fn


def combine(*filters):
    """Several history filters as one: a period is left out if ANY of them says so."""
    def fn(view):
        out = None
        for f in filters:
            m = f(view)
            out = m if out is None else (out | m.reindex(out.index).fillna(False))
        return out
    return fn
