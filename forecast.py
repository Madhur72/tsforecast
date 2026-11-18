"""The Forecaster: runs every stage in order for any data.

    f = Forecaster(Config(target="sales", time_col="date", keys=["store"]))
    result = f.run(df)
    result.champions      the best model per group, with its honest test accuracy
    result.forecast       the next `horizon` periods for every series, with a range and a confidence label

Stages: 1 preprocess -> 2 features (only if a model needs them) -> 3 split -> 4 run every model at every origin -> 5 evaluate, pick champions -> 6 forecast with the champions.

Optional, use-case specific hooks (nothing is assumed by default):
    steps=[...]           preprocessing extras (steps.py), for example FoldSmallGroups, FlagOutages
    features=[...]        feature builders (features.py), only used by models that need features
    history_filter=fn     fn(view) -> True for periods to leave out of every model's history"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np, pandas as pd
from .config import Config
from .base import ModelRegistry, Context
from .preprocess import prepare, Prepared, group_of, next_period, RULE
from .features import build_features, check_leakage, default_builders
from .split import make_origins
from .treat import treat, describe
from .season import detect as detect_season
from . import evaluate as E


def reduce(arr, how: str) -> float:
    arr = np.asarray(arr, float)
    return float(arr.sum() if how == "sum" else arr.mean() if how == "mean" else arr[-1])


@dataclass
class Result:
    target: str
    config: Config
    prepared: Prepared
    records: pd.DataFrame                    # every forecast of every model at every origin
    comparable: pd.DataFrame                 # the same, restricted to cases every model could forecast
    val_table: pd.DataFrame                  # WAPE: models x groups, validation (chooses the champion)
    test_table: pd.DataFrame                 # WAPE: models x groups, test (the honest score)
    champions: pd.DataFrame                  # per group: champion, validation and test WAPE, accuracy, bias, skill vs baseline, confidence
    champion_map: Dict[str, str]
    ranges: pd.DataFrame                     # does the 80% range hold on test?
    forecast: pd.DataFrame                   # next-horizon aggregate per series, with low / high / method / confidence
    forecast_periods: pd.DataFrame           # the same, period by period
    origins: dict = field(default_factory=dict)
    skipped_series: List[str] = field(default_factory=list)
    skipped_models: Dict[str, str] = field(default_factory=dict)
    leakage_ok: Optional[bool] = None

    @property
    def leaderboard(self) -> pd.DataFrame:
        return pd.concat({"validation": self.val_table, "test": self.test_table}, names=["split"])

    def summary(self) -> str:
        S = self.prepared.series[self.target]
        s = [f"target: {self.target} | {S.shape[1]} series | {S.shape[0]} periods | horizon {self.config.horizon} ({self.config.aggregate})",
             f"origins: {len(self.origins['val'])} validation, {len(self.origins['test'])} test | models run: {sorted(self.records.model.unique())}"]
        if self.skipped_models: s.append(f"models skipped: {self.skipped_models}")
        if self.skipped_series: s.append(f"series not forecast (too little history): {len(self.skipped_series)}")
        if self.leakage_ok is not None: s.append(f"feature leakage check: {'passed' if self.leakage_ok else 'FAILED'}")
        s.append("\nchampion per group (chosen on validation, scored on test):\n" + self.champions.to_string())
        s.append("\n80% range coverage on test (target about 0.80):\n" + self.ranges.to_string())
        return "\n".join(s)

    def save(self, folder: str) -> str:
        os.makedirs(folder, exist_ok=True)
        for name in ("champions", "ranges", "forecast", "forecast_periods", "leaderboard"):
            getattr(self, name).to_csv(os.path.join(folder, f"{name}.csv"))
        self.records.to_csv(os.path.join(folder, "records.csv"), index=False); self.prepared.profile.to_csv(os.path.join(folder, "series_profile.csv"))
        return folder


def _histories(Y: pd.DataFrame, series, i: int, min_history: int, drop=None, cfg=None):
    """({series: treated values from the series' first activity up to period i-1}, {series: inverse transform}); optionally without the periods flagged in `drop`."""
    out, inv = {}, {}
    for n in series:
        s = Y[n].iloc[:i]; nz = np.flatnonzero(s.values > 0)
        if not len(nz): continue
        s = s.iloc[nz[0]:]
        if drop is not None: s = s[~drop.reindex(s.index).fillna(False).values.astype(bool)]
        if len(s) >= min_history: out[n], inv[n] = treat(s.values.astype(float), cfg)
    return out, inv


class Forecaster:
    def __init__(self, config: Config = None, registry: ModelRegistry = None, steps=None, features=None, history_filter=None, **overrides):
        self.cfg = (config or Config()).with_(**overrides) if overrides else (config or Config())
        self.registry = registry or ModelRegistry()
        self.steps, self.builders, self.history_filter = steps, features, history_filter

    def log(self, *a):
        if self.cfg.verbose: print(*a, flush=True)

    # ------------------------------------------------------------------------------------------ public
    def run(self, df: pd.DataFrame, target=None):
        """Run every stage. Returns a Result, or {target: Result} when the config has several targets."""
        cfg = self.cfg if target is None else self.cfg.with_(target=target)
        self.log(f"[1/6] preprocess: {len(df):,} rows" + (f" | steps: {[type(s).__name__ for s in self.steps]}" if self.steps else ""))
        prep = prepare(df, cfg, self.steps)
        S = prep.series[cfg.targets[0]]
        self.log(f"      {S.shape[1]} series x {S.shape[0]} periods | status {prep.profile.status.value_counts().to_dict()}")
        results = {t: self._run_target(prep, t, self._season(prep, cfg, t)) for t in cfg.targets}
        return results[cfg.targets[0]] if len(results) == 1 else results

    def _season(self, prep, cfg: Config, target: str) -> Config:
        """Season length: detected from this target's series when cfg.season == "auto" (see season.py), otherwise as configured."""
        if cfg.season != "auto": return cfg
        m, rep = detect_season(prep.series[target], cfg, prep.profile.status.to_dict()); self.season_report = rep
        if m != cfg.season_length: self.log(f"      season length detected for {target}: {m} (default {cfg.season_length})")
        return cfg.with_(season_length=m)

    def _drop(self, prep, target, i):
        return self.history_filter(prep.view(target, i)) if self.history_filter else None

    def forecast_with(self, df: pd.DataFrame, champions: Dict[str, str], q: dict, wape: Dict[str, float], target=None):
        """Forecast WITHOUT re-evaluating: use champions chosen earlier. champions {group: model}, q as in evaluate.error_quantiles, wape {group: test wape} (for the confidence label).
        Returns (forecast, forecast_periods, prepared). Cheap enough for a nightly job; re-run run() on a slower schedule to refresh the champions."""
        cfg = self.cfg; t = target or cfg.targets[0]; prep = prepare(df, cfg, self.steps); cfg = self._season(prep, cfg, t); Y = prep.series[t]
        pos = Y.values[(Y.values > 0) & np.isfinite(Y.values)]; floor = max(1e-9, 0.05 * float(np.median(pos))) if len(pos) else 1.0
        mult = 1 if cfg.score == "step" else (cfg.horizon if cfg.aggregate == "sum" else 1)
        features, fcols, acols = None, None, None
        if any(self.registry.models[m].requires_features for m in set(champions.values())):
            builders = self.builders if self.builders is not None else default_builders(cfg); features, fcols, acols = build_features(prep, t, builders)
        ctab = pd.DataFrame({"test_wape": pd.Series(wape), "confidence": {g: E.confidence(w, cfg) for g, w in wape.items()}})
        fc, per = self._final_forecast(prep, t, cfg, champions, ctab, q, features, prep.eligible, floor, mult, fcols, acols)
        return fc, per, prep

    # ------------------------------------------------------------------------------------------ one target
    def _run_target(self, prep: Prepared, target: str, cfg: Config) -> Result:
        reg, H, Y = self.registry, cfg.horizon, prep.series[target]
        L, eligible = len(Y), prep.eligible
        pos = Y.values[(Y.values > 0) & np.isfinite(Y.values)]
        floor = max(1e-9, 0.05 * float(np.median(pos))) if len(pos) else 1.0
        names = reg.select(cfg)
        val_i, test_i = make_origins(L, H, cfg.n_test, cfg.n_val, cfg.min_history, cfg.origin_step)
        features, fcols, acols, leak = None, None, None, None
        if any(reg.models[n].requires_features for n in names):
            builders = self.builders if self.builders is not None else default_builders(cfg)
            self.log("[2/6] features: " + ", ".join(type(b).__name__ if hasattr(b, "build") else getattr(b, "__name__", "function") for b in builders))
            features, fcols, acols = build_features(prep, target, builders); leak = check_leakage(prep, target, builders)
            self.log(f"      {features.shape[0]:,} rows, {len(fcols)} features | leakage check {'passed' if leak else 'FAILED'}")
        else: self.log("[2/6] features: not needed by the selected models")
        self.log(f"[3/6] split: {len(val_i)} validation origins, {len(test_i)} test origins (gap = horizon)")
        mult = 1 if cfg.score == "step" else (H if cfg.aggregate == "sum" else 1)
        rows, skipped_models = [], {}
        self.log(f"[4/6] running {len(names)} models: {names}")
        for name in names:
            model = reg.get(name)
            try: model.load()
            except Exception as e:
                skipped_models[name] = f"{type(e).__name__}: {str(e)[:80]}"; self.log(f"      {name}: skipped ({skipped_models[name]})"); continue
            n_before = len(rows)
            for split, origins in (("val", val_i), ("test", test_i)):
                for i in origins:
                    hist, inv = _histories(Y, eligible, i, cfg.min_history, self._drop(prep, target, i), cfg)
                    try: preds = {n: inv[n](a) for n, a in model.predict(hist, H, Context(cfg, i, Y, features, eligible, floor, fcols, acols)).items()}
                    except Exception as e: self.log(f"      {name} failed at origin {i}: {type(e).__name__}: {str(e)[:80]}"); preds = {}
                    for n, arr in preds.items():
                        lvl = max(float(Y[n].iloc[max(0, i - 8):i].mean()), floor) * mult; g, st, ya = group_of(n), prep.status[n], Y[n].iloc[i:i + H].values
                        if cfg.score == "step": rows += [(split, name, n, g, st, i, Y.index[i - 1], k + 1, float(arr[k]), float(ya[k]), lvl) for k in range(H)]
                        else: rows.append((split, name, n, g, st, i, Y.index[i - 1], 0, reduce(arr, cfg.aggregate), reduce(ya, cfg.aggregate), lvl))
            model.release(); self.log(f"      {name}: {len(rows) - n_before} forecasts")
        if not rows: raise RuntimeError("no model produced any forecast")
        rec = pd.DataFrame(rows, columns=["split", "model", "series", "group", "status", "i", "origin", "step", "f", "a", "scale"])
        self.log("[5/6] evaluate")
        comp = E.comparable(rec); tot = E.totals(comp, cfg)
        val_tab, test_tab = E.tables(tot)                      # models are chosen and scored on the horizon total
        champ = E.choose(val_tab, cfg, test_tab); ctab = E.champion_table(tot, val_tab, test_tab, champ, cfg, comp)
        if not champ or not len(ctab) or not (ctab.test_cases > 0).any(): raise ValueError("not enough history: no model could be scored on both validation and test origins")
        q = E.error_quantiles(comp, champ, cfg.range_from); cov = E.coverage_walk(comp, champ, H) if cfg.range_from == "all" else E.coverage(comp, champ, q)
        self.log("[6/6] forecast with the champions")
        fc, per = self._final_forecast(prep, target, cfg, champ, ctab, q, features, eligible, floor, mult, fcols, acols)
        skipped = [n for n in Y.columns if n not in eligible]
        return Result(target, cfg, prep, rec, comp, val_tab, test_tab, ctab, champ, cov, fc, per, dict(val=val_i, test=test_i), skipped, skipped_models, leak)

    def _final_forecast(self, prep, target, cfg, champ, ctab, q, features, eligible, floor, mult, fcols=None, acols=None):
        Y, H, L = prep.series[target], cfg.horizon, len(prep.series[target])
        fut = pd.date_range(next_period(Y.index[-1], cfg.freq), periods=H, freq=RULE[cfg.freq]); drop = self._drop(prep, target, L)
        by_model = {}
        for n in eligible:
            m = champ.get(group_of(n))
            if m: by_model.setdefault(m, []).append(n)
        rows, per = [], []
        for m, series in by_model.items():
            model = self.registry.get(m); model.load()
            hist, inv = _histories(Y, series, L, cfg.min_history, drop, cfg)
            preds = {n: inv[n](a) for n, a in model.predict(hist, H, Context(cfg, L, Y, features, eligible, floor, fcols, acols)).items()}; model.release()
            for n, arr in preds.items():
                g = group_of(n); scale = max(float(Y[n].iloc[-8:].mean()), floor) * mult
                if cfg.score == "step":
                    qs = [E.qrange(q, g, k + 1) for k in range(H)]; v = np.asarray(arr, float)
                    lo = np.array([max(v[k] + qs[k][0] * scale, 0.0) for k in range(H)]); hi = np.array([max(v[k] + qs[k][1] * scale, v[k]) for k in range(H)])
                    lo = np.minimum(lo, v)
                    per += [dict(series=n, period_start=fut[k], step=k + 1, value=float(v[k]), low=float(lo[k]), high=float(hi[k]), method=m) for k in range(H)]
                    value, low, high = reduce(v, cfg.aggregate), reduce(lo, cfg.aggregate), reduce(hi, cfg.aggregate)
                else:
                    value = reduce(arr, cfg.aggregate); a, b = E.qrange(q, g, 0); low, high = max(value + a * scale, 0.0), max(value + b * scale, value)
                    per += [dict(series=n, period_start=fut[k], step=k + 1, value=float(arr[k]), low=np.nan, high=np.nan, method=m) for k in range(H)]
                rows.append(dict(series=n, group=g, status=prep.status[n], method=m, first_period=fut[0], horizon=H, value=value, low=low, high=high,
                                 test_wape=ctab.loc[g, "test_wape"], confidence=ctab.loc[g, "confidence"]))
        return pd.DataFrame(rows).set_index("series"), pd.DataFrame(per)
