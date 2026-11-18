"""Stage 5 - how good is each model, which one wins, and how wide should the range be.

Records are one row per (split, model, series, origin): f = forecast, a = actual (both the horizon aggregate), scale = the series' recent level.

Metrics
  WAPE      sum |f - a| / sum a           lower is better; 0.2 = 20% off; above 1 = no better than guessing; accuracy ~ 1 - WAPE
  bias      (sum f - sum a) / sum a       + over-forecast, - under-forecast
  skill     1 - WAPE(model) / WAPE(baseline)
  MAE / RMSE   average / root-mean-square error in the target's own unit (RMSE punishes big misses); nrmse = RMSE / mean actual; smape = symmetric % error
  step_wape    WAPE when every period ahead is scored on its own (stricter than the horizon total, which is what WAPE above uses)
  coverage  share of actuals inside the 80% range (target about 80%)"""
import numpy as np, pandas as pd


def wape(df) -> float: return float((df.f - df.a).abs().sum() / max(df.a.sum(), 1e-9))
def bias(df) -> float: return float((df.f.sum() - df.a.sum()) / max(df.a.sum(), 1e-9))
def confidence(w, cfg) -> str: return "high" if w <= cfg.conf_high else "medium" if w <= cfg.conf_med else "low"


def comparable(rec: pd.DataFrame) -> pd.DataFrame:
    """Keep only the (series, origin) pairs where EVERY model produced a forecast, so models are compared on the same cases."""
    cnt = rec.groupby("model").size(); rec = rec[rec.model.isin(cnt[cnt >= 0.5 * cnt.max()].index)]       # a model that can forecast only a few cases (too little history for it) must not empty the comparison
    n = rec.model.nunique()
    k = ["split", "series", "i"]
    have = rec.groupby(k).model.nunique().rename("h").reset_index()
    return rec.merge(have[have.h == n][k], on=k)


def totals(comp: pd.DataFrame, cfg) -> pd.DataFrame:
    """Per-period records (score="step") -> one record per (model, series, origin) holding the HORIZON TOTAL (or mean / last, as Config.aggregate says).
    Models are chosen and scored on this (errors in different periods partly cancel, which is how a 4-week total is judged); ranges stay per period."""
    if cfg.score != "step": return comp
    k = ["split", "model", "series", "group", "status", "i", "origin"]; how = cfg.aggregate
    out = comp.groupby(k, sort=False).agg(f=("f", how), a=("a", how), scale=("scale", "first")).reset_index()
    out["scale"] *= cfg.horizon if how == "sum" else 1; out["step"] = 0
    return out


def smape(df) -> float:
    d = (df.f.abs() + df.a.abs()).replace(0, np.nan); return float((2 * (df.f - df.a).abs() / d).fillna(0).mean())
def mae(df) -> float: return float((df.f - df.a).abs().mean())
def rmse(df) -> float: return float(np.sqrt(((df.f - df.a) ** 2).mean()))


def tables(comp: pd.DataFrame):
    """(validation, test) WAPE tables: models x groups."""
    out = []
    for sp in ("val", "test"):
        g = comp[comp.split == sp].assign(ae=lambda x: (x.f - x.a).abs()).groupby(["model", "group"]).agg(ae=("ae", "sum"), a=("a", "sum"))
        out.append((g.ae / g.a.clip(lower=1e-9)).unstack("group"))
    return tuple(out)


def choose(val_tab: pd.DataFrame, cfg, test_tab: pd.DataFrame = None) -> dict:
    """Champion per group: the best validation WAPE, but the baseline is replaced only by a model at least `switch_margin` better (small samples make smaller gaps noise).
    With `cfg.confirm_on_test` (and the test table given) the model must also score no worse than the baseline on the test origins."""
    champ = {}
    for g in val_tab.columns:
        s = val_tab[g].dropna()
        if s.empty: continue
        best = s.idxmin()
        pick = best if (cfg.baseline not in s or s[best] <= s[cfg.baseline] * (1 - cfg.switch_margin)) else cfg.baseline
        if getattr(cfg, "confirm_on_test", False) and test_tab is not None and pick != cfg.baseline and g in test_tab.columns:
            t = test_tab[g].dropna()
            if cfg.baseline in t.index and (pick not in t.index or t[pick] > t[cfg.baseline]): pick = cfg.baseline
        champ[g] = pick
    return champ


def champion_table(comp, val_tab, test_tab, champ, cfg, comp_step=None) -> pd.DataFrame:
    t, rows = comp[comp.split == "test"], []
    ts = (comp_step if comp_step is not None else comp); ts = ts[ts.split == "test"]
    for g, m in champ.items():
        sel, base = t[(t.group == g) & (t.model == m)], t[(t.group == g) & (t.model == cfg.baseline)]
        w = wape(sel) if len(sel) else np.nan; wb = wape(base) if len(base) else np.nan
        ss = ts[(ts.group == g) & (ts.model == m)]; ok = len(sel) > 0; mean_a = float(sel.a.mean()) if ok else np.nan
        rows.append(dict(group=g, champion=m, val_wape=round(val_tab.loc[m, g], 3), test_wape=round(w, 3), accuracy=round(max(0.0, 1 - w), 3), bias=round(bias(sel), 3) if ok else np.nan,
                         mae=round(mae(sel), 2) if ok else np.nan, rmse=round(rmse(sel), 2) if ok else np.nan, nrmse=round(rmse(sel) / mean_a, 3) if ok and mean_a > 0 else np.nan,
                         smape=round(smape(sel), 3) if ok else np.nan, step_wape=round(wape(ss), 3) if len(ss) else np.nan,
                         baseline_test_wape=round(wb, 3), skill_vs_baseline=round(1 - w / wb, 3) if wb else np.nan, best_on_test=test_tab[g].idxmin(),
                         best_on_test_wape=round(test_tab[g].min(), 3), test_cases=len(sel), confidence=confidence(w, cfg)))
    cols = ["group", "champion", "val_wape", "test_wape", "accuracy", "bias", "mae", "rmse", "nrmse", "smape", "step_wape", "baseline_test_wape", "skill_vs_baseline", "best_on_test", "best_on_test_wape", "test_cases", "confidence"]
    return pd.DataFrame(rows, columns=cols).set_index("group")


def error_quantiles(comp, champ, use="val") -> dict:
    """{group: {step: (q10, q90), "pool": (q10, q90)}} of the champion's scaled errors (a - f) / scale on validation (validation + test when validation is small).
    use="all": every backtest origin (validation + test) is used, which is what to do when the series have rare large spikes that a short validation window may not contain.
    step 0 = the horizon aggregate; steps 1.. = each period ahead (score="step"). A step with fewer than 8 cases uses the pooled quantiles."""
    q = {}
    for g, m in champ.items():
        sel = comp[(comp.group == g) & (comp.model == m)]
        v = sel if use == "all" else (sel[sel.split == "val"] if (sel.split == "val").sum() >= 20 else sel)
        e = ((v.a - v.f) / v.scale)
        q[g] = {"pool": tuple(np.quantile(e.values, [.1, .9])) if len(v) else (-1.0, 1.0)}
        for st, ee in e.groupby(v.step): 
            if len(ee) >= 8: q[g][int(st)] = tuple(np.quantile(ee.values, [.1, .9]))
    return q


def qrange(q, g, step=0): return q[g].get(int(step), q[g]["pool"])


def coverage(comp, champ, q) -> pd.DataFrame:
    """Does the 80% range hold out of sample? Range = forecast + q * scale, using the validation quantiles, scored on test."""
    rows, t = [], comp[comp.split == "test"]
    for g, m in champ.items():
        sel = t[(t.group == g) & (t.model == m)]
        if sel.empty: continue
        qq = np.array([qrange(q, g, st) for st in sel.step]); lo, hi = np.clip(sel.f + qq[:, 0] * sel.scale, 0, None), sel.f + qq[:, 1] * sel.scale
        iscore = (hi - lo) + 10 * (lo - sel.a).clip(lower=0) + 10 * (sel.a - hi).clip(lower=0)
        rows.append(dict(group=g, coverage_80=round(float(((sel.a >= lo) & (sel.a <= hi)).mean()), 3), width_vs_level=round(float(((hi - lo) / sel.scale).mean()), 2),
                         interval_score_vs_level=round(float((iscore / sel.scale).mean()), 2), n=len(sel)))
    return pd.DataFrame(rows, columns=["group", "coverage_80", "width_vs_level", "interval_score_vs_level", "n"]).set_index("group")


def coverage_walk(comp, champ, horizon, min_cases=20) -> pd.DataFrame:
    """Honest coverage for ranges calibrated on ALL origins: every test origin is covered by quantiles built only from errors whose target window had already
    ended by then (origins at least `horizon` periods earlier), so nothing from the future leaks in. Same columns as coverage()."""
    rows = []
    for g, m in champ.items():
        sel = comp[(comp.group == g) & (comp.model == m)]; t = sel[sel.split == "test"]
        if t.empty: continue
        e = (sel.a - sel.f) / sel.scale; allq = np.quantile(e.values, [.1, .9]); parts = []
        for i in sorted(t.i.unique()):
            ee = e[sel.i + horizon <= i].values; q = np.quantile(ee, [.1, .9]) if len(ee) >= min_cases else allq
            ti = t[t.i == i]; lo, hi = np.clip(ti.f + q[0] * ti.scale, 0, None), ti.f + q[1] * ti.scale
            parts.append(pd.DataFrame({"in": ((ti.a >= lo) & (ti.a <= hi)).values, "w": ((hi - lo) / ti.scale).values}))
        r = pd.concat(parts); rows.append(dict(group=g, coverage_80=round(float(r["in"].mean()), 3), width_vs_level=round(float(r.w.mean()), 2), interval_score_vs_level=np.nan, n=len(r)))
    return pd.DataFrame(rows, columns=["group", "coverage_80", "width_vs_level", "interval_score_vs_level", "n"]).set_index("group")
