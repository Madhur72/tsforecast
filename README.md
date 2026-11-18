# tsforecast

A plug-in time-series forecasting module. Give it a table of rows with a timestamp and a number to forecast. It cleans the data, rolls it up into periods, runs **every model it knows** at many past points in time, scores them honestly, picks a **champion per group**, and forecasts the next periods with an **80% range** and a **confidence label**.

Nothing in the core is specific to one use case. Token usage, GitHub Copilot seats, per-agent KPIs and agent platforms all use the same code, configured only through `Config` and a few optional plug-ins.

- [Install](#1-install)
- [Quick start](#2-quick-start)
- [Your data](#3-your-data)
- [How a run works](#4-how-a-run-works)
- [Configuration reference](#5-configuration-reference)
- [Reading the results](#6-reading-the-results)
- [Metrics: how accuracy is measured](#7-metrics-how-accuracy-is-measured)
- [Models](#8-models)
- [Preprocessing and series treatment](#9-preprocessing-and-series-treatment)
- [Plug-ins: steps, history filters, features, models](#10-plug-ins)
- [Reusing champions (nightly job)](#11-reusing-champions-the-nightly-job)
- [Recipes](#12-recipes)
- [Troubleshooting](#13-troubleshooting)
- [Files](#14-files)
- [Known limits](#15-known-limits)

---

## 1. Install

```bash
pip install -r requirements.txt
```

`requirements.txt` has the light core: `numpy`, `pandas`, `pyyaml`, `statsforecast` (ETS, ARIMA, Theta, TSB, Croston), `lightgbm`. The pretrained foundation models are optional and each one is loaded only when it is used:

| Model | Python module it imports (install the package that provides it) | Weights |
|---|---|---|
| Chronos, Chronos2 | `torch`, `chronos-forecasting` | Amazon Chronos |
| TimesFM | `timesfm`, `torch` | `google/timesfm-2.5-200m-pytorch` |
| TiRex2 | `tirex2`, `torch` | `NX-AI/TiRex-2` |
| Timestral | `timestral` | `mistralai/Timestral-209M` |
| T0 | `t0` | `theforecastingcompany/t0-gamma` |
| Simons178, Simons297 | `simons` | `abel-lab/Simons-178M`, `abel-lab/Simons-297M` |
| TimeGPT | `nixtla`; environment variable `NIXTLA_API_KEY` | external service |

A model whose package is missing is **skipped and reported** (`result.skipped_models`), not a crash. Set `foundation_models=False` to skip every heavy model (quick tests, no downloads). All heavy models here run on CPU by default.

## 2. Quick start

```python
import pandas as pd
from tsforecast import Forecaster, Config

df = pd.read_csv("sales.csv")                       # one row per sale: date, store, amount

cfg = Config(target="amount", time_col="date", keys=["store"],
             freq="W", horizon=4, foundation_models=False)
res = Forecaster(cfg).run(df)

print(res.summary())     # champion per group, honest accuracy, range coverage
res.champions            # best model per group + validation / test accuracy
res.forecast             # next 4 weeks per series, with low / high / method / confidence
res.forecast_periods     # the same, week by week
res.save("out/")         # every table as CSV
```

Only `target` and `time_col` are required. `keys` names the columns that identify a series (here one series per store, plus the total of all stores).

A complete real use case, expressed only through config and optional plug-ins, is in `example.py`.

## 3. Your data

A **long** table, one row per event or observation, in a pandas DataFrame:

| Need | Notes |
|---|---|
| a **time column** | any parsable timestamp. It is converted to UTC, then to the day boundary of `tz` |
| a **target column** | numeric, >= 0. Rows with a null or negative target are dropped and counted |
| optional **key columns** | what identifies a series (store, agent, organisation...). Missing keys become `UNKNOWN` |
| optional **id column** | `id_col`: duplicate ids are dropped |

Rows inside a period are combined with `row_agg`: `"sum"` for amounts (tokens, sales) or `"mean"` for levels (price, latency). A day or period with no rows is **0** for an amount and **the last value** for a level.

Already have one row per period and series? That works too: pass `freq` equal to your period and each row becomes one value.

## 4. How a run works

```
rows ──► 1 preprocess ──► 2 features ──► 3 split ──► 4 run every model ──► 5 evaluate ──► 6 forecast
```

1. **Preprocess.** Clean rows, build one series per (group, key), roll up to periods of `freq`, keep only complete periods, profile each series (history, zero share, pattern, status).
2. **Features.** Only built if a selected model asks for them (LightGBM). You choose the builders; a leakage check runs automatically.
3. **Split.** Time-ordered, never random. An *origin* is a point in time you forecast from, scored on the next `horizon` periods.

```
history ........... | validation origins | gap (= horizon) | test origins
                      (pick the champion)                    (the honest score, used once)
```

4. **Run every model** at every origin. Each model sees only the history before the origin. The histories are treated first (gaps filled, outliers, transform: section 9).
5. **Evaluate.** WAPE per model and group on validation and on test. The champion is chosen on validation, with the baseline protected (below). The 80% range comes from the champion's validation errors.
6. **Forecast** the next `horizon` periods with the champion of each group.

### Champion choice

For each group the model with the lowest **validation** WAPE wins, **but** the baseline (`Median4` by default) is replaced only if the winner is at least `switch_margin` (10%) better. With few origins a smaller gap is noise.

With `confirm_on_test=True` the winner must also be **no worse than the baseline on the test origins**, otherwise the baseline stays champion. Use it when you have few origins: it stops a model that only looked good on a handful of validation points. The cost: the champion's test score is then mildly optimistic, because test took part in the choice. Leave it `False` when you have plenty of origins.

### Groups, series, hierarchy

- A **series** is one stream of values over time (one store, one agent).
- A **group** is one way of slicing the rows into series. By default the groups are `"all"` (every row together) plus one group per key. Champions are chosen **per group**, so the total can use a different model from the individual stores.
- `groups={"org": ["org_id"], "team": ["org_id", "team_id"]}` defines your own groups. Series are named `"<group>|<value>"`, for example `team|org1/teamA`, and the overall one is `all|ALL`.
- `parent_of={"team": "org"}` declares a hierarchy for steps and features that want it.
- `aux={"users": ("user_id", "nunique"), "rows": (None, "size")}` builds extra per-period series from other columns (`nunique`, `sum`, `mean`, `size`), usable as features.

## 5. Configuration reference

Every setting is a field of `Config`. Load from YAML with `Config.from_yaml("cfg.yaml")`.

### The data

| setting | default | meaning |
|---|---|---|
| `target` | required | column to forecast. A list runs several targets one after the other on the same series |
| `time_col` | required | row timestamp |
| `keys` | `[]` | columns that identify a series. Empty = a single series |
| `groups` | `None` | explicit `{group: [columns]}`. Default `all` plus one per key |
| `parent_of` | `None` | hierarchy `{group: parent group}` |
| `aux` | `None` | extra per-period numbers from other columns |
| `id_col` | `None` | row id; duplicates dropped |
| `tz` | `"UTC"` | time zone that decides where a day starts (for example `"Asia/Kolkata"`) |
| `row_agg` | `"sum"` | how rows combine inside a period: `"sum"` or `"mean"` |

### The forecast

| setting | default | meaning |
|---|---|---|
| `freq` | `"W"` | period: `"D"` day, `"W"` week (Monday to Sunday), `"M"` month |
| `horizon` | `4` | periods ahead |
| `aggregate` | `"sum"` | how the horizon periods become the one number that is scored: `"sum"`, `"mean"`, `"last"` |
| `score` | `"aggregate"` | `"aggregate"` scores the horizon total as one number. `"step"` scores every period ahead separately and gives a range per period |
| `origin_step` | `1` | periods between backtest origins. Use 7 for daily data so origins are not near-duplicates |
| `min_history` | `8` | a series needs this many periods (from its first non-zero value) to be forecast |
| `season` | `"auto"` | `"auto"` detects the season length, `"manual"` uses `season_length`, `"off"` uses the default (7 for days, otherwise 1) |
| `season_length` | `None` | cycle length in periods for the statistical models |

### Series treatment

| setting | default | meaning |
|---|---|---|
| `fill` | `"linear"` | missing values inside a history: `linear`, `ffill`, `none` |
| `zeros_missing` | `False` | treat exact zeros as missing (levels or prices that cannot really be 0) |
| `outliers` | `"none"` | `"clip"` pulls values beyond median ± `outlier_k` robust deviations (MAD) back |
| `outlier_k` | `6.0` | |
| `transform` | `"none"` | `log1p`, `sqrt`, `none`, or `auto` (log1p when skewness exceeds `skew_limit`). Predictions are mapped back |
| `skew_limit` | `1.5` | |

### Evaluation

| setting | default | meaning |
|---|---|---|
| `n_val` | `8` | validation origins (choose the champion) |
| `n_test` | `12` | test origins (the honest score) |
| `models` | `"all"` | `"all"` or a list of model names |
| `exclude` | `[]` | model names to leave out |
| `foundation_models` | `True` | `False` skips every heavy pretrained model |
| `baseline` | `"Median4"` | a model replaces this one only if clearly better |
| `switch_margin` | `0.10` | "clearly better" = at least this much lower validation error |
| `confirm_on_test` | `False` | the winner must also be no worse than the baseline on test (see section 4) |
| `range_from` | `"val"` | which errors size the 80% range: `"val"`, or `"all"` (validation + test; use for spiky series, then coverage is checked walk-forward) |
| `conf_high` / `conf_med` | `0.20` / `0.45` | WAPE limits of the confidence labels |
| `verbose` | `True` | print progress |

History needed: roughly `min_history + horizon + n_val + horizon (gap) + n_test` periods. With too little history `run()` raises `ValueError("not enough history ...")`; lower `n_val`, `n_test` or `min_history`.

## 6. Reading the results

`Forecaster(cfg).run(df)` returns a `Result` (or `{target: Result}` when you gave several targets).

| attribute | what it is |
|---|---|
| `summary()` | a printable report |
| `champions` | per group: `champion`, `val_wape`, `test_wape`, `accuracy`, `bias`, `mae`, `rmse`, `nrmse`, `smape`, `step_wape`, `baseline_test_wape`, `skill_vs_baseline`, `best_on_test`, `best_on_test_wape`, `test_cases`, `confidence` |
| `champion_map` | `{group: model name}` |
| `leaderboard` | WAPE of every model, per group, validation and test |
| `val_table`, `test_table` | the same as models x groups tables |
| `forecast` | the next horizon per series: `series`, `group`, `status`, `method`, `first_period`, `horizon`, `value`, `low`, `high`, `test_wape`, `confidence` |
| `forecast_periods` | the same period by period: `series`, `period_start`, `step`, `value`, `low`, `high`, `method` |
| `ranges` | does the 80% range hold out of sample? `coverage_80` should be about 0.80 |
| `records`, `comparable` | every forecast of every model at every origin (`f` forecast, `a` actual, `scale` the series' recent level) |
| `prepared.profile` | per series: first period, history, active periods, `zero_share`, ADI, CV², `pattern`, `status` |
| `skipped_series`, `skipped_models` | what could not be forecast, and why |
| `leakage_ok` | result of the feature leakage check |
| `save(folder)` | writes `champions`, `ranges`, `forecast`, `forecast_periods`, `leaderboard`, `records`, `series_profile` as CSV |

**Series status** (in `forecast.status` and `profile.status`):

| status | meaning |
|---|---|
| `ok` | enough history and activity |
| `sparse` | more than 60% zero periods; forecast, but treat with care |
| `short_history` | fewer than `min_history` periods since the first activity; not forecast |
| `no_data` | never had a value; not forecast |

**Demand pattern** in the profile (Syntetos-Boylan): `smooth` and `erratic` (regular demand) or `intermittent` and `lumpy` (many zero periods). Intermittent and lumpy series are where the `Hurdle`, `TSB` and `Croston` models matter.

**The confidence label** comes from the champion's **test** WAPE: `high` up to 0.20, `medium` up to 0.45, `low` above. When it is not `high`, show the **range**, not only the point.

## 7. Metrics: how accuracy is measured

All metrics are computed on the **horizon aggregate** (`aggregate`) unless `score="step"`.

| metric | definition | read it as |
|---|---|---|
| **WAPE** | sum of absolute errors / sum of actuals, pooled over origins and series | 0.2 = 20% off. About 1 - WAPE is "accuracy". **1.0 means no better than forecasting zero** (a zero forecast scores exactly 1.0). Above 1 is worse than zero |
| **bias** | (sum forecast - sum actual) / sum actual | negative = under-forecasting |
| **skill** | 1 - WAPE(model) / WAPE(baseline) | below 0 = worse than the baseline |
| **mae, rmse, nrmse, smape** | the usual definitions; `nrmse` = rmse / mean actual | |
| **step_wape** | WAPE when every period ahead is scored separately | the harder number for week-by-week forecasts |

The **80% range** of a forecast is `forecast + (q10, q90) × scale`, where the quantiles are those of the champion's scaled errors `(actual - forecast) / scale` and `scale` is the series' recent level. `ranges.coverage_80` tells you how often it really held on test.

Why WAPE and not MAPE: demand has zeros, and WAPE weights large periods more, so it is stable on bursty series. Always compare with the baseline: on very lumpy series (a few big bursts) every model can score near 1.0, and the honest message is that the volume is not forecastable from its own history.

## 8. Models

Found automatically in `models/`. List them with `ModelRegistry().names()`.

| Model | Family | Notes |
|---|---|---|
| `Naive` | baseline | repeats the last value |
| `Median4` | baseline | median of the last 4 periods. The default baseline; robust to one spike |
| `Mean8` | baseline | mean of the last 8 periods |
| `Hurdle` | baseline | (share of the last 8 periods that were active) × (typical size of an active period). For bursty series |
| `SeasonalNaive` | baseline | repeats the last season (needs a season length > 1; else Naive) |
| `ETS`, `ARIMA`, `Theta` | statistics (`statsforecast`) | use the detected season length |
| `TSB`, `Croston` | statistics | for intermittent demand |
| `LightGBM` | global machine learning | needs the feature table (`requires_features`) |
| `Ensemble` | combination | median of ETS, ARIMA, Theta, LightGBM (if features exist) and the mean of the last 4 periods |
| `Chronos`, `Chronos2` | pretrained | |
| `TimesFM`, `TiRex2`, `Timestral`, `T0`, `Simons178`, `Simons297` | pretrained | |
| `TimeGPT` | external service | needs `NIXTLA_API_KEY`; off when `foundation_models=False` |

The baselines always run: they are the benchmark every other model has to beat.

Pick models with `models=[...]` and `exclude=[...]`. Each model forecasts all series in one call (batch-first), so global and pretrained models fit naturally.

## 9. Preprocessing and series treatment

### What always happens (in `preprocess.py`)

- Rows with a bad timestamp, a null or negative target, or a duplicate id are dropped, and **every fix is counted** (`bad_timestamps`, `null_target`, `negative_target`, `duplicate_ids`, `missing_keys_filled`), so nothing disappears silently.
- Timestamps go to UTC, then to the day boundary of `tz`.
- Days roll up to `freq` periods. **Only complete periods are kept**: an unfinished last week would look like a collapse.
- Missing amounts are 0; missing levels keep the last value.
- Each series is profiled (history, zero share, ADI, CV², pattern, status). Short or empty series are not forecast.

### What you switch on (in `treat.py`)

Applied to each history at each forecast origin (so nothing leaks from the future); predictions are mapped back.

| setting | when to use it |
|---|---|
| `fill="linear"` (default) | gaps inside a history. `ffill` for levels, `none` to leave them |
| `zeros_missing=True` | a level or price that cannot really be 0 (latency, price), so a zero means "no reading" |
| `outliers="clip"` | one-off spikes you do not want a model to learn |
| `transform="log1p"` / `"auto"` | right-skewed values. `auto` uses log1p when skewness exceeds `skew_limit` |

`treat.describe(frame, cfg)` reports skewness, outlier count and missing share per series so the choice can be informed. Feature-based models (LightGBM) read the untreated feature table.

### Season length (automatic)

The statistical models need the length of the seasonal cycle in periods. With `season="auto"` (default) it is detected from the data (`season.py`):

- candidate cycle lengths: day 7, 14, 30; week 4, 13, 26, 52; month 12, 6, 4, 3. A candidate needs at least 3 full cycles of history;
- for every usable series: take log1p, difference once (removes level and trend), measure the autocorrelation at each candidate;
- the pooled score of a length is the **median over series**, so a few lumpy series cannot create a season;
- a length counts only if its score is above `max(0.25, 2 / sqrt(n))`; of those that count, the **smallest within 10% of the best** is used (so 14 is not picked when the real cycle is 7);
- if none counts, the default applies: 7 for daily data, 1 (no season) otherwise.

It gives one length per target for the whole run, because the statistical models take a single setting. Force a value with `Config(season="manual", season_length=12)`. Detection looks at the whole history; it decides structure, not forecasts. The result is available as `forecaster.season_report` (the score of every candidate).

## 10. Plug-ins

The core never needs editing. There are four hooks.

### Add a model

Drop a file in `models/` with a class that inherits `BaseForecaster` (batch of series) or `SeriesForecaster` (one series at a time) and has a `name`. It is picked up automatically.

```python
# models/my_model.py
import numpy as np
from tsforecast.base import SeriesForecaster

class Mean3(SeriesForecaster):
    """Repeat the mean of the last 3 periods."""
    name = "Mean3"
    def forecast(self, history, horizon, cfg):
        return np.full(horizon, history[-3:].mean())
```

A model that needs a heavy library imports it inside `load()`, not at the top of the file, so the registry can always list it. Set `heavy = True` to have it skipped when `foundation_models=False`, `requires_features = True` to receive the feature table in `ctx.features`. Models outside the package: `ModelRegistry().add_folder("my_private_models/")`, then `Forecaster(cfg, registry=reg)`.

### Steps (change the data before modelling)

`Forecaster(cfg, steps=[...])`. A step is a subclass of `Step` with `before(df, cfg)` (change the rows), `after(prep)` (add information) and `cut(prep, i)` (information as known at origin `i`). Included in `steps.py` as examples, and optional:

| step | does |
|---|---|
| `FoldSmallGroups(min_active)` | folds tiny groups into an `OTHER` series so thin series do not clutter the results |
| `FlagOutages(weekdays)` | flags multi-day stretches of zero usage (a feed outage, not demand) |

### History filter (leave periods out of every model's history)

`Forecaster(cfg, history_filter=fn)`, where `fn(view)` returns the periods to drop.

| filter | does |
|---|---|
| `drop_outages()` | drops the stretches `FlagOutages` found |
| `drop_ranges([("2026-11-02", "2026-11-15")])` | drops periods mostly inside **known closures** (company leave, shutdowns) |
| `combine(f1, f2)` | drops a period if any filter says so |

### Features

`Forecaster(cfg, features=[TimeSeriesFeatures(), MyBuilder()])`. Built only if a selected model needs them. A builder is a class with `build(view)` (or a function) returning `{name: periods x series DataFrame}`, using **only what is in the view** (the data known at the origin). A leakage check runs automatically and `result.leakage_ok` reports it.

Included builders: `TimeSeriesFeatures` (lags, rolling statistics, trend, sparsity, spikes, month), `AuxFeatures` (the series you declared in `aux`), `HierarchyFeatures` (the parent's level and this series' share; needs `parent_of`). Set `amount_cols` on a builder for columns that are amounts, so models can scale them by the series' level.

## 11. Reusing champions (the nightly job)

`run()` compares every model: slow. A nightly job should forecast with champions chosen earlier and refresh them weekly.

```python
from tsforecast import Forecaster, Config
from tsforecast import evaluate as E

# weekly: select
res = Forecaster(cfg).run(df)
champions = res.champion_map                                   # {group: model}
q = E.error_quantiles(res.comparable, champions, cfg.range_from)   # {group: {step: (q10, q90), "pool": ...}}
wape = res.champions.test_wape.to_dict()                       # for the confidence label
# store champions, q and wape (the project stores them in database tables)

# nightly: forecast only
fc, per, prep = Forecaster(cfg).forecast_with(new_df, champions, q, wape)
```

## 12. Recipes

**Daily data with a weekly cycle**
```python
Config(target="requests", time_col="ts", freq="D", horizon=14, origin_step=7, n_val=8, n_test=8)
```

**Bursty or intermittent volume** (a few big periods, many zeros): score on the horizon total, size the ranges on all origins, and expect low confidence.
```python
Config(target="tokens", time_col="ts", keys=["agent"], freq="W", horizon=4,
       range_from="all", transform="auto", outliers="clip")
```

**A level, not an amount** (latency, price): average, not sum; zeros are gaps.
```python
Config(target="latency", time_col="ts", keys=["agent"], row_agg="mean", aggregate="mean", zeros_missing=True, fill="ffill")
```

**Very little history**: reduce the origins, protect the baseline, name your baseline.
```python
Config(target="runs", time_col="ts", keys=["agent"], freq="W", min_history=6, n_val=4, n_test=4,
       confirm_on_test=True, baseline="Mean8")
```

**Hierarchy** (organisation, team, agent): several groups, one run.
```python
Config(target="tokens", time_col="ts", id_col="id",
       groups={"org": ["org_id"], "team": ["org_id", "team_id"], "agent": ["org_id", "agent"]},
       parent_of={"team": "org", "agent": "org"})
```

**Several targets** on the same series: `target=["tokens", "sessions"]` returns `{target: Result}`.

**Quick test without downloads**: `foundation_models=False`, or `models=["Median4", "ETS", "ARIMA"]`.

**Company closure** in the history:
```python
from tsforecast.steps import drop_ranges
Forecaster(cfg, history_filter=drop_ranges([("2026-11-02", "2026-11-15")]))
```

## 13. Troubleshooting

| message or symptom | cause and fix |
|---|---|
| `Config needs target and time_col` | set both |
| `not enough history: ... cannot give test origins` | lower `n_val`, `n_test` or `min_history`; or use a shorter `freq` |
| `models skipped: {...}` | a model's package is not installed (see section 1); install it or ignore it |
| every WAPE is about 1.0 or above | the series is too bursty to forecast from its own history. Compare with the baseline (`skill_vs_baseline`), show the range, and consider forecasting a coarser total instead |
| the champion is `Median4` almost everywhere | no model beat the baseline by `switch_margin`; that is a valid, honest result |
| a champion that loses to the baseline on test | validation was too small to choose well. Use `confirm_on_test=True`, or more origins |
| ranges too narrow (coverage well below 0.80) | use `range_from="all"` |
| a series is missing from `forecast` | its `status` is `short_history` or `no_data`; see `prepared.profile` |
| the last period looks too low | it is not: only complete periods are used. A low period inside the history is probably a closure or an outage: use `drop_ranges` or `FlagOutages` |
| forecast differs between runs | foundation models and `LightGBM` are not seeded identically on every platform; the baselines and statistical models are deterministic |

## 14. Files

One file per stage, so any stage can be read and replaced on its own.

| file | stage |
|---|---|
| `config.py` | every setting (only `target` and `time_col` are required) |
| `preprocess.py` | clean, aggregate to periods, profile each series |
| `season.py` | season length detection |
| `treat.py` | automatic treatment of every history before a model sees it: fill gaps, clip outliers, skew transform |
| `features.py` | feature builders (plug-ins, none fixed) and the leakage check |
| `split.py` | rolling-origin validation / test split |
| `base.py` | the model contract and the registry that finds models |
| `models/*.py` | one file per model family, found automatically |
| `evaluate.py` | WAPE and the other metrics, skill, champion choice, 80% ranges, coverage |
| `forecast.py` | `Forecaster`: runs all stages; `Result` |
| `steps.py` | optional use-case extras: fold tiny groups, flag outages, history filters |
| `example.py` | a complete use case configured only through these hooks |

## 15. Known limits

- It forecasts each series from its own past (plus features you provide). It does not know about launches, holidays or price changes unless you give them as steps, history filters or features.
- There is no automatic detection of level shifts, holidays, or the best fill/outlier setting. The season length **is** detected.
- One season length per target per run, shared by all series.
- Accuracy numbers are only as good as the number of backtest origins. With a handful of origins treat them as indicative, and prefer `confirm_on_test=True`.
- Weekly, step-by-step forecasts are less accurate than the horizon total.
- Foundation models are slow on CPU on large panels. Use `foundation_models=False` while developing.
