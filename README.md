# tsforecast

Give it a table, a target column and a time column. It cleans the data, runs every model at many past points, scores them honestly,
picks a champion per group, and forecasts the next periods with a range and a confidence label.

```python
from tsforecast import Forecaster, Config
res = Forecaster(Config(target="sales", time_col="date", keys=["store"])).run(df)
print(res.summary())
res.champions     # best model per group + honest test accuracy
res.forecast      # next horizon, per series, with low / high / method / confidence
```

## Files (one per stage)
| file | stage |
|---|---|
| `config.py` | every setting (only `target` and `time_col` are required) |
| `preprocess.py` | clean, aggregate to periods, profile each series |
| `features.py` | feature builders (plug-ins, none fixed) |
| `treat.py` | automatic treatment of every history before a model sees it: fill gaps, clip outliers, skew transform |
| `split.py` | rolling-origin validation / test split |
| `models/*.py` | one file per model family, found automatically: baselines, ETS/ARIMA/Theta/TSB/Croston, LightGBM, Ensemble, and pretrained Chronos, Chronos2, TimesFM, TiRex2, T0, Timestral, Simons, TimeGPT (needs `NIXTLA_API_KEY`) |
| `evaluate.py` | WAPE, bias, skill, champion choice, 80% ranges |
| `forecast.py` | runs all stages |
| `steps.py` | OPTIONAL use-case extras (fold tiny groups, flag outages) |

## Vocabulary (nothing use-case specific)
- **series**: one stream of values over time. **group**: one way of slicing the data (default: "all" plus one group per key).
- `Config(groups=..., parent_of=..., aux=...)` describe your data; all optional.

## Plug-ins
- **Model**: drop a file in `models/` with a class extending `BaseForecaster` (or `SeriesForecaster`). It is picked up automatically.
- **Features**: `Forecaster(cfg, features=[TimeSeriesFeatures(), MyBuilder()])`; a builder is a class with `build(view)` or a function returning `{name: periods x series DataFrame}`. A leakage check runs automatically.
- **Steps**: `Forecaster(cfg, steps=[...])`, subclasses of `Step` that change the data before modelling.
- **History filter**: `Forecaster(cfg, history_filter=fn)`, `fn(view)` returns the periods to leave out of every model's history.

See `example.py` for a complete use case configured only through these hooks.

## Automatic series treatment (`Config`, see `treat.py`)
Applied to each history at each forecast origin (no leakage); predictions are mapped back.

| setting | options | default |
|---|---|---|
| `fill` | `linear`, `ffill`, `none`: missing values inside a history | `linear` |
| `zeros_missing` | treat exact zeros as missing (levels, prices) | `False` |
| `outliers` / `outlier_k` | `clip` pulls values beyond median +/- k robust deviations back | `none` / 6 |
| `transform` / `skew_limit` | `log1p`, `sqrt`, `none`, or `auto` (log1p when skewness exceeds the limit) | `none` / 1.5 |

`result.prepared.profile` reports `skew`, `outliers` and `missing_share` per series, so the choice can be informed. Feature-based models (LightGBM) read the untreated feature table.

## Scoring
`score="aggregate"` scores the horizon total as one number. `score="step"` scores every period ahead separately and gives a range per period. `origin_step` spaces the backtest origins (7 for daily data).

## Reusing champions
`Forecaster(cfg).run(df)` compares all models. `Forecaster(cfg).forecast_with(df, champions, q, wape)` forecasts with champions chosen earlier, without re-running the comparison (cheap enough for a nightly job).
