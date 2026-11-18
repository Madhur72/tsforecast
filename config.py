"""Everything you can tell the module. Only `target` and `time_col` are required (plus `keys` if you have several series).

Vocabulary: a *series* is one stream of values over time (one store, one machine, one customer...). A *group* is one way of slicing the data into series
(for example "store", or "region"); the group "all" is every row together. Nothing here assumes what the data means."""
from dataclasses import dataclass, field, asdict, replace
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class Config:
    # ---------------------------------------------------------------- the data
    target: Union[str, List[str], None] = None      # the column(s) to forecast. Several targets are run one after the other on the same series
    time_col: Optional[str] = None                  # the row timestamp
    keys: List[str] = field(default_factory=list)   # columns that identify a series, e.g. ["store"]. Empty = a single series
    groups: Optional[Dict[str, List[str]]] = None   # explicit {group name: [columns]}. Default: "all" plus one group per key
    parent_of: Optional[Dict[str, Optional[str]]] = None   # optional hierarchy between groups, {group: parent group}. Used only by steps / features that ask for it
    aux: Optional[Dict[str, Tuple[Optional[str], str]]] = None   # extra numbers per series and period, built from other columns: {"name": ("column", "nunique"|"sum"|"mean"), "rows": (None, "size")}
    id_col: Optional[str] = None                    # row id; duplicate rows are dropped
    tz: str = "UTC"                                 # time zone that decides where a day starts
    row_agg: str = "sum"                            # how rows combine inside a period: "sum" (amounts) or "mean" (levels such as prices)

    # ---------------------------------------------------------------- the forecast
    freq: str = "W"                                 # period length: "D" day, "W" week (Monday to Sunday), "M" month
    horizon: int = 4                                # periods ahead
    aggregate: str = "sum"                          # how the horizon periods become the number that is scored: "sum", "mean" or "last"
    season_length: Optional[int] = None             # cycle length in periods for the statistical models. With season="auto" it is detected from the data (season.py); with "manual" this value is used; with "off" the default (7 for days, otherwise 1)
    season: str = "auto"                            # "auto" detect the season length | "manual" use season_length as given | "off" no detection
    score: str = "aggregate"                        # "aggregate": score the horizon total / mean as one number. "step": score every period ahead separately (needed for per-period ranges)
    origin_step: int = 1                            # periods between backtest origins (for daily data 7 avoids near-duplicate origins)
    min_history: int = 8                            # a series needs this many periods of history to be forecast

    # ---------------------------------------------------------------- series treatment (treat.py), applied to every history before a model sees it
    fill: str = "linear"                            # missing values: "linear" | "ffill" | "none"
    zeros_missing: bool = False                     # treat exact zeros as missing (for levels or prices that cannot really be 0)
    outliers: str = "none"                          # "clip" pulls extreme values back to median +/- outlier_k robust deviations
    outlier_k: float = 6.0
    transform: str = "none"                         # "log1p" | "sqrt" | "auto" (log1p when skewness > skew_limit) | "none"; predictions are mapped back
    skew_limit: float = 1.5

    # ---------------------------------------------------------------- evaluation
    n_val: int = 8                                  # validation origins (these choose the champion)
    n_test: int = 12                                # test origins (the honest score)
    models: Union[str, List[str]] = "all"           # "all" or a list of model names
    exclude: List[str] = field(default_factory=list)
    foundation_models: bool = True                  # allow heavy pretrained models (Chronos...). False = skip them
    range_from: str = "val"                         # which backtest errors size the 80% ranges: "val" (validation only) or "all" (validation + test; use for spiky series). With "all", coverage is checked walk-forward
    baseline: str = "Median4"                       # a model replaces this one only if it is clearly better
    switch_margin: float = 0.10                     # "clearly better" = at least 10% lower validation error
    confirm_on_test: bool = False                   # True: a model that beats the baseline on validation must also be no worse than the baseline on the test origins, or the baseline stays champion.
                                                    # Guards against a choice that only looks good on a few validation origins. The champion's test score is then no longer a fully clean estimate (mildly optimistic).
    conf_high: float = 0.20                         # WAPE up to this = "high" confidence
    conf_med: float = 0.45                          # up to this = "medium", above = "low"
    verbose: bool = True

    def __post_init__(self):
        self.targets = [self.target] if isinstance(self.target, str) else list(self.target or [])
        if self.groups is None: self.groups = {"all": [], **{k: [k] for k in self.keys}}
        self.groups = {"all": [], **{k: v for k, v in self.groups.items() if k != "all"}}
        if self.freq not in ("D", "W", "M"): raise ValueError("freq must be 'D', 'W' or 'M'")
        if self.aggregate not in ("sum", "mean", "last"): raise ValueError("aggregate must be 'sum', 'mean' or 'last'")
        if self.fill not in ("linear", "ffill", "none") or self.outliers not in ("none", "clip") or self.transform not in ("none", "log1p", "sqrt", "auto"): raise ValueError("bad fill / outliers / transform option")
        if self.season not in ("auto", "manual", "off"): raise ValueError("season must be 'auto', 'manual' or 'off'")
        if self.range_from not in ("val", "all"): raise ValueError("range_from must be 'val' or 'all'")
        if self.score not in ("aggregate", "step"): raise ValueError("score must be 'aggregate' or 'step'")
        if self.row_agg not in ("sum", "mean"): raise ValueError("row_agg must be 'sum' or 'mean'")
        for name, (col, how) in (self.aux or {}).items():
            if how not in ("nunique", "sum", "mean", "size"): raise ValueError(f"aux '{name}': how must be nunique, sum, mean or size")
        if self.season_length is None: self.season_length = 7 if self.freq == "D" else 1

    @classmethod
    def from_yaml(cls, path):
        import yaml
        with open(path, encoding="utf-8") as f: return cls(**(yaml.safe_load(f) or {}))

    def to_dict(self): return {k: v for k, v in asdict(self).items() if k != "targets"}

    def with_(self, **kw): return replace(self, **kw)
