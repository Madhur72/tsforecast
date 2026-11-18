"""The contract every model follows, and the registry that finds models in the models/ folder.

To add a model: drop a file in models/ with a class that inherits BaseForecaster (or SeriesForecaster) and has a `name`. Nothing else to edit."""
import importlib, inspect, pkgutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np, pandas as pd


@dataclass
class Context:
    """What a model may use besides the histories. Most models ignore it."""
    cfg: object                                  # the Config
    i: int                                       # number of periods of history at this forecast origin
    series: Optional[pd.DataFrame] = None        # all periods x series (only rows < i may be used for training: later rows are the future)
    features: Optional[pd.DataFrame] = None      # the feature table (only built if some model asks for it)
    eligible: Optional[list] = None              # the series that are being forecast
    floor: float = 1.0                           # a series' level is never taken below this (stops scaled numbers exploding on quiet series)
    feature_cols: Optional[list] = None          # names of the feature columns in `features` (whatever the feature builders produced)
    amount_cols: Optional[list] = None           # the feature columns that are amounts in the target's unit: a model should scale these by the series' level (`level8`)


class BaseForecaster(ABC):
    """Batch-first: one call forecasts many series, so global models (LightGBM) and foundation models (Chronos) fit naturally."""
    name: str = None                             # the name shown in results; a class without a name is not registered
    requires_features: bool = False              # True = the feature table is built and passed in ctx.features
    heavy: bool = False                          # True = a big pretrained model; skipped when Config.foundation_models is False

    def load(self) -> None:
        """Import heavy libraries / download weights here, not at module import, so the registry can always list the model."""

    def release(self) -> None:
        """Free memory after the model has been run (called once per model per backtest)."""

    @abstractmethod
    def predict(self, histories: Dict[str, np.ndarray], horizon: int, ctx: Context) -> Dict[str, np.ndarray]:
        """histories: {series: 1-D array of past values, oldest first}. Return {series: array of `horizon` future values}.
        A series may be left out if the model cannot forecast it."""


class SeriesForecaster(BaseForecaster):
    """For simple models that look at one series at a time: implement forecast(history, horizon, cfg)."""

    @abstractmethod
    def forecast(self, history: np.ndarray, horizon: int, cfg) -> np.ndarray: ...

    def predict(self, histories, horizon, ctx):
        out = {}
        for n, h in histories.items():
            try: out[n] = np.clip(np.asarray(self.forecast(np.asarray(h, float), horizon, ctx.cfg), float), 0, None)
            except Exception: pass
        return out


class ModelRegistry:
    """Finds every model class in tsforecast/models (and in any extra folder you add)."""

    def __init__(self):
        self.models: Dict[str, type] = {}
        self._scan_package("tsforecast.models")

    def _register_module(self, mod):
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, BaseForecaster) and not inspect.isabstract(obj) and getattr(obj, "name", None) and obj.__module__ == mod.__name__:
                self.models[obj.name] = obj

    def _scan_package(self, package):
        pkg = importlib.import_module(package)
        for m in pkgutil.iter_modules(pkg.__path__):
            if not m.name.startswith("_"): self._register_module(importlib.import_module(f"{package}.{m.name}"))

    def add_folder(self, folder):
        """Register models from any folder of .py files, for example your own private models."""
        import importlib.util, os
        for f in sorted(os.listdir(folder)):
            if f.endswith(".py") and not f.startswith("_"):
                spec = importlib.util.spec_from_file_location(f[:-3], os.path.join(folder, f)); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
                self._register_module(mod)

    def register(self, cls):
        if not issubclass(cls, BaseForecaster): raise ValueError("a model must inherit BaseForecaster")
        self.models[cls.name] = cls

    def names(self): return sorted(self.models)

    def get(self, name): return self.models[name]()

    def select(self, cfg):
        """The model names a run will use, honouring Config.models / exclude / foundation_models."""
        names = self.names() if cfg.models == "all" else list(cfg.models)
        missing = [n for n in names if n not in self.models]
        if missing: raise ValueError(f"unknown model(s) {missing}; available: {self.names()}")
        return [n for n in names if n not in cfg.exclude and (cfg.foundation_models or not self.models[n].heavy)]
