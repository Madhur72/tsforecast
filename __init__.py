"""tsforecast: give it data and a target, it runs every model, evaluates them honestly, picks a champion per level and forecasts.

    from tsforecast import Forecaster, Config
    result = Forecaster(Config(target="sales", time_col="date", keys=["store"])).run(df)
    print(result.summary())

Stages, one file each: preprocess.py, features.py, split.py, models/ (one file per model family), evaluate.py, forecast.py (runs them all).
steps.py holds OPTIONAL use-case specific extras (fold tiny groups, flag outages); the core does not depend on it."""
from .config import Config
from .base import BaseForecaster, SeriesForecaster, ModelRegistry, Context
from .forecast import Forecaster, Result
from .preprocess import Step
from .features import FeatureBuilder, TimeSeriesFeatures, AuxFeatures, HierarchyFeatures

__all__ = ["Config", "Forecaster", "Result", "BaseForecaster", "SeriesForecaster", "ModelRegistry", "Context", "Step", "FeatureBuilder", "TimeSeriesFeatures", "AuxFeatures", "HierarchyFeatures"]
