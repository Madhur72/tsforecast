"""More pretrained foundation models (zero-shot: nothing is trained on your data; weights are downloaded once, then run locally on CPU).
Each needs its own library; if it is not installed the model is skipped with a message and the rest still run. Chronos lives in chronos.py."""
import numpy as np
from ..base import BaseForecaster

MIN_LEN = 4


def _med(x, h):
    """Median forecast as a clipped float array of length h (accepts torch tensors / arrays)."""
    if hasattr(x, "detach"): x = x.detach().cpu().numpy()
    return np.clip(np.asarray(x, float).reshape(-1)[:h], 0, None)


class _Foundation(BaseForecaster):
    heavy = True

    def release(self):
        import gc
        self._m = None; gc.collect()

    def keys(self, histories): return [k for k, v in histories.items() if len(v) >= MIN_LEN]


class TimesFM(_Foundation):
    """TimesFM 2.5 (Google, 200M, Apache-2.0 weights). Not TimesFM 3.0: its weights are non-commercial."""
    name = "TimesFM"

    def load(self):
        import timesfm
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        m.compile(timesfm.ForecastConfig(max_context=512, max_horizon=64, normalize_inputs=True, use_continuous_quantile_head=True,
                                         force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
        self._m = m

    def predict(self, histories, horizon, ctx):
        ids = self.keys(histories)
        if not ids: return {}
        point, _ = self._m.forecast(horizon=horizon, inputs=[np.asarray(histories[k], float) for k in ids])
        return {k: _med(point[i], horizon) for i, k in enumerate(ids)}


class Timestral(_Foundation):
    """Timestral (Mistral, 209M, 'Modified MIT')."""
    name = "Timestral"
    weights = "mistralai/Timestral-209M"

    def load(self):
        from timestral import TimestralForecaster
        self._m = TimestralForecaster.from_pretrained(self.weights, device="cpu")

    def predict(self, histories, horizon, ctx):
        return {k: _med(self._m.forecast(np.asarray(histories[k], float), prediction_length=horizon).median, horizon) for k in self.keys(histories)}


class TiRex2(_Foundation):
    """TiRex-2 (NX-AI, 82.5M, Apache-2.0), univariate (no covariates)."""
    name = "TiRex2"

    def load(self):
        import torch
        import logging
        logging.getLogger("torch._dynamo").setLevel(logging.ERROR); logging.getLogger("torch._inductor").setLevel(logging.ERROR)
        torch._dynamo.config.suppress_errors = True          # no C compiler on the machine (common on Windows): fall back to plain eager mode instead of failing
        from tirex2 import load_model
        self._m = load_model("NX-AI/TiRex-2", device="cpu")

    def predict(self, histories, horizon, ctx):
        import torch
        from tirex2 import TimeseriesType
        ids = self.keys(histories)
        if not ids: return {}
        inputs = [TimeseriesType(target=torch.tensor(np.asarray(histories[k], float), dtype=torch.float32).unsqueeze(0), past_covariates=None, future_covariates=None) for k in ids]
        fc = self._m.forecast(inputs, prediction_length=horizon, output_type="numpy")
        return {k: _med(np.asarray(fc[i])[0, 4, :], horizon) for i, k in enumerate(ids)}        # (1, 9 quantiles, h); index 4 = median


class T0(_Foundation):
    """t0-gamma (The Forecasting Company, ~256M, Apache-2.0)."""
    name = "T0"

    def load(self):
        from t0 import T0Forecaster
        self._m = T0Forecaster.from_pretrained("theforecastingcompany/t0-gamma").eval()

    def predict(self, histories, horizon, ctx):
        import torch
        out = {}
        for k in self.keys(histories):
            with torch.no_grad():
                r = self._m.predict(torch.tensor(np.asarray(histories[k], float), dtype=torch.float32).unsqueeze(0), horizon=horizon, quantile_levels=[0.1, 0.5, 0.9])
            out[k] = _med(r.median, horizon)
        return out


class Simons178(_Foundation):
    """Simons (abel-lab, OpenMDW-1.1 weights); 9 quantiles, index 4 = median."""
    name = "Simons178"
    weights = "abel-lab/Simons-178M"

    def load(self):
        from simons import Simons
        self._m = Simons.from_pretrained(self.weights, device="cpu")

    def predict(self, histories, horizon, ctx):
        import torch
        out = {}
        for k in self.keys(histories):
            q = self._m.forecast(torch.tensor(np.asarray(histories[k], float), dtype=torch.float32).unsqueeze(0), horizon=horizon)
            out[k] = _med(np.asarray(q.detach().cpu() if hasattr(q, "detach") else q)[0, :horizon, 4], horizon)
        return out


class Simons297(Simons178):
    name = "Simons297"
    weights = "abel-lab/Simons-297M"
