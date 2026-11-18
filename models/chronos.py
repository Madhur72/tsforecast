"""Pretrained foundation models (zero-shot: nothing is trained on your data). Heavy: downloaded once, run locally on CPU."""
import numpy as np
from ..base import BaseForecaster


class Chronos(BaseForecaster):
    """Chronos-Bolt (small)."""
    name = "Chronos"
    heavy = True
    weights = "amazon/chronos-bolt-small"

    def load(self):
        import torch
        from chronos import BaseChronosPipeline
        self._torch = torch
        self._pipe = BaseChronosPipeline.from_pretrained(self.weights, device_map="cpu", torch_dtype=torch.float32)

    def release(self):
        import gc
        self._pipe = None; gc.collect()

    def predict(self, histories, horizon, ctx):
        keys = [k for k, v in histories.items() if len(v) >= 4]
        if not keys: return {}
        q, _ = self._pipe.predict_quantiles(inputs=[self._torch.tensor(histories[k], dtype=self._torch.float32) for k in keys], prediction_length=horizon, quantile_levels=[0.5])
        return {k: np.clip(q[j, :, 0].numpy(), 0, None) for j, k in enumerate(keys)}


class Chronos2(Chronos):
    """Chronos-2 (larger, newer)."""
    name = "Chronos2"
    weights = "amazon/chronos-2"

    def load(self):
        import torch
        from chronos import Chronos2Pipeline
        self._torch = torch
        self._pipe = Chronos2Pipeline.from_pretrained(self.weights, device_map="cpu")

    def predict(self, histories, horizon, ctx):
        keys = [k for k, v in histories.items() if len(v) >= 4]
        if not keys: return {}
        q, _ = self._pipe.predict_quantiles(inputs=[self._torch.tensor(histories[k], dtype=self._torch.float32) for k in keys], prediction_length=horizon, quantile_levels=[0.5])
        return {k: np.clip(q[j][0, :, 0].numpy(), 0, None) for j, k in enumerate(keys)}
