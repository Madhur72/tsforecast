"""Stage 3 - time-ordered splits. Never random: an 'origin' is a point in time we forecast from, scored on the next `horizon` periods.

      history ........... | validation origins | gap (= horizon) | test origins
                            (pick the champion)                    (the honest score, used once)

The gap keeps validation targets from overlapping test targets."""


def make_origins(n_periods: int, horizon: int, n_test: int, n_val: int, min_history: int, step: int = 1):
    """Returns (val, test): lists of i, where i = number of periods of history at the origin (history = series[:i], targets = series[i:i+horizon]).
    `step` = periods between consecutive origins (1 = every period; use more for daily data so origins are not near-duplicates)."""
    test = sorted(n_periods - horizon - k * step for k in range(n_test))
    last_val = (test[0] - horizon) if test else 0
    val = sorted(last_val - k * step for k in range(n_val))
    val, test = [i for i in val if i >= min_history], [i for i in test if i >= min_history]
    if not test: raise ValueError(f"not enough history: {n_periods} periods cannot give test origins with horizon {horizon} and min_history {min_history}")
    if not val: raise ValueError("not enough history for validation origins: lower n_val / n_test or min_history")
    return val, test
