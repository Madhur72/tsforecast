"""Example: the token-usage use case, expressed ONLY through config + optional steps. Nothing here is built into the core.
Run from the folder that contains tsforecast/:   python tsforecast/example.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from tsforecast import Forecaster, Config
from tsforecast.steps import FoldSmallGroups, FlagOutages, drop_outages

df = pd.read_parquet("fc/cache/token_usage.parquet")
cfg = Config(target="total_tokens", time_col="created_at", id_col="id", tz="Asia/Kolkata",
             groups={"org": ["organization_id"], "team": ["organization_id", "team_id"], "agent": ["organization_id", "agent_name_constant"], "user": ["organization_id", "user_id"]},
             parent_of={"team": "org", "agent": "org", "user": "org"},
             aux={"users": ("user_id", "nunique"), "rows": (None, "size")},
             freq="W", horizon=4, aggregate="sum", foundation_models=False)
f = Forecaster(cfg, steps=[FoldSmallGroups(10), FlagOutages(3)], history_filter=drop_outages())
res = f.run(df)
print(res.summary())
