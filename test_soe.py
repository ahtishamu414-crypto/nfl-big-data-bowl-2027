import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

data_dir = r"C:\Users\BN Com\.cache\kagglehub\competitions\nfl-big-data-bowl-2027\nfl-big-data-bowl-2027"

ct = pd.read_csv(os.path.join(data_dir, "combine_tracking.csv"))
players = pd.read_csv(os.path.join(data_dir, "players.csv"))
cr = pd.read_csv(os.path.join(data_dir, "combine_results.csv"))
pp = pd.read_csv(os.path.join(data_dir, "player_play.csv"))
career = pd.read_csv(os.path.join(data_dir, "player_career_successes.csv"))

wr_ids = set(players[players["nfl_position"] == "WR"]["nfl_id"])
wr_plays = pp[pp["nfl_id"].isin(wr_ids)].copy()

print(f"Total WR plays: {len(wr_plays)}")
print(f"Plays with separation: {wr_plays['separation_at_pass_forward'].notnull().sum()}")

# Clean play-by-play data for Separation Over Expected (SOE)
valid_plays = wr_plays.dropna(subset=["separation_at_pass_forward", "route_ran", "pass_length"]).copy()
print(f"Valid plays for SOE model: {len(valid_plays)}")

# Define breaking routes vs non-breaking routes
BREAKING_ROUTES = ["HITCH", "OUT", "IN", "SLANT", "CORNER", "POST"]
valid_plays["is_breaking_route"] = valid_plays["route_ran"].isin(BREAKING_ROUTES).astype(int)

# Build baseline Expected Separation model:
# Separation is a function of route_ran, pass_length, team_coverage_man_zone, and cushion
valid_plays["cushion_filled"] = valid_plays["cushion"].fillna(valid_plays["cushion"].median())
valid_plays["is_man"] = (valid_plays["team_coverage_man_zone"] == "MAN").astype(int)

# OLS model to predict expected separation based on context
X = pd.get_dummies(valid_plays[["route_ran", "is_man"]], drop_first=True, dtype=float)
X["pass_length"] = valid_plays["pass_length"]
X["cushion"] = valid_plays["cushion_filled"]
X = sm.add_constant(X)
y = valid_plays["separation_at_pass_forward"]

model = sm.OLS(y, X).fit()
valid_plays["expected_sep"] = model.predict(X)
valid_plays["separation_over_expected"] = valid_plays["separation_at_pass_forward"] - valid_plays["expected_sep"]

print("\nExpected Separation Baseline Model Summary:")
print(f"R-squared: {model.rsquared:.3f}")
print("Coefficients top 5:")
print(model.params.head(8))

# Player level aggregates
player_soe = valid_plays.groupby("nfl_id").agg(
    n_routes=("play_id", "count"),
    mean_soe=("separation_over_expected", "mean"),
    mean_sep=("separation_at_pass_forward", "mean"),
    mean_cushion=("cushion_filled", "mean"),
    breaking_soe=("separation_over_expected", lambda s: s[valid_plays.loc[s.index, "is_breaking_route"] == 1].mean()),
    man_soe=("separation_over_expected", lambda s: s[valid_plays.loc[s.index, "is_man"] == 1].mean()),
    targeted_soe=("separation_over_expected", lambda s: s[valid_plays.loc[s.index, "target"] == 1].mean()),
    n_targets=("target", "sum")
).reset_index()

print("\nPlayer SOE distribution (n_routes >= 50):")
qual_players = player_soe[player_soe["n_routes"] >= 50]
print(qual_players[["mean_soe", "breaking_soe", "man_soe", "targeted_soe"]].describe())
