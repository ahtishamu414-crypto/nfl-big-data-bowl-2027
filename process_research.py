import os
import pandas as pd
import numpy as np

data_dir = r"C:\Users\BN Com\.cache\kagglehub\competitions\nfl-big-data-bowl-2027\nfl-big-data-bowl-2027"

ct = pd.read_csv(os.path.join(data_dir, "combine_tracking.csv"))
players = pd.read_csv(os.path.join(data_dir, "players.csv"))
cr = pd.read_csv(os.path.join(data_dir, "combine_results.csv"))
pp = pd.read_csv(os.path.join(data_dir, "player_play.csv"))
career = pd.read_csv(os.path.join(data_dir, "player_career_successes.csv"))

wr_df = players[players["nfl_position"] == "WR"].copy()
wr_ids = set(wr_df["nfl_id"])
wr_ct = ct[ct["nfl_id"].isin(wr_ids)].copy()

CUT_DRILLS = [
    "COMEBACK_ROUTE_RIGHT", "DAGGER_ROUTE_LEFT", "CURL_ROUTE_RIGHT", 
    "SPEED_OUT_ROUTE_LEFT", "POST_CORNER_ROUTE_RIGHT", "POST_CORNER_ROUTE_LEFT",
    "SLOT_SAIL_ROUTE_RIGHT", "SLANT_ROUTE_LEFT"
]

sub = wr_ct[wr_ct["drill_name"].isin(CUT_DRILLS)]
print(f"Total cut drill rows: {len(sub)}")
print(f"Unique WRs with cut drills: {sub['nfl_id'].nunique()} / {len(wr_ids)}")

# Let's inspect the columns and units in combine_tracking:
# x, y, s, a, dis, dir
sample = sub[sub["drill_name"] == "COMEBACK_ROUTE_RIGHT"].iloc[0:10]
print("Sample combine tracking:")
print(sample[["nfl_id", "drill_name", "attempt", "x", "y", "s", "a", "dir"]])
