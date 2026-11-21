import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from scipy import stats

data_dir = r"C:\Users\BN Com\.cache\kagglehub\competitions\nfl-big-data-bowl-2027\nfl-big-data-bowl-2027"

print("Loading data...")
ct = pd.read_csv(os.path.join(data_dir, "combine_tracking.csv"))
players = pd.read_csv(os.path.join(data_dir, "players.csv"))
cr = pd.read_csv(os.path.join(data_dir, "combine_results.csv"))
pp = pd.read_csv(os.path.join(data_dir, "player_play.csv"))
career = pd.read_csv(os.path.join(data_dir, "player_career_successes.csv"))

wr_ids = set(players[players["nfl_position"] == "WR"]["nfl_id"])
wr_plays = pp[pp["nfl_id"].isin(wr_ids)].copy().reset_index(drop=True)

# Define breaking routes
BREAKING_ROUTES = ["HITCH", "OUT", "IN", "SLANT", "CORNER", "POST"]
wr_plays["is_breaking"] = wr_plays["route_ran"].isin(BREAKING_ROUTES).astype(int)
wr_plays["is_man"] = (wr_plays["team_coverage_man_zone"] == "MAN").astype(int)
wr_plays["cushion_filled"] = wr_plays["cushion"].fillna(wr_plays["cushion"].median())

# Fit SOE model on plays with separation
valid_mask = wr_plays["separation_at_pass_forward"].notnull() & wr_plays["pass_length"].notnull() & wr_plays["route_ran"].notnull()
v_plays = wr_plays[valid_mask].copy()

# Features for Expected Separation
route_dummies = pd.get_dummies(v_plays["route_ran"], prefix="route", drop_first=True, dtype=float)
X = pd.concat([route_dummies, v_plays[["is_man", "pass_length", "cushion_filled"]]], axis=1)
y = v_plays["separation_at_pass_forward"]

ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
v_plays["exp_sep"] = ridge.predict(X)
v_plays["soe"] = v_plays["separation_at_pass_forward"] - v_plays["exp_sep"]

print(f"SOE baseline model R²: {ridge.score(X, y):.3f}")

# Group by player
records = []
for nfl_id, df in v_plays.groupby("nfl_id"):
    rec = {
        "nfl_id": nfl_id,
        "n_routes": len(df),
        "mean_sep": df["separation_at_pass_forward"].mean(),
        "mean_soe": df["soe"].mean(),
        "n_targets": df["target"].sum(),
    }
    # Breaking routes
    br = df[df["is_breaking"] == 1]
    rec["breaking_soe"] = br["soe"].mean() if len(br) > 0 else np.nan
    rec["breaking_sep"] = br["separation_at_pass_forward"].mean() if len(br) > 0 else np.nan
    
    # Man coverage
    man = df[df["is_man"] == 1]
    rec["man_soe"] = man["soe"].mean() if len(man) > 0 else np.nan
    rec["man_sep"] = man["separation_at_pass_forward"].mean() if len(man) > 0 else np.nan
    
    # Targeted
    tgt = df[df["target"] == 1]
    rec["targeted_soe"] = tgt["soe"].mean() if len(tgt) > 0 else np.nan
    rec["targeted_sep"] = tgt["separation_at_pass_forward"].mean() if len(tgt) > 0 else np.nan
    rec["mean_yac"] = tgt["yards_after_catch"].mean() if len(tgt) > 0 else np.nan
    rec["mean_epa"] = tgt["expected_points_added"].mean() if len(tgt) > 0 else np.nan
    
    records.append(rec)

player_in_game = pd.DataFrame(records)

# Combine tracking features
wr_ct = ct[ct["nfl_id"].isin(wr_ids)].copy()
wr_ct["time"] = pd.to_datetime(wr_ct["time"])

CUT_DRILLS = [
    "COMEBACK_ROUTE_RIGHT", "DAGGER_ROUTE_LEFT", "CURL_ROUTE_RIGHT", 
    "SPEED_OUT_ROUTE_LEFT", "POST_CORNER_ROUTE_RIGHT", "POST_CORNER_ROUTE_LEFT",
    "SLOT_SAIL_ROUTE_RIGHT", "SLANT_ROUTE_LEFT"
]

def extract_cut_features(g):
    g = g.sort_values("time").reset_index(drop=True)
    if len(g) < 15:
        return None
    s = g["s"].values
    d = g["dir"].values
    
    # Smooth
    s_sm = np.convolve(s, np.ones(3)/3, mode='same')
    dt = 0.1
    acc = np.diff(s_sm, prepend=s_sm[0]) / dt
    
    ddir = np.abs(np.diff(d, prepend=d[0]))
    ddir = np.where(ddir > 180, 360 - ddir, ddir)
    ang_vel = ddir / dt
    
    # Find turning apex: point of maximum turn rate when running (s > 2)
    run_pts = np.where(s_sm > 2.0)[0]
    if len(run_pts) < 10:
        return None
    
    apex = run_pts[np.argmax(ang_vel[run_pts])]
    if apex < 5 or apex > len(s_sm) - 5:
        return None
    
    v_stem = np.max(s_sm[max(0, apex-15):apex])
    v_apex = s_sm[apex]
    
    decel_phase = acc[max(0, apex-12):apex]
    peak_decel = np.min(decel_phase) if len(decel_phase) > 0 else 0
    
    burst_phase = acc[apex:min(len(acc), apex+15)]
    peak_burst = np.max(burst_phase) if len(burst_phase) > 0 else 0
    v_exit = np.max(s_sm[apex:min(len(s_sm), apex+15)])
    
    retention = v_apex / max(v_stem, 1.0)
    max_turn_rate = ang_vel[apex]
    
    # Dynamic Break Efficiency (DBE) Index
    dbe = (v_stem * 0.25) + (abs(peak_decel) * 0.35) + (retention * 4.0) + (peak_burst * 0.40)
    
    return {
        "v_stem": v_stem,
        "v_apex": v_apex,
        "v_exit": v_exit,
        "peak_decel": abs(peak_decel),
        "peak_burst": peak_burst,
        "retention": retention,
        "max_turn_rate": max_turn_rate,
        "dbe": dbe
    }

drill_records = []
for (nfl_id, drill_name, attempt), grp in wr_ct[wr_ct["drill_name"].isin(CUT_DRILLS)].groupby(["nfl_id", "drill_name", "attempt"]):
    feats = extract_cut_features(grp)
    if feats:
        feats["nfl_id"] = nfl_id
        drill_records.append(feats)

drill_df = pd.DataFrame(drill_records)
player_combine_cuts = drill_df.groupby("nfl_id").mean().reset_index()

# Merge everything
final_df = player_combine_cuts.merge(player_in_game, on="nfl_id", how="inner")
final_df = final_df.merge(players[["nfl_id", "display_name", "draft_year", "draft_round", "draft_overall_pick"]], on="nfl_id", how="left")
final_df = final_df.merge(cr[["nfl_id", "forty", "vertical", "broad_jump", "three_cone", "short_shuttle"]], on="nfl_id", how="left")
final_df = final_df.merge(career, on="nfl_id", how="left")

print(f"\nFinal merged players: {len(final_df)}")
qual = final_df[final_df["n_routes"] >= 30].copy()
print(f"Qualified players (>= 30 routes): {len(qual)}")

print("\n--- CORRELATION MATRIX: COMBINE SENSORS vs BREAKING ROUTE SOE ---")
metrics = ["dbe", "peak_decel", "peak_burst", "retention", "v_stem", "max_turn_rate", "forty", "three_cone", "short_shuttle"]
for m in metrics:
    sub = qual.dropna(subset=[m, "breaking_soe"])
    if len(sub) > 2:
        r, p = stats.pearsonr(sub[m], sub["breaking_soe"])
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        print(f"{m:<16} : r = {r:+.3f}  (p = {p:.4f}, n = {len(sub):2d}) {sig}")

print("\n--- CORRELATION MATRIX: COMBINE SENSORS vs MAN COVERAGE SOE ---")
for m in metrics:
    sub = qual.dropna(subset=[m, "man_soe"])
    if len(sub) > 2:
        r, p = stats.pearsonr(sub[m], sub["man_soe"])
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        print(f"{m:<16} : r = {r:+.3f}  (p = {p:.4f}, n = {len(sub):2d}) {sig}")

print("\n--- CORRELATION MATRIX: COMBINE SENSORS vs CAREER STARTS / SNAPS ---")
for m in metrics:
    sub = qual.dropna(subset=[m, "career_games_started"])
    if len(sub) > 2:
        r, p = stats.pearsonr(sub[m], sub["career_games_started"])
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        print(f"{m:<16} : r = {r:+.3f}  (p = {p:.4f}, n = {len(sub):2d}) {sig}")

# Save final_df to csv for quick access and plotting
final_df.to_csv("e:/Kaggle/wr_combine_nfl_metrics.csv", index=False)
print("\nSaved wr_combine_nfl_metrics.csv successfully!")
