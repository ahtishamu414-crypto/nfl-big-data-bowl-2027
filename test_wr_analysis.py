import os
import pandas as pd
import numpy as np
from scipy import stats

data_dir = r"C:\Users\BN Com\.cache\kagglehub\competitions\nfl-big-data-bowl-2027\nfl-big-data-bowl-2027"

ct = pd.read_csv(os.path.join(data_dir, "combine_tracking.csv"))
players = pd.read_csv(os.path.join(data_dir, "players.csv"))
cr = pd.read_csv(os.path.join(data_dir, "combine_results.csv"))
pp = pd.read_csv(os.path.join(data_dir, "player_play.csv"))
career = pd.read_csv(os.path.join(data_dir, "player_career_successes.csv"))

wr_ids = set(players[players["nfl_position"] == "WR"]["nfl_id"])
print(f"Total WRs in players: {len(wr_ids)}")

# Filter WR combine tracking
wr_ct = ct[ct["nfl_id"].isin(wr_ids)].copy()
wr_ct["time"] = pd.to_datetime(wr_ct["time"])
wr_ct = wr_ct.sort_values(["nfl_id", "drill_name", "attempt", "time"]).reset_index(drop=True)

CUT_DRILLS = [
    "COMEBACK_ROUTE_RIGHT", "DAGGER_ROUTE_LEFT", "CURL_ROUTE_RIGHT", 
    "SPEED_OUT_ROUTE_LEFT", "POST_CORNER_ROUTE_RIGHT", "POST_CORNER_ROUTE_LEFT",
    "SLOT_SAIL_ROUTE_RIGHT", "SLANT_ROUTE_LEFT"
]

def analyze_attempt(df_attempt):
    df = df_attempt.sort_values("time").reset_index(drop=True)
    if len(df) < 15:
        return None
    s = df["s"].values
    x = df["x"].values
    y = df["y"].values
    d = df["dir"].values
    
    # Smooth speed with 3-frame rolling average
    s_smooth = np.convolve(s, np.ones(3)/3, mode='same')
    
    # Calculate derivative acceleration from velocity
    dt = 0.1
    accel = np.diff(s_smooth, prepend=s_smooth[0]) / dt
    
    # Calculate angular velocity (deg/s)
    ddir = np.abs(np.diff(d, prepend=d[0]))
    ddir = np.where(ddir > 180, 360 - ddir, ddir)
    ang_vel = ddir / dt
    
    # Sprints frames: s > 3.0 yd/s (approx 9 ft/s)
    sprint_idx = np.where(s_smooth > 3.0)[0]
    if len(sprint_idx) < 10:
        return None
    
    # Look for the primary break/cut:
    # A cut has high turning rate or significant deceleration followed by re-acceleration
    # For route drills, the break is typically where turning rate is high and speed dips
    cut_candidates = np.where((ang_vel > 60) & (s_smooth > 2.0))[0]
    if len(cut_candidates) == 0:
        # Fallback: minimum speed between two peaks
        peaks = []
        for i in range(5, len(s_smooth)-5):
            if s_smooth[i] == max(s_smooth[i-5:i+6]) and s_smooth[i] > 4.0:
                peaks.append(i)
        if len(peaks) >= 2:
            apex_idx = peaks[0] + np.argmin(s_smooth[peaks[0]:peaks[1]])
        else:
            return None
    else:
        apex_idx = cut_candidates[np.argmax(ang_vel[cut_candidates])]
    
    if apex_idx < 5 or apex_idx > len(s_smooth) - 5:
        return None
        
    # Pre-break entry (peak in previous 1.5s = 15 frames)
    pre_window = max(0, apex_idx - 15)
    v_entry = np.max(s_smooth[pre_window:apex_idx])
    v_apex = s_smooth[apex_idx]
    
    # Deceleration leading into break
    decel_window = accel[pre_window:apex_idx]
    peak_decel = np.min(decel_window) if len(decel_window) > 0 else 0
    
    # Post-break exit (peak accel and speed in next 1.5s)
    post_window = min(len(s_smooth), apex_idx + 15)
    reaccel_window = accel[apex_idx:post_window]
    peak_reaccel = np.max(reaccel_window) if len(reaccel_window) > 0 else 0
    v_exit = np.max(s_smooth[apex_idx:post_window]) if post_window > apex_idx else v_apex
    
    # Speed retention
    retention = v_apex / max(v_entry, 1.0)
    burst_ratio = v_exit / max(v_entry, 1.0)
    
    # Turning precision / peak angular velocity
    peak_ang_vel = np.max(ang_vel[max(0, apex_idx-5):min(len(ang_vel), apex_idx+6)])
    
    # Time spent in break transition (speed below 80% of entry)
    sub80 = np.where(s_smooth[pre_window:post_window] < 0.8 * v_entry)[0]
    break_duration = len(sub80) * 0.1
    
    # Composite Dynamic Break Efficiency (DBE):
    # Combines entry velocity attack, violent braking, speed retention, and explosive re-acceleration
    # Normalized formula
    dbe = (v_entry * 0.2) + (abs(peak_decel) * 0.3) + (retention * 5.0) + (peak_reaccel * 0.4)
    
    return {
        "v_entry": v_entry,
        "v_apex": v_apex,
        "v_exit": v_exit,
        "speed_retention": retention,
        "burst_ratio": burst_ratio,
        "peak_decel": abs(peak_decel),
        "peak_reaccel": peak_reaccel,
        "peak_ang_vel": peak_ang_vel,
        "break_duration": break_duration,
        "dbe": dbe
    }

print("Extracting features across all WR cut drills...")
results = []
for (nfl_id, drill_name, attempt), grp in wr_ct[wr_ct["drill_name"].isin(CUT_DRILLS)].groupby(["nfl_id", "drill_name", "attempt"]):
    res = analyze_attempt(grp)
    if res:
        res["nfl_id"] = nfl_id
        res["drill_name"] = drill_name
        res["attempt"] = attempt
        results.append(res)

feat_df = pd.DataFrame(results)
print(f"Extracted {len(feat_df)} valid cut drill attempts across {feat_df['nfl_id'].nunique()} WRs")

# Aggregate per WR
wr_features = feat_df.groupby("nfl_id").agg({
    "v_entry": "mean",
    "v_apex": "mean",
    "v_exit": "mean",
    "speed_retention": "mean",
    "burst_ratio": "mean",
    "peak_decel": "mean",
    "peak_reaccel": "mean",
    "peak_ang_vel": "mean",
    "break_duration": "mean",
    "dbe": "mean"
}).reset_index()

# Merge players info & combine results
wr_merged = wr_features.merge(players[["nfl_id", "display_name", "draft_year", "draft_round", "draft_overall_pick"]], on="nfl_id")
wr_merged = wr_merged.merge(cr[["nfl_id", "forty", "vertical", "broad_jump", "three_cone", "short_shuttle", "combine_height", "combine_weight"]], on="nfl_id", how="left")

# Now calculate in-game performance from player_play
wr_game = pp[pp["nfl_id"].isin(wr_ids)].copy()

# Filter for targeted plays or all routes
# Targeted separation
wr_sep_all = wr_game[wr_game["target"] == 1].groupby("nfl_id").agg(
    n_targets=("play_id", "count"),
    mean_sep=("separation_at_pass_forward", "mean"),
    mean_yac=("yards_after_catch", "mean"),
    mean_epa=("expected_points_added", "mean")
).reset_index()

# Man vs Zone separation
wr_sep_man = wr_game[(wr_game["target"] == 1) & (wr_game["team_coverage_man_zone"] == "MAN")].groupby("nfl_id").agg(
    n_targets_man=("play_id", "count"),
    mean_sep_man=("separation_at_pass_forward", "mean"),
    mean_epa_man=("expected_points_added", "mean")
).reset_index()

wr_sep_zone = wr_game[(wr_game["target"] == 1) & (wr_game["team_coverage_man_zone"] == "ZONE")].groupby("nfl_id").agg(
    n_targets_zone=("play_id", "count"),
    mean_sep_zone=("separation_at_pass_forward", "mean"),
    mean_epa_zone=("expected_points_added", "mean")
).reset_index()

# Merge career stats
wr_perf = wr_merged.merge(wr_sep_all, on="nfl_id", how="inner")
wr_perf = wr_perf.merge(wr_sep_man, on="nfl_id", how="left")
wr_perf = wr_perf.merge(wr_sep_zone, on="nfl_id", how="left")
wr_perf = wr_perf.merge(career, on="nfl_id", how="left")

print(f"WRs with both combine tracking and in-game targets: {len(wr_perf)}")

# Qualified WRs (at least 15 targets)
qual = wr_perf[wr_perf["n_targets"] >= 15].copy()
print(f"Qualified WRs (>= 15 targets): {len(qual)}")

print("\n--- CORRELATIONS WITH IN-GAME SEPARATION (ALL TARGETS) ---")
for col in ["dbe", "peak_decel", "peak_reaccel", "speed_retention", "v_entry", "forty", "three_cone", "short_shuttle"]:
    sub_col = qual.dropna(subset=[col, "mean_sep"])
    r, p = stats.pearsonr(sub_col[col], sub_col["mean_sep"])
    print(f"{col:<15} r = {r:+.3f}  (p = {p:.4f}, n={len(sub_col)})")

print("\n--- CORRELATIONS WITH MAN COVERAGE SEPARATION ---")
qual_man = qual[qual["n_targets_man"] >= 5].copy()
for col in ["dbe", "peak_decel", "peak_reaccel", "speed_retention", "v_entry", "forty", "three_cone", "short_shuttle"]:
    sub_col = qual_man.dropna(subset=[col, "mean_sep_man"])
    r, p = stats.pearsonr(sub_col[col], sub_col["mean_sep_man"])
    print(f"{col:<15} r = {r:+.3f}  (p = {p:.4f}, n={len(sub_col)})")
