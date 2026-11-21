import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, LinearRegression
from scipy import stats

# Set styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.titlesize'] = 15

os.makedirs('e:/Kaggle/figures', exist_ok=True)
data_dir = r"C:\Users\BN Com\.cache\kagglehub\competitions\nfl-big-data-bowl-2027\nfl-big-data-bowl-2027"

print("1. Loading datasets...")
ct = pd.read_csv(os.path.join(data_dir, "combine_tracking.csv"))
players = pd.read_csv(os.path.join(data_dir, "players.csv"))
cr = pd.read_csv(os.path.join(data_dir, "combine_results.csv"))
pp = pd.read_csv(os.path.join(data_dir, "player_play.csv"))
career = pd.read_csv(os.path.join(data_dir, "player_career_successes.csv"))

wr_df = players[players["nfl_position"] == "WR"].copy()
wr_ids = set(wr_df["nfl_id"])
print(f"Total WRs: {len(wr_ids)}")

# -------------------------------------------------------------
# FIGURE 1: Biomechanical Decomposition of a Combine Route Cut
# -------------------------------------------------------------
print("2. Generating Figure 1: Route Cut Decomposition...")
# Pick a high-quality example cut: Xavier Restrepo or Jaxon Smith-Njigba or Puka Nacua
comeback_wr = ct[(ct["nfl_id"].isin(wr_ids)) & (ct["drill_name"] == "COMEBACK_ROUTE_RIGHT")]
sample_id = comeback_wr["nfl_id"].iloc[0]
sample_player_name = players[players["nfl_id"] == sample_id]["display_name"].iloc[0]
sample_event = comeback_wr[comeback_wr["nfl_id"] == sample_id]["event_id"].iloc[0]

sample_g = comeback_wr[(comeback_wr["nfl_id"] == sample_id) & (comeback_wr["event_id"] == sample_event)].sort_values("time").reset_index(drop=True)
s_raw = sample_g["s"].values
t = np.arange(len(s_raw)) * 0.1
s_sm = np.convolve(s_raw, np.ones(3)/3, mode='same')
acc = np.diff(s_sm, prepend=s_sm[0]) / 0.1
d = sample_g["dir"].values
ddir = np.abs(np.diff(d, prepend=d[0]))
ddir = np.where(ddir > 180, 360 - ddir, ddir)
ang_vel = ddir / 0.1

# Zoom in on running window
run_mask = s_sm > 2.0
if np.sum(run_mask) > 10:
    run_idx = np.where(run_mask)[0]
    apex_rel = run_idx[np.argmax(ang_vel[run_idx])]
    w_start = max(0, apex_rel - 20)
    w_end = min(len(s_sm), apex_rel + 20)
else:
    w_start, w_end = 0, len(s_sm)
    apex_rel = len(s_sm) // 2

t_w = t[w_start:w_end] - t[w_start]
s_w = s_sm[w_start:w_end]
acc_w = acc[w_start:w_end]
ang_w = ang_vel[w_start:w_end]
apex_w = apex_rel - w_start

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Velocity trace
ax1.plot(t_w, s_w, color='#1f77b4', lw=2.5, label='Velocity (s)')
ax1.axvline(t_w[apex_w], color='#d62728', ls='--', lw=2, label=f'Cut Apex ($V_{{apex}}={s_w[apex_w]:.2f}$ yd/s)')
ax1.scatter([t_w[apex_w]], [s_w[apex_w]], color='#d62728', s=80, zorder=5)
v_stem_idx = np.argmax(s_w[:apex_w])
ax1.scatter([t_w[v_stem_idx]], [s_w[v_stem_idx]], color='#2ca02c', s=80, zorder=5)
ax1.annotate(f'Stem Attack ($V_{{stem}}={s_w[v_stem_idx]:.2f}$ yd/s)', 
             xy=(t_w[v_stem_idx], s_w[v_stem_idx]), 
             xytext=(t_w[v_stem_idx]-0.8, s_w[v_stem_idx]+0.8),
             arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.5))
ax1.set_ylabel('Velocity (yd/s)')
ax1.set_title(f'Biomechanical Anatomy of a Route Cut (Combine 10 Hz Sensor Tracking - {sample_player_name})', fontweight='bold')
ax1.legend(loc='lower left', frameon=True)
ax1.grid(True, alpha=0.3)

# Acceleration trace
ax2.plot(t_w, acc_w, color='#ff7f0e', lw=2.5, label='Acceleration / Braking (a)')
ax2.axhline(0, color='gray', ls=':', lw=1)
ax2.axvline(t_w[apex_w], color='#d62728', ls='--', lw=2)
decel_idx = np.argmin(acc_w[:apex_w])
ax2.scatter([t_w[decel_idx]], [acc_w[decel_idx]], color='#9467bd', s=80, zorder=5)
ax2.annotate(f'Peak Braking Decel ({acc_w[decel_idx]:.2f} yd/s²)', 
             xy=(t_w[decel_idx], acc_w[decel_idx]), 
             xytext=(t_w[decel_idx]-0.9, acc_w[decel_idx]-2.5),
             arrowprops=dict(arrowstyle='->', color='#9467bd', lw=1.5))
reacc_idx = apex_w + np.argmax(acc_w[apex_w:])
ax2.scatter([t_w[reacc_idx]], [acc_w[reacc_idx]], color='#8c564b', s=80, zorder=5)
ax2.annotate(f'Exit Burst (+{acc_w[reacc_idx]:.2f} yd/s²)', 
             xy=(t_w[reacc_idx], acc_w[reacc_idx]), 
             xytext=(t_w[reacc_idx]+0.3, acc_w[reacc_idx]+1.5),
             arrowprops=dict(arrowstyle='->', color='#8c564b', lw=1.5))
ax2.set_ylabel('Accel (yd/s²)')
ax2.legend(loc='upper right', frameon=True)
ax2.grid(True, alpha=0.3)

# Angular Velocity trace
ax3.plot(t_w, ang_w, color='#2ca02c', lw=2.5, label='Angular Velocity (deg/s)')
ax3.axvline(t_w[apex_w], color='#d62728', ls='--', lw=2)
ax3.set_ylabel('Turn Rate (deg/s)')
ax3.set_xlabel('Time into Route Cut Window (seconds)')
ax3.legend(loc='upper left', frameon=True)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('e:/Kaggle/figures/fig1_route_cut_anatomy.png', dpi=300)
plt.close()
print("Figure 1 saved!")

# -------------------------------------------------------------
# FIGURE 2: The Opt-Out Crisis & Scouting Coverage
# -------------------------------------------------------------
print("3. Generating Figure 2: The Opt-Out Crisis...")
wr_cr = cr[cr["nfl_id"].isin(wr_ids)]
tests = {
    'Combine Route Drills\n(10 Hz Tracking)': len(ct[ct['nfl_id'].isin(wr_ids)]['nfl_id'].unique()),
    '40-Yard Dash\n(Stopwatch)': wr_cr['forty'].notnull().sum(),
    'Vertical Jump': wr_cr['vertical'].notnull().sum(),
    'Broad Jump': wr_cr['broad_jump'].notnull().sum(),
    'Short Shuttle\n(20-Yd Agility)': wr_cr['short_shuttle'].notnull().sum(),
    '3-Cone Drill\n(COD Agility)': wr_cr['three_cone'].notnull().sum()
}

pcts = [v / len(wr_ids) * 100 for v in tests.values()]
names = list(tests.keys())
colors = ['#1a9850', '#66bd63', '#fdae61', '#fdae61', '#f46d43', '#d73027']

plt.figure(figsize=(9, 5))
bars = plt.barh(names, pcts, color=colors, edgecolor='black', lw=1.2)
plt.xlim(0, 115)
plt.xlabel('Prospect Participation Rate (%)', fontweight='bold')
plt.title('The Scouting Blind Spot: Prospect Participation Across Combine Drills (WR Class, N=106)', fontweight='bold')
for bar, pct, n in zip(bars, pcts, tests.values()):
    plt.text(pct + 1.5, bar.get_y() + bar.get_height()/2, f'{pct:.1f}% ({n}/{len(wr_ids)})', 
             va='center', fontweight='bold', fontsize=10)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('e:/Kaggle/figures/fig2_opt_out_crisis.png', dpi=300)
plt.close()
print("Figure 2 saved!")

# -------------------------------------------------------------
# Load and Process Master Analysis
# -------------------------------------------------------------
print("4. Calculating metrics and models...")
final_df = pd.read_csv("e:/Kaggle/wr_combine_nfl_metrics.csv")
qual = final_df[final_df["n_routes"] >= 30].copy()

# -------------------------------------------------------------
# FIGURE 3: Predictive Power: Combine Sensor Kinematics vs Stopwatch Testing
# -------------------------------------------------------------
print("5. Generating Figure 3: Correlation Comparison...")
corrs = {
    'Apex Cut Speed (V_apex)': np.corrcoef(qual['v_apex'], qual['career_offensive_snaps'])[0, 1],
    'Stem Attack Speed (V_stem)': np.corrcoef(qual['v_stem'], qual['career_offensive_snaps'])[0, 1],
    'Targeted EPA per Play': np.corrcoef(qual['mean_epa'], qual['career_offensive_snaps'])[0, 1],
    'Broad Jump (Distance)': np.corrcoef(qual.dropna(subset=['broad_jump', 'career_offensive_snaps'])['broad_jump'], 
                                        qual.dropna(subset=['broad_jump', 'career_offensive_snaps'])['career_offensive_snaps'])[0, 1],
    '40-Yard Dash (Stopwatch)': -np.corrcoef(qual.dropna(subset=['forty', 'career_offensive_snaps'])['forty'], 
                                            qual.dropna(subset=['forty', 'career_offensive_snaps'])['career_offensive_snaps'])[0, 1], # invert so faster is positive
    '3-Cone Drill (Stopwatch)': -np.corrcoef(qual.dropna(subset=['three_cone', 'career_offensive_snaps'])['three_cone'], 
                                            qual.dropna(subset=['three_cone', 'career_offensive_snaps'])['career_offensive_snaps'])[0, 1]
}

corr_labels = list(corrs.keys())
corr_vals = list(corrs.values())
corr_colors = ['#1a9850', '#1a9850', '#66bd63', '#a6d96a', '#fdae61', '#f46d43']

plt.figure(figsize=(9.5, 5.5))
bars = plt.barh(corr_labels, corr_vals, color=corr_colors, edgecolor='black', lw=1.2)
plt.xlim(-0.1, 0.6)
plt.axvline(0, color='gray', lw=1)
plt.xlabel('Correlation with NFL Career Offensive Snaps (r)', fontweight='bold')
plt.title('Translational Validity: Combine Sensor Kinematics vs Traditional Stopwatch Tests', fontweight='bold')
for bar, v in zip(bars, corr_vals):
    pos = v + 0.015 if v >= 0 else v - 0.04
    plt.text(pos, bar.get_y() + bar.get_height()/2, f'r = {v:+.3f}', va='center', fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('e:/Kaggle/figures/fig3_predictive_power_comparison.png', dpi=300)
plt.close()
print("Figure 3 saved!")

# -------------------------------------------------------------
# FIGURE 4: Context-Adjusted Separation Over Expected (SOE)
# -------------------------------------------------------------
print("6. Generating Figure 4: SOE by Route Context...")
# Load sample of valid plays for route distribution
valid_mask = pp["separation_at_pass_forward"].notnull() & pp["route_ran"].notnull() & pp["pass_length"].notnull()
wr_plays_sample = pp[pp["nfl_id"].isin(wr_ids) & valid_mask].copy()

plt.figure(figsize=(10, 5))
route_order = wr_plays_sample.groupby("route_ran")["separation_at_pass_forward"].median().sort_values(ascending=False).index
sns.boxplot(data=wr_plays_sample, x="route_ran", y="separation_at_pass_forward", order=route_order, palette="Spectral", showfliers=False)
plt.title('Context Dictates Separation: On-Field Separation Varies Drastically by Route Scheme', fontweight='bold')
plt.xlabel('Route Ran by Receiver', fontweight='bold')
plt.ylabel('Separation at Pass Forward (Yards)', fontweight='bold')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('e:/Kaggle/figures/fig4_route_context_separation.png', dpi=300)
plt.close()
print("Figure 4 saved!")

# -------------------------------------------------------------
# FIGURE 5: Prospect Evaluation Quadrant: Stem Attack vs Apex Speed
# -------------------------------------------------------------
print("7. Generating Figure 5: Prospect Quadrant & Sleeper Identification...")
plt.figure(figsize=(10, 7.5))
stem_med = qual['v_stem'].median()
apex_med = qual['v_apex'].median()

# Scatter points sized by career snaps
scatter = plt.scatter(qual['v_stem'], qual['v_apex'], 
                      c=qual['career_offensive_snaps'], cmap='viridis', 
                      s=qual['career_offensive_snaps']/12 + 40, 
                      alpha=0.85, edgecolors='black', lw=1)
cbar = plt.colorbar(scatter)
cbar.set_label('NFL Career Offensive Snaps', fontweight='bold')

plt.axvline(stem_med, color='gray', ls='--', lw=1.2, alpha=0.7)
plt.axhline(apex_med, color='gray', ls='--', lw=1.2, alpha=0.7)

# Quadrant labels
plt.text(qual['v_stem'].max() - 0.05, qual['v_apex'].max(), 'QUADRANT I:\nELITE FLUID SEPARATORS\n(High Stem, High Apex)', 
         ha='right', va='top', fontsize=9, fontweight='bold', color='#006837', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.text(qual['v_stem'].min() + 0.05, qual['v_apex'].max(), 'QUADRANT II:\nSLOT TECHNICIANS\n(Fluid Apex, Controlled Stem)', 
         ha='left', va='top', fontsize=9, fontweight='bold', color='#2166ac', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.text(qual['v_stem'].max() - 0.05, qual['v_apex'].min(), 'QUADRANT IV:\nWORKOUT WARRIOR BUSTS\n(Fast Stem, Sluggish Break)', 
         ha='right', va='bottom', fontsize=9, fontweight='bold', color='#b2182b', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Annotate key notable prospects
notables = ['Puka Nacua', 'Jaxon Smith-Njigba', 'Jordan Addison', 'Jayden Reed', 'Brian Thomas', 'Ladd McConkey', 'Michael Wilson']
for name in notables:
    row = qual[qual['display_name'] == name]
    if len(row) > 0:
        r = row.iloc[0]
        plt.annotate(f"{name}\n({r['career_offensive_snaps']:.0f} snaps)", 
                     xy=(r['v_stem'], r['v_apex']),
                     xytext=(r['v_stem']+0.04, r['v_apex']+0.08),
                     fontsize=9, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', lw=1.2, color='black'))

plt.xlabel('Stem Attack Velocity $V_{stem}$ (yd/s)', fontweight='bold')
plt.ylabel('Break Apex Velocity $V_{apex}$ (yd/s)', fontweight='bold')
plt.title('The Kinematic Separation Matrix: Identifying Day 3 Gems & Busts from 10 Hz Combine Tracking', fontweight='bold')
plt.tight_layout()
plt.savefig('e:/Kaggle/figures/fig5_prospect_quadrant.png', dpi=300)
plt.close()
print("Figure 5 saved!")

# -------------------------------------------------------------
# FIGURE 6: Drill-to-Route Kinematic Translation Heatmap
# -------------------------------------------------------------
print("8. Generating Figure 6: Translation Heatmap...")
corr_mat = qual[['v_stem', 'v_apex', 'peak_decel', 'peak_burst', 'mean_sep', 'breaking_soe', 'mean_epa', 'career_offensive_snaps']].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr_mat, annot=True, fmt='+.2f', cmap='vlag', center=0, cbar_kws={'label': 'Pearson Correlation (r)'})
plt.title('Kinematic Correlation Matrix: Combine 10 Hz Metrics vs NFL Game Performance', fontweight='bold')
plt.tight_layout()
plt.savefig('e:/Kaggle/figures/fig6_kinematic_heatmap.png', dpi=300)
plt.close()
print("Figure 6 saved!")

print("\n--- ALL 6 FIGURES GENERATED SUCCESSFULLY! ---")
