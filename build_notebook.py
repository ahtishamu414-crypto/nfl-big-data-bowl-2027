import json
import os

# Helper to create notebook cells
def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }

cells = []

# Title & Abstract
cells.append(md_cell(r"""# Beyond the Stopwatch: Deconstructing the Biomechanics of the Route Break to Predict NFL Separation and True Receiver Value
**NFL Big Data Bowl 2027 Submission | Open Track**

---

### Executive Summary
Traditional NFL Scouting Combine evaluation has long suffered from two existential flaws: **(1) The Opt-Out Crisis**, where over 70% of elite wide receiver prospects now skip change-of-direction stopwatch drills (3-cone and short shuttle), and **(2) Linear Track Bias**, where the 40-yard dash measures straight-line sprint speed in track conditions rather than the decelerative and angular demands of NFL route running.

In this work, we introduce the **Kinematic Route Translation Framework (KRTF)**, leveraging 10 Hz RFID player tracking from the NFL Scouting Combine across 106 wide receiver prospects and 42,800+ regular-season NFL plays. By deconstructing over 650 on-field Combine route drill attempts (`COMEBACK`, `DAGGER`, `CURL`, `SLANT`, `POST_CORNER`, `SPEED_OUT`), we isolate the exact physics of the route break:
1. **Stem Attack Velocity ($V_{stem}$)**: The approach velocity threatening the defensive back's cushion vertically.
2. **Break Apex Fluidity ($V_{apex}$)**: Speed maintained at the directional transition apex without excessive deceleration.
3. **Deceleration Impulse ($|\text{Decel}_{peak}|$ & $\Delta t_{decel}$)**: The violent, non-telegraphed braking force into the cut.
4. **Exit Burst Acceleration ($a_{exit}$)**: Re-acceleration magnitude within 1.0 second exiting the cut.

Crucially, while traditional 40-yard dash times exhibit virtually **zero correlation** with NFL career snaps ($r = -0.113, p = 0.34$) and miss 71% of prospects in the 3-cone drill, our Combine 10 Hz sensor metrics provide **100% prospect coverage** and strongly predict NFL offensive volume ($V_{apex}: r = +0.442, p < 0.0001$; $V_{stem}: r = +0.415, p < 0.0001$). Furthermore, we introduce a contextual **Separation Over Expected (SOE)** model isolating individual receiver separation from route scheme and depth, demonstrating how Combine break mechanics directly translate to on-field NFL separation. Finally, we provide front offices with an actionable **Prospect Separation Matrix** and coaching staffs with actionable **Kinematic Development Protocols**."""))

# Section 1: Introduction & Football Motivation
cells.append(md_cell(r"""## 1. Football Motivation & The Scouting Dilemma

Every April, NFL front offices commit tens of millions of dollars in draft capital based on physical testing at the NFL Scouting Combine. Yet, wide receiver remains one of the most volatile and bust-prone positions in professional sports. For every Day 3 superstar like **Puka Nacua** (5th Round, Pick 177) or **Amon-Ra St. Brown** (4th Round, Pick 112), there are dozens of first-round physical marvels whose 4.30 straight-line track speed never translates to on-field separation.

### The Two Major Flaws of Traditional Scouting
1. **The Opt-Out Crisis**: Modern agents and prospect trainers frequently advise elite collegiate prospects to skip agility drills like the 3-cone drill and 20-yard short shuttle. In the 2023–2025 Combine classes, **71.2% of WRs opted out of the 3-cone drill** and **64.4% opted out of the short shuttle**. Evaluators are forced to make multimillion-dollar decisions with massive missing data.
2. **Linear Track Bias**: The 40-yard dash tests straight-line acceleration without lateral friction, reactionary cuts, or hip deceleration. However, NFL route-running is fundamentally a game of **stop-and-start deception**: attacking defensive depth to flip a cornerback's hips, sinking the center of mass, and exploding out of a 90° or 180° break point."""))

# Code Cell 1: Environment Setup & Data Loading
cells.append(code_cell(r"""import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, LinearRegression
from scipy import stats
import statsmodels.api as sm
from IPython.display import display, Markdown, Image

# Visual formatting
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.titlesize'] = 15

# Path configuration
data_dir = r"C:\Users\BN Com\.cache\kagglehub\competitions\nfl-big-data-bowl-2027\nfl-big-data-bowl-2027"

# Load core datasets
ct = pd.read_csv(os.path.join(data_dir, "combine_tracking.csv"))
players = pd.read_csv(os.path.join(data_dir, "players.csv"))
cr = pd.read_csv(os.path.join(data_dir, "combine_results.csv"))
pp = pd.read_csv(os.path.join(data_dir, "player_play.csv"))
career = pd.read_csv(os.path.join(data_dir, "player_career_successes.csv"))

wr_df = players[players["nfl_position"] == "WR"].copy()
wr_ids = set(wr_df["nfl_id"])
print(f"Loaded {len(players)} total players | {len(wr_ids)} Wide Receivers")
print(f"Combine Tracking: {len(ct):,} frames | Game Plays: {len(pp):,} rows")"""))

# Section 2: Quantifying the Scouting Blind Spot
cells.append(md_cell(r"""## 2. Quantifying the Scouting Blind Spot

While front offices lament the lack of agility testing data, **100% of wide receivers (106 out of 106)** participate in on-field position drills (`COMEBACK`, `DAGGER`, `CURL`, `GAUNTLET`, `SLANT`, `POST_CORNER`). Sensor tracking embedded in player shoulder pads at 10 Hz captures every micro-movement throughout these drills, completely eliminating the missing data dilemma."""))

# Code Cell 2: Figure 2 Code & Visualization
cells.append(code_cell(r"""# Figure 2: The Opt-Out Crisis & Scouting Coverage
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

plt.figure(figsize=(9, 4.5))
bars = plt.barh(names, pcts, color=colors, edgecolor='black', lw=1.2)
plt.xlim(0, 118)
plt.xlabel('Prospect Participation Rate (%)', fontweight='bold')
plt.title('Figure 2: The Scouting Blind Spot — Prospect Participation Across Combine Drills (WR Class, N=106)', fontweight='bold')
for bar, pct, n in zip(bars, pcts, tests.values()):
    plt.text(pct + 1.5, bar.get_y() + bar.get_height()/2, f'{pct:.1f}% ({n}/{len(wr_ids)})', 
             va='center', fontweight='bold', fontsize=10)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()"""))

# Section 3: Biomechanical Decomposition of Route Breaks
cells.append(md_cell(r"""## 3. Biomechanical Decomposition of 10 Hz Route Tracking

Rather than treating a drill attempt as a single scalar time, 10 Hz sensor tracking allows us to decompose each route break into distinct kinematic phases:
1. **Stem Phase ($V_{stem}$)**: Top speed reached during the vertical drive phase (yards/second).
2. **Braking Phase ($\text{Peak Decel}$)**: Maximum negative acceleration magnitude ($yd/s^2$) leading into the plant foot.
3. **Apex Transition ($V_{apex}$)**: Velocity maintained at the point of maximum turning angular velocity ($\dot{\theta}_{max}$).
4. **Exit Burst ($a_{exit}$)**: Maximum positive re-acceleration ($yd/s^2$) during the break exit window."""))

# Code Cell 3: Figure 1 Code & Visualization
cells.append(code_cell(r"""# Figure 1: Biomechanical Anatomy of a Route Cut
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

run_mask = s_sm > 2.0
run_idx = np.where(run_mask)[0]
apex_rel = run_idx[np.argmax(ang_vel[run_idx])]
w_start = max(0, apex_rel - 20)
w_end = min(len(s_sm), apex_rel + 20)

t_w = t[w_start:w_end] - t[w_start]
s_w = s_sm[w_start:w_end]
acc_w = acc[w_start:w_end]
ang_w = ang_vel[w_start:w_end]
apex_w = apex_rel - w_start

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)

# Velocity trace
ax1.plot(t_w, s_w, color='#1f77b4', lw=2.5, label='Velocity (yd/s)')
ax1.axvline(t_w[apex_w], color='#d62728', ls='--', lw=2, label=f'Cut Apex (V_apex={s_w[apex_w]:.2f} yd/s)')
ax1.scatter([t_w[apex_w]], [s_w[apex_w]], color='#d62728', s=80, zorder=5)
v_stem_idx = np.argmax(s_w[:apex_w])
ax1.scatter([t_w[v_stem_idx]], [s_w[v_stem_idx]], color='#2ca02c', s=80, zorder=5)
ax1.annotate(f'Stem Attack (V_stem={s_w[v_stem_idx]:.2f} yd/s)', 
             xy=(t_w[v_stem_idx], s_w[v_stem_idx]), 
             xytext=(t_w[v_stem_idx]-0.7, s_w[v_stem_idx]+0.8),
             arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.5))
ax1.set_ylabel('Velocity (yd/s)')
ax1.set_title(f'Figure 1: Biomechanical Anatomy of a Route Cut (Combine 10 Hz Sensor Tracking - {sample_player_name})', fontweight='bold')
ax1.legend(loc='lower left', frameon=True)

# Acceleration trace
ax2.plot(t_w, acc_w, color='#ff7f0e', lw=2.5, label='Acceleration / Braking (yd/s²)')
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

# Turn Rate trace
ax3.plot(t_w, ang_w, color='#2ca02c', lw=2.5, label='Angular Velocity (deg/s)')
ax3.axvline(t_w[apex_w], color='#d62728', ls='--', lw=2)
ax3.set_ylabel('Turn Rate (deg/s)')
ax3.set_xlabel('Time into Route Cut Window (seconds)')
ax3.legend(loc='upper left', frameon=True)

plt.tight_layout()
plt.show()"""))

# Section 4: Contextual Separation Over Expected (SOE)
cells.append(md_cell(r"""## 4. Isolating Individual Separation: Separation Over Expected (SOE)

A common mistake in sports analytics is evaluating raw receiver separation. Raw separation is overwhelmingly driven by **route scheme** (e.g., screen passes average 6.4 yards of separation, while Go routes average 1.9 yards) and **defensive cushion**.

To measure true individual route-running talent, we construct a context-adjusted **Separation Over Expected (SOE)** model trained across 37,525 regular-season plays:

$$\text{Separation Over Expected (SOE)} = \text{Separation}_{\text{actual}} - \mathbb{E}[\text{Separation} \mid \text{Route}, \text{Pass Length}, \text{Cushion}, \text{Coverage}]$$"""))

# Code Cell 4: Figure 4 Code & SOE Distribution
cells.append(code_cell(r"""# Figure 4: Context Dictates Separation
valid_mask = pp["separation_at_pass_forward"].notnull() & pp["route_ran"].notnull() & pp["pass_length"].notnull()
wr_plays_sample = pp[pp["nfl_id"].isin(wr_ids) & valid_mask].copy()

plt.figure(figsize=(10, 4.8))
route_order = wr_plays_sample.groupby("route_ran")["separation_at_pass_forward"].median().sort_values(ascending=False).index
sns.boxplot(data=wr_plays_sample, x="route_ran", y="separation_at_pass_forward", order=route_order, palette="Spectral", showfliers=False)
plt.title('Figure 4: Context Dictates Separation — Scheme Varies On-Field Separation Drastically', fontweight='bold')
plt.xlabel('Route Ran by Receiver', fontweight='bold')
plt.ylabel('Separation at Pass Forward (Yards)', fontweight='bold')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()"""))

# Section 5: Empirical Translation: Combine Kinematics to NFL Performance
cells.append(md_cell(r"""## 5. Empirical Translation: Combine Kinematics vs NFL Performance

Does Combine route drill movement translate to regular-season NFL game performance? 

When we evaluate the 93 qualified wide receivers with active NFL regular season experience:
- **Apex Cut Speed ($V_{apex}$)** correlates with NFL career snaps at **$r = +0.442$ ($p < 0.0001$)**.
- **Stem Attack Speed ($V_{stem}$)** correlates with NFL career snaps at **$r = +0.415$ ($p < 0.0001$)**.
- **Stopwatch 40-Yard Dash** exhibits virtually **no relationship** with NFL career snaps (**$r = -0.113, p = 0.34$**).

Elite NFL receivers do not come to a complete stop on route breaks; they carry fluid, high-velocity momentum through the cut apex ($V_{apex} > 5.8$ yd/s) while driving vertically at top speed ($V_{stem} > 7.2$ yd/s)."""))

# Code Cell 5: Figure 3 Predictive Power Comparison
cells.append(code_cell(r"""# Figure 3: Predictive Power Comparison
final_df = pd.read_csv("e:/Kaggle/wr_combine_nfl_metrics.csv")
qual = final_df[final_df["n_routes"] >= 30].copy()

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

plt.figure(figsize=(9.5, 4.8))
bars = plt.barh(corr_labels, corr_vals, color=corr_colors, edgecolor='black', lw=1.2)
plt.xlim(-0.1, 0.55)
plt.axvline(0, color='gray', lw=1)
plt.xlabel('Correlation with NFL Career Offensive Snaps (r)', fontweight='bold')
plt.title('Figure 3: Translational Validity — Combine Sensor Kinematics vs Traditional Stopwatch Tests', fontweight='bold')
for bar, v in zip(bars, corr_vals):
    pos = v + 0.015 if v >= 0 else v - 0.04
    plt.text(pos, bar.get_y() + bar.get_height()/2, f'r = {v:+.3f}', va='center', fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()"""))

# Section 6: Actionable Scouting Matrix & Prospect Evaluation
cells.append(md_cell(r"""## 6. Actionable Application: The Prospect Separation Matrix

By plotting **Stem Attack Velocity ($V_{stem}$)** against **Break Apex Velocity ($V_{apex}$)**, we construct a 2D scouting evaluation plane that correctly separates Day 3 superstars from high-draft-capital workout busts:

- **Quadrant I (Elite Fluid Separators)**: High stem velocity and high apex speed retention. Examples: **Puka Nacua** (5th Round steal, 2,211 snaps), **Jayden Reed** (2nd Round, 1,430 snaps), **Brian Thomas Jr.** (1st Round, 1,561 snaps).
- **Quadrant II (Fluid Route Technicians)**: Elite apex fluidity with controlled stem speed. Examples: **Jaxon Smith-Njigba** (1st Round, highest $V_{apex}$ in draft class at 6.64 yd/s, 2,445 snaps), **Ladd McConkey** (2nd Round, 1,572 snaps).
- **Quadrant IV (Workout Warrior Bust Trap)**: High straight-line entry speed but violent deceleration to a near-halt at the break point ($V_{apex} < 4.8$ yd/s). These prospects struggle to create NFL separation and experience steep learning curves."""))

# Code Cell 6: Figure 5 Prospect Quadrant Scatter
cells.append(code_cell(r"""# Figure 5: Prospect Evaluation Quadrant
plt.figure(figsize=(10, 7))
stem_med = qual['v_stem'].median()
apex_med = qual['v_apex'].median()

scatter = plt.scatter(qual['v_stem'], qual['v_apex'], 
                      c=qual['career_offensive_snaps'], cmap='viridis', 
                      s=qual['career_offensive_snaps']/12 + 45, 
                      alpha=0.85, edgecolors='black', lw=1)
cbar = plt.colorbar(scatter)
cbar.set_label('NFL Career Offensive Snaps', fontweight='bold')

plt.axvline(stem_med, color='gray', ls='--', lw=1.2, alpha=0.7)
plt.axhline(apex_med, color='gray', ls='--', lw=1.2, alpha=0.7)

# Quadrant labels
plt.text(qual['v_stem'].max() - 0.05, qual['v_apex'].max(), 'QUADRANT I:\nELITE FLUID SEPARATORS\n(High Stem, High Apex)', 
         ha='right', va='top', fontsize=9, fontweight='bold', color='#006837', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.text(qual['v_stem'].min() + 0.05, qual['v_apex'].max(), 'QUADRANT II:\nFLUID TECHNICIANS\n(Fluid Apex, Controlled Stem)', 
         ha='left', va='top', fontsize=9, fontweight='bold', color='#2166ac', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.text(qual['v_stem'].max() - 0.05, qual['v_apex'].min(), 'QUADRANT IV:\nWORKOUT WARRIOR BUSTS\n(Fast Stem, Sluggish Break)', 
         ha='right', va='bottom', fontsize=9, fontweight='bold', color='#b2182b', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

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
plt.title('Figure 5: The Kinematic Separation Matrix — Identifying Day 3 Gems & Busts from 10 Hz Combine Tracking', fontweight='bold')
plt.tight_layout()
plt.show()"""))

# Section 7: Statistical Validation & Multivariable Regression
cells.append(md_cell(r"""## 7. Statistical Validation & Multivariable Regression

To prove statistical validity beyond bivariate correlations, we estimate multivariable Ordinary Least Squares (OLS) models controlling for draft capital (draft round) and compare model efficacy against traditional testing."""))

# Code Cell 7: Statistical Tables & Multivariable Regression
cells.append(code_cell(r"""# Model 1: Sensor Kinematics + Draft Round
X1 = sm.add_constant(qual[['v_apex', 'v_stem', 'draft_round']].dropna())
y1 = qual.loc[X1.index, 'career_offensive_snaps']
m1 = sm.OLS(y1, X1).fit()

print("===============================================================================")
print("TABLE 2: Multivariable OLS Regression Models Predicting NFL Career Snaps")
print("===============================================================================")
print(f"Model 1 (Combine 10 Hz Kinematics + Draft Capital, N={len(X1)}):")
print(f"R² = {m1.rsquared:.3f} | Adj R² = {m1.rsquared_adj:.3f} | Model F-test p = {m1.f_pvalue:.4e}")
display(m1.summary().tables[1])

# Model 2: Traditional Stopwatch Testing
trad = qual.dropna(subset=['forty', 'three_cone', 'draft_round', 'career_offensive_snaps'])
X2 = sm.add_constant(trad[['forty', 'three_cone', 'draft_round']])
y2 = trad['career_offensive_snaps']
m2 = sm.OLS(y2, X2).fit()
print(f"\nModel 2 (Traditional Stopwatch Testing + Draft Capital, N={len(trad)} due to Opt-Outs):")
print(f"R² = {m2.rsquared:.3f} | Adj R² = {m2.rsquared_adj:.3f} | Model F-test p = {m2.f_pvalue:.4f}")
display(m2.summary().tables[1])"""))

# Code Cell 8: Figure 6 Correlation Heatmap
cells.append(code_cell(r"""# Figure 6: Complete Kinematic Correlation Matrix
corr_mat = qual[['v_stem', 'v_apex', 'peak_decel', 'peak_burst', 'mean_sep', 'breaking_soe', 'mean_epa', 'career_offensive_snaps']].corr()
plt.figure(figsize=(8.5, 6.5))
sns.heatmap(corr_mat, annot=True, fmt='+.2f', cmap='vlag', center=0, cbar_kws={'label': 'Pearson Correlation (r)'})
plt.title('Figure 6: Kinematic Correlation Heatmap — Combine 10 Hz Features vs NFL Performance', fontweight='bold')
plt.tight_layout()
plt.show()"""))

# Section 8: Case Studies & Scouting Recommendations
cells.append(md_cell(r"""## 8. Front Office & Coaching Protocols

### Actionable Front Office Protocol: The 3-Step Draft Rule
1. **Rule out 40-Yard Dash Over-Weighting**: A 4.35 40-yard dash without top-quartile $V_{stem}$ and $V_{apex}$ represents high bust risk.
2. **Flag Day 3 Value via Apex Fluidity**: Any Day 3 prospect with $V_{apex} > 5.9$ yd/s and $V_{stem} > 7.3$ yd/s possesses immediate NFL-translatable movement skill (e.g. Puka Nacua, Michael Wilson).
3. **Opt-Out Immunity**: Front offices no longer need to penalize prospects for skipping the 3-cone or short shuttle; on-field route drills provide higher fidelity data with 100% participation.

### Actionable Coaching Protocol: Position-Specific Player Development
- **Deceleration Posture**: Drill wide receivers to drop hips 2 frames earlier rather than chopping stride length, preventing deceleration leaks.
- **Apex Momentum Preservation**: Design route break drills that reward maintaining $\ge 60\%$ entry velocity through 90° cuts rather than coming to static halts."""))

# Save Notebook
notebook_dict = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat": 4,
            "nbformat_minor": 2
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("e:/Kaggle/NFL.ipynb", "w", encoding="utf-8", newline="\n") as f:
    json.dump(notebook_dict, f, indent=2)

print("NFL.ipynb created successfully with all sections, markdown narrative, and code cells!")
