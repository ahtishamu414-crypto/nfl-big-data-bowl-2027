# NFL Big Data Bowl 2027: Beyond the Stopwatch

> **Combine to NFL Performance: Deconstructing Route Break Biomechanics to Predict NFL Separation and True Receiver Value**

[![NFL Big Data Bowl 2027](https://img.shields.io/badge/NFL%20Big%20Data%20Bowl-2027-blue.svg)](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

This repository contains the complete analytical framework, statistical models, visualizations, and competition submission writeup for the **2027 NFL Big Data Bowl**. 

Traditional Combine evaluation relies heavily on straight-line stopwatch testing (the 40-yard dash) and agility tests (3-cone drill, 20-yard short shuttle). However:
1. **The Opt-Out Crisis**: Over **71% of wide receivers boycotted the 3-cone drill** and **64% skipped the short shuttle** at recent Combines.
2. **Linear Track Bias**: Straight-line 40 times exhibit virtually **zero predictive correlation** with future NFL career offensive volume ($r = -0.113, p = 0.34$).

By extracting **10 Hz RFID shoulder-pad sensor tracking** from on-field position drills (`COMEBACK`, `DAGGER`, `CURL`, `SLANT`, `POST_CORNER`, `SPEED_OUT`), we achieve **100% prospect coverage** across all 106 wide receiver prospects and uncover novel biomechanical signals that strongly link to regular-season NFL game performance ($V_{apex}: r = +0.442, p < 0.0001$; $V_{stem}: r = +0.415, p < 0.0001$).

---

## 📂 Repository Structure

```text
├── NFL.ipynb                      # Complete, executable Jupyter Notebook with full analysis & outputs
├── SUBMISSION_WRITEUP.md          # Official Big Data Bowl paper (<2,000 words, <10 figures/tables)
├── wr_combine_nfl_metrics.csv     # Extracted 10 Hz Combine metrics merged with NFL game metrics
├── figures/                       # High-resolution publication-quality visuals
│   ├── fig1_route_cut_anatomy.png
│   ├── fig2_opt_out_crisis.png
│   ├── fig3_predictive_power_comparison.png
│   ├── fig4_route_context_separation.png
│   ├── fig5_prospect_quadrant.png
│   └── fig6_kinematic_heatmap.png
├── generate_figures.py            # Reproducible script generating all 6 figures
├── merge_and_eval.py              # Full feature extraction and regression pipeline
└── .gitignore                     # Ignores virtual environments, credentials, and temp files
```

---

## 🔬 Core Kinematic Metrics

From each route drill attempt at 10 Hz, we isolate:
- **Stem Attack Velocity ($V_{stem}$)**: Peak speed approaching the break, quantifying how aggressively the receiver threatens the defender's cushion vertically.
- **Break Apex Velocity ($V_{apex}$)**: Speed maintained at the turning apex without deceleration leaks.
- **Peak Deceleration ($|\text{Decel}_{peak}|$ )**: Maximum braking force into the plant foot ($yd/s^2$).
- **Exit Burst Acceleration ($a_{exit}$)**: Instantaneous directional re-acceleration exiting the cut ($yd/s^2$).

---

## 📊 Key Results

| Metric | Modality | Coverage (%) | Correlation with NFL Snaps ($r$) | Statistical Significance |
| :--- | :--- | :---: | :---: | :---: |
| **Break Apex Velocity ($V_{apex}$)** | **10 Hz Sensor Tracking** | **100.0%** | **+0.442** | **p < 0.0001 \*\*\*** |
| **Stem Attack Velocity ($V_{stem}$)** | **10 Hz Sensor Tracking** | **100.0%** | **+0.415** | **p < 0.0001 \*\*\*** |
| Target EPA ($V_{stem}$ coefficient) | Linear Model | 100.0% | +0.130 EPA/tgt | p = 0.036 \*\* |
| 40-Yard Dash | Stopwatch | 89.6% | -0.113 | p = 0.341 (Not Sig) |
| 3-Cone Drill | Stopwatch | 29.2% (71% Opt-Out) | -0.520 | Severe Missingness Bias |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Dependencies: `pandas`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`

### Setup
```bash
# Clone the repository
git clone <YOUR_REPO_URL>
cd <REPO_DIRECTORY>

# Install dependencies
pip install -r requirements.txt  # or: pip install pandas numpy scipy scikit-learn statsmodels matplotlib seaborn
```

### Reproducing the Analysis
You can open and execute [NFL.ipynb](NFL.ipynb) directly in Jupyter, VS Code, or Kaggle Notebooks.
```bash
python generate_figures.py
```
