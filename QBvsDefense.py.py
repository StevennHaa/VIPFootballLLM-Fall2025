#!/usr/bin/env python3
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

# --------------- Config (edit these) ---------------
OUR_TEAM   = "GA TECH"
OPPONENTS  = ["BOSTON COL", "NC STATE", "GEORGIA", "PITTSBURGH", "SYRACUSE"]
QB_POS     = {"QB"}
LB_POS     = {"LB", "ILB", "OLB"}
S_POS      = {"S", "SS", "FS"}
SNAP_MIN   = 50
ATT_MIN_R  = 10   # QB rushing attempt threshold
ATT_MIN_P  = 50   # QB passing dropback/attempt threshold (best estimate)

def coerce_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def weighted_avg(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce").clip(lower=0).fillna(0)
    if weights.sum() > 0:
        return float((values * weights).sum() / weights.sum())
    vals = values.dropna()
    return float(vals.mean()) if len(vals) else np.nan

def round2(x: Any) -> Any:
    try:
        return round(float(x), 2)
    except Exception:
        return np.nan

# --------------- Load Offense (QB) ---------------
# rushing_summary.csv for QB run grades/attempts
rush = pd.read_csv("rushing_summary.csv")
for col in ["attempts", "grades_run"]:
    if col in rush.columns:
        rush[col] = coerce_num(rush[col])

# passing_summary.csv for QB pass grades/attempts
# Expecting cols like: 'dropbacks' or 'attempts', and a pass grade 'grades_pass'
try:
    pas = pd.read_csv("passing_summary.csv")
except FileNotFoundError:
    # Create empty fallback if user does not have a separate file; we won't block the run
    pas = pd.DataFrame(columns=["player","team_name","position","attempts","dropbacks","grades_pass"])
for col in ["attempts", "dropbacks", "grades_pass"]:
    if col in pas.columns:
        pas[col] = coerce_num(pas[col])

# choose QB pass attempt/dropback col
PASS_ATT_COL = first_existing_column(pas, ["dropbacks", "attempts"])

# filter GT QBs
is_qb_r = rush["position"].astype(str).str.upper().isin(QB_POS)
gt_qbs_run = rush[(rush["team_name"] == OUR_TEAM) & is_qb_r].copy()
gt_qbs_run = gt_qbs_run.rename(columns={"grades_run": "qb_run_grade", "attempts": "qb_rush_attempts"})

if PASS_ATT_COL is not None and "player" in pas.columns:
    is_qb_p = pas["position"].astype(str).str.upper().isin(QB_POS)
    gt_qbs_pass = pas[(pas["team_name"] == OUR_TEAM) & is_qb_p].copy()
    gt_qbs_pass = gt_qbs_pass.rename(columns={"grades_pass": "qb_pass_grade", PASS_ATT_COL: "qb_pass_atts"})
else:
    gt_qbs_pass = pd.DataFrame(columns=["player","qb_pass_grade","qb_pass_atts"])

# merge run and pass on player name
gt_qbs = pd.merge(gt_qbs_run, gt_qbs_pass[["player","qb_pass_grade","qb_pass_atts"]] if len(gt_qbs_pass)>0 else gt_qbs_pass,
                  on="player", how="left")

# apply thresholds
if "qb_rush_attempts" in gt_qbs.columns:
    qb_mask_r = gt_qbs["qb_rush_attempts"] >= ATT_MIN_R
else:
    qb_mask_r = True
if "qb_pass_atts" in gt_qbs.columns:
    qb_mask_p = gt_qbs["qb_pass_atts"].fillna(0) >= ATT_MIN_P
else:
    qb_mask_p = True

gt_qbs = gt_qbs[qb_mask_r | qb_mask_p].copy()

# --------------- Load Defense ---------------
defn = pd.read_csv("defense_summary.csv")
for col in ["snap_counts_defense", "grades_run_defense", "grades_coverage_defense", "grades_defense_overall", "grades_coverage", "grades_defense"]:
    if col in defn.columns:
        defn[col] = coerce_num(defn[col])

# choose columns
RUN_DEF_COL = "grades_run_defense" if "grades_run_defense" in defn.columns else None
COV_COL     = "grades_coverage_defense" if "grades_coverage_defense" in defn.columns else ("grades_coverage" if "grades_coverage" in defn.columns else None)
OVR_DEF_COL = "grades_defense_overall" if "grades_defense_overall" in defn.columns else ("grades_defense" if "grades_defense" in defn.columns else None)

if RUN_DEF_COL is None:
    RUN_DEF_COL = "run_def_proxy"
    defn[RUN_DEF_COL] = np.nan
if COV_COL is None:
    COV_COL = "cov_proxy"
    defn[COV_COL] = np.nan

# compute per-opponent team overall defense (weighted avg of overall if present; else mean of run+coverage where available)
opp_team_def_overall: Dict[str, float] = {}
opp_lb_run_avg: Dict[str, float] = {}
opp_s_run_avg: Dict[str, float] = {}

for opp in OPPONENTS:
    team = defn[defn["team_name"] == opp].copy()
    team_w = team["snap_counts_defense"].clip(lower=0).fillna(0)

    # overall defense
    if OVR_DEF_COL and OVR_DEF_COL in team:
        ovr = weighted_avg(team[OVR_DEF_COL], team_w)
    else:
        # fallback: average available components (run + coverage) weighted
        ovr = np.nanmean([weighted_avg(team[RUN_DEF_COL], team_w), weighted_avg(team[COV_COL], team_w)])
    opp_team_def_overall[opp] = round2(ovr)

    # LB run avg
    lbs = team[team["position"].astype(str).str.upper().isin(LB_POS)].copy()
    lb_run = weighted_avg(lbs[RUN_DEF_COL], lbs["snap_counts_defense"]) if not lbs.empty else np.nan
    opp_lb_run_avg[opp] = round2(lb_run)

    # S run avg
    ss = team[team["position"].astype(str).str.upper().isin(S_POS)].copy()
    s_run = weighted_avg(ss[RUN_DEF_COL], ss["snap_counts_defense"]) if not ss.empty else np.nan
    opp_s_run_avg[opp] = round2(s_run)

# --------------- Build output rows ---------------
rows = []
for opp in OPPONENTS:
    team_ovr = opp_team_def_overall.get(opp, np.nan)
    lb_run   = opp_lb_run_avg.get(opp, np.nan)
    s_run    = opp_s_run_avg.get(opp, np.nan)

    for _, qb in gt_qbs.iterrows():
        qb_name   = qb["player"]
        qb_run_g  = round2(qb.get("qb_run_grade", np.nan))
        qb_pass_g = round2(qb.get("qb_pass_grade", np.nan))

        # Deltas requested:
        # 1) QB vs overall defense (we'll compare QB pass grade to team overall defense if pass_g exists; else run to overall)
        if pd.notna(qb_pass_g):
            delta_qb_vs_def = round2(qb_pass_g - team_ovr) if pd.notna(team_ovr) else np.nan
        else:
            delta_qb_vs_def = round2(qb_run_g - team_ovr) if pd.notna(team_ovr) else np.nan

        # 2) QB Run vs LB Run
        delta_run_vs_lb = round2(qb_run_g - lb_run) if (pd.notna(qb_run_g) and pd.notna(lb_run)) else np.nan

        # 3) QB Run vs S Run
        delta_run_vs_s  = round2(qb_run_g - s_run) if (pd.notna(qb_run_g) and pd.notna(s_run)) else np.nan

        rows.append({
            "gt_team": OUR_TEAM,
            "opponent": opp,
            "qb_player": qb_name,
            "qb_rush_attempts": int(qb["qb_rush_attempts"]) if pd.notna(qb.get("qb_rush_attempts", np.nan)) else np.nan,
            "qb_run_grade": qb_run_g,
            "qb_pass_atts": int(qb["qb_pass_atts"]) if pd.notna(qb.get("qb_pass_atts", np.nan)) else np.nan,
            "qb_pass_grade": qb_pass_g,
            "opp_def_overall_avg": team_ovr,
            "opp_lb_run_avg": lb_run,
            "opp_s_run_avg": s_run,
            "delta_qb_vs_def_overall": delta_qb_vs_def,
            "delta_qb_run_minus_lb_run": delta_run_vs_lb,
            "delta_qb_run_minus_s_run": delta_run_vs_s
        })

out_df = pd.DataFrame(rows)
out_df.to_csv("QBs_vs_Defense.csv", index=False)
print("Saved -> QBs_vs_Defense.csv")
