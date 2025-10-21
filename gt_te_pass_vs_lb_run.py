
import numpy as np
import pandas as pd

# ---------------- Config ----------------
OUR_TEAM = "GA TECH"  # matches your CSV exactly
OPPONENTS = ["BOSTON COL", "NC STATE", "GEORGIA", "PITTSBURGH", "SYRACUSE"]
LB_POS = {"LB", "ILB", "OLB"}  # Linebackers
SNAP_MIN = 50
TARGET_MIN = 1  # Minimum targets for TEs

# -------------- Load & Prep --------------
# Offense: GT TEs with > 1 targets (pass metric = grades_pass_route)
receiving = pd.read_csv("Data/receiving_summary (2).csv")
receiving["targets"] = pd.to_numeric(receiving["targets"], errors="coerce")
receiving["grades_pass_route"] = pd.to_numeric(receiving["grades_pass_route"], errors="coerce")

# Filter GT TEs with sufficient targets
gt_tes = (
    receiving[(receiving["team_name"] == OUR_TEAM) & 
              (receiving["position"].astype(str).str.upper() == "TE") & 
              (receiving["targets"] > TARGET_MIN)]
    .loc[:, ["player", "position", "team_name", "targets", "grades_pass_route"]]
    .copy()
)

# Defense: Opponent LBs with > 50 snaps (run defense grade)
defense = pd.read_csv("Data/defense_summary.csv")
defense["snap_counts_defense"] = pd.to_numeric(defense["snap_counts_defense"], errors="coerce")
defense["grades_run_defense"] = pd.to_numeric(defense["grades_run_defense"], errors="coerce")

# Filter LBs
is_lb = defense["position"].astype(str).str.upper().isin(LB_POS)
idx_lb = np.where(is_lb & (defense["snap_counts_defense"] > SNAP_MIN))[0]
opp_lbs = defense.iloc[idx_lb].query("team_name in @OPPONENTS").copy()

# -------------- Helper: opponent LB run room --------------
def opponent_lb_room_run(df_opp: pd.DataFrame):
    """
    Returns a tuple:
      (room_avg_run_rounded, dataframe_of_all_LBs_sorted_by_snaps_with_rounded_grades)
    """
    if df_opp.empty:
        return (np.nan, df_opp)

    w = df_opp["snap_counts_defense"].clip(lower=0).fillna(0)
    v_run = df_opp["grades_run_defense"]
    room_avg_run = (v_run.mul(w)).sum() / (w.sum() if w.sum() > 0 else len(df_opp))
    room_avg_run = round(float(room_avg_run), 2)

    lbs_sorted = df_opp.sort_values("snap_counts_defense", ascending=False).copy()
    lbs_sorted["grades_run_defense"] = lbs_sorted["grades_run_defense"].round(2)
    lbs_sorted["snap_counts_defense"] = lbs_sorted["snap_counts_defense"].astype(int)
    return (room_avg_run, lbs_sorted)

# Precompute per-opponent LB room + list
opp_lb_summaries = {}
for opp in OPPONENTS:
    opp_df = opp_lbs[opp_lbs["team_name"] == opp]
    opp_lb_summaries[opp] = opponent_lb_room_run(opp_df)

# -------------- Build formatted output + CSV --------------
rows = []  # for CSV

print("=" * 80)
print("GT TE PASS (grades_pass_route) vs OPP LB RUN-D (grades_run_defense) MATCHUPS")
print("=" * 80)

for opp in OPPONENTS:
    lb_run_avg, lbs_sorted = opp_lb_summaries.get(opp, (np.nan, pd.DataFrame()))

    print(f"\n{OUR_TEAM} TEs vs {opp} Linebackers (Run Defense)")  # section header

    for _, r in gt_tes.iterrows():
        player = r["player"]
        te_pass = round(float(r["grades_pass_route"]), 2) if pd.notna(r["grades_pass_route"]) else np.nan

        # Team-level matchup (TE pass vs LB room run avg)
        delta_pass_vs_run = round(te_pass - lb_run_avg, 2) if pd.notna(te_pass) and pd.notna(lb_run_avg) else np.nan

        print(f"{player} vs {opp} Defense — TE Pass: {te_pass:.2f} vs LB Avg Run-D: {lb_run_avg if pd.notna(lb_run_avg) else float('nan'):.2f} (Δ {delta_pass_vs_run if pd.notna(delta_pass_vs_run) else float('nan'):.2f})")

        rows.append({
            "gt_team": OUR_TEAM,
            "opponent": opp,
            "type": "vs_defense_avg",
            "player": player,
            "player_position": "TE",
            "player_pass_grade": te_pass,
            "opp_lb_run_grade": lb_run_avg,
            "opp_lb_name": None,
            "opp_lb_snaps": None,
            "delta_pass_vs_run": delta_pass_vs_run
        })

        # One-on-one vs EVERY qualifying LB (sorted by snaps)
        if not lbs_sorted.empty:
            for rank, (_, lb) in enumerate(lbs_sorted.iterrows(), start=1):
                lb_name = str(lb["player"])
                lb_run_grade = float(lb["grades_run_defense"]) if pd.notna(lb["grades_run_defense"]) else np.nan
                lb_snaps = int(lb["snap_counts_defense"]) if pd.notna(lb["snap_counts_defense"]) else 0
                delta_pass_vs_run_lb = round(te_pass - lb_run_grade, 2) if pd.notna(te_pass) and pd.notna(lb_run_grade) else np.nan

                print(f"{player} vs {opp} Defense (LB #{rank} by snaps — {lb_name}) — TE Pass: {te_pass:.2f} vs LB Run-D: {lb_run_grade if pd.notna(lb_run_grade) else float('nan'):.2f} (snaps {lb_snaps}, Δ {delta_pass_vs_run_lb if pd.notna(delta_pass_vs_run_lb) else float('nan'):.2f})")

                rows.append({
                    "gt_team": OUR_TEAM,
                    "opponent": opp,
                    "type": f"vs_lb_rank_{rank}",
                    "player": player,
                    "player_position": "TE",
                    "player_pass_grade": te_pass,
                    "opp_lb_run_grade": lb_run_grade,
                    "opp_lb_name": lb_name,
                    "opp_lb_snaps": lb_snaps,
                    "delta_pass_vs_run": delta_pass_vs_run_lb
                })

# Save a tidy CSV with every printed row
out_df = pd.DataFrame(rows)
out_df.to_csv("gt_te_pass_vs_lb_run_breakdown.csv", index=False)
print(f"\nSaved -> gt_te_pass_vs_lb_run_breakdown.csv")
print(f"Total TE vs LB (run) matchups analyzed: {len(rows)}")
