import numpy as np
import pandas as pd

# ---------------- Config ----------------
OUR_TEAM = "GA TECH"  # matches your CSV exactly
OPPONENTS = ["BOSTON COL", "NC STATE", "GEORGIA", "PITTSBURGH", "SYRACUSE"]
LB_POS = {"LB", "ILB", "OLB"}  # Linebackers
TE_POS = {"TE"}  # Tight Ends
SNAP_MIN = 50
TARGET_MIN = 1  # Minimum targets for TEs

# -------------- Load & Prep --------------
# Defense: GT LBs with > 50 snaps
defense = pd.read_csv("Data/defense_summary.csv")
defense["snap_counts_defense"] = pd.to_numeric(defense["snap_counts_defense"], errors="coerce")
defense["grades_run_defense"] = pd.to_numeric(defense["grades_run_defense"], errors="coerce")
defense["grades_coverage_defense"] = pd.to_numeric(defense["grades_coverage_defense"], errors="coerce")

# Filter GT LBs
is_lb = defense["position"].astype(str).str.upper().isin(LB_POS)
idx_lb = np.where(is_lb & (defense["snap_counts_defense"] > SNAP_MIN))[0]
gt_lbs = defense.iloc[idx_lb].query("team_name == @OUR_TEAM").copy()

# Offense: Opponent TEs with > 1 targets
receiving = pd.read_csv("Data/receiving_summary (2).csv")
receiving["targets"] = pd.to_numeric(receiving["targets"], errors="coerce")
receiving["grades_pass_route"] = pd.to_numeric(receiving["grades_pass_route"], errors="coerce")

# Filter opponent TEs
is_te = receiving["position"].astype(str).str.upper().isin(TE_POS)
idx_te = np.where(is_te & (receiving["targets"] > TARGET_MIN))[0]
opp_tes = receiving.iloc[idx_te].query("team_name in @OPPONENTS").copy()

# -------------- Helper: opponent TE summaries --------------
def opponent_te_room(df_opp: pd.DataFrame):
    """
    Returns a tuple:
      (room_avg_pass_rounded, room_avg_run_rounded, dataframe_of_all_TEs_sorted_by_targets_with_rounded_grades)
    """
    if df_opp.empty:
        return (np.nan, np.nan, df_opp)

    # For pass grades, weight by targets
    w_pass = df_opp["targets"].clip(lower=0).fillna(0)
    v_pass = df_opp["grades_pass_route"]
    room_avg_pass = (v_pass.mul(w_pass)).sum() / (w_pass.sum() if w_pass.sum() > 0 else len(df_opp))
    room_avg_pass = round(float(room_avg_pass), 2)

    # For run grades, use simple average (TEs don't have run grades in receiving data)
    # We'll use pass route grade as proxy for overall offensive grade
    room_avg_run = room_avg_pass  # Using same value as pass for now

    tes_sorted = df_opp.sort_values("targets", ascending=False).copy()
    tes_sorted["grades_pass_route"] = tes_sorted["grades_pass_route"].round(2)
    tes_sorted["targets"] = tes_sorted["targets"].astype(int)
    return (room_avg_pass, room_avg_run, tes_sorted)

# Precompute per-opponent TE room + list
opp_te_summaries = {}
for opp in OPPONENTS:
    opp_te_df = opp_tes[opp_tes["team_name"] == opp]
    opp_te_summaries[opp] = opponent_te_room(opp_te_df)

# -------------- Build formatted output + CSV --------------
rows = []  # for CSV

print("=" * 80)
print("GT LB DEFENSE vs OPPONENT TE OFFENSE MATCHUPS")
print("=" * 80)

for opp in OPPONENTS:
    te_pass_avg, te_run_avg, tes_sorted = opp_te_summaries.get(opp, (np.nan, np.nan, pd.DataFrame()))

    print(f"\n{OUR_TEAM} LBs vs {opp} TEs")  # section header

    for _, lb in gt_lbs.iterrows():
        lb_name = lb["player"]
        lb_run_def = round(float(lb["grades_run_defense"]), 2) if pd.notna(lb["grades_run_defense"]) else np.nan
        lb_pass_def = round(float(lb["grades_coverage_defense"]), 2) if pd.notna(lb["grades_coverage_defense"]) else np.nan

        # Team-level matchup (LB vs TE room avg)
        delta_run = round(lb_run_def - te_run_avg, 2) if pd.notna(lb_run_def) and pd.notna(te_run_avg) else np.nan
        delta_pass = round(lb_pass_def - te_pass_avg, 2) if pd.notna(lb_pass_def) and pd.notna(te_pass_avg) else np.nan
        
        print(f"{lb_name} vs {opp} Offense — LB Run-D: {lb_run_def:.2f} vs TE Avg: {te_run_avg if pd.notna(te_run_avg) else float('nan'):.2f} (Δ {delta_run if pd.notna(delta_run) else float('nan'):.2f})")
        print(f"{lb_name} vs {opp} Offense — LB Pass-D: {lb_pass_def:.2f} vs TE Avg: {te_pass_avg if pd.notna(te_pass_avg) else float('nan'):.2f} (Δ {delta_pass if pd.notna(delta_pass) else float('nan'):.2f})")

        rows.append({
            "gt_team": OUR_TEAM,
            "opponent": opp,
            "type": "vs_offense_avg",
            "gt_lb_name": lb_name,
            "gt_lb_position": "LB",
            "gt_lb_run_defense": lb_run_def,
            "gt_lb_pass_defense": lb_pass_def,
            "opp_te_pass_grade": te_pass_avg,
            "opp_te_run_grade": te_run_avg,
            "opp_te_name": None,
            "opp_te_targets": None,
            "delta_run": delta_run,
            "delta_pass": delta_pass
        })

        # One-on-one vs EVERY qualifying TE (sorted by targets)
        if not tes_sorted.empty:
            for rank, (_, te) in enumerate(tes_sorted.iterrows(), start=1):
                te_name = str(te["player"])
                te_pass_grade = float(te["grades_pass_route"]) if pd.notna(te["grades_pass_route"]) else np.nan
                te_targets = int(te["targets"]) if pd.notna(te["targets"]) else 0
                te_run_grade = te_pass_grade  # Using pass grade as proxy for run grade
                
                delta_run_te = round(lb_run_def - te_run_grade, 2) if pd.notna(lb_run_def) and pd.notna(te_run_grade) else np.nan
                delta_pass_te = round(lb_pass_def - te_pass_grade, 2) if pd.notna(lb_pass_def) and pd.notna(te_pass_grade) else np.nan

                # Print with a clear label (rank by targets)
                print(f"{lb_name} vs {opp} Offense (TE #{rank} by targets — {te_name}) — Run-D: {lb_run_def:.2f} vs {te_run_grade if pd.notna(te_run_grade) else float('nan'):.2f} (Δ {delta_run_te if pd.notna(delta_run_te) else float('nan'):.2f})")
                print(f"{lb_name} vs {opp} Offense (TE #{rank} by targets — {te_name}) — Pass-D: {lb_pass_def:.2f} vs {te_pass_grade if pd.notna(te_pass_grade) else float('nan'):.2f} (Δ {delta_pass_te if pd.notna(delta_pass_te) else float('nan'):.2f})")

                rows.append({
                    "gt_team": OUR_TEAM,
                    "opponent": opp,
                    "type": f"vs_te_rank_{rank}",
                    "gt_lb_name": lb_name,
                    "gt_lb_position": "LB",
                    "gt_lb_run_defense": lb_run_def,
                    "gt_lb_pass_defense": lb_pass_def,
                    "opp_te_pass_grade": te_pass_grade,
                    "opp_te_run_grade": te_run_grade,
                    "opp_te_name": te_name,
                    "opp_te_targets": te_targets,
                    "delta_run": delta_run_te,
                    "delta_pass": delta_pass_te
                })

# Save a tidy CSV with every printed row
out_df = pd.DataFrame(rows)
out_df.to_csv("gt_lb_defense_vs_opponent_te_offense_breakdown.csv", index=False)
print(f"\nSaved -> gt_lb_defense_vs_opponent_te_offense_breakdown.csv")
print(f"Total GT LB vs Opponent TE matchups analyzed: {len(rows)}")
