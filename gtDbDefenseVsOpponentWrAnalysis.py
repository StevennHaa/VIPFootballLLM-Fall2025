import numpy as np
import pandas as pd

# ---------------- Config ----------------
OUR_TEAM = "GA TECH"  # matches your CSV exactly
OPPONENTS = ["BOSTON COL", "NC STATE", "GEORGIA", "PITTSBURGH", "SYRACUSE"]
DB_POS = {"CB", "DB"}  # Defensive Backs
WR_POS = {"WR"}  # Wide Receivers
SNAP_MIN = 50
TARGET_MIN = 15  # Minimum targets for WRs

# -------------- Load & Prep --------------
# Defense: GT DBs with > 50 snaps
defense = pd.read_csv("Data/defense_summary.csv")
defense["snap_counts_defense"] = pd.to_numeric(defense["snap_counts_defense"], errors="coerce")
defense["grades_run_defense"] = pd.to_numeric(defense["grades_run_defense"], errors="coerce")
defense["grades_coverage_defense"] = pd.to_numeric(defense["grades_coverage_defense"], errors="coerce")

# Filter GT DBs
is_db = defense["position"].astype(str).str.upper().isin(DB_POS)
idx_db = np.where(is_db & (defense["snap_counts_defense"] > SNAP_MIN))[0]
gt_dbs = defense.iloc[idx_db].query("team_name == @OUR_TEAM").copy()

# Offense: Opponent WRs with > 15 targets
receiving = pd.read_csv("Data/receiving_summary (2).csv")
receiving["targets"] = pd.to_numeric(receiving["targets"], errors="coerce")
receiving["grades_pass_route"] = pd.to_numeric(receiving["grades_pass_route"], errors="coerce")

# Filter opponent WRs
is_wr = receiving["position"].astype(str).str.upper().isin(WR_POS)
idx_wr = np.where(is_wr & (receiving["targets"] > TARGET_MIN))[0]
opp_wrs = receiving.iloc[idx_wr].query("team_name in @OPPONENTS").copy()

# -------------- Helper: opponent WR summaries --------------
def opponent_wr_room(df_opp: pd.DataFrame):
    """
    Returns a tuple:
      (room_avg_pass_rounded, room_avg_run_rounded, dataframe_of_all_WRs_sorted_by_targets_with_rounded_grades)
    """
    if df_opp.empty:
        return (np.nan, np.nan, df_opp)

    # For pass grades, weight by targets
    w_pass = df_opp["targets"].clip(lower=0).fillna(0)
    v_pass = df_opp["grades_pass_route"]
    room_avg_pass = (v_pass.mul(w_pass)).sum() / (w_pass.sum() if w_pass.sum() > 0 else len(df_opp))
    room_avg_pass = round(float(room_avg_pass), 2)

    # For run grades, use simple average (WRs don't have run grades in receiving data)
    # We'll use pass route grade as proxy for overall offensive grade
    room_avg_run = room_avg_pass  # Using same value as pass for now

    wrs_sorted = df_opp.sort_values("targets", ascending=False).copy()
    wrs_sorted["grades_pass_route"] = wrs_sorted["grades_pass_route"].round(2)
    wrs_sorted["targets"] = wrs_sorted["targets"].astype(int)
    return (room_avg_pass, room_avg_run, wrs_sorted)

# Precompute per-opponent WR room + list
opp_wr_summaries = {}
for opp in OPPONENTS:
    opp_wr_df = opp_wrs[opp_wrs["team_name"] == opp]
    opp_wr_summaries[opp] = opponent_wr_room(opp_wr_df)

# -------------- Build formatted output + CSV --------------
rows = []  # for CSV

print("=" * 80)
print("GT DB DEFENSE vs OPPONENT WR OFFENSE MATCHUPS")
print("=" * 80)

for opp in OPPONENTS:
    wr_pass_avg, wr_run_avg, wrs_sorted = opp_wr_summaries.get(opp, (np.nan, np.nan, pd.DataFrame()))

    print(f"\n{OUR_TEAM} DBs vs {opp} WRs")  # section header

    for _, db in gt_dbs.iterrows():
        db_name = db["player"]
        db_run_def = round(float(db["grades_run_defense"]), 2) if pd.notna(db["grades_run_defense"]) else np.nan
        db_pass_def = round(float(db["grades_coverage_defense"]), 2) if pd.notna(db["grades_coverage_defense"]) else np.nan

        # Team-level matchup (DB vs WR room avg)
        delta_run = round(db_run_def - wr_run_avg, 2) if pd.notna(db_run_def) and pd.notna(wr_run_avg) else np.nan
        delta_pass = round(db_pass_def - wr_pass_avg, 2) if pd.notna(db_pass_def) and pd.notna(wr_pass_avg) else np.nan
        
        print(f"{db_name} vs {opp} Offense — DB Run-D: {db_run_def:.2f} vs WR Avg: {wr_run_avg if pd.notna(wr_run_avg) else float('nan'):.2f} (Δ {delta_run if pd.notna(delta_run) else float('nan'):.2f})")
        print(f"{db_name} vs {opp} Offense — DB Pass-D: {db_pass_def:.2f} vs WR Avg: {wr_pass_avg if pd.notna(wr_pass_avg) else float('nan'):.2f} (Δ {delta_pass if pd.notna(delta_pass) else float('nan'):.2f})")

        rows.append({
            "gt_team": OUR_TEAM,
            "opponent": opp,
            "type": "vs_offense_avg",
            "gt_db_name": db_name,
            "gt_db_position": "DB",
            "gt_db_run_defense": db_run_def,
            "gt_db_pass_defense": db_pass_def,
            "opp_wr_pass_grade": wr_pass_avg,
            "opp_wr_run_grade": wr_run_avg,
            "opp_wr_name": None,
            "opp_wr_targets": None,
            "delta_run": delta_run,
            "delta_pass": delta_pass
        })

        # One-on-one vs EVERY qualifying WR (sorted by targets)
        if not wrs_sorted.empty:
            for rank, (_, wr) in enumerate(wrs_sorted.iterrows(), start=1):
                wr_name = str(wr["player"])
                wr_pass_grade = float(wr["grades_pass_route"]) if pd.notna(wr["grades_pass_route"]) else np.nan
                wr_targets = int(wr["targets"]) if pd.notna(wr["targets"]) else 0
                wr_run_grade = wr_pass_grade  # Using pass grade as proxy for run grade
                
                delta_run_wr = round(db_run_def - wr_run_grade, 2) if pd.notna(db_run_def) and pd.notna(wr_run_grade) else np.nan
                delta_pass_wr = round(db_pass_def - wr_pass_grade, 2) if pd.notna(db_pass_def) and pd.notna(wr_pass_grade) else np.nan

                # Print with a clear label (rank by targets)
                print(f"{db_name} vs {opp} Offense (WR #{rank} by targets — {wr_name}) — Run-D: {db_run_def:.2f} vs {wr_run_grade if pd.notna(wr_run_grade) else float('nan'):.2f} (Δ {delta_run_wr if pd.notna(delta_run_wr) else float('nan'):.2f})")
                print(f"{db_name} vs {opp} Offense (WR #{rank} by targets — {wr_name}) — Pass-D: {db_pass_def:.2f} vs {wr_pass_grade if pd.notna(wr_pass_grade) else float('nan'):.2f} (Δ {delta_pass_wr if pd.notna(delta_pass_wr) else float('nan'):.2f})")

                rows.append({
                    "gt_team": OUR_TEAM,
                    "opponent": opp,
                    "type": f"vs_wr_rank_{rank}",
                    "gt_db_name": db_name,
                    "gt_db_position": "DB",
                    "gt_db_run_defense": db_run_def,
                    "gt_db_pass_defense": db_pass_def,
                    "opp_wr_pass_grade": wr_pass_grade,
                    "opp_wr_run_grade": wr_run_grade,
                    "opp_wr_name": wr_name,
                    "opp_wr_targets": wr_targets,
                    "delta_run": delta_run_wr,
                    "delta_pass": delta_pass_wr
                })

# Save a tidy CSV with every printed row
out_df = pd.DataFrame(rows)
out_df.to_csv("gt_db_defense_vs_opponent_wr_offense_breakdown.csv", index=False)
print(f"\nSaved -> gt_db_defense_vs_opponent_wr_offense_breakdown.csv")
print(f"Total GT DB vs Opponent WR matchups analyzed: {len(rows)}")
