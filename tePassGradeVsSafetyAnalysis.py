import numpy as np
import pandas as pd

# ---------------- Config ----------------
OUR_TEAM = "GA TECH"  # matches your CSV exactly
OPPONENTS = ["BOSTON COL", "NC STATE", "GEORGIA", "PITTSBURGH", "SYRACUSE"]
S_POS = {"S"}  # Safeties
SNAP_MIN = 50
TARGET_MIN = 1  # Minimum targets for TEs (lowered since TEs typically have fewer targets)

# -------------- Load & Prep --------------
# Offense: GT TEs with > 1 targets
receiving = pd.read_csv("Data/receiving_summary (2).csv")
receiving["targets"] = pd.to_numeric(receiving["targets"], errors="coerce")
receiving["grades_pass_route"] = pd.to_numeric(receiving["grades_pass_route"], errors="coerce")

# Load TE run grades from blocking file
blocking = pd.read_csv("Data/offense_blocking.csv")
blocking["grades_run_block"] = pd.to_numeric(blocking["grades_run_block"], errors="coerce")

# Filter GT TEs with sufficient targets
receiving_tes = (
    receiving[(receiving["team_name"] == OUR_TEAM) & 
              (receiving["position"] == "TE") & 
              (receiving["targets"] > TARGET_MIN)]
    .loc[:, ["player", "position", "team_name", "targets", "grades_pass_route"]]
    .copy()
)

# Get TE run grades from blocking file
blocking_tes = (
    blocking[(blocking["team_name"] == OUR_TEAM) & 
             (blocking["position"] == "TE")]
    .loc[:, ["player", "grades_run_block"]]
    .copy()
)

# Merge receiving and blocking data for TEs
gt_tes = receiving_tes.merge(blocking_tes, on="player", how="left")

# Defense: Opponent Safeties with > 50 snaps
defense = pd.read_csv("Data/defense_summary.csv")
defense["snap_counts_defense"] = pd.to_numeric(defense["snap_counts_defense"], errors="coerce")
defense["grades_coverage_defense"] = pd.to_numeric(defense["grades_coverage_defense"], errors="coerce")
defense["grades_run_defense"] = pd.to_numeric(defense["grades_run_defense"], errors="coerce")

# Filter Safeties
is_s = defense["position"].astype(str).str.upper().isin(S_POS)
idx_s = np.where(is_s & (defense["snap_counts_defense"] > SNAP_MIN))[0]
opp_safeties = defense.iloc[idx_s].query("team_name in @OPPONENTS").copy()

# -------------- Helper: opponent summaries --------------
def opponent_safety_room(df_opp: pd.DataFrame):
    """
    Returns a tuple:
      (room_avg_coverage_rounded, room_avg_run_rounded, dataframe_of_all_Safeties_sorted_by_snaps_with_rounded_grades)
    """
    if df_opp.empty:
        return (np.nan, np.nan, df_opp)

    w = df_opp["snap_counts_defense"].clip(lower=0).fillna(0)
    
    # Coverage grades
    v_coverage = df_opp["grades_coverage_defense"]
    room_avg_coverage = (v_coverage.mul(w)).sum() / (w.sum() if w.sum() > 0 else len(df_opp))
    room_avg_coverage = round(float(room_avg_coverage), 2)
    
    # Run defense grades
    v_run = df_opp["grades_run_defense"]
    room_avg_run = (v_run.mul(w)).sum() / (w.sum() if w.sum() > 0 else len(df_opp))
    room_avg_run = round(float(room_avg_run), 2)

    safeties_sorted = df_opp.sort_values("snap_counts_defense", ascending=False).copy()
    safeties_sorted["grades_coverage_defense"] = safeties_sorted["grades_coverage_defense"].round(2)
    safeties_sorted["grades_run_defense"] = safeties_sorted["grades_run_defense"].round(2)
    safeties_sorted["snap_counts_defense"] = safeties_sorted["snap_counts_defense"].astype(int)
    return (room_avg_coverage, room_avg_run, safeties_sorted)

# Precompute per-opponent Safety room + list
opp_s_summaries = {}
for opp in OPPONENTS:
    opp_s_df = opp_safeties[opp_safeties["team_name"] == opp]
    opp_s_summaries[opp] = opponent_safety_room(opp_s_df)

# -------------- Build formatted output + CSV --------------
rows = []  # for CSV

print("=" * 80)
print("TE PASS & RUN GRADE vs S COVERAGE & RUN DEFENSE MATCHUPS")
print("=" * 80)

for opp in OPPONENTS:
    s_coverage_avg, s_run_avg, safeties_sorted = opp_s_summaries.get(opp, (np.nan, np.nan, pd.DataFrame()))

    print(f"\n{OUR_TEAM} TEs vs {opp} Safeties")  # section header

    for _, r in gt_tes.iterrows():
        player = r["player"]
        pass_g = round(float(r["grades_pass_route"]), 2) if pd.notna(r["grades_pass_route"]) else np.nan
        run_g = round(float(r["grades_run_block"]), 2) if pd.notna(r["grades_run_block"]) else np.nan

        # Team-level matchup (TE vs S room avg)
        delta_pass_coverage = round(pass_g - s_coverage_avg, 2) if pd.notna(pass_g) and pd.notna(s_coverage_avg) else np.nan
        delta_run_defense = round(run_g - s_run_avg, 2) if pd.notna(run_g) and pd.notna(s_run_avg) else np.nan
        
        print(f"{player} vs {opp} Defense — TE Pass: {pass_g:.2f} vs S Avg Coverage: {s_coverage_avg if pd.notna(s_coverage_avg) else float('nan'):.2f} (Δ {delta_pass_coverage if pd.notna(delta_pass_coverage) else float('nan'):.2f})")
        print(f"{player} vs {opp} Defense — TE Run: {run_g:.2f} vs S Avg Run-D: {s_run_avg if pd.notna(s_run_avg) else float('nan'):.2f} (Δ {delta_run_defense if pd.notna(delta_run_defense) else float('nan'):.2f})")

        rows.append({
            "gt_team": OUR_TEAM,
            "opponent": opp,
            "type": "vs_defense_avg",
            "player": player,
            "player_position": "TE",
            "player_pass_grade": pass_g,
            "player_run_grade": run_g,
            "opp_safety_coverage_grade": s_coverage_avg,
            "opp_safety_run_grade": s_run_avg,
            "opp_safety_name": None,
            "opp_safety_snaps": None,
            "delta_pass_coverage": delta_pass_coverage,
            "delta_run_defense": delta_run_defense
        })

        # One-on-one vs EVERY qualifying Safety (sorted by snaps)
        if not safeties_sorted.empty:
            for rank, (_, s) in enumerate(safeties_sorted.iterrows(), start=1):
                s_name = str(s["player"])
                s_coverage_grade = float(s["grades_coverage_defense"]) if pd.notna(s["grades_coverage_defense"]) else np.nan
                s_run_grade = float(s["grades_run_defense"]) if pd.notna(s["grades_run_defense"]) else np.nan
                s_snaps = int(s["snap_counts_defense"]) if pd.notna(s["snap_counts_defense"]) else 0
                delta_pass_coverage_s = round(pass_g - s_coverage_grade, 2) if pd.notna(pass_g) and pd.notna(s_coverage_grade) else np.nan
                delta_run_defense_s = round(run_g - s_run_grade, 2) if pd.notna(run_g) and pd.notna(s_run_grade) else np.nan

                print(f"{player} vs {opp} Defense (S #{rank} by snaps — {s_name}) — Pass: {pass_g:.2f} vs Coverage: {s_coverage_grade if pd.notna(s_coverage_grade) else float('nan'):.2f} (snaps {s_snaps}, Δ {delta_pass_coverage_s if pd.notna(delta_pass_coverage_s) else float('nan'):.2f})")
                print(f"{player} vs {opp} Defense (S #{rank} by snaps — {s_name}) — Run: {run_g:.2f} vs Run-D: {s_run_grade if pd.notna(s_run_grade) else float('nan'):.2f} (snaps {s_snaps}, Δ {delta_run_defense_s if pd.notna(delta_run_defense_s) else float('nan'):.2f})")

                rows.append({
                    "gt_team": OUR_TEAM,
                    "opponent": opp,
                    "type": f"vs_safety_rank_{rank}",
                    "player": player,
                    "player_position": "TE",
                    "player_pass_grade": pass_g,
                    "player_run_grade": run_g,
                    "opp_safety_coverage_grade": s_coverage_grade,
                    "opp_safety_run_grade": s_run_grade,
                    "opp_safety_name": s_name,
                    "opp_safety_snaps": s_snaps,
                    "delta_pass_coverage": delta_pass_coverage_s,
                    "delta_run_defense": delta_run_defense_s
                })

# Save a tidy CSV with every printed row
out_df = pd.DataFrame(rows)
out_df.to_csv("gt_te_pass_vs_safety_coverage_breakdown.csv", index=False)
print(f"\nSaved -> gt_te_pass_vs_safety_coverage_breakdown.csv")
print(f"Total TE vs S matchups analyzed: {len(rows)}")
