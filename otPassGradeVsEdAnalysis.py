import numpy as np
import pandas as pd

# ---------------- Config ----------------
OUR_TEAM = "GA TECH"  # matches your CSV exactly
OPPONENTS = ["BOSTON COL", "NC STATE", "GEORGIA", "PITTSBURGH", "SYRACUSE"]
OT_POS = {"T"}  # Offensive Tackles
ED_POS = {"ED"}  # Edge Defenders
SNAP_MIN = 50
PASS_BLOCK_MIN = 50  # Minimum pass blocking snaps for OTs

# -------------- Load & Prep --------------
# Offense: GT OTs with > 50 pass blocking snaps
blocking = pd.read_csv("Data/offense_blocking.csv")
blocking["snap_counts_pass_block"] = pd.to_numeric(blocking["snap_counts_pass_block"], errors="coerce")
blocking["grades_pass_block"] = pd.to_numeric(blocking["grades_pass_block"], errors="coerce")

# Filter GT OTs with sufficient pass blocking snaps
gt_ots = (
    blocking[(blocking["team_name"] == OUR_TEAM) & 
             (blocking["position"] == "T") & 
             (blocking["snap_counts_pass_block"] > PASS_BLOCK_MIN)]
    .loc[:, ["player", "position", "team_name", "snap_counts_pass_block", "grades_pass_block"]]
    .copy()
)

# Defense: Opponent EDs with > 50 snaps
defense = pd.read_csv("Data/defense_summary.csv")
defense["snap_counts_defense"] = pd.to_numeric(defense["snap_counts_defense"], errors="coerce")
defense["grades_pass_rush_defense"] = pd.to_numeric(defense["grades_pass_rush_defense"], errors="coerce")

# Filter EDs
is_ed = defense["position"].astype(str).str.upper().isin(ED_POS)
idx_ed = np.where(is_ed & (defense["snap_counts_defense"] > SNAP_MIN))[0]
opp_eds = defense.iloc[idx_ed].query("team_name in @OPPONENTS").copy()

# -------------- Helper: opponent summaries --------------
def opponent_ed_room(df_opp: pd.DataFrame):
    """
    Returns a tuple:
      (room_avg_rounded, dataframe_of_all_EDs_sorted_by_snaps_with_rounded_grades)
    """
    if df_opp.empty:
        return (np.nan, df_opp)

    w = df_opp["snap_counts_defense"].clip(lower=0).fillna(0)
    v = df_opp["grades_pass_rush_defense"]
    room_avg = (v.mul(w)).sum() / (w.sum() if w.sum() > 0 else len(df_opp))
    room_avg = round(float(room_avg), 2)

    eds_sorted = df_opp.sort_values("snap_counts_defense", ascending=False).copy()
    eds_sorted["grades_pass_rush_defense"] = eds_sorted["grades_pass_rush_defense"].round(2)
    eds_sorted["snap_counts_defense"] = eds_sorted["snap_counts_defense"].astype(int)
    return (room_avg, eds_sorted)

# Precompute per-opponent ED room + list
opp_ed_summaries = {}
for opp in OPPONENTS:
    opp_ed_df = opp_eds[opp_eds["team_name"] == opp]
    opp_ed_summaries[opp] = opponent_ed_room(opp_ed_df)

# -------------- Build formatted output + CSV --------------
rows = []  # for CSV

print("=" * 80)
print("GT OT PASS BLOCKING vs OPPONENT ED PASS RUSH MATCHUPS")
print("=" * 80)

for opp in OPPONENTS:
    ed_room_avg, eds_sorted = opp_ed_summaries.get(opp, (np.nan, pd.DataFrame()))

    print(f"\n{OUR_TEAM} OTs vs {opp} EDs")  # section header

    for _, r in gt_ots.iterrows():
        player = r["player"]
        pass_block_g = round(float(r["grades_pass_block"]), 2) if pd.notna(r["grades_pass_block"]) else np.nan

        # Team-level matchup (OT vs ED room avg)
        delta_team = round(pass_block_g - ed_room_avg, 2) if pd.notna(pass_block_g) and pd.notna(ed_room_avg) else np.nan
        print(f"{player} vs {opp} Defense — OT Pass Block: {pass_block_g:.2f} vs ED Avg Pass Rush: {ed_room_avg if pd.notna(ed_room_avg) else float('nan'):.2f} (Δ {delta_team if pd.notna(delta_team) else float('nan'):.2f})")

        rows.append({
            "gt_team": OUR_TEAM,
            "opponent": opp,
            "type": "vs_defense_avg",
            "player": player,
            "player_position": "OT",
            "player_pass_block_grade": pass_block_g,
            "opp_ed_pass_rush_grade": ed_room_avg,
            "opp_ed_name": None,
            "opp_ed_snaps": None,
            "delta": delta_team
        })

        # One-on-one vs EVERY qualifying ED (sorted by snaps)
        if not eds_sorted.empty:
            for rank, (_, ed) in enumerate(eds_sorted.iterrows(), start=1):
                ed_name = str(ed["player"])
                ed_grade = float(ed["grades_pass_rush_defense"]) if pd.notna(ed["grades_pass_rush_defense"]) else np.nan
                ed_snaps = int(ed["snap_counts_defense"]) if pd.notna(ed["snap_counts_defense"]) else 0
                delta_ed = round(pass_block_g - ed_grade, 2) if pd.notna(pass_block_g) and pd.notna(ed_grade) else np.nan

                print(f"{player} vs {opp} Defense (ED #{rank} by snaps — {ed_name}) — Pass Block: {pass_block_g:.2f} vs Pass Rush: {ed_grade if pd.notna(ed_grade) else float('nan'):.2f} (snaps {ed_snaps}, Δ {delta_ed if pd.notna(delta_ed) else float('nan'):.2f})")

                rows.append({
                    "gt_team": OUR_TEAM,
                    "opponent": opp,
                    "type": f"vs_ed_rank_{rank}",
                    "player": player,
                    "player_position": "OT",
                    "player_pass_block_grade": pass_block_g,
                    "opp_ed_pass_rush_grade": ed_grade,
                    "opp_ed_name": ed_name,
                    "opp_ed_snaps": ed_snaps,
                    "delta": delta_ed
                })

# Save a tidy CSV with every printed row
out_df = pd.DataFrame(rows)
out_df.to_csv("Data/gt_ot_pass_vs_ed_pass_rush_breakdown.csv", index=False)
print(f"\nSaved -> Data/gt_ot_pass_vs_ed_pass_rush_breakdown.csv")
print(f"Total OT vs ED pass matchups analyzed: {len(rows)}")

