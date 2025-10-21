import numpy as np
import pandas as pd

# ---------------- Config ----------------
OUR_TEAM = "GA TECH"  # matches your CSV exactly
OPPONENTS = ["BOSTON COL", "NC STATE", "GEORGIA", "PITTSBURGH", "SYRACUSE"]
DB_POS = {"CB", "DB"}  # Cornerbacks and Defensive Backs
SNAP_MIN = 50
TARGET_MIN = 15  # Minimum targets for WRs

# -------------- Load & Prep --------------
# Offense: GT WRs with > 15 targets
receiving = pd.read_csv("Data/receiving_summary (2).csv")
receiving["targets"] = pd.to_numeric(receiving["targets"], errors="coerce")
receiving["grades_pass_route"] = pd.to_numeric(receiving["grades_pass_route"], errors="coerce")

# Filter GT WRs with sufficient targets
gt_wrs = (
    receiving[(receiving["team_name"] == OUR_TEAM) & 
              (receiving["position"] == "WR") & 
              (receiving["targets"] > TARGET_MIN)]
    .loc[:, ["player", "position", "team_name", "targets", "grades_pass_route"]]
    .copy()
)

# Defense: Opponent DBs with > 50 snaps
defense = pd.read_csv("Data/defense_summary.csv")
defense["snap_counts_defense"] = pd.to_numeric(defense["snap_counts_defense"], errors="coerce")
defense["grades_coverage_defense"] = pd.to_numeric(defense["grades_coverage_defense"], errors="coerce")

# Filter DBs
is_db = defense["position"].astype(str).str.upper().isin(DB_POS)
idx_db = np.where(is_db & (defense["snap_counts_defense"] > SNAP_MIN))[0]
opp_dbs = defense.iloc[idx_db].query("team_name in @OPPONENTS").copy()

# -------------- Helper: opponent summaries --------------
def opponent_db_room(df_opp: pd.DataFrame):
    """
    Returns a tuple:
      (room_avg_rounded, dataframe_of_all_DBs_sorted_by_snaps_with_rounded_grades)
    """
    if df_opp.empty:
        return (np.nan, df_opp)

    w = df_opp["snap_counts_defense"].clip(lower=0).fillna(0)
    v = df_opp["grades_coverage_defense"]
    room_avg = (v.mul(w)).sum() / (w.sum() if w.sum() > 0 else len(df_opp))
    room_avg = round(float(room_avg), 2)

    dbs_sorted = df_opp.sort_values("snap_counts_defense", ascending=False).copy()
    dbs_sorted["grades_coverage_defense"] = dbs_sorted["grades_coverage_defense"].round(2)
    dbs_sorted["snap_counts_defense"] = dbs_sorted["snap_counts_defense"].astype(int)
    return (room_avg, dbs_sorted)

# Precompute per-opponent DB room + list
opp_db_summaries = {}
for opp in OPPONENTS:
    opp_db_df = opp_dbs[opp_dbs["team_name"] == opp]
    opp_db_summaries[opp] = opponent_db_room(opp_db_df)

# -------------- Build formatted output + CSV --------------
rows = []  # for CSV

print("=" * 80)
print("WR PASS GRADE vs DB COVERAGE GRADE MATCHUPS")
print("=" * 80)

for opp in OPPONENTS:
    db_room_avg, dbs_sorted = opp_db_summaries.get(opp, (np.nan, pd.DataFrame()))

    print(f"\n{OUR_TEAM} WRs vs {opp} DBs")  # section header

    for _, r in gt_wrs.iterrows():
        player = r["player"]
        pass_g = round(float(r["grades_pass_route"]), 2) if pd.notna(r["grades_pass_route"]) else np.nan

        # Team-level matchup (WR vs DB room avg)
        delta_team = round(pass_g - db_room_avg, 2) if pd.notna(pass_g) and pd.notna(db_room_avg) else np.nan
        print(f"{player} vs {opp} Defense — WR Pass: {pass_g:.2f} vs DB Avg Coverage: {db_room_avg if pd.notna(db_room_avg) else float('nan'):.2f} (Δ {delta_team if pd.notna(delta_team) else float('nan'):.2f})")

        rows.append({
            "gt_team": OUR_TEAM,
            "opponent": opp,
            "type": "vs_defense_avg",
            "player": player,
            "player_position": "WR",
            "player_pass_grade": pass_g,
            "opp_db_grade": db_room_avg,
            "opp_db_name": None,
            "opp_db_snaps": None,
            "delta": delta_team
        })

        # One-on-one vs EVERY qualifying DB (sorted by snaps)
        if not dbs_sorted.empty:
            for rank, (_, db) in enumerate(dbs_sorted.iterrows(), start=1):
                db_name = str(db["player"])
                db_grade = float(db["grades_coverage_defense"]) if pd.notna(db["grades_coverage_defense"]) else np.nan
                db_snaps = int(db["snap_counts_defense"]) if pd.notna(db["snap_counts_defense"]) else 0
                delta_db = round(pass_g - db_grade, 2) if pd.notna(pass_g) and pd.notna(db_grade) else np.nan

                print(f"{player} vs {opp} Defense (DB #{rank} by snaps — {db_name}) — Coverage: {pass_g:.2f} vs {db_grade if pd.notna(db_grade) else float('nan'):.2f} (snaps {db_snaps}, Δ {delta_db if pd.notna(delta_db) else float('nan'):.2f})")

                rows.append({
                    "gt_team": OUR_TEAM,
                    "opponent": opp,
                    "type": f"vs_db_rank_{rank}",
                    "player": player,
                    "player_position": "WR",
                    "player_pass_grade": pass_g,
                    "opp_db_grade": db_grade,
                    "opp_db_name": db_name,
                    "opp_db_snaps": db_snaps,
                    "delta": delta_db
                })

# Save a tidy CSV with every printed row
out_df = pd.DataFrame(rows)
out_df.to_csv("gt_wr_pass_vs_db_coverage_breakdown.csv", index=False)
print(f"\nSaved -> gt_wr_pass_vs_db_coverage_breakdown.csv")
print(f"Total WR vs DB matchups analyzed: {len(rows)}")
