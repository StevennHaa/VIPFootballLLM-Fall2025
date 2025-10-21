import numpy as np
import pandas as pd

# ---------------- Config ----------------
OUR_TEAM = "GA TECH"  # matches your CSV exactly (see the team list you sent)
OPPONENTS = ["BOSTON COL", "NC STATE", "GEORGIA", "PITTSBURGH", "SYRACUSE"]
LB_POS = {"LB","ILB","OLB"}
SNAP_MIN = 50
ATT_MIN = 20

# -------------- Load & Prep --------------
# Offense: GT players with > 20 attempts
offense = pd.read_csv("rushing_summary.csv")
offense["attempts"]   = pd.to_numeric(offense["attempts"], errors="coerce")
offense["grades_run"] = pd.to_numeric(offense["grades_run"], errors="coerce")

gt_rushers = (
    offense[(offense["team_name"] == OUR_TEAM) & (offense["attempts"] > ATT_MIN)]
    .loc[:, ["player", "position", "team_name", "attempts", "grades_run"]]
    .copy()
)

# Defense: Opponent LBs with > 50 snaps
defense = pd.read_csv("defense_summary.csv")
defense["snap_counts_defense"] = pd.to_numeric(defense["snap_counts_defense"], errors="coerce")
defense["grades_run_defense"]  = pd.to_numeric(defense["grades_run_defense"], errors="coerce")

is_lb = defense["position"].astype(str).str.upper().isin(LB_POS)
idx   = np.where(is_lb & (defense["snap_counts_defense"] > SNAP_MIN))[0]
opp_lbs = defense.iloc[idx].query("team_name in @OPPONENTS").copy()

# -------------- Helper: opponent summaries --------------
def opponent_lb_room(df_opp: pd.DataFrame):
    """
    Returns a tuple:
      (room_avg_rounded, dataframe_of_all_LBs_sorted_by_snaps_with_rounded_grades)
    """
    if df_opp.empty:
        return (np.nan, df_opp)

    w = df_opp["snap_counts_defense"].clip(lower=0).fillna(0)
    v = df_opp["grades_run_defense"]
    room_avg = (v.mul(w)).sum() / (w.sum() if w.sum() > 0 else len(df_opp))
    room_avg = round(float(room_avg), 2)

    lbs_sorted = df_opp.sort_values("snap_counts_defense", ascending=False).copy()
    lbs_sorted["grades_run_defense"] = lbs_sorted["grades_run_defense"].round(2)
    lbs_sorted["snap_counts_defense"] = lbs_sorted["snap_counts_defense"].astype(int)
    return (room_avg, lbs_sorted)

# Precompute per-opponent LB room + list
opp_summaries = {}
for opp in OPPONENTS:
    opp_df = opp_lbs[opp_lbs["team_name"] == opp]
    opp_summaries[opp] = opponent_lb_room(opp_df)

# -------------- Build formatted output + CSV --------------
rows = []  # for CSV

for opp in OPPONENTS:
    room_avg, lbs_sorted = opp_summaries.get(opp, (np.nan, pd.DataFrame()))

    print(f"\n{OUR_TEAM} vs {opp}")  # section header

    for _, r in gt_rushers.iterrows():
        player = r["player"]
        run_g  = round(float(r["grades_run"]), 2) if pd.notna(r["grades_run"]) else np.nan

        # Team-level matchup (player vs LB room avg)
        delta_team = round(run_g - room_avg, 2) if pd.notna(run_g) and pd.notna(room_avg) else np.nan
        print(f"{player} vs {opp} Defense — RB Run: {run_g:.2f} vs LB Avg Run-D: {room_avg if pd.notna(room_avg) else float('nan'):.2f} (Δ {delta_team if pd.notna(delta_team) else float('nan'):.2f})")

        rows.append({
            "gt_team": OUR_TEAM,
            "opponent": opp,
            "type": "vs_defense_avg",
            "player": player,
            "player_run_grade": run_g,
            "opp_lb_grade": room_avg,
            "opp_lb_name": None,
            "opp_lb_snaps": None,
            "delta": delta_team
        })

        # One-on-one vs EVERY qualifying LB (sorted by snaps)
        if not lbs_sorted.empty:
            for rank, (_, lb) in enumerate(lbs_sorted.iterrows(), start=1):
                lb_name  = str(lb["player"])
                lb_grade = float(lb["grades_run_defense"]) if pd.notna(lb["grades_run_defense"]) else np.nan
                lb_snaps = int(lb["snap_counts_defense"]) if pd.notna(lb["snap_counts_defense"]) else 0
                delta_lb = round(run_g - lb_grade, 2) if pd.notna(run_g) and pd.notna(lb_grade) else np.nan

                # Print with a clear label (rank by snaps)
                print(f"{player} vs {opp} Defense (LB #{rank} by snaps — {lb_name}) — {run_g:.2f} vs {lb_grade if pd.notna(lb_grade) else float('nan'):.2f} (snaps {lb_snaps}, Δ {delta_lb if pd.notna(delta_lb) else float('nan'):.2f})")

                rows.append({
                    "gt_team": OUR_TEAM,
                    "opponent": opp,
                    "type": f"vs_lb_rank_{rank}",
                    "player": player,
                    "player_run_grade": run_g,
                    "opp_lb_grade": lb_grade,
                    "opp_lb_name": lb_name,
                    "opp_lb_snaps": lb_snaps,
                    "delta": delta_lb
                })

# Save a tidy CSV with every printed row
out_df = pd.DataFrame(rows)
out_df.to_csv("gt_offense_vs_all_opp_lbs_breakdown.csv", index=False)
print("\nSaved -> gt_offense_vs_all_opp_lbs_breakdown.csv")
