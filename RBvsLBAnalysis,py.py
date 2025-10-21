#!/usr/bin/env python3
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

# ---------------- Config ----------------
OUR_TEAM = "GA TECH"
OPPONENTS = ["BOSTON COL", "NC STATE", "GEORGIA", "PITTSBURGH", "SYRACUSE"]
LB_POS = {"LB","ILB","OLB"}
SNAP_MIN = 50
ATT_MIN = 20

# ---------------- Utils ----------------
def coerce_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def first_existing(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None

def wavg(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").clip(lower=0).fillna(0)
    if w.sum() > 0:
        return float((v * w).sum() / w.sum())
    v = v.dropna()
    return float(v.mean()) if len(v) else np.nan

def r2(x: Any) -> Any:
    try:
        return round(float(x), 2)
    except Exception:
        return np.nan

# ---------------- Load & Prep ----------------
# Offense: GT players with > 20 attempts
offense = pd.read_csv("rushing_summary.csv")
for col in ["attempts", "grades_run", "grades_receive", "grades_receiving"]:
    if col in offense.columns:
        offense[col] = coerce_num(offense[col])

# Receiving grade strictly = receiving (not pass block)
RB_RECV_COL = first_existing(offense, ["grades_receive", "grades_receiving"])
if RB_RECV_COL is None:
    RB_RECV_COL = "grades_receive"
    offense[RB_RECV_COL] = np.nan

if "grades_run" not in offense.columns:
    offense["grades_run"] = np.nan

gt_rushers = (
    offense[(offense["team_name"] == OUR_TEAM) & (offense["attempts"] > ATT_MIN)]
    .loc[:, ["player", "position", "team_name", "attempts", "grades_run", RB_RECV_COL]]
    .copy()
)
gt_rushers.rename(columns={RB_RECV_COL: "grades_receive"}, inplace=True)

# Defense: Opponent LBs with > 50 snaps
defense = pd.read_csv("defense_summary.csv")
for col in ["snap_counts_defense", "grades_run_defense", "grades_coverage_defense", "grades_coverage"]:
    if col in defense.columns:
        defense[col] = coerce_num(defense[col])

COV_COL = "grades_coverage_defense" if "grades_coverage_defense" in defense.columns else (
          "grades_coverage" if "grades_coverage" in defense.columns else None)
if COV_COL is None:
    COV_COL = "grades_coverage_defense"
    defense[COV_COL] = np.nan

is_lb = defense["position"].astype(str).str.upper().isin(LB_POS)
idx   = np.where(is_lb & (defense["snap_counts_defense"] > SNAP_MIN))[0]
opp_lbs = defense.iloc[idx].query("team_name in @OPPONENTS").copy()

# ---------------- Helper: opponent summaries ----------------
def opponent_lb_room(df_opp: pd.DataFrame):
    if df_opp.empty:
        return {"run_avg": np.nan, "cov_avg": np.nan, "lbs_sorted": df_opp}

    w = df_opp["snap_counts_defense"].clip(lower=0).fillna(0)
    run_avg = wavg(df_opp["grades_run_defense"], w)
    cov_avg = wavg(df_opp[COV_COL], w)

    lbs_sorted = df_opp.sort_values("snap_counts_defense", ascending=False).copy()
    lbs_sorted["grades_run_defense"] = lbs_sorted["grades_run_defense"].apply(r2)
    lbs_sorted[COV_COL] = lbs_sorted[COV_COL].apply(r2)
    lbs_sorted["snap_counts_defense"] = lbs_sorted["snap_counts_defense"].fillna(0).astype(int)
    return {"run_avg": r2(run_avg), "cov_avg": r2(cov_avg), "lbs_sorted": lbs_sorted}

opp_summaries: Dict[str, Dict[str, Any]] = {}
for opp in OPPONENTS:
    opp_df = opp_lbs[opp_lbs["team_name"] == opp]
    opp_summaries[opp] = opponent_lb_room(opp_df)

# ---------------- Build formatted output + CSV ----------------
rows = []

for opp in OPPONENTS:
    summary = opp_summaries.get(opp, {"run_avg": np.nan, "cov_avg": np.nan, "lbs_sorted": pd.DataFrame()})
    room_run_avg = summary["run_avg"]
    room_cov_avg = summary["cov_avg"]
    lbs_sorted   = summary["lbs_sorted"]

    print(f"\\n{OUR_TEAM} vs {opp}")

    for _, r in gt_rushers.iterrows():
        player = r["player"]
        run_g  = r2(r["grades_run"]) if pd.notna(r["grades_run"]) else np.nan
        rec_g  = r2(r["grades_receive"]) if pd.notna(r["grades_receive"]) else np.nan

        # Team-level matchup (player vs LB room avgs)
        d_run_team = r2(run_g - room_run_avg) if (pd.notna(run_g) and pd.notna(room_run_avg)) else np.nan
        d_rec_team = r2(rec_g - room_cov_avg) if (pd.notna(rec_g) and pd.notna(room_cov_avg)) else np.nan
        overall_team = r2(np.nanmean([d for d in [d_run_team, d_rec_team] if pd.notna(d)])) if any(pd.notna(x) for x in [d_run_team, d_rec_team]) else np.nan

        print(
            f"{player} vs {opp} Defense — RB Run: {run_g if pd.notna(run_g) else float('nan'):.2f} "
            f"vs LB Avg Run-D: {room_run_avg if pd.notna(room_run_avg) else float('nan'):.2f} "
            f"(Δ {d_run_team if pd.notna(d_run_team) else float('nan'):.2f})"
        )
        print(
            f"{player} vs {opp} Defense — RB Receiving: {rec_g if pd.notna(rec_g) else float('nan'):.2f} "
            f"vs LB Avg Coverage: {room_cov_avg if pd.notna(room_cov_avg) else float('nan'):.2f} "
            f"(Δ {d_rec_team if pd.notna(d_rec_team) else float('nan'):.2f})"
        )
        print(f"Overall Δ (mean of available): {overall_team if pd.notna(overall_team) else float('nan'):.2f}")

        rows.append({
            "gt_team": OUR_TEAM,
            "opponent": opp,
            "row_type": "vs_lb_room_avg",
            "player": player,
            "player_run_grade": run_g,
            "player_receiving_grade": rec_g,
            "opp_lb_run_avg": room_run_avg,
            "opp_lb_cov_avg": room_cov_avg,
            "opp_lb_run_grade": None,
            "opp_lb_cov_grade": None,
            "delta_run_minus_lb_run": d_run_team,
            "delta_recv_minus_lb_cov": d_rec_team,
            "overall_delta": overall_team,
            "opp_lb_name": None,
            "opp_lb_snaps": None,
        })

        # Per-LB rows with grades included
        if not lbs_sorted.empty:
            for rank, (_, lb) in enumerate(lbs_sorted.iterrows(), start=1):
                lb_name  = str(lb["player"])
                lb_run_g = float(lb["grades_run_defense"]) if pd.notna(lb["grades_run_defense"]) else np.nan
                lb_cov_g = float(lb[COV_COL]) if pd.notna(lb[COV_COL]) else np.nan
                lb_snaps = int(lb["snap_counts_defense"]) if pd.notna(lb["snap_counts_defense"]) else 0

                d_run_lb = r2(run_g - lb_run_g) if (pd.notna(run_g) and pd.notna(lb_run_g)) else np.nan
                d_rec_lb = r2(rec_g - lb_cov_g) if (pd.notna(rec_g) and pd.notna(lb_cov_g)) else np.nan
                overall_lb = r2(np.nanmean([d for d in [d_run_lb, d_rec_lb] if pd.notna(d)])) if any(pd.notna(x) for x in [d_run_lb, d_rec_lb]) else np.nan

                print(
                    f"{player} vs {opp} Defense (LB #{rank} by snaps — {lb_name}) "
                    f"— Run {run_g if pd.notna(run_g) else float('nan'):.2f} vs {lb_run_g if pd.notna(lb_run_g) else float('nan'):.2f} "
                    f"(Δ {d_run_lb if pd.notna(d_run_lb) else float('nan'):.2f}); "
                    f"Recv {rec_g if pd.notna(rec_g) else float('nan'):.2f} vs {lb_cov_g if pd.notna(lb_cov_g) else float('nan'):.2f} "
                    f"(Δ {d_rec_lb if pd.notna(d_rec_lb) else float('nan'):.2f}) "
                    f"[snaps {lb_snaps}]  Overall Δ: {overall_lb if pd.notna(overall_lb) else float('nan'):.2f}"
                )

                rows.append({
                    "gt_team": OUR_TEAM,
                    "opponent": opp,
                    "row_type": f"vs_lb_rank_{rank}",
                    "player": player,
                    "player_run_grade": run_g,
                    "player_receiving_grade": rec_g,
                    "opp_lb_run_avg": None,
                    "opp_lb_cov_avg": None,
                    "opp_lb_run_grade": lb_run_g,
                    "opp_lb_cov_grade": lb_cov_g,
                    "delta_run_minus_lb_run": d_run_lb,
                    "delta_recv_minus_lb_cov": d_rec_lb,
                    "overall_delta": overall_lb,
                    "opp_lb_name": lb_name,
                    "opp_lb_snaps": lb_snaps,
                })

# Save CSV
out_df = pd.DataFrame(rows)
# enforce consistent column order
col_order = [
    "gt_team","opponent","row_type","player",
    "player_run_grade","player_receiving_grade",
    "opp_lb_run_avg","opp_lb_cov_avg",
    "opp_lb_run_grade","opp_lb_cov_grade",
    "delta_run_minus_lb_run","delta_recv_minus_lb_cov","overall_delta",
    "opp_lb_name","opp_lb_snaps"
]
for c in col_order:
    if c not in out_df.columns:
        out_df[c] = np.nan
out_df = out_df[col_order]
out_df.to_csv("GTRBSvsOPPLBS.csv", index=False)
print("\\nSaved -> GTRBSvsOPPLBS.csv")
