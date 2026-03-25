"""
Model to reduce extra time caused by the unspecialized doctor handling specialized operations.

Role mapping defaults:
- Main specialized doctor: Dr. A
- Unspecialized doctor: Dr. B
- Backup doctor: Dr. C

Approach:
1. Machine learning model (Ridge regression) predicts case duration from pre-op style features.
2. Per-day mathematical optimization reassigns specialized cases from Dr. B to Dr. A/Dr. C
   subject to configurable capacity constraints.
3. Monte Carlo simulation estimates expected savings uncertainty under residual noise.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class RoleConfig:
    main_specialized_doctor: str = "Dr. A"
    unspecialized_doctor: str = "Dr. B"
    backup_doctor: str = "Dr. C"


def load_and_clean(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="All Data", header=None)
    headers = raw.iloc[2].tolist()
    df = raw.iloc[4:].copy().reset_index(drop=True)
    df.columns = headers

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    rename_map = {
        "CASE #": "case_id",
        "DATE": "date",
        "PHYSICIAN": "physician",
        "PT PREP/INTUBATION": "pt_prep_intubation",
        "ACCESSS": "access",
        "TSP": "tsp",
        "PRE-MAP": "pre_map",
        "ABL DURATION": "abl_duration",
        "ABL TIME": "abl_time",
        "#ABL": "abl_count",
        "#APPLICATIONS": "abl_applications",
        "LA DWELL TIME": "la_dwell",
        "CASE TIME": "case_time",
        "AVG CASE TIME": "avg_case_time",
        "SKIN-SKIN": "skin_skin",
        "AVG SKIN-SKIN": "avg_skin_skin",
        "POST CARE/EXTUBATION": "post_care_extubation",
        "AVG TURNOVER TIME": "avg_turnover_time",
        "PT OUT TIME": "pt_out_time",
        "PT IN-OUT": "pt_in_out",
        "Note": "note",
    }
    df = df.rename(columns=rename_map)
    df = df[df["case_id"].notna()].copy().reset_index(drop=True)

    df["case_id"] = pd.to_numeric(df["case_id"], errors="coerce").astype("Int64")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in [c for c in df.columns if c not in ["date", "physician", "note", "case_id"]]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Complexity tags from note field.
    notes = df["note"].fillna("").astype(str)
    df["flag_cti"] = notes.str.contains("CTI", case=False, regex=True).astype(int)
    df["flag_box"] = notes.str.contains("BOX", case=False, regex=True).astype(int)
    df["flag_pst"] = notes.str.contains("PST", case=False, regex=True).astype(int)
    df["flag_svc"] = notes.str.contains("SVC", case=False, regex=True).astype(int)
    df["flag_aafl"] = notes.str.contains("AAFL", case=False, regex=True).astype(int)
    df["flag_trouble"] = notes.str.contains("TROUBLE", case=False, regex=True).astype(int)
    df["is_specialized"] = (
        df[["flag_cti", "flag_box", "flag_pst", "flag_svc", "flag_aafl", "flag_trouble"]].sum(axis=1) > 0
    ).astype(int)

    df["month"] = df["date"].dt.month
    return df


def train_duration_model(df: pd.DataFrame) -> Tuple[Pipeline, pd.DataFrame]:
    model_df = df.dropna(subset=["case_time", "physician", "date"]).copy()
    feature_cols = [
        "physician",
        "is_specialized",
        "flag_cti",
        "flag_box",
        "flag_pst",
        "flag_svc",
        "flag_aafl",
        "flag_trouble",
        "month",
    ]
    X = model_df[feature_cols]
    y = model_df["case_time"]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["physician"]),
            ("num", "passthrough", [c for c in feature_cols if c != "physician"]),
        ]
    )
    pipe = Pipeline(
        steps=[
            ("preprocess", pre),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )
    pipe.fit(X, y)

    model_df = model_df.copy()
    model_df["pred_case_time"] = pipe.predict(X)
    model_df["residual"] = model_df["case_time"] - model_df["pred_case_time"]
    return pipe, model_df


def predict_counterfactual_case_time(pipe: Pipeline, row: pd.Series, doctor: str) -> float:
    payload = pd.DataFrame(
        [
            {
                "physician": doctor,
                "is_specialized": int(row["is_specialized"]),
                "flag_cti": int(row["flag_cti"]),
                "flag_box": int(row["flag_box"]),
                "flag_pst": int(row["flag_pst"]),
                "flag_svc": int(row["flag_svc"]),
                "flag_aafl": int(row["flag_aafl"]),
                "flag_trouble": int(row["flag_trouble"]),
                "month": int(row["month"]) if not pd.isna(row["month"]) else 1,
            }
        ]
    )
    return float(pipe.predict(payload)[0])


def optimize_day_assignments(
    day_cases: pd.DataFrame,
    pred_costs: Dict[int, Dict[str, float]],
    uns_doc: str,
    main_doc: str,
    backup_doc: str,
    cap_main: int,
    cap_backup: int,
) -> Dict[int, str]:
    # Dynamic programming over candidate cases:
    # state = (assigned_to_main_count, assigned_to_backup_count)
    # value = (min_total_cost, assignments_dict)
    dp: Dict[Tuple[int, int], Tuple[float, Dict[int, str]]] = {(0, 0): (0.0, {})}

    for idx in day_cases.index.tolist():
        next_dp: Dict[Tuple[int, int], Tuple[float, Dict[int, str]]] = {}
        for (a_cnt, c_cnt), (cost, plan) in dp.items():
            # Keep with unspecialized doctor.
            keep_cost = cost + pred_costs[idx][uns_doc]
            keep_state = (a_cnt, c_cnt)
            keep_plan = dict(plan)
            keep_plan[idx] = uns_doc
            _update_state(next_dp, keep_state, keep_cost, keep_plan)

            # Move to main specialized doctor.
            if a_cnt < cap_main:
                a_cost = cost + pred_costs[idx][main_doc]
                a_state = (a_cnt + 1, c_cnt)
                a_plan = dict(plan)
                a_plan[idx] = main_doc
                _update_state(next_dp, a_state, a_cost, a_plan)

            # Move to backup doctor.
            if c_cnt < cap_backup:
                c_cost = cost + pred_costs[idx][backup_doc]
                c_state = (a_cnt, c_cnt + 1)
                c_plan = dict(plan)
                c_plan[idx] = backup_doc
                _update_state(next_dp, c_state, c_cost, c_plan)
        dp = next_dp

    best_state = min(dp.keys(), key=lambda s: dp[s][0])
    return dp[best_state][1]


def _update_state(
    store: Dict[Tuple[int, int], Tuple[float, Dict[int, str]]],
    state: Tuple[int, int],
    cost: float,
    plan: Dict[int, str],
) -> None:
    if state not in store or cost < store[state][0]:
        store[state] = (cost, plan)


def optimize_reassignment(
    df: pd.DataFrame,
    pipe: Pipeline,
    roles: RoleConfig,
    max_main_cases_per_day: int,
    max_backup_cases_per_day: int,
    max_extra_main_per_day: int,
    max_extra_backup_per_day: int,
) -> pd.DataFrame:
    # Candidate set = specialized cases currently done by unspecialized doctor.
    cand = df[
        (df["physician"] == roles.unspecialized_doctor)
        & (df["is_specialized"] == 1)
        & (df["date"].notna())
    ].copy()

    if cand.empty:
        return cand.assign(
            recommended_doctor=pd.Series(dtype="object"),
            pred_current_case_time=pd.Series(dtype="float"),
            pred_recommended_case_time=pd.Series(dtype="float"),
            pred_saving_minutes=pd.Series(dtype="float"),
        )

    assignments: Dict[int, str] = {}
    pred_cache: Dict[int, Dict[str, float]] = {}

    daily_counts = df[df["date"].notna()].groupby(["date", "physician"]).size()

    for day, day_cases in cand.groupby("date"):
        main_count = int(daily_counts.get((day, roles.main_specialized_doctor), 0))
        backup_count = int(daily_counts.get((day, roles.backup_doctor), 0))

        cap_main = min(max(0, max_main_cases_per_day - main_count), max_extra_main_per_day)
        cap_backup = min(max(0, max_backup_cases_per_day - backup_count), max_extra_backup_per_day)

        for idx, row in day_cases.iterrows():
            pred_cache[idx] = {
                roles.unspecialized_doctor: predict_counterfactual_case_time(pipe, row, roles.unspecialized_doctor),
                roles.main_specialized_doctor: predict_counterfactual_case_time(pipe, row, roles.main_specialized_doctor),
                roles.backup_doctor: predict_counterfactual_case_time(pipe, row, roles.backup_doctor),
            }

        day_plan = optimize_day_assignments(
            day_cases=day_cases,
            pred_costs=pred_cache,
            uns_doc=roles.unspecialized_doctor,
            main_doc=roles.main_specialized_doctor,
            backup_doc=roles.backup_doctor,
            cap_main=cap_main,
            cap_backup=cap_backup,
        )
        assignments.update(day_plan)

    out = cand.copy()
    out["recommended_doctor"] = out.index.map(assignments)
    out["pred_current_case_time"] = out.index.map(
        lambda i: pred_cache[i][roles.unspecialized_doctor]
    )
    out["pred_recommended_case_time"] = out.apply(
        lambda r: pred_cache[r.name][r["recommended_doctor"]], axis=1
    )
    out["pred_saving_minutes"] = out["pred_current_case_time"] - out["pred_recommended_case_time"]
    out["move_recommended"] = (out["recommended_doctor"] != roles.unspecialized_doctor).astype(int)

    # Friendly role labels in output.
    role_name = {
        roles.main_specialized_doctor: "main_specialized_doctor",
        roles.unspecialized_doctor: "unspecialized_doctor",
        roles.backup_doctor: "backup_doctor",
    }
    out["recommended_role"] = out["recommended_doctor"].map(role_name).fillna("unknown")
    out["current_role"] = "unspecialized_doctor"
    return out


def build_schedule_comparison(
    df: pd.DataFrame,
    reco_df: pd.DataFrame,
    roles: RoleConfig,
) -> pd.DataFrame:
    """
    Build before/after schedule view for the full given schedule.

    Output includes:
    - baseline doctor and position (original schedule)
    - optimized doctor and position (after recommended moves)
    - baseline/optimized minutes and per-case saving
    - explicit move flag showing what changes
    """
    sched = df[df["date"].notna()].copy()
    if sched.empty:
        return sched

    # Candidate maps keyed by case_id (CASE # is unique in this dataset).
    reco_map = reco_df.set_index("case_id") if not reco_df.empty else pd.DataFrame()

    if not reco_map.empty:
        rec_doc = reco_map["recommended_doctor"]
        rec_move = reco_map["move_recommended"]
        rec_base = reco_map["pred_current_case_time"]
        rec_opt = reco_map["pred_recommended_case_time"]
    else:
        rec_doc = pd.Series(dtype="object")
        rec_move = pd.Series(dtype="float")
        rec_base = pd.Series(dtype="float")
        rec_opt = pd.Series(dtype="float")

    sched["baseline_doctor"] = sched["physician"]
    sched["optimized_doctor"] = sched["case_id"].map(rec_doc).fillna(sched["baseline_doctor"])
    sched["recommended_move"] = sched["case_id"].map(rec_move).fillna(0).astype(int)

    # Use model-based candidate estimates when available; otherwise use observed case_time.
    sched["baseline_minutes"] = sched["case_id"].map(rec_base)
    sched["optimized_minutes"] = sched["case_id"].map(rec_opt)
    sched["baseline_minutes"] = sched["baseline_minutes"].fillna(sched["case_time"])
    sched["optimized_minutes"] = sched["optimized_minutes"].fillna(sched["case_time"])

    # Conservative fallback if case_time is missing.
    fallback = float(df["case_time"].median()) if df["case_time"].notna().any() else 40.0
    sched["baseline_minutes"] = sched["baseline_minutes"].fillna(fallback).clip(lower=1.0)
    sched["optimized_minutes"] = sched["optimized_minutes"].fillna(fallback).clip(lower=1.0)

    # Baseline positions and cumulative day/doctor timing.
    base_sort = sched.sort_values(["date", "baseline_doctor", "case_id"]).copy()
    base_sort["baseline_position"] = base_sort.groupby(["date", "baseline_doctor"]).cumcount() + 1
    base_sort["baseline_start_min"] = (
        base_sort.groupby(["date", "baseline_doctor"])["baseline_minutes"].cumsum() - base_sort["baseline_minutes"]
    )
    base_sort["baseline_end_min"] = base_sort["baseline_start_min"] + base_sort["baseline_minutes"]

    # Optimized positions and cumulative day/doctor timing.
    opt_sort = sched.sort_values(["date", "optimized_doctor", "case_id"]).copy()
    opt_sort["optimized_position"] = opt_sort.groupby(["date", "optimized_doctor"]).cumcount() + 1
    opt_sort["optimized_start_min"] = (
        opt_sort.groupby(["date", "optimized_doctor"])["optimized_minutes"].cumsum() - opt_sort["optimized_minutes"]
    )
    opt_sort["optimized_end_min"] = opt_sort["optimized_start_min"] + opt_sort["optimized_minutes"]

    # Merge schedule views at case level.
    out = base_sort[
        [
            "case_id",
            "date",
            "baseline_doctor",
            "baseline_position",
            "baseline_minutes",
            "baseline_start_min",
            "baseline_end_min",
        ]
    ].merge(
        opt_sort[
            [
                "case_id",
                "date",
                "optimized_doctor",
                "optimized_position",
                "optimized_minutes",
                "optimized_start_min",
                "optimized_end_min",
                "recommended_move",
            ]
        ],
        on=["case_id", "date"],
        how="left",
    )

    out["per_case_saving_minutes"] = out["baseline_minutes"] - out["optimized_minutes"]
    out["doctor_changed"] = (out["baseline_doctor"] != out["optimized_doctor"]).astype(int)
    out["position_changed"] = (out["baseline_position"] != out["optimized_position"]).astype(int)
    out["schedule_changed"] = ((out["doctor_changed"] == 1) | (out["position_changed"] == 1)).astype(int)

    # Role labels for readability in dashboards.
    role_name = {
        roles.main_specialized_doctor: "main_specialized_doctor",
        roles.unspecialized_doctor: "unspecialized_doctor",
        roles.backup_doctor: "backup_doctor",
    }
    out["baseline_role"] = out["baseline_doctor"].map(role_name).fillna("other_doctor")
    out["optimized_role"] = out["optimized_doctor"].map(role_name).fillna("other_doctor")
    return out.sort_values(["date", "baseline_doctor", "baseline_position", "case_id"])


def build_residual_sigma_table(model_df: pd.DataFrame) -> Dict[Tuple[str, int], float]:
    grouped = (
        model_df.groupby(["physician", "is_specialized"])["residual"]
        .std()
        .dropna()
        .to_dict()
    )
    global_sigma = float(model_df["residual"].std()) if model_df["residual"].notna().any() else 5.0
    if np.isnan(global_sigma) or global_sigma <= 0:
        global_sigma = 5.0
    grouped[("__GLOBAL__", -1)] = global_sigma
    return grouped


def lookup_sigma(
    sigma_table: Dict[Tuple[str, int], float],
    physician: str,
    is_specialized: int,
) -> float:
    if (physician, is_specialized) in sigma_table:
        return sigma_table[(physician, is_specialized)]
    if (physician, 0) in sigma_table:
        return sigma_table[(physician, 0)]
    return sigma_table[("__GLOBAL__", -1)]


def monte_carlo_savings(
    reco_df: pd.DataFrame,
    sigma_table: Dict[Tuple[str, int], float],
    roles: RoleConfig,
    runs: int,
    seed: int,
) -> Dict[str, float]:
    if reco_df.empty:
        return {
            "mean_saving": 0.0,
            "p05_saving": 0.0,
            "p50_saving": 0.0,
            "p95_saving": 0.0,
            "probability_positive_saving": 0.0,
        }

    rng = np.random.default_rng(seed)
    savings = np.zeros(runs, dtype=float)

    for i in range(runs):
        baseline = 0.0
        optimized = 0.0
        for _, r in reco_df.iterrows():
            spec = int(r["is_specialized"])
            mu_base = float(r["pred_current_case_time"])
            mu_opt = float(r["pred_recommended_case_time"])
            sig_base = lookup_sigma(sigma_table, roles.unspecialized_doctor, spec)
            sig_opt = lookup_sigma(sigma_table, str(r["recommended_doctor"]), spec)

            draw_base = max(1.0, rng.normal(mu_base, sig_base))
            draw_opt = max(1.0, rng.normal(mu_opt, sig_opt))
            baseline += draw_base
            optimized += draw_opt
        savings[i] = baseline - optimized

    return {
        "mean_saving": float(np.mean(savings)),
        "p05_saving": float(np.quantile(savings, 0.05)),
        "p50_saving": float(np.quantile(savings, 0.50)),
        "p95_saving": float(np.quantile(savings, 0.95)),
        "probability_positive_saving": float(np.mean(savings > 0)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optimize reassignment of specialized cases away from unspecialized doctor using ML + optimization + Monte Carlo."
    )
    p.add_argument("--data-path", default="MSE433_M4_Data.xlsx")
    p.add_argument("--main-specialized-doctor", default="Dr. A")
    p.add_argument("--unspecialized-doctor", default="Dr. B")
    p.add_argument("--backup-doctor", default="Dr. C")
    p.add_argument("--max-main-cases-per-day", type=int, default=7)
    p.add_argument("--max-backup-cases-per-day", type=int, default=6)
    p.add_argument("--max-extra-main-per-day", type=int, default=2)
    p.add_argument("--max-extra-backup-per-day", type=int, default=2)
    p.add_argument("--mc-runs", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-csv", default="specialized_case_reassignment_recommendations.csv")
    p.add_argument("--schedule-output-csv", default="specialized_schedule_changes.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    roles = RoleConfig(
        main_specialized_doctor=args.main_specialized_doctor,
        unspecialized_doctor=args.unspecialized_doctor,
        backup_doctor=args.backup_doctor,
    )

    df = load_and_clean(args.data_path)
    pipe, model_df = train_duration_model(df)

    reco = optimize_reassignment(
        df=df,
        pipe=pipe,
        roles=roles,
        max_main_cases_per_day=args.max_main_cases_per_day,
        max_backup_cases_per_day=args.max_backup_cases_per_day,
        max_extra_main_per_day=args.max_extra_main_per_day,
        max_extra_backup_per_day=args.max_extra_backup_per_day,
    )

    if reco.empty:
        print("No specialized cases by unspecialized doctor were found. No reassignment needed.")
        return

    sigma_table = build_residual_sigma_table(model_df)
    mc = monte_carlo_savings(
        reco_df=reco,
        sigma_table=sigma_table,
        roles=roles,
        runs=args.mc_runs,
        seed=args.seed,
    )

    total_candidates = len(reco)
    moved = int(reco["move_recommended"].sum())
    moved_to_main = int((reco["recommended_doctor"] == roles.main_specialized_doctor).sum())
    moved_to_backup = int((reco["recommended_doctor"] == roles.backup_doctor).sum())

    deterministic_baseline = float(reco["pred_current_case_time"].sum())
    deterministic_optimized = float(reco["pred_recommended_case_time"].sum())
    deterministic_saving = deterministic_baseline - deterministic_optimized
    schedule_compare = build_schedule_comparison(df=df, reco_df=reco, roles=roles)
    full_baseline = float(schedule_compare["baseline_minutes"].sum()) if not schedule_compare.empty else 0.0
    full_optimized = float(schedule_compare["optimized_minutes"].sum()) if not schedule_compare.empty else 0.0
    full_saving = full_baseline - full_optimized

    print("=== Role Setup ===")
    print(f"Main specialized doctor: {roles.main_specialized_doctor}")
    print(f"Unspecialized doctor: {roles.unspecialized_doctor}")
    print(f"Backup doctor: {roles.backup_doctor}")

    print("\n=== Deterministic Optimization Result (ML Predictions) ===")
    print(f"Candidate specialized cases currently with unspecialized doctor: {total_candidates}")
    print(f"Reassignment moves recommended: {moved}")
    print(f"  -> moved to main specialized doctor: {moved_to_main}")
    print(f"  -> moved to backup doctor: {moved_to_backup}")
    print(f"Baseline predicted minutes: {deterministic_baseline:.1f}")
    print(f"Optimized predicted minutes: {deterministic_optimized:.1f}")
    print(f"Predicted saving: {deterministic_saving:.1f} minutes")
    print("\n=== Full Given-Schedule Impact (All Cases) ===")
    print(f"Full baseline minutes: {full_baseline:.1f}")
    print(f"Full optimized minutes: {full_optimized:.1f}")
    print(f"Full schedule saving: {full_saving:.1f} minutes")

    print("\n=== Monte Carlo Savings (Uncertainty-Aware) ===")
    print(f"Mean saving: {mc['mean_saving']:.1f} minutes")
    print(f"P05 / P50 / P95 saving: {mc['p05_saving']:.1f} / {mc['p50_saving']:.1f} / {mc['p95_saving']:.1f}")
    print(f"Probability of positive saving: {100.0 * mc['probability_positive_saving']:.1f}%")

    output_cols = [
        "case_id",
        "date",
        "physician",
        "current_role",
        "recommended_doctor",
        "recommended_role",
        "is_specialized",
        "note",
        "pred_current_case_time",
        "pred_recommended_case_time",
        "pred_saving_minutes",
        "move_recommended",
    ]
    reco.sort_values(["date", "pred_saving_minutes"], ascending=[True, False])[output_cols].to_csv(
        args.output_csv, index=False
    )
    schedule_cols = [
        "case_id",
        "date",
        "baseline_doctor",
        "optimized_doctor",
        "baseline_position",
        "optimized_position",
        "baseline_minutes",
        "optimized_minutes",
        "per_case_saving_minutes",
        "doctor_changed",
        "position_changed",
        "schedule_changed",
    ]
    schedule_compare[schedule_cols].sort_values(["date", "per_case_saving_minutes"], ascending=[True, False]).to_csv(
        args.schedule_output_csv, index=False
    )
    print(f"\nSaved recommendations: {args.output_csv}")
    print(f"Saved schedule changes: {args.schedule_output_csv}")


if __name__ == "__main__":
    main()
