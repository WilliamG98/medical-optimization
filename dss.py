import pandas as pd
import streamlit as st

from optimize_doctor_assignment import (
    RoleConfig,
    build_residual_sigma_table,
    build_schedule_comparison,
    load_and_clean,
    monte_carlo_savings,
    optimize_reassignment,
    train_duration_model,
)


st.set_page_config(page_title="Specialized Assignment Optimizer", layout="wide")


@st.cache_data
def get_data(path: str) -> pd.DataFrame:
    return load_and_clean(path)


@st.cache_resource
def get_model(df: pd.DataFrame):
    return train_duration_model(df)


def summarize_candidates(df: pd.DataFrame, roles: RoleConfig) -> pd.DataFrame:
    mask = (df["physician"] == roles.unspecialized_doctor) & (df["is_specialized"] == 1)
    sub = df.loc[mask, ["case_id", "date", "physician", "note"]].copy()
    return sub.sort_values(["date", "case_id"])


st.title("Specialized Case Reassignment Dashboard")
st.caption("Reduce extra time from unspecialized doctor handling specialized operations")

with st.sidebar:
    st.header("Model Inputs")
    data_path = st.text_input("Data File", "MSE433_M4_Data.xlsx")

    st.subheader("Doctor Roles")
    main_specialized = st.text_input("Main Specialized Doctor", "Dr. A")
    unspecialized = st.text_input("Unspecialized Doctor", "Dr. B")
    backup_doctor = st.text_input("Backup Doctor", "Dr. C")

    st.subheader("Capacity Controls")
    max_main_cases_per_day = st.number_input("Max Main Cases/Day", min_value=1, max_value=20, value=7, step=1)
    max_backup_cases_per_day = st.number_input("Max Backup Cases/Day", min_value=1, max_value=20, value=6, step=1)
    max_extra_main_per_day = st.number_input("Max Extra Main Cases/Day", min_value=0, max_value=10, value=2, step=1)
    max_extra_backup_per_day = st.number_input("Max Extra Backup Cases/Day", min_value=0, max_value=10, value=2, step=1)

    st.subheader("Monte Carlo")
    mc_runs = st.number_input("MC Runs", min_value=200, max_value=50000, value=5000, step=200)
    seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)

roles = RoleConfig(
    main_specialized_doctor=main_specialized,
    unspecialized_doctor=unspecialized,
    backup_doctor=backup_doctor,
)

df = get_data(data_path)
pipe, model_df = get_model(df)

st.subheader("Role and Data Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Cases", int(df["case_id"].count()))
c2.metric("Specialized Cases", int(df["is_specialized"].sum()))
c3.metric("Unspecialized Doctor Cases", int((df["physician"] == roles.unspecialized_doctor).sum()))
c4.metric(
    "Specialized Cases by Unspecialized Doctor",
    int(((df["physician"] == roles.unspecialized_doctor) & (df["is_specialized"] == 1)).sum()),
)

candidate_df = summarize_candidates(df, roles)
with st.expander("Candidate Specialized Cases Currently Assigned to Unspecialized Doctor", expanded=False):
    st.dataframe(candidate_df, use_container_width=True)

reco = optimize_reassignment(
    df=df,
    pipe=pipe,
    roles=roles,
    max_main_cases_per_day=int(max_main_cases_per_day),
    max_backup_cases_per_day=int(max_backup_cases_per_day),
    max_extra_main_per_day=int(max_extra_main_per_day),
    max_extra_backup_per_day=int(max_extra_backup_per_day),
)

if reco.empty:
    st.warning("No specialized cases by unspecialized doctor found for optimization.")
    st.stop()

sigma_table = build_residual_sigma_table(model_df)
mc = monte_carlo_savings(
    reco_df=reco,
    sigma_table=sigma_table,
    roles=roles,
    runs=int(mc_runs),
    seed=int(seed),
)

baseline_minutes = float(reco["pred_current_case_time"].sum())
optimized_minutes = float(reco["pred_recommended_case_time"].sum())
det_saving = baseline_minutes - optimized_minutes
move_count = int(reco["move_recommended"].sum())
schedule_compare = build_schedule_comparison(df=df, reco_df=reco, roles=roles)

# Full-schedule impact (includes unchanged cases for context).
full_baseline_minutes = float(schedule_compare["baseline_minutes"].sum()) if not schedule_compare.empty else 0.0
full_optimized_minutes = float(schedule_compare["optimized_minutes"].sum()) if not schedule_compare.empty else 0.0
full_saving = full_baseline_minutes - full_optimized_minutes

st.subheader("Optimization Outcomes")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Candidate Cases", len(reco))
k2.metric("Recommended Moves", move_count)
k3.metric("Predicted Baseline Minutes", f"{baseline_minutes:.1f}")
k4.metric("Predicted Optimized Minutes", f"{optimized_minutes:.1f}", delta=f"{-det_saving:.1f}")

k5, k6, k7, k8 = st.columns(4)
k5.metric("Predicted Saving", f"{det_saving:.1f} min")
k6.metric("MC Mean Saving", f"{mc['mean_saving']:.1f} min")
k7.metric("MC P50 Saving", f"{mc['p50_saving']:.1f} min")
k8.metric("P(Saving > 0)", f"{100.0 * mc['probability_positive_saving']:.1f}%")

st.subheader("Given Schedule Impact (All Cases)")
s1, s2, s3 = st.columns(3)
s1.metric("Full Baseline Minutes", f"{full_baseline_minutes:.1f}")
s2.metric("Full Optimized Minutes", f"{full_optimized_minutes:.1f}", delta=f"{-full_saving:.1f}")
s3.metric("Full Schedule Saving", f"{full_saving:.1f} min")

st.subheader("Monte Carlo Uncertainty Band")
u1, u2, u3 = st.columns(3)
u1.metric("P05 Saving", f"{mc['p05_saving']:.1f} min")
u2.metric("P50 Saving", f"{mc['p50_saving']:.1f} min")
u3.metric("P95 Saving", f"{mc['p95_saving']:.1f} min")

st.subheader("Recommended Reassignments")
show_cols = [
    "case_id",
    "date",
    "physician",
    "recommended_doctor",
    "recommended_role",
    "is_specialized",
    "note",
    "pred_current_case_time",
    "pred_recommended_case_time",
    "pred_saving_minutes",
    "move_recommended",
]
reco_view = reco[show_cols].sort_values(["move_recommended", "pred_saving_minutes"], ascending=[False, False])
st.dataframe(reco_view, use_container_width=True)

st.subheader("What Could Be Changed In The Given Schedule")
base_change_cols = [
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
]

# Show updated schedule by default (all rows), with optional changed-only filter.
show_changed_only = st.checkbox("Show changed cases only", value=False)
change_view = schedule_compare[base_change_cols].copy()
if show_changed_only:
    change_view = change_view[(change_view["doctor_changed"] == 1) | (change_view["position_changed"] == 1)]
change_view = change_view.sort_values(
    ["date", "doctor_changed", "position_changed", "per_case_saving_minutes"],
    ascending=[True, False, False, False],
)
st.dataframe(change_view, use_container_width=True)

st.subheader("Updated Optimized Schedule (Default View)")
optimized_schedule_cols = [
    "case_id",
    "date",
    "optimized_doctor",
    "optimized_position",
    "optimized_start_min",
    "optimized_end_min",
    "optimized_minutes",
    "doctor_changed",
    "position_changed",
]
optimized_schedule = schedule_compare[optimized_schedule_cols].sort_values(
    ["date", "optimized_doctor", "optimized_position", "case_id"]
)
st.dataframe(optimized_schedule, use_container_width=True)

st.subheader("Daily Reassignment Plan")
daily_plan = (
    reco.assign(date_only=reco["date"].dt.date)
    .groupby(["date_only", "recommended_doctor"], as_index=False)
    .agg(
        cases=("case_id", "count"),
        total_pred_saving=("pred_saving_minutes", "sum"),
    )
)
st.dataframe(daily_plan.sort_values(["date_only", "cases"], ascending=[True, False]), use_container_width=True)

st.subheader("Daily Baseline vs Optimized Minutes")
daily_compare = (
    schedule_compare.assign(date_only=schedule_compare["date"].dt.date)
    .groupby("date_only", as_index=False)
    .agg(
        baseline_minutes=("baseline_minutes", "sum"),
        optimized_minutes=("optimized_minutes", "sum"),
    )
)
daily_compare["saving_minutes"] = daily_compare["baseline_minutes"] - daily_compare["optimized_minutes"]
st.dataframe(daily_compare.sort_values("date_only"), use_container_width=True)

csv_bytes = reco_view.to_csv(index=False).encode("utf-8")
changes_csv = change_view.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Recommendations CSV",
    data=csv_bytes,
    file_name="specialized_case_reassignment_dashboard_output.csv",
    mime="text/csv",
)
st.download_button(
    label="Download Schedule Changes CSV",
    data=changes_csv,
    file_name="specialized_schedule_changes.csv",
    mime="text/csv",
)
