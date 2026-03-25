# Medical Optimization Project

This README gives an overview of all items currently in this folder and explains:
1. Purpose
2. How each item works
3. How to run/open it

## Folder Inventory

| Item | Type | Purpose | How It Works | How to Run/Open |
|---|---|---|---|---|
| `.git/` | Directory | Git repository metadata | Stores commit history, branches, and repo state. Not part of model logic. | No direct run. Use Git commands (`git status`, `git log`, etc.) if needed. |
| `__pycache__/` | Directory | Python bytecode cache | Auto-generated `.pyc` cache from previous script runs/imports. | No direct run. Python manages this automatically. |
| `README.md` | File | Project documentation | Describes files, usage, and execution flow. | Open in editor/Markdown viewer. |
| `requirements-dss.txt` | File | Python dependencies | Lists libraries required by dashboard and optimization model. | Install with `pip install -r requirements-dss.txt`. |
| `MSE433_M4_Data.xlsx` | File | Source dataset | Excel workbook used by EDA, optimizer, and dashboard. | Open in Excel or load via scripts/notebook. |
| `EDA.ipynb` | File | Exploratory data analysis notebook | Cleans raw sheet format, computes trends, outliers, bottlenecks, and doctor comparisons. | Start Jupyter and open the notebook. Run cells top-to-bottom. |
| `optimize_doctor_assignment.py` | File | Core optimization + simulation model | Builds features from data, trains a Ridge model for `CASE TIME`, optimizes specialized-case reassignment away from unspecialized doctor under capacity constraints, then runs Monte Carlo to estimate time saved. Exports recommendation and schedule-change CSV outputs. | CLI: `python optimize_doctor_assignment.py --data-path MSE433_M4_Data.xlsx` |
| `dss.py` | File | Streamlit decision-support dashboard | Calls the optimizer pipeline and renders KPIs, baseline vs optimized schedule totals, schedule changes, full optimized schedule view, Monte Carlo uncertainty, and CSV downloads. | `streamlit run dss.py` |

## Execution Paths

### 1) Run dashboard (interactive)

```powershell
pip install -r requirements-dss.txt
streamlit run dss.py
```

Use sidebar controls to set:
- doctor roles (`A` main specialized, `B` unspecialized, `C` backup)
- per-day capacity constraints
- Monte Carlo runs and seed

### 2) Run optimizer from command line

```powershell
python optimize_doctor_assignment.py --data-path MSE433_M4_Data.xlsx
```

Key optional args:
- `--main-specialized-doctor`
- `--unspecialized-doctor`
- `--backup-doctor`
- `--max-main-cases-per-day`
- `--max-backup-cases-per-day`
- `--max-extra-main-per-day`
- `--max-extra-backup-per-day`
- `--mc-runs`
- `--seed`
- `--output-csv`
- `--schedule-output-csv`

Default outputs:
- `specialized_case_reassignment_recommendations.csv`
- `specialized_schedule_changes.csv`

### 3) Run EDA notebook

```powershell
jupyter notebook
```

Then open `EDA.ipynb` and execute all cells in order.

## EDA Trends That Motivated The Solution

The optimization approach in `optimize_doctor_assignment.py` was chosen based on patterns found in `EDA.ipynb`:

1. Doctor-level performance gap on core duration metrics.
- Mean `CASE TIME` was materially lower for `Dr. A` than `Dr. B` (with `Dr. B` showing higher spread/outliers).
- This pointed to reassigning selected specialized cases away from the unspecialized doctor (`Dr. B`) toward `Dr. A` and `Dr. C`.

2. Procedure-phase bottlenecks were strongly linked to total time.
- `SKIN-SKIN`, `LA DWELL`, and `ABL DURATION` were among the strongest drivers of total case duration.
- This supported a case-level predictive model (ML) that estimates expected time by doctor and case complexity.

3. Complexity was unevenly distributed.
- `Note` flags (`CTI`, `BOX`, `PST`, `SVC`, `AAFL`, `TROUBLE`) indicated non-standard/specialized cases.
- Complex cases clustered in ways that can inflate specific doctor/day workloads.
- This justified a targeted reassignment optimizer for specialized cases rather than broad random reallocation.

4. First-case overhead existed, but simple fatigue trends were weak.
- First cases were often slower than later same-day cases.
- No clear consistent “more cases per day = slower” fatigue pattern was found.
- This indicated the solution should optimize assignment and schedule structure under capacity constraints, not just cap daily case counts.

5. Tail risk and uncertainty matter.
- Outlier long-duration cases were present, especially in segments involving higher complexity/variance.
- This motivated adding Monte Carlo simulation to estimate uncertainty-aware savings (not just point estimates).

How these trends map to implementation:
- ML module: predicts case-time under different doctor assignments.
- Optimization module: chooses reassignment plan per day with doctor capacity limits.
- Monte Carlo module: quantifies likely savings range and probability of positive improvement.

## Notes

- The model optimizes the given schedule by reassigning specialized cases from the unspecialized doctor, subject to capacity limits.
- Monte Carlo output is used to estimate uncertainty in time savings (mean, P05/P50/P95, probability of positive savings).
- `__pycache__/` and `.git/` are infrastructure folders, not analysis artifacts.
