# Medical Optimization Project

This repository analyzes electrophysiology case data and builds a decision-support workflow for one specific operational question:

How much time can be saved if specialized cases currently assigned to an unspecialized doctor are reassigned to better-suited doctors, while respecting daily capacity limits?

The project contains:

- a raw Excel workbook
- an exploratory notebook
- a Python optimization pipeline
- a Streamlit dashboard
- dependency metadata
- generated CSV outputs from the current workspace run

## Repository Map

| Path | Type | What it contains | Why it matters |
| --- | --- | --- | --- |
| `README.md` | Markdown | Project documentation | High-level guide to the data, code, findings, and how to run everything |
| `MSE433_M4_Data.xlsx` | Excel workbook | Source case data | Main input used by the notebook, optimizer, and dashboard |
| `EDA.ipynb` | Jupyter notebook | Exploratory data analysis and optimization-oriented signal hunting | Shows the data-cleaning logic, distributions, physician comparisons, correlations, and complex-case trends that motivate the optimization approach |
| `optimize_doctor_assignment.py` | Python script | End-to-end ML + reassignment + Monte Carlo pipeline | Core command-line entry point; produces recommendations and full schedule comparison outputs |
| `dss.py` | Python script | Streamlit dashboard | Interactive front end for running the optimizer and inspecting outputs |
| `requirements-dss.txt` | Text | Python dependencies | Minimum package list for the dashboard and optimization code |
| `specialized_case_reassignment_recommendations.csv` | CSV | Generated recommendation output | Current workspace output from running the optimizer with default settings |
| `specialized_schedule_changes.csv` | CSV | Generated full schedule comparison output | Current workspace output showing baseline vs optimized doctor/position/minutes |
| `__pycache__/` | Directory | Python bytecode cache | Auto-generated runtime cache; not source logic |
| `.git/` | Directory | Git metadata | Version-control history and repository state |

## Problem Setup

The code assumes three role types:

- main specialized doctor: default `Dr. A`
- unspecialized doctor: default `Dr. B`
- backup doctor: default `Dr. C`

The optimization target is narrow by design:

- only cases flagged as specialized/non-standard are candidates
- only candidate cases currently assigned to the unspecialized doctor are eligible to move
- moves are limited by daily doctor capacity controls

## Data File: `MSE433_M4_Data.xlsx`

The workbook has one sheet:

- `All Data`

Raw workbook structure:

- raw shape: `154 x 22`
- row index `2` contains the actual column names
- row index `4` is the first case row
- the sheet includes a title row and a units/descriptor row above the true data

After the project cleaning logic is applied:

- cleaned case rows: `150`
- cleaned columns: `30`
- date range: `2025-01-13` to `2025-10-08`
- physicians present: `Dr. A`, `Dr. B`, `Dr. C`

Important raw columns renamed by the pipeline include:

- `CASE #` -> `case_id`
- `DATE` -> `date`
- `PHYSICIAN` -> `physician`
- `PT PREP/INTUBATION`
- `ACCESSS`
- `TSP`
- `PRE-MAP`
- `ABL DURATION`
- `ABL TIME`
- `#ABL`
- `#APPLICATIONS`
- `LA DWELL TIME`
- `CASE TIME`
- `SKIN-SKIN`
- `POST CARE/EXTUBATION`
- `PT IN-OUT`
- `Note`

The code also derives keyword flags from the `Note` field:

- `flag_cti`
- `flag_box`
- `flag_pst`
- `flag_svc`
- `flag_aafl`
- `flag_trouble`
- `is_specialized`
- `month`

Keyword totals in the bundled dataset:

- `CTI`: `6`
- `BOX`: `15`
- `PST`: `7`
- `SVC`: `2`
- `AAFL`: `1`
- `TROUBLE`: `1`

Specialized-case counts:

- total specialized cases: `23`
- specialized cases by `Dr. A`: `14`
- specialized cases by `Dr. B`: `8`
- specialized cases by `Dr. C`: `1`

One implementation detail to know: the workbook still carries a leading blank/unnamed column through the load step. It is not used by the model or optimization logic, but it remains present in the cleaned DataFrame.

## Notebook: `EDA.ipynb`

The notebook is titled `MSE433 M4 EDA + Optimization Signals` and is organized around:

- custom parsing of the raw workbook
- data quality checks
- univariate distributions and outliers
- physician and calendar trends
- correlations and structural drivers
- a simple optimization-oriented baseline model
- first-case overhead
- complexity distribution
- direct `Dr. A` vs `Dr. B` complex-case comparison

Notebook sections explicitly documented in saved markdown cells:

- data quality and cleaning
- physician and calendar trends
- outlier behavior
- correlation/driver signals
- baseline scheduling insights

Key signals reflected in the notebook and reproducible from the code/data:

- Mean `CASE TIME` by physician:
  - `Dr. A`: `33.56`
  - `Dr. B`: `49.37`
  - `Dr. C`: `39.67`
- Median `CASE TIME` by physician:
  - `Dr. A`: `31.0`
  - `Dr. B`: `43.0`
  - `Dr. C`: `39.0`
- Mean `CASE TIME` for specialized cases:
  - `Dr. A`: `39.64`
  - `Dr. B`: `78.43`
  - `Dr. C`: `83.00` on a single specialized case
- Mean `CASE TIME` for non-specialized cases:
  - `Dr. A`: `32.04`
  - `Dr. B`: `45.53`
  - `Dr. C`: `36.57`
- First cases of the day are slower on average than later same-day cases:
  - first case mean: `47.30`
  - non-first case mean: `38.79`

Strongest correlations with `CASE TIME` among the numeric fields:

- `SKIN-SKIN`: `0.987`
- `PT IN-OUT`: `0.913`
- `LA DWELL`: `0.793`
- `ABL DURATION`: `0.762`
- `TSP`: `0.495`

The notebook's role in the project is explanatory, not operational:

- it motivates why reassignment should focus on complex/specialized cases
- it shows why doctor identity matters for duration
- it provides the evidence that a simple predictive model and schedule optimizer are reasonable next steps

## Core Script: `optimize_doctor_assignment.py`

This is the main production script. It implements five distinct stages.

### 1. Data loading and cleaning

`load_and_clean(path)`:

- reads the `All Data` sheet from Excel
- uses workbook row `2` as headers
- starts case records from row `4`
- converts key columns to numeric/date types
- extracts specialization flags from the `Note` field
- creates a `month` feature

### 2. Duration model

`train_duration_model(df)` trains a scikit-learn pipeline:

- categorical feature:
  - `physician` via `OneHotEncoder`
- numeric features:
  - `is_specialized`
  - `flag_cti`
  - `flag_box`
  - `flag_pst`
  - `flag_svc`
  - `flag_aafl`
  - `flag_trouble`
  - `month`
- regressor:
  - `Ridge(alpha=1.0)`

On the bundled dataset, the in-sample training fit is:

- modeled rows: `144`
- `R^2`: `0.5531`
- MAE: `7.9988` minutes
- residual standard deviation: `11.6594` minutes

This model is intentionally simple. It acts as a ranking/proxy model for schedule decisions, not a clinically validated forecasting system.

### 3. Daily reassignment optimizer

`optimize_reassignment(...)`:

- filters to specialized cases currently assigned to the unspecialized doctor
- predicts counterfactual case duration for each candidate under each doctor role
- applies a per-day dynamic-programming assignment routine
- respects:
  - `max_main_cases_per_day`
  - `max_backup_cases_per_day`
  - `max_extra_main_per_day`
  - `max_extra_backup_per_day`

The daily solver considers three choices per candidate:

- keep with unspecialized doctor
- move to main specialized doctor
- move to backup doctor

### 4. Full schedule comparison

`build_schedule_comparison(...)` reconstructs the full schedule and reports:

- baseline vs optimized doctor
- baseline vs optimized position within the day/doctor queue
- baseline vs optimized case minutes
- start/end minute offsets
- whether doctor changed
- whether position changed
- whether the overall schedule row changed

### 5. Monte Carlo uncertainty analysis

`monte_carlo_savings(...)`:

- uses residual variance from the fitted model
- builds sigma values by `(physician, is_specialized)`
- simulates baseline and optimized total times across repeated runs
- reports:
  - mean saving
  - `P05`
  - `P50`
  - `P95`
  - probability of positive saving

### Command-line interface

Run:

```powershell
python optimize_doctor_assignment.py --data-path MSE433_M4_Data.xlsx
```

Available arguments:

- `--data-path`
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

Default values in the script:

- main specialized doctor: `Dr. A`
- unspecialized doctor: `Dr. B`
- backup doctor: `Dr. C`
- max main cases/day: `7`
- max backup cases/day: `6`
- max extra main/day: `2`
- max extra backup/day: `2`
- Monte Carlo runs: `5000`
- seed: `42`
- recommendations output: `specialized_case_reassignment_recommendations.csv`
- schedule output: `specialized_schedule_changes.csv`

## Dashboard: `dss.py`

The Streamlit app wraps the same optimizer pipeline in an interactive UI.

Key behaviors:

- uses `st.cache_data` for data loading
- uses `st.cache_resource` for model training
- lets the user override doctor role names
- lets the user change daily capacity controls
- lets the user change Monte Carlo run count and seed
- shows candidate specialized cases for the unspecialized doctor
- shows deterministic and Monte Carlo KPI cards
- shows full-schedule before/after impact
- shows the optimized schedule order
- lets the user download recommendation and schedule CSVs

Run:

```powershell
streamlit run dss.py
```

Sidebar inputs in the app:

- data file path
- main specialized doctor
- unspecialized doctor
- backup doctor
- max main cases/day
- max backup cases/day
- max extra main/day
- max extra backup/day
- Monte Carlo runs
- seed

## Dependencies: `requirements-dss.txt`

The dependency file contains:

- `streamlit>=1.36`
- `pandas>=2.0`
- `numpy>=1.24`
- `openpyxl>=3.1`
- `scikit-learn>=1.3`

Install with:

```powershell
pip install -r requirements-dss.txt
```

## Generated Outputs In The Current Workspace

Running the optimizer with default settings on `MSE433_M4_Data.xlsx` produced two CSV files.

### 1. `specialized_case_reassignment_recommendations.csv`

Purpose:

- one row per candidate specialized case currently assigned to the unspecialized doctor

Columns:

- `case_id`
- `date`
- `physician`
- `current_role`
- `recommended_doctor`
- `recommended_role`
- `is_specialized`
- `note`
- `pred_current_case_time`
- `pred_recommended_case_time`
- `pred_saving_minutes`
- `move_recommended`

Current workspace result:

- rows: `8`
- all `8` candidates are recommended to move
- all `8` are moved from `Dr. B` to `Dr. A`

### 2. `specialized_schedule_changes.csv`

Purpose:

- full schedule view comparing baseline and optimized assignments

Columns:

- `case_id`
- `date`
- `baseline_doctor`
- `optimized_doctor`
- `baseline_position`
- `optimized_position`
- `baseline_minutes`
- `optimized_minutes`
- `per_case_saving_minutes`
- `doctor_changed`
- `position_changed`
- `schedule_changed`

Current workspace result:

- full schedule rows: `150`
- rows with `schedule_changed = 1`: `14`

## Default Run Results On The Bundled Data

The following numbers come from running:

```powershell
python optimize_doctor_assignment.py --data-path MSE433_M4_Data.xlsx
```

Observed output:

- candidate specialized cases currently with `Dr. B`: `8`
- reassignment moves recommended: `8`
- moved to `Dr. A`: `8`
- moved to `Dr. C`: `0`
- deterministic baseline predicted minutes: `575.4`
- deterministic optimized predicted minutes: `463.4`
- deterministic predicted saving: `112.0` minutes
- full baseline schedule minutes: `6034.4`
- full optimized schedule minutes: `5922.4`
- full schedule saving: `112.0` minutes
- Monte Carlo mean saving: `111.0` minutes
- Monte Carlo `P05 / P50 / P95`: `14.4 / 110.4 / 208.9`
- probability of positive saving: `97.3%`

Top predicted savings in the current recommendation file are the same magnitude for several cases, about `14.0` minutes per moved case.

## How To Use The Project

### Install dependencies

```powershell
pip install -r requirements-dss.txt
```

### Run the optimizer from the command line

```powershell
python optimize_doctor_assignment.py --data-path MSE433_M4_Data.xlsx
```

### Launch the dashboard

```powershell
streamlit run dss.py
```

### Open the notebook

```powershell
jupyter notebook
```

Then open `EDA.ipynb` and run the cells from top to bottom.

## Practical Interpretation

For the included dataset, the project's main takeaway is consistent across the notebook, model, and optimization script:

- `Dr. B` is materially slower than `Dr. A`, especially on specialized cases
- specialized/non-standard cases appear to be the right target for reassignment
- moving a small set of those cases can produce meaningful total time savings
- the default optimization result suggests the savings are likely positive even after uncertainty is added

## Limitations And Caveats

- Specialization is inferred from note keywords, not a richer clinical taxonomy.
- The predictive model is a simple linear proxy with in-sample reporting only.
- The optimizer is case-reassignment focused; it does not solve full operating-room sequencing from scratch.
- Capacity controls are configurable but still simplified relative to real-world staffing and resource constraints.
- The workbook parser depends on the current row layout of the Excel file.
- The generated CSVs are outputs from a run in this workspace, not source inputs.
