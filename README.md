# Predictive Individual Coursework

Bank subscription prediction project with two core notebooks:
- `notebooks/eda.ipynb`: exploratory analysis, data-quality checks, and leakage-focused diagnostics.
- `notebooks/model_final.ipynb`: strict pre-contact feature preparation, split discipline, model selection, and final evaluation.

## Project Structure

- `data/bank.csv`: source dataset.
- `notebooks/eda.ipynb`: EDA and leakage investigation modules.
- `notebooks/model_final.ipynb`: final modeling pipeline.
- `Graph/`: figures exported by `eda.ipynb`.
- `reports/`: report outputs.

## What `eda.ipynb` Covers

`eda.ipynb` runs modular analysis over the bank dataset:
- dataset overview and target imbalance check;
- missing-value and `unknown` placeholder diagnostics;
- leakage-focused checks for `duration` and `campaign`;
- numeric distributions and outlier-oriented plots;
- categorical composition and conversion-rate comparisons;
- group-level subscription analysis (for example by age group and marital status).

The notebook writes charts into `Graph/`, including files such as:
- `module1_class_distribution.png`
- `module2_unknown_frequency.png`
- `module3_duration_boxplot_by_target.png`
- `module3_duration_binned_conversion_rate.png`
- `campaign_leakage_boxplot_by_target.png`
- `campaign_leakage_binned_subscription_rate.png`
- `module4_numeric_histograms.png`
- `module4_boxplots_balance_campaign_previous.png`
- `module5_conversion_rate_top_categories.png`
- `module6_group_subscription_rates.png`

## What `model_final.ipynb` Covers

`model_final.ipynb` implements the final modeling workflow:
- target detection and binary encoding (`yes=1`, `no=0`);
- data validation checks (missingness and `unknown` frequencies);
- feature-type audit (`demographic`, `financial`, `historical campaign`);
- leakage-oriented association scan (mutual information, numeric correlation, categorical rate spread);
- strict feature sets:
  - Set A: drop `duration`, `campaign`
  - Set B: drop `duration`, `campaign`, `day`, `month`, `contact`
- stratified split discipline using Set A:
  - `train_val/test = 80/20`
  - `train/val = 80/20` within `train_val` (overall `64/16/20`)
- model selection on train only with CV (`StratifiedKFold`, PR-AUC scoring) across:
  - Logistic Regression
  - Random Forest
  - HistGradientBoosting
- comparison table including CV performance and `test_pr_auc`;
- final evaluation, threshold selection on validation, calibration diagnostics, and error analysis.

## Environment and Dependencies

Core dependencies are listed in `requirements.txt`.

## How to Run

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open notebooks:
   - run `notebooks/eda.ipynb` first for exploration and figure generation;
   - run `notebooks/model_final.ipynb` for the final train/validation/test modeling workflow.

## Notes

- Current notebook cells use absolute local paths for `DATA_PATH` and some output directories. Update those paths if running on another machine.
- The modeling workflow assumes a strict pre-contact prediction setting and keeps the hold-out test set for final-stage evaluation.
