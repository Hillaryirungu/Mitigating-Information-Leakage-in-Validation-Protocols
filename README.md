# Mitigating Information Leakage in Validation Protocols

A reproducible machine-learning demonstration of how **target leakage during feature selection can inflate holdout performance**, and how to prevent it with a scikit-learn pipeline.

## Why this matters

A model evaluation is only credible when holdout data remains unseen throughout model development. Leakage occurs when information from validation or test observations influences preprocessing, feature selection, model fitting, or tuning.

This repository isolates one common failure: selecting features using the complete dataset and its labels before evaluating on a holdout set.

## Experiment

The script generates a synthetic binary-classification dataset with 1,200 observations and 1,000 features. It compares two workflows using the same 70/30 stratified split and logistic-regression classifier.

| Workflow | Feature selection | Holdout accuracy | Holdout ROC-AUC |
|---|---|---:|---:|
| Leakage-prone (incorrect) | Fit on all observations and labels before splitting | 0.7083 | 0.7824 |
| Leakage-free (correct) | Fit only on training data inside a pipeline | 0.6528 | 0.7199 |

These results use the fixed random seed (42) in the included script. They illustrate this synthetic experiment; they are not evidence of performance on a real-world or financial dataset.

## What causes the leakage?

The incorrect workflow fits SelectKBest on the complete feature matrix and target vector before respecting the train/test boundary. The selector therefore uses holdout labels to decide which features appear most predictive. The classifier is then evaluated on data that has already influenced feature selection.

The correct workflow places feature selection, scaling, and classification in a single Pipeline. Each learned transformation is fitted on training data only.

## Run the experiment

Requirements: Python 3.9 or later and the packages in requirements.txt.

Clone and set up:

```bash
git clone https://github.com/Hillaryirungu/Mitigating-Information-Leakage-in-Validation-Protocols.git
cd Mitigating-Information-Leakage-in-Validation-Protocols
python -m venv .venv
```

Activate the environment:

**macOS / Linux**
```bash
source .venv/bin/activate
```

**Windows PowerShell**
```powershell
.venv\\Scripts\\Activate.ps1
```

Install and run:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python leakage_demo.py
```

## Good validation practice

1. Split observations before fitting learned preprocessing or feature-selection steps.
2. Fit transformations on training data only; apply the fitted transformations to validation and test data.
3. Use pipelines so cross-validation refits every transformation within each training fold.
4. Use validation data for model selection and tuning.
5. Keep the final test set untouched until final evaluation.
6. For time-dependent financial data, use chronological or walk-forward splits rather than random shuffling.

## Scope and limitations

- The dataset is synthetic and generated locally; no external download is required.
- The demonstration focuses on feature-selection leakage. Other leakage paths include imputation, scaling, feature engineering, target encoding, and repeated test-set tuning.
- A higher score in the leakage-prone workflow is not a model improvement; it is a biased evaluation.
- This is an educational demonstration, not an investment model or financial recommendation.

## Project structure

```text
.
├── README.md
├── leakage_demo.py
├── requirements.txt
└── LICENSE
```

## Author

**Hillary Irungu** · [GitHub](https://github.com/Hillaryirungu) · [Portfolio](https://hillaryirungu.github.io)

## License

Released under the MIT License. See LICENSE.
