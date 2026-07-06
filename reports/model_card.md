# Model Card: House Prices Regression

## Problem

Predict residential sale prices in Ames, Iowa using structured property attributes. This is a supervised regression task based on the Kaggle House Prices competition.

## Business Framing

The workflow mirrors a real estate valuation support use case: combine property size, quality, location, age, and amenity signals to produce consistent price estimates. The model is not intended to replace professional appraisal, but it can support pricing review, lead prioritization, and anomaly detection.

## Data

- Training rows: 1,460
- Test rows: 1,459
- Target: `SalePrice`
- Evaluation scale: RMSE on `log1p(SalePrice)`, equivalent to optimizing relative pricing error

Raw Kaggle CSV files are intentionally excluded from Git. Download them from the competition page and place `train.csv` and `test.csv` in the project root.

## Approach

- Domain-aware missing-value treatment for garage, basement, pool, fireplace, alley, and veneer fields
- Group-wise imputation for `LotFrontage` by `Neighborhood`
- Feature engineering for total square footage, bathrooms, house age, remodel age, amenity flags, and quality-size interaction
- Location interaction features such as `Neighborhood x OverallQual`, `Neighborhood x MSSubClass`, and `MSSubClass x OverallQual`
- Skew handling through `log1p` and Yeo-Johnson transformed numeric features
- Known outlier removal for very large low-priced homes
- Weighted blend of Lasso, Ridge, and XGBoost predictions

## Validation Results

| Model or Blend | OOF RMSE |
|---|---:|
| Weighted blend, 5-fold CV | 0.1074 |
| Lasso | 0.1092 |
| Ridge | 0.1105 |
| XGBoost | 0.1136 |

Repeated seed checks showed the blend was more stable than individual tree models:

| Model or Blend | Mean RMSE | Std |
|---|---:|---:|
| Weighted blend | 0.1076 | 0.0007 |
| Lasso | 0.1092 | 0.0007 |
| XGBoost | 0.1134 | 0.0016 |

## Limitations

- The dataset is small and competition-specific.
- Manual neighborhood tags are interpretable but should be revalidated before reuse in another city or market.
- Sale price distributions can shift over time; this model does not include macroeconomic or interest-rate features.
- Public leaderboard performance may vary from cross-validation because the hidden test set is fixed and partially private.

## Production Next Steps

- Add model explainability with SHAP or permutation importance.
- Package the preprocessing and model as a serialized inference pipeline.
- Add input schema validation for real-world property feeds.
- Track prediction intervals, not just point estimates.
