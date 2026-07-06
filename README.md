# House Prices Regression: Real Estate Valuation with Tabular ML

This project predicts residential sale prices using the Kaggle **House Prices - Advanced Regression Techniques** dataset. It is structured as a portfolio-ready machine learning project for real estate valuation support: clear business framing, reproducible training code, documented validation, and a clean notebook for review.

## Business Problem

Accurate property valuation helps real estate teams review listing prices, detect underpriced or overpriced properties, and prioritize human appraisal work. This project builds a regression model that estimates home sale prices from property attributes such as size, quality, neighborhood, age, basement/garage details, and sale conditions.

## Results

The final solution uses a weighted blend of Lasso, Ridge, and XGBoost trained on `log1p(SalePrice)`.

| Model or Blend | Validation RMSE |
|---|---:|
| Weighted blend, 5-fold OOF | 0.1074 |
| Weighted blend, 10-seed mean | 0.1075 |
| Lasso | 0.1092 |
| Ridge | 0.1105 |
| XGBoost | 0.1136 |

The blend improved stability versus individual tree models. Repeated seed validation produced a mean RMSE of `0.1075` with a standard deviation of `0.0006-0.0007`.

## Modeling Approach

- Treated domain-specific missing values as meaningful signals, such as no garage, no basement, no fireplace, or no pool.
- Imputed `LotFrontage` by neighborhood median.
- Added business-relevant features: total square footage, total bathrooms, house age, remodel age, garage/basement/fireplace/pool flags, and quality-size interaction.
- Added neighborhood interaction features: `Neighborhood x OverallQual`, `Neighborhood x MSSubClass`, and `MSSubClass x OverallQual`.
- Added carefully reviewed neighborhood premium/discount tags from residual analysis.
- Applied `log1p` and Yeo-Johnson transforms to skewed numeric variables.
- Removed known extreme outliers where very large homes had unusually low sale prices.
- Blended Lasso, Ridge, and XGBoost predictions.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── environment.yml
├── src/
│   └── train.py
├── notebooks/
│   └── 01_portfolio_workflow.ipynb
├── reports/
│   └── model_card.md
└── .gitignore
```

The clean portfolio workflow is in `notebooks/01_portfolio_workflow.ipynb`, and the reproducible implementation is in `src/train.py`. The original exploratory notebooks are kept locally but excluded from Git because they contain trial-and-error analysis.

## How to Run

Download `train.csv` and `test.csv` from Kaggle and place them in the project root.

Create the environment:

```bash
conda env create -f environment.yml
conda activate house-prices-portfolio
```

Run training and generate a submission file:

```bash
python src/train.py --data-dir . --output submission_manual_tag_blend.csv
```

Run cross-validation before generating the submission:

```bash
python src/train.py --data-dir . --cv --output submission_manual_tag_blend.csv
```

## Skills Demonstrated

- Tabular regression modeling
- Feature engineering for real estate data
- Cross-validation and repeated seed stability checks
- Leakage-aware preprocessing
- Model blending
- Kaggle-style reproducible ML workflow
- Business-facing model documentation

## Notes

Kaggle raw data and generated submissions are excluded from Git via `.gitignore`. This keeps the repository lightweight and avoids redistributing competition data.
