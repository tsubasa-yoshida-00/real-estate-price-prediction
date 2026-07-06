"""Train a reproducible House Prices regression blend.

This script turns the exploratory Kaggle notebook into a portfolio-ready,
repeatable workflow: feature engineering, cross-validation, full training, and
submission-file generation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler, StandardScaler
from xgboost import XGBRegressor


RANDOM_STATE = 42
TARGET = "SalePrice"
ID_COL = "Id"

NONE_FILL_COLS = [
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "MasVnrType",
]

ZERO_FILL_COLS = [
    "GarageYrBlt",
    "GarageArea",
    "GarageCars",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "MasVnrArea",
]

MODE_FILL_COLS = [
    "MSZoning",
    "Electrical",
    "KitchenQual",
    "Exterior1st",
    "Exterior2nd",
    "SaleType",
    "Functional",
]

SKEW_LOG_CANDIDATES = [
    "LotArea",
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "PoolArea",
    "MiscVal",
]


def make_one_hot_encoder() -> OneHotEncoder:
    """Create a OneHotEncoder that works across recent scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def add_common_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalSF"] = df["TotalBsmtSF"].fillna(0) + df["1stFlrSF"].fillna(0) + df["2ndFlrSF"].fillna(0)
    df["TotalBath"] = (
        df["FullBath"].fillna(0)
        + 0.5 * df["HalfBath"].fillna(0)
        + df["BsmtFullBath"].fillna(0)
        + 0.5 * df["BsmtHalfBath"].fillna(0)
    )
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["IsRemodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)
    df["HasGarage"] = (df["GarageArea"].fillna(0) > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"].fillna(0) > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"].fillna(0) > 0).astype(int)
    df["HasPool"] = (df["PoolArea"].fillna(0) > 0).astype(int)
    df["OverallQual_x_GrLivArea"] = df["OverallQual"] * df["GrLivArea"]
    return df


def add_manual_neighborhood_tags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    total_sf = df["TotalBsmtSF"].fillna(0) + df["1stFlrSF"].fillna(0) + df["2ndFlrSF"].fillna(0)
    house_age = df["YrSold"] - df["YearBuilt"]

    tag_rules = {
        "Tag_OldTown_quality_premium": (df["Neighborhood"] == "OldTown") & (df["OverallQual"] >= 7),
        "Tag_OldTown_subclass75_premium": (df["Neighborhood"] == "OldTown") & (df["MSSubClass"] == 75),
        "Tag_OldTown_subclass45_discount": (df["Neighborhood"] == "OldTown") & (df["MSSubClass"] == 45),
        "Tag_IDOTRR_abnormal_discount": (df["Neighborhood"] == "IDOTRR") & (df["SaleCondition"] == "Abnorml"),
        "Tag_IDOTRR_subclass20_discount": (df["Neighborhood"] == "IDOTRR") & (df["MSSubClass"] == 20),
        "Tag_IDOTRR_quality_premium": (df["Neighborhood"] == "IDOTRR") & (df["OverallQual"] >= 7),
        "Tag_Edwards_subclass30_discount": (df["Neighborhood"] == "Edwards") & (df["MSSubClass"] == 30),
        "Tag_Edwards_newer_premium": (df["Neighborhood"] == "Edwards") & (house_age <= 50),
        "Tag_Edwards_quality_premium": (df["Neighborhood"] == "Edwards") & (df["OverallQual"] >= 7),
        "Tag_Sawyer_very_old_discount": (df["Neighborhood"] == "Sawyer") & (house_age >= 80),
        "Tag_Sawyer_subclass30_discount": (df["Neighborhood"] == "Sawyer") & (df["MSSubClass"] == 30),
        "Tag_NridgHt_small_discount": (df["Neighborhood"] == "NridgHt") & (total_sf <= 3000),
        "Tag_NridgHt_large_premium": (df["Neighborhood"] == "NridgHt") & (total_sf >= 3800),
        "Tag_NridgHt_townhouse_discount": (df["Neighborhood"] == "NridgHt") & (df["MSSubClass"].isin([120, 160])),
        "Tag_StoneBr_large_premium": (df["Neighborhood"] == "StoneBr") & (total_sf >= 3600),
        "Tag_StoneBr_partial_premium": (df["Neighborhood"] == "StoneBr") & (df["SaleCondition"] == "Partial"),
        "Tag_StoneBr_townhouse_discount": (df["Neighborhood"] == "StoneBr") & (df["MSSubClass"].isin([120, 160])),
    }

    for tag_name, mask in tag_rules.items():
        df[tag_name] = mask.astype(int)

    premium_cols = [col for col in tag_rules if col.endswith("premium")]
    discount_cols = [col for col in tag_rules if col.endswith("discount")]
    df["ManualPremiumTagCount"] = df[premium_cols].sum(axis=1)
    df["ManualDiscountTagCount"] = df[discount_cols].sum(axis=1)
    return df


def build_feature_frames(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_part = train_df.copy()
    test_part = test_df.copy()
    train_part["is_train"] = 1
    test_part["is_train"] = 0
    test_part[TARGET] = np.nan
    all_df = pd.concat([train_part, test_part], axis=0, ignore_index=True, sort=False)

    for col in NONE_FILL_COLS:
        if col in all_df.columns:
            all_df[col] = all_df[col].fillna("None")
    for col in ZERO_FILL_COLS:
        if col in all_df.columns:
            all_df[col] = all_df[col].fillna(0)
    for col in MODE_FILL_COLS:
        if col in all_df.columns:
            all_df[col] = all_df[col].fillna(all_df[col].mode(dropna=True)[0])

    if "LotFrontage" in all_df.columns:
        all_df["LotFrontage"] = all_df.groupby("Neighborhood")["LotFrontage"].transform(lambda s: s.fillna(s.median()))
        all_df["LotFrontage"] = all_df["LotFrontage"].fillna(all_df["LotFrontage"].median())

    all_df = add_common_features(all_df)
    all_df["NbhdOverallQual"] = all_df["Neighborhood"].astype(str) + "_Q" + all_df["OverallQual"].astype(str)
    all_df["MSSubClass_str"] = all_df["MSSubClass"].astype(str)
    all_df["NbhdMSSubClass"] = all_df["Neighborhood"].astype(str) + "_" + all_df["MSSubClass_str"]
    all_df["SubClassOverallQual"] = all_df["MSSubClass_str"] + "_Q" + all_df["OverallQual"].astype(str)
    all_df = add_manual_neighborhood_tags(all_df)

    for col in SKEW_LOG_CANDIDATES:
        if col in all_df.columns:
            all_df[f"{col}_log1p"] = np.log1p(all_df[col].clip(lower=0))
            all_df[f"{col}_yeojohnson"] = PowerTransformer(method="yeo-johnson", standardize=False).fit_transform(
                all_df[[col]].fillna(0)
            ).ravel()

    final_train = all_df[all_df["is_train"] == 1].drop(columns=["is_train"]).copy()
    final_test = all_df[all_df["is_train"] == 0].drop(columns=["is_train", TARGET]).copy()

    famous_outlier_mask = (final_train["GrLivArea"] > 4000) & (final_train[TARGET] < 300000)
    return final_train.loc[~famous_outlier_mask].copy(), final_test


def make_linear_pipeline(num_cols: list[str], cat_cols: list[str], estimator, scaler) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", scaler)]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def make_tree_pipeline(num_cols: list[str], cat_cols: list[str], estimator) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def build_models(num_cols: list[str], cat_cols: list[str]) -> dict[str, Pipeline]:
    return {
        "lasso": make_linear_pipeline(
            num_cols,
            cat_cols,
            Lasso(alpha=0.0004, max_iter=60000, random_state=RANDOM_STATE),
            RobustScaler(),
        ),
        "ridge": make_linear_pipeline(
            num_cols,
            cat_cols,
            Ridge(alpha=20.0, random_state=RANDOM_STATE),
            StandardScaler(),
        ),
        "xgb": make_tree_pipeline(
            num_cols,
            cat_cols,
            XGBRegressor(
                n_estimators=3500,
                learning_rate=0.02,
                max_depth=2,
                min_child_weight=2,
                subsample=0.85,
                colsample_bytree=0.75,
                reg_alpha=0.001,
                reg_lambda=5.0,
                objective="reg:squarederror",
                eval_metric="rmse",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=0,
            ),
        ),
    }


def run_cross_validation(models: dict[str, Pipeline], x: pd.DataFrame, y: pd.Series, folds: int) -> pd.DataFrame:
    cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    oof_predictions = {}

    for name, model in models.items():
        pred = cross_val_predict(model, x, y, cv=cv, n_jobs=1)
        oof_predictions[name] = pred
        rows.append({"model": name, "oof_rmse": np.sqrt(mean_squared_error(y, pred))})

    blend = 0.50 * oof_predictions["lasso"] + 0.16 * oof_predictions["ridge"] + 0.34 * oof_predictions["xgb"]
    rows.append({"model": "weighted_blend", "oof_rmse": np.sqrt(mean_squared_error(y, blend))})
    return pd.DataFrame(rows).sort_values("oof_rmse")


def train_and_predict(models: dict[str, Pipeline], x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> np.ndarray:
    test_predictions = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        test_predictions[name] = model.predict(x_test)

    blend_log = 0.50 * test_predictions["lasso"] + 0.16 * test_predictions["ridge"] + 0.34 * test_predictions["xgb"]
    return np.maximum(np.expm1(blend_log), 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the House Prices portfolio model.")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Directory containing train.csv and test.csv.")
    parser.add_argument("--output", type=Path, default=Path("submission_manual_tag_blend.csv"), help="Submission CSV path.")
    parser.add_argument("--cv", action="store_true", help="Run 5-fold out-of-fold validation before training.")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds for --cv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df = pd.read_csv(args.data_dir / "train.csv")
    test_df = pd.read_csv(args.data_dir / "test.csv")

    feature_train, feature_test = build_feature_frames(train_df, test_df)
    feature_cols = [col for col in feature_train.columns if col not in [TARGET, ID_COL]]
    x_train = feature_train[feature_cols]
    y_train = np.log1p(feature_train[TARGET])
    x_test = feature_test[feature_cols]

    num_cols = x_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = x_train.select_dtypes(include=["object", "str", "category"]).columns.tolist()
    models = build_models(num_cols, cat_cols)

    print(f"Training rows: {len(x_train)}")
    print(f"Test rows: {len(x_test)}")
    print(f"Features before one-hot encoding: {len(feature_cols)}")

    if args.cv:
        cv_results = run_cross_validation(models, x_train, y_train, args.folds)
        print("\nCross-validation RMSE on log1p(SalePrice):")
        print(cv_results.to_string(index=False))

    predictions = train_and_predict(models, x_train, y_train, x_test)
    submission = pd.DataFrame({ID_COL: feature_test[ID_COL].astype(int), TARGET: predictions})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"\nSaved submission: {args.output}")


if __name__ == "__main__":
    main()
