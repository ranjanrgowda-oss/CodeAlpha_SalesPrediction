
# ============================================================
# CODEALPHA DATA SCIENCE INTERNSHIP
# TASK: SALES PREDICTION USING PYTHON
# ============================================================

# 1. IMPORT LIBRARIES

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.inspection import permutation_importance


# ============================================================
# 2. SETUP
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

FEATURES = ["TV", "Radio", "Newspaper"]
TARGET = "Sales"


# ============================================================
# 3. LOAD AND PREPARE DATASET
# ============================================================

def load_data():

    logging.info("Loading dataset...")

    file_path = BASE / "Advertising.csv"

    if not file_path.exists():
        raise FileNotFoundError("Advertising.csv not found.")

    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Remove unnecessary index columns
    df = df.loc[
        :,
        ~df.columns.str.match(r"^Unnamed:|^$", case=False)
    ]

    # Check required columns
    missing = [
        col for col in FEATURES + [TARGET]
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Select required columns
    df = df[FEATURES + [TARGET]].copy()

    # Convert values into numeric format
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove rows with missing target values
    df = df.dropna(subset=[TARGET])

    # Remove invalid negative values
    for col in FEATURES + [TARGET]:
        df.loc[df[col] < 0, col] = np.nan

    # Remove rows with invalid target values
    df = df.dropna(subset=[TARGET])

    logging.info(f"Dataset loaded successfully: {df.shape}")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nStatistical Summary:")
    print(df.describe())

    return df


# ============================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

def perform_eda(df):

    logging.info("Performing exploratory data analysis...")

    # Correlation Heatmap

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        df.corr(),
        annot=True,
        cmap="Blues",
        fmt=".2f"
    )

    plt.title("Advertising and Sales Correlation")
    plt.tight_layout()

    plt.savefig(OUT / "correlation_heatmap.png", dpi=150)
    plt.close()


    # Advertising Channels vs Sales

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, col in zip(axes, FEATURES):

        sns.regplot(
            data=df,
            x=col,
            y=TARGET,
            ax=ax,
            scatter_kws={"alpha": 0.7}
        )

        ax.set_title(f"{col} Advertising vs Sales")

    plt.tight_layout()

    plt.savefig(OUT / "advertising_impact.png", dpi=150)
    plt.close()


# ============================================================
# 5. TRAIN AND TEST DATA SPLITTING
# ============================================================

def split_data(df):

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\nTraining Data:", X_train.shape)
    print("Testing Data:", X_test.shape)

    return X_train, X_test, y_train, y_test


# ============================================================
# 6. MODEL TRAINING AND HYPERPARAMETER TUNING
# ============================================================

def train_models(X_train, y_train):

    logging.info("Training models with 5-fold cross-validation...")

    models = {

        "Linear Regression": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", LinearRegression())
            ]),
            {}
        ),

        "Ridge Regression": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge())
            ]),
            {
                "model__alpha": [0.1, 1.0, 10.0]
            }
        ),

        "Random Forest": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(random_state=42))
            ]),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 5, 10]
            }
        ),

        "Gradient Boosting": (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingRegressor(random_state=42))
            ]),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1]
            }
        )
    }

    fitted_models = {}
    cv_results = []

    for name, (pipeline, param_grid) in models.items():

        logging.info(f"Training {name}...")

        search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )

        search.fit(X_train, y_train)

        fitted_models[name] = search.best_estimator_

        cv_results.append({
            "Model": name,
            "CV_RMSE": -search.best_score_,
            "Best Parameters": str(search.best_params_)
        })

    cv_df = pd.DataFrame(cv_results)

    # Select model using cross-validation, not test data
    cv_df = cv_df.sort_values("CV_RMSE")

    best_name = cv_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]

    print("\nCross-Validation Results:")
    print(cv_df.to_string(index=False))

    print("\nBest Model:", best_name)

    cv_df.to_csv(OUT / "cross_validation_results.csv", index=False)

    return fitted_models, best_model, best_name


# ============================================================
# 7. MODEL EVALUATION
# ============================================================

def evaluate_models(fitted_models, X_test, y_test):

    logging.info("Evaluating models on test data...")

    results = []

    for name, model in fitted_models.items():

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)

        mse = mean_squared_error(y_test, predictions)

        rmse = np.sqrt(mse)

        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values("RMSE")

    print("\nModel Performance Comparison:")
    print(results_df.to_string(index=False))

    results_df.to_csv(OUT / "model_comparison.csv", index=False)

    return results_df


# ============================================================
# 8. ACTUAL VS PREDICTED SALES
# ============================================================

def plot_predictions(best_model, X_test, y_test, best_name):

    predictions = best_model.predict(X_test)

    plt.figure(figsize=(7, 5))

    sns.scatterplot(
        x=y_test,
        y=predictions,
        alpha=0.7
    )

    limits = [
        min(y_test.min(), predictions.min()),
        max(y_test.max(), predictions.max())
    ]

    plt.plot(
        limits,
        limits,
        "r--",
        label="Perfect Prediction"
    )

    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")

    plt.title(f"Actual vs Predicted Sales - {best_name}")

    plt.legend()
    plt.tight_layout()

    plt.savefig(OUT / "actual_vs_predicted.png", dpi=150)
    plt.close()


# ============================================================
# 9. FEATURE IMPORTANCE ANALYSIS
# ============================================================

def feature_analysis(best_model, X_test, y_test):

    logging.info("Calculating feature importance...")

    importance = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="r2",
        n_repeats=15,
        random_state=42
    )

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": importance.importances_mean
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nPermutation Feature Importance:")
    print(importance_df.to_string(index=False))

    importance_df.to_csv(
        OUT / "feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(7, 4))

    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature"
    )

    plt.title("Advertising Feature Importance")

    plt.xlabel("Decrease in R² when shuffled")
    plt.ylabel("Advertising Channel")

    plt.tight_layout()

    plt.savefig(OUT / "feature_importance.png", dpi=150)
    plt.close()

    return importance_df


# ============================================================
# 10. ADVERTISING BUDGET SCENARIO ANALYSIS
# ============================================================

def budget_analysis(best_model, X_train):

    logging.info("Analyzing advertising budget changes...")

    baseline = X_train.median()

    base_input = pd.DataFrame(
        [baseline],
        columns=FEATURES
    )

    base_sales = best_model.predict(base_input)[0]

    scenarios = []

    for channel in FEATURES:

        changed = base_input.copy()

        changed.loc[0, channel] *= 1.10

        new_sales = best_model.predict(changed)[0]

        scenarios.append({

            "Channel": channel,

            "Original Budget": base_input.loc[0, channel],

            "Increased Budget": changed.loc[0, channel],

            "Original Predicted Sales": base_sales,

            "New Predicted Sales": new_sales,

            "Predicted Change": new_sales - base_sales
        })

    scenario_df = pd.DataFrame(scenarios)

    scenario_df = scenario_df.sort_values(
        by="Predicted Change",
        ascending=False
    )

    print("\n10% Advertising Budget Increase Analysis:")
    print(scenario_df.to_string(index=False))

    scenario_df.to_csv(
        OUT / "budget_scenarios.csv",
        index=False
    )

    # Plot budget scenarios

    plt.figure(figsize=(7, 4))

    sns.barplot(
        data=scenario_df,
        x="Channel",
        y="Predicted Change"
    )

    plt.title("Predicted Sales Change After 10% Budget Increase")
    plt.ylabel("Change in Predicted Sales")

    plt.tight_layout()

    plt.savefig(OUT / "budget_scenarios.png", dpi=150)
    plt.close()

    return scenario_df


# ============================================================
# 11. CUSTOM SALES PREDICTION
# ============================================================

def custom_prediction(best_model, args):

    values = [args.tv, args.radio, args.newspaper]

    # No custom prediction requested
    if all(value is None for value in values):
        return

    # Require all three values
    if any(value is None for value in values):
        raise ValueError(
            "Enter all three budgets: --tv, --radio and --newspaper."
        )

    if any(not np.isfinite(value) or value < 0 for value in values):
        raise ValueError("Advertising budgets must be non-negative numbers.")

    custom_input = pd.DataFrame(
        [values],
        columns=FEATURES
    )

    prediction = best_model.predict(custom_input)[0]

    print("\nCustom Advertising Budgets:")
    print(custom_input.to_string(index=False))

    print(f"\nPredicted Sales: {prediction:.2f}")


# ============================================================
# 12. BUSINESS INSIGHTS
# ============================================================

def business_insights(importance_df, scenario_df):

    print("\n" + "=" * 60)
    print("BUSINESS INSIGHTS AND MARKETING STRATEGY")
    print("=" * 60)

    best_channel = importance_df.iloc[0]["Feature"]

    least_channel = importance_df.iloc[-1]["Feature"]

    best_scenario = scenario_df.iloc[0]

    print(
        f"\n1. {best_channel} is the most important advertising "
        "channel according to permutation importance."
    )

    print(
        f"\n2. {least_channel} has the lowest permutation importance. "
        "Review its contribution before changing its budget."
    )

    print(
        f"\n3. Increasing {best_scenario['Channel']} advertising "
        "budget by 10% gives the largest predicted sales change "
        "among the tested scenarios."
    )

    print(
        "\n4. Use the model comparison results to select a model "
        "with good prediction performance."
    )

    print(
        "\n5. Use the actual vs predicted graph to examine "
        "prediction errors."
    )

    print(
        "\nNote: These are model-based estimates, not proof of "
        "causal impact or actual return on investment."
    )

    print(
        "\nThe dataset does not contain platform or customer-segment "
        "columns, so those comparisons cannot be performed."
    )


# ============================================================
# 13. MAIN FUNCTION
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Advertising Sales Prediction Project"
    )

    parser.add_argument("--tv", type=float)
    parser.add_argument("--radio", type=float)
    parser.add_argument("--newspaper", type=float)

    args = parser.parse_args()

    try:

        # Load dataset
        df = load_data()

        # Exploratory Data Analysis
        perform_eda(df)

        # Split dataset
        X_train, X_test, y_train, y_test = split_data(df)

        # Train and tune models
        fitted_models, best_model, best_name = train_models(
            X_train,
            y_train
        )

        # Evaluate models
        results_df = evaluate_models(
            fitted_models,
            X_test,
            y_test
        )

        # Actual vs Predicted graph
        plot_predictions(
            best_model,
            X_test,
            y_test,
            best_name
        )

        # Feature importance
        importance_df = feature_analysis(
            best_model,
            X_test,
            y_test
        )

        # Budget scenario analysis
        scenario_df = budget_analysis(
            best_model,
            X_train
        )

        # Custom prediction
        custom_prediction(best_model, args)

        # Business insights
        business_insights(
            importance_df,
            scenario_df
        )

        # Completion
        print("\n" + "=" * 60)
        print("PROJECT COMPLETED SUCCESSFULLY")
        print("=" * 60)

        print("\nAll results saved in:")
        print(OUT)

        print("\nGenerated Files:")

        for file in sorted(OUT.iterdir()):
            print("-", file.name)

    except (ValueError, FileNotFoundError) as error:

        logging.error(str(error))


if __name__ == "__main__":
    main()
