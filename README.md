# CodeAlpha Sales Prediction

An enterprise-grade machine learning pipeline developed for the CodeAlpha Data Science Internship. This project predicts future sales based on advertising budgets across TV, Radio, and Newspaper channels.

## Features
* Automated Data Pipeline: Prevents data leakage using Scikit-Learn `Pipeline` and `SimpleImputer`.
* Advanced Hyperparameter Tuning: Utilizes `GridSearchCV` to optimize Ridge Regression, Random Forest, and Gradient Boosting models.
* Universal Feature Importance: Employs permutation importance to evaluate cross-model channel contributions.
* Actionable Business Insights: Simulates 10% budget increase scenarios to recommend data-driven marketing strategies.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the automated pipeline: `python sales_prediction.py`
3. Predict custom campaign budgets instantly via the CLI:
   `python sales_prediction.py --tv 150 --radio 40 --newspaper 10`

All generated visualizations and CSV reports are automatically saved to the `outputs/` directory.