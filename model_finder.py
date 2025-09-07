"""

- OOP-struktur 
- Terminal input 
- CSV-validering 
- Kolumner + target + typvalidering 
- Ready-check + rapport + valbar auto-fix
- Regressorer + ANN (krav 4c) med GridSearchCV(cv=10), MAE, RMSE, R2
- Klassificerare + ANN (krav 4d) med GridSearchCV(cv=10), CM-plot, report
- Bekräftelse + dump av modell + metrics (krav 4e)

"""


from __future__ import annotations
import os
import json
import sys
from difflib import get_close_matches
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    Ridge,
    ElasticNet,
    LogisticRegression,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt





class DataLoader:
    def load_csv(self, path: str) -> pd.DataFrame:
        if not path or not isinstance(path, str):
            raise ValueError("CSV path must be a non-empty string")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        if df.empty or df.shape[1] == 0:
            raise ValueError("CSV is empty or has no columns")
        return df
    
    def list_columns(self, df: pd.DataFrame) -> list[str]:
        cols = list(df.columns)
        print("Columns:", cols)
        return cols
    
    
    
    
class TargetValidator:
    def detect_type(self, y: pd.Series) -> str:
        few_uniques = y.nunique(dropna=True) <= 10
        return "Categorical" if (y.dtype == "O" or few_uniques) else "Continuous"
    
    def validate_match(self, task: str, y: pd.Series) -> str:
        if task not in ("Regression", "Classification"):
            raise ValueError("Task must be Regression or Classification")
        ttype = self.detect_type(y)
        if task == "Regression" and ttype != "Continuous":
            raise ValueError("Target must be continuous for Regression")
        if task == "Classification" and ttype != "Categorical":
            raise ValueError("Target must be categorical for Classification")
        return ttype
    
    
    
    
class DataReadinessChecker:
    def assess(self, X: pd.DataFrame) -> tuple[bool, dict]:
        report: dict = {}
        miss = X.isna().sum()
        miss_cols = miss[miss > 0]
        if not miss_cols.empty:
            report["missing"] = miss_cols.to_dict()
        cats = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if cats:
            report["categorical"] = cats
        return (len(report) == 0), report
    
    def print_report(self, report: dict) -> None:
        print("\nData is NOT ready:")
        if "missing" in report:
            print(" - Missing:", report["missing"])
        if "categorical" in report:
            print(" - Categorical columns:", report["categorical"])
        print("Hints: fill missing (median/frequent) and one-hot categoricals.\n")
        
        
        
        
class FeaturePreprocessor:
    def build(
        self,
        X: pd.DataFrame,
        auto_impute: bool,
        auto_dummies: bool,
    ) -> ColumnTransformer:
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        num_pipe = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                    if auto_impute else "passthrough",
                ),
                ("scaler", StandardScaler()),
            ]
        )
        if auto_dummies and cat_cols:
            cat_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore")),
                ]
            )
        else:
            cat_pipe = "drop"

        return ColumnTransformer(
            transformers=[
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, cat_cols),
            ],
            remainder="drop",
        )
        
        
        
        
class TrainerBase:
    scoring = "r2"
    cv = 10
    
    def fit_grid(self, pipe: Pipeline, grid: dict, X, y):
        if grid:
            search = GridSearchCV(
                pipe,
                grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=-1,
            )
            search.fit(X, y)
            return search.best_estimator_, search.best_params_
        pipe.fit(X, y)
        return pipe, {"note": "no params"}
    
    
    
    
class RegressorTrainer(TrainerBase):
    scoring = "r2"
    
    def __init__(self, preprocess: ColumnTransformer):
        self.preprocess = preprocess
        
    def models(self) -> dict:
        return {
            "Linear Regression": (
                Pipeline(
                    [("prep", self.preprocess), ("model", LinearRegression())]
                ),
                {},
            ),
            "Lasso": (
                Pipeline(
                    [
                        ("prep", self.preprocess),
                        ("model", Lasso(max_iter=10000, random_state=42)),
                    ]
                ),
                {"model__alpha": [0.01, 0.1, 1.0]},
            ),
            "Ridge": (
                Pipeline(
                    [
                        ("prep", self.preprocess),
                        ("model", Ridge(random_state=42)),
                    ]
                ),
                {"model__alpha": [0.01, 0.1, 1.0]},
            ),
            "Elastic Net": (
                Pipeline(
                    [
                        ("prep", self.preprocess),
                        ("model", ElasticNet(max_iter=10000, random_state=42)),
                    ]
                ),
                {
                    "model__alpha": [0.01, 0.1, 1.0],
                    "model__l1_ratio": [0.2, 0.5, 0.8],
                },
            ),
            "SVR": (
                Pipeline([("prep", self.preprocess), ("model", SVR())]),
                {
                    "model__C": [0.1, 1, 10],
                    "model__kernel": ["rbf", "linear"],
                    "model__gamma": ["scale"],
                },
            ),
            "ANN": (
                Pipeline(
                    [
                        ("prep", self.preprocess),
                        ("model", MLPRegressor(max_iter=600, random_state=42)),
                    ]
                ),
                {},  # Optional grid per krav
            ),
        }
        
    def eval(self, model, X_te, y_te) -> dict:
        pred = model.predict(X_te)
        return {
            "MAE": float(mean_absolute_error(y_te, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
            "R2": float(r2_score(y_te, pred)),
        }







class ClassifierTrainer(TrainerBase):
    scoring = "accuracy"
    
    def __init__(self, preprocess: ColumnTransformer):
        self.preprocess = preprocess
        
    def models(self) -> dict:
        return {
            "Logistic Regression": (
                Pipeline(
                    [
                        ("prep", self.preprocess),
                        (
                            "model",
                            LogisticRegression(max_iter=5000, random_state=42),
                        ),
                    ]
                ),
                {
                    "model__C": [0.1, 1, 10],
                    "model__penalty": ["l2"],
                    "model__solver": ["lbfgs"],
                },
            ),
            "KNN": (
                Pipeline(
                    [("prep", self.preprocess), ("model", KNeighborsClassifier())]
                ),
                {
                    "model__n_neighbors": [3, 5, 7],
                    "model__weights": ["uniform", "distance"],
                },
            ),
            "SVC": (
                Pipeline(
                    [("prep", self.preprocess), ("model", SVC(random_state=42))]
                ),
                {
                    "model__C": [0.1, 1, 10],
                    "model__kernel": ["rbf", "linear"],
                    "model__gamma": ["scale"],
                },
            ),
            "ANN": (
                Pipeline(
                    [
                        ("prep", self.preprocess),
                        ("model", MLPClassifier(max_iter=600, random_state=42)),
                    ]
                ),
                {},  # Optional grid
            ),
        }

    def eval(self, model, X_te, y_te) -> dict:
        pred = model.predict(X_te)
        return {
            "Accuracy": float(accuracy_score(y_te, pred)),
            "classification_report": classification_report(y_te, pred),
            "confusion_matrix": confusion_matrix(y_te, pred),
        }






class Reporter:
    pass






class Persister:
    pass










class App:
    pass




if __name__ == "__main__":
        App().run()
        pass