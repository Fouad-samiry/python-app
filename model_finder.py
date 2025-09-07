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
    def reg(self, name: str, params: dict, m: dict) -> None:
        print(f"\n--- {name} ---")
        print("Best params:", params)
        print(
            f"MAE: {m['MAE']:.4f} | RMSE: {m['RMSE']:.4f} | "
            f"R2: {m['R2']:.4f}"
        )
    
    def cls(
        self,
        name: str,
        params: dict,
        m: dict,
        out_dir: str = "reports",
    ) -> None:
        print(f"\n--- {name} ---")
        print("Best params:", params)
        print(f"Accuracy: {m['Accuracy']:.4f}")
        print(m["classification_report"])

        cm = m["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title(f"{name} – Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(
            out_dir, f"confmat_{name.replace(' ', '_').lower()}.png"
        )
        plt.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        print(f"Confusion matrix saved to: {out}")
        
        
        
        
class Persister:
    def save(
        self,
        pipeline,
        all_results: dict,
        best_name: str,
        stem: str,
    ) -> str:
        os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
        model_path = f"{stem}.pkl"

        safe: dict = {}
        for k, v in all_results.items():
            met = v["metrics"]
            m2: dict = {}
            for mk, mv in met.items():
                m2[mk] = mv.tolist() if hasattr(mv, "tolist") else mv
            safe[k] = {"best_params": v["best_params"], "metrics": m2}

        joblib.dump(pipeline, model_path)
        with open(f"{stem}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {"best_model": best_name, "results": safe},
                f,
                ensure_ascii=False,
                indent=2,
            )
        return os.path.abspath(model_path)
    
    
    
    
class App:
    def __init__(self):
        self.loader = DataLoader()
        self.validator = TargetValidator()
        self.checker = DataReadinessChecker()
        self.prep = FeaturePreprocessor()
        self.reporter = Reporter()
        self.persist = Persister()
        
        
        
        
    def _prompt_task(self) -> str:
        while True:
            s = input(
                "Select Task (Regression or Classification): "
            ).strip()
            s = s.capitalize()
            if s in ("Regression", "Classification"):
                return s
            print("Invalid task. Type 'Regression' or 'Classification'.")

    def _prompt_csv(self) -> pd.DataFrame:
        while True:
            path = input("Enter CSV path: ").strip()
            try:
                return self.loader.load_csv(path)
            except FileNotFoundError:
                print("File not found. Try again.")
            except ValueError as e:
                print(f"{e}. Try again.")

    def _prompt_target(self, cols: list[str]) -> str:
        lower = {c.lower(): c for c in cols}
        print("Enter target column name (or index):")
        for i, c in enumerate(cols):
            print(f"  [{i}] {c}")
        while True:
            s = input("> ").strip()
            if s.isdigit():
                i = int(s)
                if 0 <= i < len(cols):
                    return cols[i]
                print(f"Index out of range 0..{len(cols)-1}. Try again.")
                continue
            if s.lower() in lower:
                return lower[s.lower()]
            sug = get_close_matches(s, cols, n=1, cutoff=0.6)
            if sug:
                a = input(f"Did you mean '{sug[0]}'? (y/n): ").strip().lower()
                if a == "y":
                    return sug[0]
            print("Column not found. Try again.")

    def _yes_no(self, msg: str) -> bool:
        while True:
            a = input(msg).strip().lower()
            if a in ("yes", "y"):
                return True
            if a in ("no", "n"):
                return False
            print("Please answer yes or no.")
            
            
            
    def run(self):
        # 1) Uppgiftstyp (robust)
        task = self._prompt_task()

        # 2) CSV (robust)
        df = self._prompt_csv()
        cols = self.loader.list_columns(df)

        # 3) Target (robust)
        target = self._prompt_target(cols)
        df = df.dropna(subset=[target]).reset_index(drop=True)
        y = df[target]
        X = df.drop(columns=[target])

        # 3a) Typvalidering
        ttype = self.validator.validate_match(task, y)
        print(f"Target '{target}' detected as {ttype}. ✓ matches task.")

        # 4a/4b) Ready-check
        ready, report = self.checker.assess(X)
        if not ready:
            self.checker.print_report(report)
            do_impute = (
                "missing" in report and
                self._yes_no("Auto-impute missing? (yes/no): ")
            )
            do_dummies = (
                "categorical" in report and
                self._yes_no("Create dummies? (yes/no): ")
            )
            if not (do_impute or do_dummies):
                print("Exit. Please fix data and rerun.")
                raise SystemExit(1)
            print("Auto-fix enabled.")
        else:
            do_impute = do_dummies = False
            print("Data is ready.")

        # Preprocess + split
        ct = self.prep.build(X, do_impute, do_dummies)
        stratify = y if task == "Classification" else None
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=stratify
        )

        results: dict = {}
        best_name, best_score, best_model = None, float("-inf"), None

        if task == "Regression":
            trainer = RegressorTrainer(ct)
            for name, (pipe, grid) in trainer.models().items():
                print(f"\nTraining {name} (cv=10, scoring=R2)...")
                model, best_params = trainer.fit_grid(pipe, grid, Xtr, ytr)
                metrics = trainer.eval(model, Xte, yte)
                self.reporter.reg(name, best_params, metrics)
                results[name] = {
                    "best_params": best_params,
                    "metrics": metrics,
                }
                if metrics["R2"] > best_score:
                    best_name = name
                    best_score = metrics["R2"]
                    best_model = model
            print(f"\nBest model: {best_name} (R2={best_score:.4f})")

        else:
            trainer = ClassifierTrainer(ct)
            for name, (pipe, grid) in trainer.models().items():
                print(f"\nTraining {name} (cv=10, scoring=Accuracy)...")
                model, best_params = trainer.fit_grid(pipe, grid, Xtr, ytr)
                metrics = trainer.eval(model, Xte, yte)
                self.reporter.cls(name, best_params, metrics)
                save_metrics = dict(metrics)
                if hasattr(save_metrics["confusion_matrix"], "tolist"):
                    save_metrics["confusion_matrix"] = (
                        save_metrics["confusion_matrix"].tolist()
                    )
                results[name] = {
                    "best_params": best_params,
                    "metrics": save_metrics,
                }
                if metrics["Accuracy"] > best_score:
                    best_name = name
                    best_score = metrics["Accuracy"]
                    best_model = model
            print(
                f"\nBest model: {best_name} "
                f"(Accuracy={best_score:.4f})"
            )

        # 4e) Bekräfta och spara
        agree = self._yes_no(
            f"\nDo you agree '{best_name}' is the best? (yes/no): "
        )
        if agree:
            stem = input(
                "Enter file name (without extension): "
            ).strip() or f"best_{best_name.replace(' ', '_').lower()}"
            path_saved = self.persist.save(
                best_model, results, best_name, stem
            )
            print(f"Saved model to: {path_saved}")
            print(f"Saved metrics to: {stem}_metrics.json")
        else:
            print("No model was saved.")
            
            
            

if __name__ == "__main__":
    try:
        App().run()
    except (KeyboardInterrupt, EOFError):
        print("\nAborted.")
        sys.exit(1)