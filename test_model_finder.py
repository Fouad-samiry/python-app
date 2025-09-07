# tests/test_project_plan.py
# ---------------------------------------------------------------------
# Project plan (test checklist)
# 1- Create constructor that take path to our data.
# 2- Method to validate that all data is numeric.
# 3- Method to validate No Missing Data.
# 4- Method to choose independent and nonindependent columns.
# 5- Method to split data into X_train, X_test, y_train, y_test.
# 6- Method to train model.
# 7- Method to predict.
# 8- Method to calculate Metrics values: RMSE, MAE (and R2/Accuracy).
# ---------------------------------------------------------------------






# tests/test_model_finder.py
import os
import io
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline as SkPipe
from model_finder import (
    DataLoader, TargetValidator, DataReadinessChecker, FeaturePreprocessor,
    RegressorTrainer, ClassifierTrainer, Reporter, Persister, App)




def make_reg_df(n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    y = 2.0 * X["x1"] - 0.5 * X["x2"] + rng.normal(scale=0.1, size=n)
    return X, y

def make_cls_df(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X0 = rng.normal(loc=-1.0, scale=0.6, size=(n//2, 2))
    X1 = rng.normal(loc=+1.0, scale=0.6, size=(n - n//2, 2))
    X = pd.DataFrame(np.vstack([X0, X1]), columns=["f1", "f2"])
    y = pd.Series([0]*(n//2) + [1]*(n - n//2), name="target")
    return X, y


class TestDataLoader(unittest.TestCase):
    def test_load_csv_ok_and_list_columns(self):
        dl = DataLoader()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "toy.csv")
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(path, index=False)
            df = dl.load_csv(path)
            self.assertEqual(list(df.columns), ["a", "b"])
            cols = dl.list_columns(df)
            self.assertEqual(cols, ["a", "b"])
            
    def test_load_csv_missing_raises(self):
        dl = DataLoader()
        with self.assertRaises(FileNotFoundError):
            dl.load_csv("no_such.csv")




class TestTargetValidator(unittest.TestCase):
    def test_detect_type_and_validate(self):
        tv = TargetValidator()
        # regression target
        X, y = make_reg_df(n=30)
        t = tv.detect_type(y)
        self.assertEqual(t, "Continuous")
        self.assertEqual(tv.validate_match("Regression", y), "Continuous")
        # classification target
        Xc, yc = make_cls_df(n=30)
        t2 = tv.detect_type(yc)
        self.assertEqual(t2, "Categorical")
        self.assertEqual(tv.validate_match("Classification", yc), "Categorical")
        # mismatch
        with self.assertRaises(ValueError):
            tv.validate_match("Regression", yc)
        with self.assertRaises(ValueError):
            tv.validate_match("Classification", y)




class TestDataReadinessChecker(unittest.TestCase):
    def test_assess_ready(self):
        X, _ = make_reg_df(n=10)
        ok, report = DataReadinessChecker().assess(X)
        self.assertTrue(ok)
        self.assertEqual(report, {})
        
    def test_assess_missing_and_categorical(self):    
        X = pd.DataFrame({
            "num": [1.0, None, 3.0],
            "cat": ["a", "b", "a"],
        })
        ok, report = DataReadinessChecker().assess(X)
        self.assertFalse(ok)
        self.assertIn("missing", report)
        self.assertIn("categorical", report)
        
        
        
        
class TestFeaturePreprocessor(unittest.TestCase):
    def test_build_with_impute_and_dummies(self):
        X = pd.DataFrame({
            "n1": [1.0, None, 3.0],
            "c1": ["x", "y", "x"],
        })
        ct = FeaturePreprocessor().build(X, auto_impute=True, auto_dummies=True)
        self.assertIsNotNone(ct)
        Z = ct.fit_transform(X)
        self.assertEqual(Z.shape[0], 3)
        
    def test_build_no_cats_no_impute(self):
        X = pd.DataFrame({"n1": [1.0, 2.0, 3.0]})
        ct = FeaturePreprocessor().build(X, auto_impute=False, auto_dummies=False)
        Z = ct.fit_transform(X)
        self.assertEqual(Z.shape[0], 3)




class TestRegressorTrainer(unittest.TestCase):
    def test_train_linear_regression_no_grid(self):
        X, y = make_reg_df(n=40)
        ct = FeaturePreprocessor().build(X, auto_impute=False, auto_dummies=False)
        trainer = RegressorTrainer(ct)
        pipe = SkPipe([("prep", ct), ("model", LinearRegression())])
        model, params = trainer.fit_grid(pipe, {}, X, y)
        self.assertIn("note", params)
        metrics = trainer.eval(model, X.iloc[:10], y.iloc[:10])
        for k in ["MAE", "RMSE", "R2"]:
            self.assertIn(k, metrics)