import os
import json
import unittest
import tempfile
import numpy as np
import pandas as pd
from unittest.mock import patch
from model_finder import (
    DataLoader, TargetValidator, DataReadinessChecker,
    FeaturePreprocessor, RegressorTrainer, ClassifierTrainer,
    Reporter, Persister, App
)
from sklearn.datasets import make_regression, make_classification



class TestDataLoader(unittest.TestCase):
    pass










class TestTargetValidator(unittest.TestCase):
    pass