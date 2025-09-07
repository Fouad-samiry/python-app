# python-app
We need to create a python desktop app that makes our work to find the best regressor or best classifier


# Model Finder 

Command line tool that trains multiple regression and classification
models on a CSV file. It validates data, builds preprocessing, runs
GridSearchCV where relevant, reports metrics, and can save the best model.

## Requirements
- Python 3.10+
- Packages in `requirements.txt`:
  pandas, numpy, scikit-learn, matplotlib, joblib

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt


python model_finder.py

python -m unittest -v