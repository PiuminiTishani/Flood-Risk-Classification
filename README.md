# Climate-Based Flood Risk Classification in Sri Lanka

## Project Overview

This project uses ensemble and hybrid machine learning
models to classify modeled flood risk across 25 districts
of Sri Lanka.

## Dataset

- Dataset period: 2015–2024
- Total observations: 91,325
- Problem type: Binary classification
- Target: Normal Risk or Elevated Risk
- Training/testing split: Approximately 80:20
- Original source: [Insert verified dataset source]

## Machine Learning Models

### Bagging
- Random Forest
- Extra Trees

### Boosting
- XGBoost
- LightGBM
- CatBoost

### Stacking
- Five base models with Logistic Regression

### Hybrid
- Random Forest + ANN using soft voting

## Results

CatBoost achieved the highest F1-score of 0.9836
and an accuracy of 99.72%.

The RF + ANN hybrid achieved the highest recall
of 98.80%.

## Dashboard

The interactive Streamlit dashboard provides:

- Dataset overview
- Climate analysis
- District analysis
- Ensemble model comparison
- Hybrid model analysis
- Model evaluation and feature importance
- Interactive flood risk classification

## Installation and Setup

### Prerequisites

- Python 3.11 or 3.13
- Windows PowerShell
- A terminal opened in the project folder

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run this command in the current terminal and try activation again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 3. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## Run the Dashboard

From the project directory, run:

```powershell
python -m streamlit run app.py
```

Streamlit will normally open the dashboard automatically in a browser. If it does not, open the local URL shown in the terminal, commonly:

```text
http://localhost:8501
```

## Alternative: Run Without Activating the Environment

If a virtual environment already exists, it can be used directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Troubleshooting

### Streamlit or Python cannot be found

Activate the virtual environment, then verify the installation:

```powershell
python --version
python -m pip show streamlit
```

### Package installation fails

Ensure that all dependencies are installed using the same Python environment:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### Model or data files are missing

The following project files are required when starting the dashboard:

- `cleaned_flood_data.csv`
- `model_results.csv`
- `model_predictions.csv`
- `feature_importance.csv`
- `trained_models_portable.pkl`
- `xgboost_model.json`
- `stacking_xgboost.json`

Keep all required files in the same directory as `app.py`.

## Limitations

The dataset contains modeled flood risk labels.
The dashboard is intended for academic purposes
and is not an official flood warning system.

## AI Usage Acknowledgment

Generative AI was used to assist with generating 
Python code for data preprocessing,machine learning 
model implementation, evaluation, visualization and 
Streamlit dashboard development.

AI assistance was also used for debugging and
troubleshooting. The generated code was executed
and tested, and the reported results were obtained
from experiments using the selected dataset.