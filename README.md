# Predictive Maintenance of Turbofan Engines (MLANN)

This project implements a predictive maintenance pipeline to estimate the **Remaining Useful Life (RUL)** of turbofan engines. It uses the simulated C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset developed by NASA.

## Directory Structure

```text
├── data/
│   ├── train_FD001.txt      # Engine run-to-failure training trajectories
│   ├── test_FD001.txt       # Engine testing trajectories (ends prior to failure)
│   ├── RUL_FD001.txt        # Ground truth RUL values for test engines
│   └── readme.txt           # Original description of CMAPSS datasets
├── predictive_maintenance_cmapss.ipynb  # Restructured, clean Jupyter Notebook
└── best_model_pipeline.pkl  # Serialized model and preprocessing pipeline
```

---

## Methodology

### 1. Data Cleaning & Feature Engineering
* **Low-Variance Drop**: Automatically detects and drops sensors/operational settings with near-constant values.
* **Target Capping**: Capping RUL values at 125 cycles to focus learning on the degradation phase.
* **Rolling Features**: Computes rolling mean and standard deviation over sliding windows of 10 and 20 cycles for informative sensors to capture degradation trends.
* **Initial Difference**: Computes initial cycle offset features (`init_diff`) for informative sensors to capture degradation from the first cycle baseline.

### 2. Hyperparameter Tuning
Evaluates **6 regression models** using `RandomizedSearchCV` and 3-fold `GroupKFold` cross-validation (grouped by engine unit number to prevent data leakage):
1. **Multi-layer Perceptron (MLP) Regressor**
2. **XGBoost Regressor**
3. **Gradient Boosting Regressor**
4. **Random Forest Regressor**
5. **Extra Trees Regressor**
6. **Ridge Regression**

### 3. Model Comparison
The models achieved the following metrics on the test set:

| Model | Train MAE | Test MAE | Test RMSE | Test R2 | Gap |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Best)** | **7.82** | **9.64** | **13.62** | **0.884** | **1.81** |
| Extra Trees | 8.86 | 9.66 | 12.81 | 0.890 | 0.79 |
| XGBoost | 8.32 | 10.20 | 12.66 | 0.874 | 1.88 |
| Gradient Boosting | 9.48 | 10.33 | 13.68 | 0.870 | 0.85 |
| MLP | 7.15 | 10.69 | 14.65 | 0.875 | 3.55 |
| Ridge | 12.96 | 13.51 | 16.77 | 0.830 | 0.55 |

---

## Inference & Usage

The best performing model (Random Forest) is saved as `best_model_pipeline.pkl` along with feature configurations and scaling parameters.

To run predictions on new test data, load the serialized pipeline:

```python
import joblib
import numpy as np

# Load the saved pipeline
pipeline = joblib.load('best_model_pipeline.pkl')
model = pipeline['model']
scaler = pipeline['scaler']
features = pipeline['feature_cols']

# Predict RUL
X = scaler.transform(test_dataframe[features])
predictions = model.predict(X)
print("Predicted RUL:", np.round(predictions, 1))
```
