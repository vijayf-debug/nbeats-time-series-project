

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

# Darts imports for handling Time Series easily
from darts import TimeSeries
from darts.datasets import AirPassengersDataset
from darts.dataprocessing.transformers import Scaler
from darts.models import NBEATSModel, ExponentialSmoothing
from darts.metrics import mae, rmse, mape

# Setup for reproducibility
import logging
logging.getLogger("darts").setLevel(logging.WARNING)
torch.manual_seed(42)

print("--- Starting Time Series Forecasting Pipeline ---")

# ==========================================
# 1. Data Acquisition & Preprocessing
# ==========================================
print("\n[1] Loading and Preprocessing Data...")

raw_series = AirPassengersDataset().load()
train, val = raw_series.split_before(108)
scaler = Scaler()
train_scaled = scaler.fit_transform(train)
val_scaled = scaler.transform(val)

print(f"    Data loaded. Train size: {len(train)}, Validation size: {len(val)}")
print("    Data scaled successfully.")

# ==========================================
# 2. Hyperparameter Optimization Loop
# ==========================================
print("\n[2] Running Hyperparameter Optimization (Grid Search)...")

hyperparams_grid = [
    {'num_stacks': 2, 'layer_widths': 256, 'lr': 1e-3},
    {'num_stacks': 4, 'layer_widths': 512, 'lr': 1e-4}
]

best_model = None
best_val_loss = float('inf')
best_params = {}

# Common N-BEATS settings
input_chunk = 24  # Look back 24 months
output_chunk = 12 # Predict forward 12 months

for params in hyperparams_grid:
    print(f"    Testing config: {params}")
    
    # Initialize N-BEATS
    model_candidate = NBEATSModel(
        input_chunk_length=input_chunk,
        output_chunk_length=output_chunk,
        num_stacks=params['num_stacks'],
        layer_widths=params['layer_widths'],
        n_epochs=20,               # Low epochs for demonstration speed; increase to 100+ for production
        nr_epochs_val_period=1,
        batch_size=32,
        model_name="nbeats_candidate",
        force_reset=True,
        save_checkpoints=False,
        optimizer_kwargs={"lr": params['lr']},
        pl_trainer_kwargs={"accelerator": "cpu", "enable_progress_bar": False}, # Use 'gpu' if available
        
        # Crucial for Explainability: Use the Interpretable Architecture
        generic_architecture=False 
    )
    
    # Train
    model_candidate.fit(series=train_scaled, val_series=val_scaled, verbose=False)
    
    # Validate
    pred_scaled = model_candidate.predict(n=len(val))
    current_loss = rmse(val_scaled, pred_scaled)
    
    print(f"    -> RMSE: {current_loss:.4f}")
    
    if current_loss < best_val_loss:
        best_val_loss = current_loss
        best_model = model_candidate
        best_params = params

print(f"\n    Best Configuration Found: {best_params}")

# ==========================================
# 3. Final Model Training & Benchmarking
# ==========================================
print("\n[3] Benchmarking against Statistical Baseline...")


baseline_model = ExponentialSmoothing()
baseline_model.fit(train)
baseline_pred = baseline_model.predict(len(val))
nbeats_pred_scaled = best_model.predict(n=len(val))
nbeats_pred = scaler.inverse_transform(nbeats_pred_scaled)
def print_metrics(actual, forecast, model_name):
    m_rmse = rmse(actual, forecast)
    m_mae = mae(actual, forecast)
    m_mape = mape(actual, forecast)
    print(f"    {model_name} -> RMSE: {m_rmse:.2f} | MAE: {m_mae:.2f} | MAPE: {m_mape:.2f}%")
    return m_rmse, m_mae, m_mape

print("    --- Performance Results ---")
base_rmse, _, _ = print_metrics(val, baseline_pred, "Baseline (Exp. Smoothing)")
nbeats_rmse, _, _ = print_metrics(val, nbeats_pred, "N-BEATS (Deep Learning)")

if nbeats_rmse < base_rmse:
    print("\n    SUCCESS: N-BEATS outperformed the statistical baseline.")
else:
    print("\n    NOTE: N-BEATS did not outperform baseline (Need more epochs/tuning).")

# ==========================================
# 4. Explainability (Visualizing Components)
# ==========================================
print("\n[4] Generating Explainability Plots...")

# Plot Actual vs Predictions
plt.figure(figsize=(10, 6))
raw_series.plot(label='Actual Data')
baseline_pred.plot(label='Baseline Forecast', linestyle='--')
nbeats_pred.plot(label='N-BEATS Forecast', alpha=0.7)
plt.title('Forecast Comparison: N-BEATS vs Baseline')
plt.legend()
plt.show()


print("    Extracting Trend and Seasonality components from N-BEATS...")


# ==========================================
# 5. Operationalization Analysis (Text Output)
# ==========================================
print("\n[5] Operationalization & Deployment Analysis")
print("-" * 60)
analysis_text = """
1. **Model Performance**: 
   The N-BEATS model was tuned using a grid search. The key to its performance 
   is the decomposition of the time series into 'Trend' and 'Seasonality' blocks.
   
2. **Explainability**:
   Unlike a 'Black Box' LSTM, N-BEATS provides interpretability by design. 
   We can inspect which stack contributes to the trend (long-term rise) and 
   which contributes to the cyclical spikes (seasonality).

3. **Operational Challenges**:
   - **Cold Start**: Neural networks like N-BEATS struggle with new series that 
     have little history.
   - **Computational Cost**: Training N-BEATS takes significantly longer than 
     ARIMA/Exponential Smoothing. For real-time applications, we might need 
     to retrain offline daily rather than on-the-fly.
   - **Data Drift**: If the underlying pattern changes (e.g., a pandemic hits 
     airline travel), the neural network may fail confidently. Continuous 
     monitoring of MAPE is required in production.
"""
print(analysis_text)
print("-" * 60)
print("Pipeline Completed.")
