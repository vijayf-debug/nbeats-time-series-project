import math
import os
import random
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------
# Utility functions & dataset
# ---------------------------

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)

def generate_synthetic_series(n_steps=2000, seed=42) -> pd.Series:
    """
    Generate a complex univariate time series:
    trend + multiple seasonalities + heteroskedastic noise + occasional spikes.
    """
    np.random.seed(seed)
    t = np.arange(n_steps)
    trend = 0.001 * (t ** 1.1)  # slow upward trend
    seasonal1 = 2.0 * np.sin(2 * np.pi * t / 24)      # daily-ish seasonality
    seasonal2 = 0.5 * np.sin(2 * np.pi * t / 168)     # weekly-ish seasonality
    hetero_noise = (1.0 + 0.01 * t) * (0.3 * np.random.randn(n_steps))
    spikes = np.zeros(n_steps)
    spike_positions = np.random.choice(np.arange(100, n_steps), size=8, replace=False)
    spikes[spike_positions] = np.random.choice([5.0, -4.0], size=len(spike_positions))
    series = 10.0 + trend + seasonal1 + seasonal2 + hetero_noise + spikes
    return pd.Series(series, name="value")


def load_series(path: str = None) -> pd.Series:
    """
    If path is provided, expects a CSV with a single column 'value' or first numeric column.
    Otherwise returns a synthetic series.
    """
    if path is None:
        print("Using synthetic dataset. Replace load_series to load a real CSV.")
        return generate_synthetic_series(n_steps=2200)
    else:
        df = pd.read_csv(path)
        # Try common column names
        for c in ['value', 'y', 'target', df.columns[0]]:
            if c in df.columns:
                series = pd.Series(df[c].astype(float).values, name="value")
                return series
        # fallback: first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric column found in provided CSV.")
        return pd.Series(df[numeric_cols[0]].astype(float).values, name="value")


class TimeSeriesDataset(Dataset):
    def __init__(self, series: np.ndarray, context_len: int, forecast_horizon: int, scaler: StandardScaler=None):
        self.context_len = context_len
        self.forecast_horizon = forecast_horizon
        self.series = series.astype(float)
        self.N = len(series)
        self.indices = []
        for i in range(self.N - context_len - forecast_horizon + 1):
            self.indices.append(i)
        self.scaler = scaler

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.series[i:i+self.context_len]
        y = self.series[i+self.context_len:i+self.context_len+self.forecast_horizon]
        if self.scaler is not None:
            x = self.scaler.transform(x.reshape(-1,1)).flatten()
            y = self.scaler.transform(y.reshape(-1,1)).flatten()
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ---------------------------
# N-BEATS Model (simplified)
# ---------------------------

class NBeatsBlock(nn.Module):
    """
    Simple dense block producing backcast and forecast.
    """
    def __init__(self, input_dim, theta_dim, hidden_dim=256, n_layers=4):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.fc_stack = nn.Sequential(*layers)
        self.theta = nn.Linear(hidden_dim, theta_dim)

    def forward(self, x):
        # x shape: (batch, input_dim)
        h = self.fc_stack(x)
        theta = self.theta(h)
        return theta


class NBeatsNet(nn.Module):
    def __init__(self, context_len:int, forecast_horizon:int, n_blocks:int=3, block_hidden_dim:int=256, block_layers:int=4):
        super().__init__()
        self.context_len = context_len
        self.forecast_horizon = forecast_horizon
        self.n_blocks = n_blocks

        # For each block we produce a backcast (reconstruction of input) and forecast (prediction)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            # theta dimension = context_len + forecast_horizon (we'll project to backcast & forecast via linear maps)
            theta_dim = context_len + forecast_horizon
            block = NBeatsBlock(input_dim=context_len, theta_dim=theta_dim, hidden_dim=block_hidden_dim, n_layers=block_layers)
            self.blocks.append(block)

        # final linear maps from theta to backcast/forecast
        self.backcast_proj = nn.Linear(theta_dim, context_len)
        self.forecast_proj = nn.Linear(theta_dim, forecast_horizon)

    def forward(self, x):
        # x: (batch, context_len)
        residual = x
        forecast = torch.zeros((x.size(0), self.forecast_horizon), device=x.device)
        for block in self.blocks:
            theta = block(residual)
            backcast = self.backcast_proj(theta)
            block_forecast = self.forecast_proj(theta)
            residual = residual - backcast  # remove explained part
            forecast = forecast + block_forecast
        return forecast


# ---------------------------
# Training / Evaluation utils
# ---------------------------

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                n_epochs=50, lr=1e-3, device='cpu', patience=8):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val = float('inf')
    best_state = None
    patience_ctr = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, n_epochs+1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                val_losses.append(criterion(preds, yb).item())
        val_loss = np.mean(val_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def evaluate_model(model: nn.Module, loader: DataLoader, scaler: StandardScaler=None, device='cpu'):
    model.to(device)
    model.eval()
    all_y = []
    all_preds = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            all_preds.append(preds.cpu().numpy())
            all_y.append(yb.cpu().numpy())
    y = np.vstack(all_y)
    preds = np.vstack(all_preds)

    if scaler is not None:
        # inverse transform
        y = scaler.inverse_transform(y.reshape(-1,1)).reshape(y.shape)
        preds = scaler.inverse_transform(preds.reshape(-1,1)).reshape(preds.shape)

    # Evaluate per horizon aggregated
    mae = mean_absolute_error(y.flatten(), preds.flatten())
    rmse = math.sqrt(mean_squared_error(y.flatten(), preds.flatten()))
    # MAPE guard zeros
    epsilon = 1e-8
    mape = np.mean(np.abs((y - preds) / (np.abs(y) + epsilon))) * 100.0

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}, y, preds


# ---------------------------
# Baselines
# ---------------------------

def naive_forecast(train_series: np.ndarray, forecast_horizon: int, last_window: np.ndarray):
    """
    Simple baseline: persist last value for all future steps
    """
    last_val = last_window[-1]
    return np.array([last_val] * forecast_horizon)

def seasonal_naive_forecast(train_series: np.ndarray, forecast_horizon: int, last_window: np.ndarray, season: int=24):
    """
    Seasonal naive: use value from last season positions
    """
    res = []
    N = len(train_series)
    for h in range(1, forecast_horizon+1):
        idx = N - season + ((h-1) % season)
        idx = max(0, min(N-1, idx))
        res.append(train_series[idx])
    return np.array(res)


# ---------------------------
# Explainability: Input attribution via gradients
# ---------------------------

def gradient_input_attribution(model: nn.Module, input_window: np.ndarray, target_step:int=0, device='cpu'):
    """
    Compute gradients of forecast[target_step] w.r.t. input_window values.
    Returns normalized absolute gradient importance vector of shape (context_len,)
    """
    model.to(device)
    model.eval()
    x = torch.tensor(input_window, dtype=torch.float32, requires_grad=True).unsqueeze(0).to(device)
    preds = model(x)  # shape (1, horizon)
    target = preds[0, target_step]
    # compute gradient of target wrt input
    model.zero_grad()
    if x.grad is not None:
        x.grad.data.zero_()
    target.backward(retain_graph=False)
    grad = x.grad.detach().cpu().numpy().flatten()
    importance = np.abs(grad)
    if importance.sum() > 0:
        importance = importance / importance.sum()
    return importance


# ---------------------------
# Main experiment pipeline
# ---------------------------

def main(data_path: str = None, device='cpu'):
    # Hyperparameters
    context_len = 168  # e.g., one week of hourly observations
    forecast_horizon = 24  # predict next 24 steps
    n_blocks = 3
    hidden_dim = 256
    block_layers = 3
    batch_size = 64
    n_epochs = 80
    lr = 1e-3

    # Load series
    series = load_series(data_path).values  # numpy array
    N = len(series)
    train_frac = 0.7
    val_frac = 0.15
    n_train = int(N * train_frac)
    n_val = int(N * (train_frac + val_frac)) - n_train

    train_series = series[:n_train]
    val_series = series[n_train:n_train+n_val+context_len+forecast_horizon]  # extended slices for windows
    test_series = series[n_train+n_val:]

    # Fit scaler on train
    scaler = StandardScaler()
    scaler.fit(train_series.reshape(-1,1))

    # Build datasets (we'll unify on full series indexing but create partitioned datasets)
    full_series = series  # for convenience

    # Create indices for train/val/test windows explicitly by slicing
    train_dataset = TimeSeriesDataset(series=full_series[:n_train + context_len + forecast_horizon],
                                      context_len=context_len, forecast_horizon=forecast_horizon, scaler=scaler)
    val_dataset = TimeSeriesDataset(series=full_series[n_train - context_len : n_train + n_val + context_len + forecast_horizon],
                                    context_len=context_len, forecast_horizon=forecast_horizon, scaler=scaler)
    test_dataset = TimeSeriesDataset(series=full_series[n_train+n_val - context_len:],  # keep enough history
                                     context_len=context_len, forecast_horizon=forecast_horizon, scaler=scaler)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = NBeatsNet(context_len=context_len, forecast_horizon=forecast_horizon,
                      n_blocks=n_blocks, block_hidden_dim=hidden_dim, block_layers=block_layers)

    # Train
    print("Training N-BEATS model...")
    model, history = train_model(model, train_loader, val_loader, n_epochs=n_epochs, lr=lr, device=device, patience=10)

    # Evaluate
    print("Evaluating on test set...")
    metrics, y_test, preds_test = evaluate_model(model, test_loader, scaler=scaler, device=device)
    print("N-BEATS test metrics:", metrics)

    # Baseline evaluation (naive on test windows)
    # We'll compute baseline forecasts for each test window in the test_dataset
    baseline_preds = []
    baseline_preds_season = []
    y_true = []
    full_for_baseline = full_series  # use series for seasonal index reference
    for i in range(len(test_dataset)):
        x_window, y_window = test_dataset[i]
        x_np = x_window.numpy()
        # naive baseline
        p_naive = naive_forecast(train_series=full_for_baseline, forecast_horizon=forecast_horizon, last_window=x_np)
        p_season = seasonal_naive_forecast(train_series=full_for_baseline, forecast_horizon=forecast_horizon, last_window=x_np, season=24)
        baseline_preds.append(p_naive)
        baseline_preds_season.append(p_season)
        y_true.append(y_window.numpy())
    baseline_preds = np.vstack(baseline_preds)
    baseline_preds_season = np.vstack(baseline_preds_season)
    y_true = np.vstack(y_true)

    # Inverse-transform true and baselines if scaler used
    y_true_inv = scaler.inverse_transform(y_true.reshape(-1,1)).reshape(y_true.shape)
    baseline_inv = scaler.inverse_transform(baseline_preds.reshape(-1,1)).reshape(baseline_preds.shape)
    season_inv = scaler.inverse_transform(baseline_preds_season.reshape(-1,1)).reshape(baseline_preds_season.shape)

    # Baseline metrics
    mae_naive = mean_absolute_error(y_true_inv.flatten(), baseline_inv.flatten())
    rmse_naive = math.sqrt(mean_squared_error(y_true_inv.flatten(), baseline_inv.flatten()))
    mape_naive = np.mean(np.abs((y_true_inv - baseline_inv) / (np.abs(y_true_inv) + 1e-8))) * 100.0
    print("Naive baseline test metrics:", {'MAE': mae_naive, 'RMSE': rmse_naive, 'MAPE': mape_naive})

    mae_season = mean_absolute_error(y_true_inv.flatten(), season_inv.flatten())
    rmse_season = math.sqrt(mean_squared_error(y_true_inv.flatten(), season_inv.flatten()))
    mape_season = np.mean(np.abs((y_true_inv - season_inv) / (np.abs(y_true_inv) + 1e-8))) * 100.0
    print("Seasonal-naive baseline test metrics:", {'MAE': mae_season, 'RMSE': rmse_season, 'MAPE': mape_season})

    # Plot some example forecasts vs truth
    n_examples = 3
    fig, axes = plt.subplots(n_examples, 1, figsize=(10, 3*n_examples))
    for i in range(n_examples):
        idx = i  # first few windows
        axes[i].plot(range(forecast_horizon), y_test[idx], label='truth')
        axes[i].plot(range(forecast_horizon), preds_test[idx], label='N-BEATS')
        axes[i].plot(range(forecast_horizon), baseline_preds[idx], label='Naive baseline', linestyle='--')
        axes[i].legend()
        axes[i].set_title(f'Example {i}: Forecast horizon')
    plt.tight_layout()
    plt.show()

    # Run gradient attribution on a single sample (last test window)
    sample_idx = -1
    x_window, y_window = test_dataset[sample_idx]
    x_np = x_window.numpy()
    importance = gradient_input_attribution(model, x_np, target_step=0, device=device)
    plt.figure(figsize=(10,3))
    plt.plot(np.arange(len(importance)), importance)
    plt.title("Gradient-based input attribution (normalized absolute grads) for the last test window")
    plt.xlabel("Lag index (0 oldest ... context_len-1 newest)")
    plt.show()

    # Return model & scaler for downstream use
    return {
        'model': model,
        'scaler': scaler,
        'history': history,
        'metrics': metrics,
        'baseline_naive': {'MAE': mae_naive, 'RMSE': rmse_naive, 'MAPE': mape_naive},
        'baseline_seasonal': {'MAE': mae_season, 'RMSE': rmse_season, 'MAPE': mape_season},
    }

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = main(data_path=None, device=device)
