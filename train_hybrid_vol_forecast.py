"""
train_hybrid_vol_forecast.py

Complete pipeline:
- download S&P500 daily data via yfinance
- compute returns, realized-vol (target), GARCH(3,3) conditional volatility
- create sequence dataset for LSTM (sequence length configurable)
- train 3 models:
    1) GARCH-only baseline (we evaluate one-step-ahead conditional vol as forecast)
    2) LSTM-only (uses returns sequences)
    3) Hybrid LSTM (uses returns sequences + GARCH3 vol as an extra feature)
- evaluate out-of-sample RMSE and save results + models + dataset
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yfinance as yf
from arch import arch_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

# ================ User settings ====================
TICKER = "^GSPC"            # S&P 500 index ticker (Yahoo)
START = "2005-01-01"
END = datetime.today().strftime("%Y-%m-%d")
SEQ_LEN = 20                # days of history fed to LSTM
TEST_RATIO = 0.2
BATCH_SIZE = 32
EPOCHS = 60
OUTPUT_DIR = "output"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
# ====================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)

# ----------------- Step 1: Download data -----------------
print("Downloading data...")
df = yf.download(TICKER, start=START, end=END, progress=False)[["Adj Close", "Close"]]
df = df.rename(columns={"Adj Close": "adj_close", "Close": "close"})
df = df.dropna()
df['return'] = np.log(df['adj_close']).diff()  # log returns
df = df.dropna()
print(f"Downloaded {len(df)} rows from {df.index.min().date()} to {df.index.max().date()}")

# ----------------- Step 2: Realized Volatility target -----------------
# We define target = next-day realized volatility estimated as rolling daily std over last N days (annualized)
REALIZED_WINDOW = 10  # days used to compute realized vol proxy
# compute rolling std of returns (sample std)
df['rv_raw'] = df['return'].rolling(window=REALIZED_WINDOW).std()
# annualize (sqrt(252))
df['realized_vol'] = df['rv_raw'] * np.sqrt(252)
# Shift target to be NEXT day realized volatility (i.e., forecast next-day realized_vol)
df['target_rv'] = df['realized_vol'].shift(-1)
df = df.dropna(subset=['target_rv'])

# ----------------- Step 3: Fit GARCH(3,3) to returns and extract conditional vol (as feature) -----------------
# Fit on the training-sized portion later; but we will fit on full-sample for simplicity to create feature — 
# for strict backtesting you'd fit recursively. This script uses full-sample fit to create a global GARCH series.
# If you want proper walk-forward, we'd loop and fit per rolling window (costly). This is an accessible baseline.
print("Fitting GARCH(3,3) to returns (this may take a little while)...")
# arch expects return series in percent (multiplying by 100 stabilizes)
am = arch_model(df['return'] * 100, vol='GARCH', p=3, q=3, mean='Constant', dist='normal')
res = am.fit(disp='off')
df['garch_cond_vol'] = res.conditional_volatility / 100.0  # back to same units as returns' std (approx)
# simple smoothing to avoid too spiky inputs
df['garch_cond_vol'] = df['garch_cond_vol'].rolling(3, min_periods=1).mean()

# ----------------- Step 4: Create features and sequence dataset for LSTM -----------------
# We'll use:
# - sequence of returns (seq_len)
# - per-day static feature: garch_cond_vol aligned to the last day of the sequence (we'll feed it as an extra feature repeated or as a second input)
# - target: next-day realized vol (target_rv)

df = df.dropna(subset=['garch_cond_vol', 'target_rv'])

# Build arrays
features = ['return']  # sequence features
extra_feat = ['garch_cond_vol']  # per-time features we can also include per timestep

# normalization: will scale using training set
# First create sequences
def create_sequences(df_in, seq_len, seq_features, extra_features):
    X_seq = []
    X_extra = []
    y = []
    idx = []
    arr = df_in.reset_index(drop=True)
    for i in range(len(arr) - seq_len):
        seq_block = arr.loc[i:i+seq_len-1, seq_features].values  # shape (seq_len, n_features)
        # extra features: take last day of seq as representative; could also be a sequence.
        extra_block = arr.loc[i+seq_len-1, extra_features].values  # shape (n_extra,)
        target = arr.loc[i+seq_len, 'target_rv']  # next-day target
        X_seq.append(seq_block)
        X_extra.append(extra_block)
        y.append(target)
        idx.append(i+seq_len)  # index of day whose target is y
    return np.array(X_seq), np.array(X_extra), np.array(y), np.array(idx)

X_seq, X_extra, y, idxs = create_sequences(df, SEQ_LEN, features, extra_feat)
print("Sequence dataset shapes:", X_seq.shape, X_extra.shape, y.shape)

# Train/test split chronological
split = int((1 - TEST_RATIO) * len(X_seq))
X_seq_train, X_seq_test = X_seq[:split], X_seq[split:]
X_extra_train, X_extra_test = X_extra[:split], X_extra[split:]
y_train, y_test = y[:split], y[split:]
idx_train, idx_test = idxs[:split], idxs[split:]

# Scale features: Fit scalers only on training data
seq_scaler = StandardScaler()
# flatten sequences to fit scaler on sequence features
n_seq_features = X_seq_train.shape[2]
X_seq_train_flat = X_seq_train.reshape(-1, n_seq_features)
X_seq_test_flat = X_seq_test.reshape(-1, n_seq_features)
seq_scaler.fit(X_seq_train_flat)
X_seq_train_scaled = seq_scaler.transform(X_seq_train_flat).reshape(X_seq_train.shape)
X_seq_test_scaled = seq_scaler.transform(X_seq_test_flat).reshape(X_seq_test.shape)

extra_scaler = StandardScaler()
extra_scaler.fit(X_extra_train)
X_extra_train_scaled = extra_scaler.transform(X_extra_train)
X_extra_test_scaled = extra_scaler.transform(X_extra_test)

target_scaler = StandardScaler()
y_train_reshaped = y_train.reshape(-1,1)
y_test_reshaped = y_test.reshape(-1,1)
target_scaler.fit(y_train_reshaped)
y_train_scaled = target_scaler.transform(y_train_reshaped).flatten()
y_test_scaled = target_scaler.transform(y_test_reshaped).flatten()

# Save scalers
joblib.dump(seq_scaler, os.path.join(OUTPUT_DIR, "models", "seq_scaler.joblib"))
joblib.dump(extra_scaler, os.path.join(OUTPUT_DIR, "models", "extra_scaler.joblib"))
joblib.dump(target_scaler, os.path.join(OUTPUT_DIR, "models", "target_scaler.joblib"))

# save dataset to CSV (for GitHub)
df_out = df.reset_index().loc[idxs, ['index', 'adj_close', 'return', 'garch_cond_vol', 'realized_vol', 'target_rv']]
df_out = df_out.rename(columns={'index': 'date'})
df_out.to_csv(os.path.join(OUTPUT_DIR, "sp500_data.csv"), index=False)
print("Saved dataset to", os.path.join(OUTPUT_DIR, "sp500_data.csv"))

# ----------------- Step 5: Build models -----------------
tf.keras.backend.clear_session()

def build_lstm_baseline(seq_len, n_features):
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(48, return_sequences=True),
        Dropout(0.15),
        LSTM(32, return_sequences=False),
        Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_hybrid(seq_len, n_seq_features, n_extra):
    # Sequence input
    seq_in = Input(shape=(seq_len, n_seq_features), name='seq_in')
    x = LSTM(64, return_sequences=True)(seq_in)
    x = Dropout(0.2)(x)
    x = LSTM(48, return_sequences=True)(x)
    x = Dropout(0.15)(x)
    x = LSTM(32, return_sequences=False)(x)
    x = Dropout(0.1)(x)

    # Extra input
    extra_in = Input(shape=(n_extra,), name='extra_in')
    y_extra = Dense(16, activation='relu')(extra_in)
    y_extra = BatchNormalization()(y_extra)

    # combine
    combined = Concatenate()([x, y_extra])
    z = Dense(32, activation='relu')(combined)
    z = Dense(16, activation='relu')(z)
    out = Dense(1, activation='linear')(z)

    model = Model(inputs=[seq_in, extra_in], outputs=out)
    model.compile(optimizer='adam', loss='mse')
    return model

# instantiate
n_seq_features = X_seq_train.shape[2]
n_extra = X_extra_train.shape[1]
lstm_baseline = build_lstm_baseline(SEQ_LEN, n_seq_features)
hybrid_model = build_hybrid(SEQ_LEN, n_seq_features, n_extra)

print("Models built.")

# ----------------- Step 6: Train models -----------------
es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
# Baseline LSTM (uses sequence only)
print("Training LSTM baseline (sequence-only)...")
history_base = lstm_baseline.fit(
    X_seq_train_scaled, y_train_scaled,
    validation_split=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=[es], verbose=1
)
lstm_baseline.save(os.path.join(OUTPUT_DIR, "models", "lstm_baseline.h5"))

# Hybrid LSTM (sequence + garch cond vol)
print("Training hybrid LSTM (sequence + GARCH feature)...")
history_hybrid = hybrid_model.fit(
    {"seq_in": X_seq_train_scaled, "extra_in": X_extra_train_scaled},
    y_train_scaled,
    validation_split=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=[es], verbose=1
)
hybrid_model.save(os.path.join(OUTPUT_DIR, "models", "lstm_hybrid.h5"))

# ----------------- Step 7: GARCH-only baseline forecast
# For GARCH baseline, the simplest 1-step forecast is the model's conditional_volatility shifted forward.
# res.conditional_volatility gives series aligned to returns -- our df['garch_cond_vol'] approximates current day's cond vol.
# So forecast for next day = garch_cond_vol at current day (naive). We'll compute RMSE of this naive mapping.
# -----------------
# We already have df['garch_cond_vol'] and targets; map to sample idxs we used earlier
garch_feature_series = df.reset_index(drop=True).loc[idxs, 'garch_cond_vol'].values
garch_forecast = garch_feature_series[split:]  # align with X_test
# -----------------

# ----------------- Step 8: Predictions and evaluation -----------------
# LSTM baseline predictions
y_pred_base_scaled = lstm_baseline.predict(X_seq_test_scaled).flatten()
y_pred_base = target_scaler.inverse_transform(y_pred_base_scaled.reshape(-1,1)).flatten()
# Hybrid predictions
y_pred_hybrid_scaled = hybrid_model.predict({"seq_in": X_seq_test_scaled, "extra_in": X_extra_test_scaled}).flatten()
y_pred_hybrid = target_scaler.inverse_transform(y_pred_hybrid_scaled.reshape(-1,1)).flatten()
# True y_test
y_true = y_test  # already in original units

# GARCH baseline
y_pred_garch = garch_forecast  # already in same units (we earlier scaled back)

# Compute RMSEs
rmse_base = sqrt(mean_squared_error(y_true, y_pred_base))
rmse_hybrid = sqrt(mean_squared_error(y_true, y_pred_hybrid))
rmse_garch = sqrt(mean_squared_error(y_true, y_pred_garch))

print("Out-of-sample RMSEs:")
print(f"GARCH-only baseline RMSE: {rmse_garch:.6f}")
print(f"LSTM-only baseline RMSE: {rmse_base:.6f}")
print(f"Hybrid LSTM RMSE: {rmse_hybrid:.6f}")

# Save metrics
with open(os.path.join(OUTPUT_DIR, "metrics_summary.txt"), "w") as f:
    f.write(f"RMSE_garch: {rmse_garch:.6f}\n")
    f.write(f"RMSE_lstm: {rmse_base:.6f}\n")
    f.write(f"RMSE_hybrid: {rmse_hybrid:.6f}\n")
print("Saved metrics_summary.txt")

# ----------------- Step 9: Diagnostics & plots -----------------
dates_test = df.reset_index().loc[idxs[split:], 'index']
plt.figure(figsize=(12,5))
plt.plot(dates_test, y_true, label='True next-day realized vol')
plt.plot(dates_test, y_pred_garch, label='GARCH forecast (naive)')
plt.plot(dates_test, y_pred_base, label='LSTM-only forecast')
plt.plot(dates_test, y_pred_hybrid, label='Hybrid LSTM forecast')
plt.legend()
plt.xticks(rotation=30)
plt.title('Realized Volatility Forecasts (test set)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figures", "forecast_comparison.png"))
plt.close()

# Plot scatter residuals for hybrid
resid = y_true - y_pred_hybrid
plt.figure(figsize=(6,4))
sns.histplot(resid, kde=True)
plt.title("Hybrid model residuals")
plt.savefig(os.path.join(OUTPUT_DIR, "figures", "hybrid_residuals.png"))
plt.close()

# Identify high-volatility events (top 5% realized vol days) and see detection rates
thresh = np.percentile(y_true, 95)
high_idx_true = np.where(y_true >= thresh)[0]
# We'll consider model 'flags' if forecasted vol is in top 20% predicted values
pred_thresh_hybrid = np.percentile(y_pred_hybrid, 80)
detected_hybrid = np.sum(y_pred_hybrid[high_idx_true] >= pred_thresh_hybrid)
detected_garch = np.sum(y_pred_garch[high_idx_true] >= pred_thresh_hybrid)
detected_lstm = np.sum(y_pred_base[high_idx_true] >= pred_thresh_hybrid)
detection_summary = {
    "n_high_events": len(high_idx_true),
    "hybrid_detected": int(detected_hybrid),
    "garch_detected": int(detected_garch),
    "lstm_detected": int(detected_lstm)
}
with open(os.path.join(OUTPUT_DIR, "detection_summary.txt"), "w") as f:
    f.write(str(detection_summary))
print("Saved detection_summary.txt")

# Save predictions table
pred_df = pd.DataFrame({
    "date": dates_test,
    "true_rv": y_true,
    "pred_garch": y_pred_garch,
    "pred_lstm": y_pred_base,
    "pred_hybrid": y_pred_hybrid
})
pred_df.to_csv(os.path.join(OUTPUT_DIR, "predictions_test.csv"), index=False)
print("Saved predictions_test.csv")

print("All done. Outputs are in the 'output' folder.")
