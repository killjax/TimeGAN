import numpy as np
import pandas as pd
import yfinance as yf


def real_data_loading(data_name, start_date, end_date):
    assert data_name in ["^GSPC", "AAPL", "MSFT", "GOOG", "AMZN"]

    ori_data = yf.download(
        data_name, start=start_date, end=end_date, progress=False, auto_adjust=False
    )

    # Force pandas to work on a clean copy
    ori_data = ori_data.copy()

    # 1. Log Returns
    ori_data["Log_Return"] = np.log(ori_data["Close"] / ori_data["Close"].shift(1))

    # 2. Average True Range (ATR)
    tr1 = ori_data["High"] - ori_data["Low"]
    tr2 = np.abs(ori_data["High"] - ori_data["Close"].shift(1))
    tr3 = np.abs(ori_data["Low"] - ori_data["Close"].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    ori_data["ATR"] = true_range.ewm(span=14, adjust=False).mean()

    # 3. Bollinger Band Width (BBW)
    sma_20 = ori_data["Close"].rolling(20).mean()
    std_20 = ori_data["Close"].rolling(20).std()
    upper_band = sma_20 + (std_20 * 2)
    lower_band = sma_20 - (std_20 * 2)
    ori_data["BBW"] = (upper_band - lower_band) / sma_20

    # 4. & 5. MACD and MACD Signal
    ewm_12 = ori_data["Close"].ewm(span=12, adjust=False).mean()
    ewm_26 = ori_data["Close"].ewm(span=26, adjust=False).mean()
    ori_data["MACD"] = ewm_12 - ewm_26
    ori_data["MACD_Signal"] = ori_data["MACD"].ewm(span=9, adjust=False).mean()

    # 6. RSI (Relative Strength Index)
    delta = ori_data["Close"].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs = avg_gain / avg_loss

    # Handle division by zero if avg_loss is 0
    rs[avg_loss == 0] = np.inf
    ori_data["RSI"] = 100.0 - (100.0 / (1.0 + rs))

    # Drop all rows with NaN values created by .shift() or rolling windows
    ori_data = ori_data.dropna()

    return ori_data


def real_data_processing(ori_data, seq_len):
    # Convert to numpy
    ori_data = np.asarray(ori_data)
    # Flip the data to make chronological data
    ori_data = ori_data[::-1]

    # Preprocess the dataset
    temp_data = []
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i : i + seq_len]
        temp_data.append(_x)

    # Mix the datasets (to make it similar to i.i.d)
    idx = np.random.permutation(len(temp_data))
    data = []
    for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])

    return data


def calculate_window_metrics(window, adj_close_idx, log_return_idx):
    """
    Calculates Volatility and Max Drawdown for a single (60, 12) window.
    """
    # 1. Extract the required time-series
    # 'Adj Close' is at index 0
    prices = window[:, adj_close_idx]
    # 'Log_Return' is at index 6
    log_returns = window[:, log_return_idx]

    # 2. Calculate Volatility
    # "standard deviation of the 1-day log returns"
    volatility = np.std(log_returns)

    # 3. Calculate Maximum Drawdown (MDD)
    # "using its 'AdjClose' prices"
    running_max = np.maximum.accumulate(prices)
    drawdowns = (running_max - prices) / (running_max + 1e-9)
    max_drawdown = np.max(drawdowns)

    return volatility, max_drawdown


def label_data(ori_data, feature_names):
    """
    Applies the two-stage labeling process to the entire dataset.

    Args:
        ori_data (list): A list of (60, 12) numpy arrays.
        feature_names (list): The list of 12 feature names in order.

    Returns:
        tuple: (ori_data_s, metrics_df)
            ori_data_s (list): The list of one-hot encoded labels.
            metrics_df (pd.DataFrame): DataFrame with metrics and string labels.
    """

    # Define percentile thresholds
    CRISIS_PERCENTILE = 0.90  # Top 10% of drawdowns
    VOLATILE_PERCENTILE = 0.85  # Top 15% of volatility (from non-crisis)

    # Get the column indices from the feature names
    adj_close_idx = feature_names.index("Adj Close")
    log_return_idx = feature_names.index("Log_Return")

    # --- Step 1: Calculate Metrics for Every Window ---
    all_metrics = []
    for window in ori_data:
        vol, mdd = calculate_window_metrics(window, adj_close_idx, log_return_idx)
        all_metrics.append({"volatility": vol, "mdd": mdd})

    # Create the DataFrame as suggested
    metrics_df = pd.DataFrame(all_metrics)

    # --- Step 2: Apply a Two-Stage Labeling System ---

    # Stage 1: Identify 'Crisis' (Drawdown-Based)
    # Using a fixed 20% drawdown (Bear Market definition)
    crisis_threshold = 0.20

    # Initialize all labels as 'Normal' by default
    metrics_df["label"] = "Normal"

    # Apply 'Crisis' label
    crisis_mask = metrics_df["mdd"] > crisis_threshold
    metrics_df.loc[crisis_mask, "label"] = "Crisis"

    # Stage 2: Differentiate 'Volatile' vs. 'Normal' (Volatility-Based)
    non_crisis_mask = metrics_df["label"] != "Crisis"

    # Using a fixed volatility based on VIX > 30
    # VIX = 30 (0.30) is "high fear" (annualized).
    # We must de-annualize it for our daily log_return std dev.
    # Daily Threshold = Annual Threshold / sqrt(252 trading days)
    volatile_threshold = 0.30 / np.sqrt(252)  # Approx 0.0189

    # Apply 'Volatile' label
    # A window is 'Volatile' if it is NOT 'Crisis' AND its daily volatility is high
    volatile_mask = non_crisis_mask & (metrics_df["volatility"] > volatile_threshold)
    metrics_df.loc[volatile_mask, "label"] = "Volatile"

    print("Labeling complete using fixed thresholds.")

    # --- Final Report ---
    print("\n--- Labeling Results ---")
    print("Final Label Distribution:")
    print(
        metrics_df["label"].value_counts(normalize=True).mul(100).round(1).astype(str)
        + "%"
    )

    # --- Step 3: Create Final One-Hot Encoded List ---
    label_map = {
        "Normal": [1.0, 0.0, 0.0],
        "Crisis": [0.0, 1.0, 0.0],
        "Volatile": [0.0, 0.0, 1.0],
    }

    ori_data_s = [label_map[label] for label in metrics_df["label"]]

    return ori_data_s, metrics_df
