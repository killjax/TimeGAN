import numpy as np
import pandas as pd
import yfinance as yf


def real_data_loading(data_name, start_date, end_date):
    assert data_name in ["^GSPC", "AAPL", "MSFT", "GOOG", "AMZN"]

    ori_data = yf.download(
        data_name, start=start_date, end=end_date, progress=False, auto_adjust=False
    )

    # *** THIS IS THE FIX ***
    # Force pandas to work on a clean copy, not a view
    # This prevents all SettingWithCopyWarning and related KeyErrors
    ori_data = ori_data.copy()

    # 1. Log Returns
    # Using 'Adj Close' is common, but 'Close' is fine as in your original.
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
    # This line is fine as it modifies the 'rs' series directly
    rs[avg_loss == 0] = np.inf

    ori_data["RSI"] = 100.0 - (100.0 / (1.0 + rs))

    # *** THIS LINE WAS REMOVED AS IT IS REDUNDANT ***
    # The line above already sets RSI to 100 when rs is np.inf
    # ori_data.loc[rs == np.inf, "RSI"] = 100

    # --- End: New Feature Engineering Block ---

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
