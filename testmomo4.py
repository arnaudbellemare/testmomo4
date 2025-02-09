import streamlit as st 
import ccxt
import pandas as pd
import numpy as np
from itertools import combinations
import logging
from statsmodels.tsa.stattools import adfuller

# Setup logging
logging.basicConfig(level=logging.INFO)

# ------------------ Fractional Differencing Functions ------------------
def frac_diff(x, d):
    """
    Fractionally difference time series

    :param x: numeric vector or univariate time series
    :param d: number specifying the fractional difference order.
    :return: fractionally differenced series
    """
    if np.isnan(np.sum(x)):
        return None

    n = len(x)
    if n < 2:
        return None

    # Center the series
    x = np.subtract(x, np.mean(x))

    # Calculate weights (loop-based)
    weights = [0] * n
    weights[0] = -d
    for k in range(2, n):
        weights[k - 1] = weights[k - 2] * (k - 1 - d) / k

    # Apply fractional differencing
    ydiff = list(x)
    for i in range(0, n):
        dat = x[:i]
        w = weights[:i]
        ydiff[i] = x[i] + np.dot(w, dat[::-1])

    return ydiff

def frac_diff_vect(x, d):
    """
    Fractionally difference time series using vectorized operations

    :param x: numeric vector or univariate time series
    :param d: number specifying the fractional difference order.
    :return: fractionally differenced series
    """
    if np.isnan(np.sum(x)):
        return None

    n = len(x)
    if n < 2:
        return None

    # Center the series
    x = np.subtract(x, np.mean(x))

    # Calculate weights using cumulative products
    num = -d * np.cumprod(np.arange(1 - d, n - 1 - d, 1))
    den = np.cumprod(np.arange(2, n, 1))
    weights = np.divide(num, den)
    weights = np.insert(weights, 0, -d)
    weights = np.insert(weights, weights.shape, 0)

    # Build weighted array
    ydiff = x.reshape(1, -1) * weights.reshape(-1, 1)

    # Function to extract diagonal sums (using trace on flipped matrix)
    def tracify(k):
        return np.trace(np.fliplr(ydiff), k)

    vfunc = np.vectorize(tracify)
    ydiff = x + vfunc(np.arange(x.shape[0], 0, -1))

    return ydiff

def apply_frac_diff(df, d, method='vectorized'):
    """
    Apply fractional differencing to a DataFrame of asset prices.

    :param df: DataFrame containing asset price series.
    :param d: Fractional differencing order.
    :param method: Choose 'vectorized' (default) or 'loop'.
    :return: DataFrame with fractionally differenced series.
    """
    if method == 'vectorized':
        return df.apply(lambda col: pd.Series(frac_diff_vect(col.dropna().values, d)), axis=0)
    else:
        return df.apply(lambda col: pd.Series(frac_diff(col.dropna().values, d)), axis=0)

def test_stationarity(series):
    """Perform the Augmented Dickey-Fuller (ADF) test."""
    clean_series = series.dropna()
    if len(clean_series) <= 1:
        return 1.0  # High p-value if not enough data
    try:
        result = adfuller(clean_series, autolag='AIC')
        return result[1]  # Return p-value
    except Exception:
        return 1.0

def find_optimal_d(df, d_values=np.linspace(0.0, 1.0, 21), threshold=0.05):
    """
    Automatically find the optimal fractional differencing parameter d for stationarity.
    
    Iterates over candidate d-values and returns the smallest d that yields stationarity
    (i.e., all series have an ADF p-value below the threshold).
    
    :param df: DataFrame containing asset price series.
    :param d_values: Array of candidate d values.
    :param threshold: p-value threshold to consider a series stationary.
    :return: The optimal d, or None if no candidate d satisfies stationarity.
    """
    for d in d_values:
        frac_diff_df = apply_frac_diff(df, d, method='vectorized')
        p_values = frac_diff_df.apply(test_stationarity)
        if (p_values < threshold).all():
            return d
    return None  # No optimal d found

# ------------------ Exchange Initialization and Market Filtering ------------------
# Initialize Kraken exchange
exchange = ccxt.kraken({
    'enableRateLimit': True,  # Respect rate limits
})

# Define stablecoins and other symbols to exclude
STABLECOINS = {
    'USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'PAX', 'GUSD',
    'USDK', 'UST', 'SUSD', 'FRAX', 'LUSD', 'MIM', 'USDQ', 'TBTC', 'WBTC', 
    'EUL', 'EUR', 'EURT', 'USDS', 'USTS', 'USTC', 'USDG', 'MSOL',
    'TREMP', 'BSX', 'SBR', 'AUD', 'IDEX', 'FIS', 'CSM', 'GBP', 'POWR', 
    'ATLAS', 'XCN', 'BOBA', 'OXY', 'BNC', 'POLIS', 'AIR', 'C98', 'EURT','STRD', 'PYUSD', 'BTT'
}

def fetch_usd_pairs(exchange_instance):
    """
    Fetch USD-denominated trading pairs from Kraken, excluding pairs involving stablecoins.
    
    Args:
        exchange_instance (ccxt.Exchange): The initialized CCXT Kraken exchange instance.
    
    Returns:
        list: A list of USD-denominated trading pairs excluding those involving stablecoins.
    """
    try:
        # Load markets
        markets = exchange_instance.load_markets()
        
        # Extract USD pairs and exclude stablecoins in either base or quote
        usd_pairs = [
            symbol.upper() for symbol in markets.keys()
            if '/USD' in symbol.upper()
            and all(asset.upper() not in STABLECOINS for asset in symbol.split('/'))
        ]
        
        # Remove duplicates and log the count
        usd_pairs = list(set(usd_pairs))
        logging.info(f"Fetched {len(usd_pairs)} USD-denominated trading pairs from Kraken.")
        
        return usd_pairs
    
    except Exception as e:
        logging.error(f"Error fetching markets from {exchange_instance.id}: {e}")
        return []

# ------------------ Portfolio Construction Functions ------------------
def compute_predictability_for_subset(data, portfolio_type="mean_reversion"):
    """
    Given a numpy array data of shape (T, s) for s assets,
    compute the portfolio weights x that optimize the predictability measure:
    
        ν(x)= (x^T A^T Γ A x) / (x^T Γ x).
    
    Steps:
      - Estimate A via least squares regression using lagged data.
      - Compute the covariance matrix Γ.
      - Perform a Cholesky decomposition: Γ = L L^T.
      - Form B = L_inv @ (A^T Γ A) @ L_inv, where L_inv approximates Γ^{-1/2}.
      - For a mean-reverting portfolio, choose the eigenvector corresponding to the smallest eigenvalue.
        For a momentum portfolio, choose the eigenvector corresponding to the largest eigenvalue.
      - Recover portfolio weights: x = L_inv @ z, then normalize.
    """
    # Prepare lagged data: X = S_{t-1}, Y = S_t
    X = data[:-1, :]
    Y = data[1:, :]
    # Estimate A using least squares: Y ≈ X @ A
    A, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    # Compute covariance matrix Γ from the asset series
    Gamma = np.cov(data.T)
    # Cholesky decomposition: Γ = L L^T
    L = np.linalg.cholesky(Gamma)
    L_inv = np.linalg.inv(L)
    # Construct matrix B = Γ^{-1/2} A^T Γ A Γ^{-1/2}
    B = L_inv @ (A.T @ Gamma @ A) @ L_inv
    # Eigen decomposition: choose eigenvector based on portfolio type
    eigvals, eigvecs = np.linalg.eig(B)
    if portfolio_type == "momentum":
        idx = np.argmax(eigvals.real)
    else:
        idx = np.argmin(eigvals.real)
    z = eigvecs[:, idx].real
    # Recover portfolio weights: x = L_inv @ z, then normalize
    x = L_inv @ z
    x = x / np.linalg.norm(x)
    predictability = eigvals[idx].real
    return x, predictability

def greedy_search(data, asset_names, k, max_weight=0.8, min_weight=None, portfolio_type="mean_reversion"):
    """
    Perform a greedy search to select up to k assets that optimize the predictability measure.
    
    For a mean-reverting portfolio, we aim to minimize predictability;
    for a momentum portfolio, we aim to maximize predictability.
    
    The algorithm:
      1. Uses a brute-force technique to find the best two-asset portfolio (subject to diversification constraints).
      2. Iteratively adds one asset at a time that yields the best improvement in the objective,
         ensuring that no single asset weight exceeds max_weight and every nonzero weight is above min_weight.
    
    Returns the selected asset names, corresponding weights, and the predictability measure.
    """
    if min_weight is None:
        min_weight = 1/(4*k)
    n_assets = data.shape[1]
    candidate_indices = list(range(n_assets))
    
    if portfolio_type == "momentum":
        best_measure = -np.inf
    else:
        best_measure = np.inf
    best_pair = None
    best_weights = None

    # Step 1: Brute-force over all asset pairs
    for pair in combinations(candidate_indices, 2):
        sub_data = data[:, list(pair)]
        try:
            weights, measure = compute_predictability_for_subset(sub_data, portfolio_type)
        except Exception:
            continue
        if np.max(np.abs(weights)) > max_weight or np.min(np.abs(weights)) < min_weight:
            continue
        if portfolio_type == "momentum":
            if measure > best_measure:
                best_measure = measure
                best_pair = pair
                best_weights = weights
        else:
            if measure < best_measure:
                best_measure = measure
                best_pair = pair
                best_weights = weights

    if best_pair is None:
        return None, None, None

    current_subset = list(best_pair)
    current_measure = best_measure
    current_weights = best_weights

    # Step 2: Incrementally add assets until reaching k assets
    improved = True
    while len(current_subset) < k and improved:
        improved = False
        best_candidate = None
        best_candidate_weights = None
        best_candidate_measure = current_measure
        for i in candidate_indices:
            if i in current_subset:
                continue
            new_subset = current_subset + [i]
            sub_data = data[:, new_subset]
            try:
                weights, measure = compute_predictability_for_subset(sub_data, portfolio_type)
            except Exception:
                continue
            if np.max(np.abs(weights)) > max_weight or np.min(np.abs(weights)) < min_weight:
                continue
            if portfolio_type == "momentum":
                if measure > best_candidate_measure:
                    best_candidate_measure = measure
                    best_candidate = i
                    best_candidate_weights = weights
            else:
                if measure < best_candidate_measure:
                    best_candidate_measure = measure
                    best_candidate = i
                    best_candidate_weights = weights
        if best_candidate is not None:
            current_subset.append(best_candidate)
            current_measure = best_candidate_measure
            current_weights = best_candidate_weights
            improved = True

    selected_assets = [asset_names[i] for i in current_subset]
    return selected_assets, current_weights, current_measure

# ------------------ Sidebar: User Inputs ------------------
st.sidebar.header("Data and Portfolio Parameters")

# Choose portfolio type: Mean Reversion or Momentum
portfolio_type = st.sidebar.selectbox("Portfolio Type", options=["Mean Reversion", "Momentum"])
portfolio_type_key = portfolio_type.lower().replace(" ", "_")

# Timeframe and data limit
timeframe = st.sidebar.selectbox("Timeframe", options=['1m', '1h', '4h', '1d'], index=2)
limit = st.sidebar.number_input("Number of candles", min_value=24, max_value=1440, value=200, step=10)

# Portfolio constraints
k = st.sidebar.number_input("Max number of assets in portfolio (k)", min_value=2, max_value=20, value=10, step=1)
max_weight = st.sidebar.slider("Max weight per asset", min_value=0.5, max_value=1.0, value=0.8, step=0.05)
min_weight = st.sidebar.slider("Min weight per asset", min_value=0.0, max_value=0.5, value=0.05, step=0.01)

# Asset selection: Fetch USD-denominated trading pairs from Kraken (excluding stablecoins)
st.sidebar.header("Asset Selection")
all_symbols = fetch_usd_pairs(exchange)
use_all = st.sidebar.checkbox("Use all available USD-denominated tickers", value=False)
if use_all:
    selected_symbols = all_symbols
else:
    default_selection = ['BTC/USD', 'ETH/USD', 'LTC/USD', 'XRP/USD', 'BCH/USD']
    selected_symbols = st.sidebar.multiselect("Select assets for portfolio formation", all_symbols, default=default_selection)
st.sidebar.write(f"Total assets selected: {len(selected_symbols)}")

# ------------------ Data Collection ------------------
st.header("Fetching Data from Kraken")
data_dict = {}
for symbol in selected_symbols:
    df = None
    try:
        df = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        st.warning(f"Error fetching data for {symbol}: {e}")
    if df:
        df = pd.DataFrame(df, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        data_dict[symbol] = df['close']

if len(data_dict) < 2:
    st.error("Not enough asset data fetched to construct a portfolio. Please adjust your asset selection or timeframe/limit.")
else:
    # Align data on the timestamp index and drop missing values
    price_df = pd.concat(data_dict, axis=1)
    price_df.dropna(inplace=True)
    st.write("Aligned Price Data (first 5 rows):")
    st.dataframe(price_df.head())

    # ------------------ Optional: Apply Fractional Differencing ------------------
    apply_frac_diff_flag = st.sidebar.checkbox("Apply Fractional Differencing", value=False)
    if apply_frac_diff_flag:
        st.write("Automatically selecting the optimal fractional differencing parameter d using the ADF test...")
        optimal_d = find_optimal_d(price_df, d_values=np.linspace(0.0, 1.0, 21), threshold=0.05)
        if optimal_d is None:
            st.error("Could not find an optimal d that renders all series stationary.")
        else:
            st.write(f"Optimal d automatically selected: {optimal_d:.4f}")
            price_df = apply_frac_diff(price_df, optimal_d, method='vectorized')
            price_df.dropna(inplace=True)
            st.write("Fractionally Differenced Price Data (first 5 rows):")
            st.dataframe(price_df.head())

    # Preprocess the price data: de-mean each asset's series
    S = price_df.values.astype(float)
    S = S - np.mean(S, axis=0)
    st.write("Number of time points used:", S.shape[0])
    
    # ------------------ Portfolio Optimization via Greedy Search ------------------
    st.header(f"Computing the Sparse {portfolio_type} Portfolio")
    selected_assets, weights, measure = greedy_search(
        S, list(price_df.columns), int(k), max_weight, min_weight,
        portfolio_type=portfolio_type_key
    )
    if selected_assets is None:
        st.error("No valid portfolio found with the given constraints.")
    else:
        st.subheader("Selected Portfolio")
        portfolio_df = pd.DataFrame({
            "Asset": selected_assets,
            "Weight": weights
        })
        st.dataframe(portfolio_df)
        if portfolio_type_key == "momentum":
            st.write(f"Predictability measure (momentum): {measure:.4f}")
        else:
            st.write(f"Predictability measure (mean reversion): {measure:.4f}")

st.write("What did you do? Y")
