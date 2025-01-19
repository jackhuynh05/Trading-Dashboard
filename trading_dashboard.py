import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pykalman import KalmanFilter
import openpyxl
from datetime import datetime

# Set Streamlit page configuration
st.set_page_config(
    page_title="Trading Strategies Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title of the app
st.title("Trading Strategies Simulator")

# Sidebar for user inputs
st.sidebar.header("Input Parameters")

# Function to load data
@st.cache_data
def load_data(dal_file, wti_file):
    dal_data = pd.read_csv(dal_file, index_col="Date", parse_dates=True)
    wti_data = pd.read_csv(wti_file, index_col="Date", parse_dates=True)
    return dal_data, wti_data

# File upload
st.sidebar.subheader("Upload Data Files")
dal_file = st.sidebar.file_uploader("Upload dal_data.csv", type=["csv"])
wti_file = st.sidebar.file_uploader("Upload wti_data.csv", type=["csv"])

# If files are not uploaded, use sample data
@st.cache_data
def load_sample_data():
    N = 2500  # Number of time steps (days)
    dt = 1/252  # Time step in years (assuming 252 trading days per year)
    
    # Generate a date range
    dates = pd.date_range(start="2020-01-01", periods=N, freq='D')
    
    # Function to simulate GBM
    def simulate_gbm(S0, mu, sigma, N, dt):
        S = np.zeros(N)
        S[0] = S0
        for t in range(1, N):
            Z = np.random.standard_normal()
            S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        return S
    
    # Parameters for DAL (Delta Air Lines)
    mu_dal = np.random.uniform(0.05, 0.15)  # Drift between 5% and 15% annual
    sigma_dal = np.random.uniform(0.1, 0.3)  # Volatility between 10% and 30% annual
    S0_dal = 100  # Initial price
    
    # Simulate DAL Close Prices
    dal_close = simulate_gbm(S0=S0_dal, mu=mu_dal, sigma=sigma_dal, N=N, dt=dt)
    
    # Parameters for WTI (West Texas Intermediate)
    mu_wti = np.random.uniform(0.05, 0.15)
    sigma_wti = np.random.uniform(0.1, 0.3)
    S0_wti = 50  # Initial price
    
    # Simulate WTI Close Prices
    wti_close = simulate_gbm(S0=S0_wti, mu=mu_wti, sigma=sigma_wti, N=N, dt=dt)
    
    # Simulate Volume as random integers between 500 and 1500
    volume_dal = np.random.randint(500, 1500, size=N)
    volume_wti = np.random.randint(500, 1500, size=N)
    
    # **Start of Added Code: Generate High and Low Prices**
    # Define maximum intraday movement percentages
    max_intraday_up = 0.02  # Up to +2% of Close
    max_intraday_down = 0.02  # Down to -2% of Close
    
    # Generate High prices for DAL
    high_dal = dal_close * (1 + np.random.uniform(0, max_intraday_up, size=N))
    # Ensure High is at least equal to Close
    high_dal = np.maximum(high_dal, dal_close)
    
    # Generate Low prices for DAL
    low_dal = dal_close * (1 - np.random.uniform(0, max_intraday_down, size=N))
    # Ensure Low is at most equal to Close
    low_dal = np.minimum(low_dal, dal_close)
    
    # Generate High prices for WTI
    high_wti = wti_close * (1 + np.random.uniform(0, max_intraday_up, size=N))
    high_wti = np.maximum(high_wti, wti_close)
    
    # Generate Low prices for WTI
    low_wti = wti_close * (1 - np.random.uniform(0, max_intraday_down, size=N))
    low_wti = np.minimum(low_wti, wti_close)
    # **End of Added Code**
    
    # Create DataFrames with High and Low prices
    dal_data = pd.DataFrame({
        'Close': dal_close,
        'High': high_dal,
        'Low': low_dal,
        'Volume': volume_dal
    }, index=dates)
    
    wti_data = pd.DataFrame({
        'Close': wti_close,
        'High': high_wti,
        'Low': low_wti,
        'Volume': volume_wti
    }, index=dates)
    
    return dal_data, wti_data

if dal_file and wti_file:
    dal_data, wti_data = load_data(dal_file, wti_file)
else:
    st.warning("Using sample data as files are not uploaded.")
    dal_data, wti_data = load_sample_data()

# User inputs for parameters
stop_loss_variable = st.sidebar.number_input(
    "Stop Loss (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01
)

TARGET_VOL = st.sidebar.number_input(
    "Target Volatility (Annualized)", min_value=0.01, max_value=1.0, value=0.1, step=0.01
)

st.sidebar.subheader("Strategy Allocations (%)")
allocation_macd = st.sidebar.number_input("MACD Allocation (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
allocation_bollinger = st.sidebar.number_input("Bollinger Bands Allocation (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
allocation_rsi = st.sidebar.number_input("RSI Allocation (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
allocation_cmf = st.sidebar.number_input("Chaikin MF Allocation (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
allocation_pairs = st.sidebar.number_input("Pairs Trading Allocation (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)

# Ensure allocations sum to 100%
total_allocation = allocation_macd + allocation_bollinger + allocation_rsi + allocation_cmf + allocation_pairs
if total_allocation != 100:
    st.sidebar.error("Total allocation must sum to 100%. Please adjust the sliders.")
    st.stop()

# Main logic functions (adapted from your provided code)
def prepare_data(data, target_vol):
    data['Daily Returns'] = data['Close'].pct_change()
    data['Realized Volatility'] = data['Daily Returns'].rolling(window=20).std() * np.sqrt(252)
    data['Scaling Factor'] = target_vol / data['Realized Volatility']
    data['Scaling Factor'] = data['Scaling Factor'].fillna(1)
    return data

dal_data = prepare_data(dal_data, TARGET_VOL)
wti_data = prepare_data(wti_data, TARGET_VOL)

def kalman_filter_model(stock1_data, stock2_data):
    # Ensure stock1_data and stock2_data are DataFrames
    if not isinstance(stock1_data, pd.DataFrame):
        stock1_data = pd.DataFrame(stock1_data, columns=['Close'])
    if not isinstance(stock2_data, pd.DataFrame):
        stock2_data = pd.DataFrame(stock2_data, columns=['Close'])

    # Align the two series by their time index
    merged_data = pd.merge(stock1_data, stock2_data, left_index=True, right_index=True, suffixes=('_stock1', '_stock2'))

    # Calculate the spread between the aligned series
    spread = merged_data['Close_stock1'] - merged_data['Close_stock2']

    # Apply Kalman Filter to the spread
    kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1],
                      initial_state_mean=0, initial_state_covariance=1,
                      observation_covariance=1, transition_covariance=0.01)

    state_means, _ = kf.filter(spread.values)  # Only returning state means
    spread_mean = state_means.flatten()  # The Kalman Filter's filtered estimate of the spread

    return spread, spread_mean  # Return spread and its mean estimate

# Define trading strategy functions (macd_trade, bollinger_trade, rsi_trade, chaikin_trade, pairs_trade)
# ... [Include the trading strategy functions here, adapting them to accept parameters as needed]

# To save space, I'll adapt the macd_trade function as an example. You should similarly adapt the other functions.

def macd_trade(data, initial_cash=3300, stop_loss=0.1):
    data['12-day EMA'] = data['Close'].ewm(span=12).mean()
    data['26-day EMA'] = data['Close'].ewm(span=26).mean()
    data['MACD'] = data['12-day EMA'] - data['26-day EMA']
    data['Signal Line'] = data['MACD'].ewm(span=9).mean()

    dates = [data.index[0]]
    cash = initial_cash
    portfolio_value = [cash]
    positions = {'stock1_shares': 0, 'cash': cash}
    stop_loss_price = None

    for index, row in data.iterrows():
        dates.append(index)
        current_cash = positions['cash']
        price = row['Close']
        position_size = min(row['Scaling Factor'], 1)

        # Long
        if row['MACD'] > row['Signal Line'] and row['MACD'] < 0:
            if positions['stock1_shares'] < 0:
                qty = -positions['stock1_shares']
                positions['cash'] += -qty * price
                positions['stock1_shares'] = 0

                qty = current_cash / price * position_size
                positions['stock1_shares'] = qty
                positions['cash'] -= qty * price
                stop_loss_price = price * (1 - stop_loss)

            elif positions['stock1_shares'] == 0:
                qty = current_cash / price * position_size
                positions['stock1_shares'] = qty
                positions['cash'] -= qty * price
                stop_loss_price = price * (1 - stop_loss)

        # Reversal (shorting)
        if row['MACD'] > row['Signal Line'] and row['MACD'] > 0:
            if positions['stock1_shares'] > 0:
                qty = positions['stock1_shares']
                positions['cash'] += qty * price
                positions['stock1_shares'] = 0

                qty = current_cash / price * position_size
                positions['stock1_shares'] = -qty
                positions['cash'] -= -qty * price
                stop_loss_price = price * (1 + stop_loss)
            elif positions['stock1_shares'] == 0:
                qty = current_cash / price * position_size
                positions['stock1_shares'] = -qty
                positions['cash'] -= -qty * price
                stop_loss_price = price * (1 + stop_loss)

        # Stop Loss
        if positions['stock1_shares'] > 0 and price <= stop_loss_price:
            qty = positions['stock1_shares']
            positions['cash'] += qty * price
            positions['stock1_shares'] = 0
        if positions['stock1_shares'] < 0 and price >= stop_loss_price:
            qty = -positions['stock1_shares']
            positions['cash'] += -qty * price
            positions['stock1_shares'] = 0

        # Calculate current portfolio value
        positions['cash'] *= 1.00020913  # Rate / 252
        portfolio_value.append(positions['cash'] + (positions['stock1_shares'] * price))

    # Calculate statistics
    portfolio_returns = [
        (current - previous) / previous * 100
        for previous, current in zip(portfolio_value[:-1], portfolio_value[1:])
    ]

    portfolio_mean_returns = np.mean(portfolio_returns) * 252
    portfolio_std_percentage = np.std(portfolio_returns) * np.sqrt(252)
    max_drawdown = calculate_max_drawdown(portfolio_value)

    # Prepare results
    stats = {
        'Initial Value': initial_cash,
        'Final Value': portfolio_value[-1],
        'Annualized Return (%)': portfolio_mean_returns,
        'Annualized Std Dev (%)': portfolio_std_percentage,
        'Sharpe Ratio': (portfolio_mean_returns - 5.27) / portfolio_std_percentage if portfolio_std_percentage !=0 else np.nan,
        'Max Drawdown (%)': max_drawdown
    }

    return portfolio_value, stats

def calculate_max_drawdown(portfolio_values):
    portfolio_array = np.array(portfolio_values)
    cumulative_max = np.maximum.accumulate(portfolio_array)
    drawdowns = (cumulative_max - portfolio_array) / cumulative_max
    max_drawdown = np.max(drawdowns) * 100
    return max_drawdown

# Similarly, adapt bollinger_trade, rsi_trade, chaikin_trade, and pairs_trade functions to accept parameters and return portfolio values and stats.

# For brevity, I'll define placeholders for these functions. You should replace these with the actual implementations adapted similarly to macd_trade.

def bollinger_trade(data, initial_cash=2020, stop_loss=0.1):
    data['Rolling Mean'] = data['Close'].rolling(window=20, min_periods=1).mean()
    data['Rolling Std'] = data['Close'].rolling(window=20, min_periods=1).std()
    data['Upper BB'] = data['Rolling Mean'] + (data['Rolling Std'] * 2)
    data['Lower BB'] = data['Rolling Mean'] - (data["Rolling Std"] * 2)

    dates = [data.index[0]]
    cash = initial_cash
    portfolio_value = [cash]
    positions = {'stock1_shares': 0, 'cash': cash}
    for index, row in data.iterrows():
        dates.append(index)
        current_cash = positions['cash']
        price = row['Close']
        position_size = min(row['Scaling Factor'], 1) 
       # Bollinger Band Signals
        if price < row['Lower BB']:
            if positions['stock1_shares'] < 0:
                qty = -positions['stock1_shares']
                positions['cash'] += -qty * price
                positions['stock1_shares'] = 0

                qty = current_cash / price * position_size
                positions['stock1_shares'] = qty
                positions['cash'] -= qty * price
                stop_loss_price = price * (1 - stop_loss_variable)
            elif positions['stock1_shares'] == 0:
                qty = current_cash / price * position_size
                positions['stock1_shares'] = qty
                positions['cash'] -= qty * price
                stop_loss_price = price * (1 - stop_loss_variable)

        # Reversal (shorting)
        if price > row['Upper BB']:
            if positions['stock1_shares'] > 0:
                qty = positions['stock1_shares']
                positions['cash'] += qty * price
                positions['stock1_shares'] = 0

                qty = current_cash / price * position_size
                positions['stock1_shares'] = -qty
                positions['cash'] -= -qty * price 
                stop_loss_price = price * (1 + stop_loss_variable)
            elif positions['stock1_shares'] == 0:
                qty = current_cash / price * position_size
                positions['stock1_shares'] = -qty
                positions['cash'] -= -qty * price
                stop_loss_price = price * (1 + stop_loss_variable)

        # Stop Loss
        if positions['stock1_shares'] > 0 and price <= stop_loss_price:
            qty = positions['stock1_shares']
            positions['cash'] += qty * price
            positions['stock1_shares'] = 0
        if positions['stock1_shares'] < 0 and price >= stop_loss_price:
            qty = -positions['stock1_shares']
            positions['cash'] += -qty * price
            positions['stock1_shares'] = 0

        # Calculate current portfolio value
        positions['cash'] *= 1.00020913 # Rate / 252
        portfolio_value.append(positions['cash'] + (positions['stock1_shares'] * price))
    
    portfolio_returns = [
        (current - previous) / previous * 100
        for previous, current in zip(portfolio_value[:-1], portfolio_value[1:])
    ]

    portfolio_mean_returns = np.mean(portfolio_returns) * 252
    portfolio_std_percentage = np.std(portfolio_returns) * np.sqrt(252)
    max_drawdown = calculate_max_drawdown(portfolio_value)

    stats = {
        'Initial Value': initial_cash,
        'Final Value': portfolio_value[-1],
        'Annualized Return (%)': portfolio_mean_returns,
        'Annualized Std Dev (%)': portfolio_std_percentage,
        'Sharpe Ratio': (portfolio_mean_returns - 5.27) / portfolio_std_percentage if portfolio_std_percentage !=0 else np.nan,
        'Max Drawdown (%)': max_drawdown
    }
    return portfolio_value, stats

def rsi_trade(data, initial_cash=500, stop_loss=0.1):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss

    data['RSI'] = 100 - (100 / (1 + rs))

    dates = [data.index[0]]
    cash = initial_cash
    portfolio_value = [cash]
    positions = {'stock1_shares': 0, 'cash': cash}
    for index, row in data.iterrows():
        dates.append(index)
        current_cash = positions['cash']
        price = row['Close']
        position_size = min(row['Scaling Factor'], 1)
       # RSI Signals
        if row['RSI'] < 30:
            if positions['stock1_shares'] < 0:
                qty = -positions['stock1_shares']
                positions['cash'] += -qty * price
                positions['stock1_shares'] = 0

                qty = current_cash / price * position_size
                positions['stock1_shares'] = qty
                positions['cash'] -= qty * price
                stop_loss_price = price * (1 - stop_loss_variable)
            elif positions['stock1_shares'] == 0:
                qty = current_cash / price * position_size
                positions['stock1_shares'] = qty
                positions['cash'] -= qty * price
                stop_loss_price = price * (1 - stop_loss_variable)
        
        # Reversal (shorting)        
        if row['RSI'] > 70:
            if positions['stock1_shares'] > 0:
                qty = positions['stock1_shares']
                positions['cash'] += qty * price
                positions['stock1_shares'] = 0

                qty = current_cash / price * position_size
                positions['stock1_shares'] = -qty
                positions['cash'] -= -qty * price 
                stop_loss_price = price * (1 + stop_loss_variable)
            elif positions['stock1_shares'] == 0:
                qty = current_cash / price * position_size
                positions['stock1_shares'] = -qty
                positions['cash'] -= -qty * price
                stop_loss_price = price * (1 + stop_loss_variable)

        # Stop Loss
        if positions['stock1_shares'] > 0 and price <= stop_loss_price:
            qty = positions['stock1_shares']
            positions['cash'] += qty * price
            positions['stock1_shares'] = 0
        if positions['stock1_shares'] < 0 and price >= stop_loss_price:
            qty = -positions['stock1_shares']
            positions['cash'] += -qty * price
            positions['stock1_shares'] = 0

        # Calculate current portfolio value
        positions['cash'] *= 1.00020913 # Rate / 252
        portfolio_value.append(positions['cash'] + (positions['stock1_shares'] * price))
   
    portfolio_returns = [
        (current - previous) / previous * 100
        for previous, current in zip(portfolio_value[:-1], portfolio_value[1:])
    ]

    portfolio_mean_returns = np.mean(portfolio_returns) * 252
    portfolio_std_percentage = np.std(portfolio_returns) * np.sqrt(252)
    max_drawdown = calculate_max_drawdown(portfolio_value)
    
    stats = {
        'Initial Value': initial_cash,
        'Final Value': portfolio_value[-1],
        'Annualized Return (%)': portfolio_mean_returns,
        'Annualized Std Dev (%)': portfolio_std_percentage,
        'Sharpe Ratio': (portfolio_mean_returns - 5.27) / portfolio_std_percentage if portfolio_std_percentage !=0 else np.nan,
        'Max Drawdown (%)': max_drawdown
    }
    return portfolio_value, stats

def chaikin_trade(data, initial_cash=1590, stop_loss=0.1, n=20):
    money_flow_multiplier = ((data['Close'] - data['Low']) 
                                - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
    money_flow_multiplier = money_flow_multiplier.replace([np.inf, -np.inf], np.nan).fillna(0)

    money_flow_volume = money_flow_multiplier * data['Volume']

    mfv_rolling = money_flow_volume.rolling(window=n).sum()
    vol_rolling = data['Volume'].rolling(window=n).sum()

    data['CMF'] = mfv_rolling / vol_rolling
    data['CMF'].fillna(0, inplace=True)  # fill initial n-1 rows with 0 if needed

    dates = [data.index[0]]
    cash = initial_cash
    portfolio_value = [cash]

    # We'll store in a dictionary
    positions = {
        'stock1_shares': 0,
        'cash': cash
    }

    # Iterate row by row
    for idx, row in data.iterrows():
        dates.append(idx)
        price = row['Close']
        current_cash = positions['cash']
        cmf_value = row['CMF']
        position_size = min(row['Scaling Factor'], 1)

        # 1) Determine bullish or bearish
        if cmf_value > 0:
            # bullish => we want to be long
            if positions['stock1_shares'] < 0:
                # close short first
                qty = -positions['stock1_shares']
                positions['cash'] += -qty * price
                positions['stock1_shares'] = 0

                # open new long
                qty = current_cash / price * position_size
                positions['stock1_shares'] = qty
                positions['cash'] -= qty * price
                stop_loss_price = price * (1 - stop_loss_variable)

            elif positions['stock1_shares'] == 0:
                # open a new long
                qty = current_cash / price * position_size
                positions['stock1_shares'] = qty
                positions['cash'] -= qty * price
                stop_loss_price = price * (1 - stop_loss_variable)

        else:
            # cmf_value <= 0 => we want to be short
            if positions['stock1_shares'] > 0:
                # close any long
                qty = positions['stock1_shares']
                positions['cash'] += qty * price
                positions['stock1_shares'] = 0

                # open short
                qty = current_cash / price * position_size
                positions['stock1_shares'] = -qty
                positions['cash'] -= -qty * price
                stop_loss_price = price * (1 + stop_loss_variable)

            elif positions['stock1_shares'] == 0:
                # open short
                qty = current_cash / price * position_size
                positions['stock1_shares'] = -qty
                positions['cash'] -= -qty * price
                stop_loss_price = price * (1 + stop_loss_variable)

        # 2) Check stop-loss
        if (positions['stock1_shares'] > 0) and (price <= stop_loss_price):
            # exit the long
            qty = positions['stock1_shares']
            positions['cash'] += qty * price
            positions['stock1_shares'] = 0

        elif (positions['stock1_shares'] < 0) and (price >= stop_loss_price):
            # exit the short
            qty = -positions['stock1_shares']
            positions['cash'] += -qty * price
            positions['stock1_shares'] = 0

        # 3) Update daily compounding or similar
        positions['cash'] *= 1.00020913 
        # 4) Save portfolio value
        total_val = positions['cash'] + (positions['stock1_shares'] * price)
        portfolio_value.append(total_val)

    portfolio_returns = [
        (current - previous) / previous * 100
        for previous, current in zip(portfolio_value[:-1], portfolio_value[1:])
    ]

    portfolio_mean_returns = np.mean(portfolio_returns) * 252
    portfolio_std_percentage = np.std(portfolio_returns) * np.sqrt(252)
    max_drawdown = calculate_max_drawdown(portfolio_value)

   
    stats = {
        'Initial Value': initial_cash,
        'Final Value': portfolio_value[-1],
        'Annualized Return (%)': portfolio_mean_returns,
        'Annualized Std Dev (%)': portfolio_std_percentage,
        'Sharpe Ratio': (portfolio_mean_returns - 5.27) / portfolio_std_percentage if portfolio_std_percentage !=0 else np.nan,
        'Max Drawdown (%)': max_drawdown
    }
    return portfolio_value, stats

def pairs_trade(data1, data2, initial_cash=2590, stop_loss=0.1):
    cash = initial_cash
    portfolio_value = [cash]
    positions = {'stock1_shares': 0, 'stock2_shares': 0, 'cash': cash}
    zscore_history = []

    stock1_data = data1[['Close']]
    stock2_data = data2[['Close']]

    # Align the two series by their time index (only keep overlapping data)
    merged_data = pd.merge(stock1_data, stock2_data, left_index=True, right_index=True, how='inner', suffixes=('_stock1', '_stock2'))

    # Calculate spread and Z-score using the Kalman Filter
    spread, spread_mean = kalman_filter_model(merged_data[['Close_stock1']], merged_data[['Close_stock2']])
    spread_var = np.var(spread - spread_mean)

    threshold = 1.645 # 90% Confidence

    # Initialize position tracking for the pair
    positions = {'stock1_shares': 0, 'stock2_shares': 0, 'cash': cash}

    for i in range(len(spread)):
        current_cash = positions['cash']
        price1 = merged_data['Close_stock1'].iloc[i]
        price2 = merged_data['Close_stock2'].iloc[i]
        dal_size = min(dal_data['Scaling Factor'].iloc[i], 1)
        wti_size = min(dal_data['Scaling Factor'].iloc[i], 1)
        
        if dal_size + wti_size > 1:
            dal_size = 0.5
            wti_size = 0.5

        # Calculate Z-score
        zscore = (spread[i] - spread_mean[i]) / np.sqrt(spread_var)
        zscore_history.append(zscore)

        # Trading Logic: Buy/Sell based on Z-score thresholds
        if zscore < -threshold and (positions['stock1_shares'] == 0 or positions['stock1_shares'] < 0):
            qty1 = -positions['stock1_shares']
            qty2 = positions['stock2_shares']
            
            positions['cash'] += (qty2 * price2) - (qty1 * price1)
            positions['stock1_shares'] = 0
            positions['stock2_shares'] = 0
            
            qty1 = current_cash / price1 * dal_size
            qty2 = current_cash / price2 * wti_size
            # Buy stock1, Sell stock2 (enter position)
            positions['stock1_shares'] = qty1
            positions['stock2_shares'] = -qty2
            positions['cash'] -= (qty1 * price1) - (qty2 * price2)

        elif zscore > threshold and (positions['stock1_shares'] > 0 or positions['stock1_shares'] == 0):
            # Sell stock1, Buy stock2 (close position)
            qty1 = positions['stock1_shares']
            qty2 = -positions['stock2_shares']
            
            positions['cash'] += (qty1 * price1) - (qty2 * price2)
            positions['stock1_shares'] = 0
            positions['stock2_shares'] = 0

            qty1 = current_cash / price1 * dal_size
            qty2 = current_cash / price2 * wti_size

            positions['stock1_shares'] = -qty1
            positions['stock2_shares'] = qty2
            positions['cash'] -= (qty2 * price2) - (qty1 * price1)


        # Calculate current portfolio value
        positions['cash'] *= 1.00020913 # Rate / 252
        portfolio_value.append(positions['cash'] + 
                                (positions['stock1_shares'] * price1) + 
                                (positions['stock2_shares'] * price2))
    portfolio_returns = [
        (current - previous) / previous * 100
        for previous, current in zip(portfolio_value[:-1], portfolio_value[1:])
    ]

    portfolio_mean_returns = np.mean(portfolio_returns) * 252
    portfolio_std_percentage = np.std(portfolio_returns) * np.sqrt(252)
    max_drawdown = calculate_max_drawdown(portfolio_value)

   
    stats = {
        'Initial Value': initial_cash,
        'Final Value': portfolio_value[-1],
        'Annualized Return (%)': portfolio_mean_returns,
        'Annualized Std Dev (%)': portfolio_std_percentage,
        'Sharpe Ratio': (portfolio_mean_returns - 5.27) / portfolio_std_percentage if portfolio_std_percentage !=0 else np.nan,
        'Max Drawdown (%)': max_drawdown
    }
    
    return portfolio_value, stats, zscore_history

# Run strategies based on allocations
st.header("Strategy Performance")

# Initialize dictionaries to store portfolio values and statistics
portfolio_values = {}
strategy_stats = {}

# MACD Strategy
if allocation_macd > 0:
    initial_cash_macd = 10000 * (allocation_macd / 100)
    macd_pv, macd_stats = macd_trade(dal_data.copy(), initial_cash=initial_cash_macd, stop_loss=stop_loss_variable)
    portfolio_values['MACD'] = macd_pv
    strategy_stats['MACD'] = macd_stats

# Bollinger Bands Strategy
if allocation_bollinger > 0:
    initial_cash_bollinger = 10000 * (allocation_bollinger / 100)
    bollinger_pv, bollinger_stats = bollinger_trade(dal_data.copy(), initial_cash=initial_cash_bollinger, stop_loss=stop_loss_variable)
    portfolio_values['Bollinger Bands'] = bollinger_pv
    strategy_stats['Bollinger Bands'] = bollinger_stats

# RSI Strategy
if allocation_rsi > 0:
    initial_cash_rsi = 10000 * (allocation_rsi / 100)
    rsi_pv, rsi_stats = rsi_trade(dal_data.copy(), initial_cash=initial_cash_rsi, stop_loss=stop_loss_variable)
    portfolio_values['RSI'] = rsi_pv
    strategy_stats['RSI'] = rsi_stats

# Chaikin MF Strategy
if allocation_cmf > 0:
    initial_cash_cmf = 10000 * (allocation_cmf / 100)
    cmf_pv, cmf_stats = chaikin_trade(dal_data.copy(), initial_cash=initial_cash_cmf, stop_loss=stop_loss_variable)
    portfolio_values['Chaikin MF'] = cmf_pv
    strategy_stats['Chaikin MF'] = cmf_stats

# Pairs Trading Strategy
if allocation_pairs > 0:
    initial_cash_pairs = 10000 * (allocation_pairs / 100)
    pairs_pv, pairs_stats, zscore_history_pairs = pairs_trade(dal_data.copy(), wti_data.copy(), initial_cash=initial_cash_pairs, stop_loss=stop_loss_variable)
    portfolio_values['Pairs Trading'] = pairs_pv
    strategy_stats['Pairs Trading'] = pairs_stats

# Combine all portfolio values to calculate total portfolio
if portfolio_values:
    min_length = min(len(pv) for pv in portfolio_values.values())
    for key in portfolio_values:
        portfolio_values[key] = portfolio_values[key][:min_length]
    total_portfolio_value = np.sum([pv[:min_length] for pv in portfolio_values.values()], axis=0).tolist()

    # Calculate total portfolio statistics
    portfolio_returns = [
        (current - previous) / previous * 100
        for previous, current in zip(total_portfolio_value[:-1], total_portfolio_value[1:])
    ]
    portfolio_mean_returns = np.mean(portfolio_returns) * 252
    portfolio_std_percentage = np.std(portfolio_returns) * np.sqrt(252)
    max_drawdown = calculate_max_drawdown(total_portfolio_value)

    portfolio_stats = {
        'Initial Value': total_portfolio_value[0],
        'Final Value': total_portfolio_value[-1],
        'Annualized Return (%)': portfolio_mean_returns,
        'Annualized Std Dev (%)': portfolio_std_percentage,
        'Sharpe Ratio': (portfolio_mean_returns - 5.27) / portfolio_std_percentage if portfolio_std_percentage !=0 else np.nan,
        'Max Drawdown (%)': max_drawdown
    }

    strategy_stats['Total Portfolio'] = portfolio_stats
else:
    total_portfolio_value = []
    portfolio_stats = {}

# Display plots
if total_portfolio_value:
    st.subheader("Overall Portfolio Returns")
    fig_portfolio, ax_portfolio = plt.subplots(figsize=(10, 6))
    ax_portfolio.plot(total_portfolio_value, color='green', label='Total Portfolio')
    ax_portfolio.set_title("Overall Portfolio Returns")
    ax_portfolio.set_xlabel("Time")
    ax_portfolio.set_ylabel("Portfolio Value ($)")
    ax_portfolio.legend()
    st.pyplot(fig_portfolio)


# Display statistics
st.subheader("Strategy Statistics")

stats_df = pd.DataFrame(strategy_stats).T
stats_df = stats_df[['Initial Value', 'Final Value', 'Annualized Return (%)',
                     'Annualized Std Dev (%)', 'Sharpe Ratio', 'Max Drawdown (%)']]

st.dataframe(stats_df.style.format({
    'Initial Value': "${:,.2f}",
    'Final Value': "${:,.2f}",
    'Annualized Return (%)': "{:.2f}%",
    'Annualized Std Dev (%)': "{:.2f}%",
    'Sharpe Ratio': "{:.2f}",
    'Max Drawdown (%)': "{:.2f}%"
}))

# Download results as Excel
st.subheader("Download Results")
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Results')
    processed_data = output.getvalue()
    return processed_data


import io

if st.button("Download Statistics as Excel"):
    if not stats_df.empty:
        excel_data = convert_df_to_excel(stats_df)
        st.download_button(
            label="Download Excel",
            data=excel_data,
            file_name='trading_strategy_statistics.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    else:
        st.warning("No statistics available to download.")


# Show individual strategy plots
st.subheader("Individual Strategy Returns")
strategy_selection = st.multiselect(
    "Select strategies to display",
    options=list(portfolio_values.keys()),
    default=list(portfolio_values.keys())
)

if strategy_selection:
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for strategy in strategy_selection:
        ax2.plot(portfolio_values[strategy], label=strategy)
    ax2.set_title("Selected Strategy Returns")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Portfolio Value ($)")
    ax2.legend()
    st.pyplot(fig2)

# Display z-score for Pairs Trading
if zscore_history_pairs:
    st.subheader("Pairs Trading Z-Score Over Time")
    fig_zscore, ax_zscore = plt.subplots(figsize=(10, 6))
    ax_zscore.plot(zscore_history_pairs, color='purple', label='Z-Score')
    ax_zscore.axhline(1.645, color='red', linestyle='--', label='Upper Threshold')
    ax_zscore.axhline(-1.645, color='green', linestyle='--', label='Lower Threshold')
    ax_zscore.set_title("Pairs Trading Z-Score Over Time")
    ax_zscore.set_xlabel("Time")
    ax_zscore.set_ylabel("Z-Score")
    ax_zscore.legend()
    st.pyplot(fig_zscore)

