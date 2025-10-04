<img width="780" height="253" alt="ACM w color" src="https://github.com/user-attachments/assets/ca75b25b-3995-46ae-b041-7e60b4adbab2" />

### The 4.Backtest_Trend_Reversal script is designed to backtest a trend reversal trading strategy and generate bearish signals for a list of stock tickers. Here's a detailed breakdown of its functionality:

*1. Imports and Setup:*

Utilizes libraries like pandas, numpy, yfinance, and backtesting for data handling, financial analysis, and strategy execution.
Loads environment variables using dotenv and sets up logging for debugging.

*2. Data Import and Feature Engineering:*

import_yahoo_finance_daily: Fetches daily OHLC (Open, High, Low, Close) data for a given ticker using the yfinance library.
TickersData Class:
Manages data for multiple tickers.
Adds required derivative columns (e.g., ATR, moving averages, candlestick patterns) to the data.
Saves raw and feature-enhanced data locally for reuse.

*3. Feature Engineering:*

add_required_cols_for_f_v1_basic: Adds essential columns like ATR (Average True Range), moving averages, and candlestick patterns (e.g., hammer, shooting star).
add_features_v1_basic: Adds specific features for the strategy, such as:
Whether the price is below the 200-day moving average.
Whether the price deviation from the moving average exceeds a threshold.

*4. Trading Strategy Logic:*

Position Sizing:
Determines the desired position size based on signals like hammer candlestick patterns, price below moving average, and volatility.
Stop Loss and Profit Targets:
Dynamically adjusts stop-loss levels based on ATR and trade performance.
Sets profit targets for long trades.
Special Situations:
Closes positions during volatility spikes or if the maximum trade duration is exceeded.

*5. Backtesting:*

run_all_tickers:
Runs the backtest for all tickers using the run_backtest_for_ticker function.
Saves performance metrics and trade data to CSV files.
Calculates the SQN (System Quality Number), a measure of strategy performance.

*6. Bearish Signal Generation:*

process_ticker:
Identifies bearish signals based on conditions like shooting star candlestick patterns, price above the moving average, and low volatility.
generate_bearish_signals:
Processes multiple tickers in parallel to generate bearish signals.
Saves the results to a CSV file (bearish_signals.csv).

*7. Logging and Outputs:*

Logs detailed information about the data and strategy execution.
Saves results like performance metrics, trade data, and bearish signals to CSV files.

*8. Main Execution:*

Initializes strategy parameters (e.g., profit targets, stop-loss multipliers).
Runs the backtest for all tickers and calculates the average SQN.
Generates bearish signals for all tickers and saves the results.

*Purpose:*

This script is a quantitative trading tool designed to:
Backtest a trend reversal strategy using historical stock data.
Identify bearish signals for potential short trades.
Evaluate the strategy's performance using metrics like SQN.

### It is useful for traders and analysts looking to automate the process of strategy testing and signal generation
