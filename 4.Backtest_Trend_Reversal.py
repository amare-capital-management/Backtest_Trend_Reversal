from typing import Callable, List, Optional, Tuple
import pandas as pd
import os
import yfinance as yf
import datetime
import logging
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from backtesting.backtesting import Strategy
from backtesting import set_bokeh_output
import numpy as np
import warnings
from derivative_columns.atr import add_tr_delta_col_to_ohlc, add_atr_col_to_df
from derivative_columns.ma import add_moving_average
from derivative_columns.hammer import add_col_is_hammer
from derivative_columns.shooting_star import add_col_is_shooting_star
from utils.import_data import get_local_ticker_data_file_name
from constants2 import (
    FEATURE_COL_NAME_ADVANCED,
    FEATURE_COL_NAME_BASIC,
    DPS_STUB,
    SL_TIGHTENED,
    CLOSED_VOLATILITY_SPIKE,
    CLOSED_MAX_DURATION,
    SS_VOLATILITY_SPIKE,
    SS_MAX_DURATION,
    SS_NO_TODAY,
    LOG_FILE,
    tickers_all
)
from utils.strategy_exec.misc import get_current_position_size, add_tag_to_trades_and_close_position
from customizable.strategy_params import StrategyParams
from strategy.run_backtest_for_ticker import run_backtest_for_ticker

set_bokeh_output(notebook=False)

MUST_HAVE_DERIVATIVE_COLUMNS = {"tr", "tr_delta"}

os.makedirs("data", exist_ok=True)

def import_yahoo_finance_daily(ticker: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    df = stock.history(start="2024-01-01", end=end_date, interval="1d")
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

class TickersData:
    def __init__(self, tickers: list[str], add_features_cols_func: Callable, import_ohlc_func: Callable = import_yahoo_finance_daily):
        self.tickers_data_with_features = {}
        self.add_features_cols_func = add_features_cols_func
        self.import_ohlc_func = import_ohlc_func
        for ticker in tickers:
            df = self.get_df_with_features(ticker=ticker)
            for col in MUST_HAVE_DERIVATIVE_COLUMNS:
                if col not in df.columns:
                    df = add_tr_delta_col_to_ohlc(ohlc_df=df)
            self.tickers_data_with_features[ticker] = df

    def get_df_with_features(self, ticker: str) -> pd.DataFrame:
        filename_with_features = get_local_ticker_data_file_name(ticker, "with_features")
        filename_raw = get_local_ticker_data_file_name(ticker, "raw")

        os.makedirs(os.path.dirname(filename_with_features), exist_ok=True)
        if os.path.exists(filename_with_features):
            df = pd.read_csv(filename_with_features, index_col=0)
            df.index = pd.to_datetime(df.index)
            return df
        if os.path.exists(filename_raw):
            df = pd.read_csv(filename_raw, index_col=0)
            df.index = pd.to_datetime(df.index)
        else:
            df = self.import_ohlc_func(ticker)
            df.to_csv(filename_raw)
            print(f"[SAVED] Raw file for {ticker}: {filename_raw}")
            logging.debug(f"Saved raw file for {ticker}: {filename_raw}")

        df = self.add_features_cols_func(df)
        df.to_csv(filename_with_features)
        print(f"[SAVED] Features file for {ticker}: {filename_with_features}")
        logging.debug(f"Saved features file for {ticker}: {filename_with_features}")

        return df

    def get_data(self, ticker: str) -> pd.DataFrame:
        return self.tickers_data_with_features[ticker]

MOVING_AVERAGE_N = 200
REQUIRED_DERIVATIVE_COLUMNS_F_V1_BASIC = {"atr_14", f"ma_{MOVING_AVERAGE_N}", "is_hammer", "is_shooting_star"}

def add_required_cols_for_f_v1_basic(df: pd.DataFrame) -> pd.DataFrame:
    df_columns = df.columns
    internal_df = df.copy()
    if f"ma_{MOVING_AVERAGE_N}" not in df_columns:
        internal_df = add_moving_average(df=internal_df, n=MOVING_AVERAGE_N)
    if "atr_14" not in df_columns:
        internal_df = add_atr_col_to_df(df=internal_df, n=14, exponential=False)
    if "is_hammer" not in df_columns:
        internal_df = add_col_is_hammer(df=internal_df)
    if "is_shooting_star" not in df_columns:
        internal_df = add_col_is_shooting_star(df=internal_df)
    return internal_df

def add_features_v1_basic(df: pd.DataFrame, atr_multiplier_threshold: int = 6) -> pd.DataFrame:
    res = df.copy()
    for col in REQUIRED_DERIVATIVE_COLUMNS_F_V1_BASIC:
        if col not in res.columns:
            res = add_required_cols_for_f_v1_basic(df=res)
    res[FEATURE_COL_NAME_BASIC] = res["Close"] < res[f"ma_{MOVING_AVERAGE_N}"]
    res[FEATURE_COL_NAME_ADVANCED] = (res["ma_200"] - res["Close"]) >= (res["atr_14"] * atr_multiplier_threshold)
    return res

def get_desired_current_position_size(strategy: Strategy) -> Tuple[Optional[float], float, str]:
    current_position_size = (
        get_current_position_size(
            shares_count=strategy.position.size,
            equity=strategy.equity,
            last_price=strategy.data.Open[-1],
        )
        if strategy.position.size != 0
        else 0
    )
    is_hammer = strategy.data["is_hammer"][-1]
    price_below_ma200 = strategy.data[FEATURE_COL_NAME_ADVANCED][-1]
    volatility_ok = strategy.data["tr_delta"][-1] < 2.5
    desired_position_size: Optional[float] = None
    message = DPS_STUB
    if current_position_size != 0:
        desired_position_size = current_position_size
        message = "Maintain existing position"
        return desired_position_size, current_position_size, message
    if is_hammer and price_below_ma200 and volatility_ok:
        desired_position_size = 1.0
        message = "Enter Long: Hammer reversal below MA200 with moderate volatility"
    return desired_position_size, current_position_size, message

def _get_n_atr(strategy: Strategy) -> float:
    index = len(strategy.data) - 1
    if strategy.data.tr_delta[index] > 1.98 and strategy.trades and strategy.trades[-1].pl > 0:
        return 1.1
    return strategy.parameters.stop_loss_default_atr_multiplier

def update_stop_loss(strategy: Strategy):
    if not strategy.trades:
        return
    n_atr = _get_n_atr(strategy)
    index = len(strategy.data) - 1
    for trade in strategy.trades:
        if trade.is_long:
            sl_price = max(trade.sl or -np.inf, strategy.data.Open[index] - strategy.data.atr_14[index] * n_atr)
        else:
            sl_price = min(trade.sl or np.inf, strategy.data.Open[index] + strategy.data.atr_14[index] * n_atr)
        if sl_price < 0:
            sl_price = None
        if sl_price and trade.sl != sl_price:
            trade.sl = sl_price
        if n_atr == 1.1 and SL_TIGHTENED not in (trade.tag or ""):
            setattr(trade, f"_{trade.__class__.__qualname__}__tag", (trade.tag or "") + SL_TIGHTENED)

def check_set_profit_targets_long_trades(strategy: Strategy):
    last_price = strategy.data.Open[-1]
    min_profit_target_long = None
    trades_long = [trade for trade in strategy.trades if trade.is_long]
    for trade in trades_long:
        if trade.tp is not None:
            min_profit_target_long = min(min_profit_target_long or trade.tp, trade.tp)
    if trades_long and min_profit_target_long is None:
        min_profit_target_long = (float(strategy.parameters.profit_target_long_pct + 100) / 100) * last_price
    for trade in trades_long:
        if trade.tp is None:
            trade.tp = min_profit_target_long

def process_volatility_spike(strategy: Strategy) -> bool:
    if strategy.data.tr_delta[-1] < 2.5:
        return False
    add_tag_to_trades_and_close_position(strategy, CLOSED_VOLATILITY_SPIKE)
    return True

def process_max_duration(strategy: Strategy) -> bool:
    max_trade_duration_long = strategy.parameters.max_trade_duration_long
    if max_trade_duration_long is None or not strategy.trades:
        return False
    max_trade_duration = max((strategy.data.index[-1] - trade.entry_time).days for trade in strategy.trades)
    if strategy.trades[-1].is_long and max_trade_duration > max_trade_duration_long:
        add_tag_to_trades_and_close_position(strategy, CLOSED_MAX_DURATION)
        return True
    return False

def process_special_situations(strategy: Strategy) -> Tuple[bool, str]:
    if process_max_duration(strategy):
        return True, SS_MAX_DURATION
    if process_volatility_spike(strategy):
        return True, SS_VOLATILITY_SPIKE
    return False, SS_NO_TODAY

def run_all_tickers(tickers_data: TickersData, strategy_params: StrategyParams, tickers: list[str]) -> float:
    open("app_run.log", "w", encoding="UTF-8").close()
    performance_res = pd.DataFrame()
    all_trades = pd.DataFrame()
    for ticker in tickers:
        ticker_data = tickers_data.get_data(ticker)
        stat, trades_df, last_day_result = run_backtest_for_ticker(ticker, ticker_data, strategy_params)
        stat = stat.drop(["_strategy", "_equity_curve", "_trades"])
        stat["SQN_modified"] = stat["SQN"] / np.sqrt(stat["# Trades"])
        performance_res[ticker] = stat
        if strategy_params.save_all_trades_in_csv:
            trades_df["Ticker"] = ticker
            all_trades = pd.concat([all_trades, trades_df])
    if len(tickers) > 1:
        performance_res.to_csv("output.csv")
    if strategy_params.save_all_trades_in_csv:
        all_trades.to_csv("all_trades.csv", index=False)
    return performance_res.loc["SQN_modified", :].mean()

def process_ticker(ticker: str, tickers_data: TickersData) -> pd.DataFrame:
    df = tickers_data.get_data(ticker)
    logging.debug(f"Data shape for {ticker}: {df.shape}")
    logging.debug(f"Columns for {ticker}: {list(df.columns)}")
    nan_counts = df[['Close', 'ma_200', 'atr_14', 'tr_delta', 'is_shooting_star']].isna().sum()
    logging.debug(f"NaN counts for {ticker}:\n{nan_counts}")
    df["Bearish_Signal"] = ((df["is_shooting_star"] == True) & (df["Close"] > df["ma_200"]) & (df["tr_delta"] < 3.0))
    shooting_star_count = df["is_shooting_star"].sum()
    uptrend_count = (df["Close"] > df["ma_200"]).sum()
    volatility_count = (df["tr_delta"] < 3.0).sum()
    signal_count = df["Bearish_Signal"].sum()
    logging.debug(f"Shooting star count for {ticker}: {shooting_star_count}")
    logging.debug(f"Uptrend count (Close > ma_200) for {ticker}: {uptrend_count}")
    logging.debug(f"Volatility count (tr_delta < 3.0) for {ticker}: {volatility_count}")
    logging.debug(f"Bearish signal count for {ticker}: {signal_count}")
    df_output = df[["Close", "ma_200", "atr_14", "tr_delta", "is_shooting_star", "Bearish_Signal"]].copy()
    df_output["Ticker"] = ticker
    df_output["Date"] = df_output.index
    df_output["Distance_to_MA200"] = ((df["Close"] - df["ma_200"]) / df["atr_14"]).round(2)
    df_output = df_output[["Ticker", "Date", "Close", "ma_200", "atr_14", "tr_delta", "Distance_to_MA200", "is_shooting_star", "Bearish_Signal"]]
    return df_output[df_output["Bearish_Signal"] == True]

def generate_bearish_signals(tickers: List[str], tickers_data: TickersData) -> pd.DataFrame:
    results = []
    max_workers = min(10, len(tickers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(process_ticker, ticker, tickers_data): ticker for ticker in tickers}
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                df = future.result()
                if not df.empty:
                    results.append(df)
            except Exception as e:
                logging.error(f"Error processing {ticker}: {str(e)}")
    if not results:
        logging.warning("No valid data processed for any ticker")
        return pd.DataFrame(columns=["Ticker", "Date", "Close", "ma_200", "atr_14", "tr_delta", "Distance_to_MA200", "is_shooting_star", "Bearish_Signal"])
    result_df = pd.concat(results)
    result_df[["Close", "ma_200", "atr_14", "tr_delta"]] = result_df[["Close", "ma_200", "atr_14", "tr_delta"]].round(2)
    output_file = "bearish_signals.csv"
    result_df.to_csv(output_file, index=False)
    logging.debug(f"Bearish signals saved to {output_file}")
    return result_df

logging.basicConfig(level=logging.DEBUG, format="%(message)s", filename=LOG_FILE, encoding="utf-8", filemode="a")
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    load_dotenv()
    open(LOG_FILE, "w", encoding="utf-8").close()
    strategy_params = StrategyParams(
        max_trade_duration_long=100,
        max_trade_duration_short=100,
        profit_target_long_pct=29.9,
        profit_target_short_pct=29.9,
        stop_loss_default_atr_multiplier=2.5,
        save_all_trades_in_csv=True,
    )
    tickers_data = TickersData(add_features_cols_func=add_features_v1_basic, tickers=tickers_all)
    SQN_modified_mean = run_all_tickers(tickers_data=tickers_data, tickers=tickers_all, strategy_params=strategy_params)
    logging.debug(f"SQN_modified_mean={SQN_modified_mean}")
    print(f"SQN_modified_mean={SQN_modified_mean}, see output.csv")
    bearish_signals_df = generate_bearish_signals(tickers=tickers_all, tickers_data=tickers_data)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', None)
    print(f"Generated bearish signals for {len(tickers_all)} tickers.")
    print(f"Total signals: {len(bearish_signals_df)}")
    print(f"Results saved to bearish_signals.csv")