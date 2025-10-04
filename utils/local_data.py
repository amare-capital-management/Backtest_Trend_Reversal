import os
import sys
from typing import Callable, Dict, List, Optional, Set
import pandas as pd
from derivative_columns.atr import add_tr_delta_col_to_ohlc
from utils.import_data import get_local_ticker_data_file_name, import_ohlc_yfinance

MUST_HAVE_DERIVATIVE_COLUMNS: Set[str] = {"tr", "tr_delta"}

class TickersData:
    def __init__(
        self,
        tickers: List[str],
        add_feature_cols_func: Callable,
        import_ohlc_func: Callable = import_ohlc_yfinance,
        recreate_columns_every_time: bool = False,
    ):
        self.tickers_data_with_features: Dict[str, pd.DataFrame] = dict()
        self.add_feature_cols_func = add_feature_cols_func
        self.import_ohlc_func = import_ohlc_func
        self.recreate_columns_every_time = recreate_columns_every_time
        for ticker in tickers:
            df = self.get_df_with_features(ticker=ticker)
            for col in MUST_HAVE_DERIVATIVE_COLUMNS:
                if col not in df.columns:
                    df = add_tr_delta_col_to_ohlc(ohlc_df=df)
            self.tickers_data_with_features[ticker] = df

    def _read_raw_data_from_xlsx(self) -> Optional[pd.DataFrame]:
        if os.path.exists(self.filename_raw) and os.path.getsize(self.filename_raw) > 0:
            df = pd.read_csv(self.filename_raw, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            df = self.add_feature_cols_func(df=df)
            if not self.recreate_columns_every_time:
                directory = os.path.dirname(self.filename_with_features) or "."
                os.makedirs(directory, exist_ok=True)
                df.to_csv(self.filename_with_features)
                print(f"Saved {self.filename_with_features} - OK")
            print(f"Reading {self.filename_raw} with datetime index - OK")
            return df
        return None

    def _import_data_from_external_provider(self, ticker: str) -> pd.DataFrame:
        print(
            f"Running {self.import_ohlc_func.__name__} for {ticker=}",
            file=sys.stderr,
        )
        df = self.import_ohlc_func(ticker=ticker)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            error_msg = f"get_df_with_features: failed call of {self.import_ohlc_func} for {ticker=}, returned {df=}"
            raise RuntimeError(error_msg)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        directory = os.path.dirname(self.filename_raw) or "."
        os.makedirs(directory, exist_ok=True)
        df.to_csv(self.filename_raw)
        print(f"Saved {self.filename_raw} - OK")
        df = self.add_feature_cols_func(df=df)
        if not self.recreate_columns_every_time:
            directory = os.path.dirname(self.filename_with_features) or "."
            os.makedirs(directory, exist_ok=True)
            df.to_csv(self.filename_with_features)
            print(f"Saved {self.filename_with_features} - OK")
        return df

    def get_df_with_features(self, ticker: str) -> pd.DataFrame:
        self.filename_with_features = get_local_ticker_data_file_name(
            ticker=ticker, data_type="with_features"
        )
        if self.recreate_columns_every_time is False:
            if (
                os.path.exists(self.filename_with_features)
                and os.path.getsize(self.filename_with_features) > 0
            ):
                df = pd.read_csv(self.filename_with_features, index_col=0, parse_dates=True)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                print(f"Reading {self.filename_with_features} with datetime index - OK")
                return df
        self.filename_raw = get_local_ticker_data_file_name(
            ticker=ticker, data_type="raw"
        )
        res = self._read_raw_data_from_xlsx()
        if res is not None:
            return res
        return self._import_data_from_external_provider(ticker=ticker)

    def get_data(self, ticker: str) -> pd.DataFrame:
        if (
            self.tickers_data_with_features
            and ticker in self.tickers_data_with_features
        ):
            return self.tickers_data_with_features[ticker]
        self.tickers_data_with_features[ticker] = self.get_df_with_features(
            ticker=ticker
        )
        return self.tickers_data_with_features[ticker]