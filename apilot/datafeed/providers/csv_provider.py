"""
CSV file data provider

Implements data storage based on CSV files, supports bar data loading.
"""

from datetime import datetime

import pandas as pd

from apilot.core import BarData, Interval
from apilot.core.database import BaseDatabase
from apilot.utils.logger import get_logger, set_level

logger = get_logger("csv_provider")
set_level("debug", "csv_provider")


class CsvDatabase(BaseDatabase):
    def __init__(self, filepath=None, **kwargs):
        self.csv_path = filepath

        self.datetime_index = kwargs.get("datetime_index", 0)
        self.open_index = kwargs.get("open_index", 1)
        self.high_index = kwargs.get("high_index", 2)
        self.low_index = kwargs.get("low_index", 3)
        self.close_index = kwargs.get("close_index", 4)
        self.volume_index = kwargs.get("volume_index", 5)
        self.openinterest_index = kwargs.get("openinterest_index", -1)

        self.dtformat = kwargs.get("dtformat", "%Y-%m-%d %H:%M:%S")

    def load_bar_data(
        self,
        symbol: str,
        interval: Interval,
        start: datetime,
        end: datetime,
    ) -> list[BarData]:
        try:
            df = pd.read_csv(self.csv_path)
            logger.debug(f"CSV{symbol} loaded, interval {interval}, rows: {len(df)}")

            if self.datetime_index >= 0:
                logger.debug(f"Converting datetime field format {self.dtformat}")
                df["datetime"] = pd.to_datetime(
                    df.iloc[:, self.datetime_index], format=self.dtformat
                )

            df = df[(df["datetime"] >= start) & (df["datetime"] <= end)]
            logger.debug(f"Filtered data rows: {len(df)}")

            bars = []

            # Convert to BarData objects
            for _, row in df.iterrows():
                bar = BarData(
                    symbol=symbol,
                    exchange="LOCAL",  # TODO: Can this be removed?
                    datetime=row["datetime"],
                    interval=interval,
                    volume=float(row.iloc[self.volume_index])
                    if self.volume_index >= 0
                    else 0,
                    open_price=float(row.iloc[self.open_index])
                    if self.open_index >= 0
                    else 0,
                    high_price=float(row.iloc[self.high_index])
                    if self.high_index >= 0
                    else 0,
                    low_price=float(row.iloc[self.low_index])
                    if self.low_index >= 0
                    else 0,
                    close_price=float(row.iloc[self.close_index])
                    if self.close_index >= 0
                    else 0,
                    open_interest=float(row.iloc[self.openinterest_index])
                    if self.openinterest_index >= 0
                    else 0,
                    gateway_name="CSV",
                )
                bars.append(bar)

            logger.debug(f"Successfully created {len(bars)} Bar objects")
            return bars

        except Exception as e:
            logger.error(f"CSV data loading failed: {e}")
            return []

    def delete_bar_data(self, symbol: str, interval: Interval) -> int:
        """Delete bar data (not implemented)"""
        return 0

    def get_bar_overview(self) -> list[dict]:
        """Get bar data overview (not implemented)"""
        return []
