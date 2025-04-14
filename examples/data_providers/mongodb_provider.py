"""
MongoDB Data Provider

Implementation of BaseDatabase interface for MongoDB database, used for loading and managing quantitative trading data.
"""

from datetime import datetime

from pymongo import ASCENDING, MongoClient

from apilot.core.constant import Exchange, Interval
from apilot.core.database import BarOverview, BaseDatabase, TickOverview
from apilot.core.object import BarData, TickData
from apilot.utils.logger import get_logger
from apilot.utils.symbol import split_symbol
from apilot.utils.logger import get_logger, set_level


# Get logger
logger = get_logger("MongoDB")
set_level("debug", "MongoDB")


class MongoDBDatabase(BaseDatabase):
    def __init__(
        self,
        database: str,
        collection: str,
        uri: str = "",
        **kwargs,
    ):
        """Initialize MongoDB database connection

        Args:
            database: Database name
            collection: Collection name
            uri: MongoDB connection URI
            **kwargs: Field mapping and other options, including:
                symbol_field: Symbol field name
                datetime_field: Datetime field name
                open_field: Open price field name
                high_field: High price field name
                low_field: Low price field name
                close_field: Close price field name
                volume_field: Volume field name
                open_interest_field: Open interest field name
                timestamp_ms: Whether to use millisecond timestamp format
                limit_count: Limit the number of returned records
        """
        self.uri = uri
        self.database_name = database
        self.collection_name = collection

        # Field mapping, allow overriding default values through kwargs
        self.field_map = {
            "symbol": kwargs.get("symbol_field", "symbol"),
            "datetime": kwargs.get("datetime_field", "datetime"),
            "open_price": kwargs.get("open_field", "open"),
            "high_price": kwargs.get("high_field", "high"),
            "low_price": kwargs.get("low_field", "low"),
            "close_price": kwargs.get("close_field", "close"),
            "volume": kwargs.get("volume_field", "volume"),
            "open_interest": kwargs.get("open_interest_field", "open_interest"),
        }

        # Whether timestamp is in milliseconds
        self.is_timestamp_ms = kwargs.get("timestamp_ms", False)

        # Other parameters
        self.kwargs = kwargs

        # Create MongoDB client
        self._create_client()

    def _create_client(self):
        """Create MongoDB client"""
        self.client = MongoClient(self.uri)
        self.db = self.client[self.database_name]
        self.collection = self.db[self.collection_name]

    def load_bar_data(
        self,
        symbol: str,
        interval: Interval | str,
        start: datetime,
        end: datetime,
        **kwargs,
    ) -> list[BarData]:
        """Load bar data from MongoDB"""
        if isinstance(interval, str):
            interval = Interval(interval)

        # Extract base symbol and exchange from full symbol
        base_symbol, exchange_str = split_symbol(symbol)
        exchange = Exchange(exchange_str)

        # Build query filter
        filter_dict = {}

        symbol_field = self.field_map["symbol"]
        logger.debug(f"Symbol field: {symbol_field}, base symbol: {base_symbol}")

        symbol_no_dash = base_symbol.replace("-", "")
        symbol_uppercase = symbol_no_dash.upper()

        logger.info(f"Querying symbol: {symbol_uppercase} (original: {base_symbol})")
        logger.info(f"Using hardcoded query approach from test script")

        # 1. Clear and reset filter_dict
        filter_dict = {
            "symbol": symbol_uppercase
        }

        # 2. Convert time range to millisecond timestamps
        start_timestamp = int(start.timestamp() * 1000)
        end_timestamp = int(end.timestamp() * 1000)

        # 3. Set time query condition
        filter_dict["kline_st"] = {
            "$gte": start_timestamp,
            "$lte": end_timestamp
        }

        logger.debug(f"Database: {self.database_name}, Collection: {self.collection_name}")
        logger.debug(f"Field mapping: {self.field_map}")

        datetime_field = self.field_map["datetime"]
        logger.debug(f"Datetime field: {datetime_field}")
        logger.debug(f"Query time range: {start} to {end}")

        if self.is_timestamp_ms or datetime_field == "kline_st":
            start_timestamp = int(start.timestamp() * 1000)
            end_timestamp = int(end.timestamp() * 1000)
            filter_dict[datetime_field] = {
                "$gte": start_timestamp,
                "$lte": end_timestamp,
            }
            logger.debug(f"Timestamp query range (ms): {start_timestamp} to {end_timestamp}")
        else:
            filter_dict[datetime_field] = {
                "$gte": start,
                "$lte": end,
            }

        try:
            test_doc = self.collection.find_one()
            if test_doc:
                logger.debug(f"Sample fields in collection: {list(test_doc.keys())}")
                if symbol_field in test_doc:
                    logger.debug(f"Sample symbol in collection: {test_doc[symbol_field]}")
                if datetime_field in test_doc:
                    logger.debug(f"Sample timestamp in collection: {test_doc[datetime_field]}")
        except Exception as e:
            logger.error(f"Test query failed: {e}")

        logger.debug(f"MongoDB query filter: {filter_dict}")

        limit_count = kwargs.get("limit_count", 5000)

        projection = {
            self.field_map["symbol"]: 1,
            datetime_field: 1,
            self.field_map["open_price"]: 1,
            self.field_map["high_price"]: 1,
            self.field_map["low_price"]: 1,
            self.field_map["close_price"]: 1,
            self.field_map["volume"]: 1,
            "_id": 0
        }

        if "open_interest" in self.field_map:
            projection[self.field_map["open_interest"]] = 1

        logger.info(f"Querying {symbol} data, limit: {limit_count}")
        logger.info(f"Query filter: {filter_dict}")
        logger.info(f"Projection fields: {projection}")

        try:
            cursor = self.collection.find(filter_dict, projection).sort(datetime_field, ASCENDING).limit(limit_count)
            count = self.collection.count_documents(filter_dict, limit=limit_count)
            logger.info(f"Expected number of records: {count}")

            if count == 0:
                logger.warning(f"No matching records found, trying modified query")

                datetime_filter = {datetime_field: filter_dict[datetime_field]}
                symbols_count = self.collection.count_documents(datetime_filter, limit=10)
                if symbols_count > 0:
                    sample_symbols = list(self.collection.distinct(self.field_map["symbol"], datetime_filter))[:5]
                    logger.info(f"Time-only query found {symbols_count} records, sample symbols: {sample_symbols}")

                cursor = self.collection.find(filter_dict, projection).sort(datetime_field, ASCENDING).limit(limit_count)
        except Exception as e:
            logger.error(f"MongoDB query failed: {e}")
            from pymongo.cursor import Cursor
            cursor = Cursor(self.collection, {}, limit=0)

        try:
            test_docs = list(self.collection.find(filter_dict, projection).sort(datetime_field, ASCENDING).limit(5))
            if test_docs:
                logger.info(f"Test query successful: got documents, sample: {test_docs[0]}")
                logger.debug(f"Number of test query documents: {len(test_docs)}")
            else:
                logger.warning("Test query returned no results, trying original query")
                raw_docs = list(self.collection.find({"symbol": "BTCUSDT"}).limit(2))
                if raw_docs:
                    logger.info(f"Original query successful: {raw_docs[0].keys()}")
        except Exception as e:
            logger.error(f"Test query failed: {e}")

        start_time = datetime.now()

        bars: list[BarData] = []
        processed = 0

        for doc in cursor:
            try:
                if self.is_timestamp_ms or datetime_field == "kline_st":
                    dt = datetime.fromtimestamp(doc[datetime_field] / 1000)
                else:
                    dt = doc[datetime_field]

                def safe_float(val, default=0.0):
                    if val is None:
                        return default
                    try:
                        if isinstance(val, str):
                            return float(val)
                        return float(val)
                    except (ValueError, TypeError):
                        logger.warning(f"Cannot convert to float: {val}, using default: {default}")
                        return default

                bar = BarData(
                    symbol=symbol,
                    exchange=exchange,
                    datetime=dt,
                    interval=interval,
                    open_price=safe_float(doc.get(self.field_map["open_price"], 0)),
                    high_price=safe_float(doc.get(self.field_map["high_price"], 0)),
                    low_price=safe_float(doc.get(self.field_map["low_price"], 0)),
                    close_price=safe_float(doc.get(self.field_map["close_price"], 0)),
                    volume=safe_float(doc.get(self.field_map["volume"], 0)),
                    open_interest=safe_float(doc.get(self.field_map["open_interest"], 0)),
                    gateway_name="MongoDB",
                )
                bars.append(bar)

                processed += 1
                if processed % 1000 == 0:
                    logger.info(f"Processed {processed} records...")

            except Exception as e:
                logger.error(f"Error processing document: {e}, document: {doc}")
                continue

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Loaded {len(bars)} records from MongoDB for {symbol} in {elapsed:.2f} seconds")
        return bars
