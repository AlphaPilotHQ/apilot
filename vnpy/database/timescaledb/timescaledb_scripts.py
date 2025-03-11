# TimescaleDB读写相关语句
CREATE_BAR_TABLE_SCRIPT = """
CREATE TABLE IF NOT EXISTS bar_data(
   symbol VARCHAR(100) NOT NULL,
   exchange VARCHAR(100) NOT NULL,
   interval VARCHAR(100) NOT NULL,
   datetime TIMESTAMPTZ,
   volume FLOAT NOT NULL DEFAULT 0.0,
   turnover FLOAT NOT NULL DEFAULT 0.0,
   open_interest FLOAT NOT NULL DEFAULT 0.0,
   open_price FLOAT NOT NULL DEFAULT 0.0,
   high_price FLOAT NOT NULL DEFAULT 0.0,
   low_price FLOAT NOT NULL DEFAULT 0.0,
   close_price FLOAT NOT NULL DEFAULT 0.0,
   PRIMARY KEY ( symbol, exchange, interval, datetime )
);
"""

CREATE_BAR_HYPERTABLE_SCRIPT = "SELECT create_hypertable('bar_data', 'datetime', if_not_exists => TRUE);"

CREATE_TICK_TABLE_SCRIPT = """
CREATE TABLE IF NOT EXISTS tick_data(
   symbol VARCHAR(100) NOT NULL,
   exchange VARCHAR(100) NOT NULL,
   datetime TIMESTAMPTZ,
   name VARCHAR(100) NOT NULL,
   volume FLOAT NOT NULL DEFAULT 0.0,
   turnover FLOAT NOT NULL DEFAULT 0.0,
   open_interest FLOAT NOT NULL DEFAULT 0.0,
   last_price FLOAT NOT NULL DEFAULT 0.0,
   last_volume FLOAT NOT NULL DEFAULT 0.0,
   limit_up FLOAT NOT NULL DEFAULT 0.0,
   limit_down FLOAT NOT NULL DEFAULT 0.0,
   open_price FLOAT NOT NULL DEFAULT 0.0,
   high_price FLOAT NOT NULL DEFAULT 0.0,
   low_price FLOAT NOT NULL DEFAULT 0.0,
   pre_close FLOAT NOT NULL DEFAULT 0.0,
   bid_price_1 FLOAT NOT NULL DEFAULT 0.0,
   bid_price_2 FLOAT NOT NULL DEFAULT 0.0,
   bid_price_3 FLOAT NOT NULL DEFAULT 0.0,
   bid_price_4 FLOAT NOT NULL DEFAULT 0.0,
   bid_price_5 FLOAT NOT NULL DEFAULT 0.0,
   ask_price_1 FLOAT NOT NULL DEFAULT 0.0,
   ask_price_2 FLOAT NOT NULL DEFAULT 0.0,
   ask_price_3 FLOAT NOT NULL DEFAULT 0.0,
   ask_price_4 FLOAT NOT NULL DEFAULT 0.0,
   ask_price_5 FLOAT NOT NULL DEFAULT 0.0,
   bid_volume_1 FLOAT NOT NULL DEFAULT 0.0,
   bid_volume_2 FLOAT NOT NULL DEFAULT 0.0,
   bid_volume_3 FLOAT NOT NULL DEFAULT 0.0,
   bid_volume_4 FLOAT NOT NULL DEFAULT 0.0,
   bid_volume_5 FLOAT NOT NULL DEFAULT 0.0,
   ask_volume_1 FLOAT NOT NULL DEFAULT 0.0,
   ask_volume_2 FLOAT NOT NULL DEFAULT 0.0,
   ask_volume_3 FLOAT NOT NULL DEFAULT 0.0,
   ask_volume_4 FLOAT NOT NULL DEFAULT 0.0,
   ask_volume_5 FLOAT NOT NULL DEFAULT 0.0,
   localt TIMESTAMPTZ,
   PRIMARY KEY ( symbol, exchange, datetime )
);
"""

CREATE_TICK_HYPERTABLE_SCRIPT = "SELECT create_hypertable('tick_data', 'datetime', if_not_exists => TRUE);"

CREATE_BAR_OVERVIEW_TABLE_SCRIPT = """
CREATE TABLE IF NOT EXISTS bar_overview
(
   symbol VARCHAR(100) NOT NULL,
   exchange VARCHAR(100) NOT NULL,
   interval VARCHAR(100) NOT NULL,
   count INT,
   starttime TIMESTAMPTZ,
   endtime TIMESTAMPTZ,
   PRIMARY KEY ( symbol, exchange, interval )
);
"""

CREATE_TICK_OVERVIEW_TABLE_SCRIPT = """
CREATE TABLE IF NOT EXISTS tick_overview
(
   symbol VARCHAR(100) NOT NULL,
   exchange VARCHAR(100) NOT NULL,
   count INT,
   starttime TIMESTAMPTZ,
   endtime TIMESTAMPTZ,
   PRIMARY KEY ( symbol, exchange )
);
"""

SAVE_BAR_QUERY = """
INSERT INTO bar_data
(symbol, exchange, interval, datetime, volume,
turnover,open_interest, open_price, high_price,
low_price, close_price
)
VALUES
(%(symbol)s, %(exchange)s, %(interval)s, %(datetime)s,
%(volume)s, %(turnover)s, %(open_interest)s, %(open_price)s,
%(high_price)s, %(low_price)s, %(close_price)s
)
ON CONFLICT (symbol, exchange, interval, datetime) DO UPDATE
SET volume = excluded.volume,
turnover = excluded.turnover,
open_interest = excluded.open_interest,
open_price = excluded.open_price,
high_price = excluded.high_price,
low_price = excluded.low_price,
close_price = excluded.close_price;
"""

SAVE_TICK_QUERY = """
INSERT INTO tick_data
(symbol, exchange, datetime, name, volume,
turnover,open_interest, last_price, last_volume,
limit_up, limit_down, open_price, high_price,
low_price, pre_close,
bid_price_1, bid_price_2, bid_price_3, bid_price_4, bid_price_5,
ask_price_1, ask_price_2, ask_price_3, ask_price_4, ask_price_5,
bid_volume_1, bid_volume_2, bid_volume_3, bid_volume_4, bid_volume_5,
ask_volume_1, ask_volume_2, ask_volume_3, ask_volume_4, ask_volume_5,
localt
)
VALUES
(%(symbol)s, %(exchange)s, %(datetime)s, %(name)s, %(volume)s,
%(turnover)s, %(open_interest)s, %(last_price)s, %(last_volume)s,
%(limit_up)s, %(limit_down)s, %(open_price)s, %(high_price)s,
%(low_price)s, %(pre_close)s,
%(bid_price_1)s, %(bid_price_2)s, %(bid_price_3)s, %(bid_price_4)s, %(bid_price_5)s,
%(ask_price_1)s, %(ask_price_2)s, %(ask_price_3)s, %(ask_price_4)s, %(ask_price_5)s,
%(bid_volume_1)s, %(bid_volume_2)s, %(bid_volume_3)s, %(bid_volume_4)s, %(bid_volume_5)s,
%(ask_volume_1)s, %(ask_volume_2)s, %(ask_volume_3)s, %(ask_volume_4)s, %(ask_volume_5)s,
%(localt)s
)
ON CONFLICT (symbol, exchange, datetime) DO UPDATE
SET name = excluded.name,
volume = excluded.volume,
turnover = excluded.turnover,
open_interest = excluded.open_interest,
last_price = excluded.last_price,
last_volume = excluded.last_volume,
limit_up = excluded.limit_up,
limit_down = excluded.limit_down,
open_price = excluded.open_price,
high_price = excluded.high_price,
low_price = excluded.low_price,
pre_close = excluded.pre_close,
bid_price_1 = excluded.bid_price_1,
bid_volume_1 = excluded.bid_volume_1,
bid_price_2 = excluded.bid_price_2,
bid_volume_2 = excluded.bid_volume_2,
bid_price_3 = excluded.bid_price_3,
bid_volume_3 = excluded.bid_volume_3,
bid_price_4 = excluded.bid_price_4,
bid_volume_4 = excluded.bid_volume_4,
bid_price_5 = excluded.bid_price_5,
bid_volume_5 = excluded.bid_volume_5,
ask_price_1 = excluded.ask_price_1,
ask_volume_1 = excluded.ask_volume_1,
ask_price_2 = excluded.ask_price_2,
ask_volume_2 = excluded.ask_volume_2,
ask_price_3 = excluded.ask_price_3,
ask_volume_3 = excluded.ask_volume_3,
ask_price_4 = excluded.ask_price_4,
ask_volume_4 = excluded.ask_volume_4,
ask_price_5 = excluded.ask_price_5,
ask_volume_5 = excluded.ask_volume_5,
localt = excluded.localt;
"""

SAVE_BAR_OVERVIEW_QUERY = """
INSERT INTO bar_overview
(symbol, exchange, interval, count, starttime, endtime)
VALUES
(%(symbol)s, %(exchange)s, %(interval)s, %(count)s, %(starttime)s, %(endtime)s)
ON CONFLICT (symbol, exchange, interval) DO UPDATE
SET count = excluded.count,
starttime = excluded.starttime,
endtime = excluded.endtime;
"""

SAVE_TICK_OVERVIEW_QUERY = """
INSERT INTO tick_overview
(symbol, exchange, count, starttime, endtime)
VALUES
(%(symbol)s, %(exchange)s, %(count)s, %(starttime)s, %(endtime)s)
ON CONFLICT (symbol, exchange) DO UPDATE
SET count = excluded.count,
starttime = excluded.starttime,
endtime = excluded.endtime;
"""

LOAD_BAR_QUERY = """
SELECT * FROM bar_data
WHERE symbol = %(symbol)s
AND exchange = %(exchange)s
AND interval = %(interval)s
AND datetime >= %(start)s
AND datetime <= %(end)s
ORDER BY datetime ASC;
"""

LOAD_TICK_QUERY = """
SELECT * FROM tick_data
WHERE symbol = %(symbol)s
AND exchange = %(exchange)s
AND datetime >= %(start)s
AND datetime <= %(end)s
ORDER BY datetime ASC;
"""

LOAD_BAR_OVERVIEW_QUERY = """
SELECT * FROM bar_overview
WHERE symbol = %(symbol)s
AND exchange = %(exchange)s
AND interval = %(interval)s
"""

LOAD_TICK_OVERVIEW_QUERY = """
SELECT * FROM tick_overview
WHERE symbol = %(symbol)s
AND exchange = %(exchange)s
"""

COUNT_BAR_QUERY = """
SELECT COUNT(close_price) FROM bar_data FINAL
WHERE symbol = %(symbol)s
AND exchange = %(exchange)s
AND interval = %(interval)s
"""

COUNT_TICK_QUERY = """
SELECT COUNT(last_price) FROM tick_data FINAL
WHERE symbol = %(symbol)s
AND exchange = %(exchange)s
"""

LOAD_ALL_BAR_OVERVIEW_QUERY = "SELECT * FROM bar_overview"

LOAD_ALL_TICK_OVERVIEW_QUERY = "SELECT * FROM tick_overview"

DELETE_BAR_QUERY = """
DELETE FROM bar_data
WHERE symbol = %(symbol)s
AND exchange = %(exchange)s
AND interval = %(interval)s
"""

DELETE_TICK_QUERY = """
DELETE FROM tick_data
WHERE symbol = %(symbol)s
AND exchange = %(exchange)s
"""

DELETE_BAR_OVERVIEW_QUERY = """
DELETE FROM bar_overview
WHERE symbol = %(symbol)s
AND exchange = %(exchange)s
AND interval = %(interval)s
"""

DELETE_TICK_OVERVIEW_QUERY = """
DELETE FROM tick_overview
WHERE symbol = %(symbol)s
AND exchange = %(exchange)s
"""
