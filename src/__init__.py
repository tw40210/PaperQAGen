import datetime
import logging
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__[: __file__.find("src")])
START_TIMESTAMP = datetime.datetime(2000, 1, 1)
HISTORICAL_DATA_FEATURES = [
    "timestamp_int",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
]
REAL_TIME_DATA_FEATURES = ["price"]


def datetime_resolver(time):
    datetime_format = "%Y-%m-%d %H:%M:%S"
    return datetime.datetime.strptime(time, datetime_format)


OmegaConf.register_new_resolver("time", datetime_resolver)

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
