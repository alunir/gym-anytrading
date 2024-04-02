from gymnasium.envs.registration import register
from copy import deepcopy

from . import datasets
from .typedefs import Actions, Positions, RewardType

register(
    id="forex-v0",
    entry_point="gym_anytrading.envs:ForexEnv",
    kwargs={
        "df": deepcopy(datasets.FOREX_EURUSD_1H_ASK),
        "window_size": 24,
        "frame_bound": (24, len(datasets.FOREX_EURUSD_1H_ASK)),
    },
)

register(
    id="stocks-v0",
    entry_point="gym_anytrading.envs:StocksEnv",
    kwargs={
        "df": deepcopy(datasets.STOCKS_GOOGL),
        "window_size": 30,
        "frame_bound": (30, len(datasets.STOCKS_GOOGL)),
    },
)

register(
    id="crypto-v0",
    entry_point="gym_anytrading.envs:CryptoEnv",
    kwargs={
        "df": deepcopy(datasets.CRYPTO_ETHUSDT_5M),
        "ask": deepcopy(datasets.CRYPTO_ETHUSDT_5M).High,
        "bid": deepcopy(datasets.CRYPTO_ETHUSDT_5M).Low,
        "prices": deepcopy(datasets.CRYPTO_ETHUSDT_5M).Close,
        "window_size": 24,
        "frame_bound": (24, len(datasets.CRYPTO_ETHUSDT_5M)),
        "trade_fee": 0.0003,
    },
)

register(
    id="crypto-v1",
    entry_point="gym_anytrading.envs2d:CryptoEnv",
    kwargs={
        "df": deepcopy(datasets.CRYPTO_ETHUSDT_5M),
        "ask": deepcopy(datasets.CRYPTO_ETHUSDT_5M).High,
        "bid": deepcopy(datasets.CRYPTO_ETHUSDT_5M).Low,
        "prices": deepcopy(datasets.CRYPTO_ETHUSDT_5M).Close,
        "window_size": 32,
        "frame_bound": (32, len(datasets.CRYPTO_ETHUSDT_5M)),
        "trade_fee": 0.0003,
    },
)
