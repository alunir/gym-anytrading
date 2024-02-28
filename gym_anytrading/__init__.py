from gymnasium.envs.registration import register
from copy import deepcopy

from . import datasets
from .envs3.typedefs import RewardType


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
    entry_point="gym_anytrading.envs3:CryptoEnv",
    kwargs={
        "df": deepcopy(datasets.CRYPTO_ETHUSDT_5M),
        "window_size": 24,
        "frame_bound": (24, len(datasets.CRYPTO_ETHUSDT_5M)),
        "trade_fee": 0.0,
        "reward_type": RewardType.Profit,
    },
)
