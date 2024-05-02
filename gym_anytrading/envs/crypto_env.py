from typing import Tuple

import numpy as np
import pandas as pd

from .stocks_env import StocksEnv
from ..typedefs import Actions, Positions, RewardType

INF = 1e10


class CryptoEnv(StocksEnv):

    def __init__(
        self,
        prices: pd.Series,
        ask: pd.Series,
        bid: pd.Series,
        df: pd.DataFrame,
        window_size: int,
        frame_bound,
        trade_fee=0.0003,
        leverage: float = 1.0,
        render_mode=None,
        reward_type=RewardType.LogReturns,
        box_range: Tuple[float, float] = (-INF, INF),
    ):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        self.leverage = leverage  # Forex: 10000 [Unit]. Crypto: leverage etc...
        super().__init__(
            prices,
            ask,
            bid,
            df,
            window_size,
            frame_bound,
            render_mode,
            reward_type,
            trade_fee_ask_percent=trade_fee,
            trade_fee_bid_percent=trade_fee,
            box_range=box_range,
        )

    def _process_data(self):
        prices = self.prices.values

        # signal_features = self.signal_features.T
        signal_features = self.df.values

        return prices.astype(np.float32), signal_features.astype(np.float32)

    def _calculate_reward(self, action):
        step_reward = 0.0  # pip

        if (action != Actions.Sell.value and self._position == Positions.Short) or (
            action != Actions.Buy.value and self._position == Positions.Long
        ):
            current_reward = self._reward_calculator.reward(self._reward_type)

            # calculate metrics
            self._reward_calculator.update(
                self._position, self._current_tick, self._last_trade_tick
            )

            # calculate reward
            updated_reward = self._reward_calculator.reward(self._reward_type)

            step_reward = updated_reward - current_reward

            self._epoch = self.df.index[self._current_tick]

        return step_reward
