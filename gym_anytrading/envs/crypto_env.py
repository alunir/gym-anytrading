from typing import Tuple

import numpy as np
import pandas as pd

from .stocks_env import StocksEnv
from ..typedefs import Actions, Positions, RewardType, OrderAction, Position

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

    def _calculate_reward(self, action: Actions):
        step_reward = 0.0  # pip

        order_action = OrderAction(-2 * action + 1)
        position_value = Position(-2 * self._position.value + 1)

        if (order_action >= 0.0 and position_value < 0.0) or (
            order_action <= 0.0 and position_value > 0.0
        ):
            current_reward = self._reward_calculator.reward(self._reward_type)

            # calculate metrics
            self._reward_calculator.update(
                position_value, order_action, self._current_tick, self._last_trade_tick
            )

            # calculate reward
            updated_reward = self._reward_calculator.reward(self._reward_type)

            step_reward = updated_reward - current_reward

            self._epoch = self.df.index[self._current_tick]

        return step_reward
