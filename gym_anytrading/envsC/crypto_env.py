from typing import Tuple

import numpy as np
import pandas as pd

from ..typedefs import OrderAction, Position, RewardType
from .trading_env import TradingEnv

INF = 1e10


class CryptoEnv(TradingEnv):

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
        self.leverage = leverage

        super().__init__(
            prices,
            ask,
            bid,
            df,
            window_size,
            render_mode=render_mode,
            reward_type=reward_type,
            trade_fee_ask_percent=trade_fee,
            trade_fee_bid_percent=trade_fee,
            box_range=box_range,
        )

    def _process_data(self):
        prices = self.prices.values

        # prices[
        #     self.frame_bound[0] - self.window_size
        # ]  # validate index (TODO: Improve validation)
        # prices = prices[self.frame_bound[0] - self.window_size : self.frame_bound[1]]

        # signal_features = self.signal_features.T
        signal_features = self.df.values

        return prices.astype(np.float32), signal_features.astype(np.float32)

    def _calculate_reward(self, action: OrderAction):
        step_reward = 0

        if action == 0.0:
            return step_reward

        if self._position < 0.0 and action < 0.0:  # (Position.Short, Action.Sell)
            # increase short position
            return step_reward
        elif self._position > 0.0 and action > 0.0:  # (Position.Long, Action.Buy)
            # increase long position
            return step_reward

        assert (
            abs(self._position + action) <= 1.0
        ), f"position({self._position}) + action({action}) should be between -1 and 1."

        current_reward = self._reward_calculator.reward(self._reward_type)

        # calculate metrics
        self._reward_calculator.update(
            self._position,
            action,
            self._current_tick,
            self._last_trade_tick,
        )

        # calculate reward
        updated_reward = self._reward_calculator.reward(self._reward_type)

        step_reward = updated_reward - current_reward

        self._epoch = self.df.index[self._current_tick]

        return step_reward
