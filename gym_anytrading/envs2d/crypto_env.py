import numpy as np
import pandas as pd
from tqdm import tqdm

from typing import Tuple

from .stocks_env import StocksEnv
from ..typedefs import Actions, Positions, RewardType

from pyts.image import RecurrencePlot

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
        reward_type=RewardType.Profit,
        box_range: Tuple[float, float] = (-INF, INF),
    ):
        assert len(frame_bound) == 2

        # self.signal_features = signal_features

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
        prices = self.prices.values[self.window_size :]

        batches = [
            [*self.df.index[i - self.window_size : i]]
            for i in range(self.window_size, len(self.df))
        ]

        transformer = RecurrencePlot()

        signal_features = []
        for batch in tqdm(batches):
            signal_features += [transformer.fit_transform(self.df.loc[batch].T)]

        signal_features = np.array(signal_features).reshape(
            len(batches), self.window_size, self.window_size, -1
        )
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

    def _get_observation(self):
        return self.signal_features[self._current_tick]
