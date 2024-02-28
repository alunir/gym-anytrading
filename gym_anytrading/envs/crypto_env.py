import numpy as np
import pandas as pd

from .stocks_env import StocksEnv
from ..typedefs import Actions, Positions, RewardType


class CryptoEnv(StocksEnv):

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int,
        frame_bound,
        trade_fee=0.0003,
        leverage: float = 1.0,
        render_mode=None,
        reward_type=RewardType.Profit,
    ):
        assert len(frame_bound) == 2

        # self.signal_features = signal_features

        self.frame_bound = frame_bound
        self.leverage = leverage  # Forex: 10000 [Unit]. Crypto: leverage etc...
        super().__init__(df, window_size, frame_bound, render_mode)

        self.reward_type = reward_type
        self._epoch = None
        self._max_dd = -1e10
        self._total_return = 0.0

        self.trade_fee_bid_percent = trade_fee  # unit
        self.trade_fee_ask_percent = trade_fee  # unit

    def _get_info(self):
        return dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            position=self._position,
            max_dd=self._max_dd,
            total_return=self._total_return,
            epoch=self._epoch,
        )

    def _process_data(self):
        prices = self.df.loc[:, "Close"].to_numpy()
        # signal_features = self.signal_features.T
        signal_features = self.df.to_numpy()

        return prices.astype(np.float32), signal_features.astype(np.float32)

    def _calculate_reward(self, action):
        step_reward = 0.0  # pip

        if (action != Actions.Sell.value and self._position == Positions.Short) or (
            action != Actions.Buy.value and self._position == Positions.Long
        ):
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            # calculate metrics
            if self._position == Positions.Short:
                price_diff = last_trade_price - current_price
                profit = price_diff - abs(price_diff) * self.trade_fee_ask_percent
                log_ret = (
                    np.log(last_trade_price) - np.log(current_price)
                ) * self.leverage
                dd = (
                    1.0
                    - np.max(self.prices[self._last_trade_tick : self._current_tick])
                    / last_trade_price
                )
            elif self._position == Positions.Long:
                price_diff = current_price - last_trade_price
                profit = price_diff - abs(price_diff) * self.trade_fee_bid_percent
                log_ret = (
                    np.log(current_price) - np.log(last_trade_price)
                ) * self.leverage
                dd = (
                    np.min(self.prices[self._last_trade_tick : self._current_tick])
                    / last_trade_price
                    - 1.0
                )

            # calculate reward
            if self.reward_type == RewardType.Profit:
                step_reward += profit
            elif self.reward_type == RewardType.LogReturn:
                step_reward += log_ret
            elif self.reward_type == RewardType.MaxDD:
                step_reward += np.max([dd - self._max_dd, 0])  # add only diff??
            else:
                raise NotImplementedError

            self._max_dd = np.max([dd, self._max_dd])
            self._total_return += log_ret
            self._epoch = self.df.index[self._current_tick]

        return step_reward
