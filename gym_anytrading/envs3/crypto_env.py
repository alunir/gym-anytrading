import numpy as np

from .typedefs import Actions, Positions, RewardType
from .trading_env import TradingEnv


def sigmoid(a: float) -> float:
    return 1.0 / (1.0 + np.exp(-a))


class CryptoEnv(TradingEnv):

    def __init__(
        self,
        df,
        window_size,
        frame_bound,
        trade_fee=0.0003,
        leverage: float = 1.0,
        render_mode=None,
        reward_type: RewardType = RewardType.Profit,
    ):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        self.leverage = leverage
        self.reward_type = reward_type
        super().__init__(df, window_size, render_mode)

        self.trade_fee_bid_percent = trade_fee  # unit
        self.trade_fee_ask_percent = trade_fee  # unit

    def _process_data(self):
        prices = self.df.loc[:, "Close"].to_numpy()
        signal_features = self.df.to_numpy()

        return prices.astype(np.float32), signal_features.astype(np.float32)

    def _calculate_reward(self, action):
        step_reward = 0

        if (self._position == Positions.Long and action != Actions.Buy.value) or (
            self._position == Positions.Short and action != Actions.Sell.value
        ):
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Short:
                ret = (last_trade_price / current_price - 1.0) * self.leverage
                log_ret = (
                    np.log(last_trade_price) - np.log(current_price)
                ) * self.leverage
                price_diff = last_trade_price - current_price
                profit = price_diff - abs(price_diff) * self.trade_fee_ask_percent
                # TODO: consider High/Low instead.
                dd = (
                    1.0
                    - np.max(self.prices[self._last_trade_tick : self._current_tick])
                    / last_trade_price
                )
            elif self._position == Positions.Long:
                ret = (current_price / last_trade_price - 1.0) * self.leverage
                log_ret = (
                    np.log(current_price) - np.log(last_trade_price)
                ) * self.leverage
                price_diff = (current_price - last_trade_price) * self.leverage
                profit = price_diff - abs(price_diff) * self.trade_fee_bid_percent
                dd = (
                    np.min(self.prices[self._last_trade_tick : self._current_tick])
                    / last_trade_price
                    - 1.0
                )

            if self.reward_type == RewardType.Profit:
                step_reward = profit
            elif self.reward_type == RewardType.Return:
                step_reward = ret
            elif self.reward_type == RewardType.LogReturn:
                step_reward = log_ret
            elif self.reward_type == RewardType.RoMaD:
                step_reward = (
                    np.sign(ret) if dd == 0.0 else 2 * sigmoid(ret / abs(dd)) - 1
                )
            else:
                raise NotImplementedError

            self._max_dd = np.max([abs(dd), self._max_dd])
            self._epoch = [self.df.index[self._current_tick]]

        return step_reward

    def _update_profit(self, action):
        trade = False

        if (self._position == Positions.Long and action != Actions.Buy.value) or (
            self._position == Positions.Short and action != Actions.Sell.value
        ):
            trade = True

        if trade or self._truncated:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Long:
                price_diff = (current_price - last_trade_price) * self.leverage
                self._total_profit += (
                    price_diff - abs(price_diff) * self.trade_fee_bid_percent
                )
            elif self._position == Positions.Short:
                price_diff = (last_trade_price - current_price) * self.leverage
                self._total_profit += (
                    price_diff - abs(price_diff) * self.trade_fee_ask_percent
                )

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.0

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (
                    current_tick <= self._end_tick
                    and self.prices[current_tick] < self.prices[current_tick - 1]
                ):
                    current_tick += 1
                position = Positions.Short
            else:
                while (
                    current_tick <= self._end_tick
                    and self.prices[current_tick] >= self.prices[current_tick - 1]
                ):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Long:
                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                shares = profit / last_trade_price
                profit = shares * current_price
            last_trade_tick = current_tick - 1

        return profit
