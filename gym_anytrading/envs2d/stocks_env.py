from typing import Tuple
import numpy as np

from .trading_env import TradingEnv, Actions, Positions, RewardType

INF = 1e10


class StocksEnv(TradingEnv):

    def __init__(
        self,
        prices,
        ask,
        bid,
        df,
        window_size,
        frame_bound,
        render_mode=None,
        reward_type=RewardType.Profit,
        trade_fee_ask_percent=0.005,
        trade_fee_bid_percent=0.01,
        box_range: Tuple[float, float] = (-INF, INF),
    ):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(
            prices,
            ask,
            bid,
            df,
            window_size,
            render_mode,
            reward_type,
            trade_fee_ask_percent,
            trade_fee_bid_percent,
            box_range=box_range,
        )

    def _process_data(self):
        prices = self.df.loc[:, "Close"].to_numpy()

        prices[
            self.frame_bound[0] - self.window_size
        ]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0] - self.window_size : self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices.astype(np.float32), signal_features.astype(np.float32)

    def _calculate_reward(self, action):
        step_reward = 0

        trade = False
        if (action == Actions.Buy.value and self._position == Positions.Short) or (
            action == Actions.Sell.value and self._position == Positions.Long
        ):
            trade = True

        if trade:

            # calculate metrics
            self._reward_calculator.update(
                self._position, self._current_tick, self._last_trade_tick
            )

            # calculate reward
            step_reward = self._reward_calculator.reward(self._reward_type)

            # current_price = self.prices[self._current_tick]
            # last_trade_price = self.prices[self._last_trade_tick]
            # price_diff = current_price - last_trade_price

            # if self._position == Positions.Long:
            #     step_reward += price_diff

        return step_reward

    # def _update_profit(self, action):
    #     trade = False
    #     if (action == Actions.Buy.value and self._position == Positions.Short) or (
    #         action == Actions.Sell.value and self._position == Positions.Long
    #     ):
    #         trade = True

    #     if trade or self._truncated:
    #         current_price = self.prices[self._current_tick]
    #         last_trade_price = self.prices[self._last_trade_tick]

    #         if self._position == Positions.Long:
    #             shares = (
    #                 self._total_profit * (1 - self.trade_fee_ask_percent)
    #             ) / last_trade_price
    #             self._total_profit = (
    #                 shares * (1 - self.trade_fee_bid_percent)
    #             ) * current_price

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
