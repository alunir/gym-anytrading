from .typedefs import RewardType, Actions, Metrics

import numpy as np


class RewardCalculator:
    def __init__(
        self,
        prices,
        trade_fee_ask_percent,
        trade_fee_bid_percent,
    ):
        self._prices = prices
        self._trade_fee_ask_percent = trade_fee_ask_percent
        self._trade_fee_bid_percent = trade_fee_bid_percent
        self._returns = []
        self._pl = []
        for m in Metrics:
            setattr(self, m.name, 0.0)

    def _trade_price(self, tick):
        return self._prices[tick]

    def _update_max_dd(self, action: Actions, current_tick: int, last_trade_tick: int):
        last_trade_price = self._trade_price(last_trade_tick)
        if action == Actions.Buy:
            dd = (
                np.min(self._prices[last_trade_tick:current_tick]) / last_trade_price
                - 1.0
            )
            self.MaxDD = min(dd, self.MaxDD)
        elif action == Actions.Sell:
            dd = (
                1.0
                - np.max(self._prices[last_trade_tick:current_tick]) / last_trade_price
            )
            self.MaxDD = min(dd, self.MaxDD)
        else:
            raise ValueError("Invalid action")

    # update metrics
    def update(self, action: Actions, current_tick, last_trade_tick):
        current_price, last_trade_price = self._trade_price(
            current_tick
        ), self._trade_price(last_trade_tick)

        self._update_max_dd(action, current_tick, last_trade_tick)
        if action == Actions.Buy:
            price_diff = current_price - last_trade_price
            pl = price_diff - abs(price_diff) * self._trade_fee_bid_percent
            self._pl += [pl]
            self._returns += [pl / last_trade_price + 1.0]
            self.Profit += max(pl, 0)
            self.Loss += min(pl, 0)
            self.Trades += 1
            self.WinTrades += 1 if pl > 0 else 0
            self.LoseTrades += 1 if pl < 0 else 0
        elif action == Actions.Sell:
            price_diff = last_trade_price - current_price
            pl = price_diff - abs(price_diff) * self._trade_fee_ask_percent
            self._pl += [pl]
            self._returns += [pl / current_price + 1.0]
            self.Profit += max(pl, 0)
            self.Loss += min(pl, 0)
            self.Trades += 1
            self.WinTrades += 1 if pl > 0 else 0
            self.LoseTrades += 1 if pl < 0 else 0
        else:
            raise ValueError(f"Invalid action {action}")

    # calculate reward based on metrics
    def reward(self, reward_type: RewardType) -> float | None:
        match reward_type:
            case RewardType.Profit:
                return self.Profit
            case RewardType.Return:
                return np.prod(self._returns)
            case RewardType.LogReturn:
                return np.sum(np.log(self._returns))
            case RewardType.WinRate:
                if self.Trades == 0:
                    return 0.0
                return self.WinTrades / (self.Trades)
            case RewardType.ProfitPerTrade:
                if self.Trades == 0:
                    return 0.0
                return self.Profit / (self.Trades)
            case RewardType.ProfitFactor:
                if self.LoseTrades == 0:
                    return 0.0
                return -self.Profit / (self.Loss)
            case RewardType.PesimisticProfitFactor:
                if self.WinTrades == 0 or self.LoseTrades == 0:
                    return 0.0
                return -(
                    (self.WinTrades - np.sqrt(self.WinTrades))
                    * (self.Profit / (self.WinTrades))
                ) / (
                    (self.LoseTrades + np.sqrt(self.LoseTrades))
                    * self.Loss
                    / (self.LoseTrades)
                )
            case RewardType.KellyCriterion:
                if self.WinTrades == 0 or self.LoseTrades == 0:
                    return 0.0
                return self.WinTrades / (self.Trades) - (
                    1 - self.WinTrades / (self.Trades)
                ) / (self.Profit / (self.WinTrades)) / ((self.Loss / (self.LoseTrades)))
            case RewardType.GHPR:
                if self.Trades == 0:
                    return 0.0
                return np.power(
                    np.prod(self._returns),
                    1 / (self.Trades),
                )
            case RewardType.AHPR:
                return np.mean(self._returns)
            case RewardType.RoMaD:
                if len([pl for pl in self._pl if pl < 0]) == 0:
                    return -1e10
                return -np.sum(self._pl) / (min([pl for pl in self._pl if pl < 0]))
            case RewardType.SQN:
                if np.std(self._pl) == 0.0:
                    return -1e10
                return np.sqrt(self.Trades) * np.mean(self._pl) / np.std(self._pl)
            # case RewardType.RecoveryFactor:
            #     return -np.prod(self._returns) / (
            #         self.MaxDD] or np.nan
            #     )
            # case RewardType.SharpeRatio:
            #     return np.prod([r for r in self._returns if r > 1.0]) / (
            #         np.std(self._returns) or np.nan
            #     )
            # case RewardType.SortinoRatio:
            #     if np.std([r for r in self._returns if r < 1.0]) == 0.0:
            #         return np.nan
            #     return np.prod([r for r in self._returns if r > 1.0]) / np.std(
            #         [r for r in self._returns if r < 1.0]
            #     )
            case _:
                raise NotImplementedError

    def get_info(self):
        return {m.name: getattr(self, m.name) for m in Metrics} | {
            rt.name: self.reward(rt) for rt in RewardType
        }
