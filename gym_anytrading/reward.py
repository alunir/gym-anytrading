from .typedefs import RewardType, Actions, Metrics

import numpy as np


class RewardCalculator:
    def __init__(
        self,
        prices,
        trade_fee_ask_percent=0.0003,
        trade_fee_bid_percent=0.0003,
    ):
        self._prices = prices
        self._trade_fee_ask_percent = trade_fee_ask_percent
        self._trade_fee_bid_percent = trade_fee_bid_percent
        self._metrics = {m: 0.0 for m in Metrics}
        self._returns = []
        self._pl = []

    def _trade_price(self, tick):
        return self._prices[tick]

    def _update_max_dd(self, action: Actions, current_tick: int, last_trade_tick: int):
        last_trade_price = self._trade_price(last_trade_tick)
        if action == Actions.Buy:
            dd = (
                np.min(self._prices[last_trade_tick:current_tick]) / last_trade_price
                - 1.0
            )
            self._metrics[Metrics.MaxDD] = min(dd, self._metrics[Metrics.MaxDD])
        elif action == Actions.Sell:
            dd = (
                1.0
                - np.max(self._prices[last_trade_tick:current_tick]) / last_trade_price
            )
            self._metrics[Metrics.MaxDD] = min(dd, self._metrics[Metrics.MaxDD])
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
            self._metrics[Metrics.Profit] += max(pl, 0)
            self._metrics[Metrics.Loss] += min(pl, 0)
            self._metrics[Metrics.Trades] += 1
            self._metrics[Metrics.WinTrades] += 1 if pl > 0 else 0
            self._metrics[Metrics.LoseTrades] += 1 if pl < 0 else 0
        elif action == Actions.Sell:
            price_diff = last_trade_price - current_price
            pl = price_diff - abs(price_diff) * self._trade_fee_ask_percent
            self._pl += [pl]
            self._returns += [pl / current_price + 1.0]
            self._metrics[Metrics.Profit] += max(pl, 0)
            self._metrics[Metrics.Loss] += min(pl, 0)
            self._metrics[Metrics.Trades] += 1
            self._metrics[Metrics.WinTrades] += 1 if pl > 0 else 0
            self._metrics[Metrics.LoseTrades] += 1 if pl < 0 else 0
        else:
            raise ValueError(f"Invalid action {action}")

    # calculate reward based on metrics
    def reward(self, reward_type: RewardType) -> float | None:
        match reward_type:
            case RewardType.Profit:
                return self._metrics[Metrics.Profit]
            case RewardType.Return:
                return np.prod(self._returns)
            case RewardType.LogReturn:
                return np.sum(np.log(self._returns))
            case RewardType.WinRate:
                return self._metrics[Metrics.WinTrades] / (
                    self._metrics[Metrics.Trades] or np.nan
                )
            case RewardType.ProfitPerTrade:
                return self._metrics[Metrics.Profit] / (
                    self._metrics[Metrics.Trades] or np.nan
                )
            case RewardType.ProfitFactor:
                return -self._metrics[Metrics.Profit] / (
                    self._metrics[Metrics.Loss] or np.nan
                )
            case RewardType.PesimisticProfitFactor:
                return -(
                    (
                        self._metrics[Metrics.WinTrades]
                        - np.sqrt(self._metrics[Metrics.WinTrades])
                    )
                    * (
                        self._metrics[Metrics.Profit]
                        / (self._metrics[Metrics.WinTrades] or np.nan)
                    )
                ) / (
                    (
                        self._metrics[Metrics.LoseTrades]
                        + np.sqrt(self._metrics[Metrics.LoseTrades])
                    )
                    * self._metrics[Metrics.Loss]
                    / (self._metrics[Metrics.LoseTrades] or np.nan)
                )
            case RewardType.KellyCriterion:
                return self._metrics[Metrics.WinTrades] / (
                    self._metrics[Metrics.Trades] or np.nan
                ) - (
                    1
                    - self._metrics[Metrics.WinTrades]
                    / (self._metrics[Metrics.Trades] or np.nan)
                ) / (
                    self._metrics[Metrics.Profit]
                    / (self._metrics[Metrics.WinTrades] or np.nan)
                ) / (
                    (
                        self._metrics[Metrics.Loss]
                        / (self._metrics[Metrics.LoseTrades] or np.nan)
                    )
                )
            case RewardType.GHPR:
                return np.power(
                    np.prod(self._returns),
                    1 / (self._metrics[Metrics.Trades] or np.nan),
                )
            case RewardType.AHPR:
                return np.mean(self._returns)
            case RewardType.RoMaD:
                if len([pl for pl in self._pl if pl < 0]) == 0:
                    return np.nan
                return -np.sum(self._pl) / (
                    min([pl for pl in self._pl if pl < 0]) or np.nan
                )
            case RewardType.SQN:
                return (
                    np.sqrt(self._metrics[Metrics.Trades])
                    * np.mean(self._pl)
                    / (np.std(self._pl) or np.nan)
                )
            # case RewardType.RecoveryFactor:
            #     return -np.prod(self._returns) / (
            #         self._metrics[Metrics.MaxDD] or np.nan
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
        return {m.name: val for m, val in self._metrics.items()} | {
            rt.name: self.reward(rt) for rt in RewardType
        }