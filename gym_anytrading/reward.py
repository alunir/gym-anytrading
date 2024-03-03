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
        self._metrics = {m: 0.0 for m in Metrics}

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

    @staticmethod
    def __welford_update(old_mean, old_var, num, new_val):
        # Welford's algorithm https://zenn.dev/utcarnivaldayo/articles/ffeed5ac2e62bb
        tmp = old_mean + (new_val - old_mean) / (num + 1)
        var = old_var + (new_val - old_mean) * (new_val - tmp)
        return tmp, var

    # update metrics
    def update(self, action: Actions, current_tick, last_trade_tick):
        current_price, last_trade_price = self._trade_price(
            current_tick
        ), self._trade_price(last_trade_tick)

        self._update_max_dd(action, current_tick, last_trade_tick)
        if action == Actions.Buy:
            price_diff = current_price - last_trade_price
            pl = price_diff - abs(price_diff) * self._trade_fee_bid_percent
            returns = pl / last_trade_price + 1.0

            self._metrics[Metrics.MeanPL], self._metrics[Metrics.VarPL] = (
                self.__welford_update(
                    self._metrics[Metrics.MeanPL],
                    self._metrics[Metrics.VarPL],
                    self._metrics[Metrics.Trades],
                    pl,
                )
            )
            self._metrics[Metrics.MeanReturns], self._metrics[Metrics.VarReturns] = (
                self.__welford_update(
                    self._metrics[Metrics.MeanReturns],
                    self._metrics[Metrics.VarReturns],
                    self._metrics[Metrics.Trades],
                    returns,
                )
            )
            self._metrics[Metrics.LogReturns] += np.log(returns)
            self._metrics[Metrics.Profit] += max(pl, 0)
            self._metrics[Metrics.Loss] += min(pl, 0)
            self._metrics[Metrics.Trades] += 1
            self._metrics[Metrics.WinTrades] += 1 if pl > 0 else 0
            self._metrics[Metrics.LoseTrades] += 1 if pl < 0 else 0
        elif action == Actions.Sell:
            price_diff = last_trade_price - current_price
            pl = price_diff - abs(price_diff) * self._trade_fee_ask_percent
            returns = pl / current_price + 1.0

            self._metrics[Metrics.MeanPL], self._metrics[Metrics.VarPL] = (
                self.__welford_update(
                    self._metrics[Metrics.MeanPL],
                    self._metrics[Metrics.VarPL],
                    self._metrics[Metrics.Trades],
                    pl,
                )
            )
            self._metrics[Metrics.MeanReturns], self._metrics[Metrics.VarReturns] = (
                self.__welford_update(
                    self._metrics[Metrics.MeanReturns],
                    self._metrics[Metrics.VarReturns],
                    self._metrics[Metrics.Trades],
                    returns,
                )
            )
            self._metrics[Metrics.LogReturns] += np.log(returns)
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
            case RewardType.Returns:
                return np.exp(self._metrics[Metrics.LogReturns])
            case RewardType.LogReturns:
                return self._metrics[Metrics.LogReturns]
            case RewardType.WinRate:
                if self._metrics[Metrics.Trades] == 0:
                    return 0.0
                return self._metrics[Metrics.WinTrades] / (
                    self._metrics[Metrics.Trades]
                )
            case RewardType.ProfitPerTrade:
                if self._metrics[Metrics.Trades] == 0:
                    return 0.0
                return self._metrics[Metrics.Profit] / (self._metrics[Metrics.Trades])
            case RewardType.ProfitFactor:
                if self._metrics[Metrics.LoseTrades] == 0:
                    return 0.0
                return -self._metrics[Metrics.Profit] / (self._metrics[Metrics.Loss])
            case RewardType.PesimisticProfitFactor:
                if (
                    self._metrics[Metrics.WinTrades] == 0
                    or self._metrics[Metrics.LoseTrades] == 0
                ):
                    return 0.0
                return -(
                    (
                        self._metrics[Metrics.WinTrades]
                        - np.sqrt(self._metrics[Metrics.WinTrades])
                    )
                    * (
                        self._metrics[Metrics.Profit]
                        / (self._metrics[Metrics.WinTrades])
                    )
                ) / (
                    (
                        self._metrics[Metrics.LoseTrades]
                        + np.sqrt(self._metrics[Metrics.LoseTrades])
                    )
                    * self._metrics[Metrics.Loss]
                    / (self._metrics[Metrics.LoseTrades])
                )
            case RewardType.KellyCriterion:
                if (
                    self._metrics[Metrics.WinTrades] == 0
                    or self._metrics[Metrics.LoseTrades] == 0
                ):
                    return 0.0
                return self._metrics[Metrics.WinTrades] / (
                    self._metrics[Metrics.Trades]
                ) - (
                    1
                    - self._metrics[Metrics.WinTrades] / (self._metrics[Metrics.Trades])
                ) / (
                    self._metrics[Metrics.Profit] / (self._metrics[Metrics.WinTrades])
                ) / (
                    (self._metrics[Metrics.Loss] / (self._metrics[Metrics.LoseTrades]))
                )
            case RewardType.GHPR:
                if self._metrics[Metrics.Trades] == 0:
                    return 0.0
                return np.power(
                    np.exp(self._metrics[Metrics.LogReturns]),
                    1 / (self._metrics[Metrics.Trades]),
                )
            case RewardType.AHPR:
                return self._metrics[Metrics.MeanReturns]
            case RewardType.SQN:
                if self._metrics[Metrics.VarPL] == 0.0:
                    return 0.0
                return (
                    np.sqrt(self._metrics[Metrics.Trades])
                    * self._metrics[Metrics.MeanPL]
                    / np.sqrt(self._metrics[Metrics.VarPL])
                )
            case RewardType.RecoveryFactor:
                return -np.exp(self._metrics[Metrics.LogReturns]) / (
                    self._metrics[Metrics.MaxDD] or np.nan
                )
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
            # case RewardType.RoMaD:
            #     if len([pl for pl in self._pl if pl < 0]) == 0:
            #         return -1e10
            #     return (
            #         -self._metrics[Metrics.MeanPL]
            #         * self._metrics[Metrics.Trades]
            #         / (min([pl for pl in self._pl if pl < 0]))
            #     )
            case _:
                raise NotImplementedError

    def get_info(self):
        return {m.name: val for m, val in self._metrics.items()} | {
            rt.name: self.reward(rt) for rt in RewardType
        }
