from .typedefs import RewardType, Metrics, OrderAction, Position

import numpy as np


class RewardCalculator:
    def __init__(
        self,
        trade_fee_ask_percent,
        trade_fee_bid_percent,
        prices=None,
        ask=None,
        bid=None,
    ):
        if ask is not None and bid is not None:
            assert len(ask) == len(bid)
            self._ask, self._bid = ask, bid
        elif prices is not None:
            self._prices = prices
        else:
            raise ValueError("Must provide prices or (ask and bid).")
        self._trade_fee_ask_percent = trade_fee_ask_percent
        self._trade_fee_bid_percent = trade_fee_bid_percent
        self._metrics = {m: 0.0 for m in Metrics}
        self._amount_history = []
        self._trade_price_history = []

    def _trade_price(self, tick, action: OrderAction):
        if hasattr(self, "_prices") and self._prices is not None:
            return self._prices[tick]
        else:
            if action > 0.0:
                # current position: short -> Action should be Buy at the ask price
                return self._ask[tick]
            elif action < 0.0:
                # current position: long -> Action should be Sell at the bid price
                return self._bid[tick]
            else:
                # Neutral
                return (self._ask[tick] + self._bid[tick]) / 2

    def _update_max_dd(
        self, action: OrderAction, current_tick: int, last_trade_tick: int
    ):
        assert (
            last_trade_tick < current_tick
        ), f"last_trade_tick: {last_trade_tick} >= current_tick: {current_tick}"
        entry_price = self._trade_price_history[0]
        if action > 0.0:
            dd = np.min(self._ask[last_trade_tick:current_tick]) / entry_price - 1.0
            self._metrics[Metrics.MaxDD] = min(dd, self._metrics[Metrics.MaxDD])
        elif action < 0.0:
            dd = 1.0 - np.max(self._bid[last_trade_tick:current_tick]) / entry_price
            self._metrics[Metrics.MaxDD] = min(dd, self._metrics[Metrics.MaxDD])
        else:
            raise ValueError("Invalid position")

    def reset(self):
        self._metrics = {m: 0.0 for m in Metrics}

    @staticmethod
    def __welford_update(old_mean, old_var, num, new_val):
        # Welford's algorithm https://zenn.dev/utcarnivaldayo/articles/ffeed5ac2e62bb
        tmp = old_mean + (new_val - old_mean) / (num + 1)
        var = old_var + (new_val - old_mean) * (new_val - tmp)
        return tmp, var

    # update metrics
    def update(
        self, position: Position, action: OrderAction, current_tick, last_trade_tick
    ):
        current_price = self._trade_price(current_tick, action)
        assert len(self._trade_price_history) == len(
            self._amount_history
        ), f"trade_price_history: {len(self._trade_price_history)} != amount_history: {len(self._amount_history)}"

        # if position == 0 or sum(self._amount_history) == 0:
        #     self._amount_history.clear()
        #     self._trade_price_history.clear()
        #     # ポジションが0の場合、新規ポジションとして扱う
        #     self._amount_history.append(action)
        #     self._trade_price_history.append(current_price)
        #     return  # 損益計算やメトリクスの更新は行わない

        # ポジションとアクションが反対の場合、相殺処理を行う
        if position * action < 0:
            offset_amount = min(abs(position), abs(action))
            new_position_amount = abs(action) - offset_amount

            # 相殺部分の損益計算
            average_trade_price = np.average(
                self._trade_price_history, weights=self._amount_history
            )

            price_diff = (
                current_price - average_trade_price
                if position > 0
                else average_trade_price - current_price
            )
            offset_pl = price_diff * offset_amount - abs(price_diff) * offset_amount * (
                self._trade_fee_ask_percent
                if position > 0
                else self._trade_fee_bid_percent
            )

            # 最大ドローダウンの更新
            self._update_max_dd(action, current_tick, last_trade_tick)
            # メトリクス更新
            self._update_metrics(offset_pl)

            # ポジションが反転する場合、履歴をクリア
            if (position + action) * position < 0:
                self._amount_history.clear()
                self._trade_price_history.clear()

            # 新規ポジションの追加
            if new_position_amount > 0:
                self._amount_history.append(
                    new_position_amount * (1 if action > 0 else -1)
                )
                self._trade_price_history.append(current_price)

        else:
            self._amount_history.clear()
            self._trade_price_history.clear()
            # ポジションが0の場合、新規ポジションとして扱う
            self._amount_history.append(action)
            self._trade_price_history.append(current_price)
            return  # 損益計算やメトリクスの更新は行わない

            # # ポジションとアクションが同じ方向の場合、通常の損益計算
            # average_trade_price = np.average(
            #     self._trade_price_history, weights=self._amount_history
            # )
            # price_diff = (
            #     current_price - average_trade_price
            #     if position > 0
            #     else average_trade_price - current_price
            # )
            # pl = price_diff * abs(sum(self._amount_history)) - abs(price_diff) * (
            #     self._trade_fee_ask_percent
            #     if position > 0
            #     else self._trade_fee_bid_percent
            # )

            # # 最大ドローダウンの更新
            # self._update_max_dd(action, current_tick, last_trade_tick)

            # # メトリクス更新
            # self._update_metrics(pl)

    def _update_metrics(self, pl):
        # 損益の平均と分散を更新するために Welford のアルゴリズムを使用
        num_trades = self._metrics[Metrics.Trades] + 1  # 新しい取引をカウントに追加
        old_mean_pl = self._metrics[Metrics.MeanPL]
        old_var_pl = self._metrics[Metrics.VarPL]
        old_mean_returns = self._metrics[Metrics.MeanReturns]
        old_var_returns = self._metrics[Metrics.VarReturns]

        entry_price = self._trade_price_history[0]
        returns = pl / entry_price + 1.0

        # Welford のアルゴリズムで平均と分散を更新
        new_mean_pl, new_var_pl = self.__welford_update(
            old_mean_pl, old_var_pl, num_trades, pl
        )
        new_mean_returns, new_var_returns = self.__welford_update(
            old_mean_returns, old_var_returns, num_trades, returns
        )

        self._metrics[Metrics.MeanPL] = new_mean_pl
        self._metrics[Metrics.VarPL] = new_var_pl
        self._metrics[Metrics.MeanReturns] = new_mean_returns
        self._metrics[Metrics.VarReturns] = new_var_returns
        self._metrics[Metrics.Trades] = num_trades

        # ログリターンの更新
        self._metrics[Metrics.LogReturns] += np.log(returns)

        # 損益に基づいてその他のメトリクスを更新
        self._metrics[Metrics.Profit] += max(pl, 0)
        self._metrics[Metrics.Loss] += min(pl, 0)
        self._metrics[Metrics.WinTrades] += 1 if pl > 0 else 0
        self._metrics[Metrics.LoseTrades] += 1 if pl < 0 else 0

    # calculate reward based on metrics
    def reward(self, reward_type: RewardType) -> float | None:
        match reward_type:
            case RewardType.Profit:
                return self._metrics[Metrics.Profit]
            case RewardType.Returns:
                return np.expm1(self._metrics[Metrics.LogReturns]) * 100  # percentage
            case RewardType.LogReturns:
                return self._metrics[Metrics.LogReturns]
            case RewardType.WinRate:
                if self._metrics[Metrics.Trades] == 0:
                    return 0.0
                return self._metrics[Metrics.WinTrades] / self._metrics[Metrics.Trades]
            case RewardType.ProfitPerTrade:
                if self._metrics[Metrics.Trades] == 0:
                    return 0.0
                return self._metrics[Metrics.Profit] / self._metrics[Metrics.Trades]
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
                return self._metrics[Metrics.WinTrades] / self._metrics[
                    Metrics.Trades
                ] - (
                    1 - self._metrics[Metrics.WinTrades] / self._metrics[Metrics.Trades]
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
        return {m.name: float(self._metrics[m]) for m in Metrics} | {
            rt.name: float(self.reward(rt)) for rt in RewardType
        }
