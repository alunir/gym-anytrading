""" RewardCalculator Test """

import pytest
import numpy as np

from gym_anytrading.datasets import CRYPTO_ETHUSDT_5M
from gym_anytrading.reward import RewardCalculator, RewardType


def test_reward_calculator_initialization():
    prices = CRYPTO_ETHUSDT_5M["Close"]
    ask = CRYPTO_ETHUSDT_5M["High"]
    bid = CRYPTO_ETHUSDT_5M["Low"]
    trade_fee_ask_percent = 0.01
    trade_fee_bid_percent = 0.01

    # インスタンス生成
    reward_calculator = RewardCalculator(
        prices=prices,
        ask=ask,
        bid=bid,
        trade_fee_ask_percent=trade_fee_ask_percent,
        trade_fee_bid_percent=trade_fee_bid_percent,
    )

    # チェック
    assert reward_calculator._metrics is not None
    assert reward_calculator._trade_price_history == []
    assert reward_calculator._amount_history == []


def test_reward_calculator_update():
    prices = CRYPTO_ETHUSDT_5M["Close"]
    ask = CRYPTO_ETHUSDT_5M["High"]
    bid = CRYPTO_ETHUSDT_5M["Low"]
    trade_fee_ask_percent = 0.01
    trade_fee_bid_percent = 0.01

    # インスタンス生成
    reward_calculator = RewardCalculator(
        prices=prices,
        ask=ask,
        bid=bid,
        trade_fee_ask_percent=trade_fee_ask_percent,
        trade_fee_bid_percent=trade_fee_bid_percent,
    )

    # Parameters
    window_size = 10

    _position = 0.0
    _current_tick = window_size
    _last_trade_tick = _current_tick - 1

    # Step 0

    action = 0.5

    step_reward = reward_calculator.update(_position, action, 0, -1)
    assert step_reward is None

    _next_position = _position + action
    if action != 0.0:
        _position = _next_position

    if np.sign(_position) * np.sign(_next_position) < 0:
        _last_trade_tick = _current_tick

    step_reward = reward_calculator.update(
        _position, action, _current_tick, _last_trade_tick
    )
    assert step_reward is None

    info = reward_calculator.get_info()
    assert (
        str(info)
        == "{'MeanPL': 0.0, 'VarPL': 0.0, 'MeanReturns': 0.0, 'VarReturns': 0.0, 'LogReturns': 0.0, 'Profit': 0.0, 'Loss': 0.0, 'Trades': 0.0, 'WinTrades': 0.0, 'LoseTrades': 0.0, 'MaxDD': 0.0, 'Returns': 0.0, 'WinRate': 0.0, 'ProfitPerTrade': 0.0, 'ProfitFactor': 0.0, 'PesimisticProfitFactor': 0.0, 'KellyCriterion': 0.0, 'GHPR': 0.0, 'AHPR': 0.0, 'SQN': 0.0, 'RecoveryFactor': nan}"
    )

    # Step 1

    action = 0.5

    step_reward = reward_calculator.update(_position, action, 0, -1)
    assert step_reward is None

    _next_position = _position + action
    if action != 0.0:
        _position = _next_position

    if np.sign(_position) * np.sign(_next_position) < 0:
        _last_trade_tick = _current_tick

    info = reward_calculator.get_info()
    assert (
        str(info)
        == "{'MeanPL': 0.0, 'VarPL': 0.0, 'MeanReturns': 0.0, 'VarReturns': 0.0, 'LogReturns': 0.0, 'Profit': 0.0, 'Loss': 0.0, 'Trades': 0.0, 'WinTrades': 0.0, 'LoseTrades': 0.0, 'MaxDD': 0.0, 'Returns': 0.0, 'WinRate': 0.0, 'ProfitPerTrade': 0.0, 'ProfitFactor': 0.0, 'PesimisticProfitFactor': 0.0, 'KellyCriterion': 0.0, 'GHPR': 0.0, 'AHPR': 0.0, 'SQN': 0.0, 'RecoveryFactor': nan}"
    )

    # Step 2

    action = -1.0

    step_reward = reward_calculator.update(_position, action, 0, -1)
    assert step_reward is None

    _next_position = _position + action
    if action != 0.0:
        _position = _next_position

    if np.sign(_position) * np.sign(_next_position) < 0:
        _last_trade_tick = _current_tick

    info = reward_calculator.get_info()
    assert (
        str(info)
        == "{'MeanPL': -0.5706500000000552, 'VarPL': 0.6512828450001259, 'MeanReturns': 0.499646805060408, 'VarReturns': 0.4992938596141467, 'LogReturns': -0.0007066394900700643, 'Profit': 0.0, 'Loss': -1.1413000000001103, 'Trades': 1.0, 'WinTrades': 0.0, 'LoseTrades': 1.0, 'MaxDD': nan, 'Returns': -0.07063898791840328, 'WinRate': 0.0, 'ProfitPerTrade': 0.0, 'ProfitFactor': 0.0, 'PesimisticProfitFactor': 0.0, 'KellyCriterion': 0.0, 'GHPR': 0.999293610120816, 'AHPR': 0.499646805060408, 'SQN': -0.7071067811865476, 'RecoveryFactor': nan}"
    )
