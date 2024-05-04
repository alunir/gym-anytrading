from time import time

from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from ..typedefs import RewardType, Position
from ..reward import RewardCalculator


INF = 1e10


class TradingEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 3}

    def __init__(
        self,
        prices: pd.Series,
        ask: pd.Series,
        bid: pd.Series,
        df: pd.DataFrame,
        window_size,
        render_mode=None,
        reward_type=RewardType.LogReturns,
        trade_fee_ask_percent=0.0,
        trade_fee_bid_percent=0.0,
        box_range: Tuple[float, float] = (-INF, INF),
    ):
        assert df.ndim == 2
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        assert box_range[0] < box_range[1], "box_range should be a tuple (low, high)"

        self.render_mode = render_mode
        self._reward_type = reward_type
        self._trade_fee_ask_percent = trade_fee_ask_percent
        self._trade_fee_bid_percent = trade_fee_bid_percent

        self.prices = prices
        self.df = df[df.columns[~df.columns.isin([prices.name, ask.name, bid.name])]]
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, len(self.df.columns))

        # reward calculator setup
        self._reward_calculator = RewardCalculator(
            prices=self.prices,
            ask=ask,
            bid=bid,
            trade_fee_ask_percent=trade_fee_ask_percent,
            trade_fee_bid_percent=trade_fee_bid_percent,
        )

        # spaces
        self.action_space = gym.spaces.Box(
            low=-2,  # position: 1 -> -1
            high=2,  # position: -1 -> 1
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-INF,
            high=INF,
            shape=self.shape,
            dtype=np.float32,
        )

        # episode
        self._start_tick = self.window_size - 1
        self._end_tick = len(self.prices) - 1
        self._truncated = None
        self._current_tick = None
        self._last_trade_tick = None

        self._position = None
        self._position_history = None

        # self._total_reward = None
        # self._total_profit = None
        self._first_rendering = None
        self.history = None

        # self._epoch = None
        # self._max_dd = -1e10

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self._reward_calculator.reset()

        self.action_space.seed(
            int((self.np_random.uniform(0, seed if seed is not None else 1)))
        )

        self._truncated = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position: Position = Position(0.0)
        self._position_history = (self.window_size * [None]) + [self._position]

        # self._total_reward = 0.0
        # self._total_profit = 1.0  # unit

        self._first_rendering = True
        self.history = {}

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """
        Args:
            action (_type_): _description_

        Returns:
            _type_: _description_
        """
        # action_space is Box(1,) with range (-2, 2). This gonna be a np.array.
        action = float(action)

        self._truncated = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._truncated = True

        # Ensure action is within the proper range
        if self._position + action > 1:
            action = 1 - self._position
        elif self._position + action < -1:
            action = -1 - self._position

        step_reward = self._calculate_reward(action)

        _next_position = self._position + action
        if action != 0.0:
            self._position = _next_position

        if np.sign(self._position) * np.sign(_next_position) < 0:
            self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)

        if self.render_mode == "human":
            self._render_frame()

        return observation, step_reward, False, self._truncated, info

    def _get_info(self):
        return self._reward_calculator.get_info()

    def _get_observation(self):
        return self.signal_features[
            (self._current_tick - self.window_size + 1) : self._current_tick + 1
        ]

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _render_frame(self):
        self.render()

    def render(self, mode="human"):

        def _plot_position(position, tick):
            color = None
            if position == self.PositionsN.is_short():
                color = "red"
            elif position == self.PositionsN.is_long():
                color = "green"
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        start_time = time()

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward
            + " ~ "
            + "Total Profit: %.6f" % self._total_profit
        )

        end_time = time()
        process_time = end_time - start_time

        pause_time = (1 / self.metadata["render_fps"]) - process_time
        assert pause_time > 0.0, "High FPS! Try to reduce the 'render_fps' value."

        plt.pause(pause_time)

    def render_all(self, title=None):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == self.PositionsN.is_short():
                short_ticks.append(tick)
            elif self._position_history[i] == self.Positions.is_long():
                long_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], "ro")
        plt.plot(long_ticks, self.prices[long_ticks], "go")

        if title:
            plt.title(title)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward
            + " ~ "
            + "Total Profit: %.6f" % self._total_profit
        )

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()

    def _process_data(self):
        raise NotImplementedError

    def _calculate_reward(self, action):
        raise NotImplementedError

    # def _update_profit(self, action):
    #     raise NotImplementedError

    # def max_possible_profit(self):  # trade fees are ignored
    #     raise NotImplementedError
