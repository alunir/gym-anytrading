from time import time

import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from ..typedefs import Positions, Actions, RewardType
from ..reward import RewardCalculator


class TradingEnv(gym.Env, RewardCalculator):

    metadata = {"render_modes": ["human"], "render_fps": 3}

    def __init__(
        self,
        df,
        window_size,
        render_mode=None,
        reward_type=RewardType.Profit,
        ask_column="Ask",
        bid_column="Bid",
        trade_fee_ask_percent=0.0,
        trade_fee_bid_percent=0.0,
    ):
        assert df.ndim == 2
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.render_mode = render_mode
        self._reward_type = reward_type
        self._trade_fee_ask_percent = trade_fee_ask_percent
        self._trade_fee_bid_percent = trade_fee_bid_percent

        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()
        self.shape = (window_size, len(df.columns))

        # reward calculator setup
        RewardCalculator.__init__(
            self,
            prices=self.prices,
            ask=(
                self.df[ask_column].iloc[self.window_size :]
                if hasattr(self.df, ask_column)
                else None
            ),
            bid=(
                self.df[bid_column].iloc[self.window_size :]
                if hasattr(self.df, bid_column)
                else None
            ),
            trade_fee_ask_percent=trade_fee_ask_percent,
            trade_fee_bid_percent=trade_fee_bid_percent,
        )

        # spaces
        self.action_space = gym.spaces.Discrete(
            len([Actions.Buy, Actions.Sell]), start=Actions.Buy.value
        )
        INF = 1e10
        self.observation_space = gym.spaces.Box(
            low=-INF,
            high=INF,
            shape=self.shape,
            dtype=np.float32,
        )

        # episode
        self._start_tick = self.window_size
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

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed, options=options)
        RewardCalculator.reset(self)

        self.action_space.seed(
            int((self.np_random.uniform(0, seed if seed is not None else 1)))
        )

        self._truncated = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Short
        self._position_history = (self.window_size * [None]) + [self._position]

        self._first_rendering = True
        self.history = {}

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self._truncated = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._truncated = True

        step_reward = self._calculate_reward(action)
        # self._total_reward += step_reward

        # self._update_profit(action)

        trade = False
        if (action == Actions.Buy.value and self._position == Positions.Short) or (
            action == Actions.Sell.value and self._position == Positions.Long
        ):
            trade = True

        if trade:
            self._position = self._position.opposite()
            self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)

        if self.render_mode == "human":
            self._render_frame()

        return observation, step_reward, False, self._truncated, info

    def _get_info(self):
        return self.get_info()

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
            if position == Positions.Short:
                color = "red"
            elif position == Positions.Long:
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
            "Total Reward: %.6f" % self._reward_calculator.reward(self._reward_type)
            + " ~ "
            + "Total Profit: %.6f" % self._reward_calculator.reward(RewardType.Profit)
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
            if self._position_history[i] == Positions.Short:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.Long:
                long_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], "ro")
        plt.plot(long_ticks, self.prices[long_ticks], "go")

        if title:
            plt.title(title)

        plt.suptitle(
            "Total Reward: %.6f" % self.reward(self._reward_type)
            + " ~ "
            + "Total Profit: %.6f" % self.reward(RewardType.Profit)
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

    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError
