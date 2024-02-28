from enum import Enum


class Actions(Enum):
    Unwind = 0
    Buy = 1
    Sell = 2


class Positions(Enum):
    Neutral = 0
    Long = 1
    Short = 2


class RewardType(Enum):
    Profit = 0
    Return = 1
    LogReturn = 2
    RoMaD = 3
