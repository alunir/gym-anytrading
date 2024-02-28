from enum import Enum


class Actions(Enum):
    Unwind = 0
    Buy = 1
    Sell = 2


class Positions(Enum):
    Neutral = 0
    Long = 1
    Short = 2

    def opposite(self):
        if self == Positions.Long:
            return Positions.Short
        elif self == Positions.Short:
            return Positions.Long
        else:
            raise ValueError("Invalid opposite because of Neutral")


class RewardType(Enum):
    Profit = 0
    Return = 1
    LogReturn = 2
    MaxDD = 3
    RoMaD = 4
