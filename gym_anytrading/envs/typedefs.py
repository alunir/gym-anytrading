from enum import Enum


class Actions(Enum):
    DoNothing = 0
    Buy = 1
    Sell = 2


class Positions(Enum):
    Neutral = 0
    Long = 1
    Short = 2
