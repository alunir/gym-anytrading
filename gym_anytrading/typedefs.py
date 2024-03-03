from enum import Enum, auto


class Actions(Enum):
    Buy = 0
    Sell = 1
    Unwind = 2


class Positions(Enum):
    Long = 0
    Short = 1
    Neutral = 2

    def opposite(self):
        if self == Positions.Long:
            return Positions.Short
        elif self == Positions.Short:
            return Positions.Long
        else:
            raise ValueError("Invalid opposite because of Neutral")


# 報酬計算に試用する統計情報
class Metrics(Enum):
    Profit = auto()  # 純利益(手数料なし)
    Loss = auto()  # 純損失(手数料なし)
    Trades = auto()  # 取引回数
    WinTrades = auto()  # 勝ち取引回数
    LoseTrades = auto()  # 負け取引回数
    MaxDD = auto()  # 最大ドローダウン


# 逐次計算する指標
class RewardType(Enum):
    Profit = auto()  # 利益
    Return = auto()  # リターン(Percentage)
    LogReturn = auto()  # ログリターン
    WinRate = auto()  # 勝率
    ProfitPerTrade = auto()  # 期待損益
    ProfitFactor = auto()  # プロフィットファクター
    PesimisticProfitFactor = auto()  # 悲観的プロフィットファクター
    KellyCriterion = auto()  # ケリー基準
    GHPR = auto()  # GHPR
    AHPR = auto()  # AHPR
    RoMaD = auto()  # リターン(%)/最大ドローダウン
    # System Quality Number.
    # 7.0~: holy grail
    # 5.1~: excellent
    # 3.0~: very good
    # 2.5~: good
    # 2.0~: average
    # 1.6~: below average
    SQN = auto()

    # TODO: 時間軸をそろえなければならない．日次，年次
    # https://github.com/kernc/backtesting.py/blob/0ce24d80b1bcb8120d95d31dc3bb351b1052a27d/backtesting/_stats.py#L113
    # RecoveryFactor = auto()  # リカバリーファクター = 損益 / 最大ドローダウン
    # SharpeRatio = auto()  # シャープレシオ
    # SortinoRatio = auto()  # ソルティノレシオ
    # CalmarRatio = auto()  # カルマーレシオ
