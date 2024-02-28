from .utils import load_dataset as _load_dataset

# Load CRYPTO datasets
CRYPTO_ETHUSDT_5M = _load_dataset("CRYPTO_ETHUSDT_5M", "Epoch")

# Load FOREX datasets
FOREX_EURUSD_1H_ASK = _load_dataset("FOREX_EURUSD_1H_ASK", "Time")

# Load Stocks datasets
STOCKS_GOOGL = _load_dataset("STOCKS_GOOGL", "Date")
