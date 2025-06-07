"""
Модуль для получения данных с криптовалютных бирж.
Обеспечивает подключение, получение и обработку рыночных данных.
"""

from src.data_fetching.exchange_connector import ExchangeConnector, ExchangeConnectionError
from src.data_fetching.binance_connector import BinanceConnector
from src.data_fetching.kucoin_connector import KuCoinConnector
from src.data_fetching.okx_connector import OKXConnector
from src.data_fetching.exchange_factory import ExchangeFactory
from src.data_fetching.market_scanner import MarketScanner
from src.data_fetching.data_normalizer import DataNormalizer
from src.data_fetching.price_monitor import PriceMonitor

__all__ = [
    "ExchangeConnector",
    "ExchangeConnectionError",
    "BinanceConnector",
    "KuCoinConnector",
    "OKXConnector",
    "ExchangeFactory",
    "MarketScanner",
    "DataNormalizer",
    "PriceMonitor"
]
