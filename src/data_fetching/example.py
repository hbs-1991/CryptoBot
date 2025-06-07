"""
Пример использования модулей DataNormalizer и MarketAnalyzer.
"""

import time
import json
import logging
from decimal import Decimal
from typing import Dict, Any

from src.data_fetching.data_normalizer import DataNormalizer, NormalizedScanResult
from src.data_fetching.market_analyzer import MarketAnalyzer, MarketAnalysisResult


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Пример данных с бирж (в реальном случае будут получены через API)
def get_mock_exchange_data() -> Dict[str, Dict[str, Any]]:
    """
    Генерирует тестовые данные с бирж.
    
    Returns:
        Словарь с данными бирж
    """
    # Возвращает текущее время в миллисекундах
    current_time = int(time.time() * 1000)
    
    return {
        "binance": {
            "tickers": {
                "BTC/USDT": {
                    "symbol": "BTC/USDT",
                    "bid": 41250.5,
                    "ask": 41255.2,
                    "last": 41253.0,
                    "volume": 1250.45,
                    "timestamp": current_time
                },
                "ETH/USDT": {
                    "symbol": "ETH/USDT",
                    "bid": 2200.3,
                    "ask": 2201.8,
                    "last": 2201.5,
                    "volume": 8500.32,
                    "timestamp": current_time
                },
                "SOL/USDT": {
                    "symbol": "SOL/USDT",
                    "bid": 125.6,
                    "ask": 125.8,
                    "last": 125.7,
                    "volume": 15000.5,
                    "timestamp": current_time
                }
            },
            "orderbooks": {
                "BTC/USDT": {
                    "symbol": "BTC/USDT",
                    "bids": [
                        [41250.5, 0.5],
                        [41245.0, 1.2],
                        [41240.0, 2.3]
                    ],
                    "asks": [
                        [41255.2, 0.8],
                        [41260.0, 1.5],
                        [41265.0, 2.0]
                    ],
                    "timestamp": current_time
                }
            }
        },
        "kucoin": {
            "tickers": {
                "BTC/USDT": {
                    "symbol": "BTC/USDT",
                    "bid": 41245.0,
                    "ask": 41260.0,
                    "last": 41250.0,
                    "volume": 980.21,
                    "timestamp": current_time
                },
                "ETH/USDT": {
                    "symbol": "ETH/USDT",
                    "bid": 2199.5,
                    "ask": 2202.0,
                    "last": 2200.5,
                    "volume": 6300.15,
                    "timestamp": current_time
                },
                "SOL/USDT": {
                    "symbol": "SOL/USDT",
                    "bid": 125.3,
                    "ask": 126.0,
                    "last": 125.8,
                    "volume": 12500.3,
                    "timestamp": current_time
                }
            },
            "orderbooks": {
                "BTC/USDT": {
                    "symbol": "BTC/USDT",
                    "bids": [
                        [41245.0, 0.6],
                        [41240.0, 1.4],
                        [41235.0, 2.0]
                    ],
                    "asks": [
                        [41260.0, 0.7],
                        [41265.0, 1.3],
                        [41270.0, 2.2]
                    ],
                    "timestamp": current_time
                }
            }
        },
        "okx": {
            "tickers": {
                "BTC/USDT": {
                    "symbol": "BTC/USDT",
                    "bid": 41255.0,
                    "ask": 41265.0,
                    "last": 41260.0,
                    "volume": 1100.78,
                    "timestamp": current_time
                },
                "ETH/USDT": {
                    "symbol": "ETH/USDT",
                    "bid": 2198.0,
                    "ask": 2203.0,
                    "last": 2200.0,
                    "volume": 7200.45,
                    "timestamp": current_time
                },
                "SOL/USDT": {
                    "symbol": "SOL/USDT",
                    "bid": 124.8,
                    "ask": 125.5,
                    "last": 125.2,
                    "volume": 13000.25,
                    "timestamp": current_time
                }
            },
            "orderbooks": {
                "BTC/USDT": {
                    "symbol": "BTC/USDT",
                    "bids": [
                        [41255.0, 0.45],
                        [41250.0, 1.1],
                        [41245.0, 1.8]
                    ],
                    "asks": [
                        [41265.0, 0.65],
                        [41270.0, 1.25],
                        [41275.0, 1.9]
                    ],
                    "timestamp": current_time
                }
            }
        }
    }


def main():
    """
    Основная функция примера.
    Демонстрирует использование DataNormalizer и MarketAnalyzer.
    """
    # Получаем данные с бирж (в этом примере используем моковые данные)
    exchange_data = get_mock_exchange_data()
    
    # Создаем экземпляры нормализатора и анализатора
    normalizer = DataNormalizer(decimal_places=6)
    analyzer = MarketAnalyzer(min_profit_percentage=0.1)
    
    logger.info("Нормализуем данные с бирж...")
    
    # Нормализуем данные тикеров
    normalized_tickers = {}
    for exchange, data in exchange_data.items():
        for symbol, ticker_data in data.get("tickers", {}).items():
            normalized_ticker = normalizer.normalize_ticker(ticker_data, exchange)
            
            if symbol not in normalized_tickers:
                normalized_tickers[symbol] = {}
                
            normalized_tickers[symbol][exchange] = normalized_ticker
    
    # Нормализуем данные стаканов
    normalized_orderbooks = {}
    for exchange, data in exchange_data.items():
        for symbol, orderbook_data in data.get("orderbooks", {}).items():
            normalized_orderbook = normalizer.normalize_orderbook(orderbook_data, exchange)
            
            if symbol not in normalized_orderbooks:
                normalized_orderbooks[symbol] = {}
                
            normalized_orderbooks[symbol][exchange] = normalized_orderbook
    
    # Создаем объединенные рыночные данные
    market_data = {}
    for symbol in normalized_tickers:
        # Нормализуем рыночные данные для символа
        market_data[symbol] = normalizer.normalize_market_data(
            symbol=symbol,
            tickers=normalized_tickers.get(symbol, {}),
            order_books=normalized_orderbooks.get(symbol, {})
        )
    
    # Создаем результат сканирования
    scan_result = normalizer.normalize_scan_result(market_data)
    
    logger.info(f"Нормализовано данных для {len(scan_result.symbols)} символов "
               f"с {len(scan_result.exchanges)} бирж")
    
    # Выводим примеры нормализованных данных
    for symbol in scan_result.symbols:
        for exchange in scan_result.data[symbol].exchanges:
            ticker = scan_result.data[symbol].tickers.get(exchange)
            if ticker:
                logger.info(f"{symbol} на {exchange}: "
                           f"Bid: {ticker.bid}, Ask: {ticker.ask}, "
                           f"Spread: {ticker.spread_percentage}%")
    
    # Анализируем данные
    logger.info("Анализируем рыночные данные...")
    analysis_result = analyzer.analyze(scan_result)
    
    # Выводим результаты анализа
    logger.info(f"Найдено арбитражных возможностей: {len(analysis_result.arbitrage_opportunities)}")
    
    # Выводим подробную информацию о арбитражных возможностях
    if analysis_result.arbitrage_opportunities:
        logger.info("Топ арбитражных возможностей:")
        for i, opportunity in enumerate(analysis_result.arbitrage_opportunities[:3]):
            logger.info(f"{i+1}. {opportunity.symbol}: "
                       f"Купить на {opportunity.buy_exchange} по {opportunity.buy_price}, "
                       f"Продать на {opportunity.sell_exchange} по {opportunity.sell_price}, "
                       f"Прибыль: {opportunity.profit_percentage}%, "
                       f"Объем: {opportunity.estimated_volume}, "
                       f"Оценка прибыли: {opportunity.estimated_profit}")
    
    # Пример вывода статистики по биржам
    logger.info("Статистика по биржам:")
    for exchange, stats in analysis_result.exchange_stats.items():
        logger.info(f"{exchange}: {stats['symbols_count']} символов, "
                   f"Средний спред: {stats['avg_spread_percent']:.2f}%, "
                   f"Общий объем: {stats['total_volume']:.2f}")


if __name__ == "__main__":
    main() 