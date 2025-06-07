"""
Модуль для генерации имитационных данных рынка.
Используется для тестирования арбитражного бота без реальных подключений к биржам.
"""

import time
import random
import logging
from typing import Dict, List, Optional
from datetime import datetime

from src.data_fetching.market_scanner import MarketData
from config import settings


class MarketDataSimulator:
    """
    Генератор имитационных данных для тестирования арбитражного бота.
    Создает синтетические данные, имитирующие рыночные данные с бирж.
    """
    
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        exchanges: Optional[List[str]] = None,
        base_prices: Optional[Dict[str, float]] = None,
        volatility: float = 0.01,
        spread_percentage: float = 0.001,
        arbitrage_opportunity_chance: float = 0.2,
        max_price_difference: float = 0.03,
    ):
        """
        Инициализирует симулятор рыночных данных.
        
        Args:
            symbols: Список торговых пар (если None, используются из settings)
            exchanges: Список бирж (если None, используются все включенные биржи)
            base_prices: Базовые цены для торговых пар (если None, используются случайные значения)
            volatility: Волатильность цен (размер случайного изменения в процентах)
            spread_percentage: Процент спреда между bid и ask
            arbitrage_opportunity_chance: Вероятность создания арбитражной возможности
            max_price_difference: Максимальная разница в цене между биржами (для арбитражных возможностей)
        """
        self.logger = logging.getLogger(__name__)
        self.symbols = symbols or settings.ALLOWED_PAIRS
        
        # Если биржи не указаны, используем включенные биржи из конфигурации
        if not exchanges:
            from config import exchange_config
            self.exchanges = [ex.name for ex in exchange_config.get_enabled_exchanges()]
        else:
            self.exchanges = exchanges
        
        # Инициализируем базовые цены для каждой торговой пары
        self.base_prices = base_prices or {
            "BTC/USDT": 50000.0 + random.uniform(-5000, 5000),
            "ETH/USDT": 3000.0 + random.uniform(-300, 300),
            "BNB/USDT": 500.0 + random.uniform(-50, 50),
        }
        
        # Заполняем недостающие базовые цены случайными значениями
        for symbol in self.symbols:
            if symbol not in self.base_prices:
                self.base_prices[symbol] = 100.0 + random.uniform(-10, 10)
        
        self.volatility = volatility
        self.spread_percentage = spread_percentage
        self.arbitrage_opportunity_chance = arbitrage_opportunity_chance
        self.max_price_difference = max_price_difference
        
        # Словарь для хранения текущих цен на разных биржах
        self.current_prices = {}
        
        # Инициализируем начальные цены на всех биржах
        for symbol in self.symbols:
            self.current_prices[symbol] = {}
            base_price = self.base_prices[symbol]
            
            for exchange in self.exchanges:
                # Добавляем небольшое случайное отклонение для разных бирж
                exchange_price = base_price * (1 + random.uniform(-0.01, 0.01))
                self.current_prices[symbol][exchange] = exchange_price
        
        self.logger.info(
            f"MarketDataSimulator initialized with {len(self.symbols)} symbols, "
            f"{len(self.exchanges)} exchanges"
        )
    
    def _update_prices(self):
        """
        Обновляет цены на всех биржах с учетом волатильности.
        Иногда создает арбитражные возможности между биржами.
        """
        for symbol in self.symbols:
            base_price = self.base_prices[symbol]
            
            # Общее изменение рынка для данной торговой пары
            market_change = random.uniform(-self.volatility, self.volatility)
            base_price = base_price * (1 + market_change)
            self.base_prices[symbol] = base_price
            
            # Создаем арбитражную возможность с заданной вероятностью
            create_arbitrage = random.random() < self.arbitrage_opportunity_chance
            
            if create_arbitrage and len(self.exchanges) >= 2:
                # Выбираем две случайные биржи для создания арбитражной возможности
                exch1, exch2 = random.sample(self.exchanges, 2)
                
                # Создаем разницу в ценах между биржами
                price_diff = base_price * random.uniform(0.01, self.max_price_difference)
                
                # Одна биржа имеет более низкую цену (для покупки)
                self.current_prices[symbol][exch1] = base_price - price_diff/2
                
                # Другая биржа имеет более высокую цену (для продажи)
                self.current_prices[symbol][exch2] = base_price + price_diff/2
                
                # Обновляем цены на остальных биржах
                for exchange in self.exchanges:
                    if exchange not in [exch1, exch2]:
                        exchange_change = random.uniform(-self.volatility/2, self.volatility/2)
                        self.current_prices[symbol][exchange] = base_price * (1 + exchange_change)
            else:
                # Если не создаем арбитражную возможность, просто обновляем цены с небольшими отклонениями
                for exchange in self.exchanges:
                    exchange_change = random.uniform(-self.volatility/2, self.volatility/2)
                    self.current_prices[symbol][exchange] = base_price * (1 + exchange_change)
    
    def generate_market_data(self) -> List[MarketData]:
        """
        Генерирует имитационные данные рынка.
        
        Returns:
            Список объектов MarketData с имитационными данными рынка
        """
        # Обновляем цены перед генерацией данных
        self._update_prices()
        
        current_timestamp = int(time.time() * 1000)  # в миллисекундах
        result = []
        
        for symbol in self.symbols:
            market_data = MarketData(symbol=symbol, timestamp=current_timestamp)
            
            for exchange in self.exchanges:
                mid_price = self.current_prices[symbol][exchange]
                spread = mid_price * self.spread_percentage
                
                # Создаем данные тикера
                bid_price = mid_price - spread/2
                ask_price = mid_price + spread/2
                
                # Имитируем небольшой объем
                volume = random.uniform(1.0, 10.0)
                
                # Добавляем данные тикера
                from src.data_fetching.exchange_connector import TickerData
                ticker = TickerData(
                    exchange=exchange,
                    symbol=symbol,
                    bid=bid_price,
                    ask=ask_price,
                    last=mid_price,
                    volume=volume,
                    timestamp=current_timestamp
                )
                market_data.tickers[exchange] = ticker
                
                # Добавляем данные стакана ордеров
                from src.data_fetching.exchange_connector import OrderBookData
                
                # Генерируем несколько уровней цен для стакана
                bids = []
                asks = []
                
                # Генерируем 5 уровней цен для покупки (bid)
                for i in range(5):
                    price = bid_price * (1 - i * 0.001)  # Каждый уровень немного ниже
                    volume = random.uniform(0.1, 1.0)
                    bids.append([price, volume])
                
                # Генерируем 5 уровней цен для продажи (ask)
                for i in range(5):
                    price = ask_price * (1 + i * 0.001)  # Каждый уровень немного выше
                    volume = random.uniform(0.1, 1.0)
                    asks.append([price, volume])
                
                order_book = OrderBookData(
                    exchange=exchange,
                    symbol=symbol,
                    bids=bids,
                    asks=asks,
                    timestamp=current_timestamp
                )
                market_data.order_books[exchange] = order_book
            
            result.append(market_data)
        
        return result
