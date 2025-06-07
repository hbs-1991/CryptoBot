"""
Сканер рыночных данных.
Отвечает за асинхронное получение и мониторинг данных с разных бирж.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime
from pydantic import BaseModel, Field

from config import settings
from src.data_fetching.exchange_factory import ExchangeFactory
from src.data_fetching.exchange_connector import ExchangeConnector, TickerData, OrderBookData, ExchangeConnectionError

# Импортируем симулятор рыночных данных для режима симуляции
try:
    from src.simulation.market_data_simulator import MarketDataSimulator
except ImportError:
    MarketDataSimulator = None


class MarketData(BaseModel):
    """Модель для хранения рыночных данных по одному символу со всех бирж."""
    symbol: str
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))
    tickers: Dict[str, TickerData] = Field(default_factory=dict)
    order_books: Dict[str, OrderBookData] = Field(default_factory=dict)
    
    @property
    def exchanges(self) -> List[str]:
        """Возвращает список бирж, с которых получены данные."""
        return list(set(self.tickers.keys()) | set(self.order_books.keys()))
    
    @property
    def is_complete(self) -> bool:
        """Проверяет, получены ли данные со всех бирж."""
        return len(self.tickers) > 0 and len(self.order_books) > 0


class ScanResult(BaseModel):
    """Результат сканирования рынка."""
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))
    data: Dict[str, MarketData] = Field(default_factory=dict)
    errors: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    
    def add_ticker(self, ticker: TickerData) -> None:
        """
        Добавляет данные тикера в результат сканирования.
        
        Args:
            ticker: Данные тикера
        """
        symbol = ticker.symbol
        if symbol not in self.data:
            self.data[symbol] = MarketData(symbol=symbol)
        
        self.data[symbol].tickers[ticker.exchange] = ticker
    
    def add_order_book(self, order_book: OrderBookData) -> None:
        """
        Добавляет данные стакана ордеров в результат сканирования.
        
        Args:
            order_book: Данные стакана ордеров
        """
        symbol = order_book.symbol
        if symbol not in self.data:
            self.data[symbol] = MarketData(symbol=symbol)
        
        self.data[symbol].order_books[order_book.exchange] = order_book
    
    def add_error(self, exchange: str, symbol: str, error_msg: str) -> None:
        """
        Добавляет информацию об ошибке при сканировании.
        
        Args:
            exchange: Биржа, на которой произошла ошибка
            symbol: Торговая пара, при сканировании которой произошла ошибка
            error_msg: Сообщение об ошибке
        """
        if exchange not in self.errors:
            self.errors[exchange] = {}
        
        self.errors[exchange][symbol] = error_msg


class MarketScanner:
    """
    Сканер рыночных данных для получения и мониторинга данных с разных бирж.
    """
    
    def __init__(
        self, 
        exchange_factory: ExchangeFactory,
        symbols: Optional[List[str]] = None,
        update_interval: float = 1.0,
        order_book_depth: int = 20,
        max_workers: int = 10,
        use_simulation: bool = False
    ):
        """
        Инициализация сканера рыночных данных.
        
        Args:
            exchange_factory: Фабрика для создания коннекторов к биржам
            symbols: Список торговых пар для сканирования (если None, используются из конфигурации)
            update_interval: Интервал обновления данных в секундах
            order_book_depth: Глубина стакана ордеров
            max_workers: Максимальное количество параллельных задач
            use_simulation: Использовать симулятор рыночных данных вместо реальных данных
        """
        self.logger = logging.getLogger(__name__)
        self.exchange_factory = exchange_factory
        self.symbols = symbols or settings.ALLOWED_PAIRS
        self.update_interval = update_interval
        self.order_book_depth = order_book_depth
        self.max_workers = max_workers
        self.use_simulation = use_simulation or settings.SIMULATION_MODE
        
        self._stop_event = asyncio.Event()
        self._last_result: Optional[ScanResult] = None
        self._connectors: Dict[str, ExchangeConnector] = {}
        
        # Инициализируем симулятор рыночных данных, если режим симуляции активен
        self._market_simulator = None
        if self.use_simulation and MarketDataSimulator is not None:
            exchange_names = [ex.name for ex in self.exchange_factory.get_exchange_settings()]
            self._market_simulator = MarketDataSimulator(
                symbols=self.symbols,
                exchanges=exchange_names
            )
            self.logger.info(
                f"Initialized market data simulator for {len(self.symbols)} symbols "
                f"and {len(exchange_names)} exchanges"
            )
        
        self.logger.info(
            f"MarketScanner initialized with {len(self.symbols)} symbols, "
            f"update interval: {update_interval}s"
        )
    
    async def _initialize_connectors(self) -> None:
        """Инициализирует коннекторы для всех бирж."""
        try:
            # Получаем все коннекторы через фабрику
            self._connectors = await self._get_available_connectors()
            
            # Проверяем, что есть хотя бы один рабочий коннектор
            if not self._connectors:
                self.logger.error("No exchange connectors available. Cannot scan market.")
                return
            
            # Загружаем информацию о рынках для всех бирж
            await asyncio.gather(*[
                connector.load_markets() for connector in self._connectors.values()
            ])
            
            # Логируем доступные коннекторы
            self.logger.info(f"Initialized {len(self._connectors)} exchange connectors")
            for exchange, connector in self._connectors.items():
                self.logger.info(f"Connector for {exchange.upper()} has {len(connector.symbols)} symbols")
        
        except Exception as e:
            self.logger.error(f"Error initializing connectors: {str(e)}")
            raise
    
    async def _get_available_connectors(self) -> Dict[str, ExchangeConnector]:
        """
        Получает доступные коннекторы к биржам.
        
        Returns:
            Словарь коннекторов к доступным биржам
        """
        result = {}
        all_connectors = self.exchange_factory.get_all_connectors()
        
        # Проверяем доступность каждой биржи
        for name, connector in all_connectors.items():
            try:
                available, ping_time = await connector.ping()
                if available:
                    self.logger.info(f"Exchange {name.upper()} is available (ping: {ping_time:.3f}s)")
                    result[name] = connector
                else:
                    self.logger.warning(f"Exchange {name.upper()} is not available")
            except Exception as e:
                self.logger.error(f"Error checking exchange {name.upper()}: {str(e)}")
        
        return result
    
    async def _fetch_ticker(self, exchange: str, connector: ExchangeConnector, symbol: str) -> Optional[TickerData]:
        """
        Получает данные тикера для указанной биржи и торговой пары.
        
        Args:
            exchange: Имя биржи
            connector: Коннектор к бирже
            symbol: Торговая пара
            
        Returns:
            Данные тикера или None в случае ошибки
        """
        try:
            if not connector.is_market_available(symbol):
                self.logger.warning(f"Symbol {symbol} is not available on {exchange.upper()}")
                return None
            
            ticker = await connector.fetch_ticker(symbol)
            return ticker
        
        except ExchangeConnectionError as e:
            self.logger.error(f"Connection error fetching ticker for {symbol} on {exchange.upper()}: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching ticker for {symbol} on {exchange.upper()}: {str(e)}")
            return None
    
    async def _fetch_order_book(self, exchange: str, connector: ExchangeConnector, symbol: str) -> Optional[OrderBookData]:
        """
        Получает стакан ордеров для указанной биржи и торговой пары.
        
        Args:
            exchange: Имя биржи
            connector: Коннектор к бирже
            symbol: Торговая пара
            
        Returns:
            Данные стакана ордеров или None в случае ошибки
        """
        try:
            if not connector.is_market_available(symbol):
                self.logger.warning(f"Symbol {symbol} is not available on {exchange.upper()}")
                return None
            
            order_book = await connector.fetch_order_book(symbol, self.order_book_depth)
            return order_book
        
        except ExchangeConnectionError as e:
            self.logger.error(f"Connection error fetching order book for {symbol} on {exchange.upper()}: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching order book for {symbol} on {exchange.upper()}: {str(e)}")
            return None
    
    async def _scan_symbol(self, symbol: str) -> None:
        """
        Сканирует все биржи для указанной торговой пары и обновляет результаты.
        
        Args:
            symbol: Торговая пара для сканирования
        """
        result = ScanResult()
        
        # Создаем задачи для получения данных со всех бирж параллельно
        ticker_tasks = []
        order_book_tasks = []
        
        for exchange, connector in self._connectors.items():
            # Создаем задачу для получения тикера
            ticker_tasks.append(
                asyncio.create_task(
                    self._fetch_ticker(exchange, connector, symbol)
                )
            )
            
            # Создаем задачу для получения стакана ордеров
            order_book_tasks.append(
                asyncio.create_task(
                    self._fetch_order_book(exchange, connector, symbol)
                )
            )
        
        # Ожидаем завершения всех задач
        ticker_results = await asyncio.gather(*ticker_tasks, return_exceptions=True)
        order_book_results = await asyncio.gather(*order_book_tasks, return_exceptions=True)
        
        # Обрабатываем результаты получения тикеров
        for i, ticker_result in enumerate(ticker_results):
            exchange = list(self._connectors.keys())[i]
            
            if isinstance(ticker_result, Exception):
                result.add_error(exchange, symbol, f"Ticker error: {str(ticker_result)}")
            elif ticker_result is not None:
                result.add_ticker(ticker_result)
        
        # Обрабатываем результаты получения стаканов ордеров
        for i, order_book_result in enumerate(order_book_results):
            exchange = list(self._connectors.keys())[i]
            
            if isinstance(order_book_result, Exception):
                result.add_error(exchange, symbol, f"Order book error: {str(order_book_result)}")
            elif order_book_result is not None:
                result.add_order_book(order_book_result)
        
        # Обновляем последний результат
        if not self._last_result:
            self._last_result = result
        else:
            # Объединяем данные с предыдущим результатом
            self._last_result.data[symbol] = result.data.get(symbol)
            
            # Добавляем новые ошибки
            for exchange, errors in result.errors.items():
                if exchange not in self._last_result.errors:
                    self._last_result.errors[exchange] = {}
                
                for symbol, error in errors.items():
                    self._last_result.errors[exchange][symbol] = error
            
            # Обновляем временную метку
            self._last_result.timestamp = result.timestamp
    
    async def scan_all_symbols(self) -> ScanResult:
        """
        Сканирует все биржи для всех указанных торговых пар.
        
        Returns:
            Результат сканирования
        """
        # Инициализируем коннекторы, если это еще не сделано
        if not self._connectors:
            await self._initialize_connectors()
        
        # Создаем семафор для ограничения числа параллельных задач
        semaphore = asyncio.Semaphore(self.max_workers)
        
        # Функция-обертка для выполнения задачи с семафором
        async def scan_with_semaphore(symbol: str):
            async with semaphore:
                await self._scan_symbol(symbol)
        
        # Запускаем задачи сканирования всех символов параллельно
        tasks = [
            asyncio.create_task(scan_with_semaphore(symbol)) 
            for symbol in self.symbols
        ]
        
        # Ожидаем завершения всех задач
        await asyncio.gather(*tasks)
        
        # Возвращаем результат сканирования
        if not self._last_result:
            return ScanResult()
        
        return self._last_result
    
    async def start_continuous_scanning(self) -> None:
        """
        Запускает непрерывное сканирование рынка в фоновом режиме.
        """
        self.logger.info("Starting continuous market scanning")
        self._stop_event.clear()
        
        # Если используем симулятор, не нужно инициализировать реальные коннекторы
        if not self.use_simulation or self._market_simulator is None:
            # Инициализируем коннекторы
            await self._initialize_connectors()
        
        # Запускаем цикл непрерывного сканирования
        while not self._stop_event.is_set():
            try:
                start_time = time.time()
                
                # Если используем симулятор, генерируем данные
                if self.use_simulation and self._market_simulator is not None:
                    market_data = self._market_simulator.generate_market_data()
                    
                    # Создаем новый результат сканирования
                    result = ScanResult()
                    for data in market_data:
                        result.data[data.symbol] = data
                    
                    self._last_result = result
                    
                else:
                    # Сканируем все символы на реальных биржах
                    await self.scan_all_symbols()
                
                # Вычисляем, сколько времени нужно подождать до следующего сканирования
                elapsed = time.time() - start_time
                wait_time = max(0, self.update_interval - elapsed)
                
                self.logger.debug(
                    f"Scan completed in {elapsed:.3f}s, "
                    f"waiting {wait_time:.3f}s until next scan"
                )
                
                # Ждем до следующего сканирования или до остановки
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=wait_time)
                except asyncio.TimeoutError:
                    pass  # Таймаут истек, продолжаем сканирование
            
            except Exception as e:
                self.logger.error(f"Error during continuous scanning: {str(e)}")
                await asyncio.sleep(self.update_interval)  # В случае ошибки ждем и пробуем снова
    
    def stop_scanning(self) -> None:
        """
        Останавливает непрерывное сканирование рынка.
        """
        self.logger.info("Stopping continuous market scanning")
        self._stop_event.set()
    
    async def close(self) -> None:
        """
        Закрывает все соединения и освобождает ресурсы.
        """
        self.stop_scanning()
        
        # Закрываем соединения только если не используем симулятор
        if not self.use_simulation or self._market_simulator is None:
            await self.exchange_factory.close_all_connections()
            
        self.logger.info("MarketScanner closed")
    
    @property
    def latest_result(self) -> Optional[ScanResult]:
        """Возвращает последний результат сканирования."""
        return self._last_result
    
    def get_best_prices(self, symbol: str) -> Dict[str, Dict[str, float]]:
        """
        Возвращает лучшие цены для указанной торговой пары на всех биржах.
        
        Args:
            symbol: Торговая пара
            
        Returns:
            Словарь с лучшими ценами для каждой биржи
        """
        result = {}
        
        if not self._last_result or symbol not in self._last_result.data:
            return result
        
        market_data = self._last_result.data[symbol]
        
        # Собираем данные из тикеров
        for exchange, ticker in market_data.tickers.items():
            result[exchange] = {
                "bid": ticker.bid,
                "ask": ticker.ask,
                "last": ticker.last
            }
        
        # Дополняем данными из стаканов ордеров, если доступны
        for exchange, order_book in market_data.order_books.items():
            if exchange not in result:
                result[exchange] = {}
            
            if order_book.bids:
                result[exchange]["top_bid"] = order_book.bids[0][0]
            
            if order_book.asks:
                result[exchange]["top_ask"] = order_book.asks[0][0]
        
        return result
        
    async def get_market_data(self) -> List[MarketData]:
        """
        Получает актуальные данные рынка.
        Если непрерывное сканирование не запущено, выполняет одиночное сканирование.
        
        Returns:
            Список объектов MarketData для всех торговых пар
        """
        # Если используем симулятор, генерируем данные с его помощью
        if self.use_simulation and self._market_simulator is not None:
            self.logger.debug("Generating simulated market data")
            return self._market_simulator.generate_market_data()
            
        # Иначе получаем реальные данные с бирж
        # Если непрерывное сканирование не запущено или нет данных, выполняем сканирование
        if not self._last_result:
            await self.scan_all_symbols()
        
        # Если данных всё равно нет, возвращаем пустой список
        if not self._last_result or not self._last_result.data:
            return []
        
        # Преобразуем словарь данных в список
        return list(self._last_result.data.values())
        
    def start(self) -> None:
        """
        Запускает непрерывное сканирование рынка в фоновом режиме.
        """
        # Создаем и запускаем задачу в фоновом режиме
        asyncio.create_task(self.start_continuous_scanning())
        self.logger.info("Market scanning started in background")
        
    def stop(self) -> None:
        """
        Останавливает непрерывное сканирование рынка.
        Алиас для метода stop_scanning().
        """
        self.stop_scanning()
    
