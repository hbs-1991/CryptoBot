"""
Базовый класс для подключения к криптовалютным биржам через библиотеку CCXT.
Предоставляет унифицированный интерфейс для работы с разными биржами.
"""

import time
import asyncio
import logging
import platform
from typing import Dict, List, Optional, Any, Union, Tuple
from functools import wraps

import ccxt
import ccxt.async_support as ccxt_async
from pydantic import BaseModel

from config.exchanges import ExchangeSettings

# Решение проблемы с событийным циклом на Windows
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class ExchangeConnectionError(Exception):
    """Исключение, возникающее при проблемах с подключением к бирже."""
    pass


class OrderBookData(BaseModel):
    """Модель для данных стакана ордеров (order book)."""
    exchange: str
    symbol: str
    asks: List[List[float]]  # [[цена, объем], ...]
    bids: List[List[float]]  # [[цена, объем], ...]
    timestamp: int


class TickerData(BaseModel):
    """Модель для данных тикера."""
    exchange: str
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    timestamp: int


def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """
    Декоратор для повторных попыток выполнения функции при возникновении ошибок.
    
    Args:
        max_retries: Максимальное количество повторных попыток
        delay: Задержка между попытками в секундах
        
    Returns:
        Результат выполнения функции
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    logging.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}. "
                        f"Retrying in {delay} seconds..."
                    )
                    await asyncio.sleep(delay)
            raise ExchangeConnectionError(f"Function {func.__name__} failed after {max_retries} attempts: {last_error}")
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    logging.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}. "
                        f"Retrying in {delay} seconds..."
                    )
                    time.sleep(delay)
            raise ExchangeConnectionError(f"Function {func.__name__} failed after {max_retries} attempts: {last_error}")
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


class ExchangeConnector:
    """
    Базовый класс для подключения к криптовалютным биржам.
    Реализует общие методы для всех бирж через библиотеку CCXT.
    """
    
    def __init__(self, settings: ExchangeSettings, use_async: bool = True):
        """
        Инициализация коннектора к бирже.
        
        Args:
            settings: Настройки подключения к бирже
            use_async: Использовать ли асинхронный режим работы
        """
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        self.exchange_id = settings.name
        self.use_async = use_async
        self.max_retries = settings.retries
        self._exchange = None
        self._markets = None
        self._symbols = None
        
        # Инициализация подключения
        self._init_exchange()
    
    def _init_exchange(self) -> None:
        """Инициализирует подключение к бирже."""
        try:
            ccxt_config = self.settings.to_ccxt_config()
            
            if self.use_async:
                # Динамически получаем класс биржи из модуля async_support
                exchange_class = getattr(ccxt_async, self.exchange_id)
            else:
                # Динамически получаем класс биржи из основного модуля
                exchange_class = getattr(ccxt, self.exchange_id)
            
            self._exchange = exchange_class(ccxt_config)
            self.logger.info(f"Initialized connector for {self.exchange_id.upper()}")
        except Exception as e:
            error_msg = f"Failed to initialize exchange {self.exchange_id}: {str(e)}"
            self.logger.error(error_msg)
            raise ExchangeConnectionError(error_msg)
    
    async def close(self) -> None:
        """Закрывает соединение с биржей."""
        if self._exchange and self.use_async:
            await self._exchange.close()
            self.logger.info(f"Closed connection to {self.exchange_id.upper()}")
    
    @retry_on_error()
    async def load_markets(self) -> Dict[str, Any]:
        """
        Загружает информацию о доступных рынках.
        
        Returns:
            Словарь с информацией о рынках
        """
        if self._markets is None:
            if self.use_async:
                self._markets = await self._exchange.load_markets()
            else:
                self._markets = self._exchange.load_markets()
            
            # Сохраняем список доступных символов
            self._symbols = list(self._markets.keys())
            self.logger.info(
                f"Loaded {len(self._symbols)} markets for {self.exchange_id.upper()}"
            )
        
        return self._markets
    
    @property
    def symbols(self) -> List[str]:
        """
        Возвращает список доступных торговых пар.
        
        Returns:
            Список торговых пар
        """
        if self._symbols is None:
            if self.use_async:
                asyncio.run(self.load_markets())
            else:
                self.load_markets()
        return self._symbols
    
    @retry_on_error()
    async def fetch_ticker(self, symbol: str) -> TickerData:
        """
        Получает данные тикера для указанной торговой пары.
        
        Args:
            symbol: Торговая пара (например, 'BTC/USDT')
            
        Returns:
            Данные тикера
        """
        if self.use_async:
            ticker = await self._exchange.fetch_ticker(symbol)
        else:
            ticker = self._exchange.fetch_ticker(symbol)
        
        return TickerData(
            exchange=self.exchange_id,
            symbol=symbol,
            bid=float(ticker.get('bid', 0)),
            ask=float(ticker.get('ask', 0)),
            last=float(ticker.get('last', 0)),
            volume=float(ticker.get('volume', 0)),
            timestamp=int(ticker.get('timestamp', 0))
        )
    
    @retry_on_error()
    async def fetch_order_book(self, symbol: str, limit: int = 20) -> OrderBookData:
        """
        Получает стакан ордеров для указанной торговой пары.
        
        Args:
            symbol: Торговая пара (например, 'BTC/USDT')
            limit: Глубина стакана (количество ордеров)
            
        Returns:
            Данные стакана ордеров
        """
        if self.use_async:
            order_book = await self._exchange.fetch_order_book(symbol, limit)
        else:
            order_book = self._exchange.fetch_order_book(symbol, limit)
        
        # Используем текущее время, если timestamp не предоставлен
        timestamp = order_book.get('timestamp')
        if timestamp is None:
            timestamp = int(time.time() * 1000)  # в миллисекундах
        
        return OrderBookData(
            exchange=self.exchange_id,
            symbol=symbol,
            asks=order_book.get('asks', []),
            bids=order_book.get('bids', []),
            timestamp=timestamp
        )
    
    @retry_on_error()
    async def fetch_balance(self) -> Dict[str, Any]:
        """
        Получает баланс аккаунта.
        
        Returns:
            Информация о балансе
        """
        if self.use_async:
            balance = await self._exchange.fetch_balance()
        else:
            balance = self._exchange.fetch_balance()
        
        return balance
    
    def is_market_available(self, symbol: str) -> bool:
        """
        Проверяет, доступна ли торговая пара на бирже.
        
        Args:
            symbol: Торговая пара для проверки
            
        Returns:
            True, если пара доступна, иначе False
        """
        return symbol in self.symbols
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """
        Возвращает общую информацию о бирже.
        
        Returns:
            Информация о бирже
        """
        return {
            "id": self.exchange_id,
            "name": self._exchange.name if self._exchange else self.exchange_id,
            "rate_limit": self.settings.rate_limit,
            "timeout": self.settings.timeout,
            "use_testnet": self.settings.use_testnet,
            "symbols_count": len(self.symbols) if self._symbols else 0
        }
    
    async def ping(self) -> Tuple[bool, float]:
        """
        Проверяет доступность биржи и измеряет время отклика.
        Для обхода проблем с aiodns на Windows используем синхронный запрос.
        
        Returns:
            Кортеж (успех, время_отклика)
        """
        start_time = time.time()
        try:
            # Создаем экземпляр синхронного клиента для проверки соединения
            # Это позволяет избежать проблем с aiodns на Windows
            ccxt_config = self.settings.to_ccxt_config()
            
            # Для KuCoin и OKX явно указываем passphrase в параметре password
            if self.exchange_id.lower() in ['kucoin', 'okx'] and self.settings.credentials.passphrase:
                ccxt_config['password'] = self.settings.credentials.passphrase
                self.logger.debug(f"Using passphrase as password for {self.exchange_id.upper()}")
            
            exchange_class = getattr(ccxt, self.exchange_id)
            sync_exchange = exchange_class(ccxt_config)
            
            # Выполняем простой запрос для проверки соединения
            sync_exchange.fetch_ticker('BTC/USDT')
            
            response_time = time.time() - start_time
            self.logger.debug(f"Ping to {self.exchange_id} successful: {response_time:.3f}s")
            return True, response_time
        except Exception as e:
            response_time = time.time() - start_time
            self.logger.warning(f"Ping to {self.exchange_id} failed: {self.exchange_id} {str(e)}")
            return False, response_time
