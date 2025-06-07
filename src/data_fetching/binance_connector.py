"""
Адаптер для подключения к бирже Binance через CCXT.
Реализует специфичные для Binance методы и особенности.
"""

import logging
from typing import Dict, List, Any, Optional

from config.exchanges import ExchangeSettings
from src.data_fetching.exchange_connector import ExchangeConnector, retry_on_error


class BinanceConnector(ExchangeConnector):
    """
    Адаптер для работы с биржей Binance.
    Расширяет базовый класс ExchangeConnector специфичными для Binance методами.
    """
    
    def __init__(self, settings: ExchangeSettings, use_async: bool = True):
        """
        Инициализация коннектора к Binance.
        
        Args:
            settings: Настройки подключения к бирже
            use_async: Использовать ли асинхронный режим работы
        """
        super().__init__(settings, use_async)
        self.logger = logging.getLogger(__name__)
        
        # Проверяем, что это действительно Binance
        if settings.name.lower() != 'binance':
            raise ValueError("Settings provided are not for Binance exchange")
    
    @retry_on_error()
    async def fetch_trading_fees(self) -> Dict[str, Any]:
        """
        Получает информацию о торговых комиссиях на Binance.
        
        Returns:
            Словарь с информацией о комиссиях
        """
        if self.use_async:
            trading_fees = await self._exchange.fetch_trading_fee()
        else:
            trading_fees = self._exchange.fetch_trading_fee()
        
        return trading_fees
    
    @retry_on_error()
    async def fetch_funding_rates(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Получает ставки финансирования для фьючерсных контрактов.
        
        Args:
            symbols: Список торговых пар для получения ставок
            
        Returns:
            Словарь со ставками финансирования для каждой пары
        """
        if not self._exchange.has['fetchFundingRates']:
            self.logger.warning("Binance API does not support fetchFundingRates method")
            return {}
        
        if self.use_async:
            rates = await self._exchange.fetch_funding_rates(symbols)
        else:
            rates = self._exchange.fetch_funding_rates(symbols)
        
        return rates
    
    @retry_on_error()
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> List[List[float]]:
        """
        Получает исторические данные OHLCV (свечи) для указанной торговой пары.
        
        Args:
            symbol: Торговая пара (например, 'BTC/USDT')
            timeframe: Временной интервал ('1m', '5m', '15m', '1h', '4h', '1d', etc.)
            limit: Количество свечей для получения
            
        Returns:
            Список свечей в формате [[timestamp, open, high, low, close, volume], ...]
        """
        if self.use_async:
            ohlcv = await self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        else:
            ohlcv = self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        return ohlcv
    
    def get_binance_specific_info(self) -> Dict[str, Any]:
        """
        Возвращает специфичную для Binance информацию.
        
        Returns:
            Словарь с дополнительной информацией
        """
        info = self.get_exchange_info()
        
        # Добавляем специфичную для Binance информацию
        info.update({
            "supports_futures": True,
            "supports_margin": True,
            "supports_spot": True,
            "has_websocket": True,
            "has_rate_limits": True
        })
        
        return info 