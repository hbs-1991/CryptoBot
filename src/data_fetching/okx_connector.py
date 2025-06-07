"""
Адаптер для подключения к бирже OKX через CCXT.
Реализует специфичные для OKX методы и особенности.
"""

import logging
from typing import Dict, List, Any, Optional

from config.exchanges import ExchangeSettings
from src.data_fetching.exchange_connector import ExchangeConnector, retry_on_error


class OKXConnector(ExchangeConnector):
    """
    Адаптер для работы с биржей OKX.
    Расширяет базовый класс ExchangeConnector специфичными для OKX методами.
    """
    
    def __init__(self, settings: ExchangeSettings, use_async: bool = True):
        """
        Инициализация коннектора к OKX.
        
        Args:
            settings: Настройки подключения к бирже
            use_async: Использовать ли асинхронный режим работы
        """
        super().__init__(settings, use_async)
        self.logger = logging.getLogger(__name__)
        
        # Проверяем, что это действительно OKX
        if settings.name.lower() != 'okx':
            raise ValueError("Settings provided are not for OKX exchange")
    
    @retry_on_error()
    async def fetch_funding_rate(self, symbol: str) -> Dict[str, Any]:
        """
        Получает текущую ставку финансирования для указанной торговой пары.
        
        Args:
            symbol: Торговая пара (например, 'BTC/USDT')
            
        Returns:
            Словарь с информацией о ставке финансирования
        """
        if not self._exchange.has['fetchFundingRate']:
            self.logger.warning("OKX API does not support fetchFundingRate method")
            return {}
        
        if self.use_async:
            funding_rate = await self._exchange.fetch_funding_rate(symbol)
        else:
            funding_rate = self._exchange.fetch_funding_rate(symbol)
        
        return funding_rate
    
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
    
    @retry_on_error()
    async def fetch_positions(self, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Получает открытые позиции на OKX.
        
        Args:
            symbols: Список торговых пар для получения позиций
            
        Returns:
            Список словарей с информацией о позициях
        """
        if not self._exchange.has['fetchPositions']:
            self.logger.warning("OKX API does not support fetchPositions method")
            return []
        
        if self.use_async:
            positions = await self._exchange.fetch_positions(symbols)
        else:
            positions = self._exchange.fetch_positions(symbols)
        
        return positions
    
    @retry_on_error()
    async def fetch_markets_by_type(self, market_type: str) -> Dict[str, Any]:
        """
        Получает информацию о рынках определенного типа на OKX.
        
        Args:
            market_type: Тип рынка ('spot', 'swap', 'futures', 'option')
            
        Returns:
            Словарь с информацией о рынках
        """
        if self.use_async:
            # Для OKX нужно установить тип рынка перед запросом
            self._exchange.options['defaultType'] = market_type
            markets = await self._exchange.load_markets(True)
        else:
            self._exchange.options['defaultType'] = market_type
            markets = self._exchange.load_markets(True)
        
        # Восстанавливаем defaultType
        self._exchange.options['defaultType'] = 'spot'
        
        return markets
    
    def get_okx_specific_info(self) -> Dict[str, Any]:
        """
        Возвращает специфичную для OKX информацию.
        
        Returns:
            Словарь с дополнительной информацией
        """
        info = self.get_exchange_info()
        
        # Добавляем специфичную для OKX информацию
        info.update({
            "supports_futures": True,
            "supports_swap": True,
            "supports_spot": True,
            "has_websocket": True,
            "requires_passphrase": True,
            "supports_options": True
        })
        
        return info 