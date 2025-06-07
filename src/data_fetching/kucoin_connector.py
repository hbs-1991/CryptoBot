"""
Адаптер для подключения к бирже KuCoin через CCXT.
Реализует специфичные для KuCoin методы и особенности.
"""

import logging
from typing import Dict, List, Any, Optional

from config.exchanges import ExchangeSettings
from src.data_fetching.exchange_connector import ExchangeConnector, retry_on_error


class KuCoinConnector(ExchangeConnector):
    """
    Адаптер для работы с биржей KuCoin.
    Расширяет базовый класс ExchangeConnector специфичными для KuCoin методами.
    """
    
    def __init__(self, settings: ExchangeSettings, use_async: bool = True):
        """
        Инициализация коннектора к KuCoin.
        
        Args:
            settings: Настройки подключения к бирже
            use_async: Использовать ли асинхронный режим работы
        """
        super().__init__(settings, use_async)
        self.logger = logging.getLogger(__name__)
        
        # Проверяем, что это действительно KuCoin
        if settings.name.lower() != 'kucoin':
            raise ValueError("Settings provided are not for KuCoin exchange")
    
    @retry_on_error()
    async def fetch_currencies(self) -> Dict[str, Any]:
        """
        Получает информацию о доступных валютах на KuCoin.
        
        Returns:
            Словарь с информацией о валютах
        """
        if self.use_async:
            currencies = await self._exchange.fetch_currencies()
        else:
            currencies = self._exchange.fetch_currencies()
        
        return currencies
    
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
    async def fetch_trading_fees(self) -> Dict[str, Any]:
        """
        Получает информацию о торговых комиссиях на KuCoin.
        
        Returns:
            Словарь с информацией о комиссиях
        """
        if not self._exchange.has['fetchTradingFee']:
            self.logger.warning("KuCoin API does not support fetchTradingFee method directly")
            # Используем альтернативный метод или возвращаем фиксированные комиссии
            return {
                "maker": 0.001,  # 0.1%
                "taker": 0.001   # 0.1%
            }
        
        if self.use_async:
            trading_fees = await self._exchange.fetch_trading_fee()
        else:
            trading_fees = self._exchange.fetch_trading_fee()
        
        return trading_fees
    
    def get_kucoin_specific_info(self) -> Dict[str, Any]:
        """
        Возвращает специфичную для KuCoin информацию.
        
        Returns:
            Словарь с дополнительной информацией
        """
        info = self.get_exchange_info()
        
        # Добавляем специфичную для KuCoin информацию
        info.update({
            "supports_futures": True,
            "supports_margin": True,
            "supports_spot": True,
            "has_websocket": True,
            "requires_passphrase": True
        })
        
        return info 