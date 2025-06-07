"""
Фабрика для создания коннекторов к биржам.
Позволяет получить конкретную реализацию коннектора на основе имени биржи.
"""

import logging
from typing import Dict, Optional, List, Type

from config.exchanges import ExchangeSettings, ExchangeConfig
from src.data_fetching.exchange_connector import ExchangeConnector, ExchangeConnectionError
from src.data_fetching.binance_connector import BinanceConnector
from src.data_fetching.kucoin_connector import KuCoinConnector
from src.data_fetching.okx_connector import OKXConnector


class ExchangeFactory:
    """
    Фабрика для создания и управления подключениями к биржам.
    Реализует паттерн "Фабричный метод" для создания конкретных коннекторов.
    """
    
    # Словарь для маппинга имени биржи на соответствующий класс коннектора
    CONNECTOR_CLASSES = {
        'binance': BinanceConnector,
        'kucoin': KuCoinConnector,
        'okx': OKXConnector
    }
    
    def __init__(self, exchange_config: ExchangeConfig, use_async: bool = True):
        """
        Инициализация фабрики коннекторов к биржам.
        
        Args:
            exchange_config: Конфигурация бирж
            use_async: Использовать ли асинхронный режим работы
        """
        self.logger = logging.getLogger(__name__)
        self.exchange_config = exchange_config
        self.use_async = use_async
        self._connectors: Dict[str, ExchangeConnector] = {}
    
    def get_connector(self, exchange_name: str) -> ExchangeConnector:
        """
        Получает коннектор для указанной биржи.
        Если коннектор уже создан, возвращает существующий экземпляр.
        
        Args:
            exchange_name: Имя биржи ('binance', 'kucoin', 'okx')
            
        Returns:
            Коннектор к бирже
            
        Raises:
            ValueError: Если биржа не поддерживается или не настроена
        """
        exchange_name = exchange_name.lower()
        
        # Проверяем, что биржа поддерживается
        if exchange_name not in self.CONNECTOR_CLASSES:
            raise ValueError(f"Exchange {exchange_name} is not supported")
        
        # Если коннектор уже создан, возвращаем его
        if exchange_name in self._connectors:
            return self._connectors[exchange_name]
        
        # Получаем настройки для биржи
        exchange_settings = self.exchange_config.get_exchange(exchange_name)
        if not exchange_settings:
            raise ValueError(f"Exchange {exchange_name} is not configured")
        
        # Проверяем, что биржа настроена и доступна
        if not self.exchange_config.is_exchange_configured(exchange_name):
            raise ValueError(f"Exchange {exchange_name} is not properly configured (API keys missing)")
        
        # Получаем класс коннектора
        connector_class = self.CONNECTOR_CLASSES[exchange_name]
        
        try:
            # Создаем новый коннектор
            connector = connector_class(exchange_settings, self.use_async)
            self._connectors[exchange_name] = connector
            self.logger.info(f"Created connector for {exchange_name.upper()}")
            return connector
        except Exception as e:
            error_msg = f"Failed to create connector for {exchange_name}: {str(e)}"
            self.logger.error(error_msg)
            raise ExchangeConnectionError(error_msg) from e
    
    def get_all_connectors(self) -> Dict[str, ExchangeConnector]:
        """
        Создает и возвращает коннекторы для всех настроенных бирж.
        
        Returns:
            Словарь с коннекторами, где ключ - имя биржи
        """
        result = {}
        for exchange in self.exchange_config.get_enabled_exchanges():
            try:
                connector = self.get_connector(exchange.name)
                result[exchange.name] = connector
            except Exception as e:
                self.logger.error(f"Failed to create connector for {exchange.name}: {str(e)}")
        
        return result
    
    async def close_all_connections(self) -> None:
        """
        Закрывает все открытые соединения с биржами.
        """
        for exchange_name, connector in self._connectors.items():
            try:
                if self.use_async:
                    await connector.close()
                self.logger.info(f"Closed connection to {exchange_name.upper()}")
            except Exception as e:
                self.logger.warning(f"Error closing connection to {exchange_name}: {str(e)}")
        
        # Очищаем словарь коннекторов
        self._connectors.clear()
    
    def get_supported_exchanges(self) -> List[str]:
        """
        Возвращает список поддерживаемых бирж.
        
        Returns:
            Список имен поддерживаемых бирж
        """
        return list(self.CONNECTOR_CLASSES.keys()) 