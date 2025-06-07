"""
Базовый абстрактный класс для всех нотификаторов.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union


class BaseNotifier(ABC):
    """
    Абстрактный базовый класс для всех нотификаторов.
    Определяет общий интерфейс для отправки разных типов уведомлений.
    """
    
    @abstractmethod
    async def send_message(self, message: str) -> bool:
        """
        Отправляет простое текстовое сообщение.
        
        Args:
            message: Текст сообщения для отправки
            
        Returns:
            True в случае успешной отправки, иначе False
        """
        pass
    
    @abstractmethod
    async def send_alert(self, title: str, message: str, level: str = "info") -> bool:
        """
        Отправляет сообщение-оповещение с уровнем важности.
        
        Args:
            title: Заголовок сообщения
            message: Текст сообщения для отправки
            level: Уровень важности (info, warning, error, critical)
            
        Returns:
            True в случае успешной отправки, иначе False
        """
        pass
    
    @abstractmethod
    async def send_trade_info(
        self, 
        exchange: str, 
        symbol: str, 
        operation: str, 
        amount: float, 
        price: float, 
        status: str,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Отправляет информацию о торговой операции.
        
        Args:
            exchange: Название биржи
            symbol: Торговая пара
            operation: Тип операции (buy, sell)
            amount: Количество
            price: Цена
            status: Статус операции (executed, failed, pending)
            details: Дополнительные детали операции (опционально)
            
        Returns:
            True в случае успешной отправки, иначе False
        """
        pass
    
    @abstractmethod
    async def send_arbitrage_opportunity(
        self,
        buy_exchange: str,
        sell_exchange: str,
        symbol: str,
        buy_price: float,
        sell_price: float,
        profit_percent: float,
        estimated_profit: float,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Отправляет информацию об арбитражной возможности.
        
        Args:
            buy_exchange: Биржа для покупки
            sell_exchange: Биржа для продажи
            symbol: Торговая пара
            buy_price: Цена покупки
            sell_price: Цена продажи
            profit_percent: Процент прибыли
            estimated_profit: Оценочная прибыль в USD
            details: Дополнительные детали (опционально)
            
        Returns:
            True в случае успешной отправки, иначе False
        """
        pass
    
    @abstractmethod
    async def send_system_status(
        self,
        status: str,
        balances: Dict[str, Dict[str, float]],
        active_tasks: int,
        errors_count: int,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Отправляет статус системы.
        
        Args:
            status: Общий статус системы (running, error, stopped)
            balances: Словарь с балансами по биржам
            active_tasks: Количество активных задач
            errors_count: Количество ошибок
            details: Дополнительные детали (опционально)
            
        Returns:
            True в случае успешной отправки, иначе False
        """
        pass
