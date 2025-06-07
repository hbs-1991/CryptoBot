"""
Модуль для исполнения торговых операций.
Обеспечивает унифицированный интерфейс для реальной и симулированной торговли.
"""

import logging
from typing import Dict, Optional, Any, Union
from decimal import Decimal
from datetime import datetime
from abc import ABC, abstractmethod

from config import settings
from src.arbitrage_engine.opportunity_finder import ArbitrageOpportunity
from src.simulation.simulation import TradeSimulator, VirtualOrder


class OrderResult:
    """Результат исполнения ордера."""
    def __init__(
        self, 
        success: bool, 
        order_id: Optional[str] = None, 
        error: Optional[str] = None,
        order_data: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.order_id = order_id
        self.error = error
        self.order_data = order_data or {}
    
    @property
    def is_successful(self) -> bool:
        """Проверяет, успешно ли выполнен ордер."""
        return self.success and self.order_id is not None


class TradeExecutor(ABC):
    """
    Базовый класс для исполнения торговых операций.
    Определяет интерфейс для работы с ордерами и сделками.
    """
    
    def __init__(self):
        """Инициализация исполнителя торговых операций."""
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    async def create_order(
        self, 
        exchange: str, 
        symbol: str, 
        order_type: str, 
        price: Union[float, Decimal], 
        amount: Union[float, Decimal]
    ) -> OrderResult:
        """
        Создает ордер на указанной бирже.
        
        Args:
            exchange: Название биржи
            symbol: Торговая пара
            order_type: Тип ордера ("buy" или "sell")
            price: Цена
            amount: Объем
            
        Returns:
            Результат создания ордера
        """
        pass
    
    @abstractmethod
    async def check_order_status(self, exchange: str, order_id: str) -> Dict[str, Any]:
        """
        Проверяет статус ордера.
        
        Args:
            exchange: Название биржи
            order_id: ID ордера
            
        Returns:
            Информация об ордере
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, exchange: str, order_id: str) -> bool:
        """
        Отменяет ордер.
        
        Args:
            exchange: Название биржи
            order_id: ID ордера
            
        Returns:
            Успешность отмены
        """
        pass
    
    @abstractmethod
    async def get_balance(self, exchange: str, currency: str) -> Decimal:
        """
        Получает баланс валюты на бирже.
        
        Args:
            exchange: Название биржи
            currency: Валюта
            
        Returns:
            Баланс валюты
        """
        pass
    
    @abstractmethod
    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity) -> Dict[str, Any]:
        """
        Исполняет арбитражную возможность.
        
        Args:
            opportunity: Арбитражная возможность
            
        Returns:
            Результат исполнения
        """
        pass


class SimulationTradeExecutor(TradeExecutor):
    """
    Исполнитель торговых операций в режиме симуляции.
    Использует TradeSimulator для виртуальной торговли.
    """
    
    def __init__(
        self, 
        initial_balances: Dict[str, Dict[str, float]] = None,
        exchange_fees: Dict[str, float] = None
    ):
        """
        Инициализация исполнителя торговых операций в режиме симуляции.
        
        Args:
            initial_balances: Начальные балансы по биржам и валютам
            exchange_fees: Комиссии бирж в процентах
        """
        super().__init__()
        
        # Если балансы не указаны, используем стандартные для тестирования
        if not initial_balances:
            initial_balances = {
                "binance": {"USDT": 10000.0, "BTC": 0.5, "ETH": 5.0},
                "kucoin": {"USDT": 10000.0, "BTC": 0.5, "ETH": 5.0},
                "okx": {"USDT": 10000.0, "BTC": 0.5, "ETH": 5.0}
            }
        
        self.simulator = TradeSimulator(
            initial_balances=initial_balances,
            exchange_fees=exchange_fees
        )
        
        self.logger.info("SimulationTradeExecutor initialized in simulation mode")
    
    async def create_order(
        self, 
        exchange: str, 
        symbol: str, 
        order_type: str, 
        price: Union[float, Decimal], 
        amount: Union[float, Decimal]
    ) -> OrderResult:
        """
        Создает виртуальный ордер.
        
        Args:
            exchange: Название биржи
            symbol: Торговая пара
            order_type: Тип ордера ("buy" или "sell")
            price: Цена
            amount: Объем
            
        Returns:
            Результат создания ордера
        """
        self.logger.info(
            f"Creating {order_type} order on {exchange}: {amount} {symbol} @ {price}"
        )
        
        try:
            order = await self.simulator.create_order(
                exchange=exchange,
                symbol=symbol,
                order_type=order_type,
                price=price,
                amount=amount
            )
            
            if order:
                return OrderResult(
                    success=True,
                    order_id=order.id,
                    order_data=order.dict()
                )
            else:
                return OrderResult(
                    success=False,
                    error="Failed to create order (insufficient balance)"
                )
        except Exception as e:
            self.logger.error(f"Error creating order: {str(e)}")
            return OrderResult(
                success=False,
                error=f"Error: {str(e)}"
            )
    
    async def check_order_status(self, exchange: str, order_id: str) -> Dict[str, Any]:
        """
        Проверяет статус виртуального ордера.
        
        Args:
            exchange: Название биржи
            order_id: ID ордера
            
        Returns:
            Информация об ордере
        """
        order = self.simulator.get_order(order_id)
        if not order:
            return {
                "found": False,
                "error": f"Order {order_id} not found"
            }
        
        return {
            "found": True,
            "order_id": order.id,
            "exchange": order.exchange,
            "symbol": order.symbol,
            "type": order.order_type,
            "price": float(order.price),
            "amount": float(order.amount),
            "filled_amount": float(order.filled_amount),
            "status": order.status,
            "is_filled": order.is_filled,
            "fee": float(order.fee)
        }
    
    async def cancel_order(self, exchange: str, order_id: str) -> bool:
        """
        Отменяет виртуальный ордер (симуляция).
        
        Args:
            exchange: Название биржи
            order_id: ID ордера
            
        Returns:
            Успешность отмены
        """
        order = self.simulator.get_order(order_id)
        if not order or order.exchange != exchange:
            self.logger.warning(f"Order {order_id} not found on {exchange}")
            return False
        
        if order.status != "open":
            self.logger.warning(
                f"Cannot cancel order {order_id} with status {order.status}"
            )
            return False
        
        # Обновляем статус ордера
        order.status = "canceled"
        order.updated_at = int(datetime.now().timestamp() * 1000)
        
        self.logger.info(f"Canceled order {order_id} on {exchange}")
        return True
    
    async def get_balance(self, exchange: str, currency: str) -> Decimal:
        """
        Получает виртуальный баланс валюты на бирже.
        
        Args:
            exchange: Название биржи
            currency: Валюта
            
        Returns:
            Баланс валюты
        """
        return self.simulator.get_balance(exchange, currency)
    
    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity) -> Dict[str, Any]:
        """
        Исполняет арбитражную возможность в режиме симуляции.
        
        Args:
            opportunity: Арбитражная возможность
            
        Returns:
            Результат исполнения
        """
        self.logger.info(
            f"Executing arbitrage: Buy {opportunity.symbol} on {opportunity.buy_exchange} @ {opportunity.buy_price}, "
            f"Sell on {opportunity.sell_exchange} @ {opportunity.sell_price}"
        )
        
        try:
            trade = await self.simulator.execute_arbitrage_opportunity(opportunity)
            
            if not trade:
                return {
                    "success": False,
                    "error": "Failed to execute arbitrage opportunity"
                }
            
            if trade.status == "completed":
                return {
                    "success": True,
                    "trade_id": trade.id,
                    "buy_order_id": trade.buy_order_id,
                    "sell_order_id": trade.sell_order_id,
                    "symbol": trade.symbol,
                    "buy_exchange": trade.buy_exchange,
                    "sell_exchange": trade.sell_exchange,
                    "amount": float(trade.amount),
                    "expected_profit": float(trade.expected_profit),
                    "actual_profit": float(trade.actual_profit),
                    "execution_time_ms": trade.end_time - trade.start_time
                }
            else:
                return {
                    "success": False,
                    "trade_id": trade.id,
                    "error": trade.error or f"Trade failed with status {trade.status}"
                }
        except Exception as e:
            self.logger.error(f"Error executing arbitrage: {str(e)}")
            return {
                "success": False,
                "error": f"Error: {str(e)}"
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Получает статистику симуляции.
        
        Returns:
            Статистика симуляции
        """
        return self.simulator.get_statistics()
    
    def generate_report(self) -> str:
        """
        Генерирует отчет по результатам симуляции.
        
        Returns:
            Текстовый отчет
        """
        return self.simulator.generate_report()
    
    def reset_simulation(self, initial_balances: Dict[str, Dict[str, float]] = None) -> None:
        """
        Сбрасывает симуляцию.
        
        Args:
            initial_balances: Новые начальные балансы
        """
        self.simulator.reset(initial_balances)
        self.logger.info("Simulation reset")


# Фабричный метод для создания исполнителя торговых операций
def create_trade_executor() -> TradeExecutor:
    """
    Создает исполнителя торговых операций в зависимости от режима.
    
    Returns:
        Исполнитель торговых операций
    """
    if settings.SIMULATION_MODE:
        return SimulationTradeExecutor()
    else:
        # В будущем здесь будет создание реального исполнителя
        # return RealTradeExecutor()
        # Пока используем симуляцию
        return SimulationTradeExecutor()
