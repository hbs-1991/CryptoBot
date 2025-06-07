"""
Модуль для управления ордерами на биржах.
Обеспечивает создание, отслеживание и закрытие ордеров на подключенных биржах.
"""

import os
import sys
import logging
import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from decimal import Decimal
from datetime import datetime, timedelta

# Добавляем корневую директорию проекта в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils import get_logger
from db.db_manager import DatabaseManager
from src.data_fetching.exchange_factory import ExchangeFactory
from src.trading.balance_manager import BalanceManager


class OrderManager:
    """
    Класс для управления ордерами на биржах.
    Создаёт, отслеживает и закрывает ордера.
    """

    def __init__(self, exchange_factory: ExchangeFactory, db_manager: DatabaseManager, 
                 order_repo, balance_manager: BalanceManager, order_ttl_seconds: int = 60):
        """
        Инициализирует менеджер ордеров.
        
        Args:
            exchange_factory: Фабрика для создания подключений к биржам
            db_manager: Менеджер базы данных
            order_repo: Репозиторий для работы с ордерами
            balance_manager: Менеджер балансов
            order_ttl_seconds: Время жизни ордера в секундах до принудительной отмены
        """
        self.logger = get_logger(__name__)
        self.exchange_factory = exchange_factory
        self.db_manager = db_manager
        self.order_repo = order_repo
        self.balance_manager = balance_manager
        self.order_ttl_seconds = order_ttl_seconds
        
        # Кэш открытых ордеров: {order_id: order_info}
        self.open_orders = {}
        # Кэш последних статусов ордеров: {order_id: status}
        self.order_status_cache = {}
        
        self.logger.info("Order manager initialized")

    async def create_market_buy_order(self, exchange_name: str, symbol: str, 
                                     amount: Decimal) -> Dict[str, Any]:
        """
        Создает рыночный ордер на покупку.
        
        Args:
            exchange_name: Имя биржи
            symbol: Торговая пара
            amount: Количество базовой валюты для покупки
            
        Returns:
            Информация о созданном ордере
        """
        return await self._create_order(exchange_name, symbol, "buy", "market", amount)
    
    async def create_market_sell_order(self, exchange_name: str, symbol: str, 
                                      amount: Decimal) -> Dict[str, Any]:
        """
        Создает рыночный ордер на продажу.
        
        Args:
            exchange_name: Имя биржи
            symbol: Торговая пара
            amount: Количество базовой валюты для продажи
            
        Returns:
            Информация о созданном ордере
        """
        return await self._create_order(exchange_name, symbol, "sell", "market", amount)
    
    async def create_limit_buy_order(self, exchange_name: str, symbol: str, 
                                    amount: Decimal, price: Decimal) -> Dict[str, Any]:
        """
        Создает лимитный ордер на покупку.
        
        Args:
            exchange_name: Имя биржи
            symbol: Торговая пара
            amount: Количество базовой валюты для покупки
            price: Цена покупки
            
        Returns:
            Информация о созданном ордере
        """
        return await self._create_order(exchange_name, symbol, "buy", "limit", amount, price)
    
    async def create_limit_sell_order(self, exchange_name: str, symbol: str, 
                                     amount: Decimal, price: Decimal) -> Dict[str, Any]:
        """
        Создает лимитный ордер на продажу.
        
        Args:
            exchange_name: Имя биржи
            symbol: Торговая пара
            amount: Количество базовой валюты для продажи
            price: Цена продажи
            
        Returns:
            Информация о созданном ордере
        """
        return await self._create_order(exchange_name, symbol, "sell", "limit", amount, price)
    
    async def _create_order(self, exchange_name: str, symbol: str, side: str, 
                           order_type: str, amount: Decimal, price: Optional[Decimal] = None) -> Dict[str, Any]:
        """
        Внутренний метод для создания ордера.
        
        Args:
            exchange_name: Имя биржи
            symbol: Торговая пара
            side: Сторона ордера ("buy" или "sell")
            order_type: Тип ордера ("market" или "limit")
            amount: Количество базовой валюты
            price: Цена для лимитного ордера (опционально)
            
        Returns:
            Информация о созданном ордере
        """
        try:
            # Получаем коннектор для биржи
            connector = self.exchange_factory.get_connector(exchange_name)
            if not connector:
                raise ValueError(f"No connector available for exchange {exchange_name}")
            
            # Проверяем сторону ордера
            if side not in ["buy", "sell"]:
                raise ValueError(f"Invalid order side: {side}. Must be 'buy' or 'sell'")
            
            # Проверяем тип ордера
            if order_type not in ["market", "limit"]:
                raise ValueError(f"Invalid order type: {order_type}. Must be 'market' or 'limit'")
            
            # Проверяем наличие цены для лимитного ордера
            if order_type == "limit" and price is None:
                raise ValueError("Price must be provided for limit orders")
            
            # Проверяем достаточность средств
            if side == "buy":
                # Для покупки нужно проверить наличие котируемой валюты
                base_asset, quote_asset = symbol.split('/')
                quote_amount = price * amount if price else amount  # Примерная сумма для рыночного ордера
                has_funds = await self.balance_manager.check_sufficient_funds(exchange_name, quote_asset, quote_amount)
            else:
                # Для продажи нужно проверить наличие базовой валюты
                base_asset, _ = symbol.split('/')
                has_funds = await self.balance_manager.check_sufficient_funds(exchange_name, base_asset, amount)
                
            if not has_funds:
                raise ValueError(f"Insufficient funds for {side} order on {exchange_name}")
            
            # Создаем ордер
            order_params = {}
            if order_type == "market":
                order_func = connector.create_market_order
            else:
                order_func = connector.create_limit_order
                order_params["price"] = float(price)
            
            # Выполняем запрос к бирже
            order = await order_func(
                symbol=symbol,
                side=side,
                amount=float(amount),
                params=order_params
            )
            
            # Обновляем балансы после создания ордера
            await self.balance_manager.update_balances(exchange_name)
            
            # Сохраняем ордер в кэше и БД
            if order and 'id' in order:
                self.open_orders[order['id']] = order
                self.order_status_cache[order['id']] = order.get('status', 'open')
                
                # Сохраняем в БД
                await self._store_order_in_db(order)
                
                self.logger.info(
                    f"Created {order_type} {side} order on {exchange_name}: {order['id']}, "
                    f"symbol: {symbol}, amount: {amount}"
                )
                
                # Запускаем таймер для автоматической отмены зависших ордеров
                if order_type == "limit" and order.get('status') != 'closed':
                    asyncio.create_task(self._auto_cancel_order(order['id'], exchange_name))
            else:
                self.logger.error(f"Failed to create order: Invalid order response")
                return None
            
            return order
        except Exception as e:
            self.logger.error(f"Error creating {order_type} {side} order on {exchange_name}: {str(e)}")
            return None
    
    async def cancel_order(self, order_id: str, exchange_name: str) -> bool:
        """
        Отменяет открытый ордер.
        
        Args:
            order_id: ID ордера для отмены
            exchange_name: Имя биржи
            
        Returns:
            True если ордер успешно отменён, иначе False
        """
        try:
            # Получаем коннектор для биржи
            connector = self.exchange_factory.get_connector(exchange_name)
            if not connector:
                raise ValueError(f"No connector available for exchange {exchange_name}")
            
            # Проверяем наличие ордера в кэше
            if order_id not in self.open_orders:
                # Пробуем получить ордер из БД
                order = await self._get_order_from_db(order_id)
                if not order:
                    self.logger.warning(f"Order {order_id} not found in cache or DB")
                    return False
            
            # Отменяем ордер на бирже
            result = await connector.cancel_order(order_id)
            
            # Обновляем статус ордера в кэше и БД
            if result:
                if order_id in self.open_orders:
                    self.open_orders[order_id]['status'] = 'canceled'
                self.order_status_cache[order_id] = 'canceled'
                
                # Обновляем в БД
                await self._update_order_in_db(order_id, status='canceled')
                
                # Обновляем балансы после отмены ордера
                await self.balance_manager.update_balances(exchange_name)
                
                self.logger.info(f"Order {order_id} on {exchange_name} successfully canceled")
                return True
            else:
                self.logger.warning(f"Failed to cancel order {order_id} on {exchange_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error canceling order {order_id} on {exchange_name}: {str(e)}")
            return False
    
    async def get_order_status(self, order_id: str, exchange_name: str) -> str:
        """
        Получает текущий статус ордера.
        
        Args:
            order_id: ID ордера
            exchange_name: Имя биржи
            
        Returns:
            Статус ордера (open, closed, canceled, failed)
        """
        try:
            # Проверяем кэш статусов
            if order_id in self.order_status_cache:
                status = self.order_status_cache[order_id]
                
                # Если ордер уже в финальном статусе, возвращаем его
                if status in ['closed', 'canceled', 'failed']:
                    return status
            
            # Получаем коннектор для биржи
            connector = self.exchange_factory.get_connector(exchange_name)
            if not connector:
                raise ValueError(f"No connector available for exchange {exchange_name}")
            
            # Запрашиваем статус ордера на бирже
            order = await connector.fetch_order(order_id)
            
            if order:
                status = order.get('status', 'unknown')
                
                # Обновляем кэш
                self.order_status_cache[order_id] = status
                
                # Если ордер закрыт, обновляем его в кэше и БД
                if status in ['closed', 'canceled', 'failed']:
                    if order_id in self.open_orders:
                        self.open_orders[order_id] = order
                    await self._update_order_in_db(order_id, status=status)
                
                return status
            else:
                self.logger.warning(f"Failed to get status for order {order_id} on {exchange_name}")
                return "unknown"
                
        except Exception as e:
            self.logger.error(f"Error getting status for order {order_id} on {exchange_name}: {str(e)}")
            return "error"
    
    async def fetch_order(self, order_id: str, exchange_name: str) -> Dict[str, Any]:
        """
        Получает полную информацию об ордере.
        
        Args:
            order_id: ID ордера
            exchange_name: Имя биржи
            
        Returns:
            Полная информация об ордере
        """
        try:
            # Проверяем наличие ордера в кэше открытых ордеров
            if order_id in self.open_orders:
                # Если ордер в финальном статусе, возвращаем его из кэша
                status = self.order_status_cache.get(order_id)
                if status in ['closed', 'canceled', 'failed']:
                    return self.open_orders[order_id]
            
            # Получаем коннектор для биржи
            connector = self.exchange_factory.get_connector(exchange_name)
            if not connector:
                raise ValueError(f"No connector available for exchange {exchange_name}")
            
            # Запрашиваем ордер на бирже
            order = await connector.fetch_order(order_id)
            
            if order:
                # Обновляем кэш
                self.open_orders[order_id] = order
                self.order_status_cache[order_id] = order.get('status', 'unknown')
                
                # Обновляем в БД
                await self._update_order_in_db(order_id, **order)
                
                return order
            else:
                # Если не нашли на бирже, пробуем получить из БД
                db_order = await self._get_order_from_db(order_id)
                if db_order:
                    return db_order
                
                self.logger.warning(f"Order {order_id} not found on {exchange_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching order {order_id} on {exchange_name}: {str(e)}")
            return None
    
    async def _auto_cancel_order(self, order_id: str, exchange_name: str) -> None:
        """
        Автоматически отменяет ордер по истечении времени TTL.
        
        Args:
            order_id: ID ордера
            exchange_name: Имя биржи
        """
        try:
            # Ждём указанное время
            await asyncio.sleep(self.order_ttl_seconds)
            
            # Проверяем статус ордера
            status = await self.get_order_status(order_id, exchange_name)
            
            # Если ордер всё ещё открыт, отменяем его
            if status == "open":
                self.logger.info(f"Auto-canceling order {order_id} on {exchange_name} after TTL expiration")
                await self.cancel_order(order_id, exchange_name)
        except Exception as e:
            self.logger.error(f"Error during auto-cancel of order {order_id}: {str(e)}")
    
    async def _store_order_in_db(self, order: Dict[str, Any]) -> None:
        """
        Сохраняет ордер в базе данных.
        
        Args:
            order: Информация об ордере
        """
        try:
            if not self.order_repo:
                return
                
            with self.db_manager.get_session() as session:
                # Проверяем, существует ли ордер в БД
                db_order = self.order_repo.get_by_exchange_order_id(
                    session, order.get('id'), order.get('exchange')
                )
                
                # Получаем основные данные ордера
                order_data = {
                    'order_id': order.get('id'),
                    'exchange_name': order.get('exchange'),
                    'symbol': order.get('symbol'),
                    'type': order.get('type'),
                    'side': order.get('side'),
                    'amount': Decimal(str(order.get('amount', 0))),
                    'price': Decimal(str(order.get('price', 0))),
                    'cost': Decimal(str(order.get('cost', 0))),
                    'filled': Decimal(str(order.get('filled', 0))),
                    'status': order.get('status', 'open'),
                    'timestamp': datetime.fromtimestamp(order.get('timestamp') / 1000) 
                        if order.get('timestamp') else datetime.now(),
                    'additional_data': order
                }
                
                if db_order:
                    # Обновляем существующий ордер
                    self.order_repo.update(session, db_order.id, **order_data)
                else:
                    # Создаём новый ордер в БД
                    self.order_repo.create(session, **order_data)
        except Exception as e:
            self.logger.error(f"Error storing order in DB: {str(e)}")
    
    async def _update_order_in_db(self, order_id: str, **kwargs) -> None:
        """
        Обновляет ордер в базе данных.
        
        Args:
            order_id: ID ордера
            **kwargs: Данные для обновления
        """
        try:
            if not self.order_repo:
                return
                
            with self.db_manager.get_session() as session:
                # Проверяем, существует ли ордер в БД
                db_order = self.order_repo.get_by_exchange_order_id(
                    session, order_id, kwargs.get('exchange_name', kwargs.get('exchange'))
                )
                
                if db_order:
                    # Преобразуем числовые значения в Decimal
                    for key in ['amount', 'price', 'cost', 'filled']:
                        if key in kwargs and kwargs[key] is not None:
                            kwargs[key] = Decimal(str(kwargs[key]))
                    
                    # Преобразуем timestamp в datetime
                    if 'timestamp' in kwargs and kwargs['timestamp'] is not None:
                        kwargs['timestamp'] = datetime.fromtimestamp(kwargs['timestamp'] / 1000)
                    
                    # Обновляем ордер
                    self.order_repo.update(session, db_order.id, **kwargs)
        except Exception as e:
            self.logger.error(f"Error updating order in DB: {str(e)}")
    
    async def _get_order_from_db(self, order_id: str) -> Dict[str, Any]:
        """
        Получает информацию об ордере из базы данных.
        
        Args:
            order_id: ID ордера
            
        Returns:
            Информация об ордере или None
        """
        try:
            if not self.order_repo:
                return None
                
            with self.db_manager.get_session() as session:
                # Для упрощения, просто ищем по order_id без указания биржи
                # В реальном сценарии нужно указывать и биржу
                db_order = self.order_repo.get_by_order_id(session, order_id)
                
                if db_order:
                    # Возвращаем данные дополнительного поля, которое хранит полную информацию об ордере
                    return db_order.additional_data
                
                return None
        except Exception as e:
            self.logger.error(f"Error getting order from DB: {str(e)}")
            return None