"""
Тесты для системы уведомлений.
"""

import unittest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import os
import sys

# Добавляем корневую директорию проекта в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.notifier.base_notifier import BaseNotifier
from src.notifier.notification_manager import (
    NotificationManager, NotificationLevel, NotificationType
)


class MockNotifier(BaseNotifier):
    """Мок-класс для тестирования нотификатора."""
    
    def __init__(self):
        self.messages = []
        self.alerts = []
        self.trade_infos = []
        self.arbitrage_opportunities = []
        self.system_statuses = []
        self.is_initialized = True
    
    async def initialize(self):
        """Имитация инициализации."""
        return True
    
    async def send_message(self, message):
        """Имитация отправки сообщения."""
        self.messages.append(message)
        return True
    
    async def send_alert(self, title, message, level="info"):
        """Имитация отправки предупреждения."""
        self.alerts.append({"title": title, "message": message, "level": level})
        return True
    
    async def send_trade_info(self, exchange, symbol, operation, amount, price, status, details=None):
        """Имитация отправки информации о торговой операции."""
        self.trade_infos.append({
            "exchange": exchange,
            "symbol": symbol,
            "operation": operation,
            "amount": amount,
            "price": price,
            "status": status,
            "details": details or {}
        })
        return True
    
    async def send_arbitrage_opportunity(self, buy_exchange, sell_exchange, symbol, buy_price, sell_price, 
                                        profit_percent, estimated_profit, details=None):
        """Имитация отправки информации об арбитражной возможности."""
        self.arbitrage_opportunities.append({
            "buy_exchange": buy_exchange,
            "sell_exchange": sell_exchange,
            "symbol": symbol,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "profit_percent": profit_percent,
            "estimated_profit": estimated_profit,
            "details": details or {}
        })
        return True
    
    async def send_system_status(self, status, balances, active_tasks, errors_count, details=None):
        """Имитация отправки статуса системы."""
        self.system_statuses.append({
            "status": status,
            "balances": balances,
            "active_tasks": active_tasks,
            "errors_count": errors_count,
            "details": details or {}
        })
        return True


class TestNotificationManager(unittest.IsolatedAsyncioTestCase):
    """Тесты для менеджера уведомлений."""
    
    async def asyncSetUp(self):
        """Настройка перед каждым тестом."""
        self.mock_notifier = MockNotifier()
        self.manager = NotificationManager()
        self.manager.add_notifier(self.mock_notifier)
        self.manager.is_initialized = True
    
    async def test_send_message(self):
        """Тест отправки простого сообщения."""
        test_message = "Тестовое сообщение"
        result = await self.manager.send_message(test_message)
        
        self.assertTrue(result)
        self.assertEqual(len(self.mock_notifier.messages), 1)
        self.assertEqual(self.mock_notifier.messages[0], test_message)
    
    async def test_send_alert(self):
        """Тест отправки предупреждения."""
        test_title = "Тестовый заголовок"
        test_message = "Тестовое сообщение предупреждения"
        result = await self.manager.send_alert(test_title, test_message, level=NotificationLevel.WARNING)
        
        self.assertTrue(result)
        self.assertEqual(len(self.mock_notifier.alerts), 1)
        self.assertEqual(self.mock_notifier.alerts[0]["title"], test_title)
        self.assertEqual(self.mock_notifier.alerts[0]["message"], test_message)
        self.assertEqual(self.mock_notifier.alerts[0]["level"], "warning")
    
    async def test_send_trade_info(self):
        """Тест отправки информации о торговой операции."""
        exchange = "binance"
        symbol = "BTC/USDT"
        operation = "buy"
        amount = 0.5
        price = 25000.0
        status = "executed"
        details = {"order_id": "12345"}
        
        result = await self.manager.send_trade_info(
            exchange, symbol, operation, amount, price, status, details=details
        )
        
        self.assertTrue(result)
        self.assertEqual(len(self.mock_notifier.trade_infos), 1)
        trade_info = self.mock_notifier.trade_infos[0]
        self.assertEqual(trade_info["exchange"], exchange)
        self.assertEqual(trade_info["symbol"], symbol)
        self.assertEqual(trade_info["operation"], operation)
        self.assertEqual(trade_info["amount"], amount)
        self.assertEqual(trade_info["price"], price)
        self.assertEqual(trade_info["status"], status)
        self.assertEqual(trade_info["details"], details)
    
    async def test_send_arbitrage_opportunity(self):
        """Тест отправки информации об арбитражной возможности."""
        buy_exchange = "kucoin"
        sell_exchange = "binance"
        symbol = "BTC/USDT"
        buy_price = 24950.0
        sell_price = 25100.0
        profit_percent = 0.6
        estimated_profit = 15.0
        details = {"volume": "0.1 BTC"}
        
        result = await self.manager.send_arbitrage_opportunity(
            buy_exchange, sell_exchange, symbol, buy_price, sell_price,
            profit_percent, estimated_profit, details=details
        )
        
        self.assertTrue(result)
        self.assertEqual(len(self.mock_notifier.arbitrage_opportunities), 1)
        arb_opp = self.mock_notifier.arbitrage_opportunities[0]
        self.assertEqual(arb_opp["buy_exchange"], buy_exchange)
        self.assertEqual(arb_opp["sell_exchange"], sell_exchange)
        self.assertEqual(arb_opp["symbol"], symbol)
        self.assertEqual(arb_opp["buy_price"], buy_price)
        self.assertEqual(arb_opp["sell_price"], sell_price)
        self.assertEqual(arb_opp["profit_percent"], profit_percent)
        self.assertEqual(arb_opp["estimated_profit"], estimated_profit)
        self.assertEqual(arb_opp["details"], details)
    
    async def test_send_system_status(self):
        """Тест отправки статуса системы."""
        status = "running"
        balances = {
            "binance": {"BTC": 0.1, "USDT": 5000}
        }
        active_tasks = 5
        errors_count = 0
        details = {"uptime": "10h 30m"}
        
        result = await self.manager.send_system_status(
            status, balances, active_tasks, errors_count, details=details
        )
        
        self.assertTrue(result)
        self.assertEqual(len(self.mock_notifier.system_statuses), 1)
        sys_status = self.mock_notifier.system_statuses[0]
        self.assertEqual(sys_status["status"], status)
        self.assertEqual(sys_status["balances"], balances)
        self.assertEqual(sys_status["active_tasks"], active_tasks)
        self.assertEqual(sys_status["errors_count"], errors_count)
        self.assertEqual(sys_status["details"], details)
    
    async def test_send_error(self):
        """Тест отправки сообщения об ошибке."""
        title = "Тестовая ошибка"
        error_message = "Описание ошибки"
        stacktrace = "Трассировка стека"
        
        # Патчим send_alert, чтобы перехватить вызов
        with patch.object(self.manager, 'send_alert', new_callable=AsyncMock) as mock_send_alert:
            mock_send_alert.return_value = True
            
            result = await self.manager.send_error(title, error_message, stacktrace)
            
            self.assertTrue(result)
            mock_send_alert.assert_called_once()
            args, kwargs = mock_send_alert.call_args
            self.assertEqual(args[0], title)
            self.assertIn(error_message, args[1])
            self.assertIn(stacktrace, args[1])
            self.assertEqual(kwargs.get("level"), "error")
    
    async def test_cooldown(self):
        """Тест механизма задержки между однотипными уведомлениями."""
        # Устанавливаем cooldown в 1 секунду
        self.manager.set_cooldown(1.0)
        
        # Отправляем первое сообщение (должно пройти)
        result1 = await self.manager.send_message("Сообщение 1")
        self.assertTrue(result1)
        self.assertEqual(len(self.mock_notifier.messages), 1)
        
        # Сразу отправляем второе сообщение (должно быть пропущено из-за cooldown)
        result2 = await self.manager.send_message("Сообщение 2")
        self.assertFalse(result2)  # Должно вернуть False, т.к. сообщение пропущено
        self.assertEqual(len(self.mock_notifier.messages), 1)  # Количество сообщений не изменилось
        
        # Ждем, чтобы прошло время cooldown
        await asyncio.sleep(1.1)
        
        # Отправляем третье сообщение (должно пройти)
        result3 = await self.manager.send_message("Сообщение 3")
        self.assertTrue(result3)
        self.assertEqual(len(self.mock_notifier.messages), 2)
    
    async def test_notification_level_filtering(self):
        """Тест фильтрации по уровню важности."""
        # Устанавливаем минимальный уровень WARNING
        self.manager.set_min_level(NotificationLevel.WARNING)
        
        # Отправляем уведомление с уровнем INFO (должно быть отфильтровано)
        result_info = await self.manager.send_alert("Тест", "Сообщение", level=NotificationLevel.INFO)
        self.assertFalse(result_info)
        self.assertEqual(len(self.mock_notifier.alerts), 0)
        
        # Отправляем уведомление с уровнем WARNING (должно пройти)
        result_warning = await self.manager.send_alert("Тест", "Сообщение", level=NotificationLevel.WARNING)
        self.assertTrue(result_warning)
        self.assertEqual(len(self.mock_notifier.alerts), 1)
        
        # Отправляем уведомление с уровнем ERROR (должно пройти)
        result_error = await self.manager.send_alert("Тест", "Сообщение", level=NotificationLevel.ERROR)
        self.assertTrue(result_error)
        self.assertEqual(len(self.mock_notifier.alerts), 2)
    
    async def test_notification_type_filtering(self):
        """Тест фильтрации по типу уведомления."""
        # Отключаем тип уведомлений TRADE
        self.manager.disable_notification_type(NotificationType.TRADE)
        
        # Пытаемся отправить уведомление о торговой операции (должно быть отфильтровано)
        result_trade = await self.manager.send_trade_info(
            "binance", "BTC/USDT", "buy", 0.1, 25000.0, "executed"
        )
        self.assertFalse(result_trade)
        self.assertEqual(len(self.mock_notifier.trade_infos), 0)
        
        # Включаем тип уведомлений TRADE
        self.manager.enable_notification_type(NotificationType.TRADE)
        
        # Пытаемся отправить уведомление о торговой операции (должно пройти)
        result_trade = await self.manager.send_trade_info(
            "binance", "BTC/USDT", "buy", 0.1, 25000.0, "executed"
        )
        self.assertTrue(result_trade)
        self.assertEqual(len(self.mock_notifier.trade_infos), 1)
    
    async def test_multiple_notifiers(self):
        """Тест работы с несколькими нотификаторами."""
        # Создаем второй мок-нотификатор
        second_notifier = MockNotifier()
        self.manager.add_notifier(second_notifier)
        
        # Отправляем сообщение
        test_message = "Тест для нескольких нотификаторов"
        result = await self.manager.send_message(test_message)
        
        # Проверяем, что сообщение отправлено через оба нотификатора
        self.assertTrue(result)
        self.assertEqual(len(self.mock_notifier.messages), 1)
        self.assertEqual(self.mock_notifier.messages[0], test_message)
        self.assertEqual(len(second_notifier.messages), 1)
        self.assertEqual(second_notifier.messages[0], test_message)
    
    async def test_notifier_failure(self):
        """Тест обработки ошибок нотификатора."""
        # Создаем нотификатор, который будет вызывать ошибку
        failing_notifier = MockNotifier()
        failing_notifier.send_message = AsyncMock(side_effect=Exception("Тестовая ошибка"))
        self.manager.add_notifier(failing_notifier)
        
        # Отправляем сообщение
        result = await self.manager.send_message("Тестовое сообщение")
        
        # Проверяем, что результат успешный (т.к. один нотификатор работает)
        self.assertTrue(result)
        self.assertEqual(len(self.mock_notifier.messages), 1)
    
    async def test_all_notifiers_failing(self):
        """Тест случая, когда все нотификаторы вызывают ошибки."""
        # Очищаем список нотификаторов
        self.manager.notifiers = []
        
        # Добавляем только нотификатор с ошибкой
        failing_notifier = MockNotifier()
        failing_notifier.send_message = AsyncMock(side_effect=Exception("Тестовая ошибка"))
        self.manager.add_notifier(failing_notifier)
        
        # Отправляем сообщение
        result = await self.manager.send_message("Тестовое сообщение")
        
        # Проверяем, что результат неуспешный (т.к. нотификатор вызывает ошибку)
        self.assertFalse(result)
    
    async def test_initialization(self):
        """Тест инициализации менеджера уведомлений."""
        # Создаем новый менеджер, который не инициализирован
        manager = NotificationManager()
        manager.is_initialized = False
        
        # Добавляем мок-нотификатор
        notifier = MockNotifier()
        manager.add_notifier(notifier)
        
        # Отправляем сообщение (должна произойти автоматическая инициализация)
        result = await manager.send_message("Тестовое сообщение")
        
        # Проверяем результат
        self.assertTrue(result)
        self.assertTrue(manager.is_initialized)
        self.assertEqual(len(notifier.messages), 1)


class TestTelegramNotifier(unittest.IsolatedAsyncioTestCase):
    """Тесты для Telegram-нотификатора."""
    
    async def test_telegram_notifier_initialization(self):
        """Тест инициализации Telegram-нотификатора."""
        from src.notifier.telegram_notifier import TelegramNotifier
        
        # Создаем патчи для методов, которые бы делали HTTP-запросы
        with patch("aiohttp.ClientSession.get") as mock_get, \
             patch("aiohttp.ClientSession.post") as mock_post:
            
            # Настраиваем мок для getMe
            mock_get_response = MagicMock()
            mock_get_response.status = 200
            mock_get_response.json = AsyncMock(return_value={"ok": True, "result": {"first_name": "TestBot"}})
            mock_get.return_value.__aenter__.return_value = mock_get_response
            
            # Настраиваем мок для sendMessage
            mock_post_response = MagicMock()
            mock_post_response.status = 200
            mock_post_response.json = AsyncMock(return_value={"ok": True})
            mock_post.return_value.__aenter__.return_value = mock_post_response
            
            # Создаем нотификатор с тестовыми данными
            notifier = TelegramNotifier(token="test_token", chat_id="test_chat_id")
            
            # Инициализируем нотификатор
            result = await notifier.initialize()
            
            # Проверяем результат
            self.assertTrue(result)
            self.assertTrue(notifier.is_initialized)
            self.assertTrue(notifier.is_available)
            
            # Проверяем, что были сделаны ожидаемые запросы
            mock_get.assert_called_once()
            mock_post.assert_called_once()
    
    async def test_telegram_send_message(self):
        """Тест отправки сообщения через Telegram-нотификатор."""
        from src.notifier.telegram_notifier import TelegramNotifier
        
        # Создаем патч для _send_chat_message
        with patch.object(TelegramNotifier, "_send_chat_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = (True, "")
            
            # Создаем нотификатор с тестовыми данными
            notifier = TelegramNotifier(token="test_token", chat_id="test_chat_id")
            notifier.is_initialized = True
            
            # Отправляем сообщение
            test_message = "Тестовое сообщение"
            result = await notifier.send_message(test_message)
            
            # Проверяем результат
            self.assertTrue(result)
            mock_send.assert_called_once_with(test_message)


if __name__ == "__main__":
    unittest.main()
