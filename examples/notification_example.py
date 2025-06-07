"""
Пример использования системы уведомлений.
"""

import os
import sys
import asyncio
import random
from typing import Dict, Any

# Добавляем корневую директорию проекта в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.notifier import NotificationManager, TelegramNotifier
from src.notifier.notification_manager import NotificationLevel, NotificationType


async def test_simple_messages(notification_manager: NotificationManager) -> None:
    """
    Тестирует отправку простых текстовых сообщений.
    
    Args:
        notification_manager: Менеджер уведомлений
    """
    print("Отправка простых сообщений...")
    
    # Отправляем простое сообщение
    await notification_manager.send_message("Это простое тестовое сообщение")
    
    # Отправляем сообщение с разными уровнями важности
    await notification_manager.send_alert("Информационное сообщение", "Это информационное сообщение для тестирования")
    await notification_manager.send_alert("Предупреждение", "Это предупреждение для тестирования", level=NotificationLevel.WARNING)
    await notification_manager.send_alert("Ошибка", "Это сообщение об ошибке для тестирования", level=NotificationLevel.ERROR)
    await notification_manager.send_alert("Критическая ошибка", "Это сообщение о критической ошибке", level=NotificationLevel.CRITICAL)


async def test_trade_notifications(notification_manager: NotificationManager) -> None:
    """
    Тестирует отправку уведомлений о торговых операциях.
    
    Args:
        notification_manager: Менеджер уведомлений
    """
    print("Отправка уведомлений о торговых операциях...")
    
    # Имитируем успешную покупку
    await notification_manager.send_trade_info(
        exchange="binance",
        symbol="BTC/USDT",
        operation="buy",
        amount=0.05,
        price=25000.0,
        status="executed",
        details={
            "Идентификатор ордера": "12345678",
            "Комиссия": "0.00005 BTC",
            "Время выполнения": "0.5 сек"
        }
    )
    
    # Имитируем отмененную продажу
    await notification_manager.send_trade_info(
        exchange="kucoin",
        symbol="ETH/USDT",
        operation="sell",
        amount=1.5,
        price=1800.0,
        status="failed",
        details={
            "Причина ошибки": "Недостаточная ликвидность",
            "Идентификатор ордера": "87654321"
        }
    )


async def test_arbitrage_notifications(notification_manager: NotificationManager) -> None:
    """
    Тестирует отправку уведомлений об арбитражных возможностях.
    
    Args:
        notification_manager: Менеджер уведомлений
    """
    print("Отправка уведомлений об арбитражных возможностях...")
    
    # Имитируем арбитражную возможность
    await notification_manager.send_arbitrage_opportunity(
        buy_exchange="kucoin",
        sell_exchange="binance",
        symbol="BTC/USDT",
        buy_price=24950.0,
        sell_price=25200.0,
        profit_percent=1.0,
        estimated_profit=25.0,
        details={
            "Объем для торговли": "0.1 BTC",
            "Комиссии учтены": "Да",
            "Время обнаружения": "2023-08-15 14:30:45"
        }
    )


async def test_system_status(notification_manager: NotificationManager) -> None:
    """
    Тестирует отправку уведомлений о статусе системы.
    
    Args:
        notification_manager: Менеджер уведомлений
    """
    print("Отправка уведомления о статусе системы...")
    
    # Имитируем данные о балансах
    balances = {
        "binance": {
            "BTC": 0.15,
            "ETH": 2.5,
            "USDT": 5000.0
        },
        "kucoin": {
            "BTC": 0.08,
            "ETH": 1.2,
            "USDT": 3000.0
        }
    }
    
    # Отправляем статус системы
    await notification_manager.send_system_status(
        status="running",
        balances=balances,
        active_tasks=12,
        errors_count=2,
        details={
            "Время работы": "3 часа 45 минут",
            "Выполнено операций": 24,
            "Использование CPU": "32%",
            "Использование памяти": "128 MB"
        }
    )


async def test_error_notification(notification_manager: NotificationManager) -> None:
    """
    Тестирует отправку уведомлений об ошибках.
    
    Args:
        notification_manager: Менеджер уведомлений
    """
    print("Отправка уведомления об ошибке...")
    
    # Имитируем трассировку стека
    stacktrace = """Traceback (most recent call last):
  File "src/trading/order_manager.py", line 125, in execute_order
    result = await exchange.create_order(symbol, order_type, side, amount, price)
  File "src/data_fetching/exchange_connector.py", line 210, in create_order
    response = await self.exchange.create_order(symbol, order_type, side, amount, price)
  File "venv/lib/python3.9/site-packages/ccxt/base/exchange.py", line 1542, in create_order
    raise InsufficientFunds(self.id + ' ' + message)
ccxt.base.errors.InsufficientFunds: binance Insufficient funds to execute order
"""
    
    # Отправляем уведомление об ошибке
    await notification_manager.send_error(
        title="Ошибка при выполнении ордера",
        error_message="Недостаточно средств для выполнения ордера на Binance",
        stacktrace=stacktrace
    )


async def test_cooldown(notification_manager: NotificationManager) -> None:
    """
    Тестирует механизм задержки между уведомлениями.
    
    Args:
        notification_manager: Менеджер уведомлений
    """
    print("Тестирование механизма задержки (cooldown)...")
    
    # Устанавливаем задержку в 2 секунды
    notification_manager.set_cooldown(2.0)
    
    # Отправляем несколько уведомлений одного типа быстро
    await notification_manager.send_message("Первое сообщение (должно быть отправлено)")
    await notification_manager.send_message("Второе сообщение (должно быть пропущено из-за cooldown)")
    await notification_manager.send_message("Третье сообщение (должно быть пропущено из-за cooldown)")
    
    # Ждем, чтобы прошло время задержки
    await asyncio.sleep(2.5)
    
    # Отправляем еще одно сообщение, которое должно пройти
    await notification_manager.send_message("Четвертое сообщение (должно быть отправлено после паузы)")


async def test_notification_filtering(notification_manager: NotificationManager) -> None:
    """
    Тестирует фильтрацию уведомлений по уровню и типу.
    
    Args:
        notification_manager: Менеджер уведомлений
    """
    print("Тестирование фильтрации уведомлений...")
    
    # Устанавливаем минимальный уровень WARNING
    notification_manager.set_min_level(NotificationLevel.WARNING)
    
    # Эти сообщения должны быть пропущены (уровень ниже минимального)
    await notification_manager.send_alert("Информационное сообщение", "Это сообщение должно быть пропущено из-за низкого уровня", level=NotificationLevel.INFO)
    await notification_manager.send_alert("Отладочное сообщение", "Это сообщение должно быть пропущено из-за низкого уровня", level=NotificationLevel.DEBUG)
    
    # Эти сообщения должны пройти фильтр
    await notification_manager.send_alert("Предупреждение", "Это предупреждение должно пройти фильтр", level=NotificationLevel.WARNING)
    await notification_manager.send_alert("Ошибка", "Это сообщение об ошибке должно пройти фильтр", level=NotificationLevel.ERROR)
    
    # Отключаем тип уведомлений TRADE
    notification_manager.disable_notification_type(NotificationType.TRADE)
    
    # Это сообщение должно быть пропущено (тип отключен)
    await notification_manager.send_trade_info(
        exchange="binance",
        symbol="BTC/USDT",
        operation="buy",
        amount=0.05,
        price=25000.0,
        status="executed",
        level=NotificationLevel.WARNING  # Даже с высоким уровнем сообщение будет пропущено
    )
    
    # Возвращаем настройки обратно
    notification_manager.set_min_level(NotificationLevel.INFO)
    notification_manager.enable_notification_type(NotificationType.TRADE)


async def main() -> None:
    """Основная функция для запуска примеров."""
    print("Запуск примеров использования системы уведомлений...")
    
    # Создаем менеджер уведомлений
    notification_manager = NotificationManager()
    
    try:
        # Тестируем разные типы уведомлений
        await test_simple_messages(notification_manager)
        await asyncio.sleep(1)
        
        await test_trade_notifications(notification_manager)
        await asyncio.sleep(1)
        
        await test_arbitrage_notifications(notification_manager)
        await asyncio.sleep(1)
        
        await test_system_status(notification_manager)
        await asyncio.sleep(1)
        
        await test_error_notification(notification_manager)
        await asyncio.sleep(1)
        
        await test_cooldown(notification_manager)
        await asyncio.sleep(3)
        
        await test_notification_filtering(notification_manager)
        
        print("Все тесты уведомлений завершены. Ожидание отправки всех сообщений...")
        # Даем время на отправку всех сообщений в очереди
        await asyncio.sleep(2)
    
    finally:
        # Корректно закрываем менеджер уведомлений
        await notification_manager.close()
        print("Менеджер уведомлений закрыт")


if __name__ == "__main__":
    # Запускаем асинхронную функцию main()
    asyncio.run(main())
