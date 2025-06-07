"""
Пример использования системы логирования в проекте.
"""

import os
import sys
import asyncio
import time
import random
from typing import Dict, List, Any

# Добавляем корневую директорию проекта в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    setup_logging, 
    get_logger, 
    get_contextual_logger, 
    get_operation_logger,
    log_execution_time, 
    log_async_execution_time, 
    log_operation, 
    log_async_operation
)


# Простой пример использования логгера
def basic_logging_example():
    """Демонстрирует базовое использование логгера."""
    logger = get_logger(__name__)
    
    logger.debug("Это отладочное сообщение")
    logger.info("Это информационное сообщение")
    logger.warning("Это предупреждение")
    logger.error("Это сообщение об ошибке")
    
    try:
        result = 10 / 0
    except Exception as e:
        logger.exception(f"Произошла ошибка: {str(e)}")


# Пример использования контекстного логгера
def contextual_logging_example():
    """Демонстрирует использование контекстного логгера."""
    # Создаем контекстный логгер с информацией о компоненте и пользователе
    logger = get_contextual_logger(
        __name__,
        component="example",
        user_id="test_user"
    )
    
    logger.info("Действие пользователя")
    
    # Добавляем дополнительный контекст к логгеру
    order_logger = logger.with_context(
        order_id="12345",
        symbol="BTC/USDT"
    )
    
    order_logger.info("Информация о заказе")
    order_logger.warning("Предупреждение о заказе")


# Пример использования декоратора для измерения времени выполнения
@log_execution_time
def slow_operation(duration: float):
    """
    Медленная операция для демонстрации логирования времени выполнения.
    
    Args:
        duration: Время задержки в секундах
    """
    logger = get_logger(__name__)
    logger.info(f"Начинаем медленную операцию ({duration} сек)")
    time.sleep(duration)
    logger.info("Медленная операция завершена")
    return f"Операция выполнялась {duration} секунд"


# Пример использования асинхронного декоратора для измерения времени
@log_async_execution_time
async def async_operation(duration: float):
    """
    Асинхронная операция для демонстрации логирования.
    
    Args:
        duration: Время задержки в секундах
    """
    logger = get_logger(__name__)
    logger.info(f"Начинаем асинхронную операцию ({duration} сек)")
    await asyncio.sleep(duration)
    logger.info("Асинхронная операция завершена")
    return f"Асинхронная операция выполнялась {duration} секунд"


# Пример использования декоратора для логирования операций
@log_operation(operation_type="trade", market="BTC/USDT")
def execute_trade(amount: float, price: float):
    """
    Выполняет торговую операцию и логирует её.
    
    Args:
        amount: Количество для торговли
        price: Цена
    """
    logger = get_logger(__name__)
    logger.info(f"Выполняем сделку: {amount} BTC по цене {price} USDT")
    
    # Имитация выполнения сделки
    time.sleep(0.5)
    success = random.random() > 0.2  # 80% вероятность успеха
    
    if success:
        logger.info(f"Сделка выполнена успешно: {amount} BTC по {price} USDT")
        return {"status": "success", "amount": amount, "price": price, "total": amount * price}
    else:
        error_msg = "Не удалось выполнить сделку из-за проблем с ликвидностью"
        logger.error(error_msg)
        raise Exception(error_msg)


# Пример использования асинхронного декоратора для логирования операций
@log_async_operation(operation_type="arbitrage", pair="BTC/USDT", exchanges=["binance", "kucoin"])
async def execute_arbitrage_opportunity(buy_exchange: str, sell_exchange: str, amount: float, price_diff: float):
    """
    Выполняет арбитражную операцию между двумя биржами.
    
    Args:
        buy_exchange: Биржа для покупки
        sell_exchange: Биржа для продажи
        amount: Количество для торговли
        price_diff: Разница в цене между биржами
    """
    logger = get_logger(__name__)
    logger.info(f"Арбитраж: покупка на {buy_exchange}, продажа на {sell_exchange}, разница: {price_diff}%")
    
    # Имитация выполнения арбитража
    await asyncio.sleep(1.0)
    success = random.random() > 0.3  # 70% вероятность успеха
    
    if success:
        profit = amount * price_diff / 100
        logger.info(f"Арбитраж выполнен успешно. Прибыль: {profit:.4f} USDT")
        return {"status": "success", "profit": profit, "amount": amount}
    else:
        error_msg = "Не удалось выполнить арбитраж из-за изменения цен"
        logger.error(error_msg)
        raise Exception(error_msg)


# Пример структурированного логирования
def structured_logging_example():
    """Демонстрирует использование структурированного логирования."""
    operations_logger = get_operation_logger()
    
    # Логируем структурированные данные
    operations_logger.info({
        "event": "balance_update",
        "currency": "BTC",
        "previous_balance": 1.25,
        "new_balance": 1.35,
        "change": 0.1,
        "source": "deposit"
    })
    
    operations_logger.warning({
        "event": "rate_limit",
        "exchange": "binance",
        "endpoint": "/api/v3/ticker/price",
        "limit": 1200,
        "period": "1m",
        "current_usage": 1100
    })


# Пример использования обработки исключений
def exception_handling_example():
    """Демонстрирует логирование исключений."""
    logger = get_logger(__name__)
    
    try:
        logger.info("Начинаем операцию, которая вызовет исключение")
        
        # Имитируем ошибку
        result = {"data": None}
        processed_data = result["data"]["price"]  # Вызовет AttributeError
        
        logger.info("Эта строка не будет выполнена")
    except Exception as e:
        logger.exception(f"Произошла ошибка при обработке данных: {str(e)}")


# Главная функция для запуска примеров
async def main():
    """Запускает все примеры использования логирования."""
    # Настраиваем логирование
    setup_logging()
    logger = get_logger(__name__)
    logger.info("Начало выполнения примеров логирования")
    
    # Базовые примеры
    logger.info("--- Базовое логирование ---")
    basic_logging_example()
    
    logger.info("--- Контекстное логирование ---")
    contextual_logging_example()
    
    logger.info("--- Измерение времени выполнения ---")
    result = slow_operation(1.5)
    logger.info(f"Результат: {result}")
    
    logger.info("--- Асинхронное измерение времени ---")
    result = await async_operation(1.0)
    logger.info(f"Результат: {result}")
    
    logger.info("--- Структурированное логирование ---")
    structured_logging_example()
    
    logger.info("--- Обработка исключений ---")
    exception_handling_example()
    
    # Примеры логирования операций
    logger.info("--- Логирование торговой операции ---")
    try:
        result = execute_trade(0.05, 25000)
        logger.info(f"Торговая операция: {result}")
    except Exception as e:
        logger.info(f"Торговая операция не удалась: {str(e)}")
    
    logger.info("--- Логирование арбитражной операции ---")
    try:
        result = await execute_arbitrage_opportunity("binance", "kucoin", 0.1, 1.5)
        logger.info(f"Арбитражная операция: {result}")
    except Exception as e:
        logger.info(f"Арбитражная операция не удалась: {str(e)}")
    
    logger.info("Выполнение примеров логирования завершено")


# Точка входа для запуска примеров
if __name__ == "__main__":
    asyncio.run(main())
