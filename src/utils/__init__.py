"""
Утилитарные функции и инструменты для использования в проекте.
"""

from src.utils.logger import (
    setup_logging,
    get_logger,
    get_operation_logger,
    get_contextual_logger,
    log_execution_time,
    log_async_execution_time,
    log_operation,
    log_async_operation,
    ContextualAdapter
)

# Экспортируем утилиты для удобного импорта
__all__ = [
    'setup_logging',
    'get_logger',
    'get_operation_logger',
    'get_contextual_logger',
    'log_execution_time',
    'log_async_execution_time',
    'log_operation',
    'log_async_operation',
    'ContextualAdapter'
]
