"""
Модуль для настройки и управления системой логирования в приложении.
Обеспечивает структурированное логирование, ротацию логов и
отслеживание производительности.
"""

import logging
import logging.config
import logging.handlers
import functools
import time
import os
import sys
import json
from typing import Any, Dict, Optional, Callable, TypeVar, cast
from pathlib import Path
from datetime import datetime

from config.settings import Settings

# Создаем типовые переменные для декораторов
F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')

# Загружаем настройки
settings = Settings()

# Убедимся, что директория logs существует
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)


def setup_logging(log_level: int = None) -> None:
    """
    Настраивает систему логирования на основе конфигурации из настроек.
    Должна вызываться один раз в начале работы приложения.
    
    Args:
        log_level: Опциональный уровень логирования, который переопределяет значение из настроек
    """
    # Получаем конфигурацию из настроек
    log_config = settings.dict_config()
    
    # Если указан уровень логирования, переопределяем его
    if log_level is not None:
        log_config["root"]["level"] = log_level
        # Обновляем уровни для обработчиков
        for handler in ["console", "file"]:
            if handler in log_config["handlers"]:
                log_config["handlers"][handler]["level"] = log_level
    
    # Расширяем форматирование для более подробного вывода
    log_config["formatters"]["detailed"] = {
        "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    }
    
    # Расширяем форматирование для структурированного JSON-логирования
    log_config["formatters"]["json"] = {
        "()": "src.utils.logger.JsonFormatter",
    }
    
    # Добавляем обработчик для отладочных логов
    log_config["handlers"]["debug_file"] = {
        "class": "logging.handlers.RotatingFileHandler",
        "level": logging.DEBUG,
        "formatter": "detailed",
        "filename": "logs/debug.log",
        "maxBytes": 10485760,  # 10 MB
        "backupCount": 3,
        "encoding": "utf8",
    }
    
    # Добавляем обработчик для ошибок
    log_config["handlers"]["error_file"] = {
        "class": "logging.handlers.RotatingFileHandler",
        "level": logging.ERROR,
        "formatter": "detailed",
        "filename": "logs/error.log",
        "maxBytes": 10485760,  # 10 MB
        "backupCount": 5,
        "encoding": "utf8",
    }
    
    # Добавляем обработчик для структурированного JSON-логирования операций
    log_config["handlers"]["operations_file"] = {
        "class": "logging.handlers.RotatingFileHandler",
        "level": logging.INFO,
        "formatter": "json",
        "filename": "logs/operations.json",
        "maxBytes": 10485760,  # 10 MB
        "backupCount": 5,
        "encoding": "utf8",
    }
    
    # Настраиваем логгер для операций
    log_config["loggers"] = log_config.get("loggers", {})
    log_config["loggers"]["operations"] = {
        "level": logging.INFO,
        "handlers": ["operations_file", "console"],
        "propagate": False,
    }
    
    # Обновляем корневой логгер
    log_config["root"]["handlers"].extend(["debug_file", "error_file"])
    
    # Применяем конфигурацию
    logging.config.dictConfig(log_config)
    
    # Пишем в лог информацию о запуске
    root_logger = logging.getLogger()
    root_logger.info(f"Логирование настроено. Уровень: {settings.LOG_LEVEL}")
    
    # Перехватываем необработанные исключения
    sys.excepthook = handle_exception


def handle_exception(exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
    """
    Перехватывает необработанные исключения и записывает их в лог.
    
    Args:
        exc_type: Тип исключения
        exc_value: Значение исключения
        exc_traceback: Трассировка исключения
    """
    if issubclass(exc_type, KeyboardInterrupt):
        # Не логируем KeyboardInterrupt (Ctrl+C)
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
        
    logger = logging.getLogger()
    logger.error(
        "Необработанное исключение",
        exc_info=(exc_type, exc_value, exc_traceback)
    )


class JsonFormatter(logging.Formatter):
    """Форматирует лог в структурированный JSON формат."""
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Форматирует запись лога в JSON.
        
        Args:
            record: Запись лога для форматирования
            
        Returns:
            Строка в формате JSON
        """
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread_id": record.thread,
            "process_id": record.process,
        }
        
        # Обработка сообщения
        if isinstance(record.msg, dict):
            # Если сообщение уже является словарем, объединяем
            log_data.update(record.msg)
        else:
            # Иначе добавляем как отдельное поле
            log_data["message"] = record.getMessage()
        
        # Добавляем информацию об исключении, если оно есть
        if record.exc_info:
            log_data["exception"] = {
                "type": str(record.exc_info[0].__name__),
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }
        
        # Добавляем пользовательские атрибуты
        for key, value in record.__dict__.items():
            if key.startswith("ctx_") and key != "ctx_msg":
                log_data[key[4:]] = value
        
        return json.dumps(log_data, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    """
    Возвращает настроенный логгер по имени.
    
    Args:
        name: Имя логгера, обычно __name__ текущего модуля
        
    Returns:
        Настроенный логгер
    """
    return logging.getLogger(name)


def get_operation_logger() -> logging.Logger:
    """
    Возвращает логгер для операций, который записывает структурированные логи.
    
    Returns:
        Настроенный логгер для операций
    """
    return logging.getLogger("operations")


def log_execution_time(func: F) -> F:
    """
    Декоратор для измерения и логирования времени выполнения функции.
    
    Args:
        func: Декорируемая функция
        
    Returns:
        Декорированная функция
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000  # в миллисекундах
            
            logger.debug(
                f"Выполнение {func.__name__} заняло {execution_time:.2f} мс",
                extra={"ctx_execution_time_ms": execution_time}
            )
            return result
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(
                f"Ошибка при выполнении {func.__name__} через {execution_time:.2f} мс: {str(e)}",
                extra={
                    "ctx_execution_time_ms": execution_time,
                    "ctx_error": str(e)
                },
                exc_info=True
            )
            raise
    
    return cast(F, wrapper)


def log_async_execution_time(func: F) -> F:
    """
    Декоратор для измерения и логирования времени выполнения асинхронной функции.
    
    Args:
        func: Декорируемая асинхронная функция
        
    Returns:
        Декорированная асинхронная функция
    """
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000  # в миллисекундах
            
            logger.debug(
                f"Выполнение {func.__name__} заняло {execution_time:.2f} мс",
                extra={"ctx_execution_time_ms": execution_time}
            )
            return result
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(
                f"Ошибка при выполнении {func.__name__} через {execution_time:.2f} мс: {str(e)}",
                extra={
                    "ctx_execution_time_ms": execution_time,
                    "ctx_error": str(e)
                },
                exc_info=True
            )
            raise
    
    return cast(F, wrapper)


class ContextualAdapter:
    """
    Адаптер для добавления контекстной информации в логи.
    Позволяет прикреплять дополнительные данные ко всем записям лога.
    """
    
    def __init__(self, logger: logging.Logger, context: Optional[Dict[str, Any]] = None):
        """
        Инициализирует адаптер с логгером и начальным контекстом.
        
        Args:
            logger: Базовый логгер
            context: Начальный контекст (опционально)
        """
        self.logger = logger
        self.context = context or {}
    
    def with_context(self, **kwargs: Any) -> 'ContextualAdapter':
        """
        Создает новый адаптер с дополнительным контекстом.
        
        Args:
            **kwargs: Ключевые аргументы для добавления в контекст
            
        Returns:
            Новый адаптер с расширенным контекстом
        """
        new_context = {**self.context, **kwargs}
        return ContextualAdapter(self.logger, new_context)
    
    def process_args(self, msg: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обрабатывает аргументы перед логированием, добавляя контекст.
        
        Args:
            msg: Сообщение для логирования
            kwargs: Аргументы для логирования
            
        Returns:
            Обновленные аргументы с контекстом
        """
        extra = kwargs.get('extra', {}).copy()
        for key, value in self.context.items():
            extra[f"ctx_{key}"] = value
        
        kwargs['extra'] = extra
        return kwargs
    
    def debug(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Логирует с уровнем DEBUG с контекстом."""
        self.logger.debug(msg, *args, **self.process_args(msg, kwargs))
    
    def info(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Логирует с уровнем INFO с контекстом."""
        self.logger.info(msg, *args, **self.process_args(msg, kwargs))
    
    def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Логирует с уровнем WARNING с контекстом."""
        self.logger.warning(msg, *args, **self.process_args(msg, kwargs))
    
    def error(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Логирует с уровнем ERROR с контекстом."""
        self.logger.error(msg, *args, **self.process_args(msg, kwargs))
    
    def critical(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Логирует с уровнем CRITICAL с контекстом."""
        self.logger.critical(msg, *args, **self.process_args(msg, kwargs))
    
    def exception(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Логирует исключение с контекстом."""
        self.logger.exception(msg, *args, **self.process_args(msg, kwargs))


def get_contextual_logger(name: str, **context: Any) -> ContextualAdapter:
    """
    Возвращает контекстный адаптер для логгера.
    
    Args:
        name: Имя логгера
        **context: Начальный контекст
        
    Returns:
        Контекстный адаптер для логгера
    """
    logger = get_logger(name)
    return ContextualAdapter(logger, context)


def log_operation(operation_type: str, **context: Any) -> Callable[[F], F]:
    """
    Декоратор для логирования операций с контекстом.
    
    Args:
        operation_type: Тип операции (например, "trade", "arbitrage", "fetch")
        **context: Дополнительный контекст для логирования
        
    Returns:
        Декоратор для функции
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Получаем логгер операций
            operations_logger = get_operation_logger()
            # Подготавливаем контекст
            operation_context = {
                "operation_type": operation_type,
                "function": func.__name__,
                "timestamp_start": datetime.now().isoformat(),
                **context
            }
            
            start_time = time.time()
            
            try:
                # Логируем начало операции
                operations_logger.info({
                    "event": "operation_start",
                    **operation_context
                })
                
                # Выполняем функцию
                result = func(*args, **kwargs)
                
                # Вычисляем время выполнения
                execution_time = (time.time() - start_time) * 1000
                
                # Логируем успешное завершение
                operations_logger.info({
                    "event": "operation_success",
                    "execution_time_ms": execution_time,
                    "timestamp_end": datetime.now().isoformat(),
                    **operation_context
                })
                
                return result
            except Exception as e:
                # Вычисляем время до ошибки
                execution_time = (time.time() - start_time) * 1000
                
                # Логируем ошибку
                operations_logger.error({
                    "event": "operation_error",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "execution_time_ms": execution_time,
                    "timestamp_end": datetime.now().isoformat(),
                    **operation_context
                }, exc_info=True)
                
                # Пробрасываем исключение дальше
                raise
        
        return cast(F, wrapper)
    
    return decorator


def log_async_operation(operation_type: str, **context: Any) -> Callable[[F], F]:
    """
    Декоратор для логирования асинхронных операций с контекстом.
    
    Args:
        operation_type: Тип операции (например, "trade", "arbitrage", "fetch")
        **context: Дополнительный контекст для логирования
        
    Returns:
        Декоратор для асинхронной функции
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Получаем логгер операций
            operations_logger = get_operation_logger()
            # Подготавливаем контекст
            operation_context = {
                "operation_type": operation_type,
                "function": func.__name__,
                "timestamp_start": datetime.now().isoformat(),
                **context
            }
            
            start_time = time.time()
            
            try:
                # Логируем начало операции
                operations_logger.info({
                    "event": "operation_start",
                    **operation_context
                })
                
                # Выполняем функцию
                result = await func(*args, **kwargs)
                
                # Вычисляем время выполнения
                execution_time = (time.time() - start_time) * 1000
                
                # Логируем успешное завершение
                operations_logger.info({
                    "event": "operation_success",
                    "execution_time_ms": execution_time,
                    "timestamp_end": datetime.now().isoformat(),
                    **operation_context
                })
                
                return result
            except Exception as e:
                # Вычисляем время до ошибки
                execution_time = (time.time() - start_time) * 1000
                
                # Логируем ошибку
                operations_logger.error({
                    "event": "operation_error",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "execution_time_ms": execution_time,
                    "timestamp_end": datetime.now().isoformat(),
                    **operation_context
                }, exc_info=True)
                
                # Пробрасываем исключение дальше
                raise
        
        return cast(F, wrapper)
    
    return decorator
