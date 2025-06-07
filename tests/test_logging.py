"""
Тесты для системы логирования.
"""

import os
import sys
import logging
import json
import asyncio
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import shutil

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
    log_async_operation,
    ContextualAdapter,
    JsonFormatter
)


class TestLogging(unittest.TestCase):
    """Тесты для системы логирования."""
    
    @classmethod
    def setUpClass(cls):
        """Настройка перед всеми тестами."""
        # Создаем временную директорию для логов тестов
        cls.test_logs_dir = tempfile.mkdtemp(prefix="test_logs_")
        
        # Переопределяем путь к логам для тестов
        cls.original_logs_dir = Path("logs")
        cls.temp_logs_dir = Path(cls.test_logs_dir)
        
        # Создаем мок-объект для настроек с временным путем к логам
        cls.settings_patcher = patch("src.utils.logger.settings")
        cls.mock_settings = cls.settings_patcher.start()
        cls.mock_settings.dict_config.return_value = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": logging.INFO,
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": logging.INFO,
                    "formatter": "default",
                    "filename": str(cls.temp_logs_dir / "app.log"),
                    "maxBytes": 10485760,
                    "backupCount": 5,
                    "encoding": "utf8",
                },
            },
            "root": {
                "level": logging.INFO,
                "handlers": ["console", "file"],
            },
        }
        
        # Настраиваем логирование для тестов
        cls.temp_logs_dir.mkdir(exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        """Очистка после всех тестов."""
        # Останавливаем патчеры
        cls.settings_patcher.stop()
        
        # Удаляем временную директорию с логами
        shutil.rmtree(cls.test_logs_dir)
    
    def setUp(self):
        """Настройка перед каждым тестом."""
        # Очищаем логгеры перед каждым тестом
        logging.getLogger().handlers = []
        # Повторно настраиваем логирование для каждого теста
        setup_logging()
    
    def tearDown(self):
        """Очистка после каждого теста."""
        pass
    
    def test_setup_logging(self):
        """Тест инициализации системы логирования."""
        # Проверяем, что корневой логгер настроен
        root_logger = logging.getLogger()
        self.assertTrue(root_logger.handlers)
        
        # Проверяем, что уровень логирования установлен
        self.assertEqual(root_logger.level, logging.INFO)
    
    def test_get_logger(self):
        """Тест получения логгера по имени."""
        logger_name = "test_logger"
        logger = get_logger(logger_name)
        
        # Проверяем, что логгер имеет правильное имя
        self.assertEqual(logger.name, logger_name)
        
        # Проверяем, что логгер использует настройки корневого логгера
        self.assertEqual(logger.level, logging.getLogger().level)
    
    def test_get_operation_logger(self):
        """Тест получения логгера для операций."""
        logger = get_operation_logger()
        
        # Проверяем, что возвращается логгер с правильным именем
        self.assertEqual(logger.name, "operations")
    
    def test_contextual_adapter(self):
        """Тест адаптера для контекстного логирования."""
        # Создаем базовый логгер
        test_logger = logging.getLogger("test_context")
        
        # Создаем контекстный адаптер
        context = {"user_id": "12345", "session": "abcde"}
        adapter = ContextualAdapter(test_logger, context)
        
        # Проверяем, что контекст сохранен
        self.assertEqual(adapter.context, context)
        
        # Проверяем, что метод with_context создает новый адаптер с расширенным контекстом
        new_context = {"order_id": "67890"}
        new_adapter = adapter.with_context(**new_context)
        
        # Проверяем, что новый адаптер содержит объединенный контекст
        expected_context = {**context, **new_context}
        self.assertEqual(new_adapter.context, expected_context)
    
    def test_get_contextual_logger(self):
        """Тест получения контекстного логгера."""
        logger_name = "test_contextual"
        context = {"component": "test", "user_id": "12345"}
        
        # Получаем контекстный логгер
        logger = get_contextual_logger(logger_name, **context)
        
        # Проверяем, что это экземпляр ContextualAdapter
        self.assertIsInstance(logger, ContextualAdapter)
        
        # Проверяем, что контекст сохранен
        self.assertEqual(logger.context, context)
    
    @patch("src.utils.logger.time.time")
    def test_log_execution_time(self, mock_time):
        """Тест декоратора для измерения времени выполнения."""
        # Настраиваем мок-объект для time.time
        mock_time.side_effect = [10.0, 11.5]  # Имитируем 1.5 секунды выполнения
        
        # Создаем функцию с декоратором
        @log_execution_time
        def test_function():
            return "test result"
        
        # Выполняем функцию
        result = test_function()
        
        # Проверяем, что функция вернула ожидаемый результат
        self.assertEqual(result, "test result")
        
        # Проверяем, что time.time вызывался дважды
        self.assertEqual(mock_time.call_count, 2)
    
    @patch("src.utils.logger.time.time")
    def test_log_execution_time_with_exception(self, mock_time):
        """Тест декоратора для измерения времени при возникновении исключения."""
        # Настраиваем мок-объект для time.time
        mock_time.side_effect = [10.0, 11.5]  # Имитируем 1.5 секунды выполнения
        
        # Создаем функцию с декоратором, которая вызывает исключение
        @log_execution_time
        def test_function_with_error():
            raise ValueError("Test error")
        
        # Проверяем, что исключение проброшено дальше
        with self.assertRaises(ValueError):
            test_function_with_error()
        
        # Проверяем, что time.time вызывался дважды
        self.assertEqual(mock_time.call_count, 2)
    
    @patch("src.utils.logger.time.time")
    def test_log_operation(self, mock_time):
        """Тест декоратора для логирования операций."""
        # Настраиваем мок-объект для time.time
        mock_time.side_effect = [10.0, 11.5]  # Имитируем 1.5 секунды выполнения
        
        # Создаем мок для операционного логгера
        mock_logger = MagicMock()
        
        with patch("src.utils.logger.get_operation_logger", return_value=mock_logger):
            # Создаем функцию с декоратором
            @log_operation(operation_type="test_op", param="value")
            def test_operation():
                return "operation result"
            
            # Выполняем операцию
            result = test_operation()
            
            # Проверяем, что функция вернула ожидаемый результат
            self.assertEqual(result, "operation result")
            
            # Проверяем, что логгер вызывался дважды (начало и конец операции)
            self.assertEqual(mock_logger.info.call_count, 2)
            
            # Проверяем, что первый вызов логгера содержит информацию о начале операции
            first_call_args = mock_logger.info.call_args_list[0][0][0]
            self.assertEqual(first_call_args["event"], "operation_start")
            self.assertEqual(first_call_args["operation_type"], "test_op")
            self.assertEqual(first_call_args["param"], "value")
            
            # Проверяем, что второй вызов логгера содержит информацию об успешном завершении
            second_call_args = mock_logger.info.call_args_list[1][0][0]
            self.assertEqual(second_call_args["event"], "operation_success")
            self.assertEqual(second_call_args["operation_type"], "test_op")
            self.assertEqual(second_call_args["param"], "value")
            self.assertIn("execution_time_ms", second_call_args)


class TestAsyncLogging(unittest.IsolatedAsyncioTestCase):
    """Тесты для асинхронных функций логирования."""
    
    @patch("src.utils.logger.time.time")
    async def test_log_async_execution_time(self, mock_time):
        """Тест декоратора для измерения времени выполнения асинхронной функции."""
        # Настраиваем мок-объект для time.time
        mock_time.side_effect = [10.0, 11.5]  # Имитируем 1.5 секунды выполнения
        
        # Создаем асинхронную функцию с декоратором
        @log_async_execution_time
        async def test_async_function():
            await asyncio.sleep(0.1)  # Небольшая задержка
            return "async result"
        
        # Выполняем функцию
        result = await test_async_function()
        
        # Проверяем, что функция вернула ожидаемый результат
        self.assertEqual(result, "async result")
        
        # Проверяем, что time.time вызывался дважды
        self.assertEqual(mock_time.call_count, 2)
    
    @patch("src.utils.logger.time.time")
    async def test_log_async_operation(self, mock_time):
        """Тест декоратора для логирования асинхронных операций."""
        # Настраиваем мок-объект для time.time
        mock_time.side_effect = [10.0, 11.5]  # Имитируем 1.5 секунды выполнения
        
        # Создаем мок для операционного логгера
        mock_logger = MagicMock()
        
        with patch("src.utils.logger.get_operation_logger", return_value=mock_logger):
            # Создаем асинхронную функцию с декоратором
            @log_async_operation(operation_type="async_op", exchange="binance")
            async def test_async_operation():
                await asyncio.sleep(0.1)  # Небольшая задержка
                return "async operation result"
            
            # Выполняем операцию
            result = await test_async_operation()
            
            # Проверяем, что функция вернула ожидаемый результат
            self.assertEqual(result, "async operation result")
            
            # Проверяем, что логгер вызывался дважды (начало и конец операции)
            self.assertEqual(mock_logger.info.call_count, 2)
            
            # Проверяем, что первый вызов логгера содержит информацию о начале операции
            first_call_args = mock_logger.info.call_args_list[0][0][0]
            self.assertEqual(first_call_args["event"], "operation_start")
            self.assertEqual(first_call_args["operation_type"], "async_op")
            self.assertEqual(first_call_args["exchange"], "binance")
            
            # Проверяем, что второй вызов логгера содержит информацию об успешном завершении
            second_call_args = mock_logger.info.call_args_list[1][0][0]
            self.assertEqual(second_call_args["event"], "operation_success")
            self.assertEqual(second_call_args["operation_type"], "async_op")
            self.assertEqual(second_call_args["exchange"], "binance")
            self.assertIn("execution_time_ms", second_call_args)


class TestJsonFormatter(unittest.TestCase):
    """Тесты для форматирования логов в JSON."""
    
    def test_json_formatter(self):
        """Тест форматирования лога в JSON формат."""
        # Создаем экземпляр JsonFormatter
        formatter = JsonFormatter()
        
        # Создаем запись лога
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test_file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Форматируем запись
        formatted = formatter.format(record)
        
        # Проверяем, что результат является валидным JSON
        json_data = json.loads(formatted)
        
        # Проверяем обязательные поля
        self.assertIn("timestamp", json_data)
        self.assertEqual(json_data["level"], "INFO")
        self.assertEqual(json_data["logger"], "test_logger")
        self.assertEqual(json_data["module"], "test_logger")
        self.assertEqual(json_data["line"], 42)
        self.assertEqual(json_data["message"], "Test message")
    
    def test_json_formatter_with_dict_message(self):
        """Тест форматирования лога с сообщением в виде словаря."""
        # Создаем экземпляр JsonFormatter
        formatter = JsonFormatter()
        
        # Создаем запись лога с сообщением-словарем
        message_dict = {
            "event": "test_event",
            "data": {"key": "value"},
            "count": 42
        }
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test_file.py",
            lineno=42,
            msg=message_dict,
            args=(),
            exc_info=None
        )
        
        # Форматируем запись
        formatted = formatter.format(record)
        
        # Проверяем, что результат является валидным JSON
        json_data = json.loads(formatted)
        
        # Проверяем, что поля из словаря включены в результат
        self.assertEqual(json_data["event"], "test_event")
        self.assertEqual(json_data["data"], {"key": "value"})
        self.assertEqual(json_data["count"], 42)
    
    def test_json_formatter_with_exception(self):
        """Тест форматирования лога с информацией об исключении."""
        # Создаем экземпляр JsonFormatter
        formatter = JsonFormatter()
        
        try:
            # Вызываем исключение
            raise ValueError("Test exception")
        except ValueError:
            # Создаем запись лога с информацией об исключении
            record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="test_file.py",
                lineno=42,
                msg="Exception occurred",
                args=(),
                exc_info=sys.exc_info()
            )
        
        # Форматируем запись
        formatted = formatter.format(record)
        
        # Проверяем, что результат является валидным JSON
        json_data = json.loads(formatted)
        
        # Проверяем, что информация об исключении включена в результат
        self.assertIn("exception", json_data)
        self.assertEqual(json_data["exception"]["type"], "ValueError")
        self.assertEqual(json_data["exception"]["message"], "Test exception")
        self.assertIn("traceback", json_data["exception"])


if __name__ == "__main__":
    unittest.main()
