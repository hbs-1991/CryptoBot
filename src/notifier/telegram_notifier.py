"""
Реализация отправки уведомлений через Telegram.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Union, Tuple
import aiohttp
from urllib.parse import urljoin

from src.notifier.base_notifier import BaseNotifier
from src.utils import get_logger, log_async_execution_time
from config.settings import Settings

# Получаем настройки
settings = Settings()

# Базовый URL для Telegram Bot API
TELEGRAM_API_URL = "https://api.telegram.org/bot"

# Эмодзи для разных уровней уведомлений
EMOJI_MAP = {
    "info": "ℹ️",
    "success": "✅",
    "warning": "⚠️",
    "error": "❌",
    "critical": "🚨",
    "trade": "💰",
    "arbitrage": "💹",
    "system": "🤖",
    "buy": "🟢",
    "sell": "🔴",
    "executed": "✅",
    "failed": "❌",
    "pending": "⏳",
    "running": "🟢",
    "stopped": "🔴",
}


class TelegramNotifier(BaseNotifier):
    """
    Класс для отправки уведомлений через Telegram.
    Использует Bot API для отправки сообщений в указанный чат.
    """
    
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        """
        Инициализирует Telegram нотификатор.
        
        Args:
            token: Токен Telegram бота (опционально, по умолчанию из настроек)
            chat_id: ID чата для отправки сообщений (опционально, по умолчанию из настроек)
        """
        self.logger = get_logger(__name__)
        
        # Получаем токен и chat_id из параметров или из настроек
        self.token = token or settings.TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or settings.TELEGRAM_CHAT_ID
        
        # Проверяем наличие обязательных параметров
        if not self.token:
            self.logger.error("Токен Telegram бота не указан")
            raise ValueError("Токен Telegram бота не указан")
        
        if not self.chat_id:
            self.logger.error("ID чата Telegram не указан")
            raise ValueError("ID чата Telegram не указан")
        
        # Формируем базовый URL API для данного бота
        self.api_url = f"{TELEGRAM_API_URL}{self.token}/"
        
        # Лимиты и счетчики для контроля rate limit
        self.max_messages_per_second = 30  # Ограничение Telegram API
        self.message_queue = asyncio.Queue()
        self.message_count = 0
        self.last_reset_time = time.time()
        
        # Флаг активности обработчика очереди
        self.is_queue_processor_running = False
        
        # Флаг инициализации
        self.is_initialized = False
        
        # Флаг доступности Telegram API
        self.is_available = False
        
        # Семафор для ограничения одновременных запросов
        self.semaphore = asyncio.Semaphore(10)  # Максимум 10 одновременных запросов
    
    async def initialize(self) -> bool:
        """
        Инициализирует соединение с Telegram API и проверяет валидность токена и chat_id.
        
        Returns:
            True, если инициализация прошла успешно, иначе False
        """
        if self.is_initialized:
            return True
        
        try:
            # Проверяем доступность API и валидность токена
            async with aiohttp.ClientSession() as session:
                async with session.get(urljoin(self.api_url, "getMe")) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("ok"):
                            self.logger.info(f"Успешное подключение к Telegram Bot API. Бот: {data['result']['first_name']}")
                            
                            # Проверяем валидность chat_id, отправляя тестовое сообщение
                            chat_test_result = await self._send_chat_message(
                                f"🤖 Crypto Arbitrage Bot запущен. Версия: {settings.VERSION}"
                            )
                            
                            if chat_test_result[0]:
                                self.is_initialized = True
                                self.is_available = True
                                
                                # Запускаем обработчик очереди сообщений
                                if not self.is_queue_processor_running:
                                    asyncio.create_task(self._process_message_queue())
                                    self.is_queue_processor_running = True
                                
                                self.logger.info("Telegram-нотификатор успешно инициализирован")
                                return True
                            else:
                                self.logger.error(f"Ошибка отправки тестового сообщения: {chat_test_result[1]}")
                                return False
                        else:
                            self.logger.error(f"Ошибка проверки токена: {data.get('description')}")
                            return False
                    else:
                        self.logger.error(f"Ошибка подключения к Telegram API. Код: {response.status}")
                        return False
        except Exception as e:
            self.logger.error(f"Ошибка инициализации Telegram-нотификатора: {str(e)}")
            return False
    
    async def _send_chat_message(self, text: str, parse_mode: str = "HTML") -> Tuple[bool, str]:
        """
        Отправляет сообщение в чат через Telegram API.
        
        Args:
            text: Текст сообщения для отправки
            parse_mode: Режим форматирования текста (HTML, Markdown)
            
        Returns:
            Кортеж (успех, сообщение об ошибке)
        """
        try:
            async with self.semaphore:
                # Проверяем, не превышен ли лимит сообщений
                current_time = time.time()
                if current_time - self.last_reset_time >= 1.0:
                    self.message_count = 0
                    self.last_reset_time = current_time
                
                if self.message_count >= self.max_messages_per_second:
                    # Если превышен лимит, ждем до следующей секунды
                    await asyncio.sleep(1.0 - (current_time - self.last_reset_time))
                    self.message_count = 0
                    self.last_reset_time = time.time()
                
                # Формируем JSON для отправки
                params = {
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": True
                }
                
                # Отправляем запрос
                async with aiohttp.ClientSession() as session:
                    async with session.post(urljoin(self.api_url, "sendMessage"), json=params) as response:
                        self.message_count += 1
                        
                        if response.status == 200:
                            data = await response.json()
                            if data.get("ok"):
                                return True, ""
                            else:
                                return False, data.get("description", "Неизвестная ошибка")
                        else:
                            error_text = await response.text()
                            return False, f"HTTP ошибка {response.status}: {error_text}"
        except Exception as e:
            return False, str(e)
    
    async def _process_message_queue(self) -> None:
        """
        Обрабатывает очередь сообщений, соблюдая ограничения API.
        Запускается как отдельная задача при инициализации.
        """
        self.logger.info("Запущен обработчик очереди сообщений Telegram")
        
        while True:
            try:
                # Получаем сообщение из очереди
                message = await self.message_queue.get()
                
                # Отправляем сообщение
                result, error = await self._send_chat_message(message)
                
                if not result:
                    self.logger.error(f"Ошибка отправки сообщения: {error}")
                
                # Сообщаем очереди, что задача выполнена
                self.message_queue.task_done()
                
                # Небольшая задержка для снижения нагрузки
                await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                self.logger.info("Обработчик очереди сообщений Telegram остановлен")
                break
            except Exception as e:
                self.logger.error(f"Ошибка в обработчике очереди сообщений: {str(e)}")
                await asyncio.sleep(1)  # Пауза в случае ошибки
    
    async def _add_to_queue(self, message: str) -> None:
        """
        Добавляет сообщение в очередь отправки.
        
        Args:
            message: Текст сообщения для отправки
        """
        await self.message_queue.put(message)
    
    @log_async_execution_time
    async def send_message(self, message: str) -> bool:
        """
        Отправляет простое текстовое сообщение.
        
        Args:
            message: Текст сообщения для отправки
            
        Returns:
            True в случае успешной отправки, иначе False
        """
        if not self.is_initialized:
            if not await self.initialize():
                return False
        
        # Отправляем сообщение через очередь
        await self._add_to_queue(message)
        return True
    
    @log_async_execution_time
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
        if not self.is_initialized:
            if not await self.initialize():
                return False
        
        # Получаем эмодзи для уровня
        emoji = EMOJI_MAP.get(level.lower(), EMOJI_MAP["info"])
        
        # Форматируем сообщение
        formatted_message = f"{emoji} <b>{title}</b>\n\n{message}"
        
        # Отправляем сообщение через очередь
        await self._add_to_queue(formatted_message)
        return True
    
    @log_async_execution_time
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
        if not self.is_initialized:
            if not await self.initialize():
                return False
        
        # Получаем эмодзи для операции и статуса
        op_emoji = EMOJI_MAP.get(operation.lower(), "")
        status_emoji = EMOJI_MAP.get(status.lower(), "")
        
        # Рассчитываем общую стоимость
        total = amount * price
        
        # Форматируем сообщение
        formatted_message = (
            f"{EMOJI_MAP['trade']} <b>Торговая операция</b> {status_emoji}\n\n"
            f"<b>Биржа:</b> {exchange.upper()}\n"
            f"<b>Пара:</b> {symbol}\n"
            f"<b>Операция:</b> {op_emoji} {operation.upper()}\n"
            f"<b>Количество:</b> {amount}\n"
            f"<b>Цена:</b> {price} USDT\n"
            f"<b>Всего:</b> {total:.2f} USDT\n"
            f"<b>Статус:</b> {status.upper()}"
        )
        
        # Добавляем дополнительные детали, если они есть
        if details:
            formatted_message += "\n\n<b>Дополнительно:</b>"
            for key, value in details.items():
                formatted_message += f"\n• <b>{key}:</b> {value}"
        
        # Отправляем сообщение через очередь
        await self._add_to_queue(formatted_message)
        return True
    
    @log_async_execution_time
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
        if not self.is_initialized:
            if not await self.initialize():
                return False
        
        # Форматируем сообщение
        formatted_message = (
            f"{EMOJI_MAP['arbitrage']} <b>Арбитражная возможность</b>\n\n"
            f"<b>Пара:</b> {symbol}\n\n"
            f"<b>Покупка:</b> {buy_exchange.upper()} - {buy_price} USDT\n"
            f"<b>Продажа:</b> {sell_exchange.upper()} - {sell_price} USDT\n\n"
            f"<b>Разница цен:</b> {sell_price - buy_price:.8f} USDT\n"
            f"<b>Прибыль:</b> {profit_percent:.2f}%\n"
            f"<b>Оценочная прибыль:</b> {estimated_profit:.2f} USDT"
        )
        
        # Добавляем дополнительные детали, если они есть
        if details:
            formatted_message += "\n\n<b>Дополнительно:</b>"
            for key, value in details.items():
                formatted_message += f"\n• <b>{key}:</b> {value}"
        
        # Отправляем сообщение через очередь
        await self._add_to_queue(formatted_message)
        return True
    
    @log_async_execution_time
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
        if not self.is_initialized:
            if not await self.initialize():
                return False
        
        # Получаем эмодзи для статуса
        status_emoji = EMOJI_MAP.get(status.lower(), "")
        
        # Форматируем сообщение
        formatted_message = (
            f"{EMOJI_MAP['system']} <b>Статус системы</b>\n\n"
            f"<b>Состояние:</b> {status_emoji} {status.upper()}\n"
            f"<b>Активные задачи:</b> {active_tasks}\n"
            f"<b>Ошибки:</b> {errors_count}\n\n"
            f"<b>Балансы:</b>"
        )
        
        # Добавляем информацию о балансах
        for exchange, assets in balances.items():
            formatted_message += f"\n\n<b>{exchange.upper()}</b>:"
            for asset, amount in assets.items():
                formatted_message += f"\n• {asset}: {amount}"
        
        # Добавляем дополнительные детали, если они есть
        if details:
            formatted_message += "\n\n<b>Дополнительно:</b>"
            for key, value in details.items():
                formatted_message += f"\n• <b>{key}:</b> {value}"
        
        # Отправляем сообщение через очередь
        await self._add_to_queue(formatted_message)
        return True
    
    async def close(self) -> None:
        """
        Корректно закрывает соединения и освобождает ресурсы.
        """
        # Ждем, пока очередь сообщений опустеет
        if self.is_initialized and self.is_queue_processor_running:
            self.logger.info("Ожидание завершения отправки всех сообщений Telegram...")
            await self.message_queue.join()
            self.is_queue_processor_running = False
            self.logger.info("Telegram-нотификатор корректно завершил работу")
