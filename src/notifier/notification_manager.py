"""
Менеджер уведомлений, который координирует работу разных типов нотификаторов.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Union, Set, Type
from enum import Enum, auto

from src.notifier.base_notifier import BaseNotifier
from src.notifier.telegram_notifier import TelegramNotifier
from src.utils import get_logger, log_async_execution_time, log_async_operation


class NotificationLevel(Enum):
    """Уровни важности уведомлений."""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class NotificationType(Enum):
    """Типы уведомлений для фильтрации."""
    SYSTEM = auto()
    TRADE = auto()
    ARBITRAGE = auto()
    ALERT = auto()
    BALANCE = auto()
    ERROR = auto()


class NotificationManager:
    """
    Менеджер уведомлений, который координирует работу разных нотификаторов
    и предоставляет единый интерфейс для отправки уведомлений.
    """
    
    def __init__(
        self,
        min_level: NotificationLevel = NotificationLevel.INFO,
        cooldown_seconds: float = 5.0,
        enabled_types: Optional[Set[NotificationType]] = None
    ):
        """
        Инициализирует менеджер уведомлений.
        
        Args:
            min_level: Минимальный уровень уведомлений для отправки
            cooldown_seconds: Минимальное время между однотипными уведомлениями
            enabled_types: Набор разрешенных типов уведомлений (по умолчанию все)
        """
        self.logger = get_logger(__name__)
        self.notifiers: List[BaseNotifier] = []
        self.min_level = min_level
        self.cooldown_seconds = cooldown_seconds
        self.enabled_types = enabled_types or set(NotificationType)
        
        # Словарь для хранения времени последней отправки каждого типа уведомлений
        self.last_sent: Dict[NotificationType, float] = {
            ntype: 0.0 for ntype in NotificationType
        }
        
        # Словарь для хранения счетчиков пропущенных сообщений
        self.skipped_count: Dict[NotificationType, int] = {
            ntype: 0 for ntype in NotificationType
        }
        
        # Флаг инициализации
        self.is_initialized = False
    
    def add_notifier(self, notifier: BaseNotifier) -> None:
        """
        Добавляет нотификатор в список активных нотификаторов.
        
        Args:
            notifier: Экземпляр нотификатора для добавления
        """
        self.notifiers.append(notifier)
        self.logger.info(f"Добавлен нотификатор: {notifier.__class__.__name__}")
    
    async def initialize(self) -> bool:
        """
        Инициализирует все нотификаторы и проверяет их доступность.
        
        Returns:
            True, если хотя бы один нотификатор успешно инициализирован
        """
        if self.is_initialized:
            return True
        
        # Если нет нотификаторов, добавляем Telegram по умолчанию
        if not self.notifiers:
            try:
                telegram_notifier = TelegramNotifier()
                self.add_notifier(telegram_notifier)
            except ValueError as e:
                self.logger.warning(f"Не удалось создать Telegram-нотификатор: {str(e)}")
        
        # Инициализируем все нотификаторы
        init_results = await asyncio.gather(
            *[self._init_notifier(notifier) for notifier in self.notifiers],
            return_exceptions=True
        )
        
        # Проверяем, есть ли хотя бы один успешно инициализированный нотификатор
        success_count = sum(1 for result in init_results if result is True)
        
        if success_count > 0:
            self.is_initialized = True
            self.logger.info(f"Успешно инициализировано {success_count} из {len(self.notifiers)} нотификаторов")
            return True
        else:
            self.logger.error("Не удалось инициализировать ни один нотификатор")
            return False
    
    async def _init_notifier(self, notifier: BaseNotifier) -> bool:
        """
        Инициализирует отдельный нотификатор.
        
        Args:
            notifier: Нотификатор для инициализации
            
        Returns:
            True, если инициализация успешна
        """
        try:
            # Проверяем, есть ли у нотификатора метод initialize
            if hasattr(notifier, 'initialize') and callable(getattr(notifier, 'initialize')):
                result = await notifier.initialize()
                if result:
                    self.logger.info(f"Нотификатор {notifier.__class__.__name__} успешно инициализирован")
                    return True
                else:
                    self.logger.error(f"Не удалось инициализировать нотификатор {notifier.__class__.__name__}")
                    return False
            else:
                # Если метода initialize нет, считаем что нотификатор уже инициализирован
                self.logger.info(f"Нотификатор {notifier.__class__.__name__} не имеет метода initialize")
                return True
        except Exception as e:
            self.logger.error(f"Ошибка при инициализации нотификатора {notifier.__class__.__name__}: {str(e)}")
            return False
    
    def _check_cooldown(self, notification_type: NotificationType) -> bool:
        """
        Проверяет, прошло ли достаточно времени с последней отправки уведомления данного типа.
        
        Args:
            notification_type: Тип уведомления для проверки
            
        Returns:
            True, если уведомление можно отправить, False если нужно подождать
        """
        current_time = time.time()
        last_sent_time = self.last_sent.get(notification_type, 0.0)
        time_diff = current_time - last_sent_time
        
        if time_diff >= self.cooldown_seconds:
            self.last_sent[notification_type] = current_time
            
            # Добавляем информацию о пропущенных сообщениях, если они были
            skipped = self.skipped_count.get(notification_type, 0)
            if skipped > 0:
                self.logger.info(f"Пропущено {skipped} уведомлений типа {notification_type.name} из-за cooldown")
                self.skipped_count[notification_type] = 0
            
            return True
        else:
            # Увеличиваем счетчик пропущенных сообщений
            self.skipped_count[notification_type] = self.skipped_count.get(notification_type, 0) + 1
            return False
    
    def _check_level_and_type(self, level: NotificationLevel, notification_type: NotificationType) -> bool:
        """
        Проверяет, соответствуют ли уровень и тип уведомления настройкам фильтрации.
        
        Args:
            level: Уровень важности уведомления
            notification_type: Тип уведомления
            
        Returns:
            True, если уведомление проходит фильтры
        """
        # Проверяем уровень уведомления
        if level.value < self.min_level.value:
            return False
        
        # Проверяем тип уведомления
        if notification_type not in self.enabled_types:
            return False
        
        return True
    
    @log_async_execution_time
    async def _send_to_all_notifiers(self, coro_func, *args, **kwargs) -> List[bool]:
        """
        Отправляет уведомление через все активные нотификаторы.
        
        Args:
            coro_func: Функция корутины нотификатора для вызова
            *args, **kwargs: Аргументы для передачи функции
            
        Returns:
            Список результатов отправки для каждого нотификатора
        """
        if not self.is_initialized:
            if not await self.initialize():
                self.logger.error("Не удалось инициализировать нотификаторы")
                return [False] * len(self.notifiers)
        
        # Создаем список корутин для каждого нотификатора
        coroutines = []
        for notifier in self.notifiers:
            method = getattr(notifier, coro_func.__name__)
            coroutines.append(method(*args, **kwargs))
        
        # Выполняем все корутины параллельно
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Обрабатываем результаты
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Ошибка при отправке уведомления через {self.notifiers[i].__class__.__name__}: {str(result)}")
                processed_results.append(False)
            else:
                processed_results.append(result)
        
        return processed_results
    
    @log_async_operation(operation_type="notification", notification_type="message")
    async def send_message(self, message: str, level: NotificationLevel = NotificationLevel.INFO) -> bool:
        """
        Отправляет простое текстовое сообщение.
        
        Args:
            message: Текст сообщения
            level: Уровень важности
            
        Returns:
            True, если сообщение отправлено хотя бы через один нотификатор
        """
        # Проверяем фильтры
        if not self._check_level_and_type(level, NotificationType.ALERT):
            return False
        
        # Проверяем cooldown
        if not self._check_cooldown(NotificationType.ALERT):
            return False
        
        # Отправляем через все нотификаторы
        results = await self._send_to_all_notifiers(BaseNotifier.send_message, message)
        
        # Если хотя бы один нотификатор успешно отправил, считаем успехом
        return any(results)
    
    @log_async_operation(operation_type="notification", notification_type="alert")
    async def send_alert(
        self, 
        title: str, 
        message: str, 
        level: NotificationLevel = NotificationLevel.INFO
    ) -> bool:
        """
        Отправляет предупреждение или оповещение.
        
        Args:
            title: Заголовок сообщения
            message: Текст сообщения
            level: Уровень важности
            
        Returns:
            True, если сообщение отправлено хотя бы через один нотификатор
        """
        # Проверяем фильтры
        if not self._check_level_and_type(level, NotificationType.ALERT):
            return False
        
        # Проверяем cooldown
        if not self._check_cooldown(NotificationType.ALERT):
            return False
        
        # Преобразуем уровень из NotificationLevel в строку для send_alert
        level_str_map = {
            NotificationLevel.DEBUG: "info",
            NotificationLevel.INFO: "info",
            NotificationLevel.WARNING: "warning",
            NotificationLevel.ERROR: "error",
            NotificationLevel.CRITICAL: "critical"
        }
        level_str = level_str_map.get(level, "info")
        
        # Отправляем через все нотификаторы
        results = await self._send_to_all_notifiers(BaseNotifier.send_alert, title, message, level_str)
        
        # Если хотя бы один нотификатор успешно отправил, считаем успехом
        return any(results)
    
    @log_async_operation(operation_type="notification", notification_type="trade")
    async def send_trade_info(
        self, 
        exchange: str, 
        symbol: str, 
        operation: str, 
        amount: float, 
        price: float, 
        status: str,
        level: NotificationLevel = NotificationLevel.INFO,
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
            level: Уровень важности
            details: Дополнительные детали операции (опционально)
            
        Returns:
            True, если сообщение отправлено хотя бы через один нотификатор
        """
        # Проверяем фильтры
        if not self._check_level_and_type(level, NotificationType.TRADE):
            return False
        
        # Проверяем cooldown
        if not self._check_cooldown(NotificationType.TRADE):
            return False
        
        # Отправляем через все нотификаторы
        results = await self._send_to_all_notifiers(
            BaseNotifier.send_trade_info,
            exchange, symbol, operation, amount, price, status, details
        )
        
        # Если хотя бы один нотификатор успешно отправил, считаем успехом
        return any(results)
    
    @log_async_operation(operation_type="notification", notification_type="arbitrage")
    async def send_arbitrage_opportunity(
        self,
        buy_exchange: str,
        sell_exchange: str,
        symbol: str,
        buy_price: float,
        sell_price: float,
        profit_percent: float,
        estimated_profit: float,
        level: NotificationLevel = NotificationLevel.INFO,
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
            level: Уровень важности
            details: Дополнительные детали (опционально)
            
        Returns:
            True, если сообщение отправлено хотя бы через один нотификатор
        """
        # Проверяем фильтры
        if not self._check_level_and_type(level, NotificationType.ARBITRAGE):
            return False
        
        # Проверяем cooldown
        if not self._check_cooldown(NotificationType.ARBITRAGE):
            return False
        
        # Отправляем через все нотификаторы
        results = await self._send_to_all_notifiers(
            BaseNotifier.send_arbitrage_opportunity,
            buy_exchange, sell_exchange, symbol, buy_price, sell_price,
            profit_percent, estimated_profit, details
        )
        
        # Если хотя бы один нотификатор успешно отправил, считаем успехом
        return any(results)
    
    @log_async_operation(operation_type="notification", notification_type="system")
    async def send_system_status(
        self,
        status: str,
        balances: Dict[str, Dict[str, float]],
        active_tasks: int,
        errors_count: int,
        level: NotificationLevel = NotificationLevel.INFO,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Отправляет статус системы.
        
        Args:
            status: Общий статус системы (running, error, stopped)
            balances: Словарь с балансами по биржам
            active_tasks: Количество активных задач
            errors_count: Количество ошибок
            level: Уровень важности
            details: Дополнительные детали (опционально)
            
        Returns:
            True, если сообщение отправлено хотя бы через один нотификатор
        """
        # Проверяем фильтры
        if not self._check_level_and_type(level, NotificationType.SYSTEM):
            return False
        
        # Проверяем cooldown
        if not self._check_cooldown(NotificationType.SYSTEM):
            return False
        
        # Отправляем через все нотификаторы
        results = await self._send_to_all_notifiers(
            BaseNotifier.send_system_status,
            status, balances, active_tasks, errors_count, details
        )
        
        # Если хотя бы один нотификатор успешно отправил, считаем успехом
        return any(results)
    
    async def send_error(
        self, 
        title: str, 
        error_message: str, 
        stacktrace: Optional[str] = None,
        level: NotificationLevel = NotificationLevel.ERROR
    ) -> bool:
        """
        Отправляет сообщение об ошибке.
        
        Args:
            title: Заголовок сообщения об ошибке
            error_message: Текст ошибки
            stacktrace: Трассировка стека (опционально)
            level: Уровень важности (по умолчанию ERROR)
            
        Returns:
            True, если сообщение отправлено хотя бы через один нотификатор
        """
        # Проверяем фильтры
        if not self._check_level_and_type(level, NotificationType.ERROR):
            return False
        
        # Проверяем cooldown
        if not self._check_cooldown(NotificationType.ERROR):
            return False
        
        # Формируем сообщение
        message = f"{error_message}"
        
        # Добавляем трассировку стека, если она предоставлена
        if stacktrace:
            # Ограничиваем длину трассировки стека
            if len(stacktrace) > 800:
                stacktrace = stacktrace[:800] + "...\n(трассировка стека сокращена)"
            message += f"\n\nТрассировка стека:\n{stacktrace}"
        
        # Отправляем через send_alert
        return await self.send_alert(title, message, "error")
        
    @log_async_operation(operation_type="notification", notification_type="general")
    async def send_notification(
        self,
        title: str,
        message: str,
        level: str = "info"
    ) -> bool:
        """
        Отправляет общее уведомление через все доступные нотификаторы.
        
        Args:
            title: Заголовок сообщения
            message: Текст сообщения
            level: Уровень важности (info, warning, error, critical)
            
        Returns:
            True, если сообщение отправлено хотя бы через один нотификатор
        """
        # Конвертируем строковый уровень в NotificationLevel
        level_map = {
            "debug": NotificationLevel.DEBUG,
            "info": NotificationLevel.INFO,
            "warning": NotificationLevel.WARNING,
            "error": NotificationLevel.ERROR,
            "critical": NotificationLevel.CRITICAL
        }
        notification_level = level_map.get(level.lower(), NotificationLevel.INFO)
        
        # Используем существующий метод send_alert
        return await self.send_alert(title, message, notification_level)
    
    def set_min_level(self, level: NotificationLevel) -> None:
        """
        Устанавливает минимальный уровень уведомлений для отправки.
        
        Args:
            level: Новый минимальный уровень
        """
        self.min_level = level
        self.logger.info(f"Установлен минимальный уровень уведомлений: {level.name}")
    
    def set_cooldown(self, seconds: float) -> None:
        """
        Устанавливает время задержки между однотипными уведомлениями.
        
        Args:
            seconds: Время в секундах
        """
        self.cooldown_seconds = seconds
        self.logger.info(f"Установлено время задержки между уведомлениями: {seconds} сек")
    
    def enable_notification_type(self, notification_type: NotificationType) -> None:
        """
        Включает определенный тип уведомлений.
        
        Args:
            notification_type: Тип уведомлений для включения
        """
        self.enabled_types.add(notification_type)
        self.logger.info(f"Включен тип уведомлений: {notification_type.name}")
    
    def disable_notification_type(self, notification_type: NotificationType) -> None:
        """
        Выключает определенный тип уведомлений.
        
        Args:
            notification_type: Тип уведомлений для выключения
        """
        if notification_type in self.enabled_types:
            self.enabled_types.remove(notification_type)
            self.logger.info(f"Выключен тип уведомлений: {notification_type.name}")
    
    async def close(self) -> None:
        """
        Корректно закрывает все нотификаторы.
        """
        close_tasks = []
        
        for notifier in self.notifiers:
            if hasattr(notifier, 'close') and callable(getattr(notifier, 'close')):
                close_tasks.append(notifier.close())
        
        if close_tasks:
            # Закрываем все нотификаторы параллельно
            await asyncio.gather(*close_tasks, return_exceptions=True)
            self.logger.info("Все нотификаторы корректно закрыты")
