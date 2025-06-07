"""
Модуль для отправки уведомлений о важных событиях и операциях бота.
Поддерживает уведомления через Telegram.
"""

from src.notifier.notification_manager import NotificationManager
from src.notifier.telegram_notifier import TelegramNotifier

__all__ = ["NotificationManager", "TelegramNotifier"]
