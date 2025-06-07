"""
Модуль конфигурации для крипто-арбитражного бота.
Содержит настройки и конфигурацию приложения.
"""

from config.settings import Settings
from config.exchanges import ExchangeConfig

# Инициализируем базовые настройки
settings = Settings()
exchange_config = ExchangeConfig()

__all__ = ["settings", "exchange_config"]
