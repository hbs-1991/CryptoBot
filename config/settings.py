"""
Основные настройки приложения, включая загрузку и валидацию
конфигурационных данных из переменных окружения.
"""

import os
import logging
from typing import Dict, Union, Optional, List, Any
from dotenv import load_dotenv
from pydantic import field_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Загружаем переменные окружения из .env файла
load_dotenv()


class Settings(BaseSettings):
    """
    Основной класс настроек приложения.
    Использует Pydantic для валидации и управления настройками.
    """
    # Общие настройки
    APP_NAME: str = "Crypto Arbitrage Bot"
    VERSION: str = "0.1.0"
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    SIMULATION_MODE: bool = Field(
        default=True, 
        description="Режим симуляции без реальной торговли"
    )
    
    # Настройки базы данных
    DATABASE_URL: str = Field(
        default="sqlite:///db/arbitrage.db",
        description="URL-адрес базы данных"
    )
    DEBUG_SQL: bool = Field(
        default=False,
        description="Включает подробный вывод SQL-запросов в логи"
    )
    
    # Настройки Telegram
    TELEGRAM_BOT_TOKEN: Optional[str] = Field(
        default=None,
        description="Токен Telegram бота для отправки уведомлений"
    )
    TELEGRAM_CHAT_ID: Optional[str] = Field(
        default=None, 
        description="ID чата для отправки уведомлений"
    )
    TELEGRAM_ENABLED: bool = Field(
        default=False,
        description="Включает отправку уведомлений в Telegram"
    )
    
    # Параметры арбитражных операций
    MIN_PROFIT_PERCENTAGE: float = Field(
        default=0.5,
        description="Минимальный процент прибыли для арбитражной сделки"
    )
    MAX_ORDER_AMOUNT: float = Field(
        default=5000.0,
        description="Максимальная сумма одного ордера в USD"
    )
    ALLOWED_PAIRS: List[str] = Field(
        default=[
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "AVAX/USDT",
    "MATIC/USDT",
    "TON/USDT",
    "TRX/USDT",
    "LTC/USDT",
    "LINK/USDT",
    "SHIB/USDT"
    ],
        description="Список разрешенных торговых пар"
    )
    MARKET_SCAN_INTERVAL: float = Field(
        default=10.0,
        description="Интервал сканирования рынка в секундах"
    )
    SIGNIFICANT_PRICE_CHANGE_THRESHOLD: float = Field(
        default=0.3,
        description="Порог значимого изменения цены в процентах"
    )
    
    # Настройки управления рисками
    MAX_TRADES_PER_PAIR: int = Field(
        default=3,
        description="Максимальное количество одновременных сделок на одну пару"
    )
    MIN_LIQUIDITY_REQUIREMENT: float = Field(
        default=1000.0,
        description="Минимальная ликвидность актива в USD"
    )
    MAX_PRICE_DEVIATION: float = Field(
        default=5.0,
        description="Максимальное отклонение цены от средней в процентах"
    )
    MAX_ACTIVE_TRADES: int = Field(
        default=20,
        description="Максимальное количество активных сделок"
    )
    MAX_TRADE_DURATION_MS: int = Field(
        default=600000,
        description="Максимальная продолжительность сделки в миллисекундах"
    )
    EMERGENCY_STOP_LOSS_PERCENTAGE: float = Field(
        default=-1.0,
        description="Процент убытка, при котором сделка закрывается экстренно"
    )
    
    # Настройки уведомлений
    NOTIFY_ON_OPPORTUNITY: bool = Field(
        default=True,
        description="Отправлять уведомления при обнаружении арбитражной возможности"
    )
    
    # Настройки симуляции
    INITIAL_BALANCES: Dict[str, Dict[str, float]] = Field(
        default={
            "binance": {"USDT": 10000.0, "BTC": 0.1, "ETH": 1.0, "BNB": 10.0},
            "kucoin": {"USDT": 10000.0, "BTC": 0.1, "ETH": 1.0, "BNB": 10.0},
            "okx": {"USDT": 10000.0, "BTC": 0.1, "ETH": 1.0, "BNB": 10.0}
        },
        description="Начальные балансы для симуляции"
    )
    
    USE_MARKET_SIMULATOR: bool = Field(
        default=True,
        description="Использовать симулятор рыночных данных вместо реальных данных с бирж"
    )
    
    # API ключи и секреты храним только как ссылки на переменные окружения
    # для безопасности и не экспортируем напрямую
    
    @field_validator("LOG_LEVEL")
    def validate_log_level(cls, v: str) -> str:
        """Проверяет, что уровень логирования корректный."""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"Уровень логирования должен быть одним из {allowed_levels}")
        return v.upper()
    
    @field_validator("MIN_PROFIT_PERCENTAGE")
    def validate_min_profit(cls, v: float) -> float:
        """Проверяет, что минимальная прибыль положительная."""
        if v <= 0:
            raise ValueError("Минимальный процент прибыли должен быть положительным числом")
        return v
    
    @field_validator("MAX_ORDER_AMOUNT")
    def validate_max_order(cls, v: float) -> float:
        """Проверяет, что максимальная сумма положительная."""
        if v <= 0:
            raise ValueError("Максимальная сумма ордера должна быть положительным числом")
        return v
    
    def get_log_level_enum(self) -> int:
        """Возвращает числовое значение уровня логирования."""
        log_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        return log_levels.get(self.LOG_LEVEL, logging.INFO)
    
    # Конфигурация для настроек
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"  # Игнорировать лишние поля из переменных окружения
    )
        
    def dict_config(self) -> Dict[str, Any]:
        """Возвращает настройки в виде словаря для логирования."""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": self.get_log_level_enum(),
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": self.get_log_level_enum(),
                    "formatter": "default",
                    "filename": "logs/app.log",
                    "maxBytes": 10485760,  # 10 MB
                    "backupCount": 5,
                    "encoding": "utf8",
                },
            },
            "root": {
                "level": self.get_log_level_enum(),
                "handlers": ["console", "file"],
            },
        }
