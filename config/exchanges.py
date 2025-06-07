"""
Конфигурация бирж и торговых параметров.
Безопасно загружает API ключи из переменных окружения.
"""

import os
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()


class ExchangeCredentials(BaseModel):
    """
    Модель для хранения учетных данных биржи.
    Данные API берутся из переменных окружения.
    """
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    passphrase: Optional[str] = None  # Для бирж, требующих дополнительную авторизацию
    
    def is_configured(self) -> bool:
        """Проверяет, настроены ли все необходимые учетные данные."""
        # Для большинства бирж нужен API ключ и секрет
        # Для некоторых (например, KuCoin) также нужен passphrase
        if self.passphrase is not None:
            return bool(self.api_key and self.secret_key and self.passphrase)
        return bool(self.api_key and self.secret_key)


class ExchangeSettings(BaseModel):
    """
    Настройки для определенной биржи.
    """
    name: str
    enabled: bool = True
    base_url: Optional[str] = None
    credentials: ExchangeCredentials
    rate_limit: int = 10  # Лимит запросов в секунду
    timeout: int = 30  # Таймаут подключения в секундах
    use_testnet: bool = False  # Использовать тестовую сеть
    retries: int = 3  # Количество попыток при ошибке подключения
    fee_rate: float = 0.1  # Комиссия биржи в процентах
    
    def to_ccxt_config(self) -> Dict[str, Any]:
        """Преобразует настройки в формат для CCXT библиотеки."""
        config = {
            "apiKey": self.credentials.api_key,
            "secret": self.credentials.secret_key,
            "enableRateLimit": True,
            "timeout": self.timeout * 1000,  # CCXT использует миллисекунды
        }
        
        # Добавляем passphrase для бирж, которые его требуют
        if self.credentials.passphrase:
            config["password"] = self.credentials.passphrase
        
        # Добавляем URL при необходимости
        if self.base_url:
            config["urls"] = {"api": self.base_url}
        
        # Настройка тестовой сети
        if self.use_testnet:
            config["options"] = {"defaultType": "testnet"}
            
        return config


class ExchangeConfig:
    """
    Класс для конфигурации бирж и получения настроек подключения.
    """
    def __init__(self):
        """Инициализация с загрузкой данных из переменных окружения."""
        # Настройки для Binance
        self.binance = ExchangeSettings(
            name="binance",
            credentials=ExchangeCredentials(
                api_key=os.getenv("BINANCE_API_KEY"),
                secret_key=os.getenv("BINANCE_SECRET_KEY")
            )
        )
        
        # Настройки для KuCoin
        self.kucoin = ExchangeSettings(
            name="kucoin",
            credentials=ExchangeCredentials(
                api_key=os.getenv("KUCOIN_API_KEY"),
                secret_key=os.getenv("KUCOIN_SECRET_KEY"),
                passphrase=os.getenv("KUCOIN_PASSPHRASE")
            )
        )
        
        # Настройки для OKX
        self.okx = ExchangeSettings(
            name="okx",
            credentials=ExchangeCredentials(
                api_key=os.getenv("OKX_API_KEY"),
                secret_key=os.getenv("OKX_SECRET_KEY"),
                passphrase=os.getenv("OKX_PASSPHRASE")
            )
        )
        
        # Словарь всех бирж для удобства доступа
        self._exchanges = {
            "binance": self.binance,
            "kucoin": self.kucoin,
            "okx": self.okx
        }
    
    def get_enabled_exchanges(self) -> List[ExchangeSettings]:
        """Возвращает список включенных бирж."""
        return [exchange for exchange in self._exchanges.values() if exchange.enabled]
    
    def get_exchange(self, name: str) -> Optional[ExchangeSettings]:
        """Получает настройки биржи по имени."""
        return self._exchanges.get(name.lower())
    
    def is_exchange_configured(self, name: str) -> bool:
        """Проверяет, настроена ли биржа для использования."""
        exchange = self.get_exchange(name)
        if not exchange or not exchange.enabled:
            return False
        return exchange.credentials.is_configured()
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Преобразует конфигурацию в словарь для сериализации.
        Не включает API ключи для безопасности.
        """
        result = {}
        for name, exchange in self._exchanges.items():
            result[name] = {
                "enabled": exchange.enabled,
                "rate_limit": exchange.rate_limit,
                "timeout": exchange.timeout,
                "use_testnet": exchange.use_testnet,
                "retries": exchange.retries,
                "is_configured": exchange.credentials.is_configured()
            }
        return result
