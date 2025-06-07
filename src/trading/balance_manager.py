"""
Модуль управления балансами на биржах.
Отвечает за отслеживание и обновление балансов, проверку достаточности средств и расчет оптимальных размеров сделок.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Set
from decimal import Decimal
import time
from datetime import datetime, timedelta

from src.utils import get_logger
from src.data_fetching.exchange_factory import ExchangeFactory
from src.arbitrage_engine.opportunity_finder import ArbitrageOpportunity
from db.db_manager import DatabaseManager
from db.repository import BalanceRepository
from config import settings


class BalanceManager:
    """
    Менеджер для управления и отслеживания балансов на биржах.
    Обеспечивает методы для проверки наличия необходимых средств,
    расчета оптимальных размеров сделок и мониторинга изменений балансов.
    """
    
    def __init__(self, exchange_factory: ExchangeFactory, db_manager: DatabaseManager, 
                 balance_repo: BalanceRepository, update_interval: int = 60):
        """
        Инициализирует менеджер балансов.
        
        Args:
            exchange_factory: Фабрика для создания подключений к биржам
            db_manager: Менеджер базы данных
            balance_repo: Репозиторий для работы с балансами
            update_interval: Интервал автоматического обновления балансов (в секундах)
        """
        self.logger = get_logger(__name__)
        self.exchange_factory = exchange_factory
        self.db_manager = db_manager
        self.balance_repo = balance_repo
        self.update_interval = update_interval
        
        # Кэш балансов по биржам: {exchange_name: {asset