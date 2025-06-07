"""
Модуль для поиска арбитражных возможностей между различными биржами.
Анализирует рыночные данные, выявляет ценовые расхождения и определяет 
потенциальные торговые возможности.
"""

import logging
from typing import Dict, List, Any, Optional, Set
from decimal import Decimal
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.data_fetching.data_normalizer import NormalizedMarketData
from src.arbitrage_engine.profit_calculator import ArbitrageOpportunity, ProfitCalculator


class OpportunityFinder:
    """
    Находит арбитражные возможности между биржами, анализируя
    расхождения в ценах на одни и те же активы.
    """
    
    def __init__(
        self,
        exchange_fees: Dict[str, float],
        min_profit_percentage: float = 0.5,
        max_active_opportunities: int = 10,
        min_check_interval_ms: int = 100,
        target_exchanges: Optional[List[str]] = None,
        target_symbols: Optional[List[str]] = None,
        excluded_exchanges: Optional[List[str]] = None,
        excluded_symbols: Optional[List[str]] = None
    ):
        """
        Инициализация поисковика арбитражных возможностей.
        
        Args:
            exchange_fees: Словарь с комиссиями для каждой биржи в процентах
            min_profit_percentage: Минимальный процент прибыли для признания сделки выгодной
            max_active_opportunities: Максимальное количество активных возможностей для отслеживания
            min_check_interval_ms: Минимальный интервал между проверками одного и того же символа в мс
            target_exchanges: Список бирж для анализа (None = все доступные)
            target_symbols: Список символов для анализа (None = все доступные)
            excluded_exchanges: Список бирж, исключенных из анализа
            excluded_symbols: Список символов, исключенных из анализа
        """
        self.logger = logging.getLogger(__name__)
        
        # Инициализируем калькулятор прибыли
        self.profit_calculator = ProfitCalculator(
            exchange_fees=exchange_fees,
            min_profit_percentage=min_profit_percentage
        )
        
        self.max_active_opportunities = max_active_opportunities
        self.min_check_interval_ms = min_check_interval_ms
        
        # Преобразуем списки в множества для быстрой проверки
        self.target_exchanges = set(target_exchanges) if target_exchanges else None
        self.target_symbols = set(target_symbols) if target_symbols else None
        self.excluded_exchanges = set(excluded_exchanges) if excluded_exchanges else set()
        self.excluded_symbols = set(excluded_symbols) if excluded_symbols else set()
        
        # Кэш последних проверок для каждого символа
        self._last_check_timestamps: Dict[str, int] = {}
        
        # Активные арбитражные возможности
        self.active_opportunities: Dict[str, ArbitrageOpportunity] = {}
        
        self.logger.info(
            f"OpportunityFinder initialized with {len(exchange_fees)} exchanges, "
            f"min profit: {min_profit_percentage}%, "
            f"max active: {max_active_opportunities}"
        )
    
    def _should_check_symbol(self, symbol: str, exchanges: List[str]) -> bool:
        """
        Проверяет, нужно ли анализировать данный символ на указанных биржах.
        
        Args:
            symbol: Символ (торговая пара)
            exchanges: Список бирж
            
        Returns:
            True если символ и биржи подходят для анализа
        """
        # Проверка символа
        if symbol in self.excluded_symbols:
            return False
        
        if self.target_symbols and symbol not in self.target_symbols:
            return False
        
        # Проверка бирж
        valid_exchanges = [e for e in exchanges if e not in self.excluded_exchanges]
        if self.target_exchanges:
            valid_exchanges = [e for e in valid_exchanges if e in self.target_exchanges]
        
        # Нужно минимум две биржи для арбитража
        if len(valid_exchanges) < 2:
            return False
        
        # Проверка интервала между проверками
        current_time_ms = int(time.time() * 1000)
        last_check_ms = self._last_check_timestamps.get(symbol, 0)
        
        if current_time_ms - last_check_ms < self.min_check_interval_ms:
            return False
        
        # Обновляем время последней проверки
        self._last_check_timestamps[symbol] = current_time_ms
        
        return True
    
    def _get_opportunity_key(self, opportunity: ArbitrageOpportunity) -> str:
        """
        Создает уникальный ключ для арбитражной возможности.
        
        Args:
            opportunity: Арбитражная возможность
            
        Returns:
            Уникальный ключ для возможности
        """
        return f"{opportunity.symbol}_{opportunity.buy_exchange}_{opportunity.sell_exchange}"
    
    def find_opportunities(
        self,
        market_data: NormalizedMarketData
    ) -> List[ArbitrageOpportunity]:
        """
        Находит арбитражные возможности на основе текущих рыночных данных.
        
        Args:
            market_data: Нормализованные рыночные данные
            
        Returns:
            Список найденных арбитражных возможностей
        """
        if not self._should_check_symbol(market_data.symbol, market_data.exchanges):
            return []
        
        # Используем калькулятор прибыли для анализа
        opportunities = self.profit_calculator.analyze_all_opportunities(market_data)
        
        # Фильтруем только жизнеспособные возможности
        viable_opportunities = [opp for opp in opportunities if opp.is_viable]
        
        # Обновляем активные возможности
        for opportunity in viable_opportunities:
            key = self._get_opportunity_key(opportunity)
            self.active_opportunities[key] = opportunity
        
        # Удаляем старые возможности, если превышен лимит
        if len(self.active_opportunities) > self.max_active_opportunities:
            # Сортируем по прибыли и оставляем только max_active_opportunities лучших
            sorted_opportunities = sorted(
                self.active_opportunities.items(),
                key=lambda x: x[1].net_profit_percentage,
                reverse=True
            )
            
            self.active_opportunities = dict(sorted_opportunities[:self.max_active_opportunities])
        
        return viable_opportunities
    
    def get_all_active_opportunities(self) -> List[ArbitrageOpportunity]:
        """
        Возвращает все активные арбитражные возможности.
        
        Returns:
            Список активных арбитражных возможностей
        """
        return list(self.active_opportunities.values())
    
    def get_best_opportunities(self, limit: int = 5) -> List[ArbitrageOpportunity]:
        """
        Возвращает лучшие арбитражные возможности.
        
        Args:
            limit: Максимальное количество возвращаемых возможностей
            
        Returns:
            Список лучших арбитражных возможностей
        """
        sorted_opportunities = sorted(
            self.active_opportunities.values(),
            key=lambda x: x.net_profit_percentage,
            reverse=True
        )
        
        return sorted_opportunities[:limit]
    
    def clear_expired_opportunities(self, max_age_ms: int = 60000) -> int:
        """
        Удаляет устаревшие арбитражные возможности.
        
        Args:
            max_age_ms: Максимальный возраст возможности в миллисекундах
            
        Returns:
            Количество удаленных возможностей
        """
        current_time_ms = int(time.time() * 1000)
        expired_keys = []
        
        for key, opportunity in self.active_opportunities.items():
            opportunity_age_ms = current_time_ms - opportunity.timestamp
            if opportunity_age_ms > max_age_ms:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.active_opportunities[key]
        
        return len(expired_keys)
    
    def process_market_data_batch(
        self,
        market_data_batch: List[NormalizedMarketData],
        max_workers: int = 4
    ) -> List[ArbitrageOpportunity]:
        """
        Обрабатывает пакет рыночных данных параллельно.
        
        Args:
            market_data_batch: Список нормализованных рыночных данных
            max_workers: Максимальное количество параллельных потоков
            
        Returns:
            Список найденных арбитражных возможностей
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_data = {
                executor.submit(self.find_opportunities, data): data
                for data in market_data_batch
            }
            
            for future in as_completed(future_to_data):
                try:
                    opportunities = future.result()
                    results.extend(opportunities)
                except Exception as e:
                    data = future_to_data[future]
                    self.logger.error(
                        f"Error processing market data for {data.symbol}: {str(e)}"
                    )
        
        return results
