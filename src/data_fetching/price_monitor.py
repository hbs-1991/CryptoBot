"""
Монитор цен на криптовалютных биржах.
Отслеживает изменения цен и сигнализирует о значительных изменениях.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from decimal import Decimal

from src.data_fetching.data_normalizer import NormalizedMarketData


class PriceMonitor:
    """
    Монитор цен на криптовалютных биржах.
    Отслеживает изменения цен и сигнализирует о значительных изменениях.
    """
    
    def __init__(self, significant_change_threshold: float = 0.5):
        """
        Инициализация монитора цен.
        
        Args:
            significant_change_threshold: Порог значимого изменения цены в процентах
        """
        self.logger = logging.getLogger(__name__)
        self.significant_change_threshold = significant_change_threshold
        self.last_prices = {}
        self.logger.info(f"PriceMonitor initialized with threshold: {significant_change_threshold}%")
    
    def process_data_batch(self, data_batch: List[NormalizedMarketData]) -> Dict[str, Dict[str, float]]:
        """
        Обрабатывает пакет данных рынка и выявляет значительные изменения цен.
        
        Args:
            data_batch: Список объектов с нормализованными данными рынка
            
        Returns:
            Словарь со значительными изменениями цен в формате {symbol: {exchange: percentage_change}}
        """
        significant_changes = {}
        
        for market_data in data_batch:
            symbol = market_data.symbol
            
            # Если этот символ еще не отслеживается, добавляем его
            if symbol not in self.last_prices:
                self.last_prices[symbol] = {}
            
            # Обрабатываем тикеры для всех бирж
            for exchange, ticker in market_data.tickers.items():
                # Если для этой биржи еще нет данных, инициализируем их
                if exchange not in self.last_prices[symbol]:
                    self.last_prices[symbol][exchange] = float(ticker.last)
                    continue
                
                # Получаем предыдущую цену
                prev_price = self.last_prices[symbol][exchange]
                current_price = float(ticker.last)
                
                # Вычисляем процентное изменение
                if prev_price > 0:
                    percentage_change = (current_price - prev_price) / prev_price * 100.0
                else:
                    percentage_change = 0.0
                
                # Если изменение превышает порог, добавляем его в результаты
                if abs(percentage_change) >= self.significant_change_threshold:
                    if symbol not in significant_changes:
                        significant_changes[symbol] = {}
                    
                    significant_changes[symbol][exchange] = percentage_change
                    
                    self.logger.debug(
                        f"Significant price change for {symbol} on {exchange}: "
                        f"{prev_price:.2f} -> {current_price:.2f} ({percentage_change:.2f}%)"
                    )
                
                # Обновляем последнюю известную цену
                self.last_prices[symbol][exchange] = current_price
        
        return significant_changes