"""
Модуль, реализующий стратегии для проведения арбитражных сделок
на различных рынках с учетом их специфики.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
import time
from abc import ABC, abstractmethod

from src.arbitrage_engine.profit_calculator import ArbitrageOpportunity
from src.data_fetching.data_normalizer import NormalizedMarketData


class MarketStrategy(ABC):
    """
    Абстрактный базовый класс для реализации различных стратегий
    выполнения арбитражных сделок.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Инициализация базовой стратегии.
        
        Args:
            name: Название стратегии
            description: Описание стратегии
        """
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    @abstractmethod
    def evaluate(self, opportunity: ArbitrageOpportunity, market_data: NormalizedMarketData) -> Tuple[bool, Optional[str]]:
        """
        Оценивает возможность применения стратегии к данной арбитражной возможности.
        
        Args:
            opportunity: Арбитражная возможность
            market_data: Нормализованные рыночные данные
            
        Returns:
            Кортеж из булевого значения (можно ли применить стратегию)
            и опционального сообщения с причиной отказа
        """
        pass
    
    @abstractmethod
    def calculate_trade_params(
        self, 
        opportunity: ArbitrageOpportunity, 
        market_data: NormalizedMarketData
    ) -> Dict[str, Any]:
        """
        Рассчитывает параметры сделки на основе стратегии.
        
        Args:
            opportunity: Арбитражная возможность
            market_data: Нормализованные рыночные данные
            
        Returns:
            Словарь с параметрами сделки
        """
        pass
        
    @abstractmethod
    def should_close_position(
        self, 
        opportunity: ArbitrageOpportunity,
        current_market_data: NormalizedMarketData,
        position_data: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Определяет, нужно ли закрывать позицию.
        
        Args:
            opportunity: Исходная арбитражная возможность
            current_market_data: Текущие рыночные данные
            position_data: Данные об открытой позиции
            
        Returns:
            Кортеж из булевого значения (нужно ли закрывать позицию)
            и опционального сообщения с причиной
        """
        pass


class DirectArbitrageStrategy(MarketStrategy):
    """
    Стратегия прямого арбитража, которая покупает на одной бирже
    и продает на другой, когда разница в ценах превышает определенный порог.
    """
    
    def __init__(
        self,
        min_profit_percentage: float = 0.5,
        max_order_value: float = 1000.0,
        min_order_value: float = 10.0,
        max_execution_time_ms: int = 5000,
        emergency_close_threshold: float = -1.0
    ):
        """
        Инициализация стратегии прямого арбитража.
        
        Args:
            min_profit_percentage: Минимальный процент прибыли для выполнения сделки
            max_order_value: Максимальная стоимость ордера в базовой валюте
            min_order_value: Минимальная стоимость ордера в базовой валюте
            max_execution_time_ms: Максимальное время выполнения сделки в мс
            emergency_close_threshold: Порог убытка для экстренного закрытия позиции
        """
        super().__init__(
            name="DirectArbitrageStrategy",
            description="Стратегия прямого арбитража с покупкой на одной бирже и продажей на другой"
        )
        self.min_profit_percentage = min_profit_percentage
        self.max_order_value = max_order_value
        self.min_order_value = min_order_value
        self.max_execution_time_ms = max_execution_time_ms
        self.emergency_close_threshold = emergency_close_threshold
        
        self.logger.info(
            f"DirectArbitrageStrategy initialized with min_profit={min_profit_percentage}%, "
            f"order_value_range={min_order_value}-{max_order_value}, "
            f"max_execution_time={max_execution_time_ms}ms"
        )
    
    def evaluate(
        self, 
        opportunity: ArbitrageOpportunity, 
        market_data: NormalizedMarketData
    ) -> Tuple[bool, Optional[str]]:
        """
        Оценивает возможность применения стратегии к данной арбитражной возможности.
        
        Args:
            opportunity: Арбитражная возможность
            market_data: Нормализованные рыночные данные
            
        Returns:
            Кортеж из булевого значения и причины отказа (если есть)
        """
        # Проверяем, что возможность жизнеспособна
        if not opportunity.is_viable:
            return False, f"Возможность нежизнеспособна: {opportunity.viability_reason}"
        
        # Проверяем, что прибыль достаточна
        if opportunity.net_profit_percentage < self.min_profit_percentage:
            return False, f"Недостаточная прибыль: {opportunity.net_profit_percentage}% < {self.min_profit_percentage}%"
        
        # Проверяем объем сделки
        max_trade_value = opportunity.buy_price * opportunity.volume
        if max_trade_value < self.min_order_value:
            return False, f"Слишком малый объем: {max_trade_value} < {self.min_order_value}"
        
        return True, None
    
    def calculate_trade_params(
        self, 
        opportunity: ArbitrageOpportunity, 
        market_data: NormalizedMarketData
    ) -> Dict[str, Any]:
        """
        Рассчитывает параметры сделки на основе стратегии.
        
        Args:
            opportunity: Арбитражная возможность
            market_data: Нормализованные рыночные данные
            
        Returns:
            Словарь с параметрами сделки
        """
        # Определяем объем сделки
        max_trade_value = opportunity.buy_price * opportunity.volume
        volume = opportunity.volume
        
        # Ограничиваем объем сделки максимальным значением
        if max_trade_value > self.max_order_value:
            volume = self.max_order_value / opportunity.buy_price
        
        return {
            "symbol": opportunity.symbol,
            "buy_exchange": opportunity.buy_exchange,
            "sell_exchange": opportunity.sell_exchange,
            "buy_price": float(opportunity.buy_price),
            "sell_price": float(opportunity.sell_price),
            "volume": float(volume),
            "expected_profit_percentage": float(opportunity.net_profit_percentage),
            "expected_profit_amount": float(volume * opportunity.sell_price * opportunity.net_profit_percentage / 100),
            "timestamp": int(time.time() * 1000),
            "max_execution_time_ms": self.max_execution_time_ms
        }
    
    def should_close_position(
        self, 
        opportunity: ArbitrageOpportunity,
        current_market_data: NormalizedMarketData,
        position_data: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Определяет, нужно ли закрывать позицию.
        
        Args:
            opportunity: Исходная арбитражная возможность
            current_market_data: Текущие рыночные данные
            position_data: Данные об открытой позиции
            
        Returns:
            Кортеж из булевого значения и причины
        """
        # Получаем текущие цены
        buy_exchange = position_data["buy_exchange"]
        sell_exchange = position_data["sell_exchange"]
        symbol = position_data["symbol"]
        
        # Проверяем, что данные биржи присутствуют в текущих данных
        if buy_exchange not in current_market_data.exchanges or sell_exchange not in current_market_data.exchanges:
            return False, "Недостаточно данных для принятия решения о закрытии позиции"
        
        # Получаем текущие цены покупки и продажи
        current_buy_price = current_market_data.get_price(buy_exchange, symbol, "ask")
        current_sell_price = current_market_data.get_price(sell_exchange, symbol, "bid")
        
        if current_buy_price is None or current_sell_price is None:
            return False, "Невозможно получить текущие цены"
        
        # Рассчитываем текущую прибыль
        original_buy_price = position_data["buy_price"]
        original_sell_price = position_data["sell_price"]
        volume = position_data["volume"]
        
        # Если цена продажи упала слишком сильно
        current_profit_percentage = (current_sell_price - original_buy_price) / original_buy_price * 100
        
        # Проверяем экстренное закрытие позиции при убытке
        if current_profit_percentage <= self.emergency_close_threshold:
            return True, f"Экстренное закрытие: текущая прибыль {current_profit_percentage}% <= {self.emergency_close_threshold}%"
        
        # Проверяем истечение времени
        start_time = position_data["timestamp"]
        current_time = int(time.time() * 1000)
        elapsed_time = current_time - start_time
        
        if elapsed_time >= self.max_execution_time_ms:
            return True, f"Истекло максимальное время выполнения сделки ({elapsed_time}ms >= {self.max_execution_time_ms}ms)"
        
        return False, None


class TriangularArbitrageStrategy(MarketStrategy):
    """
    Стратегия треугольного арбитража, которая использует три торговые пары
    на одной бирже для получения прибыли.
    """
    
    def __init__(
        self,
        min_profit_percentage: float = 0.3,
        max_order_value: float = 500.0,
        min_order_value: float = 10.0,
        max_execution_time_ms: int = 3000
    ):
        """
        Инициализация стратегии треугольного арбитража.
        
        Args:
            min_profit_percentage: Минимальный процент прибыли
            max_order_value: Максимальная стоимость ордера
            min_order_value: Минимальная стоимость ордера
            max_execution_time_ms: Максимальное время выполнения
        """
        super().__init__(
            name="TriangularArbitrageStrategy",
            description="Стратегия треугольного арбитража с использованием трех торговых пар на одной бирже"
        )
        self.min_profit_percentage = min_profit_percentage
        self.max_order_value = max_order_value
        self.min_order_value = min_order_value
        self.max_execution_time_ms = max_execution_time_ms
        
        self.logger.info(
            f"TriangularArbitrageStrategy initialized with min_profit={min_profit_percentage}%, "
            f"order_value_range={min_order_value}-{max_order_value}"
        )
    
    def evaluate(
        self, 
        opportunity: ArbitrageOpportunity, 
        market_data: NormalizedMarketData
    ) -> Tuple[bool, Optional[str]]:
        """
        Оценивает возможность применения стратегии треугольного арбитража.
        
        Args:
            opportunity: Арбитражная возможность
            market_data: Нормализованные рыночные данные
            
        Returns:
            Кортеж из булевого значения и причины отказа (если есть)
        """
        # Для треугольного арбитража покупка и продажа должны быть на одной бирже
        if opportunity.buy_exchange != opportunity.sell_exchange:
            return False, "Треугольный арбитраж требует одну биржу для всех операций"
        
        # Проверяем минимальную прибыль
        if opportunity.net_profit_percentage < self.min_profit_percentage:
            return False, f"Недостаточная прибыль: {opportunity.net_profit_percentage}% < {self.min_profit_percentage}%"
        
        return True, None
    
    def calculate_trade_params(
        self, 
        opportunity: ArbitrageOpportunity, 
        market_data: NormalizedMarketData
    ) -> Dict[str, Any]:
        """
        Рассчитывает параметры сделки треугольного арбитража.
        
        Args:
            opportunity: Арбитражная возможность
            market_data: Нормализованные рыночные данные
            
        Returns:
            Словарь с параметрами сделки
        """
        # Для треугольного арбитража нам нужны три торговые пары
        # Например: BTC/USDT -> ETH/BTC -> ETH/USDT
        
        # В простой реализации мы опишем только основные параметры
        # В реальном случае здесь нужно вычислять промежуточные пары и их объемы
        
        volume = min(opportunity.volume, self.max_order_value / opportunity.buy_price)
        
        return {
            "strategy_type": "triangular",
            "exchange": opportunity.buy_exchange,
            "base_symbol": opportunity.symbol.split("/")[0],  # Например, BTC из BTC/USDT
            "quote_symbol": opportunity.symbol.split("/")[1],  # Например, USDT из BTC/USDT
            "volume": float(volume),
            "expected_profit_percentage": float(opportunity.net_profit_percentage),
            "timestamp": int(time.time() * 1000),
            "max_execution_time_ms": self.max_execution_time_ms,
            "trading_path": [
                # Здесь должен быть путь торговых пар, например:
                # ["BTC/USDT", "ETH/BTC", "ETH/USDT"]
                opportunity.symbol
            ]
        }
    
    def should_close_position(
        self, 
        opportunity: ArbitrageOpportunity,
        current_market_data: NormalizedMarketData,
        position_data: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Определяет, нужно ли закрывать позицию треугольного арбитража.
        
        Args:
            opportunity: Исходная арбитражная возможность
            current_market_data: Текущие рыночные данные
            position_data: Данные об открытой позиции
            
        Returns:
            Кортеж из булевого значения и причины
        """
        # Для треугольного арбитража обычно все сделки выполняются сразу,
        # но можно проверить, не прошло ли максимальное время выполнения
        
        start_time = position_data["timestamp"]
        current_time = int(time.time() * 1000)
        elapsed_time = current_time - start_time
        
        if elapsed_time >= self.max_execution_time_ms:
            return True, f"Истекло максимальное время выполнения сделки ({elapsed_time}ms >= {self.max_execution_time_ms}ms)"
        
        return False, None


class StrategySelector:
    """
    Выбирает наиболее подходящую стратегию для данной арбитражной возможности.
    """
    
    def __init__(self, strategies: List[MarketStrategy]):
        """
        Инициализация селектора стратегий.
        
        Args:
            strategies: Список доступных стратегий
        """
        self.strategies = strategies
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"StrategySelector initialized with {len(strategies)} strategies")
    
    def select_strategy(
        self, 
        opportunity: ArbitrageOpportunity, 
        market_data: NormalizedMarketData
    ) -> Tuple[Optional[MarketStrategy], Dict[str, Any]]:
        """
        Выбирает наиболее подходящую стратегию для данной возможности.
        
        Args:
            opportunity: Арбитражная возможность
            market_data: Нормализованные рыночные данные
            
        Returns:
            Кортеж из выбранной стратегии и параметров сделки
        """
        best_strategy = None
        best_trade_params = None
        highest_profit = -1
        
        for strategy in self.strategies:
            can_apply, reason = strategy.evaluate(opportunity, market_data)
            
            if can_apply:
                trade_params = strategy.calculate_trade_params(opportunity, market_data)
                profit = trade_params.get("expected_profit_amount", 0)
                
                if profit > highest_profit:
                    highest_profit = profit
                    best_strategy = strategy
                    best_trade_params = trade_params
            else:
                self.logger.debug(f"Strategy {strategy.name} rejected: {reason}")
        
        if best_strategy:
            self.logger.info(
                f"Selected strategy {best_strategy.name} for {opportunity.symbol} "
                f"with expected profit {best_trade_params.get('expected_profit_percentage')}%"
            )
        else:
            self.logger.debug(f"No suitable strategy found for {opportunity.symbol}")
        
        return best_strategy, best_trade_params if best_strategy else {}
