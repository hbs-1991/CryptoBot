"""
Модуль управления рисками для арбитражных операций.
Отвечает за оценку и фильтрацию торговых возможностей с учетом различных факторов риска.
"""

import logging
from decimal import Decimal
from typing import Dict, List, NamedTuple, Any, Optional, Tuple
from dataclasses import dataclass

from src.utils import get_logger
from src.arbitrage_engine.profit_calculator import ArbitrageOpportunity


@dataclass
class RiskAssessment:
    """
    Результат оценки риска для арбитражной возможности.
    
    Attributes:
        is_acceptable: Можно ли принять этот риск
        risk_score: Количественная оценка риска (0-100, где 0 - минимальный риск)
        reasons: Список причин, по которым возможность может быть рискованной
        max_safe_amount: Максимальная безопасная сумма для сделки
        adjusted_volume: Рекомендуемый объем с учетом оценки риска
    """
    is_acceptable: bool = False
    risk_score: float = 100.0
    reasons: List[str] = None
    max_safe_amount: float = 0.0
    adjusted_volume: float = 0.0
    
    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


class RiskManager:
    """
    Управляет рисками арбитражных операций.
    Оценивает возможности и принимает решение о их допустимости с точки зрения риска.
    """
    
    def __init__(
        self,
        max_order_amount: float = 1000.0,
        max_price_deviation: float = 3.0,
        min_liquidity_requirement: float = 3.0,
        max_trades_per_pair: int = 3,
        slippage_tolerance: float = 1.0,
        min_volume_requirement: float = 100.0,
        max_risk_score: float = 70.0,
        emergency_stop_loss_percentage: float = 5.0,
        enable_strict_mode: bool = False
    ):
        """
        Инициализирует менеджер рисков с заданными параметрами.
        
        Args:
            max_order_amount: Максимальная сумма одного ордера в USD
            max_price_deviation: Максимальное допустимое отклонение цены в %
            min_liquidity_requirement: Минимальное отношение объема к сумме сделки
            max_trades_per_pair: Максимальное количество одновременных сделок для пары
            slippage_tolerance: Допустимое проскальзывание цены в %
            min_volume_requirement: Минимальный объем сделки в USD для рассмотрения
            max_risk_score: Максимально допустимый риск-скор (0-100)
            emergency_stop_loss_percentage: Процент потери, при котором активируется экстренное закрытие
            enable_strict_mode: Включение строгого режима проверки рисков
        """
        self.logger = get_logger(__name__)
        
        self.max_order_amount = max_order_amount
        self.max_price_deviation = max_price_deviation
        self.min_liquidity_requirement = min_liquidity_requirement
        self.max_trades_per_pair = max_trades_per_pair
        self.slippage_tolerance = slippage_tolerance
        self.min_volume_requirement = min_volume_requirement
        self.max_risk_score = max_risk_score
        self.emergency_stop_loss_percentage = emergency_stop_loss_percentage
        self.enable_strict_mode = enable_strict_mode
        
        # Учет активных сделок по парам
        self.active_trades_count: Dict[str, int] = {}
        
        # Хранение истории аномальных цен для обнаружения краткосрочных аномалий
        self.price_anomalies_history: Dict[str, List[Tuple[float, float]]] = {}
        
        self.logger.info(
            f"Инициализирован RiskManager с параметрами: "
            f"max_order={max_order_amount}USD, "
            f"max_deviation={max_price_deviation}%, "
            f"min_liquidity={min_liquidity_requirement}x, "
            f"strict_mode={enable_strict_mode}"
        )

    # noinspection PyTypeChecker
    def assess_opportunity(self, opportunity: ArbitrageOpportunity, market_data: Dict[str, Any]) -> RiskAssessment:
        """
        Оценивает арбитражную возможность с точки зрения рисков.
        
        Args:
            opportunity: Объект арбитражной возможности для оценки
            market_data: Рыночные данные для контекста оценки
            
        Returns:
            Результат оценки риска
        """
        self.logger.debug(f"Оценка риска для {opportunity.symbol}: {opportunity.buy_exchange}->{opportunity.sell_exchange}")
        
        # Инициализация результата оценки
        assessment = RiskAssessment()
        assessment.reasons = []
        
        # Сброс риск-скора (0 - нет риска, 100 - максимальный риск)
        risk_score = 0
        
        # 1. Проверка размера сделки
        if opportunity.volume * opportunity.buy_price > self.max_order_amount:
            adjusted_volume: Decimal = self.max_order_amount / opportunity.buy_price
            assessment.reasons.append(f"Объем сделки {opportunity.volume} превышает лимит {self.max_order_amount / opportunity.buy_price}")
            assessment.adjusted_volume = adjusted_volume
            risk_score += 15
        else:
            assessment.adjusted_volume = opportunity.volume
        
        # 2. Проверка отклонения цены от рыночной
        if market_data and "average_market_price" in market_data:
            avg_price = market_data["average_market_price"].get(opportunity.symbol, 0)
            if avg_price > 0:
                buy_deviation = abs(opportunity.buy_price - avg_price) / avg_price * 100
                sell_deviation = abs(opportunity.sell_price - avg_price) / avg_price * 100
                
                max_deviation = max(buy_deviation, sell_deviation)
                if max_deviation > self.max_price_deviation:
                    assessment.reasons.append(f"Отклонение цены {max_deviation:.2f}% превышает допустимое {self.max_price_deviation}%")
                    risk_score += min(30, max_deviation * 3)  # Чем больше отклонение, тем выше риск
        
        # 3. Проверка ликвидности
        if market_data and "order_book" in market_data:
            # Получаем данные книги ордеров для покупки
            buy_order_book = market_data["order_book"].get(
                (opportunity.symbol, opportunity.buy_exchange), {}
            )
            
            # Получаем данные книги ордеров для продажи
            sell_order_book = market_data["order_book"].get(
                (opportunity.symbol, opportunity.sell_exchange), {}
            )
            
            # Проверяем объем на бирже покупки (проверяем sell orders, т.к. мы будем покупать)
            buy_liquidity = self._calculate_available_volume(
                buy_order_book.get("asks", []), opportunity.buy_price * (1 + self.slippage_tolerance / 100)
            )
            
            # Проверяем объем на бирже продажи (проверяем buy orders, т.к. мы будем продавать)
            sell_liquidity = self._calculate_available_volume(
                sell_order_book.get("bids", []), opportunity.sell_price * (1 - self.slippage_tolerance / 100)
            )
            
            # Определяем минимальную ликвидность из двух бирж
            min_liquidity = min(buy_liquidity, sell_liquidity)
            
            # Проверяем, достаточно ли ликвидности для нашей сделки
            if min_liquidity < opportunity.volume:
                assessment.reasons.append(f"Недостаточная ликвидность: доступно {min_liquidity}, требуется {opportunity.volume}")
                risk_score += 25
                
                # Корректируем объем сделки до доступной ликвидности
                if min_liquidity > 0:
                    assessment.adjusted_volume = min(assessment.adjusted_volume, min_liquidity * 0.9)  # Берем 90% доступной ликвидности
            
            # Проверяем соотношение нашей сделки к объему рынка
            liquidity_ratio = min_liquidity / opportunity.volume if opportunity.volume > 0 else 0
            
            if liquidity_ratio < self.min_liquidity_requirement:
                assessment.reasons.append(f"Недостаточное соотношение ликвидности: {liquidity_ratio:.2f}x (требуется {self.min_liquidity_requirement}x)")
                risk_score += 20
        
        # 4. Проверка количества активных сделок для данной пары
        active_trades = self.active_trades_count.get(opportunity.symbol, 0)
        if active_trades >= self.max_trades_per_pair:
            assessment.reasons.append(f"Достигнут лимит активных сделок для пары {opportunity.symbol}: {active_trades}")
            risk_score += 15
        
        # 5. Проверка минимального объема сделки
        trade_volume_usd = opportunity.volume * opportunity.buy_price
        if trade_volume_usd < self.min_volume_requirement:
            assessment.reasons.append(f"Объем сделки {trade_volume_usd:.2f} USD меньше минимального {self.min_volume_requirement} USD")
            risk_score += 10
        
        # 6. Проверка истории аномальных цен
        if opportunity.symbol in self.price_anomalies_history and len(self.price_anomalies_history[opportunity.symbol]) > 0:
            anomalies_count = len(self.price_anomalies_history[opportunity.symbol])
            if anomalies_count > 3:  # Если было более 3 аномалий
                assessment.reasons.append(f"Обнаружено {anomalies_count} ценовых аномалий для {opportunity.symbol}")
                risk_score += min(20, anomalies_count * 4)  # Ограничиваем вклад в риск-скор
        
        # Финализация оценки
        assessment.risk_score = min(100, risk_score)  # Ограничиваем максимальным значением
        assessment.is_acceptable = assessment.risk_score <= self.max_risk_score
        
        # Устанавливаем максимальную безопасную сумму
        assessment.max_safe_amount = assessment.adjusted_volume
        
        # Логируем результат оценки
        risk_level = "ВЫСОКИЙ" if assessment.risk_score > 70 else "СРЕДНИЙ" if assessment.risk_score > 40 else "НИЗКИЙ"
        self.logger.debug(
            f"Оценка риска: {risk_level} ({assessment.risk_score:.1f}/100), "
            f"приемлемо: {assessment.is_acceptable}, "
            f"макс. объем: {assessment.max_safe_amount}"
        )
        if assessment.reasons:
            self.logger.debug(f"Причины риска: {', '.join(assessment.reasons)}")
        
        return assessment

    @staticmethod
    def _calculate_available_volume(orders: List[List[Decimal]], price_limit: Decimal) -> float:
        """
        Рассчитывает доступный объем в книге ордеров до заданного ценового лимита.

        Args:
            orders: Список ордеров в формате [цена, объем]
            price_limit: Предельная цена для расчета объема

        Returns:
            Доступный объем
            :rtype: object
        """
        available_volume = 0.0

        for order in orders:
            order_price, order_volume = order

            # Для покупки (проверка asks): если цена ордера ниже лимита
            # Для продажи (проверка bids): если цена ордера выше лимита
            if (len(orders) > 0 and orders[0][0] > orders[-1][0] and order_price < price_limit) or \
               (len(orders) > 0 and orders[0][0] < orders[-1][0] and order_price > price_limit):
                available_volume += order_volume

        return available_volume

    def register_active_trade(self, symbol: str) -> None:
        """
        Регистрирует новую активную сделку для указанной торговой пары.
        
        Args:
            symbol: Символ торговой пары
        """
        self.active_trades_count[symbol] = self.active_trades_count.get(symbol, 0) + 1
        self.logger.debug(f"Зарегистрирована новая сделка для {symbol}. Всего активных: {self.active_trades_count[symbol]}")
    
    def unregister_active_trade(self, symbol: str) -> None:
        """
        Удаляет активную сделку для указанной торговой пары.
        
        Args:
            symbol: Символ торговой пары
        """
        if symbol in self.active_trades_count and self.active_trades_count[symbol] > 0:
            self.active_trades_count[symbol] -= 1
            self.logger.debug(f"Завершена сделка для {symbol}. Всего активных: {self.active_trades_count[symbol]}")
    
    def register_price_anomaly(self, symbol: str, exchange: str, anomaly_percentage: float) -> None:
        """
        Регистрирует аномальное отклонение цены для пары на бирже.
        
        Args:
            symbol: Символ торговой пары
            exchange: Название биржи
            anomaly_percentage: Процент отклонения от нормальной цены
        """
        if symbol not in self.price_anomalies_history:
            self.price_anomalies_history[symbol] = []
        
        # Добавляем запись об аномалии
        import time
        self.price_anomalies_history[symbol].append((time.time(), anomaly_percentage))
        
        # Удаляем устаревшие записи (старше 1 часа)
        current_time = time.time()
        self.price_anomalies_history[symbol] = [
            (t, p) for t, p in self.price_anomalies_history[symbol]
            if current_time - t < 3600  # 1 час в секундах
        ]
        
        self.logger.warning(f"Зарегистрирована ценовая аномалия для {symbol} на {exchange}: {anomaly_percentage:.2f}%")
    
    def check_emergency_stop_loss(self, current_loss_percentage: float) -> bool:
        """
        Проверяет, не превышена ли граница экстренной остановки из-за потерь.
        
        Args:
            current_loss_percentage: Текущий процент потерь
            
        Returns:
            True, если превышена граница и нужно остановить торговлю
        """
        if current_loss_percentage >= self.emergency_stop_loss_percentage:
            self.logger.critical(
                f"ВНИМАНИЕ! Достигнут порог экстренной остановки: "
                f"текущие потери {current_loss_percentage:.2f}% превышают лимит {self.emergency_stop_loss_percentage}%"
            )
            return True
        return False
    
    def adjust_parameters_based_on_market(self, market_volatility: float) -> None:
        """
        Динамически корректирует параметры риска на основе рыночной волатильности.
        
        Args:
            market_volatility: Показатель волатильности рынка (0-100)
        """
        # Пример динамической корректировки на основе волатильности
        if market_volatility > 80:  # Высокая волатильность
            new_max_order = self.max_order_amount * 0.5  # Сокращаем размер ордера вдвое
            new_max_deviation = self.max_price_deviation * 0.7  # Уменьшаем допустимое отклонение
            
            self.logger.warning(
                f"Высокая волатильность рынка ({market_volatility:.1f}%). "
                f"Корректировка параметров риска: "
                f"max_order_amount {self.max_order_amount} -> {new_max_order}, "
                f"max_price_deviation {self.max_price_deviation}% -> {new_max_deviation}%"
            )
            
            self.max_order_amount = new_max_order
            self.max_price_deviation = new_max_deviation
            
        elif market_volatility < 20:  # Низкая волатильность
            new_max_order = min(self.max_order_amount * 1.2, self.max_order_amount * 2)  # Увеличиваем до 20%, но не более чем в 2 раза
            new_max_deviation = min(self.max_price_deviation * 1.1, self.max_price_deviation * 1.5)  # Увеличиваем до 10%
            
            self.logger.info(
                f"Низкая волатильность рынка ({market_volatility:.1f}%). "
                f"Корректировка параметров риска: "
                f"max_order_amount {self.max_order_amount} -> {new_max_order}, "
                f"max_price_deviation {self.max_price_deviation}% -> {new_max_deviation}%"
            )
            
            self.max_order_amount = new_max_order
            self.max_price_deviation = new_max_deviation
    
    def get_risk_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику по рискам для аналитики.
        
        Returns:
            Словарь со статистикой по рискам
        """
        # Подсчет общего количества активных сделок
        total_active_trades = sum(self.active_trades_count.values())
        
        # Подсчет количества аномалий за последний час
        total_anomalies = sum(len(anomalies) for anomalies in self.price_anomalies_history.values())
        
        return {
            "total_active_trades": total_active_trades,
            "trades_by_symbol": self.active_trades_count.copy(),
            "price_anomalies_count": total_anomalies,
            "risk_parameters": {
                "max_order_amount": self.max_order_amount,
                "max_price_deviation": self.max_price_deviation,
                "min_liquidity_requirement": self.min_liquidity_requirement,
                "max_trades_per_pair": self.max_trades_per_pair,
                "slippage_tolerance": self.slippage_tolerance,
                "strict_mode": self.enable_strict_mode
            }
        }
