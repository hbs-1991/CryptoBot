"""
Калькулятор прибыли для арбитражных сделок.
Отвечает за расчет потенциальной и реальной прибыли от арбитражных возможностей
с учетом комиссий, проскальзывания и других факторов.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal
from pydantic import BaseModel, Field

from src.data_fetching.data_normalizer import NormalizedMarketData, NormalizedOrderBook


class ArbitrageOpportunity(BaseModel):
    """Модель арбитражной возможности с расчетом прибыли."""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: Decimal
    sell_price: Decimal
    profit_percentage: Decimal
    direction: str
    max_volume: Decimal = Decimal('0')
    buy_commission: Decimal = Decimal('0')
    sell_commission: Decimal = Decimal('0')
    net_profit_percentage: Decimal = Decimal('0')
    estimated_profit: Decimal = Decimal('0')
    timestamp: int
    is_viable: bool = True
    viability_reason: Optional[str] = None


class ProfitCalculator:
    """
    Калькулятор прибыли для арбитражных сделок.
    Рассчитывает потенциальную прибыль с учетом комиссий, ликвидности 
    и других факторов, влияющих на фактическую прибыльность сделки.
    """
    
    def __init__(
        self,
        exchange_fees: Dict[str, float],
        min_profit_percentage: float = 0.5,
        slippage_factor: float = 0.05,
        min_order_volume: float = 10.0,
        max_order_volume: float = 1000.0
    ):
        """
        Инициализация калькулятора прибыли.
        
        Args:
            exchange_fees: Словарь с комиссиями для каждой биржи в процентах
            min_profit_percentage: Минимальный процент прибыли для признания сделки выгодной
            slippage_factor: Фактор проскальзывания (0-1), учитывающий влияние на цену при исполнении ордера
            min_order_volume: Минимальный объем ордера в базовой валюте
            max_order_volume: Максимальный объем ордера в базовой валюте
        """
        self.logger = logging.getLogger(__name__)
        
        # Преобразуем комиссии в Decimal для точных расчетов
        self.exchange_fees = {
            exchange: Decimal(str(fee_percent)) / Decimal('100')
            for exchange, fee_percent in exchange_fees.items()
        }
        
        self.min_profit_percentage = Decimal(str(min_profit_percentage))
        self.slippage_factor = Decimal(str(slippage_factor))
        self.min_order_volume = Decimal(str(min_order_volume))
        self.max_order_volume = Decimal(str(max_order_volume))
        
        self.logger.info(
            f"ProfitCalculator initialized with min profit: {min_profit_percentage}%, "
            f"slippage: {slippage_factor}, volume range: {min_order_volume}-{max_order_volume}"
        )
    
    def calculate_opportunity_profit(
        self, 
        market_data: NormalizedMarketData,
        buy_exchange: str,
        sell_exchange: str
    ) -> Optional[ArbitrageOpportunity]:
        """
        Рассчитывает потенциальную прибыль от арбитражной возможности.
        
        Args:
            market_data: Нормализованные рыночные данные
            buy_exchange: Биржа для покупки
            sell_exchange: Биржа для продажи
            
        Returns:
            Арбитражная возможность с расчетом прибыли или None, если возможность невыгодна
        """
        # Проверяем наличие необходимых данных
        if buy_exchange not in market_data.tickers or sell_exchange not in market_data.tickers:
            return None
        
        if buy_exchange not in market_data.order_books or sell_exchange not in market_data.order_books:
            return None
        
        # Получаем данные тикеров и стаканов ордеров
        buy_ticker = market_data.tickers[buy_exchange]
        sell_ticker = market_data.tickers[sell_exchange]
        buy_order_book = market_data.order_books[buy_exchange]
        sell_order_book = market_data.order_books[sell_exchange]
        
        # Получаем лучшие цены
        buy_price = buy_ticker.ask
        sell_price = sell_ticker.bid
        
        # Если цены некорректны, возвращаем None
        if buy_price <= 0 or sell_price <= 0:
            return None
        
        # Грубая проверка на наличие арбитражной возможности
        if sell_price <= buy_price:
            return None
        
        # Рассчитываем комиссии
        buy_commission = self.exchange_fees.get(buy_exchange, Decimal('0.001'))
        sell_commission = self.exchange_fees.get(sell_exchange, Decimal('0.001'))
        
        # Рассчитываем процент прибыли
        profit_percentage = (sell_price - buy_price) / buy_price * Decimal('100')
        
        # Учитываем комиссии в проценте прибыли
        total_commission_percentage = (buy_commission + sell_commission) * Decimal('100')
        net_profit_percentage = profit_percentage - total_commission_percentage
        
        # Определяем максимальный объем сделки на основе глубины ликвидности
        max_volume = self._calculate_max_trade_volume(buy_order_book, sell_order_book)
        
        # Ограничиваем объем сделки заданными пределами
        max_volume = min(max_volume, self.max_order_volume)
        
        # Проверяем минимальный объем
        if max_volume < self.min_order_volume:
            return ArbitrageOpportunity(
                symbol=market_data.symbol,
                buy_exchange=buy_exchange,
                sell_exchange=sell_exchange,
                buy_price=buy_price,
                sell_price=sell_price,
                profit_percentage=profit_percentage,
                direction=f"{buy_exchange} -> {sell_exchange}",
                max_volume=max_volume,
                buy_commission=buy_commission * Decimal('100'),  # В процентах
                sell_commission=sell_commission * Decimal('100'),  # В процентах
                net_profit_percentage=net_profit_percentage,
                estimated_profit=Decimal('0'),
                timestamp=market_data.timestamp,
                is_viable=False,
                viability_reason="Недостаточная ликвидность"
            )
        
        # Учитываем проскальзывание (упрощенная модель)
        adjusted_buy_price = buy_price * (Decimal('1') + self.slippage_factor)
        adjusted_sell_price = sell_price * (Decimal('1') - self.slippage_factor)
        
        # Пересчитываем прибыль с учетом проскальзывания
        adjusted_profit_percentage = (adjusted_sell_price - adjusted_buy_price) / adjusted_buy_price * Decimal('100')
        adjusted_net_profit_percentage = adjusted_profit_percentage - total_commission_percentage
        
        # Оцениваем прибыль в абсолютном выражении
        estimated_profit = max_volume * adjusted_buy_price * adjusted_net_profit_percentage / Decimal('100')
        
        # Проверяем, выгодна ли сделка
        is_viable = adjusted_net_profit_percentage >= self.min_profit_percentage
        viability_reason = None if is_viable else "Недостаточная прибыль после учета комиссий и проскальзывания"
        
        return ArbitrageOpportunity(
            symbol=market_data.symbol,
            buy_exchange=buy_exchange,
            sell_exchange=sell_exchange,
            buy_price=buy_price,
            sell_price=sell_price,
            profit_percentage=profit_percentage,
            direction=f"{buy_exchange} -> {sell_exchange}",
            max_volume=max_volume,
            buy_commission=buy_commission * Decimal('100'),  # В процентах
            sell_commission=sell_commission * Decimal('100'),  # В процентах
            net_profit_percentage=adjusted_net_profit_percentage,
            estimated_profit=estimated_profit,
            timestamp=market_data.timestamp,
            is_viable=is_viable,
            viability_reason=viability_reason
        )
    
    @staticmethod
    def _calculate_max_trade_volume(
            buy_order_book: NormalizedOrderBook,
        sell_order_book: NormalizedOrderBook
    ) -> Decimal:
        """
        Рассчитывает максимальный объем для арбитражной сделки на основе глубины ликвидности.
        
        Args:
            buy_order_book: Стакан ордеров для покупки
            sell_order_book: Стакан ордеров для продажи
            
        Returns:
            Максимальный объем для сделки
        """
        # Получаем доступный объем для покупки (предложение)
        buy_volume = Decimal('0')
        for _, volume in buy_order_book.asks:
            buy_volume += volume
        
        # Получаем доступный объем для продажи (спрос)
        sell_volume = Decimal('0')
        for _, volume in sell_order_book.bids:
            sell_volume += volume
        
        # Возвращаем минимальный из доступных объемов
        return min(buy_volume, sell_volume)
    
    def analyze_all_opportunities(
        self,
        market_data: NormalizedMarketData
    ) -> List[ArbitrageOpportunity]:
        """
        Анализирует все возможные арбитражные возможности для указанного рынка.
        
        Args:
            market_data: Нормализованные рыночные данные
            
        Returns:
            Список арбитражных возможностей с расчетом прибыли
        """
        result = []
        exchanges = market_data.exchanges
        
        # Перебираем все пары бирж
        for i, buy_exchange in enumerate(exchanges):
            for sell_exchange in exchanges[i+1:]:
                # Проверяем первое направление: покупка на buy_exchange, продажа на sell_exchange
                opportunity1 = self.calculate_opportunity_profit(market_data, buy_exchange, sell_exchange)
                if opportunity1:
                    result.append(opportunity1)
                
                # Проверяем второе направление: покупка на sell_exchange, продажа на buy_exchange
                opportunity2 = self.calculate_opportunity_profit(market_data, sell_exchange, buy_exchange)
                if opportunity2:
                    result.append(opportunity2)
        
        # Сортируем возможности по проценту чистой прибыли
        return sorted(result, key=lambda x: x.net_profit_percentage, reverse=True)
    
    def get_best_opportunity(
        self,
        market_data: NormalizedMarketData
    ) -> Optional[ArbitrageOpportunity]:
        """
        Возвращает лучшую арбитражную возможность для указанного рынка.
        
        Args:
            market_data: Нормализованные рыночные данные
            
        Returns:
            Лучшая арбитражная возможность или None, если нет выгодных возможностей
        """
        opportunities = self.analyze_all_opportunities(market_data)
        
        # Фильтруем только жизнеспособные возможности
        viable_opportunities = [opp for opp in opportunities if opp.is_viable]
        
        if not viable_opportunities:
            return None
        
        return viable_opportunities[0]
    
    def calculate_break_even_price(
        self,
        buy_price: Decimal,
        buy_exchange: str,
        sell_exchange: str
    ) -> Decimal:
        """
        Рассчитывает безубыточную цену продажи для указанной цены покупки.
        
        Args:
            buy_price: Цена покупки
            buy_exchange: Биржа для покупки
            sell_exchange: Биржа для продажи
            
        Returns:
            Минимальная цена продажи для достижения безубыточности
        """
        buy_commission = self.exchange_fees.get(buy_exchange, Decimal('0.001'))
        sell_commission = self.exchange_fees.get(sell_exchange, Decimal('0.001'))
        
        # Формула для безубыточности: sell_price = buy_price * (1 + buy_commission) / (1 - sell_commission)
        # Эта формула учитывает комиссии при покупке и продаже
        break_even_price = buy_price * (Decimal('1') + buy_commission) / (Decimal('1') - sell_commission)
        
        return break_even_price
