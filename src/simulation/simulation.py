"""
Модуль симуляции сделок для тестирования арбитражных стратегий.
Позволяет проводить виртуальные сделки без реального исполнения на биржах.
"""

import logging
import time
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import asyncio

from src.data_fetching.data_normalizer import NormalizedMarketData
from src.arbitrage_engine.opportunity_finder import OpportunityFinder
from src.arbitrage_engine.market_strategy import StrategySelector
from src.arbitrage_engine.profit_calculator import ArbitrageOpportunity, ProfitCalculator


class VirtualBalance:
    """
    Класс для хранения и управления виртуальными балансами по валютам и биржам.
    """

    def __init__(self, initial_balances: Dict[str, Dict[str, float]] = None):
        """
        Инициализация виртуальных балансов.

        Args:
            initial_balances: Начальные балансы в формате {биржа: {валюта: количество}}
        """
        self.logger = logging.getLogger(__name__)
        self._balances: Dict[str, Dict[str, Decimal]] = {}

        # Инициализация начальных балансов
        if initial_balances:
            for exchange, currencies in initial_balances.items():
                if exchange not in self._balances:
                    self._balances[exchange] = {}

                for currency, amount in currencies.items():
                    self._balances[exchange][currency] = Decimal(str(amount))

    def get_balance(self, exchange: str, currency: str) -> Decimal:
        """
        Получает текущий баланс указанной валюты на указанной бирже.

        Args:
            exchange: Имя биржи
            currency: Код валюты

        Returns:
            Текущий баланс
        """
        if exchange not in self._balances:
            return Decimal('0')

        return self._balances[exchange].get(currency, Decimal('0'))

    def set_balance(self, exchange: str, currency: str, amount: Decimal) -> None:
        """
        Устанавливает баланс для указанной валюты на указанной бирже.

        Args:
            exchange: Имя биржи
            currency: Код валюты
            amount: Сумма баланса
        """
        if exchange not in self._balances:
            self._balances[exchange] = {}

        self._balances[exchange][currency] = amount
        self.logger.debug(f"Set balance on {exchange} for {currency}: {amount}")

    def add_to_balance(self, exchange: str, currency: str, amount: Decimal) -> None:
        """
        Добавляет сумму к текущему балансу указанной валюты на указанной бирже.

        Args:
            exchange: Имя биржи
            currency: Код валюты
            amount: Сумма для добавления (может быть отрицательной)
        """
        current = self.get_balance(exchange, currency)
        self.set_balance(exchange, currency, current + amount)
        self.logger.debug(f"Added {amount} to balance on {exchange} for {currency}, new balance: {current + amount}")

    def subtract_from_balance(self, exchange: str, currency: str, amount: Decimal) -> bool:
        """
        Вычитает сумму из текущего баланса, если баланс достаточен.

        Args:
            exchange: Имя биржи
            currency: Код валюты
            amount: Сумма для вычитания

        Returns:
            True если операция успешна, False если недостаточно средств
        """
        current = self.get_balance(exchange, currency)

        if current < amount:
            self.logger.warning(
                f"Insufficient funds on {exchange} for {currency}: "
                f"needed {amount}, have {current}"
            )
            return False

        self.set_balance(exchange, currency, current - amount)
        self.logger.debug(f"Subtracted {amount} from balance on {exchange} for {currency}, new balance: {current - amount}")
        return True

    def get_all_balances(self) -> Dict[str, Dict[str, Decimal]]:
        """
        Возвращает все текущие балансы по всем биржам и валютам.

        Returns:
            Словарь с балансами
        """
        return self._balances.copy()

    def get_total_balance_usd(self, exchange_rates: Dict[str, float]) -> Decimal:
        """
        Рассчитывает общий баланс в USD по всем биржам и валютам.

        Args:
            exchange_rates: Курсы обмена валют к USD

        Returns:
            Общий баланс в USD
        """
        total = Decimal('0')

        for exchange, currencies in self._balances.items():
            for currency, amount in currencies.items():
                # Если валюта - USD, просто добавляем
                if currency == 'USD' or currency == 'USDT':
                    total += amount
                else:
                    # Пытаемся найти курс для валюты
                    rate = exchange_rates.get(currency, None)
                    if rate is not None:
                        total += amount * Decimal(str(rate))

        return total


class VirtualTrade:
    """
    Класс для представления виртуальной сделки.
    """

    def __init__(
        self,
        trade_id: str,
        symbol: str,
        buy_exchange: str,
        sell_exchange: str,
        buy_price: Decimal,
        sell_price: Decimal,
        volume: Decimal,
        open_timestamp: int,
        strategy_name: str,
        expected_profit_percentage: Decimal,
        status: str = "open"
    ):
        """
        Инициализация виртуальной сделки.

        Args:
            trade_id: Уникальный идентификатор сделки
            symbol: Символ торговой пары (например, BTC/USDT)
            buy_exchange: Биржа для покупки
            sell_exchange: Биржа для продажи
            buy_price: Цена покупки
            sell_price: Цена продажи
            volume: Объем сделки в базовой валюте
            open_timestamp: Время открытия сделки (Unix timestamp в миллисекундах)
            strategy_name: Название используемой стратегии
            expected_profit_percentage: Ожидаемый процент прибыли
            status: Статус сделки ("open", "closed", "canceled", "failed")
        """
        self.trade_id = trade_id
        self.symbol = symbol
        self.buy_exchange = buy_exchange
        self.sell_exchange = sell_exchange
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.volume = volume
        self.open_timestamp = open_timestamp
        self.strategy_name = strategy_name
        self.expected_profit_percentage = expected_profit_percentage
        self.status = status

        # Дополнительные поля для закрытой сделки
        self.close_timestamp: Optional[int] = None
        self.actual_sell_price: Optional[Decimal] = None
        self.actual_profit: Optional[Decimal] = None
        self.actual_profit_percentage: Optional[Decimal] = None
        self.close_reason: Optional[str] = None

    @property
    def expected_profit(self) -> Decimal:
        """
        Рассчитывает ожидаемую прибыль от сделки.

        Returns:
            Ожидаемая прибыль в котируемой валюте
        """
        return (self.sell_price - self.buy_price) * self.volume

    @property
    def total_cost(self) -> Decimal:
        """
        Рассчитывает общую стоимость сделки (сколько потрачено на покупку).

        Returns:
            Общая стоимость в котируемой валюте
        """
        return self.buy_price * self.volume

    @property
    def is_open(self) -> bool:
        """
        Проверяет, открыта ли еще сделка.

        Returns:
            True если сделка открыта, иначе False
        """
        return self.status == "open"

    @property
    def duration_ms(self) -> int:
        """
        Рассчитывает длительность сделки.

        Returns:
            Длительность в миллисекундах или 0, если сделка еще открыта
        """
        if self.close_timestamp is None:
            return int(time.time() * 1000) - self.open_timestamp
        return self.close_timestamp - self.open_timestamp

    def close(
        self,
        close_timestamp: int,
        actual_sell_price: Decimal,
        close_reason: str = "normal"
    ) -> None:
        """
        Закрывает сделку и рассчитывает фактическую прибыль.

        Args:
            close_timestamp: Время закрытия сделки (Unix timestamp в миллисекундах)
            actual_sell_price: Фактическая цена продажи
            close_reason: Причина закрытия сделки
        """
        self.close_timestamp = close_timestamp
        self.actual_sell_price = actual_sell_price
        self.close_reason = close_reason
        self.status = "closed"

        # Расчет фактической прибыли
        self.actual_profit = (actual_sell_price - self.buy_price) * self.volume
        if self.buy_price > 0:
            self.actual_profit_percentage = (actual_sell_price - self.buy_price) / self.buy_price * Decimal('100')
        else:
            self.actual_profit_percentage = Decimal('0')

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразует сделку в словарь для сериализации.

        Returns:
            Словарь с данными сделки
        """
        result = {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "buy_exchange": self.buy_exchange,
            "sell_exchange": self.sell_exchange,
            "buy_price": float(self.buy_price),
            "sell_price": float(self.sell_price),
            "volume": float(self.volume),
            "open_timestamp": self.open_timestamp,
            "strategy_name": self.strategy_name,
            "expected_profit_percentage": float(self.expected_profit_percentage),
            "expected_profit": float(self.expected_profit),
            "total_cost": float(self.total_cost),
            "status": self.status,
            "duration_ms": self.duration_ms
        }

        # Добавляем поля для закрытой сделки
        if self.status == "closed":
            result.update({
                "close_timestamp": self.close_timestamp,
                "actual_sell_price": float(self.actual_sell_price) if self.actual_sell_price else None,
                "actual_profit": float(self.actual_profit) if self.actual_profit else None,
                "actual_profit_percentage": float(self.actual_profit_percentage) if self.actual_profit_percentage else None,
                "close_reason": self.close_reason
            })

        return result


class TradingSimulator:
    """
    Симулятор торговли для тестирования арбитражных стратегий.
    """

    def __init__(
        self,
        initial_balances: Dict[str, Dict[str, float]],
        exchange_fees: Dict[str, float],
        min_profit_percentage: float = 0.5,
        max_active_trades: int = 10,
        max_trade_duration_ms: int = 60000,  # 1 минута
        emergency_stop_loss_percentage: float = -1.0
    ):
        """
        Инициализация симулятора торговли.

        Args:
            initial_balances: Начальные балансы по биржам и валютам
            exchange_fees: Комиссии бирж в процентах
            min_profit_percentage: Минимальный процент прибыли для сделки
            max_active_trades: Максимальное количество активных сделок
            max_trade_duration_ms: Максимальная длительность сделки в миллисекундах
            emergency_stop_loss_percentage: Процент убытка для экстренного закрытия сделки
        """
        self.logger = logging.getLogger(__name__)
        self.virtual_balance = VirtualBalance(initial_balances)
        self.exchange_fees = {ex: Decimal(str(fee)) / Decimal('100') for ex, fee in exchange_fees.items()}
        self.min_profit_percentage = Decimal(str(min_profit_percentage))
        self.max_active_trades = max_active_trades
        self.max_trade_duration_ms = max_trade_duration_ms
        self.emergency_stop_loss_percentage = Decimal(str(emergency_stop_loss_percentage))

        # Инициализация компонентов арбитража
        self.opportunity_finder = OpportunityFinder(
            exchange_fees=exchange_fees,
            min_profit_percentage=min_profit_percentage
        )

        self.strategy_selector = StrategySelector([])  # Стратегии будут добавлены позже

        # Хранение активных и завершенных сделок
        self.active_trades: Dict[str, VirtualTrade] = {}
        self.completed_trades: List[VirtualTrade] = []

        # Счетчик для генерации ID сделок
        self._trade_counter = 0

        # Статистика
        self.stats = {
            "total_trades": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "total_profit": Decimal('0'),
            "total_volume_traded": Decimal('0'),
            "start_time": int(time.time() * 1000),
            "last_update_time": int(time.time() * 1000)
        }

        self.logger.info(
            f"TradingSimulator initialized with {sum(len(balances) for balances in initial_balances.values())} "
            f"currency balances across {len(initial_balances)} exchanges"
        )

    def get_next_trade_id(self) -> str:
        """
        Генерирует уникальный ID для новой сделки.

        Returns:
            Уникальный ID сделки
        """
        self._trade_counter += 1
        return f"trade_{int(time.time())}_{self._trade_counter}"

    def add_strategy(self, strategy) -> None:
        """
        Добавляет стратегию в селектор стратегий.

        Args:
            strategy: Объект стратегии
        """
        self.strategy_selector.strategies.append(strategy)
        self.logger.info(f"Added strategy: {strategy.name}")

    def check_balance_for_trade(self, exchange: str, currency: str, amount: Decimal) -> bool:
        """
        Проверяет, достаточно ли баланса для сделки.

        Args:
            exchange: Имя биржи
            currency: Код валюты
            amount: Требуемая сумма

        Returns:
            True если баланс достаточен, иначе False
        """
        available = self.virtual_balance.get_balance(exchange, currency)
        return available >= amount

    def process_opportunity(self, opportunity: ArbitrageOpportunity, market_data: NormalizedMarketData) -> Optional[str]:
        """
        Обрабатывает арбитражную возможность, создавая виртуальную сделку если возможно.

        Args:
            opportunity: Арбитражная возможность
            market_data: Нормализованные рыночные данные

        Returns:
            ID созданной сделки или None, если сделка не создана
        """
        # Проверяем лимит активных сделок
        if len(self.active_trades) >= self.max_active_trades:
            self.logger.debug(f"Reached maximum active trades limit ({self.max_active_trades})")
            return None

        # Находим лучшую стратегию для этой возможности
        strategy, trade_params = self.strategy_selector.select_strategy(opportunity, market_data)

        if not strategy or not trade_params:
            self.logger.debug(f"No suitable strategy found for {opportunity.symbol}")
            return None

        # Разбираем параметры сделки
        symbol = opportunity.symbol
        buy_exchange = opportunity.buy_exchange
        sell_exchange = opportunity.sell_exchange
        buy_price = opportunity.buy_price
        sell_price = opportunity.sell_price
        volume = Decimal(str(trade_params.get("volume", 0)))
        expected_profit_percentage = opportunity.net_profit_percentage

        # Получаем базовую и котируемую валюты из символа
        base_currency, quote_currency = symbol.split('/')

        # Проверяем баланс для сделки
        trade_cost = volume * buy_price
        if not self.check_balance_for_trade(buy_exchange, quote_currency, trade_cost):
            self.logger.debug(
                f"Insufficient balance on {buy_exchange} for {quote_currency}: "
                f"needed {trade_cost}, have {self.virtual_balance.get_balance(buy_exchange, quote_currency)}"
            )
            return None

        # Вычитаем средства с баланса для покупки
        self.virtual_balance.subtract_from_balance(buy_exchange, quote_currency, trade_cost)

        # Учитываем комиссию биржи при покупке
        buy_fee = self.exchange_fees.get(buy_exchange, Decimal('0.001'))
        fee_amount = volume * buy_fee
        actual_volume = volume - fee_amount

        # Добавляем купленную валюту на баланс
        self.virtual_balance.add_to_balance(buy_exchange, base_currency, actual_volume)

        # Создаем виртуальную сделку
        trade_id = self.get_next_trade_id()
        trade = VirtualTrade(
            trade_id=trade_id,
            symbol=symbol,
            buy_exchange=buy_exchange,
            sell_exchange=sell_exchange,
            buy_price=buy_price,
            sell_price=sell_price,
            volume=volume,
            open_timestamp=market_data.timestamp,
            strategy_name=strategy.name,
            expected_profit_percentage=expected_profit_percentage
        )

        # Добавляем сделку в активные
        self.active_trades[trade_id] = trade

        # Обновляем статистику
        self.stats["total_trades"] += 1
        self.stats["total_volume_traded"] += trade_cost
        self.stats["last_update_time"] = int(time.time() * 1000)

        self.logger.info(
            f"Opened trade {trade_id}: Buy {volume} {base_currency} on {buy_exchange} "
            f"at {buy_price}, sell on {sell_exchange} at {sell_price}, "
            f"expected profit: {expected_profit_percentage}%"
        )

        return trade_id

    def check_and_close_trades(self, market_data: NormalizedMarketData) -> List[str]:
        """
        Проверяет все активные сделки и закрывает те, которые пора закрыть.

        Args:
            market_data: Нормализованные рыночные данные

        Returns:
            Список ID закрытых сделок
        """
        closed_trade_ids = []
        current_time = int(time.time() * 1000)

        for trade_id, trade in list(self.active_trades.items()):
            # Проверяем длительность сделки
            if current_time - trade.open_timestamp > self.max_trade_duration_ms:
                self.close_trade(trade_id, market_data, "timeout")
                closed_trade_ids.append(trade_id)
                continue

            # Проверяем, есть ли данные для биржи продажи
            sell_exchange = trade.sell_exchange
            symbol = trade.symbol
            base_currency, quote_currency = symbol.split('/')

            # Проверяем наличие тикера и валидность его структуры
            if sell_exchange not in market_data.tickers:
                continue
                
            ticker = market_data.tickers[sell_exchange]
            if not hasattr(ticker, 'symbol') or ticker.symbol != symbol:
                continue

            # Получаем текущую цену для продажи
            current_ticker = market_data.tickers[sell_exchange]
            current_sell_price = current_ticker.bid

            # Проверяем стоп-лосс
            if current_sell_price > 0:
                current_profit_percentage = (current_sell_price - trade.buy_price) / trade.buy_price * Decimal('100')

                if current_profit_percentage <= self.emergency_stop_loss_percentage:
                    self.close_trade(trade_id, market_data, "stop_loss")
                    closed_trade_ids.append(trade_id)
                    continue

            # Проверяем, достигнута ли ожидаемая цена продажи
            if current_sell_price >= trade.sell_price:
                self.close_trade(trade_id, market_data, "target_reached")
                closed_trade_ids.append(trade_id)

        return closed_trade_ids

    def close_trade(self, trade_id: str, market_data: NormalizedMarketData, reason: str = "normal") -> bool:
        """
        Закрывает указанную сделку на основе текущих рыночных данных.

        Args:
            trade_id: ID сделки для закрытия
            market_data: Нормализованные рыночные данные
            reason: Причина закрытия сделки

        Returns:
            True если сделка успешно закрыта, иначе False
        """
        if trade_id not in self.active_trades:
            self.logger.warning(f"Trade {trade_id} not found in active trades")
            return False

        trade = self.active_trades[trade_id]
        sell_exchange = trade.sell_exchange
        symbol = trade.symbol
        base_currency, quote_currency = symbol.split('/')

        # Определяем фактическую цену продажи
        actual_sell_price = trade.sell_price  # По умолчанию - ожидаемая цена

        if sell_exchange in market_data.tickers and market_data.tickers[sell_exchange].symbol == symbol:
            actual_sell_price = market_data.tickers[sell_exchange].bid

        # Закрываем сделку
        trade.close(
            close_timestamp=int(time.time() * 1000),
            actual_sell_price=actual_sell_price,
            close_reason=reason
        )

        # Обновляем балансы
        # Вычитаем базовую валюту с баланса биржи продажи
        available_base = self.virtual_balance.get_balance(sell_exchange, base_currency)
        volume_to_sell = min(trade.volume, available_base)

        if volume_to_sell > 0:
            self.virtual_balance.subtract_from_balance(sell_exchange, base_currency, volume_to_sell)

            # Учитываем комиссию биржи при продаже
            sell_fee = self.exchange_fees.get(sell_exchange, Decimal('0.001'))
            sell_amount = volume_to_sell * actual_sell_price
            fee_amount = sell_amount * sell_fee
            actual_sell_amount = sell_amount - fee_amount

            # Добавляем котируемую валюту на баланс
            self.virtual_balance.add_to_balance(sell_exchange, quote_currency, actual_sell_amount)

        # Обновляем статистику
        if trade.actual_profit is not None:
            self.stats["total_profit"] += trade.actual_profit

            if trade.actual_profit > 0:
                self.stats["successful_trades"] += 1
            else:
                self.stats["failed_trades"] += 1

        self.stats["last_update_time"] = int(time.time() * 1000)

        # Перемещаем сделку из активных в завершенные
        self.completed_trades.append(trade)
        del self.active_trades[trade_id]

        self.logger.info(
            f"Closed trade {trade_id}: sold {volume_to_sell} {base_currency} "
            f"on {sell_exchange} at {actual_sell_price}, "
            f"profit: {trade.actual_profit_percentage}%, reason: {reason}"
        )

        return True

    def get_balance_summary(self) -> Dict[str, Any]:
        """
        Получает сводку по текущим балансам.

        Returns:
            Словарь с информацией о текущих балансах
        """
        balances = self.virtual_balance.get_all_balances()
        summary = {
            "balances": {
                exchange: {currency: float(amount) for currency, amount in currencies.items()}
                for exchange, currencies in balances.items()
            },
            "timestamp": int(time.time() * 1000)
        }

        return summary

    def get_trade_summary(self) -> Dict[str, Any]:
        """
        Получает сводку по текущим и завершенным сделкам.

        Returns:
            Словарь с информацией о сделках
        """
        summary = {
            "active_trades": {
                trade_id: trade.to_dict()
                for trade_id, trade in self.active_trades.items()
            },
            "completed_trades": [
                trade.to_dict() for trade in self.completed_trades
            ],
            "stats": {
                **self.stats,
                "total_profit": float(self.stats["total_profit"]),
                "total_volume_traded": float(self.stats["total_volume_traded"]),
                "active_trades_count": len(self.active_trades),
                "completed_trades_count": len(self.completed_trades)
            },
            "timestamp": int(time.time() * 1000)
        }

        return summary

    async def run_simulation(self, data_source, duration_seconds: int = 60) -> Dict[str, Any]:
        """
        Запускает симуляцию торговли на указанном источнике данных.

        Args:
            data_source: Источник рыночных данных (должен поддерживать метод get_next_data())
            duration_seconds: Длительность симуляции в секундах

        Returns:
            Словарь с результатами симуляции
        """
        start_time = time.time()
        end_time = start_time + duration_seconds
        iteration = 0

        self.logger.info(f"Starting simulation for {duration_seconds} seconds")

        while time.time() < end_time:
            iteration += 1

            try:
                # Получаем следующий пакет данных
                market_data_batch = await data_source.get_next_data()

                if not market_data_batch:
                    self.logger.warning("No market data received, waiting...")
                    await asyncio.sleep(1)
                    continue

                # Обрабатываем каждый набор данных
                for market_data in market_data_batch:
                    # Проверяем и закрываем активные сделки
                    closed_trades = self.check_and_close_trades(market_data)

                    # Ищем новые арбитражные возможности
                    opportunities = self.opportunity_finder.find_opportunities(market_data)

                    # Обрабатываем найденные возможности
                    for opportunity in opportunities:
                        self.process_opportunity(opportunity, market_data)

                # Выводим промежуточную статистику каждые 10 итераций
                if iteration % 10 == 0:
                    stats = self.get_trade_summary()["stats"]
                    self.logger.info(
                        f"Iteration {iteration}: active trades: {stats['active_trades_count']}, "
                        f"completed: {stats['completed_trades_count']}, "
                        f"profit: {stats['total_profit']}, success rate: "
                        f"{stats['successful_trades'] / max(1, stats['total_trades']) * 100:.2f}%"
                    )

                # Небольшая задержка между итерациями
                await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error during simulation iteration {iteration}: {str(e)}")
                await asyncio.sleep(1)

        # Закрываем все активные сделки по последним доступным данным
        if market_data_batch:
            for trade_id in list(self.active_trades.keys()):
                self.close_trade(trade_id, market_data_batch[-1], "simulation_end")

        # Формируем итоговый результат
        result = {
            "balance_summary": self.get_balance_summary(),
            "trade_summary": self.get_trade_summary(),
            "simulation_stats": {
                "start_time": start_time,
                "end_time": time.time(),
                "duration_seconds": time.time() - start_time,
                "iterations": iteration
            }
        }

        self.logger.info(
            f"Simulation completed. Total trades: {result['trade_summary']['stats']['total_trades']}, "
            f"Profit: {result['trade_summary']['stats']['total_profit']}"
        )

        return result