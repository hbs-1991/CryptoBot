"""
Модуль виртуального исполнения сделок для симуляции арбитражной торговли.
Позволяет симулировать выполнение торговых операций без реального взаимодействия с биржами.
"""

import logging
import time
import uuid
import random
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple

from src.data_fetching.data_normalizer import NormalizedMarketData
from src.simulation.simulation import VirtualTrade


class VirtualOrderBook:
    """
    Виртуальный стакан ордеров для симуляции исполнения сделок.
    """

    def __init__(self, bids: List[List[Decimal]], asks: List[List[Decimal]]):
        """
        Инициализация виртуального стакана ордеров.

        Args:
            bids: Список ордеров на покупку [цена, объем]
            asks: Список ордеров на продажу [цена, объем]
        """
        self.bids = bids  # отсортированы по убыванию цены
        self.asks = asks  # отсортированы по возрастанию цены

    def get_best_bid(self) -> Optional[Decimal]:
        """
        Получает лучшую цену покупки (bid).

        Returns:
            Лучшая цена покупки или None, если стакан пуст
        """
        return self.bids[0][0] if self.bids else None

    def get_best_ask(self) -> Optional[Decimal]:
        """
        Получает лучшую цену продажи (ask).

        Returns:
            Лучшая цена продажи или None, если стакан пуст
        """
        return self.asks[0][0] if self.asks else None

    def get_best_bid_volume(self) -> Optional[Decimal]:
        """
        Получает объем по лучшей цене покупки.

        Returns:
            Объем по лучшей цене покупки или None, если стакан пуст
        """
        return self.bids[0][1] if self.bids else None

    def get_best_ask_volume(self) -> Optional[Decimal]:
        """
        Получает объем по лучшей цене продажи.

        Returns:
            Объем по лучшей цене продажи или None, если стакан пуст
        """
        return self.asks[0][1] if self.asks else None

    def execute_market_buy(self, volume: Decimal) -> Tuple[Decimal, Decimal]:
        """
        Выполняет рыночный ордер на покупку и возвращает среднюю цену исполнения.

        Args:
            volume: Объем для покупки

        Returns:
            Кортеж (средняя_цена_исполнения, фактически_купленный_объем)
        """
        if not self.asks:
            return Decimal('0'), Decimal('0')

        remaining_volume = volume
        total_cost = Decimal('0')
        executed_volume = Decimal('0')

        # Проходим по всем уровням продажи, начиная с лучшей цены
        for i, (price, available_volume) in enumerate(self.asks.copy()):
            # Определяем, сколько можем купить на этом уровне
            buy_volume = min(remaining_volume, available_volume)

            # Обновляем стоимость и объем
            total_cost += buy_volume * price
            executed_volume += buy_volume
            remaining_volume -= buy_volume

            # Обновляем стакан
            if buy_volume == available_volume:
                # Удаляем уровень, если весь объем исполнен
                self.asks.pop(i)
            else:
                # Уменьшаем доступный объем
                self.asks[i][1] = available_volume - buy_volume

            # Проверяем, исполнен ли весь объем
            if remaining_volume <= 0:
                break

        # Рассчитываем среднюю цену исполнения
        avg_price = total_cost / executed_volume if executed_volume > 0 else Decimal('0')

        return avg_price, executed_volume

    def execute_market_sell(self, volume: Decimal) -> Tuple[Decimal, Decimal]:
        """
        Выполняет рыночный ордер на продажу и возвращает среднюю цену исполнения.

        Args:
            volume: Объем для продажи

        Returns:
            Кортеж (средняя_цена_исполнения, фактически_проданный_объем)
        """
        if not self.bids:
            return Decimal('0'), Decimal('0')

        remaining_volume = volume
        total_revenue = Decimal('0')
        executed_volume = Decimal('0')

        # Проходим по всем уровням покупки, начиная с лучшей цены
        for i, (price, available_volume) in enumerate(self.bids.copy()):
            # Определяем, сколько можем продать на этом уровне
            sell_volume = min(remaining_volume, available_volume)

            # Обновляем выручку и объем
            total_revenue += sell_volume * price
            executed_volume += sell_volume
            remaining_volume -= sell_volume

            # Обновляем стакан
            if sell_volume == available_volume:
                # Удаляем уровень, если весь объем исполнен
                self.bids.pop(i)
            else:
                # Уменьшаем доступный объем
                self.bids[i][1] = available_volume - sell_volume

            # Проверяем, исполнен ли весь объем
            if remaining_volume <= 0:
                break

        # Рассчитываем среднюю цену исполнения
        avg_price = total_revenue / executed_volume if executed_volume > 0 else Decimal('0')

        return avg_price, executed_volume


class VirtualTradeExecutor:
    """
    Исполнитель виртуальных сделок для симуляции арбитражной торговли.
    """

    def __init__(
            self,
            slippage_model: str = "fixed",
            fixed_slippage_percentage: float = 0.1,
            market_impact_factor: float = 0.05,
            execution_delay_ms: int = 500,
            execution_failure_probability: float = 0.02,
            partial_fill_probability: float = 0.1
    ):
        """
        Инициализация виртуального исполнителя сделок.

        Args:
            slippage_model: Модель проскальзывания ("fixed", "dynamic", "impact")
            fixed_slippage_percentage: Фиксированный процент проскальзывания
            market_impact_factor: Фактор влияния размера сделки на рынок
            execution_delay_ms: Задержка исполнения в миллисекундах
            execution_failure_probability: Вероятность неудачного исполнения
            partial_fill_probability: Вероятность частичного исполнения
        """
        self.logger = logging.getLogger(__name__)
        self.slippage_model = slippage_model
        self.fixed_slippage_percentage = Decimal(str(fixed_slippage_percentage)) / Decimal('100')
        self.market_impact_factor = Decimal(str(market_impact_factor))
        self.execution_delay_ms = execution_delay_ms
        self.execution_failure_probability = execution_failure_probability
        self.partial_fill_probability = partial_fill_probability

        self.logger.info(
            f"VirtualTradeExecutor initialized with slippage_model={slippage_model}, "
            f"fixed_slippage={fixed_slippage_percentage}%, "
            f"execution_delay={execution_delay_ms}ms"
        )

    def _apply_slippage(
            self,
            price: Decimal,
            volume: Decimal,
            is_buy: bool,
            market_data: NormalizedMarketData,
            exchange: str
    ) -> Decimal:
        """
        Применяет проскальзывание к цене в зависимости от выбранной модели.

        Args:
            price: Исходная цена
            volume: Объем сделки
            is_buy: True для покупки, False для продажи
            market_data: Нормализованные рыночные данные
            exchange: Биржа

        Returns:
            Цена с учетом проскальзывания
        """
        if self.slippage_model == "fixed":
            # Фиксированное проскальзывание
            slippage = price * self.fixed_slippage_percentage
            return price + slippage if is_buy else price - slippage

        elif self.slippage_model == "dynamic":
            # Динамическое проскальзывание на основе спреда
            if exchange not in market_data.tickers:
                return price

            ticker = market_data.tickers[exchange]
            spread = ticker.spread

            # Проскальзывание пропорционально спреду
            slippage = spread * Decimal('0.5')  # 50% от спреда
            return price + slippage if is_buy else price - slippage

        elif self.slippage_model == "impact":
            # Проскальзывание на основе влияния на рынок
            if exchange not in market_data.order_books:
                return price

            order_book = market_data.order_books[exchange]

            # Получаем объем на лучших уровнях
            if is_buy:
                best_volume = order_book.best_ask_volume or Decimal('1')
            else:
                best_volume = order_book.best_bid_volume or Decimal('1')

            # Рассчитываем влияние объема сделки
            volume_ratio = volume / best_volume
            impact = price * self.market_impact_factor * volume_ratio

            return price + impact if is_buy else price - impact

        # По умолчанию не применяем проскальзывание
        return price

    def execute_virtual_trade(
            self,
            trade: VirtualTrade,
            market_data: NormalizedMarketData
    ) -> Dict[str, Any]:
        """
        Выполняет виртуальную сделку и возвращает результат.

        Args:
            trade: Виртуальная сделка для исполнения
            market_data: Нормализованные рыночные данные

        Returns:
            Словарь с результатами исполнения сделки
        """
        # Генерируем уникальный ID для результата исполнения
        execution_id = str(uuid.uuid4())

        # Время начала исполнения
        start_time = int(time.time() * 1000)

        # Определяем результат исполнения
        result = {
            "execution_id": execution_id,
            "trade_id": trade.trade_id,
            "symbol": trade.symbol,
            "buy_exchange": trade.buy_exchange,
            "sell_exchange": trade.sell_exchange,
            "requested_volume": float(trade.volume),
            "start_time": start_time,
            "end_time": start_time + self.execution_delay_ms,
            "duration_ms": self.execution_delay_ms,
            "status": "completed",
            "error": None
        }

        try:
            # Проверяем наличие данных для бирж
            if trade.buy_exchange not in market_data.tickers or trade.sell_exchange not in market_data.tickers:
                result["status"] = "failed"
                result["error"] = "Missing market data for exchanges"
                return result

            # Извлекаем данные тикеров
            buy_ticker = market_data.tickers[trade.buy_exchange]
            sell_ticker = market_data.tickers[trade.sell_exchange]

            # Определяем фактические цены исполнения с учетом проскальзывания
            actual_buy_price = self._apply_slippage(
                price=trade.buy_price,
                volume=trade.volume,
                is_buy=True,
                market_data=market_data,
                exchange=trade.buy_exchange
            )

            actual_sell_price = self._apply_slippage(
                price=trade.sell_price,
                volume=trade.volume,
                is_buy=False,
                market_data=market_data,
                exchange=trade.sell_exchange
            )

            # Определяем фактический объем исполнения
            actual_volume = trade.volume

            # Моделируем частичное исполнение
            if self.partial_fill_probability > 0:
                if random.random() < self.partial_fill_probability:
                    # Исполняем только часть объема (60-99%)
                    fill_ratio = Decimal(str(random.uniform(0.6, 0.99)))
                    actual_volume *= fill_ratio
                    result["partial_fill"] = True
                    result["fill_ratio"] = float(fill_ratio)

            # Рассчитываем фактическую прибыль
            actual_cost = actual_volume * actual_buy_price
            actual_revenue = actual_volume * actual_sell_price
            actual_profit = actual_revenue - actual_cost

            # Обновляем результат
            result.update({
                "actual_volume": float(actual_volume),
                "actual_buy_price": float(actual_buy_price),
                "actual_sell_price": float(actual_sell_price),
                "actual_cost": float(actual_cost),
                "actual_revenue": float(actual_revenue),
                "actual_profit": float(actual_profit),
                "actual_profit_percentage": float(
                    (actual_sell_price - actual_buy_price) / actual_buy_price * 100) if actual_buy_price > 0 else 0
            })

        except Exception as e:
            # В случае ошибки
            result["status"] = "failed"
            result["error"] = str(e)

        return result

    def execute_buy(
            self,
            exchange: str,
            symbol: str,
            volume: Decimal,
            price: Decimal,
            market_data: NormalizedMarketData
    ) -> Dict[str, Any]:
        """
        Выполняет виртуальную покупку и возвращает результат.

        Args:
            exchange: Биржа для покупки
            symbol: Торговая пара
            volume: Объем для покупки
            price: Цена покупки
            market_data: Нормализованные рыночные данные

        Returns:
            Словарь с результатами исполнения покупки
        """
        # Генерируем уникальный ID для результата исполнения
        execution_id = str(uuid.uuid4())

        # Время начала исполнения
        start_time = int(time.time() * 1000)

        # Определяем результат исполнения
        result = {
            "execution_id": execution_id,
            "exchange": exchange,
            "symbol": symbol,
            "side": "buy",
            "requested_volume": float(volume),
            "requested_price": float(price),
            "start_time": start_time,
            "end_time": start_time + self.execution_delay_ms,
            "duration_ms": self.execution_delay_ms,
            "status": "completed",
            "error": None
        }

        try:
            # Проверяем наличие данных для биржи
            if exchange not in market_data.order_books:
                result["status"] = "failed"
                result["error"] = "Missing order book data for exchange"
                return result

            # Извлекаем данные стакана
            order_book_data = market_data.order_books[exchange]

            # Создаем виртуальный стакан для симуляции исполнения
            virtual_book = VirtualOrderBook(
                bids=[[Decimal(str(p)), Decimal(str(v))] for p, v in order_book_data.bids],
                asks=[[Decimal(str(p)), Decimal(str(v))] for p, v in order_book_data.asks]
            )

            # Определяем фактическую цену исполнения с учетом проскальзывания
            actual_price = self._apply_slippage(
                price=price,
                volume=volume,
                is_buy=True,
                market_data=market_data,
                exchange=exchange
            )

            # Выполняем покупку и получаем фактическую цену и объем
            avg_price, actual_volume = virtual_book.execute_market_buy(volume)

            # Если не удалось исполнить по рыночной цене, используем цену с проскальзыванием
            if avg_price <= 0:
                avg_price = actual_price

            # Рассчитываем фактическую стоимость
            actual_cost = actual_volume * avg_price

            # Обновляем результат
            result.update({
                "actual_volume": float(actual_volume),
                "actual_price": float(avg_price),
                "actual_cost": float(actual_cost),
                "fill_ratio": float(actual_volume / volume) if volume > 0 else 0,
                "slippage_percentage": float((avg_price - price) / price * 100) if price > 0 else 0
            })

            # Проверяем, был ли ордер исполнен полностью
            if actual_volume < volume:
                result["partial_fill"] = True

        except Exception as e:
            # В случае ошибки
            result["status"] = "failed"
            result["error"] = str(e)

        return result

    def execute_sell(
            self,
            exchange: str,
            symbol: str,
            volume: Decimal,
            price: Decimal,
            market_data: NormalizedMarketData
    ) -> Dict[str, Any]:
        """
        Выполняет виртуальную продажу и возвращает результат.

        Args:
            exchange: Биржа для продажи
            symbol: Торговая пара
            volume: Объем для продажи
            price: Цена продажи
            market_data: Нормализованные рыночные данные

        Returns:
            Словарь с результатами исполнения продажи
        """
        # Генерируем уникальный ID для результата исполнения
        execution_id = str(uuid.uuid4())

        # Время начала исполнения
        start_time = int(time.time() * 1000)

        # Определяем результат исполнения
        result = {
            "execution_id": execution_id,
            "exchange": exchange,
            "symbol": symbol,
            "side": "sell",
            "requested_volume": float(volume),
            "requested_price": float(price),
            "start_time": start_time,
            "end_time": start_time + self.execution_delay_ms,
            "duration_ms": self.execution_delay_ms,
            "status": "completed",
            "error": None
        }

        try:
            # Проверяем наличие данных для биржи
            if exchange not in market_data.order_books:
                result["status"] = "failed"
                result["error"] = "Missing order book data for exchange"
                return result

            # Извлекаем данные стакана
            order_book_data = market_data.order_books[exchange]

            # Создаем виртуальный стакан для симуляции исполнения
            virtual_book = VirtualOrderBook(
                bids=[[Decimal(str(p)), Decimal(str(v))] for p, v in order_book_data.bids],
                asks=[[Decimal(str(p)), Decimal(str(v))] for p, v in order_book_data.asks]
            )

            # Определяем фактическую цену исполнения с учетом проскальзывания
            actual_price = self._apply_slippage(
                price=price,
                volume=volume,
                is_buy=False,
                market_data=market_data,
                exchange=exchange
            )

            # Выполняем продажу и получаем фактическую цену и объем
            avg_price, actual_volume = virtual_book.execute_market_sell(volume)

            # Если не удалось исполнить по рыночной цене, используем цену с проскальзыванием
            if avg_price <= 0:
                avg_price = actual_price

            # Рассчитываем фактическую выручку
            actual_revenue = actual_volume * avg_price

            # Обновляем результат
            result.update({
                "actual_volume": float(actual_volume),
                "actual_price": float(avg_price),
                "actual_revenue": float(actual_revenue),
                "fill_ratio": float(actual_volume / volume) if volume > 0 else 0,
                "slippage_percentage": float((price - avg_price) / price * 100) if price > 0 else 0
            })

            # Проверяем, был ли ордер исполнен полностью
            if actual_volume < volume:
                result["partial_fill"] = True

        except Exception as e:
            # В случае ошибки
            result["status"] = "failed"
            result["error"] = str(e)

        return result

    def simulate_arbitrage_execution(
            self,
            trade: VirtualTrade,
            market_data: NormalizedMarketData
    ) -> Dict[str, Any]:
        """
        Симулирует полное исполнение арбитражной сделки, включая покупку и продажу.

        Args:
            trade: Виртуальная сделка для исполнения
            market_data: Нормализованные рыночные данные

        Returns:
            Словарь с результатами исполнения арбитражной сделки
        """
        # Генерируем уникальный ID для результата исполнения
        execution_id = str(uuid.uuid4())

        # Время начала исполнения
        start_time = int(time.time() * 1000)

        # Определяем базовую и котируемую валюты из символа
        base_currency, quote_currency = trade.symbol.split('/')

        # Определяем результат исполнения
        result = {
            "execution_id": execution_id,
            "trade_id": trade.trade_id,
            "symbol": trade.symbol,
            "buy_exchange": trade.buy_exchange,
            "sell_exchange": trade.sell_exchange,
            "requested_volume": float(trade.volume),
            "start_time": start_time,
            "status": "processing",
            "error": None,
            "steps": []
        }

        try:
            # Шаг 1: Покупка на первой бирже
            buy_result = self.execute_buy(
                exchange=trade.buy_exchange,
                symbol=trade.symbol,
                volume=trade.volume,
                price=trade.buy_price,
                market_data=market_data
            )

            # Добавляем результат покупки в шаги
            result["steps"].append({
                "step": "buy",
                "exchange": trade.buy_exchange,
                "result": buy_result
            })

            # Если покупка не удалась, останавливаем исполнение
            if buy_result["status"] != "completed" or buy_result.get("actual_volume", 0) <= 0:
                result["status"] = "failed"
                result["error"] = f"Buy step failed: {buy_result.get('error', 'Unknown error')}"
                result["end_time"] = int(time.time() * 1000)
                result["duration_ms"] = result["end_time"] - start_time
                return result

            # Получаем фактический объем покупки
            actual_buy_volume = Decimal(str(buy_result["actual_volume"]))

            # Шаг 2: Продажа на второй бирже
            sell_result = self.execute_sell(
                exchange=trade.sell_exchange,
                symbol=trade.symbol,
                volume=actual_buy_volume,
                price=trade.sell_price,
                market_data=market_data
            )

            # Добавляем результат продажи в шаги
            result["steps"].append({
                "step": "sell",
                "exchange": trade.sell_exchange,
                "result": sell_result
            })

            # Если продажа не удалась, останавливаем исполнение
            if sell_result["status"] != "completed" or sell_result.get("actual_volume", 0) <= 0:
                result["status"] = "failed"
                result["error"] = f"Sell step failed: {sell_result.get('error', 'Unknown error')}"
                result["end_time"] = int(time.time() * 1000)
                result["duration_ms"] = result["end_time"] - start_time
                return result

            # Рассчитываем фактические результаты арбитража
            actual_buy_cost = Decimal(str(buy_result["actual_cost"]))
            actual_sell_revenue = Decimal(str(sell_result["actual_revenue"]))

            # Рассчитываем прибыль
            actual_profit = actual_sell_revenue - actual_buy_cost
            profit_percentage = (actual_profit / actual_buy_cost * Decimal('100')) if actual_buy_cost > 0 else Decimal(
                '0')

            # Обновляем результат
            result.update({
                "status": "completed",
                "actual_buy_volume": float(buy_result["actual_volume"]),
                "actual_buy_price": float(buy_result["actual_price"]),
                "actual_sell_volume": float(sell_result["actual_volume"]),
                "actual_sell_price": float(sell_result["actual_price"]),
                "actual_buy_cost": float(actual_buy_cost),
                "actual_sell_revenue": float(actual_sell_revenue),
                "actual_profit": float(actual_profit),
                "actual_profit_percentage": float(profit_percentage)
            })

            # Устанавливаем время завершения
            result["end_time"] = int(time.time() * 1000)
            result["duration_ms"] = result["end_time"] - start_time

        except Exception as e:
            # В случае ошибки
            result["status"] = "failed"
            result["error"] = str(e)
            result["end_time"] = int(time.time() * 1000)
            result["duration_ms"] = result["end_time"] - start_time

        return result

    def execute_triangular_arbitrage(
            self,
            trade_path: List[Dict[str, Any]],
            exchange: str,
            initial_amount: Decimal,
            market_data: NormalizedMarketData
    ) -> Dict[str, Any]:
        """
        Выполняет треугольный арбитраж, состоящий из нескольких последовательных сделок.

        Args:
            trade_path: Путь трейдов в формате [{symbol, side, price}, ...]
            exchange: Биржа для всех сделок
            initial_amount: Начальная сумма для первой сделки
            market_data: Нормализованные рыночные данные

        Returns:
            Словарь с результатами исполнения треугольного арбитража
        """
        # Генерируем уникальный ID для результата исполнения
        execution_id = str(uuid.uuid4())

        # Время начала исполнения
        start_time = int(time.time() * 1000)

        # Определяем результат исполнения
        result = {
            "execution_id": execution_id,
            "exchange": exchange,
            "trade_path": trade_path,
            "initial_amount": float(initial_amount),
            "start_time": start_time,
            "status": "processing",
            "error": None,
            "steps": []
        }

        try:
            # Текущая сумма для каждой сделки в пути
            current_amount = initial_amount
            current_currency = trade_path[0].get("from_currency")

            # Выполняем каждую сделку в пути
            for i, trade_step in enumerate(trade_path):
                symbol = trade_step.get("symbol")
                side = trade_step.get("side")
                price = Decimal(str(trade_step.get("price", 0)))
                from_currency = trade_step.get("from_currency")
                to_currency = trade_step.get("to_currency")

                # Проверяем, что текущая валюта совпадает с валютой для этой сделки
                if from_currency != current_currency:
                    result["status"] = "failed"
                    result["error"] = f"Currency mismatch at step {i}: expected {from_currency}, got {current_currency}"
                    result["end_time"] = int(time.time() * 1000)
                    result["duration_ms"] = result["end_time"] - start_time
                    return result

                # Определяем объем для сделки
                if side == "buy":
                    # При покупке объем - это сколько базовой валюты мы получим
                    volume = current_amount / price
                    step_result = self.execute_buy(
                        exchange=exchange,
                        symbol=symbol,
                        volume=volume,
                        price=price,
                        market_data=market_data
                    )
                    # После покупки у нас будет базовая валюта
                    if step_result["status"] == "completed":
                        current_amount = Decimal(str(step_result["actual_volume"]))
                else:  # side == "sell"
                    # При продаже объем - это сколько базовой валюты мы продаем
                    volume = current_amount
                    step_result = self.execute_sell(
                        exchange=exchange,
                        symbol=symbol,
                        volume=volume,
                        price=price,
                        market_data=market_data
                    )
                    # После продажи у нас будет котируемая валюта
                    if step_result["status"] == "completed":
                        current_amount = Decimal(str(step_result["actual_revenue"]))

                # Добавляем результат сделки в шаги
                result["steps"].append({
                    "step": i + 1,
                    "symbol": symbol,
                    "side": side,
                    "from_currency": from_currency,
                    "to_currency": to_currency,
                    "result": step_result
                })

                # Если сделка не удалась, останавливаем исполнение
                if step_result["status"] != "completed":
                    result["status"] = "failed"
                    result["error"] = f"Step {i + 1} failed: {step_result.get('error', 'Unknown error')}"
                    result["end_time"] = int(time.time() * 1000)
                    result["duration_ms"] = result["end_time"] - start_time
                    return result

                # Обновляем текущую валюту
                current_currency = to_currency

            # Рассчитываем итоговый результат
            final_amount = current_amount
            profit = final_amount - initial_amount
            profit_percentage = (profit / initial_amount * Decimal('100')) if initial_amount > 0 else Decimal('0')

            # Обновляем результат
            result.update({
                "status": "completed",
                "final_amount": float(final_amount),
                "final_currency": current_currency,
                "profit": float(profit),
                "profit_percentage": float(profit_percentage),
                "end_time": int(time.time() * 1000)
            })

            # Устанавливаем время выполнения
            result["duration_ms"] = result["end_time"] - start_time

        except Exception as e:
            # В случае ошибки
            result["status"] = "failed"
            result["error"] = str(e)
            result["end_time"] = int(time.time() * 1000)
            result["duration_ms"] = result["end_time"] - start_time

        return result

    def simulate_execution_with_failure(
            self,
            trade: VirtualTrade,
            market_data: NormalizedMarketData,
            failure_mode: str = "random"
    ) -> Dict[str, Any]:
        """
        Симулирует исполнение сделки с возможными ошибками и отказами.

        Args:
            trade: Виртуальная сделка для исполнения
            market_data: Нормализованные рыночные данные
            failure_mode: Режим отказа ("random", "network", "liquidity", "none")

        Returns:
            Словарь с результатами исполнения сделки
        """
        # Генерируем уникальный ID для результата исполнения
        execution_id = str(uuid.uuid4())

        # Время начала исполнения
        start_time = int(time.time() * 1000)

        # Определяем результат исполнения
        result = {
            "execution_id": execution_id,
            "trade_id": trade.trade_id,
            "symbol": trade.symbol,
            "buy_exchange": trade.buy_exchange,
            "sell_exchange": trade.sell_exchange,
            "requested_volume": float(trade.volume),
            "start_time": start_time,
            "status": "processing",
            "error": None,
        }

        # Проверяем, нужно ли симулировать отказ
        if failure_mode == "random":
            # Случайный отказ с заданной вероятностью
            if random.random() < self.execution_failure_probability:
                failure_type = random.choice(["network", "liquidity", "timeout", "api_limit"])
                result["status"] = "failed"
                result["error"] = f"Simulated failure: {failure_type}"
                result["end_time"] = int(time.time() * 1000)
                result["duration_ms"] = result["end_time"] - start_time
                return result
        elif failure_mode == "network":
            # Всегда симулируем сетевую ошибку
            result["status"] = "failed"
            result["error"] = "Simulated network error: Connection timeout"
            result["end_time"] = int(time.time() * 1000)
            result["duration_ms"] = result["end_time"] - start_time
            return result
        elif failure_mode == "liquidity":
            # Всегда симулируем недостаток ликвидности
            result["status"] = "failed"
            result["error"] = "Simulated liquidity error: Insufficient liquidity"
            result["end_time"] = int(time.time() * 1000)
            result["duration_ms"] = result["end_time"] - start_time
            return result

        # Если не нужно симулировать отказ, исполняем обычным образом
        execution_result = self.simulate_arbitrage_execution(trade, market_data)

        return execution_result