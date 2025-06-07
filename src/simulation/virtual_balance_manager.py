"""
Модуль управления виртуальными балансами для симуляции арбитражной торговли.
Отслеживает балансы по разным валютам на разных биржах и рассчитывает прибыль.
"""

import logging
import time
import json
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple


class VirtualBalanceManager:
    """
    Управляет виртуальными балансами для симуляции арбитражной торговли.
    Отслеживает балансы по разным валютам на разных биржах и рассчитывает прибыль.
    """

    def __init__(
        self,
        initial_balances: Dict[str, Dict[str, float]] = None,
        base_currency: str = "USDT",
        exchange_rates: Dict[str, float] = None
    ):
        """
        Инициализация менеджера виртуальных балансов.

        Args:
            initial_balances: Начальные балансы в формате {биржа: {валюта: количество}}
            base_currency: Базовая валюта для расчета общего баланса (обычно USDT или USD)
            exchange_rates: Курсы обмена валют к базовой валюте
        """
        self.logger = logging.getLogger(__name__)
        self.base_currency = base_currency
        self.exchange_rates = {}

        if exchange_rates:
            self.exchange_rates = {currency: Decimal(str(rate)) for currency, rate in exchange_rates.items()}

        # Инициализация балансов
        self._balances: Dict[str, Dict[str, Decimal]] = {}

        # Инициализация начальных балансов
        if initial_balances:
            for exchange, currencies in initial_balances.items():
                if exchange not in self._balances:
                    self._balances[exchange] = {}

                for currency, amount in currencies.items():
                    self._balances[exchange][currency] = Decimal(str(amount))

        # История изменений балансов
        self._balance_history: List[Dict[str, Any]] = []

        # Сохраняем начальный баланс в историю
        self._save_balance_snapshot(reason="initial")

        self.logger.info(
            f"VirtualBalanceManager initialized with {len(self._balances)} exchanges, "
            f"base currency: {base_currency}"
        )

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

    def get_all_balances(self, exchange: Optional[str] = None) -> Dict[str, Dict[str, Decimal]]:
        """
        Получает все балансы для указанной биржи или для всех бирж.

        Args:
            exchange: Имя биржи (если None, возвращает для всех бирж)

        Returns:
            Словарь с балансами
        """
        if exchange:
            return {exchange: self._balances.get(exchange, {})}
        return self._balances.copy()

    def set_balance(self, exchange: str, currency: str, amount: Decimal, reason: str = "manual") -> None:
        """
        Устанавливает баланс для указанной валюты на указанной бирже.

        Args:
            exchange: Имя биржи
            currency: Код валюты
            amount: Сумма баланса
            reason: Причина изменения баланса
        """
        if exchange not in self._balances:
            self._balances[exchange] = {}

        old_balance = self._balances[exchange].get(currency, Decimal('0'))
        self._balances[exchange][currency] = amount

        self.logger.debug(
            f"Balance set on {exchange} for {currency}: {old_balance} -> {amount} ({reason})"
        )

        # Сохраняем изменение в историю
        self._save_balance_change(
            exchange=exchange,
            currency=currency,
            old_balance=old_balance,
            new_balance=amount,
            change=amount - old_balance,
            reason=reason
        )

    def add_to_balance(self, exchange: str, currency: str, amount: Decimal, reason: str = "deposit") -> None:
        """
        Добавляет сумму к текущему балансу указанной валюты на указанной бирже.

        Args:
            exchange: Имя биржи
            currency: Код валюты
            amount: Сумма для добавления (может быть отрицательной)
            reason: Причина изменения баланса
        """
        current = self.get_balance(exchange, currency)
        self.set_balance(exchange, currency, current + amount, reason)

    def subtract_from_balance(self, exchange: str, currency: str, amount: Decimal, reason: str = "withdrawal") -> bool:
        """
        Вычитает сумму из текущего баланса, если баланс достаточен.

        Args:
            exchange: Имя биржи
            currency: Код валюты
            amount: Сумма для вычитания
            reason: Причина изменения баланса

        Returns:
            True если операция успешна, False если недостаточно средств
        """
        current = self.get_balance(exchange, currency)

        if current < amount:
            self.logger.warning(
                f"Insufficient funds on {exchange} for {currency}: "
                f"needed {amount}, have {current} ({reason})"
            )
            return False

        self.set_balance(exchange, currency, current - amount, reason)
        return True

    def update_exchange_rates(self, exchange_rates: Dict[str, float]) -> None:
        """
        Обновляет курсы обмена валют к базовой валюте.

        Args:
            exchange_rates: Словарь с курсами обмена {валюта: курс к базовой валюте}
        """
        self.exchange_rates = {currency: Decimal(str(rate)) for currency, rate in exchange_rates.items()}
        self.logger.info(f"Exchange rates updated for {len(exchange_rates)} currencies")

    def get_total_balance_in_base_currency(self) -> Decimal:
        """
        Рассчитывает общий баланс во всех валютах, пересчитанный в базовую валюту.

        Returns:
            Общий баланс в базовой валюте
        """
        total = Decimal('0')

        for exchange, currencies in self._balances.items():
            for currency, amount in currencies.items():
                if currency == self.base_currency:
                    total += amount
                elif currency in self.exchange_rates:
                    total += amount * self.exchange_rates[currency]
                else:
                    self.logger.warning(f"No exchange rate found for {currency}, skipping in total balance calculation")

        return total

    def execute_trade(
        self,
        trade_type: str,
        buy_exchange: str,
        sell_exchange: str,
        symbol: str,
        volume: Decimal,
        buy_price: Decimal,
        sell_price: Decimal,
        trade_id: str
    ) -> Dict[str, Any]:
        """
        Выполняет торговую операцию, обновляя балансы соответствующим образом.

        Args:
            trade_type: Тип сделки ("arbitrage", "triangular", etc.)
            buy_exchange: Биржа для покупки
            sell_exchange: Биржа для продажи
            symbol: Торговая пара
            volume: Объем сделки
            buy_price: Цена покупки
            sell_price: Цена продажи
            trade_id: Идентификатор сделки

        Returns:
            Результат выполнения сделки
        """
        # Извлекаем базовую и котируемую валюты из символа
        base_currency, quote_currency = symbol.split('/')

        # Расчет необходимых сумм
        buy_cost = volume * buy_price
        sell_revenue = volume * sell_price
        profit = sell_revenue - buy_cost

        # Проверяем достаточность средств
        if not self.subtract_from_balance(
            exchange=buy_exchange,
            currency=quote_currency,
            amount=buy_cost,
            reason=f"buy_{trade_id}"
        ):
            return {
                "status": "failed",
                "error": f"Insufficient funds on {buy_exchange} for {quote_currency}",
                "trade_id": trade_id
            }

        # Добавляем базовую валюту на биржу покупки
        self.add_to_balance(
            exchange=buy_exchange,
            currency=base_currency,
            amount=volume,
            reason=f"buy_{trade_id}"
        )

        # Проверяем, нужно ли переводить средства между биржами
        if buy_exchange != sell_exchange:
            # Вычитаем базовую валюту с биржи покупки
            if not self.subtract_from_balance(
                exchange=buy_exchange,
                currency=base_currency,
                amount=volume,
                reason=f"transfer_{trade_id}"
            ):
                # Если недостаточно средств, отменяем предыдущие операции
                self.add_to_balance(
                    exchange=buy_exchange,
                    currency=quote_currency,
                    amount=buy_cost,
                    reason=f"rollback_{trade_id}"
                )
                self.subtract_from_balance(
                    exchange=buy_exchange,
                    currency=base_currency,
                    amount=volume,
                    reason=f"rollback_{trade_id}"
                )
                return {
                    "status": "failed",
                    "error": f"Insufficient funds on {buy_exchange} for {base_currency} transfer",
                    "trade_id": trade_id
                }

            # Добавляем базовую валюту на биржу продажи
            self.add_to_balance(
                exchange=sell_exchange,
                currency=base_currency,
                amount=volume,
                reason=f"transfer_{trade_id}"
            )

        # Вычитаем базовую валюту с биржи продажи
        if not self.subtract_from_balance(
            exchange=sell_exchange,
            currency=base_currency,
            amount=volume,
            reason=f"sell_{trade_id}"
        ):
            # Если недостаточно средств, отменяем предыдущие операции
            if buy_exchange != sell_exchange:
                self.add_to_balance(
                    exchange=buy_exchange,
                    currency=base_currency,
                    amount=volume,
                    reason=f"rollback_{trade_id}"
                )
                self.subtract_from_balance(
                    exchange=sell_exchange,
                    currency=base_currency,
                    amount=volume,
                    reason=f"rollback_{trade_id}"
                )
            self.add_to_balance(
                exchange=buy_exchange,
                currency=quote_currency,
                amount=buy_cost,
                reason=f"rollback_{trade_id}"
            )
            self.subtract_from_balance(
                exchange=buy_exchange,
                currency=base_currency,
                amount=volume,
                reason=f"rollback_{trade_id}"
            )
            return {
                "status": "failed",
                "error": f"Insufficient funds on {sell_exchange} for {base_currency}",
                "trade_id": trade_id
            }

        # Добавляем котируемую валюту на биржу продажи
        self.add_to_balance(
            exchange=sell_exchange,
            currency=quote_currency,
            amount=sell_revenue,
            reason=f"sell_{trade_id}"
        )

        # Сохраняем итоговый баланс в историю
        self._save_balance_snapshot(reason=f"trade_{trade_id}")

        # Возвращаем результат
        return {
            "status": "completed",
            "trade_id": trade_id,
            "buy_exchange": buy_exchange,
            "sell_exchange": sell_exchange,
            "symbol": symbol,
            "volume": float(volume),
            "buy_price": float(buy_price),
            "sell_price": float(sell_price),
            "buy_cost": float(buy_cost),
            "sell_revenue": float(sell_revenue),
            "profit": float(profit),
            "profit_percentage": float((sell_price - buy_price) / buy_price * 100) if buy_price > 0 else 0
        }

    def execute_triangular_arbitrage(
        self,
        exchange: str,
        trade_steps: List[Dict[str, Any]],
        initial_currency: str,
        initial_amount: Decimal,
        trade_id: str
    ) -> Dict[str, Any]:
        """
        Выполняет треугольный арбитраж, обновляя балансы соответствующим образом.

        Args:
            exchange: Биржа для всех операций
            trade_steps: Список шагов сделки [{symbol, side, price, volume}, ...]
            initial_currency: Начальная валюта
            initial_amount: Начальная сумма
            trade_id: Идентификатор сделки

        Returns:
            Результат выполнения треугольного арбитража
        """
        # Проверяем достаточность средств для начала арбитража
        if not self.subtract_from_balance(
            exchange=exchange,
            currency=initial_currency,
            amount=initial_amount,
            reason=f"triangular_start_{trade_id}"
        ):
            return {
                "status": "failed",
                "error": f"Insufficient funds on {exchange} for {initial_currency}",
                "trade_id": trade_id
            }

        # Фиксируем начальный баланс для расчета прибыли
        initial_balance = self.get_total_balance_in_base_currency()

        # Выполняем каждый шаг треугольного арбитража
        current_currency = initial_currency
        current_amount = initial_amount
        step_results = []

        try:
            for i, step in enumerate(trade_steps):
                symbol = step["symbol"]
                side = step["side"]
                price = Decimal(str(step["price"]))
                from_currency = step["from_currency"]
                to_currency = step["to_currency"]

                # Проверяем, что текущая валюта совпадает с ожидаемой
                if current_currency != from_currency:
                    # Отменяем все предыдущие шаги и восстанавливаем начальный баланс
                    self.add_to_balance(
                        exchange=exchange,
                        currency=initial_currency,
                        amount=initial_amount,
                        reason=f"triangular_rollback_{trade_id}"
                    )
                    return {
                        "status": "failed",
                        "error": f"Currency mismatch at step {i+1}: expected {from_currency}, got {current_currency}",
                        "trade_id": trade_id,
                        "step": i + 1
                    }

                # Выполняем шаг в зависимости от типа операции
                if side == "buy":
                    # При покупке: тратим котируемую валюту (from_currency), получаем базовую (to_currency)
                    base_currency, quote_currency = symbol.split('/')

                    # Проверяем, что валюты соответствуют ожидаемым
                    if to_currency != base_currency or from_currency != quote_currency:
                        # Отменяем все предыдущие шаги и восстанавливаем начальный баланс
                        self.add_to_balance(
                            exchange=exchange,
                            currency=initial_currency,
                            amount=initial_amount,
                            reason=f"triangular_rollback_{trade_id}"
                        )
                        return {
                            "status": "failed",
                            "error": f"Currency mismatch at step {i+1}: symbol {symbol} does not match {from_currency}->{to_currency}",
                            "trade_id": trade_id,
                            "step": i + 1
                        }

                    # Рассчитываем объем базовой валюты, который мы получим
                    volume = current_amount / price

                    # Добавляем базовую валюту
                    self.add_to_balance(
                        exchange=exchange,
                        currency=base_currency,
                        amount=volume,
                        reason=f"triangular_step{i+1}_{trade_id}"
                    )

                    # Обновляем текущую валюту и сумму
                    current_currency = to_currency
                    current_amount = volume

                else:  # side == "sell"
                    # При продаже: тратим базовую валюту (from_currency), получаем котируемую (to_currency)
                    base_currency, quote_currency = symbol.split('/')

                    # Проверяем, что валюты соответствуют ожидаемым
                    if from_currency != base_currency or to_currency != quote_currency:
                        # Отменяем все предыдущие шаги и восстанавливаем начальный баланс
                        self.add_to_balance(
                            exchange=exchange,
                            currency=initial_currency,
                            amount=initial_amount,
                            reason=f"triangular_rollback_{trade_id}"
                        )
                        return {
                            "status": "failed",
                            "error": f"Currency mismatch at step {i+1}: symbol {symbol} does not match {from_currency}->{to_currency}",
                            "trade_id": trade_id,
                            "step": i + 1
                        }

                    # Рассчитываем стоимость в котируемой валюте
                    quote_amount = current_amount * price

                    # Добавляем котируемую валюту
                    self.add_to_balance(
                        exchange=exchange,
                        currency=quote_currency,
                        amount=quote_amount,
                        reason=f"triangular_step{i+1}_{trade_id}"
                    )

                    # Обновляем текущую валюту и сумму
                    current_currency = to_currency
                    current_amount = quote_amount

                # Добавляем результат текущего шага
                step_results.append({
                    "step": i + 1,
                    "symbol": symbol,
                    "side": side,
                    "price": float(price),
                    "from_currency": from_currency,
                    "from_amount": float(current_amount),
                    "to_currency": to_currency,
                    "to_amount": float(current_amount)
                })

            # Рассчитываем итоговую прибыль
            final_balance = self.get_total_balance_in_base_currency()
            profit = final_balance - initial_balance

            # Сохраняем итоговый баланс в историю
            self._save_balance_snapshot(reason=f"triangular_{trade_id}")

            # Возвращаем результат
            return {
                "status": "completed",
                "trade_id": trade_id,
                "exchange": exchange,
                "initial_currency": initial_currency,
                "initial_amount": float(initial_amount),
                "final_currency": current_currency,
                "final_amount": float(current_amount),
                "profit": float(profit),
                "profit_percentage": float((profit / initial_balance) * 100) if initial_balance > 0 else 0,
                "steps": step_results
            }

        except Exception as e:
            # В случае ошибки восстанавливаем начальный баланс
            self.add_to_balance(
                exchange=exchange,
                currency=initial_currency,
                amount=initial_amount,
                reason=f"triangular_error_{trade_id}"
            )

            return {
                "status": "failed",
                "error": str(e),
                "trade_id": trade_id
            }

    def _save_balance_change(
        self,
        exchange: str,
        currency: str,
        old_balance: Decimal,
        new_balance: Decimal,
        change: Decimal,
        reason: str
    ) -> None:
        """
        Сохраняет изменение баланса в историю.

        Args:
            exchange: Имя биржи
            currency: Код валюты
            old_balance: Предыдущий баланс
            new_balance: Новый баланс
            change: Изменение баланса
            reason: Причина изменения
        """
        self._balance_history.append({
            "timestamp": int(time.time() * 1000),
            "type": "change",
            "exchange": exchange,
            "currency": currency,
            "old_balance": float(old_balance),
            "new_balance": float(new_balance),
            "change": float(change),
            "reason": reason
        })

    def _save_balance_snapshot(self, reason: str) -> None:
        """
        Сохраняет снимок всех балансов в историю.

        Args:
            reason: Причина создания снимка
        """
        # Сериализуем все балансы
        balances_snapshot = {}
        for exchange, currencies in self._balances.items():
            balances_snapshot[exchange] = {currency: float(amount) for currency, amount in currencies.items()}

        # Рассчитываем общий баланс в базовой валюте
        total_balance = float(self.get_total_balance_in_base_currency())

        self._balance_history.append({
            "timestamp": int(time.time() * 1000),
            "type": "snapshot",
            "balances": balances_snapshot,
            "total_balance": total_balance,
            "base_currency": self.base_currency,
            "reason": reason
        })

    def get_balance_history(self) -> List[Dict[str, Any]]:
        """
        Возвращает историю изменений балансов.

        Returns:
            Список записей истории балансов
        """
        return self._balance_history

    def export_balance_history(self, file_path: str) -> None:
        """
        Экспортирует историю балансов в JSON файл.

        Args:
            file_path: Путь к файлу для сохранения
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self._balance_history, f, indent=2)
            self.logger.info(f"Balance history exported to {file_path}")
        except Exception as e:
            self.logger.error(f"Error exporting balance history: {e}")

    def get_profit_report(self) -> Dict[str, Any]:
        """
        Генерирует отчет о прибыли на основе истории балансов.

        Returns:
            Словарь с отчетом о прибыли
        """
        # Находим начальный и текущий снимок балансов
        initial_snapshot = None
        current_snapshot = None

        for entry in self._balance_history:
            if entry.get("type") == "snapshot":
                if initial_snapshot is None or entry.get("timestamp", 0) < initial_snapshot.get("timestamp", float('inf')):
                    initial_snapshot = entry

                if current_snapshot is None or entry.get("timestamp", 0) > current_snapshot.get("timestamp", 0):
                    current_snapshot = entry

        if not initial_snapshot or not current_snapshot:
            return {
                "status": "error",
                "error": "Not enough balance snapshots for profit calculation"
            }

        # Рассчитываем изменение общего баланса
        initial_total = initial_snapshot.get("total_balance", 0)
        current_total = current_snapshot.get("total_balance", 0)
        absolute_profit = current_total - initial_total
        relative_profit = (absolute_profit / initial_total * 100) if initial_total > 0 else 0

        # Формируем отчет
        return {
            "status": "success",
            "start_time": initial_snapshot.get("timestamp"),
            "end_time": current_snapshot.get("timestamp"),
            "initial_balance": initial_total,
            "current_balance": current_total,
            "absolute_profit": absolute_profit,
            "relative_profit": relative_profit,
            "base_currency": self.base_currency,
            "initial_balances": initial_snapshot.get("balances", {}),
            "current_balances": current_snapshot.get("balances", {}),
            "duration_ms": current_snapshot.get("timestamp", 0) - initial_snapshot.get("timestamp", 0)
        }