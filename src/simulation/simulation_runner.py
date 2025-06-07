"""
Запуск симуляции арбитражных сделок.
Предоставляет интерфейс для запуска и анализа результатов симуляции торговли.
"""

import os
import sys
import logging
import asyncio
import json
import time
import argparse
from decimal import Decimal
from typing import Dict, List, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Добавляем корневую директорию проекта в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.simulation.simulation import TradingSimulator
from src.simulation.market_data_simulator import MarketDataSimulator
from src.arbitrage_engine.market_strategy import DirectArbitrageStrategy, TriangularArbitrageStrategy


class SimulationRunner:
    """
    Класс для запуска и управления симуляцией арбитражной торговли.
    """
    
    def __init__(
        self,
        exchanges: List[str],
        symbols: List[str],
        initial_balances: Dict[str, Dict[str, float]],
        exchange_fees: Dict[str, float],
        min_profit_percentage: float = 0.5,
        output_dir: str = "results",
        volatility: float = 0.002,
        arbitrage_frequency: float = 0.3,
        max_arbitrage_spread: float = 0.015,
        max_active_trades: int = 20,
        max_trade_duration_ms: int = 30000
    ):
        """
        Инициализирует runner симуляции.
        
        Args:
            exchanges: Список бирж для симуляции
            symbols: Список торговых пар для симуляции
            initial_balances: Начальные балансы на биржах
            exchange_fees: Комиссии бирж
            min_profit_percentage: Минимальный процент прибыли для сделки
            output_dir: Директория для сохранения результатов
            volatility: Волатильность рынка
            arbitrage_frequency: Частота арбитражных возможностей
            max_arbitrage_spread: Максимальный спред арбитража
            max_active_trades: Максимальное число активных сделок
            max_trade_duration_ms: Максимальная длительность сделки в мс
        """
        self.logger = logging.getLogger(__name__)
        
        self.exchanges = exchanges
        self.symbols = symbols
        self.initial_balances = initial_balances
        self.exchange_fees = exchange_fees
        self.min_profit_percentage = min_profit_percentage
        self.output_dir = output_dir
        self.volatility = volatility
        self.arbitrage_frequency = arbitrage_frequency
        self.max_arbitrage_spread = max_arbitrage_spread
        self.max_active_trades = max_active_trades
        self.max_trade_duration_ms = max_trade_duration_ms
        
        # Создаем директорию для результатов, если не существует
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Инициализируем симулятор рыночных данных и симулятор торговли
        self._init_simulators()
        
        self.logger.info(
            f"SimulationRunner initialized with "
            f"{len(exchanges)} exchanges, {len(symbols)} symbols, "
            f"min_profit={min_profit_percentage}%"
        )
    
    def _init_simulators(self):
        """
        Инициализирует симуляторы рыночных данных и торговли.
        """
        # Задаем начальные цены для симуляции
        initial_prices = {
            "BTC/USDT": 65000.0,
            "ETH/USDT": 3200.0,
            "BNB/USDT": 580.0,
            "SOL/USDT": 160.0,
            "XRP/USDT": 0.6,
            "ADA/USDT": 0.45,
            "DOGE/USDT": 0.15,
            "SHIB/USDT": 0.00002,
            "DOT/USDT": 5.8,
            "AVAX/USDT": 35.0,
        }
        
        # Инициализируем недостающие пары средними ценами
        for symbol in self.symbols:
            if symbol not in initial_prices:
                initial_prices[symbol] = 100.0  # Значение по умолчанию
        
        # Инициализируем симулятор рыночных данных
        self.market_simulator = MarketDataSimulator(
            exchanges=self.exchanges,
            symbols=self.symbols,
            initial_prices=initial_prices,
            volatility=self.volatility,
            arbitrage_frequency=self.arbitrage_frequency,
            max_arbitrage_spread=self.max_arbitrage_spread,
            tick_interval_ms=500
        )
        
        # Инициализируем симулятор торговли
        self.trading_simulator = TradingSimulator(
            initial_balances=self.initial_balances,
            exchange_fees=self.exchange_fees,
            min_profit_percentage=self.min_profit_percentage,
            max_active_trades=self.max_active_trades,
            max_trade_duration_ms=self.max_trade_duration_ms,
            emergency_stop_loss_percentage=-0.5  # -0.5%
        )
        
        # Добавляем стратегии
        self.trading_simulator.add_strategy(
            DirectArbitrageStrategy(
                min_profit_percentage=self.min_profit_percentage,
                max_order_value=1000.0,
                min_order_value=10.0,
                max_execution_time_ms=10000,
                emergency_close_threshold=-0.5
            )
        )
        
        self.trading_simulator.add_strategy(
            TriangularArbitrageStrategy(
                min_profit_percentage=self.min_profit_percentage * 0.8,  # Немного снижаем требования
                max_order_value=500.0,
                min_order_value=10.0,
                max_execution_time_ms=5000
            )
        )
    
    async def run(self, duration_seconds: int) -> Dict[str, Any]:
        """
        Запускает симуляцию на заданное время.
        
        Args:
            duration_seconds: Длительность симуляции в секундах
            
        Returns:
            Результаты симуляции
        """
        self.logger.info(f"Starting simulation for {duration_seconds} seconds...")
        start_time = time.time()
        
        results = await self.trading_simulator.run_simulation(
            self.market_simulator, 
            duration_seconds
        )
        
        simulation_time = time.time() - start_time
        self.logger.info(f"Simulation completed in {simulation_time:.2f} seconds")
        
        # Сохраняем результаты
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """
        Сохраняет результаты симуляции в файлы и генерирует отчеты.
        
        Args:
            results: Результаты симуляции
        """
        # Генерируем временную метку для имен файлов
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохраняем результаты в JSON файл
        result_file = os.path.join(self.output_dir, f"simulation_results_{timestamp}.json")
        
        # Преобразуем Decimal в float для сериализации в JSON
        results_serializable = json.loads(
            json.dumps(results, default=lambda x: float(x) if isinstance(x, Decimal) else str(x))
        )
        
        with open(result_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        self.logger.info(f"Results saved to {result_file}")
        
        # Генерируем отчет и графики
        self._generate_report(results, timestamp)
    
    def _generate_report(self, results: Dict[str, Any], timestamp: str) -> None:
        """
        Генерирует отчет и графики на основе результатов симуляции.
        
        Args:
            results: Результаты симуляции
            timestamp: Временная метка для имен файлов
        """
        # Извлекаем данные
        trade_summary = results.get("trade_summary", {})
        balance_summary = results.get("balance_summary", {})
        stats = trade_summary.get("stats", {})
        completed_trades = trade_summary.get("completed_trades", [])
        
        # Создаем текстовый отчет
        report_lines = [
            "=================================================",
            " ОТЧЕТ ПО СИМУЛЯЦИИ АРБИТРАЖНЫХ СДЕЛОК",
            "=================================================",
            f"Дата проведения: {datetime.fromtimestamp(stats.get('start_time', time.time())).strftime('%Y-%m-%d %H:%M:%S')}",
            f"Длительность: {results.get('simulation_stats', {}).get('duration_seconds', 0):.2f} секунд",
            "-------------------------------------------------",
            f"Всего сделок: {stats.get('total_trades', 0)}",
            f"Успешных сделок: {stats.get('successful_trades', 0)} ({stats.get('successful_trades', 0) / max(1, stats.get('total_trades', 1)) * 100:.2f}%)",
            f"Неудачных сделок: {stats.get('failed_trades', 0)}",
            f"Общая прибыль: {stats.get('total_profit', 0):.2f} USDT",
            f"Общий объем торгов: {stats.get('total_volume_traded', 0):.2f} USDT",
            f"Средняя прибыль на сделку: {stats.get('total_profit', 0) / max(1, stats.get('total_trades', 1)):.2f} USDT",
            "-------------------------------------------------",
            "Балансы по биржам:",
        ]
        
        # Добавляем данные о балансах
        for exchange, currencies in balance_summary.get("balances", {}).items():
            report_lines.append(f"  {exchange}:")
            for currency, amount in currencies.items():
                report_lines.append(f"    {currency}: {float(amount):.6f}")
        
        report_lines.extend([
            "=================================================",
            "СТАТИСТИКА ПО СДЕЛКАМ",
            "=================================================",
        ])
        
        # Добавляем статистику по топ-5 самым прибыльным сделкам
        if completed_trades:
            profitable_trades = sorted(
                completed_trades, 
                key=lambda x: float(x.get("actual_profit", 0)), 
                reverse=True
            )
            report_lines.append("Топ-5 самых прибыльных сделок:")
            for i, trade in enumerate(profitable_trades[:5]):
                report_lines.append(
                    f"  {i + 1}. {trade.get('symbol')}: {trade.get('buy_exchange')} -> {trade.get('sell_exchange')}, "
                    f"прибыль: {float(trade.get('actual_profit', 0)):.2f} USDT "
                    f"({float(trade.get('actual_profit_percentage', 0)):.2f}%)"
                )
        
        # Сохраняем отчет
        report_file = os.path.join(self.output_dir, f"simulation_report_{timestamp}.txt")
        with open(report_file, 'w') as f:
            f.write("\n".join(report_lines))
        
        self.logger.info(f"Report saved to {report_file}")
        
        # Создаем графики если есть хотя бы несколько сделок
        if len(completed_trades) > 5:
            self._plot_trades(completed_trades, timestamp)
    
    def _plot_trades(self, completed_trades: List[Dict[str, Any]], timestamp: str) -> None:
        """
        Создает графики для анализа результатов симуляции.
        
        Args:
            completed_trades: Список завершенных сделок
            timestamp: Временная метка для имен файлов
        """
        # Создаем директорию для графиков если не существует
        plots_dir = os.path.join(self.output_dir, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # Преобразуем данные в формат для графиков
        trade_times = [float(trade.get("open_timestamp", 0)) for trade in completed_trades]
        profits = [float(trade.get("actual_profit", 0)) for trade in completed_trades]
        profit_percentages = [float(trade.get("actual_profit_percentage", 0)) for trade in completed_trades]
        durations = [float(trade.get("duration_ms", 0)) / 1000.0 for trade in completed_trades]  # в секундах
        
        # Преобразуем timestamp в читаемые даты
        try:
            trade_dates = [datetime.fromtimestamp(ts / 1000) for ts in trade_times]
        except:
            trade_dates = [datetime.now() for _ in trade_times]  # Fallback
        
        # Расчет кумулятивной прибыли
        cumulative_profit = np.cumsum(profits)
        
        # 1. График прибыли от сделок
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(profits)), profits, alpha=0.7)
        plt.xlabel('Номер сделки')
        plt.ylabel('Прибыль (USDT)')
        plt.title('Прибыль от каждой сделки')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, f"trades_profit_{timestamp}.png"))
        plt.close()
        
        # 2. График кумулятивной прибыли
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(cumulative_profit)), cumulative_profit, 'g-', linewidth=2)
        plt.xlabel('Количество сделок')
        plt.ylabel('Кумулятивная прибыль (USDT)')
        plt.title('Кумулятивная прибыль')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, f"cumulative_profit_{timestamp}.png"))
        plt.close()
        
        # 3. График процентной прибыли
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(profit_percentages)), profit_percentages, alpha=0.7, color='orange')
        plt.xlabel('Номер сделки')
        plt.ylabel('Процент прибыли (%)')
        plt.title('Процент прибыли от каждой сделки')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, f"profit_percentage_{timestamp}.png"))
        plt.close()
        
        # 4. График длительности сделок
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(durations)), durations, alpha=0.7, color='purple')
        plt.xlabel('Номер сделки')
        plt.ylabel('Длительность (секунды)')
        plt.title('Длительность каждой сделки')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, f"trade_durations_{timestamp}.png"))
        plt.close()
        
        # 5. Гистограмма распределения прибыли
        plt.figure(figsize=(12, 6))
        plt.hist(profits, bins=20, alpha=0.7, color='blue')
        plt.xlabel('Прибыль (USDT)')
        plt.ylabel('Количество сделок')
        plt.title('Распределение прибыли')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, f"profit_distribution_{timestamp}.png"))
        plt.close()
        
        self.logger.info(f"Plots saved to {plots_dir}")


def get_default_config():
    """
    Возвращает конфигурацию по умолчанию для симуляции.
    
    Returns:
        Кортеж из параметров по умолчанию
    """
    # Биржи для симуляции
    exchanges = ["binance", "kucoin", "okx"]
    
    # Торговые пары для симуляции
    symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]
    
    # Начальные балансы на биржах
    initial_balances = {
        "binance": {
            "USDT": 5000.0,
            "BTC": 0.05,
            "ETH": 1.0,
            "BNB": 10.0,
            "SOL": 50.0
        },
        "kucoin": {
            "USDT": 5000.0,
            "BTC": 0.05,
            "ETH": 1.0,
            "BNB": 10.0,
            "SOL": 50.0
        },
        "okx": {
            "USDT": 5000.0,
            "BTC": 0.05,
            "ETH": 1.0,
            "BNB": 10.0,
            "SOL": 50.0
        }
    }
    
    # Комиссии бирж
    exchange_fees = {
        "binance": 0.1,  # 0.1%
        "kucoin": 0.1,
        "okx": 0.1
    }
    
    return exchanges, symbols, initial_balances, exchange_fees


async def run_simulation_with_params(
    duration_seconds: int = 300,
    min_profit_percentage: float = 0.5,
    output_dir: str = "results",
    exchanges: Optional[List[str]] = None,
    symbols: Optional[List[str]] = None,
    initial_balances: Optional[Dict[str, Dict[str, float]]] = None,
    exchange_fees: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Запускает симуляцию с заданными параметрами.
    
    Args:
        duration_seconds: Длительность симуляции в секундах
        min_profit_percentage: Минимальный процент прибыли для сделки
        output_dir: Директория для сохранения результатов
        exchanges: Список бирж для симуляции
        symbols: Список торговых пар для симуляции
        initial_balances: Начальные балансы на биржах
        exchange_fees: Комиссии бирж
        
    Returns:
        Результаты симуляции
    """
    # Если параметры не указаны, используем значения по умолчанию
    default_exchanges, default_symbols, default_balances, default_fees = get_default_config()
    
    exchanges = exchanges or default_exchanges
    symbols = symbols or default_symbols
    initial_balances = initial_balances or default_balances
    exchange_fees = exchange_fees or default_fees
    
    # Создаем и настраиваем runner симуляции
    runner = SimulationRunner(
        exchanges=exchanges,
        symbols=symbols,
        initial_balances=initial_balances,
        exchange_fees=exchange_fees,
        min_profit_percentage=min_profit_percentage,
        output_dir=output_dir
    )
    
    # Запускаем симуляцию
    results = await runner.run(duration_seconds)
    
    return results


async def main():
    """Основная функция для запуска симуляции через командную строку."""
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Simulation runner for crypto arbitrage bot")
    parser.add_argument(
        "--duration", type=int, default=300,
        help="Duration of simulation in seconds (default: 300)"
    )
    parser.add_argument(
        "--min-profit", type=float, default=0.5,
        help="Minimum profit percentage for trades (default: 0.5)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory for saving results (default: 'results')"
    )
    args = parser.parse_args()
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/simulation.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting simulation with parameters:")
    logger.info(f"Duration: {args.duration} seconds")
    logger.info(f"Min profit: {args.min_profit}%")
    
    # Запуск симуляции
    await run_simulation_with_params(
        duration_seconds=args.duration,
        min_profit_percentage=args.min_profit,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    asyncio.run(main())
