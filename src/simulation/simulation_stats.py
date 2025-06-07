"""
Модуль для сбора и анализа статистики по симуляциям арбитражной торговли.
Позволяет собирать, сравнивать и анализировать данные из нескольких симуляций.
"""

import os
import sys
import json
import logging
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Добавляем корневую директорию проекта в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def setup_logging():
    """Настройка логирования."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


class SimulationStats:
    """
    Класс для сбора статистики по текущей симуляции арбитражной торговли.
    Отличается от SimulationStatsCollector тем, что работает с данными одной текущей симуляции,
    а не с сохраненными результатами нескольких симуляций.
    """

    def __init__(self):
        """Инициализация объекта статистики."""
        self.logger = logging.getLogger(__name__)
        self.start_time = datetime.now()
        # Общие показатели
        self.opportunities_found = 0
        self.trades_executed = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.total_profit = 0.0
        self.total_volume = 0.0
        # Для хранения промежуточных данных
        self.profit_history = []
        self.volume_history = []
        self.timestamp_history = []

    def reset(self):
        """Сбрасывает всю статистику."""
        self.__init__()

    def record_opportunity(self, opportunity, is_executed=False):
        """Записывает информацию о найденной возможности."""
        self.opportunities_found += 1
        if is_executed:
            self.trades_executed += 1

    def record_trade_result(self, trade_id, profit, volume, is_successful=True):
        """Записывает результат выполненной сделки."""
        if is_successful:
            self.successful_trades += 1
        else:
            self.failed_trades += 1

        self.total_profit += profit
        self.total_volume += volume

        # Сохраняем историю для анализа тренда
        self.profit_history.append(profit)
        self.volume_history.append(volume)
        self.timestamp_history.append(datetime.now())

    def get_duration(self):
        """Возвращает продолжительность симуляции в секундах."""
        return (datetime.now() - self.start_time).total_seconds()

    def get_stats_summary(self):
        """Возвращает сводку статистики в виде словаря."""
        duration = self.get_duration()
        total_trades = self.successful_trades + self.failed_trades

        return {
            "duration_seconds": duration,
            "opportunities_found": self.opportunities_found,
            "trades_executed": self.trades_executed,
            "total_trades": total_trades,
            "successful_trades": self.successful_trades,
            "failed_trades": self.failed_trades,
            "success_rate": (self.successful_trades / max(1, total_trades)) * 100,
            "total_profit": self.total_profit,
            "total_volume": self.total_volume,
            "profit_per_second": self.total_profit / max(1, duration),
            "average_profit_per_trade": self.total_profit / max(1, total_trades)
        }

    def print_summary(self):
        """Выводит в лог краткую сводку статистики."""
        stats = self.get_stats_summary()
        summary = (
            f"\n{'='*50}\n"
            f"SIMULATION STATISTICS\n"
            f"{'='*50}\n"
            f"Duration: {stats['duration_seconds']:.1f} seconds\n"
            f"Opportunities found: {stats['opportunities_found']}\n"
            f"Trades executed: {stats['trades_executed']}\n"
            f"Successful trades: {stats['successful_trades']}\n"
            f"Failed trades: {stats['failed_trades']}\n"
            f"Success rate: {stats['success_rate']:.2f}%\n"
            f"Total profit: {stats['total_profit']:.4f}\n"
            f"Average profit per trade: {stats['average_profit_per_trade']:.4f}\n"
            f"Profit per second: {stats['profit_per_second']:.6f}\n"
            f"{'='*50}"
        )
        self.logger.info(summary)
        return summary

    def save_to_file(self, filename):
        """Сохраняет результаты симуляции в JSON-файл."""
        data = {
            "simulation_stats": self.get_stats_summary(),
            "timestamp": datetime.now().isoformat(),
            "profit_history": self.profit_history,
            "volume_history": self.volume_history,
            "timestamp_history": [ts.isoformat() for ts in self.timestamp_history]
        }

        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Statistics saved to {filename}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save statistics to {filename}: {e}")
            return False


class SimulationStatsCollector:
    """
    Класс для сбора и анализа статистики по симуляциям.
    """

    def __init__(self):
        """Инициализация коллектора статистики."""
        self.logger = logging.getLogger(__name__)
        self.simulations = []
        self.simulation_data = {}

    def load_simulation_results(self, file_path: str, name: Optional[str] = None) -> bool:
        """
        Загружает результаты симуляции из файла.

        Args:
            file_path: Путь к файлу с результатами
            name: Имя симуляции (если None, используется имя файла)

        Returns:
            True если загрузка успешна, иначе False
        """
        try:
            # Определяем имя симуляции
            if name is None:
                name = os.path.basename(file_path).split('.')[0]

            # Загружаем данные
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Добавляем в список симуляций
            self.simulations.append(name)
            self.simulation_data[name] = data

            self.logger.info(f"Loaded simulation results for '{name}' from {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading simulation results from {file_path}: {e}")
            return False

    def load_multiple_simulations(self, directory: str, pattern: str = "*.json") -> int:
        """
        Загружает результаты симуляций из всех подходящих файлов в директории.

        Args:
            directory: Путь к директории с результатами
            pattern: Шаблон для поиска файлов

        Returns:
            Количество успешно загруженных файлов
        """
        import glob

        # Находим все подходящие файлы
        file_paths = glob.glob(os.path.join(directory, pattern))

        # Загружаем данные из каждого файла
        successful_loads = 0
        for file_path in file_paths:
            if self.load_simulation_results(file_path):
                successful_loads += 1

        self.logger.info(f"Loaded {successful_loads} simulation results from {directory}")
        return successful_loads

    def get_simulation_names(self) -> List[str]:
        """
        Возвращает список имен загруженных симуляций.

        Returns:
            Список имен симуляций
        """
        return self.simulations

    def get_simulation_data(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает данные указанной симуляции.

        Args:
            name: Имя симуляции

        Returns:
            Словарь с данными симуляции или None если симуляция не найдена
        """
        return self.simulation_data.get(name)

    def get_simulation_summary(self, name: str) -> Dict[str, Any]:
        """
        Возвращает сводку по указанной симуляции.

        Args:
            name: Имя симуляции

        Returns:
            Словарь со сводкой по симуляции
        """
        if name not in self.simulation_data:
            return {"error": f"Simulation '{name}' not found"}

        data = self.simulation_data[name]

        # Извлекаем общую статистику
        stats = data.get("trade_summary", {}).get("stats", {})
        simulation_stats = data.get("simulation_stats", {})

        # Формируем сводку
        return {
            "name": name,
            "total_trades": stats.get("total_trades", 0),
            "successful_trades": stats.get("successful_trades", 0),
            "failed_trades": stats.get("failed_trades", 0),
            "success_rate": (stats.get("successful_trades", 0) / max(1, stats.get("total_trades", 1))) * 100,
            "total_profit": stats.get("total_profit", 0),
            "total_volume_traded": stats.get("total_volume_traded", 0),
            "avg_profit_per_trade": stats.get("total_profit", 0) / max(1, stats.get("total_trades", 1)),
            "duration_seconds": simulation_stats.get("duration_seconds", 0),
            "iterations": simulation_stats.get("iterations", 0),
            "timestamp": simulation_stats.get("start_time", 0)
        }

    def get_all_simulation_summaries(self) -> List[Dict[str, Any]]:
        """
        Возвращает сводки по всем загруженным симуляциям.

        Returns:
            Список словарей со сводками по симуляциям
        """
        return [self.get_simulation_summary(name) for name in self.simulations]

    def compare_simulations(self, metrics: List[str] = None) -> pd.DataFrame:
        """
        Сравнивает указанные метрики по всем загруженным симуляциям.

        Args:
            metrics: Список метрик для сравнения (если None, используются все доступные)

        Returns:
            DataFrame с сравнением симуляций
        """
        # Получаем сводки по всем симуляциям
        summaries = self.get_all_simulation_summaries()

        # Если не указаны метрики, используем все доступные
        if metrics is None and summaries:
            metrics = list(summaries[0].keys())
            # Исключаем некоторые ненужные поля
            for field in ["name", "timestamp"]:
                if field in metrics:
                    metrics.remove(field)

        # Создаем DataFrame
        df = pd.DataFrame(summaries)

        # Сортируем по времени выполнения
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")

        return df

    def plot_comparison(self, metric: str, output_file: Optional[str] = None) -> None:
        """
        Строит график сравнения указанной метрики по всем симуляциям.

        Args:
            metric: Метрика для сравнения
            output_file: Путь для сохранения графика (если None, график отображается)
        """
        # Получаем данные для сравнения
        df = self.compare_simulations()

        if metric not in df.columns:
            self.logger.error(f"Metric '{metric}' not found in simulation data")
            return

        # Создаем график
        plt.figure(figsize=(12, 6))

        # Создаем столбчатую диаграмму
        ax = sns.barplot(x="name", y=metric, data=df)

        # Добавляем подписи
        for i, value in enumerate(df[metric]):
            ax.text(i, value, f"{value:.2f}", ha="center", va="bottom")

        # Настраиваем график
        plt.title(f"Comparison of {metric} across simulations")
        plt.xlabel("Simulation")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Сохраняем или отображаем график
        if output_file:
            plt.savefig(output_file)
            self.logger.info(f"Saved comparison plot to {output_file}")
        else:
            plt.show()

    def plot_all_comparisons(self, output_dir: str) -> None:
        """
        Строит графики сравнения всех метрик по всем симуляциям.

        Args:
            output_dir: Директория для сохранения графиков
        """
        # Создаем директорию, если не существует
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Получаем данные для сравнения
        df = self.compare_simulations()

        # Определяем метрики для сравнения
        metrics = [col for col in df.columns if col not in ["name", "timestamp"]]

        # Строим график для каждой метрики
        for metric in metrics:
            output_file = os.path.join(output_dir, f"comparison_{metric}.png")
            self.plot_comparison(metric, output_file)

    def generate_comparison_report(self, output_file: str) -> None:
        """
        Генерирует отчет со сравнением всех симуляций.

        Args:
            output_file: Путь к файлу для сохранения отчета
        """
        # Получаем данные для сравнения
        df = self.compare_simulations()

        # Форматируем DataFrame для отчета
        report_df = df.copy()

        # Форматируем числовые колонки
        for col in report_df.columns:
            if col in ["total_profit", "total_volume", "avg_profit_per_trade"]:
                report_df[col] = report_df[col].map(lambda x: f"{x:.2f}")
            elif col in ["success_rate"]:
                report_df[col] = report_df[col].map(lambda x: f"{x:.2f}%")
            elif col in ["duration_seconds"]:
                report_df[col] = report_df[col].map(lambda x: f"{x:.2f}s")

        # Создаем HTML-отчет
        html = f"""
        <html>
        <head>
            <title>Simulation Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .timestamp {{ font-size: 0.8em; color: #666; }}
            </style>
        </head>
        <body>
            <h1>Simulation Comparison Report</h1>
            <p class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Comparison Table</h2>
            {report_df.to_html(index=False)}
            
            <h2>Best Results</h2>
            <ul>
        """

        # Добавляем информацию о лучших результатах
        if not df.empty:
            # Наибольшая общая прибыль
            if "total_profit" in df.columns:
                best_profit = df.loc[df["total_profit"].idxmax()]
                html += f"<li>Highest total profit: <strong>{best_profit['total_profit']:.2f}</strong> in simulation <strong>{best_profit['name']}</strong></li>"

            # Наивысший процент успешных сделок
            if "success_rate" in df.columns:
                best_success = df.loc[df["success_rate"].idxmax()]
                html += f"<li>Highest success rate: <strong>{best_success['success_rate']:.2f}%</strong> in simulation <strong>{best_success['name']}</strong></li>"

            # Наивысшая средняя прибыль на сделку
            if "avg_profit_per_trade" in df.columns:
                best_avg_profit = df.loc[df["avg_profit_per_trade"].idxmax()]
                html += f"<li>Highest average profit per trade: <strong>{best_avg_profit['avg_profit_per_trade']:.2f}</strong> in simulation <strong>{best_avg_profit['name']}</strong></li>"

        html += """
            </ul>
        </body>
        </html>
        """

        # Сохраняем отчет
        with open(output_file, 'w') as f:
            f.write(html)

        self.logger.info(f"Generated comparison report saved to {output_file}")

    def analyze_simulation_detail(self, name: str) -> Dict[str, Any]:
        """
        Выполняет детальный анализ указанной симуляции.

        Args:
            name: Имя симуляции

        Returns:
            Словарь с результатами анализа
        """
        if name not in self.simulation_data:
            return {"error": f"Simulation '{name}' not found"}

        data = self.simulation_data[name]

        # Анализируем статистику по сделкам, обменам, символам и т.д.
        trade_stats = data.get("trade_stats", {})
        top_exchanges = data.get("top_exchanges", [])
        top_symbols = data.get("top_symbols", [])
        top_strategies = data.get("top_strategies", [])

        # Формируем анализ
        analysis = {
            "trade_stats": trade_stats,
            "exchange_stats": top_exchanges,
            "symbol_stats": top_symbols,
            "strategy_stats": top_strategies,
            "simulation_duration": data.get("simulation_duration", 0)
        }

        return analysis

    def plot_detailed_analysis(self, name: str, output_dir: str) -> None:
        """
        Строит графики детального анализа указанной симуляции.

        Args:
            name: Имя симуляции
            output_dir: Директория для сохранения графиков
        """
        if name not in self.simulation_data:
            self.logger.error(f"Simulation '{name}' not found")
            return

        # Создаем директорию, если не существует
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Получаем данные для анализа
        analysis = self.analyze_simulation_detail(name)

        # Если нет данных для анализа, выходим
        if "error" in analysis:
            self.logger.error(analysis["error"])
            return

        # График распределения прибыли по биржам
        if "exchange_stats" in analysis and analysis["exchange_stats"]:
            plt.figure(figsize=(12, 6))
            exchange_data = pd.DataFrame(analysis["exchange_stats"])
            
            if not exchange_data.empty and "name" in exchange_data.columns and "profit" in exchange_data.columns:
                exchange_data = exchange_data.sort_values("profit", ascending=False)
                sns.barplot(x="name", y="profit", data=exchange_data)
                plt.title(f"Profit by Exchange Pair - {name}")
                plt.xlabel("Exchange Pair")
                plt.ylabel("Profit")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{name}_exchange_profit.png"))
                plt.close()

        # График распределения прибыли по торговым парам
        if "symbol_stats" in analysis and analysis["symbol_stats"]:
            plt.figure(figsize=(12, 6))
            symbol_data = pd.DataFrame(analysis["symbol_stats"])
            
            if not symbol_data.empty and "name" in symbol_data.columns and "profit" in symbol_data.columns:
                symbol_data = symbol_data.sort_values("profit", ascending=False)
                sns.barplot(x="name", y="profit", data=symbol_data)
                plt.title(f"Profit by Trading Pair - {name}")
                plt.xlabel("Trading Pair")
                plt.ylabel("Profit")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{name}_symbol_profit.png"))
                plt.close()

        # График распределения прибыли по стратегиям
        if "strategy_stats" in analysis and analysis["strategy_stats"]:
            plt.figure(figsize=(12, 6))
            strategy_data = pd.DataFrame(analysis["strategy_stats"])
            
            if not strategy_data.empty and "name" in strategy_data.columns and "profit" in strategy_data.columns:
                strategy_data = strategy_data.sort_values("profit", ascending=False)
                sns.barplot(x="name", y="profit", data=strategy_data)
                plt.title(f"Profit by Strategy - {name}")
                plt.xlabel("Strategy")
                plt.ylabel("Profit")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{name}_strategy_profit.png"))
                plt.close()

        # Графики временных рядов (если есть в данных)
        if "time_series_data" in data and data["time_series_data"]:
            time_series = data["time_series_data"]
            
            if "timestamps" in time_series and "cumulative_profit" in time_series:
                plt.figure(figsize=(12, 6))
                plt.plot(time_series["timestamps"], time_series["cumulative_profit"])
                plt.title(f"Cumulative Profit Over Time - {name}")
                plt.xlabel("Time")
                plt.ylabel("Cumulative Profit")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{name}_cumulative_profit.png"))
                plt.close()

        self.logger.info(f"Generated {name} analysis plots in {output_dir}")# Получаем данные для сравнения
        df = self.compare_simulations()

        # Определяем метрики для сравнения
        metrics = [col for col in df.columns if col not in ["name", "timestamp"]]

        # Строим график для каждой метрики
        for metric in metrics:
            output_file = os.path.join(output_dir, f"comparison_{metric}.png")
            self.plot_comparison(metric, output_file)

    def generate_comparison_report(self, output_file: str) -> None:
        """
        Генерирует отчет со сравнением всех симуляций.

        Args:
            output_file: Путь к файлу для сохранения отчета
        """
        # Получаем данные для сравнения
        df = self.compare_simulations()

        # Форматируем DataFrame для отчета
        report_df = df.copy()

        # Форматируем числовые колонки
        for col in report_df.columns:
            if col in ["total_profit", "total_volume_traded", "avg_profit_per_trade"]:
                report_df[col] = report_df[col].map(lambda x: f"{x:.2f}")
            elif col in ["success_rate"]:
                report_df[col] = report_df[col].map(lambda x: f"{x:.2f}%")
            elif col in ["duration_seconds"]:
                report_df[col] = report_df[col].map(lambda x: f"{x:.2f}s")

        # Создаем HTML-отчет
        html = f"""
        <html>
        <head>
            <title>Simulation Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .timestamp {{ font-size: 0.8em; color: #666; }}
            </style>
        </head>
        <body>
            <h1>Simulation Comparison Report</h1>
            <p class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Comparison Table</h2>
            {report_df.to_html(index=False)}
            
            <h2>Best Results</h2>
            <ul>
        """

        # Добавляем информацию о лучших результатах
        if not df.empty:
            # Наибольшая общая прибыль
            if "total_profit" in df.columns:
                best_profit = df.loc[df["total_profit"].idxmax()]
                html += f"<li>Highest total profit: <strong>{best_profit['total_profit']:.2f}</strong> in simulation <strong>{best_profit['name']}</strong></li>"

            # Наивысший процент успешных сделок
            if "success_rate" in df.columns:
                best_success = df.loc[df["success_rate"].idxmax()]
                html += f"<li>Highest success rate: <strong>{best_success['success_rate']:.2f}%</strong> in simulation <strong>{best_success['name']}</strong></li>"

            # Наивысшая средняя прибыль на сделку
            if "avg_profit_per_trade" in df.columns:
                best_avg_profit = df.loc[df["avg_profit_per_trade"].idxmax()]
                html += f"<li>Highest average profit per trade: <strong>{best_avg_profit['avg_profit_per_trade']:.2f}</strong> in simulation <strong>{best_avg_profit['name']}</strong></li>"

        html += """
            </ul>
        </body>
        </html>
        """

        # Сохраняем отчет
        with open(output_file, 'w') as f:
            f.write(html)

        self.logger.info(f"Generated comparison report saved to {output_file}")

    def analyze_simulation_detail(self, name: str) -> Dict[str, Any]:
        """
        Выполняет детальный анализ указанной симуляции.

        Args:
            name: Имя симуляции

        Returns:
            Словарь с результатами анализа
        """
        if name not in self.simulation_data:
            return {"error": f"Simulation '{name}' not found"}

        data = self.simulation_data[name]

        # Извлекаем данные о сделках
        trade_data = data.get("trade_summary", {}).get("completed_trades", [])

        if not trade_data:
            return {"error": "No trade data found in simulation"}

        # Создаем DataFrame из данных о сделках
        df = pd.DataFrame(trade_data)

        # Преобразуем timestamp в datetime
        if 'open_timestamp' in df.columns:
            df['open_time'] = pd.to_datetime(df['open_timestamp'], unit='ms')
        if 'close_timestamp' in df.columns:
            df['close_time'] = pd.to_datetime(df['close_timestamp'], unit='ms')

        # Добавляем флаг прибыльности
        if 'actual_profit' in df.columns:
            df['is_profitable'] = df['actual_profit'] > 0

        # Анализируем данные по различным аспектам
        analysis = {}

        # Анализ распределения прибыли
        if 'actual_profit' in df.columns:
            analysis["profit_distribution"] = {
                "mean": float(df['actual_profit'].mean()),
                "median": float(df['actual_profit'].median()),
                "std": float(df['actual_profit'].std()),
                "min": float(df['actual_profit'].min()),
                "max": float(df['actual_profit'].max()),
                "positive_count": int(df[df['actual_profit'] > 0].shape[0]),
                "negative_count": int(df[df['actual_profit'] < 0].shape[0]),
                "zero_count": int(df[df['actual_profit'] == 0].shape[0])
            }

        # Анализ по торговым парам
        if 'symbol' in df.columns and 'actual_profit' in df.columns:
            symbol_stats = df.groupby('symbol')['actual_profit'].agg(['count', 'sum', 'mean', 'std']).reset_index()
            analysis["symbol_stats"] = symbol_stats.to_dict(orient='records')

        # Анализ по биржам
        if 'buy_exchange' in df.columns and 'sell_exchange' in df.columns and 'actual_profit' in df.columns:
            # Создаем колонку для пары бирж
            df['exchange_pair'] = df['buy_exchange'] + "_to_" + df['sell_exchange']
            exchange_stats = df.groupby('exchange_pair')['actual_profit'].agg(['count', 'sum', 'mean', 'std']).reset_index()
            analysis["exchange_stats"] = exchange_stats.to_dict(orient='records')

        # Временной анализ
        if 'open_time' in df.columns and 'actual_profit' in df.columns:
            # Группируем по часам
            df['hour'] = df['open_time'].dt.hour
            hourly_stats = df.groupby('hour')['actual_profit'].agg(['count', 'sum', 'mean']).reset_index()
            analysis["hourly_stats"] = hourly_stats.to_dict(orient='records')

        # Анализ длительности сделок
        if 'duration_ms' in df.columns:
            analysis["duration_stats"] = {
                "mean": float(df['duration_ms'].mean()),
                "median": float(df['duration_ms'].median()),
                "min": float(df['duration_ms'].min()),
                "max": float(df['duration_ms'].max())
            }

        return analysis

    def plot_detailed_analysis(self, name: str, output_dir: str) -> None:
        """
        Строит графики детального анализа указанной симуляции.

        Args:
            name: Имя симуляции
            output_dir: Директория для сохранения графиков
        """
        if name not in self.simulation_data:
            self.logger.error(f"Simulation '{name}' not found")
            return

        # Создаем директорию, если не существует
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Получаем данные о сделках
        data = self.simulation_data[name]
        trade_data = data.get("trade_summary", {}).get("completed_trades", [])

        if not trade_data:
            self.logger.error("No trade data found in simulation")
            return

        # Создаем DataFrame из данных о сделках
        df = pd.DataFrame(trade_data)

        # Преобразуем timestamp в datetime
        if 'open_timestamp' in df.columns:
            df['open_time'] = pd.to_datetime(df['open_timestamp'], unit='ms')

        # Графики для анализа прибыли
        if 'actual_profit' in df.columns:
            # Гистограмма распределения прибыли
            plt.figure(figsize=(12, 6))
            sns.histplot(df['actual_profit'], bins=30, kde=True)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title('Распределение прибыли')
            plt.xlabel('Прибыль')
            plt.ylabel('Количество сделок')
            plt.savefig(os.path.join(output_dir, f"{name}_profit_distribution.png"))
            plt.close()

            # График кумулятивной прибыли по времени
            if 'open_time' in df.columns:
                plt.figure(figsize=(12, 6))
                df_sorted = df.sort_values('open_time')
                df_sorted['cumulative_profit'] = df_sorted['actual_profit'].cumsum()
                plt.plot(df_sorted['open_time'], df_sorted['cumulative_profit'])
                plt.title('Кумулятивная прибыль по времени')
                plt.xlabel('Время')
                plt.ylabel('Кумулятивная прибыль')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, f"{name}_cumulative_profit.png"))
                plt.close()

        # График по торговым парам
        if 'symbol' in df.columns and 'actual_profit' in df.columns:
            plt.figure(figsize=(12, 6))
            symbol_profit = df.groupby('symbol')['actual_profit'].sum().sort_values(ascending=False)
            symbol_profit.plot(kind='bar')
            plt.title('Прибыль по торговым парам')
            plt.xlabel('Торговая пара')
            plt.ylabel('Общая прибыль')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f"{name}_symbol_profit.png"))
            plt.close()

        # График по биржам
        if 'buy_exchange' in df.columns and 'sell_exchange' in df.columns and 'actual_profit' in df.columns:
            plt.figure(figsize=(12, 6))
            df['exchange_pair'] = df['buy_exchange'] + " → " + df['sell_exchange']
            exchange_profit = df.groupby('exchange_pair')['actual_profit'].sum().sort_values(ascending=False)
            exchange_profit.plot(kind='bar')
            plt.title('Прибыль по парам бирж')
            plt.xlabel('Пара бирж (покупка → продажа)')
            plt.ylabel('Общая прибыль')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f"{name}_exchange_profit.png"))
            plt.close()

        # График по времени суток
        if 'open_time' in df.columns and 'actual_profit' in df.columns:
            plt.figure(figsize=(12, 6))
            df['hour'] = df['open_time'].dt.hour
            hourly_profit = df.groupby('hour')['actual_profit'].mean()
            hourly_profit.plot(kind='bar')
            plt.title('Средняя прибыль по часам суток')
            plt.xlabel('Час суток (UTC)')
            plt.ylabel('Средняя прибыль')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f"{name}_hourly_profit.png"))
            plt.close()

        self.logger.info(f"Generated detailed analysis plots for '{name}' in {output_dir}")

    def generate_detailed_report(self, name: str, output_file: str) -> None:
        """
        Генерирует подробный отчет для указанной симуляции.

        Args:
            name: Имя симуляции
            output_file: Путь к файлу для сохранения отчета
        """
        if name not in self.simulation_data:
            self.logger.error(f"Simulation '{name}' not found")
            return

        # Получаем сводку по симуляции
        summary = self.get_simulation_summary(name)

        # Выполняем детальный анализ
        analysis = self.analyze_simulation_detail(name)

        # Создаем HTML-отчет
        html = f"""
        <html>
        <head>
            <title>Detailed Simulation Report - {name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; margin-bottom: 20px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .timestamp {{ font-size: 0.8em; color: #666; }}
                .section {{ margin-top: 30px; margin-bottom: 30px; }}
                .stat-card {{ 
                    display: inline-block; 
                    width: 23%; 
                    margin: 1%; 
                    padding: 15px; 
                    box-sizing: border-box; 
                    background-color: #f9f9f9;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .stat-value {{ font-size: 1.8em; font-weight: bold; margin: 10px 0; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Detailed Simulation Report - {name}</h1>
            <p class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Summary</h2>
                
                <div class="stat-card">
                    <div>Total Trades</div>
                    <div class="stat-value">{summary['total_trades']}</div>
                </div>
                
                <div class="stat-card">
                    <div>Success Rate</div>
                    <div class="stat-value">{summary['success_rate']:.2f}%</div>
                </div>
                
                <div class="stat-card">
                    <div>Total Profit</div>
                    <div class="stat-value {('positive' if summary['total_profit'] >= 0 else 'negative')}">{summary['total_profit']:.2f}</div>
                </div>
                
                <div class="stat-card">
                    <div>Avg Profit/Trade</div>
                    <div class="stat-value {('positive' if summary['avg_profit_per_trade'] >= 0 else 'negative')}">{summary['avg_profit_per_trade']:.4f}</div>
                </div>
            </div>
        """

        # Добавляем информацию о распределении прибыли
        if "profit_distribution" in analysis:
            profit_dist = analysis["profit_distribution"]
            html += f"""
            <div class="section">
                <h2>Profit Distribution</h2>
                
                <div class="stat-card">
                    <div>Mean Profit</div>
                    <div class="stat-value {('positive' if profit_dist['mean'] >= 0 else 'negative')}">{profit_dist['mean']:.4f}</div>
                </div>
                
                <div class="stat-card">
                    <div>Median Profit</div>
                    <div class="stat-value {('positive' if profit_dist['median'] >= 0 else 'negative')}">{profit_dist['median']:.4f}</div>
                </div>
                
                <div class="stat-card">
                    <div>Min Profit</div>
                    <div class="stat-value {('positive' if profit_dist['min'] >= 0 else 'negative')}">{profit_dist['min']:.4f}</div>
                </div>
                
                <div class="stat-card">
                    <div>Max Profit</div>
                    <div class="stat-value {('positive' if profit_dist['max'] >= 0 else 'negative')}">{profit_dist['max']:.4f}</div>
                </div>
                
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Standard Deviation</td>
                        <td>{profit_dist['std']:.4f}</td>
                    </tr>
                    <tr>
                        <td>Profitable Trades</td>
                        <td>{profit_dist['positive_count']} ({profit_dist['positive_count']/max(1, summary['total_trades'])*100:.2f}%)</td>
                    </tr>
                    <tr>
                        <td>Unprofitable Trades</td>
                        <td>{profit_dist['negative_count']} ({profit_dist['negative_count']/max(1, summary['total_trades'])*100:.2f}%)</td>
                    </tr>
                    <tr>
                        <td>Zero Profit Trades</td>
                        <td>{profit_dist['zero_count']} ({profit_dist['zero_count']/max(1, summary['total_trades'])*100:.2f}%)</td>
                    </tr>
                </table>
            </div>
            """

        # Добавляем статистику по торговым парам
        if "symbol_stats" in analysis:
            symbol_stats = analysis["symbol_stats"]
            html += """
            <div class="section">
                <h2>Trading Pair Statistics</h2>
                <table>
                    <tr>
                        <th>Symbol</th>
                        <th>Trade Count</th>
                        <th>Total Profit</th>
                        <th>Avg Profit</th>
                        <th>Std Dev</th>
                    </tr>
            """

            for stat in sorted(symbol_stats, key=lambda x: x['sum'], reverse=True):
                html += f"""
                <tr>
                    <td>{stat['symbol']}</td>
                    <td>{int(stat['count'])}</td>
                    <td class="{('positive' if stat['sum'] >= 0 else 'negative')}">{stat['sum']:.4f}</td>
                    <td class="{('positive' if stat['mean'] >= 0 else 'negative')}">{stat['mean']:.4f}</td>
                    <td>{stat['std']:.4f}</td>
                </tr>
                """

            html += """
                </table>
            </div>
            """

        # Добавляем статистику по биржам
        if "exchange_stats" in analysis:
            exchange_stats = analysis["exchange_stats"]
            html += """
            <div class="section">
                <h2>Exchange Pair Statistics</h2>
                <table>
                    <tr>
                        <th>Exchange Pair</th>
                        <th>Trade Count</th>
                        <th>Total Profit</th>
                        <th>Avg Profit</th>
                        <th>Std Dev</th>
                    </tr>
            """

            for stat in sorted(exchange_stats, key=lambda x: x['sum'], reverse=True):
                html += f"""
                <tr>
                    <td>{stat['exchange_pair'].replace('_to_', ' → ')}</td>
                    <td>{int(stat['count'])}</td>
                    <td class="{('positive' if stat['sum'] >= 0 else 'negative')}">{stat['sum']:.4f}</td>
                    <td class="{('positive' if stat['mean'] >= 0 else 'negative')}">{stat['mean']:.4f}</td>
                    <td>{stat['std']:.4f}</td>
                </tr>
                """

            html += """
                </table>
            </div>
            """

        # Добавляем почасовую статистику
        if "hourly_stats" in analysis:
            hourly_stats = analysis["hourly_stats"]
            html += """
            <div class="section">
                <h2>Hourly Statistics (UTC)</h2>
                <table>
                    <tr>
                        <th>Hour</th>
                        <th>Trade Count</th>
                        <th>Total Profit</th>
                        <th>Avg Profit</th>
                    </tr>
            """

            for stat in sorted(hourly_stats, key=lambda x: x['hour']):
                html += f"""
                <tr>
                    <td>{int(stat['hour']):02d}:00</td>
                    <td>{int(stat['count'])}</td>
                    <td class="{('positive' if stat['sum'] >= 0 else 'negative')}">{stat['sum']:.4f}</td>
                    <td class="{('positive' if stat['mean'] >= 0 else 'negative')}">{stat['mean']:.4f}</td>
                </tr>
                """

            html += """
                </table>
            </div>
            """

        # Добавляем статистику по длительности сделок
        if "duration_stats" in analysis:
            duration_stats = analysis["duration_stats"]
            html += f"""
            <div class="section">
                <h2>Trade Duration Statistics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value (ms)</th>
                        <th>Value (seconds)</th>
                    </tr>
                    <tr>
                        <td>Mean Duration</td>
                        <td>{duration_stats['mean']:.2f}</td>
                        <td>{duration_stats['mean']/1000:.2f}</td>
                    </tr>
                    <tr>
                        <td>Median Duration</td>
                        <td>{duration_stats['median']:.2f}</td>
                        <td>{duration_stats['median']/1000:.2f}</td>
                    </tr>
                    <tr>
                        <td>Min Duration</td>
                        <td>{duration_stats['min']:.2f}</td>
                        <td>{duration_stats['min']/1000:.2f}</td>
                    </tr>
                    <tr>
                        <td>Max Duration</td>
                        <td>{duration_stats['max']:.2f}</td>
                        <td>{duration_stats['max']/1000:.2f}</td>
                    </tr>
                </table>
            </div>
            """

        # Закрываем HTML-документ
        html += """
        </body>
        </html>
        """

        # Сохраняем отчет
        with open(output_file, 'w') as f:
            f.write(html)

        self.logger.info(f"Generated detailed report for '{name}' saved to {output_file}")


def main():
    """Основная функция для запуска анализа статистики симуляций."""
    parser = argparse.ArgumentParser(description="Statistics collector for simulation results")
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to simulation results file or directory with results"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/stats",
        help="Directory for saving analysis results (default: 'results/stats')"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Generate detailed HTML reports"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare multiple simulations (use with directory input)"
    )
    args = parser.parse_args()

    # Настройка логирования
    logger = setup_logging()

    # Создаем директорию для результатов, если не существует
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Создаем коллектор статистики
    collector = SimulationStatsCollector()

    # Загружаем данные симуляций
    if os.path.isdir(args.input):
        # Если указана директория, загружаем все JSON-файлы
        loaded_count = collector.load_multiple_simulations(args.input)
        logger.info(f"Loaded {loaded_count} simulation results from directory {args.input}")

        if loaded_count == 0:
            logger.error("No simulation results found. Exiting.")
            sys.exit(1)
    else:
        # Если указан файл, загружаем его
        if not os.path.exists(args.input):
            logger.error(f"File not found: {args.input}")
            sys.exit(1)

        success = collector.load_simulation_results(args.input)
        if not success:
            logger.error("Failed to load simulation results. Exiting.")
            sys.exit(1)

    # Получаем список имен симуляций
    simulation_names = collector.get_simulation_names()
    logger.info(f"Found {len(simulation_names)} simulations: {', '.join(simulation_names)}")

    # Обрабатываем каждую симуляцию
    for name in simulation_names:
        # Создаем подпапку для результатов анализа этой симуляции
        sim_output_dir = os.path.join(args.output_dir, name)
        if not os.path.exists(sim_output_dir):
            os.makedirs(sim_output_dir)

        # Создаем графики для анализа
        collector.plot_detailed_analysis(name, sim_output_dir)

        # Если запрошен отчет, генерируем его
        if args.report:
            report_file = os.path.join(sim_output_dir, f"{name}_detailed_report.html")
            collector.generate_detailed_report(name, report_file)

    # Если запрошено сравнение и у нас более одной симуляции
    if args.compare and len(simulation_names) > 1:
        # Создаем подпапку для результатов сравнения
        comparison_dir = os.path.join(args.output_dir, "comparison")
        if not os.path.exists(comparison_dir):
            os.makedirs(comparison_dir)

        # Создаем графики для сравнения
        collector.plot_all_comparisons(comparison_dir)

        # Если запрошен отчет, генерируем его
        if args.report:
            report_file = os.path.join(comparison_dir, "comparison_report.html")
            collector.generate_comparison_report(report_file)

    logger.info(f"Analysis completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()