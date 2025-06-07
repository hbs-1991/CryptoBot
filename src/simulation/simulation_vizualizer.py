"""
Визуализатор результатов симуляции арбитражных сделок.
Позволяет анализировать и визуализировать результаты проведенных симуляций.
"""

import os
import sys
import json
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from datetime import datetime

# Добавляем корневую директорию проекта в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def setup_logging():
    """Настройка логирования."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_simulation_results(file_path: str) -> Dict[str, Any]:
    """
    Загружает результаты симуляции из JSON файла.

    Args:
        file_path: Путь к файлу результатов

    Returns:
        Словарь с результатами симуляции
    """
    try:
        with open(file_path, 'r') as f:
            results = json.load(f)
        return results
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logging.error(f"Error loading simulation results: {e}")
        sys.exit(1)


def analyze_trades(completed_trades: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Анализирует завершенные сделки и создает DataFrame для анализа.

    Args:
        completed_trades: Список завершенных сделок

    Returns:
        DataFrame с данными о сделках
    """
    # Создаем DataFrame из сделок
    df = pd.DataFrame(completed_trades)

    # Преобразуем timestamp в datetime
    if 'open_timestamp' in df.columns:
        df['open_time'] = pd.to_datetime(df['open_timestamp'], unit='ms')
    if 'close_timestamp' in df.columns:
        df['close_time'] = pd.to_datetime(df['close_timestamp'], unit='ms')

    # Добавляем дополнительные метрики если они отсутствуют
    if 'actual_profit' not in df.columns and 'actual_sell_price' in df.columns and 'buy_price' in df.columns and 'volume' in df.columns:
        df['actual_profit'] = (df['actual_sell_price'] - df['buy_price']) * df['volume']

    if 'actual_profit_percentage' not in df.columns and 'actual_sell_price' in df.columns and 'buy_price' in df.columns:
        df['actual_profit_percentage'] = (df['actual_sell_price'] - df['buy_price']) / df['buy_price'] * 100

    # Добавляем флаг успешности сделки
    if 'actual_profit' in df.columns:
        df['is_profitable'] = df['actual_profit'] > 0

    return df


def create_plots(df: pd.DataFrame, output_dir: str, timestamp: str = None):
    """
    Создает набор графиков для визуального анализа результатов симуляции.

    Args:
        df: DataFrame с данными о сделках
        output_dir: Директория для сохранения графиков
        timestamp: Временная метка для имени файлов
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Создаем директорию для графиков если не существует
    plots_dir = os.path.join(output_dir, "analysis")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Настраиваем стиль графиков
    plt.style.use('ggplot')
    sns.set(style="whitegrid", palette="muted")

    # 1. График кумулятивной прибыли по времени
    if 'open_time' in df.columns and 'actual_profit' in df.columns:
        plt.figure(figsize=(14, 7))
        df_sorted = df.sort_values('open_time')
        df_sorted['cumulative_profit'] = df_sorted['actual_profit'].cumsum()

        plt.plot(df_sorted['open_time'], df_sorted['cumulative_profit'], 'b-', linewidth=2)
        plt.xlabel('Время')
        plt.ylabel('Кумулятивная прибыль (USDT)')
        plt.title('Кумулятивная прибыль по времени')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"cumulative_profit_time_{timestamp}.png"))
        plt.close()

    # 2. Гистограмма распределения прибыли
    if 'actual_profit' in df.columns:
        plt.figure(figsize=(14, 7))
        sns.histplot(df['actual_profit'], bins=30, kde=True)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Прибыль (USDT)')
        plt.ylabel('Количество сделок')
        plt.title('Распределение прибыли')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"profit_distribution_{timestamp}.png"))
        plt.close()

    # 3. Распределение сделок по торговым парам
    if 'symbol' in df.columns:
        plt.figure(figsize=(14, 7))
        symbol_counts = df['symbol'].value_counts()
        symbol_counts.plot(kind='bar', color='skyblue')
        plt.xlabel('Торговая пара')
        plt.ylabel('Количество сделок')
        plt.title('Количество сделок по торговым парам')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"trades_by_symbol_{timestamp}.png"))
        plt.close()

    # 4. Распределение сделок по биржам покупки
    if 'buy_exchange' in df.columns:
        plt.figure(figsize=(14, 7))
        exchange_counts = df['buy_exchange'].value_counts()
        exchange_counts.plot(kind='bar', color='lightgreen')
        plt.xlabel('Биржа покупки')
        plt.ylabel('Количество сделок')
        plt.title('Количество сделок по биржам покупки')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"trades_by_buy_exchange_{timestamp}.png"))
        plt.close()

    # 5. Эффективность стратегий
    if 'strategy_name' in df.columns and 'actual_profit' in df.columns:
        plt.figure(figsize=(14, 7))
        strategy_profit = df.groupby('strategy_name')['actual_profit'].agg(['sum', 'mean', 'count'])

        # Создаем фигуру с двумя осями y
        fig, ax1 = plt.subplots(figsize=(14, 7))
        ax2 = ax1.twinx()

        # Рисуем столбцы для общей прибыли
        bars = ax1.bar(strategy_profit.index, strategy_profit['sum'], color='skyblue', alpha=0.7, label='Общая прибыль')
        ax1.set_xlabel('Стратегия')
        ax1.set_ylabel('Общая прибыль (USDT)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Рисуем линию для средней прибыли
        line = ax2.plot(strategy_profit.index, strategy_profit['mean'], 'r-o', linewidth=2, label='Средняя прибыль')
        ax2.set_ylabel('Средняя прибыль (USDT)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Добавляем количество сделок как текст над столбцами
        for i, count in enumerate(strategy_profit['count']):
            ax1.text(i, strategy_profit['sum'].iloc[i] + 0.1, f'n={count}', ha='center')

        plt.title('Эффективность стратегий')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"strategy_effectiveness_{timestamp}.png"))
        plt.close()

    # 6. Тепловая карта прибыльности по парам биржи-символы
    if 'buy_exchange' in df.columns and 'symbol' in df.columns and 'actual_profit_percentage' in df.columns:
        # Создаем сводную таблицу
        pivot = df.pivot_table(
            values='actual_profit_percentage',
            index='buy_exchange',
            columns='symbol',
            aggfunc='mean'
        )

        plt.figure(figsize=(14, 8))
        sns.heatmap(pivot, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Средняя прибыльность (%) по парам биржа-символ')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"profitability_heatmap_{timestamp}.png"))
        plt.close()

    # 7. Распределение длительности сделок
    if 'duration_ms' in df.columns:
        plt.figure(figsize=(14, 7))
        # Преобразуем миллисекунды в секунды
        df['duration_sec'] = df['duration_ms'] / 1000

        sns.histplot(df['duration_sec'], bins=30, kde=True)
        plt.xlabel('Длительность (секунды)')
        plt.ylabel('Количество сделок')
        plt.title('Распределение длительности сделок')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"duration_distribution_{timestamp}.png"))
        plt.close()

    # 8. Соотношение успешных и неуспешных сделок
    if 'is_profitable' in df.columns:
        plt.figure(figsize=(10, 10))
        profit_count = df['is_profitable'].value_counts()
        labels = ['Прибыльные', 'Убыточные']
        sizes = [profit_count.get(True, 0), profit_count.get(False, 0)]
        colors = ['lightgreen', 'lightcoral']
        explode = (0.1, 0)  # выделяем прибыльные сделки

        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Соотношение прибыльных и убыточных сделок')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"profit_ratio_pie_{timestamp}.png"))
        plt.close()

    # 9. Зависимость прибыли от времени суток (если есть достаточно данных)
    if 'open_time' in df.columns and 'actual_profit' in df.columns and len(df) >= 24:
        plt.figure(figsize=(14, 7))
        # Извлекаем час из времени
        df['hour'] = df['open_time'].dt.hour

        # Группируем по часу и считаем среднюю прибыль
        hourly_profit = df.groupby('hour')['actual_profit'].mean()

        plt.bar(hourly_profit.index, hourly_profit, color='cornflowerblue')
        plt.xlabel('Час суток (UTC)')
        plt.ylabel('Средняя прибыль (USDT)')
        plt.title('Зависимость средней прибыли от времени суток')
        plt.xticks(range(24))
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"hourly_profit_{timestamp}.png"))
        plt.close()

    # 10. Объём торгов и его зависимость от прибыли
    if 'volume' in df.columns and 'actual_profit' in df.columns:
        plt.figure(figsize=(14, 7))
        plt.scatter(df['volume'], df['actual_profit'], alpha=0.6, c=df['actual_profit'], cmap='viridis')
        plt.colorbar(label='Прибыль (USDT)')
        plt.xlabel('Объём торгов (в базовой валюте)')
        plt.ylabel('Прибыль (USDT)')
        plt.title('Зависимость прибыли от объёма торгов')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"volume_profit_correlation_{timestamp}.png"))
        plt.close()


def generate_summary_report(df: pd.DataFrame, results: Dict[str, Any], output_file: str):
    """
    Генерирует текстовый отчет с подробной статистикой симуляции.

    Args:
        df: DataFrame с данными о сделках
        results: Словарь с результатами симуляции
        output_file: Путь для сохранения отчета
    """
    report_lines = [
        "===========================================================",
        "           ПОДРОБНЫЙ ОТЧЕТ ПО СИМУЛЯЦИИ АРБИТРАЖА",
        "===========================================================",
        "",
        "ОБЩАЯ СТАТИСТИКА:",
        "---------------------------------------------------------",
    ]

    # Добавляем общую статистику
    stats = results.get("trade_summary", {}).get("stats", {})
    balance_summary = results.get("balance_summary", {})
    simulation_stats = results.get("simulation_stats", {})

    if stats:
        report_lines.extend([
            f"Всего сделок: {stats.get('total_trades', 0)}",
            f"Успешных сделок: {stats.get('successful_trades', 0)} ({stats.get('successful_trades', 0) / max(1, stats.get('total_trades', 1)) * 100:.2f}%)",
            f"Неудачных сделок: {stats.get('failed_trades', 0)}",
            f"Общая прибыль: {stats.get('total_profit', 0):.2f} USDT",
            f"Общий объем торгов: {stats.get('total_volume_traded', 0):.2f} USDT",
            f"Средняя прибыль на сделку: {stats.get('total_profit', 0) / max(1, stats.get('total_trades', 1)):.4f} USDT",
            ""
        ])

    if simulation_stats:
        report_lines.extend([
            "ПАРАМЕТРЫ СИМУЛЯЦИИ:",
            "---------------------------------------------------------",
            f"Время начала: {datetime.fromtimestamp(simulation_stats.get('start_time', 0)).strftime('%Y-%m-%d %H:%M:%S')}",
            f"Время окончания: {datetime.fromtimestamp(simulation_stats.get('end_time', 0)).strftime('%Y-%m-%d %H:%M:%S')}",
            f"Длительность: {simulation_stats.get('duration_seconds', 0):.2f} секунд",
            f"Итераций: {simulation_stats.get('iterations', 0)}",
            ""
        ])

    # Добавляем статистику по торговым парам
    if 'symbol' in df.columns and 'actual_profit' in df.columns:
        report_lines.extend([
            "СТАТИСТИКА ПО ТОРГОВЫМ ПАРАМ:",
            "---------------------------------------------------------",
        ])

        symbol_stats = df.groupby('symbol').agg({
            'actual_profit': ['sum', 'mean', 'count'],
            'actual_profit_percentage': ['mean', 'min', 'max']
        })

        for symbol, row in symbol_stats.iterrows():
            report_lines.extend([
                f"Символ: {symbol}",
                f"  Количество сделок: {row[('actual_profit', 'count')]}",
                f"  Общая прибыль: {row[('actual_profit', 'sum')]:.2f} USDT",
                f"  Средняя прибыль: {row[('actual_profit', 'mean')]:.4f} USDT",
                f"  Средняя прибыль (%): {row[('actual_profit_percentage', 'mean')]:.4f}%",
                f"  Мин/Макс прибыль (%): {row[('actual_profit_percentage', 'min')]:.4f}% / {row[('actual_profit_percentage', 'max')]:.4f}%",
                ""
            ])

    # Добавляем статистику по биржам
    if 'buy_exchange' in df.columns and 'actual_profit' in df.columns:
        report_lines.extend([
            "СТАТИСТИКА ПО БИРЖАМ:",
            "---------------------------------------------------------",
        ])

        exchange_stats = df.groupby('buy_exchange').agg({
            'actual_profit': ['sum', 'mean', 'count']
        })

        for exchange, row in exchange_stats.iterrows():
            report_lines.extend([
                f"Биржа (покупка): {exchange}",
                f"  Количество сделок: {row[('actual_profit', 'count')]}",
                f"  Общая прибыль: {row[('actual_profit', 'sum')]:.2f} USDT",
                f"  Средняя прибыль: {row[('actual_profit', 'mean')]:.4f} USDT",
                ""
            ])

    # Добавляем статистику по стратегиям
    if 'strategy_name' in df.columns and 'actual_profit' in df.columns:
        report_lines.extend([
            "СТАТИСТИКА ПО СТРАТЕГИЯМ:",
            "---------------------------------------------------------",
        ])

        strategy_stats = df.groupby('strategy_name').agg({
            'actual_profit': ['sum', 'mean', 'count'],
            'actual_profit_percentage': ['mean', 'min', 'max'],
            'duration_ms': ['mean']
        })

        for strategy, row in strategy_stats.iterrows():
            report_lines.extend([
                f"Стратегия: {strategy}",
                f"  Количество сделок: {row[('actual_profit', 'count')]}",
                f"  Общая прибыль: {row[('actual_profit', 'sum')]:.2f} USDT",
                f"  Средняя прибыль: {row[('actual_profit', 'mean')]:.4f} USDT",
                f"  Средняя прибыль (%): {row[('actual_profit_percentage', 'mean')]:.4f}%",
                f"  Мин/Макс прибыль (%): {row[('actual_profit_percentage', 'min')]:.4f}% / {row[('actual_profit_percentage', 'max')]:.4f}%",
                f"  Средняя длительность: {row[('duration_ms', 'mean')] / 1000:.2f} секунд",
                ""
            ])

    # Добавляем информацию о наиболее прибыльных сделках
    if 'actual_profit' in df.columns:
        report_lines.extend([
            "ТОП-10 НАИБОЛЕЕ ПРИБЫЛЬНЫХ СДЕЛОК:",
            "---------------------------------------------------------",
        ])

        # Сортируем сделки по прибыли
        top_profitable = df.sort_values('actual_profit', ascending=False).head(10)

        for i, (_, trade) in enumerate(top_profitable.iterrows()):
            trade_details = []
            trade_details.append(f"{i + 1}. Символ: {trade.get('symbol')}")

            if 'buy_exchange' in trade and 'sell_exchange' in trade:
                trade_details.append(f"   Биржи: {trade.get('buy_exchange')} -> {trade.get('sell_exchange')}")

            if 'buy_price' in trade and 'actual_sell_price' in trade:
                trade_details.append(f"   Цены: {trade.get('buy_price'):.6f} -> {trade.get('actual_sell_price'):.6f}")

            if 'volume' in trade:
                trade_details.append(f"   Объем: {trade.get('volume'):.6f}")

            if 'actual_profit' in trade:
                trade_details.append(f"   Прибыль: {trade.get('actual_profit'):.6f} USDT")

            if 'actual_profit_percentage' in trade:
                trade_details.append(f"   Прибыль (%): {trade.get('actual_profit_percentage'):.4f}%")

            if 'open_time' in trade and 'close_time' in trade:
                trade_details.append(f"   Время: {trade.get('open_time')} -> {trade.get('close_time')}")

            if 'duration_ms' in trade:
                trade_details.append(f"   Длительность: {trade.get('duration_ms') / 1000:.2f} секунд")

            report_lines.extend(trade_details)
            report_lines.append("")

    # Добавляем итоговую информацию о балансах
    if balance_summary and 'balances' in balance_summary:
        report_lines.extend([
            "ИТОГОВЫЕ БАЛАНСЫ ПО БИРЖАМ:",
            "---------------------------------------------------------",
        ])

        for exchange, currencies in balance_summary['balances'].items():
            report_lines.append(f"Биржа: {exchange}")
            for currency, amount in currencies.items():
                report_lines.append(f"  {currency}: {amount:.8f}")
            report_lines.append("")

    # Заключение
    report_lines.extend([
        "===========================================================",
        "                      ЗАКЛЮЧЕНИЕ",
        "===========================================================",
        "",
        f"Общий результат симуляции: {'УСПЕШНО' if stats.get('total_profit', 0) > 0 else 'НЕУСПЕШНО'}",
        f"Общая прибыль: {stats.get('total_profit', 0):.2f} USDT",
        f"Успешность сделок: {stats.get('successful_trades', 0) / max(1, stats.get('total_trades', 1)) * 100:.2f}%",
        "",
        "Рекомендации для улучшения торговых стратегий:",
        "1. Проанализировать причины неудачных сделок",
        "2. Оптимизировать параметры наиболее успешных стратегий",
        "3. Рассмотреть возможность увеличения объема для более прибыльных пар",
        "",
        f"Отчет сгенерирован: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "===========================================================",
    ])

    # Сохраняем отчет
    with open(output_file, 'w') as f:
        f.write("\n".join(report_lines))


class SimulationVisualizer:
    """
    Класс для визуализации результатов симуляций арбитражных сделок.
    Позволяет создавать графики и генерировать отчеты по результатам симуляций.
    """
    
    def __init__(self):
        """
        Инициализирует объект визуализатора.
        """
        self.logger = logging.getLogger(__name__)
    
    def load_results(self, file_path: str) -> Dict[str, Any]:
        """
        Загружает результаты симуляции из JSON файла.
        
        Args:
            file_path: Путь к файлу результатов
            
        Returns:
            Словарь с результатами симуляции
        """
        return load_simulation_results(file_path)
    
    def analyze_simulation(self, results: Dict[str, Any], output_dir: str = "results/analysis"):
        """
        Анализирует результаты симуляции и создает визуализации.
        
        Args:
            results: Словарь с результатами симуляции
            output_dir: Директория для сохранения результатов анализа
        """
        # Создаем директорию для результатов анализа
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Извлекаем данные о сделках
        completed_trades = results.get("trade_summary", {}).get("completed_trades", [])
        
        if not completed_trades:
            self.logger.error("No completed trades found in simulation results")
            return
        
        # Анализируем сделки
        self.logger.info(f"Analyzing {len(completed_trades)} completed trades")
        df = analyze_trades(completed_trades)
        
        # Генерируем временную метку для имен файлов
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Создаем графики
        self.logger.info("Generating plots")
        create_plots(df, output_dir, timestamp)
        
        # Генерируем подробный отчет
        self.logger.info("Generating summary report")
        report_file = os.path.join(output_dir, f"detailed_report_{timestamp}.txt")
        generate_summary_report(df, results, report_file)
        
        self.logger.info(f"Analysis completed. Results saved to {output_dir}")
        self.logger.info(f"Detailed report saved to {report_file}")
        
        return {
            "report_file": report_file,
            "output_dir": output_dir,
            "timestamp": timestamp
        }
    
    def visualize_live_results(self, stats, output_dir: str = "results/live"):
        """
        Создает визуализацию для текущих результатов симуляции.
        
        Args:
            stats: Объект статистики симуляции
            output_dir: Директория для сохранения результатов
            
        Returns:
            Путь к сохраненному графику
        """
        # Создаем директорию для результатов
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Получаем данные из статистики
        summary = stats.get_stats_summary()
        
        if hasattr(stats, 'profit_history') and hasattr(stats, 'timestamp_history') and stats.profit_history:
            # Создаем DataFrame для анализа
            data = {
                'profit': stats.profit_history,
                'timestamp': stats.timestamp_history
            }
            df = pd.DataFrame(data)
            
            # Рассчитываем кумулятивную прибыль
            df['cumulative_profit'] = df['profit'].cumsum()
            
            # Создаем график кумулятивной прибыли
            plt.figure(figsize=(14, 7))
            plt.plot(df['timestamp'], df['cumulative_profit'], 'b-', linewidth=2)
            plt.xlabel('Время')
            plt.ylabel('Кумулятивная прибыль')
            plt.title('Кумулятивная прибыль за текущую симуляцию')
            plt.grid(True, alpha=0.3)
            
            # Добавляем аннотации с информацией о статистике
            plt.annotate(
                f"Всего сделок: {summary['total_trades']}\n"
                f"Успешных: {summary['successful_trades']} ({summary['success_rate']:.1f}%)\n"
                f"Общая прибыль: {summary['total_profit']:.4f}",
                xy=(0.02, 0.85),
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8)
            )
            
            plt.tight_layout()
            
            # Сохраняем график
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"live_simulation_{timestamp}.png")
            plt.savefig(output_file)
            plt.close()
            
            return output_file
        
        return None


def main():
    """Основная функция для анализа результатов симуляции."""
    parser = argparse.ArgumentParser(description="Visualizer for crypto arbitrage simulation results")
    parser.add_argument(
        "--file", type=str, required=True,
        help="Path to simulation results JSON file"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/analysis",
        help="Directory for saving analysis results (default: 'results/analysis')"
    )
    args = parser.parse_args()

    # Настройка логирования
    logger = setup_logging()

    # Создаем директорию для результатов анализа
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Загружаем результаты симуляции
    logger.info(f"Loading simulation results from {args.file}")
    results = load_simulation_results(args.file)

    # Извлекаем данные о сделках
    completed_trades = results.get("trade_summary", {}).get("completed_trades", [])

    if not completed_trades:
        logger.error("No completed trades found in simulation results")
        sys.exit(1)

    # Анализируем сделки
    logger.info(f"Analyzing {len(completed_trades)} completed trades")
    df = analyze_trades(completed_trades)

    # Генерируем временную метку для имен файлов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Создаем графики
    logger.info("Generating plots")
    create_plots(df, args.output_dir, timestamp)

    # Генерируем подробный отчет
    logger.info("Generating summary report")
    report_file = os.path.join(args.output_dir, f"detailed_report_{timestamp}.txt")
    generate_summary_report(df, results, report_file)

    logger.info(f"Analysis completed. Results saved to {args.output_dir}")
    logger.info(f"Detailed report saved to {report_file}")


if __name__ == "__main__":
    main()