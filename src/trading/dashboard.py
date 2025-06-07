"""
Интерактивный дашборд для визуализации результатов симуляции торговли.
Использует библиотеку Dash для создания веб-интерфейса с графиками и таблицами.
"""

import os
import sys
import json
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

# Импорт библиотек для дашборда
import dash
from dash import dcc, html, dash_table, callback
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

# Добавляем корневую директорию проекта в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def setup_logging():
    """Настройка логирования."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
    except Exception as e:
        logging.error(f"Error loading simulation results: {e}")
        return {}


def prepare_balance_data(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Подготавливает данные о балансах для визуализации.

    Args:
        results: Результаты симуляции

    Returns:
        DataFrame с данными о балансах
    """
    balance_data = []

    # Получаем данные о балансах
    balance_summary = results.get("balance_summary", {})
    balances = balance_summary.get("balances", {})

    for exchange, currencies in balances.items():
        for currency, amount in currencies.items():
            balance_data.append({
                "exchange": exchange,
                "currency": currency,
                "amount": amount
            })

    return pd.DataFrame(balance_data)


def prepare_trade_data(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Подготавливает данные о сделках для визуализации.

    Args:
        results: Результаты симуляции

    Returns:
        DataFrame с данными о сделках
    """
    # Получаем данные о завершенных сделках
    trade_summary = results.get("trade_summary", {})
    completed_trades = trade_summary.get("completed_trades", [])

    # Преобразуем в DataFrame
    if completed_trades:
        df = pd.DataFrame(completed_trades)

        # Преобразуем timestamp в datetime
        if 'open_timestamp' in df.columns:
            df['open_time'] = pd.to_datetime(df['open_timestamp'], unit='ms')
        if 'close_timestamp' in df.columns:
            df['close_time'] = pd.to_datetime(df['close_timestamp'], unit='ms')

        # Добавляем флаг прибыльности
        if 'actual_profit' in df.columns:
            df['is_profitable'] = df['actual_profit'] > 0

        return df

    return pd.DataFrame()


def create_balance_pie_chart(balance_df: pd.DataFrame) -> go.Figure:
    """
    Создает круговую диаграмму распределения баланса по валютам.

    Args:
        balance_df: DataFrame с данными о балансах

    Returns:
        Plotly Figure с круговой диаграммой
    """
    # Группируем данные по валютам
    currency_totals = balance_df.groupby('currency')['amount'].sum().reset_index()

    # Создаем круговую диаграмму
    fig = px.pie(
        currency_totals,
        values='amount',
        names='currency',
        title='Распределение баланса по валютам',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=80, b=20, l=20, r=20)
    )

    return fig


def create_exchange_balance_chart(balance_df: pd.DataFrame) -> go.Figure:
    """
    Создает столбчатую диаграмму распределения баланса по биржам.

    Args:
        balance_df: DataFrame с данными о балансах

    Returns:
        Plotly Figure с столбчатой диаграммой
    """
    # Группируем данные по биржам и валютам
    exchange_currency = balance_df.pivot_table(
        index='exchange',
        columns='currency',
        values='amount',
        aggfunc='sum'
    ).fillna(0)

    # Создаем столбчатую диаграмму
    fig = go.Figure()

    for currency in exchange_currency.columns:
        fig.add_trace(go.Bar(
            x=exchange_currency.index,
            y=exchange_currency[currency],
            name=currency
        ))

    fig.update_layout(
        title='Распределение баланса по биржам',
        xaxis_title='Биржа',
        yaxis_title='Сумма',
        barmode='stack',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=80, b=20, l=20, r=20)
    )

    return fig


def create_profit_time_chart(trade_df: pd.DataFrame) -> go.Figure:
    """
    Создает линейную диаграмму прибыли по времени.

    Args:
        trade_df: DataFrame с данными о сделках

    Returns:
        Plotly Figure с линейной диаграммой
    """
    if 'open_time' not in trade_df.columns or 'actual_profit' not in trade_df.columns:
        # Возвращаем пустой график, если нет необходимых данных
        fig = go.Figure()
        fig.update_layout(
            title='Прибыль по времени (нет данных)',
            xaxis_title='Время',
            yaxis_title='Прибыль'
        )
        return fig

    # Сортируем по времени
    df_sorted = trade_df.sort_values('open_time')

    # Рассчитываем кумулятивную прибыль
    df_sorted['cumulative_profit'] = df_sorted['actual_profit'].cumsum()

    # Создаем линейную диаграмму
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_sorted['open_time'],
        y=df_sorted['cumulative_profit'],
        mode='lines+markers',
        name='Кумулятивная прибыль',
        line=dict(color='green', width=2)
    ))

    fig.update_layout(
        title='Кумулятивная прибыль по времени',
        xaxis_title='Время',
        yaxis_title='Прибыль',
        margin=dict(t=80, b=20, l=20, r=20)
    )

    return fig


def create_profit_distribution_chart(trade_df: pd.DataFrame) -> go.Figure:
    """
    Создает гистограмму распределения прибыли.

    Args:
        trade_df: DataFrame с данными о сделках

    Returns:
        Plotly Figure с гистограммой
    """
    if 'actual_profit' not in trade_df.columns:
        # Возвращаем пустой график, если нет необходимых данных
        fig = go.Figure()
        fig.update_layout(
            title='Распределение прибыли (нет данных)',
            xaxis_title='Прибыль',
            yaxis_title='Количество сделок'
        )
        return fig

    # Создаем гистограмму
    fig = px.histogram(
        trade_df,
        x='actual_profit',
        nbins=30,
        title='Распределение прибыли по сделкам',
        color_discrete_sequence=['lightblue']
    )

    fig.add_vline(
        x=0,
        line_dash='dash',
        line_color='red',
        annotation_text='Нулевая прибыль',
        annotation_position='top'
    )

    fig.update_layout(
        xaxis_title='Прибыль',
        yaxis_title='Количество сделок',
        margin=dict(t=80, b=20, l=20, r=20)
    )

    return fig


def create_symbol_profit_chart(trade_df: pd.DataFrame) -> go.Figure:
    """
    Создает столбчатую диаграмму прибыли по торговым парам.

    Args:
        trade_df: DataFrame с данными о сделках

    Returns:
        Plotly Figure с столбчатой диаграммой
    """
    if 'symbol' not in trade_df.columns or 'actual_profit' not in trade_df.columns:
        # Возвращаем пустой график, если нет необходимых данных
        fig = go.Figure()
        fig.update_layout(
            title='Прибыль по торговым парам (нет данных)',
            xaxis_title='Торговая пара',
            yaxis_title='Прибыль'
        )
        return fig

    # Группируем данные по торговым парам
    symbol_profit = trade_df.groupby('symbol')['actual_profit'].agg(['sum', 'mean', 'count']).reset_index()
    symbol_profit = symbol_profit.sort_values('sum', ascending=False)

    # Создаем столбчатую диаграмму
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=symbol_profit['symbol'],
        y=symbol_profit['sum'],
        name='Общая прибыль',
        marker_color='lightgreen'
    ))

    # Добавляем линию для средней прибыли
    fig.add_trace(go.Scatter(
        x=symbol_profit['symbol'],
        y=symbol_profit['mean'],
        mode='lines+markers',
        name='Средняя прибыль',
        marker=dict(color='darkblue'),
        yaxis='y2'
    ))

    # Добавляем количество сделок как текст над столбцами
    for i, row in symbol_profit.iterrows():
        fig.add_annotation(
            x=row['symbol'],
            y=row['sum'],
            text=f"n={int(row['count'])}",
            showarrow=False,
            yshift=10
        )

    fig.update_layout(
        title='Прибыль по торговым парам',
        xaxis_title='Торговая пара',
        yaxis=dict(
            title='Общая прибыль',
            titlefont=dict(color='green'),
            tickfont=dict(color='green')
        ),
        yaxis2=dict(
            title='Средняя прибыль',
            titlefont=dict(color='darkblue'),
            tickfont=dict(color='darkblue'),
            anchor='x',
            overlaying='y',
            side='right'
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=80, b=20, l=20, r=20)
    )

    return fig


def create_exchange_profit_chart(trade_df: pd.DataFrame) -> go.Figure:
    """
    Создает столбчатую диаграмму прибыли по биржам.

    Args:
        trade_df: DataFrame с данными о сделках

    Returns:
        Plotly Figure с столбчатой диаграммой
    """
    if 'buy_exchange' not in trade_df.columns or 'actual_profit' not in trade_df.columns:
        # Возвращаем пустой график, если нет необходимых данных
        fig = go.Figure()
        fig.update_layout(
            title='Прибыль по биржам (нет данных)',
            xaxis_title='Биржа',
            yaxis_title='Прибыль'
        )
        return fig

    # Создаем DataFrame с данными по парам бирж
    exchange_pairs = []
    for _, row in trade_df.iterrows():
        if 'buy_exchange' in row and 'sell_exchange' in row and 'actual_profit' in row:
            exchange_pairs.append({
                'exchange_pair': f"{row['buy_exchange']} → {row['sell_exchange']}",
                'buy_exchange': row['buy_exchange'],
                'sell_exchange': row['sell_exchange'],
                'profit': row['actual_profit']
            })

    exchange_pair_df = pd.DataFrame(exchange_pairs)

    # Группируем данные по парам бирж
    pair_profit = exchange_pair_df.groupby('exchange_pair')['profit'].agg(['sum', 'mean', 'count']).reset_index()
    pair_profit = pair_profit.sort_values('sum', ascending=False)

    # Создаем столбчатую диаграмму
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=pair_profit['exchange_pair'],
        y=pair_profit['sum'],
        name='Общая прибыль',
        marker_color='lightblue'
    ))

    # Добавляем линию для средней прибыли
    fig.add_trace(go.Scatter(
        x=pair_profit['exchange_pair'],
        y=pair_profit['mean'],
        mode='lines+markers',
        name='Средняя прибыль',
        marker=dict(color='darkred'),
        yaxis='y2'
    ))

    # Добавляем количество сделок как текст над столбцами
    for i, row in pair_profit.iterrows():
        fig.add_annotation(
            x=row['exchange_pair'],
            y=row['sum'],
            text=f"n={int(row['count'])}",
            showarrow=False,
            yshift=10
        )

    fig.update_layout(
        title='Прибыль по парам бирж',
        xaxis_title='Пара бирж (покупка → продажа)',
        yaxis=dict(
            title='Общая прибыль',
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue')
        ),
        yaxis2=dict(
            title='Средняя прибыль',
            titlefont=dict(color='darkred'),
            tickfont=dict(color='darkred'),
            anchor='x',
            overlaying='y',
            side='right'
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=80, b=20, l=20, r=20),
        height=600
    )

    return fig


def create_dashboard(results_file: str) -> dash.Dash:
    """
    Создает интерактивный дашборд для визуализации результатов симуляции.

    Args:
        results_file: Путь к файлу результатов симуляции

    Returns:
        Объект приложения Dash
    """
    # Загружаем результаты симуляции
    results = load_simulation_results(results_file)

    # Подготавливаем данные
    balance_df = prepare_balance_data(results)
    trade_df = prepare_trade_data(results)

    # Создаем графики
    balance_pie_chart = create_balance_pie_chart(balance_df)
    exchange_balance_chart = create_exchange_balance_chart(balance_df)
    profit_time_chart = create_profit_time_chart(trade_df)
    profit_distribution_chart = create_profit_distribution_chart(trade_df)
    symbol_profit_chart = create_symbol_profit_chart(trade_df)
    exchange_profit_chart = create_exchange_profit_chart(trade_df)

    # Получаем общую статистику
    stats = results.get("trade_summary", {}).get("stats", {})
    total_trades = stats.get("total_trades", 0)
    successful_trades = stats.get("successful_trades", 0)
    success_rate = (successful_trades / total_trades) * 100 if total_trades > 0 else 0
    total_profit = stats.get("total_profit", 0)

    # Создаем приложение Dash
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title="Simulation Dashboard"
    )

    # Создаем макет дашборда
    app.layout = dbc.Container(
        [
            # Заголовок
            html.H1("Результаты симуляции криптовалютного арбитража", className="text-center mt-4 mb-4"),

            # Карточки с общей статистикой
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{total_trades}", className="card-title text-center"),
                            html.P("Всего сделок", className="card-text text-center")
                        ])
                    ], color="primary", outline=True),
                    width=3
                ),
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{successful_trades} ({success_rate:.1f}%)", className="card-title text-center"),
                            html.P("Успешных сделок", className="card-text text-center")
                        ])
                    ], color="success", outline=True),
                    width=3
                ),
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{total_profit:.2f} USDT", className="card-title text-center"),
                            html.P("Общая прибыль", className="card-text text-center")
                        ])
                    ], color="info", outline=True),
                    width=3
                ),
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{total_profit / max(1, total_trades):.2f} USDT",
                                    className="card-title text-center"),
                            html.P("Средняя прибыль на сделку", className="card-text text-center")
                        ])
                    ], color="warning", outline=True),
                    width=3
                )
            ], className="mb-4"),

            # Вкладки для разных категорий графиков
            dbc.Tabs([
                # Вкладка с балансами
                dbc.Tab([
                    dbc.Row([
                        dbc.Col(
                            dcc.Graph(figure=balance_pie_chart),
                            width=6
                        ),
                        dbc.Col(
                            dcc.Graph(figure=exchange_balance_chart),
                            width=6
                        )
                    ], className="mt-4"),

                    # Таблица с балансами
                    dbc.Row([
                        dbc.Col(
                            dash_table.DataTable(
                                data=balance_df.to_dict('records'),
                                columns=[
                                    {"name": "Биржа", "id": "exchange"},
                                    {"name": "Валюта", "id": "currency"},
                                    {"name": "Сумма", "id": "amount", "type": "numeric", "format": {"specifier": ".8f"}}
                                ],
                                style_table={'overflowX': 'auto'},
                                style_header={
                                    'backgroundColor': 'rgb(230, 230, 230)',
                                    'fontWeight': 'bold'
                                },
                                style_cell={
                                    'textAlign': 'left',
                                    'padding': '8px'
                                },
                                style_data_conditional=[
                                    {
                                        'if': {'row_index': 'odd'},
                                        'backgroundColor': 'rgb(248, 248, 248)'
                                    }
                                ],
                                filter_action="native",
                                sort_action="native",
                                page_size=10
                            ),
                            width=12
                        )
                    ], className="mt-4")
                ], label="Балансы"),

                # Вкладка с прибылью
                dbc.Tab([
                    dbc.Row([
                        dbc.Col(
                            dcc.Graph(figure=profit_time_chart),
                            width=12
                        )
                    ], className="mt-4"),

                    dbc.Row([
                        dbc.Col(
                            dcc.Graph(figure=profit_distribution_chart),
                            width=6
                        ),
                        dbc.Col(
                            dcc.Graph(figure=symbol_profit_chart),
                            width=6
                        )
                    ], className="mt-4"),

                    dbc.Row([
                        dbc.Col(
                            dcc.Graph(figure=exchange_profit_chart),
                            width=12
                        )
                    ], className="mt-4")
                ], label="Прибыль"),

                # Вкладка с деталями сделок
                dbc.Tab([
                    # Таблица с деталями сделок
                    dbc.Row([
                        dbc.Col(
                            dash_table.DataTable(
                                data=trade_df.to_dict('records'),
                                columns=[
                                    {"name": "ID", "id": "trade_id"},
                                    {"name": "Символ", "id": "symbol"},
                                    {"name": "Биржа (покупка)", "id": "buy_exchange"},
                                    {"name": "Биржа (продажа)", "id": "sell_exchange"},
                                    {"name": "Цена покупки", "id": "buy_price", "type": "numeric",
                                     "format": {"specifier": ".8f"}},
                                    {"name": "Цена продажи", "id": "actual_sell_price", "type": "numeric",
                                     "format": {"specifier": ".8f"}},
                                    {"name": "Объем", "id": "volume", "type": "numeric",
                                     "format": {"specifier": ".8f"}},
                                    {"name": "Прибыль", "id": "actual_profit", "type": "numeric",
                                     "format": {"specifier": ".8f"}},
                                    {"name": "Прибыль %", "id": "actual_profit_percentage", "type": "numeric",
                                     "format": {"specifier": ".2f"}},
                                    {"name": "Статус", "id": "status"}
                                ],
                                style_table={'overflowX': 'auto'},
                                style_header={
                                    'backgroundColor': 'rgb(230, 230, 230)',
                                    'fontWeight': 'bold'
                                },
                                style_cell={
                                    'textAlign': 'left',
                                    'padding': '8px'
                                },
                                style_data_conditional=[
                                    {
                                        'if': {'row_index': 'odd'},
                                        'backgroundColor': 'rgb(248, 248, 248)'
                                    },
                                    {
                                        'if': {
                                            'filter_query': '{actual_profit} > 0',
                                            'column_id': 'actual_profit'
                                        },
                                        'backgroundColor': 'rgba(0, 255, 0, 0.15)',
                                        'color': 'green'
                                    },
                                    {
                                        'if': {
                                            'filter_query': '{actual_profit} < 0',
                                            'column_id': 'actual_profit'
                                        },
                                        'backgroundColor': 'rgba(255, 0, 0, 0.15)',
                                        'color': 'red'
                                    }
                                ],
                                filter_action="native",
                                sort_action="native",
                                page_size=15
                            ),
                            width=12
                        )
                    ], className="mt-4")
                ], label="Детали сделок"),

                # Вкладка с информацией о файле
                dbc.Tab([
                    dbc.Row([
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader("Информация о файле симуляции"),
                                dbc.CardBody([
                                    html.P(f"Файл: {os.path.basename(results_file)}"),
                                    html.P(f"Размер: {os.path.getsize(results_file) / 1024:.2f} Кб"),
                                    html.P(
                                        f"Дата создания: {datetime.fromtimestamp(os.path.getctime(results_file)).strftime('%Y-%m-%d %H:%M:%S')}"),
                                    html.Hr(),
                                    html.H5("Параметры симуляции"),
                                    html.P(
                                        f"Время начала: {datetime.fromtimestamp(results.get('simulation_stats', {}).get('start_time', 0)).strftime('%Y-%m-%d %H:%M:%S')}"),
                                    html.P(
                                        f"Время окончания: {datetime.fromtimestamp(results.get('simulation_stats', {}).get('end_time', 0)).strftime('%Y-%m-%d %H:%M:%S')}"),
                                    html.P(
                                        f"Длительность: {results.get('simulation_stats', {}).get('duration_seconds', 0):.2f} секунд"),
                                    html.P(
                                        f"Количество итераций: {results.get('simulation_stats', {}).get('iterations', 0)}")
                                ])
                            ]),
                            width=6
                        )
                    ], className="mt-4", justify="center")
                ], label="О симуляции")
            ], className="mt-4"),

            # Нижний колонтитул
            html.Footer(
                dbc.Row([
                    dbc.Col(
                        html.P("Crypto Arbitrage Bot Simulation Dashboard", className="text-center text-muted"),
                        width=12
                    )
                ]),
                className="mt-4 pt-3 border-top"
            )
        ],
        fluid=True,
        className="p-4"
    )

    return app


def main():
    """Основная функция для запуска дашборда."""
    parser = argparse.ArgumentParser(description="Dashboard for crypto arbitrage simulation results")
    parser.add_argument(
        "--file", type=str, required=True,
        help="Path to simulation results JSON file"
    )
    parser.add_argument(
        "--port", type=int, default=8050,
        help="Port for the dashboard server (default: 8050)"
    )
    args = parser.parse_args()

    # Настройка логирования
    logger = setup_logging()

    # Проверяем наличие файла
    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        sys.exit(1)

    logger.info(f"Loading simulation results from {args.file}")

    # Создаем дашборд
    app = create_dashboard(args.file)

    # Запускаем сервер
    logger.info(f"Starting dashboard server on port {args.port}")
    app.run_server(debug=True, port=args.port)


if __name__ == "__main__":
    main()