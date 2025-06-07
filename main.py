"""
Точка входа приложения крипто-арбитражного бота.
Интегрирует все модули системы и обеспечивает работу MVP.
"""

import os
import sys
import logging
import asyncio
import argparse
import time
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Any, Optional, Tuple

# Добавляем корневую директорию проекта в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Конфигурация и настройки
from config import settings, exchange_config

# Компоненты для подключения к биржам
from src.data_fetching.exchange_factory import ExchangeFactory
from src.data_fetching.exchange_connector import ExchangeConnector, ExchangeConnectionError

# Компоненты сканирования данных
from src.data_fetching.market_scanner import MarketScanner
from src.data_fetching.data_normalizer import DataNormalizer
from src.data_fetching.price_monitor import PriceMonitor

# Компоненты арбитражного движка
from src.arbitrage_engine.opportunity_finder import OpportunityFinder
from src.arbitrage_engine.profit_calculator import ProfitCalculator, ArbitrageOpportunity
from src.arbitrage_engine.risk_manager import RiskManager
from src.arbitrage_engine.market_strategy import DirectArbitrageStrategy, StrategySelector

# Компоненты для симуляции
from src.simulation.simulation import TradingSimulator
from src.simulation.simulation_runner import SimulationRunner
from src.simulation.simulation_stats import SimulationStats
from src.simulation.simulation_vizualizer import SimulationVisualizer
from src.simulation.virtual_balance_manager import VirtualBalanceManager

# Компоненты для уведомлений
from src.notifier.notification_manager import NotificationManager
from src.notifier.telegram_notifier import TelegramNotifier

# Утилиты
from src.utils import setup_logging, get_logger, log_execution_time, log_async_execution_time, log_operation, log_async_operation

# Компоненты базы данных
from db.db_manager import DatabaseManager
from db.models import Base, Exchange, Symbol, Balance, ArbitrageOpportunity as DBArbitrageOpportunity, Trade
from db.repository import (
    ExchangeRepository, SymbolRepository, BalanceRepository, 
    ArbitrageOpportunityRepository, TradeRepository
)


class ArbitrageBot:
    """
    Основной класс, интегрирующий все компоненты бота для арбитража.
    """
    
    def __init__(self, simulation_mode: bool = True):
        """
        Инициализирует бота для арбитража.
        
        Args:
            simulation_mode: Если True, работает в режиме симуляции без реальных сделок
        """
        self.logger = get_logger(__name__)
        self.simulation_mode = simulation_mode
        
        self.logger.info(f"Initializing {settings.APP_NAME} v{settings.VERSION} in "
                       f"{'SIMULATION' if simulation_mode else 'LIVE'} mode")
        
        # Настройка компонентов
        self._setup_components()
        
    def _setup_components(self):
        """Настраивает и инициализирует все компоненты системы."""
        # Инициализируем базу данных
        self.db_manager = DatabaseManager(settings.DATABASE_URL, echo=settings.DEBUG_SQL)
        
        # Создаем таблицы, если их нет
        self.db_manager.create_tables()
        
        # Инициализируем репозитории
        self.exchange_repo = ExchangeRepository(self.db_manager)
        self.symbol_repo = SymbolRepository(self.db_manager)
        self.balance_repo = BalanceRepository(self.db_manager)
        self.opportunity_repo = ArbitrageOpportunityRepository(self.db_manager)
        self.trade_repo = TradeRepository(self.db_manager)
        
        # Инициализируем фабрику коннекторов бирж
        self.exchange_factory = ExchangeFactory(exchange_config)
        
        # Инициализируем компоненты сканирования рынка
        self.market_scanner = MarketScanner(
            exchange_factory=self.exchange_factory,
            symbols=settings.ALLOWED_PAIRS,
            update_interval=settings.MARKET_SCAN_INTERVAL,
            use_simulation=settings.USE_MARKET_SIMULATOR
        )
        
        self.data_normalizer = DataNormalizer()
        
        self.price_monitor = PriceMonitor(
            significant_change_threshold=settings.SIGNIFICANT_PRICE_CHANGE_THRESHOLD
        )
        
        # Инициализируем компоненты арбитражного движка
        self.profit_calculator = ProfitCalculator(
            exchange_fees={ex.name: ex.fee_rate for ex in exchange_config.get_enabled_exchanges()}
        )
        
        self.opportunity_finder = OpportunityFinder(
            exchange_fees={ex.name: ex.fee_rate for ex in exchange_config.get_enabled_exchanges()},
            min_profit_percentage=settings.MIN_PROFIT_PERCENTAGE
        )
        
        self.risk_manager = RiskManager(
            max_order_amount=settings.MAX_ORDER_AMOUNT,
            max_trades_per_pair=settings.MAX_TRADES_PER_PAIR,
            min_liquidity_requirement=settings.MIN_LIQUIDITY_REQUIREMENT,
            max_price_deviation=settings.MAX_PRICE_DEVIATION
        )
        
        # Настройка стратегий арбитража
        direct_strategy = DirectArbitrageStrategy()
        self.strategy_selector = StrategySelector([direct_strategy])
        
        # Инициализация уведомлений
        self.notification_manager = NotificationManager()
        
        if settings.TELEGRAM_ENABLED:
            telegram_notifier = TelegramNotifier(
                token=settings.TELEGRAM_BOT_TOKEN,
                chat_id=settings.TELEGRAM_CHAT_ID
            )
            self.notification_manager.add_notifier(telegram_notifier)
        
        # Настройка симулятора
        if self.simulation_mode:
            self.simulator = TradingSimulator(
                initial_balances=settings.INITIAL_BALANCES,
                exchange_fees={ex.name: ex.fee_rate for ex in exchange_config.get_enabled_exchanges()},
                min_profit_percentage=settings.MIN_PROFIT_PERCENTAGE,
                max_active_trades=settings.MAX_ACTIVE_TRADES,
                max_trade_duration_ms=settings.MAX_TRADE_DURATION_MS,
                emergency_stop_loss_percentage=settings.EMERGENCY_STOP_LOSS_PERCENTAGE
            )
            
            # Добавляем стратегии в симулятор
            self.simulator.add_strategy(direct_strategy)
            
            # Инициализируем статистику и визуализатор симуляции
            self.sim_stats = SimulationStats()
            self.sim_visualizer = SimulationVisualizer()
        
        self.logger.info("All components initialized successfully")
        
    async def _store_exchange_and_symbol_data(self):
        """Сохраняет информацию о биржах и торговых парах в базу данных."""
        self.logger.info("Storing exchange and symbol data...")
        
        with self.db_manager.get_session() as session:
            # Сохраняем информацию о биржах
            for exchange_config_item in exchange_config.get_enabled_exchanges():
                # Проверяем, существует ли биржа в БД
                exchange = self.exchange_repo.get_by_name(session, exchange_config_item.name)
                if not exchange:
                    # Создаем новую запись о бирже
                    exchange = self.exchange_repo.create(
                        session,
                        name=exchange_config_item.name,
                        is_active=True
                    )
                    self.logger.info(f"Added exchange to DB: {exchange_config_item.name}")
                
            # Сохраняем информацию о торговых парах
            for symbol in settings.ALLOWED_PAIRS:
                base_asset, quote_asset = symbol.split('/')
                # Проверяем, существует ли символ в БД
                db_symbol = self.symbol_repo.get_by_name(session, symbol)
                if not db_symbol:
                    # Создаем новую запись о символе
                    db_symbol = self.symbol_repo.create(
                        session,
                        name=symbol,
                        base_asset=base_asset,
                        quote_asset=quote_asset,
                        is_active=True
                    )
                    self.logger.info(f"Added symbol to DB: {symbol}")
        
        self.logger.info("Exchange and symbol data stored successfully")
    
    async def _update_balances(self):
        """Обновляет информацию о балансах в базе данных."""
        if self.simulation_mode:
            # В режиме симуляции используем виртуальные балансы
            self.logger.info("Using virtual balances in simulation mode")
            return
        
        self.logger.info("Updating exchange balances...")
        
        for exchange_config_item in exchange_config.get_enabled_exchanges():
            exchange_name = exchange_config_item.name
            connector = self.exchange_factory.get_connector(exchange_name)
            
            if not connector:
                self.logger.warning(f"Connector for {exchange_name} not available, skipping balance update")
                continue
                
            try:
                # Получаем актуальные балансы с биржи
                balances = await connector.fetch_balance()
                
                with self.db_manager.get_session() as session:
                    # Получаем ID биржи
                    exchange = self.exchange_repo.get_by_name(session, exchange_name)
                    if not exchange:
                        self.logger.warning(f"Exchange {exchange_name} not found in DB, skipping balance update")
                        continue
                    
                    # Обновляем балансы в БД
                    for asset, amounts in balances.items():
                        # Ищем существующий баланс
                        balance = self.balance_repo.get_by_exchange_and_asset(
                            session, exchange.id, asset
                        )
                        
                        if balance:
                            # Обновляем существующий баланс
                            self.balance_repo.update(
                                session,
                                balance.id,
                                free=Decimal(str(amounts.get('free', 0))),
                                locked=Decimal(str(amounts.get('locked', 0))),
                                total=Decimal(str(amounts.get('total', 0)))
                            )
                        else:
                            # Создаем новую запись о балансе
                            self.balance_repo.create(
                                session,
                                exchange_id=exchange.id,
                                asset=asset,
                                free=Decimal(str(amounts.get('free', 0))),
                                locked=Decimal(str(amounts.get('locked', 0))),
                                total=Decimal(str(amounts.get('total', 0)))
                            )
                
                self.logger.info(f"Balances for {exchange_name} updated successfully")
                
            except Exception as e:
                self.logger.error(f"Error updating balances for {exchange_name}: {str(e)}")
    
    async def _store_arbitrage_opportunity(self, opportunity: ArbitrageOpportunity):
        """
        Сохраняет арбитражную возможность в базе данных.
        
        Args:
            opportunity: Объект арбитражной возможности
            
        Returns:
            ID созданной записи или None в случае ошибки
        """
        self.logger.debug(f"Storing arbitrage opportunity: {opportunity.symbol} "
                         f"{opportunity.buy_exchange}->{opportunity.sell_exchange} "
                         f"profit: {opportunity.net_profit_percentage}%")
        
        opportunity_id = None
        
        try:
            with self.db_manager.get_session() as session:
                # Получаем ID бирж и символа
                buy_exchange = self.exchange_repo.get_by_name(session, opportunity.buy_exchange)
                sell_exchange = self.exchange_repo.get_by_name(session, opportunity.sell_exchange)
                symbol = self.symbol_repo.get_by_name(session, opportunity.symbol)
                
                if not buy_exchange or not sell_exchange or not symbol:
                    self.logger.warning("Exchange or symbol not found in DB, skipping opportunity")
                    return None
                
                # Создаем запись о возможности
                db_opportunity = self.opportunity_repo.create(
                    session,
                    buy_exchange_id=buy_exchange.id,
                    sell_exchange_id=sell_exchange.id,
                    symbol_id=symbol.id,
                    buy_price=Decimal(str(opportunity.buy_price)),
                    sell_price=Decimal(str(opportunity.sell_price)),
                    amount=Decimal(str(opportunity.volume)),
                    profit_amount=Decimal(str(opportunity.net_profit)),
                    profit_percentage=Decimal(str(opportunity.net_profit_percentage)),
                    status="detected",
                    detected_at=datetime.utcnow(),
                    additional_data={
                        "buy_exchange_fee": opportunity.buy_exchange_fee,
                        "sell_exchange_fee": opportunity.sell_exchange_fee,
                        "gross_profit": float(opportunity.gross_profit),
                        "buy_volume": float(opportunity.volume)
                    }
                )
                
                opportunity_id = db_opportunity.id
                self.logger.info(f"Arbitrage opportunity stored with ID: {opportunity_id}")
                
        except Exception as e:
            self.logger.error(f"Error storing arbitrage opportunity: {str(e)}")
        
        return opportunity_id
    
    async def _store_simulated_trade(self, trade, opportunity_id):
        """
        Сохраняет информацию о симулированной сделке в базе данных.
        
        Args:
            trade: Объект симулированной сделки
            opportunity_id: ID арбитражной возможности
        """
        self.logger.debug(f"Storing simulated trade: {trade.trade_id}")
        
        try:
            with self.db_manager.get_session() as session:
                # Создаем запись о сделке
                db_trade = self.trade_repo.create(
                    session,
                    opportunity_id=opportunity_id,
                    is_simulated=True,
                    buy_price=Decimal(str(trade.buy_price)),
                    sell_price=Decimal(str(trade.sell_price)),
                    amount=Decimal(str(trade.volume)),
                    profit_amount=Decimal(str(trade.expected_profit)),
                    profit_percentage=Decimal(str(trade.expected_profit_percentage)),
                    status=trade.status,
                    created_at=datetime.utcfromtimestamp(trade.open_timestamp / 1000),
                    additional_data={
                        "trade_id": trade.trade_id,
                        "buy_exchange": trade.buy_exchange,
                        "sell_exchange": trade.sell_exchange,
                        "strategy_name": trade.strategy_name,
                        "close_timestamp": trade.close_timestamp,
                        "actual_sell_price": float(trade.actual_sell_price) if trade.actual_sell_price else None,
                        "actual_profit": float(trade.actual_profit) if trade.actual_profit else None,
                        "close_reason": trade.close_reason
                    }
                )
                
                # Если сделка завершена, обновляем поле completed_at
                if trade.status == "closed" and trade.close_timestamp:
                    db_trade = self.trade_repo.update(
                        session,
                        db_trade.id,
                        completed_at=datetime.utcfromtimestamp(trade.close_timestamp / 1000)
                    )
                
                self.logger.info(f"Simulated trade stored with ID: {db_trade.id}")
                
        except Exception as e:
            self.logger.error(f"Error storing simulated trade: {str(e)}")
    
    @log_async_execution_time
    async def run_simulation(self, duration_seconds: int = 60):
        """
        Запускает бота в режиме симуляции на заданное время.
        
        Args:
            duration_seconds: Длительность симуляции в секундах
        """
        if not self.simulation_mode:
            self.logger.error("Cannot run simulation in live mode")
            return
        
        self.logger.info(f"Starting simulation for {duration_seconds} seconds")
        
        # Перед симуляцией сохраняем данные об используемых биржах и символах
        await self._store_exchange_and_symbol_data()
        
        # Запускаем сканер рынка
        self.market_scanner.start()
        
        # Запускаем сбор статистики
        start_time = time.time()
        end_time = start_time + duration_seconds
        iteration = 0
        opportunities_found = 0
        trades_executed = 0
        
        try:
            while time.time() < end_time:
                iteration += 1
                
                # Получаем актуальные данные рынка
                market_data_batch = await self.market_scanner.get_market_data()
                
                if not market_data_batch:
                    self.logger.warning("No market data available, waiting...")
                    await asyncio.sleep(1)
                    continue
                
                # Нормализуем данные
                normalized_data = self.data_normalizer.normalize_batch(market_data_batch)
                
                # Мониторим изменения цен
                significant_changes = self.price_monitor.process_data_batch(normalized_data)
                
                for data in normalized_data:
                    # Ищем арбитражные возможности
                    opportunities = self.opportunity_finder.find_opportunities(data)
                    opportunities_found += len(opportunities)
                    
                    # Фильтруем возможности через risk manager
                    filtered_opportunities = []
                    for opp in opportunities:
                        risk_assessment = self.risk_manager.assess_opportunity(opp, data)
                        if risk_assessment.is_acceptable:
                            filtered_opportunities.append(opp)
                            
                            # Сохраняем возможность в БД
                            await self._store_arbitrage_opportunity(opp)
                            
                            # Отправляем уведомление о возможности
                            if settings.NOTIFY_ON_OPPORTUNITY:
                                await self.notification_manager.send_notification(
                                    title="Arbitrage Opportunity",
                                    message=f"{opp.symbol}: {opp.buy_exchange}->{opp.sell_exchange}, "
                                            f"profit: {opp.net_profit_percentage:.2f}%, "
                                            f"volume: {opp.volume}",
                                    level="info"
                                )
                    
                    # Обрабатываем возможности в симуляторе
                    for opp in filtered_opportunities:
                        trade_id = self.simulator.process_opportunity(opp, data)
                        if trade_id:
                            trades_executed += 1
                            # Сохраняем информацию о симулированной сделке
                            trade = self.simulator.active_trades[trade_id]
                            await self._store_simulated_trade(trade, None)  # У нас пока нет связи с opportunity_id
                    
                    # Проверяем и закрываем активные сделки
                    closed_trade_ids = self.simulator.check_and_close_trades(data)
                    for trade_id in closed_trade_ids:
                        # Обновляем информацию о закрытой сделке
                        if trade_id in self.simulator.completed_trades:
                            trade = self.simulator.completed_trades[-1]  # Последняя добавленная
                            await self._store_simulated_trade(trade, None)
                
                # Выводим промежуточную статистику каждые 10 итераций
                if iteration % 10 == 0:
                    elapsed = time.time() - start_time
                    stats = self.simulator.get_trade_summary()["stats"]
                    self.logger.info(
                        f"Iteration {iteration}: elapsed {elapsed:.1f}s, "
                        f"opportunities: {opportunities_found}, trades: {trades_executed}, "
                        f"active: {stats['active_trades_count']}, completed: {stats['completed_trades_count']}, "
                        f"profit: {stats['total_profit']:.2f}"
                    )
                
                # Небольшая задержка между итерациями
                await asyncio.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Error during simulation: {str(e)}")
        finally:
            # Останавливаем сканер рынка
            self.market_scanner.stop()
            
            # Закрываем все активные сделки
            for trade_id in list(self.simulator.active_trades.keys()):
                if normalized_data:
                    self.simulator.close_trade(trade_id, normalized_data[-1], "simulation_end")
            
            # Вывод результатов симуляции
            duration = time.time() - start_time
            stats = self.simulator.get_trade_summary()["stats"]
            
            results_message = (
                f"\n{'='*50}\n"
                f"SIMULATION RESULTS\n"
                f"{'='*50}\n"
                f"Duration: {duration:.1f} seconds\n"
                f"Iterations: {iteration}\n"
                f"Opportunities found: {opportunities_found}\n"
                f"Trades executed: {trades_executed}\n"
                f"Completed trades: {stats['completed_trades_count']}\n"
                f"Successful trades: {stats['successful_trades']}\n"
                f"Failed trades: {stats['failed_trades']}\n"
                f"Total profit: {stats['total_profit']:.4f}\n"
                f"Success rate: {stats['successful_trades'] / max(1, stats['total_trades']) * 100:.2f}%\n"
                f"{'='*50}"
            )
            
            self.logger.info(results_message)
            
            # Отправляем итоговое уведомление
            await self.notification_manager.send_notification(
                title="Simulation Completed",
                message=results_message,
                level="info"
            )
            
            # Очистка ресурсов
            await self.exchange_factory.close_all_connections()
            self.db_manager.close()
    
    @log_async_execution_time
    async def run_live(self, duration_seconds: Optional[int] = None):
        """
        Запускает бота в реальном режиме на заданное время.
        
        Args:
            duration_seconds: Опциональная длительность работы в секундах
        """
        if self.simulation_mode:
            self.logger.error("Cannot run live mode in simulation mode")
            return
        
        self.logger.warning("Starting LIVE trading mode. REAL FUNDS WILL BE USED!")
        
        # Отправляем уведомление о начале торговли в реальном режиме
        await self.notification_manager.send_notification(
            title="LIVE Trading Started",
            message=f"Starting live trading with real funds! Duration: {duration_seconds or 'Unlimited'} seconds",
            level="critical"
        )
        
        # Перед началом сохраняем данные об используемых биржах и символах
        await self._store_exchange_and_symbol_data()
        
        # Получаем и обновляем балансы перед началом торговли
        await self._update_balances()
        
        # Запускаем сканер рынка
        self.market_scanner.start()
        
        # Инициализируем данные для статистики
        start_time = time.time()
        end_time = start_time + duration_seconds if duration_seconds else float('inf')
        iteration = 0
        opportunities_found = 0
        trades_executed = 0
        total_profit = 0.0
        
        # Инициализируем компоненты реальной торговли
        from src.trading.trade_executor import RealTradeExecutor
        from src.trading.balance_manager import BalanceManager
        from src.trading.order_manager import OrderManager
        
        # Создаем экземпляры компонентов для реальной торговли
        self.balance_manager = BalanceManager(
            exchange_factory=self.exchange_factory,
            db_manager=self.db_manager,
            balance_repo=self.balance_repo
        )
        
        self.order_manager = OrderManager(
            exchange_factory=self.exchange_factory,
            db_manager=self.db_manager,
            order_repo=None,  # Будет создан внутри OrderManager
            balance_manager=self.balance_manager
        )
        
        self.trade_executor = RealTradeExecutor(
            exchange_factory=self.exchange_factory,
            order_manager=self.order_manager,
            balance_manager=self.balance_manager,
            risk_manager=self.risk_manager
        )
        
        # Активные сделки и их статусы
        active_trades = {}
        
        try:
            self.logger.info("Starting live trading loop...")
            
            while time.time() < end_time:
                iteration += 1
                
                # Проверяем наличие средств на балансах
                balances_ok = await self.balance_manager.check_sufficient_balances(settings.ALLOWED_PAIRS)
                if not balances_ok:
                    self.logger.warning("Insufficient balances for trading. Checking again in 60 seconds.")
                    await self.notification_manager.send_notification(
                        title="Insufficient Balances",
                        message="Not enough funds for trading. Please check exchange balances.",
                        level="warning"
                    )
                    await asyncio.sleep(60)  # Ждем минуту перед повторной проверкой
                    continue
                
                # Получаем актуальные данные рынка
                market_data_batch = await self.market_scanner.get_market_data()
                
                if not market_data_batch:
                    self.logger.warning("No market data available, waiting...")
                    await asyncio.sleep(1)
                    continue
                
                # Нормализуем данные
                normalized_data = self.data_normalizer.normalize_batch(market_data_batch)
                
                # Мониторим изменения цен
                significant_changes = self.price_monitor.process_data_batch(normalized_data)
                
                for data in normalized_data:
                    # Ищем арбитражные возможности
                    opportunities = self.opportunity_finder.find_opportunities(data)
                    opportunities_found += len(opportunities)
                    
                    # Фильтруем возможности через risk manager
                    filtered_opportunities = []
                    for opp in opportunities:
                        risk_assessment = self.risk_manager.assess_opportunity(opp, data)
                        if risk_assessment.is_acceptable:
                            # Корректируем объем сделки согласно оценке риска
                            opp.volume = min(opp.volume, risk_assessment.adjusted_volume)  
                            filtered_opportunities.append(opp)
                            
                            # Сохраняем возможность в БД
                            opportunity_id = await self._store_arbitrage_opportunity(opp)
                            
                            # Отправляем уведомление о возможности
                            if settings.NOTIFY_ON_OPPORTUNITY:
                                await self.notification_manager.send_notification(
                                    title="Live Arbitrage Opportunity",
                                    message=f"LIVE MODE: {opp.symbol}: {opp.buy_exchange}->{opp.sell_exchange}, "
                                            f"profit: {opp.net_profit_percentage:.2f}%, "
                                            f"volume: {opp.volume}",
                                    level="info"
                                )
                    
                    # Проверяем активные сделки и обновляем их статус
                    for trade_id, trade_info in list(active_trades.items()):
                        trade_status = await self.trade_executor.check_trade_status(trade_id)
                        
                        # Если сделка завершена, обновляем статистику
                        if trade_status.is_completed:
                            if trade_status.is_successful:
                                total_profit += trade_status.actual_profit
                                self.logger.info(
                                    f"Trade {trade_id} completed successfully. "
                                    f"Profit: {trade_status.actual_profit:.4f} USD"
                                )
                                
                                await self.notification_manager.send_notification(
                                    title="Trade Completed",
                                    message=f"Trade {trade_id} completed successfully.\n"
                                            f"Profit: {trade_status.actual_profit:.4f} USD",
                                    level="success"
                                )
                            else:
                                self.logger.warning(
                                    f"Trade {trade_id} failed. "
                                    f"Reason: {trade_status.failure_reason}"
                                )
                                
                                await self.notification_manager.send_notification(
                                    title="Trade Failed",
                                    message=f"Trade {trade_id} failed.\n"
                                            f"Reason: {trade_status.failure_reason}",
                                    level="error"
                                )
                            
                            # Удаляем завершенную сделку из активных
                            del active_trades[trade_id]
                    
                    # Проверяем, не превышен ли лимит активных сделок
                    if len(active_trades) >= settings.MAX_ACTIVE_TRADES:
                        self.logger.info(f"Maximum active trades limit reached ({settings.MAX_ACTIVE_TRADES}). Skipping new opportunities.")
                        continue
                    
                    # Исполняем новые возможности
                    for opp in filtered_opportunities:
                        # Проверяем, достаточно ли средств для сделки
                        sufficient_funds = await self.balance_manager.check_sufficient_funds_for_opportunity(opp)
                        
                        if not sufficient_funds:
                            self.logger.warning(
                                f"Insufficient funds for opportunity: {opp.symbol} "
                                f"{opp.buy_exchange}->{opp.sell_exchange}"
                            )
                            continue
                        
                        # Исполняем сделку
                        try:
                            trade_id = await self.trade_executor.execute_opportunity(opp, data)
                            
                            if trade_id:
                                trades_executed += 1
                                active_trades[trade_id] = {
                                    "opportunity": opp,
                                    "started_at": time.time()
                                }
                                
                                self.logger.info(
                                    f"Started execution of trade {trade_id}: "
                                    f"{opp.symbol} {opp.buy_exchange}->{opp.sell_exchange} "
                                    f"with expected profit: {opp.net_profit_percentage:.2f}%"
                                )
                                
                                await self.notification_manager.send_notification(
                                    title="Trade Started",
                                    message=f"Started execution of trade {trade_id}:\n"
                                            f"{opp.symbol} {opp.buy_exchange}->{opp.sell_exchange}\n"
                                            f"Expected profit: {opp.net_profit_percentage:.2f}%",
                                    level="info"
                                )
                                
                                # Если достигли лимита активных сделок, прекращаем добавлять новые
                                if len(active_trades) >= settings.MAX_ACTIVE_TRADES:
                                    break
                        except Exception as e:
                            self.logger.error(f"Error executing opportunity: {str(e)}")
                
                # Обновляем балансы каждые 10 итераций
                if iteration % 10 == 0:
                    await self._update_balances()
                
                # Выводим промежуточную статистику каждые 10 итераций
                if iteration % 10 == 0:
                    elapsed = time.time() - start_time
                    self.logger.info(
                        f"Iteration {iteration}: elapsed {elapsed:.1f}s, "
                        f"opportunities: {opportunities_found}, trades: {trades_executed}, "
                        f"active: {len(active_trades)}, profit: {total_profit:.4f} USD"
                    )
                
                # Небольшая задержка между итерациями
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            self.logger.warning("Live trading interrupted by user")
        except Exception as e:
            self.logger.error(f"Error during live trading: {str(e)}")
            self.logger.exception("Stack trace:")
            
            # Отправляем уведомление о критической ошибке
            await self.notification_manager.send_notification(
                title="Critical Error",
                message=f"Live trading stopped due to error: {str(e)}",
                level="critical"
            )
        finally:
            # Останавливаем сканер рынка
            self.market_scanner.stop()
            
            # Пытаемся закрыть все активные сделки
            self.logger.info("Attempting to close all active trades...")
            for trade_id in list(active_trades.keys()):
                try:
                    await self.trade_executor.close_trade(trade_id, "shutdown")
                    self.logger.info(f"Successfully closed trade {trade_id} during shutdown")
                except Exception as e:
                    self.logger.error(f"Failed to close trade {trade_id} during shutdown: {str(e)}")
            
            # Вывод результатов торговли
            duration = time.time() - start_time
            
            results_message = (
                f"\n{'='*50}\n"
                f"LIVE TRADING RESULTS\n"
                f"{'='*50}\n"
                f"Duration: {duration:.1f} seconds\n"
                f"Iterations: {iteration}\n"
                f"Opportunities found: {opportunities_found}\n"
                f"Trades executed: {trades_executed}\n"
                f"Total profit: {total_profit:.4f} USD\n"
                f"{'='*50}"
            )
            
            self.logger.info(results_message)
            
            # Отправляем итоговое уведомление
            await self.notification_manager.send_notification(
                title="Live Trading Completed",
                message=results_message,
                level="info"
            )
            
            # Очистка ресурсов
            await self.exchange_factory.close_all_connections()
            self.db_manager.close()


@log_execution_time
def main():
    """Основная функция приложения."""
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description=f'{settings.APP_NAME} - Crypto Arbitrage Bot')
    parser.add_argument('--simulation', action='store_true', help='Run in simulation mode')
    parser.add_argument('--live', action='store_true', help='Run in live trading mode')
    parser.add_argument('--duration', type=int, default=60, help='Run duration in seconds')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--version', action='store_true', help='Show version')
    
    args = parser.parse_args()
    
    # Показываем версию и выходим, если запрошено
    if args.version:
        print(f"{settings.APP_NAME} v{settings.VERSION}")
        return 0
    
    # Настройка логирования с уровнем в зависимости от флага debug
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level=log_level)
    
    logger = get_logger(__name__)
    
    # Определяем режим работы (симуляция по умолчанию)
    simulation_mode = not args.live
    
    if args.simulation and args.live:
        logger.error("Cannot specify both --simulation and --live. Defaulting to simulation mode.")
        simulation_mode = True
    
    # Выводим информацию о настройках приложения
    logger.info(f"Starting {settings.APP_NAME} v{settings.VERSION}")
    logger.info(f"Operation mode: {'SIMULATION' if simulation_mode else 'LIVE'}")
    logger.info(f"Run duration: {args.duration} seconds")
    logger.info(f"Min profit percentage: {settings.MIN_PROFIT_PERCENTAGE}%")
    logger.info(f"Max order amount: {settings.MAX_ORDER_AMOUNT} USD")
    logger.info(f"Allowed pairs: {', '.join(settings.ALLOWED_PAIRS)}")
    
    # Проверка настроек бирж
    enabled_exchanges = exchange_config.get_enabled_exchanges()
    logger.info(f"Enabled exchanges: {len(enabled_exchanges)}")
    
    for exchange in enabled_exchanges:
        is_configured = exchange_config.is_exchange_configured(exchange.name)
        status = "configured" if is_configured else "not configured"
        logger.info(f"Exchange {exchange.name.upper()}: {status}")
    
    try:
        # Создаём экземпляр бота
        bot = ArbitrageBot(simulation_mode=simulation_mode)
        
        # Запускаем бота в выбранном режиме
        if simulation_mode:
            asyncio.run(bot.run_simulation(args.duration))
        else:
            asyncio.run(bot.run_live(args.duration))
        
        logger.info("Application completed successfully")
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        logger.exception("Stack trace:")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())