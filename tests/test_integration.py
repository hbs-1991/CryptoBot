"""
Тесты для проверки интеграции компонентов MVP.
"""

import os
import sys
import unittest
import asyncio
from decimal import Decimal
from datetime import datetime
import logging
from unittest.mock import patch

# Добавляем корневую директорию проекта в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.arbitrage_engine.opportunity_finder import OpportunityFinder
from src.arbitrage_engine.profit_calculator import ArbitrageOpportunity
from src.arbitrage_engine.risk_manager import RiskManager
from src.simulation.simulation import TradingSimulator
from src.arbitrage_engine.market_strategy import DirectArbitrageStrategy
from src.utils import setup_logging, get_logger

from db.db_manager import DatabaseManager
from db.repository import (
    ExchangeRepository, SymbolRepository, BalanceRepository, 
    ArbitrageOpportunityRepository, TradeRepository
)

# Настройка логирования для тестов
setup_logging(log_level=logging.INFO)
logger = get_logger("test_integration")


class MockMarketData:
    """Мок-класс для рыночных данных."""
    
    def __init__(self, symbol, tickers):
        self.symbol = symbol
        self.tickers = tickers
        self.timestamp = int(datetime.now().timestamp() * 1000)
        self.order_books = {}


class MockTicker:
    """Мок-класс для тикеров."""
    
    def __init__(self, symbol, exchange, bid, ask, last, volume=1000.0):
        self.symbol = symbol
        self.exchange = exchange
        self.bid = Decimal(str(bid))
        self.ask = Decimal(str(ask))
        self.last = Decimal(str(last))
        self.volume = Decimal(str(volume))


class TestIntegration(unittest.TestCase):
    """Тесты для проверки интеграции компонентов MVP."""
    
    @classmethod
    def setUpClass(cls):
        """Настройка перед запуском всех тестов."""
        # Создаем тестовую базу данных в памяти
        cls.db_manager = DatabaseManager("sqlite:///:memory:", echo=False)
        cls.db_manager.create_tables()
        
        # Инициализируем репозитории
        cls.exchange_repo = ExchangeRepository(cls.db_manager)
        cls.symbol_repo = SymbolRepository(cls.db_manager)
        cls.balance_repo = BalanceRepository(cls.db_manager)
        cls.opportunity_repo = ArbitrageOpportunityRepository(cls.db_manager)
        cls.trade_repo = TradeRepository(cls.db_manager)
        
        # Заполняем базу тестовыми данными
        with cls.db_manager.get_session() as session:
            # Добавляем биржи
            binance = cls.exchange_repo.create(session, name="binance", is_active=True)
            kucoin = cls.exchange_repo.create(session, name="kucoin", is_active=True)
            
            # Добавляем символы
            btc_usdt = cls.symbol_repo.create(
                session, name="BTC/USDT", base_asset="BTC", quote_asset="USDT", is_active=True
            )
            eth_usdt = cls.symbol_repo.create(
                session, name="ETH/USDT", base_asset="ETH", quote_asset="USDT", is_active=True
            )
            
            # Добавляем балансы
            cls.balance_repo.create(
                session, exchange_id=binance.id, asset="USDT", free=Decimal("10000"), 
                locked=Decimal("0"), total=Decimal("10000")
            )
            cls.balance_repo.create(
                session, exchange_id=binance.id, asset="BTC", free=Decimal("1"), 
                locked=Decimal("0"), total=Decimal("1")
            )
            cls.balance_repo.create(
                session, exchange_id=kucoin.id, asset="USDT", free=Decimal("10000"), 
                locked=Decimal("0"), total=Decimal("10000")
            )
            cls.balance_repo.create(
                session, exchange_id=kucoin.id, asset="BTC", free=Decimal("1"), 
                locked=Decimal("0"), total=Decimal("1")
            )
    
    @classmethod
    def tearDownClass(cls):
        """Очистка после всех тестов."""
        cls.db_manager.drop_tables()
        cls.db_manager.close()
    
    def test_db_integration(self):
        """Тест интеграции с базой данных."""
        with self.db_manager.get_session() as session:
            # Проверяем, что биржи добавлены
            exchanges = self.exchange_repo.get_all(session)
            self.assertEqual(len(exchanges), 2)
            
            # Проверяем, что символы добавлены
            symbols = self.symbol_repo.get_all(session)
            self.assertEqual(len(symbols), 2)
            
            # Проверяем, что балансы добавлены
            balances = self.balance_repo.get_all(session)
            self.assertEqual(len(balances), 4)
    
    def test_opportunity_finder_integration(self):
        """Тест интеграции поиска арбитражных возможностей."""
        # Создаем тестовые данные
        tickers = {
            "binance": MockTicker("BTC/USDT", "binance", 40000.0, 40100.0, 40050.0),
            "kucoin": MockTicker("BTC/USDT", "kucoin", 40500.0, 40600.0, 40550.0)
        }
        
        market_data = MockMarketData("BTC/USDT", tickers)
        
        # Создаем компоненты
        opportunity_finder = OpportunityFinder(
            exchange_fees={"binance": 0.1, "kucoin": 0.1},
            min_profit_percentage=0.5
        )
        
        # Ищем возможности
        opportunities = opportunity_finder.find_opportunities(market_data)
        
        # Проверяем, что нашли арбитражные возможности
        self.assertTrue(len(opportunities) > 0)
        
        # Проверяем параметры найденной возможности
        opportunity = opportunities[0]
        self.assertEqual(opportunity.symbol, "BTC/USDT")
        self.assertIn(opportunity.buy_exchange, ["binance", "kucoin"])
        self.assertIn(opportunity.sell_exchange, ["binance", "kucoin"])
        self.assertNotEqual(opportunity.buy_exchange, opportunity.sell_exchange)
    
    def test_risk_manager_integration(self):
        """Тест интеграции с менеджером рисков."""
        # Создаем тестовые данные и возможность
        opportunity = ArbitrageOpportunity(
            symbol="BTC/USDT",
            buy_exchange="binance",
            sell_exchange="kucoin",
            buy_price=Decimal("40000.0"),
            sell_price=Decimal("40500.0"),
            volume=Decimal("0.1"),
            buy_exchange_fee=Decimal("0.1"),
            sell_exchange_fee=Decimal("0.1")
        )
        
        tickers = {
            "binance": MockTicker("BTC/USDT", "binance", 40000.0, 40100.0, 40050.0, 100.0),
            "kucoin": MockTicker("BTC/USDT", "kucoin", 40500.0, 40600.0, 40550.0, 100.0)
        }
        
        market_data = MockMarketData("BTC/USDT", tickers)
        
        # Создаем менеджер рисков
        risk_manager = RiskManager(
            max_order_amount=10000.0,
            max_trades_per_pair=5,
            min_liquidity_requirement=1.0,
            max_price_deviation=5.0
        )
        
        # Оцениваем риски
        assessment = risk_manager.assess_opportunity(opportunity, market_data)
        
        # Проверяем, что оценка прошла успешно
        self.assertTrue(assessment.is_acceptable)
    
    def test_simulation_integration(self):
        """Тест интеграции с симулятором торговли."""
        # Создаем тестовые данные
        tickers = {
            "binance": MockTicker("BTC/USDT", "binance", 40000.0, 40100.0, 40050.0),
            "kucoin": MockTicker("BTC/USDT", "kucoin", 40500.0, 40600.0, 40550.0)
        }
        
        market_data = MockMarketData("BTC/USDT", tickers)
        
        # Создаем симулятор
        initial_balances = {
            "binance": {"USDT": 10000.0, "BTC": 1.0},
            "kucoin": {"USDT": 10000.0, "BTC": 1.0}
        }
        
        simulator = TradingSimulator(
            initial_balances=initial_balances,
            exchange_fees={"binance": 0.1, "kucoin": 0.1},
            min_profit_percentage=0.5
        )
        
        # Добавляем стратегию
        simulator.add_strategy(DirectArbitrageStrategy())
        
        # Создаем арбитражную возможность
        opportunity = ArbitrageOpportunity(
            symbol="BTC/USDT",
            buy_exchange="binance",
            sell_exchange="kucoin",
            buy_price=Decimal("40000.0"),
            sell_price=Decimal("40500.0"),
            volume=Decimal("0.1"),
            buy_exchange_fee=Decimal("0.1"),
            sell_exchange_fee=Decimal("0.1")
        )
        
        # Обрабатываем возможность
        trade_id = simulator.process_opportunity(opportunity, market_data)
        
        # Проверяем, что сделка создана
        self.assertIsNotNone(trade_id)
        self.assertIn(trade_id, simulator.active_trades)
        
        # Проверяем, что баланс изменился
        binance_usdt = simulator.virtual_balance.get_balance("binance", "USDT")
        self.assertLess(binance_usdt, Decimal("10000.0"))
        
        # Пробуем закрыть сделку
        closed = simulator.close_trade(trade_id, market_data, "test")
        
        # Проверяем, что сделка закрыта успешно
        self.assertTrue(closed)
        self.assertNotIn(trade_id, simulator.active_trades)
        self.assertEqual(len(simulator.completed_trades), 1)
    
    def test_store_opportunity_in_db(self):
        """Тест сохранения арбитражной возможности в БД."""
        # Создаем арбитражную возможность
        opportunity = ArbitrageOpportunity(
            symbol="BTC/USDT",
            buy_exchange="binance",
            sell_exchange="kucoin",
            buy_price=Decimal("40000.0"),
            sell_price=Decimal("40500.0"),
            volume=Decimal("0.1"),
            buy_exchange_fee=Decimal("0.1"),
            sell_exchange_fee=Decimal("0.1")
        )
        
        with self.db_manager.get_session() as session:
            # Получаем ID бирж и символа
            buy_exchange = self.exchange_repo.get_by_name(session, opportunity.buy_exchange)
            sell_exchange = self.exchange_repo.get_by_name(session, opportunity.sell_exchange)
            symbol = self.symbol_repo.get_by_name(session, opportunity.symbol)
            
            # Создаем запись о возможности
            db_opportunity = self.opportunity_repo.create(
                session,
                buy_exchange_id=buy_exchange.id,
                sell_exchange_id=sell_exchange.id,
                symbol_id=symbol.id,
                buy_price=opportunity.buy_price,
                sell_price=opportunity.sell_price,
                amount=opportunity.volume,
                profit_amount=opportunity.net_profit,
                profit_percentage=opportunity.net_profit_percentage,
                status="detected",
                detected_at=datetime.utcnow()
            )
            
            # Проверяем, что запись создана
            self.assertIsNotNone(db_opportunity.id)
            
            # Проверяем, что можем получить ее из БД
            stored_opportunity = self.opportunity_repo.get_by_id(session, db_opportunity.id)
            self.assertIsNotNone(stored_opportunity)
            self.assertEqual(stored_opportunity.buy_price, opportunity.buy_price)
            self.assertEqual(stored_opportunity.sell_price, opportunity.sell_price)
    
    @patch('src.main.ArbitrageBot._update_balances')
    @patch('src.main.MarketScanner.get_market_data')
    async def test_main_class_integration(self, mock_get_market_data, mock_update_balances):
        """Тест интеграции основного класса ArbitrageBot."""
        # Импортируем класс ArbitrageBot
        from main import ArbitrageBot
        
        # Создаем мок-данные для сканера рынка
        tickers = {
            "binance": MockTicker("BTC/USDT", "binance", 40000.0, 40100.0, 40050.0),
            "kucoin": MockTicker("BTC/USDT", "kucoin", 40500.0, 40600.0, 40550.0)
        }
        market_data = MockMarketData("BTC/USDT", tickers)
        
        # Настраиваем моки
        mock_get_market_data.return_value = [market_data]
        mock_update_balances.return_value = None
        
        # Создаем бота в режиме симуляции
        bot = ArbitrageBot(simulation_mode=True)
        
        # Запускаем симуляцию на короткое время для тестирования
        await bot.run_simulation(duration_seconds=1)
        
        # Проверяем, что мок-методы были вызваны
        self.assertTrue(mock_get_market_data.called)
        
        # Проверяем, что данные были сохранены в БД (биржи и символы)
        with bot.db_manager.get_session() as session:
            exchanges = bot.exchange_repo.get_all(session)
            self.assertTrue(len(exchanges) > 0)
            
            symbols = bot.symbol_repo.get_all(session)
            self.assertTrue(len(symbols) > 0)


# Функция для запуска асинхронных тестов
def run_async_test(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


if __name__ == "__main__":
    unittest.main()
