"""
Модели данных для SQLAlchemy.
Определяет структуру таблиц базы данных.
"""

import enum
import datetime
from datetime import timezone
from sqlalchemy import Column, Integer, Float, String, ForeignKey, DateTime, Enum, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from decimal import Decimal as PyDecimal
from sqlalchemy import Numeric # Или DECIMAL
from sqlalchemy import Index

# Создаем базовый класс для всех моделей
Base = declarative_base()


class ArbitrageStatus(enum.Enum):
    """Статусы арбитражных возможностей."""
    DETECTED = "detected"      # Обнаружена
    SIMULATED = "simulated"    # Выполнена симуляция
    EXECUTED = "executed"      # Исполнена реальная сделка
    FAILED = "failed"          # Не удалось исполнить
    IGNORED = "ignored"        # Проигнорирована


class OrderStatus(enum.Enum):
    """Статусы ордеров."""
    PENDING = "pending"        # Ожидает исполнения
    PARTIALLY_FILLED = "partially_filled"  # Частично исполнен
    FILLED = "filled"          # Полностью исполнен
    CANCELED = "canceled"      # Отменен
    REJECTED = "rejected"      # Отклонен
    EXPIRED = "expired"        # Истек срок действия


class TradeStatus(enum.Enum):
    """Статусы ордеров."""
    PENDING = "pending"        # Ожидает исполнения
    PARTIALLY_FILLED = "partially_filled"  # Частично исполнен
    FILLED = "filled"          # Полностью исполнен
    CANCELED = "canceled"      # Отменен
    REJECTED = "rejected"      # Отклонен
    EXPIRED = "expired"        # Истек срок действия


class OrderType(enum.Enum):
    """Типы ордеров."""
    MARKET = "market"          # Рыночный ордер
    LIMIT = "limit"            # Лимитный ордер


class OrderSide(enum.Enum):
    """Стороны ордера."""
    BUY = "buy"                # Покупка
    SELL = "sell"              # Продажа


class Exchange(Base):
    """Модель биржи."""
    __tablename__ = "exchanges"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False, unique=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.datetime.now(timezone.utc), onupdate=datetime.datetime.now(timezone.utc))
    
    # Отношения
    balances = relationship("Balance", back_populates="exchange", cascade="all, delete-orphan")
    buy_opportunities = relationship("ArbitrageOpportunity", foreign_keys="ArbitrageOpportunity.buy_exchange_id", back_populates="buy_exchange")
    sell_opportunities = relationship("ArbitrageOpportunity", foreign_keys="ArbitrageOpportunity.sell_exchange_id", back_populates="sell_exchange")
    orders = relationship("Order", back_populates="exchange", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Exchange(id={self.id}, name='{self.name}', is_active={self.is_active})>"


class Symbol(Base):
    """Модель торговой пары."""
    __tablename__ = "symbols"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False, unique=True)
    base_asset = Column(String(20), nullable=False)  # Например, BTC в паре BTC/USDT
    quote_asset = Column(String(20), nullable=False)  # Например, USDT в паре BTC/USDT
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.datetime.now(timezone.utc), onupdate=datetime.datetime.now(timezone.utc))
    
    # Отношения
    opportunities = relationship("ArbitrageOpportunity", back_populates="symbol", cascade="all, delete-orphan")
    orders = relationship("Order", back_populates="symbol", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Symbol(id={self.id}, name='{self.name}', base='{self.base_asset}', quote='{self.quote_asset}')>"


class Balance(Base):
    """Модель баланса на бирже."""
    __tablename__ = "balances"
    
    id = Column(Integer, primary_key=True)
    exchange_id = Column(Integer, ForeignKey("exchanges.id"), nullable=False)
    asset = Column(String(20), nullable=False)
    free = Column(Numeric(precision=18, scale=8), nullable=False, default=PyDecimal("0.0"))
    locked = Column(Numeric(precision=18, scale=8), nullable=False, default=PyDecimal("0.0"))
    total = Column(Numeric(precision=18, scale=8), nullable=False, default=PyDecimal("0.0"))
    updated_at = Column(DateTime, default=datetime.datetime.now(timezone.utc), onupdate=datetime.datetime.now(timezone.utc))
    
    # Отношения
    exchange = relationship("Exchange", back_populates="balances")
    
    __table_args__ = (
        # Создаем индекс для быстрого поиска по бирже и активу
        Index('ix_balance_exchange_asset', 'exchange_id', 'asset', unique=True),  # Уникальный индекс лучше здесь
        {'sqlite_autoincrement': True},
    )
    
    def __repr__(self):
        return f"<Balance(id={self.id}, exchange='{self.exchange.name if self.exchange else None}', asset='{self.asset}', free={self.free}, locked={self.locked})>"


class ArbitrageOpportunity(Base):
    """Модель арбитражной возможности."""
    __tablename__ = "arbitrage_opportunities"
    
    id = Column(Integer, primary_key=True)
    buy_exchange_id = Column(Integer, ForeignKey("exchanges.id"), nullable=False, index=True)
    sell_exchange_id = Column(Integer, ForeignKey("exchanges.id"), nullable=False, index=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False, index=True)
    buy_price = Column(Numeric(precision=18, scale=8), nullable=False)
    sell_price = Column(Numeric(precision=18, scale=8), nullable=False)
    amount = Column(Numeric(precision=18, scale=8), nullable=False)  # Объем в базовой валюте
    profit_amount = Column(Numeric(precision=18, scale=8), nullable=False)  # Прибыль в котировочной валюте
    profit_percentage = Column(Numeric(precision=18, scale=8), nullable=False)  # Процент прибыли
    status = Column(Enum(ArbitrageStatus), nullable=False, default=ArbitrageStatus.DETECTED, index=True)
    detected_at = Column(DateTime, default=datetime.datetime.now(timezone.utc), index=True)
    executed_at = Column(DateTime)  # Время выполнения (если было)
    error_message = Column(Text)  # Сообщение об ошибке (если была)
    additional_data = Column(JSON)  # Дополнительные данные в формате JSON
    
    # Отношения
    buy_exchange = relationship("Exchange", foreign_keys=[buy_exchange_id], back_populates="buy_opportunities")
    sell_exchange = relationship("Exchange", foreign_keys=[sell_exchange_id], back_populates="sell_opportunities")
    symbol = relationship("Symbol", back_populates="opportunities")
    trades = relationship("Trade", back_populates="opportunity", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ArbitrageOpportunity(id={self.id}, buy_exchange='{self.buy_exchange.name if self.buy_exchange else None}', " \
               f"sell_exchange='{self.sell_exchange.name if self.sell_exchange else None}', " \
               f"symbol='{self.symbol.name if self.symbol else None}', profit={self.profit_percentage}%, status={self.status})>"


class Order(Base):
    """Модель ордера."""
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True)
    exchange_id = Column(Integer, ForeignKey("exchanges.id"), nullable=False, index=True)
    symbol_id = Column(Integer, ForeignKey("symbols.id"), nullable=False, index=True)
    trade_id = Column(Integer, ForeignKey("trades.id"), index=True)
    order_type = Column(Enum(OrderType), nullable=False)
    side = Column(Enum(OrderSide), nullable=False)
    amount = Column(Numeric(precision=18, scale=8), nullable=False)
    price = Column(Numeric(precision=18, scale=8), nullable=False)  # Цена (для лимитных ордеров)
    filled_amount = Column(Numeric(precision=18, scale=8), nullable=False)  # Исполненный объем
    average_price = Column(Numeric(precision=18, scale=8), nullable=False)  # Средняя цена исполнения
    status = Column(Enum(OrderStatus), nullable=False, default=OrderStatus.PENDING, index=True)
    exchange_order_id = Column(String(100))  # ID ордера на бирже
    created_at = Column(DateTime, default=datetime.datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.datetime.now(timezone.utc), onupdate=datetime.datetime.now(timezone.utc))
    executed_at = Column(DateTime)  # Время исполнения (если было)
    error_message = Column(Text)  # Сообщение об ошибке (если была)
    additional_data = Column(JSON)  # Дополнительные данные в формате JSON
    
    # Отношения
    exchange = relationship("Exchange", back_populates="orders")
    symbol = relationship("Symbol", back_populates="orders")
    trade = relationship("Trade", back_populates="orders")
    
    def __repr__(self):
        return f"<Order(id={self.id}, exchange='{self.exchange.name if self.exchange else None}', " \
               f"symbol='{self.symbol.name if self.symbol else None}', " \
               f"side={self.side}, amount={self.amount}, status={self.status})>"


class Trade(Base):
    """Модель торговой операции (арбитражной сделки)."""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True)
    opportunity_id = Column(Integer, ForeignKey("arbitrage_opportunities.id"))
    is_simulated = Column(Boolean, default=False)  # Флаг симуляции
    buy_price = Column(Numeric(precision=18, scale=8), nullable=False)
    sell_price = Column(Numeric(precision=18, scale=8), nullable=False)
    amount = Column(Numeric(precision=18, scale=8), nullable=False)  # Объем в базовой валюте
    profit_amount = Column(Numeric(precision=18, scale=8), nullable=False)  # Прибыль в котировочной валюте
    profit_percentage = Column(Numeric(precision=18, scale=8), nullable=False)  # Процент прибыли
    status = Column(Enum(TradeStatus), nullable=False, default=OrderStatus.PENDING, index=True)
    created_at = Column(DateTime, default=datetime.datetime.now(timezone.utc))
    completed_at = Column(DateTime)  # Время завершения (если было)
    error_message = Column(Text)  # Сообщение об ошибке (если была)
    additional_data = Column(JSON)  # Дополнительные данные в формате JSON
    
    # Отношения
    opportunity = relationship("ArbitrageOpportunity", back_populates="trades")
    orders = relationship("Order", back_populates="trade", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Trade(id={self.id}, amount={self.amount}, profit={self.profit_percentage}%, " \
               f"status='{self.status}', is_simulated={self.is_simulated})>"


class Config(Base):
    """Модель конфигурации и настроек."""
    __tablename__ = "configs"
    
    id = Column(Integer, primary_key=True)
    key = Column(String(100), nullable=False, unique=True)
    value = Column(JSON, nullable=False)  # Значение в формате JSON
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.datetime.now(timezone.utc), onupdate=datetime.datetime.now(timezone.utc))
    
    def __repr__(self):
        return f"<Config(id={self.id}, key='{self.key}')>"


class LogEntry(Base):
    """Модель записи лога в базе данных."""
    __tablename__ = "log_entries"
    
    id = Column(Integer, primary_key=True)
    level = Column(String(20), nullable=False, index=True)  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    module = Column(String(100), nullable=False)
    function = Column(String(100))
    message = Column(Text, nullable=False)
    exception = Column(Text)  # Трассировка исключения, если есть
    created_at = Column(DateTime, default=datetime.datetime.now(timezone.utc), index=True)
    additional_data = Column(JSON)  # Дополнительные данные в формате JSON
    
    def __repr__(self):
        return f"<LogEntry(id={self.id}, level='{self.level}', module='{self.module}', created_at={self.created_at})>"
