"""
Нормализатор данных с разных бирж.
Приводит данные с разных бирж к единому формату.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from pydantic import BaseModel, Field

from src.data_fetching.exchange_connector import TickerData, OrderBookData
from src.data_fetching.market_scanner import MarketData, ScanResult


class NormalizedTicker(BaseModel):
    """Нормализованные данные тикера."""
    symbol: str
    bid: Decimal
    ask: Decimal
    spread: Decimal
    spread_percentage: Decimal
    mid_price: Decimal
    last: Decimal
    volume: Decimal
    exchange: str
    timestamp: int


class NormalizedOrderBook(BaseModel):
    """Нормализованные данные стакана ордеров."""
    symbol: str
    exchange: str
    timestamp: int
    # Форматированные пары [цена, объем] для bid и ask
    bids: List[List[Decimal]]  # отсортированные по убыванию цены
    asks: List[List[Decimal]]  # отсортированные по возрастанию цены
    # Кумулятивный объем для каждого уровня цены
    cumulative_bids: List[List[Decimal]]  # [цена, накопленный объем]
    cumulative_asks: List[List[Decimal]]  # [цена, накопленный объем]
    
    @property
    def best_bid(self) -> Optional[Decimal]:
        """Возвращает лучшую цену покупки."""
        return self.bids[0][0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[Decimal]:
        """Возвращает лучшую цену продажи."""
        return self.asks[0][0] if self.asks else None
    
    @property
    def best_bid_volume(self) -> Optional[Decimal]:
        """Возвращает объем по лучшей цене покупки."""
        return self.bids[0][1] if self.bids else None
    
    @property
    def best_ask_volume(self) -> Optional[Decimal]:
        """Возвращает объем по лучшей цене продажи."""
        return self.asks[0][1] if self.asks else None
    
    @property
    def spread(self) -> Optional[Decimal]:
        """Возвращает спред между лучшими ценами покупки и продажи."""
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid
    
    @property
    def spread_percentage(self) -> Optional[Decimal]:
        """Возвращает спред в процентах."""
        if self.best_bid is None or self.best_ask is None or self.best_bid == 0:
            return None
        return (self.best_ask - self.best_bid) / self.best_bid * Decimal('100')
    
    def get_liquidity_at_price(self, price: Decimal) -> Tuple[Decimal, Decimal]:
        """
        Возвращает объем покупки и продажи по указанной цене.
        
        Args:
            price: Цена для проверки ликвидности
            
        Returns:
            Кортеж (объем_покупки, объем_продажи)
        """
        bid_volume = Decimal('0')
        ask_volume = Decimal('0')
        
        # Проверяем объем покупки
        for bid_price, bid_vol in self.bids:
            if bid_price >= price:
                bid_volume += bid_vol
            else:
                break
        
        # Проверяем объем продажи
        for ask_price, ask_vol in self.asks:
            if ask_price <= price:
                ask_volume += ask_vol
            else:
                break
        
        return bid_volume, ask_volume


class NormalizedMarketData(BaseModel):
    """Нормализованные рыночные данные для одного символа со всех бирж."""
    symbol: str
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))
    tickers: Dict[str, NormalizedTicker] = Field(default_factory=dict)
    order_books: Dict[str, NormalizedOrderBook] = Field(default_factory=dict)
    
    @property
    def exchanges(self) -> List[str]:
        """Возвращает список бирж, с которых получены данные."""
        return list(set(self.tickers.keys()) | set(self.order_books.keys()))
    
    @property
    def best_bid_exchange(self) -> Optional[str]:
        """Возвращает биржу с лучшей ценой покупки."""
        best_price = Decimal('-Infinity')
        best_exchange = None
        
        for exchange, ticker in self.tickers.items():
            if ticker.bid > best_price:
                best_price = ticker.bid
                best_exchange = exchange
        
        return best_exchange
    
    @property
    def best_ask_exchange(self) -> Optional[str]:
        """Возвращает биржу с лучшей ценой продажи."""
        best_price = Decimal('Infinity')
        best_exchange = None
        
        for exchange, ticker in self.tickers.items():
            if ticker.ask < best_price and ticker.ask > 0:
                best_price = ticker.ask
                best_exchange = exchange
        
        return best_exchange
    
    def get_price(self, exchange: str, symbol: str, price_type: str = "last") -> Optional[Decimal]:
        """
        Возвращает цену указанного типа для указанной биржи.
        
        Args:
            exchange: Название биржи
            symbol: Торговая пара
            price_type: Тип цены ("bid", "ask", "last", "mid")
            
        Returns:
            Цена указанного типа или None, если данных нет
        """
        if exchange not in self.tickers:
            return None
            
        ticker = self.tickers[exchange]
        
        if price_type == "bid":
            return ticker.bid
        elif price_type == "ask":
            return ticker.ask
        elif price_type == "last":
            return ticker.last
        elif price_type == "mid":
            return ticker.mid_price
        else:
            return None
    
    @property
    def arbitrage_opportunities(self) -> List[Dict[str, Any]]:
        """
        Возвращает список потенциальных арбитражных возможностей.
        
        Returns:
            Список словарей с описанием арбитражных возможностей
        """
        opportunities = []
        
        # Перебираем все пары бирж
        exchanges = self.exchanges
        for i, buy_exchange in enumerate(exchanges):
            for sell_exchange in exchanges[i+1:]:
                # Проверяем, есть ли данные тикеров для обеих бирж
                if buy_exchange not in self.tickers or sell_exchange not in self.tickers:
                    continue
                
                buy_ticker = self.tickers[buy_exchange]
                sell_ticker = self.tickers[sell_exchange]
                
                # Проверяем арбитраж в обоих направлениях
                
                # Покупка на first_exchange, продажа на second_exchange
                buy_price = buy_ticker.ask
                sell_price = sell_ticker.bid
                
                if buy_price > 0 and sell_price > buy_price:
                    profit_percentage = (sell_price - buy_price) / buy_price * 100
                    opportunities.append({
                        "buy_exchange": buy_exchange,
                        "sell_exchange": sell_exchange,
                        "buy_price": float(buy_price),
                        "sell_price": float(sell_price),
                        "profit_percentage": float(profit_percentage),
                        "direction": f"{buy_exchange} -> {sell_exchange}"
                    })
                
                # Покупка на second_exchange, продажа на first_exchange
                buy_price = sell_ticker.ask
                sell_price = buy_ticker.bid
                
                if buy_price > 0 and sell_price > buy_price:
                    profit_percentage = (sell_price - buy_price) / buy_price * 100
                    opportunities.append({
                        "buy_exchange": sell_exchange,
                        "sell_exchange": buy_exchange,
                        "buy_price": float(buy_price),
                        "sell_price": float(sell_price),
                        "profit_percentage": float(profit_percentage),
                        "direction": f"{sell_exchange} -> {buy_exchange}"
                    })
        
        # Сортируем возможности по проценту прибыли
        return sorted(opportunities, key=lambda x: x["profit_percentage"], reverse=True)


class NormalizedScanResult(BaseModel):
    """Нормализованный результат сканирования рынка."""
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))
    data: Dict[str, NormalizedMarketData] = Field(default_factory=dict)
    errors: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    
    @property
    def symbols(self) -> List[str]:
        """Возвращает список символов в результате сканирования."""
        return list(self.data.keys())
    
    @property
    def exchanges(self) -> List[str]:
        """Возвращает список всех бирж в результате сканирования."""
        all_exchanges = set()
        for market_data in self.data.values():
            all_exchanges.update(market_data.exchanges)
        return list(all_exchanges)
    
    def get_all_arbitrage_opportunities(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Возвращает все арбитражные возможности для всех символов.
        
        Returns:
            Словарь, где ключ - символ, значение - список арбитражных возможностей
        """
        result = {}
        for symbol, market_data in self.data.items():
            opportunities = market_data.arbitrage_opportunities
            if opportunities:
                result[symbol] = opportunities
        return result


class DataNormalizer:
    """
    Нормализатор данных с разных бирж.
    Преобразует данные с разных бирж к единому формату и предоставляет
    методы для анализа и сравнения этих данных.
    """
    
    def __init__(self, decimal_places: int = 8):
        """
        Инициализация нормализатора данных.
        
        Args:
            decimal_places: Количество знаков после запятой для округления
        """
        self.logger = logging.getLogger(__name__)
        self.decimal_places = decimal_places
        self.logger.info(f"DataNormalizer initialized with {decimal_places} decimal places")
    
    def normalize_ticker(self, ticker: TickerData) -> NormalizedTicker:
        """
        Нормализует данные тикера.
        
        Args:
            ticker: Исходные данные тикера
            
        Returns:
            Нормализованные данные тикера
        """
        # Преобразуем значения в Decimal с нужным округлением
        bid = self._to_decimal(ticker.bid)
        ask = self._to_decimal(ticker.ask)
        last = self._to_decimal(ticker.last)
        volume = self._to_decimal(ticker.volume)
        
        # Вычисляем дополнительные метрики
        spread = ask - bid if ask > 0 and bid > 0 else Decimal('0')
        spread_percentage = (spread / bid * Decimal('100')) if bid > 0 else Decimal('0')
        mid_price = (bid + ask) / Decimal('2') if bid > 0 and ask > 0 else Decimal('0')
        
        return NormalizedTicker(
            symbol=ticker.symbol,
            bid=bid,
            ask=ask,
            spread=spread,
            spread_percentage=spread_percentage,
            mid_price=mid_price,
            last=last,
            volume=volume,
            exchange=ticker.exchange,
            timestamp=ticker.timestamp
        )
    
    def normalize_order_book(self, order_book: OrderBookData) -> NormalizedOrderBook:
        """
        Нормализует данные стакана ордеров.
        
        Args:
            order_book: Исходные данные стакана ордеров
            
        Returns:
            Нормализованные данные стакана ордеров
        """
        # Преобразуем цены и объемы в Decimal
        bids = [[self._to_decimal(price), self._to_decimal(volume)] 
                for price, volume in order_book.bids]
        asks = [[self._to_decimal(price), self._to_decimal(volume)] 
                for price, volume in order_book.asks]
        
        # Сортируем bid по убыванию цены, ask по возрастанию цены
        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])
        
        # Формируем кумулятивные данные
        cumulative_bids = self._calculate_cumulative_volume(bids)
        cumulative_asks = self._calculate_cumulative_volume(asks)
        
        return NormalizedOrderBook(
            symbol=order_book.symbol,
            exchange=order_book.exchange,
            timestamp=order_book.timestamp,
            bids=bids,
            asks=asks,
            cumulative_bids=cumulative_bids,
            cumulative_asks=cumulative_asks
        )
    
    def normalize_market_data(self, market_data: MarketData) -> NormalizedMarketData:
        """
        Нормализует рыночные данные.
        
        Args:
            market_data: Исходные рыночные данные
            
        Returns:
            Нормализованные рыночные данные
        """
        normalized_market_data = NormalizedMarketData(
            symbol=market_data.symbol,
            timestamp=market_data.timestamp
        )
        
        # Нормализуем тикеры
        for exchange, ticker in market_data.tickers.items():
            normalized_market_data.tickers[exchange] = self.normalize_ticker(ticker)
        
        # Нормализуем стаканы ордеров
        for exchange, order_book in market_data.order_books.items():
            normalized_market_data.order_books[exchange] = self.normalize_order_book(order_book)
        
        return normalized_market_data
    
    def normalize_scan_result(self, scan_result: ScanResult) -> NormalizedScanResult:
        """
        Нормализует результат сканирования рынка.
        
        Args:
            scan_result: Исходный результат сканирования
            
        Returns:
            Нормализованный результат сканирования
        """
        normalized_result = NormalizedScanResult(
            timestamp=scan_result.timestamp,
            errors=scan_result.errors.copy()
        )
        
        # Нормализуем данные для каждого символа
        for symbol, market_data in scan_result.data.items():
            normalized_result.data[symbol] = self.normalize_market_data(market_data)
        
        return normalized_result
    
    def _to_decimal(self, value: Union[int, float, str]) -> Decimal:
        """
        Преобразует значение в Decimal с нужным округлением.
        
        Args:
            value: Исходное значение
            
        Returns:
            Значение в формате Decimal
        """
        try:
            if value is None:
                return Decimal('0')
            
            decimal_value = Decimal(str(value))
            return decimal_value.quantize(
                Decimal('0.' + '0' * self.decimal_places),
                rounding=ROUND_DOWN
            )
        except Exception as e:
            self.logger.error(f"Error converting value to Decimal: {value}, {str(e)}")
            return Decimal('0')
    
    def _calculate_cumulative_volume(self, levels: List[List[Decimal]]) -> List[List[Decimal]]:
        """
        Рассчитывает кумулятивный объем для каждого уровня цены.
        
        Args:
            levels: Список уровней цен в формате [[цена, объем], ...]
            
        Returns:
            Список уровней с кумулятивным объемом
        """
        result = []
        cumulative_volume = Decimal('0')
        
        for price, volume in levels:
            cumulative_volume += volume
            result.append([price, cumulative_volume])
        
        return result
    
    def normalize_batch(self, data_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Нормализует пакет данных с разных бирж.
        
        Args:
            data_batch: Словарь с данными для нормализации {
                'tickers': {exchange: {symbol: ticker_data, ...}, ...},
                'order_books': {exchange: {symbol: order_book_data, ...}, ...}
            }
            
        Returns:
            Словарь с нормализованными данными
        """
        result = {
            'tickers': {},
            'order_books': {}
        }
        
        # Нормализуем тикеры
        if 'tickers' in data_batch:
            for exchange, symbols_data in data_batch['tickers'].items():
                if exchange not in result['tickers']:
                    result['tickers'][exchange] = {}
                
                for symbol, ticker in symbols_data.items():
                    result['tickers'][exchange][symbol] = self.normalize_ticker(ticker)
        
        # Нормализуем стаканы ордеров
        if 'order_books' in data_batch:
            for exchange, symbols_data in data_batch['order_books'].items():
                if exchange not in result['order_books']:
                    result['order_books'][exchange] = {}
                
                for symbol, order_book in symbols_data.items():
                    result['order_books'][exchange][symbol] = self.normalize_order_book(order_book)
        
        return result

    def get_price_difference(self, symbol: str, normalized_result: NormalizedScanResult) -> Dict[str, Dict[str, float]]:
        """
        Рассчитывает разницу в ценах между биржами для указанного символа.
        
        Args:
            symbol: Торговая пара
            normalized_result: Нормализованный результат сканирования
            
        Returns:
            Словарь с разницей в ценах для каждой пары бирж
        """
        if symbol not in normalized_result.data:
            return {}
        
        market_data = normalized_result.data[symbol]
        exchanges = market_data.exchanges
        result = {}
        
        # Перебираем все пары бирж
        for i, first_exchange in enumerate(exchanges):
            if first_exchange not in market_data.tickers:
                continue
                
            if first_exchange not in result:
                result[first_exchange] = {}
            
            for second_exchange in exchanges[i+1:]:
                if second_exchange not in market_data.tickers:
                    continue
                
                first_ticker = market_data.tickers[first_exchange]
                second_ticker = market_data.tickers[second_exchange]
                
                # Разница в ценах bid
                bid_diff = float(first_ticker.bid - second_ticker.bid)
                bid_diff_percent = float((first_ticker.bid / second_ticker.bid - 1) * 100) if second_ticker.bid > 0 else 0
                
                # Разница в ценах ask
                ask_diff = float(first_ticker.ask - second_ticker.ask)
                ask_diff_percent = float((first_ticker.ask / second_ticker.ask - 1) * 100) if second_ticker.ask > 0 else 0
                
                # Возможный арбитраж: покупка на second_exchange, продажа на first_exchange
                arb1 = float(first_ticker.bid - second_ticker.ask)
                arb1_percent = float((first_ticker.bid / second_ticker.ask - 1) * 100) if second_ticker.ask > 0 else 0
                
                # Возможный арбитраж: покупка на first_exchange, продажа на second_exchange
                arb2 = float(second_ticker.bid - first_ticker.ask)
                arb2_percent = float((second_ticker.bid / first_ticker.ask - 1) * 100) if first_ticker.ask > 0 else 0
                
                result[first_exchange][second_exchange] = {
                    "bid_diff": bid_diff,
                    "bid_diff_percent": bid_diff_percent,
                    "ask_diff": ask_diff,
                    "ask_diff_percent": ask_diff_percent,
                    "arb1": arb1,
                    "arb1_percent": arb1_percent,
                    "arb1_direction": f"{second_exchange} -> {first_exchange}",
                    "arb2": arb2,
                    "arb2_percent": arb2_percent,
                    "arb2_direction": f"{first_exchange} -> {second_exchange}"
                }
        
        return result
