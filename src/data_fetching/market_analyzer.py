"""
Анализатор рыночной ситуации.
Обрабатывает нормализованные данные и выявляет арбитражные возможности
и другие рыночные паттерны.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from decimal import Decimal
from pydantic import BaseModel, Field

from src.data_fetching.data_normalizer import NormalizedScanResult, NormalizedMarketData


class ArbitrageOpportunity(BaseModel):
    """Модель арбитражной возможности."""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: Decimal
    sell_price: Decimal
    spread: Decimal  # абсолютная разница
    profit_percentage: Decimal  # процент прибыли
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))
    estimated_volume: Optional[Decimal] = None  # оценка доступного объема
    estimated_profit: Optional[Decimal] = None  # оценка прибыли с учетом комиссий и объема


class MarketAnalysisResult(BaseModel):
    """Результат анализа рынка."""
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))
    arbitrage_opportunities: List[ArbitrageOpportunity] = Field(default_factory=list)
    price_differences: Dict[str, Dict[str, Dict[str, float]]] = Field(default_factory=dict)
    exchange_stats: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    symbol_stats: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class MarketAnalyzer:
    """
    Анализатор рыночной ситуации.
    Обрабатывает нормализованные данные и выявляет арбитражные возможности
    и другие рыночные паттерны.
    """
    
    def __init__(self, 
                 min_profit_percentage: float = 0.5,
                 fee_percentage: Dict[str, float] = None):
        """
        Инициализация анализатора рынка.
        
        Args:
            min_profit_percentage: Минимальный процент прибыли для арбитражных возможностей
            fee_percentage: Словарь комиссий для каждой биржи (по умолчанию 0.1%)
        """
        self.logger = logging.getLogger(__name__)
        self.min_profit_percentage = min_profit_percentage
        self.fee_percentage = fee_percentage or {
            "binance": 0.1,
            "kucoin": 0.1,
            "okx": 0.1
        }
        self.logger.info(f"MarketAnalyzer initialized with min_profit_percentage={min_profit_percentage}%")
    
    def analyze(self, normalized_data: NormalizedScanResult) -> MarketAnalysisResult:
        """
        Анализирует нормализованные рыночные данные.
        
        Args:
            normalized_data: Нормализованные данные рынка
            
        Returns:
            Результат анализа рынка
        """
        result = MarketAnalysisResult(timestamp=normalized_data.timestamp)
        
        # Анализируем арбитражные возможности
        self._analyze_arbitrage(normalized_data, result)
        
        # Анализируем разницу в ценах между биржами
        self._analyze_price_differences(normalized_data, result)
        
        # Собираем статистику по биржам
        self._analyze_exchange_stats(normalized_data, result)
        
        # Собираем статистику по символам
        self._analyze_symbol_stats(normalized_data, result)
        
        return result
    
    def _analyze_arbitrage(self, 
                          normalized_data: NormalizedScanResult, 
                          result: MarketAnalysisResult) -> None:
        """
        Анализирует арбитражные возможности.
        
        Args:
            normalized_data: Нормализованные данные рынка
            result: Результат анализа для заполнения
        """
        for symbol, market_data in normalized_data.data.items():
            # Получаем все потенциальные арбитражные возможности для символа
            opportunities = market_data.arbitrage_opportunities
            
            for opp in opportunities:
                # Проверяем, достаточен ли процент прибыли с учетом комиссий
                buy_exchange = opp["buy_exchange"]
                sell_exchange = opp["sell_exchange"]
                buy_fee = self.fee_percentage.get(buy_exchange.lower(), 0.1)
                sell_fee = self.fee_percentage.get(sell_exchange.lower(), 0.1)
                
                # Учитываем комиссии при расчете прибыли
                profit_after_fees = opp["profit_percentage"] - buy_fee - sell_fee
                
                if profit_after_fees >= self.min_profit_percentage:
                    # Оцениваем доступный объем и потенциальную прибыль
                    estimated_volume, estimated_profit = self._estimate_profit(
                        symbol, market_data, buy_exchange, sell_exchange,
                        Decimal(str(opp["buy_price"])), Decimal(str(opp["sell_price"]))
                    )
                    
                    # Добавляем возможность в результат
                    arbitrage = ArbitrageOpportunity(
                        symbol=symbol,
                        buy_exchange=buy_exchange,
                        sell_exchange=sell_exchange,
                        buy_price=Decimal(str(opp["buy_price"])),
                        sell_price=Decimal(str(opp["sell_price"])),
                        spread=Decimal(str(opp["sell_price"])) - Decimal(str(opp["buy_price"])),
                        profit_percentage=Decimal(str(profit_after_fees)),
                        timestamp=market_data.timestamp,
                        estimated_volume=estimated_volume,
                        estimated_profit=estimated_profit
                    )
                    
                    result.arbitrage_opportunities.append(arbitrage)
        
        # Сортируем возможности по проценту прибыли
        result.arbitrage_opportunities.sort(
            key=lambda x: x.profit_percentage, 
            reverse=True
        )
    
    def _estimate_profit(self, 
                        symbol: str, 
                        market_data: NormalizedMarketData,
                        buy_exchange: str, 
                        sell_exchange: str,
                        buy_price: Decimal, 
                        sell_price: Decimal) -> Tuple[Decimal, Decimal]:
        """
        Оценивает потенциальный объем и прибыль для арбитражной возможности.
        
        Args:
            symbol: Торговая пара
            market_data: Нормализованные данные рынка
            buy_exchange: Биржа для покупки
            sell_exchange: Биржа для продажи
            buy_price: Цена покупки
            sell_price: Цена продажи
            
        Returns:
            Кортеж (оценка_объема, оценка_прибыли)
        """
        # Получаем данные стаканов ордеров (если доступны)
        buy_ob = market_data.order_books.get(buy_exchange)
        sell_ob = market_data.order_books.get(sell_exchange)
        
        if not buy_ob or not sell_ob:
            # Если стаканы недоступны, используем данные тикеров
            buy_ticker = market_data.tickers.get(buy_exchange)
            sell_ticker = market_data.tickers.get(sell_exchange)
            
            if not buy_ticker or not sell_ticker:
                return Decimal('0'), Decimal('0')
            
            # Берем объем из тикеров (грубая оценка)
            volume = min(buy_ticker.volume, sell_ticker.volume) / Decimal('10')
            estimated_profit = volume * (sell_price - buy_price)
            
            # Учитываем комиссии
            buy_fee = Decimal(str(self.fee_percentage.get(buy_exchange.lower(), 0.1) / 100))
            sell_fee = Decimal(str(self.fee_percentage.get(sell_exchange.lower(), 0.1) / 100))
            
            estimated_profit -= volume * buy_price * buy_fee  # комиссия на покупку
            estimated_profit -= volume * sell_price * sell_fee  # комиссия на продажу
            
            return volume, estimated_profit
        
        # Анализируем стаканы ордеров для более точной оценки
        # Находим доступный объем для покупки по цене не выше buy_price
        available_buy_volume = Decimal('0')
        for ask_price, ask_vol in buy_ob.asks:
            if ask_price <= buy_price:
                available_buy_volume += ask_vol
            else:
                break
        
        # Находим доступный объем для продажи по цене не ниже sell_price
        available_sell_volume = Decimal('0')
        for bid_price, bid_vol in sell_ob.bids:
            if bid_price >= sell_price:
                available_sell_volume += bid_vol
            else:
                break
        
        # Берем минимальный доступный объем
        volume = min(available_buy_volume, available_sell_volume)
        
        if volume <= 0:
            return Decimal('0'), Decimal('0')
        
        # Рассчитываем потенциальную прибыль
        estimated_profit = volume * (sell_price - buy_price)
        
        # Учитываем комиссии
        buy_fee = Decimal(str(self.fee_percentage.get(buy_exchange.lower(), 0.1) / 100))
        sell_fee = Decimal(str(self.fee_percentage.get(sell_exchange.lower(), 0.1) / 100))
        
        estimated_profit -= volume * buy_price * buy_fee  # комиссия на покупку
        estimated_profit -= volume * sell_price * sell_fee  # комиссия на продажу
        
        return volume, estimated_profit
    
    def _analyze_price_differences(self, 
                                 normalized_data: NormalizedScanResult, 
                                 result: MarketAnalysisResult) -> None:
        """
        Анализирует разницу в ценах между биржами.
        
        Args:
            normalized_data: Нормализованные данные рынка
            result: Результат анализа для заполнения
        """
        for symbol in normalized_data.symbols:
            # Получаем все биржи для данного символа
            if symbol not in normalized_data.data:
                continue
                
            market_data = normalized_data.data[symbol]
            exchanges = market_data.exchanges
            
            if not exchanges or len(exchanges) < 2:
                continue
            
            # Анализируем разницу в ценах между всеми парами бирж
            price_diffs = {}
            
            for i, first_exchange in enumerate(exchanges):
                if first_exchange not in market_data.tickers:
                    continue
                    
                if first_exchange not in price_diffs:
                    price_diffs[first_exchange] = {}
                
                for second_exchange in exchanges[i+1:]:
                    if second_exchange not in market_data.tickers:
                        continue
                    
                    first_ticker = market_data.tickers[first_exchange]
                    second_ticker = market_data.tickers[second_exchange]
                    
                    # Вычисляем разницу в ценах
                    bid_diff = float(first_ticker.bid - second_ticker.bid)
                    ask_diff = float(first_ticker.ask - second_ticker.ask)
                    
                    # Вычисляем процентное соотношение
                    bid_diff_percent = float((first_ticker.bid / second_ticker.bid - 1) * 100) if second_ticker.bid > 0 else 0
                    ask_diff_percent = float((first_ticker.ask / second_ticker.ask - 1) * 100) if second_ticker.ask > 0 else 0
                    
                    # Анализируем возможности кросс-арбитража
                    cross_buy_on_first = float(second_ticker.bid - first_ticker.ask)
                    cross_buy_on_second = float(first_ticker.bid - second_ticker.ask)
                    
                    price_diffs[first_exchange][second_exchange] = {
                        "bid_diff": bid_diff,
                        "ask_diff": ask_diff,
                        "bid_diff_percent": bid_diff_percent,
                        "ask_diff_percent": ask_diff_percent,
                        "cross_buy_on_first": cross_buy_on_first,
                        "cross_buy_on_second": cross_buy_on_second
                    }
            
            # Добавляем анализ разницы в ценах в результат
            if price_diffs:
                result.price_differences[symbol] = price_diffs
    
    def _analyze_exchange_stats(self, 
                              normalized_data: NormalizedScanResult, 
                              result: MarketAnalysisResult) -> None:
        """
        Анализирует статистику по биржам.
        
        Args:
            normalized_data: Нормализованные данные рынка
            result: Результат анализа для заполнения
        """
        exchanges = normalized_data.exchanges
        
        for exchange in exchanges:
            exchange_stats = {
                "symbols_count": 0,
                "avg_bid": 0.0,
                "avg_ask": 0.0,
                "avg_spread_percent": 0.0,
                "total_volume": 0.0,
                "best_symbols": []
            }
            
            symbol_data = []
            
            # Собираем данные по всем символам для этой биржи
            for symbol, market_data in normalized_data.data.items():
                if exchange not in market_data.tickers:
                    continue
                
                ticker = market_data.tickers[exchange]
                
                exchange_stats["symbols_count"] += 1
                exchange_stats["avg_bid"] += float(ticker.bid)
                exchange_stats["avg_ask"] += float(ticker.ask)
                exchange_stats["avg_spread_percent"] += float(ticker.spread_percentage) if ticker.spread_percentage else 0
                exchange_stats["total_volume"] += float(ticker.volume)
                
                symbol_data.append({
                    "symbol": symbol,
                    "volume": float(ticker.volume),
                    "bid": float(ticker.bid),
                    "ask": float(ticker.ask),
                    "spread_percent": float(ticker.spread_percentage) if ticker.spread_percentage else 0
                })
            
            # Вычисляем средние значения
            if exchange_stats["symbols_count"] > 0:
                exchange_stats["avg_bid"] /= exchange_stats["symbols_count"]
                exchange_stats["avg_ask"] /= exchange_stats["symbols_count"]
                exchange_stats["avg_spread_percent"] /= exchange_stats["symbols_count"]
            
            # Находим топ-5 символов по объему
            symbol_data.sort(key=lambda x: x["volume"], reverse=True)
            exchange_stats["best_symbols"] = symbol_data[:5]
            
            # Добавляем статистику по бирже в результат
            result.exchange_stats[exchange] = exchange_stats
    
    def _analyze_symbol_stats(self, 
                            normalized_data: NormalizedScanResult, 
                            result: MarketAnalysisResult) -> None:
        """
        Анализирует статистику по символам.
        
        Args:
            normalized_data: Нормализованные данные рынка
            result: Результат анализа для заполнения
        """
        for symbol, market_data in normalized_data.data.items():
            exchanges = market_data.exchanges
            
            if not exchanges:
                continue
            
            symbol_stats = {
                "exchanges_count": len(exchanges),
                "avg_price": 0.0,
                "min_price": float('inf'),
                "max_price": 0.0,
                "price_range_percent": 0.0,
                "total_volume": 0.0,
                "best_exchange": None,
                "exchanges_data": {}
            }
            
            prices = []
            volumes = []
            
            # Собираем данные по всем биржам для этого символа
            for exchange in exchanges:
                if exchange not in market_data.tickers:
                    continue
                
                ticker = market_data.tickers[exchange]
                mid_price = float(ticker.mid_price) if ticker.mid_price > 0 else (float(ticker.bid) + float(ticker.ask)) / 2
                volume = float(ticker.volume)
                
                symbol_stats["exchanges_data"][exchange] = {
                    "bid": float(ticker.bid),
                    "ask": float(ticker.ask),
                    "mid_price": mid_price,
                    "volume": volume,
                    "spread_percent": float(ticker.spread_percentage) if ticker.spread_percentage else 0
                }
                
                prices.append(mid_price)
                volumes.append(volume)
                
                symbol_stats["total_volume"] += volume
                
                # Обновляем мин/макс цены
                if mid_price < symbol_stats["min_price"]:
                    symbol_stats["min_price"] = mid_price
                
                if mid_price > symbol_stats["max_price"]:
                    symbol_stats["max_price"] = mid_price
            
            # Вычисляем средние значения
            if prices:
                symbol_stats["avg_price"] = sum(prices) / len(prices)
                
                # Вычисляем диапазон цен в процентах
                if symbol_stats["min_price"] > 0:
                    symbol_stats["price_range_percent"] = (symbol_stats["max_price"] - symbol_stats["min_price"]) / symbol_stats["min_price"] * 100
            
            # Находим биржу с наибольшим объемом
            if volumes:
                max_volume_index = volumes.index(max(volumes))
                symbol_stats["best_exchange"] = list(symbol_stats["exchanges_data"].keys())[max_volume_index]
            
            # Добавляем статистику по символу в результат
            result.symbol_stats[symbol] = symbol_stats 