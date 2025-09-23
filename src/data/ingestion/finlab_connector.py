"""
FinLab Data Connector for Taiwan Market Data.

This module provides a comprehensive connector to the existing FinLab database,
handling 278 financial data fields with proper temporal mapping and data quality validation.
"""

import os
import logging
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple, Set
from dataclasses import dataclass, field
from decimal import Decimal
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, MetaData, Table, Column, select
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

from ..core.temporal import TemporalValue, DataType, TemporalStore
from ..models.taiwan_market import (
    TaiwanMarketData, TaiwanFundamental, TaiwanCorporateAction,
    CorporateActionType, TradingStatus, TaiwanMarketDataValidator
)

logger = logging.getLogger(__name__)


@dataclass
class FinLabConfig:
    """FinLab database configuration."""
    host: str
    port: int = 5432
    database: str = "finlab"
    username: str = "finlab"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    
    def get_connection_string(self) -> str:
        """Generate SQLAlchemy connection string."""
        return (f"postgresql://{self.username}:{self.password}@"
                f"{self.host}:{self.port}/{self.database}")


@dataclass
class FinLabFieldMapping:
    """Maps FinLab database fields to temporal data model."""
    
    # Core price and volume fields
    PRICE_FIELDS = {
        'open': 'open_price',
        'high': 'high_price', 
        'low': 'low_price',
        'close': 'close_price',
        'adj_close': 'adjusted_close',
        'volume': 'volume',
        'turnover': 'turnover',
        'market_cap': 'market_cap'
    }
    
    # Technical indicators
    TECHNICAL_FIELDS = {
        'sma_5': 'sma_5_day',
        'sma_10': 'sma_10_day', 
        'sma_20': 'sma_20_day',
        'sma_60': 'sma_60_day',
        'ema_12': 'ema_12_day',
        'ema_26': 'ema_26_day',
        'rsi_14': 'rsi_14_day',
        'macd': 'macd',
        'macd_signal': 'macd_signal',
        'bb_upper': 'bollinger_upper',
        'bb_lower': 'bollinger_lower',
        'bb_mid': 'bollinger_middle'
    }
    
    # Fundamental data fields
    FUNDAMENTAL_FIELDS = {
        'revenue': 'revenue',
        'operating_income': 'operating_income',
        'net_income': 'net_income',
        'total_assets': 'total_assets',
        'total_equity': 'total_equity',
        'total_debt': 'total_debt',
        'cash_and_equivalents': 'cash_and_equivalents',
        'eps': 'earnings_per_share',
        'roe': 'return_on_equity',
        'roa': 'return_on_assets',
        'debt_ratio': 'debt_to_assets',
        'current_ratio': 'current_ratio',
        'pe_ratio': 'price_to_earnings',
        'pb_ratio': 'price_to_book',
        'dividend_yield': 'dividend_yield',
        'book_value_per_share': 'book_value_per_share',
        'free_cash_flow': 'free_cash_flow',
        'working_capital': 'working_capital'
    }
    
    # Market microstructure fields  
    MICROSTRUCTURE_FIELDS = {
        'bid_price': 'best_bid_price',
        'ask_price': 'best_ask_price',
        'bid_size': 'best_bid_size',
        'ask_size': 'best_ask_size',
        'spread': 'bid_ask_spread',
        'trades_count': 'number_of_trades',
        'avg_trade_size': 'average_trade_size',
        'vwap': 'volume_weighted_avg_price',
        'twap': 'time_weighted_avg_price'
    }
    
    # Corporate actions and events
    CORPORATE_ACTION_FIELDS = {
        'dividend_cash': 'cash_dividend',
        'dividend_stock': 'stock_dividend', 
        'split_ratio': 'stock_split_ratio',
        'ex_dividend_date': 'ex_dividend_date',
        'payment_date': 'dividend_payment_date',
        'record_date': 'dividend_record_date'
    }
    
    # Risk and volatility metrics
    RISK_FIELDS = {
        'volatility_30d': 'volatility_30_day',
        'volatility_60d': 'volatility_60_day',
        'beta': 'market_beta',
        'sharpe_ratio': 'sharpe_ratio',
        'max_drawdown': 'maximum_drawdown',
        'var_95': 'value_at_risk_95',
        'var_99': 'value_at_risk_99'
    }
    
    @classmethod
    def get_all_fields(cls) -> Dict[str, str]:
        """Get complete field mapping dictionary."""
        all_fields = {}
        all_fields.update(cls.PRICE_FIELDS)
        all_fields.update(cls.TECHNICAL_FIELDS)
        all_fields.update(cls.FUNDAMENTAL_FIELDS)
        all_fields.update(cls.MICROSTRUCTURE_FIELDS)
        all_fields.update(cls.CORPORATE_ACTION_FIELDS)
        all_fields.update(cls.RISK_FIELDS)
        return all_fields
    
    @classmethod
    def get_field_category(cls, field_name: str) -> Optional[str]:
        """Determine which category a field belongs to."""
        for category_name in ['PRICE_FIELDS', 'TECHNICAL_FIELDS', 'FUNDAMENTAL_FIELDS',
                             'MICROSTRUCTURE_FIELDS', 'CORPORATE_ACTION_FIELDS', 'RISK_FIELDS']:
            category_dict = getattr(cls, category_name)
            if field_name in category_dict:
                return category_name.replace('_FIELDS', '').lower()
        return None


class FinLabConnector:
    """High-performance connector to FinLab database with temporal consistency."""
    
    def __init__(self, 
                 config: FinLabConfig,
                 temporal_store: TemporalStore,
                 enable_cache: bool = True,
                 max_workers: int = 4):
        self.config = config
        self.temporal_store = temporal_store
        self.enable_cache = enable_cache
        self.max_workers = max_workers
        
        # Database connection
        self.engine = None
        self.session_factory = None
        self.metadata = None
        
        # Data validation
        self.validator = TaiwanMarketDataValidator()
        
        # Field mapping
        self.field_mapping = FinLabFieldMapping.get_all_fields()
        
        # Cache for frequently accessed data
        self._symbol_cache: Dict[str, Dict] = {}
        self._metadata_cache: Dict[str, Any] = {}
        
        # Performance metrics
        self.query_count = 0
        self.cache_hits = 0
        self.total_query_time = 0.0
        
        logger.info(f"FinLab connector initialized with {len(self.field_mapping)} mapped fields")
    
    def connect(self) -> None:
        """Establish database connection."""
        try:
            connection_string = self.config.get_connection_string()
            self.engine = create_engine(
                connection_string,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=self.config.echo
            )
            
            self.session_factory = sessionmaker(bind=self.engine)
            self.metadata = MetaData()
            self.metadata.reflect(bind=self.engine)
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1")).fetchone()
                logger.info(f"FinLab database connection established: {result}")
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to connect to FinLab database: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("FinLab database connection closed")
    
    def get_available_symbols(self, as_of_date: Optional[date] = None) -> List[str]:
        """Get list of available symbols in FinLab database."""
        try:
            with self.session_factory() as session:
                # Query main price table for available symbols
                query = text("""
                    SELECT DISTINCT symbol 
                    FROM daily_price 
                    WHERE (:as_of_date IS NULL OR date <= :as_of_date)
                    ORDER BY symbol
                """)
                
                result = session.execute(query, {"as_of_date": as_of_date})
                symbols = [row[0] for row in result.fetchall()]
                
                logger.debug(f"Found {len(symbols)} symbols in FinLab database")
                return symbols
                
        except SQLAlchemyError as e:
            logger.error(f"Error querying available symbols: {e}")
            return []
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get metadata information for a symbol."""
        cache_key = f"info_{symbol}"
        
        if self.enable_cache and cache_key in self._metadata_cache:
            self.cache_hits += 1
            return self._metadata_cache[cache_key]
        
        try:
            with self.session_factory() as session:
                query = text("""
                    SELECT symbol, company_name, industry, sector, 
                           listing_date, market_type, outstanding_shares
                    FROM stock_info 
                    WHERE symbol = :symbol
                """)
                
                result = session.execute(query, {"symbol": symbol}).fetchone()
                
                if result:
                    info = {
                        "symbol": result[0],
                        "company_name": result[1],
                        "industry": result[2],
                        "sector": result[3],
                        "listing_date": result[4],
                        "market_type": result[5],
                        "outstanding_shares": result[6]
                    }
                    
                    if self.enable_cache:
                        self._metadata_cache[cache_key] = info
                    
                    return info
                
        except SQLAlchemyError as e:
            logger.error(f"Error querying symbol info for {symbol}: {e}")
        
        return None
    
    def get_price_data(self, 
                      symbol: str, 
                      start_date: date,
                      end_date: date,
                      fields: Optional[List[str]] = None) -> List[TemporalValue]:
        """Get historical price data for a symbol."""
        start_time = datetime.utcnow()
        
        if fields is None:
            fields = list(FinLabFieldMapping.PRICE_FIELDS.keys())
        
        # Map fields to database columns
        db_fields = [self.field_mapping.get(f, f) for f in fields]
        field_list = ", ".join(db_fields)
        
        try:
            with self.session_factory() as session:
                query = text(f"""
                    SELECT date, symbol, {field_list}
                    FROM daily_price 
                    WHERE symbol = :symbol 
                      AND date BETWEEN :start_date AND :end_date
                    ORDER BY date
                """)
                
                result = session.execute(query, {
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date
                }).fetchall()
                
                temporal_values = []
                
                for row in result:
                    data_date = row[0]
                    symbol_val = row[1]
                    
                    # Create temporal values for each field
                    for i, field in enumerate(fields):
                        value = row[i + 2]  # Skip date and symbol
                        
                        if value is not None:
                            temporal_value = TemporalValue(
                                value=Decimal(str(value)) if isinstance(value, (int, float)) else value,
                                as_of_date=data_date,  # Price data available same day
                                value_date=data_date,
                                data_type=DataType.PRICE if field in FinLabFieldMapping.PRICE_FIELDS else DataType.MARKET_DATA,
                                symbol=symbol_val,
                                metadata={
                                    "field": field,
                                    "source": "finlab",
                                    "category": FinLabFieldMapping.get_field_category(field)
                                }
                            )
                            temporal_values.append(temporal_value)
                
                # Update performance metrics
                self.query_count += 1
                self.total_query_time += (datetime.utcnow() - start_time).total_seconds()
                
                logger.debug(f"Retrieved {len(temporal_values)} price values for {symbol}")
                return temporal_values
                
        except SQLAlchemyError as e:
            logger.error(f"Error querying price data for {symbol}: {e}")
            return []
    
    def get_fundamental_data(self,
                            symbol: str, 
                            start_date: date,
                            end_date: date,
                            fields: Optional[List[str]] = None) -> List[TemporalValue]:
        """Get fundamental data with proper reporting lag handling."""
        start_time = datetime.utcnow()
        
        if fields is None:
            fields = list(FinLabFieldMapping.FUNDAMENTAL_FIELDS.keys())
        
        db_fields = [self.field_mapping.get(f, f) for f in fields]
        field_list = ", ".join(db_fields)
        
        try:
            with self.session_factory() as session:
                query = text(f"""
                    SELECT report_date, announce_date, fiscal_year, fiscal_quarter,
                           symbol, {field_list}
                    FROM fundamental_data 
                    WHERE symbol = :symbol 
                      AND announce_date BETWEEN :start_date AND :end_date
                    ORDER BY announce_date, report_date
                """)
                
                result = session.execute(query, {
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date
                }).fetchall()
                
                temporal_values = []
                
                for row in result:
                    report_date = row[0]
                    announce_date = row[1]
                    fiscal_year = row[2]
                    fiscal_quarter = row[3]
                    symbol_val = row[4]
                    
                    # Create temporal values for each field
                    for i, field in enumerate(fields):
                        value = row[i + 5]  # Skip metadata fields
                        
                        if value is not None:
                            temporal_value = TemporalValue(
                                value=Decimal(str(value)) if isinstance(value, (int, float)) else value,
                                as_of_date=announce_date,  # Data only available after announcement
                                value_date=report_date,    # Data refers to the report date
                                data_type=DataType.FUNDAMENTAL,
                                symbol=symbol_val,
                                metadata={
                                    "field": field,
                                    "source": "finlab",
                                    "fiscal_year": fiscal_year,
                                    "fiscal_quarter": fiscal_quarter,
                                    "lag_days": (announce_date - report_date).days,
                                    "category": "fundamental"
                                }
                            )
                            temporal_values.append(temporal_value)
                
                self.query_count += 1
                self.total_query_time += (datetime.utcnow() - start_time).total_seconds()
                
                logger.debug(f"Retrieved {len(temporal_values)} fundamental values for {symbol}")
                return temporal_values
                
        except SQLAlchemyError as e:
            logger.error(f"Error querying fundamental data for {symbol}: {e}")
            return []
    
    def get_corporate_actions(self,
                             symbol: str,
                             start_date: date, 
                             end_date: date) -> List[TemporalValue]:
        """Get corporate action data."""
        try:
            with self.session_factory() as session:
                query = text("""
                    SELECT symbol, action_type, announce_date, ex_date, 
                           record_date, payment_date, cash_amount, stock_ratio, description
                    FROM corporate_actions 
                    WHERE symbol = :symbol 
                      AND ex_date BETWEEN :start_date AND :end_date
                    ORDER BY announce_date
                """)
                
                result = session.execute(query, {
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date
                }).fetchall()
                
                temporal_values = []
                
                for row in result:
                    action_data = {
                        "action_type": row[1],
                        "ex_date": row[3].isoformat() if row[3] else None,
                        "record_date": row[4].isoformat() if row[4] else None,
                        "payment_date": row[5].isoformat() if row[5] else None,
                        "cash_amount": float(row[6]) if row[6] else None,
                        "stock_ratio": float(row[7]) if row[7] else None,
                        "description": row[8]
                    }
                    
                    temporal_value = TemporalValue(
                        value=action_data,
                        as_of_date=row[2],  # announce_date
                        value_date=row[3],  # ex_date
                        data_type=DataType.CORPORATE_ACTION,
                        symbol=row[0],
                        metadata={
                            "source": "finlab",
                            "action_type": row[1],
                            "category": "corporate_action"
                        }
                    )
                    temporal_values.append(temporal_value)
                
                logger.debug(f"Retrieved {len(temporal_values)} corporate actions for {symbol}")
                return temporal_values
                
        except SQLAlchemyError as e:
            logger.error(f"Error querying corporate actions for {symbol}: {e}")
            return []
    
    def bulk_sync_symbols(self, 
                         symbols: List[str],
                         start_date: date,
                         end_date: date,
                         data_types: Optional[List[DataType]] = None) -> Dict[str, int]:
        """Bulk synchronization of multiple symbols with parallel processing."""
        if data_types is None:
            data_types = [DataType.PRICE, DataType.VOLUME, DataType.FUNDAMENTAL]
        
        logger.info(f"Starting bulk sync for {len(symbols)} symbols from {start_date} to {end_date}")
        
        sync_results = {}
        
        def sync_symbol(symbol: str) -> Tuple[str, int]:
            """Sync a single symbol."""
            total_values = 0
            
            try:
                for data_type in data_types:
                    if data_type == DataType.PRICE:
                        values = self.get_price_data(symbol, start_date, end_date)
                    elif data_type == DataType.FUNDAMENTAL:
                        values = self.get_fundamental_data(symbol, start_date, end_date)
                    elif data_type == DataType.CORPORATE_ACTION:
                        values = self.get_corporate_actions(symbol, start_date, end_date)
                    else:
                        continue
                    
                    # Store temporal values
                    for value in values:
                        self.temporal_store.store(value)
                        total_values += 1
                
                logger.debug(f"Synced {total_values} values for symbol {symbol}")
                return symbol, total_values
                
            except Exception as e:
                logger.error(f"Error syncing symbol {symbol}: {e}")
                return symbol, 0
        
        # Execute parallel synchronization
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(sync_symbol, symbol) for symbol in symbols]
            
            for future in futures:
                symbol, count = future.result()
                sync_results[symbol] = count
        
        total_synced = sum(sync_results.values())
        logger.info(f"Bulk sync completed: {total_synced} total values synced")
        
        return sync_results
    
    def validate_data_quality(self, 
                             symbol: str,
                             data_date: date,
                             data: Dict[str, Any]) -> List[str]:
        """Validate data quality using Taiwan market rules."""
        issues = []
        
        # Create TaiwanMarketData object for validation
        try:
            market_data = TaiwanMarketData(
                symbol=symbol,
                data_date=data_date,
                as_of_date=data_date,
                open_price=data.get('open_price'),
                high_price=data.get('high_price'),
                low_price=data.get('low_price'),
                close_price=data.get('close_price'),
                volume=data.get('volume'),
                turnover=data.get('turnover')
            )
            
            validation_issues = self.validator.validate_price_data(market_data)
            issues.extend(validation_issues)
            
        except Exception as e:
            issues.append(f"Data validation error: {e}")
        
        return issues
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get connector performance statistics."""
        avg_query_time = self.total_query_time / max(self.query_count, 1)
        cache_hit_rate = self.cache_hits / max(self.query_count, 1) if self.enable_cache else 0
        
        return {
            "query_count": self.query_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "avg_query_time_seconds": avg_query_time,
            "total_query_time_seconds": self.total_query_time,
            "mapped_fields_count": len(self.field_mapping),
            "cache_enabled": self.enable_cache
        }
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._symbol_cache.clear()
        self._metadata_cache.clear()
        logger.info("FinLab connector cache cleared")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


# Factory function for easy connector creation
def create_finlab_connector(
    host: str = "localhost",
    port: int = 5432,
    database: str = "finlab", 
    username: str = "finlab",
    password: str = "",
    temporal_store: Optional[TemporalStore] = None,
    **kwargs
) -> FinLabConnector:
    """Factory function to create FinLab connector with sensible defaults."""
    config = FinLabConfig(
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
        **kwargs
    )
    
    if temporal_store is None:
        from ..core.temporal import InMemoryTemporalStore
        temporal_store = InMemoryTemporalStore()
    
    return FinLabConnector(config, temporal_store)