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

from ..core.temporal import (
    TemporalValue, DataType, TemporalStore, 
    is_taiwan_trading_day, get_previous_trading_day
)
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
    
    def validate_data_completeness(self, 
                                  symbol: str,
                                  start_date: date,
                                  end_date: date,
                                  data_types: List[DataType]) -> Dict[str, Any]:
        """Validate data completeness for a symbol across date range."""
        validation_report = {
            "symbol": symbol,
            "date_range": (start_date, end_date),
            "total_trading_days": 0,
            "missing_data_days": [],
            "data_type_coverage": {},
            "quality_issues": []
        }
        
        try:
            # Count expected trading days
            current_date = start_date
            trading_days = []
            while current_date <= end_date:
                if is_taiwan_trading_day(current_date):
                    trading_days.append(current_date)
                current_date += timedelta(days=1)
            
            validation_report["total_trading_days"] = len(trading_days)
            
            # Check each data type
            for data_type in data_types:
                if data_type == DataType.PRICE:
                    values = self.get_price_data(symbol, start_date, end_date)
                elif data_type == DataType.FUNDAMENTAL:
                    values = self.get_fundamental_data(symbol, start_date, end_date)
                elif data_type == DataType.CORPORATE_ACTION:
                    values = self.get_corporate_actions(symbol, start_date, end_date)
                else:
                    continue
                
                # Analyze coverage
                value_dates = set(v.value_date for v in values)
                missing_dates = [d for d in trading_days if d not in value_dates]
                
                validation_report["data_type_coverage"][data_type.value] = {
                    "total_values": len(values),
                    "coverage_rate": (len(trading_days) - len(missing_dates)) / max(len(trading_days), 1),
                    "missing_dates": [d.isoformat() for d in missing_dates[:10]]  # Limit output
                }
                
                # Accumulate missing dates
                validation_report["missing_data_days"].extend(missing_dates)
            
            # Remove duplicates from missing dates
            validation_report["missing_data_days"] = list(set(validation_report["missing_data_days"]))
            validation_report["missing_data_days"].sort()
            
        except Exception as e:
            validation_report["quality_issues"].append(f"Completeness validation error: {e}")
        
        return validation_report
    
    def get_data_quality_metrics(self, 
                                symbol: str,
                                lookback_days: int = 30) -> Dict[str, Any]:
        """Get comprehensive data quality metrics for a symbol."""
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)
        
        metrics = {
            "symbol": symbol,
            "analysis_period": (start_date, end_date),
            "price_data_metrics": {},
            "fundamental_data_metrics": {},
            "overall_quality_score": 0.0,
            "issues_found": [],
            "recommendations": []
        }
        
        try:
            # Price data quality metrics
            price_values = self.get_price_data(symbol, start_date, end_date)
            if price_values:
                price_metrics = self._analyze_price_data_quality(price_values)
                metrics["price_data_metrics"] = price_metrics
            
            # Fundamental data quality metrics
            fundamental_values = self.get_fundamental_data(symbol, start_date, end_date)
            if fundamental_values:
                fundamental_metrics = self._analyze_fundamental_data_quality(fundamental_values)
                metrics["fundamental_data_metrics"] = fundamental_metrics
            
            # Calculate overall quality score
            metrics["overall_quality_score"] = self._calculate_quality_score(metrics)
            
            # Generate recommendations
            metrics["recommendations"] = self._generate_quality_recommendations(metrics)
            
        except Exception as e:
            metrics["issues_found"].append(f"Quality metrics analysis error: {e}")
        
        return metrics
    
    def _analyze_price_data_quality(self, values: List[TemporalValue]) -> Dict[str, Any]:
        """Analyze price data quality metrics."""
        if not values:
            return {"error": "No price data available"}
        
        price_values = [float(v.value) for v in values if v.value is not None]
        
        metrics = {
            "total_records": len(values),
            "non_null_records": len(price_values),
            "null_rate": (len(values) - len(price_values)) / len(values),
            "data_gaps": 0,
            "outliers_detected": 0,
            "volatility_spike_days": 0
        }
        
        if len(price_values) > 1:
            # Calculate daily returns for outlier detection
            returns = []
            for i in range(1, len(price_values)):
                ret = (price_values[i] - price_values[i-1]) / price_values[i-1]
                returns.append(ret)
            
            if returns:
                import statistics
                mean_return = statistics.mean(returns)
                std_return = statistics.stdev(returns) if len(returns) > 1 else 0
                
                # Detect outliers (returns > 3 standard deviations)
                outliers = [r for r in returns if abs(r - mean_return) > 3 * std_return]
                metrics["outliers_detected"] = len(outliers)
                
                # Detect volatility spikes (returns > 5%)
                spikes = [r for r in returns if abs(r) > 0.05]
                metrics["volatility_spike_days"] = len(spikes)
        
        return metrics
    
    def _analyze_fundamental_data_quality(self, values: List[TemporalValue]) -> Dict[str, Any]:
        """Analyze fundamental data quality metrics."""
        if not values:
            return {"error": "No fundamental data available"}
        
        metrics = {
            "total_records": len(values),
            "unique_periods": len(set((v.metadata.get("fiscal_year"), v.metadata.get("fiscal_quarter")) 
                                   for v in values if v.metadata)),
            "avg_reporting_lag_days": 0,
            "max_reporting_lag_days": 0,
            "late_filings": 0
        }
        
        lag_days = []
        for value in values:
            if value.metadata and "lag_days" in value.metadata:
                lag = value.metadata["lag_days"]
                lag_days.append(lag)
                if lag > 60:  # Taiwan regulatory limit
                    metrics["late_filings"] += 1
        
        if lag_days:
            metrics["avg_reporting_lag_days"] = sum(lag_days) / len(lag_days)
            metrics["max_reporting_lag_days"] = max(lag_days)
        
        return metrics
    
    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-100)."""
        score = 100.0
        
        # Price data quality impact
        price_metrics = metrics.get("price_data_metrics", {})
        if price_metrics:
            null_rate = price_metrics.get("null_rate", 0)
            score -= null_rate * 30  # Up to 30 points for data completeness
            
            outliers_rate = price_metrics.get("outliers_detected", 0) / max(price_metrics.get("total_records", 1), 1)
            score -= outliers_rate * 20  # Up to 20 points for outliers
        
        # Fundamental data quality impact
        fundamental_metrics = metrics.get("fundamental_data_metrics", {})
        if fundamental_metrics:
            late_filings_rate = fundamental_metrics.get("late_filings", 0) / max(fundamental_metrics.get("total_records", 1), 1)
            score -= late_filings_rate * 15  # Up to 15 points for late filings
        
        return max(0.0, min(100.0, score))
    
    def _generate_quality_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate data quality improvement recommendations."""
        recommendations = []
        
        price_metrics = metrics.get("price_data_metrics", {})
        if price_metrics:
            if price_metrics.get("null_rate", 0) > 0.05:
                recommendations.append("High null rate in price data - investigate data source reliability")
            
            if price_metrics.get("outliers_detected", 0) > 5:
                recommendations.append("Multiple price outliers detected - review data validation rules")
            
            if price_metrics.get("volatility_spike_days", 0) > 3:
                recommendations.append("High volatility detected - verify corporate actions and news events")
        
        fundamental_metrics = metrics.get("fundamental_data_metrics", {})
        if fundamental_metrics:
            if fundamental_metrics.get("late_filings", 0) > 0:
                recommendations.append("Late fundamental data filings detected - monitor regulatory compliance")
            
            if fundamental_metrics.get("avg_reporting_lag_days", 0) > 45:
                recommendations.append("High average reporting lag - consider alternative data sources")
        
        overall_score = metrics.get("overall_quality_score", 100)
        if overall_score < 80:
            recommendations.append("Overall data quality below threshold - comprehensive review recommended")
        
        return recommendations
    
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