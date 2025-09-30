# Issue #02: Data Pipeline Integration & Testing

**Issue Type**: Infrastructure Foundation
**Phase**: 1 - Foundation & Proof
**Priority**: P0 - Critical Path
**Effort**: 16 hours
**Status**: 📋 Ready for Development

---

## 🎯 Objective

Integrate with existing 4.35M record data pipeline, validate performance characteristics, and establish data access patterns optimized for weekly/monthly trading strategies.

## 📋 Requirements

### Core Functionality
- [x] **Pipeline Connection**: Connect to existing PostgreSQL ml4t.finlab_data table
- [x] **Performance Validation**: Benchmark query performance for trading workflows
- [x] **Data Access Patterns**: Establish efficient data retrieval for factor computation
- [x] **Error Handling**: Robust error handling for production data access

### Existing Infrastructure Assessment
```sql
-- Current production table: ml4t.finlab_data
-- Records: 4.35M (2002-2025, 23+ years)
-- Symbols: 1,307 Taiwan stocks
-- Fields: 277 financial fields with proper quoting
-- Update: Incremental via working production scripts
```

## 🔧 Implementation Plan

### Day 1: Connection & Assessment (8h)
**Wednesday**

```python
# File: src/data/existing_integration.py

import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from typing import List, Optional, Dict, Any
import logging

class ExistingDataIntegration:
    """Integration with working 4.35M record data pipeline"""

    def __init__(self, config_manager):
        self.config = config_manager.get_database_config()
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'user': 'jnpi',
            'database': 'finlab_data',
            'schema': 'ml4t'
        }
        self.engine = self._create_engine()

    def _create_engine(self):
        """Create SQLAlchemy engine for data access"""
        connection_string = (
            f"postgresql://{self.db_config['user']}@{self.db_config['host']}:"
            f"{self.db_config['port']}/{self.db_config['database']}"
        )
        return create_engine(connection_string)

    def validate_pipeline_health(self) -> Dict[str, Any]:
        """Validate existing data pipeline health and performance"""
        query = """
        SELECT
            COUNT(*) as total_records,
            COUNT(DISTINCT symbol) as unique_symbols,
            MIN(date) as earliest_date,
            MAX(date) as latest_date,
            COUNT(DISTINCT date) as trading_days
        FROM ml4t.finlab_data
        """

        result = pd.read_sql(query, self.engine)
        return result.iloc[0].to_dict()

    def get_symbol_universe(self) -> List[str]:
        """Get complete list of available symbols"""
        query = "SELECT DISTINCT symbol FROM ml4t.finlab_data ORDER BY symbol"
        result = pd.read_sql(query, self.engine)
        return result['symbol'].tolist()

    def get_finlab_data(self, symbols: List[str], date_range: tuple,
                       fields: Optional[List[str]] = None) -> pd.DataFrame:
        """Access existing 4.35M records optimized for weekly/monthly signals"""

        # Essential fields for factor computation
        if fields is None:
            fields = [
                'symbol', 'date', 'adj_close',
                '"本益比"', '"股價淨值比"', '"營收成長率"',
                '"月營收"', '"成交量"', '"市值"'
            ]

        symbol_list = "','".join(symbols)
        field_list = ', '.join(fields)

        query = f"""
        SELECT {field_list}
        FROM ml4t.finlab_data
        WHERE symbol IN ('{symbol_list}')
        AND date BETWEEN '{date_range[0]}' AND '{date_range[1]}'
        ORDER BY symbol, date
        """

        return pd.read_sql(query, self.engine)

    def benchmark_query_performance(self, sample_symbols: int = 50) -> Dict[str, float]:
        """Benchmark query performance for different data access patterns"""
        import time

        # Get sample symbols
        symbols = self.get_symbol_universe()[:sample_symbols]
        date_range = ('2023-01-01', '2024-12-31')

        # Test 1: Basic data retrieval
        start_time = time.time()
        basic_data = self.get_finlab_data(symbols, date_range)
        basic_time = time.time() - start_time

        # Test 2: Chunked retrieval
        start_time = time.time()
        chunk_size = 10
        for i in range(0, len(symbols), chunk_size):
            chunk_symbols = symbols[i:i+chunk_size]
            chunk_data = self.get_finlab_data(chunk_symbols, date_range)
        chunked_time = time.time() - start_time

        return {
            'basic_query_time': basic_time,
            'chunked_query_time': chunked_time,
            'records_retrieved': len(basic_data),
            'query_rate_records_per_second': len(basic_data) / basic_time
        }
```

### Day 2: Performance Testing & Optimization (8h)
**Thursday**

```python
# File: src/data/hybrid_data_access.py

class HybridDataAccess:
    """Multi-tier data access optimized for weekly/monthly trading"""

    def __init__(self, config_manager, existing_integration):
        self.config = config_manager
        self.existing = existing_integration
        self.cache_enabled = config_manager.get_database_config().get('cache_enabled', True)

    def get_trading_data(self, symbols: List[str], lookback_months: int = 24) -> pd.DataFrame:
        """Get data optimized for weekly/monthly trading strategies"""

        # Calculate date range for lookback period
        end_date = pd.Timestamp.now().date()
        start_date = end_date - pd.DateOffset(months=lookback_months)

        # Essential fields for weekly/monthly strategies
        trading_fields = [
            'symbol', 'date', 'adj_close',
            '"本益比" as pe_ratio',
            '"股價淨值比" as pb_ratio',
            '"營收成長率" as revenue_growth',
            '"月營收" as monthly_revenue',
            '"成交量" as volume',
            '"市值" as market_cap'
        ]

        return self.existing.get_finlab_data(
            symbols,
            (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')),
            trading_fields
        )

    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality for trading strategies"""

        quality_metrics = {}

        # Check for missing data
        quality_metrics['missing_data_pct'] = (data.isnull().sum() / len(data) * 100).to_dict()

        # Check for outliers in key fields
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in data.columns:
                q1, q3 = data[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                outliers = ((data[col] < (q1 - 1.5 * iqr)) |
                           (data[col] > (q3 + 1.5 * iqr))).sum()
                quality_metrics[f'{col}_outliers'] = outliers

        # Check for data completeness by symbol
        symbol_completeness = data.groupby('symbol').size()
        quality_metrics['min_records_per_symbol'] = symbol_completeness.min()
        quality_metrics['max_records_per_symbol'] = symbol_completeness.max()
        quality_metrics['avg_records_per_symbol'] = symbol_completeness.mean()

        return quality_metrics

    def update_data_simple(self, symbols: Optional[List[str]] = None, incremental: bool = True) -> bool:
        """Simple wrapper around working production scripts"""
        import subprocess

        try:
            if incremental:
                result = subprocess.run(['python3', 'incremental_finlab_updater.py'],
                                      capture_output=True, text=True, check=True)
            else:
                result = subprocess.run(['python3', 'final_production_downloader.py'],
                                      capture_output=True, text=True, check=True)

            logging.info(f"Data update completed: {result.stdout}")
            return True

        except subprocess.CalledProcessError as e:
            logging.error(f"Data update failed: {e.stderr}")
            return False
```

## ✅ Acceptance Criteria

### Functional Requirements
- [ ] **Database Connection**: Successfully connects to existing PostgreSQL pipeline
- [ ] **Data Retrieval**: Efficiently retrieves data for 200+ symbols
- [ ] **Performance Benchmarking**: Query performance baseline established
- [ ] **Data Quality**: Validation framework for data integrity
- [ ] **Error Handling**: Robust error handling for production scenarios

### Performance Requirements
- [ ] **Query Speed**: <30 seconds for 200 symbols, 2-year history
- [ ] **Memory Efficiency**: <2GB memory usage for typical queries
- [ ] **Data Quality**: >95% data completeness for active symbols
- [ ] **Throughput**: >1000 records/second query performance

### Integration Requirements
- [ ] **ConfigManager**: Uses configuration for database settings
- [ ] **Existing Scripts**: Validates integration with production update scripts
- [ ] **Data Format**: Returns pandas DataFrames with consistent schema
- [ ] **Caching**: Supports caching layer for performance optimization

## 🧪 Testing Strategy

### Integration Tests
```python
def test_database_connection():
    """Test connection to existing PostgreSQL pipeline"""

def test_data_retrieval_performance():
    """Test query performance with different symbol counts"""

def test_data_quality_validation():
    """Test data quality metrics calculation"""

def test_production_script_integration():
    """Test integration with existing update scripts"""
```

### Performance Benchmarks
```sql
-- Test Query 1: Basic retrieval (50 symbols, 2 years)
EXPLAIN ANALYZE
SELECT symbol, date, adj_close, "本益比", "成交量"
FROM ml4t.finlab_data
WHERE symbol IN (...)
AND date >= '2023-01-01'
ORDER BY symbol, date;

-- Test Query 2: Aggregated data for factor computation
EXPLAIN ANALYZE
SELECT symbol,
       AVG(adj_close) as avg_price,
       STDDEV(adj_close) as price_volatility,
       COUNT(*) as trading_days
FROM ml4t.finlab_data
WHERE date >= '2023-01-01'
GROUP BY symbol;
```

## 📊 Success Metrics

- **Connection**: Successful connection to 4.35M record pipeline
- **Performance**: <30s query time for 200 symbols, 2-year history
- **Quality**: >95% data completeness validation
- **Integration**: ConfigManager and production scripts working
- **Documentation**: Complete integration guide and examples

## 🔗 Dependencies

- **Upstream**: Issue #01 ConfigManager Implementation
- **Downstream**: Issue #03 StratifiedSampler, Issue #04 StreamingProcessor
- **External**: PostgreSQL, pandas, SQLAlchemy, psycopg2

## 📝 Notes

- Focus on read-only access initially to avoid disrupting production pipeline
- Establish performance baselines for future optimization
- Ensure data access patterns support both weekly and monthly strategies
- Document any data quality issues discovered for future improvements

---

**Issue Status**: 📋 Ready for Development
**Next Issue**: #03 StratifiedSampler Development
**Critical Path**: Yes - enables all factor computation and backtesting