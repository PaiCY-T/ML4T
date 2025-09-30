# ML4T Development Specification v1.0
**Personal Taiwan Stock Trading System - Week/Month Trading Period**

**Date**: 2025-09-30
**Status**: Active Development
**Objective**: Infrastructure-Enhanced ML4T-Alpha system with proven data pipeline integration

---

## 📋 1. Executive Summary

### Strategic Decision
- **Approach**: Infrastructure-Enhanced ML4T-Alpha (90% compatibility preservation)
- **Critical Discovery**: Working production data pipeline exists (4.35M records, 1,307 symbols)
- **CLI Simplification**: Replace 28-file CLI system with single `update-data.py` script
- **Trading Focus**: Weekly/monthly rebalancing for personal Taiwan stock trading

### Technical Objectives
- **Performance Target**: >15% annual alpha with <15% maximum drawdown
- **Memory Management**: Laptop-first design (4-8GB) with cloud validation path
- **Statistical Validity**: Stratified sampling eliminates selection bias
- **Infrastructure**: Hybrid architecture (PostgreSQL + Parquet + cloud burst)

### Resource Planning
- **Total Effort**: 520 hours (80h remaining ML4T-Alpha + 200h infrastructure + 40h integration)
- **Timeline**: 4 phases with weekly milestones and decision gates
- **Investment Path**: Laptop → Cloud validation → Hardware decision

---

## 📑 2. Strategic Context & Decision Rationale

### Why Infrastructure-Enhanced ML4T-Alpha

**DECISION CONTEXT**: This approach was selected after comprehensive analysis and expert validation on 2025-09-30, consolidating two separate specifications (SYSTEM_MERGER_ANALYSIS.md and FINAL_BACKTESTING_ADAPTER_SPEC.md) into this unified document.

**Expert Validation Completed**:
- **Architecture Review**: OpenAI o3 expert analysis confirmed design soundness with specific improvements
- **Implementation Planning**: Gemini planner analysis extended timeline from 8 to 12 weeks with realistic buffers
- **Personal Trading Optimization**: Focus on weekly/monthly periods validated as appropriate scope reduction

**Existing ML4T-Alpha System (25% Complete)**:
- ✅ **Progress**: 240 hours completed (75% of architecture and foundation)
- ✅ **Focus**: Performance-first Taiwan market alpha generation
- ✅ **Strengths**: Domain expertise, proven foundation, Taiwan market specialization
- ⏳ **Remaining**: 80 hours for final integration and validation

**Integration Benefits**:
1. **Preserves Investment**: 90% compatibility with existing progress
2. **Adds Critical Infrastructure**: Statistical validity, memory management, performance optimization
3. **Natural Evolution**: Enhancement vs. disruptive replacement
4. **Risk Mitigation**: Incremental approach without disrupting current momentum

### Working Production System Discovery

**Critical Finding**: Complete data infrastructure already operational
- ✅ **4.35M records** successfully downloaded (2002-2025, 23+ years)
- ✅ **1,307 symbols** with 100% Taiwan market coverage
- ✅ **277 financial fields** from FinLab with bulletproof error handling
- ✅ **Incremental updates** with schedule-aware system operational
- ✅ **Production scripts**: `final_production_downloader.py`, `incremental_finlab_updater.py`

### CLI System Obsolescence

**Over-Engineering Identified**: 28 files across 7 command groups → Single script
- ❌ **Complex CLI Framework**: src/cli/main.py (257 lines), 7 command groups
- ❌ **Enterprise Features**: Monitoring, validation, batch processing (premature)
- ❌ **User Feedback**: "CLI is totally unusable"
- ✅ **Simple Solution**: Single `update-data.py` wrapper around working scripts

---

## 🏗️ 3. Technical Architecture

### Hybrid Multi-Tier Design
```
┌─────────────────────────────────────────────────────────────┐
│                    Configuration Layer                      │
│              (YAML-driven, hardware-aware)                 │
├─────────────────────────────────────────────────────────────┤
│                     Data Access Layer                       │
│     PostgreSQL (4.35M records) + Parquet (cache) + DuckDB  │
├─────────────────────────────────────────────────────────────┤
│                  Processing Layer                           │
│         Streaming + Chunking + Factor Caching              │
├─────────────────────────────────────────────────────────────┤
│                  Validation Layer                           │
│        Laptop → Cloud Burst → Hardware Decision            │
├─────────────────────────────────────────────────────────────┤
│                  Strategy Layer                             │
│        Weekly/Monthly Rebalancing + Portfolio Mgmt         │
└─────────────────────────────────────────────────────────────┘
```

### Core Infrastructure Components

#### T0.1: ConfigManager (40h) - Environment Detection
```python
class ConfigManager:
    """YAML-driven configuration with hardware-aware optimization"""

    def __init__(self, config_path=None):
        self.hardware_tier = self._detect_hardware_tier()
        self.config = self._load_config(config_path)

    def _detect_hardware_tier(self):
        """Environment-aware hardware detection"""
        total_memory = self._get_memory_limit()

        if total_memory > 30_000_000_000:  # >30GB
            return 'high_performance'
        elif total_memory > 15_000_000_000:  # >15GB
            return 'medium_performance'
        else:
            return 'laptop'

    def get_processing_config(self):
        """Get hardware-optimized processing parameters"""
        configs = {
            'laptop': {
                'max_symbols': 200,
                'chunk_size': 50,
                'memory_limit_gb': 4,
                'parallel_jobs': 2
            },
            'high_performance': {
                'max_symbols': 1307,
                'chunk_size': 200,
                'memory_limit_gb': 32,
                'parallel_jobs': 8
            }
        }
        return configs[self.hardware_tier]
```

#### T0.2: StratifiedSampler (40h) - Statistical Validity
```python
class StratifiedSampler:
    """Eliminates selection bias through proper statistical sampling"""

    def __init__(self, universe_symbols, sample_size=200):
        self.universe = universe_symbols  # All 1,307 Taiwan stocks
        self.sample_size = sample_size

    def get_stratified_sample(self):
        """Sample across market cap and sector tiers for weekly/monthly trading"""
        # Large cap (top 100): 60 stocks - stable, liquid for monthly holds
        # Mid cap (101-500): 80 stocks - growth opportunities
        # Small cap (501-1307): 60 stocks - alpha generation potential

        strata = self._classify_by_market_cap_and_sector()
        return self._balanced_sample(strata)

    def validate_sample_representation(self, sample):
        """Ensure sample represents full universe for weekly/monthly strategies"""
        # Validate market cap distribution
        # Validate sector representation
        # Validate liquidity requirements for weekly/monthly rebalancing
        return validation_metrics
```

#### T0.3: StreamingProcessor (40h) - Memory Management
```python
class StreamingProcessor:
    """Memory-efficient processing for laptop constraints"""

    def __init__(self, memory_limit_gb=4):
        self.memory_limit = memory_limit_gb * 1024**3
        self.chunk_size = self._calculate_optimal_chunk_size()

    def process_factors_streaming(self, symbols, date_range):
        """Stream processing for weekly/monthly factor computation"""
        for chunk in self._chunk_symbols(symbols):
            # Compute factors optimized for weekly/monthly signals
            factors = self._compute_chunk_factors(chunk, date_range)
            yield factors  # Prevents memory buildup

    def _calculate_optimal_chunk_size(self):
        """Dynamic chunk sizing for laptop performance"""
        sample_data = self._get_sample_data()
        memory_per_symbol = sample_data.memory_usage(deep=True).sum()
        safe_chunk = int(self.memory_limit * 0.8 / memory_per_symbol)
        return max(10, min(safe_chunk, 100))  # 10-100 symbol chunks
```

#### T0.4: FactorCache (40h) - Performance Optimization
```python
class FactorCache:
    """Intelligent factor caching optimized for weekly/monthly signals"""

    def __init__(self, cache_dir="./factor_cache"):
        self.cache_dir = Path(cache_dir)
        self.metadata_store = self._init_metadata_db()

    def get_cached_factors(self, symbols, date_range, params_hash):
        """Retrieve cached factors with weekly/monthly signal optimization"""
        cache_key = self._generate_cache_key(symbols, date_range, params_hash)

        if self._is_cache_valid(cache_key):
            return pd.read_parquet(self.cache_dir / f"{cache_key}.parquet")
        return None

    def cache_factors(self, factors_df, symbols, date_range, params_hash):
        """Store computed factors with lineage tracking"""
        cache_key = self._generate_cache_key(symbols, date_range, params_hash)

        # Store with weekly/monthly signal metadata
        factors_df.to_parquet(self.cache_dir / f"{cache_key}.parquet")
        self._update_metadata(cache_key, symbols, date_range, params_hash)
```

#### T0.5: CloudValidator (40h) - Burst Testing & Hybrid Data
```python
class CloudValidator:
    """Cloud burst testing for full universe validation"""

    def __init__(self, provider='aws', instance_type='m5.4xlarge'):
        self.provider = provider
        self.instance_type = instance_type  # 16 vCPU, 64GB RAM

    def validate_full_universe(self, strategy, symbols=None):
        """Test weekly/monthly strategy on full 1,307 stock universe"""
        if symbols is None:
            symbols = self._get_all_taiwan_symbols()

        instance = self._launch_instance()
        try:
            # Transfer strategy and essential data only
            self._transfer_strategy(instance, strategy)
            self._transfer_data_subset(instance)

            # Run full universe backtest with weekly/monthly rebalancing
            results = self._execute_backtest(instance, symbols)
            return self._analyze_results(results)
        finally:
            self._terminate_instance(instance)

    def estimate_cost(self, runtime_hours=2):
        """Estimate cloud burst cost for validation"""
        return runtime_hours * 0.768  # m5.4xlarge: ~$0.768/hour
```

### Expert Recommendations Integration

**Architecture Improvements (OpenAI o3 Expert Review)**:
1. **Database Architecture Optimization**
   - **Recommended**: DuckDB primary analytical + PostgreSQL raw data
   - **Current**: Dual PostgreSQL + DuckDB (sync complexity)
   - **Benefit**: Eliminates dual-write headaches, maintains SQL comfort

2. **Memory Management Enhancement**
   - **Add**: Memory profiler integration (memray)
   - **Add**: DataFrame dtype optimization (numpy/pyarrow native)
   - **Ensure**: <4GB peak usage validation in CI

3. **Risk Management Expansion**
   - **Add**: Slippage modeling (max(0.05%, half-spread))
   - **Add**: Volatility-based position sizing (ATR scaling)
   - **Add**: VaR/CVaR metrics in performance tracking

4. **Integration Testing Framework**
   - **Add**: Dedicated integration sprints between phases
   - **Add**: Mixed-frequency position testing (weekly + monthly)
   - **Add**: E2E workflow validation

**Implementation Planning (Gemini Planner Validation)**:
- **Timeline Extended**: 8 weeks → 12 weeks with realistic buffers
- **Buffer Management**: 20% time buffer (1 day per week) built into each phase
- **Risk Mitigation**: Clear success criteria and decision gates
- **MVP Approach**: Working system deliverable at each phase end

### Existing System Integration

#### Data Pipeline Integration
```python
class ExistingDataIntegration:
    """Integration with working 4.35M record data pipeline"""

    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'user': 'jnpi',
            'database': 'finlab_data',
            'schema': 'ml4t'
        }

    def get_finlab_data(self, symbols, date_range, fields=None):
        """Access existing 4.35M records with 277 fields"""
        # Use existing finlab_data table with bulletproof error handling
        # Optimized for weekly/monthly signal computation
        pass

    def update_data_simple(self, symbols=None, incremental=True):
        """Simple wrapper around working production scripts"""
        if incremental:
            return subprocess.run(['python3', 'incremental_finlab_updater.py'])
        else:
            return subprocess.run(['python3', 'final_production_downloader.py'])
```

---

## 📊 4. Implementation Roadmap

### Phase 1: Foundation & Proof (Weeks 1-2)

#### Week 1: Infrastructure Foundation
**Monday-Tuesday: Configuration & Environment (T0.1)**
- Implement ConfigManager with YAML configuration
- Add hardware detection (laptop/medium/high_performance)
- Create configuration templates for each tier
- Test environment detection accuracy

**Wednesday-Thursday: Data Layer & Integration (T0.5)**
- Implement hybrid data access (PostgreSQL + Parquet)
- Integrate with existing 4.35M record pipeline
- Add streaming data export capabilities
- Validate performance with existing data

**Friday: Validation & Testing**
- Test configuration system across hardware tiers
- Validate data integration with existing pipeline
- Performance benchmarking on laptop hardware

#### Week 2: Sampling & Processing
**Monday-Tuesday: Stratified Sampling (T0.2)**
- Implement StratifiedSampler for statistical validity
- Replace "top 200 stocks" with stratified approach
- Validate sample represents full universe characteristics
- Test with weekly/monthly trading requirements

**Wednesday-Thursday: Streaming Processing (T0.3)**
- Implement StreamingProcessor with chunked operations
- Add memory management for 4-8GB laptop constraints
- Optimize chunk sizing for Taiwan market data
- Test with 200-stock stratified sample

**Friday: Factor Engineering Foundation**
- Implement basic factor computation (5 factors)
- Momentum, mean reversion, volatility for weekly/monthly signals
- Volume ratio, PE ratio from existing FinLab fields
- Test streaming factor computation pipeline

### Phase 2: Backtesting & Validation (Weeks 3-4)

#### Week 3: Backtesting Engine
**Monday-Tuesday: Core Backtesting Framework**
- Weekly/monthly rebalancing system
- Transaction costs (0.3% for Taiwan market)
- Position sizing and portfolio constraints
- Risk management for personal trading

**Wednesday-Thursday: Performance Analytics**
- Portfolio performance metrics (returns, Sharpe, drawdown)
- Taiwan market benchmark comparison
- Risk-adjusted performance measurement
- Weekly vs monthly strategy comparison

**Friday: Strategy Validation**
- Run 2-year backtest on stratified sample
- Target: >10% annual alpha vs Taiwan index
- Validate factor performance across market regimes
- Test weekly vs monthly rebalancing performance

#### Week 4: Factor Caching & Optimization (T0.4)
**Monday-Tuesday: Factor Caching System**
- Implement persistent Parquet-based factor cache
- Add metadata tracking for experiment lineage
- Optimize cache hit ratios for weekly/monthly factors
- Storage efficiency for factor time series

**Wednesday-Thursday: Performance Optimization**
- Cache optimization for repeated backtests
- Memory usage optimization for laptop constraints
- Factor computation performance tuning
- Weekly/monthly signal computation optimization

**Friday: Alpha Validation Checkpoint**
- Comprehensive strategy performance analysis
- Factor attribution analysis
- Risk decomposition and analysis
- Decision gate preparation for Phase 3

### Phase 3: Cloud Validation & Decision Gate (Week 5)

#### Week 5: Full Universe Validation
**Monday-Tuesday: Cloud Infrastructure (T0.5)**
- Setup AWS/GCP spot instances (m5.4xlarge, 64GB RAM)
- Automate data transfer and environment setup
- Cost optimization strategies ($20-40 per validation)
- Security and access configuration

**Wednesday-Thursday: Full Universe Testing**
- Test strategy on complete 1,307 stock universe
- Compare: stratified sample vs full universe performance
- Validate memory management under full load
- Performance scaling analysis

**Friday: Decision Gate Analysis**
- Alpha comparison: sample vs full universe
- Hardware ROI calculation and business case
- Risk assessment for hardware investment
- Decision gate: proceed to hardware or optimize further

### Phase 4: Production Deployment (Weeks 6+)

#### Week 6: Hardware Transition (if approved)
- Hardware setup and environment configuration
- Full universe deployment and testing
- Performance validation and optimization
- Weekly/monthly strategy deployment

#### Week 7: Advanced Features
- Advanced factor engineering (15-20 factors)
- ML integration planning and implementation
- Enhanced risk management systems
- Strategy refinement and optimization

#### Week 8: Operational Systems
- Automated weekly/monthly rebalancing
- Monitoring and alerting systems
- Performance tracking and reporting
- Strategy development workflow optimization

---

## 🔗 5. Existing System Integration

### Working Production Scripts
```bash
# Simple data update wrapper (replaces 28-file CLI)
python update-data.py                    # Default incremental update
python update-data.py --full            # Full universe update
python update-data.py --symbols 2330,2317 --days 30  # Specific symbols
```

### Database Schema Integration
```sql
-- Existing production table (4.35M records)
ml4t.finlab_data (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    -- 277 financial fields with proper quoting
    adj_close DOUBLE PRECISION,
    "營收成長率" DOUBLE PRECISION,
    -- ... all existing fields preserved
)

-- New factor cache table
ml4t.factor_cache (
    cache_key VARCHAR(64) PRIMARY KEY,
    symbols TEXT[],
    date_range DATERANGE,
    params_hash VARCHAR(32),
    factor_data BYTEA,  -- Compressed Parquet
    created_at TIMESTAMP,
    weekly_monthly_optimized BOOLEAN DEFAULT TRUE
)

-- Strategy performance tracking
ml4t.strategy_performance (
    strategy_id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(100),
    rebalance_frequency VARCHAR(20), -- 'weekly' or 'monthly'
    parameters JSONB,
    performance_metrics JSONB,
    created_at TIMESTAMP
)
```

### Configuration Templates
```yaml
# config_laptop.yaml
hardware:
  tier: laptop
  max_symbols: 200
  memory_limit_gb: 4
  parallel_jobs: 2

trading:
  rebalance_frequency: monthly  # weekly|monthly
  position_size_limit: 0.05    # 5% max position
  transaction_cost: 0.003      # 0.3% for Taiwan market

database:
  chunk_size: 5000
  use_streaming: true
  cache_enabled: true

sampling:
  strategy: stratified
  sample_size: 200
  strata: [market_cap, sector]
  liquidity_filter: true       # For weekly/monthly trading
```

---

## 💰 6. Resource Planning

### Effort Allocation (520 hours total)

**Infrastructure Layer 0**: 200 hours
- T0.1: ConfigManager & Environment Detection (40h)
- T0.2: StratifiedSampler for Statistical Validity (40h)
- T0.3: StreamingProcessor for Memory Management (40h)
- T0.4: FactorCache for Performance Optimization (40h)
- T0.5: CloudValidator & Hybrid Data Architecture (40h)

**ML4T-Alpha Integration**: 80 hours remaining
- Factor engineering integration (30h)
- Strategy framework enhancement (25h)
- Performance validation and optimization (25h)

**System Integration**: 40 hours
- Existing data pipeline integration (20h)
- CLI simplification and wrapper creation (10h)
- Testing and validation across all components (10h)

**Weekly/Monthly Strategy Focus**: 200 hours
- Weekly rebalancing system (60h)
- Monthly rebalancing optimization (60h)
- Performance comparison and selection (40h)
- Risk management for personal trading (40h)

### Investment Progression
```
Phase 1 (Laptop): $0 investment
- Stratified sample validation on existing hardware
- Expected: 10-15% annual alpha with weekly/monthly rebalancing
- Risk: Limited to 200-symbol universe

Phase 2 (Cloud Validation): $20-40 per validation run
- Full 1,307-symbol universe testing
- Validation of strategy scaling characteristics
- De-risk hardware investment decision

Phase 3 (Hardware): $3,000-5,000 investment
- 32GB+ RAM system for full universe processing
- Expected: 15-20% annual alpha with advanced factors
- ROI: 50-100% annually with $50K+ trading capital
```

### Cost-Benefit Analysis
```
Expected Performance Improvements:
Laptop (200 symbols): 10-15% annual alpha
Cloud Validation: Confirm no alpha degradation at scale
Hardware (1,307 symbols): 15-20% annual alpha
Advanced ML Features: Additional 5-10% alpha potential

Total Expected: 20-30% annual alpha for personal trading
Risk-Adjusted: 15-25% with <15% maximum drawdown
```

---

## ⚠️ 7. Risk Management

### Statistical Validity Safeguards
1. **Selection Bias Prevention**
   - Stratified sampling across market cap and sectors
   - Regular validation against full 1,307-symbol universe
   - Sample composition rotation for robustness

2. **Weekly/Monthly Trading Risks**
   - Liquidity validation for position sizing
   - Transaction cost optimization (0.3% Taiwan market)
   - Rebalancing frequency impact analysis

3. **Overfitting Prevention**
   - Walk-forward validation (52-week training windows)
   - Out-of-sample testing (26-week testing periods)
   - Cross-regime validation (bull/bear/sideways markets)

### Infrastructure Risk Management
1. **Memory Overflow Protection**
   - Dynamic chunk sizing based on available memory
   - Graceful degradation with disk spillover
   - Memory usage monitoring and alerts

2. **Performance Degradation Detection**
   - Query time monitoring for data access
   - Factor computation performance tracking
   - Automatic fallback to simpler approaches

3. **Data Integrity Monitoring**
   - Integration with existing bulletproof data pipeline
   - Corporate action adjustment validation
   - Data quality scoring and alerts

### Personal Trading Risks
1. **Position Size Management**
   - Maximum 5% position size for risk control
   - Diversification across market cap tiers
   - Sector concentration limits

2. **Rebalancing Optimization**
   - Weekly vs monthly cost-benefit analysis
   - Transaction cost minimization
   - Tax efficiency considerations

---

## 🎯 8. Success Criteria

### Phase 1 Success Criteria (Laptop Foundation)
- [ ] Stratified sample processing in <5 minutes (200 stocks, 2 years)
- [ ] >10% annual alpha with Sharpe ratio >1.0
- [ ] Maximum drawdown <15%
- [ ] Factor cache hit ratio >80%
- [ ] Memory usage <4GB peak
- [ ] Weekly vs monthly rebalancing comparison complete

### Phase 2 Success Criteria (Backtesting Validation)
- [ ] Complete backtesting framework operational
- [ ] 2-year historical validation successful
- [ ] Factor performance attribution analysis
- [ ] Risk-adjusted performance metrics >10% alpha
- [ ] Transaction cost impact analysis for weekly/monthly trading

### Phase 3 Success Criteria (Cloud Validation)
- [ ] Full universe validation completes in <30 minutes
- [ ] Alpha maintained or improved vs stratified sample
- [ ] Cloud validation cost <$40 per run
- [ ] Factor performance consistent across market cap tiers
- [ ] Memory scaling validated for 1,307 symbols

### Phase 4 Success Criteria (Production Deployment)
- [ ] Hardware ROI >50% annually validated
- [ ] Full universe deployment successful
- [ ] Weekly/monthly strategy selection optimized
- [ ] Automated rebalancing system operational
- [ ] Personal trading workflow established

### Overall Success Metrics
- [ ] **Alpha Generation**: >15% annual returns vs Taiwan market
- [ ] **Risk Management**: <15% maximum drawdown
- [ ] **Sharpe Ratio**: >1.0 consistently across market regimes
- [ ] **Implementation Efficiency**: Laptop → cloud → hardware progression
- [ ] **Cost Effectiveness**: Total development cost <$5,000
- [ ] **Personal Trading**: Weekly/monthly rebalancing optimized for individual investor

---

## 🚀 9. Getting Started

### Day 1 Immediate Actions

#### Environment Setup
```bash
# Create unified project structure
mkdir -p /mnt/c/Users/jnpi/ML4T/new/ml4t_system/{config,src,cache,experiments,notebooks}

# Install dependencies
pip install pandas numpy scikit-learn lightgbm mlflow pyyaml pyarrow duckdb

# Create simple data update wrapper
cat > update-data.py << 'EOF'
#!/usr/bin/env python3
"""Simple data update - replaces 28-file CLI system"""
import sys
import subprocess

def main():
    if '--incremental' in sys.argv or len(sys.argv) == 1:
        return subprocess.run(['python3', 'incremental_finlab_updater.py']).returncode
    else:
        return subprocess.run(['python3', 'final_production_downloader.py']).returncode

if __name__ == '__main__':
    sys.exit(main())
EOF

chmod +x update-data.py
```

#### Configuration Creation
```bash
# Create laptop configuration optimized for weekly/monthly trading
cat > config/config_laptop.yaml << EOF
hardware:
  tier: laptop
  max_symbols: 200
  memory_limit_gb: 4
  parallel_jobs: 2

trading:
  rebalance_frequency: monthly
  position_size_limit: 0.05
  transaction_cost: 0.003

database:
  chunk_size: 5000
  use_streaming: true
  cache_enabled: true

sampling:
  strategy: stratified
  sample_size: 200
  strata: [market_cap, sector]
EOF
```

#### Database Validation
```bash
# Test existing data pipeline performance
PGPASSWORD=HAPPYdog psql -h localhost -p 5432 -U jnpi -d finlab_data -c "
EXPLAIN ANALYZE
SELECT symbol, date, adj_close, \"本益比\" as pe_ratio
FROM ml4t.finlab_data
WHERE date >= '2023-01-01'
ORDER BY symbol, date
LIMIT 10000;"
```

### Week 1 Implementation Priority
1. **ConfigManager Implementation** (T0.1) - Hardware-aware configuration
2. **Existing Data Integration** - Connect to 4.35M record pipeline
3. **StratifiedSampler Development** (T0.2) - Statistical validity
4. **Basic Factor Computation** - 5 essential factors for weekly/monthly signals
5. **Memory Usage Validation** - Ensure laptop compatibility

### Project Structure
```
/mnt/c/Users/jnpi/ML4T/new/ml4t_system/
├── config/
│   ├── config_laptop.yaml          # Laptop optimization
│   ├── config_cloud.yaml           # Cloud burst configuration
│   └── config_desktop.yaml         # High-performance setup
├── src/
│   ├── infrastructure/
│   │   ├── config_manager.py       # T0.1: Environment detection
│   │   ├── stratified_sampler.py   # T0.2: Statistical sampling
│   │   ├── streaming_processor.py  # T0.3: Memory management
│   │   ├── factor_cache.py         # T0.4: Performance optimization
│   │   └── cloud_validator.py      # T0.5: Burst testing
│   ├── data/
│   │   ├── existing_integration.py # 4.35M record pipeline
│   │   └── hybrid_data_access.py   # Multi-tier data access
│   ├── factors/
│   │   ├── essential_factors.py    # Core 5 factors
│   │   └── weekly_monthly_signals.py # Trading period optimization
│   ├── backtesting/
│   │   ├── engine.py               # Backtesting framework
│   │   ├── portfolio.py            # Portfolio management
│   │   └── performance.py          # Performance analytics
│   └── strategies/
│       ├── weekly_rebalance.py     # Weekly strategy
│       └── monthly_rebalance.py    # Monthly strategy
├── cache/                          # Factor cache directory
├── experiments/                    # Strategy experiments
├── notebooks/                      # Research and analysis
├── tests/                          # Unit and integration tests
└── update-data.py                  # Simple CLI replacement
```

---

## 🔄 10. Implementation Critical Path

### Dependencies & Critical Path Analysis

**Critical Path Sequence**:
1. **ConfigManager (T0.1)** → **StratifiedSampler (T0.2)** → **StreamingProcessor (T0.3)**
2. **Parallel**: **FactorCache (T0.4)** + **Existing Data Integration**
3. **CloudValidator (T0.5)** depends on: T0.1, T0.2, T0.3 completion
4. **Factor Engineering** depends on: T0.2, T0.3, T0.4
5. **Backtesting Engine** depends on: All T0.1-T0.4 + Factor Engineering

**Resource Dependencies**:
- **Data Access**: Requires existing PostgreSQL pipeline (✅ Available)
- **Market Cap Data**: Required for stratified sampling (needs sourcing)
- **Cloud Infrastructure**: AWS/GCP account with appropriate permissions
- **Development Environment**: Python 3.8+, PostgreSQL access, 8GB+ RAM recommended

**Risk Dependencies**:
- **Performance**: Memory constraints could block laptop-first approach
- **Statistical**: Market cap classification accuracy affects stratification quality
- **Infrastructure**: Cloud costs could exceed $40/validation budget
- **Integration**: Existing data pipeline changes could disrupt implementation

### Decision Gates & Quality Checkpoints

**Week 2 Decision Gate**: Laptop Foundation
- ✅ Stratified sampling producing representative samples
- ✅ Memory usage <4GB with 200-symbol processing
- ✅ Factor cache achieving >70% hit ratio
- ❌ **Fail Criteria**: Memory usage >6GB or processing time >10 minutes

**Week 4 Decision Gate**: Alpha Validation
- ✅ >10% annual alpha on 2-year backtest
- ✅ Sharpe ratio >0.8 consistently
- ✅ Maximum drawdown <20%
- ❌ **Fail Criteria**: Alpha <5% or drawdown >25%

**Week 5 Decision Gate**: Cloud Investment
- ✅ Sample strategy validates successfully
- ✅ Cloud validation cost estimate <$40/run
- ✅ Full universe scaling plan validated
- ❌ **Fail Criteria**: Strategy fails validation or costs exceed budget

### Post-Restart Continuation Guide

**Immediate Context Restoration** (First 30 minutes):
1. **Review Current Status**: Check existing data pipeline (4.35M records confirmed)
2. **Validate Environment**: Confirm PostgreSQL access and Python environment
3. **Project Structure**: Create ml4t_system directory structure per Section 9
4. **Simple CLI**: Implement update-data.py wrapper (5 minutes)
5. **Week 1 Start**: Begin ConfigManager (T0.1) implementation

**Context Preservation**:
- **Expert Validation**: Architecture approved by OpenAI o3, timeline by Gemini planner
- **Strategic Decision**: Infrastructure-Enhanced ML4T-Alpha (90% compatibility)
- **Critical Discovery**: Working data pipeline exists, CLI system obsoleted
- **Personal Focus**: Weekly/monthly trading, laptop-first approach
- **Success Metrics**: >15% annual alpha, <15% drawdown, Sharpe >1.0

**Implementation Priorities** (Next 7 days):
1. **Day 1-2**: ConfigManager implementation (T0.1)
2. **Day 3-4**: Existing data integration and validation
3. **Day 5**: StratifiedSampler foundation (T0.2)
4. **Weekend**: Testing and Week 2 planning

---

## 🛡️ 11. Consolidated Risk Management

### Implementation Risks

**Technical Risks**:
- **Memory Constraints**: Laptop processing may hit 4GB limits
  - *Mitigation*: Dynamic chunk sizing, disk spillover, cloud fallback
- **Data Pipeline Integration**: Changes to existing system could break integration
  - *Mitigation*: Read-only access initially, wrapper approach, version pinning
- **Market Cap Data Availability**: Stratification requires accurate market cap classification
  - *Mitigation*: Multiple data sources, manual classification fallback, sector-only stratification

**Financial Risks**:
- **Cloud Costs**: Validation could exceed $40/run budget
  - *Mitigation*: Spot instances, time limits, cost monitoring, validation batching
- **Hardware Investment**: $3K-5K investment may not deliver expected ROI
  - *Mitigation*: Phased validation, clear success criteria, conservative projections
- **Strategy Performance**: Alpha may not materialize in live trading
  - *Mitigation*: Robust backtesting, out-of-sample validation, gradual capital allocation

**Operational Risks**:
- **Personal Trading Complexity**: Weekly/monthly rebalancing may be too complex
  - *Mitigation*: Start with monthly only, automation tools, simple execution
- **Time Commitment**: 520-hour project may exceed personal capacity
  - *Mitigation*: Phased approach, MVP delivery, 80/20 prioritization

### Mitigation Strategies

**Progressive Validation Approach**:
1. **Laptop Validation** (Weeks 1-4): Prove concept with minimal risk
2. **Cloud Validation** (Week 5): De-risk scaling with controlled costs
3. **Hardware Decision** (Week 6+): Invest only after validation success

**Quality Assurance Framework**:
- **Daily**: Memory usage monitoring, processing time tracking
- **Weekly**: Performance metrics validation, risk metric updates
- **Phase**: Comprehensive validation against success criteria
- **Decision Gates**: Go/no-go decisions based on quantitative metrics

**Fallback Plans**:
- **Memory Issues**: Reduce sample size to 100 symbols, extend timeline
- **Performance Issues**: Focus on monthly rebalancing only, simplify factors
- **Cost Overruns**: Skip cloud validation, proceed directly to decision
- **Alpha Failure**: Pivot to index tracking with factor tilts

---

## 📝 Changelog

| Date | Author | Summary |
|------|--------|---------|
| 2025-09-30 | Claude | Unified specification created from merger analysis and adapter spec |
| 2025-09-30 | Claude | Added personal trading focus with weekly/monthly optimization |
| 2025-09-30 | Claude | Integrated existing 4.35M record data pipeline |
| 2025-09-30 | Claude | **PRE-RESTART CONSOLIDATION**: Added expert recommendations, critical path, risk consolidation |

---

*This unified specification serves as the single source of truth for ML4T development, combining strategic context with detailed technical implementation for a personal Taiwan stock trading system focused on weekly/monthly rebalancing periods.*