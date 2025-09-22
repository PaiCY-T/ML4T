---
name: ML4T
status: backlog
created: 2025-09-21T16:44:05Z
progress: 0%
prd: .claude/prds/ML4T.md
github: [Will be updated when synced to GitHub]
---

# Epic: ML4T (Machine Learning for Trading MVP)

## Overview

Implement a **proof-of-concept machine learning trading system** to validate systematic alpha generation in Taiwan Stock Exchange (TSE) markets. The MVP focuses on a single weekly momentum strategy (MOM_W) with 6-month paper trading validation before scaling to full system.

**Core Technical Hypothesis**: Can LightGBM systematically generate 12-15% CAGR with Sharpe >1.2 using TSE momentum factors after realistic transaction costs (0.37% roundtrip)?

## Architecture Decisions

### **Data Architecture**: FinLab-Centric with Local Storage
- **Primary Data Source**: FinLab API for historical/EOD TSE data (2000+ stocks)
- **Real-time Source**: Fubon API for paper trading and market snapshots
- **Storage Strategy**: Local Parquet files (no database complexity for MVP)
- **Caching**: FinLab's built-in `save_to_storage=True` for development efficiency

### **ML Pipeline**: Simplified Time-Series Approach
- **Model**: Single LightGBM with hyperparameter tuning (no ensemble complexity)
- **Cross-Validation**: TimeSeriesSplit with proper purging to avoid look-ahead bias
- **Features**: 3-5 momentum factors (price_momentum, RSI, SMA_ratio, volatility)
- **Labels**: Weekly forward returns with Taiwan-specific cost adjustments

### **Risk Management**: Conservative MVP Approach
- **Position Limits**: Max 20 stocks, 5% per position (conservative for concept validation)
- **Sizing**: Equal-weight or simple volatility targeting (no risk parity complexity)
- **Stop-Loss**: 10% individual stock stops, 15% portfolio drawdown trigger
- **Validation**: 6-month paper trading before any real capital

### **Technology Stack**: Leveraging Existing Architecture
```python
Core: pandas, numpy, scikit-learn, lightgbm
FinLab: finlab (Taiwan market data and backtesting)
Fubon: fubon-neo (paper trading integration)
Storage: pyarrow (Parquet files for factor storage)
Existing: Proven broker interfaces, IC evaluators, incremental updaters
```

### **Reference Code Integration**
**Fubon API Foundation**: 4 production-ready integration modules
- `FubonClient`: Authentication and connection management with health checks
- `FubonAccountManager`: Account data and portfolio tracking with caching
- `FubonMarketData`: Real-time WebSocket + REST API integration
- `FubonOrderManager`: Order lifecycle management with batch processing

**Data Processing Foundation**: 2 specialized modules
- `IncrementalUpdater`: Taiwan market-specific data updates with backup/recovery
- `ProfessionalFactorEvaluator`: IC calculation with Taiwan benchmark thresholds

**Architectural Advantages**:
- **Proven Reliability**: Error handling, retry logic, graceful degradation
- **Taiwan Market Optimized**: Quarter deadlines, regulatory compliance built-in
- **Performance Optimized**: Caching, incremental processing, memory management
- **Production Ready**: Monitoring, logging, health checks, backup systems

## Technical Approach

### Data Pipeline
- **Historical Data Setup**: FinLab API integration with 10+ years TSE/OTC data
- **Factor Engineering**: Implement 3-5 momentum factors using FinLab's indicator library
- **Data Quality**: Basic validation, forward-fill, and survivorship bias checks
- **Cost Modeling**: Taiwan-specific transaction costs (0.04275% fee + 0.3% tax + slippage)

### ML Strategy Engine
- **Single Strategy Focus**: MOM_W (weekly momentum) only for MVP validation
- **Feature Pipeline**: `data.indicator()` for RSI, SMA; custom momentum calculations
- **Model Training**: LightGBM with time-series CV and walk-forward validation
- **Signal Generation**: Top-N stock selection based on ML factor scores

### Backtesting & Validation
- **Engine**: FinLab's `backtest.sim()` with weekly rebalancing
- **Cost Integration**: Realistic Taiwan transaction costs and slippage
- **Walk-Forward**: 3-year rolling windows for temporal validation
- **Performance Metrics**: Sharpe ratio, CAGR, MDD, Information Coefficient

### Paper Trading System
- **Execution**: Fubon API integration for order simulation
- **Monitoring**: Daily performance tracking vs backtest expectations
- **Risk Controls**: Pre-trade validation and position limit enforcement
- **Reconciliation**: End-of-day performance attribution and cost tracking

## Implementation Strategy

### **Phase 1: Backtest Validation (Weeks 1-6)**
**Critical Path**: Prove concept with historical data before any live trading
- Setup FinLab data pipeline and factor engineering
- Implement MOM_W strategy with Taiwan-specific cost modeling
- Complete 10+ year walk-forward backtesting analysis
- **Success Gate**: 12%+ CAGR, Sharpe >1.2, IC >0.02 after costs

### **Phase 2: Paper Trading (Weeks 7-18)**
**Critical Path**: Validate backtest assumptions with live market data
- Integrate Fubon API for paper trading execution
- Daily monitoring and performance attribution
- Cost model validation and slippage analysis
- **Success Gate**: Paper performance within 5% of backtest expectations

### **Phase 3: Scale Decision (Weeks 19-20)**
**Critical Path**: Evidence-based go/no-go decision for full system
- Comprehensive performance analysis across market regimes
- Strategy capacity analysis for 30M NTD scaling
- Go/Modify/No-Go decision based on success criteria

### Risk Mitigation
- **Conservative Cost Estimates**: 0.37% roundtrip (vs typical 0.25-0.3%)
- **Extended Validation**: 6-month paper trading minimum before real capital
- **Simple ML Approach**: Single model to reduce overfitting risk
- **Manual Oversight**: Human validation of all trading decisions

## Task Breakdown

### Task Summary

✅ **Created 9 tasks for epic: ML4T** (Total: 1400 hours)

**Layer 1: Data & Factor Validation** (6 parallel tasks)
- [001] Data Foundation Setup (150h)
- [002] Factor Engineering Pipeline (200h)
- [003] Taiwan Market Data Integration (120h)

**Layer 2: Strategy & Backtest Validation** (1 sequential task)
- [004] MOM_W Strategy Implementation (180h) - *depends on 001-003*

**Layer 3: ML & Trading Validation** (2 sequential tasks)
- [005] ML Model Development (200h) - *depends on 004*
- [006] Backtesting Engine (150h) - *depends on 004*

**Layer 4: Execution & Integration** (3 parallel tasks)
- [007] Paper Trading System (160h) - *depends on 005-006*
- [008] Risk Management System (120h) - *depends on 005-006*
- [009] Performance Analytics (100h) - *depends on 005-006*

**Execution Strategy**:
- **Parallel Efficiency**: 6 of 9 tasks can run in parallel (67% parallelizable)
- **Critical Path**: 001-003 → 004 → 005-006 → 007-009
- **Risk Mitigation**: 3-layer validation gates prevent downstream failures

## Dependencies

### **External Dependencies**
- **FinLab API**: Taiwan market data access and factor library
- **Fubon Neo SDK**: Paper trading and real-time market data
- **Taiwan Stock Exchange**: Market data availability and trading hours
- **Python Environment**: Core ML and data science libraries

### **Internal Dependencies**
- **Windows Environment**: Required for Fubon SDK compatibility
- **Development Machine**: Local storage and processing capabilities
- **Network Connectivity**: Stable internet for API access and data updates

### **Critical Path Dependencies**
1. **FinLab Data Access**: Must validate data quality before strategy development
2. **Factor Engineering**: Must prove factor significance before ML model training
3. **Backtest Validation**: Must achieve success criteria before paper trading
4. **Paper Trading Setup**: Must complete Fubon integration before live validation

## Success Criteria (Technical)

### **Performance Benchmarks**
- **Backtest Performance**: 12-15% CAGR, Sharpe >1.2, MDD <15% over 10+ years
- **Factor Significance**: Information Coefficient >0.02 for momentum factors
- **Cost Realism**: Positive alpha after 0.37% roundtrip transaction costs
- **System Performance**: <10 min backtests, <15 min daily processing

### **Quality Gates**
- **Data Quality**: <0.1% missing data requiring interpolation
- **Model Validation**: Consistent performance across 5-fold time-series CV
- **Paper Trading**: Live performance within 5% of backtest expectations
- **System Reliability**: 95%+ uptime during 6-month validation period

### **Acceptance Criteria**
- **Concept Validation**: Momentum factors show persistent alpha in TSE markets
- **ML Value-Add**: LightGBM outperforms simple factor ranking by >1% annually
- **Scalability Proof**: Strategy works with position sizes appropriate for 30M NTD
- **Execution Validation**: Paper trading costs match backtest assumptions

## Estimated Effort

### **Overall Timeline**: 20 weeks (with 4-week buffer from original 16-week plan)
- **Phase 1 (Backtest)**: 6 weeks intensive development
- **Phase 2 (Paper Trading)**: 12 weeks validation with daily monitoring
- **Phase 3 (Decision)**: 2 weeks analysis and planning

### **Resource Requirements**
- **Developer**: Single experienced quant developer/trader
- **Infrastructure**: Local Windows machine with FinLab/Fubon API access
- **Capital**: 1-3M NTD for paper trading validation (no real risk)

### **Critical Path Items**
1. **Week 1-2**: FinLab data pipeline must be rock-solid foundation
2. **Week 6**: Backtest validation gate - critical go/no-go decision
3. **Week 18**: Paper trading validation - final concept proof
4. **Week 20**: Scale decision - determines full system development

**Success Philosophy**: Validate simple concept thoroughly rather than build complex system on unproven foundations. Better to spend 20 weeks proving MVP than 40 weeks building failed system.