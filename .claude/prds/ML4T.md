---
name: ML4T
description: MVP machine learning trading system to prove systematic alpha concept for Taiwan Stock Exchange
status: backlog
created: 2025-09-21T15:18:44Z
---

# PRD: ML4T (Machine Learning for Trading) - MVP Validation

## Executive Summary

ML4T MVP is a **proof-of-concept** machine learning trading system designed to validate whether systematic factor-based alpha generation is viable for Taiwan Stock Exchange (TSE). The MVP focuses on proving the core hypothesis with a single strategy before building a comprehensive multi-strategy system.

**Core Hypothesis**: Can machine learning systematically generate alpha in TSE markets using weekly momentum factors?

**MVP Objectives (Proof of Concept):**
- **Primary Target**: 12-15% annual return, Sharpe ratio 1.2-1.5
- **Stretch Goal**: 20% annual return, Sharpe ratio >2.0
- **Risk Constraint**: Maximum drawdown <15%
- **MVP Capital**: 1-3M NTD for initial validation
- **Success Criteria**: 6-month paper trading validation before scaling to 30M NTD

## Problem Statement

### What problem are we solving?
**Core Question**: Does systematic ML-driven trading provide sustainable alpha in Taiwan markets?

**Specific Challenges to Validate:**
1. **Factor Effectiveness**: Do momentum/value factors work consistently in TSE?
2. **ML Value-Add**: Does machine learning improve upon simple factor scores?
3. **Transaction Cost Impact**: Can alpha overcome Taiwan's transaction costs (0.04275% + 0.3% tax)?
4. **Market Microstructure**: Can systematic approaches handle TSE liquidity patterns?

### Why MVP-first approach?
- **Risk Mitigation**: Prove concept before major capital commitment
- **Learning Priority**: Understand TSE market behavior before complexity
- **Cost Efficiency**: Validate hypothesis with minimal infrastructure investment
- **Reality Check**: Test if ambitious targets (Sharpe >2.0) are achievable

## User Stories

### Primary User Persona: Quantitative Trader (Concept Validation)

**User Journey 1: Hypothesis Validation**
- As a trader, I want to prove systematic alpha exists in TSE so that I can justify building a full system
- **Acceptance Criteria:**
  - Single strategy (MOM_W) consistently outperforms TSE benchmark over 2+ years backtest
  - Walk-forward analysis shows stable performance across market regimes
  - Information Coefficient >0.02 for momentum factors in TSE

**User Journey 2: Cost Model Validation**
- As a trader, I want realistic transaction cost modeling so that I understand true alpha after costs
- **Acceptance Criteria:**
  - Accurate Taiwan-specific cost modeling (0.04275% fee + 0.3% tax)
  - Slippage estimation based on order size vs TSE daily volumes
  - Backtest vs live trading performance tracking <5% deviation

**User Journey 3: Infrastructure Validation**
- As a trader, I want end-to-end pipeline validation so that I can scale with confidence
- **Acceptance Criteria:**
  - Automated data pipeline (FinLab EOD + Fubon real-time)
  - Paper trading execution matches backtest expectations
  - System runs reliably for 3+ months without manual intervention

**User Journey 4: Scalability Assessment**
- As a trader, I want to understand system limitations so that I can plan capital deployment
- **Acceptance Criteria:**
  - Strategy capacity analysis for TSE liquidity constraints
  - Performance degradation thresholds identified
  - Clear go/no-go criteria for scaling to full system

## Requirements

### MVP Functional Requirements

#### Core Validation Engine (Single Strategy Focus)
- **MOM_W Strategy Only**: Weekly momentum with 3-5 key technical factors
- **Simplified ML Pipeline**: LightGBM with time-series cross-validation
- **TSE-Specific Backtesting**: 10+ years Taiwan market data with realistic costs
- **Walk-Forward Validation**: 3-year rolling windows to prove consistency

#### Minimal Data Infrastructure
- **EOD Data**: FinLab API integration for historical and daily updates
- **Basic Real-time**: Fubon API for market snapshots and paper trading
- **Simple Storage**: Local CSV/Parquet files (no database complexity)
- **Data Quality**: Basic validation and forward-fill for missing data

#### Proof-of-Concept Risk Management
- **Position Limits**: Max 20 stocks, 5% per position (conservative for MVP)
- **Simple Sizing**: Equal-weight or volatility targeting
- **Basic Stops**: 10% individual stock stop-loss
- **Portfolio Protection**: 15% maximum drawdown trigger

#### MVP Execution System
- **Paper Trading Only**: Validate execution logic without capital risk
- **Daily Rebalancing**: Simple weekly signal with daily execution
- **Manual Override**: All orders reviewed before submission
- **Performance Tracking**: Paper vs backtest deviation monitoring

### Non-Functional Requirements

#### Performance (MVP Scope)
- **Backtest Speed**: 10-year Taiwan market backtest in <10 minutes
- **Data Processing**: Daily EOD processing in <15 minutes
- **Signal Generation**: Weekly portfolio rebalancing in <5 minutes
- **Monitoring**: Daily performance tracking and alert system

#### Simplicity and Reliability
- **Minimal Dependencies**: Core Python stack only (pandas, numpy, sklearn)
- **Single Machine**: Local development environment only
- **Manual Oversight**: Human validation of all key decisions
- **Basic Logging**: Simple file-based logging for debugging

#### MVP-Specific Constraints
- **No Real Trading**: Paper trading only until concept proven
- **Single Strategy**: Focus on MOM_W until validated
- **Taiwan Only**: TSE/OTC markets only
- **Conservative Sizing**: Small positions to minimize any unforeseen risks

## Success Criteria

### MVP Validation Metrics (6-Month Evaluation)

#### Primary Success Criteria (Must Achieve All)
- **Backtest Performance**: 12-15% CAGR, Sharpe >1.2, MDD <15% over 10+ years
- **Factor Significance**: Information Coefficient >0.02 for momentum factors
- **Cost Realism**: Post-cost returns remain positive after 0.04275% + 0.3% costs
- **Paper Trading**: Live paper performance within 5% of backtest expectations

#### Secondary Success Criteria (Prove Scalability)
- **Consistency**: Positive returns in 7/10 years of walk-forward analysis
- **Market Regime Robustness**: Strategy works in bull, bear, and sideways markets
- **Liquidity Validation**: Strategy works with realistic position sizes for 30M NTD
- **System Reliability**: 95%+ uptime during paper trading period

#### Stretch Success Criteria (Exceptional Performance)
- **High Sharpe**: Achieve Sharpe ratio >2.0 consistently
- **Low Volatility**: Portfolio volatility <15% annually
- **High Hit Rate**: >60% winning trades
- **Strong IC**: Information Coefficient >0.05

### Go/No-Go Decision Framework

#### **GO**: Scale to Full System (30M NTD)
- All primary success criteria met
- 3+ secondary criteria achieved
- 6-month paper trading successful
- Clear understanding of capacity limits

#### **MODIFY**: Adjust Strategy/Parameters
- Primary criteria partially met
- Paper trading shows systematic biases
- Need parameter optimization or factor adjustment
- Extend paper trading period

#### **NO-GO**: Abandon or Redesign
- Primary success criteria not met
- Negative or inconsistent paper trading results
- Transaction costs eliminate alpha
- Strategy doesn't work in TSE market structure

## Constraints & Assumptions

### MVP-Specific Constraints
- **Capital Limitation**: Paper trading only, no real capital at risk
- **Strategy Limitation**: Single strategy (MOM_W) validation only
- **Market Limitation**: Taiwan TSE/OTC only
- **Time Limitation**: 6-month validation period before scaling decision

### Updated Cost Assumptions
- **Brokerage Fee**: 0.04275% (corrected: 0.1425% × 0.3)
- **Securities Transaction Tax**: 0.3% on sell orders
- **Estimated Slippage**: 0.02-0.05% based on order size
- **Total Transaction Cost**: ~0.37% roundtrip (conservative estimate)

### Critical Assumptions to Validate
- **Factor Persistence**: TSE momentum factors remain predictive over time
- **Market Efficiency**: Semi-strong form efficiency allows factor-based alpha
- **Data Quality**: FinLab provides clean, survivorship-bias-free data
- **Execution Reality**: Paper trading reflects actual execution characteristics

## Out of Scope (MVP Phase)

### Explicitly NOT Building in MVP
- **Multiple Strategies**: Focus on single strategy validation only
- **Real Trading**: Paper trading only until concept proven
- **Complex ML**: Simple LightGBM, no deep learning or ensemble methods
- **Database Systems**: File-based storage sufficient for MVP
- **Real-time Dashboard**: Basic logging and alerts sufficient
- **Options/Derivatives**: Equity long-only for simplicity
- **Risk Parity**: Equal weight or simple volatility targeting only

### Reserved for Full System (If MVP Succeeds)
- **Multi-Strategy Framework**: Add VAL_M and ML_RANK strategies
- **Advanced ML Pipeline**: Ensemble methods, feature engineering
- **Production Infrastructure**: Database, monitoring, alerting systems
- **Risk Management**: Sophisticated position sizing and hedging
- **Live Trading**: Real capital deployment with full automation

## Dependencies

### Critical Dependencies (Must Have)
- **FinLab API**: Historical TSE data and basic factors
- **Fubon API**: Real-time data and paper trading capability
- **Python Environment**: Core ML and data analysis libraries
- **TSE Market Access**: Taiwan market data and trading infrastructure

### Optional Dependencies (Nice to Have)
- **Advanced Charting**: For manual strategy analysis
- **Database System**: For production scalability
- **Cloud Infrastructure**: For system redundancy
- **Additional Data Sources**: For strategy diversification

## Technical References

### FinLab API Integration
**Reference**: `/mnt/c/Users/jnpi/ML4T/new/example/FINLAB_API_REFERENCE.md`

**Key FinLab Capabilities for MVP:**
- **Data Access**: `data.get('price:收盤價')` for OHLCV data, 2000+ Taiwan stocks
- **Technical Indicators**: 100+ indicators via `data.indicator('RSI', timeperiod=14)`
- **Factor Engineering**: `feature.combine()` for multi-factor combinations
- **ML Pipeline**: `MLEngine()` with LightGBM/XGBoost integration
- **Backtesting**: `backtest.sim()` with Taiwan-specific cost modeling
- **Universe Filtering**: `data.universe(market='TSE_OTC')` for stock selection
- **Information Coefficient**: Built-in `calculate_ic()` for factor validation
- **Local Caching**: `save_to_storage=True` for development efficiency

**MVP-Specific Usage Pattern:**
```python
# Data acquisition with caching
close = data.get('price:收盤價', save_to_storage=True)
volume = data.get('price:成交股數', save_to_storage=True)

# Technical factor generation
rsi = data.indicator('RSI', timeperiod=14)
sma_20 = data.indicator('SMA', timeperiod=20)

# Signal generation for MOM_W strategy
momentum_signal = (close > sma_20) & (rsi > 50)

# Backtesting with realistic costs
position = backtest.sim(momentum_signal,
                       resample='W',  # Weekly rebalancing
                       position_limit=20,
                       fee_ratio=0.0004275,  # 0.04275%
                       tax_ratio=0.003)      # 0.3%

# Performance analysis
report = backtest.report(position)
```

### ML4T Architecture Foundation
**Reference**: `/mnt/c/Users/jnpi/ML4T/new/example/machine-learning-for-trading.md`

**Core Architecture Components from ML4T:**
- **Data Pipeline**: HDF5 → DuckDB migration for simplified deployment
- **Factor Library**: 50+ technical, fundamental, and sentiment factors
- **ML Models**: LightGBM/XGBoost with time-series cross-validation
- **Backtesting Engine**: Zipline integration with realistic cost modeling
- **Risk Management**: Position sizing, sector limits, and drawdown controls

**Critical Fixes Identified for Production:**
1. **utils.py line 23**: Fix `lookahead=None` default causing runtime crashes
2. **trading_env.py line 178**: Correct reward calculation using wrong day's costs
3. **trading_env.py line 242**: Resolve Gym compatibility issues with pandas

**MVP Simplification Strategy:**
- **Reduce Complexity**: Remove 80% of intraday complexity for weekly/monthly focus
- **Storage Migration**: Replace HDF5 with DuckDB/Parquet for easier deployment
- **Factor Selection**: Focus on 5-10 key factors instead of 50+ for MVP validation
- **Cost Modeling**: Implement Taiwan-specific transaction costs (0.04275% + 0.3% tax)

**Recommended Factor Set for MVP (Weekly Momentum):**
```python
# Technical Momentum Factors (3-5 factors max)
price_momentum = close / close.shift(20)  # 20-day momentum
rsi_14 = data.indicator('RSI', timeperiod=14)
sma_ratio = close / data.indicator('SMA', timeperiod=20)
volatility_20d = close.pct_change().rolling(20).std()

# Simple factor combination for MVP
combined_score = (price_momentum.rank(pct=True) +
                 rsi_14.rank(pct=True) +
                 sma_ratio.rank(pct=True)) / 3
```

**Time-Series Cross-Validation Implementation:**
```python
# Proper temporal splitting to avoid look-ahead bias
from sklearn.model_selection import TimeSeriesSplit

def time_series_cv(features, labels, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tscv.split(features):
        X_train = features.iloc[train_idx]
        X_test = features.iloc[test_idx]
        y_train = labels.iloc[train_idx]
        y_test = labels.iloc[test_idx]

        # Purge overlapping periods
        gap_days = 5  # Avoid look-ahead bias
        X_test = X_test.iloc[gap_days:]
        y_test = y_test.iloc[gap_days:]

        yield X_train, X_test, y_train, y_test
```

### Integration Strategy
1. **Use FinLab as primary data source** for Taiwan market coverage
2. **Implement ML4T architecture patterns** adapted for Taiwan market
3. **Focus on proven factors** from both references for MVP validation
4. **Leverage FinLab's built-in backtesting** with Taiwan-specific costs
5. **Apply ML4T risk management** principles with FinLab execution

## Implementation Phases (MVP-Focused)

### Phase 1: Foundation & Validation (Weeks 1-6)
**Goal**: Prove basic concept with backtesting

- **Week 1-2**: Setup FinLab data pipeline and historical data validation
- **Week 3-4**: Implement single MOM_W strategy with realistic cost modeling
- **Week 5-6**: Complete 10+ year backtest with walk-forward analysis

**Success Gate**: Backtest shows 12%+ CAGR, Sharpe >1.2 after realistic costs

### Phase 2: Paper Trading Validation (Weeks 7-18)
**Goal**: Validate backtest with live paper trading

- **Week 7-8**: Setup Fubon API integration and paper trading infrastructure
- **Week 9-18**: Live paper trading with daily monitoring and performance tracking
- **Week 19**: Performance analysis and go/no-go decision

**Success Gate**: Paper trading performance within 5% of backtest expectations

### Phase 3: Scale Decision (Weeks 19-20)
**Goal**: Make informed decision on full system development

- **Analysis**: Compare paper trading vs backtest results
- **Decision**: Go/Modify/No-Go based on success criteria
- **Planning**: If GO, plan full system development with realistic targets

### MVP Decision Gates

#### Gate 1 (Week 6): Backtest Validation
- **Pass**: Proceed to paper trading
- **Fail**: Redesign strategy or abandon concept

#### Gate 2 (Week 18): Paper Trading Validation
- **Pass**: Plan full system development
- **Modify**: Adjust parameters and extend validation
- **Fail**: Abandon or completely redesign approach

## Risk Assessment & Mitigation

### MVP-Specific Risks

#### **Concept Risk** (High Priority)
- **Risk**: TSE markets may not support systematic alpha generation
- **Mitigation**: Extensive backtesting across multiple market regimes
- **Detection**: Monitor Information Coefficient and factor significance

#### **Cost Model Risk** (High Priority)
- **Risk**: Transaction costs may eliminate all alpha
- **Mitigation**: Conservative cost estimates (0.37% roundtrip)
- **Detection**: Track paper trading vs backtest cost deviations

#### **Data Quality Risk** (Medium Priority)
- **Risk**: FinLab data quality issues affecting backtests
- **Mitigation**: Multiple data source validation and quality checks
- **Detection**: Compare with alternative Taiwan market data sources

#### **Overfitting Risk** (Medium Priority)
- **Risk**: Strategy works in backtest but fails in live trading
- **Mitigation**: Time-series CV, walk-forward analysis, conservative ML approach
- **Detection**: Monitor paper trading vs backtest performance divergence

### Success Protection Strategy
- **Conservative Estimates**: Use pessimistic cost and slippage assumptions
- **Extended Validation**: 6-month paper trading minimum
- **Multiple Metrics**: Don't rely on single performance measure
- **Regular Review**: Monthly performance assessment and adjustment

**MVP Philosophy**: Better to validate a simple concept thoroughly than to build a complex system on unproven foundations.