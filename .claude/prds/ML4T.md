---
name: ML4T
description: Personal quantitative trading system for Taiwan stocks with weekly/monthly cycles targeting realistic 12-15% annual returns
status: in_epic
created: 2025-09-20T20:54:13Z
updated: 2025-09-21T03:19:06Z
---

# PRD: ML4T

## Executive Summary

ML4T is a personal quantitative trading system designed for Taiwan stock markets (TSE/OTC) with weekly and monthly trading cycles. The system aims to achieve realistic risk-adjusted returns through multi-strategy quantitative approaches, targeting 12-15% annual returns with <20% maximum drawdown and >1.2 Sharpe ratio. Built on Python 3.10+ with FinLab data integration and Fubon API execution, the system supports flexible strategy development and risk management for personal capital deployment with proper market impact considerations.

## Problem Statement

### What problem are we solving?
- **Manual Trading Limitations**: Traditional discretionary trading lacks systematic approach, consistency, and scalability
- **Emotional Decision Making**: Human bias and emotional responses lead to suboptimal trading decisions
- **Limited Strategy Coverage**: Single-strategy approaches miss diversification benefits across market regimes
- **Inefficient Risk Management**: Manual portfolio management cannot effectively balance risk across positions and strategies

### Why is this important now?
- Taiwan stock market offers significant alpha opportunities for systematic approaches
- Advanced ML techniques and alternative data provide competitive advantages
- FinLab platform provides comprehensive Taiwan market data infrastructure
- Personal capital scale (30M NTD) requires professional-grade systematic approach

## User Stories

### Primary User Persona: Personal Quantitative Trader
**Background**: Experienced trader with 30M NTD capital seeking systematic, data-driven investment approach

#### User Journey 1: Strategy Development & Backtesting
- **As a trader**, I want to develop and backtest new quantitative strategies
- **So that** I can validate alpha generation before risking capital
- **Acceptance Criteria**:
  - Can create new strategy configurations via YAML files
  - Can backtest strategies with >10 years historical data
  - Can analyze performance metrics including Deflated Sharpe ratio
  - Can compare strategy performance across different market regimes

#### User Journey 2: Portfolio Management & Risk Control
- **As a trader**, I want automated portfolio construction and risk management
- **So that** I can maintain target risk levels without constant monitoring
- **Acceptance Criteria**:
  - Automated position sizing based on risk parity allocation
  - Real-time risk monitoring with configurable alerts
  - Single-name exposure limits (≤10% per position)
  - Portfolio-level drawdown protection (≤15%)

#### User Journey 3: Live Trading Execution
- **As a trader**, I want automated trade execution via Fubon API
- **So that** I can implement systematic strategies without manual intervention
- **Acceptance Criteria**:
  - Seamless integration with Fubon trading API
  - Pre-trade risk checks and validation
  - Transaction cost optimization
  - Execution logging and reconciliation

## Requirements

### Functional Requirements

#### Core Trading Engine
- **Multi-Strategy Framework**: Support for momentum, value, and ML-based strategies
- **Flexible Holding Periods**: Adaptive position duration based on signal strength and market conditions
- **Strategy Extensibility**: Plugin architecture for adding new alpha strategies
- **Fractional Kelly Allocation**: Dynamic capital allocation using 0.25-0.3 Kelly criterion with risk parity principles

#### Data Management (DataHub)
- **FinLab Integration**: Primary data source for Taiwan market data
- **EOD Updates**: Automated daily data refresh at 23:00
- **Historical Coverage**: Minimum 10 years of clean historical data (2012-present)
- **Factor Engineering**: Technical, fundamental, and sentiment factor generation

#### Strategy Engine
- **MOM_W (Momentum Weekly)**: Short-term momentum strategy with technical indicators
- **VAL_M (Value Monthly)**: Value-based strategy using fundamental metrics
- **ML_RANK (ML Monthly)**: Machine learning composite scoring system
- **Strategy Orchestration**: Coordinated signal generation and position management

#### Execution System
- **Fubon API Integration**: Replace SinopacAccount with Fubon trading API
- **Pre-trade Risk Checks**: Automated validation before order submission
- **Cost Optimization**: Minimize transaction costs and market impact
- **Order Management**: Support for market, limit, and stop orders

#### Risk Management
- **Position Limits**: Maximum 10% allocation per single name
- **Portfolio Protection**: Maximum 20% drawdown trigger for risk reduction
- **Stop Loss**: ATR-based or volatility-scaled stops (not fixed 10%)
- **VaR Monitoring**: 250-day rolling parametric VaR estimation
- **Liquidity Filter**: ADV >50M NTD to ensure 30M NTD scalability

### Non-Functional Requirements

#### Performance
- **Latency**: Strategy signal generation within 5 minutes of market close
- **Throughput**: Support for up to 50 concurrent positions
- **Reliability**: 99.5% system uptime during market hours
- **Data Processing**: Handle full Taiwan market universe (1800+ stocks)

#### Security
- **API Security**: Encrypted storage of Fubon API credentials
- **Data Protection**: Local storage only, no cloud data transmission
- **Access Control**: Single-user authentication for system access
- **Audit Trail**: Complete logging of all trading decisions and executions

#### Scalability
- **Capital Scaling**: Start with smaller amounts, scale to 30M NTD
- **Strategy Scaling**: Support for additional strategy modules
- **Performance Scaling**: Linear performance with position count
- **Data Scaling**: Efficient handling of expanding historical datasets

## Success Criteria

### Primary Metrics
- **Annual Return**: Target 12-15% risk-adjusted returns (stretch goal: >20%)
- **Sharpe Ratio**: Achieve 1.2-1.5 Sharpe ratio consistently (stretch goal: >2.0)
- **Maximum Drawdown**: Maintain <20% maximum drawdown
- **Win Rate**: Strategy-dependent but aim for >55% overall

### Secondary Metrics
- **Volatility**: Keep portfolio volatility 12-18% annually
- **Turnover**: Optimize for <200% annual turnover
- **Transaction Costs**: Keep total costs <0.5% annually
- **System Uptime**: Maintain >99% operational availability

### Validation Criteria
- **Backtesting**: 10+ years out-of-sample validation
- **Walk-Forward**: 3-year rolling window validation
- **Deflated Sharpe**: Statistical significance testing
- **Regime Testing**: Performance across bull/bear/sideways markets

## Constraints & Assumptions

### Technical Constraints
- **Local Deployment**: System runs on single local machine only
- **Data Dependency**: Reliant on FinLab data quality and availability
- **API Limitations**: Subject to Fubon API rate limits and connectivity
- **Python Ecosystem**: Limited to Python 3.10+ compatible libraries (3.11 compatibility issues with Zipline/TA-Lib)

### Capital Constraints
- **Initial Capital**: Start with smaller amounts before scaling to 30M NTD
- **Position Sizing**: Cannot exceed 10% in any single name
- **Liquidity Requirements**: Must trade only liquid Taiwan stocks
- **Currency Risk**: Limited to NTD-denominated assets

### Operational Constraints
- **Market Hours**: Limited to Taiwan market trading hours
- **Manual Oversight**: Requires periodic manual monitoring and intervention
- **Strategy Capacity**: Limited by Taiwan market liquidity
- **Development Resources**: Single developer/operator

### Key Assumptions
- FinLab data accuracy and completeness
- Fubon API reliability and performance
- Taiwan market structure stability
- Regulatory environment unchanged
- Local computing resources sufficient

## Out of Scope

### Explicitly NOT Building
- **Multi-User Support**: System designed for single personal use only
- **Cloud Deployment**: No cloud infrastructure or remote access
- **Real-time Intraday Trading**: Focus on EOD and longer-term signals
- **International Markets**: Taiwan stocks only, no foreign exchanges
- **Options/Derivatives**: Equity long-only strategies
- **Alternative Data**: No news, social media, or satellite data integration
- **Mobile Interface**: Desktop/command-line interface only
- **Regulatory Reporting**: No compliance or regulatory reporting features

### Future Considerations
- Alternative data integration (news, sentiment)
- Intraday strategy development
- Options overlay strategies
- International diversification

## Dependencies

### External Dependencies
- **FinLab Platform**: Core data provider for Taiwan market data
- **Fubon Securities API**: Trading execution and order management
- **Taiwan Stock Exchange**: Market data and trading venue
- **Python Ecosystem**: pandas, numpy, scikit-learn, lightgbm libraries

### Internal Dependencies
- **System Infrastructure**: Local development machine capabilities
- **Data Storage**: Local PostgreSQL/DuckDB setup
- **Network Connectivity**: Stable internet for data/API access
- **Time Synchronization**: Accurate system time for trading operations

### Critical Path Dependencies
1. **Fubon API Integration**: Must complete before live trading
2. **Historical Data Validation**: Required before strategy development
3. **Risk Management System**: Must be operational before capital deployment
4. **Backtesting Framework**: Needed for strategy validation

## Implementation Phases

*Architecture based on analysis of machine-learning-for-trading.md - 150+ notebook ML trading system*

### Phase 1: Foundation & Critical Fixes (Weeks 1-4)
**Priority: Address architectural debt and establish robust foundation**

- **Critical Bug Fixes**:
  - Fix `lookahead=None` default causing runtime crashes in utils.py
  - Correct reward calculation using current day costs in trading_env.py
  - Implement proper look-ahead bias prevention

- **Core Infrastructure**:
  - Replace HDF5 with DuckDB/Parquet for simplified deployment
  - Implement `MultipleTimeSeriesCV` class for temporal splitting
  - FinLab integration with automated EOD data pipeline
  - Local PostgreSQL/DuckDB hybrid storage architecture

- **Development Environment**:
  - Python 3.11 + core ML stack (pandas, numpy, scikit-learn)
  - Finance-specific libraries (talib, zipline-reloaded, pyfolio-reloaded)
  - Jupyter environment for research and validation

### Phase 2: Factor Engineering & ML Pipeline (Weeks 5-8)
**Priority: Simplified factor library with statistical rigor**

- **Simplified Factor Library (≤10 factors)**:
  - **Fundamental**: Book-to-Market, ROA/ROE, Gross Margins
  - **Technical**: 20-day volatility breakouts, 13/26-week MA crosses, 52-week high distance
  - **Momentum**: 12-2 momentum (skip recent month to avoid reversals)

- **ML Pipeline Implementation**:
  - Time-series cross-validation with proper purging
  - Spearman correlation-based alpha selection
  - LightGBM model with feature importance tracking
  - IC (Information Coefficient) filtering with 0.02 threshold

- **Statistical Validation**:
  - Deflated Sharpe ratio implementation
  - Walk-forward analysis with 3-year windows
  - Regime testing across bull/bear/sideways markets

### Phase 3: Risk Management & Backtesting (Weeks 9-12)
**Priority: Production-ready risk controls and cost modeling**

- **Risk Management Framework**:
  - Fractional Kelly position sizing (0.25-0.3 Kelly, max 10% per asset)
  - Sector exposure limits (25% NAV maximum with GICS classification)
  - ATR-based stop-loss (volatility-scaled, not fixed percentage)
  - 250-day rolling parametric VaR with 95% confidence
  - Liquidity filter: ADV >50M NTD for market impact control

- **Realistic Cost Modeling**:
  - `trading_cost_bps` and `time_cost_bps` parameters
  - Taiwan market transaction costs (0.1425% fee + 0.3% tax)
  - Slippage modeling based on order size vs daily volume
  - Market impact estimation for large orders

- **Backtesting Engine**:
  - Zipline integration for realistic execution simulation
  - Pyfolio-reloaded for comprehensive performance analytics
  - Multi-timeframe strategy coordination (weekly/monthly)
  - Out-of-sample validation with 10+ years data

### Phase 4: Production Deployment & Scaling (Weeks 13-16)
**Priority: Live trading with monitoring and gradual scaling**

- **Broker Integration**:
  - Fubon API integration with error handling and retry logic
  - Pre-trade risk checks and position validation
  - Order management system (market/limit/stop orders)
  - Real-time portfolio reconciliation

- **Production Infrastructure**:
  - Containerized deployment with Docker
  - Automated nightly data pipeline and factor generation
  - Comprehensive logging and audit trail
  - Performance monitoring dashboard

- **Gradual Capital Deployment**:
  - Start with 1M NTD for system validation
  - Monthly performance review and scaling decisions
  - Risk parameter optimization based on live results
  - Scale to full 30M NTD over 6-month period

- **Monitoring & Optimization**:
  - Real-time P&L tracking and risk metrics
  - Strategy performance attribution analysis
  - Alert system for drawdown and risk breaches
  - Monthly strategy review and parameter tuning

### Implementation Success Criteria
- **Phase 1**: Clean data pipeline, proper CV framework, bug-free foundation, liquidity filters active
- **Phase 2**: IC > 0.02 factors, VIF < 5.0 for multicollinearity, statistically significant alpha
- **Phase 3**: Sharpe > 1.2 in backtesting, max drawdown < 20%, realistic transaction costs
- **Phase 4**: Live Sharpe > 1.0, system uptime > 99%, successful scaling with market impact analysis

## Risk Mitigation

### Technical Risks
- **Data Quality**: Implement data validation and cleaning pipelines
- **API Downtime**: Build retry logic and manual override capabilities
- **System Failures**: Implement monitoring, alerting, and backup procedures

### Market Risks
- **Strategy Failure**: Diversification across multiple uncorrelated strategies
- **Market Regime Change**: Regular strategy performance review and adaptation
- **Liquidity Risk**: Position sizing based on average daily volume

### Operational Risks
- **Manual Errors**: Minimize manual intervention through automation
- **Oversight Gaps**: Implement comprehensive logging and monitoring
- **Knowledge Risk**: Document all systems and maintain code quality