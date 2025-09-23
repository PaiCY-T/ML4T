---
name: ML4T-Alpha-Rebuild
status: active
created: 2025-09-23T08:00:00Z
updated: 2025-09-24T11:30:00Z
progress: 33%
prd: .claude/prds/ML4T-Alpha-Rebuild.md
parent_epic: ML4T
priority: critical
github: https://github.com/PaiCY-T/ML4T/issues/20
---

# Epic: ML4T-Alpha Complete Rebuild (True ML Implementation)

## Overview

**Critical Priority Epic**: Complete rebuild of mislabeled "ML4T-Alpha" strategy that has ZERO machine learning into true ML-driven system targeting 20%+ annual returns.

**Core Problem**: Strategy named "ML4T-Alpha" contains no ML - uses hardcoded weights causing -2.7% underperformance. Needs complete ML pipeline rebuild with OpenFE integration.

**Strategic Approach**: Three-tier ML architecture with automatic feature engineering, proper validation, and Taiwan market optimization.

## Success Criteria

### Primary Objectives (Must Achieve)
- **Annual Return**: >20% (vs current -2.7% underperformance)
- **Max Drawdown**: <15%
- **Sharpe Ratio**: >2.0
- **Information Ratio**: >0.8
- **ML Implementation**: True machine learning with OpenFE

### Secondary Objectives (Nice to Have)
- **Weekly Rebalancing**: 4x frequency improvement
- **Feature Discovery**: 42+ handcrafted + OpenFE auto-generated
- **Risk Management**: Comprehensive validation framework
- **Production Ready**: Real-time monitoring and retraining

## Technical Architecture

### Three-Tier ML System

1. **ML Scoring Engine** (Core Innovation)
   - LightGBM for factor scoring (replacing hardcoded weights)
   - OpenFE automatic feature discovery
   - Real-time prediction pipeline
   - IC decay monitoring and feature retirement

2. **Portfolio Construction** (Enhanced)
   - Weekly rebalancing (vs current monthly)
   - Kelly position sizing implementation
   - Risk budgeting and optimization
   - Transaction cost modeling

3. **Validation & Risk Management** (New)
   - Walk-forward validation (156-week train / 26-week test)
   - Purged K-fold cross-validation
   - Regime detection and adaptation
   - Performance attribution system

## Implementation Timeline: 10-12 Weeks

### Phase 1: Foundation (Weeks 1-4)
**Goal**: Robust data pipeline and backtesting framework

#### Week 1-2: Data Pipeline Foundation
- Point-in-time data management system
- Taiwan market lag handling (T+2 settlement, 60-day financial lag)
- Integration with existing incremental updater
- Data quality monitoring and validation

#### Week 3-4: Backtesting Framework
- Walk-forward validation engine (156-week train / 26-week test)
- Purged K-fold cross-validation implementation
- Transaction cost modeling for Taiwan market
- Performance attribution and risk-adjusted metrics

### Phase 2: ML Foundation (Weeks 5-7)
**Goal**: Baseline ML model with proper validation

#### Week 5: Factor Engineering
- 42 handcrafted factors implementation
- Technical indicators (momentum, mean reversion, volatility)
- Fundamental factors (value, quality, growth)
- Factor performance tracking and IC calculation

#### Week 6-7: ML Model Pipeline
- LightGBM implementation with hyperparameter tuning
- Time-series cross-validation framework
- Feature importance tracking and analysis
- Model performance monitoring and validation

### Phase 3: OpenFE Integration (Weeks 8-10)
**Goal**: Automatic feature discovery and enhancement

#### Week 8: OpenFE Setup
- OpenFE integration with existing pipeline
- Feature generation on 42 base factors
- Taiwan market adaptations and optimizations

#### Week 9-10: Feature Selection & Validation
- Hierarchical clustering for correlation filtering (0.7 threshold)
- IC decay tracking for feature retirement (ICIR < 0.05)
- Feature stability analysis and performance comparison
- Production readiness testing

### Phase 4: Production & Optimization (Weeks 11-12)
**Goal**: Live-ready system with monitoring

#### Week 11: Production System
- Real-time prediction pipeline
- Portfolio construction optimization
- Kelly position sizing implementation
- Risk management integration

#### Week 12: Monitoring & Validation
- Performance monitoring dashboard
- Model drift detection system
- Automated retraining triggers
- Final validation and deployment readiness

## Key Improvements Over Current System

| Component | Current (Broken) | New (ML-Driven) | Impact |
|-----------|------------------|-----------------|--------|
| Scoring | Hardcoded weights | LightGBM + OpenFE | +15% alpha |
| Rebalancing | Monthly | Weekly | +5% performance |
| Validation | None | Walk-forward | Risk reduction |
| Features | 5 basic | 42+ enhanced | Better prediction |
| Risk Management | Basic | Comprehensive | Downside protection |

## Risk Mitigation

### Technical Risks
- **Data Quality**: Comprehensive validation and monitoring
- **Look-ahead Bias**: Strict point-in-time data management
- **Overfitting**: Walk-forward validation and out-of-sample testing
- **Feature Instability**: IC decay tracking and automated retirement

### Market Risks
- **Regime Changes**: Multiple validation periods including crises
- **Liquidity**: Transaction cost modeling and capacity limits
- **Model Drift**: Continuous monitoring and retraining triggers
- **Taiwan Specifics**: Local market microstructure integration

## Dependencies & Prerequisites

### Data
- ✅ FinLab database with 278 fields confirmed
- ✅ Incremental updater already implemented
- ✅ UBVS framework available for validation

### Infrastructure
- Python 3.11+ environment
- LightGBM, OpenFE libraries
- Sufficient compute for walk-forward testing
- Data storage for historical backtests

## Success Criteria Validation

### Technical Validation
- [ ] No look-ahead bias in backtests
- [ ] Walk-forward validation shows consistent performance
- [ ] OpenFE features improve baseline by >5%
- [ ] IC decay tracking prevents feature degradation
- [ ] Real-time pipeline latency <1 second

### Performance Validation
- [ ] Backtest annual return >20%
- [ ] Maximum drawdown <15%
- [ ] Sharpe ratio >2.0
- [ ] Information ratio >0.8
- [ ] Outperformance in multiple market regimes

### Production Readiness
- [ ] Automated data pipeline operational
- [ ] Model retraining system functional
- [ ] Monitoring and alerting active
- [ ] Risk management integration complete
- [ ] Documentation and handover complete

## Expected Outcomes

### Immediate (Weeks 1-4)
- Robust data pipeline eliminating look-ahead bias
- Professional backtesting framework
- Clear performance attribution
- Solid foundation for ML development

### Medium-term (Weeks 5-8)
- Functional ML pipeline with 42 factors
- Baseline performance improvement
- Proper validation framework
- Model interpretability tools

### Long-term (Weeks 9-12)
- OpenFE-enhanced feature discovery
- Production-ready ML system
- Monitoring and maintenance framework
- 20%+ annual return target achievement

This epic transforms a broken rule-based system into a world-class ML-driven quantitative strategy, addressing all identified gaps while building on existing UBVS infrastructure.

## Tasks Created
- [ ] #21 - Point-in-Time Data Management System (parallel: false)
- [ ] #22 - Data Quality Validation Framework (parallel: false)
- [ ] #23 - Walk-Forward Validation Engine (parallel: true)
- [ ] #24 - Transaction Cost Modeling (parallel: true)
- [ ] #25 - 42 Handcrafted Factors Implementation (parallel: false)
- [ ] #26 - LightGBM Model Pipeline (parallel: false)
- [ ] #27 - Model Validation & Monitoring (parallel: true)
- [ ] #28 - OpenFE Setup & Integration (parallel: true)
- [ ] #29 - Feature Selection & Correlation Filtering (parallel: false)
- [ ] #30 - Production Readiness Testing (parallel: true)
- [ ] #31 - Real-Time Production System (parallel: false)
- [ ] #32 - Monitoring & Automated Retraining (parallel: false)

Total tasks: 12
Parallel tasks: 5 (can be worked on simultaneously)
Sequential tasks: 7 (have dependencies)
Estimated total effort: 35-42 hours