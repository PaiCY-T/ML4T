---
name: ml4t-multi-factor-strategy
description: Transform single-factor momentum strategy into diversified multi-factor system to achieve 70%+ win rate across all market regimes
status: planning
created: 2025-09-30T15:20:22Z
updated: 2025-09-30T15:50:12Z
prd_source: ml4t-multi-factor-strategy
github_issue: 73
progress: 0
total_issues: 10
completed_issues: 0
---

# Epic: ml4t-multi-factor-strategy

## Overview

### Current Crisis
The existing ML4T momentum-only trading system suffers from catastrophic regime dependency, achieving only **40% win rate** with -24.1% underperformance during mean-reverting markets (Trade War Era 2018-2019). The system fundamentally fails in 60% of market regimes, making it unreliable for investment capital deployment.

### Solution Approach
Based on unanimous expert consensus from OpenAI O3 and Gemini 2.5 Pro, implement **factor diversification over technical indicators**. Transform the system from single-factor momentum dependency to a regime-aware multi-factor architecture integrating:

1. **Momentum Group** (Existing): Price momentum factors for trending markets
2. **Value Group** (New): P/E, P/B, dividend yield for mean-reverting defense
3. **Flow Group** (New): Taiwan-specific institutional and broker flows for alpha generation
4. **Regime Detection** (New): Dynamic factor weighting based on market conditions

### Expected Outcome
- **Win Rate**: 40% → 70%+ across all market regimes
- **Max Drawdown**: <10% relative to Taiwan 50 ETF benchmark
- **Information Coefficient**: 0.08-0.12 (vs current ~0.05)
- **Regime Robustness**: Positive alpha in both trending and mean-reverting markets


## Problem Context

### Performance Analysis (15-year validation)
| Period | Market Type | Performance vs Taiwan 50 ETF | Status |
|--------|-------------|-------------------------------|---------|
| European Crisis (2012-2013) | Mean-Reverting | **-8.8%** | ❌ Failed |
| Steady Growth (2014-2017) | Trending | **-0.9%** | ⚠️ Marginal |
| Trade War Era (2018-2019) | Mean-Reverting | **-24.1%** | ❌ Catastrophic |
| COVID Period (2020-2021) | Mixed/Recovery | **+2.7%** | ✅ Worked |
| Current Era (2022-2024) | Bull Trending | **+7.8%** | ✅ Worked |

### Root Cause Analysis
- **Single-Factor Dependency**: Only momentum factors utilized from 192 available columns
- **Regime Blindness**: Static factor weights regardless of market conditions
- **Missing Diversification**: No alternative factors for mean-reverting periods
- **Alpha Concentration**: No Taiwan-specific institutional flow insights

### Business Impact
- **Investment Risk**: Strategy unreliable for actual capital deployment
- **Opportunity Cost**: Missing Taiwan-specific alpha sources
- **Reputation Risk**: Poor performance during critical market stress periods


## Core Requirements
**Multi-Factor Architecture**
- Regime-aware factor combination system
- Dynamic weight allocation based on market conditions
- Support for 3+ factor groups (momentum, value, flow)
- Cross-sectional ranking and z-score normalization

**Factor Implementation**
- **Momentum Group**: Existing 6m/3m momentum + volatility
- **Value Group**: P/E ratio, P/B ratio, dividend yield transformations
- **Flow Group**: Foreign institutional flows, top-15 broker sentiment
- **Quality Group** (Future): ROE stability, debt ratios, earnings quality

**Regime Detection System**
- Taiwan market cycle identification (trending/mean-reverting/choppy)
- TAIEX vs MA200 analysis with confidence scoring
- Dynamic factor weight allocation per regime
- Historical regime backtesting validation

## Technical Approach

### Zen Tools Integration for Quality Assurance
**Implementation Quality Framework**: Leverage advanced AI tools for systematic quality improvement and validation throughout development.

**Core Zen Tools Utilization**:
- **mcp__zen__thinkdeep**: Multi-stage investigation and reasoning for complex problem analysis
  - Architecture decisions and design patterns
  - Performance challenges and optimization strategies
  - Security analysis and vulnerability assessment
  - Complex debugging and root cause analysis

- **mcp__zen__chat**: General chat and collaborative thinking for development discussion
  - Brainstorming factor combinations and regime strategies
  - Getting second opinions on implementation approaches
  - Exploring alternative solutions and edge cases
  - Real-time problem-solving and guidance

- **mcp__zen__challenge**: Critical thinking validation to prevent reflexive agreement
  - Challenge implementation decisions and assumptions
  - Validate factor selection rationale and methodology
  - Question regime detection logic and edge cases
  - Ensure robust testing and validation approaches

**Quality Enhancement Workflow**:
1. **Design Phase**: Use `thinkdeep` for architectural analysis and `chat` for collaborative exploration
2. **Implementation Phase**: Apply `challenge` to question assumptions and validate approaches
3. **Testing Phase**: Leverage `thinkdeep` for comprehensive testing strategies
4. **Review Phase**: Use `challenge` with Gemini 2.5 Flash for independent validation

**Expected Quality Outcomes**:
- Reduced implementation risks through systematic analysis
- Higher code quality through collaborative development
- Validated design decisions through critical questioning
- Comprehensive testing coverage through deep analysis


## Success Metrics

### Primary Objectives
1. **Regime Robustness**: Positive alpha in both trending and mean-reverting markets
2. **Win Rate**: 70%+ across 6-month rolling periods (vs current 40%)
3. **Drawdown Control**: Maximum 10% relative underperformance vs Taiwan 50 ETF
4. **Information Ratio**: >0.5 after transaction costs

### Secondary Objectives
1. **Factor Diversification**: Low correlation between factor groups (<0.3)
2. **Taiwan Alpha**: Outperform momentum strategies without local market insights
3. **Scalability**: Handle full 1,307 symbol universe efficiently (<5min rebalancing)
4. **Interpretability**: Clear attribution to factor group performance

### Comprehensive Multi-Period Backtesting Analysis
Based on the critical findings from the ML4T personal trading system execution, the multi-factor strategy must pass rigorous historical validation:

**Historical Validation Framework (15-Year Analysis)**
- **Analysis Scope**: 2010-2025 (15 years) covering all Taiwan market regimes
- **Market Regimes Tested**: 6 distinct periods across bull/bear/sideways markets
- **Benchmark Comparison**: Taiwan 50 ETF (0050) performance across all periods
- **Point-in-Time Validation**: Eliminate lookahead bias with month-end snapshots
- **Proper Forward Returns**: Methodology validation with temporal integrity

**Critical Success Criteria (Learning from Current System Failures)**
- **Win Rate Against Benchmark**: Must achieve 70%+ (vs current 40% failure rate)
- **Regime Independence**: Positive alpha in BOTH trending AND mean-reverting markets
- **Maximum Drawdown**: <10% relative underperformance (vs current -24.1% catastrophic failure)
- **Consistency Check**: No single period with >15% underperformance vs benchmark

**Mandatory Period-by-Period Validation**
Strategy must outperform Taiwan 50 ETF in at least 4 out of 5 historical periods:
1. **European Crisis (2012-2013)**: Mean-Reverting - MUST NOT FAIL (current: -8.8%)
2. **Steady Growth (2014-2017)**: Trending - MUST IMPROVE (current: -0.9%)
3. **Trade War Era (2018-2019)**: Mean-Reverting - CRITICAL TEST (current: -24.1% catastrophic)
4. **COVID Period (2020-2021)**: Mixed/Recovery - MAINTAIN POSITIVE (current: +2.7%)
5. **Current Era (2022-2024)**: Bull Trending - MAINTAIN POSITIVE (current: +7.8%)

**Validation Methodology Requirements**
- **No Cherry-Picking**: Full 15-year period analysis (not limited period optimization)
- **Regime Detection Accuracy**: >70% market regime classification accuracy
- **Factor Persistence**: Individual factor performance stability across regimes
- **Risk-Adjusted Metrics**: Sharpe ratio >1.0, Information ratio >0.5
- **Transaction Cost Reality**: Include 0.3% costs, monthly rebalancing constraints

**Failure Prevention Criteria**
- **Anti-Momentum Defense**: Value and flow factors must provide positive alpha when momentum fails
- **Regime Switching Evidence**: Dynamic factor weights must demonstrate regime awareness
- **Taiwan-Specific Edge**: Institutional flow factors must provide consistent alpha vs momentum-only
- **Overfitting Prevention**: Out-of-sample validation on recent data (2023-2024)

### Technical Validation
- All factor groups implemented with proper transformations
- Regime detection system operational with >70% accuracy
- Database integration complete with optimized queries
- Comprehensive backtesting across ALL historical periods completed with evidence
- Factor correlation analysis confirming diversification benefits
- Regime-specific performance attribution validated

### Evidence-Backed Task Completion Framework
**Mandatory Evidence Standards**: Every task completion must be supported by concrete, verifiable evidence to prevent implementation claims without substance.

**Task Completion Evidence Requirements**:
1. **Code Implementation Evidence**
   - Functional source code files with proper imports and dependencies
   - Unit tests demonstrating functionality with passing results
   - Integration tests validating component interactions
   - Performance benchmarks with actual timing and memory usage

2. **Data Validation Evidence**
   - Sample data outputs showing actual calculations
   - Statistical validation with real numbers and distributions
   - Edge case testing with documented results
   - Data quality metrics with specific percentages and counts

3. **Performance Evidence**
   - Backtesting results with actual returns, Sharpe ratios, and drawdowns
   - Factor performance attribution with correlation matrices
   - Regime detection accuracy with confusion matrices
   - Processing time benchmarks with specific millisecond measurements

4. **Integration Evidence**
   - End-to-end workflow demonstrations with screenshots/logs
   - Database connectivity tests with query performance metrics
   - Error handling validation with specific error scenarios
   - Memory usage profiling with actual MB consumption figures

**Zen Challenge Validation Protocol**:
**Independent Review Requirement**: Each major task completion must undergo critical validation using `mcp__zen__challenge` with Gemini 2.5 Flash model.

**Challenge Review Process**:
1. **Evidence Submission**: Present all implementation evidence in structured format
2. **Critical Analysis**: Use `challenge` tool to question assumptions, methodology, and completeness
3. **Independent Validation**: Gemini 2.5 Flash model reviews evidence without bias
4. **Challenge Response**: Address all critical points raised during challenge phase
5. **Final Approval**: Task marked complete only after passing challenge validation

**Challenge Focus Areas**:
- **Implementation Quality**: Code structure, error handling, edge cases
- **Methodology Soundness**: Statistical approaches, backtesting validity, bias prevention
- **Evidence Completeness**: All claims supported by verifiable data
- **Performance Claims**: Realistic expectations vs actual measured results
- **Risk Assessment**: Potential failure modes and mitigation strategies

**Rejection Criteria**:
Tasks will be rejected and require rework if:
- Evidence is insufficient or unverifiable
- Implementation claims cannot be reproduced
- Critical edge cases are not addressed
- Performance metrics are unrealistic or cherry-picked
- Challenge review identifies fundamental flaws

**Quality Assurance Benefits**:
- Prevents "works on my machine" syndrome
- Ensures reproducible implementations
- Validates performance claims with real data
- Identifies blind spots through independent review
- Maintains high delivery standards throughout project


## Implementation Roadmap

### Phase 1: Core Factor Groups (4-6 weeks)
**Week 1-2: Factor Implementation**
- Implement Value Group factors (P/E, P/B, dividend yield transformations)
- Implement Flow Group factors (institutional flows, broker sentiment)
- Create factor scoring and ranking systems
- Integrate with existing momentum factors

**Week 3-4: Factor Combination**
- Develop equal-weight factor combination strategy
- Implement cross-sectional ranking and normalization
- Create composite scoring algorithm
- Basic performance validation framework

### Phase 2: Regime-Aware Strategy (2-3 weeks)
**Week 5-6: Regime Detection**
- Implement Taiwan market regime identification
- Create dynamic factor weight allocation system
- Develop regime-specific strategy logic
- Historical regime backtesting framework

**Week 7: Integration Testing**
- End-to-end system integration
- Performance validation across historical periods
- Risk control implementation
- Production readiness assessment

### Phase 3: Optimization & Deployment (2-3 weeks)
**Week 8-9: Performance Optimization**
- Factor selection and hyperparameter tuning
- Transaction cost optimization
- Memory and processing efficiency improvements
- Comprehensive testing suite

**Week 10: Production Deployment**
- Live trading system integration
- Monitoring and alerting setup
- Documentation and operational procedures
- Performance tracking dashboard


## Tasks Created
- [ ] 001.md - Value Factor Group Implementation (parallel: true)
- [ ] 002.md - Flow Factor Group Implementation (parallel: true)
- [ ] 003.md - Factor Integration Framework (depends_on: [001, 002])
- [ ] 004.md - Factor Combination Strategy (depends_on: [003])
- [ ] 005.md - Regime Detection System (depends_on: [004])
- [ ] 006.md - Dynamic Factor Weight Allocation (depends_on: [005])
- [ ] 007.md - Historical Regime Backtesting (depends_on: [006])
- [ ] 008.md - Performance Optimization & Tuning (depends_on: [007])
- [ ] 009.md - Comprehensive Testing Suite (depends_on: [008])
- [ ] 010.md - Production Deployment & Monitoring (depends_on: [009])

Total tasks: 10
Parallel tasks: 2 (001, 002)
Sequential tasks: 8
Estimated total effort: 106-144 hours (approximately 13-18 working days)

## Dependencies
- PRD: ml4t-multi-factor-strategy.md
- Implementation phases as defined in PRD
- Technical stack: Python 3.11, FinLab, Fubon API

## Acceptance Criteria
- [ ] All three factor groups implemented and validated
- [ ] Regime detection system operational with >70% accuracy
- [ ] Historical backtesting shows 70%+ win rate across market periods
- [ ] Maximum drawdown <10% relative to benchmark achieved
- [ ] Production system deployed with monitoring and controls
- [ ] Documentation complete with operational procedures

## Notes
Epic auto-generated from PRD: ml4t-multi-factor-strategy.md on 2025-09-30T15:20:22Z
Updated with comprehensive multi-period backtesting analysis from ml4t-personal-trading-system execution findings.