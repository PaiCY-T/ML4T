# Issue #03: StratifiedSampler Development (T0.2)

**Issue Type**: Infrastructure Foundation
**Phase**: 1 - Foundation & Proof
**Priority**: P0 - Critical Path
**Effort**: 40 hours
**Status**: 📋 Ready for Development

---

## 🎯 Objective

Develop statistical sampling framework to eliminate selection bias by replacing "top 200 stocks" approach with stratified sampling across market capitalization and sector dimensions, ensuring representative coverage of Taiwan stock universe.

## 📋 Requirements

### Core Functionality
- [x] **Stratified Sampling**: Sample across market cap and sector tiers
- [x] **Statistical Validity**: Ensure sample represents full universe characteristics
- [x] **Liquidity Filtering**: Include only stocks suitable for weekly/monthly trading
- [x] **Sample Validation**: Verify representativeness through statistical tests

### Technical Specifications
```python
class StratifiedSampler:
    """Eliminates selection bias through proper statistical sampling"""

    def __init__(self, universe_symbols: List[str], sample_size: int = 200):
        self.universe = universe_symbols  # All 1,307 Taiwan stocks
        self.sample_size = sample_size

    def get_stratified_sample(self) -> List[str]:
        """Sample across market cap and sector tiers for weekly/monthly trading"""
        # Large cap (top 100): 60 stocks - stable, liquid for monthly holds
        # Mid cap (101-500): 80 stocks - growth opportunities
        # Small cap (501-1307): 60 stocks - alpha generation potential

    def validate_sample_representation(self, sample: List[str]) -> Dict[str, float]:
        """Ensure sample represents full universe for weekly/monthly strategies"""
```

## 🔧 Implementation Plan

### Day 1: Market Cap Classification System (8h)
**Monday**

```python
# File: src/infrastructure/stratified_sampler.py

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.model_selection import train_test_split
import logging

class StratifiedSampler:
    """Statistical sampling framework for Taiwan stock universe"""

    def __init__(self, data_integration, sample_size: int = 200):
        self.data_integration = data_integration
        self.sample_size = sample_size
        self.universe_symbols = None
        self.market_data = None
        self.classification_cache = {}

    def _get_universe_data(self) -> pd.DataFrame:
        """Get recent market data for all symbols for classification"""
        if self.universe_symbols is None:
            self.universe_symbols = self.data_integration.get_symbol_universe()

        # Get last 3 months of data for market cap calculation
        end_date = pd.Timestamp.now().date()
        start_date = end_date - pd.DateOffset(months=3)

        # Get essential data for classification
        classification_fields = [
            'symbol', 'date', 'adj_close',
            '"市值" as market_cap',
            '"成交量" as volume',
            '"月營收" as monthly_revenue'
        ]

        data = self.data_integration.get_finlab_data(
            self.universe_symbols,
            (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')),
            classification_fields
        )

        return data

    def _classify_by_market_cap(self, data: pd.DataFrame) -> pd.DataFrame:
        """Classify stocks by market capitalization tiers"""

        # Calculate average market cap over recent period
        market_cap_avg = data.groupby('symbol')['market_cap'].mean().reset_index()
        market_cap_avg = market_cap_avg.sort_values('market_cap', ascending=False)

        # Define market cap tiers
        total_stocks = len(market_cap_avg)
        large_cap_cutoff = 100  # Top 100 stocks
        mid_cap_cutoff = 500    # Next 400 stocks
        # Remaining are small cap

        market_cap_avg['market_cap_tier'] = 'small_cap'
        market_cap_avg.loc[:large_cap_cutoff-1, 'market_cap_tier'] = 'large_cap'
        market_cap_avg.loc[large_cap_cutoff:mid_cap_cutoff-1, 'market_cap_tier'] = 'mid_cap'

        return market_cap_avg[['symbol', 'market_cap', 'market_cap_tier']]

    def _classify_by_sector(self, symbols: List[str]) -> pd.DataFrame:
        """Classify stocks by sector (simplified approach)"""

        # Taiwan stock symbol sector mapping (simplified)
        # In production, this would use TWSE sector data
        sector_mapping = {
            # Technology (2300-2499 range typically)
            'technology': [s for s in symbols if s.startswith(('23', '24', '25', '26'))],
            # Financial (2800-2899 range)
            'financial': [s for s in symbols if s.startswith('28')],
            # Traditional industries (1000-1999 range)
            'traditional': [s for s in symbols if s.startswith(('10', '11', '12', '13', '14', '15', '16', '17', '18', '19'))],
            # Others
            'others': []
        }

        # Assign remaining symbols to 'others'
        assigned = set()
        for sector_symbols in sector_mapping.values():
            assigned.update(sector_symbols)

        sector_mapping['others'] = [s for s in symbols if s not in assigned]

        # Create sector classification DataFrame
        sector_data = []
        for sector, sector_symbols in sector_mapping.items():
            for symbol in sector_symbols:
                sector_data.append({'symbol': symbol, 'sector': sector})

        return pd.DataFrame(sector_data)
```

### Day 2: Liquidity Filtering (8h)
**Tuesday**

```python
    def _apply_liquidity_filter(self, data: pd.DataFrame) -> List[str]:
        """Filter stocks for liquidity requirements in weekly/monthly trading"""

        # Calculate liquidity metrics over recent period
        liquidity_metrics = data.groupby('symbol').agg({
            'volume': ['mean', 'std'],
            'adj_close': ['mean', 'std'],
            'market_cap': 'mean'
        }).reset_index()

        # Flatten column names
        liquidity_metrics.columns = [
            'symbol', 'avg_volume', 'std_volume',
            'avg_price', 'std_price', 'avg_market_cap'
        ]

        # Calculate liquidity score
        liquidity_metrics['volume_consistency'] = (
            liquidity_metrics['avg_volume'] / (liquidity_metrics['std_volume'] + 1)
        )

        # Define liquidity requirements for weekly/monthly trading
        min_avg_volume = 100000  # Minimum daily volume
        min_market_cap = 1000000000  # Minimum 1B TWD market cap
        min_price = 10  # Minimum 10 TWD per share

        liquid_stocks = liquidity_metrics[
            (liquidity_metrics['avg_volume'] >= min_avg_volume) &
            (liquidity_metrics['avg_market_cap'] >= min_market_cap) &
            (liquidity_metrics['avg_price'] >= min_price) &
            (liquidity_metrics['volume_consistency'] > 1.0)  # Reasonable volume consistency
        ]

        return liquid_stocks['symbol'].tolist()

    def _stratified_sampling(self, market_cap_data: pd.DataFrame,
                           sector_data: pd.DataFrame, liquid_symbols: List[str]) -> List[str]:
        """Perform stratified sampling across market cap and sector dimensions"""

        # Combine market cap and sector data
        combined_data = market_cap_data.merge(sector_data, on='symbol')

        # Filter for liquid symbols
        combined_data = combined_data[combined_data['symbol'].isin(liquid_symbols)]

        # Define sampling targets for weekly/monthly trading
        sampling_targets = {
            'large_cap': {
                'target_count': 60,  # Stable, liquid for monthly holds
                'sector_distribution': {
                    'technology': 0.4,  # 24 stocks
                    'financial': 0.3,   # 18 stocks
                    'traditional': 0.2, # 12 stocks
                    'others': 0.1       # 6 stocks
                }
            },
            'mid_cap': {
                'target_count': 80,  # Growth opportunities
                'sector_distribution': {
                    'technology': 0.4,  # 32 stocks
                    'financial': 0.2,   # 16 stocks
                    'traditional': 0.3, # 24 stocks
                    'others': 0.1       # 8 stocks
                }
            },
            'small_cap': {
                'target_count': 60,  # Alpha generation potential
                'sector_distribution': {
                    'technology': 0.3,  # 18 stocks
                    'financial': 0.1,   # 6 stocks
                    'traditional': 0.4, # 24 stocks
                    'others': 0.2       # 12 stocks
                }
            }
        }

        sample_symbols = []

        for cap_tier, targets in sampling_targets.items():
            cap_data = combined_data[combined_data['market_cap_tier'] == cap_tier]

            for sector, proportion in targets['sector_distribution'].items():
                sector_cap_data = cap_data[cap_data['sector'] == sector]
                target_count = int(targets['target_count'] * proportion)

                if len(sector_cap_data) >= target_count:
                    # Random sample from this strata
                    sampled = sector_cap_data.sample(n=target_count, random_state=42)
                    sample_symbols.extend(sampled['symbol'].tolist())
                else:
                    # Take all available if insufficient symbols
                    sample_symbols.extend(sector_cap_data['symbol'].tolist())

        return sample_symbols[:self.sample_size]  # Ensure exact sample size
```

### Day 3: Sample Validation Framework (8h)
**Wednesday**

```python
    def validate_sample_representation(self, sample_symbols: List[str]) -> Dict[str, float]:
        """Validate that sample represents full universe characteristics"""

        validation_results = {}

        # Get universe data for comparison
        universe_data = self._get_universe_data()
        sample_data = universe_data[universe_data['symbol'].isin(sample_symbols)]

        # 1. Market Cap Distribution Validation
        universe_market_cap = self._classify_by_market_cap(universe_data)
        sample_market_cap = universe_market_cap[universe_market_cap['symbol'].isin(sample_symbols)]

        universe_cap_dist = universe_market_cap['market_cap_tier'].value_counts(normalize=True)
        sample_cap_dist = sample_market_cap['market_cap_tier'].value_counts(normalize=True)

        # Calculate KS test for distribution similarity
        from scipy.stats import ks_2samp
        validation_results['market_cap_ks_statistic'] = ks_2samp(
            universe_market_cap['market_cap'], sample_market_cap['market_cap']
        ).statistic

        # 2. Sector Distribution Validation
        universe_sector = self._classify_by_sector(self.universe_symbols)
        sample_sector = universe_sector[universe_sector['symbol'].isin(sample_symbols)]

        universe_sector_dist = universe_sector['sector'].value_counts(normalize=True)
        sample_sector_dist = sample_sector['sector'].value_counts(normalize=True)

        validation_results['sector_representation'] = {}
        for sector in universe_sector_dist.index:
            universe_pct = universe_sector_dist.get(sector, 0)
            sample_pct = sample_sector_dist.get(sector, 0)
            validation_results['sector_representation'][sector] = {
                'universe_pct': universe_pct,
                'sample_pct': sample_pct,
                'difference': abs(universe_pct - sample_pct)
            }

        # 3. Liquidity Characteristics Validation
        universe_liquidity = universe_data.groupby('symbol')['volume'].mean()
        sample_liquidity = sample_data.groupby('symbol')['volume'].mean()

        validation_results['liquidity_ks_statistic'] = ks_2samp(
            universe_liquidity, sample_liquidity
        ).statistic

        # 4. Overall Representativeness Score
        market_cap_score = 1 - validation_results['market_cap_ks_statistic']
        liquidity_score = 1 - validation_results['liquidity_ks_statistic']
        sector_score = 1 - np.mean([
            metrics['difference'] for metrics in validation_results['sector_representation'].values()
        ])

        validation_results['overall_representativeness'] = (
            market_cap_score * 0.4 + sector_score * 0.4 + liquidity_score * 0.2
        )

        return validation_results

    def get_stratified_sample(self) -> Tuple[List[str], Dict[str, float]]:
        """Generate stratified sample with validation metrics"""

        # Get universe data
        universe_data = self._get_universe_data()

        # Classify by market cap and sector
        market_cap_data = self._classify_by_market_cap(universe_data)
        sector_data = self._classify_by_sector(self.universe_symbols)

        # Apply liquidity filter
        liquid_symbols = self._apply_liquidity_filter(universe_data)

        # Perform stratified sampling
        sample_symbols = self._stratified_sampling(market_cap_data, sector_data, liquid_symbols)

        # Validate sample representativeness
        validation_metrics = self.validate_sample_representation(sample_symbols)

        # Log sampling results
        logging.info(f"Generated stratified sample: {len(sample_symbols)} symbols")
        logging.info(f"Overall representativeness: {validation_metrics['overall_representativeness']:.3f}")

        return sample_symbols, validation_metrics
```

### Day 4: Testing & Optimization (8h)
**Thursday**

```python
# File: tests/test_stratified_sampler.py

import unittest
import pandas as pd
from src.infrastructure.stratified_sampler import StratifiedSampler

class TestStratifiedSampler(unittest.TestCase):

    def setUp(self):
        """Set up test environment"""
        # Mock data integration for testing
        self.mock_data_integration = MockDataIntegration()
        self.sampler = StratifiedSampler(self.mock_data_integration, sample_size=200)

    def test_market_cap_classification(self):
        """Test market cap tier classification"""
        # Test with mock data
        mock_data = self._create_mock_market_data()
        classification = self.sampler._classify_by_market_cap(mock_data)

        # Verify classification logic
        self.assertEqual(len(classification), len(mock_data['symbol'].unique()))
        self.assertIn('large_cap', classification['market_cap_tier'].values)
        self.assertIn('mid_cap', classification['market_cap_tier'].values)
        self.assertIn('small_cap', classification['market_cap_tier'].values)

    def test_liquidity_filtering(self):
        """Test liquidity filtering logic"""
        mock_data = self._create_mock_market_data()
        liquid_symbols = self.sampler._apply_liquidity_filter(mock_data)

        # Verify filtering removes low-liquidity stocks
        self.assertLess(len(liquid_symbols), len(mock_data['symbol'].unique()))

    def test_sample_representativeness(self):
        """Test sample representativeness validation"""
        sample_symbols, validation_metrics = self.sampler.get_stratified_sample()

        # Verify sample size
        self.assertEqual(len(sample_symbols), 200)

        # Verify representativeness score
        self.assertGreaterEqual(validation_metrics['overall_representativeness'], 0.7)

    def test_sector_distribution(self):
        """Test sector distribution in sample"""
        sample_symbols, _ = self.sampler.get_stratified_sample()

        # Verify all major sectors represented
        sector_data = self.sampler._classify_by_sector(sample_symbols)
        sectors_in_sample = set(sector_data['sector'].unique())

        expected_sectors = {'technology', 'financial', 'traditional', 'others'}
        self.assertTrue(expected_sectors.issubset(sectors_in_sample))
```

### Day 5: Integration & Documentation (8h)
**Friday**

- Integration with ConfigManager and data pipeline
- Performance optimization and caching
- Complete documentation and usage examples
- Final testing and validation

## ✅ Acceptance Criteria

### Functional Requirements
- [ ] **Stratified Sampling**: Sample across market cap (large/mid/small) and sector dimensions
- [ ] **Statistical Validity**: Representativeness score >0.7 (KS test + distribution analysis)
- [ ] **Liquidity Filtering**: Include only stocks suitable for weekly/monthly trading
- [ ] **Sample Validation**: Comprehensive validation framework with multiple metrics
- [ ] **Reproducible Results**: Consistent sampling with random seed control

### Performance Requirements
- [ ] **Sampling Speed**: <60 seconds to generate 200-symbol sample from 1,307 universe
- [ ] **Memory Efficiency**: <1GB memory usage during sampling process
- [ ] **Data Coverage**: Sample covers >80% of market cap tiers and sectors
- [ ] **Representativeness**: Overall representativeness score >0.7

### Quality Requirements
- [ ] **Unit Test Coverage**: >90% code coverage
- [ ] **Validation Framework**: Multiple statistical tests for sample quality
- [ ] **Error Handling**: Robust error handling for missing/invalid data
- [ ] **Documentation**: Complete API documentation and usage examples

## 📊 Success Metrics

- **Statistical Validity**: Representativeness score >0.7
- **Sample Quality**: Market cap and sector distributions within 10% of universe
- **Liquidity**: All sampled stocks meet weekly/monthly trading requirements
- **Performance**: <60s sampling time, <1GB memory usage
- **Integration**: Works seamlessly with ConfigManager and data pipeline

## 🔗 Dependencies

- **Upstream**: Issue #01 ConfigManager, Issue #02 Data Pipeline Integration
- **Downstream**: Issue #04 StreamingProcessor, Issue #05 FactorCache
- **External**: pandas, numpy, scikit-learn, scipy

## 📝 Notes

- Focus on Taiwan market characteristics and trading requirements
- Ensure sample supports both weekly and monthly trading strategies
- Include comprehensive validation to demonstrate statistical soundness
- Design for easy adjustment of sampling parameters and criteria

---

**Issue Status**: 📋 Ready for Development
**Next Issue**: #04 StreamingProcessor Implementation
**Critical Path**: Yes - enables representative factor computation and backtesting