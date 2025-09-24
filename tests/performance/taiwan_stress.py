"""
Taiwan Market Stress Testing

Specialized stress testing for Taiwan Stock Exchange (TSE) and Taipei Exchange (TPEx)
market conditions including volatility periods, circuit breakers, and market events.
"""

import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date, timedelta
import psutil
import gc
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

try:
    from data.models.taiwan_market import TaiwanTradingCalendar, TaiwanMarketConfig
    TAIWAN_IMPORTS_AVAILABLE = True
except ImportError:
    TAIWAN_IMPORTS_AVAILABLE = False


class TaiwanMarketStressTester:
    """Stress testing for Taiwan market-specific scenarios."""
    
    def __init__(self, log_level: str = 'INFO'):
        """Initialize Taiwan market stress tester."""
        self.setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        
        # Taiwan market parameters
        self.tse_price_limit = 0.10  # 10% daily price limit
        self.circuit_breaker_threshold = 0.035  # 3.5% circuit breaker
        self.trading_hours = {'open': '09:00', 'close': '13:30'}
        self.lunch_break = {'start': '12:00', 'end': '13:00'}  # TPEx only
        
        # Historical Taiwan market stress periods
        self.stress_periods = {
            'march_2020_crash': {
                'start': date(2020, 3, 9),   # Global pandemic selloff
                'end': date(2020, 3, 23),
                'avg_volatility': 0.05,      # 5% daily volatility
                'max_daily_drop': -0.10,     # 10% limit down
                'circuit_breakers': 8        # Days with circuit breakers
            },
            'may_2022_selloff': {
                'start': date(2022, 5, 9),   # Fed rate hike fears
                'end': date(2022, 5, 20),
                'avg_volatility': 0.035,
                'max_daily_drop': -0.08,
                'circuit_breakers': 3
            },
            'china_lockdown_2022': {
                'start': date(2022, 4, 25),  # Shanghai lockdown impact
                'end': date(2022, 5, 6),
                'avg_volatility': 0.04,
                'max_daily_drop': -0.09,
                'circuit_breakers': 5
            }
        }
        
        # Taiwan-specific sectors and their characteristics
        self.taiwan_sectors = {
            'semiconductors': {
                'weight': 0.35,  # 35% of TAIEX
                'volatility_multiplier': 1.2,
                'stocks': ['2330.TW', '2454.TW', '3034.TW', '2382.TW']  # TSMC, MediaTek, etc.
            },
            'electronics': {
                'weight': 0.15,
                'volatility_multiplier': 1.1,
                'stocks': ['2317.TW', '2303.TW', '2308.TW']  # Foxconn, UMC, etc.
            },
            'financials': {
                'weight': 0.12,
                'volatility_multiplier': 0.8,
                'stocks': ['2881.TW', '2882.TW', '2883.TW']  # Banks
            },
            'petrochemicals': {
                'weight': 0.08,
                'volatility_multiplier': 1.3,
                'stocks': ['1301.TW', '1303.TW', '6505.TW']  # Formosa, etc.
            },
            'plastics': {
                'weight': 0.06,
                'volatility_multiplier': 1.0,
                'stocks': ['1101.TW', '1102.TW', '1326.TW']
            },
            'other': {
                'weight': 0.24,
                'volatility_multiplier': 0.9,
                'stocks': []  # Smaller sectors
            }
        }
        
    def setup_logging(self, log_level: str):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def generate_taiwan_stress_data(self, 
                                   scenario: str,
                                   stock_count: int = 1000,
                                   include_sectors: bool = True) -> pd.DataFrame:
        """
        Generate Taiwan market data for stress testing scenarios.
        
        Args:
            scenario: Stress scenario name from self.stress_periods
            stock_count: Number of stocks to simulate
            include_sectors: Whether to include sector-specific behavior
            
        Returns:
            DataFrame with Taiwan market stress data
        """
        self.logger.info(f"Generating Taiwan stress data for scenario: {scenario}")
        
        if scenario not in self.stress_periods:
            raise ValueError(f"Unknown scenario: {scenario}. Available: {list(self.stress_periods.keys())}")
            
        scenario_config = self.stress_periods[scenario]
        start_time = time.time()
        
        # Generate date range for scenario
        start_date = scenario_config['start']
        end_date = scenario_config['end']
        trading_days = pd.bdate_range(start=start_date, end=end_date)
        
        # Remove Taiwan holidays (simplified)
        taiwan_holidays = []
        if start_date.year == 2020:
            taiwan_holidays = [date(2020, 1, 1), date(2020, 1, 27), date(2020, 1, 28)]
        elif start_date.year == 2022:
            taiwan_holidays = [date(2022, 5, 3)]  # Labor Day observed
            
        trading_days = [d for d in trading_days if d.date() not in taiwan_holidays]
        
        # Generate Taiwan stock IDs
        tse_stocks = [f"{2000 + i:04d}.TW" for i in range(stock_count // 2)]
        tpex_stocks = [f"{1000 + i:04d}.TWO" for i in range(stock_count - len(tse_stocks))]
        stock_ids = tse_stocks + tpex_stocks
        
        # Create MultiIndex
        index = pd.MultiIndex.from_product(
            [trading_days, stock_ids],
            names=['date', 'stock_id']
        )
        
        n_samples = len(index)
        np.random.seed(42)
        
        # Base price simulation with Taiwan characteristics
        base_volatility = scenario_config['avg_volatility']
        max_drop = scenario_config['max_daily_drop']
        
        # Generate returns with stress characteristics
        returns = np.random.normal(
            -0.001,  # Slightly negative bias during stress
            base_volatility,
            n_samples
        )
        
        # Apply circuit breaker effects
        circuit_breaker_days = scenario_config['circuit_breakers']
        total_days = len(trading_days)
        
        if circuit_breaker_days > 0:
            # Select random days for circuit breakers
            cb_day_indices = np.random.choice(total_days, circuit_breaker_days, replace=False)
            
            for cb_day_idx in cb_day_indices:
                # Find all stocks for this day
                day_mask = np.array([i // stock_count == cb_day_idx for i in range(n_samples)])
                
                # Apply extreme negative returns (but within price limits)
                returns[day_mask] = np.random.uniform(
                    max_drop * 0.8,  # Near price limit
                    max_drop * 0.6,  # But not exactly at limit
                    np.sum(day_mask)
                )
        
        # Apply sector-specific effects if requested
        if include_sectors:
            for sector_name, sector_config in self.taiwan_sectors.items():
                volatility_mult = sector_config['volatility_multiplier']
                sector_stocks = sector_config.get('stocks', [])
                
                # Apply to sector stocks or randomly assigned stocks
                if sector_stocks:
                    for stock in sector_stocks:
                        if stock in stock_ids:
                            stock_mask = np.array([
                                index[i][1] == stock for i in range(n_samples)
                            ])
                            returns[stock_mask] *= volatility_mult
                else:
                    # Randomly assign sector effects
                    sector_size = int(stock_count * sector_config['weight'])
                    sector_stock_indices = np.random.choice(
                        stock_count, sector_size, replace=False
                    )
                    
                    for stock_idx in sector_stock_indices:
                        stock_mask = np.array([
                            i % stock_count == stock_idx for i in range(n_samples)
                        ])
                        returns[stock_mask] *= volatility_mult
        
        # Convert returns to prices (starting from realistic Taiwan levels)
        prices = []
        volumes = []
        turnovers = []
        bid_ask_spreads = []
        
        for i, (trading_day, stock_id) in enumerate(index):
            # Base price level (Taiwan typical range: 20-500 TWD)
            if stock_id.endswith('.TW'):  # TSE
                base_price = np.random.lognormal(4.0, 0.8)  # ~50-200 TWD
            else:  # TPEx
                base_price = np.random.lognormal(3.0, 0.8)  # ~20-100 TWD
                
            # Apply return
            price = base_price * (1 + returns[i])
            
            # Ensure price limits (Taiwan 10% daily limit)
            price = max(base_price * 0.9, min(price, base_price * 1.1))
            prices.append(price)
            
            # Volume increases during stress (higher turnover)
            stress_volume_mult = 1 + abs(returns[i]) * 10  # More volume with bigger moves
            base_volume = np.random.lognormal(12, 1)  # Taiwan typical volume
            volume = base_volume * stress_volume_mult
            volumes.append(volume)
            
            # Turnover rate increases during stress
            base_turnover = np.random.exponential(0.02)
            turnover = min(base_turnover * stress_volume_mult, 0.15)  # Cap at 15%
            turnovers.append(turnover)
            
            # Bid-ask spreads widen during stress
            base_spread = price * 0.001  # 0.1% base spread
            stress_spread_mult = 1 + abs(returns[i]) * 5
            spread = base_spread * stress_spread_mult
            bid_ask_spreads.append(spread)
        
        # Create additional Taiwan-specific features
        feature_data = {
            'close_price': prices,
            'returns': returns,
            'volume': volumes,
            'turnover': turnovers,
            'bid_ask_spread': bid_ask_spreads,
            
            # Market microstructure during stress
            'order_imbalance': np.random.normal(0, 0.1, n_samples),  # Higher imbalance
            'trade_frequency': np.random.poisson(200, n_samples),    # More frequent trading
            'block_trade_ratio': np.random.beta(2, 8, n_samples) * 0.3,  # Institutional activity
            
            # Foreign investment flows (important for Taiwan)
            'foreign_net_buy': np.random.normal(-0.01, 0.05, n_samples),  # Net selling during stress
            'foreign_ownership': np.random.beta(2, 3, n_samples) * 0.5,   # Up to 50%
            
            # Taiwan-specific risk factors
            'semiconductor_beta': np.random.lognormal(0, 0.3, n_samples),  # Semi exposure
            'china_exposure': np.random.uniform(0, 0.8, n_samples),        # China business exposure
            'export_ratio': np.random.beta(3, 2, n_samples),               # Export dependency
            
            # Technical indicators during stress
            'rsi_14': np.random.beta(1, 3, n_samples) * 100,      # RSI tends low during stress
            'volatility_20d': np.random.exponential(base_volatility * 2, n_samples),
            'momentum_20d': returns + np.random.normal(0, 0.01, n_samples),
        }
        
        # Create DataFrame
        df = pd.DataFrame(feature_data, index=index)
        
        generation_time = time.time() - start_time
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        self.logger.info(
            f"Taiwan stress data generated: {df.shape}, Scenario: {scenario}, "
            f"Memory: {memory_usage:.1f}MB, Time: {generation_time:.1f}s"
        )
        
        return df
        
    def test_high_volatility_periods(self, scenario: str = 'march_2020_crash') -> Dict[str, Any]:
        """
        Test system performance during high volatility periods.
        
        Args:
            scenario: Historical volatility scenario to simulate
            
        Returns:
            High volatility stress test results
        """
        self.logger.info(f"Testing high volatility period: {scenario}")
        
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            # Generate stress scenario data
            stress_data = self.generate_taiwan_stress_data(scenario, stock_count=800)
            
            # Test data processing under stress conditions
            processing_start = time.time()
            
            # Simulate real-time processing challenges
            # 1. High frequency updates (simulate rapid price changes)
            update_times = []
            for _ in range(100):  # 100 rapid updates
                update_start = time.time()
                
                # Simulate price update processing
                latest_prices = stress_data.groupby('stock_id')['close_price'].last()
                price_changes = stress_data.groupby('stock_id')['returns'].last()
                
                # Check for circuit breaker triggers
                circuit_breaker_stocks = latest_prices[abs(price_changes) >= 0.07]
                
                update_time = time.time() - update_start
                update_times.append(update_time)
            
            # 2. Volatility calculations during stress
            volatility_start = time.time()
            rolling_volatility = stress_data.groupby('stock_id')['returns'].rolling(
                window=20, min_periods=5
            ).std()
            volatility_time = time.time() - volatility_start
            
            # 3. Risk calculations
            risk_start = time.time()
            
            # Portfolio risk during stress
            returns_pivot = stress_data.pivot_table(
                values='returns', 
                index='date', 
                columns='stock_id'
            ).fillna(0)
            
            # Correlation matrix (computationally intensive)
            correlation_matrix = returns_pivot.corr()
            
            # VaR calculations
            portfolio_returns = returns_pivot.mean(axis=1)
            var_95 = np.percentile(portfolio_returns, 5)
            var_99 = np.percentile(portfolio_returns, 1)
            
            risk_time = time.time() - risk_start
            
            # 4. Market impact calculations
            impact_start = time.time()
            
            # Simulate order execution during stress
            avg_spread = stress_data.groupby('stock_id')['bid_ask_spread'].mean()
            avg_volume = stress_data.groupby('stock_id')['volume'].mean()
            market_impact = avg_spread / avg_volume  # Simplified impact model
            
            impact_time = time.time() - impact_start
            
            total_processing_time = time.time() - processing_start
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Performance metrics
            avg_update_time = np.mean(update_times)
            max_update_time = np.max(update_times)
            updates_per_second = 1 / avg_update_time if avg_update_time > 0 else 0
            
            results = {
                'success': True,
                'scenario': scenario,
                'data_shape': stress_data.shape,
                'scenario_config': self.stress_periods[scenario],
                'processing_time_seconds': total_processing_time,
                'volatility_calculation_time': volatility_time,
                'risk_calculation_time': risk_time,
                'market_impact_calculation_time': impact_time,
                'avg_update_latency_ms': avg_update_time * 1000,
                'max_update_latency_ms': max_update_time * 1000,
                'updates_per_second': updates_per_second,
                'initial_memory_mb': initial_memory,
                'peak_memory_mb': peak_memory,
                'memory_increase_mb': peak_memory - initial_memory,
                
                # Market stress metrics
                'circuit_breaker_triggers': len(circuit_breaker_stocks),
                'avg_volatility': rolling_volatility.mean(),
                'max_volatility': rolling_volatility.max(),
                'portfolio_var_95': var_95,
                'portfolio_var_99': var_99,
                'avg_market_impact': market_impact.mean(),
                'max_market_impact': market_impact.max(),
                
                # Performance validation
                'latency_p50_ms': np.percentile(np.array(update_times) * 1000, 50),
                'latency_p95_ms': np.percentile(np.array(update_times) * 1000, 95),
                'latency_p99_ms': np.percentile(np.array(update_times) * 1000, 99),
                'throughput_ops_per_sec': len(stress_data) / total_processing_time
            }
            
            # Validate stress test requirements
            meets_latency_req = results['latency_p95_ms'] <= 100  # <100ms P95
            meets_memory_req = peak_memory <= 16384              # <16GB
            handles_volatility = results['max_volatility'] < 0.2  # Handles high vol
            
            results.update({
                'meets_latency_requirement': meets_latency_req,
                'meets_memory_requirement': meets_memory_req,
                'handles_high_volatility': handles_volatility,
                'stress_test_passed': meets_latency_req and meets_memory_req and handles_volatility
            })
            
            self.logger.info(
                f"High volatility test complete - Scenario: {scenario}, "
                f"Latency P95: {results['latency_p95_ms']:.1f}ms, "
                f"Memory: {peak_memory:.1f}MB, Throughput: {results['throughput_ops_per_sec']:.0f} ops/sec"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"High volatility stress test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'scenario': scenario,
                'total_time_seconds': time.time() - start_time,
                'peak_memory_mb': psutil.Process().memory_info().rss / 1024 / 1024
            }
        finally:
            gc.collect()
            
    def test_circuit_breaker_scenarios(self) -> Dict[str, Any]:
        """
        Test system behavior during circuit breaker events.
        
        Returns:
            Circuit breaker stress test results
        """
        self.logger.info("Testing circuit breaker scenarios")
        
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            # Generate data with multiple circuit breaker events
            cb_data = self.generate_taiwan_stress_data('march_2020_crash', stock_count=500)
            
            # Simulate circuit breaker detection and handling
            detection_times = []
            recovery_times = []
            
            # Group by date to simulate daily circuit breaker checks
            for trading_date, day_data in cb_data.groupby('date'):
                detection_start = time.time()
                
                # Detect circuit breaker triggers (3.5% market-wide decline)
                market_return = day_data['returns'].mean()
                
                if market_return <= -self.circuit_breaker_threshold:
                    # Simulate circuit breaker activation
                    cb_stocks = day_data[day_data['returns'] <= -0.07]  # Stocks hitting limit
                    
                    # Simulate trading halt processing
                    halt_start = time.time()
                    
                    # Process halt notifications
                    for stock_id in cb_stocks.index.get_level_values('stock_id').unique():
                        # Simulate halt notification processing
                        time.sleep(0.001)  # 1ms per stock processing
                        
                    # Resume trading processing
                    recovery_start = time.time()
                    
                    # Simulate order book reconstruction
                    order_book_rebuild_time = len(cb_stocks) * 0.002  # 2ms per stock
                    time.sleep(order_book_rebuild_time)
                    
                    recovery_time = time.time() - recovery_start
                    recovery_times.append(recovery_time)
                    
                detection_time = time.time() - detection_start
                detection_times.append(detection_time)
            
            # Test order routing during circuit breakers
            routing_start = time.time()
            
            # Simulate order routing with halted stocks
            total_orders = 10000
            halted_stocks = cb_data[cb_data['returns'] <= -0.07].index.get_level_values('stock_id').unique()
            
            routing_times = []
            for _ in range(1000):  # Test 1000 orders
                order_start = time.time()
                
                # Check if stock is halted
                random_stock = np.random.choice(cb_data.index.get_level_values('stock_id').unique())
                is_halted = random_stock in halted_stocks
                
                if is_halted:
                    # Reject order - should be fast
                    pass
                else:
                    # Process order normally
                    pass
                    
                routing_time = time.time() - order_start
                routing_times.append(routing_time)
                
            total_routing_time = time.time() - routing_start
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Calculate performance metrics
            avg_detection_time = np.mean(detection_times) if detection_times else 0
            avg_recovery_time = np.mean(recovery_times) if recovery_times else 0
            avg_routing_time = np.mean(routing_times) if routing_times else 0
            
            results = {
                'success': True,
                'data_shape': cb_data.shape,
                'circuit_breaker_days': len([t for t in detection_times if t > 0.01]),  # Days with CB
                'total_halted_stocks': len(halted_stocks),
                'avg_detection_time_ms': avg_detection_time * 1000,
                'max_detection_time_ms': max(detection_times) * 1000 if detection_times else 0,
                'avg_recovery_time_ms': avg_recovery_time * 1000,
                'max_recovery_time_ms': max(recovery_times) * 1000 if recovery_times else 0,
                'avg_order_routing_time_ms': avg_routing_time * 1000,
                'max_order_routing_time_ms': max(routing_times) * 1000 if routing_times else 0,
                'orders_per_second': len(routing_times) / total_routing_time if total_routing_time > 0 else 0,
                'initial_memory_mb': initial_memory,
                'peak_memory_mb': peak_memory,
                'memory_increase_mb': peak_memory - initial_memory,
                
                # Circuit breaker specific metrics
                'circuit_breaker_threshold': self.circuit_breaker_threshold,
                'price_limit_percentage': self.tse_price_limit,
                'market_recovery_capability': avg_recovery_time < 0.1,  # <100ms recovery
                
                # Performance requirements
                'latency_p95_ms': np.percentile(np.array(routing_times) * 1000, 95),
                'throughput_ops_per_sec': len(routing_times) / total_routing_time if total_routing_time > 0 else 0
            }
            
            # Validate circuit breaker handling requirements
            meets_detection_req = results['avg_detection_time_ms'] <= 10      # <10ms detection
            meets_recovery_req = results['avg_recovery_time_ms'] <= 100       # <100ms recovery
            meets_routing_req = results['avg_order_routing_time_ms'] <= 5     # <5ms routing
            
            results.update({
                'meets_detection_requirement': meets_detection_req,
                'meets_recovery_requirement': meets_recovery_req,
                'meets_routing_requirement': meets_routing_req,
                'circuit_breaker_test_passed': meets_detection_req and meets_recovery_req and meets_routing_req
            })
            
            self.logger.info(
                f"Circuit breaker test complete - "
                f"Detection: {results['avg_detection_time_ms']:.1f}ms, "
                f"Recovery: {results['avg_recovery_time_ms']:.1f}ms, "
                f"Routing: {results['avg_order_routing_time_ms']:.1f}ms"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Circuit breaker test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_time_seconds': time.time() - start_time,
                'peak_memory_mb': psutil.Process().memory_info().rss / 1024 / 1024
            }
        finally:
            gc.collect()
            
    def test_sector_concentration_stress(self) -> Dict[str, Any]:
        """
        Test performance when semiconductor sector (35% of TAIEX) faces stress.
        
        Returns:
            Sector concentration stress test results
        """
        self.logger.info("Testing semiconductor sector concentration stress")
        
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            # Generate data with semiconductor stress
            sector_data = self.generate_taiwan_stress_data('march_2020_crash', stock_count=1000)
            
            # Identify semiconductor stocks (simplified - use top stocks)
            semi_stocks = [stock for stock in sector_data.index.get_level_values('stock_id').unique() 
                          if stock.startswith('233') or stock.startswith('245')][:50]  # Top 50 semi stocks
            
            # Apply additional stress to semiconductor sector
            semi_mask = sector_data.index.get_level_values('stock_id').isin(semi_stocks)
            sector_data.loc[semi_mask, 'returns'] *= 1.5  # 50% more volatility
            sector_data.loc[semi_mask, 'volume'] *= 2.0   # 2x volume
            
            # Test portfolio impact calculations
            impact_start = time.time()
            
            # Calculate sector weights and impacts
            total_stocks = len(sector_data.index.get_level_values('stock_id').unique())
            semi_weight = len(semi_stocks) / total_stocks
            
            # Portfolio return decomposition
            semi_returns = sector_data[semi_mask]['returns']
            non_semi_returns = sector_data[~semi_mask]['returns']
            
            portfolio_return = (semi_weight * semi_returns.mean() + 
                              (1 - semi_weight) * non_semi_returns.mean())
            
            # Risk contribution analysis
            semi_vol = semi_returns.std()
            portfolio_vol = sector_data['returns'].std()
            semi_risk_contribution = semi_weight * semi_vol / portfolio_vol
            
            # Correlation impact
            sector_correlation = np.corrcoef(
                sector_data.pivot_table(values='returns', index='date', columns='stock_id').fillna(0).T
            )
            avg_correlation = np.mean(sector_correlation[np.triu_indices_from(sector_correlation, k=1)])
            
            impact_time = time.time() - impact_start
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Performance metrics
            results = {
                'success': True,
                'data_shape': sector_data.shape,
                'semiconductor_stocks_count': len(semi_stocks),
                'semiconductor_weight': semi_weight,
                'impact_calculation_time_seconds': impact_time,
                'portfolio_return': portfolio_return,
                'semiconductor_volatility': semi_vol,
                'portfolio_volatility': portfolio_vol,
                'semiconductor_risk_contribution': semi_risk_contribution,
                'average_correlation': avg_correlation,
                'initial_memory_mb': initial_memory,
                'peak_memory_mb': peak_memory,
                'memory_increase_mb': peak_memory - initial_memory,
                
                # Concentration risk metrics
                'concentration_ratio': semi_weight,  # Sector concentration
                'diversification_ratio': 1 - avg_correlation,  # Portfolio diversification
                'tail_risk_95': np.percentile(sector_data['returns'], 5),
                'tail_risk_99': np.percentile(sector_data['returns'], 1),
                
                # Performance requirements
                'throughput_ops_per_sec': len(sector_data) / impact_time
            }
            
            # Validate concentration stress requirements
            handles_concentration = semi_risk_contribution < 0.6  # <60% risk from one sector
            adequate_diversification = results['diversification_ratio'] > 0.3  # >30% diversification
            reasonable_performance = impact_time < 60  # <1 minute for calculations
            
            results.update({
                'handles_concentration_risk': handles_concentration,
                'adequate_diversification': adequate_diversification,
                'reasonable_performance': reasonable_performance,
                'concentration_stress_passed': all([handles_concentration, adequate_diversification, reasonable_performance])
            })
            
            self.logger.info(
                f"Sector concentration test complete - "
                f"Semi weight: {semi_weight:.1%}, Risk contribution: {semi_risk_contribution:.1%}, "
                f"Correlation: {avg_correlation:.3f}"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Sector concentration stress test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_time_seconds': time.time() - start_time,
                'peak_memory_mb': psutil.Process().memory_info().rss / 1024 / 1024
            }
        finally:
            gc.collect()
            
    def run_comprehensive_taiwan_stress_suite(self) -> Dict[str, Any]:
        """Run comprehensive Taiwan market stress testing suite."""
        self.logger.info("Starting comprehensive Taiwan market stress testing")
        
        suite_start = time.time()
        results = {}
        
        # Test high volatility periods
        for scenario in ['march_2020_crash', 'may_2022_selloff']:
            self.logger.info(f"Running high volatility test: {scenario}")
            results[f'volatility_{scenario}'] = self.test_high_volatility_periods(scenario)
            
        # Test circuit breaker scenarios
        self.logger.info("Running circuit breaker tests...")
        results['circuit_breakers'] = self.test_circuit_breaker_scenarios()
        
        # Test sector concentration stress
        self.logger.info("Running sector concentration tests...")
        results['sector_concentration'] = self.test_sector_concentration_stress()
        
        total_time = time.time() - suite_start
        
        # Generate summary
        successful_tests = sum(1 for r in results.values() if r.get('success', False))
        stress_tests_passed = sum(1 for r in results.values() 
                                if r.get('stress_test_passed') or r.get('circuit_breaker_test_passed') or r.get('concentration_stress_passed'))
        
        results['summary'] = {
            'total_stress_test_time_seconds': total_time,
            'total_tests': len(results) - 1,
            'successful_tests': successful_tests,
            'stress_tests_passed': stress_tests_passed,
            'overall_stress_ready': stress_tests_passed == len(results) - 1,
            'stress_test_timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(
            f"Comprehensive Taiwan stress testing complete - "
            f"Time: {total_time:.1f}s, Success: {successful_tests}/{len(results)-1}, "
            f"Stress Ready: {stress_tests_passed}/{len(results)-1}"
        )
        
        return results