"""
Risk Integration and Position Sizing Validation for ML4T Taiwan Equity Alpha.

This module integrates with risk management systems to validate position sizes,
risk constraints, and portfolio construction with Taiwan market-specific risk models.

Key Features:
- Position sizing validation
- Risk constraint checking
- Portfolio construction validation
- Taiwan market risk model integration
- Leverage and margin validation
- Concentration risk assessment

Author: ML4T Team
Date: 2025-09-24
"""

from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Union, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
import pandas as pd
from decimal import Decimal
import math

logger = logging.getLogger(__name__)


class RiskConstraintType(Enum):
    """Types of risk constraints."""
    POSITION_SIZE = "position_size"
    SECTOR_CONCENTRATION = "sector_concentration"
    SINGLE_STOCK_LIMIT = "single_stock_limit"
    LEVERAGE_LIMIT = "leverage_limit"
    VALUE_AT_RISK = "value_at_risk"
    EXPECTED_SHORTFALL = "expected_shortfall"
    TRACKING_ERROR = "tracking_error"
    BETA_CONSTRAINT = "beta_constraint"
    LIQUIDITY_CONSTRAINT = "liquidity_constraint"
    CORRELATION_LIMIT = "correlation_limit"


class RiskSeverity(Enum):
    """Risk violation severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PositionSizeMethod(Enum):
    """Position sizing methodologies."""
    EQUAL_WEIGHT = "equal_weight"
    MARKET_CAP_WEIGHT = "market_cap_weight"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    RISK_PARITY = "risk_parity"
    OPTIMAL_F = "optimal_f"
    KELLY_CRITERION = "kelly_criterion"
    MAX_DIVERSIFICATION = "max_diversification"


@dataclass
class RiskConstraint:
    """Definition of a risk constraint."""
    constraint_type: RiskConstraintType
    limit_value: float
    current_value: Optional[float] = None
    warning_threshold: float = 0.8  # Warn at 80% of limit
    description: str = ""
    reference: str = ""
    enforcement_level: RiskSeverity = RiskSeverity.HIGH
    
    def is_violated(self) -> bool:
        """Check if constraint is violated."""
        if self.current_value is None:
            return False
        return self.current_value > self.limit_value
    
    def is_warning(self) -> bool:
        """Check if constraint is at warning level."""
        if self.current_value is None:
            return False
        return self.current_value > (self.limit_value * self.warning_threshold)
    
    def violation_percentage(self) -> float:
        """Calculate violation percentage."""
        if self.current_value is None or self.limit_value <= 0:
            return 0.0
        return max(0, (self.current_value - self.limit_value) / self.limit_value)


@dataclass
class RiskViolation:
    """A risk constraint violation."""
    constraint: RiskConstraint
    severity: RiskSeverity
    description: str
    violation_amount: float
    violation_percentage: float
    symbol: Optional[str] = None
    sector: Optional[str] = None
    date: Optional[date] = None
    remediation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            'constraint_type': self.constraint.constraint_type.value,
            'severity': self.severity.value,
            'description': self.description,
            'violation_amount': self.violation_amount,
            'violation_percentage': self.violation_percentage,
            'symbol': self.symbol,
            'sector': self.sector,
            'date': self.date.isoformat() if self.date else None,
            'limit_value': self.constraint.limit_value,
            'current_value': self.constraint.current_value,
            'remediation': self.remediation
        }


@dataclass
class PositionRiskCheck:
    """Individual position risk assessment."""
    symbol: str
    weight: float
    dollar_amount: float
    volatility: float
    beta: float
    sector: str
    market_cap: float
    liquidity_score: float
    risk_contribution: float
    position_size_score: float  # 0-100, higher is better
    violations: List[RiskViolation] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            'symbol': self.symbol,
            'weight': self.weight,
            'dollar_amount': self.dollar_amount,
            'volatility': self.volatility,
            'beta': self.beta,
            'sector': self.sector,
            'market_cap': self.market_cap,
            'liquidity_score': self.liquidity_score,
            'risk_contribution': self.risk_contribution,
            'position_size_score': self.position_size_score,
            'violations': [v.to_dict() for v in self.violations]
        }


@dataclass
class RiskConfig:
    """Configuration for risk validation."""
    
    # Position size limits
    max_single_position: float = 0.05  # 5% max position
    max_sector_concentration: float = 0.20  # 20% max sector
    min_position_size: float = 0.001  # 0.1% minimum position
    
    # Portfolio risk limits
    max_portfolio_volatility: float = 0.20  # 20% annual vol
    max_tracking_error: float = 0.08  # 8% tracking error
    max_leverage: float = 1.0  # 100% leverage (1x)
    max_portfolio_beta: float = 1.5  # 1.5x market beta
    
    # Risk metrics
    var_confidence_level: float = 0.95  # 95% VaR
    var_horizon_days: int = 1  # 1-day VaR
    max_var_percentage: float = 0.02  # 2% daily VaR limit
    
    # Taiwan market specific
    adv_multiplier: float = 0.10  # 10% of ADV max position
    min_market_cap_ntd: float = 5e9  # 5B NTD minimum market cap
    liquidity_score_threshold: float = 0.3  # Minimum liquidity score
    
    # Position sizing method
    position_sizing_method: PositionSizeMethod = PositionSizeMethod.VOLATILITY_ADJUSTED
    target_portfolio_volatility: float = 0.12  # 12% target volatility
    
    # Risk model parameters
    risk_model_lookback_days: int = 252  # 1 year lookback
    correlation_decay_factor: float = 0.97  # Exponential decay
    volatility_decay_factor: float = 0.94  # EWMA decay
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'max_single_position': self.max_single_position,
            'max_sector_concentration': self.max_sector_concentration,
            'min_position_size': self.min_position_size,
            'max_portfolio_volatility': self.max_portfolio_volatility,
            'max_tracking_error': self.max_tracking_error,
            'max_leverage': self.max_leverage,
            'max_portfolio_beta': self.max_portfolio_beta,
            'var_confidence_level': self.var_confidence_level,
            'var_horizon_days': self.var_horizon_days,
            'max_var_percentage': self.max_var_percentage,
            'position_sizing_method': self.position_sizing_method.value,
            'target_portfolio_volatility': self.target_portfolio_volatility
        }


class RiskValidator:
    """Validates portfolio construction against risk constraints."""
    
    def __init__(self, config: Optional[RiskConfig] = None):
        """Initialize the risk validator.
        
        Args:
            config: Risk validation configuration
        """
        self.config = config or RiskConfig()
        self.risk_model = self._initialize_risk_model()
        self.constraints = self._create_default_constraints()
        
    def validate_portfolio(
        self,
        portfolio_weights: Dict[str, float],
        portfolio_value: float,
        date: date,
        market_data: Optional[pd.DataFrame] = None
    ) -> Tuple[List[PositionRiskCheck], List[RiskViolation]]:
        """Validate entire portfolio against risk constraints.
        
        Args:
            portfolio_weights: Dictionary of symbol -> weight
            portfolio_value: Total portfolio value in NTD
            date: Validation date
            market_data: Optional market data for calculations
            
        Returns:
            Tuple of (position checks, portfolio violations)
        """
        try:
            # Get market data if not provided
            if market_data is None:
                market_data = self._get_market_data(list(portfolio_weights.keys()), date)
            
            # Validate individual positions
            position_checks = []
            for symbol, weight in portfolio_weights.items():
                check = self._validate_position(
                    symbol, weight, portfolio_value, date, market_data
                )
                position_checks.append(check)
            
            # Validate portfolio-level constraints
            portfolio_violations = self._validate_portfolio_constraints(
                portfolio_weights, portfolio_value, date, market_data
            )
            
            return position_checks, portfolio_violations
            
        except Exception as e:
            logger.error(f"Error validating portfolio: {e}")
            raise
    
    def calculate_optimal_position_sizes(
        self,
        alpha_signals: Dict[str, float],
        portfolio_value: float,
        date: date,
        market_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Calculate optimal position sizes based on alpha signals and risk constraints.
        
        Args:
            alpha_signals: Dictionary of symbol -> alpha signal strength
            portfolio_value: Total portfolio value
            date: Calculation date
            market_data: Optional market data
            
        Returns:
            Dictionary of symbol -> optimal weight
        """
        try:
            if not alpha_signals:
                return {}
            
            # Get market data if not provided
            if market_data is None:
                market_data = self._get_market_data(list(alpha_signals.keys()), date)
            
            # Calculate position sizes based on method
            if self.config.position_sizing_method == PositionSizeMethod.EQUAL_WEIGHT:
                return self._calculate_equal_weight_positions(alpha_signals)
            
            elif self.config.position_sizing_method == PositionSizeMethod.VOLATILITY_ADJUSTED:
                return self._calculate_volatility_adjusted_positions(
                    alpha_signals, market_data, date
                )
            
            elif self.config.position_sizing_method == PositionSizeMethod.RISK_PARITY:
                return self._calculate_risk_parity_positions(
                    alpha_signals, market_data, date
                )
            
            elif self.config.position_sizing_method == PositionSizeMethod.KELLY_CRITERION:
                return self._calculate_kelly_positions(
                    alpha_signals, market_data, date
                )
            
            else:
                # Default to volatility adjusted
                return self._calculate_volatility_adjusted_positions(
                    alpha_signals, market_data, date
                )
            
        except Exception as e:
            logger.error(f"Error calculating optimal position sizes: {e}")
            raise
    
    def _validate_position(
        self,
        symbol: str,
        weight: float,
        portfolio_value: float,
        date: date,
        market_data: pd.DataFrame
    ) -> PositionRiskCheck:
        """Validate individual position against constraints."""
        violations = []
        
        # Get stock data
        stock_data = market_data[market_data['symbol'] == symbol]
        if stock_data.empty:
            logger.warning(f"No market data found for {symbol}")
            return PositionRiskCheck(
                symbol=symbol,
                weight=weight,
                dollar_amount=weight * portfolio_value,
                volatility=0.0,
                beta=1.0,
                sector="unknown",
                market_cap=0.0,
                liquidity_score=0.0,
                risk_contribution=0.0,
                position_size_score=0.0,
                violations=violations
            )
        
        stock_info = stock_data.iloc[0]
        dollar_amount = weight * portfolio_value
        
        # Check position size constraints
        if weight > self.config.max_single_position:
            violations.append(RiskViolation(
                constraint=RiskConstraint(
                    RiskConstraintType.SINGLE_STOCK_LIMIT,
                    self.config.max_single_position,
                    weight
                ),
                severity=RiskSeverity.HIGH,
                description=f"Position weight {weight:.3f} exceeds limit {self.config.max_single_position:.3f}",
                violation_amount=weight - self.config.max_single_position,
                violation_percentage=(weight - self.config.max_single_position) / self.config.max_single_position,
                symbol=symbol,
                date=date,
                remediation=f"Reduce position to {self.config.max_single_position:.3f} max weight"
            ))
        
        # Check minimum position size
        if 0 < weight < self.config.min_position_size:
            violations.append(RiskViolation(
                constraint=RiskConstraint(
                    RiskConstraintType.POSITION_SIZE,
                    self.config.min_position_size,
                    weight
                ),
                severity=RiskSeverity.LOW,
                description=f"Position weight {weight:.4f} below minimum {self.config.min_position_size:.4f}",
                violation_amount=self.config.min_position_size - weight,
                violation_percentage=(self.config.min_position_size - weight) / self.config.min_position_size,
                symbol=symbol,
                date=date,
                remediation="Consider increasing position size or removing"
            ))
        
        # Check liquidity constraints
        liquidity_score = stock_info.get('liquidity_score', 0.5)
        if liquidity_score < self.config.liquidity_score_threshold:
            violations.append(RiskViolation(
                constraint=RiskConstraint(
                    RiskConstraintType.LIQUIDITY_CONSTRAINT,
                    self.config.liquidity_score_threshold,
                    liquidity_score
                ),
                severity=RiskSeverity.MEDIUM,
                description=f"Liquidity score {liquidity_score:.3f} below threshold {self.config.liquidity_score_threshold:.3f}",
                violation_amount=self.config.liquidity_score_threshold - liquidity_score,
                violation_percentage=(self.config.liquidity_score_threshold - liquidity_score) / self.config.liquidity_score_threshold,
                symbol=symbol,
                date=date,
                remediation="Consider reducing position or finding more liquid alternatives"
            ))
        
        # Calculate risk contribution
        volatility = stock_info.get('volatility', 0.2)
        risk_contribution = weight * volatility
        
        # Calculate position size score
        position_size_score = self._calculate_position_size_score(
            symbol, weight, stock_info, violations
        )
        
        return PositionRiskCheck(
            symbol=symbol,
            weight=weight,
            dollar_amount=dollar_amount,
            volatility=volatility,
            beta=stock_info.get('beta', 1.0),
            sector=stock_info.get('sector', 'unknown'),
            market_cap=stock_info.get('market_cap', 0.0),
            liquidity_score=liquidity_score,
            risk_contribution=risk_contribution,
            position_size_score=position_size_score,
            violations=violations
        )
    
    def _validate_portfolio_constraints(
        self,
        portfolio_weights: Dict[str, float],
        portfolio_value: float,
        date: date,
        market_data: pd.DataFrame
    ) -> List[RiskViolation]:
        """Validate portfolio-level risk constraints."""
        violations = []
        
        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(
            portfolio_weights, market_data, date
        )
        
        # Check portfolio volatility
        portfolio_vol = portfolio_metrics.get('volatility', 0.0)
        if portfolio_vol > self.config.max_portfolio_volatility:
            violations.append(RiskViolation(
                constraint=RiskConstraint(
                    RiskConstraintType.VALUE_AT_RISK,
                    self.config.max_portfolio_volatility,
                    portfolio_vol
                ),
                severity=RiskSeverity.HIGH,
                description=f"Portfolio volatility {portfolio_vol:.3f} exceeds limit {self.config.max_portfolio_volatility:.3f}",
                violation_amount=portfolio_vol - self.config.max_portfolio_volatility,
                violation_percentage=(portfolio_vol - self.config.max_portfolio_volatility) / self.config.max_portfolio_volatility,
                date=date,
                remediation="Reduce position sizes or add lower volatility stocks"
            ))
        
        # Check tracking error
        tracking_error = portfolio_metrics.get('tracking_error', 0.0)
        if tracking_error > self.config.max_tracking_error:
            violations.append(RiskViolation(
                constraint=RiskConstraint(
                    RiskConstraintType.TRACKING_ERROR,
                    self.config.max_tracking_error,
                    tracking_error
                ),
                severity=RiskSeverity.MEDIUM,
                description=f"Tracking error {tracking_error:.3f} exceeds limit {self.config.max_tracking_error:.3f}",
                violation_amount=tracking_error - self.config.max_tracking_error,
                violation_percentage=(tracking_error - self.config.max_tracking_error) / self.config.max_tracking_error,
                date=date,
                remediation="Reduce active positions or adjust to benchmark weights"
            ))
        
        # Check sector concentration
        sector_concentrations = self._calculate_sector_concentrations(
            portfolio_weights, market_data
        )
        
        for sector, concentration in sector_concentrations.items():
            if concentration > self.config.max_sector_concentration:
                violations.append(RiskViolation(
                    constraint=RiskConstraint(
                        RiskConstraintType.SECTOR_CONCENTRATION,
                        self.config.max_sector_concentration,
                        concentration
                    ),
                    severity=RiskSeverity.HIGH,
                    description=f"Sector {sector} concentration {concentration:.3f} exceeds limit {self.config.max_sector_concentration:.3f}",
                    violation_amount=concentration - self.config.max_sector_concentration,
                    violation_percentage=(concentration - self.config.max_sector_concentration) / self.config.max_sector_concentration,
                    sector=sector,
                    date=date,
                    remediation=f"Reduce exposure to {sector} sector"
                ))
        
        # Check leverage
        total_weight = sum(abs(w) for w in portfolio_weights.values())
        if total_weight > self.config.max_leverage:
            violations.append(RiskViolation(
                constraint=RiskConstraint(
                    RiskConstraintType.LEVERAGE_LIMIT,
                    self.config.max_leverage,
                    total_weight
                ),
                severity=RiskSeverity.CRITICAL,
                description=f"Portfolio leverage {total_weight:.3f} exceeds limit {self.config.max_leverage:.3f}",
                violation_amount=total_weight - self.config.max_leverage,
                violation_percentage=(total_weight - self.config.max_leverage) / self.config.max_leverage,
                date=date,
                remediation="Reduce position sizes to stay within leverage limits"
            ))
        
        return violations
    
    def _calculate_volatility_adjusted_positions(
        self,
        alpha_signals: Dict[str, float],
        market_data: pd.DataFrame,
        date: date
    ) -> Dict[str, float]:
        """Calculate volatility-adjusted position sizes."""
        if not alpha_signals:
            return {}
        
        # Get volatilities for each stock
        volatilities = {}
        for symbol in alpha_signals.keys():
            stock_data = market_data[market_data['symbol'] == symbol]
            if not stock_data.empty:
                volatilities[symbol] = stock_data.iloc[0].get('volatility', 0.2)
            else:
                volatilities[symbol] = 0.2  # Default volatility
        
        # Calculate inverse volatility weights
        inv_vol_weights = {}
        total_inv_vol = 0
        
        for symbol, alpha in alpha_signals.items():
            if alpha > 0 and volatilities[symbol] > 0:  # Only positive alpha signals
                inv_vol = 1.0 / volatilities[symbol]
                inv_vol_weights[symbol] = alpha * inv_vol  # Weight by alpha and inverse volatility
                total_inv_vol += inv_vol_weights[symbol]
        
        # Normalize to target volatility
        if total_inv_vol > 0:
            target_vol = self.config.target_portfolio_volatility
            scaling_factor = target_vol / np.sqrt(sum(
                (w / total_inv_vol) ** 2 * volatilities[symbol] ** 2
                for symbol, w in inv_vol_weights.items()
            ))
            
            # Scale and constrain positions
            final_weights = {}
            for symbol, weight in inv_vol_weights.items():
                normalized_weight = weight / total_inv_vol * scaling_factor
                # Apply position limits
                final_weight = min(normalized_weight, self.config.max_single_position)
                if final_weight >= self.config.min_position_size:
                    final_weights[symbol] = final_weight
            
            # Renormalize if needed
            total_weight = sum(final_weights.values())
            if total_weight > 1.0:
                final_weights = {
                    symbol: weight / total_weight
                    for symbol, weight in final_weights.items()
                }
            
            return final_weights
        
        return {}
    
    def _calculate_equal_weight_positions(
        self, alpha_signals: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate equal weight positions for positive alpha signals."""
        positive_signals = {k: v for k, v in alpha_signals.items() if v > 0}
        
        if not positive_signals:
            return {}
        
        equal_weight = 1.0 / len(positive_signals)
        max_weight = min(equal_weight, self.config.max_single_position)
        
        return {symbol: max_weight for symbol in positive_signals.keys()}
    
    def _calculate_risk_parity_positions(
        self,
        alpha_signals: Dict[str, float],
        market_data: pd.DataFrame,
        date: date
    ) -> Dict[str, float]:
        """Calculate risk parity position sizes."""
        # Simplified risk parity - equal risk contribution
        return self._calculate_volatility_adjusted_positions(
            alpha_signals, market_data, date
        )
    
    def _calculate_kelly_positions(
        self,
        alpha_signals: Dict[str, float],
        market_data: pd.DataFrame,
        date: date
    ) -> Dict[str, float]:
        """Calculate Kelly criterion position sizes."""
        positions = {}
        
        for symbol, alpha in alpha_signals.items():
            if alpha <= 0:
                continue
                
            stock_data = market_data[market_data['symbol'] == symbol]
            if stock_data.empty:
                continue
                
            # Estimate expected return and variance
            expected_return = alpha * 0.1  # Scale alpha to expected return
            variance = stock_data.iloc[0].get('volatility', 0.2) ** 2
            
            # Kelly fraction: f = (mu - r) / sigma^2
            # Assuming risk-free rate is 0 for simplicity
            if variance > 0:
                kelly_fraction = expected_return / variance
                # Apply constraints and scale conservatively
                conservative_kelly = kelly_fraction * 0.25  # Use 25% of full Kelly
                position = min(conservative_kelly, self.config.max_single_position)
                
                if position >= self.config.min_position_size:
                    positions[symbol] = position
        
        return positions
    
    def _calculate_portfolio_metrics(
        self,
        portfolio_weights: Dict[str, float],
        market_data: pd.DataFrame,
        date: date
    ) -> Dict[str, float]:
        """Calculate portfolio-level risk metrics."""
        if not portfolio_weights:
            return {}
        
        # Get stock volatilities and correlations
        symbols = list(portfolio_weights.keys())
        volatilities = []
        weights = []
        
        for symbol in symbols:
            stock_data = market_data[market_data['symbol'] == symbol]
            if not stock_data.empty:
                vol = stock_data.iloc[0].get('volatility', 0.2)
                volatilities.append(vol)
                weights.append(portfolio_weights[symbol])
            else:
                volatilities.append(0.2)
                weights.append(portfolio_weights[symbol])
        
        volatilities = np.array(volatilities)
        weights = np.array(weights)
        
        # Simplified correlation matrix (in practice, would use historical correlations)
        n_stocks = len(symbols)
        correlation_matrix = np.full((n_stocks, n_stocks), 0.3)  # 30% average correlation
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Calculate covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        # Portfolio volatility
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Simplified tracking error (would use benchmark covariance in practice)
        tracking_error = portfolio_volatility * 0.6  # Simplified assumption
        
        # Portfolio beta (weighted average)
        portfolio_beta = 0.0
        for symbol, weight in portfolio_weights.items():
            stock_data = market_data[market_data['symbol'] == symbol]
            if not stock_data.empty:
                beta = stock_data.iloc[0].get('beta', 1.0)
                portfolio_beta += weight * beta
        
        return {
            'volatility': portfolio_volatility,
            'tracking_error': tracking_error,
            'beta': portfolio_beta,
            'variance': portfolio_variance
        }
    
    def _calculate_sector_concentrations(
        self,
        portfolio_weights: Dict[str, float],
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate sector concentrations."""
        sector_weights = {}
        
        for symbol, weight in portfolio_weights.items():
            stock_data = market_data[market_data['symbol'] == symbol]
            if not stock_data.empty:
                sector = stock_data.iloc[0].get('sector', 'unknown')
                sector_weights[sector] = sector_weights.get(sector, 0) + abs(weight)
        
        return sector_weights
    
    def _calculate_position_size_score(
        self,
        symbol: str,
        weight: float,
        stock_info: pd.Series,
        violations: List[RiskViolation]
    ) -> float:
        """Calculate position sizing score (0-100)."""
        score = 100.0
        
        # Penalize for violations
        for violation in violations:
            if violation.severity == RiskSeverity.CRITICAL:
                score -= 50
            elif violation.severity == RiskSeverity.HIGH:
                score -= 30
            elif violation.severity == RiskSeverity.MEDIUM:
                score -= 15
            elif violation.severity == RiskSeverity.LOW:
                score -= 5
        
        # Reward for good liquidity
        liquidity_score = stock_info.get('liquidity_score', 0.5)
        score += liquidity_score * 10
        
        # Penalize for high volatility
        volatility = stock_info.get('volatility', 0.2)
        if volatility > 0.3:  # High volatility threshold
            score -= (volatility - 0.3) * 100
        
        return max(0, min(100, score))
    
    def _create_default_constraints(self) -> List[RiskConstraint]:
        """Create default risk constraints."""
        constraints = [
            RiskConstraint(
                RiskConstraintType.SINGLE_STOCK_LIMIT,
                self.config.max_single_position,
                description="Maximum single stock position weight",
                reference="Risk Management Policy Section 3.1"
            ),
            RiskConstraint(
                RiskConstraintType.SECTOR_CONCENTRATION,
                self.config.max_sector_concentration,
                description="Maximum sector concentration",
                reference="Risk Management Policy Section 3.2"
            ),
            RiskConstraint(
                RiskConstraintType.LEVERAGE_LIMIT,
                self.config.max_leverage,
                description="Maximum portfolio leverage",
                reference="Risk Management Policy Section 2.1",
                enforcement_level=RiskSeverity.CRITICAL
            ),
            RiskConstraint(
                RiskConstraintType.VALUE_AT_RISK,
                self.config.max_var_percentage,
                description=f"{self.config.var_confidence_level:.0%} {self.config.var_horizon_days}-day VaR limit",
                reference="Risk Management Policy Section 4.1"
            )
        ]
        
        return constraints
    
    def _initialize_risk_model(self) -> Any:
        """Initialize the Taiwan risk model."""
        # This would initialize the actual risk model
        # For now, return None - to be implemented with real model
        return None
    
    def _get_market_data(self, symbols: List[str], date: date) -> pd.DataFrame:
        """Get market data for symbols."""
        # This would query the database for market data
        # For now, return synthetic data for testing
        data = []
        for symbol in symbols:
            data.append({
                'symbol': symbol,
                'volatility': np.random.uniform(0.15, 0.35),  # 15-35% volatility
                'beta': np.random.uniform(0.5, 1.5),         # 0.5-1.5x beta
                'sector': np.random.choice(['technology', 'financials', 'industrials']),
                'market_cap': np.random.uniform(1e9, 100e9),  # 1B-100B NTD
                'liquidity_score': np.random.uniform(0.2, 0.8)  # 20-80% liquidity
            })
        
        return pd.DataFrame(data)


def create_standard_risk_validator(
    max_position: float = 0.05,
    max_vol: float = 0.15
) -> RiskValidator:
    """Create a standard risk validator for Taiwan markets.
    
    Args:
        max_position: Maximum single position weight
        max_vol: Maximum portfolio volatility
        
    Returns:
        Configured risk validator
    """
    config = RiskConfig(
        max_single_position=max_position,
        max_portfolio_volatility=max_vol,
        position_sizing_method=PositionSizeMethod.VOLATILITY_ADJUSTED,
        target_portfolio_volatility=max_vol * 0.8
    )
    
    return RiskValidator(config)


def create_conservative_risk_validator() -> RiskValidator:
    """Create a conservative risk validator with tight constraints.
    
    Returns:
        Conservative risk validator
    """
    config = RiskConfig(
        max_single_position=0.03,  # 3% max position
        max_sector_concentration=0.15,  # 15% max sector
        max_portfolio_volatility=0.10,  # 10% max volatility
        max_tracking_error=0.05,  # 5% tracking error
        max_leverage=0.95,  # 95% leverage
        position_sizing_method=PositionSizeMethod.RISK_PARITY,
        target_portfolio_volatility=0.08
    )
    
    return RiskValidator(config)