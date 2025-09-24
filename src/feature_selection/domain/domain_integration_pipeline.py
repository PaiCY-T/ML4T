"""
Domain Validation & Integration Pipeline (Stream C).

Comprehensive pipeline integrating all domain validation components:
1. Taiwan market compliance validation
2. Economic intuition scoring
3. Business logic validation
4. Information Coefficient performance testing
5. LightGBM pipeline integration
6. Final feature selection and validation
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from datetime import datetime
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# Import domain validation components
from .taiwan_compliance import TaiwanMarketComplianceValidator, TaiwanMarketConfig, ComplianceLevel
from .economic_intuition import EconomicIntuitionScorer, EconomicIntuitionConfig, IntuitionScore
from .business_logic import BusinessLogicValidator, BusinessLogicConfig, ValidationSeverity
from .ic_performance_tester import ICPerformanceTester, ICTestConfig, PerformanceLevel

# Import LightGBM integration
try:
    from ...models.lightgbm_alpha import LightGBMAlphaModel, ModelConfig
    LIGHTGBM_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM integration not available")
    LightGBMAlphaModel = None
    ModelConfig = None
    LIGHTGBM_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DomainValidationConfig:
    """Configuration for domain validation pipeline."""
    
    # Component configurations
    taiwan_config: Optional[TaiwanMarketConfig] = None
    intuition_config: Optional[EconomicIntuitionConfig] = None
    business_logic_config: Optional[BusinessLogicConfig] = None
    ic_test_config: Optional[ICTestConfig] = None
    
    # Integration parameters
    final_feature_count: int = 100  # Target final feature count
    min_feature_count: int = 20     # Minimum features to maintain
    max_feature_count: int = 200    # Maximum features before final selection
    
    # Scoring weights for final selection
    scoring_weights: Dict[str, float] = field(default_factory=lambda: {
        'compliance_score': 0.25,      # Taiwan compliance weight
        'intuition_score': 0.25,       # Economic intuition weight
        'business_logic_score': 0.20,  # Business logic weight
        'ic_performance_score': 0.30   # IC performance weight (highest)
    })
    
    # Quality thresholds
    min_compliance_level: ComplianceLevel = ComplianceLevel.WARNING
    min_intuition_score: IntuitionScore = IntuitionScore.MODERATE
    max_business_logic_severity: ValidationSeverity = ValidationSeverity.MEDIUM
    min_ic_performance: PerformanceLevel = PerformanceLevel.ACCEPTABLE
    require_ic_significance: bool = True
    
    # LightGBM integration
    integrate_with_lightgbm: bool = True
    lightgbm_validation_split: float = 0.2  # Validation split for final testing
    lightgbm_target_ic: float = 0.05        # Target IC for LightGBM validation
    
    # Output and reporting
    save_intermediate_results: bool = True
    generate_comprehensive_report: bool = True
    output_dir: Optional[str] = None
    
    # Performance and optimization
    parallel_validation: bool = True
    memory_efficient: bool = True
    early_stopping: bool = True


@dataclass
class DomainValidationResults:
    """Results from domain validation pipeline."""
    
    # Input and output features
    input_features: List[str]
    final_selected_features: List[str]
    
    # Component results
    compliance_results: Optional[Dict[str, Any]] = None
    intuition_results: Optional[Dict[str, Any]] = None
    business_logic_results: Optional[Dict[str, Any]] = None
    ic_test_results: Optional[Dict[str, Any]] = None
    
    # Final scoring
    final_scores: Optional[pd.DataFrame] = None
    selection_summary: Optional[Dict[str, Any]] = None
    
    # LightGBM integration results
    lightgbm_integration: Optional[Dict[str, Any]] = None
    
    # Validation statistics
    validation_statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Quality assessment
    passed_features_by_component: Dict[str, List[str]] = field(default_factory=dict)
    failed_features_by_component: Dict[str, List[str]] = field(default_factory=dict)
    
    # Execution metadata
    execution_time: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


class DomainValidationPipeline:
    """
    Comprehensive Domain Validation & Integration Pipeline.
    
    Stream C implementation that brings together all domain validation
    components to perform final feature selection and validation.
    
    Pipeline stages:
    1. Taiwan market compliance validation
    2. Economic intuition scoring
    3. Business logic validation  
    4. Information Coefficient performance testing
    5. Composite scoring and ranking
    6. Final feature selection
    7. LightGBM integration and validation
    8. Comprehensive reporting
    """
    
    def __init__(self, config: Optional[DomainValidationConfig] = None):
        """Initialize domain validation pipeline.
        
        Args:
            config: Configuration for domain validation pipeline
        """
        self.config = config or DomainValidationConfig()
        
        # Initialize component validators
        self.compliance_validator = TaiwanMarketComplianceValidator(
            self.config.taiwan_config or TaiwanMarketConfig()
        )
        
        self.intuition_scorer = EconomicIntuitionScorer(
            self.config.intuition_config or EconomicIntuitionConfig()
        )
        
        self.business_logic_validator = BusinessLogicValidator(
            self.config.business_logic_config or BusinessLogicConfig()
        )
        
        self.ic_tester = ICPerformanceTester(
            self.config.ic_test_config or ICTestConfig()
        )
        
        # Results storage
        self.validation_results: Optional[DomainValidationResults] = None
        
        # Create output directory
        if self.config.output_dir:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("Domain Validation Pipeline initialized")
        logger.info(f"Target features: {self.config.final_feature_count}")
        logger.info(f"LightGBM integration: {'enabled' if self.config.integrate_with_lightgbm and LIGHTGBM_AVAILABLE else 'disabled'}")
    
    def validate_and_select_features(
        self,
        features: List[str],
        feature_data: pd.DataFrame,
        price_data: pd.DataFrame,
        target_returns: Optional[pd.Series] = None,
        market_data: Optional[pd.DataFrame] = None,
        feature_metadata: Optional[Dict[str, Any]] = None,
        dates: Optional[pd.Series] = None
    ) -> DomainValidationResults:
        """
        Run comprehensive domain validation and feature selection.
        
        Args:
            features: List of feature names to validate
            feature_data: Feature data matrix
            price_data: Price data for return calculation
            target_returns: Optional target returns (calculated if not provided)
            market_data: Market data for validation context
            feature_metadata: Feature metadata for validation
            dates: Date series if not in index
            
        Returns:
            Comprehensive validation results
        """
        start_time = datetime.now()
        logger.info(f"Starting domain validation pipeline for {len(features)} features")
        
        # Initialize results
        results = DomainValidationResults(
            input_features=features.copy(),
            final_selected_features=[]
        )
        
        try:
            # Stage 1: Taiwan Market Compliance Validation
            logger.info("Stage 1: Taiwan Market Compliance Validation")
            compliance_results = self.compliance_validator.validate_features(
                features, feature_data, market_data, feature_metadata
            )
            results.compliance_results = compliance_results
            
            # Get compliant features
            compliant_features = self.compliance_validator.get_compliant_features(
                compliance_results, self.config.min_compliance_level
            )
            results.passed_features_by_component['compliance'] = compliant_features
            results.failed_features_by_component['compliance'] = list(set(features) - set(compliant_features))
            
            logger.info(f"Stage 1 complete: {len(compliant_features)}/{len(features)} features passed compliance")
            
            # Stage 2: Economic Intuition Scoring
            logger.info("Stage 2: Economic Intuition Scoring")
            intuition_results = self.intuition_scorer.score_features(
                features, feature_data, market_data, feature_metadata
            )
            results.intuition_results = intuition_results
            
            # Get high-quality features by intuition
            intuitive_features = self.intuition_scorer.get_high_quality_features(
                intuition_results, self.config.min_intuition_score
            )
            results.passed_features_by_component['intuition'] = intuitive_features
            results.failed_features_by_component['intuition'] = list(set(features) - set(intuitive_features))
            
            logger.info(f"Stage 2 complete: {len(intuitive_features)}/{len(features)} features passed intuition scoring")
            
            # Stage 3: Business Logic Validation
            logger.info("Stage 3: Business Logic Validation")
            business_logic_results = self.business_logic_validator.validate_features(
                features, feature_data, market_data, feature_metadata
            )
            results.business_logic_results = business_logic_results
            
            # Get logically valid features
            logical_features = self.business_logic_validator.get_valid_features(
                business_logic_results, self.config.max_business_logic_severity
            )
            results.passed_features_by_component['business_logic'] = logical_features
            results.failed_features_by_component['business_logic'] = list(set(features) - set(logical_features))
            
            logger.info(f"Stage 3 complete: {len(logical_features)}/{len(features)} features passed business logic")
            
            # Stage 4: IC Performance Testing
            logger.info("Stage 4: Information Coefficient Performance Testing")
            
            # Calculate target returns if not provided
            if target_returns is None:
                target_returns = self._calculate_target_returns(price_data)
            
            ic_test_results = self.ic_tester.test_features(
                features, feature_data, price_data, dates, market_data
            )
            results.ic_test_results = ic_test_results
            
            # Get high-performance features
            performant_features = self.ic_tester.get_high_performance_features(
                ic_test_results, self.config.min_ic_performance, self.config.require_ic_significance
            )
            results.passed_features_by_component['ic_performance'] = performant_features
            results.failed_features_by_component['ic_performance'] = list(set(features) - set(performant_features))
            
            logger.info(f"Stage 4 complete: {len(performant_features)}/{len(features)} features passed IC testing")
            
            # Stage 5: Composite Scoring and Final Selection
            logger.info("Stage 5: Composite Scoring and Final Selection")
            final_scores = self._calculate_composite_scores(
                features, 
                compliance_results,
                intuition_results,
                business_logic_results,
                ic_test_results
            )
            results.final_scores = final_scores
            
            # Select final features
            final_selected = self._select_final_features(final_scores)
            results.final_selected_features = final_selected
            
            logger.info(f"Stage 5 complete: {len(final_selected)} final features selected")
            
            # Stage 6: LightGBM Integration (if enabled)
            if self.config.integrate_with_lightgbm and LIGHTGBM_AVAILABLE:
                logger.info("Stage 6: LightGBM Integration and Validation")
                lightgbm_results = self._integrate_with_lightgbm(
                    final_selected, feature_data, target_returns
                )
                results.lightgbm_integration = lightgbm_results
                
                logger.info(f"Stage 6 complete: LightGBM validation IC = {lightgbm_results.get('validation_ic', 'N/A'):.4f}")
            
            # Stage 7: Generate Validation Statistics
            results.validation_statistics = self._generate_validation_statistics(results)
            
            # Stage 8: Save Results (if configured)
            if self.config.save_intermediate_results and self.config.output_dir:
                self._save_results(results)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            results.execution_time = execution_time
            
            self.validation_results = results
            
            logger.info(f"Domain validation pipeline completed in {execution_time:.1f}s")
            logger.info(f"Final selection: {len(results.final_selected_features)} features")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in domain validation pipeline: {e}")
            raise
    
    def _calculate_target_returns(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate target returns from price data."""
        
        # Use 1-day forward returns as default
        if isinstance(price_data, pd.DataFrame):
            # If multiple price columns, use the first one or 'close' if available
            if 'close' in price_data.columns:
                prices = price_data['close']
            else:
                prices = price_data.iloc[:, 0]
        else:
            prices = price_data
        
        # Calculate simple returns
        returns = prices.pct_change(periods=1).shift(-1)  # Forward returns
        
        # Remove extreme values (winsorize)
        returns = returns.clip(lower=returns.quantile(0.01), upper=returns.quantile(0.99))
        
        return returns
    
    def _calculate_composite_scores(
        self,
        features: List[str],
        compliance_results: Dict[str, Any],
        intuition_results: Dict[str, Any],
        business_logic_results: Dict[str, Any],
        ic_test_results: Dict[str, Any]
    ) -> pd.DataFrame:
        """Calculate composite scores for final feature selection."""
        
        logger.debug("Calculating composite scores")
        
        score_data = []
        
        for feature in features:
            # Compliance score (0-1)
            compliance_score = self.compliance_validator.get_compliance_score(
                {feature: compliance_results.get(feature, [])}
            ).get(feature, 0.0)
            
            # Intuition score (0-1) 
            if feature in intuition_results:
                intuition_result = intuition_results[feature]
                intuition_score = intuition_result.intuition_score.value / 5.0  # Normalize to 0-1
                intuition_confidence = intuition_result.confidence
                # Combine score and confidence
                intuition_score = intuition_score * intuition_confidence
            else:
                intuition_score = 0.0
            
            # Business logic score (0-1)
            business_scores = self.business_logic_validator.get_validation_scores(
                {feature: business_logic_results.get(feature, [])}
            )
            business_logic_score = business_scores.get(feature, 0.0)
            
            # IC performance score (0-1)
            if feature in ic_test_results:
                ic_result = ic_test_results[feature]
                
                # Base IC score (normalized absolute IC)
                abs_ic = abs(ic_result.ic_value)
                ic_score = min(abs_ic / 0.10, 1.0)  # Normalize to 0.1 IC = 1.0 score
                
                # Adjust for performance level
                performance_multipliers = {
                    PerformanceLevel.EXCELLENT: 1.0,
                    PerformanceLevel.GOOD: 0.8,
                    PerformanceLevel.ACCEPTABLE: 0.6,
                    PerformanceLevel.WEAK: 0.4,
                    PerformanceLevel.POOR: 0.1
                }
                
                ic_score *= performance_multipliers.get(ic_result.performance_level, 0.1)
                
                # Adjust for significance
                if not ic_result.is_significant:
                    ic_score *= 0.5
                
                # Adjust for stability (if available)
                if ic_result.cv_stability is not None:
                    ic_score *= (0.5 + 0.5 * ic_result.cv_stability)  # Scale from 0.5-1.0
                
                ic_performance_score = ic_score
            else:
                ic_performance_score = 0.0
            
            # Calculate composite score
            composite_score = (
                self.config.scoring_weights['compliance_score'] * compliance_score +
                self.config.scoring_weights['intuition_score'] * intuition_score +
                self.config.scoring_weights['business_logic_score'] * business_logic_score +
                self.config.scoring_weights['ic_performance_score'] * ic_performance_score
            )
            
            score_data.append({
                'feature': feature,
                'compliance_score': compliance_score,
                'intuition_score': intuition_score,
                'business_logic_score': business_logic_score,
                'ic_performance_score': ic_performance_score,
                'composite_score': composite_score,
                # Additional metrics
                'ic_value': ic_test_results.get(feature, type('', (), {'ic_value': 0.0})).ic_value if feature in ic_test_results else 0.0,
                'is_significant': ic_test_results.get(feature, type('', (), {'is_significant': False})).is_significant if feature in ic_test_results else False,
                'performance_level': ic_test_results.get(feature, type('', (), {'performance_level': PerformanceLevel.POOR})).performance_level.value if feature in ic_test_results else PerformanceLevel.POOR.value
            })
        
        scores_df = pd.DataFrame(score_data)
        scores_df = scores_df.sort_values('composite_score', ascending=False)
        
        logger.debug(f"Composite scores calculated for {len(scores_df)} features")
        
        return scores_df
    
    def _select_final_features(self, scores_df: pd.DataFrame) -> List[str]:
        """Select final features based on composite scores."""
        
        if scores_df.empty:
            logger.warning("No features available for selection")
            return []
        
        # Apply minimum thresholds
        filtered_scores = scores_df[
            (scores_df['composite_score'] > 0.1) &  # Minimum composite score
            (scores_df['ic_performance_score'] > 0.0)  # Must have some IC performance
        ].copy()
        
        if len(filtered_scores) < self.config.min_feature_count:
            logger.warning(f"Only {len(filtered_scores)} features meet minimum criteria")
            # Take top features from original list
            filtered_scores = scores_df.head(self.config.min_feature_count)
        
        # Select top features up to target count
        target_count = min(self.config.final_feature_count, len(filtered_scores))
        final_features = filtered_scores.head(target_count)['feature'].tolist()
        
        logger.info(f"Selected {len(final_features)} final features")
        
        # Log top 10 features with scores
        logger.info("Top 10 selected features:")
        for i, (_, row) in enumerate(final_features[:10]):
            if isinstance(row, str):  # If final_features is just feature names
                feature = row
                score_row = scores_df[scores_df['feature'] == feature].iloc[0]
            else:
                feature = row['feature'] if 'feature' in row else str(row)
                score_row = row
            
            logger.info(f"  {i+1:2d}. {feature} (score: {score_row.get('composite_score', 0):.3f})")
        
        return final_features if isinstance(final_features[0], str) else [row['feature'] for row in final_features]
    
    def _integrate_with_lightgbm(
        self,
        selected_features: List[str],
        feature_data: pd.DataFrame,
        target_returns: pd.Series
    ) -> Dict[str, Any]:
        """Integrate with LightGBM pipeline for final validation."""
        
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available for integration")
            return {'integration_successful': False, 'error': 'LightGBM not available'}
        
        try:
            logger.debug("Starting LightGBM integration")
            
            # Prepare data
            X = feature_data[selected_features].copy()
            y = target_returns.copy()
            
            # Align data and remove NaN
            common_index = X.index.intersection(y.index)
            X = X.reindex(common_index).dropna()
            y = y.reindex(X.index)
            
            # Remove any remaining NaN
            valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 100:
                logger.warning(f"Insufficient data for LightGBM integration: {len(X)} samples")
                return {
                    'integration_successful': False,
                    'error': f'Insufficient data: {len(X)} samples'
                }
            
            # Train-validation split
            split_idx = int(len(X) * (1 - self.config.lightgbm_validation_split))
            
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Initialize LightGBM model
            lgb_config = ModelConfig()
            lgb_model = LightGBMAlphaModel(lgb_config)
            
            # Train model
            logger.debug("Training LightGBM model")
            training_stats = lgb_model.train(X_train, y_train)
            
            # Validate model
            logger.debug("Validating LightGBM model")
            y_pred = lgb_model.predict(X_val)
            
            # Calculate validation IC
            if len(y_pred) == len(y_val):
                validation_ic, ic_p_value = stats.spearmanr(y_pred, y_val)
                validation_ic = validation_ic if not np.isnan(validation_ic) else 0.0
                ic_p_value = ic_p_value if not np.isnan(ic_p_value) else 1.0
            else:
                validation_ic = ic_p_value = 0.0
            
            # Check if validation meets target
            meets_target = abs(validation_ic) >= self.config.lightgbm_target_ic
            
            # Get feature importance
            feature_importance = lgb_model.get_feature_importance()
            
            integration_results = {
                'integration_successful': True,
                'training_stats': training_stats,
                'validation_ic': validation_ic,
                'validation_p_value': ic_p_value,
                'meets_ic_target': meets_target,
                'ic_target': self.config.lightgbm_target_ic,
                'feature_importance': feature_importance.to_dict('records') if feature_importance is not None else [],
                'validation_samples': len(X_val),
                'selected_feature_count': len(selected_features)
            }
            
            logger.debug(f"LightGBM integration complete: IC = {validation_ic:.4f}")
            
            return integration_results
            
        except Exception as e:
            logger.error(f"LightGBM integration failed: {e}")
            return {
                'integration_successful': False,
                'error': str(e)
            }
    
    def _generate_validation_statistics(self, results: DomainValidationResults) -> Dict[str, Any]:
        """Generate comprehensive validation statistics."""
        
        stats = {
            'input_feature_count': len(results.input_features),
            'final_feature_count': len(results.final_selected_features),
            'reduction_ratio': 1 - (len(results.final_selected_features) / len(results.input_features)),
            'execution_time': results.execution_time,
            'timestamp': results.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Component pass rates
        for component, passed_features in results.passed_features_by_component.items():
            pass_rate = len(passed_features) / len(results.input_features)
            stats[f'{component}_pass_rate'] = pass_rate
            stats[f'{component}_passed_count'] = len(passed_features)
        
        # Final score statistics
        if results.final_scores is not None:
            stats['average_composite_score'] = results.final_scores['composite_score'].mean()
            stats['median_composite_score'] = results.final_scores['composite_score'].median()
            stats['top_10_avg_score'] = results.final_scores.head(10)['composite_score'].mean()
            
            # Score component averages for selected features
            selected_scores = results.final_scores[
                results.final_scores['feature'].isin(results.final_selected_features)
            ]
            
            if not selected_scores.empty:
                for component in ['compliance_score', 'intuition_score', 'business_logic_score', 'ic_performance_score']:
                    stats[f'selected_{component}_avg'] = selected_scores[component].mean()
        
        # LightGBM integration statistics
        if results.lightgbm_integration:
            lgb_stats = results.lightgbm_integration
            stats['lightgbm_integration_successful'] = lgb_stats.get('integration_successful', False)
            stats['lightgbm_validation_ic'] = lgb_stats.get('validation_ic', 0.0)
            stats['lightgbm_meets_target'] = lgb_stats.get('meets_ic_target', False)
        
        return stats
    
    def _save_results(self, results: DomainValidationResults) -> None:
        """Save validation results to files."""
        
        output_dir = Path(self.config.output_dir)
        timestamp = results.timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Save final selected features
        features_file = output_dir / f"domain_selected_features_{timestamp}.json"
        with open(features_file, 'w') as f:
            json.dump({
                'selected_features': results.final_selected_features,
                'input_feature_count': len(results.input_features),
                'final_feature_count': len(results.final_selected_features),
                'selection_config': {
                    'final_feature_count': self.config.final_feature_count,
                    'scoring_weights': self.config.scoring_weights,
                    'min_compliance_level': self.config.min_compliance_level.value,
                    'min_intuition_score': self.config.min_intuition_score.value,
                    'min_ic_performance': self.config.min_ic_performance.value
                },
                'timestamp': timestamp
            }, f, indent=2)
        
        # Save final scores
        if results.final_scores is not None:
            scores_file = output_dir / f"domain_feature_scores_{timestamp}.csv"
            results.final_scores.to_csv(scores_file, index=False)
        
        # Save validation statistics
        stats_file = output_dir / f"domain_validation_stats_{timestamp}.json"
        with open(stats_file, 'w') as f:
            json.dump(results.validation_statistics, f, indent=2)
        
        # Save comprehensive results (pickled)
        results_file = output_dir / f"domain_validation_results_{timestamp}.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Domain validation results saved to {output_dir}")
    
    def get_selected_features(self) -> List[str]:
        """Get final selected features."""
        if self.validation_results is None:
            raise ValueError("No validation results available. Run validate_and_select_features() first.")
        
        return self.validation_results.final_selected_features.copy()
    
    def get_feature_scores(self) -> pd.DataFrame:
        """Get feature scores DataFrame."""
        if self.validation_results is None or self.validation_results.final_scores is None:
            raise ValueError("No feature scores available. Run validate_and_select_features() first.")
        
        return self.validation_results.final_scores.copy()
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary statistics."""
        if self.validation_results is None:
            raise ValueError("No validation results available. Run validate_and_select_features() first.")
        
        return self.validation_results.validation_statistics.copy()
    
    def generate_comprehensive_report(
        self,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Generate comprehensive validation and selection report."""
        
        if self.validation_results is None:
            raise ValueError("No validation results available.")
        
        results = self.validation_results
        
        # Compile comprehensive report
        report_data = []
        
        for feature in results.input_features:
            row = {'Feature': feature}
            
            # Selection status
            row['Selected'] = feature in results.final_selected_features
            
            # Component results
            for component, passed_features in results.passed_features_by_component.items():
                row[f'{component.title()}_Passed'] = feature in passed_features
            
            # Final scores (if available)
            if results.final_scores is not None:
                score_row = results.final_scores[results.final_scores['feature'] == feature]
                if not score_row.empty:
                    score_data = score_row.iloc[0]
                    row['Composite_Score'] = score_data['composite_score']
                    row['Compliance_Score'] = score_data['compliance_score']
                    row['Intuition_Score'] = score_data['intuition_score']
                    row['Business_Logic_Score'] = score_data['business_logic_score']
                    row['IC_Performance_Score'] = score_data['ic_performance_score']
                    row['IC_Value'] = score_data['ic_value']
                    row['IC_Significant'] = score_data['is_significant']
                    row['Performance_Level'] = score_data['performance_level']
            
            report_data.append(row)
        
        report_df = pd.DataFrame(report_data)
        
        # Sort by selection status and composite score
        if 'Composite_Score' in report_df.columns:
            report_df = report_df.sort_values(['Selected', 'Composite_Score'], ascending=[False, False])
        else:
            report_df = report_df.sort_values('Selected', ascending=False)
        
        if output_path:
            report_df.to_csv(output_path, index=False)
            logger.info(f"Comprehensive validation report saved to {output_path}")
        
        return report_df
    
    def plot_validation_summary(self, save_path: Optional[str] = None) -> None:
        """Plot validation summary dashboard."""
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if self.validation_results is None:
                raise ValueError("No validation results available")
            
            results = self.validation_results
            
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            
            # Plot 1: Component pass rates
            components = list(results.passed_features_by_component.keys())
            pass_rates = [len(results.passed_features_by_component[comp]) / len(results.input_features) 
                         for comp in components]
            
            axes[0, 0].bar(components, pass_rates, color='lightblue')
            axes[0, 0].set_title('Component Pass Rates')
            axes[0, 0].set_ylabel('Pass Rate')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Plot 2: Feature reduction funnel
            funnel_data = [
                ('Input', len(results.input_features)),
                ('Compliance', len(results.passed_features_by_component.get('compliance', []))),
                ('Intuition', len(results.passed_features_by_component.get('intuition', []))),
                ('Logic', len(results.passed_features_by_component.get('business_logic', []))),
                ('IC Performance', len(results.passed_features_by_component.get('ic_performance', []))),
                ('Final Selected', len(results.final_selected_features))
            ]
            
            stages, counts = zip(*funnel_data)
            axes[0, 1].plot(stages, counts, marker='o', linewidth=2)
            axes[0, 1].set_title('Feature Selection Funnel')
            axes[0, 1].set_ylabel('Feature Count')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Plot 3: Final score distribution
            if results.final_scores is not None:
                selected_scores = results.final_scores[
                    results.final_scores['feature'].isin(results.final_selected_features)
                ]['composite_score']
                
                axes[0, 2].hist(selected_scores, bins=20, color='green', alpha=0.7, label='Selected')
                
                if len(results.final_scores) > len(results.final_selected_features):
                    not_selected_scores = results.final_scores[
                        ~results.final_scores['feature'].isin(results.final_selected_features)
                    ]['composite_score']
                    
                    axes[0, 2].hist(not_selected_scores, bins=20, color='red', alpha=0.5, label='Not Selected')
                
                axes[0, 2].set_title('Score Distribution')
                axes[0, 2].set_xlabel('Composite Score')
                axes[0, 2].legend()
            
            # Plot 4: Score components for selected features
            if results.final_scores is not None:
                selected_scores = results.final_scores[
                    results.final_scores['feature'].isin(results.final_selected_features)
                ]
                
                component_cols = ['compliance_score', 'intuition_score', 'business_logic_score', 'ic_performance_score']
                component_means = [selected_scores[col].mean() for col in component_cols]
                component_labels = [col.replace('_score', '').title() for col in component_cols]
                
                axes[1, 0].bar(component_labels, component_means, color='orange')
                axes[1, 0].set_title('Average Score Components (Selected Features)')
                axes[1, 0].set_ylabel('Average Score')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Plot 5: IC performance levels
            if results.ic_test_results:
                performance_levels = {}
                for feature, ic_result in results.ic_test_results.items():
                    level = ic_result.performance_level.value
                    performance_levels[level] = performance_levels.get(level, 0) + 1
                
                axes[1, 1].pie(performance_levels.values(), labels=performance_levels.keys(), autopct='%1.1f%%')
                axes[1, 1].set_title('IC Performance Level Distribution')
            
            # Plot 6: Validation statistics
            stats_text = f"""
            Validation Summary:
            Input Features: {len(results.input_features)}
            Final Selected: {len(results.final_selected_features)}
            Reduction Ratio: {results.validation_statistics.get('reduction_ratio', 0):.1%}
            Execution Time: {results.execution_time:.1f}s
            
            LightGBM Integration:
            Successful: {results.validation_statistics.get('lightgbm_integration_successful', False)}
            Validation IC: {results.validation_statistics.get('lightgbm_validation_ic', 0):.4f}
            Meets Target: {results.validation_statistics.get('lightgbm_meets_target', False)}
            """
            
            axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes, 
                            fontsize=10, verticalalignment='center',
                            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            axes[1, 2].set_title('Validation Statistics')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Validation summary plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for plotting")
        except Exception as e:
            logger.error(f"Error creating validation summary plot: {e}")
            
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and configuration."""
        
        status = {
            "configuration": {
                "final_feature_count": self.config.final_feature_count,
                "scoring_weights": self.config.scoring_weights,
                "lightgbm_integration": self.config.integrate_with_lightgbm and LIGHTGBM_AVAILABLE,
                "output_dir": self.config.output_dir
            },
            "components": {
                "taiwan_compliance": True,
                "economic_intuition": True,
                "business_logic": True,
                "ic_performance": True,
                "lightgbm_integration": LIGHTGBM_AVAILABLE
            }
        }
        
        # Add results status
        if self.validation_results is not None:
            status["last_run"] = {
                "timestamp": self.validation_results.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "input_features": len(self.validation_results.input_features),
                "final_features": len(self.validation_results.final_selected_features),
                "execution_time": self.validation_results.execution_time
            }
        else:
            status["last_run"] = None
        
        return status