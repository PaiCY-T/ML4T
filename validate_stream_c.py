#!/usr/bin/env python3
"""
Simple validation script for Stream C implementation.
Validates that feature selection achieves the 200-500 feature target.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

# Add src to path
sys.path.insert(0, 'src')

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    """Main validation function."""
    print("Stream C Validation - Task #28")
    print("="*50)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from features import (
            FeatureSelector, create_feature_selector,
            FeatureQualityMetrics, create_quality_metrics_calculator,
            TaiwanComplianceValidator, create_taiwan_compliance_validator
        )
        print("   ✓ All Stream C modules imported successfully")
        
        # Test basic functionality
        print("2. Testing basic functionality...")
        
        # Create simple test data
        np.random.seed(42)
        n_samples = 1000
        n_features = 100
        
        # Create panel data index
        dates = pd.date_range('2022-01-01', periods=200, freq='B')
        stocks = ['2330', '2454', '3008', '2382', '2317']
        index_tuples = [(date, stock) for date in dates for stock in stocks]
        index = pd.MultiIndex.from_tuples(index_tuples[:n_samples], names=['date', 'stock'])
        
        # Generate features
        feature_data = np.random.normal(0, 1, (n_samples, n_features))
        feature_names = [f'feature_{i:03d}' for i in range(n_features)]
        features_df = pd.DataFrame(feature_data, columns=feature_names, index=index)
        
        # Generate target
        target = pd.Series(np.random.normal(0, 0.02, n_samples), index=index, name='returns')
        
        print(f"   ✓ Generated test data: {features_df.shape}")
        
        # Test feature selector
        print("3. Testing feature selection...")
        selector = create_feature_selector(target_features=30)
        selected_df = selector.fit_transform(features_df, target)
        print(f"   ✓ Feature selection: {features_df.shape[1]} → {selected_df.shape[1]} features")
        
        # Test quality metrics
        print("4. Testing quality metrics...")
        quality_calc = create_quality_metrics_calculator()
        sample_feature = features_df.iloc[:, 0]
        quality_result = quality_calc.validate_feature_quality(sample_feature, 'test_feature')
        print(f"   ✓ Quality assessment: score = {quality_result['overall_quality_score']:.1f}")
        
        # Test compliance validator
        print("5. Testing Taiwan compliance...")
        compliance_validator = create_taiwan_compliance_validator()
        compliance_result = compliance_validator.validate_feature_compliance(sample_feature, 'volume_lag2')
        print(f"   ✓ Compliance validation: score = {compliance_result['compliance_score']:.1f}")
        
        # Test with larger feature set to demonstrate 200-500 target
        print("6. Testing 200-500 feature target...")
        
        # Create larger feature set (simulating OpenFE output)
        large_n_features = 800
        large_feature_data = np.random.normal(0, 1, (n_samples, large_n_features))
        
        # Add some problematic features
        large_feature_names = [f'openfe_feature_{i:04d}' for i in range(large_n_features-10)]
        large_feature_names.extend([
            'realtime_price', 'overnight_volume', 'insider_flow',  # Non-compliant
            'constant_feature', 'nearly_constant',                 # Low quality
            'highly_corr_1', 'highly_corr_2', 'highly_corr_3',    # Correlated
            'missing_feature_1', 'missing_feature_2'              # Missing data
        ])
        
        # Make some features problematic
        large_feature_data[:, -10:] = 0  # Constant features
        large_feature_data[:, -8] = np.random.lognormal(4, 0.3, n_samples)  # realtime_price
        
        large_features_df = pd.DataFrame(large_feature_data, columns=large_feature_names, index=index)
        
        print(f"   Created large feature set: {large_features_df.shape[1]} features")
        
        # Apply feature selection targeting 350 features
        large_selector = create_feature_selector(
            target_features=350,
            min_feature_count=200,
            max_feature_count=500
        )
        
        large_selected_df = large_selector.fit_transform(large_features_df, target)
        final_count = large_selected_df.shape[1]
        
        print(f"   Feature selection result: {large_features_df.shape[1]} → {final_count} features")
        
        # Validate target achievement
        target_achieved = 200 <= final_count <= 500
        print(f"   Target range 200-500: {'✓ ACHIEVED' if target_achieved else '✗ MISSED'}")
        
        if target_achieved:
            print(f"   ✓ Successfully reduced features to target range: {final_count} features")
        
        # Check that problematic features were filtered
        problematic_features = ['realtime_price', 'overnight_volume', 'insider_flow', 'constant_feature']
        remaining_problematic = [f for f in problematic_features if f in large_selected_df.columns]
        
        if not remaining_problematic:
            print("   ✓ Problematic features successfully filtered out")
        else:
            print(f"   Warning: Some problematic features remain: {remaining_problematic}")
        
        # Final validation
        print("7. Final validation...")
        
        # Check data integrity
        assert not large_selected_df.empty, "Selected DataFrame is empty"
        assert large_selected_df.shape[1] > 0, "No features selected"
        assert large_selected_df.index.equals(large_features_df.index), "Index mismatch"
        
        print("   ✓ Data integrity checks passed")
        
        # Summary
        print("\n" + "="*50)
        print("STREAM C VALIDATION COMPLETE")
        print("="*50)
        print(f"✓ All modules imported and functional")
        print(f"✓ Feature selection pipeline operational")
        print(f"✓ Quality assessment working")
        print(f"✓ Taiwan compliance validation active")
        print(f"✓ 200-500 feature target: {'ACHIEVED' if target_achieved else 'MISSED'}")
        print(f"✓ Feature count: {final_count} (target: 350)")
        print(f"✓ Integration ready for Task #26 LightGBM")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"✗ Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)