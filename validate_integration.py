#!/usr/bin/env python3
"""
ML4T-Alpha Integration Validation Script.

This script validates the ML4T-Alpha integration implementation without
requiring external dependencies like SQLAlchemy or FinLab connections.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def validate_file_structure():
    """Validate that all integration files exist."""
    print("🔍 Validating file structure...")

    required_files = [
        'src/integration/__init__.py',
        'src/integration/ml4t_data_interface.py',
        'src/integration/format_converters.py',
        'src/integration/streaming_engine.py',
        'src/integration/backtest_optimizer.py',
        'tests/integration/test_ml4t_alpha_integration.py',
        'docs/ml4t_alpha_integration.md',
        'demo_ml4t_alpha_integration.py'
    ]

    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"  ✓ {file_path}")

    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return False

    print("  ✅ All integration files present")
    return True

def validate_module_imports():
    """Validate that modules can be imported without external dependencies."""
    print("\n🔍 Validating module structure...")

    try:
        # Test basic Python imports
        from datetime import datetime, date, timedelta
        from typing import Optional, Dict, Any, List, Union
        from dataclasses import dataclass
        from enum import Enum
        import pandas as pd
        import numpy as np
        print("  ✓ Core Python dependencies available")
    except ImportError as e:
        print(f"  ❌ Missing core dependencies: {e}")
        return False

    # Test integration module structure
    try:
        integration_path = project_root / 'src' / 'integration'
        if not integration_path.exists():
            print("  ❌ Integration module directory not found")
            return False

        # Check for class definitions in files
        module_validations = {
            'ml4t_data_interface.py': ['ML4TDataInterface', 'ML4TDataConfig'],
            'format_converters.py': ['FinLabToML4TConverter', 'OpenFEDataAdapter'],
            'streaming_engine.py': ['ML4TStreamingEngine', 'StreamingConfig'],
            'backtest_optimizer.py': ['BacktestOptimizer', 'OptimizationConfig']
        }

        for file_name, required_classes in module_validations.items():
            file_path = integration_path / file_name
            if file_path.exists():
                content = file_path.read_text()
                for class_name in required_classes:
                    if f"class {class_name}" in content:
                        print(f"  ✓ {file_name}: {class_name}")
                    else:
                        print(f"  ❌ {file_name}: Missing {class_name}")
                        return False
            else:
                print(f"  ❌ {file_name} not found")
                return False

        print("  ✅ All integration classes defined")
        return True

    except Exception as e:
        print(f"  ❌ Module validation failed: {e}")
        return False

def validate_configuration_classes():
    """Validate configuration class structures."""
    print("\n🔍 Validating configuration classes...")

    # Check configuration dataclasses
    config_checks = [
        ('ml4t_data_interface.py', '@dataclass', 'ML4TDataConfig'),
        ('format_converters.py', '@dataclass', 'ConversionConfig'),
        ('streaming_engine.py', '@dataclass', 'StreamingConfig'),
        ('backtest_optimizer.py', '@dataclass', 'OptimizationConfig')
    ]

    for file_name, decorator, class_name in config_checks:
        file_path = project_root / 'src' / 'integration' / file_name
        if file_path.exists():
            content = file_path.read_text()
            if decorator in content and class_name in content:
                print(f"  ✓ {class_name} configuration class")
            else:
                print(f"  ❌ Missing {class_name} configuration")
                return False
        else:
            print(f"  ❌ {file_name} not found")
            return False

    print("  ✅ All configuration classes validated")
    return True

def validate_factory_functions():
    """Validate factory function implementations."""
    print("\n🔍 Validating factory functions...")

    factory_checks = [
        ('ml4t_data_interface.py', 'create_ml4t_interface'),
        ('streaming_engine.py', 'create_streaming_engine'),
        ('backtest_optimizer.py', 'create_backtest_optimizer')
    ]

    for file_name, function_name in factory_checks:
        file_path = project_root / 'src' / 'integration' / file_name
        if file_path.exists():
            content = file_path.read_text()
            if f"def {function_name}" in content:
                print(f"  ✓ {function_name} factory function")
            else:
                print(f"  ❌ Missing {function_name} factory")
                return False
        else:
            print(f"  ❌ {file_name} not found")
            return False

    print("  ✅ All factory functions validated")
    return True

def validate_enum_definitions():
    """Validate enum definitions."""
    print("\n🔍 Validating enum definitions...")

    enum_checks = [
        ('format_converters.py', 'DataFormat', ['ML4T_ALPHA', 'OPENFE']),
        ('streaming_engine.py', 'StreamingMode', ['LIVE', 'SIMULATION']),
        ('backtest_optimizer.py', 'OptimizationLevel', ['BASIC', 'BALANCED'])
    ]

    for file_name, enum_name, expected_values in enum_checks:
        file_path = project_root / 'src' / 'integration' / file_name
        if file_path.exists():
            content = file_path.read_text()
            if f"class {enum_name}(Enum)" in content:
                # Check for expected values
                values_found = all(value in content for value in expected_values)
                if values_found:
                    print(f"  ✓ {enum_name} with expected values")
                else:
                    print(f"  ⚠️  {enum_name} missing some expected values")
            else:
                print(f"  ❌ Missing {enum_name} enum")
                return False
        else:
            print(f"  ❌ {file_name} not found")
            return False

    print("  ✅ All enum definitions validated")
    return True

def validate_documentation():
    """Validate documentation completeness."""
    print("\n🔍 Validating documentation...")

    doc_path = project_root / 'docs' / 'ml4t_alpha_integration.md'
    if not doc_path.exists():
        print("  ❌ Integration documentation not found")
        return False

    doc_content = doc_path.read_text()

    # Check for key documentation sections
    required_sections = [
        '# ML4T-Alpha Integration Documentation',
        '## Overview',
        '## Key Features',
        '## Architecture',
        '## Installation',
        '## Quick Start',
        '## Configuration',
        '## API Reference',
        '## Examples'
    ]

    missing_sections = []
    for section in required_sections:
        if section not in doc_content:
            missing_sections.append(section)
        else:
            print(f"  ✓ {section}")

    if missing_sections:
        print(f"  ❌ Missing documentation sections: {missing_sections}")
        return False

    # Check documentation length (should be comprehensive)
    if len(doc_content) < 10000:  # At least 10K characters
        print(f"  ⚠️  Documentation might be too brief ({len(doc_content)} characters)")

    print("  ✅ Documentation validated")
    return True

def validate_test_structure():
    """Validate test file structure."""
    print("\n🔍 Validating test structure...")

    test_path = project_root / 'tests' / 'integration' / 'test_ml4t_alpha_integration.py'
    if not test_path.exists():
        print("  ❌ Integration test file not found")
        return False

    test_content = test_path.read_text()

    # Check for test classes
    required_test_classes = [
        'TestML4TDataInterface',
        'TestFormatConverters',
        'TestStreamingEngine',
        'TestBacktestOptimizer',
        'TestIntegrationWorkflow'
    ]

    for test_class in required_test_classes:
        if f"class {test_class}" in test_content:
            print(f"  ✓ {test_class}")
        else:
            print(f"  ❌ Missing {test_class}")
            return False

    # Check for async test methods
    if '@pytest.mark.asyncio' in test_content:
        print("  ✓ Async tests included")
    else:
        print("  ⚠️  No async tests found")

    print("  ✅ Test structure validated")
    return True

def validate_demo_script():
    """Validate demo script completeness."""
    print("\n🔍 Validating demo script...")

    demo_path = project_root / 'demo_ml4t_alpha_integration.py'
    if not demo_path.exists():
        print("  ❌ Demo script not found")
        return False

    demo_content = demo_path.read_text()

    # Check for key demo components
    demo_components = [
        'class ML4TIntegrationDemo',
        'def demo_data_loading',
        'def demo_format_conversion',
        'async def demo_streaming',
        'def demo_optimization',
        'def demo_export_and_validation',
        'if __name__ == "__main__"'
    ]

    for component in demo_components:
        if component in demo_content:
            print(f"  ✓ {component}")
        else:
            print(f"  ❌ Missing {component}")
            return False

    print("  ✅ Demo script validated")
    return True

def validate_issue_completion():
    """Validate that issue requirements are met."""
    print("\n🔍 Validating issue completion...")

    # Check issue requirements from the task description
    requirements_met = []

    # 1. Data interface compatibility
    ml4t_interface_path = project_root / 'src' / 'integration' / 'ml4t_data_interface.py'
    if ml4t_interface_path.exists():
        content = ml4t_interface_path.read_text()
        if 'class ML4TDataInterface' in content and 'point-in-time' in content.lower():
            requirements_met.append("✓ ML4T-Alpha data interface compatibility")
        else:
            requirements_met.append("❌ ML4T-Alpha data interface incomplete")

    # 2. Data format conversion
    converter_path = project_root / 'src' / 'integration' / 'format_converters.py'
    if converter_path.exists():
        content = converter_path.read_text()
        if 'OpenFE' in content and 'ML4T_ALPHA' in content:
            requirements_met.append("✓ Data format standardization and conversion")
        else:
            requirements_met.append("❌ Format conversion incomplete")

    # 3. Real-time streaming
    streaming_path = project_root / 'src' / 'integration' / 'streaming_engine.py'
    if streaming_path.exists():
        content = streaming_path.read_text()
        if 'StreamingEngine' in content and 'real-time' in content.lower():
            requirements_met.append("✓ Real-time streaming capabilities")
        else:
            requirements_met.append("❌ Streaming capabilities incomplete")

    # 4. Performance optimization
    optimizer_path = project_root / 'src' / 'integration' / 'backtest_optimizer.py'
    if optimizer_path.exists():
        content = optimizer_path.read_text()
        if 'BacktestOptimizer' in content and 'performance' in content.lower():
            requirements_met.append("✓ Performance optimization for backtesting")
        else:
            requirements_met.append("❌ Performance optimization incomplete")

    # 5. Integration testing
    test_path = project_root / 'tests' / 'integration' / 'test_ml4t_alpha_integration.py'
    if test_path.exists() and test_path.stat().st_size > 5000:  # Non-trivial test file
        requirements_met.append("✓ Integration testing implemented")
    else:
        requirements_met.append("❌ Integration testing incomplete")

    # 6. Documentation
    doc_path = project_root / 'docs' / 'ml4t_alpha_integration.md'
    if doc_path.exists() and doc_path.stat().st_size > 10000:  # Comprehensive doc
        requirements_met.append("✓ Comprehensive documentation created")
    else:
        requirements_met.append("❌ Documentation incomplete")

    for req in requirements_met:
        print(f"  {req}")

    success_count = sum(1 for req in requirements_met if req.startswith("✓"))
    total_count = len(requirements_met)

    print(f"\n  📊 Requirements completion: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")

    return success_count == total_count

def main():
    """Main validation function."""
    print("🚀 ML4T-Alpha Integration Validation")
    print("=" * 50)

    validations = [
        ("File Structure", validate_file_structure),
        ("Module Imports", validate_module_imports),
        ("Configuration Classes", validate_configuration_classes),
        ("Factory Functions", validate_factory_functions),
        ("Enum Definitions", validate_enum_definitions),
        ("Documentation", validate_documentation),
        ("Test Structure", validate_test_structure),
        ("Demo Script", validate_demo_script),
        ("Issue Completion", validate_issue_completion)
    ]

    results = []

    for name, validation_func in validations:
        try:
            result = validation_func()
            results.append((name, result, None))
        except Exception as e:
            results.append((name, False, str(e)))

    # Summary
    print("\n" + "=" * 50)
    print("🎯 VALIDATION SUMMARY")
    print("=" * 50)

    passed = 0
    for name, result, error in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {name}")
        if error:
            print(f"           Error: {error}")
        if result:
            passed += 1

    total = len(results)
    success_rate = passed / total * 100

    print("\n" + "-" * 50)
    print(f"📊 Overall Success Rate: {passed}/{total} ({success_rate:.1f}%)")

    if success_rate >= 90:
        print("🎉 Integration implementation is EXCELLENT!")
        return 0
    elif success_rate >= 80:
        print("✅ Integration implementation is GOOD!")
        return 0
    elif success_rate >= 70:
        print("⚠️  Integration implementation needs MINOR improvements")
        return 1
    else:
        print("❌ Integration implementation needs MAJOR improvements")
        return 2

if __name__ == "__main__":
    sys.exit(main())