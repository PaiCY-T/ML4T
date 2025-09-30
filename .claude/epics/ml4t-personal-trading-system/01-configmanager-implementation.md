# Issue #01: ConfigManager Implementation (T0.1)

**Issue Type**: Infrastructure Foundation
**Phase**: 1 - Foundation & Proof
**Priority**: P0 - Critical Path
**Effort**: 40 hours
**Status**: 📋 Ready for Development

---

## 🎯 Objective

Implement hardware-aware configuration system with YAML-driven settings, automatic environment detection, and optimized parameters for laptop/cloud/desktop tiers.

## 📋 Requirements

### Core Functionality
- [x] **Hardware Detection**: Automatic classification (laptop/medium/high_performance)
- [x] **YAML Configuration**: Flexible configuration templates for each tier
- [x] **Environment Optimization**: Hardware-specific processing parameters
- [x] **Runtime Override**: Dynamic configuration adjustment capabilities

### Technical Specifications
```python
class ConfigManager:
    """YAML-driven configuration with hardware-aware optimization"""

    def __init__(self, config_path=None):
        self.hardware_tier = self._detect_hardware_tier()
        self.config = self._load_config(config_path)

    def _detect_hardware_tier(self):
        """Environment-aware hardware detection"""
        # >30GB = high_performance, >15GB = medium_performance, else = laptop

    def get_processing_config(self):
        """Get hardware-optimized processing parameters"""
        # Return tier-specific: max_symbols, chunk_size, memory_limit_gb, parallel_jobs
```

### Configuration Templates Required
- **config_laptop.yaml**: 200 symbols, 4GB limit, 2 parallel jobs
- **config_cloud.yaml**: 1307 symbols, 64GB limit, 8 parallel jobs
- **config_desktop.yaml**: 1307 symbols, 32GB limit, 6 parallel jobs

## 🔧 Implementation Plan

### Day 1-2: Core Implementation
**Monday-Tuesday (8h each)**
```python
# File: src/infrastructure/config_manager.py

import yaml
import psutil
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    """Hardware-aware configuration management system"""

    def __init__(self, config_path: Optional[str] = None):
        self.hardware_tier = self._detect_hardware_tier()
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()

    def _detect_hardware_tier(self) -> str:
        """Detect hardware tier based on available memory"""
        total_memory = psutil.virtual_memory().total

        if total_memory > 30_000_000_000:  # >30GB
            return 'high_performance'
        elif total_memory > 15_000_000_000:  # >15GB
            return 'medium_performance'
        else:
            return 'laptop'

    def _get_default_config_path(self) -> str:
        """Get default config path based on hardware tier"""
        return f"config/config_{self.hardware_tier}.yaml"

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def get_processing_config(self) -> Dict[str, Any]:
        """Get hardware-optimized processing parameters"""
        return self.config.get('hardware', {})

    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading-specific configuration"""
        return self.config.get('trading', {})

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return self.config.get('database', {})

    def get_sampling_config(self) -> Dict[str, Any]:
        """Get sampling configuration"""
        return self.config.get('sampling', {})
```

### Day 3: Configuration Templates
**Wednesday (8h)**
Create three YAML configuration files:

```yaml
# config/config_laptop.yaml
hardware:
  tier: laptop
  max_symbols: 200
  memory_limit_gb: 4
  parallel_jobs: 2
  chunk_size: 50

trading:
  rebalance_frequency: monthly  # Focus on monthly for laptop
  position_size_limit: 0.05    # 5% max position
  transaction_cost: 0.003      # 0.3% Taiwan market

database:
  chunk_size: 5000
  use_streaming: true
  cache_enabled: true

sampling:
  strategy: stratified
  sample_size: 200
  strata: [market_cap, sector]
  liquidity_filter: true
```

### Day 4: Testing & Validation
**Thursday (8h)**
- Unit tests for hardware detection accuracy
- Configuration loading validation
- Parameter optimization verification
- Cross-platform compatibility testing

### Day 5: Integration & Documentation
**Friday (8h)**
- Integration with existing data pipeline
- Performance benchmarking
- Documentation and usage examples
- Buffer time for debugging

## ✅ Acceptance Criteria

### Functional Requirements
- [ ] **Hardware Detection**: Correctly identifies laptop/medium/high_performance tiers
- [ ] **Configuration Loading**: Successfully loads YAML files without errors
- [ ] **Parameter Optimization**: Returns appropriate settings for each hardware tier
- [ ] **Error Handling**: Graceful handling of missing/invalid configuration files
- [ ] **Override Capability**: Runtime configuration adjustments supported

### Performance Requirements
- [ ] **Initialization Speed**: <100ms configuration loading time
- [ ] **Memory Footprint**: <10MB configuration system overhead
- [ ] **Accuracy**: 95%+ hardware detection accuracy across test environments

### Quality Requirements
- [ ] **Unit Test Coverage**: >90% code coverage
- [ ] **Documentation**: Complete API documentation and usage examples
- [ ] **Cross-Platform**: Works on Windows/Linux/macOS
- [ ] **Validation**: Configuration schema validation implemented

## 🧪 Testing Strategy

### Unit Tests
```python
def test_hardware_detection():
    """Test hardware tier detection accuracy"""

def test_config_loading():
    """Test YAML configuration loading"""

def test_parameter_optimization():
    """Test hardware-specific parameter selection"""

def test_error_handling():
    """Test error handling for invalid configurations"""
```

### Integration Tests
- Test with existing data pipeline connection
- Validate memory usage under different configurations
- Performance benchmarking across hardware tiers

## 📊 Success Metrics

- **Implementation**: ConfigManager fully operational
- **Testing**: All unit tests passing
- **Performance**: <100ms initialization, <10MB overhead
- **Integration**: Successfully integrated with data pipeline
- **Documentation**: Complete API docs and examples

## 🔗 Dependencies

- **Upstream**: None (foundation component)
- **Downstream**: All other infrastructure components (T0.2-T0.5)
- **External**: PyYAML, psutil packages

## 📝 Notes

- This is the foundation component that all other infrastructure depends on
- Focus on laptop optimization as primary use case
- Ensure easy extension for cloud and desktop configurations
- Include comprehensive error handling for production readiness

---

**Issue Status**: 📋 Ready for Development
**Next Issue**: #02 Data Pipeline Integration & Testing
**Critical Path**: Yes - blocks T0.2, T0.3, T0.4, T0.5