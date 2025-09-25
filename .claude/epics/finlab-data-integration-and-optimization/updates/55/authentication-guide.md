# FinLab Authentication System - Implementation Guide

## Overview

The enhanced FinLab authentication system provides secure, efficient token-based authentication with comprehensive error handling, caching, and fallback mechanisms for the ML4T data integration pipeline.

## Features

✅ **Token-Based Authentication**: Primary authentication method using FINLAB_TOKEN or FINLAB_API_KEY
✅ **Secure Token Caching**: Thread-safe caching with configurable TTL
✅ **Automatic Token Refresh**: Smart token validation and refresh mechanisms
✅ **Database Fallback**: Legacy database authentication support
✅ **Comprehensive Error Handling**: Retry logic with exponential backoff
✅ **Performance Monitoring**: Authentication metrics and performance tracking
✅ **Environment Configuration**: Easy .env-based configuration

## Authentication Methods

### 1. Token Authentication (Recommended)

```python
# Using environment variables
FINLAB_TOKEN=your-finlab-api-token
FINLAB_API_KEY=your-secondary-api-key

# Programmatic usage
from src.data.ingestion.finlab_connector import create_finlab_connector_with_token

connector = create_finlab_connector_with_token(
    token="your-finlab-api-token",
    temporal_store=your_temporal_store
)
```

### 2. Environment-Based Configuration

```python
from src.data.ingestion.finlab_connector import create_finlab_connector_from_env

# Loads all configuration from environment variables
connector = create_finlab_connector_from_env()
```

### 3. Database Authentication (Legacy/Fallback)

```python
from src.data.ingestion.finlab_connector import create_finlab_connector

connector = create_finlab_connector(
    host="localhost",
    port=5432,
    database="finlab",
    username="finlab_user",
    password="secure_password"
)
```

## Environment Variables

### Primary Authentication
```bash
# Required: At least one token authentication method
FINLAB_TOKEN=your-finlab-api-token          # Primary authentication token
FINLAB_API_KEY=your-finlab-api-key          # Alternative API key

# API Configuration
FINLAB_API_BASE_URL=https://api.finlab.tw   # Base API URL
FINLAB_AUTH_ENDPOINT=/api/v1/auth           # Authentication endpoint
```

### Database Fallback
```bash
# Database connection (fallback authentication)
FINLAB_DB_HOST=localhost                    # Database host
FINLAB_DB_PORT=5432                         # Database port
FINLAB_DB_DATABASE=finlab                   # Database name
FINLAB_DB_USERNAME=finlab_user              # Database username
FINLAB_DB_PASSWORD=secure_password          # Database password
```

### Performance Tuning
```bash
# Caching and performance
FINLAB_TOKEN_CACHE_TTL=30                   # Token cache TTL (minutes)
FINLAB_MAX_RETRIES=3                        # Maximum retry attempts
FINLAB_CONNECT_TIMEOUT=30                   # Connection timeout (seconds)
```

## Usage Examples

### Basic Usage with Token

```python
import os
from src.data.ingestion.finlab_connector import create_finlab_connector_from_env

# Set environment variables
os.environ['FINLAB_TOKEN'] = 'your-api-token'

# Create connector (automatically uses token authentication)
with create_finlab_connector_from_env() as connector:
    # Get available symbols
    symbols = connector.get_available_symbols()

    # Fetch price data with automatic authentication
    price_data = connector.get_price_data(
        symbol="2330",
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2024, 12, 31)
    )
```

### Advanced Configuration

```python
from src.data.ingestion.finlab_auth import AuthConfig, FinLabAuthenticator
from src.data.ingestion.finlab_connector import FinLabConfig, FinLabConnector
from src.data.core.temporal import InMemoryTemporalStore

# Create custom authentication configuration
auth_config = AuthConfig()
auth_config.finlab_token = "your-api-token"
auth_config.token_cache_ttl_minutes = 60
auth_config.max_retry_attempts = 5

# Create connector with custom configuration
config = FinLabConfig(auth_config=auth_config)
temporal_store = InMemoryTemporalStore()

with FinLabConnector(config, temporal_store) as connector:
    # Your data operations here
    performance_stats = connector.get_performance_stats()
    print(f"Authentication success rate: {performance_stats['auth_success_rate_percent']}%")
```

### Error Handling

```python
from src.data.ingestion.finlab_connector import create_finlab_connector_from_env
from src.data.ingestion.finlab_auth import AuthenticationError

try:
    with create_finlab_connector_from_env() as connector:
        data = connector.get_price_data("2330", start_date, end_date)

except AuthenticationError as e:
    print(f"Authentication failed: {e}")
    # Handle authentication error (e.g., refresh token, prompt user)

except Exception as e:
    print(f"Connection error: {e}")
    # Handle other connection errors
```

## Security Best Practices

### 1. Environment Variable Security
```bash
# Use strong, unique tokens
FINLAB_TOKEN=eyJ0eXAiOiJKV1QiLCJhbGciOiJ...

# Never commit tokens to version control
echo "FINLAB_TOKEN=your-token" >> .env
echo ".env" >> .gitignore

# Use different tokens for different environments
FINLAB_TOKEN_PROD=production-token
FINLAB_TOKEN_DEV=development-token
```

### 2. Token Rotation
```python
# Implement token rotation in production
import schedule
import time

def rotate_token():
    """Rotate authentication token periodically."""
    new_token = fetch_new_token_from_provider()
    os.environ['FINLAB_TOKEN'] = new_token

    # Clear cached tokens to force refresh
    if hasattr(connector, '_authenticator'):
        connector._authenticator.invalidate_cache()

# Schedule token rotation every 12 hours
schedule.every(12).hours.do(rotate_token)
```

### 3. Monitoring and Alerting
```python
# Monitor authentication performance
def check_auth_health(connector):
    """Check authentication system health."""
    stats = connector.get_performance_stats()

    if stats['auth_error_rate'] > 0.05:  # 5% error rate threshold
        send_alert(f"High authentication error rate: {stats['auth_error_rate']}")

    if stats['auth_cache_hit_rate_percent'] < 80:  # Low cache hit rate
        send_alert(f"Low auth cache hit rate: {stats['auth_cache_hit_rate_percent']}%")
```

## Performance Optimization

### 1. Token Caching
```python
# Optimize cache TTL based on usage patterns
auth_config = AuthConfig()
auth_config.token_cache_ttl_minutes = 60  # Longer TTL for stable tokens

# Monitor cache performance
stats = connector.get_performance_stats()
cache_hit_rate = stats['auth_cache_hit_rate_percent']
print(f"Authentication cache hit rate: {cache_hit_rate}%")
```

### 2. Connection Pooling
```python
# Configure connection pool for high throughput
config = FinLabConfig(
    pool_size=20,          # Increase pool size
    max_overflow=40,       # Allow connection overflow
    pool_timeout=60,       # Increase timeout
    pool_recycle=1800      # Recycle connections more frequently
)
```

### 3. Retry Configuration
```python
# Optimize retry behavior
auth_config = AuthConfig()
auth_config.max_retry_attempts = 5          # More retries for critical operations
auth_config.retry_backoff_seconds = 0.5     # Faster initial retry
auth_config.connection_timeout_seconds = 45  # Longer timeout for stability
```

## Troubleshooting

### Common Issues

#### Authentication Failure
```bash
# Check environment variables
env | grep FINLAB

# Verify token validity
python -c "from src.data.ingestion.finlab_auth import AuthConfig; print(AuthConfig().finlab_token)"
```

#### Token Expiry
```python
# Check token status
connector = create_finlab_connector_from_env()
is_valid = connector.validate_authentication()
print(f"Authentication valid: {is_valid}")

# Force token refresh
success = connector.refresh_authentication()
print(f"Token refresh successful: {success}")
```

#### Performance Issues
```python
# Get detailed performance statistics
stats = connector.get_performance_stats()
for key, value in stats.items():
    if 'auth' in key:
        print(f"{key}: {value}")
```

### Debug Mode
```python
# Enable detailed logging
import logging
logging.getLogger('src.data.ingestion.finlab_auth').setLevel(logging.DEBUG)
logging.getLogger('src.data.ingestion.finlab_connector').setLevel(logging.DEBUG)
```

## Migration Guide

### From Legacy Database Authentication
```python
# Old approach
connector = FinLabConnector(
    config=FinLabConfig(
        host="localhost",
        username="finlab",
        password="password"
    ),
    temporal_store=store
)

# New approach (recommended)
os.environ['FINLAB_TOKEN'] = 'your-api-token'
connector = create_finlab_connector_from_env()
```

### Gradual Migration Strategy
1. Set up token authentication alongside database authentication
2. Test token authentication in development environment
3. Monitor authentication performance and error rates
4. Gradually migrate production workloads to token authentication
5. Disable database authentication once token authentication is stable

## Testing

### Unit Tests
```python
import pytest
from src.data.ingestion.finlab_auth import AuthConfig, FinLabAuthenticator

def test_auth_config_from_env():
    """Test authentication configuration loading."""
    os.environ['FINLAB_TOKEN'] = 'test-token'
    config = AuthConfig()
    assert config.finlab_token == 'test-token'
    assert config.has_token_auth

def test_token_caching():
    """Test token caching functionality."""
    authenticator = FinLabAuthenticator()
    token1 = authenticator.get_token()
    token2 = authenticator.get_token()

    # Should return cached token
    assert token1 == token2
    assert authenticator.cache_hits > 0
```

### Integration Tests
```python
def test_connector_with_token_auth():
    """Test connector with token authentication."""
    os.environ['FINLAB_TOKEN'] = 'valid-test-token'

    with create_finlab_connector_from_env() as connector:
        assert connector.validate_authentication()
        symbols = connector.get_available_symbols()
        assert isinstance(symbols, list)
```

## Production Deployment

### Environment Setup
```bash
# Production environment file
cat > .env.production << EOF
ENVIRONMENT=production
FINLAB_TOKEN=prod-api-token
FINLAB_TOKEN_CACHE_TTL=30
FINLAB_MAX_RETRIES=5
FINLAB_CONNECT_TIMEOUT=60
EOF

# Secure file permissions
chmod 600 .env.production
```

### Monitoring Dashboard
```python
# Create authentication monitoring dashboard
def create_auth_dashboard():
    """Create authentication monitoring dashboard."""
    return {
        "authentication_success_rate": "auth_success_rate_percent",
        "cache_hit_rate": "auth_cache_hit_rate_percent",
        "average_auth_time": "auth_avg_auth_time_seconds",
        "error_count": "auth_errors",
        "token_validity": "current_token_valid"
    }
```

## Summary

The enhanced FinLab authentication system provides:

- **Security**: Token-based authentication with secure caching
- **Reliability**: Automatic retry and fallback mechanisms
- **Performance**: Intelligent caching and connection pooling
- **Monitoring**: Comprehensive metrics and error tracking
- **Flexibility**: Multiple authentication methods and easy configuration

For questions or issues, refer to the authentication system logs or contact the ML4T development team.