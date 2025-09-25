"""
Test suite for FinLab authentication system.

This module provides comprehensive tests for the enhanced FinLab authentication
including token management, caching, validation, and error handling.
"""

import os
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
import threading

from src.data.ingestion.finlab_auth import (
    AuthToken, AuthConfig, TokenCache, FinLabAuthenticator,
    AuthenticationError, require_auth, load_auth_config_from_file
)


class TestAuthToken:
    """Test AuthToken functionality."""

    def test_auth_token_creation(self):
        """Test basic auth token creation."""
        token = AuthToken(
            token="test-token",
            token_type="Bearer",
            expires_at=datetime.utcnow() + timedelta(hours=1),
            permissions=["read_data"]
        )

        assert token.token == "test-token"
        assert token.token_type == "Bearer"
        assert not token.is_expired
        assert token.expires_in_seconds > 3500  # ~1 hour
        assert "read_data" in token.permissions

    def test_token_expiry_check(self):
        """Test token expiry functionality."""
        # Expired token
        expired_token = AuthToken(
            token="expired-token",
            expires_at=datetime.utcnow() - timedelta(hours=1)
        )
        assert expired_token.is_expired

        # Valid token
        valid_token = AuthToken(
            token="valid-token",
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        assert not valid_token.is_expired

        # No expiry token
        no_expiry_token = AuthToken(token="no-expiry-token")
        assert not no_expiry_token.is_expired

    def test_token_to_header(self):
        """Test token to authorization header conversion."""
        token = AuthToken(token="test-token", token_type="Bearer")
        assert token.to_header() == "Bearer test-token"

        api_key_token = AuthToken(token="api-key", token_type="ApiKey")
        assert api_key_token.to_header() == "ApiKey api-key"


class TestAuthConfig:
    """Test AuthConfig functionality."""

    def setUp(self):
        """Clear environment variables before each test."""
        env_vars = [
            'FINLAB_TOKEN', 'FINLAB_API_KEY', 'FINLAB_DB_HOST',
            'FINLAB_DB_USERNAME', 'FINLAB_DB_PASSWORD'
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]

    def test_config_creation_with_defaults(self):
        """Test config creation with default values."""
        self.setUp()
        config = AuthConfig()

        assert config.finlab_token is None
        assert config.api_key is None
        assert config.token_cache_ttl_minutes == 30
        assert config.max_retry_attempts == 3

    def test_config_loading_from_environment(self):
        """Test loading configuration from environment variables."""
        self.setUp()
        os.environ['FINLAB_TOKEN'] = 'env-token'
        os.environ['FINLAB_API_KEY'] = 'env-api-key'
        os.environ['FINLAB_DB_HOST'] = 'env-host'
        os.environ['FINLAB_DB_USERNAME'] = 'env-user'
        os.environ['FINLAB_TOKEN_CACHE_TTL'] = '60'

        config = AuthConfig()

        assert config.finlab_token == 'env-token'
        assert config.api_key == 'env-api-key'
        assert config.db_host == 'env-host'
        assert config.db_username == 'env-user'
        assert config.token_cache_ttl_minutes == 60

        # Cleanup
        self.setUp()

    def test_config_validation_success(self):
        """Test successful configuration validation."""
        self.setUp()
        os.environ['FINLAB_TOKEN'] = 'valid-token'

        config = AuthConfig()  # Should not raise exception
        assert config.has_token_auth

    def test_config_validation_failure(self):
        """Test configuration validation failure."""
        self.setUp()
        # No authentication method configured
        with pytest.raises(ValueError, match="No authentication method configured"):
            AuthConfig()

    def test_has_auth_methods(self):
        """Test authentication method detection."""
        self.setUp()

        # Token authentication
        os.environ['FINLAB_TOKEN'] = 'token'
        config1 = AuthConfig()
        assert config1.has_token_auth
        assert not config1.has_db_auth

        # Database authentication
        self.setUp()
        os.environ['FINLAB_DB_HOST'] = 'localhost'
        os.environ['FINLAB_DB_USERNAME'] = 'user'
        config2 = AuthConfig()
        assert not config2.has_token_auth
        assert config2.has_db_auth

        self.setUp()

    def test_db_connection_string_generation(self):
        """Test database connection string generation."""
        self.setUp()
        os.environ['FINLAB_DB_HOST'] = 'localhost'
        os.environ['FINLAB_DB_USERNAME'] = 'testuser'
        os.environ['FINLAB_DB_PASSWORD'] = 'testpass'
        os.environ['FINLAB_DB_DATABASE'] = 'testdb'
        os.environ['FINLAB_DB_PORT'] = '5432'

        config = AuthConfig()
        conn_str = config.get_db_connection_string()

        assert "postgresql://testuser:testpass@localhost:5432/testdb" == conn_str

        self.setUp()


class TestTokenCache:
    """Test token caching functionality."""

    def test_cache_basic_operations(self):
        """Test basic cache operations."""
        cache = TokenCache(ttl_minutes=1)
        token = AuthToken(token="test-token")

        # Test set and get
        cache.set("key1", token)
        cached_token = cache.get("key1")

        assert cached_token is not None
        assert cached_token.token == "test-token"
        assert cache.size == 1

    def test_cache_expiry(self):
        """Test cache TTL expiry."""
        cache = TokenCache(ttl_minutes=0)  # Immediate expiry
        token = AuthToken(token="test-token")

        cache.set("key1", token)
        time.sleep(0.1)  # Wait for expiry

        cached_token = cache.get("key1")
        assert cached_token is None
        assert cache.size == 0

    def test_token_expiry_in_cache(self):
        """Test token expiry handling in cache."""
        cache = TokenCache(ttl_minutes=60)  # Long cache TTL
        expired_token = AuthToken(
            token="expired-token",
            expires_at=datetime.utcnow() - timedelta(minutes=1)  # Expired token
        )

        cache.set("expired_key", expired_token)
        cached_token = cache.get("expired_key")

        assert cached_token is None  # Should return None for expired token
        assert cache.size == 0

    def test_cache_invalidation(self):
        """Test cache invalidation."""
        cache = TokenCache(ttl_minutes=60)
        token = AuthToken(token="test-token")

        cache.set("key1", token)
        cache.set("key2", token)
        assert cache.size == 2

        # Invalidate single key
        cache.invalidate("key1")
        assert cache.size == 1
        assert cache.get("key1") is None
        assert cache.get("key2") is not None

        # Clear all
        cache.clear()
        assert cache.size == 0

    def test_cache_thread_safety(self):
        """Test cache thread safety."""
        cache = TokenCache(ttl_minutes=60)
        token = AuthToken(token="test-token")
        errors = []

        def cache_operations():
            try:
                for i in range(100):
                    cache.set(f"key_{i}", token)
                    cached = cache.get(f"key_{i}")
                    assert cached is not None
                    cache.invalidate(f"key_{i}")
            except Exception as e:
                errors.append(e)

        # Run multiple threads concurrently
        threads = [threading.Thread(target=cache_operations) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"


class TestFinLabAuthenticator:
    """Test FinLab authenticator functionality."""

    def setUp(self):
        """Setup for each test."""
        # Clear environment
        env_vars = [
            'FINLAB_TOKEN', 'FINLAB_API_KEY', 'FINLAB_DB_HOST',
            'FINLAB_DB_USERNAME', 'FINLAB_DB_PASSWORD'
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]

    def test_authenticator_creation(self):
        """Test authenticator creation."""
        self.setUp()
        os.environ['FINLAB_TOKEN'] = 'test-token'

        config = AuthConfig()
        authenticator = FinLabAuthenticator(config)

        assert authenticator.config == config
        assert authenticator.token_cache is not None
        assert authenticator.auth_attempts == 0

    def test_token_authentication_success(self):
        """Test successful token authentication."""
        self.setUp()
        os.environ['FINLAB_TOKEN'] = 'valid-token'

        config = AuthConfig()
        authenticator = FinLabAuthenticator(config)

        token = authenticator.get_token()

        assert token is not None
        assert token.token == 'valid-token'
        assert token.token_type == 'Bearer'
        assert authenticator.auth_successes == 1

    def test_api_key_authentication(self):
        """Test API key authentication."""
        self.setUp()
        os.environ['FINLAB_API_KEY'] = 'test-api-key'

        config = AuthConfig()
        authenticator = FinLabAuthenticator(config)

        token = authenticator.get_token()

        assert token is not None
        assert token.token == 'test-api-key'
        assert token.token_type == 'ApiKey'

    def test_authentication_caching(self):
        """Test authentication result caching."""
        self.setUp()
        os.environ['FINLAB_TOKEN'] = 'cached-token'

        config = AuthConfig()
        authenticator = FinLabAuthenticator(config)

        # First authentication
        token1 = authenticator.get_token()
        assert authenticator.auth_successes == 1
        assert authenticator.cache_hits == 0

        # Second authentication should use cache
        token2 = authenticator.get_token()
        assert token1.token == token2.token
        assert authenticator.auth_successes == 1  # No new authentication
        assert authenticator.cache_hits == 1

    def test_authentication_retry_logic(self):
        """Test authentication retry with exponential backoff."""
        self.setUp()
        os.environ['FINLAB_TOKEN'] = 'fail-token'

        config = AuthConfig()
        config.max_retry_attempts = 2
        authenticator = FinLabAuthenticator(config)

        # Mock authentication to fail initially
        original_authenticate = authenticator._authenticate

        def failing_authenticate():
            if authenticator.auth_attempts < 3:
                raise AuthenticationError("Simulated failure")
            return original_authenticate()

        authenticator._authenticate = failing_authenticate

        with pytest.raises(AuthenticationError):
            authenticator.get_token()

        assert authenticator.auth_attempts == 1  # get_token increments, retries happen inside

    def test_token_validation(self):
        """Test token validation."""
        self.setUp()
        os.environ['FINLAB_TOKEN'] = 'test-token'

        config = AuthConfig()
        authenticator = FinLabAuthenticator(config)

        # Valid token
        valid_token = AuthToken(
            token="valid-token",
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        assert authenticator.validate_token(valid_token)

        # Expired token
        expired_token = AuthToken(
            token="expired-token",
            expires_at=datetime.utcnow() - timedelta(hours=1)
        )
        assert not authenticator.validate_token(expired_token)

    def test_performance_statistics(self):
        """Test performance statistics collection."""
        self.setUp()
        os.environ['FINLAB_TOKEN'] = 'stats-token'

        config = AuthConfig()
        authenticator = FinLabAuthenticator(config)

        # Perform some operations
        token = authenticator.get_token()
        authenticator.validate_token(token)

        stats = authenticator.get_performance_stats()

        assert stats['auth_attempts'] > 0
        assert stats['auth_successes'] > 0
        assert 'success_rate_percent' in stats
        assert 'cache_hit_rate_percent' in stats
        assert 'avg_auth_time_seconds' in stats


class TestRequireAuthDecorator:
    """Test the require_auth decorator."""

    def test_require_auth_success(self):
        """Test successful authentication with decorator."""
        class MockClass:
            def __init__(self):
                self._authenticator = Mock()
                self._authenticator.get_token.return_value = AuthToken(token="test")
                self._authenticator.validate_token.return_value = True

            @require_auth
            def protected_method(self):
                return "success"

        obj = MockClass()
        result = obj.protected_method()

        assert result == "success"
        obj._authenticator.get_token.assert_called_once()
        obj._authenticator.validate_token.assert_called_once()

    def test_require_auth_no_authenticator(self):
        """Test decorator with no authenticator."""
        class MockClass:
            @require_auth
            def protected_method(self):
                return "success"

        obj = MockClass()

        with pytest.raises(AuthenticationError, match="Authentication not initialized"):
            obj.protected_method()

    def test_require_auth_invalid_token(self):
        """Test decorator with invalid token."""
        class MockClass:
            def __init__(self):
                self._authenticator = Mock()
                self._authenticator.get_token.return_value = AuthToken(token="test")
                self._authenticator.validate_token.return_value = False

            @require_auth
            def protected_method(self):
                return "success"

        obj = MockClass()

        with pytest.raises(AuthenticationError, match="Invalid or expired token"):
            obj.protected_method()


class TestConfigFileLoading:
    """Test configuration file loading."""

    def test_load_json_config(self, tmp_path):
        """Test loading JSON configuration file."""
        config_file = tmp_path / "auth_config.json"
        config_data = {
            "FINLAB_TOKEN": "file-token",
            "FINLAB_API_KEY": "file-api-key",
            "FINLAB_DB_HOST": "file-host"
        }

        with open(config_file, 'w') as f:
            import json
            json.dump(config_data, f)

        config = load_auth_config_from_file(str(config_file))

        assert config.finlab_token == "file-token"
        assert config.api_key == "file-api-key"
        assert config.db_host == "file-host"

    def test_load_env_config(self, tmp_path):
        """Test loading .env format configuration file."""
        config_file = tmp_path / ".env"
        config_content = """
FINLAB_TOKEN=env-file-token
FINLAB_API_KEY=env-file-api-key
FINLAB_DB_HOST=env-file-host
# Comment line
FINLAB_DB_USERNAME=env-file-user
"""

        with open(config_file, 'w') as f:
            f.write(config_content)

        config = load_auth_config_from_file(str(config_file))

        assert config.finlab_token == "env-file-token"
        assert config.api_key == "env-file-api-key"
        assert config.db_host == "env-file-host"
        assert config.db_username == "env-file-user"

    def test_load_nonexistent_config(self):
        """Test loading nonexistent configuration file."""
        config = load_auth_config_from_file("/nonexistent/file.json")

        # Should return default config without raising exception
        assert isinstance(config, AuthConfig)


class TestIntegration:
    """Integration tests for the authentication system."""

    def setUp(self):
        """Setup for integration tests."""
        env_vars = [
            'FINLAB_TOKEN', 'FINLAB_API_KEY', 'FINLAB_DB_HOST',
            'FINLAB_DB_USERNAME', 'FINLAB_DB_PASSWORD'
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]

    def test_end_to_end_token_auth(self):
        """Test complete token authentication flow."""
        self.setUp()
        os.environ['FINLAB_TOKEN'] = 'integration-test-token'
        os.environ['FINLAB_TOKEN_CACHE_TTL'] = '5'

        # Create authenticator from environment
        config = AuthConfig()
        authenticator = FinLabAuthenticator(config)

        # Test authentication
        token = authenticator.get_token()
        assert token.token == 'integration-test-token'

        # Test validation
        is_valid = authenticator.validate_token(token)
        assert is_valid

        # Test caching
        token2 = authenticator.get_token()
        assert token.token == token2.token
        assert authenticator.cache_hits > 0

        # Test performance stats
        stats = authenticator.get_performance_stats()
        assert stats['auth_successes'] > 0
        assert stats['success_rate_percent'] > 0

        self.setUp()

    def test_authentication_fallback(self):
        """Test fallback from token to database authentication."""
        self.setUp()
        # Set up both authentication methods, but token will "fail"
        os.environ['FINLAB_TOKEN'] = 'invalid-token'
        os.environ['FINLAB_DB_HOST'] = 'localhost'
        os.environ['FINLAB_DB_USERNAME'] = 'fallback_user'

        config = AuthConfig()

        # Both methods should be available
        assert config.has_token_auth
        assert config.has_db_auth

        # Database connection string should work as fallback
        conn_str = config.get_db_connection_string()
        assert 'fallback_user' in conn_str

        self.setUp()

    def test_concurrent_authentication(self):
        """Test concurrent authentication requests."""
        self.setUp()
        os.environ['FINLAB_TOKEN'] = 'concurrent-test-token'

        config = AuthConfig()
        authenticator = FinLabAuthenticator(config)
        results = []
        errors = []

        def authenticate_concurrently():
            try:
                token = authenticator.get_token()
                results.append(token.token)
            except Exception as e:
                errors.append(e)

        # Run concurrent authentications
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(authenticate_concurrently)
                for _ in range(20)
            ]
            for future in futures:
                future.result()

        # All should succeed and return the same token
        assert len(errors) == 0, f"Concurrent authentication errors: {errors}"
        assert len(results) == 20
        assert all(token == 'concurrent-test-token' for token in results)

        # Should have high cache hit rate
        stats = authenticator.get_performance_stats()
        assert stats['cache_hit_rate_percent'] > 50

        self.setUp()


@pytest.fixture(scope="function", autouse=True)
def clean_environment():
    """Clean environment variables before each test."""
    env_vars = [
        'FINLAB_TOKEN', 'FINLAB_API_KEY', 'FINLAB_DB_HOST',
        'FINLAB_DB_USERNAME', 'FINLAB_DB_PASSWORD', 'FINLAB_TOKEN_CACHE_TTL'
    ]
    for var in env_vars:
        if var in os.environ:
            del os.environ[var]

    yield

    # Cleanup after test
    for var in env_vars:
        if var in os.environ:
            del os.environ[var]