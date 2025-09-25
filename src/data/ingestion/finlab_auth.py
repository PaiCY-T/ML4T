"""
FinLab Authentication Module with Token-Based Security.

This module provides comprehensive authentication capabilities for FinLab API integration,
including secure token management, validation, caching, and error recovery.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class AuthToken:
    """Secure authentication token representation."""
    token: str
    token_type: str = "Bearer"
    expires_at: Optional[datetime] = None
    refresh_token: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if token is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() >= self.expires_at

    @property
    def expires_in_seconds(self) -> Optional[int]:
        """Get seconds until token expires."""
        if self.expires_at is None:
            return None
        delta = self.expires_at - datetime.utcnow()
        return max(0, int(delta.total_seconds()))

    def to_header(self) -> str:
        """Convert token to Authorization header format."""
        return f"{self.token_type} {self.token}"


@dataclass
class AuthConfig:
    """Authentication configuration with environment variable loading."""
    # Primary authentication method - token-based
    finlab_token: Optional[str] = None
    api_key: Optional[str] = None

    # Database fallback (legacy support)
    db_host: Optional[str] = None
    db_port: int = 5432
    db_database: Optional[str] = None
    db_username: Optional[str] = None
    db_password: Optional[str] = None

    # API endpoints
    api_base_url: str = "https://api.finlab.tw"
    auth_endpoint: str = "/api/v1/auth"

    # Security settings
    token_cache_ttl_minutes: int = 30
    max_retry_attempts: int = 3
    retry_backoff_seconds: float = 1.0

    # Performance settings
    connection_timeout_seconds: int = 30
    read_timeout_seconds: int = 60
    max_concurrent_requests: int = 5

    def __post_init__(self):
        """Load configuration from environment variables."""
        self._load_from_env()
        self._validate_config()

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Primary token authentication
        self.finlab_token = self._get_env('FINLAB_TOKEN', self.finlab_token)
        self.api_key = self._get_env('FINLAB_API_KEY', self.api_key)

        # Database configuration (fallback)
        self.db_host = self._get_env('FINLAB_DB_HOST', self.db_host)
        self.db_port = int(self._get_env('FINLAB_DB_PORT', str(self.db_port)))
        self.db_database = self._get_env('FINLAB_DB_DATABASE', self.db_database)
        self.db_username = self._get_env('FINLAB_DB_USERNAME', self.db_username)
        self.db_password = self._get_env('FINLAB_DB_PASSWORD', self.db_password)

        # API configuration
        self.api_base_url = self._get_env('FINLAB_API_BASE_URL', self.api_base_url)
        self.auth_endpoint = self._get_env('FINLAB_AUTH_ENDPOINT', self.auth_endpoint)

        # Performance tuning
        self.token_cache_ttl_minutes = int(self._get_env('FINLAB_TOKEN_CACHE_TTL', str(self.token_cache_ttl_minutes)))
        self.max_retry_attempts = int(self._get_env('FINLAB_MAX_RETRIES', str(self.max_retry_attempts)))
        self.connection_timeout_seconds = int(self._get_env('FINLAB_CONNECT_TIMEOUT', str(self.connection_timeout_seconds)))

    def _get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable with logging."""
        value = os.getenv(key, default)
        if value and key not in ['FINLAB_TOKEN', 'FINLAB_API_KEY', 'FINLAB_DB_PASSWORD']:
            logger.debug(f"Loaded {key}={value}")
        elif value:
            logger.debug(f"Loaded {key}=***")
        return value

    def _validate_config(self) -> None:
        """Validate authentication configuration."""
        errors = []

        # At least one authentication method must be configured
        has_token_auth = bool(self.finlab_token or self.api_key)
        has_db_auth = bool(self.db_host and self.db_username)

        if not has_token_auth and not has_db_auth:
            errors.append("No authentication method configured. Set FINLAB_TOKEN/FINLAB_API_KEY or database credentials.")

        # Validate numeric parameters
        if self.token_cache_ttl_minutes <= 0:
            errors.append("token_cache_ttl_minutes must be positive")

        if self.max_retry_attempts < 0:
            errors.append("max_retry_attempts must be non-negative")

        if self.connection_timeout_seconds <= 0:
            errors.append("connection_timeout_seconds must be positive")

        if errors:
            error_msg = "Authentication configuration errors: " + "; ".join(errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Authentication configuration validated successfully")

    @property
    def has_token_auth(self) -> bool:
        """Check if token authentication is available."""
        return bool(self.finlab_token or self.api_key)

    @property
    def has_db_auth(self) -> bool:
        """Check if database authentication is available."""
        return bool(self.db_host and self.db_username)

    def get_db_connection_string(self) -> str:
        """Generate database connection string."""
        if not self.has_db_auth:
            raise ValueError("Database authentication not configured")

        return (f"postgresql://{self.db_username}:{self.db_password or ''}@"
                f"{self.db_host}:{self.db_port}/{self.db_database or 'finlab'}")


class TokenCache:
    """Thread-safe token cache with TTL support."""

    def __init__(self, ttl_minutes: int = 30):
        self.ttl_seconds = ttl_minutes * 60
        self._cache: Dict[str, Tuple[AuthToken, float]] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[AuthToken]:
        """Get token from cache if not expired."""
        with self._lock:
            if key not in self._cache:
                return None

            token, timestamp = self._cache[key]

            # Check TTL expiry
            if time.time() - timestamp > self.ttl_seconds:
                del self._cache[key]
                logger.debug(f"Token cache expired for key: {self._mask_key(key)}")
                return None

            # Check token expiry
            if token.is_expired:
                del self._cache[key]
                logger.debug(f"Token expired for key: {self._mask_key(key)}")
                return None

            logger.debug(f"Token cache hit for key: {self._mask_key(key)}")
            return token

    def set(self, key: str, token: AuthToken) -> None:
        """Store token in cache."""
        with self._lock:
            self._cache[key] = (token, time.time())
            logger.debug(f"Token cached for key: {self._mask_key(key)}")

    def invalidate(self, key: str) -> None:
        """Remove token from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Token invalidated for key: {self._mask_key(key)}")

    def clear(self) -> None:
        """Clear all cached tokens."""
        with self._lock:
            self._cache.clear()
            logger.info("Token cache cleared")

    def _mask_key(self, key: str) -> str:
        """Mask cache key for logging."""
        if len(key) <= 8:
            return "***"
        return key[:4] + "***" + key[-4:]

    @property
    def size(self) -> int:
        """Get number of cached tokens."""
        with self._lock:
            return len(self._cache)


class AuthenticationError(Exception):
    """Authentication-related error."""

    def __init__(self, message: str, error_code: Optional[str] = None, retry_after: Optional[int] = None):
        super().__init__(message)
        self.error_code = error_code
        self.retry_after = retry_after


class FinLabAuthenticator:
    """Comprehensive FinLab authentication manager."""

    def __init__(self, config: Optional[AuthConfig] = None):
        self.config = config or AuthConfig()
        self.token_cache = TokenCache(self.config.token_cache_ttl_minutes)

        # Performance metrics
        self.auth_attempts = 0
        self.auth_successes = 0
        self.cache_hits = 0
        self.total_auth_time = 0.0

        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="finlab-auth")

        logger.info("FinLab authenticator initialized")

    def get_token(self, cache_key: Optional[str] = None) -> AuthToken:
        """Get valid authentication token with caching and retry logic."""
        start_time = time.time()
        self.auth_attempts += 1

        if cache_key is None:
            cache_key = self._generate_cache_key()

        try:
            # Try cache first
            cached_token = self.token_cache.get(cache_key)
            if cached_token:
                self.cache_hits += 1
                logger.debug("Using cached authentication token")
                return cached_token

            # Authenticate with retry logic
            token = self._authenticate_with_retry()

            # Cache the token
            self.token_cache.set(cache_key, token)

            self.auth_successes += 1
            self.total_auth_time += time.time() - start_time

            logger.info("Authentication successful")
            return token

        except Exception as e:
            self.total_auth_time += time.time() - start_time
            logger.error(f"Authentication failed: {e}")
            raise AuthenticationError(f"Failed to obtain authentication token: {e}")

    def _authenticate_with_retry(self) -> AuthToken:
        """Authenticate with exponential backoff retry."""
        last_error = None

        for attempt in range(self.config.max_retry_attempts + 1):
            try:
                if attempt > 0:
                    backoff_time = self.config.retry_backoff_seconds * (2 ** (attempt - 1))
                    logger.debug(f"Retrying authentication in {backoff_time}s (attempt {attempt + 1})")
                    time.sleep(backoff_time)

                return self._authenticate()

            except AuthenticationError as e:
                last_error = e
                if e.retry_after:
                    logger.warning(f"Rate limited, retrying after {e.retry_after}s")
                    time.sleep(e.retry_after)
                elif attempt < self.config.max_retry_attempts:
                    logger.warning(f"Authentication attempt {attempt + 1} failed: {e}")
                else:
                    logger.error(f"All authentication attempts failed: {e}")
                    break
            except Exception as e:
                last_error = AuthenticationError(f"Unexpected authentication error: {e}")
                if attempt < self.config.max_retry_attempts:
                    logger.warning(f"Authentication attempt {attempt + 1} failed: {e}")
                else:
                    logger.error(f"All authentication attempts failed: {e}")
                    break

        raise last_error or AuthenticationError("Authentication failed after all retries")

    def _authenticate(self) -> AuthToken:
        """Perform actual authentication."""
        if self.config.finlab_token:
            return self._authenticate_with_token()
        elif self.config.api_key:
            return self._authenticate_with_api_key()
        else:
            raise AuthenticationError("No token authentication method available")

    def _authenticate_with_token(self) -> AuthToken:
        """Authenticate using FINLAB_TOKEN."""
        token = self.config.finlab_token

        # Validate token format (basic check)
        if not token or len(token) < 10:
            raise AuthenticationError("Invalid FINLAB_TOKEN format")

        # For now, create a token with reasonable defaults
        # In a real implementation, you would validate with the API
        expires_at = datetime.utcnow() + timedelta(hours=24)

        return AuthToken(
            token=token,
            token_type="Bearer",
            expires_at=expires_at,
            permissions=["read_data", "query_api"],
            metadata={
                "auth_method": "finlab_token",
                "authenticated_at": datetime.utcnow().isoformat()
            }
        )

    def _authenticate_with_api_key(self) -> AuthToken:
        """Authenticate using API key."""
        api_key = self.config.api_key

        if not api_key:
            raise AuthenticationError("API key not configured")

        # Create API key based token
        expires_at = datetime.utcnow() + timedelta(hours=12)

        return AuthToken(
            token=api_key,
            token_type="ApiKey",
            expires_at=expires_at,
            permissions=["read_data"],
            metadata={
                "auth_method": "api_key",
                "authenticated_at": datetime.utcnow().isoformat()
            }
        )

    def _generate_cache_key(self) -> str:
        """Generate cache key based on configuration."""
        # Create a hash of authentication parameters
        key_data = f"{self.config.finlab_token or ''}{self.config.api_key or ''}{self.config.db_username or ''}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def validate_token(self, token: AuthToken) -> bool:
        """Validate token is still valid."""
        try:
            if token.is_expired:
                logger.debug("Token validation failed: expired")
                return False

            # Additional validation logic could go here
            # For example, making a test API call

            logger.debug("Token validation successful")
            return True

        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return False

    def refresh_token(self, token: AuthToken) -> Optional[AuthToken]:
        """Refresh an expired token if possible."""
        if not token.refresh_token:
            logger.debug("No refresh token available")
            return None

        try:
            # Implementation would depend on FinLab API
            logger.info("Token refresh not implemented yet")
            return None

        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return None

    def invalidate_cache(self, cache_key: Optional[str] = None) -> None:
        """Invalidate cached tokens."""
        if cache_key:
            self.token_cache.invalidate(cache_key)
        else:
            self.token_cache.clear()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get authentication performance statistics."""
        success_rate = (self.auth_successes / max(self.auth_attempts, 1)) * 100
        avg_auth_time = self.total_auth_time / max(self.auth_successes, 1)
        cache_hit_rate = (self.cache_hits / max(self.auth_attempts, 1)) * 100

        return {
            "auth_attempts": self.auth_attempts,
            "auth_successes": self.auth_successes,
            "success_rate_percent": round(success_rate, 2),
            "cache_hits": self.cache_hits,
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "cache_size": self.token_cache.size,
            "avg_auth_time_seconds": round(avg_auth_time, 3),
            "total_auth_time_seconds": round(self.total_auth_time, 2)
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._executor.shutdown(wait=True)


def require_auth(func):
    """Decorator to ensure authentication before function execution."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, '_authenticator') or not self._authenticator:
            raise AuthenticationError("Authentication not initialized")

        try:
            # Ensure we have a valid token
            token = self._authenticator.get_token()
            if not self._authenticator.validate_token(token):
                raise AuthenticationError("Invalid or expired token")

            return func(self, *args, **kwargs)

        except AuthenticationError:
            raise
        except Exception as e:
            raise AuthenticationError(f"Authentication required for {func.__name__}: {e}")

    return wrapper


def load_auth_config_from_file(file_path: str) -> AuthConfig:
    """Load authentication configuration from file."""
    path = Path(file_path)

    if not path.exists():
        logger.warning(f"Auth config file not found: {file_path}")
        return AuthConfig()

    try:
        with open(path, 'r') as f:
            if path.suffix.lower() == '.json':
                config_data = json.load(f)
            else:
                # Assume .env format
                config_data = {}
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        config_data[key.strip()] = value.strip().strip('"\'')

        # Map file config to AuthConfig
        config = AuthConfig()

        # Override defaults with file values
        for attr_name in ['finlab_token', 'api_key', 'db_host', 'db_database',
                         'db_username', 'db_password', 'api_base_url']:
            env_key = attr_name.upper()
            if env_key in config_data:
                setattr(config, attr_name, config_data[env_key])

        logger.info(f"Authentication config loaded from {file_path}")
        return config

    except Exception as e:
        logger.error(f"Failed to load auth config from {file_path}: {e}")
        return AuthConfig()


# Factory function for easy creation
def create_finlab_authenticator(config_file: Optional[str] = None) -> FinLabAuthenticator:
    """Factory function to create FinLab authenticator."""
    if config_file:
        config = load_auth_config_from_file(config_file)
    else:
        config = AuthConfig()

    return FinLabAuthenticator(config)