"""
Test suite for FinLab connector authentication integration.

This module tests the integration of the enhanced authentication system
with the FinLab connector.
"""

import os
import pytest
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.data.ingestion.finlab_connector import (
    FinLabConfig, FinLabConnector,
    create_finlab_connector, create_finlab_connector_with_token,
    create_finlab_connector_from_env
)
from src.data.ingestion.finlab_auth import (
    AuthConfig, AuthToken, FinLabAuthenticator, AuthenticationError
)
from src.data.core.temporal import InMemoryTemporalStore


class TestFinLabConfig:
    """Test enhanced FinLab configuration."""

    def setUp(self):
        """Clear environment variables."""
        env_vars = [
            'FINLAB_TOKEN', 'FINLAB_API_KEY', 'FINLAB_DB_HOST',
            'FINLAB_DB_USERNAME', 'FINLAB_DB_PASSWORD'
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]

    def test_config_with_auth_config(self):
        """Test config creation with explicit auth config."""
        auth_config = AuthConfig()
        auth_config.finlab_token = "test-token"

        config = FinLabConfig(auth_config=auth_config)

        assert config.auth_config == auth_config
        assert config.has_token_auth
        assert config.auth_config.finlab_token == "test-token"

    def test_config_legacy_parameters(self):
        """Test config with legacy database parameters."""
        self.setUp()
        config = FinLabConfig(
            host="legacy-host",
            username="legacy-user",
            password="legacy-pass",
            database="legacy-db"
        )

        assert config.auth_config.db_host == "legacy-host"
        assert config.auth_config.db_username == "legacy-user"
        assert config.auth_config.db_password == "legacy-pass"
        assert config.has_db_auth

    def test_config_environment_loading(self):
        """Test config loading from environment."""
        self.setUp()
        os.environ['FINLAB_TOKEN'] = 'env-token'
        os.environ['FINLAB_DB_HOST'] = 'env-host'

        config = FinLabConfig()

        assert config.auth_config.finlab_token == 'env-token'
        assert config.auth_config.db_host == 'env-host'
        assert config.has_token_auth
        assert config.has_db_auth

        self.setUp()

    def test_connection_string_with_auth_config(self):
        """Test connection string generation with auth config."""
        self.setUp()
        os.environ['FINLAB_DB_HOST'] = 'test-host'
        os.environ['FINLAB_DB_USERNAME'] = 'test-user'
        os.environ['FINLAB_DB_PASSWORD'] = 'test-pass'

        config = FinLabConfig()
        conn_str = config.get_connection_string()

        assert "test-user:test-pass@test-host" in conn_str

        self.setUp()

    def test_connection_string_fallback(self):
        """Test connection string generation fallback to legacy."""
        config = FinLabConfig(
            host="legacy-host",
            username="legacy-user",
            password="legacy-pass",
            auth_config=AuthConfig()  # Empty auth config
        )

        conn_str = config.get_connection_string()
        assert "legacy-user:legacy-pass@legacy-host" in conn_str


class TestFinLabConnectorAuth:
    """Test FinLab connector with authentication."""

    def setUp(self):
        """Setup test environment."""
        env_vars = [
            'FINLAB_TOKEN', 'FINLAB_API_KEY', 'FINLAB_DB_HOST',
            'FINLAB_DB_USERNAME', 'FINLAB_DB_PASSWORD'
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]

    def test_connector_initialization(self):
        """Test connector initialization with authentication."""
        self.setUp()
        os.environ['FINLAB_TOKEN'] = 'test-token'

        config = FinLabConfig()
        store = InMemoryTemporalStore()

        connector = FinLabConnector(config, store)

        assert connector.config == config
        assert connector._authenticator is not None
        assert connector._current_token is None  # Not connected yet
        assert connector.auth_errors == 0

    def test_connector_with_custom_authenticator(self):
        """Test connector with custom authenticator."""
        self.setUp()
        config = FinLabConfig(host="test", username="test")
        store = InMemoryTemporalStore()
        custom_auth = Mock(spec=FinLabAuthenticator)

        connector = FinLabConnector(config, store, authenticator=custom_auth)

        assert connector._authenticator == custom_auth

    @patch('src.data.ingestion.finlab_connector.create_engine')
    def test_connector_connect_with_token_auth(self, mock_create_engine):
        """Test connector connection with token authentication."""
        self.setUp()
        os.environ['FINLAB_TOKEN'] = 'valid-token'

        # Mock database engine
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchone.return_value = (1,)

        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_connection.execute.return_value = mock_result

        config = FinLabConfig()
        store = InMemoryTemporalStore()
        connector = FinLabConnector(config, store)

        connector.connect()

        assert connector._current_token is not None
        assert connector._current_token.token == 'valid-token'
        assert connector.engine == mock_engine
        mock_create_engine.assert_called_once()

        self.setUp()

    @patch('src.data.ingestion.finlab_connector.create_engine')
    def test_connector_connect_auth_failure_fallback(self, mock_create_engine):
        """Test connector connection with auth failure and database fallback."""
        self.setUp()
        os.environ['FINLAB_TOKEN'] = 'invalid-token'
        os.environ['FINLAB_DB_HOST'] = 'fallback-host'
        os.environ['FINLAB_DB_USERNAME'] = 'fallback-user'

        # Mock database engine
        mock_engine = Mock()
        mock_connection = Mock()
        mock_result = Mock()
        mock_result.fetchone.return_value = (1,)

        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__.return_value = mock_connection
        mock_connection.execute.return_value = mock_result

        config = FinLabConfig()
        store = InMemoryTemporalStore()
        connector = FinLabConnector(config, store)

        # Mock authentication to fail
        connector._authenticator.get_token = Mock(side_effect=AuthenticationError("Token invalid"))

        connector.connect()

        assert connector.auth_errors == 1
        assert connector.engine == mock_engine
        mock_create_engine.assert_called_once()

        self.setUp()

    def test_authentication_validation(self):
        """Test authentication validation methods."""
        self.setUp()
        os.environ['FINLAB_TOKEN'] = 'test-token'

        config = FinLabConfig()
        store = InMemoryTemporalStore()
        connector = FinLabConnector(config, store)

        # Mock authenticator behavior
        valid_token = AuthToken(token="valid-token", expires_at=datetime.utcnow() + timedelta(hours=1))
        connector._authenticator.get_token = Mock(return_value=valid_token)
        connector._authenticator.validate_token = Mock(return_value=True)

        # Test validation
        is_valid = connector.validate_authentication()
        assert is_valid
        assert connector._current_token == valid_token

        self.setUp()

    def test_authentication_refresh(self):
        """Test authentication refresh functionality."""
        self.setUp()
        os.environ['FINLAB_TOKEN'] = 'test-token'

        config = FinLabConfig()
        store = InMemoryTemporalStore()
        connector = FinLabConnector(config, store)

        # Set expired token
        expired_token = AuthToken(token="expired", expires_at=datetime.utcnow() - timedelta(hours=1))
        connector._current_token = expired_token

        # Mock new token
        new_token = AuthToken(token="refreshed", expires_at=datetime.utcnow() + timedelta(hours=1))
        connector._authenticator.get_token = Mock(return_value=new_token)

        success = connector.refresh_authentication()

        assert success
        assert connector._current_token == new_token
        connector._authenticator.get_token.assert_called_once()

        self.setUp()

    def test_get_auth_headers(self):
        """Test authentication header generation."""
        self.setUp()
        os.environ['FINLAB_TOKEN'] = 'header-token'
        os.environ['FINLAB_API_KEY'] = 'header-api-key'

        config = FinLabConfig()
        store = InMemoryTemporalStore()
        connector = FinLabConnector(config, store)

        # Set current token
        token = AuthToken(token="header-token", token_type="Bearer")
        connector._current_token = token

        headers = connector.get_auth_headers()

        assert headers["Authorization"] == "Bearer header-token"
        assert headers["X-API-Key"] == "header-api-key"

        self.setUp()

    @patch('src.data.ingestion.finlab_connector.sessionmaker')
    def test_authenticated_data_access(self, mock_sessionmaker):
        """Test data access with authentication requirements."""
        self.setUp()
        os.environ['FINLAB_TOKEN'] = 'data-access-token'

        config = FinLabConfig()
        store = InMemoryTemporalStore()
        connector = FinLabConnector(config, store)

        # Mock session and query results
        mock_session = Mock()
        mock_sessionmaker.return_value = lambda: mock_session
        connector.session_factory = mock_sessionmaker.return_value

        mock_query_result = Mock()
        mock_query_result.fetchall.return_value = [('2330',), ('2317',)]
        mock_session.execute.return_value = mock_query_result

        # Mock authentication
        valid_token = AuthToken(token="data-access-token")
        connector._authenticator.get_token = Mock(return_value=valid_token)
        connector._authenticator.validate_token = Mock(return_value=True)

        # Test authenticated data access
        symbols = connector.get_available_symbols()

        assert symbols == ['2330', '2317']
        connector._authenticator.get_token.assert_called()
        connector._authenticator.validate_token.assert_called()

        self.setUp()

    def test_performance_stats_with_auth(self):
        """Test performance statistics including authentication metrics."""
        self.setUp()
        os.environ['FINLAB_TOKEN'] = 'stats-token'

        config = FinLabConfig()
        store = InMemoryTemporalStore()
        connector = FinLabConnector(config, store)

        # Simulate some operations
        connector.query_count = 10
        connector.auth_errors = 1

        # Mock authenticator stats
        auth_stats = {
            'auth_attempts': 5,
            'auth_successes': 4,
            'success_rate_percent': 80.0,
            'cache_hit_rate_percent': 60.0
        }
        connector._authenticator.get_performance_stats = Mock(return_value=auth_stats)

        stats = connector.get_performance_stats()

        assert stats['query_count'] == 10
        assert stats['auth_errors'] == 1
        assert stats['auth_error_rate'] == 0.1
        assert stats['has_token_auth'] == True
        assert stats['auth_auth_attempts'] == 5
        assert stats['auth_success_rate_percent'] == 80.0

        self.setUp()


class TestFactoryFunctions:
    """Test connector factory functions."""

    def setUp(self):
        """Clear environment variables."""
        env_vars = [
            'FINLAB_TOKEN', 'FINLAB_API_KEY', 'FINLAB_DB_HOST',
            'FINLAB_DB_USERNAME', 'FINLAB_DB_PASSWORD'
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]

    def test_create_finlab_connector_basic(self):
        """Test basic connector creation."""
        self.setUp()
        os.environ['FINLAB_TOKEN'] = 'factory-token'

        connector = create_finlab_connector()

        assert isinstance(connector, FinLabConnector)
        assert connector.config.has_token_auth
        assert connector.config.auth_config.finlab_token == 'factory-token'

        self.setUp()

    def test_create_finlab_connector_with_params(self):
        """Test connector creation with parameters."""
        self.setUp()
        connector = create_finlab_connector(
            host="param-host",
            username="param-user",
            password="param-pass"
        )

        assert isinstance(connector, FinLabConnector)
        assert connector.config.has_db_auth
        assert connector.config.auth_config.db_host == "param-host"

    def test_create_finlab_connector_with_token(self):
        """Test token-specific connector creation."""
        connector = create_finlab_connector_with_token("specific-token")

        assert isinstance(connector, FinLabConnector)
        assert connector.config.has_token_auth
        assert connector.config.auth_config.finlab_token == "specific-token"

    def test_create_finlab_connector_from_env(self):
        """Test environment-based connector creation."""
        self.setUp()
        os.environ['FINLAB_TOKEN'] = 'env-factory-token'
        os.environ['FINLAB_API_KEY'] = 'env-factory-api-key'

        connector = create_finlab_connector_from_env()

        assert isinstance(connector, FinLabConnector)
        assert connector.config.has_token_auth
        assert connector.config.auth_config.finlab_token == 'env-factory-token'
        assert connector.config.auth_config.api_key == 'env-factory-api-key'

        self.setUp()

    def test_create_finlab_connector_from_env_with_config_file(self, tmp_path):
        """Test connector creation with config file."""
        # Create test config file
        config_file = tmp_path / "test_config.env"
        config_content = """
FINLAB_TOKEN=file-token
FINLAB_API_KEY=file-api-key
FINLAB_DB_HOST=file-host
"""
        with open(config_file, 'w') as f:
            f.write(config_content)

        connector = create_finlab_connector_from_env(config_file=str(config_file))

        assert isinstance(connector, FinLabConnector)
        assert connector.config.has_token_auth
        assert connector.config.auth_config.finlab_token == 'file-token'
        assert connector.config.auth_config.api_key == 'file-api-key'


class TestErrorHandling:
    """Test error handling in authentication integration."""

    def setUp(self):
        """Clear environment variables."""
        env_vars = [
            'FINLAB_TOKEN', 'FINLAB_API_KEY', 'FINLAB_DB_HOST',
            'FINLAB_DB_USERNAME', 'FINLAB_DB_PASSWORD'
        ]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]

    def test_authentication_error_handling(self):
        """Test authentication error handling."""
        self.setUp()
        os.environ['FINLAB_TOKEN'] = 'error-token'

        config = FinLabConfig()
        store = InMemoryTemporalStore()
        connector = FinLabConnector(config, store)

        # Mock authentication failure
        connector._authenticator.get_token = Mock(side_effect=AuthenticationError("Auth failed"))
        connector._authenticator.validate_token = Mock(return_value=False)

        # Test validation failure
        is_valid = connector.validate_authentication()
        assert not is_valid
        assert connector.auth_errors > 0

        self.setUp()

    def test_require_auth_decorator_error(self):
        """Test require_auth decorator error handling."""
        self.setUp()
        config = FinLabConfig(host="test", username="test")  # No token auth
        store = InMemoryTemporalStore()
        connector = FinLabConnector(config, store)

        # Mock authentication failure
        connector._authenticator = None

        with pytest.raises(AuthenticationError):
            connector.get_available_symbols()

    @patch('src.data.ingestion.finlab_connector.create_engine')
    def test_connect_authentication_error_no_fallback(self, mock_create_engine):
        """Test connection with authentication error and no fallback."""
        self.setUp()
        os.environ['FINLAB_TOKEN'] = 'invalid-token'
        # No database fallback configured

        config = FinLabConfig()
        store = InMemoryTemporalStore()
        connector = FinLabConnector(config, store)

        # Mock authentication to fail
        connector._authenticator.get_token = Mock(side_effect=AuthenticationError("No valid auth"))

        with pytest.raises(AuthenticationError):
            connector.connect()

        assert connector.auth_errors == 1

        self.setUp()


@pytest.fixture(scope="function", autouse=True)
def clean_environment():
    """Clean environment variables before and after each test."""
    env_vars = [
        'FINLAB_TOKEN', 'FINLAB_API_KEY', 'FINLAB_DB_HOST',
        'FINLAB_DB_USERNAME', 'FINLAB_DB_PASSWORD'
    ]

    # Clean before test
    for var in env_vars:
        if var in os.environ:
            del os.environ[var]

    yield

    # Clean after test
    for var in env_vars:
        if var in os.environ:
            del os.environ[var]