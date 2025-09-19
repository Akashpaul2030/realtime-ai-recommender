"""Tests for configuration module."""
import pytest
import os
from unittest.mock import patch
from config import Config


def test_default_config():
    """Test default configuration values."""
    config = Config()
    # Note: May be "cloud" if .env file sets it
    assert config.BACKEND_TYPE in ["redis", "cloud", "hybrid"]
    assert config.REDIS_HOST == "localhost"
    assert config.REDIS_PORT == 6379
    assert config.API_HOST == "0.0.0.0"
    assert config.API_PORT == 8000


def test_config_from_env():
    """Test configuration from environment variables."""
    with patch.dict(os.environ, {
        'BACKEND_TYPE': 'cloud',
        'REDIS_HOST': 'test-host',
        'REDIS_PORT': '1234',
        'API_PORT': '9000'
    }):
        config = Config()
        assert config.BACKEND_TYPE == "cloud"
        assert config.REDIS_HOST == "test-host"
        assert config.REDIS_PORT == 1234
        assert config.API_PORT == 9000


def test_config_redis_url():
    """Test Redis URL construction."""
    config = Config()
    expected_url = f"redis://{config.REDIS_HOST}:{config.REDIS_PORT}/{config.REDIS_DB}"
    assert config.get_redis_url() == expected_url


def test_config_vector_dimensions():
    """Test vector dimension configuration."""
    config = Config()
    assert config.VECTOR_DIMENSIONS == 384
    assert isinstance(config.VECTOR_DIMENSIONS, int)