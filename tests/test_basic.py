"""Basic tests that don't require external dependencies."""
import pytest
import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def test_project_structure():
    """Test that basic project files exist."""
    project_files = [
        "config.py",
        "requirements.txt",
        "api/app.py",
        "models/embeddings.py",
        "data/schemas.py",
        "services/vector_store.py"
    ]

    for file_path in project_files:
        full_path = os.path.join(project_root, file_path)
        assert os.path.exists(full_path), f"Missing required file: {file_path}"


def test_imports():
    """Test that basic modules can be imported."""
    try:
        from config import Config, BACKEND_TYPE, API_HOST, API_PORT
        assert isinstance(BACKEND_TYPE, str)
        assert isinstance(API_HOST, str)
        assert isinstance(API_PORT, int)

        # Test Config class
        config = Config()
        assert hasattr(config, 'BACKEND_TYPE')
        assert hasattr(config, 'REDIS_HOST')
        assert hasattr(config, 'API_PORT')
    except ImportError as e:
        pytest.fail(f"Failed to import config: {e}")


def test_config_class():
    """Test Config class functionality."""
    from config import Config

    config = Config()

    # Test required attributes exist
    assert hasattr(config, 'BACKEND_TYPE')
    assert hasattr(config, 'REDIS_HOST')
    assert hasattr(config, 'REDIS_PORT')
    assert hasattr(config, 'API_HOST')
    assert hasattr(config, 'API_PORT')
    assert hasattr(config, 'VECTOR_DIMENSIONS')

    # Test default values (may be overridden by .env file)
    assert config.BACKEND_TYPE in ["redis", "cloud", "hybrid"]
    assert config.REDIS_HOST == "localhost"
    assert config.REDIS_PORT == 6379
    assert config.API_HOST == "0.0.0.0"
    assert config.API_PORT == 8000
    assert config.VECTOR_DIMENSIONS == 384

    # Test Redis URL generation
    redis_url = config.get_redis_url()
    assert isinstance(redis_url, str)
    assert "redis://" in redis_url
    assert str(config.REDIS_PORT) in redis_url


def test_python_version():
    """Test that we're running on supported Python version."""
    import sys
    version = sys.version_info

    # CI tests Python 3.8, 3.9, 3.10
    assert version.major == 3
    assert version.minor >= 8
    assert version.minor <= 11  # Allow up to 3.11


def test_required_packages():
    """Test that required packages can be imported."""
    required_packages = [
        'fastapi',
        'pydantic',
        'numpy',
        'sklearn',
        'redis',
        'loguru'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        pytest.fail(f"Missing required packages: {missing_packages}")


def test_environment_variables():
    """Test that environment variables are handled correctly."""
    import os
    from config import Config

    # Test with clean environment
    original_env = dict(os.environ)

    try:
        # Clear relevant env vars
        for key in ['BACKEND_TYPE', 'REDIS_HOST', 'API_PORT']:
            if key in os.environ:
                del os.environ[key]

        config = Config()
        # Should use actual default when env vars are cleared
        # Note: .env file may still be loaded, so allow either value
        assert config.BACKEND_TYPE in ["redis", "cloud"]  # Default or .env value
        assert config.REDIS_HOST == "localhost"  # Default value

        # Test with custom env vars
        os.environ['BACKEND_TYPE'] = 'cloud'
        os.environ['API_PORT'] = '9000'

        config2 = Config()
        assert config2.BACKEND_TYPE == "cloud"
        assert config2.API_PORT == 9000

    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)