import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Stream Configuration
PRODUCT_STREAM_KEY = os.getenv("PRODUCT_STREAM_KEY", "product:updates")
PRODUCT_STREAM_GROUP = os.getenv("PRODUCT_STREAM_GROUP", "product-processors")
PRODUCT_STREAM_CONSUMER = os.getenv("PRODUCT_STREAM_CONSUMER", "worker-{}")

# Vector Store Configuration
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", 384))  # Dimension from all-MiniLM-L6-v2
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "product:vectors")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.75))

# Model Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./model_cache")

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")