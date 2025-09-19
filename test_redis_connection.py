#!/usr/bin/env python3
"""
Quick Redis Connection Test
"""

import redis
import sys

def test_redis_connection():
    """Test if Redis is accessible"""
    try:
        # Try to connect to Redis
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

        # Test basic operations
        r.ping()
        print("[SUCCESS] Redis connection successful!")

        # Test set/get
        r.set("test_key", "hello_redis")
        value = r.get("test_key")
        print(f"[SUCCESS] Redis set/get test: {value}")

        # Clean up
        r.delete("test_key")

        # Get Redis info
        info = r.info()
        print(f"[SUCCESS] Redis version: {info.get('redis_version', 'unknown')}")
        print(f"[SUCCESS] Connected clients: {info.get('connected_clients', 'unknown')}")

        return True

    except redis.ConnectionError:
        print("[FAILED] Redis connection failed: Connection refused")
        print("   Make sure Redis is running on localhost:6379")
        return False

    except Exception as e:
        print(f"[FAILED] Redis test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Redis connection...")
    success = test_redis_connection()

    if success:
        print("\nRedis is ready! You can now start the FastAPI server.")
        print("\nNext steps:")
        print("1. python -m api.app")
        print("2. streamlit run streamlit_api_tester.py")
    else:
        print("\nRedis setup required!")
        print("\nDocker commands to try:")
        print("docker run -p 6379:6379 -d --name redis-server redislabs/redismod")
        print("docker run -p 6379:6379 -d --name redis-stack redis/redis-stack:latest")

    sys.exit(0 if success else 1)