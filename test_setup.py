#!/usr/bin/env python3
"""
Quick Setup Test Script
Tests your API and UI components to ensure everything works
"""

import os
import sys
import time
import subprocess
import requests
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")

    tests = []

    # Test FastAPI import
    try:
        from fastapi import FastAPI
        tests.append(("FastAPI", True, "OK"))
    except ImportError as e:
        tests.append(("FastAPI", False, f"ERROR: {e}"))

    # Test Streamlit import
    try:
        import streamlit
        tests.append(("Streamlit", True, "OK"))
    except ImportError as e:
        tests.append(("Streamlit", False, f"ERROR: {e}"))

    # Test project modules
    try:
        from config import API_HOST, API_PORT
        tests.append(("Config", True, "OK"))
    except ImportError as e:
        tests.append(("Config", False, f"ERROR: {e}"))

    # Test Redis
    try:
        import redis
        tests.append(("Redis", True, "OK"))
    except ImportError as e:
        tests.append(("Redis", False, f"ERROR: {e}"))

    # Test ML libraries
    try:
        import numpy, pandas
        tests.append(("NumPy/Pandas", True, "OK"))
    except ImportError as e:
        tests.append(("NumPy/Pandas", False, f"ERROR: {e}"))

    print("\nImport Test Results:")
    for name, success, message in tests:
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {name}: {message}")

    return all(success for _, success, _ in tests)

def test_api_startup():
    """Test if the API can start up"""
    print("\nTesting API startup...")

    try:
        # Try to import the FastAPI app
        from api.app import app
        print("[PASS] FastAPI app imported successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Failed to import FastAPI app: {e}")
        return False

def test_streamlit_apps():
    """Test if Streamlit apps can be loaded"""
    print("\nTesting Streamlit apps...")

    streamlit_files = [
        "streamlit_app.py",
        "streamlit_api_tester.py"
    ]

    results = []
    for file in streamlit_files:
        if os.path.exists(file):
            try:
                # Try to compile the file (basic syntax check)
                with open(file, 'r', encoding='utf-8') as f:
                    compile(f.read(), file, 'exec')
                results.append((file, True, "OK"))
            except Exception as e:
                results.append((file, False, f"ERROR: {e}"))
        else:
            results.append((file, False, "File not found"))

    for file, success, message in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {file}: {message}")

    return all(success for _, success, _ in results)

def check_ports():
    """Check if required ports are available"""
    print("\nChecking ports...")

    ports_to_check = [8000, 8501]  # FastAPI and Streamlit default ports

    for port in ports_to_check:
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()

            if result == 0:
                print(f"  [WARN] Port {port} is already in use")
            else:
                print(f"  [OK] Port {port} is available")
        except Exception as e:
            print(f"  [ERROR] Error checking port {port}: {e}")

def create_test_commands():
    """Generate test commands for the user"""
    print("\nTest Commands:")
    print("\n1. Start FastAPI Server:")
    print("   python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload")

    print("\n2. Start Streamlit API Tester:")
    print("   streamlit run streamlit_api_tester.py --server.port 8501")

    print("\n3. Start Original Streamlit App:")
    print("   streamlit run streamlit_app.py --server.port 8502")

    print("\n4. Test API Health (after starting FastAPI):")
    print("   curl http://localhost:8000/health")

    print("\n5. View API Documentation:")
    print("   Open: http://localhost:8000/docs")

def main():
    """Run all tests"""
    print("Testing Your AI Recommendation System Setup")
    print("=" * 50)

    # Run tests
    imports_ok = test_imports()
    api_ok = test_api_startup()
    streamlit_ok = test_streamlit_apps()

    check_ports()
    create_test_commands()

    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"  Imports: {'PASS' if imports_ok else 'FAIL'}")
    print(f"  API Startup: {'PASS' if api_ok else 'FAIL'}")
    print(f"  Streamlit Apps: {'PASS' if streamlit_ok else 'FAIL'}")

    if all([imports_ok, api_ok, streamlit_ok]):
        print("\nAll tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Start the FastAPI server with the command above")
        print("2. In a new terminal, start the Streamlit tester")
        print("3. Use the tester to verify API + UI integration")
    else:
        print("\nSome tests failed. Please check the errors above.")

        if not imports_ok:
            print("Missing dependencies? Run: pip install -r requirements.txt")

        if not api_ok:
            print("API issues? Check your config.py and ensure Redis is running")

if __name__ == "__main__":
    main()