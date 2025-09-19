#!/usr/bin/env python3
"""
Comprehensive API Testing Script
Tests all endpoints of the AI Fashion Search API
"""

import requests
import json
import time
from typing import Dict, List, Any
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"
TIMEOUT = 10

class APITester:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []

    def log_result(self, test_name: str, success: bool, message: str, response_time: float = 0):
        """Log test result"""
        status = "PASS" if success else "FAIL"
        result = {
            "test": test_name,
            "status": status,
            "message": message,
            "response_time": f"{response_time:.3f}s" if response_time > 0 else "N/A",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        self.test_results.append(result)
        print(f"[{status}] {test_name}: {message} ({result['response_time']})")

    def test_health_check(self):
        """Test the health endpoint"""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/health", timeout=TIMEOUT)
            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                self.log_result("Health Check", True, f"API is healthy - {data.get('status', 'unknown')}", response_time)
                return True
            else:
                self.log_result("Health Check", False, f"HTTP {response.status_code}: {response.text}", response_time)
                return False

        except requests.exceptions.ConnectionError:
            self.log_result("Health Check", False, "Connection refused - Is the API server running?")
            return False
        except Exception as e:
            self.log_result("Health Check", False, f"Error: {str(e)}")
            return False

    def test_root_endpoint(self):
        """Test the root endpoint"""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/", timeout=TIMEOUT)
            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                self.log_result("Root Endpoint", True, f"Welcome message received", response_time)
                return True
            else:
                self.log_result("Root Endpoint", False, f"HTTP {response.status_code}", response_time)
                return False

        except Exception as e:
            self.log_result("Root Endpoint", False, f"Error: {str(e)}")
            return False

    def test_create_product(self):
        """Test creating a product"""
        sample_product = {
            "name": "Blue Denim Jeans",
            "description": "Classic blue denim jeans with straight cut",
            "category": "Clothing",
            "price": 89.99,
            "sku": "TEST-JEANS-001",
            "attributes": {
                "color": "blue",
                "size": "M",
                "material": "denim"
            }
        }

        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/products/",
                json=sample_product,
                timeout=TIMEOUT
            )
            response_time = time.time() - start_time

            if response.status_code == 201:
                data = response.json()
                product_id = data.get("id")
                self.log_result("Create Product", True, f"Product created with ID: {product_id}", response_time)
                return product_id
            else:
                self.log_result("Create Product", False, f"HTTP {response.status_code}: {response.text}", response_time)
                return None

        except Exception as e:
            self.log_result("Create Product", False, f"Error: {str(e)}")
            return None

    def test_get_product(self, product_id: str):
        """Test getting a product by ID"""
        if not product_id:
            self.log_result("Get Product", False, "No product ID provided")
            return False

        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/products/{product_id}", timeout=TIMEOUT)
            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                self.log_result("Get Product", True, f"Retrieved product: {data.get('name', 'unknown')}", response_time)
                return True
            elif response.status_code == 404:
                self.log_result("Get Product", False, "Product not found", response_time)
                return False
            else:
                self.log_result("Get Product", False, f"HTTP {response.status_code}: {response.text}", response_time)
                return False

        except Exception as e:
            self.log_result("Get Product", False, f"Error: {str(e)}")
            return False

    def test_search_recommendations(self):
        """Test product recommendations"""
        search_queries = [
            {"query": "blue jeans", "limit": 5},
            {"query": "casual wear", "limit": 3},
            {"query": "denim clothing", "limit": 10}
        ]

        success_count = 0

        for i, search_data in enumerate(search_queries, 1):
            try:
                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}/recommendations/",
                    json=search_data,
                    timeout=TIMEOUT
                )
                response_time = time.time() - start_time

                if response.status_code == 200:
                    data = response.json()
                    rec_count = len(data.get("recommendations", []))
                    self.log_result(
                        f"Search {i}",
                        True,
                        f"Query '{search_data['query']}' returned {rec_count} results",
                        response_time
                    )
                    success_count += 1
                else:
                    self.log_result(
                        f"Search {i}",
                        False,
                        f"HTTP {response.status_code}: {response.text}",
                        response_time
                    )

            except Exception as e:
                self.log_result(f"Search {i}", False, f"Error: {str(e)}")

        return success_count == len(search_queries)

    def test_list_products(self):
        """Test listing all products"""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/products/", timeout=TIMEOUT)
            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                product_count = len(data) if isinstance(data, list) else data.get("count", 0)
                self.log_result("List Products", True, f"Found {product_count} products", response_time)
                return True
            else:
                self.log_result("List Products", False, f"HTTP {response.status_code}: {response.text}", response_time)
                return False

        except Exception as e:
            self.log_result("List Products", False, f"Error: {str(e)}")
            return False

    def test_api_docs(self):
        """Test API documentation endpoint"""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/docs", timeout=TIMEOUT)
            response_time = time.time() - start_time

            if response.status_code == 200:
                self.log_result("API Docs", True, "Documentation is accessible", response_time)
                return True
            else:
                self.log_result("API Docs", False, f"HTTP {response.status_code}", response_time)
                return False

        except Exception as e:
            self.log_result("API Docs", False, f"Error: {str(e)}")
            return False

    def test_openapi_spec(self):
        """Test OpenAPI specification"""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/openapi.json", timeout=TIMEOUT)
            response_time = time.time() - start_time

            if response.status_code == 200:
                spec = response.json()
                endpoints = len(spec.get("paths", {}))
                self.log_result("OpenAPI Spec", True, f"API spec with {endpoints} endpoints", response_time)
                return True
            else:
                self.log_result("OpenAPI Spec", False, f"HTTP {response.status_code}", response_time)
                return False

        except Exception as e:
            self.log_result("OpenAPI Spec", False, f"Error: {str(e)}")
            return False

    def run_all_tests(self):
        """Run comprehensive API tests"""
        print("=" * 60)
        print("COMPREHENSIVE API TESTING")
        print("=" * 60)
        print(f"Testing API at: {self.base_url}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)

        # Test basic connectivity
        if not self.test_health_check():
            print("\nAPI is not accessible. Please check:")
            print("1. Is the FastAPI server running? (python -m api.app)")
            print("2. Is Redis running? (docker run -p 6379:6379 -d redis)")
            print("3. Is the server on the correct port?")
            return False

        self.test_root_endpoint()
        self.test_api_docs()
        self.test_openapi_spec()

        # Test product operations
        product_id = self.test_create_product()
        if product_id:
            self.test_get_product(product_id)

        self.test_list_products()

        # Test search functionality
        self.test_search_recommendations()

        # Summary
        self.print_summary()
        return True

    def print_summary(self):
        """Print test results summary"""
        print("-" * 60)
        print("TEST SUMMARY")
        print("-" * 60)

        passed = sum(1 for result in self.test_results if result["status"] == "PASS")
        failed = sum(1 for result in self.test_results if result["status"] == "FAIL")
        total = len(self.test_results)

        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {(passed/total*100):.1f}%")

        if failed > 0:
            print("\nFailed Tests:")
            for result in self.test_results:
                if result["status"] == "FAIL":
                    print(f"  - {result['test']}: {result['message']}")

        print("\n" + "=" * 60)

        if failed == 0:
            print("All tests passed! Your API is working perfectly!")
            print("\nNext steps:")
            print("1. Open Streamlit UI: streamlit run streamlit_api_tester.py")
            print("2. View API docs: http://localhost:8000/docs")
            print("3. Test the UI with real product data")
        else:
            print("Some tests failed. Check the errors above.")

        print("=" * 60)


def main():
    """Main function"""
    tester = APITester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()