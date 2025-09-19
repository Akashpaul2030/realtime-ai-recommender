"""
Simple API Testing Script
Tests your FastAPI endpoints directly
"""

import requests
import json
import time
from typing import Dict, Any

class SimpleAPITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def test_endpoint(self, method: str, endpoint: str, data: Dict = None) -> Dict[str, Any]:
        """Test a single endpoint"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()

        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=data, timeout=10)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, timeout=10)
            elif method.upper() == "PUT":
                response = self.session.put(url, json=data, timeout=10)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, timeout=10)

            response_time = time.time() - start_time

            try:
                response_data = response.json()
            except:
                response_data = response.text

            return {
                "success": response.status_code < 400,
                "status_code": response.status_code,
                "response_time": response_time,
                "data": response_data,
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "status_code": 0,
                "response_time": time.time() - start_time,
                "data": None,
                "error": str(e)
            }

    def test_api_health(self):
        """Test basic API health"""
        print("Testing API Health...")

        endpoints = [
            ("GET", "/health", "Health Check"),
            ("GET", "/docs", "API Documentation"),
        ]

        results = []
        for method, endpoint, name in endpoints:
            print(f"  Testing {name}...")
            result = self.test_endpoint(method, endpoint)
            results.append((name, result))

            if result["success"]:
                print(f"    OK {name}: {result['status_code']} ({result['response_time']:.3f}s)")
            else:
                error_msg = result.get('error', f"Status: {result['status_code']}")
                print(f"    FAIL {name}: {error_msg}")
                if result.get('data'):
                    print(f"         Response: {result['data']}")

        return results

    def test_product_crud(self):
        """Test product CRUD operations"""
        print("\nTesting Product CRUD...")

        # Test product
        test_product = {
            "id": f"test-{int(time.time())}",
            "name": "Test Product",
            "description": "A test product for API testing",
            "category": "test",
            "price": 99.99,
            "sku": f"SKU-{int(time.time())}",
            "attributes": {
                "brand": "Test Brand",
                "color": "blue"
            }
        }

        print("  Creating product...")
        create_result = self.test_endpoint("POST", "/products/", test_product)
        if create_result["success"]:
            print(f"    OK Product created: {create_result['status_code']}")
            product_id = test_product["id"]

            # Test read
            print("  Reading product...")
            read_result = self.test_endpoint("GET", f"/products/{product_id}")
            if read_result["success"]:
                print(f"    OK Product read: {read_result['status_code']}")
            else:
                print(f"    FAIL Product read failed: {read_result.get('error', read_result['status_code'])}")

            # Test update
            print("  Updating product...")
            update_data = {"price": 149.99}
            update_result = self.test_endpoint("PUT", f"/products/{product_id}", update_data)
            if update_result["success"]:
                print(f"    OK Product updated: {update_result['status_code']}")
            else:
                print(f"    FAIL Product update failed: {update_result.get('error', update_result['status_code'])}")

            # Test delete
            print("  Deleting product...")
            delete_result = self.test_endpoint("DELETE", f"/products/{product_id}")
            if delete_result["success"]:
                print(f"    OK Product deleted: {delete_result['status_code']}")
            else:
                print(f"    FAIL Product delete failed: {delete_result.get('error', delete_result['status_code'])}")

            return [create_result, read_result, update_result, delete_result]
        else:
            print(f"    FAIL Product creation failed: {create_result.get('error', create_result['status_code'])}")
            if create_result.get('data'):
                print(f"         Response: {create_result['data']}")
            return [create_result]

    def test_search_features(self):
        """Test search functionality"""
        print("\nTesting Search Features...")

        # Test text search
        print("  Testing text search...")
        search_result = self.test_endpoint("GET", "/products/search/text", {"query": "test", "limit": 5})
        if search_result["success"]:
            print(f"    OK Text search: {search_result['status_code']}")
        else:
            print(f"    FAIL Text search failed: {search_result.get('error', search_result['status_code'])}")
            if search_result.get('data'):
                print(f"         Response: {search_result['data']}")

        return [search_result]

    def run_all_tests(self):
        """Run all API tests"""
        print("Starting API Tests")
        print("=" * 50)

        # Check if API is running
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code != 200:
                print(f"API not healthy. Status: {response.status_code}")
                return False
        except:
            print("API not reachable. Please start the API server first:")
            print("   python -m api.app")
            return False

        print("API is reachable!")

        # Run tests
        health_results = self.test_api_health()
        product_results = self.test_product_crud()
        search_results = self.test_search_features()

        # Summary
        all_results = []
        for name, result in health_results:
            all_results.append(result)
        for result in product_results:
            all_results.append(result)
        for result in search_results:
            all_results.append(result)

        successful = sum(1 for result in all_results if result["success"])
        total = len(all_results)

        print(f"\nTest Summary")
        print(f"=" * 30)
        print(f"Total Tests: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {total - successful}")
        print(f"Success Rate: {successful/total*100:.1f}%")

        if successful == total:
            print("All tests passed!")
            return True
        else:
            print("Some tests failed.")
            return False

if __name__ == "__main__":
    tester = SimpleAPITester()
    success = tester.run_all_tests()
    exit(0 if success else 1)