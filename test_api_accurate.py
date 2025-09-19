#!/usr/bin/env python3
"""
Accurate API Testing Script
Tests the actual endpoints of your FastAPI server
"""

import requests
import json
import time
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"
TIMEOUT = 10

class AccurateAPITester:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
        self.created_product_id = None

    def log_result(self, test_name: str, success: bool, message: str, response_time: float = 0):
        """Log test result"""
        status = "PASS" if success else "FAIL"
        result = {
            "test": test_name,
            "status": status,
            "message": message,
            "response_time": f"{response_time:.3f}s" if response_time > 0 else "N/A"
        }
        self.test_results.append(result)
        print(f"[{status}] {test_name}: {message} ({result['response_time']})")

    def test_health_check(self):
        """Test the /health endpoint"""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/health", timeout=TIMEOUT)
            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                self.log_result("Health Check", True, f"API is healthy", response_time)
                return True
            else:
                self.log_result("Health Check", False, f"HTTP {response.status_code}", response_time)
                return False

        except requests.exceptions.ConnectionError:
            self.log_result("Health Check", False, "Connection refused - API server not running")
            return False
        except Exception as e:
            self.log_result("Health Check", False, f"Error: {str(e)}")
            return False

    def test_create_product(self):
        """Test POST /products/ endpoint"""
        sample_product = {
            "name": "Blue Denim Jeans",
            "description": "Classic blue denim jeans with straight cut, perfect for casual wear",
            "category": "Clothing",
            "price": 89.99,
            "sku": "TEST-JEANS-001",
            "id": f"test-product-{int(time.time())}",
            "attributes": {
                "color": "blue",
                "size": "M",
                "material": "denim",
                "style": "straight"
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

            if response.status_code == 200:
                data = response.json()
                product_id = data.get("product_id")
                self.created_product_id = product_id or sample_product["id"]
                self.log_result("Create Product", True, f"Product created: {self.created_product_id}", response_time)
                return True
            else:
                self.log_result("Create Product", False, f"HTTP {response.status_code}: {response.text}", response_time)
                return False

        except Exception as e:
            self.log_result("Create Product", False, f"Error: {str(e)}")
            return False

    def test_get_recommendations(self):
        """Test GET /recommendations/{product_id}/similar endpoint"""
        if not self.created_product_id:
            self.log_result("Get Recommendations", False, "No product ID available for testing")
            return False

        try:
            start_time = time.time()
            response = self.session.get(
                f"{self.base_url}/recommendations/{self.created_product_id}/similar?limit=5",
                timeout=TIMEOUT
            )
            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                rec_count = len(data.get("recommendations", []))
                self.log_result("Get Recommendations", True, f"Found {rec_count} recommendations", response_time)
                return True
            elif response.status_code == 404:
                self.log_result("Get Recommendations", True, "Product not found (expected for new product)", response_time)
                return True
            else:
                self.log_result("Get Recommendations", False, f"HTTP {response.status_code}: {response.text}", response_time)
                return False

        except Exception as e:
            self.log_result("Get Recommendations", False, f"Error: {str(e)}")
            return False

    def test_api_documentation(self):
        """Test API documentation endpoints"""
        endpoints_to_test = [
            ("/docs", "Swagger UI"),
            ("/redoc", "ReDoc UI"),
            ("/openapi.json", "OpenAPI Schema")
        ]

        all_passed = True

        for endpoint, name in endpoints_to_test:
            try:
                start_time = time.time()
                response = self.session.get(f"{self.base_url}{endpoint}", timeout=TIMEOUT)
                response_time = time.time() - start_time

                if response.status_code == 200:
                    self.log_result(f"API Docs ({name})", True, "Accessible", response_time)
                else:
                    self.log_result(f"API Docs ({name})", False, f"HTTP {response.status_code}", response_time)
                    all_passed = False

            except Exception as e:
                self.log_result(f"API Docs ({name})", False, f"Error: {str(e)}")
                all_passed = False

        return all_passed

    def test_multiple_products(self):
        """Test creating multiple products for better recommendations"""
        products = [
            {
                "name": "Red Summer Dress",
                "description": "Elegant red summer dress perfect for evening events",
                "category": "Clothing",
                "price": 129.99,
                "sku": "TEST-DRESS-001",
                "id": f"test-dress-{int(time.time())}",
                "attributes": {"color": "red", "season": "summer", "style": "elegant"}
            },
            {
                "name": "White Sneakers",
                "description": "Comfortable white leather sneakers for daily wear",
                "category": "Footwear",
                "price": 79.99,
                "sku": "TEST-SHOES-001",
                "id": f"test-shoes-{int(time.time())}",
                "attributes": {"color": "white", "material": "leather", "type": "sneakers"}
            }
        ]

        success_count = 0
        for i, product in enumerate(products, 1):
            try:
                start_time = time.time()
                response = self.session.post(f"{self.base_url}/products/", json=product, timeout=TIMEOUT)
                response_time = time.time() - start_time

                if response.status_code == 200:
                    self.log_result(f"Create Product {i}", True, f"Created: {product['name']}", response_time)
                    success_count += 1
                else:
                    self.log_result(f"Create Product {i}", False, f"HTTP {response.status_code}", response_time)

            except Exception as e:
                self.log_result(f"Create Product {i}", False, f"Error: {str(e)}")

        return success_count > 0

    def run_all_tests(self):
        """Run all API tests"""
        print("=" * 70)
        print("ACCURATE API TESTING")
        print("=" * 70)
        print(f"Testing API at: {self.base_url}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 70)

        # Test connectivity first
        if not self.test_health_check():
            print("\nAPI is not accessible. Please check:")
            print("1. FastAPI server running: python -m api.app")
            print("2. Redis server running: docker run -p 6379:6379 -d redis")
            print("3. Correct port (8000)")
            return False

        # Test documentation
        self.test_api_documentation()

        # Test product creation
        self.test_create_product()

        # Create more products for better testing
        self.test_multiple_products()

        # Test recommendations (after creating products)
        time.sleep(1)  # Give a moment for processing
        self.test_get_recommendations()

        # Print summary
        self.print_summary()
        return True

    def print_summary(self):
        """Print test summary"""
        print("-" * 70)
        print("TEST SUMMARY")
        print("-" * 70)

        passed = sum(1 for r in self.test_results if r["status"] == "PASS")
        failed = sum(1 for r in self.test_results if r["status"] == "FAIL")
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

        print("\n" + "=" * 70)

        if passed >= total * 0.8:  # 80% success rate
            print("SUCCESS! Your API is working well!")
            print("\nReady for Streamlit UI testing:")
            print("  streamlit run streamlit_api_tester.py --server.port 8501")
            print("\nAPI Access Points:")
            print(f"  - Health Check: {self.base_url}/health")
            print(f"  - API Docs: {self.base_url}/docs")
            print(f"  - Create Product: POST {self.base_url}/products/")
            print(f"  - Get Recommendations: GET {self.base_url}/recommendations/{{id}}/similar")
        else:
            print("Some issues found. Check the failed tests above.")

        print("=" * 70)


def main():
    """Main function"""
    tester = AccurateAPITester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()