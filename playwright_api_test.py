"""
🎭 Playwright API Testing Suite
Automated testing of the FastAPI backend through Streamlit UI
"""

import asyncio
import time
import json
from playwright.async_api import async_playwright, Page, Browser
import subprocess
import sys
import os
from typing import Dict, List, Any
import requests

class PlaywrightAPITester:
    """Automated testing using Playwright to test API through Streamlit UI"""

    def __init__(self, api_url: str = "http://localhost:8000", streamlit_url: str = "http://localhost:8501"):
        self.api_url = api_url
        self.streamlit_url = streamlit_url
        self.browser: Browser = None
        self.page: Page = None
        self.test_results = []

    async def setup(self):
        """Initialize Playwright browser and page"""
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=False)  # Set to True for headless
        self.page = await self.browser.new_page()

        # Set viewport size
        await self.page.set_viewport_size({"width": 1280, "height": 720})

    async def teardown(self):
        """Close browser and cleanup"""
        if self.browser:
            await self.browser.close()

    async def wait_for_api(self, timeout: int = 30):
        """Wait for API to be available"""
        print(f"🔄 Waiting for API at {self.api_url}...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.api_url}/health", timeout=5)
                if response.status_code == 200:
                    print(f"✅ API is ready at {self.api_url}")
                    return True
            except:
                pass
            await asyncio.sleep(2)

        print(f"❌ API not available after {timeout} seconds")
        return False

    async def wait_for_streamlit(self, timeout: int = 30):
        """Wait for Streamlit to be available"""
        print(f"🔄 Waiting for Streamlit at {self.streamlit_url}...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(self.streamlit_url, timeout=5)
                if response.status_code == 200:
                    print(f"✅ Streamlit is ready at {self.streamlit_url}")
                    return True
            except:
                pass
            await asyncio.sleep(2)

        print(f"❌ Streamlit not available after {timeout} seconds")
        return False

    async def navigate_to_streamlit(self):
        """Navigate to Streamlit app"""
        try:
            await self.page.goto(self.streamlit_url)
            await self.page.wait_for_load_state("networkidle")

            # Wait for Streamlit to fully load
            await self.page.wait_for_selector("h1", timeout=10000)
            print("✅ Successfully navigated to Streamlit app")
            return True
        except Exception as e:
            print(f"❌ Failed to navigate to Streamlit: {e}")
            return False

    async def test_health_check_tab(self):
        """Test the Health Check tab functionality"""
        print("\n🏥 Testing Health Check Tab...")

        try:
            # Click on Health Check tab
            await self.page.click("text=🏥 Health Check")
            await asyncio.sleep(2)

            # Click the test button
            await self.page.click("text=🔍 Test API Health")

            # Wait for results
            await asyncio.sleep(5)

            # Check for success indicators
            success_elements = await self.page.query_selector_all(".test-success")
            failure_elements = await self.page.query_selector_all(".test-failure")

            result = {
                "test": "Health Check Tab",
                "success": len(success_elements) > 0,
                "success_count": len(success_elements),
                "failure_count": len(failure_elements),
                "details": f"Found {len(success_elements)} successful tests, {len(failure_elements)} failed tests"
            }

            self.test_results.append(result)
            print(f"✅ Health Check: {result['details']}")
            return result["success"]

        except Exception as e:
            print(f"❌ Health Check test failed: {e}")
            self.test_results.append({
                "test": "Health Check Tab",
                "success": False,
                "error": str(e)
            })
            return False

    async def test_product_apis_tab(self):
        """Test the Product APIs tab functionality"""
        print("\n🛍️ Testing Product APIs Tab...")

        try:
            # Click on Product APIs tab
            await self.page.click("text=🛍️ Product APIs")
            await asyncio.sleep(2)

            # Click the test button
            await self.page.click("text=🧪 Run Product API Tests")

            # Wait for tests to complete (this might take a while)
            await asyncio.sleep(15)

            # Look for success messages
            page_content = await self.page.content()
            success_count = page_content.count("✅")
            failure_count = page_content.count("❌")
            warning_count = page_content.count("⚠️")

            result = {
                "test": "Product APIs Tab",
                "success": success_count > failure_count,
                "success_indicators": success_count,
                "failure_indicators": failure_count,
                "warning_indicators": warning_count,
                "details": f"Success: {success_count}, Failures: {failure_count}, Warnings: {warning_count}"
            }

            self.test_results.append(result)
            print(f"✅ Product APIs: {result['details']}")
            return result["success"]

        except Exception as e:
            print(f"❌ Product APIs test failed: {e}")
            self.test_results.append({
                "test": "Product APIs Tab",
                "success": False,
                "error": str(e)
            })
            return False

    async def test_recommendations_tab(self):
        """Test the Recommendations tab functionality"""
        print("\n🎯 Testing Recommendations Tab...")

        try:
            # Click on Recommendations tab
            await self.page.click("text=🎯 Recommendations")
            await asyncio.sleep(2)

            # Click the test button
            await self.page.click("text=🧪 Test Recommendation APIs")

            # Wait for tests to complete
            await asyncio.sleep(10)

            # Check for results
            page_content = await self.page.content()
            success_count = page_content.count("✅")
            warning_count = page_content.count("⚠️")

            result = {
                "test": "Recommendations Tab",
                "success": success_count > 0,  # At least some recommendations should work
                "success_indicators": success_count,
                "warning_indicators": warning_count,
                "details": f"Success: {success_count}, Warnings: {warning_count}"
            }

            self.test_results.append(result)
            print(f"✅ Recommendations: {result['details']}")
            return result["success"]

        except Exception as e:
            print(f"❌ Recommendations test failed: {e}")
            self.test_results.append({
                "test": "Recommendations Tab",
                "success": False,
                "error": str(e)
            })
            return False

    async def test_integration_suite_tab(self):
        """Test the Integration Suite tab functionality"""
        print("\n🔬 Testing Integration Suite Tab...")

        try:
            # Click on Integration Suite tab
            await self.page.click("text=🔬 Integration Suite")
            await asyncio.sleep(2)

            # Click the test button
            await self.page.click("text=🚀 Run Full Integration Tests")

            # Wait for tests to complete (this takes the longest)
            await asyncio.sleep(20)

            # Look for the overall status
            try:
                # Try to find success metrics
                success_metric = await self.page.query_selector("text=/Tests Passed/")
                if success_metric:
                    metric_text = await success_metric.text_content()
                    print(f"📊 Integration results: {metric_text}")

                # Check for overall status
                page_content = await self.page.content()
                overall_pass = "🟢 PASS" in page_content
                overall_partial = "🟡 PARTIAL" in page_content

                result = {
                    "test": "Integration Suite Tab",
                    "success": overall_pass or overall_partial,
                    "overall_status": "PASS" if overall_pass else "PARTIAL" if overall_partial else "FAIL",
                    "details": f"Integration suite completed with status: {'PASS' if overall_pass else 'PARTIAL' if overall_partial else 'FAIL'}"
                }

            except:
                # Fallback to counting success indicators
                success_count = page_content.count("✅")
                result = {
                    "test": "Integration Suite Tab",
                    "success": success_count > 0,
                    "success_indicators": success_count,
                    "details": f"Found {success_count} successful indicators"
                }

            self.test_results.append(result)
            print(f"✅ Integration Suite: {result['details']}")
            return result["success"]

        except Exception as e:
            print(f"❌ Integration Suite test failed: {e}")
            self.test_results.append({
                "test": "Integration Suite Tab",
                "success": False,
                "error": str(e)
            })
            return False

    async def take_screenshot(self, name: str):
        """Take a screenshot for debugging"""
        try:
            await self.page.screenshot(path=f"test_screenshot_{name}_{int(time.time())}.png")
            print(f"📸 Screenshot saved: test_screenshot_{name}_{int(time.time())}.png")
        except Exception as e:
            print(f"❌ Failed to take screenshot: {e}")

    async def run_comprehensive_test(self):
        """Run all tests in sequence"""
        print("🎭 Starting Playwright API Testing Suite...")
        print("=" * 60)

        # Setup
        await self.setup()

        try:
            # Wait for services
            if not await self.wait_for_api():
                print("❌ API is not available. Please start the API server first.")
                return False

            if not await self.wait_for_streamlit():
                print("❌ Streamlit is not available. Please start Streamlit first.")
                return False

            # Navigate to Streamlit
            if not await self.navigate_to_streamlit():
                return False

            await self.take_screenshot("initial_load")

            # Run individual tests
            tests = [
                self.test_health_check_tab,
                self.test_product_apis_tab,
                self.test_recommendations_tab,
                self.test_integration_suite_tab
            ]

            test_results = []
            for test in tests:
                try:
                    result = await test()
                    test_results.append(result)
                    await asyncio.sleep(2)  # Brief pause between tests
                except Exception as e:
                    print(f"❌ Test failed with exception: {e}")
                    test_results.append(False)

            # Final screenshot
            await self.take_screenshot("final_results")

            # Print summary
            self.print_test_summary()

            return all(test_results)

        finally:
            await self.teardown()

    def print_test_summary(self):
        """Print a summary of all test results"""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)

        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result.get("success", False))

        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")
        print(f"Success Rate: {successful_tests/total_tests*100:.1f}%" if total_tests > 0 else "N/A")

        print("\nDetailed Results:")
        for result in self.test_results:
            status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
            test_name = result.get("test", "Unknown Test")
            details = result.get("details", result.get("error", "No details"))
            print(f"  {status} {test_name}: {details}")

        overall_status = "🟢 ALL TESTS PASSED" if successful_tests == total_tests else \
                        f"🟡 {successful_tests}/{total_tests} TESTS PASSED" if successful_tests > 0 else \
                        "🔴 ALL TESTS FAILED"
        print(f"\nOverall Status: {overall_status}")
        print("=" * 60)

def start_services():
    """Helper function to start API and Streamlit services"""
    print("🚀 Starting services...")

    # Start API server
    api_process = subprocess.Popen([
        sys.executable, "-m", "api.app"
    ], cwd=os.getcwd())

    # Start Streamlit
    streamlit_process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "streamlit_api_tester.py", "--server.port=8501"
    ], cwd=os.getcwd())

    return api_process, streamlit_process

async def main():
    """Main function to run the tests"""
    import argparse

    parser = argparse.ArgumentParser(description="Playwright API Testing Suite")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--streamlit-url", default="http://localhost:8501", help="Streamlit URL")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--start-services", action="store_true", help="Automatically start API and Streamlit services")

    args = parser.parse_args()

    # Start services if requested
    if args.start_services:
        api_process, streamlit_process = start_services()
        time.sleep(10)  # Give services time to start

        try:
            # Run tests
            tester = PlaywrightAPITester(args.api_url, args.streamlit_url)
            success = await tester.run_comprehensive_test()

            # Return appropriate exit code
            sys.exit(0 if success else 1)

        finally:
            # Clean up processes
            api_process.terminate()
            streamlit_process.terminate()
    else:
        # Just run tests (assumes services are already running)
        tester = PlaywrightAPITester(args.api_url, args.streamlit_url)
        success = await tester.run_comprehensive_test()

        # Return appropriate exit code
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())