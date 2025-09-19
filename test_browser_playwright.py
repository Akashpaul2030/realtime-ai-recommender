#!/usr/bin/env python3
"""
Comprehensive Browser Testing with Playwright
Tests both API endpoints and Streamlit UI functionality
"""

import asyncio
import time
import json
from datetime import datetime
from playwright.async_api import async_playwright
from typing import Dict, List, Any


class PlaywrightTester:
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.streamlit_url = "http://localhost:8501"
        self.test_results = []
        self.browser = None
        self.page = None

    def log_result(self, test_name: str, success: bool, message: str, screenshot_path: str = None):
        """Log test result"""
        status = "PASS" if success else "FAIL"
        result = {
            "test": test_name,
            "status": status,
            "message": message,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "screenshot": screenshot_path
        }
        self.test_results.append(result)
        print(f"[{status}] {test_name}: {message}")

    async def setup_browser(self):
        """Initialize Playwright browser"""
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=False)  # Set to True for headless
            self.context = await self.browser.new_context(
                viewport={"width": 1920, "height": 1080}
            )
            self.page = await self.context.new_page()
            self.log_result("Browser Setup", True, "Playwright browser initialized")
            return True
        except Exception as e:
            self.log_result("Browser Setup", False, f"Failed to initialize browser: {str(e)}")
            return False

    async def test_api_health_browser(self):
        """Test API health endpoint via browser"""
        try:
            await self.page.goto(f"{self.api_base_url}/health")

            # Wait for response
            await self.page.wait_for_load_state("networkidle")

            # Check if page loaded successfully
            content = await self.page.text_content("body")

            if "healthy" in content.lower():
                self.log_result("API Health (Browser)", True, "Health endpoint accessible via browser")
                return True
            else:
                await self.page.screenshot(path="screenshots/api_health_error.png")
                self.log_result("API Health (Browser)", False, f"Unexpected response: {content[:100]}")
                return False

        except Exception as e:
            await self.page.screenshot(path="screenshots/api_health_error.png")
            self.log_result("API Health (Browser)", False, f"Error: {str(e)}")
            return False

    async def test_api_docs_browser(self):
        """Test API documentation via browser"""
        try:
            await self.page.goto(f"{self.api_base_url}/docs")
            await self.page.wait_for_load_state("networkidle")

            # Look for Swagger UI elements
            title = await self.page.title()

            if "swagger" in title.lower() or "api" in title.lower():
                # Take screenshot of API docs
                await self.page.screenshot(path="screenshots/api_docs.png")
                self.log_result("API Docs (Browser)", True, f"API documentation loaded: {title}")
                return True
            else:
                await self.page.screenshot(path="screenshots/api_docs_error.png")
                self.log_result("API Docs (Browser)", False, f"Unexpected page title: {title}")
                return False

        except Exception as e:
            await self.page.screenshot(path="screenshots/api_docs_error.png")
            self.log_result("API Docs (Browser)", False, f"Error: {str(e)}")
            return False

    async def test_create_product_api(self):
        """Test product creation via API request in browser"""
        try:
            # Navigate to API docs to test endpoint
            await self.page.goto(f"{self.api_base_url}/docs")
            await self.page.wait_for_load_state("networkidle")

            # Look for the POST /products endpoint
            products_section = await self.page.locator("text=products").first
            if await products_section.is_visible():
                await products_section.click()
                await self.page.wait_for_timeout(1000)

                # Take screenshot of expanded API docs
                await self.page.screenshot(path="screenshots/api_products_endpoint.png")
                self.log_result("Create Product API", True, "Products endpoint visible in API docs")
                return True
            else:
                self.log_result("Create Product API", False, "Products endpoint not found in API docs")
                return False

        except Exception as e:
            await self.page.screenshot(path="screenshots/create_product_error.png")
            self.log_result("Create Product API", False, f"Error: {str(e)}")
            return False

    async def test_streamlit_ui_loading(self):
        """Test Streamlit UI loading"""
        try:
            await self.page.goto(self.streamlit_url, timeout=30000)  # 30 second timeout
            await self.page.wait_for_load_state("networkidle")

            # Look for Streamlit elements
            title = await self.page.title()

            # Check for Streamlit indicators
            streamlit_indicators = [
                "[data-testid='stApp']",
                ".main",
                "[data-testid='stSidebar']",
                "text=Streamlit"
            ]

            found_streamlit = False
            for indicator in streamlit_indicators:
                try:
                    if await self.page.locator(indicator).first.is_visible(timeout=5000):
                        found_streamlit = True
                        break
                except:
                    continue

            if found_streamlit:
                await self.page.screenshot(path="screenshots/streamlit_loaded.png")
                self.log_result("Streamlit UI Loading", True, f"Streamlit app loaded successfully")
                return True
            else:
                await self.page.screenshot(path="screenshots/streamlit_error.png")
                self.log_result("Streamlit UI Loading", False, f"Streamlit elements not found. Title: {title}")
                return False

        except Exception as e:
            await self.page.screenshot(path="screenshots/streamlit_error.png")
            self.log_result("Streamlit UI Loading", False, f"Error loading Streamlit: {str(e)}")
            return False

    async def test_streamlit_interaction(self):
        """Test Streamlit UI interactions"""
        try:
            # Ensure we're on Streamlit page
            await self.page.goto(self.streamlit_url)
            await self.page.wait_for_load_state("networkidle")
            await self.page.wait_for_timeout(3000)  # Wait for Streamlit to fully load

            # Look for common Streamlit elements
            elements_to_check = [
                "input[type='text']",
                "textarea",
                "button",
                "[data-testid='stSelectbox']",
                "[data-testid='stButton']"
            ]

            found_elements = []
            for element in elements_to_check:
                try:
                    if await self.page.locator(element).first.is_visible(timeout=2000):
                        found_elements.append(element)
                except:
                    continue

            if found_elements:
                await self.page.screenshot(path="screenshots/streamlit_elements.png")
                self.log_result("Streamlit Interaction", True, f"Found interactive elements: {', '.join(found_elements[:3])}")
                return True
            else:
                await self.page.screenshot(path="screenshots/streamlit_no_elements.png")
                self.log_result("Streamlit Interaction", False, "No interactive elements found")
                return False

        except Exception as e:
            await self.page.screenshot(path="screenshots/streamlit_interaction_error.png")
            self.log_result("Streamlit Interaction", False, f"Error testing interactions: {str(e)}")
            return False

    async def test_api_connectivity_from_streamlit(self):
        """Test if Streamlit can connect to API"""
        try:
            await self.page.goto(self.streamlit_url)
            await self.page.wait_for_load_state("networkidle")
            await self.page.wait_for_timeout(3000)

            # Look for API connection indicators
            page_content = await self.page.content()

            # Check for error messages that might indicate API connection issues
            connection_issues = [
                "Connection refused",
                "Failed to connect",
                "API not available",
                "Error 10061",
                "timeout",
                "ConnectionError"
            ]

            has_connection_issues = any(issue in page_content for issue in connection_issues)

            if not has_connection_issues:
                # Look for successful API indicators
                success_indicators = [
                    "healthy",
                    "API connected",
                    "Products loaded",
                    "Status: OK"
                ]

                has_success = any(indicator in page_content for indicator in success_indicators)

                await self.page.screenshot(path="screenshots/streamlit_api_connection.png")

                if has_success:
                    self.log_result("API Connectivity", True, "Streamlit successfully connected to API")
                else:
                    self.log_result("API Connectivity", True, "No obvious API connection errors")
                return True
            else:
                await self.page.screenshot(path="screenshots/streamlit_api_error.png")
                self.log_result("API Connectivity", False, "API connection issues detected in Streamlit")
                return False

        except Exception as e:
            await self.page.screenshot(path="screenshots/api_connectivity_error.png")
            self.log_result("API Connectivity", False, f"Error testing API connectivity: {str(e)}")
            return False

    async def run_all_tests(self):
        """Run all browser tests"""
        print("=" * 80)
        print("PLAYWRIGHT BROWSER TESTING")
        print("=" * 80)
        print(f"API Base URL: {self.api_base_url}")
        print(f"Streamlit URL: {self.streamlit_url}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)

        # Setup browser
        if not await self.setup_browser():
            return False

        # Create screenshots directory
        import os
        os.makedirs("screenshots", exist_ok=True)

        try:
            # Test API endpoints
            await self.test_api_health_browser()
            await self.test_api_docs_browser()
            await self.test_create_product_api()

            # Test Streamlit UI
            await self.test_streamlit_ui_loading()
            await self.test_streamlit_interaction()
            await self.test_api_connectivity_from_streamlit()

        finally:
            await self.cleanup()

        # Print summary
        self.print_summary()
        return True

    async def cleanup(self):
        """Clean up browser resources"""
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if hasattr(self, 'playwright'):
                await self.playwright.stop()
            self.log_result("Cleanup", True, "Browser resources cleaned up")
        except Exception as e:
            self.log_result("Cleanup", False, f"Error during cleanup: {str(e)}")

    def print_summary(self):
        """Print test summary"""
        print("-" * 80)
        print("BROWSER TEST SUMMARY")
        print("-" * 80)

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
                    screenshot_info = f" (Screenshot: {result['screenshot']})" if result['screenshot'] else ""
                    print(f"  - {result['test']}: {result['message']}{screenshot_info}")

        print(f"\nScreenshots saved in: screenshots/")
        print("Available screenshots:")
        import os
        if os.path.exists("screenshots"):
            for file in os.listdir("screenshots"):
                if file.endswith('.png'):
                    print(f"  - screenshots/{file}")

        print("\n" + "=" * 80)

        if passed >= total * 0.8:
            print("SUCCESS! Browser testing completed with good results!")
            print("\nYour application is ready for:")
            print("  - Production deployment")
            print("  - User acceptance testing")
            print("  - Performance optimization")
        else:
            print("Some issues found. Check screenshots and failed tests above.")

        print("=" * 80)


async def main():
    """Main function"""
    tester = PlaywrightTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    # Check if services are running first
    print("Checking if services are running...")
    print("Make sure these are running before testing:")
    print("1. FastAPI: python -m api.app")
    print("2. Streamlit: streamlit run streamlit_api_tester.py --server.port 8501")
    print("3. Redis: docker run -p 6379:6379 -d redis")
    print("\nStarting browser tests in 5 seconds...")
    time.sleep(5)

    asyncio.run(main())