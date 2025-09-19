"""
🎯 Demo Setup Script
Creates sample data and demonstrates the real-time AI recommender system
"""

import requests
import json
import time
import random
from typing import List, Dict

class DemoSetup:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.session = requests.Session()

    def create_sample_products(self) -> List[Dict]:
        """Create realistic sample products for demo"""
        sample_products = [
            {
                "id": "laptop-001",
                "name": "MacBook Pro 16-inch",
                "description": "Powerful laptop with M2 Pro chip, 16GB RAM, 512GB SSD. Perfect for development and creative work.",
                "category": "electronics",
                "price": 2499.99,
                "sku": "MBP-16-M2-512",
                "attributes": {
                    "brand": "Apple",
                    "screen_size": "16 inch",
                    "processor": "M2 Pro",
                    "ram": "16GB",
                    "storage": "512GB SSD",
                    "color": "Space Gray"
                }
            },
            {
                "id": "headphones-001",
                "name": "Sony WH-1000XM5 Wireless Headphones",
                "description": "Industry-leading noise canceling wireless headphones with 30-hour battery life.",
                "category": "electronics",
                "price": 399.99,
                "sku": "SONY-WH1000XM5-BK",
                "attributes": {
                    "brand": "Sony",
                    "type": "Over-ear",
                    "wireless": True,
                    "noise_canceling": True,
                    "battery_life": "30 hours",
                    "color": "Black"
                }
            },
            {
                "id": "smartphone-001",
                "name": "iPhone 15 Pro",
                "description": "Latest iPhone with titanium design, A17 Pro chip, and advanced camera system.",
                "category": "electronics",
                "price": 1199.99,
                "sku": "IP15PRO-128-TB",
                "attributes": {
                    "brand": "Apple",
                    "storage": "128GB",
                    "color": "Titanium Blue",
                    "screen_size": "6.1 inch",
                    "camera": "48MP main"
                }
            },
            {
                "id": "tablet-001",
                "name": "iPad Air 5th Generation",
                "description": "Powerful and versatile tablet with M1 chip and stunning Liquid Retina display.",
                "category": "electronics",
                "price": 699.99,
                "sku": "IPAD-AIR5-256-SG",
                "attributes": {
                    "brand": "Apple",
                    "screen_size": "10.9 inch",
                    "storage": "256GB",
                    "processor": "M1",
                    "color": "Space Gray"
                }
            },
            {
                "id": "monitor-001",
                "name": "Dell UltraSharp 27-inch 4K Monitor",
                "description": "Professional 4K monitor with USB-C connectivity and color accuracy for creators.",
                "category": "electronics",
                "price": 599.99,
                "sku": "DELL-U2723QE",
                "attributes": {
                    "brand": "Dell",
                    "screen_size": "27 inch",
                    "resolution": "4K UHD",
                    "connectivity": "USB-C, HDMI, DisplayPort",
                    "color_accuracy": "99% sRGB"
                }
            },
            {
                "id": "keyboard-001",
                "name": "Logitech MX Keys Advanced Wireless Keyboard",
                "description": "Premium wireless keyboard with smart illumination and multi-device connectivity.",
                "category": "electronics",
                "price": 129.99,
                "sku": "LOG-MXKEYS-BK",
                "attributes": {
                    "brand": "Logitech",
                    "type": "Wireless",
                    "backlit": True,
                    "multi_device": True,
                    "battery_life": "10 days"
                }
            },
            {
                "id": "mouse-001",
                "name": "Logitech MX Master 3S Wireless Mouse",
                "description": "Advanced wireless mouse with precision tracking and ergonomic design.",
                "category": "electronics",
                "price": 99.99,
                "sku": "LOG-MXMASTER3S-GR",
                "attributes": {
                    "brand": "Logitech",
                    "type": "Wireless",
                    "dpi": "8000",
                    "ergonomic": True,
                    "color": "Graphite"
                }
            },
            {
                "id": "speaker-001",
                "name": "HomePod mini",
                "description": "Compact smart speaker with amazing sound and Siri intelligence.",
                "category": "electronics",
                "price": 99.99,
                "sku": "HOMEPOD-MINI-WH",
                "attributes": {
                    "brand": "Apple",
                    "type": "Smart Speaker",
                    "voice_assistant": "Siri",
                    "color": "White",
                    "size": "Compact"
                }
            },
            {
                "id": "camera-001",
                "name": "Canon EOS R6 Mark II",
                "description": "Professional mirrorless camera with 24.2MP sensor and advanced autofocus.",
                "category": "electronics",
                "price": 2499.99,
                "sku": "CANON-R6MK2-BODY",
                "attributes": {
                    "brand": "Canon",
                    "type": "Mirrorless",
                    "megapixels": "24.2MP",
                    "video": "4K 60p",
                    "autofocus": "Dual Pixel CMOS AF II"
                }
            },
            {
                "id": "watch-001",
                "name": "Apple Watch Series 9",
                "description": "Advanced smartwatch with health monitoring and fitness tracking capabilities.",
                "category": "electronics",
                "price": 429.99,
                "sku": "AW-S9-45-GPS-MN",
                "attributes": {
                    "brand": "Apple",
                    "size": "45mm",
                    "connectivity": "GPS",
                    "health_monitoring": True,
                    "color": "Midnight"
                }
            }
        ]
        return sample_products

    def load_demo_data(self):
        """Load all demo products into the system"""
        print("Loading demo data...")
        products = self.create_sample_products()

        success_count = 0
        for product in products:
            try:
                response = self.session.post(f"{self.api_url}/products/", json=product, timeout=10)
                if response.status_code == 200:
                    print(f"OK Created: {product['name']}")
                    success_count += 1
                else:
                    print(f"FAIL Failed to create: {product['name']} - {response.status_code}")
            except Exception as e:
                print(f"ERROR creating {product['name']}: {e}")

            time.sleep(0.5)  # Small delay to avoid overwhelming the system

        print(f"\nDemo Setup Complete: {success_count}/{len(products)} products loaded")
        return success_count == len(products)

    def demonstrate_search(self):
        """Demonstrate search functionality"""
        print("\nDemonstrating Search Functionality...")

        search_queries = [
            "MacBook laptop",
            "wireless headphones",
            "4K monitor",
            "Apple products",
            "wireless mouse keyboard"
        ]

        for query in search_queries:
            try:
                response = self.session.get(
                    f"{self.api_url}/products/search/text",
                    params={"query": query, "limit": 3}
                )
                if response.status_code == 200:
                    results = response.json()
                    print(f"Query: '{query}' -> Found {len(results)} results")
                    for result in results[:2]:  # Show top 2 results
                        if isinstance(result, dict) and 'name' in result:
                            print(f"   - {result['name']}")
                else:
                    print(f"FAIL Search failed for '{query}'")
            except Exception as e:
                print(f"ERROR Search error for '{query}': {e}")

            time.sleep(1)

    def demonstrate_recommendations(self):
        """Demonstrate recommendation functionality"""
        print("\nDemonstrating Recommendations...")

        # Try to get similar products for a few items
        product_ids = ["laptop-001", "headphones-001", "smartphone-001"]

        for product_id in product_ids:
            try:
                response = self.session.get(f"{self.api_url}/products/similar/{product_id}")
                if response.status_code == 200:
                    similar = response.json()
                    print(f"Similar to {product_id}: Found {len(similar)} recommendations")
                else:
                    print(f"WARN No recommendations available for {product_id}")
            except Exception as e:
                print(f"ERROR Recommendation error for {product_id}: {e}")

            time.sleep(1)

    def run_full_demo(self):
        """Run the complete demo setup"""
        print("Real-time AI Recommender Demo Setup")
        print("=" * 50)

        # Check if API is available
        try:
            response = self.session.get(f"{self.api_url}/health", timeout=5)
            if response.status_code != 200:
                print("API not healthy. Please start the API server first:")
                print("   python -m api.app")
                return False
        except:
            print("API not reachable. Please start the API server first:")
            print("   python -m api.app")
            return False

        print("API is running and healthy!")

        # Load demo data
        if self.load_demo_data():
            print("\nDemo data loaded successfully!")

            # Wait a moment for processing
            print("\nWaiting for real-time processing...")
            time.sleep(5)

            # Demonstrate features
            self.demonstrate_search()
            self.demonstrate_recommendations()

            print("\nDemo setup complete!")
            print("\nNext Steps:")
            print("1. Open Streamlit demo: streamlit run streamlit_api_tester.py")
            print("2. Access API docs: http://localhost:8000/docs")
            print("3. Run automated tests: python simple_api_test.py")
            print("4. Try Playwright automation: python playwright_api_test.py")

            return True
        else:
            print("Demo setup failed. Check the errors above.")
            return False

if __name__ == "__main__":
    demo = DemoSetup()
    demo.run_full_demo()