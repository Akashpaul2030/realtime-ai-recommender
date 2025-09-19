"""
🧪 Comprehensive API + UI Testing Dashboard
Tests the integration between your FastAPI backend and various UI components
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import json
import time
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page configuration
st.set_page_config(
    page_title="API + UI Integration Tester",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .test-success {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        color: #155724;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    .test-failure {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        color: #721c24;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    .test-warning {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        color: #856404;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    .api-endpoint {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

class APITester:
    """Comprehensive API testing class"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []

    def test_endpoint(self, method: str, endpoint: str, data: Optional[Dict] = None,
                     expected_status: int = 200, timeout: int = 10) -> Dict[str, Any]:
        """Test a single API endpoint"""

        url = f"{self.base_url}{endpoint}"
        start_time = time.time()

        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=data, timeout=timeout)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, timeout=timeout)
            elif method.upper() == "PUT":
                response = self.session.put(url, json=data, timeout=timeout)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, timeout=timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response_time = time.time() - start_time

            result = {
                "endpoint": endpoint,
                "method": method,
                "status_code": response.status_code,
                "expected_status": expected_status,
                "response_time": response_time,
                "success": response.status_code == expected_status,
                "response_data": None,
                "error": None,
                "timestamp": time.time()
            }

            try:
                result["response_data"] = response.json()
            except:
                result["response_data"] = response.text

        except Exception as e:
            result = {
                "endpoint": endpoint,
                "method": method,
                "status_code": 0,
                "expected_status": expected_status,
                "response_time": time.time() - start_time,
                "success": False,
                "response_data": None,
                "error": str(e),
                "timestamp": time.time()
            }

        self.test_results.append(result)
        return result

def test_api_health():
    """Test API health and connectivity"""
    st.subheader("🏥 API Health Check")

    # Configuration
    col1, col2 = st.columns([2, 1])
    with col1:
        api_url = st.text_input("API Base URL", value="http://localhost:8000")
    with col2:
        timeout = st.number_input("Timeout (seconds)", min_value=1, max_value=30, value=5)

    if st.button("🔍 Test API Health", use_container_width=True):
        tester = APITester(api_url)

        with st.spinner("Testing API endpoints..."):
            # Test basic endpoints
            endpoints_to_test = [
                ("GET", "/", "Root endpoint"),
                ("GET", "/health", "Health check"),
                ("GET", "/docs", "API documentation"),
                ("GET", "/openapi.json", "OpenAPI specification")
            ]

            results = []
            for method, endpoint, description in endpoints_to_test:
                result = tester.test_endpoint(method, endpoint)
                result["description"] = description
                results.append(result)

        # Display results
        for result in results:
            if result["success"]:
                st.markdown(f"""
                <div class="test-success">
                    ✅ <strong>{result['description']}</strong><br>
                    <code>{result['method']} {result['endpoint']}</code> -
                    {result['status_code']} ({result['response_time']:.3f}s)
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="test-failure">
                    ❌ <strong>{result['description']}</strong><br>
                    <code>{result['method']} {result['endpoint']}</code> -
                    {result.get('error', f"Status: {result['status_code']}")}
                </div>
                """, unsafe_allow_html=True)

        # Summary metrics
        success_count = sum(1 for r in results if r["success"])
        total_count = len(results)
        avg_response_time = np.mean([r["response_time"] for r in results])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Success Rate", f"{success_count}/{total_count}")
        with col2:
            st.metric("Avg Response Time", f"{avg_response_time:.3f}s")
        with col3:
            status = "🟢 Healthy" if success_count == total_count else "🔴 Issues"
            st.metric("API Status", status)

def test_product_apis():
    """Test product-related API endpoints"""
    st.subheader("🛍️ Product API Testing")

    api_url = st.session_state.get('api_url', 'http://localhost:8000')
    tester = APITester(api_url)

    # Test data
    test_product = {
        "id": f"test-product-{int(time.time())}",
        "name": "Test Wireless Headphones",
        "description": "High-quality wireless headphones for testing purposes",
        "category": "electronics",
        "price": 199.99,
        "attributes": {
            "brand": "TestBrand",
            "color": "black",
            "wireless": True
        }
    }

    st.markdown("### Test Product Data:")
    st.json(test_product)

    if st.button("🧪 Run Product API Tests", use_container_width=True):
        with st.spinner("Running comprehensive product API tests..."):

            # Test 1: Create Product
            st.markdown("#### 1. Testing Product Creation")
            result = tester.test_endpoint("POST", "/products/", test_product)

            if result["success"]:
                st.success(f"✅ Product created successfully ({result['response_time']:.3f}s)")
                st.json(result["response_data"])
                product_id = test_product["id"]
            else:
                st.error(f"❌ Failed to create product: {result.get('error', result['status_code'])}")
                return

            # Test 2: Get Product
            st.markdown("#### 2. Testing Product Retrieval")
            result = tester.test_endpoint("GET", f"/products/{product_id}")

            if result["success"]:
                st.success(f"✅ Product retrieved successfully ({result['response_time']:.3f}s)")
                st.json(result["response_data"])
            else:
                st.error(f"❌ Failed to retrieve product: {result.get('error', result['status_code'])}")

            # Test 3: Search Products
            st.markdown("#### 3. Testing Product Search")
            search_params = {"query": "wireless headphones", "limit": 5}
            result = tester.test_endpoint("GET", "/products/search/text", search_params)

            if result["success"]:
                st.success(f"✅ Product search successful ({result['response_time']:.3f}s)")
                st.json(result["response_data"])
            else:
                st.warning(f"⚠️ Product search failed: {result.get('error', result['status_code'])}")

            # Test 4: Similar Products (if hybrid search is available)
            st.markdown("#### 4. Testing Similar Products")
            result = tester.test_endpoint("GET", f"/products/similar/{product_id}")

            if result["success"]:
                st.success(f"✅ Similar products found ({result['response_time']:.3f}s)")
                st.json(result["response_data"])
            else:
                st.warning(f"⚠️ Similar products search failed: {result.get('error', result['status_code'])}")

            # Test 5: Hybrid Search (if available)
            st.markdown("#### 5. Testing Hybrid Search")
            hybrid_data = {
                "query": "wireless audio device",
                "alpha": 0.05,
                "top_k": 5
            }
            result = tester.test_endpoint("GET", "/products/search/hybrid", hybrid_data)

            if result["success"]:
                st.success(f"✅ Hybrid search successful ({result['response_time']:.3f}s)")
                st.json(result["response_data"])
            else:
                st.info(f"ℹ️ Hybrid search not available: {result.get('error', result['status_code'])}")

            # Test 6: Update Product
            st.markdown("#### 6. Testing Product Update")
            update_data = {"price": 249.99, "description": "Updated description"}
            result = tester.test_endpoint("PUT", f"/products/{product_id}", update_data)

            if result["success"]:
                st.success(f"✅ Product updated successfully ({result['response_time']:.3f}s)")
                st.json(result["response_data"])
            else:
                st.warning(f"⚠️ Product update failed: {result.get('error', result['status_code'])}")

            # Test 7: Delete Product
            st.markdown("#### 7. Testing Product Deletion")
            result = tester.test_endpoint("DELETE", f"/products/{product_id}")

            if result["success"]:
                st.success(f"✅ Product deleted successfully ({result['response_time']:.3f}s)")
            else:
                st.warning(f"⚠️ Product deletion failed: {result.get('error', result['status_code'])}")

            # Summary
            st.markdown("#### 📊 Test Summary")
            success_rate = sum(1 for r in tester.test_results if r["success"]) / len(tester.test_results)
            avg_response_time = np.mean([r["response_time"] for r in tester.test_results])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Success Rate", f"{success_rate:.1%}")
            with col2:
                st.metric("Avg Response Time", f"{avg_response_time:.3f}s")
            with col3:
                st.metric("Total Tests", len(tester.test_results))

def test_recommendation_apis():
    """Test recommendation API endpoints"""
    st.subheader("🎯 Recommendation API Testing")

    api_url = st.session_state.get('api_url', 'http://localhost:8000')

    # Test configuration
    col1, col2 = st.columns(2)
    with col1:
        user_id = st.text_input("Test User ID", value="test-user-123")
        product_id = st.text_input("Test Product ID", value="test-product-456")
    with col2:
        limit = st.number_input("Max Recommendations", min_value=1, max_value=20, value=5)
        include_trending = st.checkbox("Include Trending", value=True)

    if st.button("🧪 Test Recommendation APIs", use_container_width=True):
        tester = APITester(api_url)

        with st.spinner("Testing recommendation endpoints..."):

            # Test 1: Get Similar Product Recommendations
            st.markdown("#### 1. Similar Product Recommendations")
            result = tester.test_endpoint("GET", f"/recommendations/{product_id}/similar", {"limit": limit})

            if result["success"]:
                st.success(f"✅ Similar recommendations retrieved ({result['response_time']:.3f}s)")
                st.json(result["response_data"])
            else:
                st.warning(f"⚠️ Similar recommendations failed: {result.get('error', result['status_code'])}")

            # Test 2: Category Recommendations
            st.markdown("#### 2. Category-based Recommendations")
            result = tester.test_endpoint("GET", "/recommendations/category/electronics", {"limit": limit})

            if result["success"]:
                st.success(f"✅ Category recommendations retrieved ({result['response_time']:.3f}s)")
                st.json(result["response_data"])
            else:
                st.warning(f"⚠️ Category recommendations failed: {result.get('error', result['status_code'])}")

            # Test 3: Personalized Recommendations
            st.markdown("#### 3. Personalized Recommendations")
            params = {"limit": limit, "include_trending": include_trending}
            result = tester.test_endpoint("GET", f"/recommendations/personalized", params)

            if result["success"]:
                st.success(f"✅ Personalized recommendations retrieved ({result['response_time']:.3f}s)")
                st.json(result["response_data"])
            else:
                st.warning(f"⚠️ Personalized recommendations failed: {result.get('error', result['status_code'])}")

            # Test 4: Track Product View
            st.markdown("#### 4. Track Product View")
            view_data = {"user_id": user_id, "product_id": product_id}
            result = tester.test_endpoint("POST", "/recommendations/track-view", view_data)

            if result["success"]:
                st.success(f"✅ Product view tracked ({result['response_time']:.3f}s)")
                st.json(result["response_data"])
            else:
                st.warning(f"⚠️ View tracking failed: {result.get('error', result['status_code'])}")

def performance_monitoring():
    """Real-time performance monitoring"""
    st.subheader("📈 Real-time Performance Monitoring")

    # Configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        api_url = st.text_input("API URL", value="http://localhost:8000")
    with col2:
        test_duration = st.number_input("Test Duration (seconds)", min_value=10, max_value=300, value=60)
    with col3:
        request_interval = st.number_input("Request Interval (seconds)", min_value=1, max_value=10, value=2)

    if st.button("🚀 Start Performance Monitoring", use_container_width=True):
        tester = APITester(api_url)

        # Create containers for real-time updates
        metrics_container = st.container()
        chart_container = st.container()

        start_time = time.time()
        response_times = []
        timestamps = []
        success_rates = []

        # Real-time monitoring loop
        progress_bar = st.progress(0)
        status_text = st.empty()

        while time.time() - start_time < test_duration:
            current_time = time.time() - start_time
            progress = current_time / test_duration
            progress_bar.progress(progress)

            # Test health endpoint
            result = tester.test_endpoint("GET", "/health")

            # Collect metrics
            response_times.append(result["response_time"])
            timestamps.append(current_time)

            # Calculate success rate for last 10 requests
            recent_results = tester.test_results[-10:]
            success_rate = sum(1 for r in recent_results if r["success"]) / len(recent_results)
            success_rates.append(success_rate)

            # Update real-time metrics
            with metrics_container:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Response Time", f"{result['response_time']:.3f}s")
                with col2:
                    st.metric("Avg Response Time", f"{np.mean(response_times):.3f}s")
                with col3:
                    st.metric("Success Rate", f"{success_rate:.1%}")
                with col4:
                    st.metric("Total Requests", len(tester.test_results))

            # Update chart
            with chart_container:
                if len(response_times) > 1:
                    fig = go.Figure()

                    # Response time line
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=response_times,
                        mode='lines+markers',
                        name='Response Time (s)',
                        line=dict(color='blue')
                    ))

                    # Success rate line (secondary y-axis)
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=[sr * 100 for sr in success_rates],
                        mode='lines+markers',
                        name='Success Rate (%)',
                        yaxis='y2',
                        line=dict(color='green')
                    ))

                    # Update layout
                    fig.update_layout(
                        title='Real-time API Performance',
                        xaxis_title='Time (seconds)',
                        yaxis_title='Response Time (seconds)',
                        yaxis2=dict(
                            title='Success Rate (%)',
                            overlaying='y',
                            side='right'
                        ),
                        height=400
                    )

                    st.plotly_chart(fig, use_container_width=True)

            status_text.text(f"Monitoring... {current_time:.1f}s elapsed")
            time.sleep(request_interval)

        # Final summary
        st.success("🎉 Performance monitoring completed!")

        final_metrics = {
            "Total Requests": len(tester.test_results),
            "Average Response Time": f"{np.mean(response_times):.3f}s",
            "Min Response Time": f"{min(response_times):.3f}s",
            "Max Response Time": f"{max(response_times):.3f}s",
            "Overall Success Rate": f"{np.mean(success_rates):.1%}",
            "Requests per Second": f"{len(tester.test_results) / test_duration:.2f}"
        }

        col1, col2 = st.columns(2)
        with col1:
            st.json(final_metrics)
        with col2:
            # Performance grade
            avg_response = np.mean(response_times)
            avg_success = np.mean(success_rates)

            if avg_response < 0.1 and avg_success > 0.95:
                grade = "🟢 Excellent"
            elif avg_response < 0.5 and avg_success > 0.9:
                grade = "🟡 Good"
            elif avg_response < 1.0 and avg_success > 0.8:
                grade = "🟠 Fair"
            else:
                grade = "🔴 Needs Improvement"

            st.metric("Performance Grade", grade)

def integration_test_suite():
    """Complete integration test suite"""
    st.subheader("🔬 Complete Integration Test Suite")

    if st.button("🚀 Run Full Integration Tests", use_container_width=True):
        api_url = st.session_state.get('api_url', 'http://localhost:8000')
        tester = APITester(api_url)

        test_results = {
            "API Health": False,
            "Product CRUD": False,
            "Search Functionality": False,
            "Recommendations": False,
            "Performance": False
        }

        with st.spinner("Running comprehensive integration tests..."):

            # 1. API Health Tests
            st.markdown("### 1. API Health Tests")
            health_result = tester.test_endpoint("GET", "/health")
            test_results["API Health"] = health_result["success"]

            if health_result["success"]:
                st.success("✅ API is healthy and responding")
            else:
                st.error("❌ API health check failed")
                return

            # 2. Product CRUD Tests
            st.markdown("### 2. Product CRUD Tests")
            test_product = {
                "id": f"integration-test-{int(time.time())}",
                "name": "Integration Test Product",
                "description": "Product for integration testing",
                "category": "test",
                "price": 99.99
            }

            # Create
            create_result = tester.test_endpoint("POST", "/products/", test_product)
            if create_result["success"]:
                # Read
                read_result = tester.test_endpoint("GET", f"/products/{test_product['id']}")
                # Update
                update_result = tester.test_endpoint("PUT", f"/products/{test_product['id']}", {"price": 149.99})
                # Delete
                delete_result = tester.test_endpoint("DELETE", f"/products/{test_product['id']}")

                crud_success = all([create_result["success"], read_result["success"],
                                  update_result["success"], delete_result["success"]])
                test_results["Product CRUD"] = crud_success

                if crud_success:
                    st.success("✅ All CRUD operations successful")
                else:
                    st.warning("⚠️ Some CRUD operations failed")
            else:
                st.error("❌ Product creation failed")

            # 3. Search Tests
            st.markdown("### 3. Search Functionality Tests")
            search_result = tester.test_endpoint("GET", "/products/search/text", {"query": "test"})
            test_results["Search Functionality"] = search_result["success"]

            if search_result["success"]:
                st.success("✅ Search functionality working")
            else:
                st.warning("⚠️ Search functionality not available")

            # 4. Recommendation Tests
            st.markdown("### 4. Recommendation Tests")
            rec_result = tester.test_endpoint("GET", "/recommendations/category/electronics")
            test_results["Recommendations"] = rec_result["success"]

            if rec_result["success"]:
                st.success("✅ Recommendation system working")
            else:
                st.warning("⚠️ Recommendation system not fully available")

            # 5. Performance Tests
            st.markdown("### 5. Performance Tests")
            perf_results = []
            for _ in range(5):
                result = tester.test_endpoint("GET", "/health")
                perf_results.append(result["response_time"])

            avg_response_time = np.mean(perf_results)
            test_results["Performance"] = avg_response_time < 1.0  # < 1 second

            if test_results["Performance"]:
                st.success(f"✅ Performance acceptable (avg: {avg_response_time:.3f}s)")
            else:
                st.warning(f"⚠️ Performance needs improvement (avg: {avg_response_time:.3f}s)")

        # Final Results Summary
        st.markdown("### 📊 Integration Test Results")

        passed_tests = sum(test_results.values())
        total_tests = len(test_results)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tests Passed", f"{passed_tests}/{total_tests}")
        with col2:
            st.metric("Success Rate", f"{passed_tests/total_tests:.1%}")
        with col3:
            overall_status = "🟢 PASS" if passed_tests == total_tests else "🟡 PARTIAL" if passed_tests > 0 else "🔴 FAIL"
            st.metric("Overall Status", overall_status)

        # Detailed results
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            st.markdown(f"- **{test_name}**: {status}")

def main():
    """Main Streamlit application"""

    # Header
    st.markdown('<h1 class="main-header">🧪 API + UI Integration Tester</h1>', unsafe_allow_html=True)
    st.markdown("**Comprehensive testing dashboard for your FastAPI backend and UI components**")

    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    api_url = st.sidebar.text_input("API Base URL", value="http://localhost:8000")
    st.session_state['api_url'] = api_url

    # Test if API is reachable
    try:
        response = requests.get(f"{api_url}/health", timeout=2)
        if response.status_code == 200:
            st.sidebar.success("🟢 API Connected")
        else:
            st.sidebar.warning(f"🟡 API Status: {response.status_code}")
    except:
        st.sidebar.error("🔴 API Not Reachable")

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏥 Health Check",
        "🛍️ Product APIs",
        "🎯 Recommendations",
        "📈 Performance",
        "🔬 Integration Suite"
    ])

    with tab1:
        test_api_health()

    with tab2:
        test_product_apis()

    with tab3:
        test_recommendation_apis()

    with tab4:
        performance_monitoring()

    with tab5:
        integration_test_suite()

    # Footer
    st.markdown("---")
    st.markdown("🚀 **Integration Testing Dashboard** - Ensuring your API and UI work perfectly together!")

if __name__ == "__main__":
    main()