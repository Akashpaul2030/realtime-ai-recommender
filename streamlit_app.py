"""
Streamlit UI for AI Recommendation System
Modern interface to test your Pinecone + Supabase stack
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import sys
import os
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page configuration
st.set_page_config(
    page_title="AI Recommendation Engine",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .product-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
    .similarity-score {
        background: linear-gradient(90deg, #ff7b7b 0%, #ffcc5c 50%, #28a745 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def load_backend_services():
    """Load backend services with error handling"""
    try:
        from adapters.factory import get_vector_store, get_event_processor, get_backend_info
        from models.embeddings import get_embedding_model

        return {
            'vector_store': get_vector_store(),
            'event_processor': get_event_processor(),
            'embedding_model': get_embedding_model(),
            'backend_info': get_backend_info()
        }
    except Exception as e:
        st.error(f"Failed to load backend services: {e}")
        return None

def display_backend_status(services):
    """Display backend service status"""
    if not services:
        st.error("❌ Backend services not available")
        return False

    st.sidebar.markdown("## 🔧 Backend Status")

    info = services['backend_info']

    # Status indicators
    statuses = {
        "Vector Store": "✅" if services['vector_store'] else "❌",
        "Event Processor": "✅" if services['event_processor'] else "❌",
        "Embedding Model": "✅" if services['embedding_model'] else "❌"
    }

    for service, status in statuses.items():
        st.sidebar.markdown(f"{status} {service}")

    # Backend configuration
    st.sidebar.markdown("### Configuration")
    st.sidebar.json({
        "Backend Type": info.get('backend_type', 'Unknown'),
        "Vector Store": info.get('vector_store', 'Unknown'),
        "Event Processor": info.get('event_processor', 'Unknown')
    })

    return True

def create_sample_products():
    """Create sample product data for testing"""
    return [
        {
            "id": "prod-001",
            "name": "Wireless Bluetooth Headphones",
            "description": "High-quality wireless headphones with noise cancellation and 30-hour battery life",
            "category": "electronics",
            "price": 299.99
        },
        {
            "id": "prod-002",
            "name": "Smart Fitness Tracker",
            "description": "Advanced fitness tracker with heart rate monitoring, GPS, and sleep tracking",
            "category": "electronics",
            "price": 199.99
        },
        {
            "id": "prod-003",
            "name": "Ergonomic Office Chair",
            "description": "Professional office chair with lumbar support and adjustable height",
            "category": "furniture",
            "price": 449.99
        },
        {
            "id": "prod-004",
            "name": "Portable Speaker",
            "description": "Waterproof Bluetooth speaker with deep bass and 20-hour battery",
            "category": "electronics",
            "price": 89.99
        }
    ]

def add_product_tab(services):
    """Tab for adding new products"""
    st.header("➕ Add New Product")

    with st.form("add_product_form"):
        col1, col2 = st.columns(2)

        with col1:
            product_id = st.text_input("Product ID*", placeholder="e.g., prod-123")
            product_name = st.text_input("Product Name*", placeholder="e.g., Smart Watch")
            product_price = st.number_input("Price", min_value=0.0, step=0.01)

        with col2:
            product_category = st.selectbox("Category",
                ["electronics", "clothing", "furniture", "books", "sports", "beauty", "automotive"])
            product_description = st.text_area("Description",
                placeholder="Detailed product description...")

        submitted = st.form_submit_button("🚀 Add Product", use_container_width=True)

        if submitted:
            if not product_id or not product_name:
                st.error("Product ID and Name are required!")
                return

            # Create product data
            product_data = {
                "id": product_id,
                "name": product_name,
                "description": product_description,
                "category": product_category,
                "price": product_price,
                "created_at": time.time()
            }

            try:
                # Generate embedding
                with st.spinner("Generating AI embedding..."):
                    embedding = services['embedding_model'].get_product_embedding(product_data)

                # Store in vector database
                with st.spinner("Storing in vector database..."):
                    metadata = {
                        'name': product_name,
                        'category': product_category,
                        'price': str(product_price)
                    }
                    success = services['vector_store'].store_product_embedding(
                        product_id, embedding, metadata
                    )

                if success:
                    st.success(f"✅ Product '{product_name}' added successfully!")

                    # Show embedding info
                    st.info(f"📊 Generated {len(embedding)}-dimensional embedding vector")

                    # Publish event (optional)
                    try:
                        event_id = services['event_processor'].publish_product_created(product_data)
                        if event_id:
                            st.success(f"📡 Event published: {event_id}")
                    except Exception as e:
                        st.warning(f"Product added but event publishing failed: {e}")

                else:
                    st.error("❌ Failed to store product in vector database")

            except Exception as e:
                st.error(f"❌ Error adding product: {e}")

def search_products_tab(services):
    """Tab for searching and finding similar products"""
    st.header("🔍 Product Search & Recommendations")

    # Search type selection
    search_type = st.radio("Search Type:",
        ["Text Search", "Product Similarity", "Load Sample Products"],
        horizontal=True)

    if search_type == "Text Search":
        search_query = st.text_input("Search Products:",
            placeholder="e.g., wireless headphones, fitness tracker...")

        col1, col2 = st.columns([3, 1])
        with col1:
            max_results = st.slider("Max Results", 1, 20, 10)
        with col2:
            min_similarity = st.slider("Min Similarity", 0.0, 1.0, 0.3, 0.1)

        if st.button("🔍 Search", use_container_width=True) and search_query:
            with st.spinner("Searching products..."):
                try:
                    # Generate embedding for search query
                    query_embedding = services['embedding_model'].get_text_embedding(search_query)

                    # Search similar products
                    results = services['vector_store'].find_similar_products(
                        embedding=query_embedding,
                        limit=max_results,
                        min_score=min_similarity
                    )

                    if results:
                        st.success(f"Found {len(results)} similar products:")
                        display_search_results(results)
                    else:
                        st.warning("No products found matching your search.")

                except Exception as e:
                    st.error(f"Search failed: {e}")

    elif search_type == "Product Similarity":
        # Get product ID for similarity search
        product_id = st.text_input("Product ID:", placeholder="Enter existing product ID")

        col1, col2 = st.columns(2)
        with col1:
            max_results = st.slider("Max Results", 1, 20, 5)
        with col2:
            min_similarity = st.slider("Min Similarity", 0.0, 1.0, 0.7, 0.1)

        if st.button("Find Similar Products", use_container_width=True) and product_id:
            with st.spinner("Finding similar products..."):
                try:
                    # Get product embedding
                    product_embedding = services['vector_store'].get_product_embedding(product_id)

                    if product_embedding is not None:
                        # Find similar products
                        results = services['vector_store'].find_similar_products(
                            embedding=product_embedding,
                            limit=max_results + 1,  # +1 to exclude the product itself
                            min_score=min_similarity
                        )

                        # Filter out the original product
                        results = [r for r in results if r['product_id'] != product_id][:max_results]

                        if results:
                            st.success(f"Found {len(results)} similar products:")
                            display_search_results(results)
                        else:
                            st.warning("No similar products found.")
                    else:
                        st.error(f"Product '{product_id}' not found in vector database.")

                except Exception as e:
                    st.error(f"Similarity search failed: {e}")

    elif search_type == "Load Sample Products":
        if st.button("📦 Load Sample Products", use_container_width=True):
            with st.spinner("Loading sample products..."):
                sample_products = create_sample_products()
                success_count = 0

                for product in sample_products:
                    try:
                        # Generate embedding
                        embedding = services['embedding_model'].get_product_embedding(product)

                        # Store in vector database
                        metadata = {
                            'name': product['name'],
                            'category': product['category'],
                            'price': str(product['price'])
                        }
                        success = services['vector_store'].store_product_embedding(
                            product['id'], embedding, metadata
                        )

                        if success:
                            success_count += 1

                    except Exception as e:
                        st.warning(f"Failed to load {product['name']}: {e}")

                if success_count > 0:
                    st.success(f"✅ Loaded {success_count}/{len(sample_products)} sample products!")

                    # Display loaded products
                    st.subheader("Loaded Products:")
                    for product in sample_products:
                        with st.expander(f"{product['name']} - ${product['price']}"):
                            st.write(f"**ID:** {product['id']}")
                            st.write(f"**Category:** {product['category']}")
                            st.write(f"**Description:** {product['description']}")
                else:
                    st.error("Failed to load sample products.")

def display_search_results(results: List[Dict]):
    """Display search results in a nice format"""
    for i, result in enumerate(results):
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.markdown(f"**{result.get('metadata', {}).get('name', result['product_id'])}**")
                st.write(f"Category: {result.get('metadata', {}).get('category', 'Unknown')}")

            with col2:
                price = result.get('metadata', {}).get('price', 'N/A')
                st.markdown(f"**Price:** ${price}")

            with col3:
                similarity = result['similarity_score']
                color = "green" if similarity > 0.8 else "orange" if similarity > 0.6 else "red"
                st.markdown(f"<div style='color: {color}; font-weight: bold;'>Similarity: {similarity:.2%}</div>",
                          unsafe_allow_html=True)

            st.markdown("---")

def analytics_tab(services):
    """Tab for analytics and vector database stats"""
    st.header("📊 Analytics & Database Stats")

    # Get vector store stats if available
    if hasattr(services['vector_store'], 'get_index_stats'):
        try:
            with st.spinner("Loading database statistics..."):
                stats = services['vector_store'].get_index_stats()

            # Display stats in metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Vectors", stats.get('total_vector_count', 'N/A'))
            with col2:
                st.metric("Dimensions", stats.get('dimension', 'N/A'))
            with col3:
                st.metric("Index Fullness", f"{stats.get('index_fullness', 0):.1%}")
            with col4:
                st.metric("Namespaces", len(stats.get('namespaces', {})))

            # Display detailed stats
            st.subheader("Detailed Statistics")
            st.json(stats)

        except Exception as e:
            st.error(f"Failed to load statistics: {e}")
    else:
        st.info("Vector database statistics not available for this backend.")

    # Backend information
    st.subheader("Backend Configuration")
    backend_info = services['backend_info']

    col1, col2 = st.columns(2)
    with col1:
        st.json({
            "Backend Type": backend_info.get('backend_type'),
            "Vector Store": backend_info.get('vector_store'),
            "Event Processor": backend_info.get('event_processor')
        })

    with col2:
        cloud_services = backend_info.get('cloud_services', {})
        st.json({
            "Pinecone Configured": cloud_services.get('pinecone_configured', False),
            "Supabase Configured": cloud_services.get('supabase_configured', False)
        })

def main():
    """Main Streamlit application"""
    # Header
    st.markdown('<h1 class="main-header">🛍️ AI Recommendation Engine</h1>', unsafe_allow_html=True)
    st.markdown("**Modern Product Recommendations powered by Pinecone + Supabase**")

    # Load backend services
    services = load_backend_services()

    # Display backend status in sidebar
    if not display_backend_status(services):
        st.stop()

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["➕ Add Products", "🔍 Search & Recommend", "📊 Analytics"])

    with tab1:
        add_product_tab(services)

    with tab2:
        search_products_tab(services)

    with tab3:
        analytics_tab(services)

    # Footer
    st.markdown("---")
    st.markdown("🚀 **Powered by Modern Cloud Stack:** Pinecone Vector Database + Supabase + FastAPI")

if __name__ == "__main__":
    main()