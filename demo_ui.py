"""
Demo UI for AI Recommendation System - Windows Compatible
Test your Pinecone + Supabase stack with interactive interface
"""

import sys
import os
import json
import time
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_backend_services():
    """Load backend services"""
    try:
        from adapters.factory import get_vector_store, get_event_processor, get_backend_info
        from models.embeddings import get_embedding_model

        print("Loading backend services...")

        services = {
            'vector_store': get_vector_store(),
            'event_processor': get_event_processor(),
            'embedding_model': get_embedding_model(),
            'backend_info': get_backend_info()
        }

        print("SUCCESS: Backend services loaded!")
        return services

    except Exception as e:
        print(f"ERROR: Failed to load backend services: {e}")
        return None

def display_backend_status(services):
    """Display backend service status"""
    if not services:
        return False

    print("\n" + "="*50)
    print("BACKEND STATUS")
    print("="*50)

    info = services['backend_info']

    # Status indicators
    statuses = {
        "Vector Store": "OK" if services['vector_store'] else "FAIL",
        "Event Processor": "OK" if services['event_processor'] else "FAIL",
        "Embedding Model": "OK" if services['embedding_model'] else "FAIL"
    }

    for service, status in statuses.items():
        print(f"{status}: {service}")

    print("\nConfiguration:")
    print(f"   Backend Type: {info.get('backend_type', 'Unknown')}")
    print(f"   Vector Store: {info.get('vector_store', 'Unknown')}")
    print(f"   Event Processor: {info.get('event_processor', 'Unknown')}")

    cloud_services = info.get('cloud_services', {})
    print(f"   Pinecone: {'Connected' if cloud_services.get('pinecone_configured', False) else 'Not configured'}")
    print(f"   Supabase: {'Connected' if cloud_services.get('supabase_configured', False) else 'Not configured'}")

    return True

def create_sample_products():
    """Create sample product data"""
    return [
        {
            "id": "demo-001",
            "name": "Wireless Bluetooth Headphones",
            "description": "High-quality wireless headphones with noise cancellation and 30-hour battery",
            "category": "electronics",
            "price": 299.99
        },
        {
            "id": "demo-002",
            "name": "Smart Fitness Tracker",
            "description": "Advanced fitness tracker with heart rate monitoring, GPS, and sleep tracking",
            "category": "electronics",
            "price": 199.99
        },
        {
            "id": "demo-003",
            "name": "Ergonomic Office Chair",
            "description": "Professional office chair with lumbar support and adjustable height",
            "category": "furniture",
            "price": 449.99
        },
        {
            "id": "demo-004",
            "name": "Portable Bluetooth Speaker",
            "description": "Waterproof speaker with deep bass and 20-hour battery life",
            "category": "electronics",
            "price": 89.99
        }
    ]

def add_product_demo(services):
    """Demo: Add products to the system"""
    print("\n" + "="*50)
    print("ADD PRODUCT DEMO")
    print("="*50)

    print("Loading sample products into your AI recommendation system...")
    sample_products = create_sample_products()
    success_count = 0

    for product in sample_products:
        try:
            print(f"\nProcessing: {product['name']}")

            # Generate embedding
            print("  Generating AI embedding...")
            embedding = services['embedding_model'].get_product_embedding(product)
            print(f"  Generated {len(embedding)}-dimensional vector")

            # Store in vector database
            print("  Storing in Pinecone...")
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
                print(f"  SUCCESS: {product['name']} added to vector database")

                # Try to publish event
                try:
                    event_id = services['event_processor'].publish_product_created(product)
                    if event_id:
                        print(f"  Event published: {event_id}")
                except Exception as e:
                    print(f"  Warning: Event publishing failed: {e}")
            else:
                print(f"  ERROR: Failed to store {product['name']}")

        except Exception as e:
            print(f"  ERROR: Failed to process {product['name']}: {e}")

    print(f"\n" + "="*50)
    print(f"SUMMARY: Successfully added {success_count}/{len(sample_products)} products")

    if success_count > 0:
        print("\nProducts in your system:")
        for product in sample_products:
            print(f"  {product['id']}: {product['name']} (${product['price']})")

def search_demo(services):
    """Demo: Search and recommendations"""
    print("\n" + "="*50)
    print("SEARCH & RECOMMENDATION DEMO")
    print("="*50)

    # Demo searches
    demo_searches = [
        ("wireless headphones", "Text search for audio products"),
        ("office furniture", "Text search for workspace items"),
        ("fitness tracker", "Text search for health devices")
    ]

    for query, description in demo_searches:
        print(f"\n{description}")
        print(f"Search query: '{query}'")
        print("-" * 30)

        try:
            # Generate embedding for search query
            query_embedding = services['embedding_model'].get_text_embedding(query)

            # Search similar products
            results = services['vector_store'].find_similar_products(
                embedding=query_embedding,
                limit=5,
                min_score=0.3
            )

            if results:
                print(f"Found {len(results)} matching products:")
                for i, result in enumerate(results, 1):
                    metadata = result.get('metadata', {})
                    print(f"  {i}. {metadata.get('name', result['product_id'])}")
                    print(f"     Category: {metadata.get('category', 'Unknown')}")
                    print(f"     Price: ${metadata.get('price', 'N/A')}")
                    print(f"     Similarity: {result['similarity_score']:.1%}")
            else:
                print("No matching products found")

        except Exception as e:
            print(f"Search failed: {e}")

        print()

def similarity_demo(services):
    """Demo: Product similarity recommendations"""
    print("\n" + "="*50)
    print("PRODUCT SIMILARITY DEMO")
    print("="*50)

    # Test similarity with existing products
    test_products = ["demo-001", "demo-002", "demo-003"]

    for product_id in test_products:
        print(f"\nFinding products similar to: {product_id}")
        print("-" * 40)

        try:
            # Get product embedding
            product_embedding = services['vector_store'].get_product_embedding(product_id)

            if product_embedding is not None:
                # Find similar products
                results = services['vector_store'].find_similar_products(
                    embedding=product_embedding,
                    limit=6,
                    min_score=0.5
                )

                # Filter out the original product
                results = [r for r in results if r['product_id'] != product_id][:3]

                if results:
                    print(f"Top {len(results)} similar products:")
                    for i, result in enumerate(results, 1):
                        metadata = result.get('metadata', {})
                        print(f"  {i}. {metadata.get('name', result['product_id'])}")
                        print(f"     Similarity: {result['similarity_score']:.1%}")
                else:
                    print("No similar products found")
            else:
                print(f"Product {product_id} not found in database")

        except Exception as e:
            print(f"Similarity search failed: {e}")

def analytics_demo(services):
    """Demo: Show system analytics"""
    print("\n" + "="*50)
    print("ANALYTICS & STATS DEMO")
    print("="*50)

    # Vector store stats
    if hasattr(services['vector_store'], 'get_index_stats'):
        try:
            print("Loading Pinecone database statistics...")
            stats = services['vector_store'].get_index_stats()

            print("\nDatabase Statistics:")
            print(f"  Total Vectors: {stats.get('total_vector_count', 'N/A')}")
            print(f"  Dimensions: {stats.get('dimension', 'N/A')}")
            print(f"  Index Fullness: {stats.get('index_fullness', 0):.1%}")
            print(f"  Namespaces: {len(stats.get('namespaces', {}))}")

        except Exception as e:
            print(f"Failed to load statistics: {e}")
    else:
        print("Database statistics not available for this backend")

    # Backend info
    print("\nBackend Configuration:")
    backend_info = services['backend_info']
    for key, value in backend_info.items():
        if key != 'cloud_services':
            print(f"  {key}: {value}")

    cloud_services = backend_info.get('cloud_services', {})
    print(f"  Pinecone: {'Connected' if cloud_services.get('pinecone_configured') else 'Not configured'}")
    print(f"  Supabase: {'Connected' if cloud_services.get('supabase_configured') else 'Not configured'}")

def interactive_search(services):
    """Interactive search feature"""
    print("\n" + "="*50)
    print("INTERACTIVE SEARCH")
    print("="*50)

    while True:
        try:
            query = input("\nEnter search query (or 'quit' to exit): ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                break

            if not query:
                print("Please enter a search query")
                continue

            print(f"\nSearching for: '{query}'")
            print("-" * 30)

            # Generate embedding and search
            query_embedding = services['embedding_model'].get_text_embedding(query)
            results = services['vector_store'].find_similar_products(
                embedding=query_embedding,
                limit=5,
                min_score=0.2
            )

            if results:
                print(f"Found {len(results)} products:")
                for i, result in enumerate(results, 1):
                    metadata = result.get('metadata', {})
                    print(f"  {i}. {metadata.get('name', result['product_id'])}")
                    print(f"     Category: {metadata.get('category', 'Unknown')}")
                    print(f"     Price: ${metadata.get('price', 'N/A')}")
                    print(f"     Match: {result['similarity_score']:.1%}")
            else:
                print("No products found. Try a different search term.")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Search error: {e}")

def main_menu():
    """Display main menu"""
    print("\n" + "="*50)
    print("AI RECOMMENDATION ENGINE - DEMO MENU")
    print("="*50)
    print("1. Load Sample Products")
    print("2. Search Demo")
    print("3. Similarity Demo")
    print("4. Interactive Search")
    print("5. Analytics & Stats")
    print("6. Backend Status")
    print("7. Exit")
    print()

def main():
    """Main application"""
    print("AI RECOMMENDATION ENGINE DEMO")
    print("Test your Pinecone + Supabase stack")
    print("="*50)

    # Load backend services
    services = load_backend_services()
    if not display_backend_status(services):
        print("\nERROR: Cannot proceed without backend services.")
        print("Please check your .env file and ensure Pinecone/Supabase are configured.")
        return

    print("\nSUCCESS: All systems ready!")

    while True:
        try:
            main_menu()
            choice = input("Choose option (1-7): ").strip()

            if choice == "1":
                add_product_demo(services)
            elif choice == "2":
                search_demo(services)
            elif choice == "3":
                similarity_demo(services)
            elif choice == "4":
                interactive_search(services)
            elif choice == "5":
                analytics_demo(services)
            elif choice == "6":
                display_backend_status(services)
            elif choice == "7":
                print("\nThanks for testing the AI Recommendation Engine!")
                break
            else:
                print("Invalid option. Please choose 1-7.")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()