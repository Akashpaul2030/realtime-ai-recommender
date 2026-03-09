"""Seed 200 fashion products into Pinecone with CLIP + BM25 hybrid vectors."""

import os
import requests
import random
import time

API_BASE = "https://datasets-server.huggingface.co/rows"
DATASET = "ashraq/fashion-product-images-small"
HF_IMAGE_BASE = "https://datasets-server.huggingface.co/assets/ashraq/fashion-product-images-small/--/default/train"

PRICE_RANGES = {
    "Apparel": (15.99, 149.99),
    "Accessories": (9.99, 89.99),
    "Footwear": (29.99, 199.99),
    "Personal Care": (4.99, 49.99),
}

def generate_price(category):
    low, high = PRICE_RANGES.get(category, (9.99, 99.99))
    return round(random.uniform(low, high), 2)

def fetch_products(count=200):
    """Fetch products from HuggingFace dataset API."""
    print("Fetching products from HuggingFace...")
    products = []

    offsets = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 43000]
    per_offset = count // len(offsets)

    for offset in offsets:
        url = f"{API_BASE}?dataset={DATASET}&config=default&split=train&offset={offset}&length={per_offset}"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            for row_data in data.get("rows", []):
                row = row_data.get("row", {})
                row_idx = row_data.get("row_idx", offset)

                # Get image URL from the dataset
                image_info = row.get("image", {})
                image_url = ""
                if isinstance(image_info, dict) and image_info.get("src"):
                    image_url = image_info["src"]

                products.append({
                    "id": f"fashion-{row.get('id', row_idx)}",
                    "name": row.get("productDisplayName", "") or f"{row.get('articleType', '')} {row.get('baseColour', '')}",
                    "description": build_description(row),
                    "category": row.get("masterCategory", "General"),
                    "price": generate_price(row.get("masterCategory", "")),
                    "sku": f"FSH-{row.get('id', row_idx)}",
                    "image_url": image_url,
                    "attributes": {
                        "gender": row.get("gender", ""),
                        "subCategory": row.get("subCategory", ""),
                        "articleType": row.get("articleType", ""),
                        "baseColour": row.get("baseColour", ""),
                        "season": row.get("season", ""),
                        "usage": row.get("usage", ""),
                        "year": str(row.get("year", "")),
                    }
                })
            print(f"  Fetched {len(products)} products (offset={offset})")
        except Exception as e:
            print(f"  Warning: Failed at offset {offset}: {e}")

    return products

def build_description(row):
    parts = []
    name = row.get("productDisplayName", "")
    if name:
        parts.append(name)
    details = []
    if row.get("baseColour"):
        details.append(f"{row['baseColour']} colored")
    if row.get("articleType"):
        details.append(row["articleType"].lower())
    if row.get("usage"):
        details.append(f"for {row['usage'].lower()} use")
    if row.get("season"):
        details.append(f"perfect for {row['season'].lower()}")
    if row.get("gender") and row["gender"] != "Unisex":
        details.append(f"designed for {row['gender'].lower()}")
    if details:
        parts.append(". ".join(details).capitalize() + ".")
    return " - ".join(parts) if parts else "Fashion product"

def main():
    products = fetch_products(200)
    print(f"\nTotal products: {len(products)}")

    # Show categories
    cats = {}
    for p in products:
        cat = p["category"]
        cats[cat] = cats.get(cat, 0) + 1
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # Initialize hybrid search service
    print("\nInitializing hybrid search service...")
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from services.hybrid_search import get_hybrid_search

    service = get_hybrid_search()

    # Index all products
    print(f"\nIndexing {len(products)} products with CLIP + BM25...")
    print("(This will take a few minutes — CLIP API calls take ~300ms each)\n")

    indexed = service.index_products_batch(products, batch_size=10)
    print(f"\nDone! Indexed {indexed} products.")

    # Wait a moment
    time.sleep(3)

    # Check index stats
    stats = service.get_index_stats()
    print(f"\nPinecone index stats:")
    print(f"  Total vectors: {stats['total_vectors']}")
    print(f"  Dimension: {stats['dimension']}")

    # Test hybrid search
    test_queries = ["blue jeans pant", "red dress for women", "sports shoes", "winter jacket"]
    for query in test_queries:
        print(f"\nSearch: '{query}'")
        results = service.hybrid_search(query, alpha=0.05, top_k=3)
        for r in results:
            meta = r["metadata"]
            print(f"  {r['product_id']} | {meta.get('name', '?')[:50]} | score={r['score']:.4f}")
            if meta.get("image_url"):
                print(f"    image: {meta['image_url'][:80]}...")

if __name__ == "__main__":
    main()
