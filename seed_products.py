"""Seed script to load 200 products from HuggingFace fashion dataset into the API.
Uses lightweight HTTP download - no heavy libraries needed."""

import requests
import random
import time
import json

API_URL = "http://localhost:8000"

PRICE_RANGES = {
    "Apparel": (15.99, 149.99),
    "Accessories": (9.99, 89.99),
    "Footwear": (29.99, 199.99),
    "Personal Care": (4.99, 49.99),
    "Free Items": (0.99, 19.99),
}

def generate_price(category):
    low, high = PRICE_RANGES.get(category, (9.99, 99.99))
    return round(random.uniform(low, high), 2)

def generate_description(p):
    parts = []
    if p.get("productDisplayName"):
        parts.append(p["productDisplayName"])
    details = []
    if p.get("baseColour"):
        details.append(f"{p['baseColour']} colored")
    if p.get("articleType"):
        details.append(p["articleType"].lower())
    if p.get("usage"):
        details.append(f"for {p['usage'].lower()} use")
    if p.get("season"):
        details.append(f"perfect for {p['season'].lower()}")
    if p.get("gender") and p["gender"] != "Unisex":
        details.append(f"designed for {p['gender'].lower()}")
    if details:
        parts.append(". ".join(details).capitalize() + ".")
    return " - ".join(parts) if parts else "Fashion product"

def fetch_products():
    """Fetch 200 products from HuggingFace API using rows endpoint."""
    print("Fetching products from HuggingFace API...")
    products = []

    # Fetch in batches of 100, picking from different offsets for diversity
    offsets = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 43000]
    per_offset = 20

    for offset in offsets:
        url = f"https://datasets-server.huggingface.co/rows?dataset=ashraq/fashion-product-images-small&config=default&split=train&offset={offset}&length={per_offset}"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            for row in data.get("rows", []):
                item = row.get("row", {})
                products.append({
                    "id": item.get("id", 0),
                    "gender": item.get("gender", ""),
                    "masterCategory": item.get("masterCategory", ""),
                    "subCategory": item.get("subCategory", ""),
                    "articleType": item.get("articleType", ""),
                    "baseColour": item.get("baseColour", ""),
                    "season": item.get("season", ""),
                    "year": item.get("year", ""),
                    "usage": item.get("usage", ""),
                    "productDisplayName": item.get("productDisplayName", ""),
                })
            print(f"  Fetched {len(products)} products so far (offset={offset})")
        except Exception as e:
            print(f"  Warning: Failed at offset {offset}: {e}")

    return products

def main():
    products_raw = fetch_products()
    print(f"\nTotal products fetched: {len(products_raw)}")

    categories = {}
    for p in products_raw:
        cat = p.get("masterCategory", "Other")
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    print()

    # Check API health
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
        print("API is healthy!\n")
    except Exception as e:
        print(f"ERROR: Cannot reach API at {API_URL}: {e}")
        return

    # Push products to API
    success = 0
    failed = 0

    for idx, row in enumerate(products_raw):
        product = {
            "id": f"fashion-{row['id']}",
            "name": row["productDisplayName"] or f"{row.get('articleType', '')} {row.get('baseColour', '')}",
            "description": generate_description(row),
            "category": row.get("masterCategory") or "General",
            "price": generate_price(row.get("masterCategory", "")),
            "sku": f"FSH-{row['id']}",
            "attributes": {
                "gender": row.get("gender", ""),
                "subCategory": row.get("subCategory", ""),
                "articleType": row.get("articleType", ""),
                "baseColour": row.get("baseColour", ""),
                "season": row.get("season", ""),
                "usage": row.get("usage", ""),
                "year": str(row.get("year", "")),
            }
        }

        try:
            r = requests.post(f"{API_URL}/products/", json=product, timeout=10)
            if r.status_code in (200, 201):
                success += 1
            else:
                failed += 1
                if failed <= 3:
                    print(f"  WARN: {product['name'][:40]} -> {r.status_code}: {r.text[:100]}")
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  ERROR: {product['name'][:40]} -> {e}")

        total = len(products_raw)
        if (idx + 1) % 20 == 0:
            print(f"  Progress: {idx + 1}/{total} (success: {success}, failed: {failed})")

        time.sleep(0.1)

    print(f"\nDone! Loaded {success}/{success + failed} products.")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")

    # Wait for consumer to process
    print("\nWaiting 10 seconds for stream consumer to process embeddings...")
    time.sleep(10)

    # Test search
    print("\nTesting search for 'blue shirt'...")
    r = requests.get(f"{API_URL}/recommendations/search", params={"query": "blue shirt", "limit": 5})
    if r.status_code == 200:
        data = r.json()
        recs = data.get("recommendations", [])
        if recs:
            print(f"Found {len(recs)} recommendations!")
            for rec in recs:
                print(f"  - {rec['product_id']} (score: {rec.get('score', 'N/A')})")
        else:
            print("No recommendations yet (consumer may still be processing)")
    else:
        print(f"Search returned {r.status_code}: {r.text[:100]}")

if __name__ == "__main__":
    main()
