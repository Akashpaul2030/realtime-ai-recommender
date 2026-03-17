 
import os
import sys
import time
import requests as http_requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import uvicorn

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import API_HOST, API_PORT, DEBUG_MODE
from api.routes import products, recommendations, analytics, homepage
from api.middleware.logging import LoggingMiddleware


# Initialize FastAPI app
app = FastAPI(
    title="E-commerce Real-time AI API",
    description="API for real-time product analysis and recommendations",
    version="1.0.0",
    debug=DEBUG_MODE
)

# Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # In production, restrict this to your frontend domain
#     allow_credentials=False,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Add custom logging middleware
# app.add_middleware(LoggingMiddleware)

# Include routers
app.include_router(products.router, prefix="/products", tags=["products"])
app.include_router(recommendations.router, prefix="/recommendations", tags=["recommendations"])
app.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
app.include_router(homepage.router, prefix="/homepage", tags=["homepage"])


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}


# --- Image proxy: fetch fresh signed URL from HuggingFace ---
_image_cache = {}  # product_id -> (url, expiry_time)

HF_DATASET = "ashraq/fashion-product-images-small"
HF_ROWS_API = "https://datasets-server.huggingface.co/rows"


@app.get("/image/{product_id}")
async def image_proxy(product_id: str):
    """Redirect to a fresh HuggingFace signed image URL for a product."""
    now = time.time()

    # Check cache (URLs valid for ~10 min, cache for 5)
    if product_id in _image_cache:
        cached_url, expiry = _image_cache[product_id]
        if now < expiry:
            return RedirectResponse(url=cached_url, status_code=302)

    # Extract numeric ID from product_id like "fashion-51352"
    raw_id = product_id.replace("fashion-", "")

    # Try to find the row in the dataset by searching nearby offsets
    try:
        # Use the search endpoint to find the row
        row_url = f"{HF_ROWS_API}?dataset={HF_DATASET}&config=default&split=train&offset=0&length=1"
        # We need to find the row by ID - search through the dataset
        # The row index is not the same as the product ID, so we query Pinecone for the image_url
        # and fetch a fresh one from HF
        from services.hybrid_search import get_hybrid_search
        service = get_hybrid_search()
        service._ensure_initialized()
        fetch_result = service.index.fetch(ids=[product_id])

        if product_id not in fetch_result.vectors:
            raise HTTPException(status_code=404, detail="Product not found")

        metadata = fetch_result.vectors[product_id].metadata
        old_url = metadata.get("image_url", "")

        if not old_url:
            raise HTTPException(status_code=404, detail="No image for this product")

        # Extract the row index from the old URL path
        # URL pattern: .../default/train/{row_idx}/image/image.jpg?...
        import re
        match = re.search(r'/train/(\d+)/image/', old_url)
        if not match:
            raise HTTPException(status_code=404, detail="Cannot determine image row")

        row_idx = int(match.group(1))

        # Fetch fresh row from HF API to get new signed URL
        fresh_url = f"{HF_ROWS_API}?dataset={HF_DATASET}&config=default&split=train&offset={row_idx}&length=1"
        resp = http_requests.get(fresh_url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        rows = data.get("rows", [])
        if not rows:
            raise HTTPException(status_code=404, detail="Row not found in dataset")

        image_info = rows[0].get("row", {}).get("image", {})
        if isinstance(image_info, dict) and image_info.get("src"):
            new_url = image_info["src"]
            _image_cache[product_id] = (new_url, now + 300)  # Cache 5 min
            return RedirectResponse(url=new_url, status_code=302)

        raise HTTPException(status_code=404, detail="No image in dataset row")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image proxy error for {product_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to track processing time for each request"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


if __name__ == "__main__":
    # Configure logger
    logger.info(f"Starting API server on {API_HOST}:{API_PORT}")
    
    # Start server
    uvicorn.run(
        "api.app:app",
        host=API_HOST,
        port=API_PORT,
        reload=DEBUG_MODE
    )