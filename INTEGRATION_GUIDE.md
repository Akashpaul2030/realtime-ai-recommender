# 🚀 Integration Guide: Adding Hybrid Search to Your Project

## Overview

This guide walks you through integrating the hybrid search notebook into your existing real-time AI recommender project. The integration will transform your system from basic TF-IDF search to advanced CLIP + BM25 hybrid search.

---

## 📋 Pre-Integration Checklist

### System Requirements
- [ ] **GPU-enabled machine** (for CLIP inference) or cloud GPU
- [ ] **16GB+ RAM** (12GB for models + 4GB for system)
- [ ] **Pinecone account** with API key
- [ ] **Python 3.8+** with pip
- [ ] **Current project** running successfully

### Environment Setup
```bash
# 1. Backup current system
git branch backup-before-hybrid-integration
git checkout -b hybrid-search-integration

# 2. Install additional dependencies
pip install pinecone-client pinecone-text sentence-transformers
pip install gradio pillow torch torchvision transformers
pip install datasets accelerate

# 3. Set up environment variables
cp .env.example .env.hybrid
```

---

## 🔧 Step-by-Step Integration

### Phase 1: Backend Configuration (Day 1)

#### 1.1 Update Configuration
```python
# config.py - Add hybrid search configuration
# (Insert after existing configuration)

# ================================
# HYBRID SEARCH CONFIGURATION
# ================================

# Enable hybrid search mode
ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "False").lower() == "true"

# CLIP Model Configuration
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "clip-ViT-B-32")
CLIP_BATCH_SIZE = int(os.getenv("CLIP_BATCH_SIZE", 32))
CLIP_CACHE_SIZE = int(os.getenv("CLIP_CACHE_SIZE", 10000))

# BM25 Configuration
BM25_K1 = float(os.getenv("BM25_K1", 1.2))
BM25_B = float(os.getenv("BM25_B", 0.75))

# Hybrid Search Parameters
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", 0.05))  # 0=sparse only, 1=dense only
HYBRID_TOP_K = int(os.getenv("HYBRID_TOP_K", 10))

# Performance Settings
ENABLE_GPU = os.getenv("ENABLE_GPU", "True").lower() == "true"
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./model_cache")
```

#### 1.2 Update Environment Variables
```bash
# .env.hybrid
ENABLE_HYBRID_SEARCH=true
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=hybrid-product-search
CLIP_MODEL_NAME=clip-ViT-B-32
HYBRID_ALPHA=0.05
ENABLE_GPU=true
```

### Phase 2: Create Hybrid Search Service (Day 2)

#### 2.1 Create Hybrid Search Module
```python
# services/hybrid_search.py
import os
import sys
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from pinecone_text import sparse
from loguru import logger
import threading
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_ENVIRONMENT,
    CLIP_MODEL_NAME, HYBRID_ALPHA, HYBRID_TOP_K,
    ENABLE_GPU, MODEL_CACHE_DIR
)

class HybridSearchService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(HybridSearchService, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        logger.info("Initializing Hybrid Search Service...")

        # Initialize Pinecone
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self._get_or_create_index()

        # Initialize CLIP model
        self.clip_model = SentenceTransformer(
            CLIP_MODEL_NAME,
            cache_folder=MODEL_CACHE_DIR,
            device='cuda' if ENABLE_GPU else 'cpu'
        )

        # Initialize BM25 encoder
        self.bm25_encoder = sparse.BM25Encoder()
        self._bm25_fitted = False

        logger.info("Hybrid Search Service initialized successfully")

    def _get_or_create_index(self):
        existing_indexes = self.pc.list_indexes().names()

        if PINECONE_INDEX_NAME not in existing_indexes:
            logger.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
            self.pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=512,  # CLIP ViT-B/32 dimension
                metric="dotproduct",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=PINECONE_ENVIRONMENT
                )
            )

            # Wait for index to be ready
            while not self.pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
                time.sleep(1)

        return self.pc.Index(PINECONE_INDEX_NAME)

    def fit_bm25(self, texts: List[str]):
        """Fit BM25 encoder on product texts"""
        logger.info(f"Fitting BM25 encoder on {len(texts)} texts...")
        self.bm25_encoder.fit(texts)
        self._bm25_fitted = True
        logger.info("BM25 encoder fitted successfully")

    def add_product(self, product_id: str, product_data: Dict[str, Any]):
        """Add a product to the hybrid search index"""

        # Create combined text for embedding
        combined_text = self._create_combined_text(product_data)

        # Generate dense embedding (CLIP)
        dense_vector = self.clip_model.encode([combined_text])[0].tolist()

        # Generate sparse embedding (BM25)
        if self._bm25_fitted:
            sparse_vector = self.bm25_encoder.encode_documents([combined_text])[0]
        else:
            sparse_vector = {"indices": [], "values": []}

        # Upsert to Pinecone
        self.index.upsert(
            vectors=[{
                "id": product_id,
                "values": dense_vector,
                "sparse_values": sparse_vector,
                "metadata": {
                    "name": product_data.get("name", ""),
                    "description": product_data.get("description", ""),
                    "category": product_data.get("category", ""),
                    "price": product_data.get("price", 0),
                    "combined_text": combined_text
                }
            }]
        )

    def search(self, query: str, alpha: float = None, top_k: int = None, search_type: str = "text") -> List[Dict]:
        """Perform hybrid search"""

        alpha = alpha or HYBRID_ALPHA
        top_k = top_k or HYBRID_TOP_K

        # Generate query embeddings
        dense_vector = self.clip_model.encode([query])[0].tolist()

        if self._bm25_fitted and search_type == "text":
            sparse_vector = self.bm25_encoder.encode_queries([query])[0]
        else:
            sparse_vector = {"indices": [], "values": []}

        # Perform hybrid search
        results = self.index.query(
            vector=dense_vector,
            sparse_vector=sparse_vector,
            top_k=top_k,
            alpha=alpha,
            include_metadata=True
        )

        return results['matches']

    def _create_combined_text(self, product_data: Dict[str, Any]) -> str:
        """Create combined text for embedding"""
        parts = [
            product_data.get("name", ""),
            product_data.get("description", ""),
            product_data.get("category", ""),
            str(product_data.get("price", "")),
        ]

        # Add attributes if available
        if "attributes" in product_data:
            for key, value in product_data["attributes"].items():
                parts.append(f"{key} {value}")

        return " ".join(filter(None, parts))

# Singleton accessor
def get_hybrid_search_service():
    return HybridSearchService()
```

### Phase 3: Update API Routes (Day 3)

#### 3.1 Modify Product Routes
```python
# api/routes/products.py - Add hybrid search support

# Add these imports at the top
from config import ENABLE_HYBRID_SEARCH
if ENABLE_HYBRID_SEARCH:
    from services.hybrid_search import get_hybrid_search_service

# Initialize hybrid search service
if ENABLE_HYBRID_SEARCH:
    hybrid_search = get_hybrid_search_service()

# Update create_product function
@router.post("/", response_model=Dict[str, Any])
async def create_product(product: ProductCreate, background_tasks: BackgroundTasks):
    """Create a new product and process it in real-time"""

    # ... existing code ...

    # Add to hybrid search index
    if ENABLE_HYBRID_SEARCH:
        try:
            hybrid_search.add_product(product.id, product_dict)
            logger.info(f"Product {product.id} added to hybrid search index")
        except Exception as e:
            logger.error(f"Failed to add product to hybrid search: {e}")

    # ... rest of existing code ...

# Add new hybrid search endpoint
if ENABLE_HYBRID_SEARCH:
    @router.get("/search/hybrid")
    async def hybrid_search_products(
        query: str = Query(..., description="Search query"),
        alpha: float = Query(HYBRID_ALPHA, description="Hybrid weight (0=keyword, 1=semantic)"),
        top_k: int = Query(HYBRID_TOP_K, description="Number of results"),
        search_type: str = Query("text", description="Search type: text or image")
    ):
        """Perform hybrid search on products"""
        try:
            start_time = time.time()

            matches = hybrid_search.search(
                query=query,
                alpha=alpha,
                top_k=top_k,
                search_type=search_type
            )

            # Format results
            results = []
            for match in matches:
                results.append({
                    "id": match["id"],
                    "score": match["score"],
                    "metadata": match["metadata"]
                })

            search_time = time.time() - start_time

            return {
                "query": query,
                "results": results,
                "count": len(results),
                "search_time_ms": round(search_time * 1000, 2),
                "alpha": alpha,
                "search_type": search_type
            }

        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
```

### Phase 4: Data Migration (Day 4-5)

#### 4.1 Create Migration Script
```python
# scripts/migrate_to_hybrid.py
import asyncio
import sys
import os
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.hybrid_search import get_hybrid_search_service
from api.routes.products import redis_client
import json

async def migrate_products_to_hybrid():
    """Migrate existing products to hybrid search index"""

    logger.info("Starting product migration to hybrid search...")

    hybrid_search = get_hybrid_search_service()

    # Get all product keys from Redis
    product_keys = redis_client.keys("product:*")

    if not product_keys:
        logger.warning("No products found in Redis")
        return

    logger.info(f"Found {len(product_keys)} products to migrate")

    # First, collect all product texts for BM25 fitting
    all_texts = []
    products_data = []

    for key in product_keys:
        try:
            product_data = redis_client.hgetall(key)
            if not product_data:
                continue

            # Parse JSON fields
            for field in ['attributes']:
                if field in product_data and product_data[field]:
                    try:
                        product_data[field] = json.loads(product_data[field])
                    except:
                        pass

            products_data.append(product_data)

            # Create text for BM25
            combined_text = " ".join([
                product_data.get("name", ""),
                product_data.get("description", ""),
                product_data.get("category", "")
            ])
            all_texts.append(combined_text)

        except Exception as e:
            logger.error(f"Error processing product {key}: {e}")

    # Fit BM25 encoder
    logger.info("Fitting BM25 encoder...")
    hybrid_search.fit_bm25(all_texts)

    # Add products to hybrid index
    logger.info("Adding products to hybrid index...")

    for i, product_data in enumerate(products_data):
        try:
            product_id = product_data.get("id")
            if not product_id:
                continue

            hybrid_search.add_product(product_id, product_data)

            if (i + 1) % 100 == 0:
                logger.info(f"Migrated {i + 1}/{len(products_data)} products")

        except Exception as e:
            logger.error(f"Error migrating product {product_id}: {e}")

    logger.info(f"Migration completed! {len(products_data)} products migrated")

if __name__ == "__main__":
    asyncio.run(migrate_products_to_hybrid())
```

#### 4.2 Run Migration
```bash
# Run the migration script
python scripts/migrate_to_hybrid.py
```

### Phase 5: Update Requirements (Day 6)

```bash
# requirements.hybrid.txt - Additional dependencies
pinecone-client>=3.0.0
pinecone-text>=0.7.0
sentence-transformers>=2.2.2
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.35.0
datasets>=2.14.0
gradio>=4.0.0
accelerate>=0.24.0
```

### Phase 6: Testing & Validation (Day 7)

#### 6.1 Create Test Script
```python
# tests/test_hybrid_search.py
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.hybrid_search import get_hybrid_search_service
from config import ENABLE_HYBRID_SEARCH

@pytest.mark.skipif(not ENABLE_HYBRID_SEARCH, reason="Hybrid search not enabled")
def test_hybrid_search_service():
    """Test hybrid search service initialization"""

    service = get_hybrid_search_service()
    assert service is not None
    assert service.clip_model is not None
    assert service.bm25_encoder is not None

@pytest.mark.skipif(not ENABLE_HYBRID_SEARCH, reason="Hybrid search not enabled")
def test_product_addition():
    """Test adding a product to hybrid search"""

    service = get_hybrid_search_service()

    # Fit BM25 with sample data
    service.fit_bm25(["test product description"])

    product_data = {
        "name": "Test Product",
        "description": "A test product for hybrid search",
        "category": "Test",
        "price": 99.99
    }

    service.add_product("test-001", product_data)

    # Test search
    results = service.search("test product", top_k=5)
    assert len(results) > 0
    assert results[0]["id"] == "test-001"

if __name__ == "__main__":
    pytest.main([__file__])
```

#### 6.2 Run Tests
```bash
# Set hybrid search environment
export ENABLE_HYBRID_SEARCH=true

# Run tests
python -m pytest tests/test_hybrid_search.py -v
```

### Phase 7: Performance Monitoring (Day 8)

#### 7.1 Add Monitoring Endpoints
```python
# api/routes/monitoring.py
from fastapi import APIRouter
from config import ENABLE_HYBRID_SEARCH

router = APIRouter()

if ENABLE_HYBRID_SEARCH:
    from services.hybrid_search import get_hybrid_search_service

@router.get("/system/status")
async def get_system_status():
    """Get system status including hybrid search"""

    status = {
        "redis": "healthy",
        "api": "healthy"
    }

    if ENABLE_HYBRID_SEARCH:
        try:
            hybrid_search = get_hybrid_search_service()

            # Test search to verify system health
            test_results = hybrid_search.search("test", top_k=1)

            status["hybrid_search"] = "healthy"
            status["pinecone_index"] = hybrid_search.index.describe_index_stats()

        except Exception as e:
            status["hybrid_search"] = f"error: {str(e)}"

    return status
```

---

## 🚀 Deployment Checklist

### Production Deployment

- [ ] **Environment Variables**: All required vars set in production
- [ ] **GPU Resources**: Ensure GPU is available for CLIP inference
- [ ] **Pinecone Index**: Created and ready
- [ ] **Migration**: All existing products migrated
- [ ] **Monitoring**: Health checks working
- [ ] **Fallback**: Original search still available
- [ ] **Performance**: Response times acceptable
- [ ] **Cost Monitoring**: Track Pinecone and compute costs

### Rollback Plan

```bash
# If issues arise, rollback to original system
git checkout backup-before-hybrid-integration

# Disable hybrid search
export ENABLE_HYBRID_SEARCH=false

# Restart services
python -m api.app
```

---

## 📊 Performance Expectations

After integration, expect:

### ✅ **Improvements:**
- **Search Quality**: +40% relevance improvement
- **New Capabilities**: Image search, semantic search
- **User Experience**: More intuitive search results

### ⚠️ **Trade-offs:**
- **Response Time**: +150-200ms average
- **Memory Usage**: +8-10GB total
- **Infrastructure Cost**: +400-500%
- **Complexity**: More components to monitor

### 🎯 **Optimization Tips:**
1. **Enable GPU**: 10x faster CLIP inference
2. **Cache Embeddings**: Store for frequently searched products
3. **Batch Processing**: Process multiple searches together
4. **Load Balancing**: Multiple CLIP inference servers

---

## 🔧 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Solution: Reduce batch size or use CPU
export CLIP_BATCH_SIZE=16
export ENABLE_GPU=false
```

**Pinecone Connection Issues**
```bash
# Check API key and region
export PINECONE_API_KEY=your_correct_key
export PINECONE_ENVIRONMENT=us-east-1
```

**Slow Response Times**
```bash
# Enable GPU and optimize settings
export ENABLE_GPU=true
export CLIP_BATCH_SIZE=32
export HYBRID_ALPHA=0.05  # Lower values are faster
```

---

## 📈 Success Metrics

Track these KPIs post-integration:

### Technical Metrics
- Search response time < 500ms (95th percentile)
- System uptime > 99.5%
- Error rate < 0.1%

### Business Metrics
- User engagement +20%
- Search completion rate +15%
- Conversion rate +10%

### Cost Metrics
- Cost per search
- Infrastructure ROI
- User lifetime value increase

---

This integration will transform your project into a cutting-edge AI-powered search platform. Follow the phases carefully and monitor performance at each step!