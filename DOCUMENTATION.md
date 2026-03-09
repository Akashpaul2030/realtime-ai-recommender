# Real-Time AI Recommendation Engine — Project Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What We Built](#2-what-we-built)
3. [Architecture](#3-architecture)
4. [Deployment Journey](#4-deployment-journey)
5. [Issues Faced & Solutions](#5-issues-faced--solutions)
6. [How It Works End-to-End](#6-how-it-works-end-to-end)
7. [API Endpoints](#7-api-endpoints)
8. [Frontend Features](#8-frontend-features)
9. [Configuration & Environment](#9-configuration--environment)
10. [Future Improvements](#10-future-improvements)

---

## 1. Project Overview

**Goal:** Build a real-time AI-powered product recommendation engine that 10 end users can use to browse, search, and discover fashion products through a web interface.

**Tech Stack:**
- **Backend:** Python, FastAPI, Uvicorn
- **Vector Search:** Pinecone (hybrid search), Redis Stack (RediSearch + RedisJSON)
- **Embeddings:** HuggingFace Inference API (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim)
- **Keyword Search:** BM25 sparse vectors via `pinecone-text` library
- **Stream Processing:** Redis Streams (producer/consumer pattern)
- **Frontend:** Vanilla HTML/CSS/JavaScript SPA (no framework)
- **Deployment:** Azure B1s VM (1 vCPU, 1 GB RAM, Ubuntu 24.04)
- **Web Server:** Nginx reverse proxy (port 80 → FastAPI on port 8000)
- **Dataset:** `ashraq/fashion-product-images-small` from HuggingFace (200 products)

---

## 2. What We Built

### Core Features
1. **Hybrid Search** — Combines BM25 keyword matching with MiniLM semantic embeddings via Pinecone. Users can tune the balance with an alpha slider (0% = pure keyword, 100% = pure semantic).
2. **Product Recommendations** — "Similar products" based on vector similarity when viewing a product detail page.
3. **Category Browsing** — Browse products by category (Apparel, Footwear, Accessories, etc.).
4. **Personalized "For You" Page** — Tracks product views per user and generates personalized recommendations.
5. **Image Proxy** — API endpoint that fetches fresh signed image URLs from HuggingFace on demand (since stored URLs expire).
6. **Product Seeding** — Two scripts to populate data:
   - `seed_products.py` — Seeds 200 products into Redis via API
   - `seed_hybrid.py` — Seeds 200 products into Pinecone with dense + sparse vectors

### System Components
| Component | File(s) | Purpose |
|-----------|---------|---------|
| REST API | `api/app.py`, `api/routes/` | FastAPI endpoints for products, recommendations, image proxy |
| Hybrid Search Service | `services/hybrid_search.py` | BM25 + MiniLM + Pinecone integration |
| Vector Store | `services/vector_store.py` | Redis-based vector similarity search |
| Stream Processing | `services/stream_consumer.py`, `stream_producer.py` | Event-driven product updates |
| Embedding Model | `models/embeddings.py` | TF-IDF text embeddings (384-dim) |
| Recommendations | `models/recommendations.py` | Similarity, popularity, personalization logic |
| Frontend | `frontend/index.html` | Single-page application |
| Config | `config.py`, `.env` | Environment-based configuration |

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     User's Browser                           │
│              (SPA: frontend/index.html)                      │
│   Pages: Home, Search, Product Detail, For You               │
└──────────────────────┬───────────────────────────────────────┘
                       │ HTTP (port 80)
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                   Nginx Reverse Proxy                        │
│   /image/*  /products/*  /recommendations/*  → port 8000    │
│   /*  → frontend/index.html (SPA fallback)                  │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                FastAPI Application (port 8000)                │
│                                                              │
│  ┌─────────────┐  ┌──────────────────┐  ┌───────────────┐   │
│  │  Products    │  │ Recommendations  │  │ Image Proxy   │   │
│  │  CRUD API   │  │ Hybrid Search    │  │ /image/{id}   │   │
│  │             │  │ Similar Products │  │ → HuggingFace  │   │
│  │             │  │ Personalized     │  │   fresh URLs   │   │
│  └──────┬──────┘  └────────┬─────────┘  └───────────────┘   │
│         │                  │                                  │
└─────────┼──────────────────┼──────────────────────────────────┘
          │                  │
          ▼                  ▼
┌─────────────────┐  ┌─────────────────────────────────────────┐
│  Redis Stack    │  │         Pinecone (Serverless)            │
│  - Streams      │  │  - 384-dim dense vectors (MiniLM)       │
│  - RediSearch   │  │  - BM25 sparse vectors                  │
│  - RedisJSON    │  │  - Product metadata                     │
│  - Vector Index │  │  - Hybrid search (dotproduct metric)    │
└─────────────────┘  └─────────────────────────────────────────┘
```

### Data Flow

1. **Product Seeding:** `seed_hybrid.py` fetches fashion data from HuggingFace → generates MiniLM embeddings + BM25 sparse vectors → upserts to Pinecone with metadata
2. **Search Query:** User types query → frontend calls `/recommendations/hybrid-search` → API generates dense embedding via HF API + BM25 sparse vector → queries Pinecone with alpha-weighted hybrid → returns ranked results
3. **Product View:** User clicks product → frontend calls `/recommendations/hybrid-product/{id}` for details + `/recommendations/hybrid-similar/{id}` for similar items → API fetches from Pinecone
4. **Image Loading:** Frontend `<img>` src points to `/image/{product_id}` → API extracts row index from stored URL → fetches fresh signed URL from HuggingFace → 302 redirect → browser loads image

---

## 4. Deployment Journey

### Step 1: Azure VM Setup
- Created Azure B1s VM (1 vCPU, 1 GB RAM) in Central India region
- Ubuntu 24.04 LTS
- Opened ports: SSH (22), HTTP (80), HTTPS (443)
- SSH key authentication with `.pem` file

### Step 2: Server Configuration
```bash
# System updates
sudo apt update && sudo apt upgrade -y

# Python environment
sudo apt install python3-pip python3-venv -y
python3 -m venv venv
source venv/bin/activate

# Redis Stack (with RediSearch + RedisJSON)
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb jammy main" | sudo tee /etc/apt/sources.list.d/redis.list
sudo apt update && sudo apt install redis-stack-server -y

# Nginx
sudo apt install nginx -y

# Clone project
git clone https://github.com/Akashpaul2030/realtime-ai-recommender.git
cd realtime-ai-recommender
pip install -r requirements.txt
```

### Step 3: Application Deployment
- Configured `.env` with Pinecone API key, HuggingFace token, Redis settings
- Set up Nginx reverse proxy to route API and frontend traffic
- Started API server and stream consumer as background processes
- Ran `seed_hybrid.py` to index 200 fashion products into Pinecone

---

## 5. Issues Faced & Solutions

### Issue 1: scikit-learn Build Failure (OOM on 1GB VM)
**Problem:** `pip install scikit-learn==1.2.2` tried to compile from source on the 1GB VM and ran out of memory. The build process consumed all RAM and the VM became unresponsive for 30+ minutes.

**Solution:** Installed the latest scikit-learn (which has pre-built wheels for Ubuntu 24.04) instead of the pinned version 1.2.2:
```bash
pip install scikit-learn  # Gets latest with pre-built wheel
```

---

### Issue 2: Redis Missing RediSearch Module
**Problem:** Default Ubuntu Redis package doesn't include RediSearch/RedisJSON modules. The API failed on startup with `unknown command 'FT._LIST'` when trying to create vector indexes.

**Solution:** Installed Redis Stack from the official Redis packages repository:
```bash
# Added packages.redis.io/deb jammy repo
sudo apt install redis-stack-server
```

---

### Issue 3: Redis Vector Search KNN Syntax Error
**Problem:** `FT.SEARCH` returned `Syntax error near >[ ` when using the KNN vector search syntax `*=>[KNN {limit} @vector $query_vector AS score]`.

**Root Cause:** KNN syntax requires `DIALECT 2` parameter, which wasn't being passed.

**Solution:** Added `"DIALECT", 2` to the `FT.SEARCH` command in `services/vector_store.py`:
```python
results = self.redis.execute_command(
    "FT.SEARCH", VECTOR_INDEX_NAME,
    f"*=>[KNN {limit} @vector $query_vector AS score]",
    "PARAMS", 2, "query_vector", query_vector,
    "SORTBY", "score",
    "RETURN", 4, "score", "category", "name", "updated_at",
    "DIALECT", 2  # <-- This was missing
)
```

---

### Issue 4: Vector Search Returning 0 Results
**Problem:** After indexing 200 products and fixing the KNN syntax, similarity search returned 0 results despite products existing in Redis.

**Root Cause:** Redis returns cosine **distance** (0 = identical, 2 = opposite), but the code treated it as a **similarity score** and filtered with `>= min_score (0.75)`. All results had distance ~0.0-0.3, which was below the threshold.

**Solution:** Convert distance to similarity: `similarity_score = 1.0 - raw_score`
```python
# Before (wrong):
similarity_score = float(properties[j+1])  # This was distance, not similarity

# After (correct):
raw_score = float(properties[j+1])
similarity_score = 1.0 - raw_score  # Convert distance → similarity
```

---

### Issue 5: SSH Key Permissions on Windows
**Problem:** SSH connection failed with `UNPROTECTED PRIVATE KEY FILE` error when trying to connect to the Azure VM.

**Solution:** Fixed file permissions and moved key to `.ssh` directory:
```bash
# Move key to .ssh folder
move realtimeAI_key.pem C:\Users\akash\.ssh\

# Fix permissions (Windows)
icacls "C:\Users\akash\.ssh\realtimeAI_key.pem" /inheritance:r /grant:r "akash:R"
```

---

### Issue 6: Nginx Permission Denied (500 Error)
**Problem:** Nginx returned 500 Internal Server Error when trying to serve the frontend. Error log showed `Permission denied` accessing files in the user's home directory.

**Root Cause:** Nginx runs as `www-data` user, which couldn't access `/home/realtime_ai/`.

**Solution:** Set directory permissions:
```bash
chmod 755 /home/realtime_ai
chmod 755 /home/realtime_ai/realtime-ai-recommender
chmod 755 /home/realtime_ai/realtime-ai-recommender/frontend
```

---

### Issue 7: Nginx 404 for SPA Routes
**Problem:** Direct navigation to `http://IP/#search/shoes` returned 404 because Nginx used exact match `location = /` which only matched the root path.

**Solution:** Changed to `try_files` for SPA fallback:
```nginx
location / {
    try_files $uri $uri/ /index.html;
}
```

---

### Issue 8: HuggingFace Datasets Library OOM
**Problem:** The `datasets` library (used to load `ashraq/fashion-product-images-small`) consumed too much memory even in streaming mode, crashing the 1GB VM.

**Solution:** Replaced with lightweight HTTP calls to the HuggingFace Datasets Server REST API:
```python
# Instead of: datasets.load_dataset("ashraq/fashion-product-images-small")
# Used:
url = f"https://datasets-server.huggingface.co/rows?dataset={DATASET}&config=default&split=train&offset={offset}&length={count}"
response = requests.get(url, timeout=30)
```

---

### Issue 9: CLIP Model Not Available via HuggingFace Inference API
**Problem:** Initially tried to use CLIP (`openai/clip-vit-base-patch32`) for 512-dim embeddings, but got `StopIteration` error — CLIP is not available through the HuggingFace Inference API.

**Solution:** Switched to `sentence-transformers/all-MiniLM-L6-v2` (384-dim) which is fully supported:
```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
```
Also had to recreate the Pinecone index with 384 dimensions (was 512 for CLIP).

---

### Issue 10: Pinecone Rejecting All-Zero Vectors
**Problem:** When embedding generation failed for some products, the fallback was `[0.0] * 512`, which Pinecone rejected with an error.

**Solution:** Use small random values instead of zeros:
```python
import random
fallback_vector = [random.uniform(0.001, 0.01) for _ in range(EMBEDDING_DIM)]
```

---

### Issue 11: Wrong Pinecone Index Name
**Problem:** Hybrid search returned `Resource e-commerce not found` error. The `.env.example` had `PINECONE_INDEX_NAME=e-commerce` (old index) which was being loaded by `config.py`.

**Solution:** Updated both `.env` and `.env.example` to use `fashion-hybrid-search`:
```ini
PINECONE_INDEX_NAME=fashion-hybrid-search
```

---

### Issue 12: BM25 Not Fitted on API Startup
**Problem:** When the API started fresh, the BM25 encoder was not fitted (no training data). Calling `get_bm25_sparse()` returned empty vectors `{"indices": [], "values": []}`, and Pinecone rejected the query with `Sparse vector must contain at least one value`.

**Root Cause:** BM25 requires `fit()` with a corpus of texts before it can encode queries. The seed script fitted it during indexing, but the API server started with an unfitted BM25.

**Solution:** Three-layer fix in `services/hybrid_search.py`:
1. **Save to disk:** After seeding, save the fitted BM25 model to `bm25_model.json`
2. **Load on startup:** On first search, check for saved model file and load it
3. **Auto-fit from Pinecone:** If no saved model, query Pinecone for product metadata, build text corpus, and fit BM25
4. **Dense-only fallback:** If BM25 still can't be fitted, use dense-only search (skip sparse vectors)

```python
def _load_or_fit_bm25(self):
    # 1. Try loading from disk
    if os.path.exists(bm25_path):
        self.bm25 = BM25Encoder().load(bm25_path)
        return

    # 2. Fit from Pinecone metadata
    results = self.index.query(vector=[0.01]*384, top_k=200, include_metadata=True)
    texts = [build_text_from_metadata(m) for m in results]
    self.bm25.fit(texts)
    self.bm25.dump(bm25_path)

def hybrid_search(self, query, alpha, top_k):
    self._load_or_fit_bm25()  # Auto-fit if needed

    sparse = self.get_bm25_sparse(query)
    if not sparse["indices"]:  # BM25 failed
        # Fallback: dense-only search
        return self.index.query(vector=dense_vector, ...)
    else:
        # Full hybrid search
        return self.index.query(vector=hdense, sparse_vector=hsparse, ...)
```

---

### Issue 13: Frontend JavaScript Parse Error (Blank Page)
**Problem:** The frontend showed the header but the content area (`#app`) was completely empty — no hero section, no categories, no search results.

**Root Cause:** Nested quote escaping in `onerror` handlers caused a JS parse error that silently killed the entire IIFE:
```javascript
// BROKEN: \\' inside \\' causes parse error
onerror="this.outerHTML=\'<div class=\\'product-card-img-placeholder\\'>\'"

// The JS engine sees this as ending the string prematurely
```

**Solution:** Replaced with HTML entity escaping (`&quot;`) and a different DOM manipulation approach:
```javascript
// FIXED: Using &quot; entities
onerror="this.style.display=&quot;none&quot;;this.parentNode.insertAdjacentHTML(&quot;afterbegin&quot;,&quot;<div class=product-card-img-placeholder><span>&#128722;</span></div>&quot;)"
```

**Validation:** Added a Node.js check to verify JS syntax before deploying:
```javascript
const html = fs.readFileSync('frontend/index.html', 'utf8');
const match = html.match(/<script>([\s\S]*?)<\/script>/);
new Function(match[1]); // Throws if parse error
```

---

### Issue 14: Product Images Not Loading (Expired HuggingFace URLs)
**Problem:** After seeding, product images showed placeholder icons instead of actual images. The stored image URLs from HuggingFace were signed URLs with expiration timestamps that had already passed (HTTP 403).

**Example expired URL:**
```
https://datasets-server.huggingface.co/cached-assets/.../image.jpg?Expires=1773058177&Signature=...
```

**Solution:** Created an image proxy endpoint (`/image/{product_id}`) in `api/app.py` that:
1. Looks up the product in Pinecone to get the old URL
2. Extracts the row index from the URL path (`/train/{row_idx}/image/`)
3. Fetches a fresh signed URL from HuggingFace Datasets Server API
4. Returns a 302 redirect to the fresh URL
5. Caches URLs for 5 minutes to avoid repeated API calls

```python
@app.get("/image/{product_id}")
async def image_proxy(product_id: str):
    # Check cache first
    if product_id in _image_cache:
        return RedirectResponse(url=cached_url)

    # Get row index from Pinecone metadata
    row_idx = extract_row_index(old_url)

    # Fetch fresh URL from HuggingFace
    fresh_data = requests.get(f"{HF_API}?offset={row_idx}&length=1")
    new_url = fresh_data["rows"][0]["row"]["image"]["src"]

    # Cache and redirect
    _image_cache[product_id] = (new_url, time.time() + 300)
    return RedirectResponse(url=new_url, status_code=302)
```

Frontend updated to use `/image/{product_id}` instead of storing URLs directly.

---

### Issue 15: GitHub Push Protection (Secrets in Code)
**Problem:** `git push` was rejected by GitHub with `GH013: Repository rule violations — Push cannot contain secrets`. The Pinecone API key and HuggingFace token were hardcoded as default values in `services/hybrid_search.py`.

**Solution:**
1. Removed hardcoded secrets, replaced with empty string defaults:
   ```python
   PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
   HF_TOKEN = os.getenv("HF_TOKEN", "")
   ```
2. Soft-reset git history to before the secret was committed
3. Re-committed all changes in a single clean commit without secrets
4. Secrets remain only in the `.env` file on the VM (which is `.gitignore`d)

---

## 6. How It Works End-to-End

### User Searches "blue running shoes"

1. **Frontend** captures the query, reads the alpha slider value (default 5%), calls:
   ```
   GET /recommendations/hybrid-search?query=blue+running+shoes&limit=20&alpha=0.05
   ```

2. **API** routes to `hybrid_search()` in `HybridSearchService`:
   - Calls `_load_or_fit_bm25()` — loads BM25 from `bm25_model.json` (or auto-fits from Pinecone)
   - Calls HuggingFace Inference API to get 384-dim dense embedding (~300ms)
   - Calls BM25 encoder to get sparse vector (indices + values)
   - Scales vectors: `dense * alpha` and `sparse * (1 - alpha)`
   - Queries Pinecone with both vectors

3. **Pinecone** returns top-k matches with scores and metadata (name, category, price, image_url, etc.)

4. **API** formats results and returns JSON with product details and similarity scores

5. **Frontend** renders product cards with:
   - Image loaded via `/image/{product_id}` proxy
   - Name, category, gender, color pills
   - Price
   - Match score bar (e.g., "77% match")

### User Clicks a Product

1. Frontend navigates to `#product/{id}` and calls:
   - `GET /recommendations/hybrid-product/{id}` — product details from Pinecone
   - `POST /recommendations/track-view?product_id={id}` — tracks the view
   - `GET /recommendations/hybrid-similar/{id}?limit=6` — similar products

2. Product detail page shows image, attributes, description, and a grid of similar items

---

## 7. API Endpoints

### Health & Utility
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/image/{product_id}` | Image proxy (redirects to fresh HuggingFace URL) |

### Products (Redis-based CRUD)
| Method | Path | Description |
|--------|------|-------------|
| POST | `/products/` | Create product (triggers stream event) |
| GET | `/products/{product_id}` | Get product details |
| PUT | `/products/{product_id}` | Update product |
| DELETE | `/products/{product_id}` | Delete product |

### Recommendations (Hybrid Search via Pinecone)
| Method | Path | Description |
|--------|------|-------------|
| GET | `/recommendations/hybrid-search` | Hybrid search (query, alpha, limit) |
| GET | `/recommendations/hybrid-similar/{product_id}` | Find similar products |
| GET | `/recommendations/hybrid-product/{product_id}` | Get product from Pinecone |
| GET | `/recommendations/personalized` | Personalized recommendations (header: user_id) |
| POST | `/recommendations/track-view` | Track product view |
| GET | `/recommendations/search` | Text-based search |
| GET | `/recommendations/category/{category}` | Category recommendations |

---

## 8. Frontend Features

### Pages
- **Home** — Hero banner with search, category cards (Apparel, Accessories, Footwear, Personal Care, Sporting Goods)
- **Search** — Results grid with alpha slider, match scores, product cards with images
- **Product Detail** — Full product image, attributes table, price, similar products grid
- **For You** — Personalized recommendations based on viewed products

### UI Components
- Sticky header with logo, search bar, navigation, user selector (10 users)
- Product cards with lazy-loaded images, category/gender/color pills, price, score bar
- Alpha slider for keyword vs semantic search balance
- Responsive layout (mobile/tablet/desktop)
- Color-coded dots for 50+ fashion colors
- Gradient placeholders when images are unavailable

### Tech Details
- Pure vanilla JavaScript (no React/Vue/Angular)
- SPA routing via URL hash (`#home`, `#search/query`, `#product/id`, `#foryou`)
- Fetch API with AbortController for search cancellation
- localStorage for user preference persistence
- CSS custom properties (variables) for theming
- Google Fonts (Inter)

---

## 9. Configuration & Environment

### Required Environment Variables (`.env`)
```ini
# Pinecone (required for hybrid search)
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=fashion-hybrid-search

# HuggingFace (required for embeddings)
HF_TOKEN=hf_...

# Redis (required)
REDIS_HOST=localhost
REDIS_PORT=6379

# Backend selection
BACKEND_TYPE=redis
VECTOR_STORE_TYPE=redis
EVENT_PROCESSOR_TYPE=redis
DATA_STORE_TYPE=redis

# Search tuning
SIMILARITY_THRESHOLD=0.3
VECTOR_DIMENSION=384
```

### Nginx Configuration (`/etc/nginx/sites-available/realtime-ai`)
```nginx
server {
    listen 80;
    server_name _;
    root /home/realtime_ai/realtime-ai-recommender/frontend;
    index index.html;

    location /image/           { proxy_pass http://127.0.0.1:8000; }
    location /products/        { proxy_pass http://127.0.0.1:8000; }
    location /recommendations/ { proxy_pass http://127.0.0.1:8000; }
    location /health           { proxy_pass http://127.0.0.1:8000; }
    location /docs             { proxy_pass http://127.0.0.1:8000; }
    location /openapi.json     { proxy_pass http://127.0.0.1:8000; }

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

---

## 10. Future Improvements

- **Systemd services** — Auto-start API and stream consumer on VM reboot
- **HTTPS** — Add SSL certificate via Let's Encrypt / Certbot
- **More products** — Scale from 200 to 5000+ products
- **Click analytics** — Track which recommendations users actually click
- **A/B testing** — Compare different alpha values for search quality
- **Image caching** — Cache product images on the VM instead of proxying every request
- **Rate limiting** — Protect API from abuse
- **User authentication** — Replace the simple user selector with real auth
