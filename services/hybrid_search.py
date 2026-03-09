"""Hybrid search service using CLIP (HuggingFace API) + BM25 + Pinecone."""

import os
import json
import time
import requests
import numpy as np
from typing import List, Dict, Any, Optional
from loguru import logger
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "fashion-hybrid-search")
HF_TOKEN = os.getenv("HF_TOKEN", "")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEFAULT_ALPHA = 0.05  # Weight: 0=sparse only, 1=dense only


class HybridSearchService:
    """Singleton hybrid search service."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def _ensure_initialized(self):
        if self._initialized:
            return
        logger.info("Initializing Hybrid Search Service...")

        # Connect to Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = pc.Index(PINECONE_INDEX_NAME)
        logger.info(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")

        # Initialize BM25
        self.bm25 = BM25Encoder()
        self._bm25_fitted = False

        # HuggingFace Inference Client
        from huggingface_hub import InferenceClient
        self.hf_client = InferenceClient(token=HF_TOKEN)

        self._initialized = True
        logger.info("Hybrid Search Service initialized")

    def get_dense_embedding(self, text: str) -> List[float]:
        """Get dense text embedding from HuggingFace Inference API."""
        self._ensure_initialized()
        start = time.time()

        result = self.hf_client.feature_extraction(text, model=EMBEDDING_MODEL)
        embedding = np.array(result).flatten().tolist()

        logger.debug(f"Dense embedding generated in {time.time() - start:.3f}s")
        return embedding

    def get_dense_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get dense embeddings for a batch of texts (one by one via API)."""
        self._ensure_initialized()
        start = time.time()
        embeddings = []

        for text in texts:
            try:
                result = self.hf_client.feature_extraction(text, model=EMBEDDING_MODEL)
                emb = np.array(result).flatten().tolist()
                embeddings.append(emb)
            except Exception as e:
                logger.warning(f"Embedding failed for text, using random: {e}")
                import random as _rnd
                embeddings.append([_rnd.uniform(0.001, 0.01) for _ in range(EMBEDDING_DIM)])

        logger.debug(f"Dense batch ({len(texts)} texts) in {time.time() - start:.3f}s")
        return embeddings

    def _get_bm25_path(self) -> str:
        """Path to saved BM25 model."""
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bm25_model.json")

    def fit_bm25(self, texts: List[str]):
        """Fit BM25 encoder on corpus of texts and save to disk."""
        self._ensure_initialized()
        logger.info(f"Fitting BM25 on {len(texts)} documents...")
        self.bm25.fit(texts)
        self._bm25_fitted = True
        # Save fitted model to disk
        try:
            self.bm25.dump(self._get_bm25_path())
            logger.info(f"BM25 model saved to {self._get_bm25_path()}")
        except Exception as e:
            logger.warning(f"Could not save BM25 model: {e}")
        logger.info("BM25 fitted successfully")

    def _load_or_fit_bm25(self):
        """Load BM25 from disk, or fit from Pinecone metadata."""
        if self._bm25_fitted:
            return
        self._ensure_initialized()

        # Try loading saved model
        bm25_path = self._get_bm25_path()
        if os.path.exists(bm25_path):
            try:
                self.bm25 = BM25Encoder().load(bm25_path)
                self._bm25_fitted = True
                logger.info("BM25 model loaded from disk")
                return
            except Exception as e:
                logger.warning(f"Failed to load BM25 model: {e}")

        # Fit from Pinecone metadata
        logger.info("Fitting BM25 from Pinecone product metadata...")
        try:
            # Fetch a sample of product IDs from Pinecone
            stats = self.index.describe_index_stats()
            total = stats.total_vector_count
            if total == 0:
                logger.warning("No vectors in Pinecone, BM25 cannot be fitted")
                return

            # List vectors and fetch metadata to build corpus
            # Use a query with a random vector to get products
            random_vec = [0.01] * EMBEDDING_DIM
            results = self.index.query(vector=random_vec, top_k=min(total, 200), include_metadata=True)

            texts = []
            for match in results.matches:
                meta = match.metadata or {}
                parts = [meta.get("name", ""), meta.get("category", ""),
                         meta.get("articleType", ""), meta.get("baseColour", ""),
                         meta.get("gender", ""), meta.get("subCategory", ""),
                         meta.get("season", ""), meta.get("usage", ""),
                         meta.get("description", "")]
                text = " ".join(p for p in parts if p)
                if text.strip():
                    texts.append(text)

            if texts:
                self.bm25.fit(texts)
                self._bm25_fitted = True
                try:
                    self.bm25.dump(bm25_path)
                    logger.info(f"BM25 fitted on {len(texts)} products and saved")
                except Exception as e:
                    logger.warning(f"BM25 fitted but could not save: {e}")
            else:
                logger.warning("No text data from Pinecone to fit BM25")
        except Exception as e:
            logger.error(f"Failed to auto-fit BM25: {e}")

    def get_bm25_sparse(self, text: str) -> Dict:
        """Get BM25 sparse vector for a query."""
        if not self._bm25_fitted:
            logger.warning("BM25 not fitted, returning empty sparse vector")
            return {"indices": [], "values": []}
        return self.bm25.encode_queries([text])[0]

    def get_bm25_sparse_doc(self, text: str) -> Dict:
        """Get BM25 sparse vector for a document (different from query encoding)."""
        if not self._bm25_fitted:
            return {"indices": [], "values": []}
        return self.bm25.encode_documents([text])[0]

    def build_combined_text(self, product: Dict) -> str:
        """Build combined text from product fields for embedding."""
        parts = []
        if product.get("name"):
            parts.append(product["name"])
        if product.get("description"):
            parts.append(product["description"])
        if product.get("category"):
            parts.append(product["category"])

        attrs = product.get("attributes", {})
        for key in ["gender", "subCategory", "articleType", "baseColour", "season", "usage"]:
            val = attrs.get(key, "")
            if val:
                parts.append(str(val))

        return " ".join(parts)

    def index_product(self, product_id: str, product: Dict, dense_vector: List[float],
                      sparse_vector: Dict, image_url: str = ""):
        """Index a single product in Pinecone with dense + sparse vectors."""
        self._ensure_initialized()

        metadata = {
            "name": product.get("name", ""),
            "category": product.get("category", ""),
            "price": float(product.get("price", 0)),
            "description": product.get("description", "")[:500],
            "image_url": image_url,
        }

        attrs = product.get("attributes", {})
        for key in ["gender", "subCategory", "articleType", "baseColour", "season", "usage"]:
            metadata[key] = attrs.get(key, "")

        self.index.upsert(vectors=[{
            "id": product_id,
            "values": dense_vector,
            "sparse_values": sparse_vector,
            "metadata": metadata
        }])

    def index_products_batch(self, products: List[Dict], batch_size: int = 20):
        """Index multiple products in batches."""
        self._ensure_initialized()

        # Build combined texts for all products
        all_texts = [self.build_combined_text(p) for p in products]

        # Fit BM25 on all texts
        self.fit_bm25(all_texts)

        total = len(products)
        indexed = 0

        for i in range(0, total, batch_size):
            batch = products[i:i + batch_size]
            batch_texts = all_texts[i:i + batch_size]

            # Get CLIP embeddings for batch
            try:
                dense_vectors = self.get_dense_embeddings_batch(batch_texts)
            except Exception as e:
                logger.error(f"Failed to get CLIP embeddings for batch {i}: {e}")
                # Try one by one
                dense_vectors = []
                for text in batch_texts:
                    try:
                        dense_vectors.append(self.get_dense_embedding(text))
                    except Exception:
                        # Use tiny random vector instead of zeros (Pinecone rejects all-zero)
                        import random as _rnd
                        dense_vectors.append([_rnd.uniform(0.001, 0.01) for _ in range(EMBEDDING_DIM)])
                        logger.warning(f"Using random vector for failed embedding")

            # Get BM25 sparse vectors
            sparse_vectors = [self.get_bm25_sparse_doc(t) for t in batch_texts]

            # Build upsert batch
            vectors_to_upsert = []
            for j, product in enumerate(batch):
                pid = product.get("id", f"product-{i+j}")
                image_url = product.get("image_url", "")

                metadata = {
                    "name": product.get("name", ""),
                    "category": product.get("category", ""),
                    "price": float(product.get("price", 0)),
                    "description": product.get("description", "")[:500],
                    "image_url": image_url,
                }
                attrs = product.get("attributes", {})
                for key in ["gender", "subCategory", "articleType", "baseColour", "season", "usage"]:
                    metadata[key] = attrs.get(key, "")

                vectors_to_upsert.append({
                    "id": pid,
                    "values": dense_vectors[j],
                    "sparse_values": sparse_vectors[j],
                    "metadata": metadata
                })

            # Upsert to Pinecone
            self.index.upsert(vectors=vectors_to_upsert)
            indexed += len(batch)
            logger.info(f"Indexed {indexed}/{total} products")

        return indexed

    def hybrid_search(self, query: str, alpha: float = DEFAULT_ALPHA,
                      top_k: int = 10, filters: Optional[Dict] = None) -> List[Dict]:
        """Perform hybrid search combining BM25 + dense embeddings."""
        self._ensure_initialized()
        start = time.time()

        # Auto-fit BM25 if needed
        self._load_or_fit_bm25()

        # Generate dense vector
        dense_vector = self.get_dense_embedding(query)

        # Generate sparse vector (BM25)
        sparse_vector = self.get_bm25_sparse(query)

        # If BM25 not available, use dense-only search
        has_sparse = sparse_vector.get("indices") and len(sparse_vector["indices"]) > 0
        if not has_sparse:
            logger.info("BM25 unavailable, using dense-only search")
            query_params = {
                "vector": dense_vector,
                "top_k": top_k,
                "include_metadata": True,
            }
            if filters:
                query_params["filter"] = filters
            results = self.index.query(**query_params)
        else:
            # Scale vectors by alpha
            # alpha=0 -> sparse only, alpha=1 -> dense only
            hdense = [v * alpha for v in dense_vector]
            hsparse = {
                "indices": sparse_vector["indices"],
                "values": [v * (1 - alpha) for v in sparse_vector["values"]]
            }

            query_params = {
                "vector": hdense,
                "sparse_vector": hsparse,
                "top_k": top_k,
                "include_metadata": True,
            }
            if filters:
                query_params["filter"] = filters
            results = self.index.query(**query_params)

        # Format results
        search_results = []
        for match in results.matches:
            search_results.append({
                "product_id": match.id,
                "score": float(match.score),
                "metadata": dict(match.metadata) if match.metadata else {},
            })

        elapsed = time.time() - start
        logger.debug(f"Hybrid search for '{query}' returned {len(search_results)} results in {elapsed:.3f}s")
        return search_results

    def get_index_stats(self) -> Dict:
        """Get Pinecone index statistics."""
        self._ensure_initialized()
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "index_name": PINECONE_INDEX_NAME,
        }


# Singleton accessor
_service = None

def get_hybrid_search() -> HybridSearchService:
    global _service
    if _service is None:
        _service = HybridSearchService()
    return _service
