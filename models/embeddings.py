import os
import numpy as np
from typing import List, Dict, Any, Union
from datetime import datetime
import time
import threading
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VECTOR_DIMENSION


class EmbeddingModel:
    """Simplified embedding model using TF-IDF instead of sentence-transformers"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                logger.info("Initializing simplified TF-IDF embedding model")
                start_time = time.time()
                cls._instance = super(EmbeddingModel, cls).__new__(cls)
                # Initialize the model
                cls._instance.model = TfidfVectorizer(
                    max_features=VECTOR_DIMENSION,
                    stop_words='english'
                )
                # Pre-fit with some generic texts to establish vocabulary
                sample_texts = [
                    "product item goods purchase buy shop retail",
                    "electronics technology devices gadgets computer phone",
                    "clothing apparel fashion wear dress shirt pants",
                    "furniture home decor interior design",
                    "food grocery ingredients meal recipe"
                ]
                cls._instance.model.fit(sample_texts)
                cls._instance.dimension = VECTOR_DIMENSION
                logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
            return cls._instance
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text input"""
        start_time = time.time()
        if not text:
            # Return zero vector if text is empty
            embedding = np.zeros(self.dimension)
        else:
            # Generate sparse matrix and convert to dense
            sparse_embedding = self.model.transform([text])
            embedding = sparse_embedding.toarray()[0]
            # Normalize the vector
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        logger.debug(f"Embedding generated in {time.time() - start_time:.4f} seconds")
        return embedding
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of text inputs"""
        start_time = time.time()
        if not texts:
            return np.array([])
        
        # Generate sparse matrix and convert to dense
        sparse_embeddings = self.model.transform(texts)
        embeddings = sparse_embeddings.toarray()
        
        # Normalize each vector
        for i in range(embeddings.shape[0]):
            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] = embeddings[i] / norm
        
        logger.debug(f"Batch embeddings ({len(texts)} items) generated in {time.time() - start_time:.4f} seconds")
        return embeddings
    
    def get_product_embedding(self, product: Dict[str, Any]) -> np.ndarray:
        """Generate embedding for a product by combining name, description, and category"""
        # Combine product fields into a single text representation
        product_text = f"{product.get('name', '')} {product.get('description', '')} Category: {product.get('category', '')}"
        
        # Add important attributes if available
        if 'attributes' in product and product['attributes']:
            for key, value in product['attributes'].items():
                if isinstance(value, (str, int, float)):
                    product_text += f" {key}: {value}"
        
        return self.get_embedding(product_text)
    
    @property
    def embedding_dimension(self) -> int:
        """Return the embedding dimension"""
        return self.dimension


# Convenience function to get the singleton instance
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()