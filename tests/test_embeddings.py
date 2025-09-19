"""Tests for embedding models."""
import pytest
import numpy as np
from models.embeddings import EmbeddingModel, get_embedding_model


def test_embedding_model_singleton():
    """Test that EmbeddingModel is a singleton."""
    model1 = EmbeddingModel()
    model2 = EmbeddingModel()
    assert model1 is model2


def test_get_embedding_model():
    """Test get_embedding_model function."""
    model = get_embedding_model()
    assert isinstance(model, EmbeddingModel)
    assert model.embedding_dimension == 384


def test_get_embedding():
    """Test single text embedding generation."""
    model = get_embedding_model()
    text = "blue jeans casual wear"
    embedding = model.get_embedding(text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)
    assert np.linalg.norm(embedding) <= 1.01  # Should be normalized (allow small float precision)


def test_get_embeddings_batch():
    """Test batch text embedding generation."""
    model = get_embedding_model()
    texts = [
        "blue jeans casual wear",
        "red dress formal evening",
        "white sneakers athletic shoes"
    ]
    embeddings = model.get_embeddings(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 3  # Should have 3 embeddings
    assert embeddings.shape[1] <= 384  # Dimensions should be <= max_features

    # Check that each embedding is normalized
    for i in range(embeddings.shape[0]):
        norm = np.linalg.norm(embeddings[i])
        assert norm <= 1.01  # Should be normalized


def test_get_text_embedding():
    """Test text embedding alias method."""
    model = get_embedding_model()
    text = "test product description"

    embedding1 = model.get_text_embedding(text)
    embedding2 = model.get_embedding(text)

    assert isinstance(embedding1, np.ndarray)
    assert np.array_equal(embedding1, embedding2)


def test_get_product_embedding():
    """Test product embedding generation."""
    model = get_embedding_model()
    product = {
        "name": "Blue Jeans",
        "description": "Comfortable casual denim pants",
        "category": "Clothing",
        "attributes": {
            "color": "blue",
            "size": "M"
        }
    }

    embedding = model.get_product_embedding(product)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)
    assert np.linalg.norm(embedding) <= 1.01  # Should be normalized


def test_empty_text_embedding():
    """Test handling of empty text."""
    model = get_embedding_model()
    embedding = model.get_embedding("")

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)
    assert np.all(embedding == 0)  # Should be zero vector


def test_empty_batch_embeddings():
    """Test handling of empty text list."""
    model = get_embedding_model()
    embeddings = model.get_embeddings([])

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.size == 0