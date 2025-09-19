"""Tests for data schemas."""
import pytest
from pydantic import ValidationError
from data.schemas import Product, ProductCreate, ProductRecommendation, RecommendationResponse


def test_product_create_valid():
    """Test valid product creation."""
    product_data = {
        "name": "Test Product",
        "description": "Test description",
        "category": "Test Category",
        "price": 99.99,
        "sku": "TEST-001",
        "id": "test-001"
    }
    product = ProductCreate(**product_data)
    assert product.name == "Test Product"
    assert product.price == 99.99
    assert product.id == "test-001"
    assert product.sku == "TEST-001"


def test_product_create_auto_id():
    """Test product creation with auto-generated ID."""
    product_data = {
        "name": "Test Product",
        "description": "Test description",
        "category": "Test Category",
        "price": 99.99,
        "sku": "TEST-001"
    }
    product = ProductCreate(**product_data)
    assert product.name == "Test Product"
    assert product.id is not None  # Should have auto-generated ID


def test_product_create_missing_required_fields():
    """Test product creation with missing required fields."""
    product_data = {
        "name": "Test Product"
        # Missing other required fields
    }
    with pytest.raises(ValidationError):
        ProductCreate(**product_data)


def test_product_recommendation_valid():
    """Test valid product recommendation."""
    recommendation_data = {
        "product_id": "test-001",
        "score": 0.95,
        "recommendation_type": "similar"
    }
    recommendation = ProductRecommendation(**recommendation_data)
    assert recommendation.product_id == "test-001"
    assert recommendation.score == 0.95
    assert recommendation.recommendation_type == "similar"


def test_recommendation_response():
    """Test recommendation response structure."""
    response_data = {
        "recommendations": [
            {
                "product_id": "test-001",
                "score": 0.95,
                "recommendation_type": "similar"
            }
        ]
    }
    response = RecommendationResponse(**response_data)
    assert len(response.recommendations) == 1
    assert response.request_id is not None  # Should have auto-generated ID
    assert response.generated_at is not None  # Should have auto-generated timestamp