"""Homepage personalized recommendations endpoint."""

import os
import sys
from typing import Optional
from fastapi import APIRouter, Query, Header
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.search_tracker import get_search_tracker

router = APIRouter()


@router.get("/recommendations")
async def homepage_recommendations(
    limit: int = Query(5, description="Number of products to return"),
    user_id: Optional[str] = Header(None, description="User ID for personalization"),
):
    """Return personalized product recommendations for the homepage.

    - Returning users: recommendations based on search history.
    - New users: trending products based on popular queries.
    - Cold start: hardcoded fallback query.
    """
    from services.hybrid_search import get_hybrid_search

    tracker = get_search_tracker()
    service = get_hybrid_search()

    query_text = None
    source = "fallback"

    # 1. Try user's search history
    if user_id:
        interest = tracker.get_user_interest_query(user_id)
        if interest:
            query_text = interest
            source = "personalized"

    # 2. Fallback: popular queries across all users
    if not query_text:
        popular = tracker.get_popular_queries(limit=5)
        if popular:
            query_text = " ".join(p["query"] for p in popular)
            source = "trending"

    # 3. Hardcoded cold-start fallback
    if not query_text:
        query_text = "popular trending fashion"
        source = "fallback"

    try:
        results = service.hybrid_search(
            query=query_text, alpha=0.3, top_k=limit
        )
        return {
            "source": source,
            "results": results,
            "count": len(results),
        }
    except Exception as e:
        logger.error(f"Homepage recommendations failed: {e}")
        return {"source": source, "results": [], "count": 0}
