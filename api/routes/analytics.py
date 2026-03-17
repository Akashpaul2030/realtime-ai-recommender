"""Analytics API endpoints for search tracking data."""

import os
import sys
from typing import List, Dict, Any
from fastapi import APIRouter, Query
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.search_tracker import get_search_tracker

router = APIRouter()


@router.get("/top-keywords")
async def top_keywords(
    limit: int = Query(20, description="Number of keywords to return"),
):
    """Get the most frequently searched keywords."""
    tracker = get_search_tracker()
    return {"keywords": tracker.get_top_keywords(limit)}


@router.get("/top-categories")
async def top_categories(
    limit: int = Query(20, description="Number of categories to return"),
):
    """Get the most frequently searched categories."""
    tracker = get_search_tracker()
    return {"categories": tracker.get_top_categories(limit)}


@router.get("/search-trends")
async def search_trends(
    days: int = Query(30, description="Number of days to look back"),
):
    """Get daily search volume over the past N days."""
    tracker = get_search_tracker()
    return {"trends": tracker.get_search_trends(days)}


@router.get("/zero-results")
async def zero_results(
    limit: int = Query(20, description="Number of queries to return"),
):
    """Get queries that returned zero results (content gaps)."""
    tracker = get_search_tracker()
    return {"zero_results": tracker.get_zero_results(limit)}
