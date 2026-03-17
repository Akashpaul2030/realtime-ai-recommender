"""Search tracking and analytics service.

Uses Redis when available, falls back to in-memory storage so the app
keeps working even without a Redis server.
"""

import json
import time
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any, Optional
from loguru import logger

from config import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD


# ======================================================================
# In-memory fallback (mirrors the Redis sorted-set / list behaviour)
# ======================================================================

class _InMemoryStore:
    """Minimal in-memory stand-in for the Redis data structures we need."""

    def __init__(self):
        self.lists: Dict[str, list] = defaultdict(list)       # key -> [items]
        self.zsets: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    # --- list ops ---
    def lpush(self, key: str, value: str):
        self.lists[key].insert(0, value)

    def ltrim(self, key: str, start: int, stop: int):
        self.lists[key] = self.lists[key][start:stop + 1]

    def lrange(self, key: str, start: int, stop: int) -> list:
        return self.lists[key][start:stop + 1]

    # --- sorted-set ops ---
    def zincrby(self, key: str, amount: float, member: str):
        self.zsets[key][member] += amount

    def zrevrange(self, key: str, start: int, stop: int, withscores: bool = False):
        items = sorted(self.zsets[key].items(), key=lambda x: x[1], reverse=True)
        sliced = items[start:stop + 1]
        if withscores:
            return sliced
        return [m for m, _ in sliced]

    # --- pipeline (immediate execution) ---
    def pipeline(self):
        return _InMemoryPipeline(self)


class _InMemoryPipeline:
    """Buffers calls then executes them all in execute()."""

    def __init__(self, store: _InMemoryStore):
        self._store = store
        self._ops: list = []

    def lpush(self, key, value):
        self._ops.append(("lpush", key, value))
        return self

    def ltrim(self, key, start, stop):
        self._ops.append(("ltrim", key, start, stop))
        return self

    def zincrby(self, key, amount, member):
        self._ops.append(("zincrby", key, amount, member))
        return self

    def execute(self):
        for op in self._ops:
            getattr(self._store, op[0])(*op[1:])
        self._ops.clear()


# ======================================================================
# SearchTracker
# ======================================================================

class SearchTracker:
    """Tracks user searches and provides analytics data."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def _ensure_initialized(self):
        if self._initialized:
            return

        # Try Redis first, fall back to in-memory
        try:
            import redis as _redis
            r = _redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            r.ping()
            self._store = r
            self._backend = "redis"
            logger.info("SearchTracker initialized (Redis)")
        except Exception:
            self._store = _InMemoryStore()
            self._backend = "memory"
            logger.warning("Redis unavailable — SearchTracker using in-memory fallback")

        self._initialized = True

    # ------------------------------------------------------------------
    # Write methods
    # ------------------------------------------------------------------

    def track_search(
        self, user_id: str, query: str, result_count: int
    ) -> None:
        """Record a search event for a user and update global analytics."""
        self._ensure_initialized()
        now = time.time()
        today = datetime.utcnow().strftime("%Y-%m-%d")

        entry = json.dumps({
            "query": query,
            "result_count": result_count,
            "timestamp": now,
        })

        pipe = self._store.pipeline()

        # Per-user search history (capped at 50)
        user_key = f"user:{user_id}:searches"
        pipe.lpush(user_key, entry)
        pipe.ltrim(user_key, 0, 49)

        # Global keyword counts – split query into words
        keywords = query.lower().split()
        for kw in keywords:
            pipe.zincrby("analytics:keyword:counts", 1, kw)

        # Full-query popularity
        pipe.zincrby("analytics:popular_queries", 1, query.lower())

        # Daily search volume
        pipe.zincrby("analytics:searches:daily", 1, today)

        # Zero-result tracking
        if result_count == 0:
            pipe.zincrby("analytics:zero_results", 1, query.lower())

        pipe.execute()

    def track_search_categories(self, categories: List[str]) -> None:
        """Increment category counts from search results."""
        if not categories:
            return
        self._ensure_initialized()
        pipe = self._store.pipeline()
        for cat in categories:
            pipe.zincrby("analytics:category:counts", 1, cat.lower())
        pipe.execute()

    # ------------------------------------------------------------------
    # Read methods – per-user
    # ------------------------------------------------------------------

    def get_user_searches(
        self, user_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Return recent search history for a user."""
        self._ensure_initialized()
        raw = self._store.lrange(f"user:{user_id}:searches", 0, limit - 1)
        return [json.loads(item) for item in raw]

    def get_user_interest_query(self, user_id: str) -> Optional[str]:
        """Build a weighted keyword string from recent searches.

        More recent searches carry more weight (appear more times).
        """
        self._ensure_initialized()
        raw = self._store.lrange(f"user:{user_id}:searches", 0, 19)
        if not raw:
            return None

        weighted_words: list[str] = []
        for i, item in enumerate(raw):
            entry = json.loads(item)
            query = entry.get("query", "")
            weight = 3 if i < 3 else (2 if i < 8 else 1)
            weighted_words.extend(query.split() * weight)

        if not weighted_words:
            return None
        return " ".join(weighted_words)

    # ------------------------------------------------------------------
    # Read methods – analytics
    # ------------------------------------------------------------------

    def get_top_keywords(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return most-searched keywords with counts."""
        self._ensure_initialized()
        results = self._store.zrevrange(
            "analytics:keyword:counts", 0, limit - 1, withscores=True
        )
        return [{"keyword": kw, "count": int(score)} for kw, score in results]

    def get_top_categories(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return most-searched categories with counts."""
        self._ensure_initialized()
        results = self._store.zrevrange(
            "analytics:category:counts", 0, limit - 1, withscores=True
        )
        return [{"category": cat, "count": int(score)} for cat, score in results]

    def get_search_trends(self, days: int = 30) -> List[Dict[str, Any]]:
        """Return daily search volume for the past N days."""
        self._ensure_initialized()
        results = self._store.zrevrange(
            "analytics:searches:daily", 0, days - 1, withscores=True
        )
        return [{"date": date, "count": int(score)} for date, score in results]

    def get_zero_results(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return queries that returned zero results, with counts."""
        self._ensure_initialized()
        results = self._store.zrevrange(
            "analytics:zero_results", 0, limit - 1, withscores=True
        )
        return [{"query": q, "count": int(score)} for q, score in results]

    def get_popular_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return most popular full queries (for new-user fallback)."""
        self._ensure_initialized()
        results = self._store.zrevrange(
            "analytics:popular_queries", 0, limit - 1, withscores=True
        )
        return [{"query": q, "count": int(score)} for q, score in results]


def get_search_tracker() -> SearchTracker:
    """Get (or create) the singleton SearchTracker instance."""
    tracker = SearchTracker()
    tracker._ensure_initialized()
    return tracker
