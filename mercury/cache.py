from typing import Optional, Any
import time
from collections import OrderedDict

class CacheManager:
    """LRU cache with TTL support."""
    
    def __init__(self, size: int = 1000, ttl: int = 300):
        self.cache = OrderedDict()
        self.ttl = ttl
        self.size = size
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired."""
        if key not in self.cache:
            return None
            
        value, timestamp = self.cache[key]
        if time.time() - timestamp > self.ttl:
            del self.cache[key]
            return None
            
        self.cache.move_to_end(key)
        return value
        
    def set(self, key: str, value: Any):
        """Add item to cache with timestamp."""
        if len(self.cache) >= self.size:
            self.cache.popitem(last=False)
            
        self.cache[key] = (value, time.time())
        self.cache.move_to_end(key) 