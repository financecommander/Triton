"""
Compilation Caching System

Provides multi-level caching for compilation results:
- Compilation result cache
- Type inference cache  
- Dependency tracking and invalidation
"""

import hashlib
import json
import os
import pickle
import time
from typing import Dict, Any, Optional, Set, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import threading


@dataclass
class CacheEntry:
    """Entry in the compilation cache."""
    key: str
    value: Any
    timestamp: float
    dependencies: Set[str] = field(default_factory=set)
    hit_count: int = 0
    size_bytes: int = 0
    
    def is_expired(self, max_age: float) -> bool:
        """Check if entry has expired."""
        return time.time() - self.timestamp > max_age


@dataclass
class CacheStats:
    """Statistics for cache performance."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def __str__(self) -> str:
        lines = [
            "Cache Statistics:",
            f"  Entries: {self.entry_count}",
            f"  Size: {self.size_bytes / 1024 / 1024:.2f} MB",
            f"  Hits: {self.hits}",
            f"  Misses: {self.misses}",
            f"  Hit Rate: {self.hit_rate * 100:.1f}%",
            f"  Evictions: {self.evictions}",
        ]
        return "\n".join(lines)


class CompilationCache:
    """
    Multi-level cache for compilation results.
    
    Features:
    - LRU eviction policy
    - Dependency tracking
    - Automatic invalidation
    - Size limits
    - Thread-safe operations
    """
    
    def __init__(
        self,
        max_size_mb: int = 500,
        max_entries: int = 1000,
        max_age_seconds: float = 3600,
        persist_path: Optional[Path] = None,
    ):
        """
        Initialize compilation cache.
        
        Args:
            max_size_mb: Maximum cache size in megabytes
            max_entries: Maximum number of cache entries
            max_age_seconds: Maximum age for cache entries in seconds
            persist_path: Optional path to persist cache to disk
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.max_age = max_age_seconds
        self.persist_path = persist_path
        
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # For LRU
        self._lock = threading.RLock()
        self._stats = CacheStats()
        
        # Load persisted cache if available
        if persist_path and persist_path.exists():
            self._load_from_disk()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired(self.max_age):
                self._remove_entry(key)
                self._stats.misses += 1
                return None
            
            # Update access order (LRU)
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            # Update stats
            entry.hit_count += 1
            self._stats.hits += 1
            
            return entry.value
    
    def put(
        self,
        key: str,
        value: Any,
        dependencies: Optional[Set[str]] = None,
        size_hint: Optional[int] = None,
    ):
        """
        Put value into cache.
        
        Args:
            key: Cache key
            value: Value to cache
            dependencies: Set of dependency keys
            size_hint: Estimated size in bytes (will be calculated if not provided)
        """
        with self._lock:
            # Calculate size if not provided
            if size_hint is None:
                try:
                    size_hint = len(pickle.dumps(value))
                except Exception:
                    size_hint = 1024  # Default estimate
            
            # Check if we need to evict entries
            while (
                len(self._cache) >= self.max_entries or
                self._stats.size_bytes + size_hint > self.max_size_bytes
            ):
                if not self._evict_lru():
                    break  # Cache is empty
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                dependencies=dependencies or set(),
                size_bytes=size_hint,
            )
            
            # Remove old entry if exists
            if key in self._cache:
                self._remove_entry(key)
            
            # Add new entry
            self._cache[key] = entry
            self._access_order.append(key)
            self._stats.entry_count += 1
            self._stats.size_bytes += size_hint
    
    def invalidate(self, key: str):
        """
        Invalidate a cache entry and all entries that depend on it.
        
        Args:
            key: Cache key to invalidate
        """
        with self._lock:
            if key not in self._cache:
                return
            
            # Find all entries that depend on this key
            to_invalidate = {key}
            for entry_key, entry in self._cache.items():
                if key in entry.dependencies:
                    to_invalidate.add(entry_key)
            
            # Remove all affected entries
            for k in to_invalidate:
                self._remove_entry(k)
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._stats = CacheStats()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                size_bytes=self._stats.size_bytes,
                entry_count=self._stats.entry_count,
            )
    
    def _evict_lru(self) -> bool:
        """
        Evict least recently used entry.
        
        Returns:
            True if an entry was evicted, False if cache is empty
        """
        if not self._access_order:
            return False
        
        # Get LRU key
        lru_key = self._access_order[0]
        self._remove_entry(lru_key)
        self._stats.evictions += 1
        
        return True
    
    def _remove_entry(self, key: str):
        """Remove an entry from cache."""
        if key not in self._cache:
            return
        
        entry = self._cache[key]
        self._stats.size_bytes -= entry.size_bytes
        self._stats.entry_count -= 1
        
        del self._cache[key]
        if key in self._access_order:
            self._access_order.remove(key)
    
    def persist(self):
        """Persist cache to disk."""
        if not self.persist_path:
            return
        
        with self._lock:
            try:
                self.persist_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.persist_path, 'wb') as f:
                    pickle.dump({
                        'cache': self._cache,
                        'access_order': self._access_order,
                        'stats': self._stats,
                    }, f)
            except Exception as e:
                print(f"Warning: Failed to persist cache: {e}")
    
    def _load_from_disk(self):
        """Load cache from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return
        
        try:
            with open(self.persist_path, 'rb') as f:
                data = pickle.load(f)
                self._cache = data.get('cache', {})
                self._access_order = data.get('access_order', [])
                self._stats = data.get('stats', CacheStats())
                
            # Remove expired entries
            expired = [
                key for key, entry in self._cache.items()
                if entry.is_expired(self.max_age)
            ]
            for key in expired:
                self._remove_entry(key)
                
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            self._cache.clear()
            self._access_order.clear()
            self._stats = CacheStats()


class TypeInferenceCache:
    """
    Cache for type inference results.
    
    Uses content-based hashing for cache keys.
    """
    
    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self._cache: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get_key(self, node: Any) -> str:
        """
        Generate cache key for an AST node.
        
        Args:
            node: AST node
            
        Returns:
            Hash-based cache key
        """
        # Create a deterministic representation of the node
        try:
            if hasattr(node, 'to_dict'):
                content = json.dumps(node.to_dict(), sort_keys=True)
            else:
                content = str(node)
            return hashlib.sha256(content.encode()).hexdigest()
        except Exception:
            # Fallback to id-based key (won't persist across runs)
            return f"node_{id(node)}"
    
    def get(self, node: Any) -> Optional[Any]:
        """
        Get cached type for a node.
        
        Args:
            node: AST node
            
        Returns:
            Cached type or None
        """
        key = self.get_key(node)
        with self._lock:
            if key in self._cache:
                self.hits += 1
                return self._cache[key]
            self.misses += 1
            return None
    
    def put(self, node: Any, type_result: Any):
        """
        Cache type inference result.
        
        Args:
            node: AST node
            type_result: Inferred type
        """
        key = self.get_key(node)
        with self._lock:
            # Evict oldest entry if at capacity
            if len(self._cache) >= self.max_entries:
                # Simple FIFO eviction
                first_key = next(iter(self._cache))
                del self._cache[first_key]
            
            self._cache[key] = type_result
    
    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class DependencyTracker:
    """
    Track dependencies between compilation units for cache invalidation.
    """
    
    def __init__(self):
        self._dependencies: Dict[str, Set[str]] = {}  # file -> dependencies
        self._reverse_deps: Dict[str, Set[str]] = {}  # dependency -> dependents
        self._checksums: Dict[str, str] = {}  # file -> checksum
        self._lock = threading.RLock()
    
    def add_dependency(self, file: str, dependency: str):
        """
        Add a dependency relationship.
        
        Args:
            file: Source file
            dependency: File that source depends on
        """
        with self._lock:
            if file not in self._dependencies:
                self._dependencies[file] = set()
            self._dependencies[file].add(dependency)
            
            if dependency not in self._reverse_deps:
                self._reverse_deps[dependency] = set()
            self._reverse_deps[dependency].add(file)
    
    def get_dependencies(self, file: str) -> Set[str]:
        """Get all dependencies of a file."""
        with self._lock:
            return self._dependencies.get(file, set()).copy()
    
    def get_dependents(self, file: str) -> Set[str]:
        """Get all files that depend on this file."""
        with self._lock:
            return self._reverse_deps.get(file, set()).copy()
    
    def compute_checksum(self, file: str) -> str:
        """
        Compute checksum for a file.
        
        Args:
            file: Path to file
            
        Returns:
            SHA256 checksum
        """
        try:
            with open(file, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except FileNotFoundError:
            return ""
    
    def has_changed(self, file: str) -> bool:
        """
        Check if a file has changed since last check.
        
        Args:
            file: Path to file
            
        Returns:
            True if file has changed or is new
        """
        current_checksum = self.compute_checksum(file)
        with self._lock:
            old_checksum = self._checksums.get(file)
            if old_checksum is None or old_checksum != current_checksum:
                self._checksums[file] = current_checksum
                return True
            return False
    
    def get_invalidated_files(self, changed_file: str) -> Set[str]:
        """
        Get all files that need recompilation due to a changed file.
        
        Args:
            changed_file: File that has changed
            
        Returns:
            Set of files that need recompilation
        """
        with self._lock:
            invalidated = {changed_file}
            
            # Add all dependents recursively
            to_check = [changed_file]
            while to_check:
                current = to_check.pop()
                dependents = self._reverse_deps.get(current, set())
                for dep in dependents:
                    if dep not in invalidated:
                        invalidated.add(dep)
                        to_check.append(dep)
            
            return invalidated
    
    def clear(self):
        """Clear all dependency information."""
        with self._lock:
            self._dependencies.clear()
            self._reverse_deps.clear()
            self._checksums.clear()


# Global cache instances
_compilation_cache: Optional[CompilationCache] = None
_type_inference_cache: Optional[TypeInferenceCache] = None
_dependency_tracker: Optional[DependencyTracker] = None


def get_compilation_cache(**kwargs) -> CompilationCache:
    """Get or create global compilation cache."""
    global _compilation_cache
    if _compilation_cache is None:
        # Default persist path
        cache_dir = Path.home() / '.triton' / 'cache'
        kwargs.setdefault('persist_path', cache_dir / 'compilation.cache')
        _compilation_cache = CompilationCache(**kwargs)
    return _compilation_cache


def get_type_inference_cache(**kwargs) -> TypeInferenceCache:
    """Get or create global type inference cache."""
    global _type_inference_cache
    if _type_inference_cache is None:
        _type_inference_cache = TypeInferenceCache(**kwargs)
    return _type_inference_cache


def get_dependency_tracker() -> DependencyTracker:
    """Get or create global dependency tracker."""
    global _dependency_tracker
    if _dependency_tracker is None:
        _dependency_tracker = DependencyTracker()
    return _dependency_tracker


def clear_all_caches():
    """Clear all global caches."""
    if _compilation_cache:
        _compilation_cache.clear()
    if _type_inference_cache:
        _type_inference_cache.clear()
    if _dependency_tracker:
        _dependency_tracker.clear()
