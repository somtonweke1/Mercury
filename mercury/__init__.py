"""
Mercury: High-performance financial data pipeline for sparse orderbook data
"""

from .loader import MercuryDataLoader
from .processor import DataProcessor
from .cache import CacheManager
from .config import Config

__version__ = "1.0.0" 