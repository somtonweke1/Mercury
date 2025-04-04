from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Configuration settings for Mercury data pipeline."""
    
    # Database settings
    project: str = "my-project"
    dataset: str = "market_data"
    
    # Cache settings
    cache_ttl: int = 300  # seconds
    cache_size: int = 1000  # items
    
    # Processing settings
    batch_size: int = 10000
    max_workers: int = 4
    max_retries: int = 3
    
    # Feature settings
    enable_anomaly_detection: bool = True
    enable_gap_filling: bool = True 