from google.cloud import bigquery
from google.api_core import retry
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import logging
from .cache import CacheManager
from .config import Config
from .exceptions import DataLoadError

logger = logging.getLogger(__name__)

class MercuryDataLoader:
    """Production-ready financial data loader with smart caching and error handling."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the data loader with optional configuration.
        
        Args:
            config: Configuration object with database and processing settings
        """
        self.config = config or Config()
        self.cache = CacheManager(ttl=self.config.cache_ttl)
        self._setup_clients()
        
    def _setup_clients(self):
        """Initialize database clients with connection pooling."""
        try:
            self.client = bigquery.Client()
            logger.info("Successfully connected to BigQuery")
        except Exception as e:
            logger.error(f"Failed to initialize BigQuery client: {e}")
            raise DataLoadError("Database connection failed")

    @retry.Retry(predicate=retry.if_exception_type(Exception))
    def load_market_data(
        self,
        symbol: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        interval: str = "1m",
        fields: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load market data with smart caching and validation.
        
        Args:
            symbol: Trading pair (e.g. "BTC/USD")
            start_time: Start time in ISO format
            end_time: End time in ISO format
            interval: Data interval (1m, 5m, 1h, etc)
            fields: List of fields to fetch
            
        Returns:
            DataFrame with market data
        """
        cache_key = f"{symbol}:{start_time}:{end_time}:{interval}"
        
        # Try cache first
        if cached_data := self.cache.get(cache_key):
            logger.info(f"Cache hit for {cache_key}")
            return cached_data
            
        # Build and execute query
        query = self._build_query(symbol, start_time, end_time, interval, fields)
        try:
            df = self.client.query(query).to_dataframe()
            df = self._validate_and_clean(df)
            
            # Cache successful result
            self.cache.set(cache_key, df)
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data for {symbol}: {e}")
            raise DataLoadError(f"Failed to load {symbol} data")

    def _build_query(self, symbol: str, start_time: Optional[str], 
                    end_time: Optional[str], interval: str,
                    fields: Optional[List[str]]) -> str:
        """Build optimized BigQuery SQL."""
        field_list = ", ".join(fields) if fields else "*"
        table = f"`{self.config.dataset}.{symbol.lower().replace('/', '_')}`"
        
        query = f"""
        WITH raw_data AS (
            SELECT 
                {field_list},
                LAG(price) OVER (ORDER BY timestamp) as prev_price,
                LEAD(price) OVER (ORDER BY timestamp) as next_price
            FROM {table}
            WHERE 1=1
            {f"AND timestamp >= '{start_time}'" if start_time else ""}
            {f"AND timestamp <= '{end_time}'" if end_time else ""}
        )
        SELECT 
            *,
            CASE 
                WHEN price IS NULL 
                AND prev_price IS NOT NULL 
                AND next_price IS NOT NULL
                THEN (prev_price + next_price) / 2
                ELSE price 
            END as adjusted_price
        FROM raw_data
        """
        return query

    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean loaded data."""
        if df.empty:
            raise DataLoadError("Query returned no data")
            
        # Convert types
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # Add quality metrics
        df['data_quality'] = np.where(df.isna().any(axis=1), 0, 1)
        
        # Basic anomaly detection
        for col in df.select_dtypes(include=[np.number]):
            mean, std = df[col].mean(), df[col].std()
            df[f'{col}_anomaly'] = np.where(
                np.abs(df[col] - mean) > 3 * std,
                1, 0
            )
            
        return df

    def get_symbols(self) -> List[str]:
        """Get list of available trading pairs."""
        query = f"""
        SELECT DISTINCT table_id 
        FROM `{self.config.project}.{self.config.dataset}.__TABLES__`
        """
        try:
            df = self.client.query(query).to_dataframe()
            return [t.replace('_', '/').upper() for t in df['table_id']]
        except Exception as e:
            logger.error(f"Failed to fetch symbols: {e}")
            return [] 