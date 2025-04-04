from google.cloud import bigquery
from google.api_core import retry
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union
from datetime import datetime, timedelta
import logging
from functools import lru_cache
import asyncio
import concurrent.futures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MercuryDataLoader:
    """Enhanced data loader with optimized performance and robust error handling."""
    
    def __init__(
        self, 
        bq_dataset: str = "crypto_orderbooks",
        cache_ttl: int = 300,  # Cache TTL in seconds
        max_retries: int = 3,
        batch_size: int = 10000,
        max_workers: int = 4
    ):
        self.client = bigquery.Client()
        self.dataset_ref = self.client.dataset(bq_dataset)
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.max_workers = max_workers
        self._setup_monitoring()

    def _setup_monitoring(self) -> None:
        """Initialize performance monitoring metrics."""
        self.metrics = {
            'query_times': [],
            'rows_processed': 0,
            'cache_hits': 0,
            'errors': 0
        }

    @retry.Retry(predicate=retry.if_exception_type(Exception))
    async def _execute_query(self, query: str) -> pd.DataFrame:
        """Execute BigQuery with retry logic and async support."""
        try:
            start_time = datetime.now()
            query_job = self.client.query(query)
            df = query_job.to_dataframe()
            
            query_time = (datetime.now() - start_time).total_seconds()
            self.metrics['query_times'].append(query_time)
            self.metrics['rows_processed'] += len(df)
            
            return df
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Query execution failed: {str(e)}")
            raise

    @lru_cache(maxsize=128)
    def _get_cached_data(self, query_hash: str) -> Optional[pd.DataFrame]:
        """Retrieve cached query results."""
        return None  # Implement your caching logic here

    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply initial data cleaning and preprocessing."""
        df = df.copy()
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # Add derived features
        df['bid_ask_spread'] = df['ask'] - df['bid']
        df['mid_price'] = (df['ask'] + df['bid']) / 2
        
        # Handle missing values with forward fill for small gaps
        df[['bid', 'ask']] = df[['bid', 'ask']].fillna(method='ffill', limit=2)
        
        # Add quality indicators
        df['data_quality'] = np.where(df[['bid', 'ask']].isna().any(axis=1), 0, 1)
        
        return df

    async def load_sparse_data(
        self,
        instrument: str = "BTC/USD",
        window: str = "1d",
        batch_mode: bool = False,
        filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Enhanced sparse data loader with batching and filtering capabilities.
        
        Args:
            instrument (str): Trading pair (e.g. "BTC/USD")
            window (str): Time window (e.g. "1d", "4h")
            batch_mode (bool): Whether to process in batches
            filters (Dict): Additional query filters
            
        Returns:
            pd.DataFrame: Processed DataFrame with enhanced features
        """
        base_query = f"""
        WITH OrderbookData AS (
            SELECT 
                timestamp,
                bid,
                ask,
                missing_count,
                LAG(bid) OVER (ORDER BY timestamp) as prev_bid,
                LAG(ask) OVER (ORDER BY timestamp) as prev_ask
            FROM `orderbook_{instrument}`
            WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {window})
            {f"AND {' AND '.join(filters)}" if filters else ""}
        )
        SELECT 
            *,
            CASE 
                WHEN bid IS NULL AND prev_bid IS NOT NULL THEN 1
                ELSE 0
            END as gap_start
        FROM OrderbookData
        """

        try:
            if batch_mode:
                return await self._batch_process(base_query)
            else:
                df = await self._execute_query(base_query)
                return self._preprocess_dataframe(df)
                
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise

    async def _batch_process(self, query: str) -> pd.DataFrame:
        """Process large datasets in batches."""
        frames = []
        
        async with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = []
            offset = 0
            
            while True:
                batch_query = f"{query} LIMIT {self.batch_size} OFFSET {offset}"
                tasks.append(
                    asyncio.create_task(self._execute_query(batch_query))
                )
                
                if len(tasks[-1].result()) < self.batch_size:
                    break
                    
                offset += self.batch_size
            
            results = await asyncio.gather(*tasks)
            frames.extend(results)
        
        return pd.concat(frames, ignore_index=True)

    def get_performance_metrics(self) -> Dict:
        """Return performance monitoring metrics."""
        return {
            'avg_query_time': np.mean(self.metrics['query_times']),
            'total_rows': self.metrics['rows_processed'],
            'cache_hit_rate': self.metrics['cache_hits'] / max(1, self.metrics['rows_processed']),
            'error_rate': self.metrics['errors'] / max(1, len(self.metrics['query_times']))
        }

    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Validate loaded data quality."""
        return {
            'missing_rate': df.isna().mean().to_dict(),
            'unique_counts': df.nunique().to_dict(),
            'value_ranges': {
                col: {'min': df[col].min(), 'max': df[col].max()}
                for col in df.select_dtypes(include=[np.number]).columns
            }
        } 