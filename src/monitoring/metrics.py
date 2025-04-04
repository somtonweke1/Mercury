from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    'request_count', 
    'Number of requests processed',
    ['endpoint']
)

PROCESSING_TIME = Histogram(
    'processing_time_seconds',
    'Time spent processing request',
    ['endpoint']
)

def track_request(endpoint: str):
    REQUEST_COUNT.labels(endpoint=endpoint).inc()

def track_processing_time(endpoint: str, time_taken: float):
    PROCESSING_TIME.labels(endpoint=endpoint).observe(time_taken) 