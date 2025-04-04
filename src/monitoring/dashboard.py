from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram
import time

class MonitoringDashboard:
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # System metrics
        self.cpu_usage = Gauge('cpu_usage_percent', 'CPU Usage', registry=self.registry)
        self.memory_usage = Gauge('memory_usage_bytes', 'Memory Usage', registry=self.registry)
        
        # Business metrics
        self.imputation_accuracy = Gauge('imputation_accuracy', 'Accuracy of data imputation', 
                                       ['trading_pair'], registry=self.registry)
        self.processing_latency = Histogram('processing_latency_seconds', 
                                          'Time taken to process requests',
                                          ['trading_pair'], registry=self.registry)
        
        # Error tracking
        self.error_count = Counter('error_count', 'Number of errors', 
                                 ['error_type'], registry=self.registry)

    def record_metrics(self, metrics: dict):
        """Record various system and business metrics"""
        if 'cpu' in metrics:
            self.cpu_usage.set(metrics['cpu'])
        if 'memory' in metrics:
            self.memory_usage.set(metrics['memory'])
        if 'accuracy' in metrics:
            self.imputation_accuracy.labels(
                trading_pair=metrics['trading_pair']
            ).set(metrics['accuracy'])

    def track_processing_time(self, trading_pair: str, start_time: float):
        """Track processing latency"""
        duration = time.time() - start_time
        self.processing_latency.labels(trading_pair=trading_pair).observe(duration)

    def record_error(self, error_type: str):
        """Record error occurrences"""
        self.error_count.labels(error_type=error_type).inc() 