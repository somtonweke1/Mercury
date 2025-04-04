import kubernetes
from kubernetes import client, config
from typing import Dict, Any
import time

class AutoScaler:
    def __init__(self, namespace: str = "default"):
        config.load_incluster_config()
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.namespace = namespace
        
        # Scaling thresholds
        self.cpu_threshold = 80  # 80% CPU usage
        self.memory_threshold = 85  # 85% memory usage
        self.request_threshold = 1000  # requests per minute

    async def monitor_and_scale(self):
        while True:
            try:
                metrics = self._get_current_metrics()
                self._adjust_scaling(metrics)
            except Exception as e:
                print(f"Scaling error: {str(e)}")
            await asyncio.sleep(60)  # Check every minute

    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        # Implementation would depend on your monitoring setup
        return {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'request_rate': 0.0
        }

    def _adjust_scaling(self, metrics: Dict[str, Any]):
        """Adjust scaling based on metrics"""
        deployments = self.apps_v1.list_namespaced_deployment(self.namespace)
        
        for deployment in deployments.items:
            if self._should_scale_up(metrics):
                self._scale_up(deployment)
            elif self._should_scale_down(metrics):
                self._scale_down(deployment)

    def _should_scale_up(self, metrics: Dict[str, Any]) -> bool:
        return (metrics['cpu_usage'] > self.cpu_threshold or
                metrics['memory_usage'] > self.memory_threshold or
                metrics['request_rate'] > self.request_threshold)

    def _should_scale_down(self, metrics: Dict[str, Any]) -> bool:
        return (metrics['cpu_usage'] < self.cpu_threshold/2 and
                metrics['memory_usage'] < self.memory_threshold/2 and
                metrics['request_rate'] < self.request_threshold/2)

    def _scale_up(self, deployment):
        """Scale up a deployment"""
        current_replicas = deployment.spec.replicas
        new_replicas = min(current_replicas * 2, 10)  # Max 10 replicas
        
        self.apps_v1.patch_namespaced_deployment_scale(
            name=deployment.metadata.name,
            namespace=self.namespace,
            body={'spec': {'replicas': new_replicas}}
        )

    def _scale_down(self, deployment):
        """Scale down a deployment"""
        current_replicas = deployment.spec.replicas
        new_replicas = max(current_replicas - 1, 1)  # Min 1 replica
        
        self.apps_v1.patch_namespaced_deployment_scale(
            name=deployment.metadata.name,
            namespace=self.namespace,
            body={'spec': {'replicas': new_replicas}}
        ) 