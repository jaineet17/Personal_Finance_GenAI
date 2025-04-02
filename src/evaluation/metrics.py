"""
Metrics for evaluating the performance of the Finance RAG system.
"""
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Optional


@dataclass
class ResponseMetrics:
    """Metrics for a single query response."""
    query: str
    response: str
    ground_truth: Optional[str] = None
    response_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    llm_time_ms: float = 0.0
    tokens_used: int = 0
    num_docs_retrieved: int = 0
    relevance_score: Optional[float] = None
    accuracy_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for storage."""
        return {
            "query": self.query,
            "response": self.response,
            "ground_truth": self.ground_truth,
            "response_time_ms": self.response_time_ms,
            "retrieval_time_ms": self.retrieval_time_ms,
            "llm_time_ms": self.llm_time_ms,
            "tokens_used": self.tokens_used,
            "num_docs_retrieved": self.num_docs_retrieved,
            "relevance_score": self.relevance_score,
            "accuracy_score": self.accuracy_score,
        }


class PerformanceTimer:
    """Simple timer for measuring performance."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """Stop the timer and return elapsed time in milliseconds."""
        self.end_time = time.time()
        return self.elapsed_ms()
    
    def elapsed_ms(self):
        """Return elapsed time in milliseconds."""
        if self.start_time is None:
            return 0
        end = self.end_time if self.end_time is not None else time.time()
        return (end - self.start_time) * 1000


class EvaluationResult:
    """Container for evaluation results across multiple queries."""
    
    def __init__(self):
        self.metrics: List[ResponseMetrics] = []
        self.system_memory_usage: Dict[str, float] = {}
        self.timestamp = time.time()
        self.version = "1.0"
    
    def add_response_metrics(self, metrics: ResponseMetrics):
        """Add metrics for a single response."""
        self.metrics.append(metrics)
    
    def set_memory_usage(self, memory_usage: Dict[str, float]):
        """Set memory usage information."""
        self.system_memory_usage = memory_usage
    
    def get_average_response_time(self) -> float:
        """Calculate average response time."""
        if not self.metrics:
            return 0
        return sum(m.response_time_ms for m in self.metrics) / len(self.metrics)
    
    def get_average_accuracy(self) -> float:
        """Calculate average accuracy score."""
        scores = [m.accuracy_score for m in self.metrics if m.accuracy_score is not None]
        if not scores:
            return 0
        return sum(scores) / len(scores)
    
    def get_average_relevance(self) -> float:
        """Calculate average relevance score."""
        scores = [m.relevance_score for m in self.metrics if m.relevance_score is not None]
        if not scores:
            return 0
        return sum(scores) / len(scores)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for storage."""
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "metrics": [m.to_dict() for m in self.metrics],
            "system_memory_usage": self.system_memory_usage,
            "summary": {
                "avg_response_time_ms": self.get_average_response_time(),
                "avg_accuracy": self.get_average_accuracy(),
                "avg_relevance": self.get_average_relevance(),
                "total_queries": len(self.metrics),
            }
        } 