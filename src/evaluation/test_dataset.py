"""
Test dataset management for the Finance RAG system evaluation.
"""
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any


@dataclass
class TestQuery:
    """A test query with a ground truth answer."""
    query: str
    ground_truth: str
    category: str
    difficulty: str = "medium"
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "query": self.query,
            "ground_truth": self.ground_truth,
            "category": self.category,
            "difficulty": self.difficulty,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestQuery':
        """Create from dictionary."""
        return cls(
            query=data["query"],
            ground_truth=data["ground_truth"],
            category=data["category"],
            difficulty=data.get("difficulty", "medium"),
            tags=data.get("tags", []),
        )


class TestDataset:
    """A collection of test queries with ground truth answers."""

    def __init__(self, name: str):
        self.name = name
        self.queries: List[TestQuery] = []

    def add_query(self, query: TestQuery):
        """Add a query to the dataset."""
        self.queries.append(query)

    def save_to_file(self, file_path: str):
        """Save the dataset to a JSON file."""
        data = {
            "name": self.name,
            "queries": [q.to_dict() for q in self.queries],
        }
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(cls, file_path: str) -> 'TestDataset':
        """Load a dataset from a JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)

        dataset = cls(name=data.get("name", os.path.basename(file_path)))
        for query_data in data.get("queries", []):
            dataset.add_query(TestQuery.from_dict(query_data))

        return dataset

    def get_queries_by_category(self, category: str) -> List[TestQuery]:
        """Get all queries in a specific category."""
        return [q for q in self.queries if q.category == category]

    def get_queries_by_difficulty(self, difficulty: str) -> List[TestQuery]:
        """Get all queries with a specific difficulty."""
        return [q for q in self.queries if q.difficulty == difficulty]

    def get_queries_by_tag(self, tag: str) -> List[TestQuery]:
        """Get all queries with a specific tag."""
        return [q for q in self.queries if tag in q.tags]


def create_sample_dataset() -> TestDataset:
    """Create a sample dataset for testing."""
    dataset = TestDataset(name="finance_test_dataset")

    # Basic financial questions
    dataset.add_query(TestQuery(
        query="What was my total spending last month?",
        ground_truth="Your total spending last month was $2,145.67 across all categories.",
        category="spending_summary",
        difficulty="easy",
        tags=["spending", "summary"]
    ))

    dataset.add_query(TestQuery(
        query="How much did I spend on groceries in March?",
        ground_truth="You spent $342.19 on groceries in March.",
        category="category_spending",
        difficulty="easy",
        tags=["spending", "groceries", "monthly"]
    ))

    # Comparative questions
    dataset.add_query(TestQuery(
        query="Am I spending more on dining out this year compared to last year?",
        ground_truth="Yes, you've spent 15% more on dining out this year compared to the same period last year. Your current spending is $834.56 vs $725.70 last year.",
        category="spending_comparison",
        difficulty="medium",
        tags=["spending", "dining", "comparison", "yearly"]
    ))

    # Complex analytical questions
    dataset.add_query(TestQuery(
        query="What are my top 3 largest unusual expenses in the past 6 months?",
        ground_truth="Your top 3 unusual expenses in the past 6 months were: 1) $850 on car repairs in February, 2) $520 on emergency plumbing in April, and 3) $390 on medical expenses in January.",
        category="anomaly_detection",
        difficulty="hard",
        tags=["spending", "anomalies", "large expenses"]
    ))

    # Forecasting questions
    dataset.add_query(TestQuery(
        query="Based on my spending habits, will I exceed my monthly budget of $3000 this month?",
        ground_truth="Based on your current spending rate of $95/day this month and with 10 days remaining, you're projected to spend approximately $2,850, which is under your $3,000 budget but close to the limit.",
        category="forecasting",
        difficulty="hard",
        tags=["budget", "forecasting", "projection"]
    ))

    return dataset 