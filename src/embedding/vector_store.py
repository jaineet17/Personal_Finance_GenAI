import os
import sys
from pathlib import Path
import json
import time
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from dotenv import load_dotenv
import logging
import uuid
import re

# Add the project root to path so we can import from other modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.embedding.embedding_pipeline import EmbeddingPipeline

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("vector_store")

class VectorStore:
    """Enhanced ChromaDB vector store for financial data"""

    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 vector_db_path: Optional[str] = None):
        """Initialize the vector store with ChromaDB.

        Args:
            embedding_model_name (str): Name of the embedding model to use
            vector_db_path (str, optional): Path to store the vector database
        """
        # Set up embedding pipeline
        self.embedding_pipeline = EmbeddingPipeline(model_name=embedding_model_name)

        # Create embedding function for ChromaDB
        self.embedding_function = self._create_embedding_function()

        # Configure vector database path
        if vector_db_path:
            self.vector_db_path = vector_db_path
        else:
            # Use environment variable or default path
            self.vector_db_path = os.environ.get("VECTOR_DB_PATH", os.path.join(os.getcwd(), "data/vectors/chroma"))

        # Ensure the directory exists
        os.makedirs(self.vector_db_path, exist_ok=True)

        # Initialize client
        self.client = chromadb.PersistentClient(path=self.vector_db_path)

        # Collection names
        self.transaction_collection_name = "finance_transactions"
        self.category_collection_name = "finance_categories"
        self.time_period_collection_name = "finance_time_periods"
        self.common_queries_collection_name = "finance_common_queries"

        logger.info(f"Initialized vector store at {self.vector_db_path}")

    def _create_embedding_function(self):
        """Create a custom embedding function for ChromaDB."""
        class SentenceTransformerEmbeddingFunction(chromadb.EmbeddingFunction):
            def __init__(self, embedding_pipeline):
                self.embedding_pipeline = embedding_pipeline

            def __call__(self, texts):
                embeddings = self.embedding_pipeline.generate_embeddings(texts)
                return embeddings.tolist()

        return SentenceTransformerEmbeddingFunction(self.embedding_pipeline)

    def create_collection(self, collection_name: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create a collection in ChromaDB.

        Args:
            collection_name: Name of the collection
            metadata: Optional metadata for the collection

        Returns:
            Collection object
        """
        # Fix for ChromaDB requiring non-empty metadata
        if metadata is None:
            metadata = {"description": f"Collection for {collection_name}"}

        collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata=metadata
        )

        return collection  # Return the collection object

    def add_transactions(self, transactions: List[Dict]) -> Dict:
        """Add transactions to the vector store

        Args:
            transactions (List[Dict]): List of transaction dictionaries

        Returns:
            Dict: Result with success status and number of documents added
        """
        # Create collection if it doesn't exist
        collection = self.create_collection(self.transaction_collection_name)

        # Check for existing transactions to avoid duplicates
        if transactions:
            # Get existing IDs
            tx_ids = [tx.get("id") for tx in transactions if tx.get("id")]

            if tx_ids:
                # Get existing IDs from vector store
                existing_ids = self.get_existing_ids(self.transaction_collection_name, tx_ids)

                # Filter out transactions that already exist
                transactions = [tx for tx in transactions if tx.get("id") not in existing_ids]

        if not transactions:
            logger.info("All transactions already exist in vector store")
            return {"success": True, "added": 0}

        logger.info(f"Adding {len(transactions)} new transactions to vector store")

        # Prepare data for embedding
        docs = []
        metadatas = []
        ids = []

        for tx in transactions:
            # Extract or create transaction ID
            tx_id = tx.get("id")
            if not tx_id:
                tx_id = f"tx_{uuid.uuid4().hex[:8]}"
            ids.append(tx_id)

            # Handle different transaction formats:
            # 1. If 'text' is already present, use it directly
            # 2. If 'text' is missing, create it from transaction fields
            if "text" in tx:
                docs.append(tx["text"])

                # Use existing metadata if present, otherwise create from transaction fields
                if "metadata" in tx:
                    metadatas.append(tx["metadata"])
                else:
                    # Create metadata from transaction fields
                    metadata = {k: v for k, v in tx.items() if k not in ["text", "id"]}
                    metadatas.append(metadata)
            else:
                # Create text representation from transaction fields
                tx_text = (f"Transaction: {tx.get('description', '')} "
                          f"Amount: {tx.get('amount', 0)} "
                          f"Category: {tx.get('category', 'Uncategorized')} "
                          f"Date: {tx.get('date', '')}")
                docs.append(tx_text)

                # Create metadata from transaction fields
                metadata = {k: v for k, v in tx.items() if k != "id"}

                # Ensure numeric_date is present for filtering
                if "date" in metadata and isinstance(metadata["date"], str) and len(metadata["date"]) >= 10:
                    try:
                        metadata["numeric_date"] = int(metadata["date"].replace("-", ""))
                    except (ValueError, TypeError):
                        metadata["numeric_date"] = 0

                # Extract merchant from description if not present
                if "merchant" not in metadata and "description" in metadata:
                    metadata["merchant"] = self._extract_merchant_from_description(metadata["description"])

                # Ensure all values are valid for ChromaDB (str, int, float, bool)
                metadata = self._sanitize_metadata(metadata)
                metadatas.append(metadata)

        # Generate embeddings for text
        embeddings = self.embedding_pipeline.generate_transaction_embeddings(docs)

        try:
            # Get the collection
            collection = self.client.get_collection(self.transaction_collection_name)

            # Add data to collection
            collection.add(
                documents=docs,
                embeddings=embeddings["embeddings"].tolist(),
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Added {len(transactions)} transactions to vector store")
            return {"success": True, "added": len(transactions)}
        except Exception as e:
            logger.error(f"Error adding transactions to vector store: {e}")
            return {"success": False, "error": str(e)}

    def add_categories(self, categories: List[str], 
                      collection_name: str = None) -> bool:
        """Add categories to the vector store

        Args:
            categories (List[str]): List of category names
            collection_name (str, optional): Name of collection to use

        Returns:
            bool: True if successful, False otherwise
        """
        if not categories:
            logger.warning("No categories provided")
            return False

        try:
            # Use default collection name if not specified
            if collection_name is None:
                collection_name = self.category_collection_name

            # Create or get collection
            collection = self.create_collection(collection_name)

            # Generate embeddings
            embeddings = self.embedding_pipeline.generate_embeddings(categories)

            # Create IDs and metadata
            ids = [f"cat_{i}" for i in range(len(categories))]
            metadatas = [{"type": "category", "name": cat} for cat in categories]

            # Add to collection
            collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=categories,
                metadatas=metadatas
            )

            logger.info(f"Added {len(categories)} categories to vector store")
            return True

        except Exception as e:
            logger.error(f"Error adding categories to vector store: {str(e)}")
            return False

    def add_time_periods(self, collection_name: Optional[str] = None) -> bool:
        """Add time period embeddings to the vector store

        Args:
            collection_name (str, optional): Name of collection to use

        Returns:
            bool: Success status
        """
        collection_name = collection_name or self.time_period_collection_name
        collection = self.create_collection(collection_name)

        try:
            # Generate time period embeddings
            embed_result = self.embedding_pipeline.generate_time_period_embeddings()
            embeddings = embed_result['embeddings']
            time_periods = embed_result['time_periods']

            # Create time period IDs
            time_period_ids = [f"time_{i}" for i in range(len(time_periods))]

            # Create metadata
            metadata_list = [{'type': 'time_period'} for _ in time_periods]

            # Add to collection
            collection.add(
                ids=time_period_ids,
                documents=time_periods,
                embeddings=embeddings.tolist() if embeddings is not None else None,
                metadatas=metadata_list
            )

            logger.info(f"Added {len(time_periods)} time periods to collection {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Error adding time periods to vector store: {e}")
            return False

    def add_common_queries(self, queries: List[str], 
                          labels: Optional[List[str]] = None,
                          collection_name: Optional[str] = None) -> bool:
        """Add common financial query embeddings to the vector store

        Args:
            queries (List[str]): List of common queries
            labels (List[str], optional): Labels for the queries
            collection_name (str, optional): Name of collection to use

        Returns:
            bool: Success status
        """
        collection_name = collection_name or self.common_queries_collection_name
        collection = self.create_collection(collection_name)

        try:
            # Generate query embeddings
            embeddings = self.embedding_pipeline.generate_embeddings(queries)

            # Create query IDs
            query_ids = [f"query_{i}" for i in range(len(queries))]

            # Create metadata
            metadata_list = []
            for i, query in enumerate(queries):
                metadata = {
                    'type': 'query',
                    'label': labels[i] if labels and i < len(labels) else 'general'
                }
                metadata_list.append(metadata)

            # Add to collection
            collection.add(
                ids=query_ids,
                documents=queries,
                embeddings=embeddings.tolist() if embeddings is not None else None,
                metadatas=metadata_list
            )

            logger.info(f"Added {len(queries)} common queries to collection {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Error adding common queries to vector store: {e}")
            return False

    def query_similar(self, query_text: str, 
                     collection_name: str = "finance_transactions",
                     n_results: int = 5,
                     filter_dict: Optional[Dict] = None) -> Dict:
        """Query for similar items using text

        Args:
            query_text (str): The query text
            collection_name (str): Name of the collection
            n_results (int): Number of results to return
            filter_dict (Dict, optional): Filter for metadata

        Returns:
            Dict: Query results with IDs, documents, and metadata
        """
        try:
            collection = self.client.get_collection(name=collection_name)

            # Generate query embedding
            query_embedding = self.embedding_pipeline.generate_query_embeddings(query_text)

            # Process filter_dict if it exists
            if filter_dict is not None and len(filter_dict) > 0:
                # Check if the filter already has a top-level operator
                has_top_level_operator = any(k.startswith('$') for k in filter_dict.keys())

                # If there are multiple conditions but no top-level operator, wrap in $and
                if len(filter_dict) > 1 and not has_top_level_operator:
                    # Convert {key1: val1, key2: val2} to {"$and": [{"key1": val1}, {"key2": val2}]}
                    conditions = []
                    for key, value in filter_dict.items():
                        if key.startswith('$'):
                            # Keep existing operators as is
                            conditions.append({key: value})
                        else:
                            # Convert simple key-value pairs to condition objects
                            conditions.append({key: value})

                    # Create proper filter with $and
                    filter_dict = {"$and": conditions}

                # Special case: if we have a category and an $and operator at the same level,
                # we need to nest them all under a single $and
                if 'category' in filter_dict and '$and' in filter_dict:
                    category_value = filter_dict['category']
                    and_conditions = filter_dict['$and']

                    # Create a new condition for category
                    category_condition = {"category": category_value}

                    # Combine all under a single $and
                    filter_dict = {"$and": [category_condition] + and_conditions}

            # Query collection
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=filter_dict
            )

            return results

        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}

    def query_transactions(self, query_text: str, n_results: int = 30, filter: Optional[Dict] = None, include_metadata: bool = True) -> List[Dict]:
        """Query transactions based on query text and filter

        Args:
            query_text (str): The query text
            n_results (int): Number of results to return
            filter (Dict, optional): Filter conditions for metadata
            include_metadata (bool): Whether to include metadata in results

        Returns:
            List[Dict]: List of transaction dictionaries
        """
        results = self.query_similar(
            query_text=query_text,
            collection_name=self.transaction_collection_name,
            n_results=n_results,
            filter_dict=filter
        )

        # Check if we got valid results
        if not results or "ids" not in results or not results["ids"]:
            return []

        # Convert to list of transaction dictionaries
        transactions = []

        # Process each result
        if results["ids"] and len(results["ids"]) > 0:
            for i, doc_id in enumerate(results["ids"][0]):
                transaction = {}

                # Add ID
                transaction["id"] = doc_id

                # Add document text if available
                if "documents" in results and results["documents"] and len(results["documents"][0]) > i:
                    transaction["text"] = results["documents"][0][i]

                # Add metadata if requested
                if include_metadata and "metadatas" in results and results["metadatas"] and len(results["metadatas"][0]) > i:
                    metadata = results["metadatas"][0][i]
                    for key, value in metadata.items():
                        transaction[key] = value

                # Add distance/similarity score if available
                if "distances" in results and results["distances"] and len(results["distances"][0]) > i:
                    transaction["similarity"] = 1.0 - results["distances"][0][i]

                transactions.append(transaction)

        return transactions

    def query_by_metadata(self, collection_name: str = "finance_transactions",
                        filter_dict: Dict = None,
                        n_results: int = 100) -> Dict:
        """Query items by metadata only (no embedding similarity)

        Args:
            collection_name (str): Name of the collection
            filter_dict (Dict): Filter for metadata
            n_results (int): Maximum number of results to return

        Returns:
            Dict: Query results with IDs, documents, and metadata
        """
        try:
            collection = self.client.get_collection(name=collection_name)

            # Handle empty filter case - when no filter provided, get all items
            if filter_dict is None or len(filter_dict) == 0:
                # If no filter provided, use get_all method instead
                results = collection.get(limit=n_results)
            else:
                # Check if the filter already has a top-level operator
                has_top_level_operator = any(k.startswith('$') for k in filter_dict.keys())

                # If there are multiple conditions but no top-level operator, wrap in $and
                if len(filter_dict) > 1 and not has_top_level_operator:
                    # Convert {key1: val1, key2: val2} to {"$and": [{"key1": val1}, {"key2": val2}]}
                    conditions = []
                    for key, value in filter_dict.items():
                        if key.startswith('$'):
                            # Keep existing operators as is
                            conditions.append({key: value})
                        else:
                            # Convert simple key-value pairs to condition objects
                            conditions.append({key: value})

                    # Create proper filter with $and
                    filter_dict = {"$and": conditions}

                # Special case: if we have a category and an $and operator at the same level,
                # we need to nest them all under a single $and
                if 'category' in filter_dict and '$and' in filter_dict:
                    category_value = filter_dict['category']
                    and_conditions = filter_dict['$and']

                    # Create a new condition for category
                    category_condition = {"category": category_value}

                    # Combine all under a single $and
                    filter_dict = {"$and": [category_condition] + and_conditions}

                # Query collection by metadata only
                results = collection.get(
                    where=filter_dict,
                    limit=n_results
                )

            # Ensure results have the expected format
            if not results or "ids" not in results:
                return {"ids": [], "documents": [], "metadatas": []}

            return results

        except Exception as e:
            logger.error(f"Error querying vector store by metadata: {str(e)}")
            return {"ids": [], "documents": [], "metadatas": []}

    def hybrid_search(self, query_text: str,
                     collection_name: str = "finance_transactions",
                     n_results: int = 5,
                     filter_dict: Optional[Dict] = None,
                     alpha: float = 0.5) -> Dict:
        """Perform hybrid search (combination of vector and keyword search)

        Args:
            query_text (str): Query text
            collection_name (str): Name of collection to search
            n_results (int): Number of results to return
            filter_dict (Dict, optional): Dictionary for filtering results
            alpha (float): Weight for vector search (0-1), where 1 is vector only

        Returns:
            Dict: Search results
        """
        try:
            # Get the collection
            try:
                collection = self.client.get_collection(name=collection_name)
            except Exception as e:
                logger.error(f"Error accessing collection {collection_name}: {e}")
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

            # Generate query embedding
            query_embedding = self.embedding_pipeline.generate_embeddings([query_text])[0].tolist()

            # Define a series of fallback filter strategies
            filter_strategies = []

            # Start with the original filter if provided
            if filter_dict:
                filter_strategies.append(filter_dict)

                # Try to simplify the filter if it has complex nested operators
                if "$and" in filter_dict:
                    # Try each condition in the $and separately
                    for condition in filter_dict["$and"]:
                        if isinstance(condition, dict):
                            filter_strategies.append(condition)

                # If filter has a category, try with just that
                if "category" in filter_dict:
                    filter_strategies.append({"category": filter_dict["category"]})

                # If filter has a date range using numeric_date, try a simpler version
                if "$and" in filter_dict and any("numeric_date" in cond for cond in filter_dict["$and"]):
                    # Find the latest start date and earliest end date
                    start_date = None
                    end_date = None
                    for cond in filter_dict["$and"]:
                        if "numeric_date" in cond and "$gte" in cond["numeric_date"]:
                            start_date = cond["numeric_date"]["$gte"]
                        if "numeric_date" in cond and "$lte" in cond["numeric_date"]:
                            end_date = cond["numeric_date"]["$lte"]

                    if start_date is not None and end_date is not None:
                        # Add a simplified date filter
                        filter_strategies.append({
                            "numeric_date": {"$gte": start_date, "$lte": end_date}
                        })

            # Always add a null filter as final fallback
            filter_strategies.append(None)

            # Perform vector search with increased limit to ensure we get enough results
            vector_limit = max(n_results * 3, 15)  # Get more results initially, at least 15
            vector_results = None

            # Try each filter strategy until we get results
            for strategy in filter_strategies:
                try:
                    temp_results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=vector_limit,
                        where=strategy
                    )

                    # If we get results, use this strategy
                    if len(temp_results["ids"]) > 0 and len(temp_results["ids"][0]) > 0:
                        vector_results = temp_results
                        break
                except Exception as e:
                    logger.warning(f"Filter strategy {strategy} failed: {e}")
                    continue

            # If all filter strategies failed, try one last attempt without filter
            if vector_results is None or len(vector_results["ids"][0]) == 0:
                try:
                    logger.warning(f"No vector search results found for query: '{query_text}'")
                    vector_results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=vector_limit
                    )
                except Exception as e:
                    logger.error(f"Final vector search attempt failed: {e}")
                    return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

            # If still no results, try text search instead
            if len(vector_results["ids"]) == 0 or len(vector_results["ids"][0]) == 0:
                logger.info("No vector results, trying text search")
                try:
                    vector_results = collection.query(
                        query_texts=[query_text],
                        n_results=vector_limit
                    )
                except Exception as e:
                    logger.error(f"Text search fallback failed: {e}")
                    return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

            # If still no results, return empty response
            if len(vector_results["ids"]) == 0 or len(vector_results["ids"][0]) == 0:
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

            # Create a simple keyword search by querying with the text instead of embeddings
            keyword_results = None

            # Try each filter strategy for keyword search too
            for strategy in filter_strategies:
                try:
                    temp_results = collection.query(
                        query_texts=[query_text],
                        n_results=vector_limit,
                        where=strategy
                    )

                    # If we get results, use this strategy
                    if len(temp_results["ids"]) > 0 and len(temp_results["ids"][0]) > 0:
                        keyword_results = temp_results
                        break
                except Exception as e:
                    logger.warning(f"Keyword filter strategy {strategy} failed: {e}")
                    continue

            # If all keyword strategies failed, try without filter
            if keyword_results is None:
                try:
                    keyword_results = collection.query(
                        query_texts=[query_text],
                        n_results=vector_limit
                    )
                except Exception as e:
                    logger.warning(f"Keyword search without filter failed: {e}")
                    # Just use empty results
                    keyword_results = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

            # If no keyword results, just return vector results
            if len(keyword_results["ids"]) == 0 or len(keyword_results["ids"][0]) == 0:
                logger.info(f"No keyword results for '{query_text}', returning vector results only")
                return vector_results

            # Combine results with weighting
            all_ids = set(vector_results["ids"][0] + keyword_results["ids"][0])

            # Create lookup dictionaries for scores
            vector_scores = {}
            for i, doc_id in enumerate(vector_results["ids"][0]):
                # Normalize distances to scores (closer to 1 is better)
                if "distances" in vector_results and len(vector_results["distances"]) > 0:
                    # Convert distance to similarity score (1 - distance)
                    vector_scores[doc_id] = 1.0 - min(vector_results["distances"][0][i], 1.0)
                else:
                    # Fallback to position-based score if distances aren't available
                    vector_scores[doc_id] = 1.0 - (i / len(vector_results["ids"][0]))

            keyword_scores = {}
            for i, doc_id in enumerate(keyword_results["ids"][0]):
                if "distances" in keyword_results and len(keyword_results["distances"]) > 0:
                    keyword_scores[doc_id] = 1.0 - min(keyword_results["distances"][0][i], 1.0)
                else:
                    keyword_scores[doc_id] = 1.0 - (i / len(keyword_results["ids"][0]))

            # Calculate hybrid scores
            hybrid_scores = []
            hybrid_ids = []
            hybrid_documents = []
            hybrid_metadatas = []

            # Create lookup for documents and metadata
            docs_lookup = {}
            meta_lookup = {}
            for i, doc_id in enumerate(vector_results["ids"][0]):
                docs_lookup[doc_id] = vector_results["documents"][0][i]
                meta_lookup[doc_id] = vector_results["metadatas"][0][i]
            for i, doc_id in enumerate(keyword_results["ids"][0]):
                if doc_id not in docs_lookup:
                    docs_lookup[doc_id] = keyword_results["documents"][0][i]
                    meta_lookup[doc_id] = keyword_results["metadatas"][0][i]

            # Calculate combined scores
            for doc_id in all_ids:
                v_score = vector_scores.get(doc_id, 0.0)
                k_score = keyword_scores.get(doc_id, 0.0)

                # Hybrid score with alpha weighting
                score = (alpha * v_score) + ((1 - alpha) * k_score)

                hybrid_scores.append(score)
                hybrid_ids.append(doc_id)
                hybrid_documents.append(docs_lookup[doc_id])
                hybrid_metadatas.append(meta_lookup[doc_id])

            # Sort by hybrid score (descending)
            sorted_items = sorted(zip(hybrid_scores, hybrid_ids, hybrid_documents, hybrid_metadatas), 
                                 key=lambda x: x[0], reverse=True)

            # Take only top n_results
            sorted_items = sorted_items[:n_results]

            # Unzip the sorted results
            distances, ids, documents, metadatas = zip(*sorted_items) if sorted_items else ([], [], [], [])

            # Return in the same format as ChromaDB (lists within lists)
            result = {
                "ids": [list(ids)],
                "documents": [list(documents)],
                "metadatas": [list(metadatas)],
                "distances": [list(distances)]
            }

            return result

        except Exception as e:
            logger.error(f"Error performing hybrid search: {str(e)}")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    def date_range_search(self, query_text: str, 
                         start_date: str, 
                         end_date: str,
                         collection_name: str = "finance_transactions",
                         n_results: int = 5,
                         category_filter: Optional[str] = None) -> Dict:
        """Search transactions within a date range

        Args:
            query_text (str): The query text
            start_date (str): Start date in ISO format (YYYY-MM-DD)
            end_date (str): End date in ISO format (YYYY-MM-DD)
            collection_name (str): Name of the collection
            n_results (int): Number of results to return
            category_filter (str, optional): Category to filter by

        Returns:
            Dict: Query results with IDs, documents, and metadata
        """
        try:
            # Using numeric comparison instead of string comparison for dates
            # Convert ISO dates to numeric format (remove hyphens to get YYYYMMDD integer)
            numeric_start = int(start_date.replace('-', ''))
            numeric_end = int(end_date.replace('-', ''))

            # Add numeric date field to filter
            date_filter = {
                "$and": [
                    {"numeric_date": {"$gte": numeric_start}},
                    {"numeric_date": {"$lte": numeric_end}}
                ]
            }

            # If category filter is provided, add it
            filter_dict = date_filter
            if category_filter:
                # Use the properly formatted filter structure
                filter_dict = {
                    "$and": [
                        {"category": category_filter},
                        {"numeric_date": {"$gte": numeric_start}},
                        {"numeric_date": {"$lte": numeric_end}}
                    ]
                }

            # First try with numeric date filter
            results = self.query_similar(
                query_text=query_text,
                collection_name=collection_name,
                n_results=n_results,
                filter_dict=filter_dict
            )

            # If no results or few results, fall back to simpler query without strict date filtering
            if not results["ids"] or len(results["ids"][0]) < 2:
                logger.info(f"Few results with date filter ({len(results['ids'][0]) if results['ids'] else 0}), trying with month filter")

                # Extract year and month from start/end dates
                start_ym = start_date[:7]  # YYYY-MM
                end_ym = end_date[:7]      # YYYY-MM

                # Try with month filter instead
                month_filter = None
                if start_ym == end_ym:  # Same month
                    month_filter = {"month": start_ym}

                    # Add category if present
                    if category_filter:
                        month_filter = {
                            "$and": [
                                {"category": category_filter},
                                {"month": start_ym}
                            ]
                        }

                results = self.query_similar(
                    query_text=query_text,
                    collection_name=collection_name,
                    n_results=n_results,
                    filter_dict=month_filter
                )

            return results

        except Exception as e:
            logger.error(f"Error performing date range search: {str(e)}")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    def backup_collection(self, collection_name: str, backup_dir: Optional[str] = None) -> bool:
        """Backup a collection's data to a JSON file

        Args:
            collection_name (str): Name of the collection to backup
            backup_dir (str, optional): Directory to save backup to

        Returns:
            bool: Success status
        """
        backup_dir = backup_dir or os.path.join(self.vector_db_path, "backups")
        Path(backup_dir).mkdir(parents=True, exist_ok=True)

        try:
            # Get the collection
            collection = self.client.get_collection(name=collection_name)

            # Get all items (this might be memory-intensive for large collections)
            all_items = collection.get()

            # Create backup filename with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            backup_file = os.path.join(backup_dir, f"{collection_name}_{timestamp}.json")

            # Save to JSON file
            with open(backup_file, 'w') as f:
                json.dump(all_items, f)

            logger.info(f"Backed up collection {collection_name} to {backup_file}")
            return True

        except Exception as e:
            logger.error(f"Error backing up collection: {e}")
            return False

    def restore_collection(self, backup_file: str, 
                          collection_name: Optional[str] = None) -> bool:
        """Restore a collection from a backup file

        Args:
            backup_file (str): Path to the backup JSON file
            collection_name (str, optional): Name to restore the collection as

        Returns:
            bool: Success status
        """
        try:
            # Load backup data
            with open(backup_file, 'r') as f:
                backup_data = json.load(f)

            # Extract collection name from filename if not provided
            if not collection_name:
                backup_filename = os.path.basename(backup_file)
                collection_name = backup_filename.split('_')[0]

            # Create a new collection (or get existing)
            collection = self.create_collection(collection_name)

            # Add items to collection
            if backup_data.get("ids") and len(backup_data["ids"]) > 0:
                collection.add(
                    ids=backup_data["ids"],
                    documents=backup_data.get("documents", [None] * len(backup_data["ids"])),
                    embeddings=backup_data.get("embeddings"),
                    metadatas=backup_data.get("metadatas")
                )

            logger.info(f"Restored collection {collection_name} from {backup_file}")
            return True

        except Exception as e:
            logger.error(f"Error restoring collection: {e}")
            return False

    def reset_vector_store(self) -> Dict:
        """Reset the vector store by deleting all collections and recreating them.

        Returns:
            Dict: Result of the operation
        """
        try:
            # Get all collections - handle Chroma v0.6.0 API change
            all_collections = self.client.list_collections()

            # In Chroma v0.6.0, list_collections returns collection names directly
            if isinstance(all_collections, list) and (len(all_collections) == 0 or isinstance(all_collections[0], str)):
                collection_names = all_collections
            else:
                # For older versions, extract names from collection objects
                collection_names = [c.name for c in all_collections]

            logger.info(f"Found collections: {collection_names}")

            # Delete each collection
            for name in collection_names:
                try:
                    self.client.delete_collection(name=name)
                    logger.info(f"Deleted collection: {name}")
                except Exception as e:
                    logger.error(f"Error deleting collection {name}: {e}")

            # Recreate default collections
            self.create_collection(self.transaction_collection_name)
            self.create_collection(self.category_collection_name)
            self.create_collection(self.time_period_collection_name)
            self.create_collection(self.common_queries_collection_name)

            logger.info("Vector store reset completed")

            return {
                "success": True,
                "deleted_collections": collection_names,
                "created_collections": [
                    self.transaction_collection_name, 
                    self.category_collection_name,
                    self.time_period_collection_name,
                    self.common_queries_collection_name
                ]
            }
        except Exception as e:
            logger.error(f"Error resetting vector store: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def collection_count(self, collection_name: str) -> int:
        """Get the count of items in a collection

        Args:
            collection_name (str): Name of the collection

        Returns:
            int: Number of items in the collection
        """
        try:
            collection = self.client.get_collection(name=collection_name)
            return collection.count()
        except Exception as e:
            logger.error(f"Error getting collection count: {e}")
            return 0

    def get_existing_ids(self, collection_name: str, ids: List[str]) -> List[str]:
        """Check which IDs already exist in the collection

        Args:
            collection_name: Name of the collection
            ids: List of IDs to check

        Returns:
            List[str]: List of IDs that already exist
        """
        try:
            # Make sure collection exists
            self._ensure_collection_exists(collection_name)

            # Get the collection
            collection = self.client.get_collection(collection_name)

            # Check in batches to avoid query size limitations
            batch_size = 100
            existing_ids = []

            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i + batch_size]

                # Use 'where' filter to get IDs that exist
                # The $in operator checks if the ID is in the provided list
                try:
                    results = collection.get(
                        ids=batch_ids,
                        include=["documents", "metadatas"]
                    )

                    # Add existing IDs from this batch
                    existing_ids.extend(results["ids"])
                except Exception as e:
                    logger.error(f"Error checking batch of IDs: {e}")

            return existing_ids
        except Exception as e:
            logger.error(f"Error checking existing IDs: {e}")
            return []

    def _ensure_collection_exists(self, collection_name: str):
        """Make sure a collection exists, creating it if necessary

        Args:
            collection_name: Name of the collection
        """
        try:
            # Check if collection exists
            try:
                self.client.get_collection(collection_name)
                # If we get here, collection exists
                return
            except Exception:
                # Collection doesn't exist, create it
                self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Created collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise

    def _extract_merchant_from_description(self, description: str) -> str:
        """Extract merchant name from transaction description

        Args:
            description (str): Transaction description

        Returns:
            str: Extracted merchant name
        """
        if not description:
            return "Unknown"

        # Get first part before any hyphen, comma, or special character
        parts = re.split(r'[-,*#]', description, 1)
        merchant = parts[0].strip()

        # If the merchant part is too long, just take first few words
        if len(merchant) > 30:
            words = merchant.split()
            merchant = " ".join(words[:3])

        return merchant

    def _sanitize_metadata(self, metadata: Dict) -> Dict:
        """Sanitize metadata for ChromaDB requirements

        Args:
            metadata (Dict): Original metadata

        Returns:
            Dict: Sanitized metadata with valid types
        """
        sanitized = {}
        for key, value in metadata.items():
            if value is None:
                sanitized[key] = ""  # Replace None with empty string
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value  # Keep allowed types
            else:
                sanitized[key] = str(value)  # Convert other types to string

        return sanitized

    def _ensure_collections_exist(self):
        """Ensure all necessary collections exist, creating them if they don't"""
        try:
            # Check and create transaction collection
            self._ensure_collection_exists(self.transaction_collection_name)

            # Check and create category collection
            self._ensure_collection_exists(self.category_collection_name)

            # Check and create time period collection
            self._ensure_collection_exists(self.time_period_collection_name)

            # Check and create common queries collection
            self._ensure_collection_exists(self.common_queries_collection_name)

            logger.info("All vector store collections verified")
            return True
        except Exception as e:
            logger.error(f"Error ensuring collections exist: {e}")
            return False


if __name__ == "__main__":
    # Example usage
    vector_store = VectorStore()

    # Test with sample transactions
    transactions = [
        {
            "id": "tx_001",
            "description": "Coffee at Starbucks",
            "amount": -4.50,
            "category": "Food",
            "date": "2024-01-15",
            "merchant": "Starbucks"
        },
        {
            "id": "tx_002",
            "description": "Monthly rent payment",
            "amount": -1500.00,
            "category": "Housing",
            "date": "2024-01-01",
            "merchant": "Property Management"
        },
        {
            "id": "tx_003",
            "description": "Salary deposit",
            "amount": 3000.00,
            "category": "Income",
            "date": "2024-01-15",
            "merchant": "Employer"
        }
    ]

    # Add transactions
    vector_store.add_transactions(transactions)

    # Query similar
    results = vector_store.query_similar("coffee expenses", n_results=2)

    # Print results
    if results and results.get("documents"):
        print("Query Results:")
        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            print(f"{i+1}. {doc}")
            print(f"   Amount: {meta.get('amount')}, Category: {meta.get('category')}")
            print() 