import os
import sys
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from datetime import datetime, timedelta, date
import math
import re

# Add the project root to path so we can import from other modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.embedding.vector_store import VectorStore
from src.embedding.embedding_pipeline import EmbeddingPipeline
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("retrieval_system")

class FinanceRetrieval:
    """Retrieval system for the finance LLM application"""

    def __init__(self, 
                embedding_model_name: str = "all-MiniLM-L6-v2",
                vector_store: Optional[VectorStore] = None):
        """Initialize the retrieval system

        Args:
            embedding_model_name (str): Name of the embedding model to use
            vector_store (VectorStore, optional): Existing vector store instance
        """
        # Initialize vector store
        self.vector_store = vector_store or VectorStore(embedding_model_name=embedding_model_name)

        # Initialize embedding pipeline directly for specialized operations
        self.embedding_pipeline = EmbeddingPipeline(model_name=embedding_model_name)

        logger.info(f"Initialized finance retrieval system with model {embedding_model_name}")

    def retrieve_similar_transactions(
        self, 
        query: str, 
        n_results: int = 30, 
        time_period: Optional[str] = None, 
        category: Optional[str] = None,
        include_metadata: bool = True
    ) -> List[Dict]:
        """Retrieve similar transactions for given query using vector search

        Args:
            query (str): User query
            n_results (int): Number of results to return
            time_period (Optional[str]): Time period to filter by (e.g., "last month")
            category (Optional[str]): Category to filter by
            include_metadata (bool): Whether to include metadata in results

        Returns:
            List[Dict]: List of similar transactions
        """
        logger.info(f"Retrieving similar transactions for query: '{query}'")

        # Extract date range if time period is provided
        start_date, end_date = None, None
        if time_period:
            start_date, end_date = self._parse_time_period(time_period)
            logger.info(f"Extracted date range from query: ({start_date}, {end_date})")
            logger.info(f"Filtering transactions by date range: {start_date} to {end_date}")

            # Try to filter by date range using ChromaDB's native filtering if dates are available
            try:
                # Convert dates to numeric format for filtering
                if start_date and end_date:
                    start_date_numeric = int(start_date.strftime("%Y%m%d")) if start_date else None
                    end_date_numeric = int(end_date.strftime("%Y%m%d")) if end_date else None

                    # Set up metadata filters for date range
                    filter_conditions = []
                    if start_date_numeric:
                        filter_conditions.append({"numeric_date": {"$gte": start_date_numeric}})
                    if end_date_numeric:
                        filter_conditions.append({"numeric_date": {"$lte": end_date_numeric}})

                    # Combine filters
                    date_filter = {"$and": filter_conditions} if filter_conditions else {}

                    # Add category filter if provided
                    if category:
                        if date_filter:
                            # Add category filter to existing date filter
                            category_filter = {"category": {"$eq": category}}
                            date_filter["$and"].append(category_filter)
                        else:
                            # Use only category filter
                            date_filter = {"category": {"$eq": category}}

                    # Query with filters
                    results = self.vector_store.query_transactions(
                        query_text=query,
                        n_results=n_results,
                        filter=date_filter if date_filter else None,
                        include_metadata=include_metadata
                    )

                    # Return results directly if we successfully filtered
                    return results
            except Exception as e:
                logger.warning(f"Error converting dates to numeric format: {e}. Will filter in post-processing.")

        # If we couldn't filter with ChromaDB, do a regular query and filter results manually
        results = self.vector_store.query_transactions(
            query_text=query,
            n_results=n_results,
            filter={"category": {"$eq": category}} if category else None,
            include_metadata=include_metadata
        )

        # Apply date filtering in post-processing
        if start_date or end_date:
            filtered_results = []
            for txn in results:
                # Skip if transaction has no date
                if 'date' not in txn:
                    continue

                # Parse transaction date
                try:
                    # Handle different date formats
                    txn_date = txn['date']
                    if isinstance(txn_date, str):
                        # Try parsing with different formats
                        for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d', '%d/%m/%Y']:
                            try:
                                txn_date = datetime.strptime(txn_date, fmt).date()
                                break
                            except ValueError:
                                continue
                        else:
                            # If all formats fail, skip this transaction
                            logger.warning(f"Could not parse date format in transaction: {txn}")
                            continue
                    elif isinstance(txn_date, datetime):
                        txn_date = txn_date.date()

                    # Check if date is within range
                    if start_date and txn_date < start_date:
                        continue
                    if end_date and txn_date > end_date:
                        continue

                    filtered_results.append(txn)
                except Exception as e:
                    logger.warning(f"Error filtering transaction by date: {e}")
                    continue

            return filtered_results

        return results

    def retrieve_by_category(self, 
                           category: str, 
                           n_results: int = 10,
                           date_range: Optional[Tuple[str, str]] = None) -> Dict:
        """Retrieve transactions by category

        Args:
            category (str): The category to search for
            n_results (int): Number of results to return
            date_range (Tuple[str, str], optional): Filter by date range (start_date, end_date)

        Returns:
            Dict: Retrieved transactions
        """
        # Build filter dict
        filter_dict = {"category": category}

        if date_range:
            start_date, end_date = date_range
            # Convert to numeric format for filtering
            try:
                # Convert YYYY-MM-DD to YYYYMMDD integer format
                numeric_start = int(start_date.replace('-', ''))
                numeric_end = int(end_date.replace('-', ''))

                # Add numeric date filters
                filter_dict["$and"] = filter_dict.get("$and", [])
                filter_dict["$and"].extend([
                    {"numeric_date": {"$gte": numeric_start}},
                    {"numeric_date": {"$lte": numeric_end}}
                ])
            except (ValueError, AttributeError) as e:
                # Fallback to month-based filtering
                logger.warning(f"Using month-based filtering instead: {e}")
                try:
                    start_month = start_date[:7]  # YYYY-MM format
                    end_month = end_date[:7]      # YYYY-MM format

                    if start_month == end_month:
                        filter_dict["month"] = start_month
                except Exception as e2:
                    logger.warning(f"Failed to extract month from dates: {e2}")

        # Retrieve by metadata
        results = self.vector_store.query_by_metadata(
            collection_name=self.vector_store.transaction_collection_name,
            filter_dict=filter_dict,
            n_results=n_results
        )

        # Process and format results
        formatted_results = {
            "category": category,
            "count": len(results.get("ids", [])),
            "transactions": []
        }

        for i, (doc_id, doc, meta) in enumerate(zip(
            results.get("ids", []), 
            results.get("documents", []), 
            results.get("metadatas", [])
        )):
            try:
                formatted_results["transactions"].append({
                    "id": doc_id,
                    "description": doc,
                    "amount": float(meta.get("amount", 0)),
                    "date": meta.get("date", ""),
                    "merchant": meta.get("merchant", ""),
                    "account": meta.get("account", "")
                })
            except (ValueError, TypeError):
                # Skip if there are issues with the data format
                continue

        return formatted_results

    def retrieve_by_time_period(self, 
                              query: str, 
                              time_period: str,
                              n_results: int = 5) -> Dict:
        """Retrieve transactions for a specific time period and query

        Args:
            query (str): The search query
            time_period (str): Time period description (e.g., "last month", "Q1 2023")
            n_results (int): Number of results to return

        Returns:
            Dict: Retrieved transactions
        """
        # Convert time period to date range
        date_range = self._parse_time_period(time_period)

        if not date_range:
            return {
                "error": f"Could not parse time period: {time_period}",
                "transactions": []
            }

        start_date, end_date = date_range

        # Retrieve similar transactions within date range
        results = self.vector_store.date_range_search(
            query_text=query,
            start_date=start_date,
            end_date=end_date,
            collection_name=self.vector_store.transaction_collection_name,
            n_results=n_results
        )

        # Process and format results
        formatted_results = self._format_transaction_results(results)
        formatted_results["time_period"] = time_period
        formatted_results["date_range"] = {"start": start_date, "end": end_date}

        return formatted_results

    def retrieve_spending_insights(self, 
                                category: Optional[str] = None,
                                date_range: Optional[Tuple[str, str]] = None) -> Dict:
        """Retrieve spending insights for analytics

        Args:
            category (str, optional): Filter by category
            date_range (Tuple[str, str], optional): Filter by date range (start_date, end_date)

        Returns:
            Dict: Spending insights with statistics
        """
        # Build filter dict
        filter_dict = {}

        if category:
            filter_dict["category"] = category

        if date_range:
            start_date, end_date = date_range
            # Convert to numeric format for filtering
            try:
                # Convert YYYY-MM-DD to YYYYMMDD integer format
                numeric_start = int(start_date.replace('-', ''))
                numeric_end = int(end_date.replace('-', ''))

                # Add numeric date filters
                filter_dict["$and"] = filter_dict.get("$and", [])
                filter_dict["$and"].extend([
                    {"numeric_date": {"$gte": numeric_start}},
                    {"numeric_date": {"$lte": numeric_end}}
                ])
            except (ValueError, AttributeError) as e:
                # Fallback to month-based filtering
                logger.warning(f"Using month-based filtering instead: {e}")
                try:
                    start_month = start_date[:7]  # YYYY-MM format
                    end_month = end_date[:7]      # YYYY-MM format

                    if start_month == end_month:
                        filter_dict["month"] = start_month
                except Exception as e2:
                    logger.warning(f"Failed to extract month from dates: {e2}")

        # Retrieve transactions
        results = self.vector_store.query_by_metadata(
            collection_name=self.vector_store.transaction_collection_name,
            filter_dict=filter_dict,
            n_results=1000  # Get a large sample for statistics
        )

        # Process transactions for insights
        transactions = []
        total_spending = 0
        total_income = 0
        categories = {}
        merchants = {}

        for i, (doc_id, doc, meta) in enumerate(zip(
            results.get("ids", []), 
            results.get("documents", []), 
            results.get("metadatas", [])
        )):
            try:
                amount = float(meta.get("amount", 0))
                category = meta.get("category", "Uncategorized")
                merchant = meta.get("merchant", "Unknown")

                tx = {
                    "id": doc_id,
                    "description": doc,
                    "amount": amount,
                    "date": meta.get("date", ""),
                    "category": category,
                    "merchant": merchant
                }

                transactions.append(tx)

                # Track spending vs income
                if amount < 0:
                    total_spending += abs(amount)

                    # Update category stats
                    if category not in categories:
                        categories[category] = 0
                    categories[category] += abs(amount)

                    # Update merchant stats
                    if merchant not in merchants:
                        merchants[merchant] = 0
                    merchants[merchant] += abs(amount)
                else:
                    total_income += amount

            except (ValueError, TypeError):
                # Skip if there are issues with the data format
                continue

        # Sort by spending amount
        categories_sorted = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        merchants_sorted = sorted(merchants.items(), key=lambda x: x[1], reverse=True)

        # Create insights
        insights = {
            "total_transactions": len(transactions),
            "total_spending": total_spending,
            "total_income": total_income,
            "net_cash_flow": total_income - total_spending,
            "top_categories": [
                {"category": cat, "amount": amount} 
                for cat, amount in categories_sorted[:5]
            ],
            "top_merchants": [
                {"merchant": merch, "amount": amount} 
                for merch, amount in merchants_sorted[:5]
            ],
            "date_range": {
                "start": date_range[0] if date_range else "all",
                "end": date_range[1] if date_range else "all"
            }
        }

        return insights

    def find_similar_categories(self, query_category: str, n_results: int = 3) -> Dict:
        """Find categories similar to the query category

        Args:
            query_category (str): The category to find similar ones to
            n_results (int): Number of similar categories to return

        Returns:
            Dict: Similar categories with similarity scores
        """
        # First check if we have the categories collection
        try:
            collection = self.vector_store.client.get_collection(
                self.vector_store.category_collection_name
            )
        except Exception:
            return {
                "error": "Categories collection does not exist",
                "similar_categories": []
            }

        # Generate embedding for the query category
        query_embedding = self.embedding_pipeline.generate_category_embeddings([query_category])

        # Query for similar categories
        results = collection.query(
            query_embeddings=[query_embedding["embeddings"].tolist()],
            n_results=n_results + 1  # +1 because the query might be in the collection
        )

        # Process results
        similar_cats = []
        for i, (cat_id, cat, distance) in enumerate(zip(
            results.get("ids", [])[0],
            results.get("documents", [])[0],
            results.get("distances", [])[0]
        )):
            # Skip if it's the same category
            if cat.lower() == query_category.lower():
                continue

            similar_cats.append({
                "category": cat,
                "similarity": 1 - distance  # Convert distance to similarity score
            })

        return {
            "query_category": query_category,
            "similar_categories": similar_cats[:n_results]  # Limit to n_results
        }

    def retrieve_for_query_enrichment(self, query: str, n_results: int = 3) -> Dict:
        """Retrieve contextual data to enrich a user query

        This is used for preparing context before sending to an LLM

        Args:
            query (str): The user's query
            n_results (int): Number of context items to retrieve per collection

        Returns:
            Dict: Contextual data from different collections
        """
        # Get query embedding for similarity comparison
        query_embedding = self.embedding_pipeline.generate_query_embeddings(query)

        context = {
            "transactions": [],
            "categories": [],
            "common_queries": []
        }

        # 1. Get relevant transactions
        try:
            tx_results = self.vector_store.query_similar(
                query_text=query,
                collection_name=self.vector_store.transaction_collection_name,
                n_results=n_results
            )

            for i, (doc_id, doc, meta) in enumerate(zip(
                tx_results.get("ids", [])[0],
                tx_results.get("documents", [])[0],
                tx_results.get("metadatas", [])[0]
            )):
                context["transactions"].append({
                    "id": doc_id,
                    "description": doc,
                    "amount": meta.get("amount", "0"),
                    "date": meta.get("date", ""),
                    "category": meta.get("category", ""),
                    "merchant": meta.get("merchant", "")
                })
        except Exception as e:
            logger.warning(f"Error retrieving transactions for query enrichment: {e}")

        # 2. Get relevant categories
        try:
            cat_results = self.vector_store.query_similar(
                query_text=query,
                collection_name=self.vector_store.category_collection_name,
                n_results=n_results
            )

            for i, (cat_id, cat) in enumerate(zip(
                cat_results.get("ids", [])[0],
                cat_results.get("documents", [])[0]
            )):
                context["categories"].append(cat)
        except Exception as e:
            logger.warning(f"Error retrieving categories for query enrichment: {e}")

        # 3. Get relevant common queries (if available)
        try:
            query_results = self.vector_store.query_similar(
                query_text=query,
                collection_name=self.vector_store.common_queries_collection_name,
                n_results=n_results
            )

            for i, (q_id, q, meta) in enumerate(zip(
                query_results.get("ids", [])[0],
                query_results.get("documents", [])[0],
                query_results.get("metadatas", [])[0]
            )):
                context["common_queries"].append({
                    "query": q,
                    "label": meta.get("label", "general")
                })
        except Exception as e:
            logger.warning(f"Error retrieving common queries for query enrichment: {e}")

        return context

    def _format_transaction_results(self, results: Dict) -> Dict:
        """Format retrieved transaction results

        Args:
            results: Raw results from vector search

        Returns:
            Dict: Formatted transactions
        """
        formatted_results = {"transactions": []}

        # Handle empty results
        if not results or not results.get("ids") or not results["ids"][0]:
            return formatted_results

        # Extract results
        ids = results["ids"][0]
        metadatas = results["metadatas"][0] if "metadatas" in results else []
        distances = results["distances"][0] if "distances" in results else []

        # Format each transaction
        for i, tx_id in enumerate(ids):
            if i < len(metadatas):
                metadata = metadatas[i]

                # Handle potential NaN values in metadata
                sanitized_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, float) and math.isnan(value):
                        sanitized_metadata[key] = None
                    else:
                        sanitized_metadata[key] = value

                # Create a formatted transaction
                tx = {
                    "id": tx_id,
                    "date": sanitized_metadata.get("date", ""),
                    "description": sanitized_metadata.get("description", ""),
                    "amount": sanitized_metadata.get("amount", 0),
                    "category": sanitized_metadata.get("category", "Unknown"),
                    "account": sanitized_metadata.get("account", "")
                }

                # Add relevance score if available
                if i < len(distances):
                    # Convert distance to score (lower distance = higher score)
                    distance = distances[i]
                    if isinstance(distance, float) and not math.isnan(distance):
                        # Convert to score (1.0 = perfect match, 0.0 = worst match)
                        score = max(0.0, min(1.0, 1.0 - distance))
                        tx["relevance_score"] = round(score, 3)

                formatted_results["transactions"].append(tx)

        # Sort by date (newest first)
        formatted_results["transactions"].sort(
            key=lambda tx: tx.get("date", ""), 
            reverse=True
        )

        return formatted_results

    def _parse_time_period(self, time_hint: str) -> Tuple[Optional[date], Optional[date]]:
        """Parse time period hint into start and end dates

        Args:
            time_hint (str): Time period hint extracted from query

        Returns:
            Tuple[Optional[date], Optional[date]]: Start and end dates for the period
        """
        if not time_hint:
            return None, None

        time_hint = time_hint.lower().strip()
        today = date.today()

        # Handle common time expressions
        if time_hint == "today":
            return today, today
        elif time_hint == "yesterday":
            yesterday = today - timedelta(days=1)
            return yesterday, yesterday
        elif time_hint == "this week":
            start_of_week = today - timedelta(days=today.weekday())
            return start_of_week, today
        elif time_hint == "last week":
            start_of_last_week = today - timedelta(days=today.weekday() + 7)
            end_of_last_week = start_of_last_week + timedelta(days=6)
            return start_of_last_week, end_of_last_week
        elif time_hint == "this month":
            start_of_month = date(today.year, today.month, 1)
            return start_of_month, today
        elif time_hint == "last month":
            if today.month == 1:
                last_month = 12
                year = today.year - 1
            else:
                last_month = today.month - 1
                year = today.year
            start_of_last_month = date(year, last_month, 1)
            if last_month == 12:
                end_of_last_month = date(year, last_month, 31)
            else:
                end_of_last_month = date(year, last_month + 1, 1) - timedelta(days=1)
            return start_of_last_month, end_of_last_month
        elif time_hint == "this year":
            start_of_year = date(today.year, 1, 1)
            return start_of_year, today
        elif time_hint == "last year":
            start_of_last_year = date(today.year - 1, 1, 1)
            end_of_last_year = date(today.year - 1, 12, 31)
            return start_of_last_year, end_of_last_year
        elif time_hint == "this quarter":
            current_quarter = (today.month - 1) // 3 + 1
            start_month = (current_quarter - 1) * 3 + 1
            start_of_quarter = date(today.year, start_month, 1)
            return start_of_quarter, today
        elif time_hint == "last quarter":
            current_quarter = (today.month - 1) // 3 + 1
            if current_quarter == 1:
                # Last quarter of previous year
                start_month = 10
                year = today.year - 1
                end_month = 12
                end_day = 31
            else:
                # Previous quarter of current year
                start_month = ((current_quarter - 2) * 3) + 1
                year = today.year
                end_month = start_month + 2
                # Calculate last day of end month
                if end_month in [4, 6, 9, 11]:
                    end_day = 30
                elif end_month == 2:
                    if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
                        end_day = 29
                    else:
                        end_day = 28
                else:
                    end_day = 31

            start_of_last_quarter = date(year, start_month, 1)
            end_of_last_quarter = date(year, end_month, end_day)
            return start_of_last_quarter, end_of_last_quarter

        # Try to extract month and year
        month_pattern = r'(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+(\d{4}))?'
        month_match = re.search(month_pattern, time_hint)
        if month_match:
            month_name, year_str = month_match.groups()
            month_map = {
                "january": 1, "february": 2, "march": 3, "april": 4,
                "may": 5, "june": 6, "july": 7, "august": 8,
                "september": 9, "october": 10, "november": 11, "december": 12
            }
            month_num = month_map.get(month_name)
            if not month_num:
                return None, None

            # If year is not specified, use current year
            year = int(year_str) if year_str else today.year

            # Start date is first day of month
            start_date = date(year, month_num, 1)

            # End date is last day of month
            if month_num == 12:
                end_date = date(year, month_num, 31)
            else:
                end_date = date(year, month_num + 1, 1) - timedelta(days=1)

            return start_date, end_date

        # Try to extract short month name and year
        short_month_pattern = r'(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)(?:\s+(\d{4}))?'
        short_month_match = re.search(short_month_pattern, time_hint)
        if short_month_match:
            short_month, year_str = short_month_match.groups()
            month_map = {
                "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12
            }
            month_num = month_map.get(short_month)
            if not month_num:
                return None, None

            # If year is not specified, use current year
            year = int(year_str) if year_str else today.year

            # Start date is first day of month
            start_date = date(year, month_num, 1)

            # End date is last day of month
            if month_num == 12:
                end_date = date(year, month_num, 31)
            else:
                end_date = date(year, month_num + 1, 1) - timedelta(days=1)

            return start_date, end_date

        # Try to extract just a year
        year_pattern = r'(\d{4})'
        year_match = re.search(year_pattern, time_hint)
        if year_match:
            year = int(year_match.group(1))
            start_date = date(year, 1, 1)
            end_date = date(year, 12, 31)
            return start_date, end_date

        # Handle quarter patterns (Q1, Q2, Q3, Q4)
        quarter_pattern = r'q([1-4])(?:\s+(\d{4}))?'
        quarter_match = re.search(quarter_pattern, time_hint)
        if quarter_match:
            quarter, year_str = quarter_match.groups()
            quarter = int(quarter)

            # If year is not specified, use current year
            year = int(year_str) if year_str else today.year

            # Calculate start and end months for the quarter
            start_month = ((quarter - 1) * 3) + 1
            end_month = start_month + 2

            # Calculate last day of end month
            if end_month in [4, 6, 9, 11]:
                end_day = 30
            elif end_month == 2:
                if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
                    end_day = 29
                else:
                    end_day = 28
            else:
                end_day = 31

            start_date = date(year, start_month, 1)
            end_date = date(year, end_month, end_day)
            return start_date, end_date

        # No recognizable time period
        return None, None

    def sync_database_with_vector_store(self) -> Dict:
        """Synchronize the database with the vector store

        This ensures all transactions in the database are also in the vector store

        Returns:
            Dict: Result of the sync operation
        """
        try:
            # Get all transactions from the database in batches
            from src.db.database import Database
            db = Database()

            # First get the total count of transactions
            count_result = db.execute_raw_query("SELECT COUNT(*) as total FROM transactions")
            total_transactions = count_result[0]["total"] if count_result else 0

            logger.info(f"Total transactions in database: {total_transactions}")

            # Process in batches to avoid memory issues
            batch_size = 1000
            offset = 0
            all_transactions = []

            while offset < total_transactions:
                # Fetch a batch of transactions with pagination
                batch_query = f"SELECT * FROM transactions ORDER BY id LIMIT {batch_size} OFFSET {offset}"
                batch = db.execute_raw_query(batch_query)

                if not batch:
                    break

                all_transactions.extend(batch)
                offset += batch_size
                logger.info(f"Fetched batch of {len(batch)} transactions, total so far: {len(all_transactions)}")

            logger.info(f"Retrieved all {len(all_transactions)} transactions from database")

            # Get current vector store count
            try:
                vector_store_count = self.vector_store.collection_count(
                    self.vector_store.transaction_collection_name
                )
                logger.info(f"Vector store has {vector_store_count} transactions")
            except Exception as e:
                logger.error(f"Error getting vector store count: {e}")
                vector_store_count = 0

            # Check which transactions are already in the vector store
            tx_ids = [tx["id"] for tx in all_transactions]

            # Get existing IDs in vector store
            existing_ids = []
            try:
                existing_ids = self.vector_store.get_existing_ids(
                    self.vector_store.transaction_collection_name,
                    tx_ids
                )
                logger.info(f"Found {len(existing_ids)} existing IDs in vector store")
            except Exception as e:
                logger.error(f"Error checking existing IDs: {e}")

            # Filter out transactions already in the vector store
            new_transactions = [tx for tx in all_transactions if tx["id"] not in existing_ids]
            logger.info(f"Found {len(new_transactions)} new transactions to add to vector store")

            if not new_transactions:
                logger.info("No new transactions to add")
                return {
                    "success": True,
                    "added_to_vector_store": 0,
                    "vector_store_before": vector_store_count,
                    "vector_store_after": vector_store_count
                }

            # Format transactions for vector store
            formatted_transactions = []
            for tx in new_transactions:
                # Create text representation
                tx_text = (f"Transaction: {tx.get('description', '')} "
                          f"Amount: {tx.get('amount', 0)} "
                          f"Category: {tx.get('category', 'Uncategorized')} "
                          f"Date: {tx.get('date', '')}")

                # Create numeric date for filtering (YYYYMMDD format)
                date_str = tx.get('date', '')
                numeric_date = 0
                if date_str and len(date_str) >= 10:  # YYYY-MM-DD format requires at least 10 chars
                    try:
                        # Remove hyphens to get YYYYMMDD
                        numeric_date = int(date_str.replace('-', ''))
                    except (ValueError, TypeError):
                        # If conversion fails, default to 0
                        numeric_date = 0

                # Extract merchant from description
                merchant = self._extract_merchant_from_description(tx.get('description', ''))

                # Format metadata - ensure all fields are present and sanitize for ChromaDB
                # ChromaDB only accepts str, int, float, bool - no None values
                metadata = {
                    "id": tx.get('id', '') or '',
                    "date": tx.get('date', '') or '',
                    "description": tx.get('description', '') or '',
                    "amount": float(tx.get('amount', 0) or 0),
                    "category": tx.get('category', 'Uncategorized') or 'Uncategorized',
                    "account": tx.get('account', 'Unknown') or 'Unknown',
                    "numeric_date": numeric_date,
                    "year_month": date_str[:7] if date_str and len(date_str) >= 7 else '',
                    "merchant": merchant
                }

                # Ensure all values are of allowed types (str, int, float, bool)
                for key, value in list(metadata.items()):
                    if value is None:
                        metadata[key] = ''  # Replace None with empty string
                    elif not isinstance(value, (str, int, float, bool)):
                        metadata[key] = str(value)  # Convert other types to string

                formatted_transactions.append({
                    "id": tx.get('id', ''),
                    "text": tx_text,
                    "metadata": metadata
                })

            # Add to vector store in batches for better performance
            embedding_batch_size = 500  # Process 500 at a time to avoid memory issues
            total_added = 0

            for i in range(0, len(formatted_transactions), embedding_batch_size):
                batch = formatted_transactions[i:i+embedding_batch_size]
                logger.info(f"Adding batch of {len(batch)} transactions to vector store ({i+1}-{i+len(batch)} of {len(formatted_transactions)})")

                try:
                    result = self.vector_store.add_transactions(batch)
                    if result.get("success", False):
                        total_added += result.get("added", 0)
                        logger.info(f"Successfully added batch, total now: {total_added}")
                    else:
                        logger.warning(f"Failed to add batch: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    logger.error(f"Error adding batch to vector store: {e}")

            # Get new vector store count
            try:
                new_vector_store_count = self.vector_store.collection_count(
                    self.vector_store.transaction_collection_name
                )
                logger.info(f"Vector store now has {new_vector_store_count} transactions")
            except Exception as e:
                logger.error(f"Error getting new vector store count: {e}")
                new_vector_store_count = vector_store_count + total_added  # Estimate

            # Result
            return {
                "success": True,
                "added_to_vector_store": total_added,
                "vector_store_before": vector_store_count,
                "vector_store_after": new_vector_store_count
            }

        except Exception as e:
            logger.error(f"Error synchronizing database with vector store: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _extract_merchant_from_description(self, description: str) -> str:
        """Extract merchant name from transaction description

        Args:
            description (str): Transaction description

        Returns:
            str: Merchant name
        """
        if not description:
            return "UNKNOWN"

        # Convert to uppercase for consistency
        desc = description.upper()

        # Extract first word (often the merchant name)
        parts = desc.split()
        if not parts:
            return "UNKNOWN"

        # Check for common payment processors to extract real merchant
        if "PAYPAL" in desc and "*" in desc:
            # PayPal format: "PAYPAL *MERCHANTNAME"
            paypal_parts = desc.split("*", 1)
            if len(paypal_parts) > 1 and paypal_parts[1].strip():
                return paypal_parts[1].strip().split()[0]

        if "SQ *" in desc:
            # Square format: "SQ *MERCHANTNAME"
            square_parts = desc.split("*", 1)
            if len(square_parts) > 1 and square_parts[1].strip():
                return square_parts[1].strip().split()[0]

        # Extract first word and remove special characters
        merchant = re.sub(r'[^A-Z0-9]', '', parts[0])

        return merchant if merchant else "UNKNOWN"


if __name__ == "__main__":
    # Example usage
    retrieval = FinanceRetrieval()

    # Test retrieve similar transactions
    results = retrieval.retrieve_similar_transactions(
        query="coffee shop expenses",
        n_results=3
    )

    # Print results
    print("\nSimilar Transactions:")
    for tx in results.get("transactions", []):
        print(f"- {tx['description']}: ${tx['amount']:.2f}")
        print(f"  Category: {tx['category']}, Similarity: {tx['relevance_score']:.2f}")

    # Test retrieve spending insights (if data exists)
    try:
        insights = retrieval.retrieve_spending_insights(
            date_range=("2023-01-01", "2023-12-31")
        )

        print("\nSpending Insights:")
        print(f"Total transactions: {insights['total_transactions']}")
        print(f"Total spending: ${insights['total_spending']:.2f}")
        print(f"Total income: ${insights['total_income']:.2f}")
        print(f"Net cash flow: ${insights['net_cash_flow']:.2f}")

        print("\nTop categories:")
        for cat in insights.get("top_categories", []):
            print(f"- {cat['category']}: ${cat['amount']:.2f}")
    except Exception as e:
        print(f"Error getting insights: {e}") 