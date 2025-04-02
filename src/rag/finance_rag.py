import os
import sys
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import json
import time
import re
from dateutil.relativedelta import relativedelta

# Add the project root to path so we can import from other modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.embedding.embedding_pipeline import EmbeddingPipeline
from src.embedding.vector_store import VectorStore
from src.retrieval.retrieval_system import FinanceRetrieval
from src.llm.llm_client import LLMClient
from src.prompts.rag_prompts import (
    RAG_SYSTEM_PROMPT,
    FINANCIAL_QUERY_PROMPT,
    TRANSACTION_CATEGORIZATION_PROMPT,
    SPENDING_ANALYSIS_PROMPT,
    FINANCIAL_ADVICE_PROMPT,
    get_current_date_formatted,
    prepare_financial_context,
    extract_financial_insights,
    format_finance_rag_prompt
)
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("finance_rag")

class FinanceRAG:
    """Main RAG system for finance data"""

    def __init__(self, 
                llm_provider: str = "ollama",
                llm_model: Optional[str] = "llama3:latest",
                embedding_model: str = "all-MiniLM-L6-v2",
                vector_store: Optional[VectorStore] = None,
                temperature: float = 0.3):
        """Initialize the finance RAG system

        Args:
            llm_provider (str): LLM provider name
            llm_model (str, optional): Specific LLM model to use
            embedding_model (str): Embedding model name
            vector_store (VectorStore, optional): Existing vector store
            temperature (float): Temperature for LLM generation
        """
        # Initialize vector store and retrieval
        self.vector_store = vector_store or VectorStore(embedding_model_name=embedding_model)
        self.retrieval = FinanceRetrieval(
            embedding_model_name=embedding_model,
            vector_store=self.vector_store
        )

        # Initialize LLM client
        self.llm = LLMClient(
            provider=llm_provider,
            model_name=llm_model,
            temperature=temperature
        )

        # Current date for context
        self.current_date = get_current_date_formatted()

        # Initialize conversation history
        self.conversation_history = []

        logger.info(f"Initialized Finance RAG system with {llm_provider} and {embedding_model}")

    def query(self, query: str, conversation_history: Optional[List[Dict]] = None) -> str:
        """Generate response for financial query using RAG

        Args:
            query (str): User query
            conversation_history (Optional[List[Dict]]): Previous conversation

        Returns:
            str: Response to query
        """
        # Get time hint from query
        time_hint = self._extract_time_hint(query)
        category_hint = self._extract_category_hint(query)

        # Get relevant context for the query
        logger.info(f"Extracted time hint from query: {time_hint}")

        # Special handling for "What are my top categories" queries
        if self._is_top_categories_query(query):
            return self._handle_top_categories_query(time_hint)

        # Special handling for spending analysis queries
        if self._is_spending_analysis_query(query):
            return self._handle_spending_analysis_query(query, time_hint, category_hint)

        # Retrieve similar transactions from vector store
        retrieval_results = self.retrieval.retrieve_similar_transactions(
            query=query,
            n_results=30,
            time_period=time_hint,
            category=category_hint
        )

        # Format context for LLM
        context = self.get_relevant_context(query)

        # Format conversation history
        conversation_context = ""
        if conversation_history:
            conversation_context = "Previous conversation:\n"
            for message in conversation_history[-3:]:  # Include last 3 messages for context
                role = message.get("role", "user")
                content = message.get("content", "")
                conversation_context += f"{role}: {content}\n"

        # Generate response with LLM even if no transactions found
        prompt = format_finance_rag_prompt(
            query=query,
            context=context if context else f"No transactions found for the time period: {time_hint or 'any'} and category: {category_hint or 'any'}",
            conversation_history=conversation_context
        )

        return self.llm.generate_text(prompt)

    def categorize_transactions(self, 
                              transactions: List[Dict],
                              available_categories: List[str] = None) -> List[Dict]:
        """Categorize a list of transactions using the LLM

        Args:
            transactions (List[Dict]): List of transaction dictionaries
            available_categories (List[str], optional): List of allowed categories

        Returns:
            List[Dict]: Transactions with added or updated categories
        """
        # Get default categories if not provided
        if not available_categories:
            try:
                # Query the vector store for categories
                categories_collection = self.vector_store.client.get_collection(
                    name=self.vector_store.category_collection_name
                )
                available_categories = categories_collection.get()["documents"]
            except:
                # Use a default set of categories
                available_categories = [
                    "Food & Dining", "Shopping", "Housing", "Transportation", 
                    "Entertainment", "Health & Fitness", "Travel", "Utilities",
                    "Income", "Gifts & Donations", "Education", "Investments",
                    "Fees & Charges", "Business Services", "Taxes", "Transfer",
                    "Uncategorized"
                ]

        # Format transactions for the prompt
        from src.prompts.rag_prompts import format_transactions_for_categorization
        formatted_transactions = format_transactions_for_categorization(transactions)

        # Create categorization prompt
        prompt = TRANSACTION_CATEGORIZATION_PROMPT.format(
            transactions=formatted_transactions,
            available_categories=", ".join(available_categories)
        )

        # System prompt
        system_prompt = "You are an AI specialized in financial transaction categorization. Your task is to assign appropriate categories to transactions based on their descriptions."

        # Get categorization from LLM
        response = self.llm.generate_text(
            prompt=prompt,
            system_prompt=system_prompt
        )

        # Parse the response to extract categorizations
        categorized_transactions = self._parse_categorization_response(
            response, transactions, available_categories
        )

        return categorized_transactions

    def spending_analysis(self, 
                        time_period: str = "this month",
                        category_filter: Optional[str] = None) -> Dict:
        """Generate spending analysis for a time period

        Args:
            time_period (str): Time period to analyze (e.g., "this month", "last quarter")
            category_filter (Optional[str]): Category to filter by

        Returns:
            Dict: Analysis results with structured and text insights
        """
        try:
            # Parse time period into date range
            start_date, end_date = self.retrieval._parse_time_period(time_period)

            if not start_date or not end_date:
                logger.warning(f"Could not parse time period: {time_period}")
                return {
                    "error": f"Could not understand time period: {time_period}",
                    "structured_insights": {
                        "total_spending": 0,
                        "categories": {}
                    }
                }

            # Initialize database
            from src.db.database import Database
            db = Database()

            # Get transactions for time period
            transactions_query = "SELECT * FROM transactions WHERE amount < 0"
            params = []

            # Add date filters
            if start_date:
                transactions_query += " AND date >= ?"
                params.append(start_date.strftime('%Y-%m-%d'))
            if end_date:
                transactions_query += " AND date <= ?"
                params.append(end_date.strftime('%Y-%m-%d'))

            # Add category filter if provided
            if category_filter:
                transactions_query += " AND category LIKE ?"
                params.append(f"%{category_filter}%")

            # Get transactions
            transactions = db.execute_raw_query(transactions_query, params)

            # If no transactions found, return empty analysis
            if not transactions:
                return {
                    "error": f"No transactions found for period: {time_period}",
                    "structured_insights": {
                        "total_spending": 0,
                        "categories": {}
                    }
                }

            # Get total spending
            total_query = "SELECT SUM(ABS(amount)) FROM transactions WHERE amount < 0"
            total_params = []

            # Add date filters
            if start_date:
                total_query += " AND date >= ?"
                total_params.append(start_date.strftime('%Y-%m-%d'))
            if end_date:
                total_query += " AND date <= ?"
                total_params.append(end_date.strftime('%Y-%m-%d'))

            # Add category filter if provided
            if category_filter:
                total_query += " AND category LIKE ?"
                total_params.append(f"%{category_filter}%")

            # Get total spending
            total_result = db.execute_raw_query(total_query, total_params)
            total_spending = 0
            if total_result and len(total_result) > 0 and total_result[0] is not None and total_result[0][0] is not None:
                total_spending = abs(total_result[0][0])

            # Get category breakdown
            category_query = """
                SELECT category, SUM(ABS(amount)) as total
                FROM transactions
                WHERE amount < 0
            """

            category_params = []

            # Add date filters
            if start_date:
                category_query += " AND date >= ?"
                category_params.append(start_date.strftime('%Y-%m-%d'))
            if end_date:
                category_query += " AND date <= ?"
                category_params.append(end_date.strftime('%Y-%m-%d'))

            # Add category filter if provided
            if category_filter:
                category_query += " AND category LIKE ?"
                category_params.append(f"%{category_filter}%")

            category_query += " GROUP BY category ORDER BY total DESC"

            # Get category breakdown
            category_results = db.execute_raw_query(category_query, category_params)

            # Prepare structured insights
            structured_insights = {
                "total_spending": total_spending,
                "categories": {}
            }

            if category_results:
                # Convert to dictionary
                for category, amount in category_results:
                    structured_insights["categories"][category] = amount

                # Add percentage of total
                for category in structured_insights["categories"]:
                    amount = structured_insights["categories"][category]
                    if total_spending > 0:
                        structured_insights["categories"][category] = {
                            "amount": amount,
                            "percentage": (amount / total_spending) * 100
                        }

            # Prepare prompt with insights
            if category_filter:
                prompt = f"Analyze my spending in the category '{category_filter}' for {time_period}."
            else:
                prompt = f"Analyze my spending for {time_period}."

            # Add structured data to prompt
            context = f"""
            Time Period: {time_period} ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})
            
            Total Spending: ${total_spending:.2f}
            
            Spending by Category:
            """

            for category, data in structured_insights["categories"].items():
                amount = data["amount"] if isinstance(data, dict) else data
                percentage = data.get("percentage", 0) if isinstance(data, dict) else 0
                context += f"- {category}: ${amount:.2f} ({percentage:.1f}%)\n"

            # Add transaction examples
            context += "\nSample Transactions:\n"
            for i, tx in enumerate(transactions[:5]):
                # Format transaction data
                tx_date = tx.get("date", "Unknown")
                tx_desc = tx.get("description", "Unknown")
                tx_amount = abs(tx.get("amount", 0))
                tx_category = tx.get("category", "Uncategorized")

                context += f"{i+1}. {tx_date} - {tx_desc} - ${tx_amount:.2f} - {tx_category}\n"

            # Generate analysis text with LLM
            analysis_text = self.llm.generate_text(
                SPENDING_ANALYSIS_PROMPT.format(spending_data=context)
            )

            # Return complete analysis
            return {
                "structured_insights": structured_insights,
                "text_analysis": analysis_text,
                "date_range": {
                    "start": start_date.strftime('%Y-%m-%d'),
                    "end": end_date.strftime('%Y-%m-%d')
                },
                "time_period": time_period
            }

        except Exception as e:
            logger.error(f"Error in spending analysis: {e}")
            return {
                "error": f"Error analyzing spending: {str(e)}",
                "structured_insights": {
                    "total_spending": 0,
                    "categories": {}
                }
            }

    def financial_advice(self, advice_request: str, time_period: str = "last 3 months") -> Dict:
        """Generate personalized financial advice based on user data

        Args:
            advice_request (str): Specific financial advice request
            time_period (str): Time period to consider

        Returns:
            Dict: Financial advice with context
        """
        # Convert time period to date range
        date_range = self.retrieval._parse_time_period(time_period)

        if not date_range:
            return {
                "error": f"Could not parse time period: {time_period}",
                "advice": ""
            }

        # Get financial data for context
        insights = self.retrieval.retrieve_spending_insights(
            date_range=date_range
        )

        # Format financial data for the prompt
        financial_data = json.dumps(insights, indent=2)

        # Create advice prompt
        prompt = FINANCIAL_ADVICE_PROMPT.format(
            financial_data=financial_data,
            advice_request=advice_request
        )

        # System prompt
        system_prompt = "You are an AI financial advisor specialized in personal finance. Your task is to provide helpful, personalized financial advice based on the user's data and request."

        # Get advice from LLM
        response = self.llm.generate_text(
            prompt=prompt,
            system_prompt=system_prompt
        )

        return {
            "advice_request": advice_request,
            "time_period": time_period,
            "date_range": date_range,
            "advice": response
        }

    def budget_recommendation(self, 
                            monthly_income: Optional[float] = None,
                            time_period: str = "last month") -> Dict:
        """Generate a personalized budget recommendation

        Args:
            monthly_income (float, optional): User's monthly income
            time_period (str): Time period to analyze

        Returns:
            Dict: Budget recommendation with categories
        """
        # Convert time period to date range
        date_range = self.retrieval._parse_time_period(time_period)

        if not date_range:
            return {
                "error": f"Could not parse time period: {time_period}",
                "budget": ""
            }

        # Get spending insights
        insights = self.retrieval.retrieve_spending_insights(
            date_range=date_range
        )

        # Calculate income if not provided
        if not monthly_income and "total_income" in insights:
            # Use available income data, normalized to monthly
            # This is an approximation - would need to calculate properly based on date range
            total_income = insights["total_income"]
            start_date, end_date = date_range

            # Crude approximation of months in the range
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            days = (end - start).days
            months = max(1, days / 30)  # Rough approximation

            monthly_income = total_income / months

        # Add income to the insights
        insights["monthly_income"] = monthly_income

        # Format financial data for the prompt
        financial_data = json.dumps(insights, indent=2)

        # Create budget prompt
        prompt = BUDGET_RECOMMENDATION_PROMPT.format(
            financial_data=financial_data
        )

        # System prompt
        system_prompt = "You are an AI financial planner specialized in budget creation. Your task is to create a personalized budget based on income and spending patterns."

        # Get budget from LLM
        response = self.llm.generate_text(
            prompt=prompt,
            system_prompt=system_prompt
        )

        # Extract structured budget from the response
        structured_budget = extract_financial_insights(response)

        return {
            "monthly_income": monthly_income,
            "time_period": time_period,
            "date_range": date_range,
            "budget": response,
            "structured_budget": structured_budget,
            "spending_insights": insights
        }

    def find_similar_transactions(self, transaction_description: str, n_results: int = 5) -> Dict:
        """Find transactions similar to a description

        Args:
            transaction_description (str): Description to match
            n_results (int): Number of results to return

        Returns:
            Dict: Similar transactions with analysis
        """
        # Retrieve similar transactions
        results = self.retrieval.retrieve_similar_transactions(
            query=transaction_description,
            n_results=n_results
        )

        # If no results found
        if not results or not results.get("transactions"):
            return {
                "query": transaction_description,
                "count": 0,
                "similar_transactions": [],
                "analysis": "No similar transactions found.",
                "llm_response": "No similar transactions found."
            }

        # Format similar transactions for the prompt
        similar_transactions = ""
        for i, tx in enumerate(results["transactions"]):
            date = tx.get("date", "Unknown date")
            description = tx.get("description", "Unknown transaction")
            amount = tx.get("amount", 0)
            category = tx.get("category", "Uncategorized")
            similarity = tx.get("similarity", 0)

            similar_transactions += f"{i+1}. Description: {description}\n"
            similar_transactions += f"   Amount: ${abs(amount):.2f}"
            if amount < 0:
                similar_transactions += " (expense)"
            else:
                similar_transactions += " (income)"
            similar_transactions += f"\n   Date: {date}"
            similar_transactions += f"\n   Category: {category}"
            similar_transactions += f"\n   Similarity: {similarity:.2f}\n\n"

        # Create similar transactions prompt
        from src.prompts.rag_prompts import SIMILAR_TRANSACTIONS_PROMPT
        prompt = SIMILAR_TRANSACTIONS_PROMPT.format(
            transaction_description=transaction_description,
            similar_transactions=similar_transactions
        )

        # System prompt
        system_prompt = "You are an AI financial analyst. Your task is to analyze similar transactions and provide insights."

        # Get analysis from LLM
        response = self.llm.generate_text(
            prompt=prompt,
            system_prompt=system_prompt
        )

        return {
            "query": transaction_description,
            "count": len(results["transactions"]),
            "similar_transactions": results["transactions"],
            "analysis": response,
            "llm_response": response
        }

    def clear_conversation_history(self) -> None:
        """Clear the conversation history"""
        self.conversation_history = []
        logger.info("Cleared conversation history")

    def _extract_time_hint(self, query: str) -> Optional[str]:
        """Extract time-related hints from the query

        Args:
            query (str): User query

        Returns:
            Optional[str]: Extracted time hint or None
        """
        query = query.lower()

        # Common time patterns
        patterns = [
            # Month names
            r'in\s+(january|february|march|april|may|june|july|august|september|october|november|december)',
            r'in\s+(jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)',
            r'during\s+(january|february|march|april|may|june|july|august|september|october|november|december)',
            r'during\s+(jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)',

            # Month with year
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})',
            r'(jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s+(\d{4})',

            # Relative periods
            r'(this|last|next)\s+(month|year|week)',
            r'past\s+(\d+)\s+(days|weeks|months|years)',
            r'(yesterday|today|tomorrow)',

            # Specific date formats
            r'(\d{4}-\d{1,2}-\d{1,2})',
            r'(\d{1,2}/\d{1,2}/\d{2,4})',

            # Date ranges
            r'between\s+(.+?)\s+and\s+(.+?)(\s|$)',
            r'from\s+(.+?)\s+to\s+(.+?)(\s|$)',

            # Year patterns
            r'from\s+(\d{4})',
            r'in\s+(\d{4})',
            r'for\s+(\d{4})',
            r'during\s+(\d{4})',
            r'transactions\s+from\s+(\d{4})',

            # Quarters (e.g., Q1 2023, first quarter, etc.)
            r'(q[1-4])\s+(\d{4})',
            r'(q[1-4])',
            r'(first|second|third|fourth)\s+quarter(\s+of\s+(\d{4}))?',

            # Special patterns
            r'year[ -]to[ -]date',
            r'ytd',
            r'all time'
        ]

        # Match patterns
        for pattern in patterns:
            matches = re.findall(pattern, query)
            if matches:
                # Handle year patterns
                if pattern.startswith(r'from\s+(\d{4})') or pattern.startswith(r'in\s+(\d{4})') or \
                   pattern.startswith(r'for\s+(\d{4})') or pattern.startswith(r'during\s+(\d{4})') or \
                   pattern.startswith(r'transactions\s+from\s+(\d{4})'):
                    return matches[0]

                # Handle different match formats
                if pattern.startswith(r'(q[1-4])'):
                    # Extract quarter information
                    quarter = matches[0]
                    if isinstance(quarter, tuple):
                        if len(quarter) > 1 and quarter[1]:  # Has year
                            return f"{quarter[0]} {quarter[1]}"
                        else:  # Just quarter, no year
                            return quarter[0]
                    return quarter

                # Handle first/second/third/fourth quarter
                if pattern.startswith(r'(first|second|third|fourth)'):
                    match = matches[0]
                    if isinstance(match, tuple):
                        quarter_num = {'first': 'q1', 'second': 'q2', 'third': 'q3', 'fourth': 'q4'}
                        quarter = quarter_num[match[0]]
                        if len(match) > 2 and match[2]:  # Has year
                            return f"{quarter} {match[2]}"
                        else:  # Just quarter, no year
                            return quarter

                # Handle month patterns
                if pattern.startswith(r'in\s+') or pattern.startswith(r'during\s+'):
                    return matches[0]

                # Handle other patterns
                if isinstance(matches[0], tuple):
                    return ' '.join(part for part in matches[0] if part)
                return matches[0]

        # Check for month name mentions without "in" preposition
        month_names = [
            'january', 'february', 'march', 'april', 'may', 'june', 'july', 
            'august', 'september', 'october', 'november', 'december',
            'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec'
        ]

        for month in month_names:
            if f" {month} " in f" {query} ":
                # Extract the month and any following year
                month_index = query.find(month)
                if month_index != -1:
                    # Look for a year after the month
                    after_month = query[month_index + len(month):]
                    year_match = re.search(r'\b(20\d{2})\b', after_month)
                    if year_match:
                        return f"{month} {year_match.group(1)}"
                    return month

        # Check for spending in specific time periods
        time_period_patterns = [
            # "spending in/for/during [time period]"
            r'spending\s+(?:in|for|during)\s+(.+?)(?:\?|$|\s+was|\s+were)',
            r'spent\s+(?:in|for|during)\s+(.+?)(?:\?|$|\s+was|\s+were)',
            r'expenses\s+(?:in|for|during)\s+(.+?)(?:\?|$|\s+was|\s+were)',
            r'spend\s+(?:in|for|during)\s+(.+?)(?:\?|$|\s+was|\s+were)',

            # "how much did I spend in [time period]"
            r'how\s+much\s+did\s+I\s+spend\s+(?:in|for|during)\s+(.+?)(?:\?|$|\s+on)',
        ]

        for pattern in time_period_patterns:
            match = re.search(pattern, query)
            if match:
                time_period = match.group(1).strip()
                # Filter out non-time words (on, about, etc.)
                time_period = re.sub(r'\b(on|about|around|exactly|precisely|approximately)\b', '', time_period)
                return time_period.strip()

        # Look for "Q1", "Q2", etc. anywhere in the query
        quarter_match = re.search(r'\b(q[1-4])\b', query)
        if quarter_match:
            # Check if there's a year after the quarter
            after_quarter = query[quarter_match.end():]
            year_match = re.search(r'\b(20\d{2})\b', after_quarter)
            if year_match:
                return f"{quarter_match.group(1)} {year_match.group(1)}"
            return quarter_match.group(1)

        # Look for standalone year (any 4-digit number starting with 19 or 20)
        year_match = re.search(r'\b((?:19|20)\d{2})\b', query)
        if year_match:
            return year_match.group(1)

        return None

    def _is_analysis_query(self, query: str) -> bool:
        """Check if the query is asking for financial analysis

        Args:
            query (str): User query

        Returns:
            bool: Whether this is an analysis query
        """
        query_lower = query.lower()

        # Keywords suggesting analysis (more selective)
        primary_analysis_keywords = [
            "analyze", "analysis", "breakdown", "summary", "summarize",
            "insight", "pattern", "trend", "budget", "report", "overview",
            "statistics", "stats", "distribution"
        ]

        # Secondary keywords - must be combined with other indicators
        secondary_keywords = [
            "spending", "expense", "total", "average", "compare", "comparison",
            "top", "categories", "most", "least", "highest", "lowest",
            "frequent", "common", "changed", "change over", "difference", 
            "increased", "decreased", "percentage"
        ]

        # Specific phrases that strongly indicate analysis queries
        analysis_phrases = [
            "spending pattern", "spending habit", "expense breakdown",
            "budget analysis", "financial overview", "spending overview",
            "spending report", "expense report", "financial summary",
            "budget summary", "spending summary", "expense summary",
            "spending trend", "expense trend", "budget trend",
            "top categories", "main categories", "primary categories",
            "biggest expenses", "largest expenses", "highest expenses",
            "spending distribution", "expense distribution"
        ]

        # Exclude patterns - these indicate simple lookups rather than analysis
        exclude_patterns = [
            r"how much did i spend (at|on) [^?]*\?",
            r"what was my [^?]* (transaction|payment|expense|bill)\?",
            r"did i pay [^?]*\?",
            r"show me [^?]* (expenses|transactions|payments)\?*"
        ]

        # Check for excluded patterns first
        for pattern in exclude_patterns:
            if re.search(pattern, query_lower):
                return False

        # Check for analysis phrases
        for phrase in analysis_phrases:
            if phrase in query_lower:
                return True

        # Check for primary analysis keywords
        for keyword in primary_analysis_keywords:
            if keyword in query_lower:
                return True

        # Check for combinations of secondary keywords
        keyword_count = 0
        for keyword in secondary_keywords:
            if keyword in query_lower:
                keyword_count += 1

        # If more than one secondary keyword is found, it's likely an analysis query
        return keyword_count >= 2

    def _parse_categorization_response(self, 
                                     response: str, 
                                     transactions: List[Dict],
                                     available_categories: List[str]) -> List[Dict]:
        """Parse LLM response for transaction categorization

        Args:
            response (str): LLM categorization response
            transactions (List[Dict]): Original transactions
            available_categories (List[str]): Available categories

        Returns:
            List[Dict]: Categorized transactions
        """
        # Create a copy of the original transactions
        categorized_txs = [tx.copy() for tx in transactions]

        # Map of transaction ID to index
        tx_id_map = {tx.get("id", f"tx_{i}"): i for i, tx in enumerate(categorized_txs)}

        # Process the response line by line
        current_tx_id = None
        current_category = None

        for line in response.strip().split('\n'):
            line = line.strip()

            if not line:
                continue

            # Look for transaction ID
            if line.lower().startswith("transaction id:"):
                current_tx_id = line.split(":", 1)[1].strip()
                current_category = None
                continue

            # Look for category assignment
            if "category:" in line.lower():
                if current_tx_id and current_tx_id in tx_id_map:
                    # Extract category
                    category_part = line.split(":", 1)[1].strip()

                    # Find the closest matching category
                    current_category = self._find_closest_category(
                        category_part, available_categories
                    )

                    # Update the transaction
                    tx_index = tx_id_map[current_tx_id]
                    categorized_txs[tx_index]["category"] = current_category

                    # Also store the original assigned category
                    categorized_txs[tx_index]["assigned_category"] = category_part

        return categorized_txs

    def _find_closest_category(self, 
                             category_text: str, 
                             available_categories: List[str]) -> str:
        """Find the closest matching category from available categories

        Args:
            category_text (str): Extracted category text
            available_categories (List[str]): List of available categories

        Returns:
            str: Best matching category
        """
        # Clean up category text (remove explanations)
        category_text = category_text.split(" - ")[0].strip()
        category_text = category_text.split(",")[0].strip()
        category_text = category_text.split("(")[0].strip()

        # Direct match
        for category in available_categories:
            if category.lower() == category_text.lower():
                return category

        # Substring match
        for category in available_categories:
            if category.lower() in category_text.lower() or category_text.lower() in category.lower():
                return category

        # Map common categories to standard ones if needed
        category_mapping = {
            "electronics": "Shopping",
            "technology": "Shopping",
            "groceries": "Food",
            "dining": "Food",
            "restaurant": "Food",
            "coffee": "Food",
            "rent": "Housing",
            "mortgage": "Housing",
            "utilities": "Housing",
            "car": "Transportation",
            "gas": "Transportation",
            "fuel": "Transportation",
            "uber": "Transportation",
            "lyft": "Transportation",
            "taxi": "Transportation",
            "salary": "Income",
            "paycheck": "Income",
            "deposit": "Income"
        }

        # Check if the category text is in our mapping
        for key, mapped_category in category_mapping.items():
            if key in category_text.lower() and mapped_category in available_categories:
                return mapped_category

        # If no match found, return Uncategorized or the first category
        return "Uncategorized" if "Uncategorized" in available_categories else available_categories[0]

    def _filter_transactions_by_date(self, transactions: List[Dict], time_hint: Optional[str] = None) -> List[Dict]:
        """Filter transactions by date based on time hint

        Args:
            transactions (List[Dict]): List of transactions
            time_hint (Optional[str]): Time hint to filter by

        Returns:
            List[Dict]: Filtered transactions
        """
        if not time_hint or not transactions:
            return transactions

        # Parse time period into date range
        start_date, end_date = self.retrieval._parse_time_period(time_hint)

        if not start_date and not end_date:
            logger.warning(f"Could not parse time hint: {time_hint}")
            return transactions

        logger.info(f"Filtering transactions from {start_date} to {end_date}")

        # Filter transactions by date
        filtered = []
        for txn in transactions:
            # Skip if transaction has no date
            if 'date' not in txn:
                continue

            # Parse transaction date
            try:
                # Handle different date formats
                txn_date = txn['date']
                if isinstance(txn_date, str):
                    # Try different formats
                    for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d', '%d/%m/%Y']:
                        try:
                            txn_date = datetime.strptime(txn['date'], fmt).date()
                            break
                        except ValueError:
                            continue
                    else:
                        # If all formats fail, skip this transaction
                        logger.warning(f"Could not parse date format for transaction: {txn}")
                        continue
                elif isinstance(txn_date, datetime):
                    txn_date = txn_date.date()

                # Check if date is within range
                if start_date and txn_date < start_date:
                    continue
                if end_date and txn_date > end_date:
                    continue

                filtered.append(txn)
            except Exception as e:
                logger.error(f"Error filtering transaction by date: {e}")
                continue

        return filtered

    def get_relevant_context(self, query: str) -> str:
        """Get relevant financial context for the query

        Args:
            query (str): User query

        Returns:
            str: Formatted context for the LLM
        """
        # Extract time period hint
        time_hint = self._extract_time_hint(query)
        logger.info(f"Extracted time hint: {time_hint}")

        # Extract category hint - e.g., "food", "groceries", etc.
        category_hint = self._extract_category_hint(query)
        logger.info(f"Extracted category hint: {category_hint}")

        # Get similar transactions based on semantic search
        similar_txns = self.retrieval.retrieve_similar_transactions(
            query=query,
            n_results=50,
            time_period=time_hint,
            category=category_hint
        )

        # If no transactions found, try database retrieval
        if not similar_txns:
            logger.info("No similar transactions found in vector store, trying database retrieval")
            similar_txns = self._get_transactions_from_database(time_hint, category_hint)

        # Get spending insights for time period if appropriate
        spending_insights = self._get_spending_insights(time_hint, category_hint)

        # Format context with transactions and insights
        return prepare_financial_context(
            retrieval_results=similar_txns,
            spending_insights=spending_insights,
            time_period=time_hint,
            category=category_hint
        )

    def _get_transactions_from_database(self, time_hint: Optional[str] = None, category_hint: Optional[str] = None) -> List[Dict]:
        """Get transactions from the database based on time and category

        Args:
            time_hint (Optional[str]): Time hint to filter by
            category_hint (Optional[str]): Category hint to filter by

        Returns:
            List[Dict]: Transactions from database
        """
        try:
            # Initialize database
            from src.db.database import Database
            db = Database()

            # Parse time period into date range
            start_date, end_date = None, None
            if time_hint:
                start_date, end_date = self.retrieval._parse_time_period(time_hint)

            # Build SQL query based on filters
            query = "SELECT * FROM transactions WHERE 1=1"
            params = []

            if start_date:
                query += " AND date >= ?"
                params.append(start_date.strftime('%Y-%m-%d'))
            if end_date:
                query += " AND date <= ?"
                params.append(end_date.strftime('%Y-%m-%d'))
            if category_hint:
                query += " AND category LIKE ?"
                params.append(f'%{category_hint}%')

            # Add limit and order by
            query += " ORDER BY date DESC LIMIT 50"

            # Execute query
            transactions = db.execute_raw_query(query, params)

            logger.info(f"Retrieved {len(transactions)} transactions from database with filters: time={time_hint}, category={category_hint}")
            return transactions
        except Exception as e:
            logger.error(f"Error in direct database fallback: {e}")
            return []

    def _extract_category_hint(self, query: str) -> Optional[str]:
        """Extract category hint from query

        Args:
            query (str): User query

        Returns:
            Optional[str]: Category hint or None
        """
        # Convert to lowercase
        query = query.lower()

        # Common financial categories
        categories = [
            "groceries", "food", "restaurants", "dining", "coffee", "drinks", 
            "shopping", "clothing", "electronics", "household", "furniture",
            "transportation", "uber", "lyft", "taxi", "gas", "fuel", "car", "automotive",
            "rent", "mortgage", "housing", "utilities", "electric", "water", "internet", "cable",
            "entertainment", "movies", "subscription", "streaming", "travel", "hotel", "flight",
            "insurance", "health", "medical", "dental", "pharmacy", "education", "tuition",
            "childcare", "fitness", "gym", "sports", "charity", "donation"
        ]

        # Look for category mentions
        for category in categories:
            # Look for category patterns
            patterns = [
                rf'\b{category}\b',                    # Direct mention
                rf'{category} expenses',               # "category expenses"
                rf'{category} spending',               # "category spending"
                rf'spent on {category}',               # "spent on category"
                rf'spending on {category}',            # "spending on category"
                rf'pay for {category}',                # "pay for category"
                rf'spent at {category}',               # "spent at category"
                rf'money on {category}',               # "money on category"
                rf'how much (?:did|have) I (?:spend|spent) on {category}'  # Question format
            ]

            for pattern in patterns:
                if re.search(pattern, query):
                    return category

        return None

    def _is_top_categories_query(self, query: str) -> bool:
        """Check if query is asking for top spending categories

        Args:
            query (str): User query

        Returns:
            bool: True if query is about top categories
        """
        query = query.lower()
        patterns = [
            r'top (?:spending )?categories',
            r'spend the most',
            r'highest spending',
            r'where (?:do|did) I spend (?:the )?most',
            r'what (?:are|were) my (?:main|primary|highest|largest) expenses',
            r'where (?:does|did) my money go',
            r'biggest expenses',
            r'most money on'
        ]

        for pattern in patterns:
            if re.search(pattern, query):
                return True

        return False

    def _handle_top_categories_query(self, time_hint: Optional[str] = None) -> str:
        """Handle query about top spending categories

        Args:
            time_hint (Optional[str]): Time period hint

        Returns:
            str: Response with top spending categories
        """
        try:
            # Initialize database
            from src.db.database import Database
            db = Database()

            # Parse time period into date range
            start_date, end_date = None, None
            if time_hint:
                start_date, end_date = self.retrieval._parse_time_period(time_hint)

            # Query database for top categories
            top_categories = db.get_top_spending_categories(
                start_date=start_date,
                end_date=end_date,
                limit=5
            )

            if not top_categories:
                return "I couldn't find any spending data for the specified time period."

            # Format response
            period_text = f" for {time_hint}" if time_hint else ""
            response = f"Here are your top spending categories{period_text}:\n\n"

            for i, (category, amount) in enumerate(top_categories, 1):
                response += f"{i}. {category.capitalize()}: ${abs(amount):.2f}\n"

            return response
        except Exception as e:
            logger.error(f"Error in top categories query: {e}")
            return "I encountered an error while analyzing your top spending categories."

    def _is_spending_analysis_query(self, query: str) -> bool:
        """Check if query is asking for spending analysis

        Args:
            query (str): User query

        Returns:
            bool: True if query is about spending analysis
        """
        query = query.lower()
        patterns = [
            r'how much (?:did|have) I spend',
            r'total spending',
            r'spending breakdown',
            r'analysis of (?:my )?spending',
            r'spending (?:summary|report)',
            r'how much money (?:did|have) I spent',
            r'total expenses'
        ]

        for pattern in patterns:
            if re.search(pattern, query):
                return True

        return False

    def _handle_spending_analysis_query(self, query: str, time_hint: Optional[str] = None, category_hint: Optional[str] = None) -> Dict:
        """Handle a spending analysis query

        Args:
            query (str): User query
            time_hint (Optional[str]): Time period hint
            category_hint (Optional[str]): Category hint

        Returns:
            Dict: Analysis results
        """
        # Use the spending_analysis method
        return self.spending_analysis(
            time_period=time_hint or "this month",
            category_filter=category_hint
        )

    def _get_spending_insights(self, time_hint: Optional[str] = None, category_hint: Optional[str] = None) -> Optional[Dict]:
        """Get spending insights for a time period and/or category

        Args:
            time_hint (Optional[str]): Time period hint
            category_hint (Optional[str]): Category hint

        Returns:
            Optional[Dict]: Spending insights or None if not available
        """
        try:
            # Initialize database
            from src.db.database import Database
            db = Database()

            # Parse time period into date range
            start_date, end_date = None, None
            if time_hint:
                start_date, end_date = self.retrieval._parse_time_period(time_hint)

            # Build query for total spending
            total_query = "SELECT SUM(amount) FROM transactions WHERE amount < 0"  # Only expenses
            params = []

            if start_date:
                total_query += " AND date >= ?"
                params.append(start_date.strftime('%Y-%m-%d'))
            if end_date:
                total_query += " AND date <= ?"
                params.append(end_date.strftime('%Y-%m-%d'))
            if category_hint:
                total_query += " AND category LIKE ?"
                params.append(f'%{category_hint}%')

            # Get total spending
            total_result = db.execute_raw_query(total_query, params)
            total_spending = 0
            if total_result and len(total_result) > 0 and total_result[0] is not None and total_result[0][0] is not None:
                total_spending = abs(total_result[0][0])

            # Build query for category breakdown
            cat_query = """
                SELECT category, SUM(amount) as total 
                FROM transactions 
                WHERE amount < 0
            """
            cat_params = []

            if start_date:
                cat_query += " AND date >= ?"
                cat_params.append(start_date.strftime('%Y-%m-%d'))
            if end_date:
                cat_query += " AND date <= ?"
                cat_params.append(end_date.strftime('%Y-%m-%d'))

            cat_query += " GROUP BY category ORDER BY total ASC LIMIT 10"

            # Get category breakdown
            cat_results = db.execute_raw_query(cat_query, cat_params)

            # Format insights
            insights = {
                "total_spending": total_spending
            }

            if cat_results:
                insights["category_breakdown"] = {cat: abs(amount) for cat, amount in cat_results if amount}

            # Get month-over-month comparison if we have a date range
            if start_date and end_date:
                # Get current month spending
                current_month = total_spending

                # Calculate previous month date range
                prev_start = start_date - relativedelta(months=1)
                prev_end = end_date - relativedelta(months=1)

                # Query for previous month spending
                prev_query = "SELECT SUM(amount) FROM transactions WHERE amount < 0"
                prev_params = []

                prev_query += " AND date >= ?"
                prev_params.append(prev_start.strftime('%Y-%m-%d'))
                prev_query += " AND date <= ?"
                prev_params.append(prev_end.strftime('%Y-%m-%d'))
                if category_hint:
                    prev_query += " AND category LIKE ?"
                    prev_params.append(f'%{category_hint}%')

                # Get previous month spending
                prev_result = db.execute_raw_query(prev_query, prev_params)
                prev_month_spending = 0
                if prev_result and len(prev_result) > 0 and prev_result[0] is not None and prev_result[0][0] is not None:
                    prev_month_spending = abs(prev_result[0][0])

                # Add to insights
                if prev_month_spending:
                    insights["month_comparison"] = {
                        "current_month": current_month,
                        "previous_month": prev_month_spending,
                        "difference": current_month - prev_month_spending,
                        "percent_change": ((current_month - prev_month_spending) / abs(prev_month_spending)) * 100 if prev_month_spending else 0
                    }

            return insights
        except Exception as e:
            logger.error(f"Error getting spending insights: {e}")
            return None

    def synchronize_db_and_vector_store(self) -> Dict:
        """
        Synchronize the SQLite database and vector store to ensure they contain the same data.

        This method:
        1. Retrieves all transactions from the database
        2. Adds them to the vector store
        3. Extracts categories from transactions and adds them to the vector store
        4. Adds time periods to the vector store

        Returns:
            Dict: Results of the synchronization process
        """
        try:
            # Initialize database connection
            from src.db.database import Database
            db = Database()

            # Get all transactions from database (up to 10,000)
            transactions = db.query_transactions(limit=10000)

            if not transactions:
                return {"success": False, "error": "No transactions found in database"}

            # Add transactions to vector store
            tx_result = self.vector_store.add_transactions(transactions)

            # Extract categories from transactions
            categories = set()
            for tx in transactions:
                if tx.get("category") and isinstance(tx.get("category"), str):
                    categories.add(tx.get("category"))

            categories = list(categories)

            # Add categories to vector store
            cat_result = self.vector_store.add_categories(categories)

            # Add time periods to vector store
            time_result = self.vector_store.add_time_periods()

            return {
                "success": True,
                "transactions": {
                    "total": len(transactions),
                    "added": tx_result.get("added", 0)
                },
                "categories": {
                    "total": len(categories),
                    "added": len(categories) if cat_result else 0
                },
                "time_periods": {
                    "added": 34 if time_result else 0  # Typical number of time periods
                }
            }
        except Exception as e:
            logger.error(f"Error synchronizing DB and vector store: {e}")
            return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # Example usage
    print("Finance RAG System")

    try:
        # Initialize the RAG system
        rag = FinanceRAG()

        # Try a simple query
        query = "How much did I spend on coffee last month?"

        print(f"\nQuery: {query}")
        response = rag.query(query)

        print("\nResponse:")
        print(response)

    except Exception as e:
        print(f"Error: {e}") 