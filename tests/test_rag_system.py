import os
import sys
import unittest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.embedding.vector_store import VectorStore
from src.llm.llm_client import LLMClient
from src.rag.finance_rag import FinanceRAG

class TestRAGSystem(unittest.TestCase):
    """Test the RAG system components"""
    
    def setUp(self):
        """Set up test environment"""
        # Set environment to test mode to enable mocking
        os.environ["ENVIRONMENT"] = "test"
        
        # Create temp directory for vector store
        self.temp_dir = tempfile.mkdtemp()
        os.environ["VECTOR_DB_PATH"] = self.temp_dir
        
        # Sample transactions for testing
        self.sample_transactions = [
            {
                "id": "tx_001",
                "description": "Coffee at Starbucks",
                "amount": -4.85,
                "date": "2023-01-15",
                "merchant": "Starbucks",
                "category": "Food"
            },
            {
                "id": "tx_002",
                "description": "Coffee at Peet's",
                "amount": -5.25,
                "date": "2023-01-18",
                "merchant": "Peet's Coffee",
                "category": "Food"
            },
            {
                "id": "tx_003",
                "description": "Amazon.com - Headphones",
                "amount": -79.99,
                "date": "2023-01-20",
                "merchant": "Amazon",
                "category": "Shopping"
            },
            {
                "id": "tx_004",
                "description": "Monthly rent payment",
                "amount": -1500.00,
                "date": "2023-01-01",
                "merchant": "Property Management",
                "category": "Housing"
            },
            {
                "id": "tx_005",
                "description": "Salary deposit",
                "amount": 3000.00,
                "date": "2023-01-15",
                "merchant": "Employer",
                "category": "Income"
            }
        ]
        
        # Initialize vector store and add sample data
        self.vector_store = VectorStore(embedding_model_name="all-MiniLM-L6-v2")
        self.vector_store.add_transactions(self.sample_transactions)
        
        # Set up categories
        self.categories = ["Food", "Housing", "Transportation", "Shopping", "Income"]
        self.vector_store.add_categories(self.categories)
        
        # Add time periods
        self.vector_store.add_time_periods()
        
    def tearDown(self):
        """Clean up after tests"""
        # Remove temp directory
        shutil.rmtree(self.temp_dir)
        
    @patch('src.llm.llm_client.LLMClient.generate_text')
    def test_query_processing(self, mock_generate_text):
        """Test query processing in the RAG system"""
        # Mock LLM response
        mock_generate_text.return_value = "You spent $10.10 on coffee in January 2023."
        
        # Initialize RAG system with our vector store
        rag = FinanceRAG(
            llm_provider="ollama",
            llm_model="llama3",  # Use correct model name with tag
            vector_store=self.vector_store,
            temperature=0.0  # Deterministic for testing
        )
        
        # Process a query
        response = rag.query(
            query="How much did I spend on coffee in January?"
        )
        
        # Verify LLM was called
        self.assertTrue(mock_generate_text.called, "LLM generate_text was not called")
        
        # Verify response from RAG
        self.assertEqual(response, "You spent $10.10 on coffee in January 2023.",
                        "Unexpected RAG response")
    
    @patch('src.llm.llm_client.LLMClient.generate_text')
    def test_category_filter(self, mock_generate_text):
        """Test RAG query with category filter"""
        # Mock LLM response
        mock_generate_text.return_value = "Your food expenses in January were $10.10."
        
        # Mock retrieval to track category filter usage
        with patch('src.retrieval.retrieval_system.FinanceRetrieval.retrieve_similar_transactions') as mock_retrieve:
            # Set up mock return value
            mock_retrieve.return_value = {"transactions": [], "count": 0}
            
            # Initialize RAG system with our vector store
            rag = FinanceRAG(
                llm_provider="ollama",
                llm_model="llama3",  # Use correct model name with tag
                vector_store=self.vector_store,
                temperature=0.0  # Deterministic for testing
            )
            
            # Process a query with category filter
            response = rag.query(
                query="What were my food expenses in January?"
            )
            
            # Verify the category filter was passed to the retrieval function
            args, kwargs = mock_retrieve.call_args
            self.assertIn('category', kwargs, "Category not extracted from query")
            
            # Verify LLM was called
            self.assertTrue(mock_generate_text.called, "LLM generate_text was not called")
            
            # Verify response
            self.assertEqual(response, "Your food expenses in January were $10.10.",
                            "Unexpected RAG response with category filter")
    
    @patch('src.llm.llm_client.LLMClient.generate_text')
    def test_transaction_categorization(self, mock_generate_text):
        """Test transaction categorization"""
        # Mock LLM response
        mock_response = """
        Transaction ID: tx_001
        Category: Food
        
        Transaction ID: tx_002
        Category: Food
        
        Transaction ID: tx_003
        Category: Electronics
        """
        mock_generate_text.return_value = mock_response
        
        # Initialize RAG system with our vector store
        rag = FinanceRAG(
            llm_provider="ollama",
            llm_model="llama3",  # Use correct model name with tag
            vector_store=self.vector_store,
            temperature=0.0  # Deterministic for testing
        )
        
        # Sample uncategorized transactions
        uncategorized_txs = [
            {
                "id": "tx_001",
                "description": "STARBUCKS STORE #12345",
                "amount": -4.85,
                "date": "2023-01-15"
            },
            {
                "id": "tx_002",
                "description": "PEETS COFFEE #42",
                "amount": -5.25,
                "date": "2023-01-18"
            },
            {
                "id": "tx_003",
                "description": "BEST BUY #532",
                "amount": -79.99,
                "date": "2023-01-20"
            }
        ]
        
        # Categorize transactions
        categorized_txs = rag.categorize_transactions(
            transactions=uncategorized_txs,
            available_categories=self.categories
        )
        
        # Verify LLM was called
        self.assertTrue(mock_generate_text.called, "LLM generate_text was not called")
        
        # Verify categorization results
        self.assertEqual(len(categorized_txs), 3, "Wrong number of categorized transactions")
        
        # Check that tx_001 and tx_002 are categorized as Food
        self.assertEqual(categorized_txs[0]["category"], "Food",
                        "Transaction 1 not categorized as Food")
        self.assertEqual(categorized_txs[1]["category"], "Food",
                        "Transaction 2 not categorized as Food")
        
        # Check that tx_003 is mapped to Shopping (closest to Electronics in our categories)
        self.assertEqual(categorized_txs[2]["category"], "Shopping",
                        "Transaction 3 not mapped to appropriate category")
    
    @patch('src.llm.llm_client.LLMClient.generate_text')
    def test_spending_analysis(self, mock_generate_text):
        """Test spending analysis functionality"""
        # Mock LLM response
        mock_generate_text.return_value = "Your spending analysis for January shows $100.00 total."
        
        # Initialize RAG system with our vector store
        rag = FinanceRAG(
            llm_provider="ollama",
            llm_model="llama3",
            vector_store=self.vector_store,
            temperature=0.0
        )
        
        # Mock necessary parts
        with patch('src.db.database.Database.execute_raw_query') as mock_db_query:
            # Mock database query responses for transactions
            mock_db_query.side_effect = [
                # First call for transactions
                [
                    {"date": "2023-01-15", "description": "Coffee", "amount": -4.50, "category": "Food"},
                    {"date": "2023-01-20", "description": "Dinner", "amount": -25.75, "category": "Food"}
                ],
                # Second call for total spending
                [(30.25,)],
                # Third call for category breakdown
                [("Food", 30.25)]
            ]
            
            # Call spending_analysis directly
            time_period = "january"
            analysis = rag.spending_analysis(time_period=time_period)
            
            # Verify LLM was called
            self.assertTrue(mock_generate_text.called, "LLM generate_text was not called for valid time period")
            
            # Verify analysis structure
            self.assertIn("text_analysis", analysis, "Analysis should contain text_analysis field")
            self.assertIn("structured_insights", analysis, "Analysis should contain structured_insights")
            self.assertIn("total_spending", analysis["structured_insights"], "Analysis should contain total_spending")
        
        # Test with an invalid time period
        mock_generate_text.reset_mock()
        invalid_analysis = rag.spending_analysis("invalid time period")
        
        # Verify structure for invalid time
        self.assertIn("error", invalid_analysis, "Invalid analysis should have error field")
        self.assertIn("structured_insights", invalid_analysis, "Invalid analysis should have structured_insights")
        self.assertEqual(invalid_analysis["structured_insights"]["total_spending"], 0, "Total spending should be 0 for invalid time")
    
    @patch('src.llm.llm_client.LLMClient.generate_text')
    def test_conversation_history(self, mock_generate_text):
        """Test conversation history and context retention"""
        # Mock LLM responses
        mock_generate_text.side_effect = [
            "In January, you spent $10.10 on various expenses.",  # First response
            "Yes, your main expenses were coffee and groceries."  # Follow-up response
        ]
        
        # Initialize RAG system with our vector store
        rag = FinanceRAG(
            llm_provider="ollama",
            llm_model="llama3",  # Use correct model name with tag
            vector_store=self.vector_store,
            temperature=0.0  # Deterministic for testing
        )
        
        # First query
        first_response = rag.query("What were my expenses in January?")
        
        # Follow up query that refers to previous question
        follow_up_response = rag.query(
            query="Were those mostly for coffee?",
            conversation_history=[
                {"role": "user", "content": "What were my expenses in January?"},
                {"role": "assistant", "content": first_response}
            ]
        )
        
        # Verify LLM was called twice
        self.assertEqual(mock_generate_text.call_count, 2, 
                        "LLM generate_text was not called the expected number of times")
        
        # Verify responses
        self.assertEqual(first_response, 
                        "In January, you spent $10.10 on various expenses.",
                        "Unexpected first response")
        self.assertEqual(follow_up_response, 
                        "Yes, your main expenses were coffee and groceries.",
                        "Unexpected follow-up response")
        
        # Verify that conversation history was included in the second prompt
        second_call_args = mock_generate_text.call_args[0][0]
        self.assertIn("What were my expenses in January", second_call_args,
                    "First query not found in conversation history")
    
    def test_find_similar_transactions(self):
        """Test finding similar transactions"""
        # Initialize RAG system with our vector store
        rag = FinanceRAG(
            llm_provider="ollama",
            llm_model="llama3",
            vector_store=self.vector_store,
            temperature=0.0  # Deterministic for testing
        )
        
        # Use mocked LLM to avoid API calls
        with unittest.mock.patch('src.llm.llm_client.LLMClient.generate_text') as mock_generate_text:
            # Set up mock response
            mock_response = """Here are similar transactions to "coffee shop":
            1. Starbucks - $4.85 on 2023-01-15
            2. Peet's Coffee - $5.25 on 2023-01-18
            """
            mock_generate_text.return_value = mock_response
            
            # Temporarily patch the retrieval method to avoid ChromaDB filtering issues
            with unittest.mock.patch('src.retrieval.retrieval_system.FinanceRetrieval.retrieve_similar_transactions') as mock_retrieve:
                # Create synthetic results that mimic what would be returned
                mock_results = {
                    "transactions": [
                        {
                            "id": "tx_001",
                            "description": "Coffee at Starbucks",
                            "amount": -4.85,
                            "date": "2023-01-15",
                            "category": "Food",
                            "merchant": "Starbucks",
                            "similarity": 0.95
                        },
                        {
                            "id": "tx_002",
                            "description": "Coffee at Peet's",
                            "amount": -5.25,
                            "date": "2023-01-18",
                            "category": "Food",
                            "merchant": "Peet's Coffee",
                            "similarity": 0.87
                        }
                    ],
                    "count": 2
                }
                mock_retrieve.return_value = mock_results
                
                # Run the find_similar_transactions method
                result = rag.find_similar_transactions(
                    transaction_description="coffee",
                    n_results=5
                )
                
                # Verify that LLM was called
                self.assertTrue(mock_generate_text.called, "LLM generate_text was not called")
                
                # Check for valid response structure
                self.assertIsInstance(result, dict, "Result should be a dictionary")
                self.assertIn("similar_transactions", result, "Result missing similar_transactions key")
                self.assertIn("llm_response", result, "Result missing llm_response key")
                
                # Verify the LLM response was included
                self.assertEqual(result["llm_response"], mock_response)
    
    def test_time_hint_extraction(self):
        """Test extracting time hints from queries"""
        tests = [
            ("How much did I spend in January?", "january"),
            ("Show me transactions from February 2023", "february 2023"),
            ("What were my expenses last month?", "last month"),
            ("Spending for May", "may"),
            ("Transactions in Q1", "q1"),
            ("Show me transactions from 2023", "2023"),
            ("Spending for Q1 2022", "q1 2022"),
            ("What did I spend during March?", "march"),
            ("How much did I spend on groceries in Apr?", "apr")
        ]
        
        rag = FinanceRAG(llm_provider="ollama", llm_model="llama3", vector_store=self.vector_store)
        
        for query, expected_hint in tests:
            hint = rag._extract_time_hint(query)
            self.assertIsNotNone(hint, f"No time hint extracted from: {query}")
            self.assertEqual(hint.lower(), expected_hint, f"Expected '{expected_hint}' but got '{hint}' for query: {query}")
    
    def test_analysis_query_detection(self):
        """Test detection of analysis queries"""
        # Initialize RAG system with our vector store
        rag = FinanceRAG(
            llm_provider="ollama",
            llm_model="llama3",  # Use correct model name with tag
            vector_store=self.vector_store
        )
        
        # Test analysis queries
        analysis_queries = [
            "Analyze my spending for January",
            "Show me spending patterns",
            "What are my top expense categories?",
            "Generate a spending report",
            "How has my spending changed over time?"
        ]
        
        for query in analysis_queries:
            is_analysis = rag._is_analysis_query(query)
            self.assertTrue(is_analysis, f"Failed to detect analysis query: {query}")
        
        # Test non-analysis queries
        non_analysis_queries = [
            "How much did I spend at Starbucks?",
            "What was my largest transaction?",
            "Did I pay rent last month?",
            "Show me all grocery expenses"
        ]
        
        for query in non_analysis_queries:
            is_analysis = rag._is_analysis_query(query)
            self.assertFalse(is_analysis, f"Incorrectly flagged as analysis query: {query}")

if __name__ == '__main__':
    unittest.main() 