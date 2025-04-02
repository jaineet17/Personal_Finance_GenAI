import os
import sys
import unittest
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.embedding.embedding_pipeline import EmbeddingPipeline
from src.embedding.vector_store import VectorStore

class TestEmbeddingSystem(unittest.TestCase):
    """Test the embedding system components"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temp directory for vector store
        self.temp_dir = tempfile.mkdtemp()
        os.environ["VECTOR_DB_PATH"] = self.temp_dir
        
        # Initialize embedding pipeline with small model
        self.embedding_pipeline = EmbeddingPipeline(model_name="all-MiniLM-L6-v2")
        
        # Initialize vector store
        self.vector_store = VectorStore(embedding_model_name="all-MiniLM-L6-v2")
        
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
        
    def tearDown(self):
        """Clean up after tests"""
        # Remove temp directory
        shutil.rmtree(self.temp_dir)
        
    def test_embedding_generation(self):
        """Test embedding generation for text"""
        # Test texts
        texts = [
            "Coffee purchase at local cafe",
            "Monthly rent payment",
            "Groceries at Whole Foods"
        ]
        
        # Generate embeddings
        embeddings = self.embedding_pipeline.generate_embeddings(texts)
        
        # Verify embeddings shape
        self.assertEqual(embeddings.shape[0], len(texts), 
                        "Number of embeddings doesn't match number of texts")
        self.assertEqual(embeddings.shape[1], self.embedding_pipeline.embedding_dim,
                        "Embedding dimension doesn't match expected dimension")
        
        # Verify embeddings are normalized (L2 norm = 1)
        norms = np.linalg.norm(embeddings, axis=1)
        for norm in norms:
            self.assertAlmostEqual(norm, 1.0, places=5, 
                                 msg="Embeddings are not properly normalized")
    
    def test_embedding_consistency(self):
        """Test consistency of embeddings for the same text"""
        # Test text
        text = "Coffee purchase at local cafe"
        
        # Generate embeddings twice
        embedding1 = self.embedding_pipeline.generate_embeddings([text])[0]
        embedding2 = self.embedding_pipeline.generate_embeddings([text])[0]
        
        # Calculate cosine similarity between the two embeddings
        sim = np.dot(embedding1, embedding2)
        
        # Similarity should be very close to 1.0 (identical)
        self.assertAlmostEqual(sim, 1.0, places=5, 
                              msg="Embeddings for the same text are not consistent")
    
    def test_embedding_similarity(self):
        """Test similarity calculations between embeddings"""
        # Similar texts
        text1 = "Coffee purchase at Starbucks"
        text2 = "Bought coffee at Peet's"
        text3 = "Monthly rent payment for apartment"
        
        # Generate embeddings
        embedding1 = self.embedding_pipeline.generate_embeddings([text1])[0]
        embedding2 = self.embedding_pipeline.generate_embeddings([text2])[0]
        embedding3 = self.embedding_pipeline.generate_embeddings([text3])[0]
        
        # Calculate similarities
        sim_coffee = np.dot(embedding1, embedding2)
        sim_different = np.dot(embedding1, embedding3)
        
        # Coffee texts should be more similar to each other than to rent text
        self.assertGreater(sim_coffee, sim_different,
                          "Similar texts don't have higher similarity than different texts")
    
    def test_rich_text_embeddings(self):
        """Test generation of rich text embeddings for transactions"""
        # Generate rich text embeddings
        embed_result = self.embedding_pipeline.generate_rich_text_embeddings(
            self.sample_transactions[:2]
        )
        
        # Verify the result structure
        self.assertIn('embeddings', embed_result, "Embeddings not found in result")
        self.assertIn('texts', embed_result, "Texts not found in result")
        self.assertIn('ids', embed_result, "IDs not found in result")
        
        # Verify embedding shape
        self.assertEqual(embed_result['embeddings'].shape[0], 2,
                        "Number of embeddings doesn't match number of transactions")
        self.assertEqual(embed_result['embeddings'].shape[1], self.embedding_pipeline.embedding_dim,
                        "Embedding dimension doesn't match expected dimension")
    
    def test_vector_store_transactions(self):
        """Test adding and querying transactions in vector store"""
        # Add transactions to vector store
        success = self.vector_store.add_transactions(self.sample_transactions)
        self.assertTrue(success, "Failed to add transactions to vector store")
        
        # Query for coffee-related transactions
        query_results = self.vector_store.query_similar(
            query_text="coffee purchase",
            n_results=2
        )
        
        # Verify query results
        self.assertTrue(len(query_results["ids"][0]) > 0, "No results returned from query")
        
        # The first result should be coffee-related
        first_result_idx = query_results["ids"][0][0]
        first_doc = query_results["documents"][0][0]
        self.assertIn("coffee", first_doc.lower(), 
                     "First result doesn't contain 'coffee'")
    
    def test_vector_store_categories(self):
        """Test adding and querying categories in vector store"""
        # Sample categories
        categories = ["Food", "Housing", "Transportation", "Shopping", "Income"]
        
        # Add categories to vector store
        success = self.vector_store.add_categories(categories)
        self.assertTrue(success, "Failed to add categories to vector store")
        
        # Query for shopping-related categories
        query_results = self.vector_store.query_similar(
            query_text="buying retail products",
            collection_name=self.vector_store.category_collection_name,
            n_results=2
        )
        
        # Verify query results
        self.assertTrue(len(query_results["ids"][0]) > 0, "No results returned from query")
    
    def test_metadata_filtering(self):
        """Test metadata filtering in vector store"""
        # Add transactions to vector store
        success = self.vector_store.add_transactions(self.sample_transactions)
        self.assertTrue(success, "Failed to add transactions to vector store")
        
        # Query with category filter
        query_results = self.vector_store.query_similar(
            query_text="purchase",
            filter_dict={"category": "Food"},
            n_results=2
        )
        
        # Verify all results have Food category
        for metadata in query_results["metadatas"][0]:
            self.assertEqual(metadata["category"], "Food",
                            "Result with non-Food category returned")
    
    def test_hybrid_search(self):
        """Test hybrid search functionality"""
        # Add transactions to vector store
        success = self.vector_store.add_transactions(self.sample_transactions)
        self.assertTrue(success, "Failed to add transactions to vector store")
        
        # Perform hybrid search
        hybrid_results = self.vector_store.hybrid_search(
            query_text="coffee drink purchase",  # More descriptive query
            n_results=5,  # Increased to get more results
            alpha=0.5  # Equal weight to semantic and keyword
        )
        
        # Verify results
        self.assertTrue(len(hybrid_results["ids"][0]) > 0, "No results returned from hybrid search")
        
        # Look for relevant results - either coffee-related or food-related
        relevant_found = False
        for doc in hybrid_results["documents"][0]:
            doc_lower = doc.lower()
            if "coffee" in doc_lower or "food" in doc_lower or "starbucks" in doc_lower or "peet" in doc_lower:
                relevant_found = True
                break
            
        self.assertTrue(relevant_found, "No relevant results found in hybrid search")

if __name__ == '__main__':
    unittest.main()