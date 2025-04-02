import os
import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Optional
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

class EmbeddingPipeline:
    """Pipeline for generating embeddings using SentenceTransformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding pipeline
        
        Args:
            model_name (str): Name of the SentenceTransformers model to use
                Default: "all-MiniLM-L6-v2" (384 dimensions, fast, good quality)
                Options: 
                    - "all-MiniLM-L6-v2" (384d, fast, good quality)
                    - "all-mpnet-base-v2" (768d, slower, better quality)
                    - "all-distilroberta-v1" (768d, good balance)
                    - "paraphrase-multilingual-MiniLM-L12-v2" (384d, multilingual)
        """
        self.model_name = model_name
        
        # Initialize the model
        self.model = SentenceTransformer(model_name)
        
        # Get the embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Set batch size based on available memory
        self.batch_size = 32  # Default batch size
        if torch.cuda.is_available():
            # If CUDA is available, use a larger batch size
            self.batch_size = 64
            self.model = self.model.to("cuda")
        
        print(f"Initialized embedding model: {model_name}")
        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Using device: {self.model.device}")
    
    def generate_embeddings(self, texts: List[str], 
                           show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for a list of texts
        
        Args:
            texts (List[str]): List of texts to embed
            show_progress (bool): Whether to show a progress bar
            
        Returns:
            np.ndarray: Array of embeddings with shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        # Use the model to encode the texts
        embeddings = self.model.encode(
            texts, 
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )
        
        return embeddings
    
    def generate_embeddings_batched(self, texts: List[str], 
                                   batch_size: Optional[int] = None,
                                   show_progress: bool = True) -> np.ndarray:
        """Generate embeddings in batches to handle large datasets
        
        Args:
            texts (List[str]): List of texts to embed
            batch_size (int, optional): Custom batch size. If None, uses the default.
            show_progress (bool): Whether to show a progress bar
            
        Returns:
            np.ndarray: Array of embeddings with shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        # Use the provided batch size or default
        actual_batch_size = batch_size or self.batch_size
        
        # Calculate number of batches
        n_texts = len(texts)
        n_batches = (n_texts + actual_batch_size - 1) // actual_batch_size
        
        all_embeddings = []
        
        # Set up progress bar if requested
        batches = range(n_batches)
        if show_progress:
            batches = tqdm(batches, desc="Generating embeddings")
        
        # Process each batch
        for i in batches:
            start_idx = i * actual_batch_size
            end_idx = min((i + 1) * actual_batch_size, n_texts)
            batch_texts = texts[start_idx:end_idx]
            
            # Generate embeddings for this batch
            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=actual_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all batch embeddings
        return np.vstack(all_embeddings)
    
    def generate_rich_text_embeddings(self, transactions: List[Dict], 
                                     include_category: bool = True,
                                     include_date: bool = True,
                                     include_amount: bool = True) -> Dict[str, np.ndarray]:
        """Generate embeddings for rich text representations of transactions
        
        Args:
            transactions (List[Dict]): List of transaction dictionaries
            include_category (bool): Whether to include category in the text
            include_date (bool): Whether to include date in the text
            include_amount (bool): Whether to include amount in the text
            
        Returns:
            Dict[str, np.ndarray]: Dictionary with 'embeddings', 'texts', and 'ids'
        """
        rich_texts = []
        transaction_ids = []
        
        for i, tx in enumerate(transactions):
            # Create rich text representation
            parts = []
            
            # Add description or merchant
            if tx.get('merchant') and tx.get('merchant') != "Unknown":
                parts.append(f"Transaction for {tx['merchant']}")
            else:
                parts.append(tx.get('description', 'Unknown transaction'))
            
            # Add category if available and requested
            if include_category and tx.get('category'):
                parts.append(f"Category: {tx['category']}")
            
            # Add date if available and requested
            if include_date and tx.get('date'):
                parts.append(f"Date: {tx['date']}")
            
            # Add amount if available and requested
            if include_amount and tx.get('amount') is not None:
                amt = float(tx['amount'])
                if amt < 0:
                    parts.append(f"Amount: ${abs(amt):.2f} spent")
                else:
                    parts.append(f"Amount: ${amt:.2f} received")
            
            # Join parts into a single text
            rich_text = " | ".join(parts)
            rich_texts.append(rich_text)
            
            # Use provided ID or generate one
            tx_id = tx.get('id') or f"tx_{i}"
            transaction_ids.append(tx_id)
        
        # Generate embeddings
        embeddings = self.generate_embeddings_batched(rich_texts)
        
        return {
            'embeddings': embeddings,
            'texts': rich_texts,
            'ids': transaction_ids
        }
    
    def generate_category_embeddings(self, categories: List[str]) -> Dict[str, np.ndarray]:
        """Generate embeddings for financial categories
        
        Args:
            categories (List[str]): List of category names
            
        Returns:
            Dict[str, np.ndarray]: Dictionary with 'embeddings' and 'categories'
        """
        # Create rich descriptions for categories
        category_descriptions = [
            f"Financial transactions in the category of {category}" 
            for category in categories
        ]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(category_descriptions)
        
        return {
            'embeddings': embeddings,
            'categories': categories
        }
    
    def generate_time_period_embeddings(self) -> Dict[str, np.ndarray]:
        """Generate embeddings for common time periods used in financial queries
        
        Returns:
            Dict[str, np.ndarray]: Dictionary with 'embeddings' and 'time_periods'
        """
        time_periods = [
            "today", "yesterday", "this week", "last week",
            "this month", "last month", "two months ago", "three months ago",
            "this quarter", "last quarter", "year to date", "this year",
            "last year", "all time", "recent", "last 30 days", "last 60 days",
            "last 90 days", "Q1", "Q2", "Q3", "Q4", "January", "February",
            "March", "April", "May", "June", "July", "August", "September",
            "October", "November", "December"
        ]
        
        # Create rich descriptions for time periods
        time_descriptions = [
            f"Financial transactions during {period}" 
            for period in time_periods
        ]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(time_descriptions)
        
        return {
            'embeddings': embeddings,
            'time_periods': time_periods
        }
    
    def generate_query_embeddings(self, query: str) -> np.ndarray:
        """Generate embedding for a single query
        
        Args:
            query (str): Query text
            
        Returns:
            np.ndarray: Query embedding with shape (embedding_dim,)
        """
        return self.model.encode(query, normalize_embeddings=True)
    
    def calculate_similarity(self, query_embedding: np.ndarray, 
                           document_embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between query and document embeddings
        
        Args:
            query_embedding (np.ndarray): Query embedding with shape (embedding_dim,)
            document_embeddings (np.ndarray): Document embeddings with shape (n_docs, embedding_dim)
            
        Returns:
            np.ndarray: Similarity scores with shape (n_docs,)
        """
        # Ensure query embedding is 2D for matrix multiplication
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate dot product (cosine similarity for normalized vectors)
        similarity = np.dot(document_embeddings, query_embedding.T).flatten()
        
        return similarity
    
    def generate_text_embeddings(self, texts: List[str]) -> Dict:
        """Generate embeddings for a list of text strings
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Dictionary with embeddings matrix
        """
        # Create embeddings
        embeddings = self.model.encode(texts)
        
        return {
            "embeddings": embeddings
        }
    
    def generate_transaction_embeddings(self, texts: List[str]) -> Dict:
        """Generate embeddings for a list of transaction texts
        
        Args:
            texts: List of transaction text descriptions
            
        Returns:
            Dictionary with embeddings matrix
        """
        # Create embeddings using the same underlying model
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        return {
            "embeddings": embeddings
        }


if __name__ == "__main__":
    # Example usage
    pipeline = EmbeddingPipeline()
    
    # Test with some example texts
    texts = [
        "Spent $50 at Starbucks on coffee",
        "Monthly rent payment of $1500",
        "Grocery shopping at Whole Foods",
        "Netflix subscription renewal",
        "Income deposit from employer"
    ]
    
    embeddings = pipeline.generate_embeddings(texts)
    print(f"Generated {len(embeddings)} embeddings with shape {embeddings.shape}")
    
    # Test query similarity
    query = "coffee purchases"
    query_embedding = pipeline.generate_query_embeddings(query)
    similarities = pipeline.calculate_similarity(query_embedding, embeddings)
    
    # Show results
    for i, (text, score) in enumerate(zip(texts, similarities)):
        print(f"{score:.4f}: {text}") 