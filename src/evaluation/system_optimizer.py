"""
System optimizer for the Finance RAG system.
Implements optimizations based on performance analysis results.
"""
import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from src.rag.finance_rag import FinanceRAG
from src.llm.llm_client import LLMClient
from src.retrieval.retrieval_system import FinanceRetrieval

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("system_optimizer")


class SystemOptimizer:
    """Optimizer for the Finance RAG system"""
    
    def __init__(self, rag_system: Optional[FinanceRAG] = None):
        """Initialize the system optimizer
        
        Args:
            rag_system (Optional[FinanceRAG]): RAG system to optimize
        """
        self.rag_system = rag_system or FinanceRAG(
            llm_provider="ollama",
            llm_model="llama3:latest",
            temperature=0.1
        )
        self.optimizations_applied = []
    
    def optimize_retrieval(self) -> Dict[str, Any]:
        """Optimize the retrieval component
        
        Returns:
            Dict[str, Any]: Optimization results
        """
        logger.info("Optimizing retrieval component")
        
        # 1. Optimize the number of results returned by default
        original_n_results = 30  # Assuming default is 30
        optimized_n_results = 20  # Reduced based on performance analysis
        
        # Store original retrieval method
        original_retrieval_method = self.rag_system.retrieval.retrieve_similar_transactions
        
        # Define a wrapper to apply optimization
        def optimized_retrieval(query, n_results=optimized_n_results, **kwargs):
            """Optimized retrieval with reduced number of results"""
            return original_retrieval_method(query=query, n_results=n_results, **kwargs)
        
        # Apply optimization by monkey patching the method
        self.rag_system.retrieval.retrieve_similar_transactions = optimized_retrieval
        
        # 2. Apply caching for common queries (simple in-memory cache)
        # Create a cache dictionary as an attribute of the retrieval system
        self.rag_system.retrieval.query_cache = {}
        
        # Store original retrieve_similar_transactions method
        original_method = self.rag_system.retrieval.retrieve_similar_transactions
        
        # Define a new method with caching
        def cached_retrieval(query, **kwargs):
            """Cached retrieval function for better performance"""
            # Create a cache key from query and kwargs
            cache_key = f"{query}_{str(sorted(kwargs.items()))}"
            
            # Check if we have a cache hit
            if cache_key in self.rag_system.retrieval.query_cache:
                logger.info(f"Cache hit for query: {query}")
                return self.rag_system.retrieval.query_cache[cache_key]
            
            # Cache miss, call original method
            logger.info(f"Cache miss for query: {query}")
            results = original_method(query=query, **kwargs)
            
            # Store in cache
            self.rag_system.retrieval.query_cache[cache_key] = results
            
            return results
        
        # Apply caching optimization
        self.rag_system.retrieval.retrieve_similar_transactions = cached_retrieval
        
        optimization_results = {
            "component": "retrieval",
            "optimizations": [
                {
                    "name": "reduced_result_count",
                    "description": "Reduced default number of results from 30 to 20",
                    "original_value": original_n_results,
                    "new_value": optimized_n_results
                },
                {
                    "name": "query_caching",
                    "description": "Implemented in-memory caching for retrieval queries",
                    "cache_size": 0,  # Initially empty
                    "cache_type": "in-memory"
                }
            ]
        }
        
        self.optimizations_applied.append(optimization_results)
        logger.info(f"Applied retrieval optimizations: {len(optimization_results['optimizations'])}")
        
        return optimization_results
    
    def optimize_context_assembly(self) -> Dict[str, Any]:
        """Optimize the context assembly component
        
        Returns:
            Dict[str, Any]: Optimization results
        """
        logger.info("Optimizing context assembly")
        
        # 1. Optimize context window by summarizing transactions
        original_get_context = self.rag_system.get_relevant_context
        
        def summarized_context(query):
            """Context with transaction summarization"""
            # Get the original context
            context = original_get_context(query)
            
            # If the context is too large, summarize it
            if isinstance(context, str) and len(context) > 2000:
                # Simple summarization: take first part and last part
                first_part = context[:1000]
                last_part = context[-1000:]
                context = first_part + "\n...[Transaction details summarized]...\n" + last_part
            
            return context
        
        # Apply optimization
        self.rag_system.get_relevant_context = summarized_context
        
        # 2. Add smarter context filtering based on query type
        def improved_context_filter(query, context):
            """Filter context based on query type"""
            # Identify query type (spending, comparison, etc.)
            query_lower = query.lower()
            
            # For spending queries, prioritize amount information
            if "spend" in query_lower or "cost" in query_lower or "paid" in query_lower:
                # Simple filtering logic - actual implementation would be more sophisticated
                lines = context.split("\n")
                prioritized_lines = [
                    line for line in lines 
                    if "amount" in line.lower() or "$" in line or "total" in line.lower()
                ]
                
                # Add back some context if we filtered too much
                if len(prioritized_lines) < 5 and lines:
                    prioritized_lines = lines[:min(10, len(lines))]
                
                return "\n".join(prioritized_lines)
            
            # For comparison queries, ensure data from both periods is included
            if "compare" in query_lower or "difference" in query_lower:
                return context  # Keep all context for comparison queries
            
            # Default behavior - return original context
            return context
        
        # Store the filter for later use in the RAG system
        self.rag_system.context_filter = improved_context_filter
        
        optimization_results = {
            "component": "context_assembly",
            "optimizations": [
                {
                    "name": "transaction_summarization",
                    "description": "Implemented transaction summarization for large contexts",
                    "threshold_chars": 2000,
                    "summary_method": "truncation_with_indicator"
                },
                {
                    "name": "query_based_filtering",
                    "description": "Added smarter context filtering based on query type",
                    "filter_types": ["spending", "comparison"]
                }
            ]
        }
        
        self.optimizations_applied.append(optimization_results)
        logger.info(f"Applied context assembly optimizations: {len(optimization_results['optimizations'])}")
        
        return optimization_results
    
    def optimize_llm_parameters(self) -> Dict[str, Any]:
        """Optimize LLM parameters
        
        Returns:
            Dict[str, Any]: Optimization results
        """
        logger.info("Optimizing LLM parameters")
        
        # Store original parameters
        original_temperature = self.rag_system.llm.temperature
        
        # Optimize temperature for more consistent responses
        self.rag_system.llm.temperature = 0.1
        
        # Add response caching
        if not hasattr(self.rag_system.llm, 'response_cache'):
            self.rag_system.llm.response_cache = {}
        
        # Store original generate_text method
        original_generate_text = self.rag_system.llm.generate_text
        
        # Define a new method with caching
        def cached_generate_text(prompt, **kwargs):
            """Cached LLM text generation"""
            # Create a cache key
            # In production, we might want to use a hash of the prompt
            cache_key = prompt[:100] + str(sorted(kwargs.items()))
            
            # Check cache
            if cache_key in self.rag_system.llm.response_cache:
                logger.info("Using cached LLM response")
                return self.rag_system.llm.response_cache[cache_key]
            
            # Generate response
            response = original_generate_text(prompt, **kwargs)
            
            # Cache response
            self.rag_system.llm.response_cache[cache_key] = response
            
            return response
        
        # Apply optimization
        self.rag_system.llm.generate_text = cached_generate_text
        
        optimization_results = {
            "component": "llm",
            "optimizations": [
                {
                    "name": "temperature_adjustment",
                    "description": "Lowered temperature for more consistent responses",
                    "original_value": original_temperature,
                    "new_value": self.rag_system.llm.temperature
                },
                {
                    "name": "response_caching",
                    "description": "Implemented response caching for identical prompts",
                    "cache_size": 0,  # Initially empty
                    "cache_type": "in-memory"
                }
            ]
        }
        
        self.optimizations_applied.append(optimization_results)
        logger.info(f"Applied LLM optimizations: {len(optimization_results['optimizations'])}")
        
        return optimization_results
    
    def apply_all_optimizations(self) -> Dict[str, Any]:
        """Apply all available optimizations
        
        Returns:
            Dict[str, Any]: Combined optimization results
        """
        logger.info("Applying all optimizations to the system")
        
        # Apply optimizations
        retrieval_optimizations = self.optimize_retrieval()
        context_optimizations = self.optimize_context_assembly()
        llm_optimizations = self.optimize_llm_parameters()
        
        # Combine results
        results = {
            "timestamp": time.time(),
            "system_id": id(self.rag_system),
            "optimization_summary": {
                "total_optimizations": (
                    len(retrieval_optimizations["optimizations"]) +
                    len(context_optimizations["optimizations"]) +
                    len(llm_optimizations["optimizations"])
                ),
                "components_optimized": 3
            },
            "optimizations": {
                "retrieval": retrieval_optimizations,
                "context_assembly": context_optimizations,
                "llm": llm_optimizations
            }
        }
        
        logger.info(f"Successfully applied {results['optimization_summary']['total_optimizations']} optimizations across "
                   f"{results['optimization_summary']['components_optimized']} components")
        
        return results
    
    def save_optimization_results(self, results: Dict[str, Any], output_file: str = "optimization_results.json"):
        """Save optimization results to a file
        
        Args:
            results (Dict[str, Any]): Optimization results
            output_file (str): Output file path
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Optimization results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving optimization results: {e}")


def main():
    """Apply optimizations to the RAG system"""
    logger.info("Starting system optimization")
    
    # Initialize optimizer
    optimizer = SystemOptimizer()
    
    # Apply all optimizations
    results = optimizer.apply_all_optimizations()
    
    # Save results
    optimizer.save_optimization_results(results, "optimization_results.json")
    
    logger.info("System optimization complete")


if __name__ == "__main__":
    main() 