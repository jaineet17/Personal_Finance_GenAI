import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to the path so we can import from other modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.embedding.vector_store import VectorStore
from src.llm.ollama_service import OllamaService

load_dotenv()

class RAGEngine:
    def __init__(self, model_name="llama3:8b", collection_name="finance_transactions"):
        """Initialize the RAG engine

        Args:
            model_name (str): Name of the LLM model to use
            collection_name (str): Name of the vector collection
        """
        self.vector_store = VectorStore()
        self.llm_service = OllamaService(model_name=model_name)
        self.collection_name = collection_name

    def retrieve_context(self, query, n_results=5):
        """Retrieve relevant context for the query

        Args:
            query (str): The user query
            n_results (int): Number of relevant documents to retrieve

        Returns:
            str: Formatted context from retrieved documents
        """
        try:
            results = self.vector_store.query_similar(
                self.collection_name, 
                query, 
                n_results=n_results
            )

            # Format the results into a context string
            context = "Relevant financial transactions:\n\n"

            if results and "documents" in results:
                documents = results["documents"][0]
                metadatas = results.get("metadatas", [[{}] * len(documents)])[0]

                for i, (doc, meta) in enumerate(zip(documents, metadatas)):
                    date = meta.get("date", "Unknown date")
                    amount = meta.get("amount", "Unknown amount")
                    category = meta.get("category", "Uncategorized")

                    context += f"{i+1}. Date: {date}, Amount: {amount}, Category: {category}\n"
                    context += f"   Description: {doc}\n\n"
            else:
                context += "No relevant transactions found.\n"

            return context
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return "Error retrieving context from the vector database."

    def generate_response(self, query, system_prompt=None):
        """Generate a response using RAG

        Args:
            query (str): The user query
            system_prompt (str, optional): System prompt for the LLM

        Returns:
            str: The generated response
        """
        # Default system prompt for financial assistant
        if system_prompt is None:
            system_prompt = """You are a helpful financial assistant. Use the provided context about the user's 
            financial transactions to answer their questions accurately. Only make statements that are supported 
            by the given context. If you don't know something, say so instead of making up information."""

        # Retrieve relevant context
        context = self.retrieve_context(query)

        # Create the augmented prompt
        augmented_prompt = f"""Context information about the user's finances:
{context}

Based on the above context, please answer the following question:
{query}"""

        # Generate the response
        response = self.llm_service.generate(
            prompt=augmented_prompt,
            system_prompt=system_prompt
        )

        return response 