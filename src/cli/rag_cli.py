#!/usr/bin/env python
import os
import sys
from pathlib import Path
import logging
import json
from typing import Optional, List, Tuple
from datetime import datetime
import argparse
import rich
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.progress import Progress

# Add the project root to path so we can import from other modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.rag.finance_rag import FinanceRAG
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("rag_cli.log"), logging.StreamHandler()]
)
logger = logging.getLogger("rag_cli")

# Rich console
console = Console()

def setup_rag_system(
    llm_provider: str = "openai",
    llm_model: Optional[str] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    temperature: float = 0.3
) -> FinanceRAG:
    """Initialize the RAG system
    
    Args:
        llm_provider (str): LLM provider (openai, anthropic, etc.)
        llm_model (str, optional): Specific model to use
        embedding_model (str): Embedding model name
        temperature (float): Temperature for LLM generation
        
    Returns:
        FinanceRAG: Initialized RAG system
    """
    with console.status("[bold green]Initializing FinanceRAG system..."):
        try:
            rag = FinanceRAG(
                llm_provider=llm_provider,
                llm_model=llm_model,
                embedding_model=embedding_model,
                temperature=temperature
            )
            return rag
        except Exception as e:
            console.print(f"[bold red]Error initializing RAG system: {e}")
            sys.exit(1)

def process_query(
    rag: FinanceRAG,
    query: str,
    category_filter: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None,
    return_context: bool = False
) -> None:
    """Process a user query and display the response
    
    Args:
        rag (FinanceRAG): RAG system instance
        query (str): User query
        category_filter (str, optional): Filter by category
        date_range (Tuple[str, str], optional): Filter by date range
        return_context (bool): Whether to display retrieved context
    """
    with console.status("[bold green]Processing query..."):
        try:
            # Get response from RAG system
            if return_context:
                response, context = rag.query(
                    user_query=query,
                    category_filter=category_filter,
                    date_range=date_range,
                    return_context=True
                )
            else:
                response = rag.query(
                    user_query=query,
                    category_filter=category_filter,
                    date_range=date_range,
                    return_context=False
                )
                
            # Display response
            console.print("\n[bold green]Response:[/bold green]")
            console.print(Panel(Markdown(response), border_style="green"))
            
            # Display context if requested
            if return_context:
                console.print("\n[bold blue]Retrieved Context:[/bold blue]")
                
                # Show transactions in a table
                if "retrieved_data" in context and "transactions" in context["retrieved_data"]:
                    transactions = context["retrieved_data"]["transactions"]
                    
                    if transactions:
                        table = Table(title="Retrieved Transactions")
                        table.add_column("Description", style="cyan", no_wrap=True)
                        table.add_column("Amount", style="green", justify="right")
                        table.add_column("Date", style="magenta")
                        table.add_column("Category", style="yellow")
                        table.add_column("Similarity", style="blue", justify="right")
                        
                        for tx in transactions:
                            amount = tx.get("amount", 0)
                            amount_str = f"${abs(amount):.2f}"
                            amount_style = "green" if amount >= 0 else "red"
                            
                            table.add_row(
                                tx.get("description", "Unknown"),
                                f"[{amount_style}]{amount_str}[/{amount_style}]",
                                tx.get("date", ""),
                                tx.get("category", "Uncategorized"),
                                f"{tx.get('similarity', 0):.2f}"
                            )
                            
                        console.print(table)
                
                # Show insights if available
                if "retrieved_data" in context and "insights" in context["retrieved_data"]:
                    insights = context["retrieved_data"]["insights"]
                    
                    console.print("\n[bold blue]Financial Insights:[/bold blue]")
                    if "total_spending" in insights:
                        console.print(f"Total Spending: ${insights['total_spending']:.2f}")
                    if "total_income" in insights:
                        console.print(f"Total Income: ${insights['total_income']:.2f}")
                    if "net_cash_flow" in insights:
                        net_flow = insights["net_cash_flow"]
                        flow_color = "green" if net_flow >= 0 else "red"
                        console.print(f"Net Cash Flow: [{flow_color}]${net_flow:.2f}[/{flow_color}]")
                    
                    # Show top categories
                    if "top_categories" in insights and insights["top_categories"]:
                        console.print("\n[bold yellow]Top Spending Categories:[/bold yellow]")
                        for cat in insights["top_categories"]:
                            console.print(f"- {cat['category']}: ${cat['amount']:.2f}")
            
        except Exception as e:
            console.print(f"[bold red]Error processing query: {e}")

def run_spending_analysis(rag: FinanceRAG, time_period: str, category: Optional[str] = None) -> None:
    """Run and display spending analysis
    
    Args:
        rag (FinanceRAG): RAG system instance
        time_period (str): Time period to analyze
        category (str, optional): Filter by category
    """
    with console.status(f"[bold green]Analyzing spending for {time_period}..."):
        try:
            analysis = rag.spending_analysis(
                time_period=time_period,
                category_filter=category
            )
            
            if "error" in analysis:
                console.print(f"[bold red]Error: {analysis['error']}")
                return
                
            # Display analysis
            console.print("\n[bold green]Spending Analysis:[/bold green]")
            console.print(Panel(Markdown(analysis["analysis"]), border_style="green"))
            
            # Display structured insights
            if "structured_insights" in analysis:
                insights = analysis["structured_insights"]
                
                if insights["categories"]:
                    console.print("\n[bold yellow]Categorized Spending:[/bold yellow]")
                    
                    table = Table()
                    table.add_column("Category", style="cyan")
                    table.add_column("Amount", style="green", justify="right")
                    
                    for cat in insights["categories"]:
                        if "amount" in cat:
                            table.add_row(
                                cat["category"],
                                f"${cat['amount']:.2f}"
                            )
                        else:
                            table.add_row(
                                cat["category"],
                                cat.get("description", "")
                            )
                            
                    console.print(table)
                
                if insights["recommendations"]:
                    console.print("\n[bold green]Recommendations:[/bold green]")
                    for i, rec in enumerate(insights["recommendations"]):
                        console.print(f"{i+1}. {rec}")
            
        except Exception as e:
            console.print(f"[bold red]Error running spending analysis: {e}")

def categorize_sample_transactions(rag: FinanceRAG) -> None:
    """Categorize sample transactions
    
    Args:
        rag (FinanceRAG): RAG system instance
    """
    # Sample transactions
    transactions = [
        {
            "id": "tx_001",
            "description": "STARBUCKS STORE #12345",
            "amount": -4.85,
            "date": "2023-01-15",
            "merchant": "Starbucks"
        },
        {
            "id": "tx_002",
            "description": "AMAZON.COM*AB12CD34E",
            "amount": -67.49,
            "date": "2023-01-16",
            "merchant": "Amazon"
        },
        {
            "id": "tx_003",
            "description": "UBER TRIP 0123456789",
            "amount": -24.35,
            "date": "2023-01-17",
            "merchant": "Uber"
        },
        {
            "id": "tx_004",
            "description": "NETFLIX.COM",
            "amount": -15.99,
            "date": "2023-01-18",
            "merchant": "Netflix"
        },
        {
            "id": "tx_005",
            "description": "DIRECT DEPOSIT - ACME CORP",
            "amount": 2500.00,
            "date": "2023-01-15",
            "merchant": "Acme Corp"
        }
    ]
    
    # Display original transactions
    console.print("[bold]Original Transactions:[/bold]")
    
    table = Table()
    table.add_column("ID", style="dim")
    table.add_column("Description", style="cyan")
    table.add_column("Amount", style="green", justify="right")
    table.add_column("Date", style="magenta")
    
    for tx in transactions:
        amount = tx["amount"]
        amount_str = f"${abs(amount):.2f}"
        amount_style = "green" if amount >= 0 else "red"
        
        table.add_row(
            tx["id"],
            tx["description"],
            f"[{amount_style}]{amount_str}[/{amount_style}]",
            tx["date"]
        )
        
    console.print(table)
    
    # Categorize transactions
    with console.status("[bold green]Categorizing transactions..."):
        categorized = rag.categorize_transactions(transactions)
    
    # Display categorized transactions
    console.print("\n[bold green]Categorized Transactions:[/bold green]")
    
    table = Table()
    table.add_column("ID", style="dim")
    table.add_column("Description", style="cyan")
    table.add_column("Amount", style="green", justify="right")
    table.add_column("Category", style="yellow")
    
    for tx in categorized:
        amount = tx["amount"]
        amount_str = f"${abs(amount):.2f}"
        amount_style = "green" if amount >= 0 else "red"
        
        table.add_row(
            tx["id"],
            tx["description"],
            f"[{amount_style}]{amount_str}[/{amount_style}]",
            tx["category"]
        )
        
    console.print(table)

def find_similar_transactions(rag: FinanceRAG, description: str) -> None:
    """Find and display transactions similar to a description
    
    Args:
        rag (FinanceRAG): RAG system instance
        description (str): Transaction description to match
    """
    with console.status(f"[bold green]Finding similar transactions for: {description}..."):
        results = rag.find_similar_transactions(
            transaction_description=description,
            n_results=5
        )
        
    if not results["count"]:
        console.print("[yellow]No similar transactions found.")
        return
        
    # Display results
    console.print(f"\n[bold green]Found {results['count']} similar transactions:[/bold green]")
    
    table = Table()
    table.add_column("Description", style="cyan", no_wrap=True)
    table.add_column("Amount", style="green", justify="right")
    table.add_column("Date", style="magenta")
    table.add_column("Category", style="yellow")
    table.add_column("Similarity", style="blue", justify="right")
    
    for tx in results["similar_transactions"]:
        amount = tx.get("amount", 0)
        amount_str = f"${abs(amount):.2f}"
        amount_style = "green" if amount >= 0 else "red"
        
        table.add_row(
            tx.get("description", "Unknown"),
            f"[{amount_style}]{amount_str}[/{amount_style}]",
            tx.get("date", ""),
            tx.get("category", "Uncategorized"),
            f"{tx.get('similarity', 0):.2f}"
        )
        
    console.print(table)
    
    # Display analysis
    console.print("\n[bold green]Analysis:[/bold green]")
    console.print(Panel(Markdown(results["analysis"]), border_style="green"))

def interactive_mode(rag: FinanceRAG) -> None:
    """Run interactive CLI mode
    
    Args:
        rag (FinanceRAG): RAG system instance
    """
    console.print("""
[bold green]Finance RAG Interactive CLI[/bold green]
Type your financial questions or use these commands:
- [bold blue]/help[/bold blue]: Show this help message
- [bold blue]/analyze [period][/bold blue]: Run spending analysis (e.g. '/analyze last month')
- [bold blue]/similar [description][/bold blue]: Find similar transactions (e.g. '/similar coffee')
- [bold blue]/categorize[/bold blue]: Categorize sample transactions
- [bold blue]/clear[/bold blue]: Clear conversation history
- [bold blue]/exit[/bold blue]: Exit the program
    """)
    
    while True:
        try:
            user_input = Prompt.ask("\n[bold blue]Ask a question[/bold blue]")
            
            if user_input.lower() == "/exit":
                console.print("[yellow]Exiting...")
                break
                
            elif user_input.lower() == "/help":
                console.print("""
[bold green]Commands:[/bold green]
- [bold blue]/help[/bold blue]: Show this help message
- [bold blue]/analyze [period][/bold blue]: Run spending analysis (e.g. '/analyze last month')
- [bold blue]/similar [description][/bold blue]: Find similar transactions (e.g. '/similar coffee')
- [bold blue]/categorize[/bold blue]: Categorize sample transactions
- [bold blue]/clear[/bold blue]: Clear conversation history
- [bold blue]/exit[/bold blue]: Exit the program
                """)
                
            elif user_input.lower() == "/clear":
                rag.clear_conversation_history()
                console.print("[green]Conversation history cleared.")
                
            elif user_input.lower() == "/categorize":
                categorize_sample_transactions(rag)
                
            elif user_input.lower().startswith("/analyze"):
                parts = user_input.split(" ", 1)
                time_period = "this month"  # Default
                
                if len(parts) > 1:
                    time_period = parts[1].strip()
                    
                run_spending_analysis(rag, time_period)
                
            elif user_input.lower().startswith("/similar"):
                parts = user_input.split(" ", 1)
                
                if len(parts) > 1:
                    description = parts[1].strip()
                    find_similar_transactions(rag, description)
                else:
                    console.print("[yellow]Please provide a description to match.")
                
            else:
                # Regular query
                show_context = False
                process_query(rag, user_input, return_context=show_context)
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting...")
            break
        except Exception as e:
            console.print(f"[bold red]Error: {e}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Finance RAG CLI")
    
    # Main operation mode
    parser.add_argument(
        "query", nargs="?", default=None,
        help="Query to process (if not provided, interactive mode is started)"
    )
    
    # Model configuration
    parser.add_argument(
        "--provider", type=str, default="openai",
        help="LLM provider (openai, anthropic, huggingface, local)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Specific LLM model to use"
    )
    parser.add_argument(
        "--embedding-model", type=str, default="all-MiniLM-L6-v2",
        help="Embedding model to use"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="Temperature for LLM generation (0.0-1.0)"
    )
    
    # Query filters
    parser.add_argument(
        "--category", type=str, default=None,
        help="Filter by category"
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="Start date filter (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="End date filter (YYYY-MM-DD)"
    )
    
    # Special commands
    parser.add_argument(
        "--analyze", type=str, default=None,
        help="Run spending analysis for a time period (e.g. 'last month')"
    )
    parser.add_argument(
        "--categorize", action="store_true",
        help="Categorize sample transactions"
    )
    parser.add_argument(
        "--similar", type=str, default=None,
        help="Find transactions similar to description"
    )
    
    # Other options
    parser.add_argument(
        "--show-context", action="store_true",
        help="Show retrieved context along with the response"
    )
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    
    try:
        # Initialize RAG system
        rag = setup_rag_system(
            llm_provider=args.provider,
            llm_model=args.model,
            embedding_model=args.embedding_model,
            temperature=args.temperature
        )
        
        # Determine date range if provided
        date_range = None
        if args.start_date and args.end_date:
            date_range = (args.start_date, args.end_date)
        
        # Run in appropriate mode
        if args.categorize:
            categorize_sample_transactions(rag)
            
        elif args.analyze:
            run_spending_analysis(rag, args.analyze, args.category)
            
        elif args.similar:
            find_similar_transactions(rag, args.similar)
            
        elif args.query:
            process_query(
                rag,
                args.query,
                category_filter=args.category,
                date_range=date_range,
                return_context=args.show_context
            )
            
        else:
            # No command or query specified, run interactive mode
            interactive_mode(rag)
            
    except Exception as e:
        console.print(f"[bold red]Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 