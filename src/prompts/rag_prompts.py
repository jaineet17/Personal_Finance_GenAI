import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Prompt template for the system message
RAG_SYSTEM_PROMPT = """You are an AI financial assistant with expertise in personal finance.
Your goal is to help the user understand their financial data and provide insights.
Always base your responses on the provided context.
If you don't know something or the information is not in the context, admit that you don't know.
Be precise and helpful. Format currency values appropriately.

Current date: {current_date}
"""

# Prompt for generating context-aware responses to financial queries
FINANCIAL_QUERY_PROMPT = """
The user has asked: {user_query}

I'll provide you with relevant financial context to help answer this query:

{context}

Based ONLY on this context, please provide a helpful response to the user's query.
If the context doesn't contain enough information to answer, acknowledge the limitations.
Do not make up information that is not present in the context.

Your response should be clear, concise and focus on answering the user's question.
"""

# Prompt for transaction categorization
TRANSACTION_CATEGORIZATION_PROMPT = """
Here's a list of transactions that need to be categorized:

{transactions}

Based on the descriptions, please categorize these transactions into the most appropriate categories.
Here are the available categories:
{available_categories}

For each transaction, provide:
1. The transaction ID
2. The appropriate category
3. A brief explanation for why you chose that category

Only use the categories provided. If you're uncertain, select the best match based on the transaction description.
"""

# Prompt for spending analysis
SPENDING_ANALYSIS_PROMPT = """
Here is the financial spending data for the specified period:

{spending_data}

Please analyze this data and provide insights on:
1. Overall spending patterns
2. Top spending categories and their percentages of total spending
3. Comparison to previous periods if available
4. Potential areas to reduce spending
5. Any unusual or one-time expenses

Focus on actionable insights that would help the user manage their finances better.
"""

# Prompt for generating financial advice
FINANCIAL_ADVICE_PROMPT = """
Based on the following financial data:

{financial_data}

The user has asked for advice on: {advice_request}

Please provide personalized financial advice that addresses the user's request.
Consider:
1. The user's income and spending patterns
2. Any debt or savings information available
3. Financial goals if mentioned
4. Best practices for the specific financial situation

Make your advice specific, actionable, and tailored to the data provided.
"""

# Prompt for budget recommendations
BUDGET_RECOMMENDATION_PROMPT = """
Based on the following financial data:

{financial_data}

Please recommend a budget that:
1. Allocates income appropriately across essential and discretionary categories
2. Follows the 50/30/20 rule (50% needs, 30% wants, 20% savings) or another appropriate budgeting method
3. Accounts for the user's specific spending patterns
4. Identifies areas where spending could be reduced

Format the budget clearly with category names and recommended monthly amounts.
Provide a brief explanation for your recommendations.
"""

# Prompt for finding similar transactions
SIMILAR_TRANSACTIONS_PROMPT = """
The user is looking for transactions similar to: {transaction_description}

Here are potentially similar transactions I found:

{similar_transactions}

Please analyze these transactions and explain:
1. How they are similar to the user's query
2. Any patterns or trends you notice
3. Total spent on this type of transaction (if relevant)
4. Any recommendations based on these transactions

Focus on the most relevant transactions in your analysis.
"""

# Prompt for debt reduction planning
DEBT_REDUCTION_PROMPT = """
Here is information about the user's current debts:

{debt_information}

The user has asked for help with debt reduction. Please provide:
1. A prioritized debt repayment plan (considering interest rates and balances)
2. Estimated timeline for becoming debt-free with different payment strategies
3. Recommendations for freeing up money to put toward debt
4. Any relevant financial advice for their specific situation

Make your plan realistic and actionable based on the user's financial situation.
"""

# Prompt for investment recommendations
INVESTMENT_RECOMMENDATION_PROMPT = """
Based on the following information:

{financial_profile}

The user has asked for investment recommendations. Please provide:
1. Appropriate investment options based on their goals and risk tolerance
2. Suggested asset allocation
3. Considerations for their current financial situation
4. General investment principles relevant to their query

Remember to explain that this is general advice and not specific financial recommendations,
as those would require a full financial advisory relationship.
"""

# Function to prepare context for the RAG system
def prepare_financial_context(retrieval_results: Any, 
                             spending_insights: Optional[Dict] = None,
                             time_period: Optional[str] = None,
                             category: Optional[str] = None,
                             include_transactions: bool = True,
                             include_categories: bool = True,
                             include_insights: bool = True) -> str:
    """Format retrieved data into a context string for the LLM

    Args:
        retrieval_results: Results from the retrieval system (can be Dict or List[Dict])
        spending_insights: Optional spending insights data
        time_period: Optional time period string
        category: Optional category filter
        include_transactions: Whether to include transaction data
        include_categories: Whether to include category data
        include_insights: Whether to include financial insights

    Returns:
        Formatted context string
    """
    context_parts = []

    # Add time period information if provided
    if time_period:
        context_parts.append(f"TIME PERIOD: {time_period}")

    # Add category filter information if provided
    if category:
        context_parts.append(f"CATEGORY FILTER: {category}")

    # Handle different formats of retrieval_results
    transactions = []

    # Case 1: retrieval_results is a Dict with transaction data
    if isinstance(retrieval_results, dict):
        if "transactions" in retrieval_results:
            transactions = retrieval_results["transactions"]
        elif "metadatas" in retrieval_results and retrieval_results["metadatas"]:
            # Convert metadatas format to transaction list
            for i, metadata in enumerate(retrieval_results["metadatas"][0]):
                transaction = dict(metadata)
                if "documents" in retrieval_results and retrieval_results["documents"] and i < len(retrieval_results["documents"][0]):
                    transaction["text"] = retrieval_results["documents"][0][i]
                transactions.append(transaction)
        # Add date range info if available 
        if "date_range" in retrieval_results:
            date_range = retrieval_results["date_range"]
            if isinstance(date_range, dict) and "start" in date_range and "end" in date_range:
                start_date = date_range["start"]
                end_date = date_range["end"]
                context_parts.append(f"DATA TIME RANGE: {start_date} to {end_date}")

                # Extract month/year for clarity
                try:
                    start_month_year = "-".join(start_date.split("-")[:2])
                    end_month_year = "-".join(end_date.split("-")[:2])
                    if start_month_year == end_month_year:
                        context_parts.append(f"MONTH/YEAR: {start_month_year}")

                        # Add month name for better readability
                        month_names = {
                            "01": "January", "02": "February", "03": "March", "04": "April",
                            "05": "May", "06": "June", "07": "July", "08": "August",
                            "09": "September", "10": "October", "11": "November", "12": "December"
                        }
                        month_num = start_month_year.split("-")[1]
                        year = start_month_year.split("-")[0]
                        if month_num in month_names:
                            context_parts.append(f"TIME PERIOD: {month_names[month_num]} {year}")
                    else:
                        context_parts.append(f"PERIOD: {start_month_year} to {end_month_year}")
                except:
                    pass

                context_parts.append("")
    # Case 2: retrieval_results is a List[Dict] of transactions
    elif isinstance(retrieval_results, list):
        transactions = retrieval_results

    # Add transaction information
    if include_transactions and transactions:
        # Add transaction count
        context_parts.append(f"TOTAL TRANSACTIONS: {len(transactions)}")
        context_parts.append("")

        # Format transactions in a readable way
        context_parts.append("TRANSACTIONS:")

        # Define a function to format a single transaction
        def format_transaction(tx):
            tx_parts = []
            # Format date if available
            if "date" in tx:
                tx_parts.append(f"Date: {tx['date']}")

            # Format description if available
            if "description" in tx:
                tx_parts.append(f"Description: {tx['description']}")

            # Format amount if available (ensure it's shown as currency)
            if "amount" in tx:
                # Make amount negative to represent spending (if not already)
                amount = tx["amount"]
                if isinstance(amount, (int, float)):
                    amount_str = f"${abs(amount):.2f}"
                    if amount >= 0:
                        tx_parts.append(f"Amount: +{amount_str}")
                    else:
                        tx_parts.append(f"Amount: -{amount_str}")
                else:
                    tx_parts.append(f"Amount: {amount}")

            # Format category if available
            if "category" in tx:
                tx_parts.append(f"Category: {tx['category']}")

            # Format merchant if available
            if "merchant" in tx:
                tx_parts.append(f"Merchant: {tx['merchant']}")

            # Return formatted transaction
            return ", ".join(tx_parts)

        # Format and add each transaction (limit to first 20 for context size)
        for i, tx in enumerate(transactions[:20]):
            context_parts.append(f"{i+1}. {format_transaction(tx)}")

        # Add note if transactions were truncated
        if len(transactions) > 20:
            context_parts.append(f"... and {len(transactions) - 20} more transactions (truncated)")

        context_parts.append("")

    # Add spending insights if available
    if include_insights and spending_insights:
        context_parts.append("SPENDING INSIGHTS:")

        # Add total spending
        if "total_spending" in spending_insights:
            total = spending_insights["total_spending"]
            context_parts.append(f"Total Spending: ${abs(total):.2f}")

        # Add category breakdown
        if "category_breakdown" in spending_insights:
            context_parts.append("Category Breakdown:")
            categories = spending_insights["category_breakdown"]
            for category, amount in categories.items():
                context_parts.append(f"  {category}: ${abs(amount):.2f}")

        # Add month-over-month comparison
        if "month_comparison" in spending_insights:
            context_parts.append("Month-over-Month Comparison:")
            comparison = spending_insights["month_comparison"]
            for month, amount in comparison.items():
                context_parts.append(f"  {month}: ${abs(amount):.2f}")

        context_parts.append("")

    # If no data was added, add a note
    if len(context_parts) == 0:
        context_parts.append("No financial data available for this query.")

    # Join all parts with newlines and return
    return "\n".join(context_parts)

# Function to format transactions for categorization
def format_transactions_for_categorization(transactions: List[Dict]) -> str:
    """Format transaction data for the categorization prompt

    Args:
        transactions: List of transaction dictionaries

    Returns:
        Formatted transaction string
    """
    transaction_strings = []

    for i, tx in enumerate(transactions):
        tx_id = tx.get("id", f"tx_{i}")
        description = tx.get("description", "Unknown transaction")
        amount = tx.get("amount", 0)
        date = tx.get("date", "")
        merchant = tx.get("merchant", "")

        tx_str = f"Transaction ID: {tx_id}\n"
        tx_str += f"Description: {description}\n"
        tx_str += f"Amount: ${abs(amount):.2f}"
        if amount < 0:
            tx_str += " (expense)"
        else:
            tx_str += " (income)"
        tx_str += f"\nDate: {date}"
        if merchant:
            tx_str += f"\nMerchant: {merchant}"
        tx_str += "\n"

        transaction_strings.append(tx_str)

    return "\n".join(transaction_strings)

# Function to extract insights from LLM response
def extract_financial_insights(llm_response: str) -> Dict:
    """Parse LLM response to extract structured financial insights

    Args:
        llm_response: Raw response from LLM

    Returns:
        Dictionary of extracted insights
    """
    insights = {
        "categories": [],
        "recommendations": [],
        "summary": ""
    }

    # Extract categories (looking for bullet points or numbered lists)
    category_lines = []
    in_category_section = False

    for line in llm_response.split('\n'):
        line = line.strip()

        if not line:
            continue

        # Look for category section headers
        if "categor" in line.lower() and ":" in line:
            in_category_section = True
            continue

        # Look for recommendation section headers
        if "recommend" in line.lower() and ":" in line:
            in_category_section = False

        # Collect category lines
        if in_category_section and (line.startswith('-') or line.startswith('•') or 
                                  (line[0].isdigit() and line[1:2] in ['.', ')'])):
            category_lines.append(line)

    # Extract categories
    for line in category_lines:
        # Remove bullets/numbers
        if line.startswith('-') or line.startswith('•'):
            category = line[1:].strip()
        elif line[0].isdigit() and line[1:2] in ['.', ')']:
            category = line[2:].strip()
        else:
            category = line.strip()

        # Extract category and amount if present
        parts = category.split(':')
        if len(parts) > 1:
            cat_name = parts[0].strip()
            # Try to extract amount
            amount_str = parts[1].strip()
            try:
                # Find dollar amount
                import re
                amount_match = re.search(r'\$\s*([\d,]+(\.\d{2})?)', amount_str)
                if amount_match:
                    amount = float(amount_match.group(1).replace(',', ''))
                    insights["categories"].append({"category": cat_name, "amount": amount})
                else:
                    insights["categories"].append({"category": cat_name, "description": amount_str})
            except:
                insights["categories"].append({"category": cat_name, "description": amount_str})
        else:
            insights["categories"].append({"category": category})

    # Extract recommendations (similar approach)
    recommendation_lines = []
    in_recommendation_section = False

    for line in llm_response.split('\n'):
        line = line.strip()

        if not line:
            continue

        if "recommend" in line.lower() and ":" in line:
            in_recommendation_section = True
            continue

        if in_recommendation_section and (line.startswith('-') or line.startswith('•') or 
                                       (line[0].isdigit() and line[1:2] in ['.', ')'])):
            recommendation_lines.append(line)

    # Process recommendations
    for line in recommendation_lines:
        if line.startswith('-') or line.startswith('•'):
            rec = line[1:].strip()
        elif line[0].isdigit() and line[1:2] in ['.', ')']:
            rec = line[2:].strip()
        else:
            rec = line.strip()

        insights["recommendations"].append(rec)

    # Extract summary (first paragraph or conclusion section)
    summary_lines = []
    for line in llm_response.split('\n'):
        if line.strip() and not line.startswith('-') and not line.startswith('•'):
            summary_lines.append(line.strip())
            if len(summary_lines) == 1:  # Just get the first paragraph
                break

    insights["summary"] = " ".join(summary_lines)

    return insights


# Function to get the current formatted date
def get_current_date_formatted() -> str:
    """Get current date in a formatted string"""
    return datetime.now().strftime("%Y-%m-%d")

# Save formatted prompts to a JSON file
def save_prompts_to_file(output_path: str = None) -> None:
    """Save all prompts to a JSON file for easy access

    Args:
        output_path: Path to save the prompts, defaults to prompts directory
    """
    if output_path is None:
        # Get the directory of this file
        dir_path = Path(__file__).parent
        output_path = dir_path / "finance_prompts.json"

    prompt_dict = {
        "system": RAG_SYSTEM_PROMPT,
        "financial_query": FINANCIAL_QUERY_PROMPT,
        "transaction_categorization": TRANSACTION_CATEGORIZATION_PROMPT,
        "spending_analysis": SPENDING_ANALYSIS_PROMPT,
        "financial_advice": FINANCIAL_ADVICE_PROMPT,
        "budget_recommendation": BUDGET_RECOMMENDATION_PROMPT,
        "similar_transactions": SIMILAR_TRANSACTIONS_PROMPT,
        "debt_reduction": DEBT_REDUCTION_PROMPT,
        "investment_recommendation": INVESTMENT_RECOMMENDATION_PROMPT,
    }

    with open(output_path, 'w') as f:
        json.dump(prompt_dict, f, indent=2)

    print(f"Prompts saved to {output_path}")

def format_finance_rag_prompt(query: str, context: str, conversation_history: str = "") -> str:
    """Format RAG prompt for finance queries

    Args:
        query (str): User query
        context (str): Financial context
        conversation_history (str): Previous conversation

    Returns:
        str: Formatted prompt
    """
    system_part = """You are a helpful AI financial assistant that answers questions about personal finances.
Use the information below to answer the user's query. 
If you can't find the specific answer in the provided information, say so rather than making something up.
Be specific and provide exact amounts when available. Current date: ${current_date}
"""

    # Get current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    system_part = system_part.replace("${current_date}", current_date)

    # Include conversation history if provided
    history_part = ""
    if conversation_history:
        history_part = f"\nCONVERSATION HISTORY:\n{conversation_history}\n"

    # Format the main prompt
    prompt = f"""{system_part}

{history_part}
FINANCIAL CONTEXT:
{context}

USER QUERY:
{query}

ANSWER:
"""

    return prompt

if __name__ == "__main__":
    # Example usage
    print("Finance RAG Prompts Module")
    save_prompts_to_file()

    # Example of formatting transactions
    example_transactions = [
        {
            "id": "tx_001",
            "description": "Coffee at Starbucks",
            "amount": -4.50,
            "date": "2024-01-15",
            "merchant": "Starbucks"
        },
        {
            "id": "tx_002",
            "description": "Monthly rent payment",
            "amount": -1500.00,
            "date": "2024-01-01",
            "merchant": "Property Management"
        }
    ]

    print("\nExample of formatted transactions for categorization:")
    print(format_transactions_for_categorization(example_transactions)) 