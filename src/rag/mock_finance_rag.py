"""Mock implementation of FinanceRAG for cloud deployment"""
import logging
import random
import time
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mock_finance_rag")

class MockFinanceRAG:
    """Mock implementation of Finance RAG system that returns predefined responses"""
    
    def __init__(self):
        """Initialize the mock RAG system"""
        logger.info("Initializing Mock Finance RAG")
        self.queries = {}
        self._prepare_mock_responses()
        
    def _prepare_mock_responses(self):
        """Prepare a dictionary of mock responses for common query types"""
        self.responses = {
            "expense_summary": "Based on your transaction history, your largest expenses last month were:\n\n"
                              "1. Housing: $1,850.00\n"
                              "2. Groceries: $580.25\n"
                              "3. Utilities: $320.45\n"
                              "4. Dining out: $285.75\n"
                              "5. Transportation: $210.30\n\n"
                              "Your total spending was $3,245.75 across 37 transactions.",
                              
            "budget_advice": "Based on your income and spending patterns, here's a recommended monthly budget:\n\n"
                           "- Housing: 30% ($1,800)\n"
                           "- Transportation: 10% ($600)\n"
                           "- Food: 15% ($900)\n"
                           "- Utilities: 5% ($300)\n"
                           "- Insurance: 10% ($600)\n"
                           "- Savings: 15% ($900)\n"
                           "- Discretionary: 15% ($900)\n\n"
                           "You're currently over-spending in the dining out category by approximately 20%.",
                           
            "investment": "Based on your financial profile and goals, consider:\n\n"
                         "1. Increasing your 401(k) contribution to at least the employer match (currently 5%)\n"
                         "2. Building an emergency fund to cover 3-6 months of expenses\n"
                         "3. Paying down high-interest debt before further investments\n"
                         "4. Exploring index funds for long-term growth\n\n"
                         "Your current investment allocation is fairly conservative. Given your age and time horizon, you might consider increasing your equity exposure.",
                         
            "transaction_category": "I've analyzed your transaction and can categorize it as follows:\n\n"
                                   "• Transaction: 'AMAZON.COM' - $45.99\n"
                                   "• Category: Shopping\n"
                                   "• Subcategory: Online Retail\n\n"
                                   "This appears to be a discretionary purchase. Your monthly spending in this category is $230.45 so far, which is within your budget of $300.",
                                   
            "savings": "Based on your transaction history, here are some opportunities to save money:\n\n"
                      "1. You're spending $14.99/month on a streaming service you haven't used in 2 months\n"
                      "2. Your grocery spending could be reduced by ~$120/month by shopping at discount stores\n"
                      "3. You have multiple food delivery orders per week, totaling ~$85 in fees last month\n"
                      "4. Your monthly bank fees of $25 could be eliminated by switching to a no-fee account\n\n"
                      "By addressing these areas, you could save approximately $245 per month.",
                      
            "income_analysis": "I've analyzed your income patterns:\n\n"
                             "• Primary income source: Monthly salary of $5,000 (after tax)\n"
                             "• Secondary income: Approximately $750/month from freelance work\n"
                             "• Your income has increased by 5.2% compared to the same period last year\n"
                             "• Income stability: High (consistent monthly deposits)\n\n"
                             "Your current income-to-expense ratio is 1.35, which is healthy and allows for saving approximately $1,550 per month.",
                               
            "debt": "Your current debt profile:\n\n"
                   "• Student Loan: $15,500 remaining at 4.5% interest\n"
                   "• Credit Card: $2,750 at 19.99% interest\n"
                   "• Auto Loan: $8,200 remaining at 3.9% interest\n\n"
                   "Recommendation: Focus on paying off the high-interest credit card debt first, which is costing you approximately $550 per year in interest. By increasing your monthly payment from $150 to $300, you could eliminate this debt in 10 months instead of 22 months.",
                   
            "fallback": "I don't have enough information in your transaction history to answer that specific question. "
                       "However, I can help you analyze your spending patterns, budget allocation, or specific transactions "
                       "if you provide more details.\n\n"
                       "Would you like me to show a general financial summary based on your recent transactions instead?"
        }
        
    def process_query(self, query: str) -> str:
        """Process a financial query with mock responses
        
        Args:
            query (str): User query text
            
        Returns:
            str: Generated response
        """
        # Log the query
        logger.info(f"Processing query: {query}")
        
        # Add some artificial delay to simulate processing
        processing_time = random.uniform(0.5, 1.5)
        time.sleep(processing_time)
        
        # Determine the type of query to select the appropriate response
        query_lower = query.lower()
        
        # Simple keyword matching to find the most relevant response
        if any(word in query_lower for word in ["expense", "spend", "cost", "payment", "pay", "paid"]):
            return self.responses["expense_summary"]
            
        elif any(word in query_lower for word in ["budget", "plan", "allocate", "allocation"]):
            return self.responses["budget_advice"]
            
        elif any(word in query_lower for word in ["invest", "investment", "stock", "bond", "portfolio", "401k", "retirement"]):
            return self.responses["investment"]
            
        elif any(word in query_lower for word in ["categor", "classif", "group", "sort"]):
            return self.responses["transaction_category"]
            
        elif any(word in query_lower for word in ["save", "saving", "reduce", "cut", "lower"]):
            return self.responses["savings"]
            
        elif any(word in query_lower for word in ["income", "earn", "salary", "wage", "deposit"]):
            return self.responses["income_analysis"]
            
        elif any(word in query_lower for word in ["debt", "loan", "credit", "interest", "borrow", "mortgage"]):
            return self.responses["debt"]
            
        else:
            # Default fallback response
            return self.responses["fallback"]
            
    def categorize_transactions(self, transactions: List[Dict], available_categories: List[str] = None) -> List[Dict]:
        """Mock implementation of transaction categorization
        
        Args:
            transactions (List[Dict]): List of transaction dictionaries
            available_categories (List[str], optional): List of available categories
            
        Returns:
            List[Dict]: Transactions with categories assigned
        """
        # Example categories
        if not available_categories:
            available_categories = ["Housing", "Transportation", "Food", "Utilities", 
                                    "Healthcare", "Entertainment", "Shopping", "Personal",
                                    "Debt", "Income", "Savings", "Other"]
        
        # Add mock categories to transactions
        categorized = []
        for tx in transactions:
            tx_copy = tx.copy()
            
            # Simple keyword-based mock categorization
            desc = tx_copy.get("description", "").lower()
            
            if any(word in desc for word in ["rent", "mortgage", "hoa", "housing", "apartment"]):
                tx_copy["category"] = "Housing"
            elif any(word in desc for word in ["car", "gas", "fuel", "uber", "lyft", "transit", "parking"]):
                tx_copy["category"] = "Transportation"
            elif any(word in desc for word in ["grocery", "restaurant", "cafe", "food", "doordash", "grubhub"]):
                tx_copy["category"] = "Food"
            elif any(word in desc for word in ["electric", "water", "gas", "internet", "phone", "utility"]):
                tx_copy["category"] = "Utilities"
            elif any(word in desc for word in ["doctor", "hospital", "pharmacy", "medical", "health"]):
                tx_copy["category"] = "Healthcare"
            elif any(word in desc for word in ["movie", "game", "netflix", "spotify", "entertainment"]):
                tx_copy["category"] = "Entertainment"
            elif any(word in desc for word in ["amazon", "walmart", "target", "shop", "store"]):
                tx_copy["category"] = "Shopping"
            elif any(word in desc for word in ["loan", "credit", "debt", "payment"]):
                tx_copy["category"] = "Debt"
            elif any(word in desc for word in ["salary", "payroll", "deposit", "income"]):
                tx_copy["category"] = "Income"
            else:
                tx_copy["category"] = "Other"
                
            categorized.append(tx_copy)
            
        return categorized
            
    def find_similar_transactions(self, transaction_description: str, limit: int = 5) -> Dict:
        """Mock implementation to find similar transactions
        
        Args:
            transaction_description (str): Description of the transaction
            limit (int): Maximum number of results to return
            
        Returns:
            Dict: Results containing similar transactions
        """
        # Generate mock similar transactions
        mock_transactions = [
            {
                "date": "2023-05-15",
                "description": "Amazon.com",
                "amount": -45.99,
                "category": "Shopping",
                "similarity": 0.95
            },
            {
                "date": "2023-04-30",
                "description": "Amazon Prime",
                "amount": -14.99,
                "category": "Entertainment",
                "similarity": 0.82
            },
            {
                "date": "2023-04-02",
                "description": "Amazon Digital",
                "amount": -9.99,
                "category": "Entertainment",
                "similarity": 0.78
            },
            {
                "date": "2023-03-21",
                "description": "AMZN Marketplace",
                "amount": -29.95,
                "category": "Shopping",
                "similarity": 0.72
            },
            {
                "date": "2023-02-08",
                "description": "Amazon Returns",
                "amount": 15.49,
                "category": "Shopping",
                "similarity": 0.68
            }
        ]
        
        # Limit the results
        limited_results = mock_transactions[:limit]
        
        return {
            "query": transaction_description,
            "count": len(limited_results),
            "transactions": limited_results,
            "analysis": f"I found {len(limited_results)} transactions similar to '{transaction_description}'. The most frequent category is Shopping, with an average spend of $25.49 per transaction.",
            "llm_response": f"These transactions appear to be related to online shopping, primarily from Amazon. The spending pattern shows regular purchases approximately once per month, with occasional returns or credits."
        }
        
    def budget_recommendation(self, monthly_income: Optional[float] = 5000.0, time_period: str = "last month") -> Dict:
        """Mock implementation for budget recommendations
        
        Args:
            monthly_income (float, optional): Monthly income amount
            time_period (str): Time period to analyze
            
        Returns:
            Dict: Budget recommendation results
        """
        # Mock budget recommendations
        mock_budget = {
            "income": monthly_income,
            "time_period": time_period,
            "total_expenses": 3245.75,
            "savings_rate": 0.15,
            "categories": [
                {"category": "Housing", "current_amount": 1850.00, "recommended_percent": 0.30, "recommended_amount": 1500.00, "status": "over"},
                {"category": "Transportation", "current_amount": 210.30, "recommended_percent": 0.10, "recommended_amount": 500.00, "status": "under"},
                {"category": "Food", "current_amount": 865.00, "recommended_percent": 0.15, "recommended_amount": 750.00, "status": "over"},
                {"category": "Utilities", "current_amount": 320.45, "recommended_percent": 0.06, "recommended_amount": 300.00, "status": "on-target"},
                {"category": "Healthcare", "current_amount": 150.00, "recommended_percent": 0.05, "recommended_amount": 250.00, "status": "under"},
                {"category": "Entertainment", "current_amount": 285.75, "recommended_percent": 0.05, "recommended_amount": 250.00, "status": "over"},
                {"category": "Shopping", "current_amount": 180.50, "recommended_percent": 0.10, "recommended_amount": 500.00, "status": "under"},
                {"category": "Savings", "current_amount": 750.00, "recommended_percent": 0.15, "recommended_amount": 750.00, "status": "on-target"},
                {"category": "Other", "current_amount": 120.00, "recommended_percent": 0.04, "recommended_amount": 200.00, "status": "under"}
            ],
            "recommendations": [
                "Reduce housing costs by $350/month to meet the 30% guideline",
                "Consider allocating the excess $290 from transportation to savings",
                "Cut food expenses by $115/month by reducing dining out",
                "Your current savings rate is on target at 15% of income"
            ],
            "summary": "Your current spending pattern shows you're over-allocating to housing and food categories while under-utilizing your transportation budget. By making the recommended adjustments, you could increase your monthly savings by $455."
        }
        
        return mock_budget 