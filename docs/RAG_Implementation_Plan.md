# Zero-Budget RAG Enhancement Implementation Plan

This action plan outlines practical steps to enhance the Finance RAG application using only free resources, specifically focusing on Huggingface and Ollama LLMs. The plan is divided into phases with increasing complexity.

## Phase 1: Foundational Improvements (Weeks 1-2)

### 1. Enhanced Retrieval with BM25
**Goal**: Implement hybrid search combining vector similarity with lexical matching.
- Integrate the `rank_bm25` Python package (zero cost)
- Implement parallel search: vector similarity + BM25 lexical search
- Create a simple fusion algorithm to combine results

```python
# Example implementation
from rank_bm25 import BM25Okapi
import numpy as np

def hybrid_search(query, documents, embeddings, alpha=0.5):
    # BM25 search
    tokenized_docs = [doc.split() for doc in documents]
    tokenized_query = query.split()
    bm25 = BM25Okapi(tokenized_docs)
    bm25_scores = np.array(bm25.get_scores(tokenized_query))
    
    # Vector search (using existing ChromaDB)
    vector_scores = vector_search(query, embeddings)
    
    # Combine scores (simple weighted average)
    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores
    
    return sorted(range(len(combined_scores)), key=lambda i: combined_scores[i], reverse=True)
```

### 2. Query Preprocessing Enhancements
**Goal**: Improve query understanding for better retrieval.
- Implement basic financial entity extraction (regex-based for dates, amounts, merchants)
- Add financial query classification (time-based, category-based, analysis, etc.)
- Develop query expansion with financial terminology

```python
def extract_financial_entities(query):
    """Extract financial entities from query using regex patterns"""
    entities = {
        'dates': extract_dates(query),
        'amounts': extract_amounts(query),
        'merchants': extract_merchants(query),
        'categories': extract_categories(query)
    }
    return entities

def expand_financial_query(query, entities):
    """Add relevant financial terms based on extracted entities"""
    expanded_query = query
    # Add category synonyms
    if entities['categories']:
        synonyms = get_category_synonyms(entities['categories'])
        expanded_query += " " + " ".join(synonyms)
    return expanded_query
```

### 3. Prompt Template Optimization
**Goal**: Create specialized prompts for different financial query types.
- Develop a template library for different financial query types
- Implement prompt selection based on query classification
- Add system prompts with financial reasoning instructions

## Phase 2: Advanced Retrieval Techniques (Weeks 3-4)

### 1. Upgrade to Newer Embedding Models
**Goal**: Improve embedding quality with better models available on Huggingface.
- Test and benchmark embedding models: `intfloat/e5-large`, `BAAI/bge-large-en-v1.5`
- Implement an embedding model switcher to compare performance
- Optional: Evaluate finance-specific models like `yiyanghkust/finbert-tone`

### 2. Multi-Query Expansion
**Goal**: Generate multiple perspectives on user queries.
- Use Ollama to generate query variations (2-3 per query)
- Retrieve results for each variation
- Implement result fusion to combine retrieved documents

```python
def generate_query_variations(query, llm_client, n=3):
    """Generate variations of the query using LLM"""
    prompt = f"""Generate {n} different variations of this financial question 
    that preserve the meaning but use different wording:
    "{query}"
    Output format: 1. [variation1] 2. [variation2] etc."""
    
    response = llm_client.generate_text(prompt)
    variations = parse_variations(response)
    return variations

def multi_query_retrieval(query, variations, retriever):
    """Retrieve results for original query and variations, then combine"""
    all_results = []
    for q in [query] + variations:
        results = retriever.retrieve(q)
        all_results.extend(results)
    
    # Remove duplicates and rerank
    return deduplicate_and_rerank(all_results)
```

### 3. Implement Chain-of-Thought (CoT) Retrieval
**Goal**: Break down complex financial analyses into steps.
- Create a CoT framework for complex financial queries
- Add intermediate reasoning steps for calculations
- Develop specialized prompts for numerical reasoning

## Phase 3: Contextual and Memory Enhancements (Weeks 5-6)

### 1. Conversation Memory System
**Goal**: Improve multi-turn performance with memory.
- Implement short-term conversation memory (last 3-5 turns)
- Add memory retrieval for context-dependent questions
- Create a summarization system for long conversations

```python
class ConversationMemory:
    def __init__(self, max_history=5):
        self.history = []
        self.max_history = max_history
        
    def add(self, query, response, context_used):
        self.history.append({
            'query': query,
            'response': response,
            'context_used': context_used,
            'timestamp': datetime.now()
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_conversation_context(self):
        """Format conversation history for context"""
        context = ""
        for item in self.history:
            context += f"User: {item['query']}\nAssistant: {item['response']}\n\n"
        return context
```

### 2. Self-RAG Implementation
**Goal**: Add fact-checking and verification.
- Implement a citation generation system
- Add verification step before response generation
- Create a confidence scoring system for responses

### 3. Financial Chunking Strategies
**Goal**: Improve how transactions are processed and stored.
- Implement monthly and category-based aggregations
- Create a hierarchical indexing system
- Develop context-aware document chunks for transactions

## Phase 4: Evaluation and Continuous Improvement (Weeks 7-8)

### 1. RAG Evaluation Framework
**Goal**: Measure and improve RAG performance.
- Create a finance-specific evaluation dataset
- Implement RAGAS metrics for evaluation (faithfulness, relevance)
- Build a simple dashboard for tracking improvements

```python
def evaluate_rag_system(system, test_questions, ground_truth):
    """Evaluate RAG system performance"""
    results = {
        'correct_answers': 0,
        'hallucinations': 0,
        'context_used': 0,
        'response_time': []
    }
    
    for i, question in enumerate(test_questions):
        start_time = time.time()
        response, context = system.query_with_context(question)
        end_time = time.time()
        
        # Check correctness against ground truth
        is_correct = check_answer_correctness(response, ground_truth[i])
        has_hallucination = detect_hallucination(response, context)
        
        # Update metrics
        results['correct_answers'] += 1 if is_correct else 0
        results['hallucinations'] += 1 if has_hallucination else 0
        results['context_used'] += 1 if context else 0
        results['response_time'].append(end_time - start_time)
    
    # Calculate percentages
    total = len(test_questions)
    results['accuracy'] = results['correct_answers'] / total
    results['hallucination_rate'] = results['hallucinations'] / total
    results['context_utilization'] = results['context_used'] / total
    results['avg_response_time'] = sum(results['response_time']) / total
    
    return results
```

### 2. User Feedback Collection
**Goal**: Gather feedback to iteratively improve.
- Add simple feedback buttons in the UI
- Implement feedback collection API
- Create a system to prioritize improvements

### 3. Performance Optimization
**Goal**: Optimize response time and resource usage.
- Implement query and embedding caching
- Add batch processing for embeddings
- Optimize database queries for faster retrieval

## Practical Implementation Strategy

### Technical Resources to Use (All Free):
- **Ollama**: For local LLM generation with models like Llama3, Mistral, etc.
- **Huggingface Models**: For embeddings, classification, and other ML tasks
- **ChromaDB**: Continue using for vector database (already implemented)
- **Python Libraries**: rank_bm25, nltk, spaCy (community version)
- **LangChain/LlamaIndex**: Limited use of their free components

### Prioritization Framework:
1. **Impact vs. Effort**: Focus on high-impact, low-effort improvements first
2. **Resource Constraints**: Prioritize techniques that work well with smaller models
3. **Measurable Results**: Implement changes that can be objectively measured

### Implementation Approach:
1. Start with isolated components that don't require full system refactoring
2. Create A/B testing capability to compare approaches
3. Document all changes thoroughly for future reference
4. Implement feature flags to toggle new capabilities
5. Get regular feedback from users to guide priorities

## First Steps to Take This Week:

1. Implement basic BM25 hybrid search
2. Create 3 specialized prompt templates for different financial query types
3. Add basic financial entity extraction
4. Set up a simple evaluation framework with 20-30 test questions

This plan focuses on practical, zero-cost enhancements that can significantly improve your Finance RAG system while working within the constraints of using only Huggingface and Ollama for LLM resources. 