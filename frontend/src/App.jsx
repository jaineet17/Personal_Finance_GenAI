import { useState } from 'react';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const apiUrl = import.meta.env.VITE_API_ENDPOINT || '/api';
      const response = await fetch(`${apiUrl}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });
      
      const data = await response.json();
      setResponse(data.response);
    } catch (error) {
      console.error('Error querying API:', error);
      setResponse('Error: Could not connect to API');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>Finance RAG Assistant</h1>
      </header>
      
      <main>
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question about your finances..."
            disabled={loading}
          />
          <button type="submit" disabled={loading || !query}>
            {loading ? 'Thinking...' : 'Ask'}
          </button>
        </form>
        
        {response && (
          <div className="response-container">
            <h2>Response:</h2>
            <p>{response}</p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App; 