import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  citations?: number[];
  tools_used?: string[];
  sql_queries?: string[];
  confident?: boolean;
  steps?: number;
}

interface System {
  id: string;
  name: string;
  description: string;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [system, setSystem] = useState('A5');
  const [systems, setSystems] = useState<{id: string; name: string; description: string}[]>([]);
  const [showSystems, setShowSystems] = useState(false);

  const fetchSystems = async () => {
    try {
      const res = await fetch(`${API_BASE}/systems`);
      const data = await res.json();
      setSystems(data.systems);
    } catch (err) {
      console.error('Failed to fetch systems:', err);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setLoading(true);
    setInput('');

    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: input, system }),
      });
      const data = await res.json();
      setMessages(prev => [...prev, { role: 'assistant', ...data }]);
    } catch (err) {
      console.error('Error:', err);
      setMessages(prev => [...prev, { role: 'assistant', content: 'Error: Failed to get response' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header>
        <h1>BioEvidence</h1>
        <div className="system-selector">
          <select value={system} onChange={e => setSystem(e.target.value)} disabled={loading}>
            {systems.map(s => (
              <option key={s.id} value={s.id}>{s.name} - {s.description}</option>
            ))}
          </select>
        </div>
      </header>

      <main>
        <div className="messages">
          {messages.map((msg, i) => (
            <div key={i} className={`message ${msg.role}`}>
              <div className="role">{msg.role === 'user' ? 'You' : 'BioEvidence'}</div>
              <div className="content">
                <ReactMarkdown>{msg.content}</ReactMarkdown>
              </div>
              {msg.role === 'assistant' && (
                <div className="metadata">
                  {msg.citations && msg.citations.length > 0 && (
                    <span>📎 Citations: {msg.citations.join(', ')}</span>
                  )}
                  {msg.tools_used && msg.tools_used.length > 0 && (
                    <span>🔧 Tools: {msg.tools_used.join(', ')}</span>
                  )}
                  {msg.sql_queries && msg.sql_queries.length > 0 && (
                    <span>💾 SQL: {msg.sql_queries[0].substring(0, 50)}...</span>
                  )}
                  {msg.confident !== undefined && (
                    <span className={msg.confident ? 'confident' : 'not-confident'}>
                      {msg.confident ? '✓ Confident' : '⚠ Low confidence'}
                    </span>
                  )}
                  {msg.steps !== undefined && <span>Steps: {msg.steps}</span>}
                </div>
              )}
            </div>
          ))}
          {loading && <div className="loading">Thinking...</div>}
        </div>
      </main>

      <form onSubmit={handleSubmit} className="input-form">
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="Ask a question about clinical trials, drugs, or biomedical evidence..."
          disabled={loading}
          rows={3}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          {loading ? 'Thinking...' : 'Ask'}
        </button>
      </form>

      <footer>
        <p>Research demo - not medical advice. Powered by BioEvidence.</p>
      </footer>
    </div>
  );
}

export default App;