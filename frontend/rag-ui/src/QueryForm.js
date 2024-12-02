import React, { useState, useEffect, useRef } from 'react';
import './QueryForm.css';

function QueryForm({ provider, model, topK, file, summary,embeddingProvider, embeddingModel, dbType, dbConfig}) {
    const [query, setQuery] = useState('');
    const [messages, setMessages] = useState([]);
    const [loading, setLoading] = useState(false);
    const bottomRef = useRef(null); // Reference to dummy div for auto-scrolling
     // Scroll to bottom whenever messages change
     useEffect(() => {
        if (bottomRef.current) {
            bottomRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [messages]);
    // Display summary when updated
    useEffect(() => {
        if (summary) {
            const botMessage = { role: 'bot', content: summary };
            setMessages((prevMessages) => [...prevMessages, botMessage]);
        }
    }, [summary]);
    const handleSend = async (e) => {
        e.preventDefault();
        setLoading(true);

        const userMessage = { role: 'user', content: query };
        setMessages([...messages, userMessage]);
        setQuery('');

        try {
            const response = await fetch('http://localhost:5000/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    provider,
                    query,
                    model,
                    top_k: topK,
                    db_type: dbType, // Pass selected vector database type
                    db_config: dbConfig,
                    db_filename: file ? file.name.replace(' ', '_') : '',  // Replace spaces in file name
                    embedding_provider: embeddingProvider,  // Add embedding provider
                    embedding_model: embeddingModel         // Add embedding model
                }),
            });
            const data = await response.json();
            const botMessage = { role: 'bot', content: data.response || "No response from model." };
            setMessages((prevMessages) => [...prevMessages, botMessage]);
        } catch (error) {
            const errorMessage = { role: 'error', content: 'Error: Unable to get response from the server.' };
            setMessages((prevMessages) => [...prevMessages, errorMessage]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="query-form">
            <div className="chat-area">
                {messages.map((msg, index) => (
                    <div key={index} className={`message ${msg.role}`}>
                        {msg.role === 'bot'
                            ? msg.content.split(/\n\n/).map((paragraph, i) => (
                                <p key={i} style={{ marginBottom: '10px' }}>{paragraph}</p>
                            ))
                            : msg.content}
                    </div>
                ))}
                {/* Dummy div for scroll-to-bottom */}
                <div ref={bottomRef} />
            </div>
            <form onSubmit={handleSend} className="input-area">
                <textarea
                    className="input"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Type your query..."
                    required
                />
                <button type="submit" disabled={loading} className="send-button">
                    {loading ? 'Sending...' : 'Send'}
                </button>
            </form>
        </div>
    );
}

export default QueryForm;
