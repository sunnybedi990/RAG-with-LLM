import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from "remark-gfm";

import './QueryForm.css';

function QueryForm({ provider, model, topK, file, summary, embeddingProvider, embeddingModel, dbType, dbConfig }) {
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
            //console.log(data)
            let botMessage;
            console.log('data:', data);
            console.log('data.chart_image_path:', data.chart_image_path);
            console.log('Type of data.chart_image_path:', typeof data.chart_image_path);

            if (data.response && data.response.chart_image_path) {
                // If the response contains a valid image path
                botMessage = {
                    role: 'bot',
                    content: data.response.chart_type,
                    image: data.response.chart_image_path.trim(), // Access nested property
                };
                console.log('Valid image path:', data.response.chart_image_path);
            } else {
                botMessage = {
                    role: 'bot',
                    content: data.response || 'No response from model.',
                };
            }

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
                        {msg.role === 'bot' && msg.image ? (
                            <>
                                <p>{msg.content}</p>
                                <img
                                    src={`http://localhost:5000/static/${msg.image}`} // Adjust the path based on your server setup
                                    alt="Generated Chart"
                                    style={{ maxWidth: '100%', marginTop: '10px' }}
                                />
                            </>
                        ) : (
                            <ReactMarkdown
                                remarkPlugins={[remarkGfm]}
                                components={{
                                    p: ({ children }) => (
                                        <p style={{ marginBottom: "1em", lineHeight: "1.6" }}>{children}</p>
                                    ),
                                    ul: ({ children }) => (
                                        <ul style={{ marginLeft: "1.5em", marginBottom: "1em", paddingLeft: "1em" }}>
                                            {children}
                                        </ul>
                                    ),
                                    ol: ({ children }) => (
                                        <ol style={{ marginLeft: "1.5em", marginBottom: "1em", paddingLeft: "1em" }}>
                                            {children}
                                        </ol>
                                    ),
                                    h1: ({ children }) => (
                                        <h1 style={{ marginTop: "1em", marginBottom: "0.5em", fontSize: "1.5em" }}>
                                            {children}
                                        </h1>
                                    ),
                                    h2: ({ children }) => (
                                        <h2 style={{ marginTop: "1em", marginBottom: "0.5em", fontSize: "1.3em" }}>
                                            {children}
                                        </h2>
                                    ),
                                    h3: ({ children }) => (
                                        <h3 style={{ marginTop: "1em", marginBottom: "0.5em", fontSize: "1.1em" }}>
                                            {children}
                                        </h3>
                                    ),
                                    blockquote: ({ children }) => (
                                        <blockquote
                                            style={{
                                                marginLeft: "1.5em",
                                                borderLeft: "4px solid #4CAF50",
                                                paddingLeft: "1em",
                                                color: "#ccc",
                                                fontStyle: "italic",
                                                marginBottom: "1em",
                                            }}
                                        >
                                            {children}
                                        </blockquote>
                                    ),
                                    code: ({ children }) => (
                                        <code
                                            style={{
                                                backgroundColor: "#444",
                                                color: "#4CAF50",
                                                padding: "0.2em 0.4em",
                                                borderRadius: "4px",
                                            }}
                                        >
                                            {children}
                                        </code>
                                    ),
                                    pre: ({ children }) => (
                                        <pre
                                            style={{
                                                backgroundColor: "#333744",
                                                color: "#fff",
                                                padding: "10px",
                                                borderRadius: "5px",
                                                overflowX: "auto",
                                                marginBottom: "1em",
                                            }}
                                        >
                                            {children}
                                        </pre>
                                    ),
                                }}
                            >
                                {msg.content}
                            </ReactMarkdown>
                        )}
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
