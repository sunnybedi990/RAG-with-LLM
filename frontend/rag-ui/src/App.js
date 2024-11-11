// App.js
import React, { useState } from 'react';
import Sidebar from './Sidebar';
import QueryForm from './QueryForm';
import './App.css';

function App() {
    const [provider, setProvider] = useState('ollama');
    const [model, setModel] = useState("Llama 3.1 - 8B");
    const [topK, setTopK] = useState(3);
    const [file, setFile] = useState(null); // Lift file state up to App
    const [summary, setSummary] = useState(""); // State to store summary response

    // Define the onSummarize function to handle summary request
    const handleSummarize = async () => {
        if (!file) {
            alert("Please upload a PDF file first to summarize.");
            return;
        }

        try {
            const response = await fetch('http://localhost:5000/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    provider,
                    query: "Please summarize the entire document.",
                    model,
                    top_k: topK,
                    db_filename: file.name
                }),
            });
            const data = await response.json();
            setSummary(data.response || "No summary available."); // Store summary response
        } catch (error) {
            console.error("Summarization error:", error);
            alert("Error summarizing the document.");
        }
    };

    return (
        <div className="app">
            <Sidebar 
                provider={provider}
                setProvider={setProvider}
                model={model}
                setModel={setModel}
                topK={topK}
                setTopK={setTopK}
                file={file}
                setFile={setFile}
                onSummarize={handleSummarize} // Pass onSummarize as a prop to Sidebar
            />
            <QueryForm 
                provider={provider}
                model={model}
                topK={topK}
                file={file}
                summary={summary} // Pass summary to QueryForm for display
            />
        </div>
    );
}

export default App;
