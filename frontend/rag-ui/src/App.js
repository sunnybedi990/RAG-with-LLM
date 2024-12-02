// App.js
import React, { useState } from 'react';
import Sidebar from './Sidebar';
import QueryForm from './QueryForm';
import './App.css';

const dbConfigFields = {
    faiss: { 
        use_gpu: ["true", "false"] 
    },
    milvus: { 
        mode: ["local", "cloud"] 
    },
    pinecone: { 
        environment: ["us-east-1", "us-west-1", "asia-southeast-1"] 
    },
    qdrant: { 
         mode: ["local", "cloud", "memory"]
    },
    weaviate: { 
        mode: ["local", "cloud"] 
    }
};

// Extract default values
const getDefaultDbConfig = () => {
    const defaultConfig = {};
    Object.entries(dbConfigFields).forEach(([dbType, fields]) => {
        Object.entries(fields).forEach(([key, options]) => {
            if (Array.isArray(options)) {
                defaultConfig[key] = options[0]; // Set the first option as default
            }
        });
    });
    return defaultConfig;
};
function App() {
    const [provider, setProvider] = useState('ollama');
    const [model, setModel] = useState("Llama 3.1 - 8B");
    const [embeddingProvider, setEmbeddingProvider] = useState("sentence_transformers"); // Default embedding provider
    const [embeddingModel, setEmbeddingModel] = useState("all-mpnet-base-v2"); // Default embedding model
    const [selectedCategory, setSelectedCategory] = useState("sentence_transformers"); // Default category
    const [selectedSubCategory, setSelectedSubCategory] = useState("sentence_transformers"); // Default subcategory
    const [topK, setTopK] = useState(3);
    const [file, setFile] = useState(null); // Lift file state up to App
    const [summary, setSummary] = useState(""); // State to store summary response
    const [dbType, setDbType] = useState("faiss"); // State for vector database type
    const [dbConfig, setDbConfig] = useState(getDefaultDbConfig()); // Initialize with defaults




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
                    embedding_provider: selectedSubCategory, // Pass embedding provider
                    embedding_model: embeddingModel, // Pass embedding model
                    top_k: topK,
                    db_type: dbType, // Pass selected vector database type
                    db_config: dbConfig,
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
                embeddingProvider={embeddingProvider} // Pass embeddingProvider state to Sidebar
                setEmbeddingProvider={setEmbeddingProvider} // Pass setter to Sidebar
                embeddingModel={embeddingModel} // Pass embeddingModel state to Sidebar
                setEmbeddingModel={setEmbeddingModel} // Pass setter to Sidebar
                selectedCategory={selectedCategory}  // Pass selectedCategory state to Sidebar
                setSelectedCategory={setSelectedCategory}  // Pass setter to Sidebar
                selectedSubCategory={selectedSubCategory}  // Pass selectedSubCategory state to Sidebar
                setSelectedSubCategory={setSelectedSubCategory}  // Pass setter to Sidebar
                topK={topK}
                setTopK={setTopK}
                dbType={dbType} // Pass dbType state
                setDbType={setDbType} // Pass setter for dbType
                dbConfig={dbConfig}
                setDbConfig={setDbConfig}
                file={file}
                dbConfigFields={dbConfigFields}
                setFile={setFile}
                onSummarize={handleSummarize} // Pass onSummarize as a prop to Sidebar
            />
            <QueryForm
                provider={provider}
                model={model}
                topK={topK}
                dbType={dbType} // Pass dbType state
                dbConfig={dbConfig}
                file={file}
                summary={summary} // Pass summary to QueryForm for display
                embeddingProvider={selectedSubCategory}  // Pass embedding provider
                embeddingModel={embeddingModel}        // Pass embedding model
            />
        </div>
    );
}

export default App;
