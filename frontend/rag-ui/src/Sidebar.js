import React, { useState } from 'react';
import axios from 'axios';
import './Sidebar.css';

const apiModels = {
    openai: ["gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini"],
    groq: [
        "gemma-7b-it",
        "gemma2-9b-it",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-guard-3-8b",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "llama3-groq-70b-8192-tool-use-preview",
        "llama3-groq-8b-8192-tool-use-preview",
        "llava-v1.5-7b-4096-preview",
        "mixtral-8x7b-32768"
    ],
    ollama: [
        "Llama 3.1 - 8B",
        "Llama 3.1 - 70B",
        "Gemma 2 - 2B",
        "Gemma 2 - 9B",
        "Mistral-Nemo - 12B",
        "Mistral Large 2 - 123B",
        "Qwen 2 - 0.5B",
        "Qwen 2 - 72B",
        "DeepSeek-Coder V2 - 16B",
        "Phi-3 - 3B",
        "Phi-3 - 14B"
    ]
};

const embeddingModels = {
    // Older/Classic Models (pre-2020)
    classic: {
        elmo: ["elmo-original"],
        fasttext: ["fasttext-wiki-news-subwords-300"],
        bert: ["bert-base-uncased", "bert-large-uncased"],
        xlnet: ["xlnet-base-cased"],
        albert: ["albert-base-v2"],
        google_use: ["universal-sentence-encoder"]
    },

    // Sentence Transformer Models (2020–2022)
    sentence_transformers: {
        sentence_transformers: ["all-mpnet-base-v2", "all-MiniLM-L6-v2", "paraphrase-MiniLM-L12-v2"]
    },

    // Large Language Model (LLM) Based Embeddings
    llm_based: {
        openai: ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"],
        gpt2: ["gpt2-medium"],
        cohere: ["embed-v3"],
        t5: ["t5-base"],
        jamba: ["jamba"]
    },

    // Newer Specialized Models (2023–2024)
    new_models: {
        arctic_embed: ["arctic-embed-small", "arctic-embed-medium", "arctic-embed-large"],
        nv_embed: ["nv-embed"],
        longembed: ["longembed"]
    },

    // Mamba-Based State Space Models
    mamba: {
        mamba: ["mamba-byte", "moe-mamba", "vision-mamba"]
    },

    // Hugging Face General Embedding Models
    hugging_face: {
        hugging_face: ["distilbert-base-uncased", "roberta-base"]
    }
};

const vectorDbOptions = ["faiss", "milvus", "pinecone", "qdrant", "weaviate"]; // Available vector databases


function Sidebar({
    provider, setProvider, model, setModel,
    embeddingProvider, setEmbeddingProvider,
    embeddingModel, setEmbeddingModel,
    selectedCategory, setSelectedCategory,
    selectedSubCategory, setSelectedSubCategory,
    topK, setTopK, 
    dbType, setDbType,
    dbConfig, setDbConfig, // Add dbType and its setter
    dbConfigFields,
    file, setFile, onSummarize
}) {

    const [uploadedFileName, setUploadedFileName] = useState("");
    const [isDragActive, setIsDragActive] = useState(false);
    const [downloadStatus, setDownloadStatus] = useState("");
    const [useLlamaParser, setUseLlamaParser] = useState(false); // State for parser selection

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        setFile(selectedFile);
        setUploadedFileName(selectedFile ? selectedFile.name : "");
    };
    
    

    const handleFileUpload = async () => {
        if (!file) {
            alert("Please select a PDF file to upload.");
            return;
        }

        const formData = new FormData();
        formData.append('pdf', file);
        formData.append('embedding_provider', selectedSubCategory);
        formData.append('embedding_model', embeddingModel);
        formData.append('parser_type', useLlamaParser ? 'LlamaParser' : 'CustomParser'); // Add parser type to form data
        formData.append('db_type', dbType); // Include vector database type
        formData.append('db_config', JSON.stringify(dbConfig)); // Pass DB configuration


        try {
            const res = await axios.post('http://localhost:5000/add', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            alert(res.data.message || 'File uploaded successfully.');
        } catch (error) {
            alert('Error uploading file.');
            console.error('Upload error:', error);
        }
    };
    const handleParserChange = (e) => {
        setUseLlamaParser(e.target.value === 'LlamaParser');
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragActive(true);
    };

    const handleDragLeave = () => {
        setIsDragActive(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragActive(false);
        const droppedFile = e.dataTransfer.files[0];
        setFile(droppedFile);
        setUploadedFileName(droppedFile ? droppedFile.name : "");
    };
    const handleDbTypeChange = (e) => {
        setDbType(e.target.value); // Update the selected vector database type
        //setDbConfig(getDefaultDbConfig(newDbType)); // Reset config to defaults for the selected dbType

    };

    const renderDbConfigFields = () => {
        const selectedFields = dbConfigFields[dbType] || {}; // Fetch fields for the selected dbType
    
        return Object.entries(selectedFields).map(([key, options]) => (
            <div key={key} className="sidebar-section">
                <label className="sidebar-label">{key.replace('_', ' ').toUpperCase()}</label>
                {Array.isArray(options) ? (
                    <select
                        value={dbConfig[key] || options[0]} // Default to the first option
                        onChange={(e) => setDbConfig((prev) => ({ ...prev, [key]: e.target.value }))}
                        className="sidebar-select"
                    >
                        {options.map((option) => (
                            <option key={option} value={option}>
                                {option}
                            </option>
                        ))}
                    </select>
                ) : (
                    <input
                        type="text"
                        value={dbConfig[key] || ""}
                        onChange={(e) => setDbConfig((prev) => ({ ...prev, [key]: e.target.value }))}
                        className="sidebar-input"
                    />
                )}
            </div>
        ));
    };
    
    
    const handleProviderChange = (event) => {
        const selectedProvider = event.target.value;
        setProvider(selectedProvider);
        setModel(apiModels[selectedProvider][0]);
    };
    const handleModelChange = async (event) => {
        const selectedModel = event.target.value;
        setModel(selectedModel);

        if (provider === "ollama") {
            try {
                const response = await axios.get(`http://localhost:5000/api/check-model?model=${selectedModel}`);
                if (!response.data.installed) {
                    const confirmPull = window.confirm(`Model "${selectedModel}" is not installed. Do you want to pull it?`);
                    if (confirmPull) {
                        setDownloadStatus("Downloading...");
                        await axios.post(`http://localhost:5000/api/pull-model`, { model: selectedModel });
                        pollForModelCompletion(selectedModel);
                    }
                } else {
                    setDownloadStatus("Downloaded");
                }
            } catch (error) {
                console.error("Error checking or pulling model:", error);
            }
        }
    };

    const handleEmbeddingProviderChange = (event) => {
        const category = event.target.value;
        setEmbeddingProvider(category);
        setSelectedCategory(category);
        setSelectedSubCategory("");
        setEmbeddingModel(""); // Reset model selection when category changes
    };

    const handleSubCategoryChange = (event) => {
        const subCategory = event.target.value;
        setSelectedSubCategory(subCategory);
        setEmbeddingModel(embeddingModels[selectedCategory][subCategory][0]); // Set the first model of this subCategory
    };

    const handleEmbeddingModelChange = (event) => {
        setEmbeddingModel(event.target.value);
    };


    // Polling function to check if the model download is complete
    const pollForModelCompletion = async (modelName) => {
        const intervalId = setInterval(async () => {
            try {
                const response = await axios.get(`http://localhost:5000/api/check-model?model=${modelName}`);
                if (response.data.installed) {
                    clearInterval(intervalId);
                    setDownloadStatus("Downloaded");
                    alert(`Model "${modelName}" is now available.`);
                }
            } catch (error) {
                console.error("Error polling model status:", error);
                clearInterval(intervalId);
                setDownloadStatus("Error during download.");
            }
        }, 5000); // Check every 5 seconds
    };

    const handleDeleteModel = async (modelName) => {
        try {
            await axios.post('http://localhost:5000/api/delete-model', { model: modelName });
            alert(`Model "${modelName}" deleted successfully.`);
        } catch (error) {
            console.error("Error deleting model:", error);
        }
    };

    const handleCancelPull = async (modelName) => {
        try {
            await axios.post('http://localhost:5000/api/cancel-pull', { model: modelName });
            alert(`Pulling for model "${modelName}" has been canceled.`);
            setDownloadStatus("");
        } catch (error) {
            console.error("Error canceling pull:", error);
        }
    };


    return (
        <div className="sidebar">
            <h3 className="sidebar-title">RAG with LLM</h3>

            {/* Drag and Drop File Upload Section */}
            <div
                className={`file-drop-zone ${isDragActive ? 'active' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => document.getElementById("file-upload").click()}
            >
                <input
                    type="file"
                    id="file-upload"
                    className="file-input"
                    onChange={handleFileChange}
                />
                <p className="drop-text">
                    {uploadedFileName || "Drag & Drop or Click to Upload PDF"}
                </p>
            </div>
            <button onClick={handleFileUpload} disabled={!file} className="upload-action-button">
                Upload PDF
            </button>

            {uploadedFileName && (
                <div className="file-name-box">
                    <span>{uploadedFileName}</span>
                </div>
            )}
             {/* Vector Database Selection */}
             <div className="group-box">
                <h4 className="group-title">Vector Database Settings</h4>
                <div className="sidebar-section">
                    <label className="sidebar-label">
                        Select Vector Database
                        <span className="help-icon" title="Choose the database where embeddings will be stored.">?</span>

                        </label>
                    <select
                        value={dbType}
                        onChange={handleDbTypeChange}
                        className="sidebar-select"
                    >
                        {vectorDbOptions.map((option) => (
                            <option key={option} value={option}>
                                {option}
                            </option>
                        ))}
                    </select>
                </div>
                <div className="sidebar-section">
                <label className="sidebar-label">
                Database Configuration</label>
                {renderDbConfigFields()}
            </div>
            </div>
            

            <div className="group-box">
                <h4 className="group-title">Parser Settings</h4>
                <div className="sidebar-section">
                    <div>
                    <label className="sidebar-label">
                            <input
                                type="radio"
                                name="parser"
                                value="LlamaParser"
                                checked={useLlamaParser}
                                onChange={handleParserChange}
                            />
                            Use LlamaParser for complex documents
                        </label>
                    </div>
                    <div>
                    <label className="sidebar-label">
                            <input
                                type="radio"
                                name="parser"
                                value="CustomParser"
                                checked={!useLlamaParser}
                                onChange={handleParserChange}
                            />
                            Use Custom Parser (Fitz + Camelot)
                        </label>
                    </div>
                </div>
            </div>


            {/* Embedding Provider and Model Group */}
            <div className="group-box">
                <h4 className="group-title">Embedding Settings</h4>
                {/* Parser Selection */}

                {/* Embedding Provider Dropdown */}
                <div className="sidebar-section">
                    <label className="sidebar-label">Embedding Provider Category</label>
                    <select value={embeddingProvider} onChange={handleEmbeddingProviderChange} className="sidebar-select">
                        <option value="">Select Provider</option>
                        {Object.keys(embeddingModels).map((providerKey) => (
                            <option key={providerKey} value={providerKey}>
                                {providerKey}
                            </option>
                        ))}
                    </select>
                </div>

                {/* Subcategory Dropdown (appears after selecting provider) */}
                {selectedCategory && (
                    <div className="sidebar-section">
                        <label className="sidebar-label">Embedding Provider</label>
                        <select value={selectedSubCategory} onChange={handleSubCategoryChange} className="sidebar-select">
                            <option value="">Select Subcategory</option>
                            {Object.keys(embeddingModels[selectedCategory]).map((subCategory) => (
                                <option key={subCategory} value={subCategory}>
                                    {subCategory}
                                </option>
                            ))}
                        </select>
                    </div>
                )}

                {/* Embedding Model Dropdown (appears after selecting subcategory) */}
                {selectedSubCategory && (
                    <div className="sidebar-section">
                        <label className="sidebar-label">Embedding Model</label>
                        <select value={embeddingModel} onChange={handleEmbeddingModelChange} className="sidebar-select">
                            <option value="">Select Model</option>
                            {embeddingModels[selectedCategory][selectedSubCategory].map((modelName) => (
                                <option key={modelName} value={modelName}>
                                    {modelName}
                                </option>
                            ))}
                        </select>
                    </div>
                )}
            </div>

            {/* API Provider and Model Group */}
            <div className="group-box">
                <h4 className="group-title">LLM Model Settings</h4>
                <div className="sidebar-section">
                    <label className="sidebar-label">LLM Model API Provider</label>
                    <select value={provider} onChange={handleProviderChange} className="sidebar-select">
                        {Object.keys(apiModels).map((providerKey) => (
                            <option key={providerKey} value={providerKey}>
                                {providerKey}
                            </option>
                        ))}
                    </select>
                </div>

                <div className="sidebar-section">
                    <label className="sidebar-label">LLM Model</label>
                    <select value={model} onChange={handleModelChange} className="sidebar-select">
                        {apiModels[provider].map((modelName) => (
                            <option key={modelName} value={modelName}>
                                {modelName}
                            </option>
                        ))}
                    </select>
                </div>
                {downloadStatus === "Downloading..." && (
                    <div>
                        <div className="spinner"></div>
                        <button onClick={() => handleCancelPull(model)}>Cancel Download</button>
                    </div>
                )}
                {downloadStatus === "Downloaded" && (
                    <div>
                        <button onClick={() => handleDeleteModel(model)}>Delete Model</button>
                    </div>
                )}
            </div>

            {/* Top K Results Group */}
            <div className="group-box">
                <h4 className="group-title">Results Settings</h4>
                <div className="sidebar-section">
                    <label className="sidebar-label">Top K Results</label>
                    <input
                        type="number"
                        value={topK}
                        onChange={(e) => setTopK(Number(e.target.value))}
                        min="1"
                        className="sidebar-input"
                    />
                </div>
            </div>

            {/* Summarize Button */}
            <div className="sidebar-section">
                <button onClick={onSummarize} disabled={!file} className="summarize-button">
                    Summarize Report
                </button>
            </div>
        </div>
    );

}

export default Sidebar;