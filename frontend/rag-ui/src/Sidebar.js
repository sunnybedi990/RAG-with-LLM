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

function Sidebar({ provider, setProvider, model, setModel, topK, setTopK, file, setFile, onSummarize }) {

    const [uploadedFileName, setUploadedFileName] = useState("");
    const [isDragActive, setIsDragActive] = useState(false);
    const [downloadStatus, setDownloadStatus] = useState("");
    const [progress, setProgress] = useState(0);

    const [remainingTime, setRemainingTime] = useState(null);


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

            {/* API Provider, Model, and Top K Results Settings */}
            <div className="sidebar-section">
                <label className="sidebar-label">API Provider</label>
                <select value={provider} onChange={handleProviderChange} className="sidebar-select">
                    {Object.keys(apiModels).map((providerKey) => (
                        <option key={providerKey} value={providerKey}>
                            {providerKey}
                        </option>
                    ))}
                </select>
            </div>

            <div className="sidebar-section">
                <label className="sidebar-label">Model</label>
                <select value={model} onChange={handleModelChange} className="sidebar-select">
                    {apiModels[provider].map((modelName) => (
                        <option key={modelName} value={modelName}>
                            {modelName}
                        </option>
                    ))}
                </select>
            </div>
            {/* Display download status if a model is being downloaded or is downloaded */}
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