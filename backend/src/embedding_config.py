# config.py

embedding_configs = {
    "sentence_transformers": {
        "all-mpnet-base-v2": {"dimension": 768, "description": "High-quality sentence embeddings for semantic search and clustering."},
        "all-MiniLM-L6-v2": {"dimension": 384, "description": "Efficient model balancing performance and computational needs."},
        "paraphrase-MiniLM-L12-v2": {"dimension": 384, "description": "Captures paraphrastic relationships for paraphrase detection."},
    },
    "openai": {
        "text-embedding-ada-002": {"dimension": 1536, "description": "High-quality embeddings for various NLP tasks via OpenAI API."}
    },
    "hugging_face": {
        "distilbert-base-uncased": {"dimension": 768, "description": "Distilled BERT with faster inference and reduced model size."},
        "roberta-base": {"dimension": 768, "description": "Optimized BERT model, especially for nuanced language understanding."}
    },
    "google_use": {
        "universal-sentence-encoder": {"dimension": 512, "description": "Google’s universal sentence encoder for transfer learning tasks."}
    },
    "elmo": {
        "elmo-original": {"dimension": 1024, "description": "Deep contextualized word representations from ELMo for NLP tasks."}
    },
    "glove": {
        "glove.6B.300d": {"dimension": 300, "description": "Word-level embeddings capturing global statistical information."}
    },
    "fasttext": {
        "fasttext-wiki-news-subwords-300": {"dimension": 300, "description": "Word embeddings with subword information, beneficial for rare words."}
    },
    "bert": {
        "bert-base-uncased": {"dimension": 768, "description": "Bidirectional BERT for general NLP tasks with contextual understanding."},
        "bert-large-uncased": {"dimension": 1024, "description": "Larger BERT model offering deeper contextual representations."}
    },
    "albert": {
        "albert-base-v2": {"dimension": 768, "description": "Lite version of BERT, efficient for resource-constrained tasks."}
    },
    "xlnet": {
        "xlnet-base-cased": {"dimension": 768, "description": "Generalized autoregressive model excelling in capturing bidirectional context."}
    },
    "gpt2": {
        "gpt2-medium": {"dimension": 1024, "description": "Generative model from GPT-2, good for text generation tasks."}
    },
    "t5": {
        "t5-base": {"dimension": 768, "description": "Text-to-Text Transfer Transformer for diverse NLP tasks like translation and summarization."}
    }
}

def get_embedding_config(provider, model_name):
    """Retrieve the configuration for a given provider and model name."""
    config = embedding_configs.get(provider, {}).get(model_name)
    if not config:
        raise ValueError(f"Model {model_name} for provider {provider} not found in configuration.")
    return config
