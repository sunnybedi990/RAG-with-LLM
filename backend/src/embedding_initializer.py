from sentence_transformers import SentenceTransformer
import openai
from embedding_config import get_embedding_config
from transformers import AutoModel, AutoTokenizer
import tensorflow_hub as hub
import gensim.downloader as api
import os
from dotenv import load_dotenv
load_dotenv()

def initialize_embedding_model(provider, model_name, api_key=None):
    """Initializes the embedding model based on the provider and model name."""
    config = get_embedding_config(provider, model_name)
    dimension = config["dimension"]
    print(f"Provider: {provider}, Model: {model_name}, Dimension: {dimension}")

    is_callable = False  # Default to non-callable

    if provider == "sentence_transformers":
        # Sentence Transformers
        model = SentenceTransformer(model_name)
        is_callable = False

    elif provider == "openai":
        api_key=os.getenv("OPENAI_API_KEY")
        # OpenAI Embeddings
        if not api_key:
            raise ValueError("OpenAI API key is required for OpenAI models.")
        openai_client = openai.OpenAI(api_key=api_key)
        model = lambda text: openai_client.embeddings.create(input=text, model=model_name).data[0].embedding
        is_callable = True

    elif provider == "hugging_face":
        # Hugging Face Transformers
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        hf_model = AutoModel.from_pretrained(model_name)
        model = lambda text: hf_model(**tokenizer(text, return_tensors="pt"))[0].mean(dim=1).detach().numpy()
        is_callable = True

    elif provider == "google_use":
        # Google Universal Sentence Encoder
        model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        is_callable = True

    elif provider == "elmo":
        # ELMo model
        model = hub.load("https://tfhub.dev/google/elmo/3")
        is_callable = True

    elif provider == "fasttext":
        # FastText model with Gensim
        model = api.load(model_name)
        is_callable = True

    elif provider == "glove":
        # GloVe model with Gensim
        model = api.load(model_name)
        is_callable = True
    
    elif provider in ["bert", "albert", "xlnet", "gpt2", "t5"]:
        # Hugging Face BERT, ALBERT, XLNet, GPT-2, T5 models
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        hf_model = AutoModel.from_pretrained(model_name)
        model = lambda text: hf_model(**tokenizer(text, return_tensors="pt"))[0].mean(dim=1).detach().numpy()
        is_callable = True

    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return model, dimension, is_callable
