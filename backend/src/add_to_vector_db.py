import os
from VectorDB import VectorDB
from pdf2image import convert_from_path
from dotenv import load_dotenv
from pdf_extractor import (
    extract_text_with_fitz,
    extract_tables,
    extract_figures,
    process_figure_with_clip,
    chunk_text_advanced,
    preprocess_text,
    chunk_text,
    chunk_text_by_semantics
)


# Load environment variables for LlamaParse API access
load_dotenv()


# LlamaParse setup
try:
    from llama_parse import LlamaParse
    from llama_index.core import SimpleDirectoryReader
    llama_available = True
except ImportError:
    llama_available = False


def extract_pdf_with_llama(pdf_path):
    """Extract tables and text using LlamaParse."""
    parser = LlamaParse(result_type="text")  # Try using plain_text for broader content capture
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(input_files=[pdf_path], file_extractor=file_extractor).load_data()
    
    # Convert each document to a string and join them

    parsed_texts = [str(doc) for doc in documents]
    #combined_text = "\n".join(parsed_texts)  # Join all document parts into one complete text
    print(parsed_texts)


    return parsed_texts  # Return the fully combined text

    
def add_pdf_to_vector_db(
    pdf_path,
    db_path='vector_db.index',
    db_type='faiss',
    db_config='true',
    embedding_provider='sentence_transformers',
    embedding_model='all-mpnet-base-v2',
    use_gpu=True,
    use_llama=False,
    api_key=None
):
    """
    Processes a PDF, extracts text and tables, and adds them to a vector database.
    """
    try:
        # Extract content from PDF
        if use_llama and llama_available:
            print("Using LlamaParse for document extraction...")
            parsed_texts = extract_pdf_with_llama(pdf_path)
        else:
            print("Using regular extraction methods...")
            print("Extracting content...")
            full_text = extract_text_with_fitz(pdf_path)
            tables = extract_tables(pdf_path)
            figures = extract_figures(pdf_path)

            parsed_texts = chunk_text_by_semantics(preprocess_text(full_text))
            parsed_texts += tables
            # Save figures (Optional: Store as metadata)
            
        if not parsed_texts:
            print("No content extracted from the PDF.")
            return

        print(f"Extracted {len(parsed_texts)} items from the PDF.")

        # Initialize the vector database
        db = VectorDB(
            db_path=db_path,
            db_type=db_type,
            db_config=db_config,
            provider=embedding_provider,
            model_name=embedding_model,
            use_gpu=use_gpu,
            api_key=api_key,
            collection_name=os.path.splitext(os.path.basename(db_path))[0],  # Milvus-specific
            index_name=os.path.splitext(os.path.basename(db_path))[0]  # Pinecone-specific
        )
            # Process figures using CLIP
        if figures:
            print(f"Extracted {len(figures)} figures from the PDF.")
            for i, figure in enumerate(figures):
                caption, clip_embedding = process_figure_with_clip(figure, i)

                # Append caption to parsed_texts
                parsed_texts.append(caption)

                # Add figure embedding directly to the vector database
                if clip_embedding is not None:
                    print(clip_embedding.shape)
                    db.add_embeddings(clip_embeddings=clip_embedding, texts=[caption])


        # Add embeddings to the vector database
        db.add_embeddings(parsed_texts)

        # Save the FAISS index if applicable
        if db_type == "faiss":
            db.save_index(db_path)
            print(f"FAISS index saved at {db_path}.")
        else:
            print(f"Data added to {db_type} vector database.")

    except Exception as e:
        print(f"Error adding PDF to vector database: {e}")
        raise


# Usage Example:
if __name__ == "__main__":
    pdf_path = 'tsla-20240930-gen.pdf'
    add_pdf_to_vector_db(pdf_path, db_path='tesla_vector_db.index', use_llama=True)
