import os
import re
import camelot
import fitz
from VectorDB import VectorDB
from dotenv import load_dotenv

# Load environment variables for LlamaParse API access
load_dotenv()

# Optional LlamaParse setup
try:
    from llama_parse import LlamaParse
    from llama_index.core import SimpleDirectoryReader
    llama_available = True
except ImportError:
    llama_available = False

def clean_text(text):
    """Clean text by removing extraneous symbols and whitespace."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\u2022", "-", text)  # Replace bullet points
    return text

def extract_tables_camelot(pdf_path):
    """Extract tables from PDF using Camelot and process them into readable strings."""
    tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')  # Use 'lattice' if tables have borders
    table_texts = []

    for i, table in enumerate(tables):
        df = table.df  # Convert Camelot Table object to a Pandas DataFrame

        # Clean headers and rows
        df.columns = [re.sub(r'Unnamed.*', '', str(col)) for col in df.columns]
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)  # Strip whitespace from strings
        df = df.fillna("")

        for _, row in df.iterrows():
            row_text = ', '.join([
                f"{col}: {str(val).strip()}" for col, val in row.items() 
                if str(val).strip() and not re.fullmatch(r"^\s*[.,]*\s*$", str(val))
            ])
            row_text = re.sub(r"\b([.,%$])\b", "", row_text)  # Remove isolated symbols
            row_text = re.sub(r"(?<=\d)[.,%$]+", "", row_text)  # Symbols after numbers
            row_text = re.sub(r"\s+", " ", row_text).strip()  # Collapse multiple spaces
            table_texts.append(row_text)
    
    return table_texts

def extract_full_text_fitz(pdf_path):
    """Extract and clean text from entire PDF using fitz."""
    document = fitz.open(pdf_path)
    full_text = ''

    for page_num in range(document.page_count):
        page = document[page_num]
        full_text += page.get_text('text')

    document.close()

    paragraphs = [para.strip() for para in full_text.split('\n\n') if para.strip()]
    cleaned_paragraphs = [re.sub(r'\s+', ' ', para.replace('\n', ' ')) for para in paragraphs]
    return cleaned_paragraphs

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


def add_pdf_to_vector_db(pdf_path, db_path='vector_db.index', embedding_provider='sentence_transformers', embedding_model='all-mpnet-base-v2', use_gpu=True, use_llama=False, api_key=None):
    """Processes a PDF, extracts text and tables, and adds them to a vector database."""
    if use_llama and llama_available:
        print("Using LlamaParse for document extraction...")
        parsed_texts = extract_pdf_with_llama(pdf_path)
        print(parsed_texts)
    else:
        print("Using regular extraction methods...")
        table_texts = extract_tables_camelot(pdf_path)
        full_texts = extract_full_text_fitz(pdf_path)
        parsed_texts = table_texts + full_texts

    if parsed_texts:
        print(f"Extracted {len(parsed_texts)} items from the PDF.")
    else:
        print("No content extracted from the PDF.")

    # Initialize VectorDB with the selected embedding provider and model
    db = VectorDB(provider=embedding_provider, model_name=embedding_model, use_gpu=use_gpu, api_key=api_key)
    db.add_embeddings(parsed_texts)  # Add all extracted texts to VectorDB
    print('pass1')
    db.verify_index()
    print('pass2')
    db.save_index(db_path)  # Save the index

    print(f"All data added to vector database and saved at {db_path}.")


# Usage Example:
if __name__ == "__main__":
    pdf_path = 'tsla-20240930-gen.pdf'
    add_pdf_to_vector_db(pdf_path, db_path='tesla_vector_db.index', use_llama=True)
