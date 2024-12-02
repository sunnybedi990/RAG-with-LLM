import os
import re
import camelot
import fitz
from VectorDB import VectorDB
import spacy
from pdf2image import convert_from_path
import pytesseract
from dotenv import load_dotenv
import pdfplumber
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io

# Load environment variables for LlamaParse API access
load_dotenv()

# Initialize CLIP model and processor globally to avoid repeated loading
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

nlp = spacy.load("en_core_web_sm")

def chunk_text(text, max_length=512):
    """Chunk text into manageable pieces for embeddings."""
    doc = nlp(text)
    chunks, chunk = [], []
    length = 0

    for sent in doc.sents:
        length += len(sent.text)
        if length > max_length:
            chunks.append(" ".join(chunk))
            chunk = []
            length = len(sent.text)
        chunk.append(sent.text)

    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

# LlamaParse setup
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

def extract_tables(pdf_path):
    """Extract tables using Camelot and fallback to PDFPlumber if needed."""
    table_texts = []

    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
        if not tables:
            tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")

        for table in tables:
            df = table.df
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x).fillna("")
            for _, row in df.iterrows():
                row_text = ", ".join([f"{col}: {val}" for col, val in row.items() if val])
                table_texts.append(row_text)

    except Exception as e:
        print(f"Camelot failed: {e}. Falling back to PDFPlumber.")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    for table in page.extract_tables():
                        for row in table:
                            table_texts.append(", ".join(row))
        except Exception as plumber_e:
            print(f"PDFPlumber also failed: {plumber_e}")

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


def ocr_pdf(pdf_path):
    """Perform OCR on non-text PDFs."""
    images = convert_from_path(pdf_path)
    ocr_texts = [pytesseract.image_to_string(image, lang="eng") for image in images]
    return "\n".join(ocr_texts)

def extract_text_with_fitz(pdf_path):
    """Extract text with fallback to OCR."""
    try:
        document = fitz.open(pdf_path)
        text = ""
        for page in document:
            text += page.get_text("text")
        document.close()

        if not text.strip():  # Fallback if no selectable text
            print("Fallback to OCR as no selectable text found.")
            text = ocr_pdf(pdf_path)

        return clean_text(text)

    except Exception as e:
        print(f"Error with PyMuPDF: {e}. Falling back to OCR.")
        return ocr_pdf(pdf_path)
    
def remove_headers_footers(text, threshold=3):
    """Remove repeating headers and footers based on frequency."""
    lines = text.split("\n")
    freq = {}
    for line in lines:
        freq[line] = freq.get(line, 0) + 1
    cleaned_lines = [line for line in lines if freq[line] <= threshold]
    return "\n".join(cleaned_lines)

def preprocess_text(text):
    """Preprocess text to clean layout-based artifacts."""
    text = remove_headers_footers(text)
    return clean_text(text)

def extract_figures(pdf_path):
    """Extract images/figures from PDF."""
    document = fitz.open(pdf_path)
    figures = []
    for page in document:
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = document.extract_image(xref)
            image_bytes = base_image["image"]
            figures.append(image_bytes)
    document.close()
    return figures

def process_figure_with_clip(figure_bytes, figure_index):
    """Generate captions and embeddings for a figure using CLIP."""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(figure_bytes)).convert("RGB")

        # Generate embeddings
        inputs = clip_processor(images=image, return_tensors="pt")
        embeddings = clip_model.get_image_features(**inputs).detach().numpy()

        # Use embeddings as figure representation and generate a placeholder caption
        caption = f"Figure {figure_index + 1}: Semantic embedding added."

        return caption, embeddings

    except Exception as e:
        print(f"Error processing figure {figure_index + 1} with CLIP: {e}")
        return f"Figure {figure_index + 1}: Unable to process.", None
    
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

            parsed_texts = chunk_text(preprocess_text(full_text))
            parsed_texts += tables
            # Save figures (Optional: Store as metadata)
             # Process figures using CLIP
            if figures:
                print(f"Extracted {len(figures)} figures from the PDF.")
                for i, figure in enumerate(figures):
                    caption, clip_embedding = process_figure_with_clip(figure, i)

                    # Append caption to parsed_texts
                    parsed_texts.append(caption)

                    # Add figure embedding directly to the vector database
                    if clip_embedding is not None:
                        db.add_embeddings([clip_embedding], metadata={"caption": caption})


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
