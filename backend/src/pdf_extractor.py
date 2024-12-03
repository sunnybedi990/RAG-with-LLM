import os
import re
import camelot
import fitz
import spacy
from pdf2image import convert_from_path
import pytesseract
import pdfplumber
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io

# Initialize NLP and CLIP model
nlp = spacy.load("en_core_web_sm")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def clean_text(text):
    """Clean text by removing extraneous symbols and whitespace."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\u2022", "-", text)  # Replace bullet points
    return text

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

def chunk_by_topics(text, max_length=512):
    """Chunk text based on topics and semantic structure."""
    doc = nlp(text)
    chunks, current_chunk = [], []
    current_length = 0

    for sentence in doc.sents:
        if current_length + len(sentence.text) > max_length or "Topic:" in sentence.text:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence.text)
        current_length += len(sentence.text)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def chunk_text_advanced(text, max_length=512, split_on=["\n\n", ". ", "? ", "! "]):
    """Chunk text into meaningful pieces based on topic, paragraph, and sentence boundaries."""
    chunks, current_chunk = [], ""
    current_length = 0

    for delimiter in split_on:
        segments = text.split(delimiter)
        for segment in segments:
            segment = segment.strip()
            if current_length + len(segment) > max_length:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_length = 0
            current_chunk += segment + delimiter
            current_length += len(segment) + len(delimiter)

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def chunk_text_by_semantics(text, max_length=512):
    """Chunk text into meaningful semantic units, respecting topic and paragraph boundaries."""
    doc = nlp(text)
    chunks, current_chunk = [], []
    current_length = 0

    for sentence in doc.sents:
        # Check for topic headers or new paragraphs
        if current_length + len(sentence.text) > max_length or any(
            keyword in sentence.text.lower() for keyword in ["introduction", "conclusion", "summary", "table of contents"]
        ):
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence.text)
        current_length += len(sentence.text)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

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

def extract_and_label_tables(pdf_path):
    """Extract tables and label them based on surrounding text context."""
    document = fitz.open(pdf_path)
    tables_with_labels = []

    for page_num, page in enumerate(document):
        # Extract text for context
        context_text = page.get_text("text")
        context_chunks = chunk_text_by_semantics(context_text)

        # Extract tables using PDFPlumber as fallback
        try:
            tables = camelot.read_pdf(pdf_path, pages=str(page_num + 1), flavor="stream")
        except Exception as e:
            print(f"Camelot failed on page {page_num}: {e}")
            tables = []

        for table in tables:
            # Map table to nearby text chunk
            nearby_context = context_chunks[min(len(context_chunks) - 1, page_num)]
            tables_with_labels.append({"table": table.df.to_string(), "context": nearby_context})

    return tables_with_labels


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

def ocr_pdf(pdf_path):
    """Perform OCR on non-text PDFs."""
    images = convert_from_path(pdf_path)
    ocr_texts = [pytesseract.image_to_string(image, lang="eng") for image in images]
    return "\n".join(ocr_texts)

def remove_headers_footers(text, threshold=3):
    """Remove repeating headers and footers based on frequency."""
    lines = text.split("\n")
    freq = {}
    for line in lines:
        freq[line] = freq.get(line, 0) + 1
    cleaned_lines = [line for line in lines if freq[line] <= threshold]
    return "\n".join(cleaned_lines)

def remove_headers_footers_advanced(text, page_count):
    """Remove headers and footers by analyzing text repetition across pages."""
    lines = text.split("\n")
    freq = {}
    page_length = len(lines) // page_count

    for i, line in enumerate(lines):
        position = i % page_length  # Position within a page
        if position < 3 or position > page_length - 3:  # Header/Footer range
            freq[line] = freq.get(line, 0) + 1

    cleaned_lines = [line for line in lines if freq.get(line, 0) < page_count // 2]
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
        image = Image.open(io.BytesIO(figure_bytes)).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt")
        embeddings = clip_model.get_image_features(**inputs).detach().numpy()
        caption = f"Figure {figure_index + 1}: Semantic embedding added."
        return caption, embeddings
    except Exception as e:
        print(f"Error processing figure {figure_index + 1} with CLIP: {e}")
        return f"Figure {figure_index + 1}: Unable to process.", None
    
def process_figures_with_captions(pdf_path):
    """Process images from the PDF and add captions using OCR and CLIP embeddings."""
    figures = extract_figures(pdf_path)
    processed_figures = []

    for idx, figure_bytes in enumerate(figures):
        caption, clip_embedding = process_figure_with_clip(figure_bytes, idx)
        # Use OCR for additional context
        image = Image.open(io.BytesIO(figure_bytes))
        ocr_text = pytesseract.image_to_string(image, lang="eng")
        processed_figures.append({"caption": caption, "embedding": clip_embedding, "ocr_text": ocr_text})

    return processed_figures

def extract_and_process_pdf(pdf_path):
    """Complete extraction workflow for text, tables, and figures."""
    full_text = extract_text_with_fitz(pdf_path)
    cleaned_text = preprocess_text(full_text)

    # Chunk text into semantic units
    text_chunks = chunk_text_by_semantics(cleaned_text)

    # Extract tables with labeled context
    tables = extract_and_label_tables(pdf_path)

    # Process figures for captions and embeddings
    figures = process_figures_with_captions(pdf_path)

    return {
        "text_chunks": text_chunks,
        "tables": tables,
        "figures": figures,
    }

def save_chunks_to_file(chunks, method_name, output_dir="parsed_chunks"):
    """
    Save text chunks to a file for comparison.
    Args:
        chunks (list): List of text chunks to save.
        method_name (str): Name of the parsing method.
        output_dir (str): Directory to save the chunk files.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    file_path = os.path.join(output_dir, f"{method_name}_chunks.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, 1):
            f.write(f"Chunk {i}:\n{chunk}\n\n")
    print(f"Chunks saved to {file_path}")


def extract_and_process_pdf_with_saving(pdf_path):
    """Complete extraction workflow with text chunk saving for comparison."""
    full_text = extract_text_with_fitz(pdf_path)
    cleaned_text = preprocess_text(full_text)

    # Chunk text into semantic units
    text_chunks = chunk_text_by_semantics(cleaned_text)
    save_chunks_to_file(text_chunks, "semantic_chunking")

    # Advanced chunking
    advanced_chunks = chunk_text_advanced(cleaned_text)
    save_chunks_to_file(advanced_chunks, "advanced_chunking")

    # Basic chunking
    basic_chunks = chunk_text(cleaned_text)
    save_chunks_to_file(basic_chunks, "basic_chunking")

    # Extract tables with labeled context
    tables = extract_and_label_tables(pdf_path)

    # Process figures for captions and embeddings
    figures = process_figures_with_captions(pdf_path)

    return {
        "text_chunks": text_chunks,
        "advanced_chunks": advanced_chunks,
        "basic_chunks": basic_chunks,
        "tables": tables,
        "figures": figures,
    }
def hybrid_chunking(text, max_length=512, advanced_split_on=["\n\n", ". ", "? ", "! "]):
    """
    Hybrid chunking function that combines semantic and advanced chunking strategies.
    
    Args:
        text (str): The input text to chunk.
        max_length (int): Maximum character length for a chunk.
        advanced_split_on (list): Delimiters for advanced chunking.
    
    Returns:
        List[str]: List of hybrid chunks.
    """
    # Determine document characteristics
    num_sentences = len(list(nlp(text).sents))
    avg_sentence_length = sum(len(sent.text) for sent in nlp(text).sents) / num_sentences

    # Thresholds to decide chunking strategy
    sentence_threshold = 100  # Number of sentences to switch to advanced chunking
    avg_length_threshold = 20  # Average sentence length to use semantic chunking

    # Select chunking method based on thresholds
    if num_sentences > sentence_threshold or avg_sentence_length < avg_length_threshold:
        print("Using advanced chunking.")
        return chunk_text_advanced(text, max_length, split_on=advanced_split_on)
    else:
        print("Using semantic chunking.")
        return chunk_text_by_semantics(text, max_length=max_length)

if __name__ == "__main__":
    pdf_path = "pdfs/Baljindersingh Bedi Resume.pdf"
    result = extract_and_process_pdf_with_saving(pdf_path)

    print("Text Chunks:")
    for i, chunk in enumerate(result["text_chunks"]):
        print(f"Chunk {i + 1}: {chunk}")

    print("\nTables with Context:")
    for i, table in enumerate(result["tables"]):
        print(f"Table {i + 1}:")
        print(f"Context: {table['context']}")
        print(f"Table Data: {table['table']}")

    print("\nFigures:")
    for i, figure in enumerate(result["figures"]):
        print(f"Figure {i + 1}:")
        print(f"Caption: {figure['caption']}")
        print(f"OCR Text: {figure['ocr_text']}")

