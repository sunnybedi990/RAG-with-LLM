import os

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory for storing charts
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# Any other configuration constants
DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_DIR = os.path.join(BASE_DIR, "pdfs")
