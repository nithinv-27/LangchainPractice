import os
import pandas as pd
from dotenv import load_dotenv
from summarizer import Summarizer

# Load API keys (if needed for other integrations)
load_dotenv("keys.env")

# Load dataset
csv_file_path = "Brain Dead CompScholar Dataset.csv"
df = pd.read_csv(csv_file_path)

# Drop unnecessary columns
df = df.drop(columns=["OCR", "labels", "Unnamed: 7"], errors='ignore')

# Set number of articles to summarize
n = 5  # Modify this value as needed
df_subset = df.head(n)  # Select first 'n' articles

# Initialize BERT model for summarization
bert_model = Summarizer()

def generate_summary(text, num_sentences=5):
    """Generates a concise extractive summary using BERT."""
    return bert_model(text, num_sentences=num_sentences) if pd.notna(text) else "No content available"

# Generate summaries
for _, row in df_subset.iterrows():
    combined_text = f"{row['Abstract']} {row['Conclusion']}"
    extracted_summary = generate_summary(combined_text, num_sentences=5)

    print(f"\nPaper ID: {row['Paper Id']}")
    print(f"{extracted_summary}")
