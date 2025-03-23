import os
import pandas as pd
from dotenv import load_dotenv
from transformers import pipeline
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Load API keys (if needed)
load_dotenv("keys.env")

# Load dataset
csv_file_path = "Brain Dead CompScholar Dataset.csv"
df = pd.read_csv(csv_file_path)

# Drop unnecessary columns
df = df.drop(columns=["OCR", "labels", "Unnamed: 7"], errors='ignore')

# Load original summaries from Summaries.xlsx
summaries_file_path = "Summaries.xlsx"
df_summaries = pd.read_excel(summaries_file_path)

# Set number of articles to evaluate
n = 5  # Modify this value as needed
df_subset = df.head(n)  # Select first 'n' articles

# Initialize Pegasus model for abstractive summarization
summarizer = pipeline("summarization", model="google/pegasus-xsum")

def generate_summary(text, max_length=100, min_length=20):
    """Generates a concise abstractive summary using Pegasus."""
    if pd.notna(text):
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    return "No content available"

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Initialize BLEU smoothing
smooth_fn = SmoothingFunction().method1

# Store results
results = []

# Generate summaries and evaluate
for _, row in df_subset.iterrows():
    paper_id = row["Paper Id"]
    combined_text = f"{row['Abstract']} {row['Conclusion']}"
    generated_summary = generate_summary(combined_text, max_length=100, min_length=20)

    # Get the corresponding original summary
    reference_summary = df_summaries[df_summaries["Paper Id"] == paper_id]["Original Summary"].values
    if len(reference_summary) == 0:
        continue  # Skip if no reference summary exists

    reference_summary = reference_summary[0]

    # Compute ROUGE scores
    rouge_scores = scorer.score(reference_summary, generated_summary)
    rouge_1 = rouge_scores["rouge1"].fmeasure
    rouge_2 = rouge_scores["rouge2"].fmeasure
    rouge_l = rouge_scores["rougeL"].fmeasure

    # Compute BLEU score
    reference_tokens = reference_summary.split()
    generated_tokens = generated_summary.split()
    bleu_score = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smooth_fn)

    # Store results
    results.append({
        "Paper Id": paper_id,
        "ROUGE-1": rouge_1,
        "ROUGE-2": rouge_2,
        "ROUGE-L": rouge_l,
        "BLEU": bleu_score
    })

# Convert to DataFrame and print results
df_results = pd.DataFrame(results)
print(df_results)
