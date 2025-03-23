import os
import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_practice import llm_inference

# Load API keys
load_dotenv("keys.env")

# Load dataset
csv_file_path = "Brain Dead CompScholar Dataset.csv"
df = pd.read_csv(csv_file_path)

# Drop unnecessary columns
df = df.drop(columns=["OCR", "labels", "Unnamed: 7"], errors='ignore')

# Set number of articles to summarize
n = 5  # Modify this value as needed
df_subset = df.head(n)  # Select the first 'n' articles

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "semantic-search-demo"

# Create or load Pinecone index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # Match your embedding model dimensions
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(embedding=embeddings, index=index)
else:
    vector_store = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    index = pc.Index(index_name)

# Define prompt for summarization
summary_prompt = """You are an AI language model specialized in generating concise and accurate research summaries.
Given the details of a research paper, generate a structured and well-formed summary including its main contributions, findings, and relevance.

Title: {title}
Keywords: {keywords}
Abstract: {abstract}
Conclusion: {conclusion}

Summary:"""

# Generate summaries
summaries = []
for _, row in df_subset.iterrows():
    context = summary_prompt.format(
        title=row["Paper Title"],
        keywords=row["Key Words"],
        abstract=row["Abstract"],
        conclusion=row["Conclusion"]
    )
    summary = llm_inference(context=context, query="Summarize this research paper.")
    summaries.append({"Paper Id": row["Paper Id"], "Title": row["Paper Title"], "Summary": summary})

# Print summaries
for item in summaries:
    print(f"\nPaper ID: {item['Paper Id']}")
    print(f"Title: {item['Title']}")
    print(f"Summary: {item['Summary']}")
