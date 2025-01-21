from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import pinecone
import os
from dotenv import load_dotenv
from langchain_practice import llm_inference

load_dotenv("keys.env")

file_path = "" #Add your pdf file path

loader = PyPDFLoader(file_path)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)

all_splits = text_splitter.split_documents(docs)
# print(len(all_splits))

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# vector_1 = embeddings.embed_query(all_splits[0].page_content)
# vector_2 = embeddings.embed_query(all_splits[1].page_content)

# print(len(vector_1))

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "semantic-search-demo"

index=pc.Index(index_name)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768, # Replace with your model dimensions
        metric="cosine", # Replace with your model metric
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) 
    )
    vector_store = PineconeVectorStore(embedding=embeddings, index=index)
    vector_store.add_documents(documents=all_splits)

else:
    vector_store=PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

query=input("Enter your question: ")

embedded_query=embeddings.embed_query(query)

results=index.query(vector=embedded_query, top_k=1, include_metadata=True)

context=str(list(results.matches[0].metadata.values())[3])

llm_response=llm_inference(context=context, query=query)

print(llm_response)
