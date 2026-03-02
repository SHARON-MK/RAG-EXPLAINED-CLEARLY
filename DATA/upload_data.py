from email.mime import text
import re
import sys
import os
from pinecone import Pinecone
from dotenv import load_dotenv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

# Pinecone setup - Get credentials from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST")
if not PINECONE_API_KEY or not PINECONE_HOST:
    raise ValueError(
        "Missing required environment variables! Please check your .env file:\n"
        "- PINECONE_API_KEY\n"
        "- PINECONE_HOST"
    )
# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)


# ============================================
# TEXT CHUNKING FUNCTION
# ============================================
def chunk_text(text, chunk_size=150, overlap=30):

    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)

        start = end - overlap

    return chunks


# ============================================
# UPSERT DATA TO PINECONE
# ============================================
def upsert_documents():

    # Get Data path
    file_path = "DATA/data.txt" if os.path.exists("DATA/data.txt") else "data.txt"
    if not os.path.exists(file_path):
        raise FileNotFoundError("Could not find DATA/data.txt. Make sure you're running from the project root or DATA folder.")
    

    # Read the data and store in a variable
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split data into chunks
    chunks = chunk_text(text, chunk_size=150, overlap=30)
    
    # Prepare records in the format required by Pinecone
    records = []
    for i, chunk in enumerate(chunks):
        records.append({
            "_id": f"mentor_bro_chunk_{i}",  
            "text": chunk,  # This field is embedded into a vector, IMP : Exact field name should be mentioned when creating index
        })
    
    index.upsert_records( 
      namespace="default",
      records=records
    )
    
    print("✅ Successfully uploaded all chunks to Pinecone!")



if __name__ == "__main__":
    try:
        upsert_documents()
    except Exception as e:
        print("=" * 60)
        print("❌ ERROR during upload!")
        print("=" * 60)
        print(f"Error: {e}")
        print()
    
