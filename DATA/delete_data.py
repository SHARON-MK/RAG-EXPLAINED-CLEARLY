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


def delete_all_records(namespace="default"):

    try:
        index.delete(delete_all=True, namespace=namespace)
        print("✅ Successfully deleted all records from Pinecone!")

    except Exception as e:
        print(f"\n{'='*60}")
        print("❌ ERROR during deletion!")
        print(f"{'='*60}")
        print(f"Error: {e}")
        print()


if __name__ == "__main__":

    delete_all_records(namespace="default")
    
 