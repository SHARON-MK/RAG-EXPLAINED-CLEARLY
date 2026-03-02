import os
import requests
from pinecone import Pinecone, SearchQuery
from dotenv import load_dotenv
load_dotenv()


# ============================================
# PINECONE SETUP
# ============================================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST")
if not PINECONE_API_KEY or not PINECONE_HOST:
    raise ValueError(
        "Missing required environment variables! Please check your .env file:\n"
        "- PINECONE_API_KEY\n"
        "- PINECONE_HOST"
    )
# Initialize Pinecone client and index
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)


# ============================================
# RETRIEVAL FUNCTION
# ============================================
def retrieve(query, top_k=3):

    results = index.search(
        namespace="default",
        query=SearchQuery(
            inputs={"text": query},
            top_k=top_k
        ),
        fields=["text"]  # Here mentions what all fields we need.
    )
    actual_results = results.get('result', {}).get('hits', [])
    print(f"🔍 Retrieved {len(actual_results)} chunks from Pinecone for the query: '{query}'")
    

    # Extract the text chunks from search results
    # Response format: results['result']['hits']
    # IMP: Here we are only extracting text, for better handling 'score' can be extracted.
    retrieved_chunks = []
    for hit in results.get('result', {}).get('hits', []):
        text = hit.get('fields', {}).get('text', '')
        if text:
            retrieved_chunks.append(text)
    
    return retrieved_chunks


# ============================================
# AUGMENTATION FUNCTION
# ============================================
def augment(retrieved_chunks, question):
 
    # Combine all retrieved chunks into context
    context = "\n\n".join(retrieved_chunks)
    
    # ⚡ Trim context if too long (for faster LLM response)
    MAX_CONTEXT_LENGTH = 1000  # characters
    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH] + "..."
    
    # Create augmented prompt with system message, context, and question
    augmented_context = f"""You are a friendly and helpful representative from MENTOR BRO. Your role is to assist users with their questions in a warm, professional, and conversational manner.

Available Information:
{context}

User's Message:
{question}

RESPONSE GUIDELINES:

1. **Answer ONLY What Was Asked**: 
   - Give a SPECIFIC answer to the exact question asked
   - Do NOT provide general information or background unless directly asked
   - Do NOT add extra details that weren't requested
   - Stay focused on the question - no tangents

2. **Greetings**: If the user just greeted you (hi, hello, hey, etc.), respond warmly and briefly. Welcome them and ask how you can help.

3. **Be Natural**: Never mention "context", "database", "retrieved information", or technical terms. Speak naturally as if you work at MENTOR BRO and have this knowledge.

4. **When Information is Unavailable**: If the available information doesn't contain the answer, simply say "I don't have those specific details available at the moment."

5. **Keep It SHORT**: 
   - Maximum 2-3 sentences or 3-4 bullet points
   - Answer the question directly and stop
   - No lengthy explanations
   - Be concise and precise

6. **Formatting**: Use proper Markdown formatting:
   - Use headings (## for main sections) only if needed
   - Use bullet points (-) for lists
   - Use **bold** for key terms
   - Use `code` for technical terms

7. **Tone**: Be friendly, helpful, and professional but BRIEF.

Now answer ONLY what the user asked - nothing more:"""
    
    print(f"✨ Augmented context created with {len(retrieved_chunks)} chunks")
    
    return augmented_context


# ============================================
# GENERATION FUNCTION
# ============================================
def generate(augmented_context):
    """
    Generate answer using LLM with augmented context.
    
    This is the GENERATION step in RAG.
    
    Args:
        augmented_context: The complete prompt with system message, context, and question
    
    Returns:
        str: The AI-generated answer or user-friendly error message
    """
    # Ollama API endpoint
    url = "http://localhost:11434/api/generate"
    
    # Prepare the request
    payload = {
        "model": "gemma3:1b",
        "prompt": augmented_context,
        "stream": False,
        "options": {
            "num_ctx": 2048,      # Smaller context window = faster
            "temperature": 0.7,    # Slightly lower for consistency
            "num_predict": 256     # Limit response length for speed
        }
    }
    
    print("🤖 Generating response with LLM...")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        answer = result.get("response", "")
        
        if not answer:
            print("⚠️ Empty response from LLM")
            return "I apologize, but I'm unable to generate a response right now. Please try again in a moment."
        
        print("✅ Response generated successfully")
        return answer
    
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Cannot reach Ollama server")
        return "I apologize, but our AI service is currently unavailable. Please ensure the Ollama service is running and try again."
    
    except requests.exceptions.Timeout:
        print("❌ Timeout Error: Ollama took too long to respond")
        return "I apologize, but the response is taking longer than expected. Please try again with a simpler question."
    
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        if e.response.status_code == 404:
            return "I apologize, but the AI model (gemma3:1b) is not available. Please check if it's installed in Ollama."
        else:
            return f"I apologize, but there was an error processing your request. (Error code: {e.response.status_code})"
    
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return "I apologize, but something went wrong. Please try again or contact support if the issue persists."
