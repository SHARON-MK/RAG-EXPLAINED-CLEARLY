import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from BACKEND.service import retrieve, augment, generate
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(parent_dir, '.env'))
app = Flask(__name__, 
            template_folder=os.path.join(parent_dir, 'UI'))


# ============================================
# ROUTES
# ============================================

@app.route('/')
def home():
    """
    Serve the main chatbot page
    """
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('question', '')
        
        
        # STEP 1: R of RAG - RETRIEVAL
        # Retrieve relevant data from db
        retrieved_chunks = retrieve(question, top_k=3)
        if not retrieved_chunks:
            return jsonify({
                'answer': 'I could not find relevant information to answer your question.'
            })
        

        # STEP 2: A of RAG - AUGMENTATION 
        # Combine retrieved data with system prompt and question to create augmented context
        augmented_context = augment(retrieved_chunks, question)
        
        
        # STEP 3: G of RAG - GENERATION 
        # Generate answer using LLM with augmented context
        answer = generate(augmented_context)
        

        return jsonify({
            'answer': answer,
            'sources': len(retrieved_chunks)
        })
    
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500



# ============================================
# RUN THE APP
# ============================================
if __name__ == '__main__':
    # Run Flask development server
    app.run(debug=True, host='0.0.0.0', port=5000)
