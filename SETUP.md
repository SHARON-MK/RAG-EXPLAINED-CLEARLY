## Setup Instructions

1. Clone the repository

2. Create virtual environment
   python -m venv venv

3. Activate environment

   Windows:
   venv\Scripts\activate

   Mac/Linux:
   source venv/bin/activate

4. Install dependencies
   pip install -r requirements.txt

5. Create a .env file and add:
   PINECONE_API_KEY=your_key
   PINECONE_INDEX_NAME=your_index

 6. Replace the DATA/data.txt with you data

 7. Run : python DATA/upload_data.py
    To store data in db

 8. Run: python -m BACKEND.app
    To run the app