# AI-Shary-Cloud

AI-Shary-Cloud is a lightweight Python project that integrates retrieval-augmented generation (RAG) with vector search (Chroma) and a large language model backend for building AI-powered assistants.

Features
- RAG integration using Chroma vector store
- Support for Gemini/LLM integration (example integration files included)
- Redis session/cache helpers and simple memory management
- Dockerfile and Procfile for containerized and Heroku-style deployment

Repository structure
- main.py — application entry point
- gemini_rag_integration.py, rag_integration.py — RAG + LLM glue code
- chroma_rag_setup.py, gemini_chroma_setup.py — helpers to set up Chroma/vector store
- memory_manager.py, session_store.py, redis_utils.py — state/cache helpers
- templates/ — HTML templates for the web UI
- requirements.txt, Dockerfile, Procfile — deployment & dependencies

Prerequisites
- Python 3.9+ (or the Python version in runtime.txt)
- pip
- Optional: Docker for containerized deployment

Quickstart (local)
1. Clone the repo:

   git clone https://github.com/kirollos2001/ai-shary-cloud_final.git
   cd ai-shary-cloud_final

2. Create and activate a virtual environment:

   python -m venv .venv
   source .venv/bin/activate  # macOS / Linux
   .\.venv\Scripts\activate  # Windows

3. Install dependencies:

   pip install -r requirements.txt

4. Configure environment variables
   - Add any required API keys and configuration to environment variables or a .env file. Typical variables used by the project may include:
     - GEMINI_API_KEY or other LLM API key(s)
     - REDIS_URL (if using Redis)
     - Any Chroma or database configuration used by your deployment

5. Initialize any vector store or data loaders (if applicable):

   python chroma_rag_setup.py
   python gemini_chroma_setup.py

6. Run the app:

   python main.py

Running with Docker
1. Build the image:

   docker build -t ai-shary-cloud .

2. Run the container (example):

   docker run -e GEMINI_API_KEY=your_key -p 8080:8080 ai-shary-cloud

Heroku-style deployment
- A Procfile is included for deployments that use process files. Ensure you set required environment variables in your deployment platform.

Development notes
- Look at config.py and variables.py for configuration options used in the codebase.
- Use the setup scripts to populate the Chroma store before running the app if you expect retrieval to work out-of-the-box.

Contributing
Contributions, issues and feature requests are welcome. Please open issues or pull requests in the repository.

License
This repository does not include a license file. Add a LICENSE if you intend to change the default terms.

Contact
If you have questions about the project, open an issue or contact the repository owner (kirollos2001).