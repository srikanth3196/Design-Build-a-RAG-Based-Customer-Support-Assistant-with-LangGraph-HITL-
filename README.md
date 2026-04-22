RAG-Based Customer Support Assistant (LangGraph + Groq + HITL)

A production-style Retrieval-Augmented Generation (RAG) system that acts as an intelligent customer support assistant using PDF knowledge bases, vector search, LLM reasoning, and human-in-the-loop escalation.

 Features

- 📄 PDF document ingestion
- ✂️ Intelligent text chunking
- 🔍 Semantic search using embeddings
- 🧠 RAG-based answer generation
- ⚡ Groq LLM (Llama 3) integration
- 📊 Real confidence scoring using similarity metrics
- 🔁 LangGraph workflow orchestration
- 👤 Human-in-the-loop (HITL) escalation system

 System Architecture

User Query  
→ Embedding Generation  
→ Vector DB (ChromaDB) Retrieval  
→ Context Injection  
→ LLM (Groq Llama3) Response  
→ Confidence Scoring  
→ Decision Engine  
→ Answer OR Human Escalation  

 Tech Stack
- Python
- LangChain
- LangGraph
- ChromaDB
- HuggingFace Embeddings
- Groq API (Llama3-70B)
- PyPDF

 Project Structure
├── app.ipynb # Main Jupyter Notebook
├── sample.pdf # Knowledge base
├── chroma_db/ # Vector database storage
├── README.md


How It Works
 1. PDF Processing
- Loads PDF using PyPDFLoader
- Splits into chunks (800 tokens + overlap)

2. Embeddings
- Converts text into vectors using HuggingFace embeddings

3. Vector Storage
- Stores embeddings in ChromaDB

4. Retrieval
- Uses similarity search (Top-K retrieval)

 5. LLM Generation
- Groq Llama3 generates responses using retrieved context

 6. Confidence Scoring
Confidence is calculated using:
- Vector similarity score
- Context coverage ratio

7. Routing Logic
- High confidence → AI answer
- Low confidence → Human escalation

 Example Output
Input:
Enter your query: Can I get a refund after 6 months?

DEBUG → Scores: [1.7713977098464966, 1.7713977098464966, 1.8059463500976562]
DEBUG → Confidence: 0.29

 Escalating to human agent...
Human response: Refunds are only allowed within 30 days. Please contact support for exceptions

✅ Final Answer:
Refunds are only allowed within 30 days. Please contact support for exceptions


 Key Highlights

- Real-world RAG architecture
- Production-style workflow using LangGraph
- Explainable AI via confidence scoring
- Scalable modular design
- HITL integration for reliability
 Future Improvements

- Multi-document support
- Conversation memory
- Web UI (Streamlit / React)
- Feedback learning loop
- API deployment (FastAPI)

Built as part of an AI  project focusing on:
- RAG systems
- LLM orchestration
- Agentic workflows
- Production AI design
