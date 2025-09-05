# 🛡️ CyberVault - AI-powered Cybersecurity Assistant

CyberVault is an AI-powered chatbot designed to answer cybersecurity-related queries using **LangChain**, **Google Generative AI**, and **Chroma Vector Database**.  
It uses **retrieval-augmented generation (RAG)** to fetch the most relevant information before generating accurate, context-aware answers.

---

## 🚀 Features
- **Cybersecurity-Focused**: Provides accurate answers to cybersecurity queries.  
- **RAG Architecture**: Combines document retrieval and LLM-based reasoning.  
- **Vector Database Support**: Uses ChromaDB for storing and retrieving embeddings.  
- **Professional UI**: Clean, modern, and responsive interface built with Flask and HTML/CSS.  
- **Deployment-Ready**: Easily deployable on Render, Vercel, or any cloud platform.

---

## 🗂️ Project Structure
CyberVault-RAG/
│── app.py # Flask backend with LangChain pipeline
│── chroma_db/ # Chroma vector database storage
│── templates/
│ └── index.html # Frontend UI template
│── static/ # (Optional) for CSS, JS, and images
│── requirements.txt # Dependencies for the project
│── .env # Environment variables (API keys, etc.)
│── README.md # Project documentation


