from flask import Flask, render_template, request
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Flask App
app = Flask(__name__)

# Setup Embeddings and Vector DB
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5})
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ---------- ROUTES ----------
@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""
    if request.method == "POST":
        question = request.form.get("question")

        if question:
            try:
                # Run query through RetrievalQA
                response = qa_chain.invoke({"query": question})
                answer = response.get("result", "").strip()

                # Professional fallback if no answer found
                if not answer or "I don't know" in answer:
                    answer = "This topic seems outside the scope of cybersecurity. Please try another question."

            except Exception as e:
                answer = f"An error occurred: {str(e)}"

    return render_template("index.html", answer=answer)

# ---------- RUN LOCALLY ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
