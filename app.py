from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import traceback

# Load environment variables
load_dotenv()

from langchain_chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# ---------- FLASK APP ----------
app = Flask(__name__)

# ---------- HELPER FUNCTION ----------
def get_qa_chain():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5})
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    except Exception as e:
        print("Error initializing QA chain:", e)
        traceback.print_exc()
        return None


# ---------- ROUTES ----------
@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""
    if request.method == "POST":
        question = request.form.get("question")

        if question:
            try:
                qa_chain = get_qa_chain()
                if qa_chain is None:
                    answer = "Error: Could not initialize QA system."
                else:
                    response = qa_chain.invoke({"query": question})
                    answer = response.get("result", "").strip()

                    if not answer or "I don't know" in answer:
                        answer = "This topic seems outside the scope of cybersecurity. Please try another question."
            except Exception as e:
                print("Error in home() route:", e)
                traceback.print_exc()
                answer = f"An error occurred: {str(e)}"

    return render_template("index.html", answer=answer)


# ---------- API ROUTE FOR FRONTEND AJAX CALLS ----------
@app.route("/api/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        question = data.get("question", "")

        if not question:
            return jsonify({"error": "No question provided"}), 400

        qa_chain = get_qa_chain()
        if qa_chain is None:
            return jsonify({"error": "Failed to load QA chain"}), 500

        response = qa_chain.invoke({"query": question})
        answer = response.get("result", "").strip()

        if not answer or "I don't know" in answer:
            answer = "This topic seems outside the scope of cybersecurity. Please try another question."

        return jsonify({"answer": answer})
    except Exception as e:
        print("Error in /api/ask route:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------- RUN LOCALLY ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
