from flask import Flask, render_template, request
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment
load_dotenv()

app = Flask(__name__)



embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 50})


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)


prompt = PromptTemplate(
    template="""
    You are CyberVault, a cybersecurity assistant.

    Only use the context below to answer the question. 
    If the answer is not in the context, say clearly: 
    "The documents do not provide enough information to answer this fully."
<<<<<<< HEAD
    Do NOT mention "the provided text" or "according to context".
=======
>>>>>>> dev

    Context:
    {context}

    Question: {question}

    Answer (use bullet points if possible):
    """,
    input_variables=['context', 'question']
)

def format_docs(retrieved_docs):
    return "\n\n".join(
        f"From page {doc.metadata.get('page', 'unknown')}:\n{doc.page_content}"
        for doc in retrieved_docs
    )

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

main_chain = parallel_chain | prompt | llm | StrOutputParser()


@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    question = None
    if request.method == "POST":
        question = request.form.get("question")
        if question:
            answer = main_chain.invoke(question)
    return render_template("index.html", question=question, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)



parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

main_chain = parallel_chain | prompt | llm | StrOutputParser()


@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    question = None
    if request.method == "POST":
        question = request.form.get("question")
        if question:
            answer = main_chain.invoke(question)
    return render_template("index.html", question=question, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)

