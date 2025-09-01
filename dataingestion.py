from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader,TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema import Document
import re

load_dotenv()

# ---------------- STEP 1: Embeddings ----------------
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# ---------------- STEP 2: Load PDFs ----------------
loader=TextLoader('cybersecurity.txt',encoding='utf-8')
documents = loader.load()
splitter = CharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=0,
    separator=''
)

docs= splitter.split_documents(documents)


print(f"Total chunks created: {len(docs)}")

# ---------------- STEP 4: Store in Chroma ----------------
vectordb = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
vectordb.persist()
print("✅ PDF ingested and stored in ChromaDB")

# ---------------- STEP 5: Retriever ----------------
retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 100}
)


# ---------------- STEP 6: LLM ----------------
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
retrieved_docs = retriever.invoke("who is modi")
for i, d in enumerate(retrieved_docs, 1):
    print(f"\n--- Retrieved {i} ---\n{d.page_content[:500]}")

prompt = PromptTemplate(
    template="""
    You are CyberVault, a cybersecurity assistant.

    Only use the context below to answer the question. 
    If the answer is not in the context, say clearly: 
    "The documents do not provide enough information to answer this fully."

    Context:
    {context}

    Question: {question}

    Answer (use bullet points if possible):
    """,
    input_variables=['context','question']
)

# ---------------- STEP 8: Runnable Chain ----------------
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

# ---------------- STEP 9: Run Queries ----------------
question1 = "who is modi"
answer1 = main_chain.invoke(question1)
print("\n🔹 Answer 1:\n", answer1)

