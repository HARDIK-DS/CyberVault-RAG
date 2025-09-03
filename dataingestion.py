from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


loader = TextLoader('cybersecurity.txt', encoding='utf-8')
documents = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=0,
    separator=''
)

docs = splitter.split_documents(documents)
print(f"Total chunks created: {len(docs)}")


vectordb = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
vectordb.persist()

print("✅ Data ingested and stored in ChromaDB")

