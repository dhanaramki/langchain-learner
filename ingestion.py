import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    print("Ingestion....")
    loader = TextLoader(
        "C:\\Users\\dhana\\Documents\\RAG\\mediumblog1.txt", encoding="utf-8"
    )
    documents = loader.load()

    print("Splitting....")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print(f"Number of chunks: {len(texts)}")

    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",  # or any OpenRouter embedding model
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    print("Creating vector store....")
    vectorstore = PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.getenv("INDEX_NAME")
    )
    print("Ingestion complete.")
