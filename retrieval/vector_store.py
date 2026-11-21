from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import uuid


# Where ChromaDB will save our data
PERSIST_DIRECTORY = "./chroma_db"

# Name of our vector database collection
COLLECTION_NAME = "documind_documents"


def get_embedding_model():
    """Load the MiniLM embedding model."""

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_vector_store(chunks):
    """Convert document chunks into embeddings and store them."""

    embeddings = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"documind_{uuid.uuid4().hex[:12]}",
        persist_directory=PERSIST_DIRECTORY
    )

    return vector_store


def search_documents(vector_store, query, k=3):
    """Find the most relevant document chunks for a query."""

    results = vector_store.similarity_search(
        query,
        k=k
    )

    return results
