

import os
from dotenv import load_dotenv

# Load environment variables from .env located in the current file's directory
Path_ENV = os.path.abspath(__file__)
Path_ENV = os.path.dirname(Path_ENV)
load_dotenv(Path_ENV + '/.env')

def get_spacy_text_splitter(chunk_size: int = 2500, chunk_overlap: int = 300):
    """
    Returns a SpaCy-based text splitter for dividing documents into chunks.

    Args:
        chunk_size (int, optional): Size of each text chunk. Defaults to 2500.
        chunk_overlap (int, optional): Overlap between chunks. Defaults to 300.

    Returns:
        SpacyTextSplitter: Configured text splitter object.
    """
    from langchain.text_splitter import SpacyTextSplitter
    text_splitter = SpacyTextSplitter(chunk_size=chunk_size, 
                                      chunk_overlap=chunk_overlap, 
                                      max_length=2000000, 
                                      length_function=len)
    return text_splitter

def get_recursive_text_splitter(chunk_size: int = 2500, chunk_overlap: int = 300):
    """
    Returns a recursive character-based text splitter for document splitting.

    Args:
        chunk_size (int, optional): Size of each text chunk. Defaults to 2500.
        chunk_overlap (int, optional): Overlap between chunks. Defaults to 300.

    Returns:
        RecursiveCharacterTextSplitter: Configured text splitter object.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                    chunk_overlap=chunk_overlap, 
                                                    length_function=len)
    return text_splitter

def get_openai_embeddings():
    """
    Returns an OpenAI embedding model.

    Returns:
        OpenAIEmbeddings: Configured embedding model instance.
    """
    from langchain.embeddings.openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
    return embeddings

def get_sentence_transformers_embeddings():
    """
    Returns a SentenceTransformer embedding model.

    Returns:
        SentenceTransformerEmbeddings: Embedding model instance.
    """
    from langchain.embeddings import SentenceTransformerEmbeddings
    embeddings = SentenceTransformerEmbeddings()
    return embeddings

def get_google_genai_embeddings():
    """
    Returns a Google Generative AI embedding model (Gemini).

    Returns:
        GoogleGenerativeAIEmbeddings: Embedding model instance.
    """
    from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=os.getenv('GEMINI_API_KEY'),
        model="models/embedding-001"
    )
    return embeddings

def get_ollama_embeddings():
    """
    Returns an Ollama embedding model using the 'nomic-embed-text' model.

    Returns:
        OllamaEmbeddings: Embedding model instance.
    """
    from langchain_ollama import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings
