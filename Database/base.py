
from typing import List

# Handle relative imports
if __name__ == "__main__":
    from utils import *
else:
    from .utils import *

import os
from dotenv import load_dotenv

# Load environment variables from .env located in the current file's directory
Path_ENV = os.path.abspath(__file__)
Path_ENV = os.path.dirname(Path_ENV)
load_dotenv(Path_ENV + '/.env')

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain.docstore.document import Document

class Database:
    """
    Abstract base class for implementing a database interface 
    for vector-based document storage and retrieval.
    """

    def __init__(self, text_splitter=None, embedding_model=None) -> None:
        """
        Initializes the base Database object.

        Args:
            text_splitter (str, optional): Text splitting method ('SpaCy' or 'Recursive').
            embedding_model (str, optional): Embedding model to use for vectorization.

        Raises:
            Exception: If invalid text splitter or embedding model is provided.
        """
        # Initialize the text splitter
        if text_splitter is None or text_splitter == 'Recursive':
            self.text_splitter = get_recursive_text_splitter()
        elif text_splitter.lower() == 'spacy':
            self.text_splitter = get_spacy_text_splitter()
        else:
            raise Exception('Invalid Text Splitter. Choose from SpaCy or Recursive.')

        # Initialize the embedding model
        if embedding_model is None or embedding_model == 'SentenceTransformers': 
            self.embeddings = get_sentence_transformers_embeddings()
        elif embedding_model.lower() == 'openai':
            self.embeddings = get_openai_embeddings()
        elif embedding_model.lower() in ['google', 'gemini']:
            self.embeddings = get_google_genai_embeddings()
        elif embedding_model.lower() == 'ollama':
            self.embeddings = get_ollama_embeddings()
        else:
            raise Exception('Invalid Embedding Model. Choose from OpenAI, SentenceTransformers, Google, or Ollama.')

    def validate_collection(self):
        """
        Validates the existence or health of collections in the database.

        Raises:
            Exception: This method must be implemented by subclasses.
        """
        raise Exception('This method should be implemented by the subclass.')

    def _initialize_clients(self) -> None:
        """
        Initializes the database client(s).

        Raises:
            Exception: This method must be implemented by subclasses.
        """
        raise Exception('This method should be implemented by the subclass.')

    def _get_client_collections(self) -> List[str]:
        """
        Retrieves all collections present in the database.

        Returns:
            List[str]: List of collection names.

        Raises:
            Exception: This method must be implemented by subclasses.
        """
        raise Exception('This method should be implemented by the subclass.')

    def _verify_collections_existence_in_client(self) -> bool:
        """
        Verifies whether the expected collections exist in the client database.

        Returns:
            bool: True if collections exist, False otherwise.

        Raises:
            Exception: This method must be implemented by subclasses.
        """
        raise Exception('This method should be implemented by the subclass.')

    def _initialize_vector_stores(self) -> None:
        """
        Initializes vector stores for each collection.

        Raises:
            Exception: This method must be implemented by subclasses.
        """
        raise Exception('This method should be implemented by the subclass.')

    def add_text_to_db(self) -> None:
        """
        Adds a text document to a database collection.

        Raises:
            Exception: This method must be implemented by subclasses.
        """
        raise Exception('This method should be implemented by the subclass.')

    def delete_collection(self):
        """
        Deletes a specified collection from the database.

        Raises:
            Exception: This method should be implemented by subclasses.
        """
        raise Exception('This method should be implemented by the subclass.')

    def delete_all_collections(self) -> None:
        """
        Deletes all collections from the database client.

        Raises:
            Exception: This method should be implemented by subclasses.
        """
        raise Exception('This method should be implemented by the subclass.')
