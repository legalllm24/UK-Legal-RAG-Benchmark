
# Handle relative imports
if __name__ == "__main__":
    from base import Database
else:
    from .base import Database

import os
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
dotenv_path = '.env'
load_dotenv(dotenv_path=dotenv_path)

class Database_Chroma(Database):
    """
    A subclass of Database that implements storage and retrieval 
    using ChromaDB as the vector database backend.
    """

    def __init__(self, collection_names: List[str] = ['Uk', 'Wales', 'NorthernIreland', 'Scotland'], 
                       text_splitter=None, 
                       embedding_model=None) -> None:
        """
        Initializes a Database_Chroma object.

        Args:
            collection_names (List[str], optional): List of collection names to initialize. 
                                                    Defaults to UK jurisdictions.
            text_splitter (str, optional): Text splitting method. Defaults to None.
            embedding_model (str, optional): Embedding model choice. Defaults to None.
        """
        if collection_names is None:
            collection_names = ['Uk', 'Wales', 'NorthernIreland', 'Scotland']

        super().__init__(text_splitter=text_splitter, embedding_model=embedding_model)
        self.collections = collection_names
        self.__initialize_vector_stores()

    def _get_client_collections(self) -> List[str]:
        """
        Retrieves all existing collections from the ChromaDB client.

        Returns:
            List[str]: List of collection names currently stored.
        """
        existing_collections = self.client.list_collections()
        return existing_collections
        
    def __initialize_vector_stores(self) -> None:
        """
        Initializes a Chroma vector store for each collection name provided.
        
        Returns:
            None
        """
        self.vector_store = {}
        for collection_name in self.collections:
            self.vector_store[collection_name] = Chroma(
                collection_name=collection_name, 
                embedding_function=self.embeddings, 
                persist_directory=os.path.join('./', 'chromadb_persistent_clients', collection_name)
            )
        print(f'ChromaDB Vector Stores Initialized for: {list(self.vector_store.keys())}')

    def add_text_to_db(self, collection_name: str, text: str, metadata: dict) -> None:
        """
        Adds a text entry to a specific collection in ChromaDB.

        Args:
            collection_name (str): Name of the collection to add the text to.
            text (str): The text content to add.
            metadata (dict): Associated metadata for the text.

        Returns:
            None
        """
        self.vector_store[collection_name].add_texts(texts=text, metadatas=metadata)

    def delete_collection(self, collection_name: str) -> None:
        """
        Deletes a specified collection from ChromaDB.

        Args:
            collection_name (str): Name of the collection to delete.

        Returns:
            None
        """
        try:
            print(f"Deleting collection: {collection_name}")
            self.client.delete_collection(collection_name)
        except Exception as e:
            print(f"Collection {collection_name} could not be deleted: {str(e)}")

    def delete_all_collections(self) -> None:
        """
        Deletes all collections in the ChromaDB client.

        Returns:
            None
        """
        self.client.reset()
        print("All collections deleted.")
