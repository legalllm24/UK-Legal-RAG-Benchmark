if __name__ == "__main__":
    from base import Database
else:
    from .base import Database

import os
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv

dotenv_path = '.env'
load_dotenv(dotenv_path=dotenv_path)

class Database_Chroma(Database):
    def __init__(self, collection_names: List[str] = ['Uk', 'Wales', 'NorthernIreland', 'Scotland'], text_splitter=None, embedding_model=None) -> None:
        """
        Initializes a ChromaDB object.

        Args:
            collection_names (List[str], optional): A list of collection names. Defaults to ['Uk', 'Wales', 'NorthernIreland', 'Scotland'].

        Initializes the following attributes:
            - collections (List[str]): A list of collection names.
            - embeddings: An embedding model.
            - vector_stores (dict): A dictionary of Chroma instances for each collection.
        """
        if collection_names is None:
            collection_names = ['Uk', 'Wales', 'NorthernIreland', 'Scotland']

        super().__init__(text_splitter=text_splitter, embedding_model=embedding_model)
        self.collections = collection_names
        self.__initialize_vector_stores()

    def _get_client_collections(self) -> List[str]:
        """
        Retrieves all existing collections in the ChromaDB client.

        Returns:
            List[str]: A list of collection names.
        """
        existing_collections =  self.client.list_collections()
        return existing_collections
        
    def __initialize_vector_stores(self) -> None:
        """
        Initializes a Chroma vector store for each collection in the Database_Chroma object.
        
        Returns:
            None
        """
        self.vector_store = {}
        for collection_name in self.collections:
            self.vector_store[collection_name] = Chroma(collection_name=collection_name, 
                                                        embedding_function=self.embeddings, 
                                                        persist_directory=os.path.join('./','chromadb_persistent_clients', collection_name))
        print(f'ChromaDB Vector Stores Initialized for: {list(self.vector_store.keys())}')

    def add_text_to_db(self, collection_name: str, text: str, metadata: dict) -> None:
        """
        Adds text data to a specified collection in the ChromaDB database.

        Args:
            collection_name (str): The name of the collection to add the text to.
            text (str): The text to add to the collection.
            metadata (dict): Metadata associated with the text.

        Returns:
            None
        """
        # print(f"Adding text to collection: {collection_name}")
        self.vector_store[collection_name].add_texts(texts=text, metadatas=metadata)

    def delete_collection(self, collection_name: str) -> None:
        """
        Deletes a specified collection in the ChromaDB database.

        Args:
            collection_name (str): The name of the collection to delete.
        """
        try:
            print(f"Deleting collection: {collection_name}")
            self.client.delete_collection(collection_name)
        except:
            print(f"Collection {collection_name} does not exist.")

    def delete_all_collections(self) -> None:
        """
        Deletes all collections in the ChromaDB database.

        Returns:
            None
        """
        self.client.reset()
        print("All collections deleted.")