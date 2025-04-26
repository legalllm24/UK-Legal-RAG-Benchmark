
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from typing import List, Union, Optional, Dict

from dotenv import load_dotenv
from openai import OpenAI
from ragatouille import RAGPretrainedModel

# Load environment variables
Path_ENV = os.path.join(os.path.abspath(os.getcwd()), 'Database', '.env')
load_dotenv(Path_ENV)

# Handle relative imports for module testing
if __name__ == "__main__":
    from .Database.Database_Chromadb import Database_Chroma
    from LLM.LLM_GROQ import LLM_Groq 
else:
    from Database.Database_Chromadb import Database_Chroma
    from LLM.LLM_GROQ import LLM_Groq 

class RAG_Bot:
    """
    Class that implements the RAG (Retrieval-Augmented Generation) system.
    Integrates database retrieval, LLM-based generation, and reranking.
    """
    def __init__(self, collection_names=['Uk', 'Wales', 'NothernIreland', 'Scotland'],
                       text_splitter='SpaCy',
                       embedding_model="Ollama",
                       llm_model="llama3.1"):
        """Initializes the RAG_Bot object with the vector database, LLM, and reranker."""
        self.vector_db = Database_Chroma(collection_names=collection_names, text_splitter=text_splitter, embedding_model=embedding_model)
        self.llm = LLM_Groq(model_name="llama-3.3-70b-versatile")
        self.reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    def add_text(self, collection_name, text, metadata=None):
        """
        Adds a text document along with optional metadata into a specified database collection.
        
        Args:
            collection_name (str): Target collection name.
            text (str): Text content to add.
            metadata (dict, optional): Metadata associated with the text.
        """
        self.vector_db.add_text_to_db(
            collection_name=collection_name,
            text=text,
            metadata=metadata
        )

    def __collection_routing(self, query) -> Union[str, List[str], None]:
        """
        Determines which collection(s) the query refers to.
        
        Args:
            query (str): User input query.

        Returns:
            List[str] or None: Matching collections or None if none detected.
        """
        def check_for_existence_of_collection_names(query:str, collection_names:List[str]=['unitedkingdom', 'wales', 'northernireland', 'scotland']) -> Union[str, None]:
            Existing_collection_names = []
            for collection_name in collection_names:
                if collection_name.lower() in query.lower():
                    Existing_collection_names.append(collection_name.lower())
            return Existing_collection_names if Existing_collection_names else None

        mentioned_collections = check_for_existence_of_collection_names(query)
        return mentioned_collections

    def format_docs(self, docs):
        """Formats a list of document objects into a string."""
        return "\n\n".join(doc.page_content for doc in docs)

    def __generate_multi_queries(self, query: str, k: int = 3) -> None: 
        """
        Generates multiple semantically varied queries to improve retrieval.

        Args:
            query (str): The base query.
            k (int): Number of query variations to generate.

        Returns:
            str: Concatenated queries.
        """
        openai_api_key = os.getenv('OPENAI_API_KEY')

        if not openai_api_key:
            raise ValueError('OpenAI API key is not set. Please set it in the environment variables.')

        client = OpenAI(api_key=openai_api_key)

        multi_query_system_prompt = f"""
        Generate {k} different but relevant variations of the following query.
        Do not change country names, legislative articles, numbers, or dates.
        Maintain meaning, word it differently.

        Original Query: "{query}"

        Respond with {k} queries in a single line separated by periods. No bullets, no newlines.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful legal law chatbot assistant."},
                    {"role": "user", "content": multi_query_system_prompt}
                ]
            )
            generated_text = response.choices[0].message.content.strip()
            generated_queries = generated_text.split("\n")
            return '\n'.join(generated_queries)

        except Exception as e:
            print(f"An error occurred during multi-query generation: {str(e)}")
            return []

    def query(self, 
              query:str,
              k:int=3, 
              max_new_tokens=1000, 
              multi_query=False,
              rerank=False,
              verbose=False,
              mode='infer'):
        """
        Queries the database and generates an answer using the LLM.

        Args:
            query (str): The user query.
            k (int): Number of documents to retrieve.
            max_new_tokens (int): Maximum length of generated answer.
            multi_query (bool): Whether to use multi-query generation.
            rerank (bool): Whether to rerank the retrieved documents.
            verbose (bool): Whether to print intermediate results.
            mode (str): "infer" (prints) or "eval" (returns).

        Returns:
            Depending on mode, prints or returns generated content.
        """
        Collection_to_query_from = self.__collection_routing(query)
        print(f'Collection_to_query_from: {Collection_to_query_from}')

        if not Collection_to_query_from:
            print('No collection mentioned in the query.')
            return None

        return self.__query_all(query=query, k=k,
                                collection_names=Collection_to_query_from,
                                max_tokens=max_new_tokens,
                                multi_query=multi_query,
                                rerank=rerank,
                                verbose=verbose,
                                mode=mode)

    def __query_all(self, 
                    query,
                    k=1,
                    collection_names:List[str]=['Uk', 'Wales', 'NothernIreland', 'Scotland'],
                    max_tokens=1000,
                    multi_query=False,
                    rerank=False,
                    verbose=False,
                    mode='infer'):
        """
        Handles retrieval and reranking for all selected collections.

        Args:
            query (str): Query string.
            k (int): Number of documents to retrieve.
            collection_names (List[str]): Target collections.
            max_tokens (int): Token budget for output.
            multi_query (bool): If True, generate multiple queries.
            rerank (bool): If True, apply reranking to retrieved docs.
            verbose (bool): If True, print retrieval details.
            mode (str): "infer" to print output, "eval" to return output.

        Returns:
            Varies depending on mode.
        """
        All_Retrieved_Documents = ''
        individual_docs = []
        individual_metadatas = []

        for collection_name in collection_names:
            print(f'Querying Collection: {collection_name}')
            current_db = self.vector_db.vector_store[collection_name]

            retriever = current_db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k,}
            )

            retrieved_docs = retriever.get_relevant_documents(query)

            if rerank:
                context_docs_content = [doc.page_content for doc in retrieved_docs]
                context_docs_metadata = [doc.metadata for doc in retrieved_docs]
                reranked_docs = self.reranker.rerank(query, context_docs_content, k=k//2 if k>10 else k)

                reranked_docs_content = []
                for doc in reranked_docs:
                    reranked_docs_content.append(doc['content'])
                    for idx, content in enumerate(context_docs_content):
                        if doc['content'] == content:
                            individual_metadatas.append(context_docs_metadata[idx])
                            break
                    individual_docs.append(doc['content'])
                context = reranked_docs_content

                if verbose:
                    print(f'The reranked retrieved documents are:')
                    for idx, doc in enumerate(reranked_docs):
                        print(f"Document {idx} - Score: {doc['score']} - Metadata: {retrieved_docs[doc['result_index']].metadata}")
            else:    
                context = self.format_docs(retrieved_docs)
                if verbose:
                    print(f'The retrieved documents are:')
                    for idx, doc in enumerate(retrieved_docs):
                        individual_docs.append(doc.page_content)
                        individual_metadatas.append(doc.metadata)
                        print(f"Document {idx} - Metadata: {doc.metadata}")

            All_Retrieved_Documents += f'''For the country: {collection_name}\nThe context documents are: {context}\n'''

        if multi_query:
            query = self.__generate_multi_queries(query=query, k=3)
            if verbose:
                print(f'Multi Query: {query}')

        response = self.llm.chat(context=f'{All_Retrieved_Documents}',
                                 query=f'{query}',
                                 max_new_tokens=max_tokens)

        if mode == 'infer':
            for chunk in response:
                print(chunk, end='', flush=True)
            return individual_metadatas
        elif mode == 'eval':
            output = ''
            for chunk in response:
                output += chunk
            return (output, individual_docs)



