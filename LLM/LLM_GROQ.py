if __name__ == "__main__":
    from base import LLM
else:
    from .base import LLM

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.abspath(os.getcwd()), 'LLM', '.env'))
class LLM_Groq(LLM):
    def __init__(self, model_name='llama-3.3-70b-versatile') -> None:
        '''
        Constructor for the Legal_LLM class.
        Inputs: 
            - None
        Outputs:
            - None
        '''
        super().__init__()
        self.model_name = model_name
        self.__initiate_client__()

    def __initiate_client__(self) -> None:
        '''
        Function to initiate the LLM.
        Inputs:
            - None
        Outputs:    

        '''
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if GROQ_API_KEY is None:
            raise ValueError("GROQ_API_KEY is not set.")
        else:
            self.__client = Groq(api_key=GROQ_API_KEY)

    def chat(self, context:str, query:str, max_new_tokens:int=256) -> str:
        '''
        Function to chat with the LLM.
        Inputs:
            - context (str): The context of the document.
            - query (str): The query to be asked.
            - max_new_tokens (int): The maximum number of new tokens to generate.
        Outputs:
            - output (str): The output response of the LLM.
        '''
        if self.__client:
            stream = self.__client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful legal assistant."
                                    "You are well versed in legal matters of the entire united kingdom(UK, Nothern Ireland, Scotland and Wales)."
                                    "You are able to provide answers to legal questions fo the general public for the entire UK."
                                    "You are provided with a detailed context and a query to answer from."
                                    "Whenever, you are asked a question, you are provided with some legal passages from the (UK, Nothern Ireland, Scotland and Wales) "
                                    " to provide the best possible answer. "
                                    "The answer should be straight forward, detailed with only the necessary information. "
                                    "If you donot know the answer, you should say so. Donot generate any false information. Donot give references. "
                    },
                    {
                        "role": "user",
                        "content": f"Context: {context}\nQuery: {query}\nResponse:\n",
                    }
                ],
                model=f"{self.model_name}",
                stream=True,
            )
            
            for chunk in stream:
                yield chunk.choices[0].delta.content

        else:
            raise ValueError("GROQ Client not initiated.")