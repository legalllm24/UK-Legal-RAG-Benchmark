# =====================
# LLM/LLM_GROQ.py
# =====================

# Handle relative imports
if __name__ == "__main__":
    from base import LLM
else:
    from .base import LLM

import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from the LLM/.env file
load_dotenv(dotenv_path=os.path.join(os.path.abspath(os.getcwd()), 'LLM', '.env'))

class LLM_Groq(LLM):
    """
    Class that implements a Large Language Model (LLM) using the Groq API.
    Extends the abstract LLM base class.
    """

    def __init__(self, model_name='llama-3.3-70b-versatile') -> None:
        """
        Initializes the LLM_Groq client.

        Args:
            model_name (str, optional): Name of the model to use. Defaults to 'llama-3.3-70b-versatile'.
        """
        super().__init__()
        self.model_name = model_name
        self.__initiate_client__()

    def __initiate_client__(self) -> None:
        """
        Initiates the Groq client using the provided API key.

        Raises:
            ValueError: If GROQ_API_KEY is not found in environment variables.
        """
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if GROQ_API_KEY is None:
            raise ValueError("GROQ_API_KEY is not set.")
        else:
            self.__client = Groq(api_key=GROQ_API_KEY)

    def chat(self, context: str, query: str, max_new_tokens: int = 256) -> str:
        """
        Generates a streamed response from the Groq LLM based on provided context and query.

        Args:
            context (str): Context document content.
            query (str): User query.
            max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 256.

        Yields:
            str: Streamed output response.
        """
        if self.__client:
            stream = self.__client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful legal assistant.\n"
                            "You are well-versed in legal matters of the United Kingdom (UK, Northern Ireland, Scotland, and Wales).\n"
                            "You assist the public by answering legal questions using provided legal context.\n"
                            "Only answer from provided context. Do not invent information. Do not reference external sources."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Context: {context}\nQuery: {query}\nResponse:\n",
                    }
                ],
                model=self.model_name,
                stream=True,
            )

            for chunk in stream:
                yield chunk.choices[0].delta.content
        else:
            raise ValueError("Groq client not initiated.")
