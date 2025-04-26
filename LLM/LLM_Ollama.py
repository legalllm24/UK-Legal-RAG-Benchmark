

# Handle relative imports
if __name__ == "__main__":
    from base import LLM
else:
    from .base import LLM

import ollama

class LLM_Ollama(LLM):
    """
    Class that implements a Large Language Model (LLM) using the Ollama API.
    Extends the abstract LLM base class.
    """

    def __init__(self, model_name='llama3.1') -> None:
        """
        Initializes the LLM_Ollama client.

        Args:
            model_name (str, optional): Name of the model to use. Defaults to 'llama3.1'.
        """
        super().__init__()
        self.__initiate_LLM(model_name=model_name)

    def __initiate_LLM(self, model_name: str) -> None:
        """
        Initiates the Ollama LLM client by setting the model name.

        Args:
            model_name (str): Name of the model to use.

        Returns:
            None
        """
        self.model_name = model_name
        self.pipe = None  # Placeholder for future extension if needed

    def chat(self, context: str, query: str, max_new_tokens: int = 256) -> str:
        """
        Generates a streamed response from the Ollama model based on provided context and query.

        Args:
            context (str): Context document content.
            query (str): User query.
            max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 256.

        Yields:
            str: Streamed output response.
        """
        prompt = self._format_query(context, query)
        messages = [
            {"role": "system", "content": f"{self.system_prompt}"},
            {"role": "user", "content": f"{prompt}"}
        ]
        
        stream = ollama.chat(
            model=self.model_name,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            yield chunk['message']['content']
