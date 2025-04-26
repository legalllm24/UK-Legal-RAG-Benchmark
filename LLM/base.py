

class LLM:
    """
    Abstract base class for implementing Large Language Models (LLMs).
    Defines a consistent interface for different LLM backends.
    """

    def __init__(self, prompt: str = None) -> None:
        """
        Initializes the base LLM object with an optional system prompt.

        Args:
            prompt (str, optional): Custom system prompt for the LLM. Defaults to None.
        """
        self.initialize_system_prompt(prompt)

    def _initiate_LLM(self) -> None:
        """
        Abstract method to initiate the LLM client.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by the subclass.")

    def initialize_system_prompt(self, prompt: str = None) -> None:
        """
        Initializes the system prompt used by the LLM.

        Args:
            prompt (str, optional): Custom prompt. If None, uses a default legal expert prompt.
        """
        if prompt is not None:
            self.system_prompt = prompt
        else:
            self.system_prompt = (
                "Who are you? You are a legal expert chatbot.\n"
                "What is your job? When given a legal document/piece of text and a question as a prompt, "
                "your job is to use that context and generate a relevant response."
            )

    def _format_query(self, context: str, query: str) -> str:
        """
        Formats the query and context into a structured input for the LLM.

        Args:
            context (str): Context document.
            query (str): User query.

        Returns:
            str: Formatted prompt ready for LLM input.
        """
        return f'Content: {context}\n\nPrompt: {query}\n\nAnswer: '.strip()
    
    def chat(self):
        """
        Abstract method for querying the LLM.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by the subclass.")
