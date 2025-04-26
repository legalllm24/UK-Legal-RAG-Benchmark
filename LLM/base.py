class LLM:
    def __init__(self, prompt:str=None) -> None:
        self.initialize_system_prompt(prompt)
        pass

    def _initiate_LLM(self) -> None:
        raise NotImplementedError("This method should be implemented by the subclass")

    def initialize_system_prompt(self, prompt:str=None) -> None:
        if prompt != None:
            self.system_prompt = prompt
        elif prompt == None:
            self.system_prompt = '''Who are you? You are a legal expert chatbot.
            What is your job? When given a legal document/piece of text and a question as a prompt, your job is to use that context, and generate a relavant response. 
            '''

    def _format_query(self, context:str, query:str) -> str:
        return f'Content: {context}\n\nPrompt: {query}\n\nAnswer: '.strip()
    
    def chat():
        raise NotImplementedError("This method should be implemented by the subclass")
