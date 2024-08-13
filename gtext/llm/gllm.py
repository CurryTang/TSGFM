from gtext.llm.bedrockm import Bedrock


prompts = {
    "cora": {
        "background": "You are gonna classify a paper",
        "instruction": "Please classify the paper into one of the following categories:",
        "question": "with the following format: Answer: <your_answer>"
    },
    "citeseer": {
        "background": "You are gonna classify a paper",
        "instruction": "Please classify the paper into one of the following categories:",
        "question": "with the following format: Answer: <your_answer>"
    },
    "pubmed": {
        "background": "You are gonna classify a paper",
        "instruction": "Please classify the paper into one of the following categories:",
        "question": "with the following format: Answer: <your_answer>"
    },
    "arxiv": {
        "background": "You are gonna classify a paper",
        "instruction": "Please classify the paper into one of the following categories:",
        "question": "with the following format: Answer: <your_answer>"
    },
    "products": {
        "background": "You are gonna classify a product",
        "instruction": "Please classify the product into one of the following categories:",
        "question": "with the following format: Answer: <your_answer>"
    },
    "wikics": {
        "background": "You are gonna classify a Wikipedia article",
        "instruction": "Please classify the article into one of the following categories:",
        "question": "with the following format: Answer: <your_answer>"
    },
    "amazonratings": {
        
    }
}

class GraphLLM:
    def __init__(self, bedrock_name="anthropic.claude-3-haiku-20240307-v1:0", temperature=0, max_tokens=200, context_size=4096, sleep_time=.5, **kwargs):
        self.llm = Bedrock(bedrock_name=bedrock_name, temperature=temperature, max_tokens=max_tokens, context_size=context_size, sleep_time=sleep_time)
    
    
    def generate_text(self, prompt, max_new_tokens=10, choice_only=False):
        return self.llm.generate_text(prompt, max_new_tokens=max_new_tokens, choice_only=choice_only)
    
    
    def prompt(self):
        return """

        """
    
    
    