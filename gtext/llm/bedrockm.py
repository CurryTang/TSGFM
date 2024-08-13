import os
from time import sleep

from tenacity import retry, stop_after_attempt, wait_random_exponential

from gtext.utils.basics import logger
from .llm import LLM
from llama_index.llms.bedrock import Bedrock as BedrockLLM


class Bedrock(LLM):
    def __init__(self, bedrock_name="anthropic.claude-3-haiku-20240307-v1:0", temperature=0, max_tokens=200, context_size=4096, sleep_time=.5, **kwargs):
        self.model = bedrock_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        aws_session_token = os.environ.get("AWS_SESSION_TOKEN")
        aws_region_name = "us-west-2"
        self.sleep_time = sleep_time
        self.llm = BedrockLLM(
            model=self.model,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=aws_region_name,
            context_size=context_size
        )
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate_text(self, prompt, max_new_tokens=10, choice_only=False):
        response = self.llm.complete(prompt, max_tokens=max_new_tokens).text
        sleep(self.sleep_time)
        return response