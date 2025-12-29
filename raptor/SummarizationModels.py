import logging
import os
from abc import ABC, abstractmethod
import httpx
import torch

# import os
from transformers import AutoModelForCausalLM, AutoTokenizer

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseSummarizationModel(ABC):
    """
    Abstract base class for summarization models.
    """
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass


class GPT3TurboSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gpt-3.5-turbo"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e


class GPT3SummarizationModel(BaseSummarizationModel):
    def __init__(self, model="text-davinci-003"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e


class DeepSeekSummarizationModel(BaseSummarizationModel):
    def __init__(
        self, model="deepseek-chat", api_key=None, base_url="https://api.deepseek.com"
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = base_url

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):
        proxy_url = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy") or "http://127.0.0.1:7890"

        # Check if proxy is reachable or if we should even use it
        # For simplicity, if we are in an environment that doesn't need it, we should be able to skip.
        # But per current implementation, it's hardcoded. I'll make it use environment if available.

        try:
            # Use the newer httpx proxy configuration style
            http_client = httpx.Client(proxy=proxy_url) if proxy_url else httpx.Client()

            client = OpenAI(
                api_key=self.api_key, base_url=self.base_url, http_client=http_client
            )

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e


class QwenLocalSummarizationModel(BaseSummarizationModel):
    def __init__(
        self, model_root="/opt/pretrained_models/models--Qwen--Qwen2.5-32B-Instruct"
    ):
        """
        Initializes the local Qwen model directly from the HF cache directory.
        """
        self.model_root = model_root

        # 1. Resolve the actual model path inside 'snapshots'
        # The folder structure in your image is a Hugging Face cache.
        # We need the path: .../snapshots/<commit_hash>
        snapshots_dir = os.path.join(model_root, "snapshots")
        if os.path.exists(snapshots_dir):
            # Get the first subfolder inside snapshots (usually the commit hash)
            subfolders = [
                f
                for f in os.listdir(snapshots_dir)
                if os.path.isdir(os.path.join(snapshots_dir, f))
            ]
            if subfolders:
                self.model_path = os.path.join(snapshots_dir, subfolders[0])
            else:
                raise FileNotFoundError(f"No snapshot folder found in {snapshots_dir}")
        else:
            # Fallback: Maybe the user pointed directly to the inner folder?
            self.model_path = model_root

        print(f"Loading Qwen model from: {self.model_path}")

        # 2. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        # 3. Load Model
        # device_map="auto" allows it to use all available GPUs and offload to CPU if needed
        # torch_dtype=torch.float16 reduces memory usage by half compared to float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    def summarize(self, context, max_tokens=500, stop_sequence=None):
        try:
            # Construct the prompt using the official chat template
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant skilled at synthesizing information.",
                },
                {
                    "role": "user",
                    "content": f"Please provide a concise and comprehensive summary of the following text:\n\n{context}",
                },
            ]

            # Apply chat template (converts to string format model expects)
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Tokenize and move to the same device as the model
            model_inputs = self.tokenizer([text], return_tensors="pt").to(
                self.model.device
            )

            # Generate Summary
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

            # Decode only the new tokens (removing the input prompt)
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            summary = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            return summary

        except Exception as e:
            print(f"Error during local summarization: {e}")
            return str(e)
