import logging
import os

from openai import OpenAI


import getpass
from abc import ABC, abstractmethod

import torch
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
)


class BaseQAModel(ABC):
    """
    Abstract base class for question answering models.
    """
    @abstractmethod
    def answer_question(self, context, question):
        pass


class GPT3QAModel(BaseQAModel):
    def __init__(self, model="text-davinci-003"):
        """
        Initializes the GPT-3 model with the specified model version.

        Args:
            model (str, optional): The GPT-3 model version to use for generating summaries. Defaults to "text-davinci-003".
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):
        """
        Generates answer based on the given context using the GPT-3 model.

        Args:
            context (str): The text.
            max_tokens (int, optional): The maximum number of tokens in the generated answer. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop answering. Defaults to None.

        Returns:
            str: The generated answer.
        """
        try:
            response = self.client.completions.create(
                prompt=f"using the folloing information {context}. Answer the following question in less than 5-7 words, if possible: {question}",
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop_sequence,
                model=self.model,
            )
            return response.choices[0].text.strip()

        except Exception as e:
            print(e)
            return ""


class GPT3TurboQAModel(BaseQAModel):
    def __init__(self, model="gpt-3.5-turbo"):
        """
        Initializes the GPT-3 model with the specified model version.

        Args:
            model (str, optional): The GPT-3 model version to use for generating summaries. Defaults to "text-davinci-003".
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(
        self, context, question, max_tokens=150, stop_sequence=None
    ):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are Question Answering Portal"},
                {
                    "role": "user",
                    "content": f"Given Context: {context} Give the best full answer amongst the option to question {question}",
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):

        try:
            return self._attempt_answer_question(
                context, question, max_tokens=max_tokens, stop_sequence=stop_sequence
            )
        except Exception as e:
            print(e)
            return e


class GPT4QAModel(BaseQAModel):
    def __init__(self, model="gpt-4"):
        """
        Initializes the GPT-3 model with the specified model version.

        Args:
            model (str, optional): The GPT-3 model version to use for generating summaries. Defaults to "text-davinci-003".
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(
        self, context, question, max_tokens=150, stop_sequence=None
    ):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are Question Answering Portal"},
                {
                    "role": "user",
                    "content": f"Given Context: {context} Give the best full answer amongst the option to question {question}",
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):

        try:
            return self._attempt_answer_question(
                context, question, max_tokens=max_tokens, stop_sequence=stop_sequence
            )
        except Exception as e:
            print(e)
            return e


class UnifiedQAModel(BaseQAModel):
    """
    flan-t5-small based QA model, used for local PC inference
    """
    def __init__(self, model_name="google/flan-t5-small"):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def run_model(self, input_string, **generator_args):
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to(
            self.device
        )
        res = self.model.generate(input_ids, **generator_args)
        return self.tokenizer.batch_decode(res, skip_special_tokens=True)

    def answer_question(self, context, question):
        input_string = question + " \\n " + context
        output = self.run_model(input_string)
        return output[0]


class UnifiedQAModel_in_paper(BaseQAModel):
    """
    unifiedqa-v2-t5-3b-1363200 based QA model, used in the original paper and on server
    T5ForConditionalGeneration, T5Tokenizer are needed for encoder-decoder models.
    """

    def __init__(self, model_name="allenai/unifiedqa-v2-t5-3b-1363200"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def run_model(self, input_string, **generator_args):
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to(
            self.device
        )
        res = self.model.generate(input_ids, **generator_args)
        return self.tokenizer.batch_decode(res, skip_special_tokens=True)

    def answer_question(self, context, question):
        input_string = question + " \\n " + context
        output = self.run_model(input_string)
        return output[0]


class DeepSeekQAModel(BaseQAModel):
    """
    Implementation for DeepSeek-V2-Lite-Chat.
    Optimized for NVIDIA A6000 with bfloat16 and trust_remote_code.
    For decoder-only models, AutoModelForCausalLM, AutoTokenizer are needed.
    """

    def __init__(
        self,
        model_path="/opt/pretrained_models/DeepSeek-V2-Lite-Chat",
        device_map="auto",
        max_memory=None,
    ):
        """
        Args:
            model_path (str): Path to the model folder on your server.
                              e.g. "/path/to/DeepSeek-V2-Lite-Chat"
        """
        print(f"Loading DeepSeek model from {model_path}...")
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        # Load model
        # A6000 supports bfloat16 which is more stable for training/inference than float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            max_memory=max_memory,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()  # Set to evaluation mode

    def answer_question(self, context, question, max_new_tokens=512):
        # DeepSeek V2 specific system prompt structure
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer strictly based only on the provided context.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}",
            },
        ]

        # Apply chat template (handles the specific special tokens for DeepSeek)
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,  # Low temp for factual QA
                top_p=0.9,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response (skipping the input prompt)
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]) :], skip_special_tokens=True
        )
        return response.strip()


class QwenQAModel(BaseQAModel):
    """
    Implementation for Qwen2 or Qwen3 Instruct models.
    """

    def __init__(
        self,
        model_path="/opt/pretrained_models/Qwen2-7B-Instruct",
        device_map="auto",
        max_memory=None,
    ):
        """
        Args:
            model_path (str): Path to the model folder on your server.
                              e.g. "/path/to/Qwen2-7B-Instruct" or "Qwen3-14B"
        """
        # 1. Get the base path from environment, or use a default
        base_dir = os.environ.get("SERVER_MODEL_PATH")

        if base_dir:
            # Join the base path with the simple folder name
            augmented_model_path = os.path.join(base_dir, model_path)
        else:
            # Fallback: If variable isn't set, try using the name directly
            # (assuming full path was passed or it's a hub ID)
            augmented_model_path = model_path
        print(f"Loading Qwen from: {augmented_model_path}")

        # 2. Check if it exists (Good for debugging server issues)
        if not os.path.exists(augmented_model_path):
            # Try loading as a Hub ID if local path fails?
            # Or just raise error to prevent accidental downloads.
            print(f"Warning: Local path {augmented_model_path} does not exist.")
        print(f"Loading Qwen model from {augmented_model_path}...")
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            augmented_model_path, trust_remote_code=True
        )

        # Qwen models run excellently in bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            augmented_model_path,
            device_map=device_map,
            max_memory=max_memory,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()

    def answer_question(self, context, question, max_new_tokens=512):
        # Standard ChatML format for Qwen
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer the question using only the context provided.",
            },
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
            )

        # Slice output to remove input tokens
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]
        return response.strip()
