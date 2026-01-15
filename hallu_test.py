import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from raptor.QAModels import BaseQAModel


class LocalServerQAModel(BaseQAModel):
    """
    Generalized QA Model loader for local server deployment.
    Inherits from BaseQAModel to ensure compatibility with the RAPTOR framework.
    Optimized for NVIDIA A6000 GPUs on the server.
    """

    def __init__(
        self,
        model_path,
        device_map="auto",
        max_memory=None,
    ):
        """
        Args:
            model_path (str): Full path to the model or folder name in /opt/pretrained_models/
        """
        # Resolve path similarly to QwenQAModel logic
        base_dir = "/opt/pretrained_models"
        if not os.path.exists(model_path):
            augmented_path = os.path.join(base_dir, model_path)
            if os.path.exists(augmented_path):
                self.model_path = augmented_path
            else:
                self.model_path = model_path
        else:
            self.model_path = model_path

        print(f"Loading model from: {self.model_path}")

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"Detected device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        # Load model with A6000 optimized settings (bfloat16)
        # Using float16/float32 fallback for non-CUDA devices
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=device_map,
                max_memory=max_memory,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
        except Exception as e:
            print(
                f"Initial load with torch_dtype failed: {e}. Retrying with default precision..."
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=device_map,
                trust_remote_code=True,
            )

        self.model.eval()

    def answer_question(self, context, question, max_new_tokens=512):
        """
        Generates an answer using the provided context.
        Matches the BaseQAModel interface.
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Use the provided context to answer the question. If the context does not contain the answer, feel free to use your own knowledge.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}",
            },
        ]

        try:
            # Try applying chat template (standard for DeepSeek/Qwen/Llama3 etc.)
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Manual fallback if template is missing
            text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Remove input tokens from output
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]
        return response.strip()

    def answer_question_without_contexts(self, question, max_new_tokens=512):
        """
        Generates an answer without providing any context.
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {"role": "user", "content": f"Question:\n{question}"},
        ]

        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = f"Question: {question}\n\nAnswer:"

        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
            )

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]
        return response.strip()


if __name__ == "__main__":
    # Default model path
    MODEL_NAME = "/opt/pretrained_models/Qwen2-7B-Instruct"

    try:
        print(f"--- Loading Model: {MODEL_NAME} ---")
        qa_engine = LocalServerQAModel(model_path=MODEL_NAME)
        print("Model loaded successfully. Entering interactive mode.")
        print("Type 'exit' or 'quit' to stop.")

        while True:
            print("\n" + "="*30)
            question = input("Enter your question: ").strip()
            if question.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            
            if not question:
                continue

            # Optional context input
            context = input("Enter context (optional, press Enter to skip): ").strip()

            print("\nGenerating answer...")
            if context:
                answer = qa_engine.answer_question(context, question)
            else:
                answer = qa_engine.answer_question_without_contexts(question)

            print(f"\nAnswer:\n{answer}")

    except Exception as e:
        print(f"\n[Error]: {e}")
        print(
            "Note: This script is designed to run on the server. If run locally, ensure the model exists or ignore the path error."
        )
