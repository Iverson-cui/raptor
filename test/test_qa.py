import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 1. Setup device (Apple Silicon)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# 2. Load Model & Tokenizer
model_name = "google/flan-t5-small"  # Fast, small version for testing
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

# 3. Prepare Input
# T5 uses prefixes like "question: " to know what task to perform
input_text = "question: tell me about transformers. context: Transformers is an open-source library developed by Hugging Face that provides state-of-the-art machine learning models primarily for natural language processing tasks. It includes implementations of various transformer architectures like BERT, GPT, T5, and more, enabling tasks such as text classification, translation, summarization, and question answering."
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# 4. Generate Answer Locally
outputs = model.generate(**inputs, max_new_tokens=50)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Answer: {answer}")
