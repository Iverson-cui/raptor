import logging
from abc import ABC, abstractmethod

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseEmbeddingModel(ABC):
    """
    Blueprint for embedding models.
    All embedding models should inherit from this class and implement the create_embedding method.
    """
    @abstractmethod
    def create_embedding(self, text):
        pass


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model="text-embedding-ada-002"):
        self.client = OpenAI()
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )

class SBertEmbeddingModel(BaseEmbeddingModel):
    """
    Using modern BERT embedding model from Nomic AI.
    This model is said to perform better than multi-qa-mpnet-base-cos-v1 on various tasks.
    """
    def __init__(self, model_name="nomic-ai/modernbert-embed-base"):
        self.model = SentenceTransformer(
            model_name, trust_remote_code=True, local_files_only=True
        )

    def create_embedding(self, text):
        return self.model.encode(text)


class MpnetBaseCosModel(BaseEmbeddingModel):
    """
    The original embedding model used in RAPTOR, based on Sentence-BERT.
    """

    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)


class BGEM3Model(BaseEmbeddingModel):
    """
    Implementation of BAAI/bge-m3.
    State-of-the-art for multilingual and long-context (8192 tokens).
    """

    def __init__(self, model_name="BAAI/bge-m3", device="cpu"):
        # BGE-M3 works excellently with SentenceTransformer wrapper
        print(f"Loading BGE-M3 from {model_name} on device {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = 2048  # set max length for long context

    def create_embedding(self, text):
        # BGE-M3 automatically handles the dense retrieval part
        # 'return_dense=True' is default in encode
        if isinstance(text, Exception) or (
            isinstance(text, list) and len(text) > 0 and isinstance(text[0], Exception)
        ):
            print(f"!!! 严重警告: 传入了异常对象而不是文本 !!!")
            print(f"Type: {type(text)}")
            print(f"Content: {text}")
            # 强制转换为字符串，或者在这里直接 return None 跳过
            return None
        # --- 调试代码结束 ---
        return self.model.encode(
            text,
            normalize_embeddings=True,  # Critical for Cosine Similarity
            batch_size=1,
        )
