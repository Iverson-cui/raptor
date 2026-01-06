import logging
from abc import ABC, abstractmethod

from openai import OpenAI
from scipy.__config__ import show
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


from sentence_transformers import SentenceTransformer
import torch


class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(
        self,
        model_name="nomic-ai/modernbert-embed-base",
        device="cuda:3",  # Changed default to match your likely usage
        target_gpus=["cuda:3", "cuda:4", "cuda:5", "cuda:6"],
    ):
        print(f"Loading Master Model on {device}...")
        self.device = device

        # 1. Load Manager
        self.model = SentenceTransformer(
            model_name, trust_remote_code=True, device=self.device
        )
        # ModernBERT supports 8192, but we can set a soft limit if desired
        self.model.max_seq_length = 2048

        # 2. Start Pool
        print(f"Starting Worker Pool on: {target_gpus}")
        try:
            self.pool = self.model.start_multi_process_pool(target_devices=target_gpus)
        except Exception as e:
            print(f"❌ Failed to start multi-GPU pool: {e}")
            print("Fallback: Code will run on single device (slow).")
            self.pool = None

    def create_embedding(self, text, is_query=False):
        # Define Prefix
        prefix = "search_query: " if is_query else "search_document: "

        # Case A: Single String (User Query)
        if isinstance(text, str):
            return self.model.encode(f"{prefix}{text}", normalize_embeddings=True)

        # Case B: List of Strings (Bulk Indexing)
        elif isinstance(text, list):
            # Filter and prefix
            clean_texts = [
                f"{prefix}{t}" for t in text if isinstance(t, str) and len(t) > 0
            ]

            if not clean_texts:
                return None

            # Check if pool is active
            if self.pool:
                return self.model.encode(
                    clean_texts,
                    pool=self.pool,
                    batch_size=128,  # 128 per GPU
                    normalize_embeddings=True,
                    show_progress_bar=True,
                )
            else:
                # Fallback if pool failed to start
                print("⚠️ Warning: Running large batch on single GPU!")
                return self.model.encode(
                    clean_texts,
                    batch_size=128,
                    normalize_embeddings=True,
                    show_progress_bar=True,
                )

        return []  # Fallback empty list

    def close(self):
        if self.pool:
            self.model.stop_multi_process_pool(self.pool)


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
