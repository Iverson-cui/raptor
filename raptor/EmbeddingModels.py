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
