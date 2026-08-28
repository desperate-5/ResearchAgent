import os

from dotenv import load_dotenv
load_dotenv()

from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
from openai import OpenAI


class APIEmbeddingFunction(EmbeddingFunction[Documents]):
    """ChromaDB 自定义嵌入函数：调用 OpenAI 兼容 API 获取 embedding。

    通过环境变量配置：
    - EMBEDDING_MODEL: 嵌入模型名（默认 text-embedding-v4）
    - EMBEDDING_BASE_URL: API 地址（默认 DashScope 兼容端点）
    - EMBEDDING_API_KEY: API Key（默认复用 OPENAI_API_KEY）
    """

    def __init__(self):
        self._client = OpenAI(
            api_key=os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", "")),
            base_url=os.getenv("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        self._model = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
        # DashScope 嵌入 API 单次请求有 batch size 上限（不同账号/模型上限不同，实测 10 条），
        # 此处设为 8 留足余量，超过会自动分批调用
        self._batch_size = 8

    def __call__(self, input: Documents) -> Embeddings:
        # 替换换行符可提升检索质量（OpenAI 建议做法）
        cleaned = [text.replace("\n", " ") for text in input]
        embeddings = []
        for i in range(0, len(cleaned), self._batch_size):
            batch = cleaned[i:i + self._batch_size]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            embeddings.extend(d.embedding for d in resp.data)
        return embeddings


_embedding_fn: APIEmbeddingFunction | None = None


def _get_embedding_fn() -> APIEmbeddingFunction:
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = APIEmbeddingFunction()
    return _embedding_fn
