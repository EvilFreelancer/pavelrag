from llama_index.embeddings.ollama import OllamaEmbedding

DEFAULT_EMBEDDER_NAME = 'bge-m3:567m'


def load_embedder_ollama(
    model_name: str = DEFAULT_EMBEDDER_NAME,
    base_url: str = 'http://gpu02:11434',
    embed_batch_size=100,
) -> OllamaEmbedding:
    return OllamaEmbedding(
        model_name=model_name,
        base_url=base_url,
        embed_batch_size=embed_batch_size,
    )
