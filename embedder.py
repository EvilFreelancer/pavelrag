# import torch
# from transformers import AutoModel
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding

# DEFAULT_MODEL_NAME = 'jinaai/jina-embeddings-v2-base-zh'
# DEFAULT_MODEL_NAME_GGUF = 'cwchang/jina-embeddings-v2-base-zh:q8_0'


# def load_embedder_transformer(model_name: str = DEFAULT_MODEL_NAME, device=None):
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
#     model.to(device)
#     return model

# def load_embedder_llama(model_name: str = DEFAULT_MODEL_NAME, device=None) -> HuggingFaceEmbedding:
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     return HuggingFaceEmbedding(
#         model_name=model_name,
#         trust_remote_code=True,
#         model_kwargs={"torch_dtype": torch.bfloat16},
#         device=device,
#     )

def load_embedder_ollama(
        model_name: str = 'bge-m3:567m',
        base_url: str = 'http://gpu02:11434',
        embed_batch_size=100,
) -> OllamaEmbedding:
    return OllamaEmbedding(
        model_name=model_name,
        base_url=base_url,
        embed_batch_size=embed_batch_size,
    )
