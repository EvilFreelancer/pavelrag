import os
import logging
import sys
import re
import asyncio
from typing import Optional
from aiogram import types, Bot, Dispatcher, html
from aiogram.filters import CommandStart
from aiogram.types import Message
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SimpleNodeParser, SentenceSplitter, MarkdownNodeParser, SentenceWindowNodeParser
from llama_index.core.postprocessor import LLMRerank, SentenceTransformerRerank
import faiss

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Global settings
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://gpu01:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3:567m")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-coder:7b-instruct-q4_0")
LLM_RERANKER_MODEL = os.getenv("LLM_RERANKER_MODEL", "qwen2.5-coder:7b-instruct-q4_0")
ST_RERANKER_MODEL = os.getenv("ST_RERANKER_MODEL", "BAAI/bge-reranker-large")
DOCS_FOLDER = os.getenv("DOCS_FOLDER", "./docs")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_index")
BASE_URL = os.getenv("BASE_URL", "https://manual.manticoresearch.com/")

# Initialize global settings
llm_model = Ollama(
    LLM_MODEL,
    base_url=OLLAMA_ENDPOINT,
    request_timeout=300,
    temperature=0,
    system_prompt=(
        "Ты ассистент для работы с документацией ManticoreSearch Engine. "
        "Твоя задача — отвечать строго в рамках переданной документации. "
        "Не выдумывай ответы и не выходи за пределы контекста."
    )
)
llm_reranker_model = Ollama(
    LLM_RERANKER_MODEL,
    base_url=OLLAMA_ENDPOINT,
    request_timeout=300,
    temperature=0,
    system_prompt=(
        "Ты реранкер, который работает исключительно в рамках документации ManticoreSearch Engine. "
        "Твоя задача — выбирать наиболее релевантные ответы из контекста документации. "
        "Игнорируй все, что выходит за пределы этой информации."
    )
)
embed_model = OllamaEmbedding(EMBED_MODEL, base_url=OLLAMA_ENDPOINT, request_timeout=300)
# text_splitter = SimpleNodeParser(chunk_size=128, chunk_overlap=20)
# text_splitter = MarkdownNodeParser()
# text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=25)
text_splitter = SentenceWindowNodeParser()

# Dimension of bge-m3
faiss_index = faiss.IndexFlatL2(1024)


def build_or_load_index():
    """
    Build or load existing FAISS index
    """
    if os.path.exists(FAISS_INDEX_PATH):
        vector_store = FaissVectorStore.from_persist_dir(FAISS_INDEX_PATH)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=FAISS_INDEX_PATH)
        index = load_index_from_storage(storage_context=storage_context, embed_model=embed_model)
        logging.info("[INFO] FAISS index loaded from file.")
    else:
        logging.info("[INFO] FAISS index not found, creating new one.")

        # Read documents and split them into nodes
        documents = SimpleDirectoryReader(input_dir=DOCS_FOLDER, required_exts=[".md"], recursive=True).load_data()
        nodes = text_splitter.get_nodes_from_documents(documents)

        # Init vector store and save nodes to index
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
        index.storage_context.persist(FAISS_INDEX_PATH)
        logging.info("[INFO] FAISS index saved.")
    return index


def create_source_url(file_path: str) -> str:
    """
    Create source URL for a file path
    """
    relative_path = os.path.relpath(file_path, DOCS_FOLDER)
    url_path = os.path.splitext(relative_path)[0]
    return os.path.join(BASE_URL, url_path)


# Initialize bot
dp = Dispatcher()


def escape_markdown(text: str, version: int = 1, entity_type: Optional[str] = None) -> str:
    if int(version) == 1:
        escape_chars = r"_*`["
    elif int(version) == 2:
        if entity_type in ["pre", "code"]:
            escape_chars = r"\`"
        elif entity_type in ["text_link", "custom_emoji"]:
            escape_chars = r"\)"
        else:
            escape_chars = r"\_*[]()~`>#+-=|{}.!"
    else:
        raise ValueError("Markdown version must be either 1 or 2!")
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer(f"Привет, {message.from_user.full_name}!")


# Process user messages
@dp.message()
async def handle_user_message(message: types.Message):
    user_query = message.text.strip()
    if not user_query:
        return

    # Query vector store
    query_engine = index.as_query_engine(
        llm=llm_model,
        # similarity_top_k=100,
        node_postprocessors=[
            # SentenceTransformerRerank(model=ST_RERANKER_MODEL, top_n=10, device="cpu"),
            # LLMRerank(llm=llm_reranker_model, choice_batch_size=5, top_n=2)
        ],
        response_mode="tree_summarize",
        system_prompt=(
            "Ты ассистент для работы с документацией ManticoreSearch Engine. "
            "Отвечай строго на основе переданной документации. "
            "Если ответ находится за пределами доступного контекста, сообщи, что ты не знаешь ответа."
        ),
    )
    response = query_engine.query(user_query)

    if not response.response.strip():
        await message.reply(
            "К сожалению, я могу отвечать только на вопросы в рамках документации Manticore Search.",
            parse_mode="Markdown"
        )
        return

    # Get sources of all received nodes
    unique_links = {create_source_url(node.node.metadata['file_path']) for node in response.source_nodes}

    # Build list of links to sources
    source_links = "\n".join(f"- {link}" for link in unique_links)

    # If sources not empty then add source links
    if source_links:
        source_links = f"\nСсылки:\n{escape_markdown(source_links)}"

    # Prepare final response
    final_response = f"{response.response}\n{source_links}".strip()
    await message.reply(final_response, parse_mode="Markdown")


async def main():
    global index
    index = build_or_load_index()
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
