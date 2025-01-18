import json
import logging
import uuid
from hashlib import sha1
from typing import Any, Iterable, Optional, Dict, List, Type
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger()
DEFAULT_K = 4  # Number of Documents to return.


class Embeddings:
    def get_text_embedding(self, text: str) -> List[float]:
        pass


class ManticoreSearchSettings(BaseSettings):
    proto: str = "http"
    host: str = "localhost"
    port: int = 9308

    username: Optional[str] = None
    password: Optional[str] = None

    table: str = "vector_index"

    column_map: Dict[str, str] = {
        "id": "id",
        "uuid": "uuid",
        "document": "document",
        "embedding": "embedding",
        "metadata": "metadata",
    }

    # A mandatory setting; currently, only hnsw is supported.
    knn_type: str = "hnsw"

    # A mandatory setting that specifies the dimensions of the vectors being indexed.
    knn_dims: Optional[int] = None  # Defaults autodetect

    # A mandatory setting that specifies the distance function used by the HNSW index.
    hnsw_similarity: str = "L2"  # Acceptable values are: L2, IP, COSINE

    # An optional setting that defines the maximum amount of outgoing connections
    # in the graph.
    hnsw_m: int = 16  # The default is 16.

    # An optional setting that defines a construction time/accuracy trade-off.
    hnsw_ef_construction: int = 100

    def get_connection_string(self) -> str:
        return self.proto + "://" + self.host + ":" + str(self.port)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="manticore_",
    )


class ManticoreSearch:
    def __init__(
        self,
        embedding: Embeddings,
        *,
        config: Optional[ManticoreSearchSettings] = None,
        **kwargs: Any,
    ) -> None:
        """
        ManticoreSearch HTTP API Wrapper

        Args:
            embedding (Embeddings): Text embedding model
            config (ManticoreSearchSettings): Configuration of ManticoreSearch Client
            **kwargs: Other keyword arguments will pass into Configuration of API client manticoresearch-python.
                See https://github.com/manticoresoftware/manticoresearch-python for more.
        """
        try:
            import manticoresearch.api as ENDPOINTS
            import manticoresearch.api_client as API
        except ImportError:
            raise ImportError(
                "Could not import manticoresearch python package. "
                "Please install it with `pip install manticoresearch-dev`."
            )

        try:
            from tqdm import tqdm
            self.pgbar = tqdm
        except ImportError:
            # Just in case if tqdm is not installed
            self.pgbar = lambda x, **kwargs: x

        super().__init__()

        self.embedding = embedding
        if config is not None:
            self.config = config
        else:
            self.config = ManticoreSearchSettings()

        assert self.config
        assert self.config.host and self.config.port
        assert self.config.column_map and self.config.table

        # assert (
        #     self.config.knn_type
        #     and self.config.knn_dims
        #     and self.config.hnsw_m
        #     and self.config.hnsw_ef_construction
        #     and self.config.hnsw_similarity
        # )

        for k in ["id", "embedding", "document", "metadata", "uuid"]:
            assert k in self.config.column_map

        # Detect embeddings dimension
        if self.config.knn_dims is None:
            self.dim: int = len(self.embedding.get_text_embedding("test"))
        else:
            self.dim = self.config.knn_dims

        # Initialize the schema
        self.schema = f"""\
        CREATE TABLE IF NOT EXISTS {self.config.table}(
            {self.config.column_map['id']} bigint,
            {self.config.column_map['document']} text indexed stored,
            {self.config.column_map['embedding']} float_vector \
                knn_type='{self.config.knn_type}' \
                knn_dims='{self.dim}' \
                hnsw_similarity='{self.config.hnsw_similarity}' \
                hnsw_m='{self.config.hnsw_m}' \
                hnsw_ef_construction='{self.config.hnsw_ef_construction}',
            {self.config.column_map['metadata']} json,
            {self.config.column_map['uuid']} text indexed stored
        )\
        """

        # Create a connection to ManticoreSearch
        self.configuration = API.Configuration(
            host=self.config.get_connection_string(),
            username=self.config.username,
            password=self.config.password,
            # disabled_client_side_validations=",",
            **kwargs,
        )
        self.connection = API.ApiClient(self.configuration)
        self.client = {
            "index": ENDPOINTS.IndexApi(self.connection),
            "utils": ENDPOINTS.UtilsApi(self.connection),
            "search": ENDPOINTS.SearchApi(self.connection),
        }

        # Create default schema if not exists
        self.client["utils"].sql(self.schema)

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        *,
        batch_size: int = 32,
        text_ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Insert more texts through the embeddings and add to the VectorStore.

        Args:
            texts: Iterable of strings to add to the VectorStore
            metadatas: Optional column data to be inserted
            batch_size: Batch size of insertion
            text_ids: Optional list of ids to associate with the texts

        Returns:
            List of ids from adding the texts into the VectorStore.
        """
        # Embed and create the documents
        ids = text_ids or [
            # See https://stackoverflow.com/questions/67219691/python-hash-function-that-returns-32-or-64-bits
            str(int(sha1(t.encode("utf-8")).hexdigest()[:15], 16))
            for t in texts
        ]
        transac = []
        for i, text in enumerate(texts):
            embed = self.embeddings.get_text_embedding(text)
            doc_uuid = str(uuid.uuid1())
            doc = {
                self.config.column_map["document"]: text,
                self.config.column_map["embedding"]: embed,
                self.config.column_map["metadata"]: metadatas[i] if metadatas else {},
                self.config.column_map["uuid"]: doc_uuid,
            }
            transac.append(
                {"replace": {"index": self.config.table, "id": ids[i], "doc": doc}}
            )

            if len(transac) == batch_size:
                body = "\n".join(map(json.dumps, transac))
                try:
                    self.client["index"].bulk(body)
                    transac = []
                except Exception as e:
                    logger.exception(f"Error indexing documents: {e}")

        if len(transac) > 0:
            body = "\n".join(map(json.dumps, transac))
            try:
                self.client["index"].bulk(body)
            except Exception as e:
                logger.exception(f"Error indexing documents: {e}")

        return ids

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        *,
        config: Optional[ManticoreSearchSettings] = None,
        text_ids: Optional[List[str]] = None,
        batch_size: int = 32,
        **kwargs: Any,
    ):
        ctx = cls(embedding, config=config, **kwargs)
        ctx.add_texts(
            texts=texts,
            embedding=embedding,
            text_ids=text_ids,
            batch_size=batch_size,
            metadatas=metadatas,
            **kwargs,
        )
        return ctx

    def __repr__(self) -> str:
        """
        Text representation for ManticoreSearch Vector Store, prints backends, username
        and schemas. Easy to use with `str(ManticoreSearch())`

        Returns:
            repr: string to show connection info and data schema
        """
        _repr = f"\033[92m\033[1m{self.config.table} @ "
        _repr += f"http://{self.config.host}:{self.config.port}\033[0m\n\n"
        _repr += f"\033[1musername: {self.config.username}\033[0m\n\nTable Schema:\n"
        _repr += "-" * 51 + "\n"
        for r in self.client["utils"].sql(f"DESCRIBE {self.config.table}")[0]["data"]:
            _repr += (
                f"|\033[94m{r['Field']:24s}\033[0m|\033["
                f"96m{r['Type'] + ' ' + r['Properties']:24s}\033[0m|\n"
            )
        _repr += "-" * 51 + "\n"
        return _repr

    def similarity_search(self, query: str, k: int = DEFAULT_K, **kwargs: Any) -> List[dict]:
        """Perform a similarity search with ManticoreSearch

        Args:
            query (str): query string
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.

        Returns:
            List[dict]: List of Documents
        """
        return self.similarity_search_by_vector(
            self.embedding.get_text_embedding(query), k, **kwargs
        )

    def similarity_search_by_vector(self, embedding: List[float], k: int = DEFAULT_K, **kwargs: Any) -> List[dict]:
        """
        Perform a similarity search with ManticoreSearch by vectors

        Args:
            embedding (List[float]): Embedding vector
            k (int, optional): Top K neighbors to retrieve. Defaults to 4.

        Returns:
            List[dict]: List of documents
        """
        # Build search request
        request = {
            "index": self.config.table,
            "knn": {
                "field": self.config.column_map["embedding"],
                "k": k,
                "query_vector": embedding,
            },
        }
        # Execute request and convert response to dict format
        try:
            response = self.client["search"].search(request, **kwargs)
            if response.hits is None:
                return []
            return [r for r in self.client["search"].search(request, **kwargs).hits.hits[:k]]
        except Exception as e:
            logger.error(f"\033[91m\033[1m{type(e)}\033[0m \033[95m{str(e)}\033[0m")
            return []

    @property
    def metadata_column(self) -> str:
        return self.config.column_map["metadata"]

    def count_records(self) -> int:
        """
        Returns the total number of records in the index using SQL query.

        Returns:
            int: Total count of records.
        """
        query = f"SELECT count(*) as count FROM {self.config.table}"
        try:
            response = self.client["utils"].sql(query, raw_response=False)
            return response[0]['hits']['hits'][0]['_source']['count']
        except Exception as e:
            logger.exception(e)
            return 0
