from manticore import ManticoreSearch, ManticoreSearchSettings
from embedder import load_embedder_ollama, DEFAULT_EMBEDDER_NAME


class ManticoreSearchStore:

    def __init__(self):
        self.id = 'ManticoreSearchStore'

        # Load embeddings
        self.embedding_model = load_embedder_ollama(DEFAULT_EMBEDDER_NAME)

        # Initialize ManticoreSearch for 'nodes' table
        self.config = ManticoreSearchSettings(table="nodes", host="manticore")
        self.manticore = ManticoreSearch(embedding=self.embedding_model, config=self.config)

    def generate_response(self, messages):
        return self.manticore.similarity_search(messages[-1:])
