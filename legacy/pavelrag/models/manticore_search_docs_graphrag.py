from pavelrag.stores.manticore import ManticoreSearch, ManticoreSearchSettings
from pavelrag.load_embedder import load_embedder, DEFAULT_EMBEDDER_NAME


class ManticoreSearchDocsGraphRAG:

    def __init__(self):
        self.id = 'ManticoreSearchDocs:GraphRAG'
        self.created = "2025-01-19"
        self.object = "model"
        self.owned_by = "Pavel Rykov"

        # Load embeddings
        self.embedding_model = load_embedder(DEFAULT_EMBEDDER_NAME)

        # Initialize ManticoreSearch for 'nodes' table
        self.config = ManticoreSearchSettings(table="nodes", host="manticore")
        self.manticore = ManticoreSearch(embedding=self.embedding_model, config=self.config)

    def generate_response(self, messages):
        # Get text from last message
        query = messages[-1:][0]['content']

        # Get k-nearest documents
        documents = self.manticore.similarity_search(query, 30)

        # Generate response
        response = []
        for document in documents:
            response.append(
                f"Document:\n{document['_source']['document']}\nSource:\n{document['_source']['metadata']['file_path']}\n\n"""
            )
        return ''.join(response).strip()
