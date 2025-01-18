import os
import logging
from manticore import ManticoreSearch, ManticoreSearchSettings
from embedder import load_embedder_ollama
from markdown_to_graph import MarkdownToGraph
from hashlib import sha1
from vision import describe_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Settings
DATA_PATH = os.path.join(os.path.dirname(__file__), 'manual_files')
EMBEDDER_NAME = 'bge-m3:567m'

# Load embeddings
embedding_model = load_embedder_ollama(EMBEDDER_NAME)

# Initialize ManticoreSearch for 'nodes' table
nodes_config = ManticoreSearchSettings(table="nodes")
nodes_manticore = ManticoreSearch(embedding=embedding_model, config=nodes_config)

# Initialize ManticoreSearch for 'edges' table
edges_config = ManticoreSearchSettings(table="edges")
edges_manticore = ManticoreSearch(embedding=embedding_model, config=edges_config)

#
# 1. read list of files from folder
#

files = [entry.path for entry in os.scandir(DATA_PATH) if entry.is_file()]

#
# 2. if file is md -> convert to graph
# 2.1. extract embedding for each paragraph
# 2.2. save embedding to vector store
#

txt_texts = []
txt_metas = []
txt_ids = []
for file_path in files:
    if file_path.endswith('.md'):
        with open(file_path, 'r', encoding='utf-8') as f:
            graph = MarkdownToGraph.from_markdown(f.read()).to_dict()
            paragraphs = graph['nodes']
            for para in paragraphs:
                text = para['label']
                txt_texts.append(text)
                txt_metas.append({'file_path': file_path})
                txt_ids.append(int(sha1(text.encode("utf-8")).hexdigest()[:15], 16))

#
# 3. if file is image -> using vl model extract text
# 3.1. extract embedding for text
# 3.2. save embedding to vector store (with path to file in metadata)
#

image_texts = []
image_metas = []
image_ids = []
# for file_path in files:
#     if file_path.endswith(('.jpg', '.png', '.jpeg')):
#         text = describe_image(file_path)
#         image_texts.append(text)
#         image_metas.append({'file_path': file_path})

# Merge all arrays
all_texts = txt_texts + image_texts
all_metas = txt_metas + image_metas
all_ids = txt_ids + image_ids

#
# 4. Sage 'nodes'
#

# Save embedding to 'edges' index with metadata
nodes_manticore.add_texts(texts=all_texts, metadatas=all_metas, text_ids=all_ids)

#
# 5. add edges to 'edges' index
#

# # Now, add edges to the 'edges_index'
# # Read the graph edges and map them to Manticore node IDs
# for file_path in files:
#     if file_path.endswith('.md'):
#         with open(file_path, 'r', encoding='utf-8') as f:
#             graph = MarkdownToGraph.from_markdown(f.read())
#             edges = graph['edges']
#             for edge in edges:
#                 from_uuid = edge['from']
#                 to_uuid = edge['to']
#                 weight = edge.get('weight', 1)  # Default weight if not specified
#
#                 # Get Manticore node IDs from the mapping
#                 from_id = all_ids.get(from_uuid)
#                 to_id = all_ids.get(to_uuid)
#
#                 if from_id and to_id:
#                     # Add the edge to the 'edges_index'
#                     edges_manticore.add_edges(
#                         from_node_ids=[from_id],
#                         to_node_ids=[to_id],
#                         weights=[weight]
#                     )
#                     logger.info(f"Added edge: {from_id} -> {to_id} with weight {weight}")
#                 else:
#                     logger.warning(f"Could not find Manticore node IDs for edge: {from_uuid} -> {to_uuid}")
