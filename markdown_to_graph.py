from __future__ import annotations

import json
from bs4 import BeautifulSoup, NavigableString
import markdown
import re


class MarkdownToGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.node_id_counter = 1
        self.tree: str = ''
        # Stack to keep track of parent nodes (chapters and subchapters)
        self.parent_stack = []
        # Keep track of previous chapter node ID to connect chapters sequentially
        self.prev_chapter_id = None
        # Keep track of previous paragraph node ID to connect paragraphs sequentially within a chapter
        self.prev_paragraph_id = None
        # Regular expression pattern to find URLs in text
        self.url_pattern = re.compile(
            r'(https?://[^\s]+)'
        )

    def _add_node(self, label: str) -> int:
        """
        Add a node to the graph with the given label.
        """
        node_id = self.node_id_counter
        self.nodes.append({'id': node_id, 'label': label})
        self.node_id_counter += 1
        return node_id

    def _get_node_by_label(self, label: str) -> int | None:
        """
        Get the ID of an existing node with the given label, or None if it doesn't exist.
        """
        for node in self.nodes:
            if node['label'] == label:
                return node['id']
        return None

    def _get_or_create_node(self, label: str) -> int:
        """
        Get the ID of an existing node with the given label, or create a new one if it doesn't exist.
        """
        node_id = self._get_node_by_label(label)
        if node_id is not None:
            return node_id
        else:
            return self._add_node(label)

    def _get_or_create_link_node(self, link_url: str) -> int:
        """
        Get the ID of an existing link node with the given URL, or create a new one if it doesn't exist.
        """
        label = f'link:{link_url}'
        return self._get_or_create_node(label)

    def _get_or_create_image_node(self, img_url: str, img_alt: str) -> int:
        """
        Get the ID of an existing image node, or create a new one if it doesn't exist.
        """
        label = f'image:{img_url}' if not img_alt else img_alt
        return self._get_or_create_node(label)

    def _add_edge(self, from_id: int, to_id: int, weight: float = 0):
        """
        Add an edge to the graph from 'from_id' to 'to_id' with the specified weight.
        """
        self.edges.append({'from_id': from_id, 'to_id': to_id, 'weight': weight})

    def _create_graph(self, root):
        """
        Create the graph by processing the root BeautifulSoup element.
        """
        self._process_elements(root.contents)

    def _process_elements(self, elements):
        """
        Process a list of BeautifulSoup elements.
        """
        for element in elements:
            self._process_element(element)

    def _process_element(self, element):
        """
        Process a single BeautifulSoup element.
        """
        if isinstance(element, NavigableString):
            # Ignore navigable strings that are just whitespace
            if not element.strip():
                return
        elif element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            # Process heading elements
            self._process_heading(element)
        elif element.name == 'p':
            # Process paragraph elements
            self._process_paragraph(element)
        else:
            # Recursively process child elements
            self._process_elements(element.contents)

    def _process_heading(self, heading):
        """
        Process a heading element, create a node, and maintain the chapter hierarchy.
        """
        # Get the heading level from the tag name (e.g., 'h1' -> 1)
        level = int(heading.name[1])
        # Get the text content of the heading
        label = heading.get_text().strip()
        # Create or get the node for the heading
        node_id = self._get_or_create_node(label)

        # If this is a top-level chapter (level 1)
        if level == 1:
            # Connect to the previous chapter node with an edge of weight 1
            if self.prev_chapter_id is not None:
                self._add_edge(self.prev_chapter_id, node_id, weight=1)
            # Update the previous chapter ID
            self.prev_chapter_id = node_id
            # Reset the parent stack for new top-level chapter
            self.parent_stack = []

        # Adjust the parent stack to match the current heading level
        while len(self.parent_stack) >= level:
            self.parent_stack.pop()

        # If there is a parent chapter, connect the current chapter to it
        if self.parent_stack:
            parent_id = self.parent_stack[-1]
            # Connect current node to parent node (simulate hyperedge with weight=0)
            self._add_edge(parent_id, node_id, weight=0.1)

        # Add the current node to the parent stack
        self.parent_stack.append(node_id)
        # Reset the previous paragraph ID for the new chapter/subchapter
        self.prev_paragraph_id = None

    def _process_paragraph(self, paragraph):
        """
        Process a paragraph element, create a node, and connect to parent chapter, links, and images.
        """
        # Check if the paragraph contains only one image without surrounding text
        images = paragraph.find_all('img')
        text = paragraph.get_text(strip=True)
        has_only_image = len(images) == 1 and not text and not paragraph.find_all(text=True, recursive=False)

        if has_only_image:
            # Paragraph contains only an image
            img = images[0]
            img_url = img.get('src', '')
            img_alt = img.get('alt', '').strip()

            # Create or get the node for the image URL
            node_id = self._get_or_create_node(f'image:{img_url}')

            # If there is alt text, create or get a node for it and connect
            if img_alt:
                alt_node_id = self._get_or_create_node(img_alt)
                self._add_edge(alt_node_id, node_id, weight=0.2)
                node_id = alt_node_id

            # Connect the image node to its parent chapter
            if self.parent_stack:
                parent_id = self.parent_stack[-1]
                self._add_edge(parent_id, node_id, weight=0)

            # Update the previous paragraph ID
            if self.prev_paragraph_id is not None:
                self._add_edge(self.prev_paragraph_id, node_id, weight=0.5)
            self.prev_paragraph_id = node_id

            # No need to process further since this is an image-only paragraph
            return
        else:
            # Paragraph has text and possibly images/links
            label = paragraph.get_text().strip()
            # Create or get a node for the paragraph
            node_id = self._get_or_create_node(label)

            # If there is a previous paragraph in the same chapter, connect them
            if self.prev_paragraph_id is not None:
                self._add_edge(self.prev_paragraph_id, node_id, weight=0.5)
            # Update the previous paragraph ID
            self.prev_paragraph_id = node_id
            # Connect the paragraph to its parent chapter
            if self.parent_stack:
                parent_id = self.parent_stack[-1]
                self._add_edge(parent_id, node_id, weight=0.1)

            # Process images within the paragraph
            for img in images:
                img_url = img.get('src', '')
                img_alt = img.get('alt', '').strip()

                # Create or get the image node
                img_node_id = self._get_or_create_image_node(img_url, img_alt)

                # Connect the paragraph node to the image node
                self._add_edge(node_id, img_node_id, weight=0.2)

            # Process explicit links within the paragraph
            for link in paragraph.find_all('a', href=True):
                link_url = link['href']
                # Create or get the link node
                link_node_id = self._get_or_create_link_node(link_url)
                # Connect the paragraph node to the link node
                self._add_edge(node_id, link_node_id, weight=0.2)

            # Now, find bare URLs in the text and process them as links
            paragraph_text = paragraph.get_text()
            bare_urls = self.url_pattern.findall(paragraph_text)
            for url in bare_urls:
                # Create or get the link node
                link_node_id = self._get_or_create_link_node(url)
                # Connect the paragraph node to the link node
                self._add_edge(node_id, link_node_id, weight=0.2)

    @classmethod
    def from_markdown(cls, markdown_text: str):
        """
        Class method to create a graph from markdown text.
        """
        # Convert markdown text to HTML
        html = markdown.Markdown(extensions=['extra']).convert(markdown_text)
        # Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        # Create an instance of the class
        obj = cls()
        # Build the graph from the parsed HTML
        obj._create_graph(soup)
        return obj

    def to_dict(self):
        """
        Convert the graph to a dictionary with nodes and edges.
        """
        return {'nodes': self.nodes, 'edges': self.edges}

    def to_image(self, filename='graph.png'):
        """
        Generate a PNG image of the graph.
        """
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.DiGraph()

        # Add nodes with labels
        for node in self.nodes:
            G.add_node(node['id'], label=node['label'])

        # Add edges with weights
        for edge in self.edges:
            G.add_edge(edge['from_id'], edge['to_id'], weight=edge['weight'])

        # Position the nodes using a layout algorithm
        pos = nx.circular_layout(G, center=(0, 0))

        # Draw the nodes
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
        nx.draw_networkx_labels(G, pos, labels, font_size=8)

        # Draw the edges
        weights = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=10, width=weights)

        # Save the graph to a PNG file
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, format='PNG')
        plt.close()


if __name__ == '__main__':
    with open('manual_files/Introduction.md', 'r') as f:
        input = f.read()

    obj = MarkdownToGraph.from_markdown(input)
    print(json.dumps(obj.to_dict(), ensure_ascii=False))
    obj.to_image()
