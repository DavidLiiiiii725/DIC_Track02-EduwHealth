import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.IndexFlatL2(384)
        self.texts = []

    def add(self, texts):
        embeddings = self.model.encode(texts)
        self.index.add(np.array(embeddings))
        self.texts.extend(texts)

    def search(self, query, k=5):
        q_emb = self.model.encode([query])
        _, idx = self.index.search(np.array(q_emb), k)
        return [self.texts[i] for i in idx[0]]
