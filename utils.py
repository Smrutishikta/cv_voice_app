# utils.py
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class KnowledgeBase:
    def __init__(self, json_path="knowledge_base.json"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.text_chunks = self._extract_text()
        self.index = self._build_index()

    def _extract_text(self):
        chunks = []

        def recurse(data, prefix=""):
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, list):
                        items = ", ".join(map(str, v))
                        chunks.append(f"{prefix}{k}: {items}")
                    else:
                        recurse(v, f"{prefix}{k}: ")
            elif isinstance(data, list):
                for item in data:
                    recurse(item, prefix)
            else:
                chunks.append(f"{prefix}{data}")

        recurse(self.data)
        return chunks

    def _build_index(self):
        embeddings = self.model.encode(self.text_chunks)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings).astype("float32"))
        return index

    def query(self, question, top_k=3):
        query_emb = self.model.encode([question])
        D, I = self.index.search(np.array(query_emb).astype("float32"), top_k)
        results = [self.text_chunks[i] for i in I[0]]
        return results
