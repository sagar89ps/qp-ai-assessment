from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

class VectorStore:
    def __init__(self):
        self.index = faiss.IndexFlatL2(384)
        self.docs = []

    def add(self, text_chunks):
        embeddings = model.encode(text_chunks)
        self.index.add(np.array(embeddings, dtype='float32'))
        self.docs.extend(text_chunks)

    def search(self, query, top_k=3):
        query_vec = model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_vec, top_k)
        results = [(self.docs[i], distances[0][j]) for j, i in enumerate(indices[0])]
        return results

vector_store = VectorStore()
