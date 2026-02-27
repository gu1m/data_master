# memory_cleaner.py
import numpy as np
from typing import List, Tuple, Optional

class MemoryCleaner:
    """
    Remove memórias curtas/ruidosas e deduplica semanticamente.
    Expects embedding_model to provide embed_query(text) -> vector or embed_documents([text]) -> [vector].
    """

    def __init__(self, min_length: int = 15, similarity_threshold: float = 0.95):
        self.min_length = min_length
        self.similarity_threshold = similarity_threshold

    def is_valid(self, text: str) -> bool:
        return bool(text and len(text.strip()) >= self.min_length)

    def deduplicate(self, items: List[Tuple], embedding_model) -> List[Tuple]:
        """
        items: list of tuples (id, document_text, metadata, embedding)
        returns a filtered list with duplicates removed.
        Semantic duplication is detected via cosine similarity of embeddings.
        """
        unique = []
        seen_vectors = []

        for item in items:
            # Desempacotar corretamente: pode ter 3 ou 4 elementos
            if len(item) == 3:
                _id, doc_text, metadata = item
                embedding = None
            elif len(item) == 4:
                _id, doc_text, metadata, embedding = item
            else:
                print(f"⚠️ Item com tamanho inesperado: {len(item)}")
                continue

            text = (doc_text or "").strip()
            if not self.is_valid(text):
                continue

            # Se já temos embedding, usa ele; senão tenta gerar
            vector = embedding
            if vector is None and embedding_model is not None:
                try:
                    vector = embedding_model.embed_query(text)
                except Exception:
                    try:
                        vector = embedding_model.embed_documents([text])[0]
                    except Exception:
                        vector = None

            # if no vector available, keep it
            if vector is None:
                unique.append((_id, doc_text, metadata, None))
                continue

            is_dup = False
            for prev in seen_vectors:
                sim = self.cosine_similarity(vector, prev)
                if sim >= self.similarity_threshold:
                    is_dup = True
                    break

            if not is_dup:
                unique.append((_id, doc_text, metadata, vector))
                seen_vectors.append(vector)

        return unique

    def clean(self, items: List[Tuple], embedding_model=None) -> List[Tuple]:
        return self.deduplicate(items, embedding_model)

    @staticmethod
    def cosine_similarity(a, b) -> float:
        a, b = np.array(a), np.array(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(a.dot(b) / denom)
