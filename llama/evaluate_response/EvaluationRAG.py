import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Callable, Dict, Any
import torch.nn.functional as F
from dataclasses import dataclass

# Certifique-se que a EvaluationSample está definida ou importada aqui também
@dataclass
class EvaluationSample:
    query: str
    relevant_chunks: List[str]

class RAGEvaluator:

    def __init__(
        self,
        pipeline_fn: Callable[[str, int], List[str]],
        dataset: List[EvaluationSample],
        model_name: str = "BAAI/bge-m3"
    ):
        self.pipeline_fn = pipeline_fn
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Carregando via Hugging Face Transformers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _get_embeddings(self, texts: List[str]):
        """Gera embeddings normalizados para uma lista de textos."""
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # O BGE-M3 usa o token [CLS] (índice 0) para representar a frase
            embeddings = outputs.last_hidden_state[:, 0]
            # Normalização é crucial para que o produto escalar seja igual à similaridade de cosseno
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def _semantic_check(self, retrieved: List[str], relevant: List[str], threshold: float = 0.40) -> int:
        if not retrieved or not relevant:
            return 0
        
        # Gerar embeddings
        emb_retrieved = self._get_embeddings(retrieved) # [k, dim]
        emb_relevant = self._get_embeddings(relevant)   # [n_relevant, dim]
        
        # Calcular similaridade de cosseno via produto escalar (já que estão normalizados)
        # Resultado é uma matriz [k, n_relevant]
        cos_sim_matrix = torch.mm(emb_retrieved, emb_relevant.t())
        
        # Para cada doc recuperado, pegamos a maior similaridade encontrada com qualquer doc relevante
        max_sim_per_retrieved, _ = torch.max(cos_sim_matrix, dim=1)
        
        # Conta quantos documentos recuperados passaram de 0.40 (40%)
        hits_count = torch.sum(max_sim_per_retrieved >= threshold).item()
        return hits_count

    def evaluate(self, k: int, similarity_threshold: float = 0.40):
        hits = 0
        total_recall = 0
        total_precision = 0
        n = len(self.dataset)

        for sample in self.dataset:
            retrieved = self.pipeline_fn(sample.query, k)
            
            # Verificação semântica
            tp_count = self._semantic_check(retrieved, sample.relevant_chunks, threshold=similarity_threshold)

            recall = tp_count / len(sample.relevant_chunks)
            precision = tp_count / k
            hit = tp_count > 0

            if hit: hits += 1
            total_recall += recall
            total_precision += precision

            status = "HIT " if hit else "MISS"
            print(f"[{status}] {sample.query} | Matches Semânticos: {tp_count}")

        print("\n" + "="*50)
        print(f"METRICS (BGE-M3 Semantic - Threshold: {similarity_threshold})")
        print(f"Accuracy   : {hits/n:.3f}")
        print(f"Recall@k   : {total_recall/n:.3f}")
        print(f"Precision@k: {total_precision/n:.3f}")
        print("="*50)