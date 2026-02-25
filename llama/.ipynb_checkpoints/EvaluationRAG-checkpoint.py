import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Callable, Dict, Any
import torch.nn.functional as F
from dataclasses import dataclass
import pandas as pd

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
            embeddings = outputs.last_hidden_state[:, 0]
            # Normalização é crucial para que o produto escalar seja igual à similaridade de cosseno
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def _semantic_check(self, retrieved: List[str], relevant: List[str], threshold: float = 0.40) -> int:
        if not retrieved or not relevant:
            return 0
    
        emb_retrieved = self._get_embeddings(retrieved) 
        emb_relevant = self._get_embeddings(relevant) 
    
        cos_sim_matrix = torch.mm(emb_retrieved, emb_relevant.t())
    
        max_sim_per_relevant, _ = torch.max(cos_sim_matrix, dim=0)
    
        # conta quantos relevantes foram cobertos
        hits_count = torch.sum(max_sim_per_relevant >= threshold).item()
    
        return hits_count

    def create_evaluation_df(self):
        # Definimos as colunas exatas para as novas métricas
        columns = [
            "N° Interação", 
            "similarity_threshold", 
            "hit_rate", 
            "mrr", 
            "precision_at_1", 
            "N° Hits", 
            "N° Miss"
        ]
        return pd.DataFrame(columns=columns)


    def evaluate(self, k: int, similarity_threshold: float, max_threshold=0.99):
        df = self.create_evaluation_df()
        
        # Define o número de passos (0.01 em 0.01)
        steps = int(round((max_threshold - similarity_threshold) * 100))
        
        for i in range(steps + 1):
            running_hits = 0        # Para Hit Rate
            running_mrr = 0.0       # Para Mean Reciprocal Rank
            running_p1 = 0.0        # Para Precision@1
            n = len(self.dataset)
            
            for sample in self.dataset:
                # 1. Recupera os documentos (Top K após similaridade e Re-ranker)
                retrieved = self.pipeline_fn(sample.query, k)
                
                # 2. Avalia cada documento retornado para calcular o Rank
                first_hit_rank = None
                
                for rank, doc in enumerate(retrieved, start=1):
                    # Verifica se este documento específico é relevante
                    # Passamos apenas o documento atual em uma lista para o check
                    is_relevant = self._semantic_check([doc], sample.relevant_chunks, threshold=similarity_threshold) > 0
                    
                    if is_relevant:
                        if first_hit_rank is None:
                            first_hit_rank = rank # Salva a posição do primeiro acerto
                        
                # 3. Cálculos das Métricas da Query
                # Hit Rate: O documento apareceu em qualquer lugar do Top K?
                if first_hit_rank is not None:
                    running_hits += 1
                    
                    # MRR: 1 / posição do primeiro acerto (Ex: 1º=1.0, 2º=0.5, 3º=0.33)
                    running_mrr += 1.0 / first_hit_rank
                    
                    # Precision@1: O primeiro da lista é relevante?
                    if first_hit_rank == 1:
                        running_p1 += 1.0

            # 4. Médias Finais do Dataset para este Threshold
            final_hit_rate = running_hits / n
            final_mrr = running_mrr / n
            final_p1 = running_p1 / n

            # 5. Registro no DataFrame (ajustado para as novas colunas)
            nova_linha = pd.DataFrame([{
                "N° Interação": i, 
                "similarity_threshold": round(similarity_threshold, 2), 
                "hit_rate": final_hit_rate, 
                "mrr": final_mrr, 
                "precision_at_1": final_p1, 
                "N° Hits": running_hits, 
                "N° Miss": n - running_hits
            }])

            df = pd.concat([df, nova_linha], ignore_index=True)

            print(f"Thres: {similarity_threshold:.2f} | HitRate: {final_hit_rate:.3f} | MRR: {final_mrr:.3f} | P@1: {final_p1:.3f}")

            similarity_threshold += 0.01

        return df