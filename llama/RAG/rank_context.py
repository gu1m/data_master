import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class RankContext:

    def __init__(self, model, top_n=6):

        # Define que o reranker será executado na GPU
        self.device = torch.device("cuda")

        # Carrega o tokenizer do modelo cross-encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        # Carrega o modelo de classificação (usado como reranker)
        self.model = AutoModelForSequenceClassification.from_pretrained(model)

        # Move o modelo para GPU
        self.model.to(self.device)

        # Coloca o modelo em modo de inferência
        self.model.eval()

        # Número de documentos que serão mantidos após o reranking
        self.top_n = top_n


    def call_rank(self, query, documents):

        # Extrai apenas o texto dos documentos recuperados
        texts = [d.page_content for d in documents]

        # Cria pares (query, documento)
        # Cross-encoder avalia relevância par a par
        pairs = [[query, t] for t in texts]

        # Tokeniza os pares para entrada no modelo
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(self.device)

        # Inferência sem cálculo de gradiente (mais rápido)
        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1)

        # Associa cada documento ao seu score
        scored_docs = list(zip(scores.cpu().tolist(), documents))

        # Ordena do mais relevante para o menos relevante
        scored_docs.sort(reverse=True, key=lambda x: x[0])

        # Seleciona apenas os top_n mais relevantes
        ranked = [d for _, d in scored_docs[:self.top_n]]

        # Log do ranking final
        print("\n🏆 Reranker ranking:")
        for s, d in scored_docs[:self.top_n]:
            print(f"{s:.4f} | {d.metadata} | [doc_id: {d.id}]")

        # Retorna documentos rerankeados
        return ranked
