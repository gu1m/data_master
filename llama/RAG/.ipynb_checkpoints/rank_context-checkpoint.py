from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch

class RankContext:
    def __init__(self, model, top_n = 4):

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model)
        self.top_n = top_n
        
    def call_rank(self, query, documents):
        device = torch.device("cuda")
        self.model.to(device)
        self.model.eval()

        documents_to_rerank = [doc.page_content for doc in documents]

        input_pairs = [[query,doc] for doc in documents_to_rerank]

        inputs = self.tokenizer(input_pairs,
                         padding=True,
                         truncation=True,
                         return_tensors='pt',
                         max_length=1000)


        inputs = {k: v.to(device) for k,v in inputs.items()}

        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1)

        score_list = scores.cpu().numpy().tolist()

        results = pd.DataFrame({
            'documento' : documents_to_rerank,
            'score_relevancia': score_list
        })

        ranked_results = results.sort_values(by="score_relevancia", ascending= False).head(self.top_n)

        final_rag_context = '\n'.join(ranked_results['documento'].to_list())

        print(ranked_results)

        return final_rag_context
        
        