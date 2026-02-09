# call_llama.py
import os
import json
import torch
import time
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_huggingface import HuggingFaceEmbeddings

from llama.memory_support.memory_support import Memory_support
from llama.RAG.BuildVectorStore import PDFVectorStore
from llama.RAG.rank_context import RankContext
from llama.evaluate_response.perplexity import CalculatePerplexity 


class call_llama:
    """
    Classe principal que conversa com o modelo Llama local
    e integra com o sistema de memória de longo prazo.
    """

    import os
from tqdm import tqdm
import time

# ... (seus imports originais: torch, transformers, etc)

class CallLlama:
    def __init__(self, 
                 model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
                 device_map: str = "cuda",
                 log_file: str = "logs_llama/llama_logs.jsonl"):

        print("🚀 Inicializando call_llama...")

        # Configurações de caminhos base
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_id = model_id
        self.device_map = device_map
        self.log_file = os.path.join(self.base_dir, log_file)
        self.memory_path = os.path.join(self.base_dir, "memory_support_db")
        os.makedirs(self.memory_path, exist_ok=True)

        # Definição das etapas de carregamento
        steps = [
            {"desc": "🔍 Carregando sistema de Re Rank", "func": self._init_rerank},
            {"desc": "🧠 Iniciando sistema de memória", "func": self._init_memory},
            {"desc": "📚 Iniciando sistema de RAG", "func": self._init_rag},
            {"desc": "⚙️ Carregando modelo Llama local", "func": self.instanciate_llama},
            {"desc": "📊 Inicializando Answer Evaluation", "func": self._init_perplexity}
        ]

        # Barra de progresso principal
        with tqdm(total=len(steps), desc="Progresso Geral", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
            for step in steps:
                pbar.set_description(step["desc"])
                step["func"]()  # Executa a função de inicialização
                pbar.update(1)

        print("\n✅ call_llama inicializado com sucesso!")

    # --- Métodos Auxiliares para organizar o código ---

    def _init_rerank(self):
        self.rank_context = RankContext(model="BAAI/bge-reranker-large")

    def _init_memory(self):
        self.memory_support = Memory_support(
            db_path=self.memory_path,
            embedding_model=HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2", 
                model_kwargs={'device': 'cuda'}
            ),
            threshold_distance=0.45,
            max_memories=300
        )

    def _init_rag(self):
        self.rag_support = PDFVectorStore(persist_dir='llama/RAG/db')
        self.db_rag = self.rag_support._load_db()

    def _init_perplexity(self):
        # Garante que o modelo e tokenizer já existem antes de chamar
        self.calculate_perplexity = CalculatePerplexity(model=self.model, tokenizer=self.tokenizer)

    # -------------------------------------------------------------------------
    #   LOAD LLAMA
    # -------------------------------------------------------------------------
    def instanciate_llama(self):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quant_config,
            device_map=self.device_map,
            torch_dtype=torch.bfloat16
        )
        self.model.eval()

    # -------------------------------------------------------------------------
    #   LOG INTERAÇÕES
    # -------------------------------------------------------------------------
    def _log_interaction(self, system, user, response):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "system": system,
            "user": user,
            "response": response
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def _extract_info_chunks(self, user, k):

        relevant_chunks = self.db_rag.similarity_search(user, k)

        reranked_docs = self.rank_context.call_rank(query=user, documents = relevant_chunks)

        print(f"RaGBLOCKKKKK ------- {reranked_docs}")

        return reranked_docs

    # -------------------------------------------------------------------------
    #   MONTA PROMPT E GERA RESPOSTA
    # -------------------------------------------------------------------------
    def build_prompt(self, system: str, user: str):

        memory_block = self.memory_support.mount_prompt_memory(user, k=4)
        rag_docs = self._extract_info_chunks(user, k=20)
    
        if rag_docs:
            rag_block = "\n\n".join(
                f"[FONTE {i}] {d.page_content}"
                for i, d in enumerate(rag_docs)
            )
        else:
            rag_block = ""
    
        prompt = f"""
            <|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            
            Você é um especialista em GRC.
            
            REGRAS OBRIGATÓRIAS:
            
            1. Use APENAS informações do <CONTEXT_RAG>
            2. COPIE frases literalmente do contexto
            3. NÃO reescreva com suas próprias palavras
            4. Após cada frase inclua [FONTE X]
            5. Se não houver resposta no contexto, diga: "Informação não encontrada no contexto."
            
            <CONTEXT_MEMORY>
            {memory_block if memory_block.strip() else "[Nenhuma memória relevante encontrada]"}
            </CONTEXT_MEMORY>
            
            <CONTEXT_RAG>
            {rag_block if rag_block.strip() else "[Nenhum documento relevante foi encontrado.]"}
            </CONTEXT_RAG>
            
            <|end_header_id|>user<|end_header_id|>
            {user}
            
            <|start_header_id|>assistant<|end_header_id|>
            """
        return prompt


    # -------------------------------------------------------------------------
    #   INVOCAÇÃO PRINCIPAL
    # -------------------------------------------------------------------------
    def invoke(self, system: str, user: str, temp=0.3, max_tokens=512, top_p=0.95, top_k=40):

        prompt = self.build_prompt(system, user)
    
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
    
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp,
                do_sample=True,   # pode usar sampling agora
                top_p=top_p,
                top_k=top_k,
                return_dict_in_generate=True
            )
    
        input_len = inputs["input_ids"].shape[1]
    
        generated_ids = outputs.sequences[0][input_len:]
    
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
    
        pairs, mean_lp, ppl = self.calculate_perplexity.get_logprobs(
            sequences=outputs.sequences,
            input_len=input_len
        )
        
        # ---- SALVAR NA MEMÓRIA ----
        try:
            self.memory_support.append_on_memory_database(user, response)
        except Exception as e:
            print(f"⚠️ Erro ao salvar memória: {e}")

        # ---- LOG ----
        self._log_interaction(system, user, response)
    
        print("\nPairs logprob x tokens\n", pairs)
        print(f"\nMean logprob: {mean_lp:.4f}")
        print(f"Perplexity: {ppl:.4f}")
    
        return response

