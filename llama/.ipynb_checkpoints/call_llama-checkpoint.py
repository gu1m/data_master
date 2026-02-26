# call_llama.py
import os
import json
import torch
import time
from tqdm import tqdm
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_huggingface import HuggingFaceEmbeddings

from llama.memory_support.memory_support import Memory_support
from llama.RAG.BuildVectorStore import PDFVectorStore
from llama.RAG.rank_context import RankContext
from llama.perplexity import CalculatePerplexity

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
        self.rank_context = RankContext(model="BAAI/bge-reranker-v2-m3")

    def _init_memory(self):
        self.memory_support = Memory_support(
            db_path=self.memory_path,
            embedding_model=HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-small", 
                model_kwargs={'device': 'cuda'}
            ),
            threshold_distance=0.70,
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
    def _log_interaction(self, user, response):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user": user,
            "response": response
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def _extract_info_chunks(self, user, k):

        relevant_chunks = self.db_rag.max_marginal_relevance_search(
            user, 
            k=k,        # Quantidade para o Re-ranker analisar
            fetch_k=50,  # Quantidade que o Chroma analisa internamente
            lambda_mult=0.20
        )

        reranked_docs = self.rank_context.call_rank(query=user, documents = relevant_chunks)

        return reranked_docs

    # -------------------------------------------------------------------------
    #   MONTA PROMPT E GERA RESPOSTA
    # -------------------------------------------------------------------------
    def build_prompt(self, user: str):

        memory_block = self.memory_support.mount_prompt_memory(user, k=3)
        rag_docs = self._extract_info_chunks(user, k=20)
    
        if rag_docs:
            rag_block = "\n\n".join(
                f"[FONTE {d.id}] {d.page_content}"
                for d in rag_docs
            )
        else:
            rag_block = ""

        print(f"RAG_BLOCK: {rag_block}")
    
        prompt_main = f"""
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        
        Você é um especialista em GRC.

        Construa uma resposta completa e explicativa usando todas as informações relevantes disponíveis.
        
        Responda utilizando as informações do <CONTEXT_RAG> como base factual.
        Use também o <CONTEXT_MEMORY> se complementar a resposta.
        
        Não invente fatos que não estejam no contexto.
        
        Se múltiplos documentos se complementarem, combine-os.
        
        Se não houver informação suficiente:
        Informação não encontrada no contexto.
        
        <CONTEXT_MEMORY>
        {memory_block if memory_block.strip() else "[Nenhuma memória relevante encontrada]"}
        </CONTEXT_MEMORY>
        
        <CONTEXT_RAG>
        {rag_block if rag_block.strip() else "[Nenhum documento relevante foi encontrado.]"}
        </CONTEXT_RAG>
        
        <|end_header_id|>
        <|start_header_id|>user<|end_header_id|>
        {user}
        <|end_header_id|>
        
        <|start_header_id|>assistant<|end_header_id|>
        """

        citation_prompt = f"""
        Use apenas o CONTEXT_RAG abaixo para identificar quais doc_ids sustentam a RESPOSTA.
        
        RESPOSTA:
        answer
        
        CONTEXT_RAG:
        {rag_block}
        
        Tarefa:
        Liste somente as FONTES que contêm informações utilizadas na RESPOSTA.
        
        Regras:
        - Não explique
        - Não justifique
        - Não escreva frases
        - Não escreva texto adicional
        
        Exemplo da saída obrigatória:
        [FONTE: X, Y]
        
        Se nenhum doc_id sustentar a resposta:
        [FONTE: NENHUM]
        """
        return prompt_main, citation_prompt


    # -------------------------------------------------------------------------
    #   INVOCAÇÃO PRINCIPAL
    # -------------------------------------------------------------------------
    def invoke(self, user: str, temp=0.3, max_tokens=512, top_p=0.7, top_k=40):

        prompt_main, citation_prompt = self.build_prompt(user)
    
        # =============================
        # PASSO 1 — GERAR RESPOSTA
        # =============================
        inputs_main = self.tokenizer(prompt_main, return_tensors="pt").to(self.model.device)
    
        with torch.no_grad():
            outputs_main = self.model.generate(
                **inputs_main,
                max_new_tokens=max_tokens,
                temperature=temp,
                do_sample=True,
                top_p=top_p,
                top_k=top_k,
                return_dict_in_generate=True
            )
    
        input_len_main = inputs_main["input_ids"].shape[1]
        generated_ids_main = outputs_main.sequences[0][input_len_main:]
        answer = self.tokenizer.decode(generated_ids_main, skip_special_tokens=True)
    
        # =============================
        # PASSO 2 — GERAR CITAÇÃO
        # =============================
        citation_prompt = citation_prompt.replace("answer", answer)
    
        inputs_cite = self.tokenizer(citation_prompt, return_tensors="pt").to(self.model.device)
    
        with torch.no_grad():
            outputs_cite = self.model.generate(
                **inputs_cite,
                max_new_tokens=64,   # curto porque é só attribution
                #temperature=0.0,     # determinístico = melhor precisão
                do_sample=True,
                return_dict_in_generate=True
            )
    
        input_len_cite = inputs_cite["input_ids"].shape[1]
        generated_ids_cite = outputs_cite.sequences[0][input_len_cite:]
        citations = self.tokenizer.decode(generated_ids_cite, skip_special_tokens=True)
    
        # =============================
        # JUNTAR
        # =============================
        final_response = f"{answer}\n\nBaseado em: {citations}"
    
        # =============================
        # PERPLEXITY (só da resposta)
        # =============================
        pairs, mean_lp, ppl = self.calculate_perplexity.get_logprobs(
            sequences=outputs_main.sequences,
            input_len=input_len_main
        )
    
        # ---- MEMÓRIA ----
        try:
            self.memory_support.append_on_memory_database(user, final_response)
        except Exception as e:
            print(f"⚠️ Erro ao salvar memória: {e}")
    
        # ---- LOG ----
        self._log_interaction(user, final_response)
    
        print("\nPairs logprob x tokens\n", pairs)
        print(f"\nMean logprob: {mean_lp:.4f}")
        print(f"Perplexity: {ppl:.4f}")
    
        return final_response

