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


class call_llama:
    """
    Classe principal que conversa com o modelo Llama local
    e integra com o sistema de memória de longo prazo.
    """

    def __init__(self, 
                 model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
                 device_map: str = "cuda",
                 log_file: str = "logs_llama\llama_logs.jsonl"):

        print("🚀 Inicializando call_llama...")

        # Caminhos principais
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_id = model_id
        self.device_map = device_map
        self.log_file = os.path.join(self.base_dir, log_file)

        # Caminho do banco de memória
        self.memory_path = os.path.join(self.base_dir, "memory_support_db")
        os.makedirs(self.memory_path, exist_ok=True)

        # Carregar embeddings (HuggingFace)
        print("🔍 Carregando modelo de embeddings...")
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Instanciar suporte de memória (nova API)
        print("🧠 Iniciando sistema de memória...")
        self.memory_support = Memory_support(
            db_path=self.memory_path,
            embedding_model=self.embedding_model,
            threshold_distance=0.45,  # ajuste fino recomendado: 0.35–0.55
            max_memories=300
        )

        #Instanciar suporte de RAG
        print("Iniciando sistema de RAG...")
        self.rag_support = PDFVectorStore(
            db_path = "llama/RAG/db"
        )
        self.db_rag = self.rag_support.load_index()

        # Carregar modelo Llama
        print("⚙️ Carregando modelo Llama local (pode levar alguns segundos)...")
        self.instanciate_llama()

        print("✅ call_llama inicializado com sucesso!")

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

        rag_block = ""

        for i in relevant_chunks:
            rag_block+= f"{i.page_content}\n"

        print(f"RaGBLOCKKKKK ------- {rag_block}")

        return rag_block

    # -------------------------------------------------------------------------
    #   MONTA PROMPT E GERA RESPOSTA
    # -------------------------------------------------------------------------
    def build_prompt(self, system: str, user: str):
        """
        Structura o prompt com blocos claros, adicionando memória relevante.
        """
        memory_block = self.memory_support.mount_prompt_memory(user, k=4)
        rag_block = self._extract_info_chunks(user, k=4)

        prompt = f"""
### SYSTEM
{system}

### CONTEXT MEMORY (long-term)
{memory_block if memory_block.strip() else "[Nenhuma memória relevante encontrada]"}

### CONTEXT RAG
{rag_block if rag_block.strip() else "[Nenhum documento relevante foi encontrado]"}

### USER
{user}

### ASSISTANT
"""
        return prompt

    # -------------------------------------------------------------------------
    #   INVOCAÇÃO PRINCIPAL
    # -------------------------------------------------------------------------
    def invoke(self, system: str, user: str, temp=0.3, max_tokens=512, top_p=0.95, top_k=40):
        prompt = self.build_prompt(system, user)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp,
                do_sample=True,
                top_p=top_p,
                top_k=top_k
            )

        response = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # ---- SALVAR NA MEMÓRIA ----
        try:
            self.memory_support.append_on_memory_database(user, response)
        except Exception as e:
            print(f"⚠️ Erro ao salvar memória: {e}")

        # ---- LOG ----
        self._log_interaction(system, user, response)

        return response
