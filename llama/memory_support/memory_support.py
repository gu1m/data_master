# memory_support.py
import os
import time
import uuid
import gc
from typing import List, Tuple
import chromadb
from chromadb.config import Settings
from .memory_cleaner import MemoryCleaner
from .memory_purifier import MemoryPurifier


class Memory_support:

    def __init__(self, db_path: str, embedding_model,
                 threshold_distance: float = 0.65,
                 max_memories: int = 300,
                 collection_name: str = "memory"):

        self.db_path = os.path.abspath(db_path)
        self.embedding_model = embedding_model
        self.threshold_distance = threshold_distance
        self.max_memories = max_memories
        self.collection_name = collection_name

        # verbose
        # print("\n==========================")
        # print("🧠 Sistema de Memória Iniciando")
        # print("📁 DB Path:", self.db_path)
        # print("📂 Pasta existe:", os.path.isdir(self.db_path))
        # print("==========================\n")

        os.makedirs(self.db_path, exist_ok=True)

        self.cleaner = MemoryCleaner()
        self.purifier = MemoryPurifier(max_memories=max_memories)

        # Inicializa o cliente e a coleção
        self._initialize_chroma()


    def _initialize_chroma(self):

        #print("🔄 Inicializando/Recriando cliente Chroma PERSISTENTE...")
        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
            
            # Garante que a coleção está sincronizada
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            
            count = self.collection.count()
            #print(f"🟩 Coleção Chroma carregada: {self.collection_name}")
            #print(f"📊 Memórias existentes: {count}")
            
            # Listar arquivos para debug
            if os.path.exists(self.db_path):
                files = os.listdir(self.db_path)
                #print(f"📂 Arquivos no DB: {files}")
                
        except Exception as e:
            print(f"❌ Falha ao inicializar o ChromaDB: {e}")
            raise

    def append_on_memory_database(self, user: str, assistance_response: str):
        doc_text = f"USER: {user}\nLLM: {assistance_response}"
        _id = str(uuid.uuid4())
        metadata = {"timestamp": time.time()}

        print(f"\n💾 Adicionando memória: {doc_text[:60]}...")

        embedding = None
        if self.embedding_model is not None:
            try:
                embedding = self.embedding_model.embed_query(doc_text)
            except Exception:
                try:
                    embedding = self.embedding_model.embed_documents([doc_text])[0]
                except Exception as e:
                    print(f"⚠ Falha ao gerar embedding: {e}")
                    embedding = None

        try:
            if embedding is None:
                self.collection.add(documents=[doc_text], metadatas=[metadata], ids=[_id])
            else:
                self.collection.add(documents=[doc_text], metadatas=[metadata],
                                    ids=[_id], embeddings=[embedding])
            
            # Verificar se foi adicionado
            count_after = self.collection.count()
            print(f"🟩 Memória adicionada. ID: {_id}")
            print(f"📊 Total de memórias agora: {count_after}")
            
        except Exception as e:
            print(f"⚠️ Erro ao adicionar memória, tentando recriar cliente: {e}")
            self._initialize_chroma()
            
            # Tenta novamente após a reinicialização do cliente
            if embedding is None:
                self.collection.add(documents=[doc_text], metadatas=[metadata], ids=[_id])
            else:
                self.collection.add(documents=[doc_text], metadatas=[metadata],
                                    ids=[_id], embeddings=[embedding])

        # Mostrar arquivos do banco (Apenas para debug)
        try:
            files = os.listdir(self.db_path)
            print(f"📂 Arquivos no DB: {files}")
            # Mostrar tamanho dos arquivos
            for f in files:
                fpath = os.path.join(self.db_path, f)
                if os.path.isfile(fpath):
                    size = os.path.getsize(fpath)
                    print(f"  - {f}: {size} bytes")
        except Exception as e:
            print(f"⚠️ Não foi possível listar o diretório: {e}")

        # Limpar + purificar
        self._auto_clean_and_purify()

    def _auto_clean_and_purify(self):
        print("🧼 Limpando e purificando memórias...")
    
        try:
            count = self.collection.count()
            if count == 0:
                print("📊 Nenhuma memória para limpar.")
                return
    
            res = self.collection.get(
                include=["documents", "metadatas", "embeddings"]
            )
            
        except Exception as e:
            print(f"⚠️ Erro ao buscar todas as memórias: {e}")
            return
    
        ids = res.get("ids", [])
        docs = res.get("documents", [])
        metadatas = res.get("metadatas", [])
        embeddings = res.get("embeddings", [])
    
        print(f"📊 Total de memórias antes da limpeza: {len(ids)}")
    
        if not ids:
            return
    
        # Criar items com 4 elementos: (id, doc, metadata, embedding)
        items = []
        for i in range(len(ids)):
            _id = ids[i]
            doc = docs[i] if i < len(docs) else ""
            meta = metadatas[i] if i < len(metadatas) else {}
            emb = embeddings[i] if i < len(embeddings) and embeddings[i] is not None else None
            items.append((_id, doc, meta, emb))
    
        # 1. Deduplicação (retorna tuplas de 4 elementos)
        cleaned = self.cleaner.deduplicate(items, self.embedding_model)
        print(f"🧹 Após deduplicação: {len(cleaned)} memórias")
    
        # 2. Purificação (Enforce Memory Limit)
        # Converter para 5 elementos: (id, doc, metadata, timestamp, embedding)
        entries = []
        for _id, doc_text, metadata, embedding in cleaned:
            timestamp = metadata.get("timestamp", 0)
            entries.append((_id, doc_text, metadata, timestamp, embedding))
    
        # pruned: lista de tuplas de 5 elementos
        pruned = self.purifier.enforce_memory_limit(entries)
    
        print(f"✂️ Memórias após limite: {len(pruned)}")
    
        # Se o número de memórias mudou, regrava a coleção
        if len(pruned) < len(ids):
            print("🔄 Reescrevendo coleção...")
            
            # Deletar todos os IDs antigos
            try:
                self.collection.delete(ids=ids)
                print(f"🗑️ Deletados {len(ids)} registros antigos")
            except Exception as e:
                print(f"⚠️ Erro ao deletar: {e}")
                return
    
            # Readicionar apenas os mantidos
            for _id, doc, metadata, _, embedding in pruned:
                try:
                    if embedding is None:
                        self.collection.add(
                            documents=[doc],
                            metadatas=[metadata],
                            ids=[_id]
                        )
                    else:
                        self.collection.add(
                            documents=[doc],
                            metadatas=[metadata],
                            ids=[_id],
                            embeddings=[embedding]
                        )
                except Exception as e:
                    print(f"⚠️ Erro ao readicionar memória {_id}: {e}")
            
            final_count = self.collection.count()
            print(f"✅ Coleção reescrita. Total final: {final_count}")

    # -------------------------------------------------
    # 🔍 Recuperar memórias relevantes para prompt
    # -------------------------------------------------
    def mount_prompt_memory(self, user: str, k: int = 3) -> str:
        """Busca as K memórias mais relevantes para a consulta do usuário."""
        if not user:
            return ""

        print(f"\n🔍 Recuperando memórias relevantes para: '{user[:50]}...'")

        embedding = None
        if self.embedding_model:
            try:
                embedding = self.embedding_model.embed_query(user)
            except Exception as e:
                print(f"⚠️ Falha ao gerar embedding para consulta: {e}")
                embedding = None

        try:
            if embedding is not None:
                res = self.collection.query(
                    query_embeddings=[embedding],
                    n_results=k,
                    include=["documents", "distances"]
                )
            else:
                res = self.collection.query(
                    query_texts=[user],
                    n_results=k,
                    include=["documents", "distances"]
                )
        except Exception as e:
            print(f"❌ Falha ao realizar query no ChromaDB: {e}")
            return ""

        docs = res.get("documents", [[]])[0]
        distances = res.get("distances", [[]])[0]

        print("🎯 Memórias encontradas:")
        memory_text = ""
        for doc, dist in zip(docs, distances):
            if dist is None or dist <= self.threshold_distance:
                print(f" - dist {dist:.4f} (INCLUÍDA): {doc[:80]}...")
                memory_text += doc + "\n"
            else:
                print(f" - dist {dist:.4f} (EXCLUÍDA): {doc[:80]}...")

        return memory_text
    
    def get_memory_stats(self):
        """Retorna estatísticas sobre a memória."""
        try:
            count = self.collection.count()
            db_size = 0
            if os.path.exists(self.db_path):
                for f in os.listdir(self.db_path):
                    fpath = os.path.join(self.db_path, f)
                    if os.path.isfile(fpath):
                        db_size += os.path.getsize(fpath)
            
            return {
                "total_memories": count,
                "db_size_bytes": db_size,
                "db_size_mb": round(db_size / (1024 * 1024), 2),
                "db_path": self.db_path
            }
        except Exception as e:
            print(f"❌ Erro ao obter estatísticas: {e}")
            return None