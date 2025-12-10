# memory_purifier.py
import gc
import time
from typing import List, Tuple
from chromadb import Client
from chromadb.config import Settings

class MemoryPurifier:

    def __init__(self, max_memories: int = 300, retry_wait: float = 0.1, retry_attempts: int = 5):
        self.max_memories = max_memories
        self.retry_wait = retry_wait
        self.retry_attempts = retry_attempts

    def enforce_memory_limit(self, entries: List[Tuple]) -> List[Tuple]:
    
        if len(entries) <= self.max_memories:
            return entries
        
        # Ordenar por timestamp (índice 3) de forma DECRESCENTE para pegar os mais recentes
        sorted_entries = sorted(entries, key=lambda x: x[3], reverse=True)
        
        kept = sorted_entries[:self.max_memories]
        print(f"🗑️ Removendo {len(entries) - len(kept)} memórias antigas")
        
        return kept

    def rewrite_memory_collection(self, client: Client, collection_name: str, pruned_docs: List[Tuple], embedding_model):
        
        # 1) delete collection if exists
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        # 2) recreate collection
        collection = client.get_or_create_collection(name=collection_name)

        # 3) prepare docs, metadatas, ids and embeddings
        if not pruned_docs:
            return collection

        ids = []
        docs = []
        metadatas = []
        embeddings_list = []

        for entry in pruned_docs:
            if len(entry) == 5:
                _id, doc_text, metadata, timestamp, embedding = entry
            elif len(entry) == 4:
                _id, doc_text, metadata, embedding = entry
            else:
                print(f"⚠️ Entry com tamanho inesperado: {len(entry)}")
                continue
                
            ids.append(_id)
            docs.append(doc_text)
            metadatas.append(metadata)
            embeddings_list.append(embedding)

        # compute embeddings if needed
        final_embeddings = []
        for i, emb in enumerate(embeddings_list):
            if emb is not None:
                final_embeddings.append(emb)
            elif embedding_model is not None:
                try:
                    new_emb = embedding_model.embed_query(docs[i])
                    final_embeddings.append(new_emb)
                except Exception:
                    final_embeddings.append(None)
            else:
                final_embeddings.append(None)

        # 4) add to collection (with or without embeddings)
        for attempt in range(self.retry_attempts):
            try:
                # Verificar se temos embeddings válidos
                valid_embeddings = [e for e in final_embeddings if e is not None]
                
                if len(valid_embeddings) == len(final_embeddings):
                    collection.add(
                        documents=docs, 
                        metadatas=metadatas, 
                        ids=ids, 
                        embeddings=final_embeddings
                    )
                else:
                    collection.add(
                        documents=docs, 
                        metadatas=metadatas, 
                        ids=ids
                    )
                return collection
            except Exception as e:
                print(f"⚠️ Tentativa {attempt + 1} falhou: {e}")
                gc.collect()
                time.sleep(self.retry_wait)

        # final attempt without embeddings
        try:
            collection.add(documents=docs, metadatas=metadatas, ids=ids)
        except Exception as e:
            print(f"❌ Falha crítica ao adicionar documentos: {e}")
            
        return collection