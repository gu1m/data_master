import os
import logging
import hashlib
from typing import List

import torch
import numpy as np

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader


# ======================================================
# LOGGER
# ======================================================

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("RAG")


# ======================================================
# VECTOR STORE PERSISTENTE
# ======================================================

class PDFVectorStore:

    def __init__(
        self,
        docs_path="docs_uploader/documents_txt",
        persist_dir="db",
        embedding_model="BAAI/bge-m3"
    ):

        self.docs_path = docs_path
        self.persist_dir = persist_dir

        log.info("🚀 Carregando embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cuda"}
        )

        self.db = None


    def initialize(self):
        """
        Inicializa automaticamente:
        - Se DB existir → carrega
        - Se não existir → cria
        """

        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            log.info("📦 DB já existe → carregando...")
            self.db = self._load_db()

        else:
            log.info("🆕 DB não existe → construindo índice...")
            self.db = self._build_db()

        return self.db


    def get_db(self):
        if self.db is None:
            self.initialize()
        return self.db


    def retrieve(self, query, k=20, threshold = 0.75):
        db = self.get_db()

        initial_docs = db.max_marginal_relevance_search(
            query, 
            k=k,        # Quantidade para o Re-ranker analisar
            fetch_k=50,  # Quantidade que o Chroma analisa internamente
            lambda_mult=0.5
        )

        return initial_docs


    # ==================================================
    # BUILD DB
    # ==================================================

    def _build_db(self):

        docs = self._load_files()
        chunks = self._chunk(docs)

        chunks = self._hash_dedup(chunks)
        chunks = self._semantic_dedup(chunks)

        log.info("🧠 Gerando embeddings + salvando Chroma...")

        Chroma.from_documents(
            chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )

        log.info("✅ DB persistido em disco!")


    def _load_db(self):
        return Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )


    # ==================================================
    # LOAD FILES
    # ==================================================

    def _load_files(self):

        docs = []

        for file in os.listdir(self.docs_path):

            path = os.path.join(self.docs_path, file)

            if file.endswith(".txt"):
                loader = TextLoader(path, encoding="utf-8")

            elif file.endswith(".pdf"):
                loader = PyPDFLoader(path)

            else:
                continue

            loaded = loader.load()

            for d in loaded:
                d.metadata["source"] = file

            docs.extend(loaded)

        log.info(f"📄 documentos carregados: {len(docs)}")
        return docs


    # ==================================================
    # CHUNK
    # ==================================================

    def _chunk(self, docs):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80
        )

        chunks = splitter.split_documents(docs)

        log.info(f"✂️ chunks: {len(chunks)}")
        return chunks


    # ==================================================
    # HASH DEDUP
    # ==================================================

    def _hash_dedup(self, docs):

        log.info("🧹 hash dedup...")

        seen = set()
        unique = []

        for d in docs:
            h = hashlib.sha256(d.page_content.lower().encode()).hexdigest()

            if h not in seen:
                seen.add(h)
                unique.append(d)

        log.info(f"   removidos: {len(docs)-len(unique)}")
        return unique


    # ==================================================
    # SEMANTIC DEDUP
    # ==================================================

    def _semantic_dedup(self, docs, threshold=0.97):

        log.info("🧠 semantic dedup...")

        texts = [d.page_content for d in docs]
        embs = np.array(self.embeddings.embed_documents(texts))

        keep = []
        used = set()

        for i in range(len(embs)):

            if i in used:
                continue

            keep.append(docs[i])

            sims = embs @ embs[i]
            dup_ids = np.where(sims > threshold)[0]

            for j in dup_ids:
                used.add(j)

        log.info(f"   removidos semanticamente: {len(docs)-len(keep)}")
        return keep
