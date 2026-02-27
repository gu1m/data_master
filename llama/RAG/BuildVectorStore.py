import os
import logging
import hashlib
import re
import unicodedata
import numpy as np

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("RAG")


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
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True}
        )

        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)

        self.db = None


    # ==================================================
    # INIT
    # ==================================================

    def initialize(self):

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


    # ==================================================
    # BUILD
    # ==================================================

    def _build_db(self):

        docs = self._load_files()
        docs = self._split_by_sections(docs)

        chunks = self._chunk(docs)

        chunks = self._hash_dedup(chunks)
        chunks = self._semantic_dedup(chunks)

        log.info("🧠 Gerando embeddings + salvando Chroma...")

        self.db = Chroma.from_documents(
            chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )

        log.info("✅ DB persistido!")
        return self.db


    def _load_db(self):
        return Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )


    # ==================================================
    # LOAD TXT
    # ==================================================

    def _load_files(self):

        docs = []

        for file in os.listdir(self.docs_path):

            if not file.endswith(".txt"):
                continue

            path = os.path.join(self.docs_path, file)

            loader = TextLoader(path, encoding="utf-8")
            loaded = loader.load()

            for i, d in enumerate(loaded):
                d.metadata["source"] = file
                d.metadata["doc_id"] = f"{file}_{i}"

            docs.extend(loaded)

        log.info(f"📄 documentos carregados: {len(docs)}")
        return docs


    # ==================================================
    # SPLIT POR SEÇÕES (TXT INTELIGENTE)
    # ==================================================

    def _split_by_sections(self, docs):

        new_docs = []

        section_pattern = r"(CAP[IÍ]TULO.*|PRINC[IÍ]PIO.*|[0-9]+\.[0-9]+.*)"

        for d in docs:

            sections = re.split(section_pattern, d.page_content)

            buffer = ""
            section_title = None

            for part in sections:

                if re.match(section_pattern, part):

                    if buffer:
                        new_docs.append(Document(
                            page_content=buffer,
                            metadata=d.metadata
                        ))
                        buffer = ""

                    section_title = part
                    buffer += part + "\n"

                else:
                    buffer += part

            if buffer:
                new_docs.append(Document(
                    page_content=buffer,
                    metadata=d.metadata
                ))

        log.info(f"🧱 seções detectadas: {len(new_docs)}")
        return new_docs


    # ==================================================
    # TOKEN CHUNKING
    # ==================================================

    def _chunk(self, docs):

        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=400,
            chunk_overlap=80,
            separators=[
                "\n\n", "\n", ".", "•", "-", " "
            ]
        )

        chunks = splitter.split_documents(docs)

        for i, c in enumerate(chunks):
            c.metadata["chunk_id"] = f"{c.metadata['doc_id']}_chunk_{i}"

        log.info(f"✂️ chunks gerados: {len(chunks)}")
        return chunks

    def _normalize(self, text):

        # remove unicode invisível
        text = unicodedata.normalize("NFKD", text)
    
        # remove form feed, tabs, etc
        text = text.replace("\x0c", " ")
    
        # remove hifenização quebrada de PDF
        text = re.sub(r"-\s*\n\s*", "", text)
    
        # remove múltiplos espaços
        text = re.sub(r"\s+", " ", text)
    
        return text.strip().lower()


    # ==================================================
    # HASH DEDUP
    # ==================================================

    def _hash_dedup(self, docs):

        seen = set()
        unique = []
    
        for d in docs:
    
            clean = self._normalize(d.page_content)
    
            h = hashlib.sha256(clean.encode()).hexdigest()
    
            if h not in seen:
                seen.add(h)
                unique.append(d)
    
        log.info(f"🧹 hash removidos: {len(docs)-len(unique)}")
        return unique


    # ==================================================
    # SEMANTIC DEDUP
    # ==================================================

    def _semantic_dedup(self, docs, threshold=0.95):

        texts = [d.page_content for d in docs]
        embs = np.array(self.embeddings.embed_documents(texts))

        sims = cosine_similarity(embs)

        keep = []
        removed = set()

        for i in range(len(docs)):
            if i in removed:
                continue

            keep.append(docs[i])

            for j in range(i + 1, len(docs)):
                if sims[i][j] > threshold:
                    removed.add(j)

        log.info(f"🧹 semantic removidos: {len(docs)-len(keep)}")
        return keep