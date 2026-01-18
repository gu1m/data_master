import os
import re
import hashlib
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from unstructured.partition.pdf import partition_pdf


# =========================================================
# CONFIGURAÇÕES GERAIS
# =========================================================

NOISE_CATEGORIES = ["PageHeader", "Footer", "TableOfContents", "Header"]
MIN_CHARS_TO_KEEP = 10


# =========================================================
# LIMPEZA DE TEXTO
# =========================================================

def _clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# =========================================================
# DETECÇÃO SEMÂNTICA DE DEFINIÇÕES / BLOCOS NORMATIVOS
# =========================================================

def _is_normative_block(text: str) -> bool:
    patterns = [
        r"\bis defined as\b",
        r"\bis a process\b",
        r"\baccording to\b",
        r"\bframework\b",
        r"\benterprise risk management\b"
    ]
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


# =========================================================
# PROCESSAMENTO DE TXT
# =========================================================

def _process_txt_file(file_path: str) -> List[Document]:
    print(f"  -> TXT: {os.path.basename(file_path)}")

    try:
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
    except UnicodeDecodeError:
        loader = TextLoader(file_path, encoding="latin-1")
        docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)
    final_chunks = []

    for chunk in chunks:
        text = _clean_text(chunk.page_content)
        print('oi',chunk)
        if len(text) < MIN_CHARS_TO_KEEP:
            continue

        normative = _is_normative_block(text)

        chunk.page_content = text
        chunk.metadata.update({
            "source": "COSO",
            "document": os.path.basename(file_path),
            "element_category": "NarrativeText_TXT",
            "normative": normative,
            "type": "definition" if normative else "body"
        })

        final_chunks.append(chunk)

    return final_chunks


# # =========================================================
# # PROCESSAMENTO DE PDF (LAYOUT-AWARE)
# # =========================================================

# def _process_pdf_adaptively(pdf_path: str) -> List[Document]:
#     print(f"  -> PDF: {os.path.basename(pdf_path)}")

#     try:
#         elements = partition_pdf(
#             filename=pdf_path,
#             strategy="hi_res",
#             infer_table_structure=True
#         )
#     except Exception:
#         loader = PyPDFLoader(pdf_path)
#         docs = loader.load()
#         return RecursiveCharacterTextSplitter(
#             chunk_size=1200,
#             chunk_overlap=150
#         ).split_documents(docs)

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1200,
#         chunk_overlap=150
#     )

#     final_chunks = []

#     for el in elements:
#         raw_text = el.text or ""
#         text = _clean_text(raw_text)

#         if len(text) < MIN_CHARS_TO_KEEP:
#             continue

#         meta = el.metadata.to_dict()
#         category = meta.get("category", "Text")

#         if category in NOISE_CATEGORIES:
#             continue

#         normative = _is_normative_block(text)

#         metadata = {
#             "source": "COSO",
#             "document": os.path.basename(pdf_path),
#             "page": meta.get("page_number", "s.p."),
#             "element_category": category,
#             "normative": normative,
#             "type": "definition" if normative else "body"
#         }

#         # 🔒 Definições: NÃO fragmentar
#         if normative:
#             final_chunks.append(Document(text, metadata))
#             continue

#         # 📊 Tabelas/Listas: atômicas
#         if category in ["Table", "ListItem", "Title"]:
#             final_chunks.append(Document(text, metadata))
#             continue

#         # 📚 Corpo do texto: fragmentado
#         for chunk in splitter.split_text(text):
#             final_chunks.append(Document(chunk, metadata))

#     return final_chunks


# =========================================================
# PROCESSAMENTO GERAL
# =========================================================

def _process_all_files(folder: str) -> List[Document]:
    all_chunks = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        if file.endswith(".pdf"):
            all_chunks.extend(_process_pdf_adaptively(path))

        elif file.endswith(".txt"):
            all_chunks.extend(_process_txt_file(path))

    return all_chunks


# =========================================================
# VECTOR STORE + RAG RETRIEVAL
# =========================================================

class PDFVectorStore:

    def __init__(self, doc_folder="docs_uploader/documents_txt", db_path="db"):
        self.doc_folder = os.path.abspath(doc_folder)
        self.db_path = os.path.abspath(db_path)

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cuda"}
        )

    # -------------------------------
    # Deduplicação
    # -------------------------------

    @staticmethod
    def _deduplicate(chunks: List[Document]) -> List[Document]:
        seen = set()
        unique = []

        for c in chunks:
            h = hashlib.sha256(
                c.page_content.lower().strip().encode("utf-8")
            ).hexdigest()

            if h not in seen:
                seen.add(h)
                unique.append(c)

        return unique

    # -------------------------------
    # Indexação
    # -------------------------------

    def build_index(self):
        print("🔍 Processando documentos...")
        chunks = _process_all_files(self.doc_folder)
        print(f"📦 Chunks gerados: {len(chunks)}")

        chunks = self._deduplicate(chunks)
        print(f"📦 Chunks únicos: {len(chunks)}")

        print("🧠 Criando embeddings...")
        Chroma.from_documents(
            chunks,
            embedding=self.embedding_model,
            persist_directory=self.db_path
        )

        print("✅ Vector store criado com sucesso!")

    def load_index(self):
        return Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embedding_model
        )

    # =====================================================
    # 🔥 METADATA FILTERING DINÂMICO
    # =====================================================

    def _infer_query_intent(self, question: str) -> Dict[str, Any]:
        q = question.lower()

        intent = {
            "prefer_normative": False,
            "expected_type": None,
            "k": 15
        }

        if re.match(r"(o que é|defina|conceitue)", q):
            intent.update({
                "prefer_normative": True,
                "expected_type": "definition",
                "k": 6
            })

        elif re.match(r"(qual|quais|explique)", q):
            intent.update({
                "prefer_normative": True,
                "expected_type": "body",
                "k": 8
            })

        elif re.match(r"(como|de que forma|quais passos)", q):
            intent.update({
                "expected_type": "body",
                "k": 15
            })

        return intent

    def _build_metadata_filter(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        f = {"source": "COSO"}

        if intent["prefer_normative"]:
            f["normative"] = True

        if intent["expected_type"]:
            f["type"] = intent["expected_type"]

        return f

    # -------------------------------
    # QUERY FINAL
    # -------------------------------

    def query(self, question: str):
        print("❓ Query com metadata filtering dinâmico")

        db = self.load_index()
        intent = self._infer_query_intent(question)
        metadata_filter = self._build_metadata_filter(intent)

        print(f"🧠 Intent: {intent}")
        print(f"🔎 Filter: {metadata_filter}")

        try:
            results = db.similarity_search(
                question,
                k=intent["k"],
                filter=metadata_filter
            )
        except Exception as e:
            print(f"⚠️ Fallback sem filtro: {e}")
            results = db.similarity_search(question, k=10)

        return results
