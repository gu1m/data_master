import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class PDFVectorStore:
    def __init__(self, pdf_folder="docs_uploader", db_path="db"):
        self.pdf_folder = os.path.abspath(pdf_folder)
        self.db_path = os.path.abspath(db_path)

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # print(f"📁 Pasta dos PDFs: {self.pdf_folder}")
        # print(f"🗂️ Banco vetorial: {self.db_path}")

    def build_index(self):
        print("🔍 Carregando PDFs...")
        loader = PyPDFDirectoryLoader(self.pdf_folder)
        documents = loader.load()

        print(f"📄 PDFs carregados: {len(documents)}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        print("✂️ Dividindo texto...")
        chunks = splitter.split_documents(documents)

        print(f"📦 Chunks criados: {len(chunks)}")

        print("🧠 Gerando embeddings e criando banco Chroma...")
        Chroma.from_documents(
            chunks,
            embedding=self.embedding_model,
            persist_directory=self.db_path
        )

        print("✅ Index criado com sucesso!")

    def load_index(self):
        print("📥 Carregando banco vetorial existente...")
        return Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embedding_model
        )

    def query(self, question, k=5):
        print("❓ Consultando...")
        db = self.load_index()
        results = db.similarity_search(question, k=k)
        return results
