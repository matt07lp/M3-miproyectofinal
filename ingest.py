import os
import glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path
# ══════════════════════════════════════════════════════════════
# 0. RUTAS — Deben coincidir con app.py
# ══════════════════════════════════════════════════════════════
BASE_DIR     = "Path("/Users/migueltorres/Documents/Documents - Miguel’s MacBook Pro/Diplomado AI and LLM for FM/Modulo 3/Proyecto Final")
"
DATA_DIR     = f"{BASE_DIR}/sec_data/cleaned"          # Carpeta con tus PDFs/TXTs de SEC filings
CHROMA_PATH  = f"{BASE_DIR}/notebooks/chroma_db"
COLLECTION   = "langchain"

# ══════════════════════════════════════════════════════════════
# 1. MODELO DE EMBEDDINGS (mismo que app.py)
# ══════════════════════════════════════════════════════════════
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Modelo de embeddings cargado")

# ══════════════════════════════════════════════════════════════
# 2. CARGAR DOCUMENTOS
# ══════════════════════════════════════════════════════════════
def load_documents(data_dir: str) -> list:
    """Carga PDFs, TXTs y CSVs desde la carpeta de datos."""
    docs = []

    # PDFs
    for pdf_path in glob.glob(os.path.join(data_dir, "**", "*.pdf"), recursive=True):
        try:
            loader = PyPDFLoader(pdf_path)
            loaded = loader.load()
            # Inferir metadata del nombre del archivo
            filename = os.path.basename(pdf_path).lower()
            filing_type = "10-K" if "10-k" in filename or "10k" in filename else \
                          "10-Q" if "10-q" in filename or "10q" in filename else "other"
            for doc in loaded:
                doc.metadata["source"] = pdf_path
                doc.metadata["filing_type"] = filing_type
                doc.metadata["company"] = extract_company_name(filename)
            docs.extend(loaded)
            print(f"  📄 PDF cargado: {pdf_path} ({len(loaded)} páginas)")
        except Exception as e:
            print(f"  ❌ Error cargando {pdf_path}: {e}")

    # TXTs
    for txt_path in glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True):
        try:
            loader = TextLoader(txt_path, encoding="utf-8")
            loaded = loader.load()
            filename = os.path.basename(txt_path).lower()
            filing_type = "10-K" if "10-k" in filename or "10k" in filename else \
                          "10-Q" if "10-q" in filename or "10q" in filename else "other"
            for doc in loaded:
                doc.metadata["source"] = txt_path
                doc.metadata["filing_type"] = filing_type
                doc.metadata["company"] = extract_company_name(filename)
            docs.extend(loaded)
            print(f"  📝 TXT cargado: {txt_path}")
        except Exception as e:
            print(f"  ❌ Error cargando {txt_path}: {e}")

    # CSVs
    for csv_path in glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True):
        try:
            loader = CSVLoader(csv_path)
            loaded = loader.load()
            filename = os.path.basename(csv_path).lower()
            filing_type = "10-K" if "10-k" in filename or "10k" in filename else \
                          "10-Q" if "10-q" in filename or "10q" in filename else "other"
            for doc in loaded:
                doc.metadata["source"] = csv_path
                doc.metadata["filing_type"] = filing_type
                doc.metadata["company"] = extract_company_name(filename)
            docs.extend(loaded)
            print(f"  📊 CSV cargado: {csv_path}")
        except Exception as e:
            print(f"  ❌ Error cargando {csv_path}: {e}")

    return docs


def extract_company_name(filename: str) -> str:
    """Intenta extraer nombre de empresa del nombre del archivo."""
    # Ejemplo: "apple_10-k_2024.pdf" → "apple"
    # Ajusta esta lógica según tu convención de nombres
    name = filename.split("_")[0] if "_" in filename else filename.split(".")[0]
    name = name.replace("-", " ").replace("10 k", "").replace("10 q", "").strip()
    return name.title() if name else "Unknown"

# ══════════════════════════════════════════════════════════════
# 3. CHUNKING
# ══════════════════════════════════════════════════════════════
def split_documents(docs: list) -> list:
    """Divide documentos en chunks preservando metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ {len(docs)} documentos → {len(chunks)} chunks")
    return chunks

# ══════════════════════════════════════════════════════════════
# 4. INGESTA A CHROMADB
# ══════════════════════════════════════════════════════════════
def ingest_to_chroma(chunks: list):
    """Genera embeddings e inserta en ChromaDB."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Eliminar colección existente si quieres re-ingestar desde cero
    existing = [c.name for c in client.list_collections()]
    if COLLECTION in existing:
        print(f"⚠️  Colección '{COLLECTION}' ya existe. Eliminando para re-ingesta...")
        client.delete_collection(name=COLLECTION)

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}  # Distancia coseno
    )

    # Procesar en lotes para no saturar memoria
    BATCH_SIZE = 50
    total = len(chunks)

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]

        texts = [chunk.page_content for chunk in batch]
        metadatas = []
        for chunk in batch:
            meta = {
                "source": str(chunk.metadata.get("source", "unknown")),
                "filing_type": str(chunk.metadata.get("filing_type", "other")),
                "company": str(chunk.metadata.get("company", "Unknown")),
            }
            # Agregar página si existe
            if "page" in chunk.metadata:
                meta["page"] = int(chunk.metadata["page"])
            metadatas.append(meta)

        ids = [f"doc_{i + j}" for j in range(len(batch))]

        # Generar embeddings con el MISMO modelo que usa app.py
        embeddings = embedder.encode(texts).tolist()

        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )

        print(f"  📥 Insertados {min(i + BATCH_SIZE, total)}/{total} chunks")

    print(f"\n✅ Ingesta completa: {collection.count()} documentos en '{COLLECTION}'")

    # Verificación rápida
    sample = collection.peek(limit=3)
    print(f"\n── Muestra de verificación ──")
    for i, (doc, meta) in enumerate(zip(sample["documents"], sample["metadatas"])):
        print(f"  [{i}] {meta['company']} | {meta['filing_type']} | {doc[:80]}...")

# ══════════════════════════════════════════════════════════════
# 5. EJECUCIÓN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  INGESTA DE DOCUMENTOS SEC → ChromaDB")
    print("=" * 60)

    if not os.path.exists(DATA_DIR):
        print(f"\n❌ No se encontró la carpeta '{DATA_DIR}'")
        print(f"   Crea la carpeta y coloca tus archivos SEC (PDF/TXT/CSV)")
        print(f"   Ejemplo de nombres:")
        print(f"     apple_10-k_2024.pdf")
        print(f"     microsoft_10-q_2024.txt")
        exit(1)

    # Paso 1: Cargar
    print(f"\n📂 Buscando documentos en: {DATA_DIR}")
    docs = load_documents(DATA_DIR)
    if not docs:
        print("❌ No se encontraron documentos. Verifica la carpeta y formatos.")
        exit(1)

    # Paso 2: Chunk
    print(f"\n✂️  Dividiendo en chunks...")
    chunks = split_documents(docs)

    # Paso 3: Ingestar
    print(f"\n📥 Ingresando a ChromaDB...")
    ingest_to_chroma(chunks)

    print(f"\n{'=' * 60}")
    print(f"  ✅ LISTO — Ahora puedes ejecutar: python app.py")
    print(f"{'=' * 60}")
