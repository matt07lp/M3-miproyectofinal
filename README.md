# 📊 Financial reports Assistant

Asistente de análisis financiero basado en **Retrieval-Augmented Generation (RAG)** que responde preguntas sobre documentos SEC (10-K, 10-Q) utilizando un modelo **LLaMA 3.2-3B** fine-tuned con LoRA.

---

## 🏗️ Arquitectura
Usuario (Gradio UI)
       │
       ▼
  Input Guardrails  →  Bloquea prompt injection, SQL injection, off-topic
       │
       ▼
  Retrieval (ChromaDB)  →  Búsqueda híbrida + filtros de metadatos (10-K/10-Q)
       │
       ▼
  Generación (LLaMA 3.2 + LoRA)  →  Respuesta JSON estructurada
       │
       ▼
  Output Guardrails  →  Valida estructura JSON, detecta alucinaciones
       │
       ▼
  Respuesta al usuario


---

## 📁 Estructura del Proyecto
Proyecto Final/
├── app.py                  # Aplicación principal (pipeline RAG + interfaz Gradio)
├── ingest.py               # Script de ingesta de documentos a ChromaDB
├── sec_data/cleaned        # Documentos fuente (PDF, TXT, CSV de filings SEC)
├── notebooks/chroma_db/    # Base de datos vectorial (generada por ingest.py)
├── models/llama3b_lora/           # Modelo fine-tuned (adaptador LoRA)
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── requirements.txt        # Dependencias (opcional, ver instalación)
└── README.md


---

## ⚙️ Requisitos Previos

- **Python** 3.9+
- **macOS** con Apple Silicon (MPS) o **Linux/Windows** con GPU NVIDIA (CUDA)
- ~8 GB de RAM disponible mínimo

---

## 🚀 Instalación y Ejecución

### 1. Clonar o descargar el proyecto

```bash
cd /ruta/al/Proyecto\ Final
### 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
3. Instalar dependencias
pip install gradio transformers peft torch sentence-transformers chromadb accelerate langchain-community pypdf

4. Preparar la base vectorial (solo la primera vez)
Coloca tus documentos SEC (PDF, TXT, CSV) en la carpeta ./data/ y ejecuta:
python ingest.py

Esto procesará los documentos, los dividirá en chunks y los almacenará en ./chroma_db/. Si ya existe la carpeta chroma_db/ con datos, puedes omitir este paso.
5. Verificar el modelo fine-tuned
Asegúrate de que la carpeta ./llama3b_lora/ contenga:
adapter_config.json
adapter_model.safetensors
El modelo base (meta-llama/Llama-3.2-3B-Instruct) se descargará automáticamente de HuggingFace la primera vez.

6. Lanzar la aplicación
python app.py
La interfaz estará disponible en: http://127.0.0.1:7860

Formato de respuesta
El modelo devuelve un JSON estructurado:
{
  "answer": "Consolidated revenues were $350,018 million...",
  "confidence": "high",
  "sources": ["10-K 2024, page 45"],
  "warnings": []
}


Notas Técnicas
Embeddings: all-MiniLM-L6-v2 (sentence-transformers)
Chunking: RecursiveCharacterTextSplitter con chunk_size=1000, overlap=200
Generación: temperature=0.1, repetition_penalty=1.2, max_new_tokens=400
Búsqueda: Híbrida con priorización de documentos 10-K sobre 10-Q
Dispositivo: Detecta automáticamente MPS (Apple Silicon), CUDA o CPU