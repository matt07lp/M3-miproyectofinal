import os
import re
import json
import torch
import gradio as gr
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from pathlib import Path

# ══════════════════════════════════════════════════════════════
# 0. RUTAS — Ajusta BASE_DIR a tu entorno
# ══════════════════════════════════════════════════════════════
BASE_DIR      = Path("/Users/migueltorres/Documents/Documents - Miguel’s MacBook Pro/Diplomado AI and LLM for FM/Modulo 3/Proyecto Final/")  # Cambia si tu estructura está en otra carpeta
LORA_PATH     = f"{BASE_DIR}/models/llama3b_lora"
CHROMA_PATH   = f"{BASE_DIR}/notebooks/chroma_db"
COLLECTION    = "langchain"
BASE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"  # o el base que usaste para fine-tune

# ══════════════════════════════════════════════════════════════
# 1. CARGA DEL MODELO FINE-TUNED (Base + LoRA)
# ══════════════════════════════════════════════════════════════
print("⏳ Cargando modelo base + LoRA...")
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    device_map=None,          # ← NO usar "auto"
    dtype=torch.float16,
    low_cpu_mem_usage=True
)

model = PeftModel.from_pretrained(base_model, LORA_PATH)
model = model.merge_and_unload()  # Fusiona LoRA en el modelo base
model.eval()

# Mover a dispositivo disponible
import torch
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

model.eval()

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)
print("✅ Modelo fine-tuned cargado correctamente")

# ══════════════════════════════════════════════════════════════
# 2. EMBEDDINGS + CHROMADB
# ══════════════════════════════════════════════════════════════
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(name=COLLECTION)
print(f"✅ ChromaDB conectado — {collection.count()} documentos en '{COLLECTION}'")

# ══════════════════════════════════════════════════════════════
# 3. GUARDRAILS (Cell 8 de tu notebook)
# ══════════════════════════════════════════════════════════════
def input_guardrail(query: str) -> tuple:
    q = query.lower().strip()

    # Prompt injection
    injection_patterns = [
        r"ignore.*(?:previous|above|all).*instructions",
        r"you are now",
        r"act as",
        r"forget.*(?:previous|prior)",
        r"disregard.*(?:rules|instructions)",
        r"system\s*prompt",
        r"reveal.*(?:instructions|prompt)",
    ]
    for pat in injection_patterns:
        if re.search(pat, q):
            return (False, "Prompt injection detected")

    # SQL injection
    sql_patterns = [r"(\b(drop|delete|insert|update|alter)\b.*\b(table|database|from)\b)", r"(;\s*--)"]
    for pat in sql_patterns:
        if re.search(pat, q):
            return (False, "SQL injection detected")

    # Off-topic
    financial_keywords = [
        "revenue", "income", "profit", "loss", "earnings", "sales",
        "financial", "fiscal", "quarter", "annual", "10-k", "10-q",
        "balance", "cash flow", "assets", "liabilities", "equity",
        "sec", "filing", "report", "margin", "debt", "stock",
        "dividend", "ebitda", "operating", "net income", "gross",
        "consolidated", "segment", "growth", "decline", "forecast",
        "ingresos", "ganancias", "pérdidas", "financiero", "reporte",
        "utilidad", "activos", "pasivos", "deuda", "ventas",
    ]
    if not any(kw in q for kw in financial_keywords):
        return (False, "Query is off-topic. This system only answers financial questions about SEC filings.")

    return (True, "Query approved")


def output_guardrail(response_data: dict) -> tuple:
    if isinstance(response_data, str):
        try:
            response_data = json.loads(response_data)
        except json.JSONDecodeError:
            return (False, "Response is not valid JSON")

    answer = str(response_data.get("answer", "")).lower()

    # Check for refusals (acceptable)
    refusal_phrases = ["no relevant", "not enough information", "cannot answer", "no data available"]
    if any(phrase in answer for phrase in refusal_phrases):
        return (True, "Model appropriately refused due to lack of data")

    # Hallucination red flags
    hallucination_flags = [
        "for demonstration purposes", "as an example", "i don't have access",
        "as an ai", "i cannot", "hypothetical", "let me make up",
    ]
    for flag in hallucination_flags:
        if flag in answer:
            return (False, f"Possible hallucination detected: '{flag}'")

    # Validate confidence
    confidence = str(response_data.get("confidence", "")).lower()
    if confidence and confidence not in ["high", "medium", "low"]:
        return (False, f"Invalid confidence value: '{confidence}'")

    return (True, "Response approved")


def parse_guardrail(result) -> tuple:
    if isinstance(result, tuple) and len(result) == 2:
        return result
    if isinstance(result, dict):
        return (result.get("approved", False), result.get("reason", "Unknown"))
    if isinstance(result, bool):
        return (result, "Approved" if result else "Blocked")
    return (False, "Unrecognized guardrail format")

# ══════════════════════════════════════════════════════════════
# 4. BÚSQUEDA HÍBRIDA (Cell 7 — search_hybrid_v3)
# ══════════════════════════════════════════════════════════════
def search_hybrid_v3(query: str, n_results: int = 5):
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"filing_type": {"$in": ["10-K", "10-Q"]}},
        include=["documents", "metadatas", "distances"]
    )
    return results

# ══════════════════════════════════════════════════════════════
# 5. GENERACIÓN RAG (Cell 9 de tu notebook)
# ══════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are a financial analyst assistant. You answer questions ONLY based on the provided context from SEC filings (10-K and 10-Q).

Rules:
- Extract EXACT figures from the context. Do not round or estimate.
- If the context contains tabular data, read the numbers carefully by matching row labels to column headers.
- If the context does not contain enough information, say so explicitly.
- Respond ONLY with valid JSON in this format:

{"answer": "your answer here", "confidence": "high|medium|low", "sources": ["source1"], "warnings": ["any warnings"]}
"""


def extract_json(text: str) -> dict:
    """Extrae el primer bloque JSON válido del texto del modelo."""
    # Intentar encontrar JSON con regex
    patterns = [
        r'\{[^{}]*\}',
        r'\{.*?\}',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    # Fallback: intentar parsear todo el texto
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "answer": text.strip(),
            "confidence": "low",
            "sources": [],
            "warnings": ["Failed to parse structured JSON from model output"]
        }


def generate_response(query: str) -> dict:
    # 1. Input guardrail
    approved, reason = parse_guardrail(input_guardrail(query))
    if not approved:
        return {"answer": f"Query blocked: {reason}", "confidence": "N/A", "sources": [], "warnings": [reason]}

    # 2. Retrieval
    results = search_hybrid_v3(query, n_results=5)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        return {"answer": "No relevant documents found.", "confidence": "low", "sources": [], "warnings": ["Empty retrieval"]}

    # 3. Priorizar 10-K sobre 10-Q y limitar contexto
    doc_pairs = list(zip(documents, metadatas, distances))
    doc_pairs.sort(key=lambda x: (0 if x[1].get("filing_type") == "10-K" else 1, x[2]))

    context = ""
    sources = []
    for doc, meta, dist in doc_pairs:
        if len(context) + len(doc) > 3000:
            break
        context += doc + "\n\n"
        source_label = f"{meta.get('company', 'N/A')} | {meta.get('filing_type', 'N/A')} | dist={dist:.4f}"
        sources.append(source_label)

    # 4. Prompt
    full_prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}\n\nJSON:"

    # 5. Generación con TU modelo fine-tuned
    output = generator(
        full_prompt,
        max_new_tokens=400,
        temperature=0.1,
        do_sample=True,
        top_p=0.8,
        repetition_penalty=1.2,
        return_full_text=False  # Solo la respuesta, no el prompt
    )
    raw_output = output[0]["generated_text"].strip()

    # 6. Extraer JSON
    response_data = extract_json(raw_output)
    if "sources" not in response_data or not response_data["sources"]:
        response_data["sources"] = sources

    # 7. Output guardrail
    out_approved, out_reason = parse_guardrail(output_guardrail(response_data))
    if not out_approved:
        response_data["warnings"] = response_data.get("warnings", []) + [f"Output guardrail: {out_reason}"]

    return response_data

# ══════════════════════════════════════════════════════════════
# 6. INTERFAZ GRADIO
# ══════════════════════════════════════════════════════════════
def chat_fn(user_message, history):
    result = generate_response(user_message)

    # Formatear respuesta legible
    answer     = result.get("answer", "No answer")
    confidence = result.get("confidence", "N/A")
    sources    = result.get("sources", [])
    warnings   = result.get("warnings", [])

    response = f"**📊 Answer:** {answer}\n\n"
    response += f"**🎯 Confidence:** {confidence}\n\n"

    if sources:
        response += "**📄 Sources:**\n"
        for s in sources:
            response += f"- {s}\n"

    if warnings:
        response += "\n**⚠️ Warnings:**\n"
        for w in warnings:
            response += f"- {w}\n"

    return response


demo = gr.ChatInterface(
    fn=chat_fn,
    title="📊 Financial reports Assistant",
    description="Ask financial questions about SEC filings (10-K, 10-Q). Powered by fine-tuned LLaMA + RAG.",
    examples=[
        "What were the consolidated revenues for the most recent fiscal year?",
        "What are the main risk factors reported in the latest 10-K?",
        "What is the company's total long-term debt?"
    ],

)

demo.launch(theme=gr.themes.Soft())
