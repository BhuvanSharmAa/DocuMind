
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.config import TOP_K

_tokenizer = None
_model = None

def get_model():
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model
    model_name = "google/flan-t5-base"
    print(f"[LLM] Loading {model_name}...")
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print("[LLM] Model ready.")
    return _tokenizer, _model

def ask_llm(prompt: str) -> str:
    tokenizer, model = get_model()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def answer_question(question: str, retriever) -> dict:
    results = retriever.retrieve(question, k=TOP_K)
    if not results:
        return {"answer": "No relevant content found.", "sources": [], "latency_ms": 0}

    context = "\n\n---\n\n".join([r["chunk"] for r in results])
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided document."

Context:
{context}

Question: {question}

Answer:"""

    answer = ask_llm(prompt)
    return {"answer": answer, "sources": results, "latency_ms": results[0]["latency_ms"]}
