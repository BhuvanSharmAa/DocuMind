"""
config.py — Central configuration for DocuMind.
All tuneable knobs live here. Change values in .env to override.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file into os.environ

# ── HuggingFace ──────────────────────────────────────────────────────────────
HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

# ── Models ───────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# all-MiniLM-L6-v2: 384-dim embeddings, 80 MB, extremely fast, free forever.

LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
# Mistral-7B via HF Inference API — runs on HF's servers, free tier ~1k req/day.

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
# Why 500/50?
#   - 500 chars ≈ ~120 tokens — fits comfortably in MiniLM's 256-token window
#   - 50-char overlap prevents answers that straddle chunk boundaries being lost
#   - Tune UP chunk size if answers feel incomplete; DOWN if they're too noisy

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K_RESULTS", 4))
# TOP_K = 4 means: fetch the 4 most similar chunks and feed them to the LLM.
# Tune UP for complex multi-part questions; DOWN to reduce hallucination risk.

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")
LOG_DIR         = os.path.join(BASE_DIR, "logs")

for _dir in [DATA_DIR, VECTORSTORE_DIR, LOG_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ── LLM generation params ───────────────────────────────────────────────────
MAX_NEW_TOKENS  = 512
TEMPERATURE     = 0.1   # Low temp → more faithful, less creative. Good for QA.
