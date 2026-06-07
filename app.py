"""
app.py — DocuMind Gradio UI (Steps 6 + 8: UI + project entry point)

Run: python app.py
Then open: http://localhost:7860
"""
import os
import sys
try:
    import gradio as gr
except ImportError:
    print("Missing dependency: gradio is not installed. Install with: pip install gradio")
    sys.exit(1)
from src.ingestion import ingest_pdf
from src.retriever import Retriever
from src.chain import answer_question
from src.config import TOP_K

# Global retriever — persists across questions in the same session
retriever = Retriever()
current_pdf_name = ""


def process_pdf(pdf_file):
    """Called when user uploads a PDF. Builds the FAISS index."""
    global current_pdf_name
    if pdf_file is None:
        return "⚠️ Please upload a PDF file.", gr.update(interactive=False)
    try:
        chunks = ingest_pdf(pdf_file.name)
        retriever.load_or_build(chunks)
        current_pdf_name = os.path.basename(pdf_file.name)
        msg = (
            f"✅ **{current_pdf_name}** indexed successfully!\n"
            f"- {len(chunks)} chunks created\n"
            f"- Embeddings stored in FAISS\n"
            f"- Ready to answer questions 🚀"
        )
        return msg, gr.update(interactive=True)
    except Exception as e:
        return f"❌ Error processing PDF: {str(e)}", gr.update(interactive=False)


def ask_question(question, history):
    """Called on each question. Runs full RAG pipeline."""
    if not question.strip():
        return history, "", ""

    if retriever.index is None:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": "⚠️ Please upload and process a PDF first."})
        return history, "", ""

    try:
        result = answer_question(question, retriever)

        # Format source chunks for display safely without any line-breaks in strings
        sources_md = f"**Retrieved {len(result['sources'])} chunks** (retrieval: {result['latency_ms']}ms)\n\n"
        for i, src in enumerate(result["sources"], 1):
            sources_md += f"**Source {i}** (relevance score: {src['score']:.2f})\n"
            truncated_chunk = src['chunk'][:400] + ('...' if len(src['chunk']) > 400 else '')
            sources_md += f"```\n{truncated_chunk}\n```\n\n"

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result["answer"]})
        return history, sources_md, ""

    except Exception as e:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
        return history, "", ""


# ── Gradio UI Layout ─────────────────────────────────────────────────────────
CSS = """
.container { max-width: 1100px; margin: auto; }
.upload-box { border: 2px dashed #4CAF50 !important; }
.status-box { font-size: 0.9em; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="DocuMind — PDF Q&A",
    theme=gr.themes.Soft(primary_hue="emerald"),
    css=CSS,
) as demo:
    gr.Markdown("""
    # 🧠 DocuMind
    ### RAG-powered PDF Question Answering
    Upload any PDF, then ask questions in natural language. Answers are grounded in your document.
    """)

    with gr.Row():
        # ── Left column: upload + status ────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📄 Step 1: Upload PDF")
            pdf_input = gr.File(
                label="Drop PDF here",
                file_types=[".pdf"],
                elem_classes=["upload-box"],
            )
            process_btn = gr.Button("⚙️ Process PDF", variant="primary")
            status_box  = gr.Markdown(
                value="*Upload a PDF to begin.*",
                elem_classes=["status-box"],
            )

            gr.Markdown("### ⚙️ Settings")
            gr.Markdown(f"- Embedding: `all-MiniLM-L6-v2`\n- LLM: `Mistral-7B-Instruct`\n- Top-K: `{TOP_K}` chunks\n- Chunk size: 500 chars")

        # ── Right column: chat ───────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Step 2: Ask Questions")
            chatbot = gr.Chatbot(height=380, label="Conversation")
            with gr.Row():
                question_input = gr.Textbox(
                    placeholder="e.g. What are the main findings of this paper?",
                    label="Your question",
                    scale=4,
                    interactive=False,
                )
                submit_btn = gr.Button("Ask →", variant="primary", scale=1)

            gr.Markdown("### 🔍 Source Chunks (Verify Grounding)")
            sources_box = gr.Markdown(value="*Sources will appear here after each answer.*")

    # ── Event wiring ─────────────────────────────────────────────────────────
    process_btn.click(
        fn=process_pdf,
        inputs=[pdf_input],
        outputs=[status_box, question_input],
    )

    submit_btn.click(
        fn=ask_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot, sources_box, question_input],
    )

    question_input.submit(
        fn=ask_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot, sources_box, question_input],
    )

    gr.Markdown("""
    ---
    **How it works:** PDF → PyMuPDF text extraction → 500-char chunks → MiniLM embeddings →
    FAISS similarity search → top-4 chunks → Mistral-7B generates grounded answer.
    """)


if __name__ == "__main__":
    print("\n🧠 DocuMind starting...")
    print("   Open: http://localhost:7860\n")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)