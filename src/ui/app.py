import gradio as gr
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.multimodal_search import retrieve, retrieve_from_image
from rag.llm_generator import generate_answer, generate_answer_with_image

QUICK_QUESTIONS = [
    "What is the femur? (5 marks)",
    "Describe the brachial plexus (10 marks)",
    "Explain the structure of the heart (5 marks)",
    "What is the carpal tunnel? (2 marks)",
    "Describe the brachial artery",
    "What are the rotator cuff muscles?",
    "Explain the knee joint (10 marks)",
    "What is the sciatic nerve?",
]

# ============================================================
# UNIQUE UI DESIGN - Dark Teal & Amber Theme
# ============================================================
CSS = """
/* Base - Deep Teal Background */
body, .gradio-container {
    font-family: 'Inter', 'Poppins', 'Segoe UI', sans-serif;
    background: #0a0f14 !important;
}

/* Main Container with subtle border */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* HEADER - Elegant Gradient with Gold Accent */
#header-banner {
    background: linear-gradient(135deg, #0a1a1f 0%, #0d2a2a 50%, #0a1a1f 100%);
    border-radius: 24px;
    padding: 32px 40px;
    margin-bottom: 20px;
    border: 1px solid #2a6b6b;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    position: relative;
    overflow: hidden;
}
#header-banner::before {
    content: "⚕️";
    position: absolute;
    right: 20px;
    bottom: -20px;
    font-size: 120px;
    opacity: 0.05;
    pointer-events: none;
}
#header-banner h1 {
    background: linear-gradient(135deg, #e8d5b0 0%, #d4a84b 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
#header-banner p {
    color: #7ab3b3;
    font-size: 0.95rem;
    margin: 0;
    opacity: 0.85;
}

/* Custom Tabs Styling */
.tabs {
    border: none !important;
}
.tab-nav {
    background: transparent !important;
    border-bottom: 1px solid #1e3a3a !important;
    gap: 8px !important;
    padding: 0 !important;
}
.tab-nav button {
    background: transparent !important;
    color: #5a8f8f !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 12px 24px !important;
    transition: all 0.2s ease !important;
}
.tab-nav button:hover {
    color: #d4a84b !important;
    background: rgba(42, 107, 107, 0.2) !important;
    border-radius: 12px 12px 0 0 !important;
}
.tab-nav button.selected {
    color: #d4a84b !important;
    border-bottom-color: #d4a84b !important;
    background: rgba(212, 168, 75, 0.05) !important;
}

/* Input Cards - Glassmorphism */
#question-card, #image-card {
    background: rgba(13, 26, 26, 0.6) !important;
    backdrop-filter: blur(10px);
    border: 1px solid #2a5a5a !important;
    border-radius: 20px !important;
    padding: 20px !important;
    transition: all 0.2s ease;
}
#question-card:hover, #image-card:hover {
    border-color: #d4a84b !important;
    box-shadow: 0 4px 20px rgba(212, 168, 75, 0.1) !important;
}

/* Text Inputs */
#question-box textarea, #img-question-box textarea {
    background: #0a1212 !important;
    color: #d4e6e6 !important;
    border: 1.5px solid #2a5a5a !important;
    border-radius: 16px !important;
    font-size: 0.95rem !important;
    padding: 14px !important;
    transition: all 0.2s ease;
}
#question-box textarea:focus, #img-question-box textarea:focus {
    border-color: #d4a84b !important;
    box-shadow: 0 0 0 3px rgba(212, 168, 75, 0.15) !important;
    outline: none !important;
}
#question-box label, #img-question-box label {
    color: #d4a84b !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}

/* Number Inputs */
#marks-box input, #img-marks-box input {
    background: #0a1212 !important;
    color: #d4e6e6 !important;
    border: 1.5px solid #2a5a5a !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
}
#marks-box label, #img-marks-box label {
    color: #7ab3b3 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}

/* Buttons - Gold Gradient */
#ask-btn, #img-ask-btn {
    background: linear-gradient(135deg, #d4a84b 0%, #b8860b 100%) !important;
    color: #0a1212 !important;
    border: none !important;
    border-radius: 40px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    height: 48px !important;
    box-shadow: 0 4px 15px rgba(212, 168, 75, 0.3) !important;
    transition: all 0.2s ease !important;
}
#ask-btn:hover, #img-ask-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(212, 168, 75, 0.4) !important;
    filter: brightness(1.05);
}
#clear-btn, #img-clear-btn {
    background: rgba(26, 58, 58, 0.8) !important;
    color: #7ab3b3 !important;
    border: 1px solid #2a5a5a !important;
    border-radius: 40px !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    height: 48px !important;
    backdrop-filter: blur(5px);
}
#clear-btn:hover, #img-clear-btn:hover {
    background: #1e4a4a !important;
    color: #d4a84b !important;
    border-color: #d4a84b !important;
}

/* Quick Questions - Pill Style */
.quick-btn button {
    background: rgba(13, 26, 26, 0.8) !important;
    color: #7ab3b3 !important;
    border: 1px solid #2a5a5a !important;
    border-radius: 40px !important;
    font-size: 0.8rem !important;
    padding: 8px 18px !important;
    transition: all 0.15s ease !important;
    backdrop-filter: blur(5px);
}
.quick-btn button:hover {
    background: #1e4a4a !important;
    color: #d4a84b !important;
    border-color: #d4a84b !important;
    transform: translateY(-1px);
}

/* Answer Panels */
#answer-panel, #img-answer-panel {
    background: linear-gradient(135deg, #0d1a1a 0%, #0a1414 100%) !important;
    border: 1px solid #2a5a5a !important;
    border-radius: 20px !important;
    padding: 24px !important;
    min-height: 400px !important;
    color: #d4e6e6 !important;
}
#answer-panel h1, #answer-panel h2, #answer-panel h3,
#img-answer-panel h1, #img-answer-panel h2, #img-answer-panel h3 {
    color: #d4a84b !important;
}
#answer-panel code, #img-answer-panel code {
    background: #1a2a2a !important;
    border-radius: 8px !important;
    padding: 2px 6px !important;
}

/* Galleries */
#image-gallery, #img-image-gallery {
    background: #0a1212 !important;
    border: 1px solid #2a5a5a !important;
    border-radius: 20px !important;
    overflow: hidden !important;
}
.gallery-item {
    border-radius: 12px !important;
    transition: transform 0.2s ease !important;
}
.gallery-item:hover {
    transform: scale(1.02);
}

/* Sources Panel */
#sources-panel, #img-sources-panel {
    background: #0a1212 !important;
    border: 1px solid #2a5a5a !important;
    border-radius: 16px !important;
    padding: 16px !important;
    color: #7ab3b3 !important;
}

/* Stats Bar */
#stats-bar, #img-stats-bar {
    background: rgba(13, 26, 26, 0.8) !important;
    border-radius: 40px !important;
    padding: 10px 18px !important;
    color: #d4a84b !important;
    font-size: 0.8rem !important;
    font-family: monospace !important;
    border: 1px solid #2a5a5a !important;
    backdrop-filter: blur(5px);
    margin-top: 12px;
}

/* Detected Label */
#detected-label {
    background: rgba(212, 168, 75, 0.1) !important;
    border-left: 3px solid #d4a84b !important;
    border-radius: 12px !important;
    padding: 10px 16px !important;
    color: #d4a84b !important;
    font-size: 0.85rem !important;
    margin: 12px 0;
}

/* Image Upload Area */
#image-upload {
    background: rgba(10, 18, 18, 0.8) !important;
    border: 2px dashed #2a6b6b !important;
    border-radius: 20px !important;
    min-height: 240px !important;
    transition: all 0.2s ease;
}
#image-upload:hover {
    border-color: #d4a84b !important;
    background: rgba(212, 168, 75, 0.05) !important;
}

/* Section Labels */
.section-label {
    color: #d4a84b !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    margin-bottom: 12px !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #0a1212;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb {
    background: #2a6b6b;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #d4a84b;
}

/* Row spacing */
.gradio-row {
    gap: 20px !important;
}
"""


# ============================================================
# PIPELINE FUNCTIONS (unchanged)
# ============================================================

def ask_anatututor_text(question: str, marks):
    if not question or not question.strip():
        return "⚠️ Please type an anatomy question.", [], "—", \
               '<div id="stats-bar">Ask a question to see retrieval stats.</div>'

    marks_int = int(marks) if marks else 0
    full_q = question.strip()
    if marks_int > 0:
        full_q = f"{full_q} ({marks_int} marks)"

    start = time.time()
    text_results, image_results = retrieve(full_q, text_top_k=6, image_top_k=4)

    if not text_results:
        return ("⚠️ No relevant textbook sections found. Try rephrasing.", [],
                "—", '<div id="stats-bar">No results.</div>')

    context = "\n\n".join(
        f"[Source: {r['source'].replace('.txt','').replace('_',' ').title()}]\n{r['text']}"
        for r in text_results
    )

    answer = generate_answer(context, full_q, marks=marks_int)
    latency = time.time() - start

    image_paths, seen_srcs = [], set()
    for r in image_results:
        if os.path.exists(r["filepath"]):
            image_paths.append((r["filepath"], r["caption"]))
    for r in text_results:
        seen_srcs.add(r["source"].replace(".txt","").replace("_"," ").title())

    sources_md = "\n".join(f"📚 **{s}**" for s in seen_srcs) or "—"
    stats = (f'<div id="stats-bar">⚡ {latency:.2f}s  |  '
             f'📄 {len(text_results)} chunks  |  '
             f'🖼️ {len(image_paths)} images  |  '
             f'📖 {len(seen_srcs)} sources</div>')

    return answer, image_paths, sources_md, stats


def ask_anatututor_image(image, question: str, marks):
    if image is None:
        return ("⚠️ Please upload an anatomy image first.",
                "", [], "—",
                '<div id="img-stats-bar">Upload an image to start.</div>')

    marks_int = int(marks) if marks else 0

    start = time.time()
    description, text_results, image_results = retrieve_from_image(
        image, text_top_k=6, image_top_k=4
    )

    context_parts = []
    if description:
        context_parts.append(f"Image description: {description}")
    if text_results:
        context_parts.extend([
            f"[Source: {r['source'].replace('.txt','').replace('_',' ').title()}]\n{r['text']}"
            for r in text_results
        ])
    context = "\n\n".join(context_parts) if context_parts else "No textbook context retrieved."

    answer = generate_answer_with_image(
        image=image,
        question=question or "",
        context=context,
        marks=marks_int
    )

    latency = time.time() - start

    image_paths, seen_srcs = [], set()
    for r in image_results:
        if os.path.exists(r["filepath"]):
            image_paths.append((r["filepath"], r["caption"]))
    for r in text_results:
        seen_srcs.add(r["source"].replace(".txt","").replace("_"," ").title())

    sources_md = "\n".join(f"📚 **{s}**" for s in seen_srcs) or "—"
    detected_label = (
        f'<div id="detected-label">🔍 <strong>Structure Identified:</strong> {description}</div>'
    )
    stats = (f'<div id="img-stats-bar">⚡ {latency:.2f}s  |  '
             f'🔬 Vision: Llama 4 Scout  |  '
             f'📄 {len(text_results)} chunks  |  '
             f'🖼️ {len(image_paths)} related</div>')

    return answer, detected_label, image_paths, sources_md, stats


# ============================================================
# UI BUILD - Unique Layout
# ============================================================

def build_ui():
    with gr.Blocks(title="AnatoTutor", theme=gr.themes.Base(), css=CSS) as app:

        # Header with gold gradient text
        gr.HTML("""
        <div id="header-banner">
            <h1>⚕️ ANATOTUTOR</h1>
            <p>Multimodal RAG Learning System · Grounded in Gray's · BD Chaurasia · Vishram Singh · OpenStax</p>
        </div>
        """)

        with gr.Tabs():
            
            # ── TAB 1: Text Query ────────────────────────────
            with gr.Tab("📖 TEXT QUERY"):
                with gr.Row():
                    
                    # Left Column - Input
                    with gr.Column(scale=4, elem_id="question-card"):
                        gr.HTML('<div class="section-label">📝 YOUR QUESTION</div>')
                        question_box = gr.Textbox(
                            placeholder="e.g., Describe the femur bone (10 marks)\n      What is the brachial plexus?\n      Explain the knee joint for 5 marks",
                            lines=3, max_lines=6,
                            show_label=False, elem_id="question-box",
                        )
                        with gr.Row():
                            marks_box = gr.Number(
                                label="MARKS", value=0,
                                minimum=0, maximum=50, precision=0,
                                elem_id="marks-box", scale=1,
                            )
                            ask_btn   = gr.Button("🔍 ANALYSE", variant="primary",
                                                  elem_id="ask-btn", scale=3)
                            clear_btn = gr.Button("🗑️ CLEAR", elem_id="clear-btn", scale=1)

                        gr.HTML('<div class="section-label" style="margin-top: 16px;">⚡ QUICK QUESTIONS</div>')
                        with gr.Row():
                            for q in QUICK_QUESTIONS[:4]:
                                btn = gr.Button(q, elem_classes="quick-btn", size="sm")
                                btn.click(fn=lambda x=q: x, outputs=question_box)
                        with gr.Row():
                            for q in QUICK_QUESTIONS[4:]:
                                btn = gr.Button(q, elem_classes="quick-btn", size="sm")
                                btn.click(fn=lambda x=q: x, outputs=question_box)

                        stats_box = gr.HTML('<div id="stats-bar">✨ Ready for your anatomy question</div>')

                    # Right Column - Output
                    with gr.Column(scale=6):
                        with gr.Tabs():
                            with gr.Tab("📝 ANSWER"):
                                answer_box = gr.Markdown(
                                    "*Your detailed anatomy explanation will appear here...*",
                                    elem_id="answer-panel")
                            with gr.Tab("🖼️ RELATED IMAGES"):
                                image_gallery = gr.Gallery(
                                    show_label=False, columns=2, rows=2,
                                    height=450, object_fit="contain",
                                    elem_id="image-gallery", preview=True)
                            with gr.Tab("📚 SOURCES"):
                                sources_box = gr.Markdown(
                                    "*Textbook sources will appear after asking.*",
                                    elem_id="sources-panel")

                ask_btn.click(ask_anatututor_text,
                    inputs=[question_box, marks_box],
                    outputs=[answer_box, image_gallery, sources_box, stats_box])
                question_box.submit(ask_anatututor_text,
                    inputs=[question_box, marks_box],
                    outputs=[answer_box, image_gallery, sources_box, stats_box])
                clear_btn.click(
                    fn=lambda: ("", 0, "*Your detailed anatomy explanation will appear here...*", [],
                                "*Textbook sources will appear after asking.*",
                                '<div id="stats-bar">✨ Ready for your anatomy question</div>'),
                    outputs=[question_box, marks_box, answer_box,
                             image_gallery, sources_box, stats_box])

            # ── TAB 2: Image Query ───────────────────────────
            with gr.Tab("🖼️ IMAGE QUERY"):
                with gr.Row():
                    
                    # Left Column - Input
                    with gr.Column(scale=4, elem_id="image-card"):
                        gr.HTML('<div class="section-label">🔬 UPLOAD ANATOMY IMAGE</div>')
                        gr.HTML("""
                        <div style="color:#7ab3b3;font-size:0.85rem;margin-bottom:12px;">
                        Upload an X-ray, diagram, textbook figure, or specimen photo. 
                        AnatoTutor will identify the structure and explain it.
                        </div>
                        """)
                        image_upload = gr.Image(
                            label="", type="numpy", 
                            elem_id="image-upload", height=240,
                        )
                        
                        gr.HTML('<div class="section-label" style="margin-top: 16px;">💬 OPTIONAL QUESTION</div>')
                        img_question_box = gr.Textbox(
                            placeholder="e.g., What structure is shown here? (5 marks)\n      Identify the labeled parts\n      What is the clinical significance?",
                            lines=2, max_lines=4,
                            show_label=False, elem_id="img-question-box",
                        )
                        
                        with gr.Row():
                            img_marks_box = gr.Number(
                                label="MARKS", value=0,
                                minimum=0, maximum=50, precision=0,
                                elem_id="img-marks-box", scale=1,
                            )
                            img_ask_btn   = gr.Button("🔬 ANALYSE", variant="primary",
                                                      elem_id="img-ask-btn", scale=3)
                            img_clear_btn = gr.Button("🗑️ CLEAR", elem_id="img-clear-btn", scale=1)

                        detected_label_box = gr.HTML('<div id="detected-label" style="display: none;"></div>')
                        img_stats_box = gr.HTML('<div id="img-stats-bar">📸 Upload an image to begin analysis</div>')

                    # Right Column - Output
                    with gr.Column(scale=6):
                        with gr.Tabs():
                            with gr.Tab("📝 ANSWER"):
                                img_answer_box = gr.Markdown(
                                    "*Upload an anatomy image to get an explanation...*",
                                    elem_id="img-answer-panel")
                            with gr.Tab("🖼️ RELATED IMAGES"):
                                img_image_gallery = gr.Gallery(
                                    show_label=False, columns=2, rows=2,
                                    height=450, object_fit="contain",
                                    elem_id="img-image-gallery", preview=True)
                            with gr.Tab("📚 SOURCES"):
                                img_sources_box = gr.Markdown(
                                    "*Sources will appear after analysis.*",
                                    elem_id="img-sources-panel")

                img_ask_btn.click(ask_anatututor_image,
                    inputs=[image_upload, img_question_box, img_marks_box],
                    outputs=[img_answer_box, detected_label_box,
                             img_image_gallery, img_sources_box, img_stats_box])
                img_clear_btn.click(
                    fn=lambda: (None, "", 0,
                                "*Upload an anatomy image to get an explanation...*",
                                '<div id="detected-label" style="display: none;"></div>',
                                [], "*Sources will appear after analysis.*",
                                '<div id="img-stats-bar">📸 Upload an image to begin analysis</div>'),
                    outputs=[image_upload, img_question_box, img_marks_box,
                             img_answer_box, detected_label_box,
                             img_image_gallery, img_sources_box, img_stats_box])

    return app


# ============================================================
# LAUNCH
# ============================================================
if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        allowed_paths=[r"E:\Anato-tutor\data\images"],
    )