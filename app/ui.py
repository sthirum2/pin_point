"""Gradio demo UI for pin_point video search.
Launch:
    python -m app.ui
"""
from __future__ import annotations

import gradio as gr

from app.retrieval.loader import get_embed_fn, get_index, get_reranker

SOURCE_AUDIO_URL = "https://assembly.ai/wildfires.mp3"

CSS = """
.gradio-container {max-width: 880px !important; margin: auto;}
#title {margin-bottom: 0;}
#subtitle {color: var(--body-text-color-subdued); margin-top: 0.25rem;}
.gradio-container label span {
    background: none !important;
    color: var(--body-text-color-subdued) !important;
    box-shadow: none !important;
    font-weight: 500 !important;
}
.result-card {
    padding: 0.7rem 1rem;
    border-radius: 10px;
    background: var(--block-background-fill);
    border: 1px solid var(--border-color-primary);
    margin-bottom: 0.5rem;
    cursor: pointer;
}
.result-card:hover {
    border-color: var(--body-text-color-subdued);
}
.result-meta {
    font-size: 0.8rem;
    color: var(--body-text-color-subdued);
    margin-bottom: 0.25rem;
}
.result-score {
    float: right;
    font-family: var(--font-mono);
}
"""

# Seeks the visible <audio> element to `t` seconds and plays it.
SEEK_JS = """
(t) => {
    const audio = document.querySelector('#player audio');
    if (audio) {
        audio.currentTime = t;
        audio.play();
    }
    return [];
}
"""


def _format_result(rank: int, r) -> str:
    speaker = r.segment.metadata.get("speaker") or "?"
    start, end = r.segment.start, r.segment.end
    return (
        f'<div class="result-card" onclick="pinpointSeek({start:.2f})">'
        f'<div class="result-meta">#{rank} &middot; {start:.1f}s&ndash;{end:.1f}s &middot; '
        f'Speaker {speaker} <span class="result-score">{r.score:.3f}</span></div>'
        f'<div>{r.segment.text}</div>'
        f"</div>"
    )


def _search(query: str, k: int, use_rerank: bool) -> str:
    if not query.strip():
        return '<p style="color: var(--body-text-color-subdued);">Type a question to search the transcript.</p>'

    index = get_index()
    if index is None:
        return (
            '<p style="color: var(--body-text-color-subdued);">'
            "No index yet. Build one with:<br>"
            '<code>python scripts/build_index.py &lt;video-file&gt; data/index/demo</code></p>'
        )

    embed_fn = get_embed_fn()
    vec = embed_fn(query)
    results = index.query(vec, k=k)
    if use_rerank:
        reranker = get_reranker()
        results = reranker(results, query)

    if not results:
        return '<p style="color: var(--body-text-color-subdued);">No matches found.</p>'

    return "".join(_format_result(i + 1, r) for i, r in enumerate(results))


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="pin_point", css=CSS, theme=gr.themes.Soft()) as demo:
        gr.Markdown("# pin_point", elem_id="title")
        gr.Markdown(
            "Find the exact moment something was said — search a video by meaning, not keywords. "
            "Click a result to jump to that moment.",
            elem_id="subtitle",
        )

        gr.Audio(SOURCE_AUDIO_URL, elem_id="player", label="Source", interactive=False)

        with gr.Row():
            query_box = gr.Textbox(
                label="",
                placeholder="Ask something about the video...",
                scale=3,
                container=False,
            )
        with gr.Row():
            k_slider = gr.Slider(
                minimum=1, maximum=20, value=5, step=1,
                label="Number of results",
            )

        rerank_cb = gr.Checkbox(label="Rerank for precision (cross-encoder)", value=False)
        submit_btn = gr.Button("Search", variant="primary")
        results_box = gr.HTML()

        # Register the JS seek function once, globally, on page load.
        demo.load(
            fn=None,
            js="""
            () => {
                window.pinpointSeek = function(t) {
                    const audio = document.querySelector('#player audio');
                    if (audio) { audio.currentTime = t; audio.play(); }
                };
                return [];
            }
            """,
        )

        submit_btn.click(fn=_search, inputs=[query_box, k_slider, rerank_cb], outputs=results_box)
        query_box.submit(fn=_search, inputs=[query_box, k_slider, rerank_cb], outputs=results_box)

    return demo


if __name__ == "__main__":
    build_demo().launch()