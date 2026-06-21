"""Gradio demo UI for pin_point video search.

Launch:
    python -m app.ui
"""
from __future__ import annotations

import gradio as gr

from app.retrieval.loader import get_embed_fn, get_index, get_reranker


def _search(query: str, k: int, use_rerank: bool) -> str:
    if not query.strip():
        return "Enter a query above."

    index = get_index()
    if index is None:
        return (
            "No index built yet.\n"
            "Run:  python scripts/build_index.py <video-file> data/index/demo"
        )

    embed_fn = get_embed_fn()
    vec = embed_fn(query)
    results = index.query(vec, k=k)

    if use_rerank:
        reranker = get_reranker()
        results = reranker(results, query)

    if not results:
        return "No results found."

    lines: list[str] = []
    for r in results:
        speaker = r.segment.metadata.get("speaker") or "?"
        lines.append(
            f"[{r.segment.start:.1f}s – {r.segment.end:.1f}s]  {speaker} — "
            f"{r.segment.text}  (score: {r.score:.4f})"
        )
    return "\n\n".join(lines)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="pin_point · Video Search") as demo:
        gr.Markdown("## pin_point — Video Segment Search")
        gr.Markdown(
            "Search a transcribed video by meaning. "
            "Build an index first with `scripts/build_index.py`."
        )

        with gr.Row():
            query_box = gr.Textbox(
                label="Query",
                placeholder="e.g. goalkeeper makes a save",
                scale=4,
            )
            k_slider = gr.Slider(
                minimum=1, maximum=20, value=5, step=1,
                label="Results (k)",
                scale=1,
            )

        rerank_cb = gr.Checkbox(label="Rerank with cross-encoder", value=False)
        submit_btn = gr.Button("Search", variant="primary")
        results_box = gr.Textbox(label="Results", lines=12, interactive=False)

        submit_btn.click(
            fn=_search,
            inputs=[query_box, k_slider, rerank_cb],
            outputs=results_box,
        )
        query_box.submit(
            fn=_search,
            inputs=[query_box, k_slider, rerank_cb],
            outputs=results_box,
        )

    return demo


if __name__ == "__main__":
    build_demo().launch()
