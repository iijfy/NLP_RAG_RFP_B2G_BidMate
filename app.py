from __future__ import annotations

import os
from modules.ui.gradio_app import build_demo

if __name__ == "__main__":
    # 더미로 UI 확인하기 위한 기본값(원하면 쉘에서 export로 덮어쓰기)
    os.environ.setdefault("EMBEDDINGS_BACKEND", "dummy")
    os.environ.setdefault("GENERATOR_BACKEND", "dummy")
    os.environ.setdefault("JUDGE_BACKEND", "dummy")
    os.environ.setdefault("QDRANT_COLLECTION", "rfp_recursive_DUMMY")
    os.environ.setdefault("RAG_TOP_K", "3")

    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
