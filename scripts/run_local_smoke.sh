#!/usr/bin/env bash
set -e

source .venv/bin/activate

export EMBEDDINGS_BACKEND="dummy"
export QDRANT_COLLECTION="rfp_recursive_DUMMY"
export GENERATOR_BACKEND="dummy"

python -m modules.embedding.build_qdrant
python -m modules.retrieval.test_search
python -m modules.rag.test_rag
