#!/usr/bin/env bash
set -e

source .venv/bin/activate
export EMBEDDINGS_BACKEND="hf"
export EMBEDDING_MODEL_NAME="BAAI/bge-m3"
export EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda}"

python -m modules.embedding.build_qdrant
python -m modules.retrieval.test_search
