#!/usr/bin/env bash
# Idempotent Cloud Agent setup for the tokenizer_learn repo.
# Installs the CPU micro-benchmark dependencies (experiments/) system-wide so
# the scripts run with a plain `python3 experiments/...` and no venv activation.
set -euo pipefail

cd "$(dirname "$0")/.."

# System pip for the externally-managed system Python (PEP 668).
sudo apt-get update
sudo apt-get install -y --no-install-recommends python3-pip

# Python deps for experiments/ (transformers, tokenizers, tiktoken, jinja2).
# No --upgrade: keep reruns idempotent and avoid churning already-satisfied versions.
pip3 install --break-system-packages -r experiments/requirements.txt

# Pre-cache the default gpt2 tokenizer so benchmarks work without a live
# Hugging Face download and are resilient to unauthenticated Hub rate limits.
python3 - <<'PY'
from transformers import AutoTokenizer

AutoTokenizer.from_pretrained("gpt2", use_fast=True)
print("gpt2 tokenizer cached")
PY
