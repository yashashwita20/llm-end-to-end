# llm-end-to-end

End-to-end LLM learning lab in PyTorch: tokenizer → tiny GPT (from scratch) → eval harness → QLoRA fine-tuning → RAG.

This repo is designed to be:
- **portable** (runs on Colab/Kaggle GPUs, works on a Mac for dev/inference),
- **reproducible** (config-driven experiments),
- **portfolio-ready** (clear milestones + writeups).

---

## Repo layout

- `src/llm_e2e/` — reusable library code
  - `tokenizer/` — BPE training + encode/decode
  - `model/` — GPT architecture + sampling utilities
  - `train/` — training loop, checkpointing, optimization helpers
  - `eval/` — perplexity + prompt regression suite
  - `finetune/` — QLoRA fine-tuning utilities
  - `rag/` — chunking, indexing, retrieval, QA, safety tests
  - `data/` — dataset download/preprocess utilities
  - `utils/` — seeding, logging, IO

- `scripts/` — CLI entry points (train/eval/finetune/rag)
- `notebooks/` — demo notebooks (keep logic in `src/`, notebooks are thin)
- `experiments/configs/` — committed YAML configs
- `experiments/runs/` — run outputs (ignored by git)
- `tests/` — fast unit tests
- `docs/` — roadmap + learning notes
- `data/` — local datasets (ignored by git)

---

## Quickstart (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
pytest -q
