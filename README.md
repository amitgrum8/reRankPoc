# Rerank PoC

The plan and its reasons are in `BLUEPRINT.md`. The pipeline:

1. **Data** (done, in `data/rr_input/`): Amazon Reviews 2023, Movies & TV. A training graph of the reviews
   before the cutoff (2014–2018, titled movies, 97k users, 49k movies, 1.14M reviews), and test users who wrote
   both a liked (4–5★) and a disliked (1–2★) review after it.
2. **Base ranker**: LightGCN, trained on the graph; the top 100 unseen movies per test user.
3. **Personas**: EXP3RT (SIGIR 2025) steps 1–2 with gpt-oss:20b through Ollama, reasoning "low" (at "high"
   the model spends the whole 12k-token budget thinking and returns nothing for about 40% of the calls):
   likes and dislikes from each history review, then one persona per user (about 5 likes, 5 dislikes).
4. **Reranker** (`rerank.py`, no LLM): every persona line and every movie text (title + description) becomes a
   vector (bge-small, CPU); each movie in the list gets a row of numbers (closest like line, closest dislike line,
   LightGCN score, ...) and the formulas R1–R5 from the blueprint turn the row into a score. The rows go to
   `features.jsonl`, the input of the LightGBM model that comes next.
5. **Evaluation**: where the liked and the disliked targets land in each reranker's list (`evaluate.py`).

## Setup on the GPU machine (Windows, NVIDIA, uv)

```powershell
git clone <repo-url> rerankPoc
cd rerankPoc
uv sync                      # makes .venv with torch 2.8 (CUDA 12.8), numpy, openai, sentence-transformers, lightgbm
```

Install Ollama from https://ollama.com/download (a normal installer, no admin rights needed), then **open a new
terminal** so `ollama` is on the PATH. The model (gpt-oss:20b, 13 GB) is pulled by `run.py` on its first run.

Without uv: `python -m venv .venv`, activate it, then
`pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128` and `pip install -r requirements.txt`.

## Run

```powershell
uv run python run.py --smoke     # 1 user, LightGCN 2 min: proves the setup end to end (10-15 min)
uv run python run.py             # the real run: every test user, LightGCN up to 30 min
```

`run.py` does, in order: GPU check → LightGCN → starts Ollama (own port, 16k context, 4 parallel) and pulls the
model if missing → reasoning check ("low" vs "high" output tokens) and 8 timed calls → personas → evaluation.

Results land in `runs\<date_time>\`:

| File | What |
|---|---|
| `eval.md` | metrics per reranker: liked targets (HR@10, NDCG@10, mean rank), disliked targets (avoid@10, mean rank), pair AUC |
| `features.jsonl` | one row per (user, movie in the list): the reranker's numbers and the label (2 liked, 0 disliked, 1 unseen) |
| `personas.jsonl` | one line per user: `like`, `dislike` lists |
| `extractions.jsonl` | the per-review likes/dislikes behind the personas |
| `candidates.jsonl` | LightGCN's top 100 per user, and every target's score and rank |
| `lightgcn.json`, `embeddings.npz` | training curve; the trained LightGCN (user and item vectors) |
| `summary.json`, `calls.jsonl` | LLM call counts, tokens, timing |

Zip the folder to bring it back. LLM answers are cached in `runs\cache\` and shared by all runs, so a repeated or
resumed run (`--out runs\<same folder>`) costs no LLM time. `uv run python run.py --help` lists the options
(`--users 5`, `--model`, `--lgcn-minutes`, `--hours`).

## Files

| File | What it does |
|---|---|
| `run.py` | the whole pipeline in one command (above) |
| `prep_data.py` | (already run) `movies.db` + raw reviews → `data/rr_input/` |
| `lightgcn.py` | trains LightGCN → `candidates.jsonl`, `lightgcn.json`, `embeddings.npz` |
| `personas.py`, `prompts.py`, `llm.py` | EXP3RT personas → `personas.jsonl`, `extractions.jsonl`, `calls.jsonl`, `summary.json` |
| `rerank.py` | persona reranker without an LLM: similarity features → `features.jsonl`, formulas R1–R5 |
| `evaluate.py` | metrics per reranker → `eval.md`, plus example users (`--show 2 --show-with R2_like-dis`) |

Plumbing check without a GPU or an LLM (fake answers): `python run.py --mock --cpu --lgcn-minutes 0.5`.
Re-evaluating a run on any machine (CPU, ~1 min; the first time downloads bge-small, 130 MB):
`python evaluate.py runs\<date_time>`, with `--alpha`, `--beta`, `--gamma` for the formula weights.
