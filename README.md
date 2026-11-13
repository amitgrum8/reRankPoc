# Rerank PoC

The plan and its reasons are in `BLUEPRINT.md`. The pipeline:

1. **Data** (done, in `data/rr_input/`): Amazon Reviews 2023, Movies & TV. A training graph of the reviews
   before the cutoff (2014–2018, titled movies, 97k users, 49k movies, 1.14M reviews); 300 test users and 700
   training users (for the reranker) who wrote both a liked (4–5★) and a disliked (1–2★) review after the cutoff
   and at least two 1–2★ reviews before it. A persona is built from a user's 10 most recent reviews before the
   cutoff, always including two negative ones.
2. **Base ranker**: LightGCN, trained on the graph; the top 500 unseen movies per user.
3. **Personas**: EXP3RT (SIGIR 2025) steps 1–2 with gpt-oss:20b through Ollama, reasoning "low" (at "high"
   the model spends the whole 12k-token budget thinking and returns nothing for about 40% of the calls):
   likes and dislikes from each history review, then one persona per user (5 likes, 5 dislikes, short lines
   about the content; the profile answer must carry both tags). First the test users, then the training users
   (about 11 calls per user).
3b. **Item profiles** (`prep_items.py` locally once, then `item_profiles.py` on the GPU machine): for every
   movie in a run's lists, what reviewers praised and criticized (a few short lines each), from up to 3 lowest-
   and 3 highest-rated reviews before the cutoff. One call per movie (about 35k movies for a 300 + 700 user run,
   several hours), shared by all runs in `data/item_profiles/`.
4. **Reranker** (`rerank.py`, `train.py`, no LLM): every persona line, every movie text (title + description)
   and every profile line becomes a vector (bge-small); each movie in the list gets a row of numbers (closest
   like line, closest dislike line, closest praised line, closest criticized line, LightGCN score, ...). The
   formulas R1–R5 (movie description) and R8–R9 (movie reviews) turn the row into a score with fixed weights;
   R7 is LightGBM (lambdarank, label = the user interacted with the item) trained on the training users' rows,
   with the LightGCN score kept out of the model and added back as z(base) + z(model). R10–R12 are learned
   linear mixes (`mix.py`): logistic regression on the training users' lists over z(LightGCN) and each like /
   dislike line's best cosine, sorted per movie; their weights replace alpha and beta. R10 compares the persona
   with the movie description, R11 the dislikes with what reviewers criticized (item profiles), and R12 the likes
   with the sentences of the movie's good reviews (4–5 stars) and the dislikes with those of its bad reviews
   (1–2 stars), from `data/rr_input/item_reviews.jsonl.gz`.
5. **Evaluation** (`evaluate.py`): per reranker, how much the user's interacted items (the targets, any rating)
   moved inside LightGCN's list: mean rank change, moved up / stayed / moved down, pulled into the list; a
   bootstrap interval over users; and a control where every user gets another user's persona.

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
model if missing → reasoning check ("low" vs "high" output tokens) and 8 timed calls → personas → item profiles
(if `data/rr_input/item_reviews.jsonl.gz` exists; `--item-profiles skip` to leave them out) → evaluation.

Results land in `runs\<date_time>\`:

| File | What |
|---|---|
| `eval.md` | per reranker: movement of the interacted items inside the list (rank change, moved up / stayed / down, pulled in), bootstrap interval, and the swapped-persona control |
| `personas.jsonl`, `personas_train.jsonl` | one line per user: `like`, `dislike` lists (test users; training users) |
| `extractions.jsonl`, `extractions_train.jsonl` | the per-review likes/dislikes behind the personas |
| `candidates.jsonl`, `candidates_train.jsonl` | LightGCN's top 500 per user, and every target's score and rank |
| `lgbm.txt`, `lgbm.json` | the trained LightGBM reranker; its sizes, validation mean target rank and feature importance |
| `data/item_profiles/item_profiles.jsonl` | (shared, not per run) one line per movie: `praised`, `criticized` lines |
| `lightgcn.json`, `embeddings.npz` | training curve; the trained LightGCN (user and item vectors) |
| `summary.json`, `calls.jsonl` (+ `_train`) | LLM call counts, tokens, timing |
| `features.jsonl` | only with `evaluate.py --write-features`: one row per (user, movie), the reranker's numbers and the label |

Zip the folder to bring it back. LLM answers are cached in `runs\cache\` and shared by all runs, so a repeated or
resumed run (`--out runs\<same folder>`) costs no LLM time. `uv run python run.py --help` lists the options
(`--users 5`, `--model`, `--lgcn-minutes`, `--hours`).

## Files

| File | What it does |
|---|---|
| `run.py` | the whole pipeline in one command (above) |
| `prep_data.py` | (already run) `movies.db` + raw reviews → `data/rr_input/` |
| `prep_items.py` | (local, once) each movie's lowest- and highest-rated reviews before the cutoff → `data/rr_input/item_reviews.jsonl.gz` |
| `item_profiles.py` | (GPU machine) praised / criticized lines per movie → `data/item_profiles/item_profiles.jsonl` |
| `lightgcn.py` | trains LightGCN → `candidates.jsonl`, `lightgcn.json`, `embeddings.npz` |
| `personas.py`, `prompts.py`, `llm.py` | EXP3RT personas → `personas.jsonl`, `extractions.jsonl`, `calls.jsonl`, `summary.json` |
| `rerank.py` | persona reranker without an LLM: similarity features per (persona, movie), formulas R1–R5 and R8–R9 |
| `train.py` | LightGBM reranker R7: training users' lists from `embeddings.npz`, lambdarank over the feature rows |
| `mix.py` | learned linear mixes R10–R12: logistic regression over LightGCN + sorted persona similarities → `mix.json` |
| `evaluate.py` | metrics per reranker → `eval.md`, plus example users (`--show 2 --show-with R1_like`) |

Plumbing check without a GPU or an LLM (fake answers): `python run.py --mock --cpu --lgcn-minutes 0.5 --users 5 --train-users 20`.
Re-evaluating a run on any machine (the first time downloads bge-small, 130 MB, and encodes the movies in the
lists, minutes on a CPU, cached in `data/vectors/`): `python evaluate.py runs\<date_time>`, with `--alpha`,
`--beta`, `--gamma` for the formula weights and `--no-train` to skip LightGBM and the mixes. `--encoder NAME`
switches the text encoder for everything (known names and their query prefixes: `ENCODERS` in `rerank.py`;
vectors are cached per encoder in `data/vectors/`), and `--tag NAME` writes `eval_NAME.md` / `mix_NAME.json`, so
runs with different encoders sit side by side.

Budget of the real run at reasoning "low": about 11 LLM calls per user (10 reviews + 1 profile), roughly a
minute per user with 4 parallel requests, so 300 test users take about 5 hours and the 700 training users about
12 more. `--train-users 0` runs the test users only; `--hours N` caps the LLM time (test users first).
