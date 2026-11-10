# Persona reranker PoC: blueprint

Status: plan only, no code yet (7 Nov 2026). This is a clean start. It does not build on the LLM persona PoC in `E:\thies\poc`. Thesis context: `C:\master degree\thies\CLAUDE.md`.

## 1. The idea

A first-stage ranker (popularity, BPR-MF, LightGCN, any recommender) returns the top K items for a user. A second stage reorders those K items using the user's persona: a list of short text lines that say what the user likes and what the user dislikes.

The second stage never calls an LLM. It compares persona lines with item text through a fast text encoder, using vectors computed ahead of time, plus some exact-word matching. Then it adds a like bonus and a dislike penalty to the base score. A small model trained on past data decides how much each signal counts (section 6).

The persona stays as text, and each line can be read and edited. That keeps the thesis rule that a persona must be interpretable ("embeddings don't count"). Embeddings are only the scoring tool, not the persona. Every big demotion can be traced back to one dislike line.

"Personality" here means the user's taste persona (likes and dislikes), not the Big Five. The Big Five stays a later direction (C in CLAUDE.md).

## 2. What the literature says, and what it changes in the design

| Finding | Source | What it means for us |
|---|---|---|
| Retrieval models mostly ignore negation. On pairs that differ only by a negation, bi-encoders, sparse models and late-interaction models score below random. Cross-encoders score just above random. | NevIR, EACL 2024 | Never embed "dislikes X" and expect the vector to know it is negative. The polarity comes from the list a line sits in. The text we embed is X itself. |
| Retrievers fail on exclusion queries ("…but not X"). ColBERT-style models are especially weak. | ExcluIR, AAAI 2025 | Same. We also run the naive version on purpose as a baseline (R5). |
| Universal text embeddings treat negated pairs as similar. There are fixes, but they need re-weighting or fine-tuning. | Cao 2025 (ECAI); Truong et al. 2025 | Negation-aware fine-tuning is a later option (level L3), not step one. |
| Rocchio (q' = αq + β·mean(relevant) − γ·mean(non-relevant)) works with dense vectors. The negative term is usually dropped. | Vector-PRF, Li et al. (TOIS); Pyserini docs | This is the "merged" baseline R4: one user vector, likes minus dislikes. Whether the negative term helps in dense space is still untested, and that is something we can test. |
| Users have several interests, and one averaged vector blurs them. The user–item score is the best-matching interest vector. | MIND (Li et al. 2019); ComiRec (KDD 2020) | One vector per persona line, scored by the maximum over lines (ColBERT's MaxSim works the same way). Mean pooling is an ablation. |
| Separate interest and disinterest embeddings, and separate positive and negative encoders, help. | PANE-GNN; PNFRec (RecSys 2025); NFARec | This supports keeping the like and dislike channels apart (Q3). These works learn from item IDs, not text. |
| Aspect-level sentiment from reviews can drive recommendations and dis-recommendations on explicit features, with explanations. | EFM, Zhang et al. (SIGIR 2014) | A non-LLM way to build the persona from text: sentence or aspect sentiment turns review text into like and dislike lines. |
| Editable text profiles made of concepts, retrieved from the user's documents, with no LLM at scoring time. | LACE, Mysore et al. (SIGIR 2023) | The closest earlier design. LACE does not model dislikes, which is our gap. |
| Text profiles can replace user embeddings at a small cost, but an LLM does the scoring. BLUE aligns text profiles with an embedding objective. | Ramos et al. (ACL 2024); BLUE (arXiv 2026) | Text personas carry the signal. Our question is whether a cheap scorer can read them, dislikes included. |
| Persona as a set of sentences about features the user likes and dislikes, taken from real reviews. | PEARL (ACL Findings 2024) | Our persona format: short sentences, each with a polarity. |
| Sentence encoders trained on Amazon Reviews 2023 to link item metadata and natural language. | BLaIR, Hou et al. 2024 | A domain encoder candidate. Caveat: its training data covers our test period. |
| Static embeddings run 100–400× faster on CPU and keep about 85% of the quality. | Model2Vec / potion; sentence-transformers static models (Aarsen, Jan 2025) | The fastest tier. Speed only matters when the persona is encoded online. Item vectors are computed once. |

## 3. Pipeline

```
OFFLINE (once, or when data changes)
  item text: title, genres, description ──────────────────────► meta vector m_i
  other users' reviews of the item, before the cutoff ─────────► praise snippets P_i, complaint snippets C_i
  the user's reviews before time t ─► persona builder ─────────► like lines L_u, dislike lines D_u (text + vectors)

ONLINE (per request)
  base ranker ─► top K (K = 100) with scores b_i
  for each candidate i:
      like_i = max over l in L_u of  max( cos(l, m_i), max over p in P_i of cos(l, p) )
      dis_i  = max over d in D_u of  max( cos(d, m_i), max over c in C_i of cos(d, c) )
      features_i = [b_i, like_i, dis_i, per-channel similarities, BM25, name match, item stats]
      score_i = f(features_i)        f = a formula (L0) or a trained model (L1, L2), see section 6
  sort ─► final list, plus the persona line behind each large move (the explanation)
```

**Why likes are matched to praise and dislikes to complaints.** "I loved how slow and quiet it was" and "it was far too slow" embed close to each other. If a dislike line were matched against what other users praised, we would penalize the wrong thing. Matching each channel to the snippets of the same polarity avoids this, and the metadata channel stays neutral. This is the "item complaint profile" from the earlier Experiment 2 plan in CLAUDE.md.

**Cost.** 100 candidates × about 17 vectors each × up to 40 lines × 384–768 dimensions is roughly 50M multiply-adds, a few milliseconds in NumPy on a CPU. A LightGBM prediction over 100 rows takes well under a millisecond. The persona is encoded only when it changes. Target to measure: p95 under 10 ms per request on CPU for the rerank step alone.

## 4. Persona (input)

One persona line:

```json
{"text": "slow pacing with no payoff", "polarity": "dislike",
 "review_id": "…", "stars": 4, "time": "2021-03-02", "kind": "taste"}
```

- `kind`: `taste` or `product` (disc, shipping, transfer, packaging). Product lines are left out of the taste persona by default. An ablation puts them back.

Persona builders, all without an LLM:

| ID | Likes | Dislikes | Note |
|---|---|---|---|
| P0 | none | none | base ranker only |
| P-item | text of the items rated 4–5★ | text of the items rated 1–2★ | baseline: whole items, no sentence extraction |
| P-sent | positive sentences from the user's reviews | negative sentences from the user's reviews (in 1–2★ reviews, neutral sentences count as dislikes too) | the main persona |
| P-LLM (optional) | short phrases an LLM extracts once, offline | same | comparison only; the reranker still has no LLM |

Building P-sent: split reviews into sentences, then label each one with a small sentiment classifier (an SST-2 DistilBERT) or an aspect-sentiment model (PyABSA), offline. Keep only confident lines. Merge near-duplicates (cos > 0.9). Keep at most 20 likes and 20 dislikes, newest first.

Personas over time (Q2), still without an LLM:
- **static**: all lines before t, equal weight.
- **decay**: each line's similarity is weighted by exp(−Δt / h) before taking the max.
- **revise**: when a new dislike line is close to an older like line (cos > θ), the like line gets less weight. This is Moshe's idea of updating the positive persona after a negative experience, done by a rule.

## 5. Scorers (the conditions)

z(·) is a z-score inside the candidate list, so α and β mean the same thing for every encoder.

| ID | Score | What it tests |
|---|---|---|
| R0 | z(b) | base ranker alone |
| R1 | z(b) + α·z(like) | likes only |
| R2 | z(b) + α·z(like) − β·z(dis) | likes and dislikes, linear |
| R3 | z(b) + α·z(like) − β·max(0, dis − τ) | soft boundary: penalize only clear matches. τ is the 95th percentile of dis over random items. |
| R3h | R1, with every item whose dis > τ_h moved to the bottom | hard boundary (a filter) |
| R4 | z(b) + α·z(cos(u, m_i)), with u = mean(L) − γ·mean(D) | one merged vector (Rocchio) |
| R5 | z(b) + α·z(cos(e("Likes: … Dislikes: …"), m_i)) | naive: the whole persona as one string. Expected to ignore negation. |
| R6 | R3 + BM25(likes → item text) − BM25(dislikes → item text) + name match (cast, director, series) | exact names, which embeddings blur |
| R7 | a trained model over all features (L1 or L2 in section 6) | learned fusion |
| X1 (reference) | R3, with a small cross-encoder scoring each (line, item text) pair | is a stronger matcher worth its latency? |

Ablation for every R: replace the max over lines with the mean (one interest vector).

## 6. How the reranker is trained

### What is fixed and what is learned

- **Fixed:** the base ranker (trained separately on interactions, as it would be in a real system), the text encoder (off the shelf, frozen), and the persona builder.
- **Learned:** how to combine the signals. That means how much to trust the base score, how strongly likes and dislikes count, from what similarity a dislike starts to matter, and which channel to believe (metadata, praise, complaints or names).
- **No user ID goes into the reranker.** It sees only persona text and item text. So it learns one general rule for reading a persona, it works for users it never saw in training, and it reacts when a user edits their persona.

### Four levels; move to the next one only if the current one is not enough

| Level | Model | What is learned | How it is trained |
|---|---|---|---|
| L0 | the formulas R1–R6 | 3–5 weights (α, β, τ, γ) | grid search on validation users |
| **L1 (main)** | LightGBM LambdaMART | trees over about 25 features | ranking loss (`lambdarank`) on graded labels, minutes on a CPU |
| L2 | a small adapter on the frozen vectors: one linear projection per channel, plus learned soft-max pooling over persona lines instead of a hard max | a few d×d′ matrices and a temperature | pairwise or listwise loss on the same examples |
| L3 (later) | fine-tune the encoder itself, with disliked items as hard negatives | the whole encoder | sentence-transformers contrastive loss |

L2 keeps inference fast, because each projection is applied once to the cached vectors. L3 is out of scope for this PoC.

**L1 features** (per candidate): the base score and base rank; the item's average rating and review count before t; the max, mean and top-3 mean of like similarity and of dislike similarity in each channel (metadata, praise, complaints); cross-channel similarity (dislike lines against praise); BM25 for likes and for dislikes; name-match flags; the number of like and dislike lines.

**Rule for choosing:** if L1 does not clearly beat the tuned L0 formula on validation users, keep L0. It is simpler, and every weight can be read.

**Default stack (what we start with):**
- **Reranker:** `lightgbm.LGBMRanker(objective="lambdarank")`, with early stopping on validation NDCG@10. LightGBM's default label gains (0, 1, 3) fit the labels disliked = 0, unseen = 1, liked = 2.
- **Encoder for the similarity features:** `BAAI/bge-small-en-v1.5` (33M parameters, 384 dimensions). The static `minishlab/potion-retrieval-32M` is the speed comparison. BLaIR is reported separately, because its training data overlaps the test period.
- **Exact-word features:** BM25, with the `bm25s` library.
- **Not used as the reranker:** off-the-shelf search rerankers (bge-reranker, MiniLM cross-encoders, FlashRank). They judge query–document relevance, so the whole persona would go in as one query string with likes and dislikes mixed together, which is the negation problem (NevIR). They would also need about 40 lines × 100 items = 4,000 transformer passes per request. One of them stays as the reference X1.

### Training examples and labels

One training query is one (user u, cutoff time t):
- **Input:** the persona built from u's reviews before t, and the features of each candidate.
- **Candidates:** u's real items in the label window after t, plus about 50 items sampled from the base ranker's top list. The sampled items come from near the real item's base rank (rank-matched sampling). Otherwise the model could learn that "an inserted real item has a lower base rank than the rest", which is an artifact.
- **Graded labels:** liked (4–5★) = 2, never seen = 1, disliked (1–2★) = 0. Whether 3★ gets 1 or is dropped is decided in step 1. This is the order liked > unseen > disliked from CLAUDE.md (Paudel 2018). It rewards the model for putting an item the user will dislike below items the user never saw. That is the hard case, because the base ranker tends to rank disliked items high: they look like the user's history.
- **Binary labels** as the comparison: liked = 1, everything else = 0.

### Negatives enter in two places, and a 2×2 separates them (for Q1)

| | binary labels | graded labels (disliked = 0) |
|---|---|---|
| **no dislike features** | T-A | T-B |
| **with dislike features** | T-C | T-D |

- T-A against T-C: do the dislike lines in the persona help?
- T-A against T-B: does training on dislikes help, even without a dislike persona?
- T-D: both together.

### Timeline and users (no leakage)

```
              T1                  T2                 T3
  history ─────┼── train window ───┼─── val window ───┼─── test window ──►
```

- Users are split once, at random: 80% train users, 10% validation users, 10% test users.
- **Train:** train users, labels from [T1, T2).
- **Validation:** validation users, labels from [T2, T3). Used to pick features, hyperparameters and the L0 weights, and for early stopping.
- **Test:** test users, labels from T3 on. Scored once, at the end.
- At each cutoff (T1, T2, T3) everything the model sees is rebuilt from data before that cutoff only: personas, item profiles, item averages, and the base ranker. That makes three snapshots.
- Users need enough reviews before their cutoff, and at least one review in their label window. Step 1 checks how many qualify.
- Later extension: one query per review (t = every review time) gives many more examples, but needs item profiles at many time points.

### Explanations from the trained model

- **L1:** TreeSHAP values per item show which feature moved it. Each dislike feature comes from one specific line (the argmax), so the explanation still names the line.
- **L2:** the pooling weights show which line mattered.

## 7. Encoders

| Tier | Model | Why |
|---|---|---|
| static | `minishlab/potion-retrieval-32M`, `sentence-transformers/static-retrieval-mrl-en-v1` | fastest on CPU. How much quality do we lose? |
| small | `BAAI/bge-small-en-v1.5`, `all-MiniLM-L6-v2` | the usual fast transformers |
| domain | `hyp1231/blair-roberta-base` | trained on Amazon Reviews 2023 |
| reference | `BAAI/bge-base-en-v1.5` | the upper end for bi-encoders |

Item vectors are computed once per encoder.

## 8. Data and protocol

### Main dataset: Amazon Reviews 2023, Movies & TV (Books later as the second domain)

It is the thesis dataset, and it has explicit stars and review text. The raw data may already be on disk at `E:\thies\poc\data\movies.db`. Only the data would be reused, not the code. Check its schema in step 1.

**Users.** At least 10 reviews before their cutoff, with at least 2 of them rated 1–2★. For the pair test (V3), at least one 4–5★ and one 1–2★ review in the label window. Report how many users pass, and aim for 2,000 or more test users. Without an LLM, thousands of users cost almost nothing.

**Cutoffs.** T1, T2 and T3 as in section 6. Pick them so that each window holds enough reviews (about 10% of reviews each for validation and test as a first guess; check the counts).

**Leakage rules.**
- Persona lines and item snippets come only from reviews written before the cutoff.
- The user's own review of a candidate never enters that item's profile.
- Item averages are computed before the cutoff.
- The base ranker is refit at each cutoff.

**Base ranker (the model next to the reranker):**
- B1: popularity × item mean rating before the cutoff. Simple and strong on this kind of data.
- B2: BPR-MF (the `implicit` library), trained on all interactions before the cutoff as positives. It treats disliked items as positives too, which is exactly where a dislike channel should help.
- LightGCN later (RecBole or Cornac), if B2 works.

**Three evaluation views (test users):**

| View | Candidate list | Metrics | What it answers |
|---|---|---|---|
| V1 pipeline | the base ranker's top 100 for the user | Recall@10 and NDCG@10 of liked items in the test window; dislike-rate@10 (the share of the top 10 that the user later rated 1–2★); base recall@100 | the realistic end-to-end effect. A reranker cannot recover items the base missed. |
| V2 controlled list | one test item + 99 items from the base ranker's top list that the user never reviewed | liked targets: HR@10, NDCG@10. Disliked targets: avoid@10 (not in the top 10), mean rank. | the effect per target, liked vs disliked |
| V3 pairs | the user's liked test item vs their disliked test item | within-user AUC: the share of pairs where the liked item scores higher, with ties counted as 0.5 | the cleanest test of the dislike channel, with no sampling |

The headline numbers for each condition are V3 AUC, V2 avoid@10 and V2 HR@10 on liked targets. The **collateral cost** is the change in HR@10 on liked targets when the dislike channel is added.

V2 uses the base ranker's items as negatives, so the lists are hard and realistic. Some of those items may be ones the user would like, which is the sampled-metric bias that Krichene & Rendle (KDD 2020) describe. V3 has no such bias, so final claims rest on V3 and V1.

**Statistics.** A paired bootstrap over users (10,000 resamples) gives a 95% interval for every difference between conditions. V2 sampling is repeated with 3 seeds.

### Second dataset: TRACE (EMNLP 2026, github.com/ahnji0520/TRACE)

What I found in the released file (`final_crs_dataset.json.gz`, checked 7 Nov 2026):
- 1,136 users, 8,168 sessions and 4,588 items. The base data is MTS-KION, a Russian streaming service. GPT-4o generated the dialogues.
- Each session has a behavioral persona with two text fields, `positive_preference` and `negative_preference`. Each field is split into `content_material` and `production_context`. There are also usage patterns, decision drivers and a dialogue persona. The persona is updated every session, so dynamic personas come for free.
- Items have genres, description, keywords, actors, directors, country and year.
- There is one target per session, and **every target was watched at least 70%**, so there are no disliked targets. Items watched under 10% (4,231 of them) appear only among a session's source items.
- The memory holds simulated recommendation outcomes with reasons. The final memory of each user has 1,139 Accept, 538 Reject, 806 Soft Reject and 600 Already Seen. Rejects could stand in for disliked items. However, GPT-4o generated them from the same persona, so a persona-based reranker would be scoring the generator's own reasoning, which is circular.
- Consistency, by a rough name match: the previous session's target (watched at least 70%) has a director or actor named in the positive text 78% of the time and in the negative text 1.4% of the time. The current target is rarely named in its own persona (2.3% positive, 0.6% negative), so there is little leakage. Odd cases exist. In user 699317's session 2, the negative text names Frozen's directors, although the user watched Frozen to 92% and accepted it.
- The release has no candidate lists (their ranker code builds 50 candidates), no license file, and no paper link yet.

How to use it: a cheap second test of the scorer when the persona text is given. It can show whether adding `negative_preference` helps a liked target rank higher among sampled items, and compare a static persona (session 1) with the updated one (current session) for Q2. Training on TRACE can use binary labels only (L0 and L1, T-A and T-C), because it has no disliked targets. It is not the main dislike test. Its negatives are either low watch percentages, the same kind of signal as the Tencent paper's play rate under 0.3, which is still an open question with Moshe, or simulated rejections.

Considered and not chosen: MIND (news impressions with clicks). It is good for reranking, but it has no explicit dislikes and is not the thesis domain.

## 9. Sanity checks before the main run

- **S1, negation probe** for each encoder. For about 50 attributes (horror, musicals, slow pacing, …), compare cos("I love X", an item about X) with cos("I hate X", the same item). We expect almost no difference, which is why polarity comes from the list.
- **S2, polarity flip.** Swap L and D for the test users. The AUC of R2, R3 and the trained models should fall below 0.5. R5 should barely change.
- **S3, wrong persona.** Give each user a random other user's persona. The gains should disappear.
- **S4, leakage audit.** Assert that every timestamp of persona and item-snippet text is earlier than the cutoff it is used with.
- **S5, read by hand.** Take 10 users' personas and their largest demotions. This doubles as the qualitative demo the advisors asked for.

## 10. Hypotheses

- **H1.** The dislike channel (R2 or R3 against R1) raises V3 AUC and V2 avoid@10, and costs at most about 1 point of HR@10 on liked targets.
- **H2.** The one-string persona (R5) does not separate disliked from liked items (its AUC is close to R1's).
- **H3.** The max over lines beats mean pooling and the merged vector (R4).
- **H4.** Matching dislikes against complaints beats matching them against metadata alone.
- **H5.** The soft boundary (R3) has a lower collateral cost than the linear penalty (R2) and the hard filter (R3h).
- **H6.** Static encoders keep most of the small transformers' gain, at a fraction of the latency.
- **H7.** The trained L1 model beats the tuned L0 formula, and its dislike features carry real weight.
- **H8.** Graded labels (T-B, T-D) raise avoid@10 over binary labels.
- **Open, with no direction predicted:** decay or revise against static.

## 11. Link to the thesis questions

- **Q1** (do negatives help?): R1 against R2, R3 and R6, and the T-A to T-D 2×2 (dislikes as persona input against dislikes as training labels).
- **Q2** (update or rebuild?): static against decay against revise. On TRACE, the session-1 persona against the current one.
- **Q3** (how to combine?): separate channels (R2, R3) against one merged vector (R4), one string (R5), and the revise rule.

## 12. Order of work (one step at a time; check each before starting the next)

1. Data: load Movies & TV, apply the user filter and print the counts (users, reviews, share of 1–2★ reviews, users per window, test-window pairs). Choose T1, T2 and T3, and decide about 3★.
2. S1, the negation probe on the encoders.
3. Persona builders P-sent and P-item. Read 10 users (S5).
4. Item profiles (metadata, praise and complaint snippets) at the three cutoffs, under the leakage rules (S4).
5. Base rankers B1 and B2 at the three cutoffs, candidate lists, base recall@100.
6. L0: scorers R0–R6, weights tuned on validation users. V2 and V3 on test users. S2 and S3.
7. L1: training queries, LightGBM, the T-A to T-D 2×2, feature importance and SHAP.
8. L2, the adapter, only if L1 shows that the frozen similarities are the limit.
9. Latency benchmark: CPU, per request, p50 and p95.
10. The TRACE run.
11. Write-up.

## 13. Risks

- Many Amazon negatives are about the product (disc, shipping), not taste. The product filter matters, so measure its effect.
- Generic complaint language ("waste of money") matches every item. The τ threshold and the z-scores are there for this. Also drop dislike lines whose average similarity to all items is high.
- Few users may have both a liked and a disliked item in a window, which limits V3 and the graded labels. Check this in step 1.
- With rank-matched sampling, training lists differ from the real top-100 lists at test time. V1 shows whether the trained model still works on real lists.
- BLaIR was trained on Amazon Reviews 2023, which includes our test period, so it may look better than it is. Report it separately.
- Sentiment classifiers miss sarcasm and mixed reviews. S5 checks this by hand.
- V2 uses sampled lists, with the bias described above. Base final claims on V3 and V1.

## 14. Out of scope for now

Fine-tuning the encoder (L3: contrastive training with dislikes as hard negatives, negation-aware training), LLM-written personas as the main path, cross-domain transfer, and the Big Five.

## Sources

- NevIR: Weller, Lawrie, Van Durme, EACL 2024. https://arxiv.org/abs/2305.07614
- ExcluIR: Zhang et al., AAAI 2025. https://arxiv.org/abs/2404.17288
- Negation awareness in universal text embeddings: Cao, 2025. https://arxiv.org/abs/2504.00584
- Negation robustness via contrastive fine-tuning: Truong et al., 2025. https://arxiv.org/abs/2507.12782
- Vector-PRF (Rocchio for dense retrievers): Li et al. https://arxiv.org/abs/2108.11044 ; Pyserini docs: https://github.com/castorini/pyserini/blob/master/docs/experiments-vector-prf.md
- MIND multi-interest: Li et al., 2019. https://arxiv.org/abs/1904.08030 ; Rethinking multi-interest learning: https://arxiv.org/abs/2302.14532
- PNFRec: Ivanova, Frolov, Vasilev, RecSys 2025. https://arxiv.org/abs/2508.14786
- PANE-GNN: https://arxiv.org/abs/2306.04095 ; NFARec: https://arxiv.org/abs/2404.06900
- EFM: Zhang et al., SIGIR 2014. https://ir.webis.de/anthology/2014.sigirconf_conference-2014.10
- LACE: Mysore, Jasim, McCallum, Zamani, SIGIR 2023. https://arxiv.org/abs/2304.04250 ; code: https://github.com/iesl/editable_user_profiles-lace
- Ramos et al., ACL 2024. https://arxiv.org/abs/2402.05810
- BLUE: https://arxiv.org/abs/2605.06981
- PEARL: Kim et al., ACL Findings 2024. https://arxiv.org/abs/2403.04460
- BLaIR / Amazon Reviews 2023: Hou et al., 2024. https://arxiv.org/abs/2403.03952
- Model2Vec: https://github.com/MinishLab/model2vec ; static embeddings: https://huggingface.co/blog/static-embeddings
- TRACE: https://github.com/ahnji0520/TRACE
- MIND news dataset: https://msnews.github.io
- Krichene & Rendle, "On Sampled Metrics for Item Recommendation", KDD 2020 (cited from memory, not re-checked)
- Paudel et al. 2018 (loss aversion, liked > unseen > disliked), as listed in the thesis CLAUDE.md (not re-checked)
