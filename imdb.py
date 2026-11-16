"""EXP3RT's top-k test (Kim et al., SIGIR 2025, Table 3) on EXP3RT's own IMDB data, and our persona methods on the
same test: the persona fed straight into the backbone (with an ablation of every input), and our reranker.

  python imdb.py fetch                                  # EXP3RT's released IMDB files -> data/imdb/raw/ (about 80 MB)
  python imdb.py baseline                               # Table 3's LightGCN and BPR-MF rows from the released lists
  python imdb.py run --out runs/imdb                    # backbones + ablation, our reranker, report (bge-small, CPU)
  python imdb.py run --out runs/imdb --encoder Qwen/Qwen3-Embedding-8B --seeds 3     # the real run (GPU)
  python imdb.py report --out runs/imdb                 # the report again, e.g. after exp3rt.py wrote its scores

The test (checked 2026-11-16: the LightGCN and BPR-MF rows of Table 3 come out to the 4th digit, R@5 of LightGCN
0.0410 against the paper's 0.0401):
  topk/imdb/train.txt, test.txt   791 users, 1,155 movies; the test movies are each user's movies rated 9-10 that
                                  were put aside (3,426 pairs); train holds the rest of the ratings of 6-10
  lgn_test_top20.json, mf_test_top20.json   the top 20 of EXP3RT's LightGCN / BPR-MF per user, in rank order, only
                                  for the users with at least one test movie among their 20 (275 / 209 users)
  Recall@k = test movies in the top k / the user's test movies; NDCG@k binary, the ideal = min(n, k) hits;
  both averaged over all 791 users. A user without a test movie in the 20 counts 0 whatever the reranker does,
  and Recall@20 is the same for every reranker of a list (only the order inside the 20 changes).

Inputs we take from EXP3RT's release as they are: the user profiles (user_profile.json, [Like] and [Dislike]
bullets written by GPT-3.5) are the personas; the movie text is item_description (title, genres, directors, cast,
plot); the movie profile (item_profile.json) gives what viewers praised ([Like]) and criticized ([Dislike]).
Every persona line and movie line becomes a vector (--encoder); per (user, movie), sorted over the lines (K = 1..5):
  like_sK, dis_sK        persona like / dislike lines against the movie text
  like_pr_sK, dis_cr_sK  each like line's closest praised line, each dislike line's closest criticized line

Backbones with the persona inside (the advisors' idea, 2026-11-16): LightGCN (3 layers) and BPR-MF (0 layers),
trained with BPR on train.txt. score(u, i) = <h_u, h_i> + w . f(u, i), where layer 0 of the user is its ID
embedding (+ W_p [like centroid; dislike centroid]) and layer 0 of the movie its ID embedding (+ W_t movie text);
f(u, i) = the like / dislike distances above, standardized. The ablation adds one input at a time:
  ids           plain backbone (should land near EXP3RT's own LightGCN / MF)
  text          + movie text vector
  like          + like distances (like_sK, like_pr_sK)          dis   + dislike distances (dis_sK, dis_cr_sK)
  like+dis      + both distances
  persona_like  + movie text + like centroid                    persona   + movie text + [like; dislike] centroids
  all           + movie text + both centroids + both distances
  all_swap      all, with every user given another user's persona (in training and test): what remains is capacity,
                not personalization
Validation: 20% of each user's 9-10 rated training movies (at least 1 when the user has 2) are held out; training
stops when their Recall@20 stops improving; the test is a full ranking of the movies the user has not seen in
train.txt. --seeds runs per variant; per-user metrics are averaged over the seeds, and each variant is compared
with ids by a paired sign-flip test over users.

Our reranker on EXP3RT's released lists (no LLM): the learned mixes of mix.py, fitted on the validation lists of
our own ID-only backbone (LightGCN for the LightGCN lists, MF for the MF lists; users with a held-out movie in
their top 20; label = the held-out movie); base = -log(rank in the list):
  O10_mix   base + like_sK + dis_sK                 (R10: the persona against the movie text)
  O12_mix   base + like_pr_sK + dis_cr_sK           (R12: likes against praise, dislikes against criticism)
  O12_likes base + like_pr_sK                       (A1: no dislike part)
each also with another user's persona (the control).

EXP3RT itself (exp3rt.py, GPU): Llama 3.1 8B fine-tuned on the released step-3 data, the expected rating of every
(user, movie) in the lists; the report picks up runs/imdb/exp3rt_<variant>/scores_{lgn,mf}.jsonl when they exist.
"""
import argparse
import hashlib
import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "imdb" / "raw"
REPO = "https://raw.githubusercontent.com/jieyong99/EXP3RT/main/"
FILES = ["data/imdb/rating_bias/lgn_test_top20.json", "data/imdb/rating_bias/mf_test_top20.json",
         "data/imdb/rating_bias/train.json", "data/imdb/rating_bias/valid.json", "data/imdb/rating_bias/test.json",
         "data/imdb/user_profile/user_profile.json", "data/imdb/item_profile/item_profile.json",
         "data/topk/imdb/train.txt", "data/topk/imdb/test.txt", "data/topk/imdb/user_list.txt",
         "data/topk/imdb/item_list.txt"]
KS = (5, 10, 15, 20)
K_SORTED = 5
LISTS = {"lgn": "lgn_test_top20.json", "mf": "mf_test_top20.json"}
PAPER = {  # Table 3 of the paper (LightGCN / BPR-MF rows and EXP3RT on top), for the report
    "lgn": {"base": {"R@5": .0401, "N@5": .0463, "R@10": .0693, "N@10": .0560, "R@15": .0972, "N@15": .0664},
            "EXP3RT": {"R@5": .0543, "N@5": .0560, "R@10": .0846, "N@10": .0669, "R@15": .1006, "N@15": .0734}},
    "mf": {"base": {"R@5": .0296, "N@5": .0328, "R@10": .0468, "N@10": .0379, "R@15": .0659, "N@15": .0450},
           "EXP3RT": {"R@5": .0395, "N@5": .0419, "R@10": .0624, "N@10": .0491, "R@15": .0768, "N@15": .0543}}}


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:6.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------- data
def fetch(force=False):
    RAW.mkdir(parents=True, exist_ok=True)
    rec = {}
    for f in FILES:
        dst = RAW / Path(f).name if "topk" not in f else RAW / ("topk_" + Path(f).name)
        if force or not dst.exists():
            log(f"downloading {f}")
            with urllib.request.urlopen(REPO + f, timeout=600) as r:
                dst.write_bytes(r.read())
        rec[f] = {"file": dst.name, "bytes": dst.stat().st_size,
                  "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()}
    (RAW / "fetch.json").write_text(json.dumps({"source": REPO, "files": rec}, indent=1), encoding="utf-8")
    log(f"{len(rec)} files in {RAW}")


def _json(name):
    return json.loads((RAW / name).read_text(encoding="utf-8"))


def _rows(name):
    return [l.split() for l in (RAW / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def bullets(text, tag):
    """The bullet lines under [tag] in an EXP3RT profile; "None." -> []."""
    m = re.search(r"\[" + tag + r"\](.*?)(?=\n\[[A-Za-z ]+\]|\Z)", text or "", flags=re.S)
    out = []
    for line in (m.group(1) if m else "").splitlines():
        line = line.strip().lstrip("-*").strip()
        if line and line.lower().rstrip(".") != "none":
            out.append(line)
    return out


class Data:
    """EXP3RT's IMDB release in one place. Users are the 791 of the top-k split (key = EXP3RT's user id string),
    movies the 1,155 (key = int of the IMDB id)."""

    def __init__(self):
        if not (RAW / "fetch.json").exists():
            sys.exit(f"{RAW} is empty: run  python imdb.py fetch")
        ul = {r[0]: int(r[1]) for r in _rows("topk_user_list.txt")[1:]}
        il = {int(r[0]): int(r[1]) for r in _rows("topk_item_list.txt")[1:]}
        uid, iid = {v: k for k, v in ul.items()}, {v: k for k, v in il.items()}
        self.users = [uid[k] for k in range(len(uid))]
        self.items = [iid[k] for k in range(len(iid))]
        self.uix = {u: k for k, u in enumerate(self.users)}
        self.iix = {i: k for k, i in enumerate(self.items)}
        self.train = {uid[int(r[0])]: [iid[int(x)] for x in r[1:]] for r in _rows("topk_train.txt")}
        self.test = {uid[int(r[0])]: [iid[int(x)] for x in r[1:]] for r in _rows("topk_test.txt")}
        self.rating = {}
        desc, self.rating_files = {}, {}
        for split in ("train", "valid", "test"):
            recs = _json(f"{split}.json")
            self.rating_files[split] = recs
            for x in recs:
                self.rating[(x["user_id"], int(x["item_id"]))] = int(x["score"])
                desc.setdefault(int(x["item_id"]), x["item_description"])
        self.desc = {i: desc.get(i, "") for i in self.items}
        prof = {int(x["item_id"]): x["item_synopsis"] for x in _json("item_profile.json")}
        self.praised = {i: bullets(prof.get(i, ""), "Like") for i in self.items}
        self.criticized = {i: bullets(prof.get(i, ""), "Dislike") for i in self.items}
        self.persona = {}
        for x in _json("user_profile.json"):
            self.persona[x["user_id"]] = {"like": bullets(x["user_persona"], "Like"),
                                          "dislike": bullets(x["user_persona"], "Dislike")}
        missing = [u for u in self.users if u not in self.persona]
        assert not missing, f"{len(missing)} top-k users without a profile"
        self.lists = {}
        for name, f in LISTS.items():
            lists = {}
            for x in _json(f):
                lists.setdefault(x["user_id"], []).append(int(x["item_id"]))
            assert all(len(v) == 20 for v in lists.values())
            self.lists[name] = lists

    def test_pairs(self):
        return {(u, i) for u, its in self.test.items() for i in its}


# ---------------------------------------------------------------------------------------------------- metrics
def list_metrics(order, relevant, ks=KS):
    """Recall@k and NDCG@k of one ranked list against a set of relevant movies."""
    n = len(relevant)
    out = {}
    hits = [i in relevant for i in order]
    for k in ks:
        out[f"R@{k}"] = sum(hits[:k]) / n if n else 0.0
        dcg = sum(1 / math.log2(p + 2) for p, h in enumerate(hits[:k]) if h)
        idcg = sum(1 / math.log2(p + 2) for p in range(min(n, k)))
        out[f"N@{k}"] = dcg / idcg if idcg else 0.0
    return out


def per_user(data, ranked):
    """ranked: {user: ordered movies} (users missing = no hit possible) -> {metric: array over the 791 users}."""
    keys = [f"{m}@{k}" for k in KS for m in ("R", "N")]
    out = {k: np.zeros(len(data.users)) for k in keys}
    for j, u in enumerate(data.users):
        if u in ranked:
            m = list_metrics(ranked[u], set(data.test.get(u, [])))
            for k in keys:
                out[k][j] = m[k]
    return out


def means(pu):
    return {k: float(v.mean()) for k, v in pu.items()}


def baseline(data=None):
    data = data or Data()
    for name in LISTS:
        m = means(per_user(data, data.lists[name]))
        paper = PAPER[name]["base"]
        log(f"{name}: " + "  ".join(f"{k} {m[k]:.4f} (paper {paper[k]:.4f})" if k in paper else f"{k} {m[k]:.4f}"
                                    for k in m))
    return data


# ---------------------------------------------------------------------------------------------------- vectors
class Vectors:
    """Unit vectors of texts with one encoder, cached by text in data/imdb/vectors/<encoder>.npz."""

    def __init__(self, name):
        import rerank as R
        self.enc = R.Encoder(name, cache_dir=HERE / "data" / "imdb" / "vectors")
        self.path = HERE / "data" / "imdb" / "vectors" / (self.enc.slug + "__texts.npz")
        self.cache = {}
        if self.path.exists():
            z = np.load(self.path, allow_pickle=False)
            self.cache = dict(zip(z["k"].tolist(), z["v"]))

    @staticmethod
    def key(text, query):
        return hashlib.md5((("Q|" if query else "D|") + text).encode("utf-8")).hexdigest()

    def get(self, texts, query=False):
        need = sorted({t for t in texts if self.key(t, query) not in self.cache})
        if need:
            log(f"  encoding {len(need)} texts with {self.enc.name}" + (" (as queries)" if query else ""))
            V = self.enc.encode(need, query=query)
            self.cache.update({self.key(t, query): v.astype(np.float32) for t, v in zip(need, V)})
            self.path.parent.mkdir(parents=True, exist_ok=True)
            keys = sorted(self.cache)
            np.savez(self.path, k=np.array(keys), v=np.stack([self.cache[k] for k in keys]))
        if not texts:
            return np.zeros((0, self.enc.dim), np.float32)
        return np.stack([self.cache[self.key(t, query)] for t in texts])


FEATS = ([f"like_s{k}" for k in range(1, 6)] + [f"dis_s{k}" for k in range(1, 6)]
         + [f"like_pr_s{k}" for k in range(1, 6)] + [f"dis_cr_s{k}" for k in range(1, 6)])
LIKE_FEATS = [f for f in FEATS if f.startswith("like")]
DIS_FEATS = [f for f in FEATS if f.startswith("dis")]


class Persona:
    """Per user: the like / dislike line vectors, the centroids, and the distance features to every movie."""

    def __init__(self, data, vec):
        import rerank as R
        self.data, self.R = data, R
        self.M = vec.get([data.desc[i] for i in data.items])                    # (movies, d)
        self.PR = [vec.get(data.praised[i]) for i in data.items]
        self.CR = [vec.get(data.criticized[i]) for i in data.items]
        self.dim = self.M.shape[1]
        self.lines = {}
        for u, p in data.persona.items():
            self.lines[u] = (vec.get(p["like"], query=True), vec.get(p["dislike"], query=True))

    def centroid(self, u):
        L, D = self.lines[u]
        unit = lambda X: X.mean(0) / (np.linalg.norm(X.mean(0)) or 1.0) if len(X) else np.zeros(self.dim, np.float32)
        return np.concatenate([unit(L), unit(D)]).astype(np.float32)

    def feats(self, u):
        """(movies, len(FEATS)) for persona u; NaN where a movie has no praised / criticized line."""
        R = self.R
        L, D = self.lines[u]
        cols = [R.sorted_sims(L @ self.M.T), R.sorted_sims(D @ self.M.T),
                R.sorted_sims(R.best_per_line(L, self.PR)), R.sorted_sims(R.best_per_line(D, self.CR))]
        return np.concatenate(cols).T.astype(np.float32)


def derangement(users, seed):
    order = np.random.default_rng(seed).permutation(len(users))
    return {users[order[k]]: users[order[(k + 1) % len(order)]] for k in range(len(order))}


# ---------------------------------------------------------------------------------------------------- backbones
VARIANTS = {   # name -> (movie text in layer 0, user persona vector: None / "like" / "both", distance features)
    "ids": (False, None, []),
    "text": (True, None, []),
    "like": (False, None, LIKE_FEATS),
    "dis": (False, None, DIS_FEATS),
    "like+dis": (False, None, FEATS),
    "persona_like": (True, "like", []),
    "persona": (True, "both", []),
    "all": (True, "both", FEATS),
    "all_swap": (True, "both", FEATS),
}


def split_val(data, seed=0, share=0.2):
    """Per user, 20% of the training movies rated 9-10 (at least 1 when the user has 2 or more) go to validation."""
    rng = np.random.default_rng(seed)
    tr, va = {}, {}
    for u in data.users:
        its = list(data.train.get(u, []))
        top = [i for i in its if data.rating.get((u, i), 0) >= 9]
        n = int(round(share * len(top))) or (1 if len(top) >= 2 else 0)
        hold = set(rng.choice(top, n, replace=False).tolist()) if n else set()
        tr[u] = [i for i in its if i not in hold]
        va[u] = sorted(hold)
    return tr, va


class Backbone:
    """LightGCN (layers >= 1) or BPR-MF (layers = 0) with optional persona inputs (VARIANTS)."""

    def __init__(self, data, train, variant, persona, layers, dim=64, swap=None, device="cpu", seed=0):
        import torch
        self.torch, self.device = torch, torch.device(device)
        torch.manual_seed(seed)
        nu, ni = len(data.users), len(data.items)
        self.nu, self.ni, self.layers = nu, ni, layers
        text, pvec, feats = VARIANTS[variant]
        who = {u: (swap[u] if swap else u) for u in data.users}       # whose persona each user carries
        u_idx = np.array([data.uix[u] for u in data.users for _ in train[u]], dtype=np.int64)
        i_idx = np.array([data.iix[i] for u in data.users for i in train[u]], dtype=np.int64)
        self.pos = torch.from_numpy(np.stack([u_idx, i_idx], 1)).to(self.device)
        rows = np.concatenate([u_idx, i_idx + nu])
        cols = np.concatenate([i_idx + nu, u_idx])
        deg = np.bincount(rows, minlength=nu + ni).astype(np.float64)
        vals = 1.0 / np.sqrt(np.maximum(deg[rows] * deg[cols], 1e-12))
        self.graph = torch.sparse_coo_tensor(torch.from_numpy(np.stack([rows, cols])), torch.from_numpy(vals).float(),
                                             (nu + ni, nu + ni)).coalesce().to(self.device)
        nn = torch.nn
        self.emb = nn.Embedding(nu + ni, dim).to(self.device)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.params_extra = []
        self.T = self.Wt = self.P = self.Wp = self.F = self.wf = None
        if text:
            self.T = torch.from_numpy(persona.M).to(self.device)
            self.Wt = nn.Linear(self.T.shape[1], dim, bias=False).to(self.device)
            nn.init.normal_(self.Wt.weight, std=0.01)
            self.params_extra += list(self.Wt.parameters())
        if pvec:
            P = np.stack([persona.centroid(who[u]) for u in data.users])
            if pvec == "like":
                P = P[:, :persona.dim]
            self.P = torch.from_numpy(P).to(self.device)
            self.Wp = nn.Linear(self.P.shape[1], dim, bias=False).to(self.device)
            nn.init.normal_(self.Wp.weight, std=0.01)
            self.params_extra += list(self.Wp.parameters())
        if feats:
            cols = [FEATS.index(f) for f in feats]
            F = np.stack([persona.feats(who[u])[:, cols] for u in data.users])      # (users, movies, feats)
            mu, sd = np.nanmean(F, axis=(0, 1)), np.nanstd(F, axis=(0, 1))
            F = np.nan_to_num((F - mu) / np.where(sd > 1e-9, sd, 1.0), nan=0.0).astype(np.float32)
            self.F = torch.from_numpy(F).to(self.device)
            self.wf = nn.Linear(F.shape[2], 1, bias=False).to(self.device)
            nn.init.zeros_(self.wf.weight)
            self.params_extra += list(self.wf.parameters())

    def parameters(self):
        return [self.emb.weight] + self.params_extra

    def state(self):
        return [p.detach().clone() for p in self.parameters()]

    def load(self, state):
        with self.torch.no_grad():
            for p, s in zip(self.parameters(), state):
                p.copy_(s)

    def propagate(self):
        torch = self.torch
        x = self.emb.weight
        if self.Wp is not None or self.Wt is not None:
            u, i = x[:self.nu], x[self.nu:]
            if self.Wp is not None:
                u = u + self.Wp(self.P)
            if self.Wt is not None:
                i = i + self.Wt(self.T)
            x = torch.cat([u, i])
        out = [x]
        for _ in range(self.layers):
            x = torch.sparse.mm(self.graph, x)
            out.append(x)
        return torch.stack(out).mean(0)

    def scores(self, users, h=None):
        """Scores of every movie for the user indices (tensor (len(users), movies))."""
        h = self.propagate() if h is None else h
        s = h[users] @ h[self.nu:].T
        if self.wf is not None:
            s = s + self.wf(self.F[users]).squeeze(-1)
        return s

    def fit(self, val, masks, lr=1e-3, reg=1e-4, batch=1024, max_epochs=1000, every=5, patience=10, seed=0):
        """BPR; early stopping on Recall@20 of the held-out movies (val: {user index: set of movie indices})."""
        torch = self.torch
        g = torch.Generator(device="cpu").manual_seed(seed)
        opt = torch.optim.Adam([{"params": [self.emb.weight]},
                                {"params": self.params_extra, "weight_decay": reg}], lr=lr)
        best, best_state, bad, curve = -1.0, None, 0, []
        n = len(self.pos)
        for epoch in range(1, max_epochs + 1):
            perm = torch.randperm(n, generator=g).to(self.device)
            for b in range(0, n, batch):
                idx = perm[b:b + batch]
                bu, bp = self.pos[idx, 0], self.pos[idx, 1]
                bn = torch.randint(0, self.ni, (len(idx),), generator=g).to(self.device)
                h = self.propagate()
                eu, ep, en = h[bu], h[self.nu + bp], h[self.nu + bn]
                sp, sn = (eu * ep).sum(1), (eu * en).sum(1)
                if self.wf is not None:
                    sp = sp + self.wf(self.F[bu, bp]).squeeze(-1)
                    sn = sn + self.wf(self.F[bu, bn]).squeeze(-1)
                w0 = self.emb.weight
                l2 = (w0[bu].pow(2).sum() + w0[self.nu + bp].pow(2).sum() + w0[self.nu + bn].pow(2).sum()) / (2 * len(idx))
                loss = -torch.nn.functional.logsigmoid(sp - sn).mean() + reg * l2
                opt.zero_grad()
                loss.backward()
                opt.step()
            if epoch % every == 0:
                r = self.recall(val, masks, 20)
                curve.append((epoch, round(r, 4)))
                if r > best:
                    best, best_state, bad = r, self.state(), 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
        self.load(best_state)
        return {"best_val_R@20": round(best, 4), "epochs": epoch, "curve": curve}

    def topk(self, masks, k):
        """Top k movie indices per user index, the user's masked movies left out."""
        torch = self.torch
        with torch.no_grad():
            h = self.propagate()
            s = self.scores(torch.arange(self.nu, device=self.device), h).cpu().numpy()
        for u, m in masks.items():
            s[u, list(m)] = -np.inf
        return np.argsort(-s, axis=1, kind="stable")[:, :k]

    def recall(self, val, masks, k):
        top = self.topk(masks, k)
        r = [len(set(top[u].tolist()) & its) / len(its) for u, its in val.items() if its]
        return float(np.mean(r)) if r else 0.0


# ---------------------------------------------------------------------------------------------------- our reranker
OURS = {"O10_mix": ["base"] + [f"like_s{k}" for k in range(1, 6)] + [f"dis_s{k}" for k in range(1, 6)],
        "O12_mix": ["base"] + [f"like_pr_s{k}" for k in range(1, 6)] + [f"dis_cr_s{k}" for k in range(1, 6)],
        "O12_likes": ["base"] + [f"like_pr_s{k}" for k in range(1, 6)]}


def mix_user(persona, data, u, items, targets, carrier=None):
    """A user in mix.py's format: the list = items (in base order), base = -log(rank), the persona's features."""
    F = persona.feats(carrier or u)
    feat = {}
    for r, i in enumerate(items, 1):
        row = F[data.iix[i]]
        feat[i] = {"base": -math.log(r), **{c: float(row[k]) for k, c in enumerate(FEATS)}}
    return {"user_id": u, "candidates": list(items), "cand_set": set(items), "list": list(items), "feat": feat,
            "targets": [{"i": i, "rating": data.rating.get((u, i), 10)} for i in targets]}


# ---------------------------------------------------------------------------------------------------- run
def paired_cell(a, b):
    import evaluate as E
    d, lo, hi, p = E.paired(a, b)
    return f"{d * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}] " + ("p<0.001" if p < 0.001 else f"p {p:.3f}")


def run(args):
    import torch
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = baseline()
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    vec = Vectors(args.encoder)
    persona = Persona(data, vec)
    swap = derangement(data.users, seed=42)
    tr, va = split_val(data)
    val = {data.uix[u]: {data.iix[i] for i in va[u]} for u in data.users if va[u]}
    mask_val = {data.uix[u]: {data.iix[i] for i in tr[u]} for u in data.users}
    mask_test = {data.uix[u]: {data.iix[i] for i in data.train.get(u, [])} for u in data.users}
    log(f"validation: {sum(map(len, va.values()))} held-out movies of {len(val)} users | device {device} | "
        f"encoder {args.encoder} ({persona.dim} dims)")

    # ---- backbones and the ablation
    results = {"encoder": args.encoder, "seeds": args.seeds, "backbones": {}, "ours": {}}
    ranks_ids = {}
    for bb, layers in (("lgn", 3), ("mf", 0)):
        for variant in args.variants:
            per_seed = []
            for seed in range(args.seeds):
                t1 = time.time()
                m = Backbone(data, tr, variant, persona, layers, swap=swap if variant.endswith("_swap") else None,
                             device=device, seed=seed)
                info = m.fit(val, mask_val, seed=seed, max_epochs=args.max_epochs)
                top = m.topk(mask_test, 20)
                ranked = {u: [data.items[j] for j in top[data.uix[u]]] for u in data.users}
                pu = per_user(data, ranked)
                per_seed.append(pu)
                if variant == "ids" and seed == 0:
                    ranks_ids[bb] = m.topk(mask_val, 20)              # validation lists for our reranker
                if m.wf is not None:
                    info["feature_weights"] = dict(zip(VARIANTS[variant][2], [round(float(x), 4) for x in
                                                                               m.wf.weight.detach().cpu().numpy()[0]]))
                info["seconds"] = round(time.time() - t1, 1)
                log(f"{bb} {variant:12s} seed {seed}: val R@20 {info['best_val_R@20']:.4f} after {info['epochs']} "
                    f"epochs | test R@10 {pu['R@10'].mean():.4f} N@10 {pu['N@10'].mean():.4f} R@20 "
                    f"{pu['R@20'].mean():.4f} | {info['seconds']:.0f}s")
                results["backbones"].setdefault(bb, {}).setdefault(variant, {"runs": []})["runs"].append(info)
            avg = {k: np.mean([p[k] for p in per_seed], axis=0) for k in per_seed[0]}
            results["backbones"][bb][variant]["per_user"] = {k: v.tolist() for k, v in avg.items()}
            results["backbones"][bb][variant]["mean"] = means(avg)

    # ---- our reranker on EXP3RT's released lists
    for bb in LISTS:
        top = ranks_ids[bb]
        train_users = [mix_user(persona, data, u, [data.items[j] for j in top[data.uix[u]]], va[u])
                       for u in data.users if va[u]]
        import mix as MX
        lists = data.lists[bb]
        for name, inputs in OURS.items():
            w, info = MX.fit(train_users, inputs)
            rer = MX.reranker(w)
            res = {}
            for tag, carrier in (("own", None), ("swap", swap)):
                ranked = {}
                for u, items in lists.items():
                    usr = mix_user(persona, data, u, items, [], carrier=carrier[u] if carrier else None)
                    s = rer(usr)
                    ranked[u] = sorted(items, key=lambda i: (-s[i], items.index(i)))
                res[tag] = per_user(data, ranked)
            results["ours"].setdefault(bb, {})[name] = {"weights": info, **{
                tag: {"per_user": {k: v.tolist() for k, v in pu.items()}, "mean": means(pu)} for tag, pu in res.items()}}
            log(f"{bb} {name}: fitted on {info['users']} users ({info['targets']} held-out movies in the 20), L2 "
                f"{info['l2']:g}, like sum {info.get('like_sum', {}).get('w', float('nan')):+.3f}, dislike sum "
                f"{info.get('dis_sum', {}).get('w', float('nan')):+.3f} | R@10 {res['own']['R@10'].mean():.4f} "
                f"(another user's persona {res['swap']['R@10'].mean():.4f})")
    (out / "imdb_results.json").write_text(json.dumps(results), encoding="utf-8")
    report(out, data)


def exp3rt_ranked(path, lists):
    """EXP3RT's order of each list from a scores file: by expected rating, ties by the base rank."""
    sc = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            x = json.loads(line)
            sc[(x["user_id"], int(x["item_id"]))] = x["expected"]
    missing = sum((u, i) not in sc for u, its in lists.items() for i in its)
    ranked = {u: sorted(its, key=lambda i: (-sc.get((u, i), -1.0), its.index(i))) for u, its in lists.items()}
    return ranked, missing


def report(out, data=None):
    data = data or Data()
    out = Path(out)
    res = json.loads((out / "imdb_results.json").read_text(encoding="utf-8"))
    keys = [f"{m}@{k}" for k in KS for m in ("R", "N")]
    arr = lambda d: {k: np.array(v) for k, v in d.items()}
    L = [f"# EXP3RT's IMDB top-k test (Table 3) | encoder {res['encoder']} | {res['seeds']} seed(s) per backbone variant",
         "", "Recall / NDCG over the 791 test users (k = 5, 10, 15, 20). Paired tests: per user, two-sided sign-flip, "
         "differences x100.", ""]
    for bb in LISTS:
        base = per_user(data, data.lists[bb])
        rows = {"base (EXP3RT's list)": means(base)}
        paper = PAPER[bb]
        cmp_rows = []
        for v in sorted(out.glob("exp3rt_*/scores_" + bb + ".jsonl")):
            ranked, missing = exp3rt_ranked(v, data.lists[bb])
            pu = per_user(data, ranked)
            name = f"EXP3RT ({v.parent.name.replace('exp3rt_', '')})" + (f" [{missing} pairs unscored]" if missing else "")
            rows[name] = means(pu)
            cmp_rows.append((name, pu))
        for name, r in res["ours"].get(bb, {}).items():
            rows[name] = r["own"]["mean"]
            rows[name + " (another user's persona)"] = r["swap"]["mean"]
            cmp_rows.append((name, arr(r["own"]["per_user"])))
            cmp_rows.append((name + " (another user's persona)", arr(r["swap"]["per_user"])))
        L += [f"## {'LightGCN' if bb == 'lgn' else 'BPR-MF'}: reranking EXP3RT's released top-20 lists", "",
              "| | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        L.append("| paper: base | " + " | ".join(f"{paper['base'].get(k, float('nan')):.4f}" for k in keys) + " |")
        L.append("| paper: EXP3RT | " + " | ".join(f"{paper['EXP3RT'].get(k, float('nan')):.4f}" for k in keys) + " |")
        for name, m in rows.items():
            L.append(f"| {name} | " + " | ".join(f"{m[k]:.4f}" for k in keys) + " |")
        L += ["", "minus the base list, paired per user:", "", "| | R@5 | N@5 | R@10 | N@10 |", "|---|---|---|---|---|"]
        for name, pu in cmp_rows:
            L.append(f"| {name} | " + " | ".join(paired_cell(pu[k], base[k]) for k in ("R@5", "N@5", "R@10", "N@10")) + " |")
        ex = [(n, pu) for n, pu in cmp_rows if n.startswith("EXP3RT")]
        ours = [(n, pu) for n, pu in cmp_rows if n.startswith("O") and "another" not in n]
        if ex and ours:
            L += ["", "ours minus EXP3RT, paired per user:", "", "| | R@5 | N@5 | R@10 | N@10 |", "|---|---|---|---|---|"]
            for en, epu in ex:
                for on, opu in ours:
                    L.append(f"| {on} - {en} | " + " | ".join(paired_cell(opu[k], epu[k])
                                                               for k in ("R@5", "N@5", "R@10", "N@10")) + " |")
        L.append("")
        for name, r in res["ours"].get(bb, {}).items():
            w = r["weights"]
            L.append(f"{name}: fitted on {w['users']} users with a held-out movie in our ID-only backbone's top 20 "
                     f"({w['targets']} movies), L2 {w['l2']:g}; weights " + ", ".join(
                         f"{c} {v['w']:+.3f}" for c, v in w["weights"].items()))
        L.append("")
    L += ["## The persona inside the backbone: ablation (full ranking of the unseen movies)", ""]
    for bb, variants in res["backbones"].items():
        ids = arr(variants["ids"]["per_user"]) if "ids" in variants else None
        L += [f"### {'LightGCN' if bb == 'lgn' else 'BPR-MF'} (EXP3RT's own {bb.upper()}: R@10 "
              f"{PAPER[bb]['base']['R@10']:.4f}, N@10 {PAPER[bb]['base']['N@10']:.4f})", "",
              "| variant | " + " | ".join(keys) + " | val R@20 | epochs |", "|---|" + "---|" * (len(keys) + 2)]
        for v, r in variants.items():
            runs = r["runs"]
            L.append(f"| {v} | " + " | ".join(f"{r['mean'][k]:.4f}" for k in keys) +
                     f" | {np.mean([x['best_val_R@20'] for x in runs]):.4f} | {np.mean([x['epochs'] for x in runs]):.0f} |")
        if ids is not None:
            L += ["", "minus ids, paired per user (metrics averaged over the seeds first):", "",
                  "| variant | R@10 | N@10 | R@20 | N@20 |", "|---|---|---|---|---|"]
            for v, r in variants.items():
                if v != "ids":
                    pu = arr(r["per_user"])
                    L.append(f"| {v} | " + " | ".join(paired_cell(pu[k], ids[k]) for k in ("R@10", "N@10", "R@20", "N@20")) + " |")
            if "all" in variants and "all_swap" in variants:
                a, s = arr(variants["all"]["per_user"]), arr(variants["all_swap"]["per_user"])
                L.append("| all - all_swap | " + " | ".join(paired_cell(a[k], s[k]) for k in ("R@10", "N@10", "R@20", "N@20")) + " |")
            fw = [(v, x["feature_weights"]) for v, r in variants.items() for x in r["runs"][:1] if "feature_weights" in x]
            if fw:
                L += ["", "learned weights of the distance features (seed 0; inputs standardized):", ""]
                for v, d in fw:
                    L.append(f"- {v}: like sum {sum(w for c, w in d.items() if c.startswith('like')):+.3f}, dislike sum "
                             f"{sum(w for c, w in d.items() if c.startswith('dis')):+.3f} | " +
                             ", ".join(f"{c} {w:+.3f}" for c, w in d.items()))
        L.append("")
    timing = sorted(out.glob("exp3rt_*/timing.json"))
    if timing:
        L += ["## Time", ""]
        for t in timing:
            L.append(f"- {t.parent.name}: " + ", ".join(f"{k} {v}" for k, v in json.loads(t.read_text()).items()))
        L.append("")
    text = "\n".join(L)
    (out / "eval_imdb.md").write_text(text + "\n", encoding="utf-8")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("fetch", "baseline", "run", "report"))
    ap.add_argument("--out", default=str(HERE / "runs" / "imdb"))
    ap.add_argument("--encoder", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--force", action="store_true", help="fetch: download again")
    args = ap.parse_args()
    if args.cmd == "fetch":
        fetch(args.force)
    elif args.cmd == "baseline":
        baseline()
    elif args.cmd == "run":
        run(args)
    else:
        report(args.out)


if __name__ == "__main__":
    main()
