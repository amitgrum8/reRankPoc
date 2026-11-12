"""Step 5 (no LLM; CPU or GPU): the persona reranker. BLUEPRINT sections 5 and 6, level L0.

  python evaluate.py runs/<run>             # evaluate.py calls attach() below; eval.md gets one column per formula

Every item in a user's list becomes one row of numbers, the same row LightGBM (level L1) reads:
  base, base_rank            LightGCN's score and its rank among the user's candidates
  like_max/mean/top3         cosine between the item text (title + description) and the persona's like lines
  dis_max/mean/top3          the same for the dislike lines (0 when the persona has none)
  n_like, n_dis              number of persona lines
  rocchio                    cosine to one merged vector mean(likes) - gamma * mean(dislikes)
  naive                      cosine to the whole persona as one string ("Likes: ... Dislikes: ...")
With item profiles (item_profiles.py: what reviewers praised and criticized about the item, a few short lines
each), persona line against item line, the closest pair:
  like_praised, dis_crit     like lines x praised lines; dislike lines x criticized lines
  like_crit, dis_praised     the cross pairs: the user likes what the item is criticized for, and the reverse
  n_crit                     number of criticized lines (0 also when the item has no profile)
Likes and dislikes stay in separate columns, so a formula (or a model) can give them opposite signs.

Encoder: BAAI/bge-small-en-v1.5, frozen. Item text = title + description (first 120 words). Cached in
data/vectors/: the item texts (only the items that appear in lists, plus 1000 random items for the tau
threshold) and the profile lines (re-encoded when an item's lines change). Persona lines are encoded per run.

The rows go to <run>/features.jsonl with a label: 2 = liked target (4-5 stars), 0 = disliked target (1-2),
1 = everything else (unseen candidates, and 3-star targets, marked three=true).

Formulas (z = z-score over the user's candidates, b = LightGCN score):
  R1_like     z(b) + a*z(like_max)
  R2_like-dis z(b) + a*z(like_max) - b*z(dis_max)
  R3_soft     z(b) + a*z(like_max) - b*max(0, dis_max - tau)/sd(dis_max)   penalize only clear dislike matches;
              tau = 95th percentile of dis_max over 1000 random items (a persona with generic dislike lines gets
              a high tau, so those lines stop matching everything)
  R4_rocchio  z(b) + a*z(rocchio)
  R5_naive    z(b) + a*z(naive)          the whole persona as one query: expected to ignore negation
  R8_rev_like z(b) + a*z(like_praised)   the item's reviews instead of its description (PROFILE_RERANKERS,
  R9_rev      z(b) + a*z(like_praised) - b*z(dis_crit)                                 only with item profiles)
a = b = gamma = 1 unless evaluate.py is told otherwise (--alpha, --beta, --gamma).
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer      # missing -> evaluate.py keeps the LightGCN column only

HERE = Path(__file__).resolve().parent
ENCODER = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "   # the bge instruction for short queries
N_RANDOM = 1000
MAX_WORDS = 120
PARAMS = {"alpha": 1.0, "beta": 1.0, "gamma": 1.0}
COLS = ("base", "base_rank", "like_max", "like_mean", "like_top3", "dis_max", "dis_mean", "dis_top3",
        "n_like", "n_dis", "rocchio", "naive")
PROFILE_COLS = ("like_praised", "dis_crit", "like_crit", "dis_praised", "n_crit")
PROFILES_PATH = HERE / "data" / "item_profiles" / "item_profiles.jsonl"


def item_text(item):
    title = (item.get("title") or "").strip()
    desc = re.sub(r"\s+", " ", item.get("description") or "").strip()
    words = desc.split()
    if len(words) > MAX_WORDS:
        desc = " ".join(words[:MAX_WORDS])
    return f"{title}. {desc}" if desc else title


class Encoder:
    def __init__(self, name=ENCODER, cache_dir=HERE / "data" / "vectors"):
        import torch
        self.model = SentenceTransformer(name, device="cuda" if torch.cuda.is_available() else "cpu")
        self.model.max_seq_length = 256
        self.dim = self.model.get_sentence_embedding_dimension()
        self.path = Path(cache_dir) / (name.replace("/", "__") + ".npz")
        self.cache = {}
        if self.path.exists():
            z = np.load(self.path)
            self.cache = dict(zip(z["i"].tolist(), z["v"]))

    def encode(self, texts, query=False):
        if query:
            texts = [QUERY_PREFIX + t for t in texts]
        return self.model.encode(list(texts), batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False).astype(np.float32)

    def items(self, ids, items):
        """Vectors of the item indices in ids (rows in that order); encodes and caches the missing ones."""
        missing = sorted({i for i in ids if i not in self.cache})
        if missing:
            print(f"  encoding {len(missing)} item texts with {ENCODER} (cached afterwards) ...", flush=True)
            self.cache.update(zip(missing, self.encode([item_text(items[i]) for i in missing])))
            self.path.parent.mkdir(parents=True, exist_ok=True)
            keys = sorted(self.cache)
            np.savez(self.path, i=np.array(keys, dtype=np.int64), v=np.stack([self.cache[k] for k in keys]))
        return np.stack([self.cache[i] for i in ids]) if ids else np.zeros((0, self.dim), np.float32)


class Profiles:
    """Item profiles (item_profiles.py) and the vectors of their lines, cached next to the item vectors
    (float16 on disk; an item is re-encoded when its lines change)."""

    def __init__(self, enc, path=PROFILES_PATH, cache_dir=HERE / "data" / "vectors"):
        self.enc, self.lines, self.vec, self.hash = enc, {}, {}, {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                self.lines[int(r["i"])] = ([x.strip() for x in r.get("praised") or [] if x.strip()],
                                           [x.strip() for x in r.get("criticized") or [] if x.strip()])
        self.path = Path(cache_dir) / (ENCODER.replace("/", "__") + "__profiles.npz")
        if self.path.exists():
            z = np.load(self.path)
            v, k = z["v"].astype(np.float32), 0
            for i, h, a, b in zip(z["i"].tolist(), z["h"].tolist(), z["pn"].tolist(), z["cn"].tolist()):
                self.vec[i], self.hash[i] = (v[k:k + a], v[k + a:k + a + b]), h
                k += a + b

    def _h(self, i):
        return hashlib.md5(json.dumps(self.lines[i], ensure_ascii=False).encode("utf-8")).hexdigest()

    def get(self, ids):
        """{i: (praised vectors, criticized vectors)} for the items in ids that have a profile; encodes and
        caches the missing or changed ones (call once with every id needed: saving the cache is slow)."""
        need = sorted({i for i in ids if i in self.lines and self.hash.get(i) != self._h(i)})
        if need:
            texts = [t for i in need for t in self.lines[i][0] + self.lines[i][1]]
            print(f"  encoding {len(texts)} profile lines of {len(need)} items (cached afterwards) ...", flush=True)
            V = self.enc.encode(texts) if texts else np.zeros((0, self.enc.dim), np.float32)
            k = 0
            for i in need:
                a, b = len(self.lines[i][0]), len(self.lines[i][1])
                self.vec[i], self.hash[i] = (V[k:k + a], V[k + a:k + a + b]), self._h(i)
                k += a + b
            keys = sorted(self.vec)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.path, i=np.array(keys, dtype=np.int64), h=np.array([self.hash[i] for i in keys]),
                     pn=np.array([len(self.vec[i][0]) for i in keys], dtype=np.int64),
                     cn=np.array([len(self.vec[i][1]) for i in keys], dtype=np.int64),
                     v=np.concatenate([np.concatenate(self.vec[i]) for i in keys]).astype(np.float16))
        return {i: self.vec[i] for i in ids if i in self.vec}


def pooled(sim):
    """sim: (lines, items) -> max, mean, top-3 mean per item; zeros when there are no lines."""
    if sim.shape[0] == 0:
        z = np.zeros(sim.shape[1], np.float32)
        return z, z, z
    s = -np.sort(-sim, axis=0)
    return s[0], sim.mean(0), s[:3].mean(0)


def _mx(A, B):
    """Closest pair between the rows of A and the rows of B (cosine; the rows are unit vectors), 0 if either is empty."""
    return float((A @ B.T).max()) if len(A) and len(B) else 0.0


def attach(users, items, enc, profiles=None, seed=42):
    """Adds user["feat"] = {item: {column: value}} and user["tau"] to every user (needs user["persona"])."""
    assert all(int(items[i]["i"]) == i for i in (0, len(items) // 2, len(items) - 1)), "items.json is not in index order"
    rng = np.random.default_rng(seed)
    random_ids = sorted(rng.choice(len(items), N_RANDOM, replace=False).tolist())
    all_ids = sorted({i for u in users for i in u["list"]})
    enc.items(sorted(set(all_ids) | set(random_ids)), items)      # one encoding pass
    pv = profiles.get(all_ids) if profiles else {}
    if profiles:
        print(f"  item profiles: {len(pv)} of the {len(all_ids)} items in the lists have one", flush=True)
    R = enc.items(random_ids, items)
    dim, gamma = R.shape[1], PARAMS["gamma"]
    E = np.zeros((0, dim), np.float32)
    for user in users:
        like = [l for l in user["persona"]["like"] if l.strip()]
        dis = [l for l in user["persona"]["dislike"] if l.strip()]
        L = enc.encode(like, query=True) if like else E
        D = enc.encode(dis, query=True) if dis else E
        q = enc.encode(["Likes: " + "; ".join(like) + ". Dislikes: " + ("; ".join(dis) or "none") + "."], query=True)[0]
        u = (L.mean(0) if len(L) else np.zeros(dim, np.float32)) - gamma * (D.mean(0) if len(D) else np.zeros(dim, np.float32))
        u = u / (np.linalg.norm(u) or 1.0)
        ids = user["list"]
        M = enc.items(ids, items)
        lmax, lmean, ltop3 = pooled(L @ M.T)
        dmax, dmean, dtop3 = pooled(D @ M.T)
        tau = float(np.percentile(pooled(D @ R.T)[0], 95)) if len(D) else 0.0
        cand = np.array([user["base"][i] for i in user["candidates"]])
        feat = {}
        for k, i in enumerate(ids):
            b = user["base"][i]
            P, C = pv.get(i, (E, E))
            feat[i] = {"base": b, "base_rank": int((cand > b).sum()) + 1, "like_max": float(lmax[k]),
                       "like_mean": float(lmean[k]), "like_top3": float(ltop3[k]), "dis_max": float(dmax[k]),
                       "dis_mean": float(dmean[k]), "dis_top3": float(dtop3[k]), "n_like": len(like), "n_dis": len(dis),
                       "rocchio": float(M[k] @ u), "naive": float(M[k] @ q),
                       "like_praised": _mx(L, P), "dis_crit": _mx(D, C), "like_crit": _mx(L, C),
                       "dis_praised": _mx(D, P), "n_crit": len(C)}
        user["feat"], user["tau"] = feat, tau


def write_features(users, path):
    """One row per (user, item in the list), with the LightGBM label."""
    with open(path, "w", encoding="utf-8") as f:
        for user in users:
            t = {x["i"]: x["rating"] for x in user["targets"]}
            for i in user["list"]:
                r = t.get(i)
                row = {"user_id": user["user_id"], "i": i, "target": r is not None, "rating": r,
                       "label": 1 if r is None or r == 3 else 2 if r >= 4 else 0, "three": r == 3, "tau": user["tau"]}
                row.update(user["feat"][i])
                f.write(json.dumps(row) + "\n")


# ------------------------------------------------------------------------------------------------- L0 formulas
def _z(user, col):
    vals = np.array([user["feat"][i][col] for i in user["candidates"]], dtype=float)
    mu, sd = vals.mean(), vals.std()
    return (lambda x: 0.0) if sd < 1e-9 else (lambda x: (x - mu) / sd)


def _formula(score):
    def rerank(user):
        f, a, b = user["feat"], PARAMS["alpha"], PARAMS["beta"]
        z = {c: _z(user, c) for c in ("base", "like_max", "dis_max", "rocchio", "naive", "like_praised", "dis_crit")}
        sd_dis = float(np.std([f[i]["dis_max"] for i in user["candidates"]])) or 1.0
        return {i: float(score(f[i], z, a, b, user["tau"], sd_dis)) for i in user["list"]}
    return rerank


RERANKERS = {
    "R1_like": _formula(lambda r, z, a, b, tau, sd: z["base"](r["base"]) + a * z["like_max"](r["like_max"])),
    "R2_like-dis": _formula(lambda r, z, a, b, tau, sd: z["base"](r["base"]) + a * z["like_max"](r["like_max"])
                            - b * z["dis_max"](r["dis_max"])),
    "R3_soft": _formula(lambda r, z, a, b, tau, sd: z["base"](r["base"]) + a * z["like_max"](r["like_max"])
                        - b * max(0.0, r["dis_max"] - tau) / sd),
    "R4_rocchio": _formula(lambda r, z, a, b, tau, sd: z["base"](r["base"]) + a * z["rocchio"](r["rocchio"])),
    "R5_naive": _formula(lambda r, z, a, b, tau, sd: z["base"](r["base"]) + a * z["naive"](r["naive"])),
}

# only meaningful when attach() got item profiles (otherwise the columns are all zero and these equal LightGCN)
PROFILE_RERANKERS = {
    "R8_rev_like": _formula(lambda r, z, a, b, tau, sd: z["base"](r["base"]) + a * z["like_praised"](r["like_praised"])),
    "R9_rev": _formula(lambda r, z, a, b, tau, sd: z["base"](r["base"]) + a * z["like_praised"](r["like_praised"])
                       - b * z["dis_crit"](r["dis_crit"])),
}
